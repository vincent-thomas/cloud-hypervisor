// Copyright © 2026 The Cloud Hypervisor Authors
//
// SPDX-License-Identifier: Apache-2.0

//! On demand restore: serving page faults, snapshotting the restored VM, and
//! restoring from snapshot chains.
//!
//! A directory written by the snapshot mode holds every page. A snapshot taken
//! while serving an on demand restore holds only the pages this daemon
//! populated: its `populated-<slot>` bitmap lists them, and its `base` file
//! names the directory the restore came from. A page resolves to the newest
//! directory in that chain that holds it. Slot numbers can change across
//! restores, so regions are matched between directories by guest address.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use log::{info, warn};
use vm_memory::mmap::MmapRegion;
use vm_memory::{
    Address, Bytes, FileOffset, GuestAddress, GuestMemoryRegion, GuestRegionMmap,
    MemoryRegionAddress,
};
use vm_migration::protocol::{Command, MemoryRange, Request, Response};
use vmm::VmMigrationConfig;
use vmm::migration::SNAPSHOT_STATE_FILE;
use vmm::sparse::copy_region;

use crate::{
    Error, MIGRATION_CONFIG_FILENAME, Result, expect_command, memory_slot_filename, recv_memory_fd,
    slot_info,
};

const PAGE_SIZE: u64 = 4096;
const BASE_FILENAME: &str = "base";

fn populated_filename(slot: u32) -> String {
    format!("populated-{slot}")
}

fn bit(bitmap: &[u64], page: u64) -> bool {
    bitmap[(page / 64) as usize] & (1 << (page % 64)) != 0
}

/// One guest RAM region of one snapshot directory.
struct LayerRegion {
    file: File,
    /// Pages this directory holds. `None` when it holds every page.
    populated: Option<Vec<u64>>,
}

/// Regions of one snapshot directory, keyed by guest address.
struct Layer {
    regions: Vec<(u64, LayerRegion)>,
}

impl Layer {
    fn region(&self, gpa: u64) -> Result<&LayerRegion> {
        self.regions
            .iter()
            .find(|(g, _)| *g == gpa)
            .map(|(_, r)| r)
            .ok_or(Error::RegionNotInSnapshot(gpa))
    }
}

/// A snapshot directory and its bases, newest first.
pub(crate) struct Chain {
    layers: Vec<Layer>,
}

impl Chain {
    pub(crate) fn load(dir: &Path) -> Result<Self> {
        let mut layers = Vec::new();
        let mut next = Some(dir.to_path_buf());
        while let Some(dir) = next {
            let config_bytes =
                fs::read(dir.join(MIGRATION_CONFIG_FILENAME)).map_err(Error::ReadFile)?;
            let config: VmMigrationConfig = serde_json::from_slice(&config_bytes)?;
            let mut regions = Vec::new();
            for (slot, gpa, _size, _file_offset) in slot_info(&config)? {
                let file =
                    File::open(dir.join(memory_slot_filename(slot))).map_err(Error::ReadFile)?;
                let populated = match fs::read(dir.join(populated_filename(slot))) {
                    Ok(bytes) => Some(
                        bytes
                            .as_chunks::<8>()
                            .0
                            .iter()
                            .map(|w| u64::from_le_bytes(*w))
                            .collect(),
                    ),
                    Err(e) if e.kind() == io::ErrorKind::NotFound => None,
                    Err(e) => return Err(Error::ReadFile(e)),
                };
                regions.push((gpa, LayerRegion { file, populated }));
            }
            next = match fs::read_to_string(dir.join(BASE_FILENAME)) {
                Ok(base) => Some(PathBuf::from(base.trim_end())),
                Err(e) if e.kind() == io::ErrorKind::NotFound => None,
                Err(e) => return Err(Error::ReadFile(e)),
            };
            layers.push(Layer { regions });
        }
        info!("Loaded a snapshot chain of {} directories", layers.len());
        Ok(Self { layers })
    }

    /// Copy the region at `gpa` into `dst` at `dst_offset`, keeping the
    /// holes of the oldest directory.
    pub(crate) fn populate(&self, gpa: u64, size: u64, dst: &File, dst_offset: u64) -> Result<()> {
        let mut layers = self.layers.iter().rev();
        let oldest = layers.next().ok_or(Error::RegionNotInSnapshot(gpa))?;
        copy_region(&oldest.region(gpa)?.file, 0, dst, dst_offset, size)
            .map_err(Error::CopyMemory)?;
        let mut page = vec![0u8; PAGE_SIZE as usize];
        for layer in layers {
            let region = layer.region(gpa)?;
            let Some(populated) = &region.populated else {
                copy_region(&region.file, 0, dst, dst_offset, size).map_err(Error::CopyMemory)?;
                continue;
            };
            for index in (0..size / PAGE_SIZE).filter(|&p| bit(populated, p)) {
                let offset = index * PAGE_SIZE;
                region
                    .file
                    .read_exact_at(&mut page, offset)
                    .map_err(Error::CopyMemory)?;
                dst.write_all_at(&page, dst_offset + offset)
                    .map_err(Error::CopyMemory)?;
            }
        }
        Ok(())
    }

    /// Read one page of the region at `gpa` from the newest directory holding it.
    fn read_page(&self, gpa: u64, offset: u64, page: &mut [u8]) -> Result<()> {
        for layer in &self.layers {
            let region = layer.region(gpa)?;
            if region
                .populated
                .as_ref()
                .is_none_or(|p| bit(p, offset / PAGE_SIZE))
            {
                return region
                    .file
                    .read_exact_at(page, offset)
                    .map_err(Error::CopyMemory);
            }
        }
        Err(Error::RegionNotInSnapshot(gpa))
    }
}

/// A guest RAM region served on demand.
pub(crate) struct OnDemandSlot {
    region: GuestRegionMmap,
    /// Device and inode of the memfd backing the region.
    memfd_id: (u64, u64),
    /// Pages written into the memfd.
    populated: Vec<AtomicU64>,
}

impl OnDemandSlot {
    pub(crate) fn new(memfd: &File, gpa: u64, size: u64, file_offset: u64) -> Result<Self> {
        // Map the same memfd CH maps, so our page writes are visible to it.
        let fo = FileOffset::new(memfd.try_clone().map_err(Error::CloneMemfd)?, file_offset);
        let mmap = MmapRegion::build(
            Some(fo),
            size as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
        )
        .map_err(Error::Mmap)?;
        let region = GuestRegionMmap::new(mmap, GuestAddress(gpa)).ok_or(Error::GuestRegion)?;
        let metadata = memfd.metadata().map_err(Error::ReadFile)?;
        let words = size.div_ceil(PAGE_SIZE * 64) as usize;
        Ok(Self {
            region,
            memfd_id: (metadata.dev(), metadata.ino()),
            populated: (0..words).map(|_| AtomicU64::new(0)).collect(),
        })
    }

    fn gpa(&self) -> u64 {
        self.region.start_addr().raw_value()
    }

    fn contains(&self, gpa: u64, len: u64) -> bool {
        gpa >= self.gpa() && gpa.saturating_add(len) <= self.gpa() + self.region.len()
    }

    fn is_populated(&self, page: u64) -> bool {
        self.populated[(page / 64) as usize].load(Ordering::Acquire) & (1 << (page % 64)) != 0
    }

    fn set_populated(&self, page: u64) {
        self.populated[(page / 64) as usize].fetch_or(1 << (page % 64), Ordering::AcqRel);
    }

    /// Whether the page is still in the memfd, rather than discarded.
    fn is_resident(&self, page: u64) -> Result<bool> {
        let addr = self
            .region
            .get_host_address(MemoryRegionAddress(page * PAGE_SIZE))
            .map_err(Error::WriteGuestMemory)?;
        let mut vec = 0u8;
        // SAFETY: `addr` is a page-aligned address inside our mapping of the
        // memfd, and `vec` has room for the one page queried.
        let ret = unsafe { libc::mincore(addr.cast(), PAGE_SIZE as usize, &mut vec) };
        if ret != 0 {
            return Err(Error::Mincore(io::Error::last_os_error()));
        }
        Ok(vec & 1 != 0)
    }
}

/// Serves the pages of an on demand restore and snapshots of the restored VM.
pub(crate) struct Pager {
    slots: Vec<OnDemandSlot>,
    chain: Chain,
    /// The directory the restore came from, the base of every snapshot.
    base_dir: PathBuf,
    /// Serializes populating pages against writing a snapshot.
    lock: Mutex<()>,
}

impl Pager {
    pub(crate) fn new(slots: Vec<OnDemandSlot>, chain: Chain, base_dir: &Path) -> Result<Self> {
        Ok(Self {
            slots,
            chain,
            base_dir: base_dir.canonicalize().map_err(Error::ReadFile)?,
            lock: Mutex::new(()),
        })
    }

    fn populate(&self, gpa: u64, len: u64) -> Result<()> {
        let slot = self
            .slots
            .iter()
            .find(|s| s.contains(gpa, len))
            .ok_or(Error::PageFaultUnmapped(gpa, len))?;
        let offset = gpa - slot.gpa();
        let _guard = self.lock.lock().unwrap();
        let mut page = vec![0u8; PAGE_SIZE as usize];
        for index in offset / PAGE_SIZE..(offset + len).div_ceil(PAGE_SIZE) {
            let address = MemoryRegionAddress(index * PAGE_SIZE);
            if slot.is_populated(index) {
                // The guest may have modified this page, so never overwrite
                // it. CH only asks again after the guest discarded the page
                // (e.g. through the balloon), and then its content is moot.
                if !slot.is_resident(index)? {
                    page.fill(0);
                    slot.region
                        .write_slice(&page, address)
                        .map_err(Error::WriteGuestMemory)?;
                }
                continue;
            }
            self.chain
                .read_page(slot.gpa(), index * PAGE_SIZE, &mut page)?;
            slot.region
                .write_slice(&page, address)
                .map_err(Error::WriteGuestMemory)?;
            slot.set_populated(index);
        }
        Ok(())
    }

    pub(crate) fn serve_page_faults(&self, stream: &mut UnixStream) -> Result<()> {
        let mut served: u64 = 0;
        loop {
            let req = match Request::read_from(stream) {
                Ok(r) => r,
                Err(e) => {
                    info!("Serve loop: socket closed after {served} PageFault(s) ({e:?})");
                    return Ok(());
                }
            };
            match req.command() {
                Command::PageFault => {
                    let range = MemoryRange::read_from(stream).map_err(Error::Protocol)?;
                    served += 1;
                    if served <= 5 || served.is_power_of_two() {
                        info!(
                            "PageFault #{served}: gpa={:#x} len={}",
                            range.gpa, range.length
                        );
                    }
                    self.populate(range.gpa, range.length)?;
                    Response::ok().write_to(stream).map_err(Error::Protocol)?;
                }
                #[expect(deprecated)] // last sent in v52
                Command::Abandon => {
                    info!("Serve loop: received Abandon, exiting");
                    Response::ok().write_to(stream).ok();
                    return Ok(());
                }
                c => return Err(Error::UnexpectedCommand(c, "a PageFault")),
            }
        }
    }

    /// Accept snapshots of the restored VM on `listener`, writing each one to
    /// a numbered subdirectory of `snapshot_dir`.
    pub(crate) fn serve_snapshots(self: Arc<Self>, listener: UnixListener, snapshot_dir: PathBuf) {
        thread::spawn(move || {
            for n in 1.. {
                let mut stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(e) => {
                        warn!("Snapshot listener failed: {e}");
                        return;
                    }
                };
                let out = snapshot_dir.join(n.to_string());
                match self.receive_snapshot(&mut stream, &out) {
                    Ok(()) => info!("Snapshot persisted to {out:?}"),
                    Err(e) => warn!("Snapshot {n} failed: {e:?}"),
                }
            }
        });
    }

    fn receive_snapshot(&self, stream: &mut UnixStream, out: &Path) -> Result<()> {
        fs::create_dir_all(out).map_err(Error::CreateOutputDir)?;
        expect_command(stream, Command::Start, "Start")?;
        Response::ok().write_to(stream).map_err(Error::Protocol)?;

        // CH slot number -> index of our slot backed by the same memfd.
        let mut slots: Vec<(u32, usize)> = Vec::new();
        let mut have_config = false;
        let mut have_state = false;
        loop {
            let req = Request::read_from(stream).map_err(Error::Protocol)?;
            match req.command() {
                Command::MemoryFd => {
                    let (slot, file) = recv_memory_fd(stream)?;
                    let metadata = file.metadata().map_err(Error::ReadFile)?;
                    let id = (metadata.dev(), metadata.ino());
                    // Only the memfds this daemon populated can be described
                    // relative to its base.
                    let Some(index) = self.slots.iter().position(|s| s.memfd_id == id) else {
                        Response::error().write_to(stream).ok();
                        return Err(Error::ForeignMemoryFd(slot));
                    };
                    slots.push((slot, index));
                }
                Command::Config | Command::State => {
                    let mut buf = vec![0u8; req.length() as usize];
                    stream.read_exact(&mut buf).map_err(Error::ReadPayload)?;
                    let name = if req.command() == Command::Config {
                        have_config = true;
                        MIGRATION_CONFIG_FILENAME
                    } else {
                        have_state = true;
                        SNAPSHOT_STATE_FILE
                    };
                    fs::write(out.join(name), &buf).map_err(Error::WriteFile)?;
                }
                Command::CompletePaused | Command::Complete => {
                    if !have_config || !have_state {
                        return Err(Error::PrematureCompletion("Config and State"));
                    }
                    if slots.len() != self.slots.len() {
                        Response::error().write_to(stream).ok();
                        return Err(Error::MissingSlot(slots.len() as u32));
                    }
                    // Persist before ACKing: CH may resume the VM right after.
                    self.write_snapshot(out, &slots)?;
                    Response::ok().write_to(stream).map_err(Error::Protocol)?;
                    return Ok(());
                }
                c => return Err(Error::UnexpectedCommand(c, "a snapshot command")),
            }
            Response::ok().write_to(stream).map_err(Error::Protocol)?;
        }
    }

    fn write_snapshot(&self, out: &Path, slots: &[(u32, usize)]) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        for &(slot, index) in slots {
            let source = &self.slots[index];
            let file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(out.join(memory_slot_filename(slot)))
                .map_err(Error::WriteFile)?;
            file.set_len(source.region.len())
                .map_err(Error::WriteFile)?;
            let pages = source.region.len() / PAGE_SIZE;
            let mut buf = Vec::new();
            let mut page = 0;
            while page < pages {
                if !source.is_populated(page) {
                    page += 1;
                    continue;
                }
                let start = page;
                while page < pages && source.is_populated(page) {
                    page += 1;
                }
                buf.resize(((page - start) * PAGE_SIZE) as usize, 0);
                source
                    .region
                    .read_slice(&mut buf, MemoryRegionAddress(start * PAGE_SIZE))
                    .map_err(Error::WriteGuestMemory)?;
                file.write_all_at(&buf, start * PAGE_SIZE)
                    .map_err(Error::WriteFile)?;
            }
            file.sync_all().map_err(Error::WriteFile)?;
            let bitmap: Vec<u8> = source
                .populated
                .iter()
                .flat_map(|w| w.load(Ordering::Acquire).to_le_bytes())
                .collect();
            fs::write(out.join(populated_filename(slot)), bitmap).map_err(Error::WriteFile)?;
        }
        fs::write(
            out.join(BASE_FILENAME),
            self.base_dir.as_os_str().as_encoded_bytes(),
        )
        .map_err(Error::WriteFile)?;
        Ok(())
    }
}
