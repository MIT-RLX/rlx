// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One copy of a weight across executables, on unified memory.
//!
//! Each executable has its own weight buffer, laid out for its own params, so
//! two executables of one model (a decode step, a prompt prefill, a media
//! prefill) used to hold a copy each — three copies of the language model.
//! Under a sharing scope ([`set_scope`], one per package's weights) the weight
//! buffer is a page-aligned VM region wrapped as a no-copy `MTLBuffer`, and
//! each weight in it is backed by a Mach memory entry kept in a process-wide
//! registry by `(scope, name, dtype, bytes)`: the first executable to place a
//! weight makes the entry over its own pages, every later one maps the same
//! pages at its own offset. The encoders see one buffer and offsets as before.
//!
//! Only externalized weights take part (params every consumer of which reads
//! through the weight buffer: matmul weights, embedding tables), so a param
//! written while running (a cache) is never shared. A mapped weight's pages
//! outlive the executable that made them: the entry holds them until the last
//! executable mapping it drops.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

#[allow(non_camel_case_types)]
type mach_port_t = u32;
#[allow(non_camel_case_types)]
type kern_return_t = i32;

const VM_FLAGS_ANYWHERE: i32 = 0x1;
const VM_FLAGS_FIXED: i32 = 0x0;
const VM_FLAGS_OVERWRITE: i32 = 0x4000;
const VM_PROT_RW: i32 = 0x1 | 0x2;
const MAP_MEM_NAMED_CREATE: i32 = 0x0002_0000;
const MAP_MEM_LEDGER_TAGGED: i32 = 0x0000_2000;
const VM_LEDGER_TAG_DEFAULT: i32 = 0x1;
const VM_INHERIT_SHARE: u32 = 0;

unsafe extern "C" {
    static mach_task_self_: mach_port_t;
    static vm_page_size: usize;
    fn mach_vm_allocate(task: mach_port_t, addr: *mut u64, size: u64, flags: i32) -> kern_return_t;
    fn mach_vm_deallocate(task: mach_port_t, addr: u64, size: u64) -> kern_return_t;
    fn mach_make_memory_entry_64(
        task: mach_port_t,
        size: *mut u64,
        offset: u64,
        permission: i32,
        object_handle: *mut mach_port_t,
        parent_entry: mach_port_t,
    ) -> kern_return_t;
    #[allow(clippy::too_many_arguments)]
    fn mach_vm_map(
        task: mach_port_t,
        address: *mut u64,
        size: u64,
        mask: u64,
        flags: i32,
        object: mach_port_t,
        offset: u64,
        copy: i32,
        cur_protection: i32,
        max_protection: i32,
        inheritance: u32,
    ) -> kern_return_t;
    fn mach_port_deallocate(task: mach_port_t, name: mach_port_t) -> kern_return_t;
    fn mach_memory_entry_ownership(
        entry: mach_port_t,
        owner: mach_port_t,
        ledger_tag: i32,
        ledger_flags: i32,
    ) -> kern_return_t;
}

fn task() -> mach_port_t {
    // SAFETY: a process-constant port name the runtime initializes.
    unsafe { mach_task_self_ }
}

/// The VM page size (16 KiB on Apple silicon).
pub(crate) fn page_size() -> usize {
    // SAFETY: a process-constant the runtime initializes.
    unsafe { vm_page_size }
}

/// `n` rounded up to whole pages.
pub(crate) fn page_round(n: usize) -> usize {
    n.div_ceil(page_size()) * page_size()
}

/// An executable's weight region: anonymous, page-aligned, freed on drop
/// (pages mapped from shared entries stay alive through their entries).
pub(crate) struct VmRegion {
    addr: u64,
    size: u64,
}

impl VmRegion {
    pub(crate) fn new(size: usize) -> Option<Self> {
        let size = page_round(size.max(1)) as u64;
        let mut addr = 0u64;
        // SAFETY: an out-parameter for a fresh anonymous allocation.
        let kr = unsafe { mach_vm_allocate(task(), &mut addr, size, VM_FLAGS_ANYWHERE) };
        (kr == 0).then_some(Self { addr, size })
    }

    pub(crate) fn ptr(&self) -> *mut u8 {
        self.addr as *mut u8
    }

    pub(crate) fn len(&self) -> usize {
        self.size as usize
    }
}

impl Drop for VmRegion {
    fn drop(&mut self) {
        // SAFETY: the region this struct allocated; nothing reads it after.
        unsafe { mach_vm_deallocate(task(), self.addr, self.size) };
    }
}

// SAFETY: plain address bookkeeping; the memory is shared with the GPU, which
// the executable synchronizes as for any of its buffers.
unsafe impl Send for VmRegion {}
unsafe impl Sync for VmRegion {}

/// A weight's pages, as a memory entry another region can map.
pub(crate) struct SharedPages {
    port: mach_port_t,
    size: u64,
}

impl Drop for SharedPages {
    fn drop(&mut self) {
        // SAFETY: the send right this struct owns.
        unsafe { mach_port_deallocate(task(), self.port) };
    }
}

impl SharedPages {
    /// `len` (whole pages) of fresh zeroed memory as an entry this task owns:
    /// charged to its footprint once, however many regions map it. (Pages
    /// of a plain anonymous region shared through an entry are charged once
    /// per mapping — ten mappings of 512 MB read as 5.4 GB — though they are
    /// one copy; a per-process limit such as iOS's would see the larger.)
    fn create(len: usize) -> Option<Self> {
        let mut size = len as u64;
        let mut port: mach_port_t = 0;
        // SAFETY: out-parameters; a new named object, no parent.
        let kr = unsafe {
            mach_make_memory_entry_64(
                task(),
                &mut size,
                0,
                VM_PROT_RW | MAP_MEM_NAMED_CREATE | MAP_MEM_LEDGER_TAGGED,
                &mut port,
                0,
            )
        };
        if kr != 0 {
            return None;
        }
        let made = Self { port, size };
        // SAFETY: the entry just made; this task as its owner.
        let kr = unsafe { mach_memory_entry_ownership(port, task(), VM_LEDGER_TAG_DEFAULT, 0) };
        (kr == 0 && size >= len as u64).then_some(made)
    }

    /// Map these pages over `region[offset..offset + size]`, replacing what
    /// was there.
    fn map_into(&self, region: &VmRegion, offset: usize) -> bool {
        if offset as u64 + self.size > region.size {
            return false;
        }
        let mut at = region.addr + offset as u64;
        // SAFETY: a fixed, in-bounds range of a region this process owns.
        let kr = unsafe {
            mach_vm_map(
                task(),
                &mut at,
                self.size,
                0,
                VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
                self.port,
                0,
                0,
                VM_PROT_RW,
                VM_PROT_RW,
                VM_INHERIT_SHARE,
            )
        };
        kr == 0 && at == region.addr + offset as u64
    }
}

// SAFETY: a port name and a size; Mach ports are process-wide.
unsafe impl Send for SharedPages {}
unsafe impl Sync for SharedPages {}

/// A shared weight's identity: its scope (a package's weights), name, dtype
/// and byte length.
type Key = (String, String, rlx_ir::DType, usize);

fn registry() -> &'static Mutex<HashMap<Key, Weak<SharedPages>>> {
    static R: std::sync::OnceLock<Mutex<HashMap<Key, Weak<SharedPages>>>> =
        std::sync::OnceLock::new();
    R.get_or_init(Default::default)
}

static SCOPE: Mutex<Option<String>> = Mutex::new(None);

/// Share weights among executables compiled under `scope` (the same name
/// means the same values within it); `None` stops sharing.
pub fn set_scope(scope: Option<String>) {
    *SCOPE.lock().expect("scope lock") = scope;
}

/// The current sharing scope.
pub fn scope() -> Option<String> {
    SCOPE.lock().expect("scope lock").clone()
}

/// Back `region[offset..offset + page_round(nbytes)]` with the shared copy of
/// weight `name` in `scope`: map it when one is live (`(entry, true)`: its
/// values are already there), else make the shared copy, zeroed, and map
/// that (`(entry, false)`: the caller uploads). `None`: the VM calls failed
/// and the range stays private.
pub(crate) fn place(
    scope: &str,
    name: &str,
    dtype: rlx_ir::DType,
    nbytes: usize,
    region: &VmRegion,
    offset: usize,
) -> Option<(Arc<SharedPages>, bool)> {
    let key = (scope.to_string(), name.to_string(), dtype, nbytes);
    let mut reg = registry().lock().expect("shared weights lock");
    if let Some(live) = reg.get(&key).and_then(Weak::upgrade)
        && live.map_into(region, offset)
    {
        return Some((live, true));
    }
    let made = Arc::new(SharedPages::create(page_round(nbytes))?);
    if !made.map_into(region, offset) {
        return None;
    }
    reg.retain(|_, w| w.strong_count() > 0);
    reg.insert(key, Arc::downgrade(&made));
    Some((made, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" {
        fn task_info(
            task: mach_port_t,
            flavor: i32,
            info: *mut i32,
            count: *mut u32,
        ) -> kern_return_t;
    }

    /// This task's `phys_footprint` (`TASK_VM_INFO`).
    fn footprint() -> u64 {
        let mut info = [0i32; 128];
        let mut count = info.len() as u32;
        // SAFETY: a buffer of `count` naturals for TASK_VM_INFO (22).
        unsafe { task_info(task(), 22, info.as_mut_ptr(), &mut count) };
        // `phys_footprint` is the 19th u64 of `task_vm_info`; the buffer is
        // i32s, so it may sit off an 8-byte boundary.
        unsafe { std::ptr::read_unaligned((info.as_ptr() as *const u64).add(18)) }
    }

    #[test]
    fn a_weight_mapped_many_times_is_charged_once() {
        let len = 256 << 20;
        let before = footprint();
        let regions: Vec<VmRegion> = (0..6).map(|_| VmRegion::new(len).unwrap()).collect();
        let (first, _) = place("charge", "w", rlx_ir::DType::F32, len, &regions[0], 0).unwrap();
        unsafe { std::ptr::write_bytes(regions[0].ptr(), 3, len) };
        let held: Vec<_> = regions[1..]
            .iter()
            .map(|r| place("charge", "w", rlx_ir::DType::F32, len, r, 0).unwrap())
            .collect();
        assert!(
            held.iter()
                .all(|(p, mapped)| *mapped && Arc::ptr_eq(p, &first))
        );
        assert!(
            regions
                .iter()
                .all(|r| unsafe { *r.ptr().add(len - 1) } == 3)
        );
        let grew = footprint().saturating_sub(before);
        assert!(
            grew < 2 * len as u64,
            "six mappings of {len} B charged {grew} B"
        );
    }

    #[test]
    fn a_mapped_weight_is_the_same_pages_and_outlives_its_maker() {
        let page = page_size();
        let a = VmRegion::new(3 * page).unwrap();
        let (made, mapped) = place("t", "w", rlx_ir::DType::F32, page + 8, &a, page).unwrap();
        assert!(!mapped);
        let w = unsafe { std::slice::from_raw_parts_mut(a.ptr().add(page), page + 8) };
        w.iter_mut()
            .enumerate()
            .for_each(|(i, v)| *v = (i % 251) as u8);

        let b = VmRegion::new(4 * page).unwrap();
        let (again, mapped) = place("t", "w", rlx_ir::DType::F32, page + 8, &b, 2 * page).unwrap();
        assert!(mapped && Arc::ptr_eq(&made, &again));
        let seen = unsafe { std::slice::from_raw_parts(b.ptr().add(2 * page), page + 8) };
        assert!(seen.iter().enumerate().all(|(i, &v)| v == (i % 251) as u8));
        // One copy: a write through either region shows in the other.
        unsafe { *b.ptr().add(2 * page) = 77 };
        assert_eq!(w[0], 77);
        // Another scope or size is another weight.
        let (_, mapped) = place("u", "w", rlx_ir::DType::F32, page + 8, &b, 0).unwrap();
        assert!(!mapped);

        drop(made);
        drop(a);
        let seen = unsafe { std::slice::from_raw_parts(b.ptr().add(2 * page), page + 8) };
        assert_eq!(seen[0], 77);
        assert!(
            seen[1..]
                .iter()
                .enumerate()
                .all(|(i, &v)| v == ((i + 1) % 251) as u8)
        );
        drop(again);
        assert!(
            registry()
                .lock()
                .unwrap()
                .get(&("t".into(), "w".into(), rlx_ir::DType::F32, page + 8))
                .and_then(Weak::upgrade)
                .is_none()
        );
    }
}
