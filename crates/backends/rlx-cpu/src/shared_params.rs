// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One copy of a weight across CPU executables.
//!
//! Each executable owns one arena and every kernel addresses it by offset, so
//! two executables of one model (a prefill graph, a decode graph per KV
//! bucket) each held a full copy of the weights. Under a sharing scope
//! ([`set_scope`], one label per model's weights) the arena is a page-aligned
//! anonymous mapping, and each large weight's page-aligned slot is backed by a
//! shared-memory object kept in a process-wide registry by
//! `(scope, name, dtype, bytes)`: the first executable to place a weight makes
//! the object, every later one maps the same pages at its own offset. Kernels
//! still see one buffer and the same offsets.
//!
//! The Metal backend does the same with Mach memory entries
//! (`rlx_metal::shared_weights`); this is the POSIX version. Elsewhere (wasm,
//! Windows) [`SUPPORTED`] is false and the registry is compiled but inert.

#![cfg_attr(not(unix), allow(dead_code))]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

static SCOPE: Mutex<Option<String>> = Mutex::new(None);

/// Set the sharing scope for executables compiled from now on (`None` turns
/// sharing off). Every executable compiled under one label must load the same
/// model's weights: a weight is matched by name, dtype and size alone.
pub fn set_scope(scope: Option<String>) {
    *SCOPE.lock().expect("scope lock") = scope;
}

/// The current sharing scope.
pub fn scope() -> Option<String> {
    SCOPE.lock().expect("scope lock").clone()
}

/// Whether this platform can share weight pages.
pub const SUPPORTED: bool = cfg!(unix);

/// The VM page size; slots shared through [`MappedArena::map_shared`] must
/// start on one.
pub fn page_size() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: sysconf has no preconditions.
        let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if p > 0 {
            return p as usize;
        }
    }
    16 * 1024
}

fn page_round(n: usize) -> usize {
    let p = page_size();
    n.div_ceil(p) * p
}

type Key = (String, String, u8, usize);

/// A shared-memory object holding one weight; closed when the last
/// executable mapping it drops (the mappings themselves keep the pages).
struct SharedWeight {
    #[cfg(unix)]
    fd: libc::c_int,
}

impl Drop for SharedWeight {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: `fd` is owned by this value and closed exactly once.
        unsafe {
            libc::close(self.fd);
        }
    }
}

static REGISTRY: Mutex<Option<HashMap<Key, Weak<SharedWeight>>>> = Mutex::new(None);

/// A page-aligned, zero-filled arena whose ranges can be backed by shared
/// weight pages.
pub struct MappedArena {
    ptr: *mut u8,
    len: usize,
    mapped_len: usize,
    held: Vec<Arc<SharedWeight>>,
}

// SAFETY: the mapping is owned by this value; access is through `&`/`&mut`.
unsafe impl Send for MappedArena {}
unsafe impl Sync for MappedArena {}

impl MappedArena {
    /// A zeroed anonymous mapping of at least `len` bytes. `None` when the
    /// platform has no mmap or the call fails.
    pub fn new(len: usize) -> Option<Self> {
        #[cfg(unix)]
        {
            let mapped_len = page_round(len.max(1));
            // SAFETY: an anonymous private mapping with no address hint.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    mapped_len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            if ptr == libc::MAP_FAILED {
                return None;
            }
            Some(Self {
                ptr: ptr.cast(),
                len,
                mapped_len,
                held: Vec::new(),
            })
        }
        #[cfg(not(unix))]
        {
            let _ = len;
            None
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` maps `mapped_len >= len` readable bytes for our lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`; `&mut self` makes the borrow exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Back `[offset, offset + page_round(nbytes))` with the shared copy of
    /// weight `name`. Returns `Some(true)` when another executable already
    /// made it (its values are there), `Some(false)` when this call made it
    /// (zeroed; the caller uploads), and `None` when the range stays private
    /// (a failed system call). `offset` must be page-aligned and the rounded
    /// range must lie inside the arena.
    pub fn map_shared(
        &mut self,
        scope: &str,
        name: &str,
        dtype_tag: u8,
        nbytes: usize,
        offset: usize,
    ) -> Option<bool> {
        #[cfg(unix)]
        {
            let size = page_round(nbytes);
            if !offset.is_multiple_of(page_size()) || offset + size > self.mapped_len {
                return None;
            }
            let key: Key = (scope.to_string(), name.to_string(), dtype_tag, nbytes);
            let mut reg = REGISTRY.lock().expect("registry lock");
            let reg = reg.get_or_insert_with(HashMap::new);
            let (weight, existed) = match reg.get(&key).and_then(Weak::upgrade) {
                Some(w) => (w, true),
                None => {
                    let w = Arc::new(create_object(size)?);
                    reg.insert(key, Arc::downgrade(&w));
                    (w, false)
                }
            };
            // SAFETY: the target range lies inside our own mapping (checked
            // above), so MAP_FIXED only replaces pages this arena owns.
            let p = unsafe {
                libc::mmap(
                    self.ptr.add(offset).cast(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    weight.fd,
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                return None;
            }
            self.held.push(weight);
            Some(existed)
        }
        #[cfg(not(unix))]
        {
            let _ = (scope, name, dtype_tag, nbytes, offset);
            None
        }
    }
}

impl Drop for MappedArena {
    fn drop(&mut self) {
        #[cfg(unix)]
        // SAFETY: unmaps exactly the region `new` mapped (shared sub-ranges
        // included); nothing borrows it past `self`.
        unsafe {
            libc::munmap(self.ptr.cast(), self.mapped_len);
        }
    }
}

/// An unnamed shared-memory object of `size` bytes.
#[cfg(unix)]
fn create_object(size: usize) -> Option<SharedWeight> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    // A unique name, unlinked straight away: the object lives on through the
    // descriptor and the mappings, and nothing else can open it.
    let name = format!(
        "/rlx-w-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let cname = std::ffi::CString::new(name).ok()?;
    // SAFETY: plain POSIX calls on a descriptor this function owns.
    unsafe {
        let fd = libc::shm_open(
            cname.as_ptr(),
            libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
            0o600 as libc::c_uint,
        );
        if fd < 0 {
            return None;
        }
        libc::shm_unlink(cname.as_ptr());
        if libc::ftruncate(fd, size as libc::off_t) != 0 {
            libc::close(fd);
            return None;
        }
        Some(SharedWeight { fd })
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn a_weight_mapped_twice_is_one_copy() {
        let page = page_size();
        let mut a = MappedArena::new(page * 4).unwrap();
        let mut b = MappedArena::new(page * 3).unwrap();
        assert_eq!(a.map_shared("t", "w", 0, page + 8, page), Some(false));
        a.as_mut_slice()[page..page + 4].copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(b.map_shared("t", "w", 0, page + 8, 0), Some(true));
        assert_eq!(&b.as_slice()[..4], &[1, 2, 3, 4]);
        // A write through either is seen by both; private pages stay private.
        b.as_mut_slice()[1] = 9;
        assert_eq!(a.as_slice()[page + 1], 9);
        a.as_mut_slice()[0] = 7;
        assert_eq!(b.as_slice()[page * 2 + 1], 0);
        // Another scope or size is another weight.
        let mut c = MappedArena::new(page * 2).unwrap();
        assert_eq!(c.map_shared("u", "w", 0, page + 8, 0), Some(false));
        assert_eq!(c.as_slice()[0], 0);
        // Misaligned or out-of-range requests stay private.
        assert_eq!(c.map_shared("t", "w", 0, 8, 1), None);
        assert_eq!(c.map_shared("t", "w", 0, page * 4, 0), None);
    }
}
