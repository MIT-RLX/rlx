// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! rocTX shim — AMD's NVTX equivalent.
//!
//! `libroctx64.so` exposes `roctxRangePush` / `roctxRangePop` /
//! `roctxMarkA`. We wrap each `Step` dispatch in a scoped range so
//! `rocprof` / `Omnitrace` / `rocm-smi` traces show step boundaries
//! cleanly — same UX win as the NVTX ranges in rlx-cuda.
//!
//! Best-effort: if `libroctx64` isn't loadable (which is the common
//! case on hosts without a profiler attached), `RoctxRuntime::load()`
//! returns `None` and the wrapper is a no-op. Zero overhead in the
//! happy path.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{CString, c_char, c_int};
use std::sync::Arc;
use std::sync::OnceLock;

use libloading::Library;

type FnRoctxRangePush = unsafe extern "C" fn(*const c_char) -> c_int;
type FnRoctxRangePop = unsafe extern "C" fn() -> c_int;

pub struct RoctxRuntime {
    _lib: Library,
    pub range_push: FnRoctxRangePush,
    pub range_pop: FnRoctxRangePop,
}

unsafe impl Send for RoctxRuntime {}
unsafe impl Sync for RoctxRuntime {}

impl RoctxRuntime {
    pub fn load() -> Option<Arc<Self>> {
        unsafe {
            let lib = Library::new("libroctx64.so")
                .or_else(|_| Library::new("libroctx64.so.1"))
                .ok()?;
            macro_rules! sym {
                ($name:literal, $ty:ty) => {{
                    let s: libloading::Symbol<$ty> = lib.get($name).ok()?;
                    *s.into_raw()
                }};
            }
            let rt = RoctxRuntime {
                range_push: sym!(b"roctxRangePush", FnRoctxRangePush),
                range_pop: sym!(b"roctxRangePop", FnRoctxRangePop),
                _lib: lib,
            };
            Some(Arc::new(rt))
        }
    }
}

/// Process-wide rocTX runtime. None if libroctx64 isn't available.
pub fn roctx_runtime() -> Option<Arc<RoctxRuntime>> {
    static RUNTIME: OnceLock<Option<Arc<RoctxRuntime>>> = OnceLock::new();
    RUNTIME.get_or_init(RoctxRuntime::load).clone()
}

/// Scoped range guard. Drop calls `roctxRangePop`. Constructed via
/// `scoped_range(name)` — a no-op when libroctx64 isn't loaded.
pub struct ScopedRange {
    runtime: Option<Arc<RoctxRuntime>>,
}

impl Drop for ScopedRange {
    fn drop(&mut self) {
        if let Some(rt) = self.runtime.take() {
            unsafe {
                let _ = (rt.range_pop)();
            }
        }
    }
}

pub fn scoped_range(name: &str) -> ScopedRange {
    if let Some(rt) = roctx_runtime() {
        let cstr = CString::new(name).unwrap_or_else(|_| CString::new("rlx::?").unwrap());
        unsafe {
            let _ = (rt.range_push)(cstr.as_ptr());
        }
        ScopedRange { runtime: Some(rt) }
    } else {
        ScopedRange { runtime: None }
    }
}
