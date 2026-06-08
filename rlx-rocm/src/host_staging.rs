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

// RLX — versatile ML compiler + runtime.
//
// Pageable or pinned host staging for faster H2D/D2H on the ROCm run hot path.

use std::sync::Arc;

use crate::hip::{HipDeviceptr, HipError, HipRuntime};

/// Host-side f32 buffer used for input upload / output download.
pub enum F32HostSlot {
    Pageable(Vec<f32>),
    Pinned {
        rt: Arc<HipRuntime>,
        ptr: *mut f32,
        len: usize,
    },
}

impl F32HostSlot {
    pub fn new(rt: &Arc<HipRuntime>, len: usize, pinned: bool) -> Self {
        if pinned && let (Some(malloc), Some(_free)) = (rt.hip_host_malloc, rt.hip_host_free) {
            let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
            let bytes = len * std::mem::size_of::<f32>();
            let err = unsafe { malloc(&mut raw, bytes, 0) };
            if err.ok().is_ok() && !raw.is_null() {
                return Self::Pinned {
                    rt: Arc::clone(rt),
                    ptr: raw as *mut f32,
                    len,
                };
            }
        }
        Self::Pageable(vec![0.0f32; len])
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Pageable(v) => v.len(),
            Self::Pinned { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn copy_from_host(&mut self, data: &[f32]) {
        match self {
            Self::Pageable(v) => {
                debug_assert!(data.len() <= v.len());
                v[..data.len()].copy_from_slice(data);
            }
            Self::Pinned { ptr, len, .. } => {
                debug_assert!(data.len() <= *len);
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), *ptr, data.len());
                }
            }
        }
    }

    pub fn htod(&self, rt: &HipRuntime, dst: HipDeviceptr, len: usize) -> Result<(), HipError> {
        debug_assert!(len <= self.len());
        let bytes = len * std::mem::size_of::<f32>();
        match self {
            Self::Pageable(v) => unsafe {
                (rt.hip_memcpy_htod)(dst, v.as_ptr() as *const _, bytes).ok()
            },
            Self::Pinned { ptr, .. } => unsafe {
                (rt.hip_memcpy_htod)(dst, *ptr as *const _, bytes).ok()
            },
        }
    }

    pub fn dtoh(&mut self, rt: &HipRuntime, src: HipDeviceptr, len: usize) -> Result<(), HipError> {
        debug_assert!(len <= self.len());
        let bytes = len * std::mem::size_of::<f32>();
        match self {
            Self::Pageable(v) => unsafe {
                (rt.hip_memcpy_dtoh)(v.as_mut_ptr() as *mut _, src, bytes).ok()
            },
            Self::Pinned { ptr, .. } => unsafe {
                (rt.hip_memcpy_dtoh)(*ptr as *mut _, src, bytes).ok()
            },
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        match self {
            Self::Pageable(v) => v.as_slice(),
            Self::Pinned { ptr, len, .. } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
        }
    }

    pub fn copy_into(&self, dst: &mut [f32]) {
        let src = self.as_slice();
        debug_assert!(dst.len() <= src.len());
        dst.copy_from_slice(&src[..dst.len()]);
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }
}

impl Drop for F32HostSlot {
    fn drop(&mut self) {
        if let Self::Pinned { rt, ptr, .. } = self
            && let Some(free) = rt.hip_host_free
        {
            unsafe {
                let _ = free(*ptr as *mut _);
            }
        }
    }
}
