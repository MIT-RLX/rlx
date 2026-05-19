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

//! MIOpen shim — forward 2D convolution path.
//!
//! Bounded scope: conv2d only. conv1d / conv3d stay on the custom
//! direct-conv kernels for now — MIOpen's nd-conv API differs more
//! from cuDNN's `set_convolutionnd_descriptor` than the 2D case
//! does. Subsequent commits add nd support if profiling shows it
//! pays off.
//!
//! Mirrors the cuDNN tier in `rlx-cuda`: per-call descriptor
//! creation + algorithm heuristic + workspace allocation +
//! `miopenConvolutionForward` + descriptor cleanup. Falls back to
//! the custom kernel cleanly on any setup error.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{c_int, c_void};
use std::ptr;
use std::sync::Arc;

use libloading::Library;

use crate::hip::HipStream;

pub type MiopenHandle = *mut c_void;
pub type MiopenTensorDescriptor = *mut c_void;
pub type MiopenConvDescriptor = *mut c_void;

/// `miopenConvAlgoPerf_t` — the find result. Layout matches AMD's
/// public C struct in `miopen.h`. Only the first field (`fwd_algo`)
/// matters for us; the rest is opaque padding.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MiopenConvAlgoPerf {
    pub fwd_algo: c_int,
    pub time: f32,
    pub memory: usize,
}

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MiopenError(pub c_int);

impl MiopenError {
    pub fn ok(self) -> Result<(), MiopenError> {
        if self.0 == 0 { Ok(()) } else { Err(self) }
    }
}

impl std::fmt::Display for MiopenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "miopenStatus({})", self.0)
    }
}

impl std::error::Error for MiopenError {}

/// `miopenDataType_t::miopenFloat = 1` (matches AMD enum).
const MIOPEN_FLOAT: c_int = 1;
/// `miopenConvolutionMode_t::miopenConvolution = 0`.
const MIOPEN_CONVOLUTION: c_int = 0;

// ── Function-pointer signatures ──────────────────────────────────────

type FnMiopenCreate = unsafe extern "C" fn(*mut MiopenHandle) -> MiopenError;
type FnMiopenDestroy = unsafe extern "C" fn(MiopenHandle) -> MiopenError;
type FnMiopenSetStream = unsafe extern "C" fn(MiopenHandle, HipStream) -> MiopenError;

type FnMiopenCreateTensorDesc = unsafe extern "C" fn(*mut MiopenTensorDescriptor) -> MiopenError;
type FnMiopenSet4dTensorDesc =
    unsafe extern "C" fn(MiopenTensorDescriptor, c_int, c_int, c_int, c_int, c_int) -> MiopenError;
type FnMiopenSetTensorDesc = unsafe extern "C" fn(
    MiopenTensorDescriptor,
    c_int,
    c_int,
    *const c_int,
    *const c_int,
) -> MiopenError;
type FnMiopenDestroyTensorDesc = unsafe extern "C" fn(MiopenTensorDescriptor) -> MiopenError;

type FnMiopenCreateConvDesc = unsafe extern "C" fn(*mut MiopenConvDescriptor) -> MiopenError;
type FnMiopenInitConvDesc = unsafe extern "C" fn(
    MiopenConvDescriptor,
    c_int, // mode
    c_int,
    c_int, // pad_h, pad_w
    c_int,
    c_int, // stride_h, stride_w
    c_int,
    c_int, // dilation_h, dilation_w
) -> MiopenError;
type FnMiopenInitConvNdDesc = unsafe extern "C" fn(
    MiopenConvDescriptor,
    c_int,        // spatial dims (= conv rank)
    *const c_int, // pads[]
    *const c_int, // strides[]
    *const c_int, // dilations[]
    c_int,        // mode
) -> MiopenError;
type FnMiopenSetConvGroupCount = unsafe extern "C" fn(MiopenConvDescriptor, c_int) -> MiopenError;
type FnMiopenDestroyConvDesc = unsafe extern "C" fn(MiopenConvDescriptor) -> MiopenError;

type FnMiopenFindConvFwdAlgo = unsafe extern "C" fn(
    MiopenHandle,
    MiopenTensorDescriptor,
    *const c_void,
    MiopenTensorDescriptor,
    *const c_void,
    MiopenConvDescriptor,
    MiopenTensorDescriptor,
    *mut c_void,
    c_int,                   // requested algo count
    *mut c_int,              // returned algo count
    *mut MiopenConvAlgoPerf, // perf results
    *mut c_void,
    usize, // workspace + size
    bool,  // exhaustive search
) -> MiopenError;

type FnMiopenGetConvFwdWorkspace = unsafe extern "C" fn(
    MiopenHandle,
    MiopenTensorDescriptor,
    MiopenTensorDescriptor,
    MiopenConvDescriptor,
    MiopenTensorDescriptor,
    *mut usize,
) -> MiopenError;

type FnMiopenConvFwd = unsafe extern "C" fn(
    MiopenHandle,
    *const c_void, // alpha
    MiopenTensorDescriptor,
    *const c_void, // x_desc, x
    MiopenTensorDescriptor,
    *const c_void, // w_desc, w
    MiopenConvDescriptor,
    c_int,         // algo
    *const c_void, // beta
    MiopenTensorDescriptor,
    *mut c_void, // y_desc, y
    *mut c_void,
    usize, // workspace + size
) -> MiopenError;

// ── Loaded runtime ───────────────────────────────────────────────────

pub struct MiopenRuntime {
    _lib: Library,
    pub create: FnMiopenCreate,
    pub destroy: FnMiopenDestroy,
    pub set_stream: FnMiopenSetStream,
    pub create_tensor_desc: FnMiopenCreateTensorDesc,
    pub set_4d_tensor_desc: FnMiopenSet4dTensorDesc,
    pub set_tensor_desc: FnMiopenSetTensorDesc,
    pub destroy_tensor_desc: FnMiopenDestroyTensorDesc,
    pub create_conv_desc: FnMiopenCreateConvDesc,
    pub init_conv_desc: FnMiopenInitConvDesc,
    pub init_conv_nd_desc: FnMiopenInitConvNdDesc,
    pub set_conv_group_count: FnMiopenSetConvGroupCount,
    pub destroy_conv_desc: FnMiopenDestroyConvDesc,
    pub find_conv_fwd_algo: FnMiopenFindConvFwdAlgo,
    pub get_conv_fwd_workspace: FnMiopenGetConvFwdWorkspace,
    pub conv_fwd: FnMiopenConvFwd,
}

unsafe impl Send for MiopenRuntime {}
unsafe impl Sync for MiopenRuntime {}

impl MiopenRuntime {
    pub fn load() -> Option<Arc<Self>> {
        unsafe {
            let lib = Library::new("libMIOpen.so")
                .or_else(|_| Library::new("libMIOpen.so.1"))
                .or_else(|_| Library::new("libMIOpen.so.0"))
                .ok()?;
            macro_rules! sym {
                ($name:literal, $ty:ty) => {{
                    let s: libloading::Symbol<$ty> = lib.get($name).ok()?;
                    *s.into_raw()
                }};
            }
            let rt = MiopenRuntime {
                create: sym!(b"miopenCreate", FnMiopenCreate),
                destroy: sym!(b"miopenDestroy", FnMiopenDestroy),
                set_stream: sym!(b"miopenSetStream", FnMiopenSetStream),
                create_tensor_desc: sym!(b"miopenCreateTensorDescriptor", FnMiopenCreateTensorDesc),
                set_4d_tensor_desc: sym!(b"miopenSet4dTensorDescriptor", FnMiopenSet4dTensorDesc),
                set_tensor_desc: sym!(b"miopenSetTensorDescriptor", FnMiopenSetTensorDesc),
                destroy_tensor_desc: sym!(
                    b"miopenDestroyTensorDescriptor",
                    FnMiopenDestroyTensorDesc
                ),
                create_conv_desc: sym!(
                    b"miopenCreateConvolutionDescriptor",
                    FnMiopenCreateConvDesc
                ),
                init_conv_desc: sym!(b"miopenInitConvolutionDescriptor", FnMiopenInitConvDesc),
                init_conv_nd_desc: sym!(
                    b"miopenInitConvolutionNdDescriptor",
                    FnMiopenInitConvNdDesc
                ),
                set_conv_group_count: sym!(
                    b"miopenSetConvolutionGroupCount",
                    FnMiopenSetConvGroupCount
                ),
                destroy_conv_desc: sym!(
                    b"miopenDestroyConvolutionDescriptor",
                    FnMiopenDestroyConvDesc
                ),
                find_conv_fwd_algo: sym!(
                    b"miopenFindConvolutionForwardAlgorithm",
                    FnMiopenFindConvFwdAlgo
                ),
                get_conv_fwd_workspace: sym!(
                    b"miopenConvolutionForwardGetWorkSpaceSize",
                    FnMiopenGetConvFwdWorkspace
                ),
                conv_fwd: sym!(b"miopenConvolutionForward", FnMiopenConvFwd),
                _lib: lib,
            };
            Some(Arc::new(rt))
        }
    }
}

/// MIOpen handle bound to a stream.
pub struct MiopenContext {
    pub runtime: Arc<MiopenRuntime>,
    pub handle: MiopenHandle,
}

unsafe impl Send for MiopenContext {}
unsafe impl Sync for MiopenContext {}

impl MiopenContext {
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // stream is opaque; we only pass it to FFI
    pub fn new(runtime: &Arc<MiopenRuntime>, stream: HipStream) -> Option<Self> {
        unsafe {
            let mut handle: MiopenHandle = ptr::null_mut();
            (runtime.create)(&mut handle).ok().ok()?;
            (runtime.set_stream)(handle, stream).ok().ok()?;
            Some(Self {
                runtime: runtime.clone(),
                handle,
            })
        }
    }
}

impl Drop for MiopenContext {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = (self.runtime.destroy)(self.handle);
            }
        }
    }
}

/// Forward conv3d via MIOpen's nd-descriptor path. NCDHW input/
/// output, 5-D filter (KCDHW), 3-D pads / strides / dilations.
/// Returns Err on any setup failure so the caller can fall through
/// to the custom direct-conv kernel.
///
/// # Safety
///
/// `arena_dev_ptr`, `workspace_dev_ptr`, and the offsets into them
/// must point to live HIP device memory of the right shape;
/// `miopen` must be a valid context bound to the active HIP stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn conv3d_forward(
    miopen: &MiopenContext,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    n: u32,
    c_in: u32,
    c_out: u32,
    d: u32,
    h: u32,
    w: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dd: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    in_off_f32: u32,
    w_off_f32: u32,
    out_off_f32: u32,
) -> Result<(), MiopenError> {
    let rt = &miopen.runtime;
    unsafe {
        let mut x_desc: MiopenTensorDescriptor = ptr::null_mut();
        let mut w_desc: MiopenTensorDescriptor = ptr::null_mut();
        let mut y_desc: MiopenTensorDescriptor = ptr::null_mut();
        let mut conv_desc: MiopenConvDescriptor = ptr::null_mut();

        let mut setup_and_run = || -> Result<(), MiopenError> {
            // 5-D row-major strides for NCDHW.
            let x_dims: [c_int; 5] = [n as _, c_in as _, d as _, h as _, w as _];
            let x_strides: [c_int; 5] = [
                (c_in * d * h * w) as _,
                (d * h * w) as _,
                (h * w) as _,
                w as _,
                1,
            ];
            let y_dims: [c_int; 5] = [n as _, c_out as _, d_out as _, h_out as _, w_out as _];
            let y_strides: [c_int; 5] = [
                (c_out * d_out * h_out * w_out) as _,
                (d_out * h_out * w_out) as _,
                (h_out * w_out) as _,
                w_out as _,
                1,
            ];
            let f_dims: [c_int; 5] = [
                c_out as _,
                (c_in / groups.max(1)) as _,
                kd as _,
                kh as _,
                kw as _,
            ];
            let f_strides: [c_int; 5] = [
                ((c_in / groups.max(1)) * kd * kh * kw) as _,
                (kd * kh * kw) as _,
                (kh * kw) as _,
                kw as _,
                1,
            ];
            let pads: [c_int; 3] = [pd as _, ph as _, pw as _];
            let strides: [c_int; 3] = [sd as _, sh as _, sw as _];
            let dilations: [c_int; 3] = [dd as _, dh as _, dw as _];

            (rt.create_tensor_desc)(&mut x_desc).ok()?;
            (rt.create_tensor_desc)(&mut y_desc).ok()?;
            (rt.create_tensor_desc)(&mut w_desc).ok()?;
            (rt.create_conv_desc)(&mut conv_desc).ok()?;

            (rt.set_tensor_desc)(x_desc, MIOPEN_FLOAT, 5, x_dims.as_ptr(), x_strides.as_ptr())
                .ok()?;
            (rt.set_tensor_desc)(y_desc, MIOPEN_FLOAT, 5, y_dims.as_ptr(), y_strides.as_ptr())
                .ok()?;
            (rt.set_tensor_desc)(w_desc, MIOPEN_FLOAT, 5, f_dims.as_ptr(), f_strides.as_ptr())
                .ok()?;
            (rt.init_conv_nd_desc)(
                conv_desc,
                3,
                pads.as_ptr(),
                strides.as_ptr(),
                dilations.as_ptr(),
                MIOPEN_CONVOLUTION,
            )
            .ok()?;
            if groups > 1 {
                (rt.set_conv_group_count)(conv_desc, groups as c_int).ok()?;
            }

            let mut needed: usize = 0;
            (rt.get_conv_fwd_workspace)(
                miopen.handle,
                w_desc,
                x_desc,
                conv_desc,
                y_desc,
                &mut needed,
            )
            .ok()?;
            if needed > workspace_size {
                return Err(MiopenError(-1));
            }

            let x_ptr = (arena_dev_ptr + (in_off_f32 as u64) * 4) as *const c_void;
            let w_ptr = (arena_dev_ptr + (w_off_f32 as u64) * 4) as *const c_void;
            let y_ptr = (arena_dev_ptr + (out_off_f32 as u64) * 4) as *mut c_void;
            let workspace_ptr = workspace_dev_ptr as *mut c_void;

            let mut returned: c_int = 0;
            let mut perf = std::mem::MaybeUninit::<MiopenConvAlgoPerf>::uninit();
            (rt.find_conv_fwd_algo)(
                miopen.handle,
                x_desc,
                x_ptr,
                w_desc,
                w_ptr,
                conv_desc,
                y_desc,
                y_ptr,
                1,
                &mut returned,
                perf.as_mut_ptr(),
                workspace_ptr,
                workspace_size,
                false,
            )
            .ok()?;
            if returned == 0 {
                return Err(MiopenError(-2));
            }
            let algo = perf.assume_init().fwd_algo;

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            (rt.conv_fwd)(
                miopen.handle,
                &alpha as *const _ as *const c_void,
                x_desc,
                x_ptr,
                w_desc,
                w_ptr,
                conv_desc,
                algo,
                &beta as *const _ as *const c_void,
                y_desc,
                y_ptr,
                workspace_ptr,
                workspace_size,
            )
            .ok()?;
            Ok(())
        };

        let result = setup_and_run();

        if !conv_desc.is_null() {
            let _ = (rt.destroy_conv_desc)(conv_desc);
        }
        if !y_desc.is_null() {
            let _ = (rt.destroy_tensor_desc)(y_desc);
        }
        if !w_desc.is_null() {
            let _ = (rt.destroy_tensor_desc)(w_desc);
        }
        if !x_desc.is_null() {
            let _ = (rt.destroy_tensor_desc)(x_desc);
        }
        result
    }
}

/// Forward conv2d with NCHW layout. Returns Ok on success, Err on
/// any setup or runtime failure so the caller can fall through to
/// the custom direct-conv kernel.
///
/// # Safety
///
/// `arena_dev_ptr`, `workspace_dev_ptr`, and the offsets into them
/// must point to live HIP device memory of the right shape;
/// `miopen` must be a valid context bound to the active HIP stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn conv2d_forward(
    miopen: &MiopenContext,
    workspace_dev_ptr: u64,
    workspace_size: usize,
    arena_dev_ptr: u64,
    n: u32,
    c_in: u32,
    c_out: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    groups: u32,
    in_off_f32: u32,
    w_off_f32: u32,
    out_off_f32: u32,
) -> Result<(), MiopenError> {
    let rt = &miopen.runtime;
    unsafe {
        let mut x_desc: MiopenTensorDescriptor = ptr::null_mut();
        let mut w_desc: MiopenTensorDescriptor = ptr::null_mut();
        let mut y_desc: MiopenTensorDescriptor = ptr::null_mut();
        let mut conv_desc: MiopenConvDescriptor = ptr::null_mut();

        // Wrap the body in a closure so cleanup runs on every exit path.
        let mut setup_and_run = || -> Result<(), MiopenError> {
            (rt.create_tensor_desc)(&mut x_desc).ok()?;
            (rt.set_4d_tensor_desc)(
                x_desc,
                MIOPEN_FLOAT,
                n as c_int,
                c_in as c_int,
                h as c_int,
                w as c_int,
            )
            .ok()?;
            (rt.create_tensor_desc)(&mut w_desc).ok()?;
            (rt.set_4d_tensor_desc)(
                w_desc,
                MIOPEN_FLOAT,
                c_out as c_int,
                (c_in / groups.max(1)) as c_int,
                kh as c_int,
                kw as c_int,
            )
            .ok()?;
            (rt.create_tensor_desc)(&mut y_desc).ok()?;
            (rt.set_4d_tensor_desc)(
                y_desc,
                MIOPEN_FLOAT,
                n as c_int,
                c_out as c_int,
                h_out as c_int,
                w_out as c_int,
            )
            .ok()?;
            (rt.create_conv_desc)(&mut conv_desc).ok()?;
            (rt.init_conv_desc)(
                conv_desc,
                MIOPEN_CONVOLUTION,
                ph as c_int,
                pw as c_int,
                sh as c_int,
                sw as c_int,
                /*dilation_h*/ 1,
                /*dilation_w*/ 1,
            )
            .ok()?;
            if groups > 1 {
                (rt.set_conv_group_count)(conv_desc, groups as c_int).ok()?;
            }

            // Sanity: ensure the workspace fits before kicking off the
            // find — the find call wants a workspace big enough for at
            // least one candidate algorithm.
            let mut needed: usize = 0;
            (rt.get_conv_fwd_workspace)(
                miopen.handle,
                w_desc,
                x_desc,
                conv_desc,
                y_desc,
                &mut needed,
            )
            .ok()?;
            if needed > workspace_size {
                return Err(MiopenError(-1)); // generic "doesn't fit" sentinel
            }

            let x_ptr = (arena_dev_ptr + (in_off_f32 as u64) * 4) as *const c_void;
            let w_ptr = (arena_dev_ptr + (w_off_f32 as u64) * 4) as *const c_void;
            let y_ptr = (arena_dev_ptr + (out_off_f32 as u64) * 4) as *mut c_void;
            let workspace_ptr = workspace_dev_ptr as *mut c_void;

            let mut returned: c_int = 0;
            let mut perf = std::mem::MaybeUninit::<MiopenConvAlgoPerf>::uninit();
            (rt.find_conv_fwd_algo)(
                miopen.handle,
                x_desc,
                x_ptr,
                w_desc,
                w_ptr,
                conv_desc,
                y_desc,
                y_ptr,
                1,
                &mut returned,
                perf.as_mut_ptr(),
                workspace_ptr,
                workspace_size,
                false, // not exhaustive — heuristic-only
            )
            .ok()?;
            if returned == 0 {
                return Err(MiopenError(-2));
            }
            let algo = perf.assume_init().fwd_algo;

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            (rt.conv_fwd)(
                miopen.handle,
                &alpha as *const _ as *const c_void,
                x_desc,
                x_ptr,
                w_desc,
                w_ptr,
                conv_desc,
                algo,
                &beta as *const _ as *const c_void,
                y_desc,
                y_ptr,
                workspace_ptr,
                workspace_size,
            )
            .ok()?;
            Ok(())
        };

        let result = setup_and_run();

        if !conv_desc.is_null() {
            let _ = (rt.destroy_conv_desc)(conv_desc);
        }
        if !y_desc.is_null() {
            let _ = (rt.destroy_tensor_desc)(y_desc);
        }
        if !w_desc.is_null() {
            let _ = (rt.destroy_tensor_desc)(w_desc);
        }
        if !x_desc.is_null() {
            let _ = (rt.destroy_tensor_desc)(x_desc);
        }

        result
    }
}
