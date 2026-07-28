// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Safe-ish Rust wrapper over the Objective-C CoreML shim (csrc/
// coreml_shim.m). Apple platforms only.

use std::ffi::{CString, c_char, c_int, c_void};
use std::path::Path;

use crate::{CoremlError, Result};

#[allow(non_camel_case_types)]
type RlxCoremlModel = c_void;

unsafe extern "C" {
    fn rlx_coreml_load(
        mlpackage_path: *const c_char,
        compute_units: c_int,
        compiled_cache_path: *const c_char,
        err: *mut c_char,
        err_len: c_int,
    ) -> *mut RlxCoremlModel;

    fn rlx_coreml_predict(
        model: *mut RlxCoremlModel,
        n_inputs: c_int,
        in_names: *const *const c_char,
        in_data: *const *const c_void,
        in_shapes: *const *const i64,
        in_ranks: *const c_int,
        in_dtypes: *const c_int,
        n_outputs: c_int,
        out_names: *const *const c_char,
        out_data: *const *mut f32,
        out_len: *const c_int,
        err: *mut c_char,
        err_len: c_int,
    ) -> c_int;

    fn rlx_coreml_free(model: *mut RlxCoremlModel);

    fn rlx_coreml_compute_plan(
        model: *mut RlxCoremlModel,
        counts: *mut c_int,
        err: *mut c_char,
        err_len: c_int,
    ) -> c_int;
}

const ERR_LEN: usize = 1024;

fn read_err(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// An owned, loaded CoreML model. Freed on drop.
pub struct CoremlModel {
    raw: *mut RlxCoremlModel,
}

// The underlying MLModel is thread-safe for prediction; we serialise
// access through `&mut self` on the executable anyway.
unsafe impl Send for CoremlModel {}

impl CoremlModel {
    /// Compile + load the `.mlpackage` at `path` with the given compute
    /// unit policy code (see [`crate::ComputeUnits::code`]).
    ///
    /// `cache_path`, when set, is a `.mlmodelc` location reused across
    /// loads — a present cache skips the (expensive) compile.
    pub fn load(path: &Path, compute_units: i32, cache_path: Option<&Path>) -> Result<Self> {
        let cpath = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| CoremlError::Runtime("path contains NUL".into()))?;
        let ccache = match cache_path {
            Some(p) => CString::new(p.to_string_lossy().as_bytes())
                .map_err(|_| CoremlError::Runtime("cache path contains NUL".into()))?,
            None => CString::new("").unwrap(),
        };
        let mut err = [0 as c_char; ERR_LEN];
        // SAFETY: both C strings outlive the call; err buffer is ERR_LEN.
        let raw = unsafe {
            rlx_coreml_load(
                cpath.as_ptr(),
                compute_units,
                ccache.as_ptr(),
                err.as_mut_ptr(),
                ERR_LEN as c_int,
            )
        };
        if raw.is_null() {
            return Err(CoremlError::Runtime(format!(
                "CoreML load failed: {}",
                read_err(&err)
            )));
        }
        Ok(CoremlModel { raw })
    }

    /// Run one prediction. `inputs`/`outputs` are `(feature_name, shape,
    /// buffer)` tuples; output buffers are written in place and must be
    /// pre-sized to the expected element count. Input buffers may be f32
    /// or f16 (`half::f16` bit patterns) per `dtypes`.
    pub fn predict(
        &mut self,
        inputs: &[(CString, Vec<i64>, &[u8], i32)],
        outputs: &mut [(CString, &mut [f32])],
    ) -> Result<()> {
        let in_names: Vec<*const c_char> = inputs.iter().map(|(n, _, _, _)| n.as_ptr()).collect();
        let in_data: Vec<*const c_void> = inputs
            .iter()
            .map(|(_, _, d, _)| d.as_ptr() as *const c_void)
            .collect();
        let in_shapes: Vec<*const i64> = inputs.iter().map(|(_, s, _, _)| s.as_ptr()).collect();
        let in_ranks: Vec<c_int> = inputs.iter().map(|(_, s, _, _)| s.len() as c_int).collect();
        let in_dtypes: Vec<c_int> = inputs.iter().map(|(_, _, _, dt)| *dt).collect();

        let out_names: Vec<*const c_char> = outputs.iter().map(|(n, _)| n.as_ptr()).collect();
        let out_len: Vec<c_int> = outputs.iter().map(|(_, b)| b.len() as c_int).collect();
        let out_data: Vec<*mut f32> = outputs.iter_mut().map(|(_, b)| b.as_mut_ptr()).collect();

        let mut err = [0 as c_char; ERR_LEN];
        // SAFETY: every pointer vector outlives the call and matches the
        // declared counts; output buffers are sized via `out_len`.
        let rc = unsafe {
            rlx_coreml_predict(
                self.raw,
                inputs.len() as c_int,
                in_names.as_ptr(),
                in_data.as_ptr(),
                in_shapes.as_ptr(),
                in_ranks.as_ptr(),
                in_dtypes.as_ptr(),
                outputs.len() as c_int,
                out_names.as_ptr(),
                out_data.as_ptr(),
                out_len.as_ptr(),
                err.as_mut_ptr(),
                ERR_LEN as c_int,
            )
        };
        if rc != 0 {
            return Err(CoremlError::Runtime(format!(
                "CoreML predict failed ({rc}): {}",
                read_err(&err)
            )));
        }
        Ok(())
    }

    /// Query per-device op counts `{cpu, gpu, ane, unknown}` via
    /// MLComputePlan. Returns `None` when the OS is too old to support it.
    pub fn compute_plan(&mut self) -> Option<[i32; 4]> {
        let mut counts = [0 as c_int; 4];
        let mut err = [0 as c_char; ERR_LEN];
        // SAFETY: counts is a 4-int buffer; err is ERR_LEN bytes.
        let rc = unsafe {
            rlx_coreml_compute_plan(
                self.raw,
                counts.as_mut_ptr(),
                err.as_mut_ptr(),
                ERR_LEN as c_int,
            )
        };
        if rc != 0 {
            return None;
        }
        Some(counts)
    }
}

impl Drop for CoremlModel {
    fn drop(&mut self) {
        // SAFETY: raw came from rlx_coreml_load and is freed exactly once.
        unsafe { rlx_coreml_free(self.raw) };
    }
}
