// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! GPU-resident batched LU + solve via a custom Metal kernel.
//!
//! ## What this is
//!
//! A scaffold that pairs the MSL kernel in
//! `cpp/kernels/batched_lu_solve.metal` with rlx-mlx's `MlxKernel`
//! registry. Flips on a real Apple-GPU dispatch path for
//! `Op::BatchedDenseSolve` instead of the MLX-CPU-stream fallback
//! used today (see `lower.rs`'s comment on `Op::BatchedDenseSolve`).
//!
//! ## Status: dispatch wired, no-pivot kernel only
//!
//! The Rust trait impl, the FFI surface to `mc::fast::metal_kernel`,
//! and the MSL kernel are all live. basic test in
//! `tests/batched_lu_solve.rs` confirms a small batched system runs
//! on the Apple GPU and matches a hand-computed reference.
//!
//! Remaining work before this becomes the default lowering for
//! `Op::BatchedDenseSolve`:
//!
//! 1. **Partial pivoting** — current MSL is pivot-free Doolittle.
//!    Works for diagonally-dominant matrices (most KCL-built MNAs)
//!    but fails on general dense systems. Pattern to add: per
//!    elimination step, SIMD argmax over column `k`, threadgroup-
//!    broadcast the pivot row, swap rows `k` and `pivot`
//!    cooperatively, then proceed. ~2 days.
//! 2. **Lowering switch in `lower.rs`** — currently
//!    `Op::BatchedDenseSolve` still routes to MLX-CPU-stream
//!    `ops::solve`. Flip it to look up `KERNEL_NAME` in the global
//!    registry and dispatch this kernel when n ≤ 90 (threadgroup-mem
//!    cap), falling back to the CPU solve otherwise. ~1 day.
//! 3. **Parity test against `rlx-cpu`'s `DenseSolveF64`** — cast
//!    f32 result to f64, expect ≤ 1e-5 relative drift on the matrices
//!    from `spike-divider-mna/examples/mna_f32_precision.rs`. ~1 day.
//! 4. **Throughput bench** at (B=1k, n=20), (B=10k, n=20), (B=1k,
//!    n=80) vs the MLX-CPU-stream baseline. Expected 20–80× win on
//!    the middle two cases. ~1 day.

#![cfg(target_os = "macos")]

use std::ffi::CString;
use std::sync::Arc;

use rlx_ir::Shape;

use crate::array::{Array, MlxError, check};
use crate::ffi::{self, MlxDtype, mlx_array_t};
use crate::op_registry::{MlxKernel, register_mlx_kernel};

/// MSL source for the batched LU+solve kernel. Loaded at compile time
/// so the source ships in the binary; the host passes it to MLX's
/// `fast::metal_kernel` which compiles it on first dispatch (cached
/// for subsequent calls with the same source + header).
const KERNEL_MSL: &str = include_str!("../cpp/kernels/batched_lu_solve.metal");

/// Canonical name under which lowering looks this kernel up. Picked
/// to match the rlx-ir `Op::BatchedDenseSolve` namespace so a future
/// generic "this op has a custom kernel" check can find it.
///
/// Underscores not dots — MLX feeds this directly into the generated
/// Metal function name (`custom_kernel_<name>`), and Metal identifiers
/// (like all C-family languages) don't allow dots.
pub const KERNEL_NAME: &str = "rlx_linalg_batched_dense_solve_metal_f32";

pub struct BatchedLuSolveMetal;

impl MlxKernel for BatchedLuSolveMetal {
    fn name(&self) -> &str {
        KERNEL_NAME
    }

    fn execute(
        &self,
        inputs: &[&Array],
        output_shape: &Shape,
        _attrs: &[u8],
    ) -> Result<Array, MlxError> {
        // Inputs by contract: [A: [B, n, n], b: [B, n]].
        if inputs.len() != 2 {
            return Err(MlxError(format!(
                "{KERNEL_NAME}: expected 2 inputs (A, b), got {}",
                inputs.len()
            )));
        }
        // Output shape contract: [B, n] f32.
        let out_dims = output_shape.dims();
        if out_dims.len() != 2 {
            return Err(MlxError(format!(
                "{KERNEL_NAME}: expected rank-2 output [B, n], got rank {}",
                out_dims.len()
            )));
        }
        let b_dim = out_dims[0].unwrap_static();
        let n_dim = out_dims[1].unwrap_static();
        // Threadgroup-memory cap: NMAX² + NMAX f32 ≤ 32 KB → NMAX ≤ 90.
        // Pad NMAX up to a power of two (with a floor of 8) so the row
        // stride is friendly to coalesced loads / SIMD writes.
        let nmax = next_pow2(n_dim).max(8);
        if nmax > 128 {
            return Err(MlxError(format!(
                "{KERNEL_NAME}: n={n_dim} exceeds NMAX cap of 128 \
                 (threadgroup memory bound at f32). Lowering should \
                 fall back to MLX-CPU `ops::solve` for n in this range."
            )));
        }
        let header = format!("#define NMAX {nmax}\n");

        let name_c = CString::new(KERNEL_NAME).unwrap();
        let source_c = CString::new(KERNEL_MSL).unwrap();
        let header_c = CString::new(header).unwrap();
        let in_a_c = CString::new("A").unwrap();
        let in_b_c = CString::new("b").unwrap();
        let out_c = CString::new("x").unwrap();

        let in_name_ptrs: [*const std::os::raw::c_char; 2] = [in_a_c.as_ptr(), in_b_c.as_ptr()];
        let in_array_ptrs: [*mut mlx_array_t; 2] = [inputs[0].ptr, inputs[1].ptr];
        let out_shape_i32: [std::os::raw::c_int; 2] =
            [b_dim as std::os::raw::c_int, n_dim as std::os::raw::c_int];

        let mut out_handle: *mut mlx_array_t = std::ptr::null_mut();
        let rc = unsafe {
            ffi::rlx_mlx_op_metal_kernel_dispatch(
                name_c.as_ptr(),
                source_c.as_ptr(),
                header_c.as_ptr(),
                in_name_ptrs.as_ptr(),
                /*n_inputs=*/ 2,
                out_c.as_ptr(),
                in_array_ptrs.as_ptr(),
                out_shape_i32.as_ptr(),
                /*output_rank=*/ 2,
                MlxDtype::F32,
                // MLX's `grid` is the *total* threads (it forwards to
                // MTLComputeCommandEncoder::dispatchThreads), not the
                // count of threadgroups. We want B threadgroups of n
                // threads each — so the grid is B*n threads total,
                // and the threadgroup size n carves them into groups.
                /*grid*/
                (b_dim * n_dim) as std::os::raw::c_int,
                1,
                1,
                /*tg*/ n_dim as std::os::raw::c_int,
                1,
                1,
                &mut out_handle,
            )
        };
        check(rc)?;
        Ok(Array::from_raw(out_handle))
    }
}

fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Register the kernel so a future lowering arm in `lower.rs` can pick
/// it up via `lookup_mlx_kernel(KERNEL_NAME)`. Currently called from
/// nowhere — the lowering path still routes `Op::BatchedDenseSolve`
/// through `ops::solve` (MLX-CPU-stream). To switch over, either
/// invoke this in a `ctor`-style one-time init, or add an explicit
/// `register_default_kernels()` entry point and call it from the
/// crate's first use.
pub fn register() {
    register_mlx_kernel(Arc::new(BatchedLuSolveMetal));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op_registry::lookup_mlx_kernel;

    #[test]
    fn registration_round_trips() {
        register();
        let k = lookup_mlx_kernel(KERNEL_NAME).expect("kernel must be findable after register()");
        assert_eq!(k.name(), KERNEL_NAME);
    }

    #[test]
    fn msl_source_present() {
        // Catches a `cargo publish` or refactor that drops the .metal
        // file from the package: include_str! would fail at compile
        // time, but a copy that ships the .rs without the .metal
        // would also be wrong, so spot-check the file's distinctive
        // contents. (No function-name probe — MLX body-only contract
        // means the source has no [[kernel]] declaration to grep for.)
        assert!(KERNEL_MSL.contains("threadgroup float Aloc"));
        assert!(KERNEL_MSL.contains("threadgroup_position_in_grid"));
        assert!(KERNEL_MSL.contains("Doolittle LU"));
    }
}
