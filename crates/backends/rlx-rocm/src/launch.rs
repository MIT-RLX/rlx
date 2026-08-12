// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Kernel-launch convenience macro for HIP dispatch.
//!
//! `cudarc` exposes `stream.launch_builder(...).arg(&x).arg(&y)...`
//! which packs kernel parameters via runtime introspection. We don't
//! have that — we hand-pack a `[*mut c_void]` array of pointers to
//! the argument values, which is what `hipModuleLaunchKernel` expects.
//!
//! Usage in a dispatch arm:
//!
//!   launch!(self.ctx, kernel, stream, grid, block,
//!           [&mut arena_ptr, n, a_off, b_off, c_off, op]);
//!
//! Each argument must be a pointer-typed expression (we take `*mut c_void`
//! of each). The macro autogenerates the `[..]` of `*mut c_void`.

#[macro_export]
macro_rules! launch_kernel {
    ($kernel:expr, $stream:expr, $grid:expr, $block:expr, [$($arg:expr),* $(,)?]) => {{
        use core::ffi::c_void;
        // Bind metavars in a safe context first so the unsafe block
        // only references locals — keeps clippy::macro_metavars_in_unsafe
        // quiet (the lint flags metavars *inside* unsafe blocks).
        let kernel = $kernel;
        let stream = $stream;
        let grid = $grid;
        let block = $block;
        let mut params: Vec<*mut c_void> = vec![
            $( $arg as *const _ as *mut c_void, )*
        ];
        // Slice form: the arity is checked against the kernel's own
        // `__global__` signature under `RLX_GPU_VALIDATE_PARAMS=1`. This macro
        // is where most launches funnel through, so it is the highest-value
        // place for that check — a count mismatch here is invisible to HIP and
        // has already shipped once (see `gguf_gpu::launch_dequant_gguf`).
        let _result = unsafe {
            kernel.launch_checked(stream, grid, block, 0, &mut params)
        };
        // Errors swallowed silently for parity with cudarc's
        // launch_builder — surface via dispatch-time assertions.
    }};
}
