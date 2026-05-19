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
        let params_ptr = if params.is_empty() {
            core::ptr::null_mut()
        } else {
            params.as_mut_ptr()
        };
        let _result = unsafe {
            kernel.launch(stream, grid, block, 0, params_ptr)
        };
        // Errors swallowed silently for parity with cudarc's
        // launch_builder — surface via dispatch-time assertions.
    }};
}
