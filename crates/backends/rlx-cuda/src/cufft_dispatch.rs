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

//! cuFFT-backed FFT for the `cufft` feature.
//!
//! NVIDIA's cuFFT operates on **interleaved** `cufftComplex` (`float2`), while
//! the RLX arena stores each FFT row as a **planar** 2N real block
//! `[re[0..n) | im[0..n)]`. So the path is: pack planar → interleaved scratch,
//! run an in-place cuFFT C2C, then unpack interleaved → planar applying the FFT
//! norm scale (cuFFT is unnormalized — same convention as the native butterfly,
//! which also multiplies by `norm_scale` only at the end). cuFFT's
//! `CUFFT_FORWARD = -1` / `CUFFT_INVERSE = +1` match the native sign convention,
//! so results are numerically equivalent (up to fp rounding).
//!
//! Plans are cached per `(n, batch)`; one scratch buffer is grown as needed and
//! reused across calls. Toggle off at runtime with `RLX_FFT_CUFFT=0` to fall
//! back to the native kernels (handy for A/B benchmarking in one binary).

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::cufft::{CudaFft, result as cufft_result, sys as cufft_sys};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtrMut, LaunchConfig, PushKernelArg,
};

use crate::fft_dispatch::row_grid;
use crate::kernels::{dispatch_grid_1d, fft_pack_interleave_kernel, fft_unpack_planar_kernel};

/// Per-executor cuFFT state: cached plans + a reusable interleaved scratch.
#[derive(Default)]
pub struct CufftState {
    plans: HashMap<(i32, i32), CudaFft>,
    scratch: Option<CudaSlice<f32>>,
}

impl CufftState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure the interleaved scratch holds at least `total` f32 (= 2·outer·n).
    fn ensure_scratch(&mut self, stream: &Arc<CudaStream>, total: usize) {
        let have = self.scratch.as_ref().map_or(0, CudaSlice::len);
        if have < total {
            self.scratch = Some(
                stream
                    .alloc_zeros::<f32>(total)
                    .expect("rlx-cuda: cufft scratch alloc failed"),
            );
        }
    }

    /// Get (creating + caching if needed) the cuFFT plan handle for `(n, batch)`.
    fn plan_handle(
        &mut self,
        stream: &Arc<CudaStream>,
        n: i32,
        batch: i32,
    ) -> cufft_sys::cufftHandle {
        self.plans
            .entry((n, batch))
            .or_insert_with(|| {
                CudaFft::plan_1d(n, cufft_sys::cufftType::CUFFT_C2C, batch, stream.clone())
                    .expect("rlx-cuda: cufftPlan1d failed")
            })
            .handle()
    }
}

/// Decide whether to route a size-`n` FFT through cuFFT vs the native butterfly
/// kernels. Controlled by `RLX_FFT_CUFFT`:
/// - unset / `1` / `on` / `smart` → **smart** (default): cuFFT only for `n > 1024`,
///   where the native path falls back to a multi-kernel scheme (3 global
///   round-trips) that cuFFT beats even after the planar⇄interleaved conversion.
///   For `n ≤ 1024` the native single fused kernel is faster than cuFFT + the
///   conversion tax, so native is kept.
/// - `always` / `all` → cuFFT for every eligible size (benchmarking).
/// - `0` / `off` / `false` → never (always native).
pub fn cufft_should_use(n: u32) -> bool {
    match rlx_ir::env::var("RLX_FFT_CUFFT").as_deref() {
        Some("0") | Some("off") | Some("OFF") | Some("false") | Some("FALSE") => false,
        Some("always") | Some("all") | Some("ALWAYS") => true,
        _ => n > 1024,
    }
}

/// Run a batched pow-2 (or any-`n`) f32 C2C FFT via cuFFT over the device arena.
/// Offsets are in f32 elements. Layout matches [`crate::fft_dispatch::run_fft_gpu`].
#[allow(clippy::too_many_arguments)]
pub fn run_fft_cufft(
    _ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    state: &mut CufftState,
    buffer: &mut CudaSlice<f32>,
    src_off: u32,
    dst_off: u32,
    outer: u32,
    n: u32,
    inverse: bool,
    norm_scale: f32,
) {
    let total = outer as usize * n as usize * 2;
    state.ensure_scratch(stream, total);
    let handle = state.plan_handle(stream, n as i32, outer as i32);
    let scratch = state
        .scratch
        .as_mut()
        .expect("rlx-cuda: cufft scratch missing");

    let (row_y, row_z) = row_grid(outer);
    let (grid_x, block) = dispatch_grid_1d(n, 256);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, row_y, row_z),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    // 1. Planar [re|im] → interleaved scratch.
    {
        let kernel = fft_pack_interleave_kernel(_ctx);
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&mut *scratch)
            .arg(&src_off)
            .arg(&n)
            .arg(&outer);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: fft_pack_interleave launch failed");
        }
    }

    // 2. In-place cuFFT C2C on the scratch. cuFFT's CUFFT_FORWARD / CUFFT_INVERSE
    // are C `#define`s (not bound by cudarc): -1 forward, +1 inverse — matching
    // the native butterfly's sign convention. Scoped so the device-ptr borrow
    // guard drops before the unpack kernel reborrows `scratch`.
    {
        let direction: std::ffi::c_int = if inverse { 1 } else { -1 };
        let (sptr, _record) = scratch.device_ptr_mut(stream);
        unsafe {
            cufft_result::exec_c2c(
                handle,
                sptr as *mut cufft_sys::cufftComplex,
                sptr as *mut cufft_sys::cufftComplex,
                direction,
            )
            .expect("rlx-cuda: cufftExecC2C failed");
        }
    }

    // 3. Interleaved scratch → planar [re|im], applying the norm scale.
    {
        let kernel = fft_unpack_planar_kernel(_ctx);
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&mut *scratch)
            .arg(&dst_off)
            .arg(&n)
            .arg(&outer)
            .arg(&norm_scale);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: fft_unpack_planar launch failed");
        }
    }
}
