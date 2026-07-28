// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Raw-GPU custom-kernel registry for `Op::Custom` on ROCm.
//!
//! The ROCm analogue of `rlx_cuda`'s `CudaGpuKernel` (and `rlx_metal`'s
//! `MetalGpuKernel` / `rlx_wgpu`'s `WgpuGpuKernel`): a `RocmGpuKernel` hipRTC-
//! compiles a HIP-C kernel and launches it **directly against the arena device
//! buffer** — no D2H/H2D roundtrip. HIP compiles the same source shape as CUDA,
//! so a `CudaGpuKernel`'s CUDA-C usually works verbatim as a `RocmGpuKernel`.
//!
//! ## Fixed launch signature
//!
//! The executor binds the whole arena as `float* arena` and passes operand
//! offsets as scalar args (element offsets, so `arena + off` indexes f32s). A
//! kernel must have **exactly** this `extern "C"` signature (up to 4 inputs;
//! unused offset/len args are 0, `n_inputs` says how many are real):
//!
//! ```c
//! extern "C" __global__ void <entry>(
//!     float* arena,
//!     unsigned out_off, unsigned out_len, unsigned n_inputs,
//!     unsigned in0_off, unsigned in0_len,
//!     unsigned in1_off, unsigned in1_len,
//!     unsigned in2_off, unsigned in2_len,
//!     unsigned in3_off, unsigned in3_len);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::device::RocmContext;
use crate::hip::HipKernel;
use crate::kernels::compile;

/// Max tensor inputs a raw ROCm custom kernel may take (matches the fixed launch
/// signature). Ops with more inputs fall through to the panic (register a CPU
/// host kernel or an `OpExtension::lower` rule instead).
pub const MAX_INPUTS: usize = 4;

/// A raw-GPU ROCm custom kernel: hipRTC-compiled HIP-C launched straight against
/// the arena buffer, no host roundtrip. Register under the same `name` used in
/// `Op::Custom` / `OpExtension::name`. See the module docs for the fixed launch
/// signature the HIP-C must follow.
pub trait RocmGpuKernel: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;

    /// HIP-C source containing an `extern "C" __global__` entry named by
    /// [`entry_point`](Self::entry_point) with the fixed signature (module docs).
    fn hip_c(&self) -> &str;

    /// Kernel entry-point symbol (default `"rlx_custom"`).
    fn entry_point(&self) -> &str {
        "rlx_custom"
    }

    /// Threads per block (default 256). Grid is `ceil(out_len / block_size)`.
    fn block_size(&self) -> u32 {
        256
    }
}

struct Registry {
    kernels: Mutex<HashMap<String, Arc<dyn RocmGpuKernel>>>,
    compiled: Mutex<HashMap<String, Arc<HipKernel>>>,
}

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| Registry {
        kernels: Mutex::new(HashMap::new()),
        compiled: Mutex::new(HashMap::new()),
    })
}

/// Register a raw-GPU ROCm custom kernel (takes precedence over the host path).
pub fn register_rocm_gpu_kernel(k: Arc<dyn RocmGpuKernel>) {
    let name = k.name().to_string();
    let mut g = registry().kernels.lock().unwrap();
    if g.contains_key(&name) {
        eprintln!("rlx-rocm: RocmGpuKernel '{name}' was already registered — replacing");
    }
    g.insert(name, k);
}

/// Whether a raw-GPU ROCm kernel is registered for `name`.
pub fn has_gpu_kernel(name: &str) -> bool {
    registry().kernels.lock().unwrap().contains_key(name)
}

/// Look up a registered kernel by name.
pub fn lookup(name: &str) -> Option<Arc<dyn RocmGpuKernel>> {
    registry().kernels.lock().unwrap().get(name).cloned()
}

/// Get (hipRTC-compiling + caching on first use) the compiled kernel for `k`.
pub fn get_or_build(ctx: &Arc<RocmContext>, k: &dyn RocmGpuKernel) -> Arc<HipKernel> {
    let name = k.name();
    if let Some(c) = registry().compiled.lock().unwrap().get(name) {
        return Arc::clone(c);
    }
    let built = Arc::new(compile(ctx, k.hip_c(), k.entry_point()));
    registry()
        .compiled
        .lock()
        .unwrap()
        .insert(name.to_string(), Arc::clone(&built));
    built
}
