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

//! Raw-GPU custom-kernel registry for `Op::Custom` on CUDA.
//!
//! Companion to the host-delegate path (`onnx_custom_host` → `rlx-gpu-host`,
//! which stages operands off-GPU and runs an `rlx-cpu` reference kernel). A
//! `CudaGpuKernel` instead compiles a CUDA-C kernel via NVRTC and launches it
//! **directly against the arena device buffer** — no D2H/H2D roundtrip. The CUDA
//! analogue of `rlx_metal`'s `MetalGpuKernel` / `rlx_wgpu`'s `WgpuGpuKernel`. A
//! registered GPU kernel takes precedence over a host one.
//!
//! ## Fixed launch signature
//!
//! The executor binds the whole arena as `float* arena` and passes the operand
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
//!     unsigned in3_off, unsigned in3_len,
//!     unsigned e0, unsigned e1, unsigned e2, unsigned e3);
//! ```
//!
//! Index the output at `arena[out_off + i]` and input `j` at
//! `arena[inJ_off + i]`, guarded by `i < out_len`. The trailing `e0..e3` are
//! optional runtime extras ([`CudaGpuKernel::extras`]); unused kernels may
//! ignore them (pass-through zeros).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::CudaContext;
use rlx_ir::Shape;

use crate::kernels::{CudaKernel, compile};

/// Max tensor inputs a raw CUDA custom kernel may take (matches the fixed launch
/// signature). Ops with more inputs fall through to the host-delegate path.
pub const MAX_INPUTS: usize = 4;

/// A raw-GPU CUDA custom kernel: NVRTC-compiled CUDA-C launched straight against
/// the arena buffer, no host roundtrip. Register under the same `name` used in
/// `Op::Custom` / `OpExtension::name`. See the module docs for the fixed launch
/// signature the CUDA-C must follow.
pub trait CudaGpuKernel: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;

    /// CUDA-C source containing an `extern "C" __global__` entry named by
    /// [`entry_point`](Self::entry_point) with the fixed signature (module docs).
    fn cuda_c(&self) -> &str;

    /// Kernel entry-point symbol (default `"rlx_custom"`).
    fn entry_point(&self) -> &str {
        "rlx_custom"
    }

    /// Threads per block (default 256). Grid is
    /// [`grid_blocks`](Self::grid_blocks)`(launch_elems, block_size)`.
    fn block_size(&self) -> u32 {
        256
    }

    /// Optional runtime extras (`e0..e3` in the fixed signature). Default zeros.
    /// Override to pass shape / active-window hints resolved at launch time.
    fn extras(&self, _attrs: &[u8], _out_shape: &Shape) -> [u32; 4] {
        [0, 0, 0, 0]
    }

    /// Elements that define the 1-D launch domain (default = `out_len`). Override
    /// when the kernel indexes a coarser domain (e.g. one thread / block per
    /// channel row).
    fn launch_elems(&self, out_len: u32, _extras: [u32; 4]) -> u32 {
        out_len
    }

    /// Number of CUDA blocks for `launch_elems` + `block_size`. Default is
    /// `ceil(launch_elems / block_size)` (one thread per element). Override to
    /// `launch_elems` when each block cooperates on one element (reductions).
    fn grid_blocks(&self, launch_elems: u32, block_size: u32) -> u32 {
        launch_elems.div_ceil(block_size.max(1)).max(1)
    }
}

struct Registry {
    kernels: Mutex<HashMap<String, Arc<dyn CudaGpuKernel>>>,
    compiled: Mutex<HashMap<String, Arc<CudaKernel>>>,
}

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| Registry {
        kernels: Mutex::new(HashMap::new()),
        compiled: Mutex::new(HashMap::new()),
    })
}

/// Register a raw-GPU CUDA custom kernel (takes precedence over a host-delegate
/// kernel of the same name).
pub fn register_cuda_gpu_kernel(k: Arc<dyn CudaGpuKernel>) {
    let name = k.name().to_string();
    let mut g = registry().kernels.lock().unwrap();
    if g.contains_key(&name) {
        eprintln!("rlx-cuda: CudaGpuKernel '{name}' was already registered — replacing");
    }
    g.insert(name, k);
}

/// Whether a raw-GPU CUDA kernel is registered for `name`.
pub fn has_gpu_kernel(name: &str) -> bool {
    registry().kernels.lock().unwrap().contains_key(name)
}

/// Look up a registered kernel by name.
pub fn lookup(name: &str) -> Option<Arc<dyn CudaGpuKernel>> {
    registry().kernels.lock().unwrap().get(name).cloned()
}

/// Get (NVRTC-compiling + caching on first use) the compiled kernel for `k`.
pub fn get_or_build(ctx: &Arc<CudaContext>, k: &dyn CudaGpuKernel) -> Arc<CudaKernel> {
    let name = k.name();
    if let Some(c) = registry().compiled.lock().unwrap().get(name) {
        return Arc::clone(c);
    }
    let built = Arc::new(compile(ctx, k.cuda_c(), k.entry_point()));
    registry()
        .compiled
        .lock()
        .unwrap()
        .insert(name.to_string(), Arc::clone(&built));
    built
}
