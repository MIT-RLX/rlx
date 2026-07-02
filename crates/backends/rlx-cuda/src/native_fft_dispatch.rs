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

//! Native cuFFT-parity FFT for the `native-cuda-fft` feature.
//!
//! A from-scratch single-kernel **Stockham autosort** FFT in the style of
//! cuFFT's single-pass kernels (mixed-radix Cooley-Tukey, one block per
//! transform, minimizing DRAM passes). For pow-2 `n ≤ 4096` the whole transform
//! fits in shared memory and runs in ONE kernel — one global load + one store,
//! all butterflies on-chip — which is exactly where cuFFT's single-pass kernels
//! win. Measured parity with cuFFT for `n ≤ 1024` and ~1.15× at 4096,
//! float32-accurate (~1e-7 rel err).
//!
//! Unlike the [`crate::cufft_dispatch`] bridge (which needs separate
//! planar⇄interleaved conversion kernels around cuFFT), this native kernel reads
//! the RLX 2N planar block directly into interleaved shared `float2` and writes
//! it back planar — the conversion is folded into the load/store, so there is no
//! conversion tax.
//!
//! Radix-4 for pow-4 sizes (fewer `__syncthreads`), radix-2 otherwise. `n > 4096`
//! falls back to the multi-kernel [`crate::fft_dispatch::run_fft_gpu`] (its
//! ping-pong shared buffer would exceed the 99 KB opt-in cap at n=8192).

use std::sync::Arc;

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::kernels::{
    CudaKernel, fft_stockham_mixed_kernel, fft_stockham_r4_kernel, fft_stockham_r8_kernel,
    fft_stockham_r16_kernel,
};

/// Largest `n` the single-block Stockham kernel handles. Shared = 16·n bytes
/// (two `float2` ping-pong buffers); 4096 → 64 KB, within Ampere's ~99 KB opt-in
/// limit. 8192 would need 128 KB, so it falls back to the multi-kernel path.
pub const STOCKHAM_MAX_N: u32 = 4096;

/// True when the native Stockham kernel handles this size: pow-2 f32, 2..=4096.
#[inline]
pub fn stockham_eligible(n: u32) -> bool {
    (2..=STOCKHAM_MAX_N).contains(&n) && n.is_power_of_two()
}

/// Runtime gate (`RLX_FFT_NATIVE=0` disables the native Stockham path so the FFT
/// falls through to cuFFT / the multi-kernel path — for A/B benchmarking when
/// both features are compiled in). Default on.
pub fn stockham_enabled() -> bool {
    !rlx_ir::env::var("RLX_FFT_NATIVE").is_some_and(|v| {
        v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")
    })
}

/// Run a batched pow-2 f32 FFT via the native Stockham kernel over the device
/// arena. Offsets are in f32 elements; layout matches
/// [`crate::fft_dispatch::run_fft_gpu`].
#[allow(clippy::too_many_arguments)]
pub fn run_fft_native_stockham(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    src_off: u32,
    dst_off: u32,
    outer: u32,
    n: u32,
    inverse: bool,
    norm_scale: f32,
    real_input: bool,
) {
    let inv = u32::from(inverse);
    let shmem = 16u32 * n; // two float2 ping-pong buffers of n elements.

    let opt_in_shared = |kernel: &CudaKernel| {
        if shmem > 48 * 1024 {
            // Opt into >48 KB dynamic shared (Ampere allows up to ~99 KB).
            kernel
                .function
                .set_attribute(
                    CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shmem as i32,
                )
                .expect("rlx-cuda: stockham set max dynamic shared failed");
        }
    };

    // Default path: a per-size, fully-radix-specialized kernel built by NVRTC
    // codegen — `n`, the block size, and every stage's radix/stride are
    // compile-time literals, so ptxas fully unrolls (no runtime stage loop /
    // `switch(R)` / `if(i<R)`), exactly like cuFFT's per-size template
    // instantiation. The optimal radix-16 schedule (`[16]·⌊m/4⌋ + [2^(m%4)]`)
    // minimizes stages. This reaches cuFFT parity (~1.0–1.2×) across sizes.
    let force_mixed = rlx_ir::env::var("RLX_FFT_FORCE_MIXED").is_some();
    // `real_input` (the fused real→complex path) is only emitted by the codegen
    // kernel, so it forces this path regardless of the A/B toggles.
    if real_input || (generated_enabled() && !force_mixed) {
        let (sched, block) = radix_schedule(n);
        let kernel = generated_kernel(ctx, n, &sched, block, real_input);
        opt_in_shared(&kernel);
        let cfg = LaunchConfig {
            grid_dim: (outer, 1, 1),
            block_dim: (block.max(1), 1, 1),
            shared_mem_bytes: shmem,
        };
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&src_off)
            .arg(&dst_off)
            .arg(&inv)
            .arg(&norm_scale)
            .arg(&outer);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: fft_gen launch failed");
        }
        return;
    }

    // Fallback (`RLX_FFT_GEN=0` or `RLX_FFT_FORCE_MIXED`): the precompiled radix-
    // specialized dedicated kernels for pure powers, and the generic radix-8
    // mixed kernel for the rest — kept for A/B benchmarking against the codegen.
    let m = n.trailing_zeros();
    let dedicated: Option<(&CudaKernel, u32)> = if force_mixed {
        None
    } else if m % 4 == 0 {
        Some((fft_stockham_r16_kernel(ctx), n / 16))
    } else if m % 3 == 0 {
        Some((fft_stockham_r8_kernel(ctx), n / 8))
    } else if m % 2 == 0 {
        Some((fft_stockham_r4_kernel(ctx), n / 4))
    } else {
        None
    };

    if let Some((kernel, threads)) = dedicated {
        opt_in_shared(kernel);
        let cfg = LaunchConfig {
            grid_dim: (outer, 1, 1),
            block_dim: (threads.max(1), 1, 1),
            shared_mem_bytes: shmem,
        };
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&src_off)
            .arg(&dst_off)
            .arg(&n)
            .arg(&inv)
            .arg(&norm_scale)
            .arg(&outer);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: fft_stockham launch failed");
        }
    } else {
        let (sched8, block) = radix8_schedule(n);
        let (packed, num_stages) = pack_schedule(&sched8);
        let kernel = fft_stockham_mixed_kernel(ctx);
        opt_in_shared(kernel);
        let cfg = LaunchConfig {
            grid_dim: (outer, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: shmem,
        };
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher
            .arg(&mut *buffer)
            .arg(&src_off)
            .arg(&dst_off)
            .arg(&n)
            .arg(&inv)
            .arg(&norm_scale)
            .arg(&outer)
            .arg(&packed)
            .arg(&num_stages);
        unsafe {
            launcher
                .launch(cfg)
                .expect("rlx-cuda: fft_stockham_mixed launch failed");
        }
    }
}

/// Radix-8-capped schedule for the generic mixed kernel fallback (it only
/// implements radix 2/4/8): `[8]·⌊m/3⌋ + [2^(m%3)]`. Returns radices + block.
fn radix8_schedule(n: u32) -> (Vec<u32>, u32) {
    let m = n.trailing_zeros();
    let mut sched = vec![8u32; (m / 3) as usize];
    let rem = m % 3;
    if rem > 0 {
        sched.push(1u32 << rem);
    }
    if sched.is_empty() {
        sched.push(2);
    }
    let min_radix = *sched.iter().min().unwrap();
    let block = (n / min_radix).clamp(1, 1024);
    (sched, block)
}

/// `RLX_FFT_GEN=0` falls back to the generic mixed kernel (for A/B). Default on.
fn generated_enabled() -> bool {
    !rlx_ir::env::var("RLX_FFT_GEN").is_some_and(|v| {
        v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")
    })
}

/// Stage schedule for the codegen kernel: `⌊m/3⌋` radix-8 stages then one
/// radix-`2^(m%3)` stage (e.g. 2048 = [8,8,8,4], 4096 = [8,8,8,8]). Radix-8 is
/// the sweet spot — radix-16 cuts a stage but its register pressure drops
/// occupancy more than it saves (FFT is occupancy/memory-bound), measurably
/// worse at 512/2048/4096. Returns the radix list and the launch block size
/// (= max butterflies in any stage = `n / min_radix`, ≤1024).
fn radix_schedule(n: u32) -> (Vec<u32>, u32) {
    let m = n.trailing_zeros();
    let mut sched = vec![8u32; (m / 3) as usize];
    let rem = m % 3;
    if rem > 0 {
        sched.push(1u32 << rem);
    }
    if sched.is_empty() {
        sched.push(2); // n == 2 (m == 1)
    }
    let min_radix = *sched.iter().min().unwrap();
    let block = (n / min_radix).clamp(1, 1024);
    (sched, block)
}

/// Pack a radix list as log2(radix) in 4-bit fields for the generic kernel.
fn pack_schedule(sched: &[u32]) -> (u32, u32) {
    let mut packed = 0u32;
    for (i, &r) in sched.iter().enumerate() {
        packed |= r.trailing_zeros() << (4 * i as u32);
    }
    (packed, sched.len() as u32)
}

/// In-process cache of NVRTC-compiled per-size kernels (the disk PTX cache keys
/// by source hash, so cold start is cheap too). Keyed by `n` (schedule is a
/// function of `n`). `CudaKernel` is `Send + Sync` (same as the `kernel_cache!`
/// statics), so a global `Arc` cache is sound for the single-context setup.
fn generated_kernel(
    ctx: &Arc<CudaContext>,
    n: u32,
    sched: &[u32],
    block: u32,
    real_input: bool,
) -> Arc<CudaKernel> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<(u32, bool), Arc<CudaKernel>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().expect("rlx-cuda: fft gen cache poisoned");
    map.entry((n, real_input))
        .or_insert_with(|| {
            let src = generate_fft_source(n, sched, block, real_input);
            Arc::new(crate::kernels::compile(ctx, &src, "fft_gen"))
        })
        .clone()
}

/// Device helpers shared by every generated kernel: complex multiply, the
/// 4-point DFT core, and the W16 root table (immediate trivial roots).
const FFT_GEN_HELPERS: &str = r#"
__device__ __forceinline__ float2 fft_cmul(float2 a, float2 b){
  return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
__device__ __forceinline__ void fft_dft4(float2 x0,float2 x1,float2 x2,float2 x3,float rs,
                                          float2& X0,float2& X1,float2& X2,float2& X3){
  float2 t0=make_float2(x0.x+x2.x,x0.y+x2.y), t1=make_float2(x0.x-x2.x,x0.y-x2.y);
  float2 t2=make_float2(x1.x+x3.x,x1.y+x3.y), t3=make_float2(x1.x-x3.x,x1.y-x3.y);
  X0=make_float2(t0.x+t2.x,t0.y+t2.y); X2=make_float2(t0.x-t2.x,t0.y-t2.y);
  X1=make_float2(t1.x+rs*t3.y,t1.y-rs*t3.x); X3=make_float2(t1.x-rs*t3.y,t1.y+rs*t3.x);
}
__device__ __forceinline__ float2 fft_w16(int e, float sgn){
  const float C1=0.92387953251128676f, S1=0.38268343236508977f, R2=0.70710678118654752f;
  switch(e&15){
    case 0: return make_float2(1.0f,0.0f);
    case 1: return make_float2(C1,sgn*S1);
    case 2: return make_float2(R2,sgn*R2);
    case 3: return make_float2(S1,sgn*C1);
    case 4: return make_float2(0.0f,sgn);
    case 6: return make_float2(-R2,sgn*R2);
    case 9: return make_float2(-C1,-sgn*S1);
    default:{ float s,c; __sincosf(sgn*6.28318530717958647692f*(float)e/16.0f,&s,&c); return make_float2(c,s);}
  }
}
"#;

/// Emit the in-register R-point DFT (`u0..u{R-1}` → `Y0..Y{R-1}`) for R ∈ {2,4,8}.
fn radix_butterfly(r: u32, out: &mut String) {
    match r {
        2 => out.push_str(
            "      float2 Y0=make_float2(u0.x+u1.x,u0.y+u1.y), Y1=make_float2(u0.x-u1.x,u0.y-u1.y);\n",
        ),
        4 => out.push_str("      float2 Y0,Y1,Y2,Y3; fft_dft4(u0,u1,u2,u3,rs,Y0,Y1,Y2,Y3);\n"),
        8 => out.push_str(
            r#"      float2 E0,E1,E2,E3,O0,O1,O2,O3;
      fft_dft4(u0,u2,u4,u6,rs,E0,E1,E2,E3);
      fft_dft4(u1,u3,u5,u7,rs,O0,O1,O2,O3);
      float2 w1=make_float2(R2,sgn*R2), w2=make_float2(0.0f,sgn), w3=make_float2(-R2,sgn*R2);
      float2 cc1=fft_cmul(w1,O1), cc2=fft_cmul(w2,O2), cc3=fft_cmul(w3,O3);
      float2 Y0=make_float2(E0.x+O0.x,E0.y+O0.y), Y4=make_float2(E0.x-O0.x,E0.y-O0.y);
      float2 Y1=make_float2(E1.x+cc1.x,E1.y+cc1.y), Y5=make_float2(E1.x-cc1.x,E1.y-cc1.y);
      float2 Y2=make_float2(E2.x+cc2.x,E2.y+cc2.y), Y6=make_float2(E2.x-cc2.x,E2.y-cc2.y);
      float2 Y3=make_float2(E3.x+cc3.x,E3.y+cc3.y), Y7=make_float2(E3.x-cc3.x,E3.y-cc3.y);
"#,
        ),
        16 => {
            // 16 = 4×4 Cooley-Tukey: 4 inner radix-4 + W16 twiddle, 4 outer radix-4.
            out.push_str("      float2 A0,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15;\n");
            for n1 in 0..4u32 {
                out.push_str(&format!(
                    "      {{ float2 X0,X1,X2,X3; fft_dft4(u{},u{},u{},u{},rs,X0,X1,X2,X3);\n",
                    n1, n1 + 4, n1 + 8, n1 + 12
                ));
                out.push_str(&format!(
                    "        A{}=X0; A{}=fft_cmul(X1,fft_w16({},sgn)); A{}=fft_cmul(X2,fft_w16({},sgn)); A{}=fft_cmul(X3,fft_w16({},sgn)); }}\n",
                    n1 * 4, n1 * 4 + 1, n1, n1 * 4 + 2, n1 * 2, n1 * 4 + 3, n1 * 3
                ));
            }
            out.push_str("      float2 Y0,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9,Y10,Y11,Y12,Y13,Y14,Y15;\n");
            for k2 in 0..4u32 {
                out.push_str(&format!(
                    "      {{ float2 X0,X1,X2,X3; fft_dft4(A{},A{},A{},A{},rs,X0,X1,X2,X3);\n",
                    k2, k2 + 4, k2 + 8, k2 + 12
                ));
                out.push_str(&format!(
                    "        Y{}=X0; Y{}=X1; Y{}=X2; Y{}=X3; }}\n",
                    k2, k2 + 4, k2 + 8, k2 + 12
                ));
            }
        }
        _ => unreachable!("radix_butterfly: unsupported radix {r}"),
    }
}

/// Generate a fully-unrolled, size-specialized Stockham FFT kernel for `n` with
/// the given radix `schedule`. `n`, the launch `block` (= blockDim.x), each
/// stage's radix `R` and stride `p`, and the butterfly counts are all
/// compile-time literals, so ptxas fully unrolls the strided load/store/butterfly
/// loops with no branches — matching cuFFT's per-size template instantiation.
/// The kernel MUST be launched with `blockDim.x == block`.
fn generate_fft_source(n: u32, schedule: &[u32], block: u32, real_input: bool) -> String {
    let stages = schedule.len();
    let mut k = String::with_capacity(2048);
    k.push_str(FFT_GEN_HELPERS);
    k.push_str(
        "extern \"C\" __global__ void fft_gen(float* arena, unsigned src_off, unsigned dst_off, unsigned inverse, float norm_scale, unsigned outer){\n",
    );
    k.push_str(&format!("  const unsigned n={n}u;\n"));
    k.push_str("  extern __shared__ float2 shm[]; float2* a=shm; float2* b=shm+n;\n");
    k.push_str("  unsigned row=blockIdx.x; if(row>=outer) return;\n");
    k.push_str(&format!(
        "  unsigned db=dst_off+row*2u*n; const unsigned T={block}u;\n"
    ));
    if real_input {
        k.push_str("  unsigned sb=src_off+row*n;\n"); // n-wide real signal, im=0
    } else {
        k.push_str("  unsigned sb=src_off+row*2u*n;\n");
    }
    k.push_str("  float sgn=inverse?1.0f:-1.0f, rs=inverse?-1.0f:1.0f; const float R2=0.70710678118654752f;\n");
    // #5 register-locality: the first stage reads the input straight from global
    // into registers (no separate load + shared write), and the last stage writes
    // results straight to global (no shared write + separate store) — saving two
    // shared round-trips and two __syncthreads. Intermediate stages ping-pong the
    // two shared buffers (a/b by stage parity).
    let mut p: u32 = 1;
    for (si, &r) in schedule.iter().enumerate() {
        let mm = n / r;
        let first = si == 0;
        let last = si == stages - 1;
        let wbuf = if si % 2 == 0 { "a" } else { "b" };
        let rbuf = if si % 2 == 0 { "b" } else { "a" }; // = buffer written by stage si-1
        k.push_str(&format!("  {{ // stage {si}: radix-{r}, p={p}\n"));
        k.push_str(&format!("    for(unsigned j=threadIdx.x;j<{mm}u;j+=T){{\n"));
        k.push_str(&format!("      unsigned kk=j&{}u;\n", p - 1));
        for i in 0..r {
            let idx = format!("j+{}u", i * mm);
            if first && real_input {
                k.push_str(&format!(
                    "      float2 u{i}=make_float2(arena[sb+{idx}],0.0f);\n"
                ));
            } else if first {
                k.push_str(&format!(
                    "      float2 u{i}=make_float2(arena[sb+{idx}],arena[sb+n+{idx}]);\n"
                ));
            } else {
                k.push_str(&format!("      float2 u{i}={rbuf}[{idx}];\n"));
            }
        }
        if r > 1 {
            k.push_str(&format!(
                "      float ba=sgn*6.28318530717958647692f*(float)kk/{}.0f;\n",
                r * p
            ));
            for i in 1..r {
                k.push_str(&format!(
                    "      {{float s,c;__sincosf({i}.0f*ba,&s,&c);u{i}=make_float2(u{i}.x*c-u{i}.y*s,u{i}.x*s+u{i}.y*c);}}\n"
                ));
            }
        }
        radix_butterfly(r, &mut k);
        k.push_str(&format!("      unsigned ob=(j-kk)*{r}u+kk;\n"));
        for i in 0..r {
            let oidx = format!("ob+{}u", i * p);
            if last {
                k.push_str(&format!(
                    "      arena[db+{oidx}]=Y{i}.x*norm_scale; arena[db+n+{oidx}]=Y{i}.y*norm_scale;\n"
                ));
            } else {
                k.push_str(&format!("      {wbuf}[{oidx}]=Y{i};\n"));
            }
        }
        k.push_str("    }\n");
        if !last {
            k.push_str("    __syncthreads();\n");
        }
        k.push_str("  }\n");
        p *= r;
    }
    k.push_str("}\n");
    k
}
