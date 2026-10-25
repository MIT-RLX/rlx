// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Persistent-kernel prototype: does one long-running cooperative kernel that
//! walks a list of ops beat per-op launches and CUDA-graph replay on a
//! dispatch-bound decode tail?
//!
//! The workload is `LAYERS` repetitions of a decode-shaped elementwise/norm
//! chain over `[ROWS, H]`:
//!
//! ```text
//! r = x + y ; n = rms_norm(r, g) ; s = silu(n) ; x = s * n
//! ```
//!
//! Three ways to run the same `4·LAYERS` ops, all on one non-null stream:
//!
//! * **eager** — one kernel launch per op;
//! * **graph** — the eager launches captured once into a CUDA graph and replayed;
//! * **persistent** — ONE cooperative launch. Every block loops over an op list
//!   in device memory, executing its share of each op, with a software grid
//!   barrier between ops (co-residency guaranteed by the cooperative launch).
//!
//! Outputs are checked against eager before timing.
//!
//! ```sh
//! cargo run --release -p rlx-cuda --example persistent_vm
//! ROWS=1 H=4096 LAYERS=28 cargo run --release -p rlx-cuda --example persistent_vm
//! ```

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use rlx_ir::Tick;
use std::sync::Arc;

const SRC: &str = r#"
extern "C" {

__device__ __forceinline__ void op_add(float* m, int a, int b, int o, int n, int tid, int nthr) {
    for (int i = tid; i < n; i += nthr) m[o + i] = m[a + i] + m[b + i];
}
__device__ __forceinline__ void op_mul(float* m, int a, int b, int o, int n, int tid, int nthr) {
    for (int i = tid; i < n; i += nthr) m[o + i] = m[a + i] * m[b + i];
}
__device__ __forceinline__ void op_silu(float* m, int a, int o, int n, int tid, int nthr) {
    for (int i = tid; i < n; i += nthr) { float v = m[a + i]; m[o + i] = v / (1.0f + expf(-v)); }
}

// Block-per-row RMSNorm; rows are strided over `nblk` blocks.
__device__ void op_rms(float* m, int src, int g, int o, int rows, int h, float eps,
                       int blk, int nblk) {
    __shared__ float red[32];
    for (int r = blk; r < rows; r += nblk) {
        const float* x = m + src + r * h;
        float s = 0.f;
        for (int i = threadIdx.x; i < h; i += blockDim.x) s += x[i] * x[i];
        for (int k = 16; k > 0; k >>= 1) s += __shfl_xor_sync(0xffffffff, s, k);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
        __syncthreads();
        if (threadIdx.x < 32) {
            float t = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
            for (int k = 16; k > 0; k >>= 1) t += __shfl_xor_sync(0xffffffff, t, k);
            if (threadIdx.x == 0) red[0] = rsqrtf(t / h + eps);
        }
        __syncthreads();
        float inv = red[0];
        for (int i = threadIdx.x; i < h; i += blockDim.x) m[o + r * h + i] = x[i] * inv * m[g + i];
        __syncthreads();
    }
}

// ── one kernel per op (eager / graph) ──
__global__ void k_add(float* m, int a, int b, int o, int n) {
    op_add(m, a, b, o, n, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x);
}
__global__ void k_mul(float* m, int a, int b, int o, int n) {
    op_mul(m, a, b, o, n, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x);
}
__global__ void k_silu(float* m, int a, int o, int n) {
    op_silu(m, a, o, n, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x);
}
__global__ void k_rms(float* m, int src, int g, int o, int rows, int h, float eps) {
    op_rms(m, src, g, o, rows, h, eps, blockIdx.x, gridDim.x);
}

// ── persistent ──
// Sense-free generation barrier: arrive on `count`; the last arriver resets
// it and bumps `gen`; everyone else spins until `gen` moves.
__device__ void grid_barrier(unsigned* count, volatile unsigned* gen, unsigned nblk) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned g = *gen;
        __threadfence();
        if (atomicAdd(count, 1u) == nblk - 1) {
            *count = 0;
            __threadfence();
            atomicAdd((unsigned*)gen, 1u);
        } else {
            while (*gen == g) { }
        }
        __threadfence();
    }
    __syncthreads();
}

// prog: 8 ints per op = {kind, a, b, o, n_or_rows, h, eps_bits, _}
// kind 0=add 1=mul 2=silu 3=rms
__global__ void persistent(const int* prog, int nops, float* m, unsigned* bar) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthr = gridDim.x * blockDim.x;
    for (int p = 0; p < nops; ++p) {
        const int* in = prog + 8 * p;
        switch (in[0]) {
            case 0: op_add(m, in[1], in[2], in[3], in[4], tid, nthr); break;
            case 1: op_mul(m, in[1], in[2], in[3], in[4], tid, nthr); break;
            case 2: op_silu(m, in[1], in[3], in[4], tid, nthr); break;
            case 3: op_rms(m, in[1], in[2], in[3], in[4], in[5], __int_as_float(in[6]),
                           blockIdx.x, gridDim.x); break;
        }
        if (p + 1 < nops) grid_barrier(bar, bar + 1, gridDim.x);
    }
}

}
"#;

#[derive(Clone, Copy)]
enum Kind {
    Add = 0,
    Mul = 1,
    Silu = 2,
    Rms = 3,
}

#[derive(Clone, Copy)]
struct Instr {
    kind: Kind,
    a: i32,
    b: i32,
    o: i32,
    n: i32,
    h: i32,
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Kernels {
    add: CudaFunction,
    mul: CudaFunction,
    silu: CudaFunction,
    rms: CudaFunction,
    persistent: CudaFunction,
}

const EPS: f32 = 1e-6;
const BLOCK: u32 = 256;

fn launch_eager(
    stream: &Arc<CudaStream>,
    k: &Kernels,
    m: &CudaSlice<f32>,
    prog: &[Instr],
    rows: i32,
) {
    for ins in prog {
        let elem_cfg = LaunchConfig {
            grid_dim: ((ins.n as u32).div_ceil(BLOCK).max(1), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            match ins.kind {
                Kind::Add | Kind::Mul => {
                    let f = if matches!(ins.kind, Kind::Add) {
                        &k.add
                    } else {
                        &k.mul
                    };
                    stream
                        .launch_builder(f)
                        .arg(m)
                        .arg(&ins.a)
                        .arg(&ins.b)
                        .arg(&ins.o)
                        .arg(&ins.n)
                        .launch(elem_cfg)
                }
                Kind::Silu => stream
                    .launch_builder(&k.silu)
                    .arg(m)
                    .arg(&ins.a)
                    .arg(&ins.o)
                    .arg(&ins.n)
                    .launch(elem_cfg),
                Kind::Rms => stream
                    .launch_builder(&k.rms)
                    .arg(m)
                    .arg(&ins.a)
                    .arg(&ins.b)
                    .arg(&ins.o)
                    .arg(&rows)
                    .arg(&ins.h)
                    .arg(&EPS)
                    .launch(LaunchConfig {
                        grid_dim: (rows as u32, 1, 1),
                        block_dim: (BLOCK, 1, 1),
                        shared_mem_bytes: 0,
                    }),
            }
        }
        .expect("launch");
    }
}

fn time_us(iters: usize, stream: &Arc<CudaStream>, mut f: impl FnMut()) -> f64 {
    for _ in 0..10 {
        f();
    }
    stream.synchronize().unwrap();
    let t0 = Tick::now();
    for _ in 0..iters {
        f();
    }
    stream.synchronize().unwrap();
    Tick::now().elapsed_ns(t0) as f64 / 1e3 / iters as f64
}

fn main() {
    let (rows, h, layers, iters) = (
        env("ROWS", 1),
        env("H", 1024),
        env("LAYERS", 28),
        env("ITERS", 500),
    );
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.new_stream().unwrap();
    // Per-slice event tracking is cudarc's multi-stream safety net; it also
    // breaks stream capture and adds host work per launch. One stream here.
    unsafe { ctx.disable_event_tracking() };

    let ptx = cudarc::nvrtc::compile_ptx_with_opts(SRC, Default::default()).expect("NVRTC");
    let module = ctx.load_module(ptx).unwrap();
    let k = Kernels {
        add: module.load_function("k_add").unwrap(),
        mul: module.load_function("k_mul").unwrap(),
        silu: module.load_function("k_silu").unwrap(),
        rms: module.load_function("k_rms").unwrap(),
        persistent: module.load_function("persistent").unwrap(),
    };

    // Arena layout (floats): x, y, g, r, n, s.
    let n = rows * h;
    let (x, y, g, r, nn, s) = (0, n, 2 * n, 2 * n + h, 3 * n + h, 4 * n + h);
    let total = 5 * n + h;
    let mut prog = Vec::with_capacity(4 * layers);
    let (n32, h32) = (n as i32, h as i32);
    for _ in 0..layers {
        let i = |kind, a: usize, b: usize, o: usize| Instr {
            kind,
            a: a as i32,
            b: b as i32,
            o: o as i32,
            n: n32,
            h: h32,
        };
        prog.push(i(Kind::Add, x, y, r));
        prog.push(Instr {
            n: rows as i32,
            ..i(Kind::Rms, r, g, nn)
        });
        prog.push(i(Kind::Silu, nn, 0, s));
        prog.push(i(Kind::Mul, s, nn, x));
    }
    let encoded: Vec<i32> = prog
        .iter()
        .flat_map(|p| {
            [
                p.kind as i32,
                p.a,
                p.b,
                p.o,
                p.n,
                p.h,
                EPS.to_bits() as i32,
                0,
            ]
        })
        .collect();

    let init: Vec<f32> = (0..total)
        .map(|i| ((i as f32) * 0.013).sin() * 0.5 + 0.1)
        .collect();
    let mut arena = stream.clone_htod(&init).unwrap();
    let prog_dev = stream.clone_htod(&encoded).unwrap();
    let mut bar = stream.alloc_zeros::<u32>(2).unwrap();

    let sms = ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .unwrap() as u32;
    let per_sm = k
        .persistent
        .occupancy_max_active_blocks_per_multiprocessor(BLOCK, 0, None)
        .unwrap()
        .max(1);
    let blocks = env("BLOCKS", (sms * per_sm.min(2)) as usize) as u32;
    let pcfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let nops = prog.len() as i32;
    let launch_persistent =
        |stream: &Arc<CudaStream>, m: &mut CudaSlice<f32>, bar: &mut CudaSlice<u32>| unsafe {
            stream
                .launch_builder(&k.persistent)
                .arg(&prog_dev)
                .arg(&nops)
                .arg(m)
                .arg(bar)
                .launch_cooperative(pcfg)
                .expect("cooperative launch");
        };

    // ── correctness: one forward each from the same initial arena ──
    let rows32 = rows as i32;
    launch_eager(&stream, &k, &arena, &prog, rows32);
    let want = stream.clone_dtoh(&arena).unwrap();
    stream.memcpy_htod(&init, &mut arena).unwrap();
    launch_persistent(&stream, &mut arena, &mut bar);
    let got = stream.clone_dtoh(&arena).unwrap();
    let err = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
        .fold(0f32, f32::max);
    assert!(err < 1e-5, "persistent vs eager: max rel err {err}");

    // ── timing ──
    let eager = time_us(iters, &stream, || {
        launch_eager(&stream, &k, &arena, &prog, rows32)
    });
    stream
        .begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
        .unwrap();
    launch_eager(&stream, &k, &arena, &prog, rows32);
    let graph = stream
        .end_capture(cudarc::driver::sys::CUgraphInstantiate_flags(0))
        .unwrap()
        .expect("captured graph");
    graph.upload().unwrap();
    let replay = time_us(iters, &stream, || graph.launch().unwrap());
    let persistent = time_us(iters, &stream, || {
        launch_persistent(&stream, &mut arena, &mut bar)
    });

    println!(
        "rows={rows} h={h} layers={layers} ops={} blocks={blocks} ({sms} SMs) err={err:.1e} | \
         eager={eager:.1}us graph={replay:.1}us persistent={persistent:.1}us | \
         per-op: eager={:.2} graph={:.2} persistent={:.2} us",
        prog.len(),
        eager / prog.len() as f64,
        replay / prog.len() as f64,
        persistent / prog.len() as f64,
    );
}
