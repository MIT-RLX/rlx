// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode-shape SDPA kernels launched directly on one device buffer, so the
//! timing is kernel time and not the per-run Q/K/V upload `bench_attention`
//! pays. Every kernel is also scored against an f64 host reference (scoring
//! kernels against each other hides which one drifted).
//!
//! Compares `attention` (tiled flash, head_dim <= 128), `attention_warp`
//! (one 32-lane group per row) and `attention_warp_split` (one block per row,
//! seq_k split across its groups) at several split counts.
//!
//! cargo run --release -p rlx-cuda --example bench_decode_attention

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use rlx_cuda::kernels::{attention_kernel, attention_warp_kernel, attention_warp_split_kernel};

#[derive(Clone, Copy)]
struct Case {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    /// 0 none · 1 causal · 3 sliding window · 4 additive bias
    mask_kind: u32,
    window: u32,
}

#[derive(Clone, Copy)]
enum Kernel {
    Flash,
    Warp,
    Split(u32),
    /// Whatever `attention_kv_split_groups` picks for the shape.
    Auto,
}

impl Kernel {
    fn name(self) -> String {
        match self {
            Kernel::Flash => "flash".into(),
            Kernel::Warp => "warp".into(),
            Kernel::Split(g) => format!("split×{g}"),
            Kernel::Auto => "auto".into(),
        }
    }
}

struct Layout {
    q: u32,
    k: u32,
    v: u32,
    out: u32,
    mask: u32,
    total: usize,
}

// BHSD: q [B,H,Sq,D], k/v [B,H,Sk,D], out like q, bias mask [B,Sq,Sk].
fn layout(c: Case) -> Layout {
    let (b, h, sq, sk, d) = (
        c.batch as usize,
        c.heads as usize,
        c.seq_q as usize,
        c.seq_k as usize,
        c.head_dim as usize,
    );
    let q = 0;
    let k = q + b * h * sq * d;
    let v = k + b * h * sk * d;
    let out = v + b * h * sk * d;
    let mask = out + b * h * sq * d;
    let total = mask + b * sq * sk;
    Layout {
        q: q as u32,
        k: k as u32,
        v: v as u32,
        out: out as u32,
        mask: mask as u32,
        total,
    }
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn reference(c: Case, l: &Layout, a: &[f32], scale: f64) -> Vec<f64> {
    let (b, h, sq, sk, d) = (
        c.batch as usize,
        c.heads as usize,
        c.seq_q as usize,
        c.seq_k as usize,
        c.head_dim as usize,
    );
    let mut out = vec![0f64; b * h * sq * d];
    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..sq {
                let q_pos = qi + (sk - sq);
                let qb = l.q as usize + ((bi * h + hi) * sq + qi) * d;
                let mut scores = vec![f64::NEG_INFINITY; sk];
                for (s, score) in scores.iter_mut().enumerate() {
                    let masked = match c.mask_kind {
                        1 => s > q_pos,
                        3 => s > q_pos || q_pos - s > c.window as usize,
                        _ => false,
                    };
                    if masked {
                        continue;
                    }
                    let kb = l.k as usize + ((bi * h + hi) * sk + s) * d;
                    let dot: f64 = (0..d).map(|e| a[qb + e] as f64 * a[kb + e] as f64).sum();
                    *score = dot * scale;
                    if c.mask_kind == 4 {
                        *score += a[l.mask as usize + (bi * sq + qi) * sk + s] as f64;
                    }
                }
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = scores.iter().map(|&x| (x - m).exp()).collect();
                let z: f64 = w.iter().sum();
                let ob = ((bi * h + hi) * sq + qi) * d;
                for s in 0..sk {
                    let vb = l.v as usize + ((bi * h + hi) * sk + s) * d;
                    for e in 0..d {
                        out[ob + e] += w[s] / z * a[vb + e] as f64;
                    }
                }
            }
        }
    }
    out
}

fn launch(
    stream: &Arc<CudaStream>,
    f: &CudaFunction,
    arena: &mut CudaSlice<f32>,
    c: Case,
    l: &Layout,
    kernel: Kernel,
    scale: f32,
) {
    let (h, sq, sk, d) = (c.heads, c.seq_q, c.seq_k, c.head_dim);
    let (q_bs, q_hs, q_ss) = (h * sq * d, sq * d, d);
    let (k_bs, k_hs, k_ss) = (h * sk * d, sk * d, d);
    // Bias mask [B, Sq, Sk]: per-batch, shared across heads.
    let (seq_q_stride, seq_k_stride, mask_bs, mask_hs) = (sk, 1u32, sq * sk, 0u32);
    let scale_bits = scale.to_bits();
    let softcap_bits = 0f32.to_bits();
    let rows = c.batch * h * sq;
    let kernel = match kernel {
        Kernel::Auto => match rlx_gpu_kernels::attention_kv_split_groups(rows, sk, d, None) {
            0 if d <= 128 => Kernel::Flash,
            0 => Kernel::Warp,
            g => Kernel::Split(g),
        },
        k => k,
    };
    let cfg = match kernel {
        Kernel::Auto => unreachable!(),
        Kernel::Flash => LaunchConfig {
            grid_dim: (sq.div_ceil(16), c.batch * h, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        },
        Kernel::Warp => LaunchConfig {
            grid_dim: (rows.div_ceil(4), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        },
        Kernel::Split(g) => LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (g * 32, 1, 1),
            shared_mem_bytes: rlx_gpu_kernels::attention_kv_split_smem_bytes(g, d),
        },
    };
    let mut lb = stream.launch_builder(f);
    lb.arg(arena)
        .arg(&c.batch)
        .arg(&h)
        .arg(&sq)
        .arg(&sk)
        .arg(&d)
        .arg(&l.q)
        .arg(&l.k)
        .arg(&l.v)
        .arg(&l.out)
        .arg(&l.mask)
        .arg(&c.mask_kind)
        .arg(&scale_bits)
        .arg(&c.window)
        .arg(&seq_q_stride)
        .arg(&seq_k_stride)
        .arg(&mask_bs)
        .arg(&mask_hs)
        .arg(&q_bs)
        .arg(&q_hs)
        .arg(&q_ss)
        .arg(&k_bs)
        .arg(&k_hs)
        .arg(&k_ss)
        .arg(&k_bs)
        .arg(&k_hs)
        .arg(&k_ss)
        .arg(&q_bs)
        .arg(&q_hs)
        .arg(&q_ss)
        .arg(&softcap_bits);
    unsafe { lb.launch(cfg) }.expect("attention launch failed");
}

fn run_case(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, c: Case, kernels: &[Kernel]) -> bool {
    let l = layout(c);
    let host = fill(l.total, (c.seq_k as u64) << 16 | c.head_dim as u64);
    let scale = 1.0 / (c.head_dim as f32).sqrt();
    let want = reference(c, &l, &host, scale as f64);
    let n_out = want.len();
    let mut ok = true;
    let mut line = format!(
        "  B={} H={:>2} Sq={} Sk={:>5} D={:>3} mask={}",
        c.batch, c.heads, c.seq_q, c.seq_k, c.head_dim, c.mask_kind
    );
    for &kernel in kernels {
        if matches!(kernel, Kernel::Flash) && c.head_dim > 128 {
            continue;
        }
        let rows = c.batch * c.heads * c.seq_q;
        let f = match kernel {
            Kernel::Auto => {
                match rlx_gpu_kernels::attention_kv_split_groups(rows, c.seq_k, c.head_dim, None) {
                    0 if c.head_dim <= 128 => &attention_kernel(ctx).function,
                    0 => &attention_warp_kernel(ctx).function,
                    _ => &attention_warp_split_kernel(ctx).function,
                }
            }
            Kernel::Flash => &attention_kernel(ctx).function,
            Kernel::Warp => &attention_warp_kernel(ctx).function,
            Kernel::Split(_) => &attention_warp_split_kernel(ctx).function,
        };
        let mut arena = stream.clone_htod(&host).expect("upload");
        launch(stream, f, &mut arena, c, &l, kernel, scale);
        let got = stream.clone_dtoh(&arena).expect("download");
        let o = l.out as usize;
        let err = (0..n_out)
            .map(|i| (got[o + i] as f64 - want[i]).abs())
            .fold(0f64, f64::max);
        if err.is_nan() || err > 2e-5 {
            ok = false;
        }

        let iters = 200;
        for _ in 0..10 {
            launch(stream, f, &mut arena, c, &l, kernel, scale);
        }
        stream.synchronize().expect("sync");
        let t0 = Instant::now();
        for _ in 0..iters {
            launch(stream, f, &mut arena, c, &l, kernel, scale);
        }
        stream.synchronize().expect("sync");
        let us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;
        line.push_str(&format!("  {}={us:>7.1}µs({err:.0e})", kernel.name()));
    }
    println!("{line}{}", if ok { "" } else { "  FAIL" });
    ok
}

fn main() {
    if !rlx_cuda::is_available() {
        println!("CUDA not available on this host — exiting.");
        return;
    }
    let ctx = CudaContext::new(0).expect("cuda context");
    let stream = ctx.default_stream();
    let kernels = [
        Kernel::Flash,
        Kernel::Warp,
        Kernel::Split(2),
        Kernel::Split(4),
        Kernel::Split(8),
        Kernel::Split(16),
        Kernel::Auto,
    ];
    let mut cases = Vec::new();
    // Qwen3-0.6B (16×128) and Qwen3.5-0.8B (10×256 after GQA expand) decode.
    for &(h, d) in &[(16u32, 128u32), (10, 256), (8, 64), (32, 128)] {
        for &sk in &[64u32, 320, 1024, 4096] {
            cases.push(Case {
                batch: 1,
                heads: h,
                seq_q: 1,
                seq_k: sk,
                head_dim: d,
                mask_kind: 0,
                window: 0,
            });
        }
    }
    // Masks across split boundaries, ragged head_dim, spec-verify rows, batch.
    cases.push(Case {
        batch: 1,
        heads: 10,
        seq_q: 4,
        seq_k: 1000,
        head_dim: 256,
        mask_kind: 1,
        window: 0,
    });
    cases.push(Case {
        batch: 1,
        heads: 8,
        seq_q: 1,
        seq_k: 2000,
        head_dim: 128,
        mask_kind: 3,
        window: 300,
    });
    cases.push(Case {
        batch: 2,
        heads: 4,
        seq_q: 2,
        seq_k: 777,
        head_dim: 200,
        mask_kind: 4,
        window: 0,
    });
    cases.push(Case {
        batch: 1,
        heads: 2,
        seq_q: 1,
        seq_k: 5,
        head_dim: 80,
        mask_kind: 0,
        window: 0,
    });
    cases.push(Case {
        batch: 1,
        heads: 4,
        seq_q: 1,
        seq_k: 3000,
        head_dim: 512,
        mask_kind: 1,
        window: 0,
    });
    cases.push(Case {
        batch: 4,
        heads: 16,
        seq_q: 1,
        seq_k: 1024,
        head_dim: 128,
        mask_kind: 0,
        window: 0,
    });
    cases.push(Case {
        batch: 1,
        heads: 16,
        seq_q: 32,
        seq_k: 512,
        head_dim: 128,
        mask_kind: 1,
        window: 0,
    });
    cases.push(Case {
        batch: 8,
        heads: 16,
        seq_q: 1,
        seq_k: 1024,
        head_dim: 128,
        mask_kind: 0,
        window: 0,
    });

    println!("decode SDPA kernels: time per launch (max |err| vs f64)");
    let failures = cases
        .iter()
        .filter(|&&c| !run_case(&ctx, &stream, c, &kernels))
        .count();
    if failures > 0 {
        println!("{failures} case(s) exceeded 2e-5 vs the f64 reference");
        std::process::exit(1);
    }
}
