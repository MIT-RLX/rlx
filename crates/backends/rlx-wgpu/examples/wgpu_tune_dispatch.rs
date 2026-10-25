// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fill the GPU dispatch table for **this wgpu adapter** by measurement — the
//! wgpu twin of `rlx-cuda` / `rlx-metal` `tune_dispatch`.
//!
//! For f32 graphs the live decision is `F32` (`matmul_wide_vec4`) vs `CoopF32`
//! (cooperative-matrix 8×8) vs, on discrete Vulkan, `CoopF16Vk`. The cascade
//! prefers coop whenever it is eligible — but coop pads `m` to 32, so at decode
//! (`m = 1`) it computes 32 rows to keep one: MAX's tensor-core waste argument
//! (`uwgmma-flash-decoding.md`). Whether the vec4 GEMV wins there is a property
//! of the adapter, which is what this measures.
//!
//! Gates (same discipline as the other tuners): output within 1e-3 relative of
//! the default's; ≥ [`MIN_SPEEDUP`] on the tuned shape and ≥ [`MIN_HOLDOUT`] on
//! every held-out shape of its bucket. Eligibility is enforced by the backend —
//! an override it cannot run degrades to the default, which then measures as
//! ~1.0× and is never installed.
//!
//! Timing: wgpu exposes no per-pass GPU span here, so one graph holds [`REPS`]
//! independent matmuls against *distinct* weights. That amortizes submit +
//! readback over many kernels and keeps the weight working set far past the
//! SLC, so a decode GEMV is timed from DRAM, not from cache. Default and
//! candidate run **interleaved** (see `pair`), scored by the median per-sample
//! ratio, with a default-vs-default control per bucket.
//!
//! ```sh
//! cargo run --release -p rlx-wgpu --example wgpu_tune_dispatch            # writes the user cache
//! cargo run --release -p rlx-wgpu --example wgpu_tune_dispatch -- --dry-run
//! ```

use rlx_gpu_dispatch::dispatch::{
    self, Choice, GpuArch, OverrideRecord, WGPU_MATMUL_VARIANTS, WgpuMatmul, Workload,
};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};
use std::time::Instant;

const WARMUP: usize = 2;
const ITERS: usize = 9;
/// Wall-clock timing here is noisy: the first M4 Pro run measured the default
/// kernel against itself at 1.21×. Thresholds sit above that floor, so a
/// "win" cannot be the same kernel timed twice.
const MIN_SPEEDUP: f64 = 1.25;
const MIN_HOLDOUT: f64 = 1.10;
const TOL: f32 = 1e-3;
/// A bucket whose default-vs-default control strays further than this from
/// 1.0 is reported as contended and installs nothing.
const MAX_CONTROL_DRIFT: f64 = 0.10;
/// Total weight bytes per timed graph; `REPS` is derived from it.
const WEIGHT_BUDGET: usize = 384 << 20;
const REPS_MAX: usize = 16;

struct Domain {
    label: &'static str,
    tune: (usize, usize, usize),
    holdout: &'static [(usize, usize, usize)],
}

/// Shapes are 32-aligned in `n` and 8-aligned in `k` where coop is meant to be
/// eligible; holdouts stay inside the tune bucket (asserted).
const DOMAIN: &[Domain] = &[
    Domain {
        label: "decode, hidden 1024 → 3072",
        tune: (1, 1024, 3072),
        holdout: &[(1, 1536, 2560), (1, 1024, 3584)],
    },
    Domain {
        label: "decode, 3072 → 1024",
        tune: (1, 3072, 1024),
        holdout: &[(1, 2560, 1536), (1, 3584, 1024)],
    },
    Domain {
        label: "decode, fused qkv / gate-up",
        tune: (1, 1024, 4096),
        holdout: &[(1, 1024, 6144), (1, 1536, 5120)],
    },
    Domain {
        label: "decode, vocab head",
        tune: (1, 1024, 151936),
        holdout: &[(1, 1536, 151936), (1, 1024, 200064)],
    },
    Domain {
        label: "decode, hidden 4096",
        tune: (1, 4096, 4096),
        holdout: &[(1, 5120, 6144), (1, 4096, 7168)],
    },
    Domain {
        label: "tiny batch / spec verify",
        tune: (4, 1024, 3072),
        holdout: &[(5, 1536, 2560), (7, 1024, 3584)],
    },
    Domain {
        label: "small batch",
        tune: (16, 2048, 2048),
        holdout: &[(24, 3072, 2048), (31, 2048, 3584)],
    },
    Domain {
        label: "short prefill",
        tune: (128, 1024, 3072),
        holdout: &[(160, 1536, 2560), (192, 1024, 3584)],
    },
    Domain {
        label: "medium prefill",
        tune: (512, 2048, 2048),
        holdout: &[(768, 3072, 2048), (640, 2048, 3584)],
    },
];

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5)
        .collect()
}

fn reps(k: usize, n: usize) -> usize {
    (WEIGHT_BUDGET / (k * n * 4)).clamp(1, REPS_MAX)
}

/// `REPS` matmuls of one `x` against distinct params `w0..`.
fn graph(m: usize, k: usize, n: usize, r: usize) -> Graph {
    let mut g = Graph::new("tune_mm");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let outs = (0..r)
        .map(|i| {
            let w = g.param(format!("w{i}"), Shape::new(&[k, n], DType::F32));
            g.matmul(x, w, Shape::new(&[m, n], DType::F32))
        })
        .collect();
    g.set_outputs(outs);
    g
}

/// One arm of a pair: a graph compiled with `v` pinned (or the default).
struct Arm {
    c: rlx_runtime::CompiledGraph,
    r: usize,
}

fn arm(arch: &GpuArch, v: Option<WgpuMatmul>, (m, k, n): (usize, usize, usize)) -> Arm {
    dispatch::clear_overrides();
    if let Some(v) = v {
        dispatch::set_override(
            Workload::Matmul { m, k, n }.key(arch),
            Choice::WgpuMatmul(v),
        )
        .expect("legal override");
    }
    // The path is chosen at compile time, so the table may be cleared after.
    let r = reps(k, n);
    let mut c = Session::new(Device::Gpu).compile(graph(m, k, n, r));
    dispatch::clear_overrides();
    for i in 0..r {
        c.set_param(&format!("w{i}"), &fill(k * n, 0.3 + i as f32));
    }
    Arm { c, r }
}

impl Arm {
    fn time(&mut self, x: &[f32]) -> (Vec<f32>, f64) {
        let t = Instant::now();
        let mut o = self.c.run(&[("x", x)]);
        let ms = t.elapsed().as_secs_f64() * 1e3 / self.r as f64;
        (o.swap_remove(0), ms)
    }
}

/// `(a_out, b_out, a_ms, b_ms, speedup)` — `a` and `b` timed **interleaved**,
/// alternating which runs first, scored by the median per-sample `a/b` ratio.
/// Contention that comes and goes mid-run hits both halves of a pair alike;
/// timing each arm in its own block let a background job turn a same-kernel
/// comparison into a 1.21× "win" on the first sweep.
fn pair(
    arch: &GpuArch,
    a: Option<WgpuMatmul>,
    b: Option<WgpuMatmul>,
    shape: (usize, usize, usize),
) -> (Vec<f32>, Vec<f32>, f64, f64, f64) {
    let mut aa = arm(arch, a, shape);
    let mut bb = arm(arch, b, shape);
    let x = fill(shape.0 * shape.1, 0.7);
    for _ in 0..WARMUP {
        aa.time(&x);
        bb.time(&x);
    }
    let (mut am, mut bm, mut ratio) = (Vec::new(), Vec::new(), Vec::new());
    let (mut ao, mut bo) = (Vec::new(), Vec::new());
    for i in 0..ITERS {
        let (ra, rb) = if i % 2 == 0 {
            let ra = aa.time(&x);
            (ra, bb.time(&x))
        } else {
            let rb = bb.time(&x);
            (aa.time(&x), rb)
        };
        ratio.push(ra.1 / rb.1);
        am.push(ra.1);
        bm.push(rb.1);
        (ao, bo) = (ra.0, rb.0);
    }
    let med = |mut v: Vec<f64>| {
        v.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
        v[v.len() / 2]
    };
    (ao, bo, med(am), med(bm), med(ratio))
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let denom = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / denom
}

fn main() {
    if !is_available(Device::Gpu) {
        eprintln!("wgpu_tune_dispatch: no wgpu adapter");
        return;
    }
    let dry_run = std::env::args().any(|a| a == "--dry-run");
    let arch = rlx_wgpu::tuning::gpu_arch().clone(); // loads shipped + user cache
    // `CoopF16Vk` is the discrete-Vulkan path; elsewhere the backend refuses the
    // override and runs the default, so timing it would compare the default
    // against itself (that produced a bogus 1.09× "win" on Metal).
    let vulkan = arch.as_str().starts_with("wgpu-Vulkan");
    dispatch::clear_overrides();
    println!("wgpu matmul dispatch tuning on {}\n", arch.as_str());

    let mut winners: Vec<(Workload, OverrideRecord)> = Vec::new();
    for d in DOMAIN {
        let (m, k, n) = d.tune;
        let tw = Workload::Matmul { m, k, n };
        for &(a, b, c) in d.holdout {
            assert_eq!(
                Workload::Matmul { m: a, k: b, n: c }.bucket(),
                tw.bucket(),
                "{}: holdout {a}x{b}x{c} outside the tune bucket",
                d.label
            );
        }
        let (ref_out, _, def_ms, _, ctl) = pair(&arch, None, None, d.tune);
        let contended = (ctl - 1.0).abs() > MAX_CONTROL_DRIFT;
        println!(
            "{:<28} {m}x{k}x{n}  bucket {}  default {def_ms:.4} ms/mm  control {ctl:.3}x{}",
            d.label,
            tw.bucket().encode(),
            if contended {
                "  CONTENDED — nothing installed"
            } else {
                ""
            }
        );
        let mut best: Option<(WgpuMatmul, f64)> = None;
        for &v in WGPU_MATMUL_VARIANTS {
            // F16 / Bf16Packed / Coop16 are not table-selectable for f32 graphs.
            if !matches!(
                v,
                WgpuMatmul::F32 | WgpuMatmul::CoopF32 | WgpuMatmul::CoopF16Vk
            ) || (v == WgpuMatmul::CoopF16Vk && !vulkan)
            {
                continue;
            }
            let (_, out, _, ms, sp) = pair(&arch, None, Some(v), d.tune);
            let err = rel_err(&out, &ref_out);
            let note = if err > TOL {
                "DISAGREES — excluded"
            } else if sp >= MIN_SPEEDUP {
                "candidate"
            } else {
                ""
            };
            println!("    {v:<10?} {ms:.4} ms  {sp:>5.2}x  err {err:.1e}  {note}");
            if !contended && err <= TOL && sp >= MIN_SPEEDUP && best.is_none_or(|(_, b)| sp > b) {
                best = Some((v, sp));
            }
        }
        let Some((win, win_sp)) = best else {
            println!("    → keep default\n");
            continue;
        };
        let mut ok = true;
        for &h in d.holdout {
            let (bo, co, _, _, sp) = pair(&arch, None, Some(win), h);
            let err = rel_err(&co, &bo);
            let pass = sp >= MIN_HOLDOUT && err <= TOL;
            println!(
                "    holdout {}x{}x{}: {sp:.2}x  err {err:.1e}  {}",
                h.0,
                h.1,
                h.2,
                if pass { "ok" } else { "FAIL" }
            );
            ok &= pass;
        }
        if ok {
            let sp = win_sp as f32;
            println!("    → {win:?} {sp:.2}x\n");
            winners.push((
                tw,
                OverrideRecord {
                    choice: Choice::WgpuMatmul(win),
                    speedup: Some(sp),
                    holdouts: d.holdout.len() as u16,
                },
            ));
        } else {
            println!("    → REJECTED by holdouts\n");
        }
    }

    dispatch::clear_overrides();
    for (w, r) in &winners {
        dispatch::set_override_measured(w.key(&arch), *r).expect("legal override");
    }
    println!("{} of {} buckets improved.", winners.len(), DOMAIN.len());
    println!("\n# shipped-table rows (rlx_gpu_dispatch::shipped):");
    print!("{}", dispatch::save_overrides());
    if !dry_run {
        match rlx_wgpu::tuning::save_tuning_cache() {
            Some(p) => println!("\nwrote {}", p.display()),
            None => println!("\ncache NOT written"),
        }
    }
}
