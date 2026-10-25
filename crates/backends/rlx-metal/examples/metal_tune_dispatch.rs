// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fill the GPU dispatch table for **this Apple GPU** by measurement — the
//! Metal twin of `rlx-cuda/examples/tune_dispatch.rs`.
//!
//! On CUDA the tiled kernel sits behind cuBLAS, so its table only steers the
//! fallback. Metal has no vendor BLAS in front: `cost::pick_sgemm` chooses
//! between MPS and seven hand-written kernels for every f32 matmul, from a
//! cascade of thresholds tuned on whichever chip its author had. This measures
//! every *eligible* variant per shape bucket and keeps a winner only when:
//!
//! * **it agrees with the default** — relative max-abs error ≤ 1e-3 against the
//!   default variant's output (MPS and the simdgroup kernels reduce in different
//!   orders, so bit-exactness is not the bar here, unlike CUDA's tile sweep);
//! * **it generalizes** — ≥ [`MIN_SPEEDUP`] on the tuned shape *and* ≥
//!   [`MIN_HOLDOUT`] on every held-out shape of the same bucket (CAKE §6);
//! * **eligibility allows it** at the tune shape and every holdout
//!   (`MetalHwModel::sgemm_eligible`; `Simd4x4` at `m % 32 != 0` overruns C).
//!
//! Timing is **paired and interleaved** (see `pair`) with a default-vs-default
//! control per bucket; a bucket whose control drifts > 10% installs nothing.
//! Each sample is the sgemm thunk's own GPU span (`RLX_METAL_THUNK_PROFILE`), not the
//! run — a run's span includes the input upload, which swamps a decode GEMV.
//! MPSGraph whole-graph execution is disabled so the thunk path (what decode
//! runs) is what gets measured. Before every sample a 256 MB streaming pass
//! evicts the SLC so a 4 MB decode working set is not timed warm. Median of
//! [`ITERS`].
//!
//! ```sh
//! just throttle
//! cargo run --release -p rlx-metal --example metal_tune_dispatch
//! cargo run --release -p rlx-metal --example metal_tune_dispatch -- --dry-run   # no cache write
//! ```
//! Prints the validated rows in the shipped-table format
//! (`rlx_gpu_dispatch::shipped`) as well as writing the user cache.

#[cfg(target_os = "macos")]
fn main() {
    imp::main();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("metal_tune_dispatch: Metal is macOS-only");
}

#[cfg(target_os = "macos")]
mod imp {
    use rlx_gpu_dispatch::dispatch::{
        self, Choice, GpuArch, METAL_SGEMM_VARIANTS, MetalSgemm, OverrideRecord, Workload,
    };
    use rlx_ir::{DType, Graph, Shape};
    use rlx_metal::backend::MetalExecutable;
    use rlx_metal::cost::{SgemmVariant, hw_model};
    use rlx_metal::{install_runtime_config, runtime_config};

    const WARMUP: usize = 3;
    const ITERS: usize = 15;
    /// A winner must beat the default by this much on the tuned shape.
    const MIN_SPEEDUP: f64 = 1.05;
    /// …and must not lose on any held-out shape of the bucket.
    const MIN_HOLDOUT: f64 = 1.02;
    /// A bucket whose default-vs-default control pair strays further than this
    /// from 1.0 is reported as contended and installs nothing.
    const MAX_CONTROL_DRIFT: f64 = 0.10;
    /// Relative max-abs disagreement with the default's output.
    const TOL: f32 = 1e-3;

    struct Domain {
        label: &'static str,
        tune: (usize, usize, usize),
        holdout: &'static [(usize, usize, usize)],
    }

    /// Declared before tuning; each holdout is asserted into its tune bucket.
    /// Sizes follow the small LMs this machine actually runs (Qwen3/3.5 at
    /// hidden 1024–4096) plus the CUDA tuner's aspect-ratio cases.
    const DOMAIN: &[Domain] = &[
        Domain {
            label: "decode, hidden 1024 → 3072",
            tune: (1, 1024, 3072),
            holdout: &[(1, 1536, 2500), (1, 1024, 3500)],
        },
        Domain {
            label: "decode, 3072 → 1024",
            tune: (1, 3072, 1024),
            holdout: &[(1, 2500, 1536), (1, 3500, 1024)],
        },
        Domain {
            label: "decode, hidden 4096",
            tune: (1, 4096, 4096),
            holdout: &[(1, 5120, 6144), (1, 4096, 7000)],
        },
        Domain {
            label: "decode, mlp up",
            tune: (1, 2048, 8192),
            holdout: &[(1, 3000, 9000), (1, 2048, 15000)],
        },
        Domain {
            label: "decode, vocab head",
            tune: (1, 1024, 151936),
            holdout: &[(1, 1536, 151936), (1, 1024, 200000)],
        },
        Domain {
            label: "tiny batch / spec verify",
            tune: (4, 1024, 3072),
            holdout: &[(5, 1536, 2500), (7, 1024, 3500)],
        },
        Domain {
            label: "small batch",
            tune: (16, 2048, 2048),
            holdout: &[(24, 3072, 2048), (16, 2048, 3584)],
        },
        Domain {
            label: "short prefill",
            tune: (128, 1024, 3072),
            holdout: &[(192, 1536, 2560), (128, 1024, 3584)],
        },
        Domain {
            label: "medium prefill",
            tune: (512, 2048, 2048),
            holdout: &[(768, 3072, 2048), (640, 2048, 3584)],
        },
        Domain {
            label: "large prefill",
            tune: (2048, 2048, 2048),
            holdout: &[(3072, 2048, 2560), (2048, 3840, 2048)],
        },
        Domain {
            label: "tall, short k (dW)",
            tune: (4096, 512, 512),
            holdout: &[(5120, 512, 512), (6144, 512, 704)],
        },
        Domain {
            label: "fat k, small mn",
            tune: (192, 8192, 512),
            holdout: &[(192, 9216, 512), (128, 12288, 512)],
        },
    ];

    fn to_dispatch(v: SgemmVariant) -> MetalSgemm {
        match v {
            SgemmVariant::Mps => MetalSgemm::Mps,
            SgemmVariant::Simd4x4 => MetalSgemm::Simd4x4,
            SgemmVariant::Simd64 => MetalSgemm::Simd64,
            SgemmVariant::Simd64SplitK => MetalSgemm::Simd64SplitK,
            SgemmVariant::Simd => MetalSgemm::Simd,
            SgemmVariant::SimdPadded => MetalSgemm::SimdPadded,
            SgemmVariant::Tiled => MetalSgemm::Tiled,
            SgemmVariant::Naive => MetalSgemm::Naive,
        }
    }

    fn from_dispatch(v: MetalSgemm) -> SgemmVariant {
        match v {
            MetalSgemm::Mps => SgemmVariant::Mps,
            MetalSgemm::Simd4x4 => SgemmVariant::Simd4x4,
            MetalSgemm::Simd64 => SgemmVariant::Simd64,
            MetalSgemm::Simd64SplitK => SgemmVariant::Simd64SplitK,
            MetalSgemm::Simd => SgemmVariant::Simd,
            MetalSgemm::SimdPadded => SgemmVariant::SimdPadded,
            MetalSgemm::Tiled => SgemmVariant::Tiled,
            MetalSgemm::Naive => SgemmVariant::Naive,
        }
    }

    fn fill(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5)
            .collect()
    }

    fn mm_graph(m: usize, k: usize, n: usize) -> Graph {
        let mut g = Graph::new("tune_mm");
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let w = g.param("w", Shape::new(&[k, n], DType::F32));
        let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);
        g
    }

    /// Streams 256 MB through the GPU so the next sample starts with a cold SLC.
    struct Flusher {
        exe: MetalExecutable,
        x: Vec<f32>,
    }

    impl Flusher {
        fn new() -> Self {
            const N: usize = 64 << 20; // 64M f32 = 256 MB in, 256 MB out
            let mut g = Graph::new("slc_flush");
            let x = g.input("x", Shape::new(&[N], DType::F32));
            let y = g.activation(
                rlx_ir::op::Activation::Relu,
                x,
                Shape::new(&[N], DType::F32),
            );
            g.set_outputs(vec![y]);
            Self {
                exe: MetalExecutable::compile(g),
                x: vec![0.5; N],
            }
        }
        fn flush(&mut self) {
            let _ = self.exe.run(&[("x", &self.x)]);
        }
    }

    /// A paired, interleaved comparison of two variants on one shape.
    struct Pair {
        a_out: Vec<f32>,
        b_out: Vec<f32>,
        a_ms: f64,
        b_ms: f64,
        /// Median over samples of `a_ms / b_ms` — > 1 means `b` is faster.
        speedup: f64,
    }

    /// Time `a` and `b` **interleaved** on one executable: every sample runs both,
    /// alternating which goes first, and the score is the median per-sample
    /// ratio. Metal resolves the variant at encode time, so flipping the table
    /// between runs switches kernels. Contention that starts or stops mid-run
    /// hits both halves of a pair alike and cancels in the ratio — the first M4
    /// Pro sweep timed each variant in its own block, a background job came back
    /// halfway, and MPS "lost" 2.09x to a kernel that is really 0.93x of it.
    /// `None` when `b` is > `cutoff`× slower on the first sample.
    fn pair(
        arch: &GpuArch,
        flusher: &mut Flusher,
        a: SgemmVariant,
        b: SgemmVariant,
        (m, k, n): (usize, usize, usize),
        cutoff: Option<f64>,
    ) -> Option<Pair> {
        let key = Workload::Matmul { m, k, n }.key(arch);
        let select = |v: SgemmVariant| {
            dispatch::clear_overrides();
            dispatch::set_override(key.clone(), Choice::MetalSgemm(to_dispatch(v)))
                .expect("legal override");
            assert_eq!(
                hw_model().pick_sgemm(m, k, n),
                v,
                "override for {v:?} ignored"
            );
        };
        let mut exe = MetalExecutable::compile(mm_graph(m, k, n));
        exe.set_param("w", &fill(k * n, 0.3));
        let x = fill(m * k, 0.7);
        let mut run = |v: SgemmVariant, flusher: &mut Flusher| -> Option<(Vec<f32>, f64)> {
            select(v);
            flusher.flush();
            rlx_metal::thunk_profile::reset();
            let out = exe.run(&[("x", &x)]).remove(0);
            Some((out, rlx_metal::thunk_profile::total_ms()?))
        };
        for _ in 0..WARMUP {
            run(a, flusher)?;
            run(b, flusher)?;
        }
        let (mut am, mut bm, mut ratio) = (Vec::new(), Vec::new(), Vec::new());
        let (mut a_out, mut b_out) = (Vec::new(), Vec::new());
        for i in 0..ITERS {
            let (ra, rb) = if i % 2 == 0 {
                let ra = run(a, flusher)?;
                (ra, run(b, flusher)?)
            } else {
                let rb = run(b, flusher)?;
                (run(a, flusher)?, rb)
            };
            if i == 0
                && let Some(c) = cutoff
                && rb.1 > ra.1 * c
            {
                return None;
            }
            ratio.push(ra.1 / rb.1);
            am.push(ra.1);
            bm.push(rb.1);
            (a_out, b_out) = (ra.0, rb.0);
        }
        let med = |mut v: Vec<f64>| {
            v.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
            v[v.len() / 2]
        };
        Some(Pair {
            a_out,
            b_out,
            a_ms: med(am),
            b_ms: med(bm),
            speedup: med(ratio),
        })
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let denom = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / denom
    }

    pub fn main() {
        let dry_run = std::env::args().any(|a| a == "--dry-run");
        // Thunk path, one command buffer per thunk with its own GPU span.
        let mut cfg = runtime_config();
        cfg.disable_mpsgraph = true;
        install_runtime_config(cfg);
        rlx_ir::env::set("RLX_METAL_THUNK_PROFILE", "1");
        rlx_metal::thunk_profile::set_auto_print(false);

        // Load any existing cache now, then clear, so a persisted override can
        // never sneak back in mid-sweep as the "default" (the CUDA tuner's bug).
        let arch = rlx_metal::tuning::gpu_arch().clone();
        dispatch::clear_overrides();
        let hw = hw_model();
        println!(
            "Metal sgemm dispatch tuning on {} ({:?}); {ITERS} samples, SLC flushed per sample\n",
            arch.as_str(),
            hw.gpu_family
        );
        let mut flusher = Flusher::new();
        let mut winners: Vec<(Workload, OverrideRecord, &str)> = Vec::new();

        for d in DOMAIN {
            let (m, k, n) = d.tune;
            let tune_w = Workload::Matmul { m, k, n };
            for &(hm, hk, hn) in d.holdout {
                let hw_ = Workload::Matmul {
                    m: hm,
                    k: hk,
                    n: hn,
                };
                assert_eq!(
                    hw_.bucket(),
                    tune_w.bucket(),
                    "{}: holdout {hm}x{hk}x{hn} is not in the tune bucket",
                    d.label
                );
            }
            dispatch::clear_overrides();
            let def = hw.pick_sgemm_default(m, k, n);
            // Control: the default against itself. Its ratio is this bucket's
            // noise floor; a bucket whose control strays is not measured.
            let Some(ctl) = pair(&arch, &mut flusher, def, def, d.tune, None) else {
                println!(
                    "{:<28} default {def:?} produced no timing — skipped",
                    d.label
                );
                continue;
            };
            let ref_out = ctl.a_out.clone();
            let def_ms = ctl.a_ms;
            let contended = (ctl.speedup - 1.0).abs() > MAX_CONTROL_DRIFT;
            println!(
                "{:<28} {m}x{k}x{n}  bucket {}  default {def:?} {def_ms:.4} ms  control {:.3}x{}",
                d.label,
                tune_w.bucket().encode(),
                ctl.speedup,
                if contended {
                    "  CONTENDED — nothing installed"
                } else {
                    ""
                }
            );
            let mut best: Option<(SgemmVariant, f64)> = None;
            for &cand in METAL_SGEMM_VARIANTS {
                let v = from_dispatch(cand);
                if v == def {
                    continue;
                }
                // Eligible at the tune shape and on at least one holdout. Where a
                // bucket shape is ineligible the override is refused at runtime
                // and the default runs, so it needs no validation there.
                let eligible_holdouts = d
                    .holdout
                    .iter()
                    .filter(|h| hw.sgemm_eligible(v, h.0, h.1, h.2))
                    .count();
                if !hw.sgemm_eligible(v, m, k, n) || eligible_holdouts == 0 {
                    println!("    {v:<14?} ineligible (tune shape or every holdout)");
                    continue;
                }
                match pair(&arch, &mut flusher, def, v, d.tune, Some(4.0)) {
                    None => println!("    {v:<14?} >4x slower on first sample — dropped"),
                    Some(p) => {
                        let (out, ms) = (p.b_out, p.b_ms);
                        let err = rel_err(&out, &ref_out);
                        let sp = p.speedup;
                        let verdict = if err > TOL {
                            "DISAGREES — excluded"
                        } else if sp >= MIN_SPEEDUP {
                            "candidate"
                        } else {
                            ""
                        };
                        println!("    {v:<14?} {ms:.4} ms  {sp:>5.2}x  err {err:.1e}  {verdict}");
                        if !contended
                            && err <= TOL
                            && sp >= MIN_SPEEDUP
                            && best.is_none_or(|(_, b)| sp > b)
                        {
                            best = Some((v, sp));
                        }
                    }
                }
            }
            let Some((win, win_sp)) = best else {
                println!("    → keep default\n");
                continue;
            };
            // Held-out validation: the winner must hold on shapes it never saw.
            let mut ok = true;
            let mut checked = 0u16;
            for &h in d.holdout {
                if !hw.sgemm_eligible(win, h.0, h.1, h.2) {
                    println!(
                        "    holdout {}x{}x{}: {win:?} ineligible — default runs there",
                        h.0, h.1, h.2
                    );
                    continue;
                }
                checked += 1;
                dispatch::clear_overrides();
                let hdef = hw.pick_sgemm_default(h.0, h.1, h.2);
                match pair(&arch, &mut flusher, hdef, win, h, None) {
                    Some(p) => {
                        let sp = p.speedup;
                        let err = rel_err(&p.b_out, &p.a_out);
                        let pass = sp >= MIN_HOLDOUT && err <= TOL;
                        println!(
                            "    holdout {}x{}x{}: {hdef:?} {:.4} vs {win:?} {:.4}  {sp:.2}x  err {err:.1e}  {}",
                            h.0,
                            h.1,
                            h.2,
                            p.a_ms,
                            p.b_ms,
                            if pass { "ok" } else { "FAIL" }
                        );
                        ok &= pass;
                    }
                    _ => ok = false,
                }
            }
            if ok {
                let sp = win_sp as f32;
                println!("    → {win:?} {sp:.2}x, {checked} holdouts confirm\n");
                winners.push((
                    tune_w,
                    OverrideRecord {
                        choice: Choice::MetalSgemm(to_dispatch(win)),
                        speedup: Some(sp),
                        holdouts: checked,
                    },
                    d.label,
                ));
            } else {
                println!("    → REJECTED by holdouts, keep default\n");
            }
        }

        dispatch::clear_overrides();
        for (w, rec, _) in &winners {
            dispatch::set_override_measured(w.key(&arch), *rec).expect("legal override");
        }
        println!("{} of {} buckets improved.", winners.len(), DOMAIN.len());
        println!("\n# shipped-table rows (rlx_gpu_dispatch::shipped):");
        print!("{}", dispatch::save_overrides());
        if !dry_run {
            match rlx_metal::tuning::save_tuning_cache() {
                Some(p) => println!("\nwrote {}", p.display()),
                None => println!("\ncache NOT written (persistence disabled or failed)"),
            }
        }
    }
}
