//! A BF16 **activation** must round-trip and compute correctly on every backend.
//!
//! Low-precision tensors live in the arena in two different layouts — PACKED at
//! 2 B/elem or WIDENED to 4 B — and which one a backend gets is a property of
//! its `ArenaWidthPolicy`, not of the dtype. Every kernel that touches such a
//! tensor has to agree with the planner about which layout it is. When they
//! disagreed, an `AutoMixedBf16` graph produced NaN on CPU and Metal (and the
//! NaN surfaced at an innocent downstream node, because the mismatch either
//! overran the slot or decoded f32 words as packed bf16 pairs — every odd
//! element survived, every even one came back as mantissa junk).
//!
//! These tests pin the three things that must hold for a bf16 activation:
//!   1. casting to BF16 actually loses the low mantissa bits (the policy is real),
//!   2. an op that consumes one computes the same answer as f32 would, and
//!   3. writing one does not clobber its neighbour in the arena.
use rlx_ir::infer::GraphExt;
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

fn devices() -> Vec<Device> {
    // Every push below is cfg-gated, so a cpu-only feature combination never
    // mutates `v` — `just lint`'s feature pass sees an unused `mut` there while
    // `--all-targets` alone does not. Removing the `mut` would instead break
    // the metal / mlx / gpu builds. Same idiom as `cpu_fft.rs`,
    // `cpu_gru_parity.rs` and `host_fallback_never_nops.rs`.
    #[allow(unused_mut)]
    let mut v = vec![Device::Cpu];
    #[cfg(feature = "metal")]
    v.push(Device::Metal);
    #[cfg(feature = "mlx")]
    v.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    v.push(Device::Gpu);
    v
}

/// Host-side bf16 rounding, round-to-nearest-even — what a `Cast(F32 -> BF16)`
/// has to reproduce however the tensor is stored.
fn bf16(v: f32) -> f32 {
    let b = v.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 {
        return v;
    }
    f32::from_bits((b + (((b >> 16) & 1) + 0x7fff)) & 0xffff_0000)
}

/// The regression itself: a graph rewritten by `AutoMixedPrecision` with
/// `AutoMixedBf16` must stay finite and track the f32 answer.
///
/// This is what broke. The policy leaves bf16 tensors in compute (matmul) and
/// data-movement (narrow/concat/transpose) positions; the CPU/Metal planners
/// sized some of those slots at 2 B/elem while the kernels that read them
/// compute in f32, so the graph produced NaN — and it surfaced at an innocent
/// downstream node, not at the op that got the layout wrong.
#[test]
fn auto_mixed_bf16_graph_tracks_f32() {
    // Serialize GPU access: these build a `Session` on a real device, and
    // without the guard they race every other GPU test in this binary.
    let _gpu = common::serialize_gpu();
    use rlx_compile::precision::{AutoMixedPrecision, PrecisionPolicy};
    use rlx_fusion::pass::Pass;
    use rlx_ir::op::BinaryOp;

    let (m, k, n) = (32usize, 24usize, 16usize);
    let f = DType::F32;
    let mut g = Graph::new("amp");
    let x = g.input("x", Shape::new(&[m, k], f));
    let w1 = g.param("w1", Shape::new(&[k, n], f));
    let w2 = g.param("w2", Shape::new(&[n, n], f));
    let h = g.matmul(x, w1, Shape::new(&[m, n], f));
    let hs = g.activation(Activation::Silu, h, Shape::new(&[m, n], f));
    // Data movement over a compute result: narrow + concat + transpose.
    let half = Shape::new(&[m, n / 2], f);
    let lo = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 1,
            start: 0,
            len: n / 2,
        },
        vec![hs],
        half.clone(),
    );
    let hi = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 1,
            start: n / 2,
            len: n / 2,
        },
        vec![hs],
        half,
    );
    let swapped = g.add_node(
        rlx_ir::Op::Concat { axis: 1 },
        vec![hi, lo],
        Shape::new(&[m, n], f),
    );
    let h2 = g.matmul(swapped, w2, Shape::new(&[m, n], f));
    let out = g.binary(BinaryOp::Add, h2, hs, Shape::new(&[m, n], f));
    g.set_outputs(vec![out]);

    let xv: Vec<f32> = (0..m * k).map(|i| 0.05 * ((i % 11) as f32 - 5.0)).collect();
    let w1v: Vec<f32> = (0..k * n).map(|i| 0.03 * ((i % 7) as f32 - 3.0)).collect();
    let w2v: Vec<f32> = (0..n * n).map(|i| 0.04 * ((i % 5) as f32 - 2.0)).collect();
    let amp = AutoMixedPrecision::new(PrecisionPolicy::AutoMixedBf16).run(g.clone());
    assert!(
        amp.nodes()
            .iter()
            .any(|nd| nd.shape.dtype() == DType::BF16 && !matches!(nd.op, rlx_ir::Op::Param { .. })),
        "the policy produced no bf16 activation — nothing under test"
    );

    for d in devices() {
        let run = |graph: Graph| {
            let mut s = Session::new(d).compile(graph);
            s.set_param("w1", &w1v);
            s.set_param("w2", &w2v);
            s.run(&[("x", &xv[..])])[0].clone()
        };
        let base = run(g.clone());
        let got = run(amp.clone());
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{d:?}: AutoMixedBf16 produced non-finite values"
        );
        let err = base
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = base.iter().fold(0f32, |a, b| a.max(b.abs())).max(1e-6);
        assert!(
            err / scale < 0.02,
            "{d:?}: AutoMixedBf16 diverged from f32 by {err} (scale {scale})"
        );
    }
}

/// A bf16 activation that is *consumed* stays finite and stays close to the f32
/// answer. It is deliberately NOT asserted to be bit-identical to a host bf16
/// round-trip: when the cast fuses into its consumer the chain keeps the region's
/// f32 working precision, which is a fidelity choice, not a layout bug.
#[test]
fn consumed_bf16_activation_tracks_f32() {
    // Serialize GPU access: these build a `Session` on a real device, and
    // without the guard they race every other GPU test in this binary.
    let _gpu = common::serialize_gpu();
    let n = 64usize;
    let f = DType::F32;
    let mut g = Graph::new("consume");
    let x = g.input("x", Shape::new(&[n], f));
    let xb = g.cast(x, DType::BF16);
    let act = g.activation(Activation::Relu, xb, Shape::new(&[n], DType::BF16));
    let back = g.cast(act, DType::F32);
    g.set_outputs(vec![back]);

    let xv: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.0013717).collect();
    for d in devices() {
        let mut s = Session::new(d).compile(g.clone());
        let got = s.run(&[("x", &xv[..])])[0].clone();
        assert!(got.iter().all(|v| v.is_finite()), "{d:?}: non-finite");
        let err = xv
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        // One bf16 ulp near 1.0 is 2^-8; allow a couple.
        assert!(err < 0.01, "{d:?}: bf16 activation diverged by {err}");
    }
}

#[test]
fn matmul_over_bf16_activations_matches_f32() {
    // Serialize GPU access: these build a `Session` on a real device, and
    // without the guard they race every other GPU test in this binary.
    let _gpu = common::serialize_gpu();
    let (m, k, n) = (64usize, 16usize, 24usize);
    let f = DType::F32;
    let mut g = Graph::new("mm");
    let x = g.input("x", Shape::new(&[m, k], f));
    let w = g.param("w", Shape::new(&[k, n], f));
    // BOTH operands are cast activations — the case that used to take the
    // packed dequant-on-the-fly GEMM and read widened f32 as bf16 bits.
    let xb = g.cast(x, DType::BF16);
    let wb = g.cast(w, DType::BF16);
    let mm = g.matmul(xb, wb, Shape::new(&[m, n], DType::BF16));
    let out = g.cast(mm, DType::F32);
    g.set_outputs(vec![out]);

    let xv: Vec<f32> = (0..m * k).map(|i| 0.01 * ((i % 7) as f32 - 3.0)).collect();
    let wv: Vec<f32> = (0..k * n).map(|i| 0.02 * ((i % 5) as f32 - 2.0)).collect();
    // Reference: the same product with both operands rounded to bf16.
    let mut want = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for t in 0..k {
                acc += bf16(xv[i * k + t]) * bf16(wv[t * n + j]);
            }
            want[i * n + j] = acc;
        }
    }
    for d in devices() {
        let mut s = Session::new(d).compile(g.clone());
        s.set_param("w", &wv);
        let got = s.run(&[("x", &xv[..])])[0].clone();
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{d:?}: non-finite matmul"
        );
        let err = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 2e-3, "{d:?}: bf16 matmul off by {err}");
    }
}

#[test]
fn bf16_output_does_not_clobber_its_neighbour() {
    // Serialize GPU access: these build a `Session` on a real device, and
    // without the guard they race every other GPU test in this binary.
    let _gpu = common::serialize_gpu();
    let (m, k, n, c) = (64usize, 16usize, 24usize, 512usize);
    let f = DType::F32;
    let mut g = Graph::new("canary");
    let x = g.input("x", Shape::new(&[m, k], f));
    let w = g.param("w", Shape::new(&[k, n], f));
    let can = g.input("can", Shape::new(&[c], f));
    // Produced before the bf16 op and read after it, so it stays live across it.
    let c0 = g.activation(Activation::Relu, can, Shape::new(&[c], f));
    let xb = g.cast(x, DType::BF16);
    let wb = g.cast(w, DType::BF16);
    let mm = g.matmul(xb, wb, Shape::new(&[m, n], DType::BF16));
    let mf = g.cast(mm, DType::F32);
    let keep = g.activation(Activation::Relu, c0, Shape::new(&[c], f));
    g.set_outputs(vec![keep, mf]);

    let xv: Vec<f32> = (0..m * k).map(|i| 0.01 * ((i % 7) as f32 - 3.0)).collect();
    let wv: Vec<f32> = (0..k * n).map(|i| 0.02 * ((i % 5) as f32 - 2.0)).collect();
    let cv: Vec<f32> = (0..c).map(|i| 1.0 + i as f32).collect();
    for d in devices() {
        let mut s = Session::new(d).compile(g.clone());
        s.set_param("w", &wv);
        let o = s.run(&[("x", &xv[..]), ("can", &cv[..])]);
        let bad = cv
            .iter()
            .zip(&o[0])
            .filter(|(a, b)| (**a - **b).abs() > 1e-4)
            .count();
        assert_eq!(bad, 0, "{d:?}: bf16 op clobbered {bad} canary elements");
    }
}
