// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `PrecisionPolicy::AlwaysF16` on a single `Op::Attention`, against a CPU f32
//! reference.
//!
//! The failure this pins is not imprecision: the Metal output came back as
//! **exact zeros** under f16 and ~3.3e38 under bf16, while `mixed` (which keeps
//! Compute and DataMovement at f32) was fine. So the bug is in how a low
//! precision *tag* reaches the attention kernel, not in f16 arithmetic.
//!
//! An f16 attention output is allowed to be loose — f16 carries ~3 decimal
//! digits and softmax sums over the key axis — so the tolerance is generous.
//! What it must not be is all-zero, non-finite, or unrelated to the reference.

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompileOptions, Device, PrecisionPolicy, Session};

const B: usize = 1;
const H: usize = 2;
const S: usize = 8;
const D: usize = 16;

fn build() -> Graph {
    let mut g = Graph::new("attn_precision_policy");
    let dim = H * D;
    let q = g.input("q", Shape::new(&[B, S, dim], DType::F32));
    let k = g.input("k", Shape::new(&[B, S, dim], DType::F32));
    let v = g.input("v", Shape::new(&[B, S, dim], DType::F32));
    let shape = rlx_ir::shape::attention_shape(g.shape(q));
    let out = g.attention_kind(q, k, v, H, D, MaskKind::None, shape);
    g.set_outputs(vec![out]);
    g
}

fn data(seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..B * S * H * D)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 9) as f32 / 4_194_304.0 - 1.0) * 0.5
        })
        .collect()
}

fn run(device: Device, policy: Option<PrecisionPolicy>) -> Vec<f32> {
    let (q, k, v) = (data(3), data(7), data(11));
    let feed: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
    let g = build();
    match policy {
        None => Session::new(device).compile(g).run(&feed)[0].clone(),
        Some(p) => {
            let opts = CompileOptions::new().policy(p);
            Session::new(device).compile_with(g, &opts).run(&feed)[0].clone()
        }
    }
}

fn check(label: &str, got: &[f32], want: &[f32]) {
    let zeros = got.iter().filter(|v| **v == 0.0).count();
    let nonfinite = got.iter().filter(|v| !v.is_finite()).count();
    let peak = want.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "{label}: zeros {zeros}/{}, non-finite {nonfinite}, max|Δ| {worst:.3e}, rel {:.3e}",
        got.len(),
        worst / peak.max(1e-30)
    );
    assert_eq!(nonfinite, 0, "{label}: {nonfinite} non-finite outputs");
    assert!(
        zeros < got.len() / 2,
        "{label}: {zeros}/{} outputs are exactly zero — the attention output \
         collapsed, which is what a low-precision tag reaching the wrong kernel \
         path looks like",
        got.len()
    );
    // f16 attention: loose, but the answer must be the same answer.
    assert!(
        worst <= 5e-2 * peak,
        "{label}: max|Δ| {worst:.3e} against peak {peak:.3e} — not f16 rounding"
    );
}

#[test]
fn attention_under_low_precision_policies_matches_f32() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let want = run(Device::Cpu, None);
    for (label, policy) in [
        ("metal f32", None),
        ("metal mixed", Some(PrecisionPolicy::AutoMixed)),
        ("metal f16", Some(PrecisionPolicy::AlwaysF16)),
    ] {
        check(label, &run(Device::Metal, policy), &want);
    }
}

// ── AMP sweep over the transformer path ─────────────────────────────────────

/// Pseudo-random values of a given length, in a range low precision handles.
fn data_n(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 9) as f32 / 4_194_304.0 - 1.0) * 0.5
        })
        .collect()
}

/// Run one graph under `AlwaysF16` and `AutoMixedBf16` on Metal against a CPU
/// f32 reference.
///
/// Every operand is an `Op::Input` and is fed. An unfed `Op::Param` defaults to
/// zeros, which makes the reference all-zero and the comparison vacuous — the
/// first draft of this sweep did exactly that and "passed" a case whose
/// expected output was entirely zeros.
fn sweep_case<F>(name: &str, inputs: &[(&str, Vec<usize>)], build: F)
where
    F: FnOnce(&mut Graph, &[rlx_ir::NodeId]) -> rlx_ir::NodeId,
{
    let mut g = Graph::new(name);
    let ids: Vec<rlx_ir::NodeId> = inputs
        .iter()
        .map(|(n, dims)| g.input(*n, Shape::new(dims, DType::F32)))
        .collect();
    let out = build(&mut g, &ids);
    g.set_outputs(vec![out]);

    let payloads: Vec<Vec<f32>> = inputs
        .iter()
        .enumerate()
        .map(|(i, (_, dims))| data_n(dims.iter().product(), 3 + i as u32 * 7))
        .collect();
    let feed: Vec<(&str, &[f32])> = inputs
        .iter()
        .zip(&payloads)
        .map(|((n, _), v)| (*n, v.as_slice()))
        .collect();

    let want = Session::new(Device::Cpu).compile(g.clone()).run(&feed)[0].clone();
    let peak = want.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    assert!(
        peak > 1e-6,
        "{name}: the f32 reference is ~all zeros ({peak:.2e}) — the case proves nothing"
    );

    // Only f16. BF16 compute is refused outright on Metal — see
    // `bf16_compute_is_refused_not_silently_wrong` below.
    {
        let label = "f16";
        let opts = CompileOptions::new().policy(PrecisionPolicy::AlwaysF16);
        let got = Session::new(Device::Metal)
            .compile_with(g.clone(), &opts)
            .run(&feed)[0]
            .clone();
        let zeros = got.iter().filter(|v| **v == 0.0).count();
        let nonfinite = got.iter().filter(|v| !v.is_finite()).count();
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "{name} [{label}]: zeros {zeros}/{}, non-finite {nonfinite}, rel {:.2e}",
            got.len(),
            worst / peak
        );
        assert_eq!(
            nonfinite, 0,
            "{name} [{label}]: {nonfinite} non-finite outputs"
        );
        assert!(
            zeros < got.len() / 2,
            "{name} [{label}]: {zeros}/{} outputs exactly zero — an f32 kernel \
             reading f16 bytes looks exactly like this",
            got.len()
        );
        assert!(
            worst <= 5e-2 * peak,
            "{name} [{label}]: rel {:.2e} — not low-precision rounding",
            worst / peak
        );
    }
}

/// Ops the precision pass is WILLING to downcast, exercised on Metal under a
/// low-precision policy.
///
/// The `force_f32` list in `rlx-compile/src/precision.rs` is a hand-maintained
/// record of which ops Metal has f16 kernels for — backend knowledge kept in a
/// backend-agnostic crate. It drifted: `FusedAttentionBlock` was listed and the
/// unfused `Op::Attention` was not, so one attention op under f16 returned exact
/// zeros while the list looked complete. Nothing structural prevents the next
/// omission, so cover the ops an AMP transformer actually runs through, and add
/// a case whenever the pass learns a new op.
#[test]
fn amp_sweep_over_the_transformer_path() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    use rlx_ir::infer::GraphExt;
    let dim = H * D;
    let rows = B * S;

    sweep_case(
        "matmul",
        &[("a", vec![rows, dim]), ("b", vec![dim, dim])],
        |g, n| g.matmul(n[0], n[1], Shape::new(&[rows, dim], DType::F32)),
    );
    sweep_case("softmax", &[("a", vec![rows, dim])], |g, n| g.sm(n[0], -1));
    sweep_case(
        "rms_norm",
        &[("a", vec![rows, dim]), ("b", vec![dim]), ("c", vec![dim])],
        |g, n| g.rms_norm(n[0], n[1], n[2], 1e-5),
    );
    // `layer_norm2d` is NCHW — give it a 4-D tensor rather than the [rows, dim]
    // the other cases use.
    sweep_case(
        "layer_norm2d",
        &[
            ("a", vec![1, dim, 2, 4]),
            ("b", vec![dim]),
            ("c", vec![dim]),
        ],
        |g, n| g.layer_norm2d(n[0], n[1], n[2], 1e-5),
    );
    sweep_case(
        "gelu_then_add",
        &[("a", vec![rows, dim]), ("b", vec![rows, dim])],
        |g, n| {
            let h = g.gelu(n[0]);
            g.add(h, n[1])
        },
    );
    // A whole block: the shape that previously NaN'd in `silu` after a matmul.
    sweep_case(
        "transformer_block",
        &[
            ("a", vec![rows, dim]),
            ("b", vec![dim, dim]),
            ("c", vec![dim]),
        ],
        |g, n| {
            let h = g.matmul(n[0], n[1], Shape::new(&[rows, dim], DType::F32));
            let one = g.param("ones", Shape::new(&[dim], DType::F32));
            let normed = g.rms_norm(h, one, n[2], 1e-5);
            let up = g.matmul(normed, n[1], Shape::new(&[rows, dim], DType::F32));
            let act = g.silu(up);
            let down = g.matmul(act, n[1], Shape::new(&[rows, dim], DType::F32));
            g.add(h, down)
        },
    );

    sweep_case(
        "silu_then_mul",
        &[("a", vec![rows, dim]), ("b", vec![rows, dim])],
        |g, n| {
            let h = g.silu(n[0]);
            g.mul(h, n[1])
        },
    );
}

/// BF16 compute has to be an error, not a number.
///
/// `HalfFlag` is `{F32, F16}` and maps BF16 to F32, so a bf16-tagged node runs
/// an f32 kernel over bf16 bytes. Before this was refused, a 32×32 matmul under
/// `AutoMixedBf16` returned a non-finite value with relative error 2.3e38 and no
/// diagnostic at all. A backend that cannot execute something must say so.
#[test]
fn bf16_compute_is_refused_not_silently_wrong() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let opts = CompileOptions::new().policy(PrecisionPolicy::AutoMixedBf16);
    let err = std::panic::catch_unwind(|| Session::new(Device::Metal).compile_with(build(), &opts))
        .err()
        .expect("AutoMixedBf16 was accepted on Metal — it returns non-finite garbage");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("bf16"),
        "refused, but not with a message naming bf16: {msg}"
    );
}
