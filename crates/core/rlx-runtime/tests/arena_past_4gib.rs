// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A training step whose arena is past 4 GiB gives on every GPU what it gives
//! on the CPU. A frozen embedding table of `BIG_ARENA_ROWS` rows (default
//! 4.6M × 256 f32 ≈ 4.4 GiB) comes first in the arena, so every activation and
//! gradient after it sits past 4 GiB: gather → RMSNorm → q/k/v → RoPE →
//! attention → output → SiLU MLP → logits → cross-entropy, then the backward.
//! This guards the 32-bit element offsets the GPU kernels take (rlx-rocm
//! refuses unaudited steps past 4 GiB).
//!
//! Allocates ~5 GiB per backend, so it runs only with `BIG_ARENA_TEST=1`.
//! `BIG_ARENA_ROWS=1000` runs the same graph small (to check it builds).

use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_runtime::{Device, Session};

const S: usize = 16;
const H: usize = 2;
const DH: usize = 32;
const D: usize = 256;
const HID: usize = H * DH;
const F: usize = 128;
const V: usize = 96;

fn f32s(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn rand(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
        })
        .collect()
}

/// The forward graph, its trained params, and the loss.
fn graph(rows: usize) -> (Graph, Vec<rlx_ir::NodeId>) {
    let mut g = Graph::new("big_arena_step");
    // First, so the planner puts it at the bottom of the arena.
    let embed = g.param("embed", f32s(&[rows, D]));
    let tokens = g.input("tokens", Shape::new(&[S], DType::I32));
    let labels = g.input("labels", f32s(&[S]));
    let cos = g.input("cos", f32s(&[S, DH / 2]));
    let sin = g.input("sin", f32s(&[S, DH / 2]));
    let gamma = g.param("gamma", f32s(&[D]));
    let beta = g.param("beta", f32s(&[D]));
    let wq = g.param("wq", f32s(&[D, HID]));
    let wk = g.param("wk", f32s(&[D, HID]));
    let wv = g.param("wv", f32s(&[D, HID]));
    let wo = g.param("wo", f32s(&[HID, D]));
    let w1 = g.param("w1", f32s(&[D, F]));
    let w2 = g.param("w2", f32s(&[F, D]));
    let head = g.param("head", f32s(&[D, V]));

    let x = g.gather_(embed, tokens, 0); // [S, D]
    let x = g.reshape_(x, vec![1, S as i64, D as i64]);
    let h = g.rms_norm(x, gamma, beta, 1e-6);
    let heads = |g: &mut Graph, w, rope: bool| {
        let t = g.mm(h, w); // [1, S, HID]
        let t = if rope { g.rope(t, cos, sin, DH) } else { t };
        let t = g.reshape_(t, vec![1, S as i64, H as i64, DH as i64]);
        g.transpose_(t, vec![0, 2, 1, 3]) // [1, H, S, DH]
    };
    let q = heads(&mut g, wq, true);
    let k = heads(&mut g, wk, true);
    let v = heads(&mut g, wv, false);
    let a = g.add_node(
        Op::Attention {
            num_heads: H,
            head_dim: DH,
            v_head_dim: None,
            mask_kind: rlx_ir::op::MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        f32s(&[1, H, S, DH]),
    );
    let a = g.transpose_(a, vec![0, 2, 1, 3]);
    let a = g.reshape_(a, vec![1, S as i64, HID as i64]);
    let o = g.mm(a, wo);
    let x = g.add(x, o);
    let m = g.mm(x, w1);
    let m = g.silu(m);
    let m = g.mm(m, w2);
    let x = g.add(x, m);
    let x = g.reshape_(x, vec![S as i64, D as i64]);
    let logits = g.mm(x, head);
    let per = g.add_node(
        Op::SoftmaxCrossEntropyWithLogits,
        vec![logits, labels],
        f32s(&[S]),
    );
    let loss = g.reduce(
        per,
        ReduceOp::Mean,
        vec![0],
        false,
        Shape::scalar(DType::F32),
    );
    g.set_outputs(vec![loss]);
    (g, vec![gamma, wq, wk, wv, wo, w1, w2, head, beta])
}

fn run(device: Device, rows: usize, embed: &[f32]) -> Vec<Vec<f32>> {
    let (fwd, wrt) = graph(rows);
    let bwd = rlx_autodiff::grad_with_loss(&fwd, &wrt);
    let mut c = Session::new(device).compile(bwd);
    c.set_param("embed", embed);
    let ones = |n: usize| vec![1.0f32; n];
    c.set_param("gamma", &ones(D));
    c.set_param("beta", &vec![0.0; D]);
    for (i, (n, len)) in [
        ("wq", D * HID),
        ("wk", D * HID),
        ("wv", D * HID),
        ("wo", HID * D),
        ("w1", D * F),
        ("w2", F * D),
        ("head", D * V),
    ]
    .iter()
    .enumerate()
    {
        c.set_param(n, &rand(*len, 100 + i as u64, 0.2));
    }
    // Tokens from the top of the table: rows far past 4 GiB when it is big.
    let tokens: Vec<f32> = (0..S).map(|i| (rows - 1 - i * 7) as f32).collect();
    let labels: Vec<f32> = (0..S).map(|i| ((i * 5) % V) as f32).collect();
    let (mut cs, mut sn) = (Vec::new(), Vec::new());
    for p in 0..S {
        for j in 0..DH / 2 {
            let a = p as f32 / 10_000f32.powf(2.0 * j as f32 / DH as f32);
            cs.push(a.cos());
            sn.push(a.sin());
        }
    }
    c.run(&[
        ("tokens", &tokens),
        ("labels", &labels),
        ("cos", &cs),
        ("sin", &sn),
        ("d_output", &[1.0]),
    ])
}

#[test]
fn a_training_step_past_4_gib_matches_the_cpu() {
    if std::env::var("BIG_ARENA_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set BIG_ARENA_TEST=1 (allocates ~5 GiB per backend)");
        return;
    }
    let rows: usize = std::env::var("BIG_ARENA_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4_600_000);
    eprintln!(
        "embedding table: {rows} × {D} f32 = {:.2} GiB",
        (rows * D * 4) as f64 / (1u64 << 30) as f64
    );
    let embed = rand(rows * D, 7, 1.0);
    let want = run(Device::Cpu, rows, &embed);
    assert!(
        want[0][0].is_finite() && want[0][0] > 0.0,
        "CPU loss {:?}",
        want[0]
    );
    let mut checked = 0;
    let mut bad = Vec::new();
    for dev in rlx_runtime::available_devices() {
        if dev == Device::Cpu || !rlx_runtime::devices_for(&graph(rows).0).contains(&dev) {
            continue;
        }
        let got = run(dev, rows, &embed);
        assert_eq!(got.len(), want.len(), "{dev:?}: output count");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let scale = w.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            let err = g
                .iter()
                .zip(w)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("{dev:?} output {i}: max |Δ| {err:.2e} (scale {scale:.2e})");
            if err > 2e-3 * scale {
                eprintln!(
                    "  got  {:?}\n  want {:?}",
                    &g[..g.len().min(6)],
                    &w[..w.len().min(6)]
                );
            }
            if err > 2e-3 * scale {
                bad.push(format!(
                    "{dev:?} output {i}: {err:.2e} vs scale {scale:.2e}"
                ));
            }
        }
        checked += 1;
    }
    eprintln!("checked {checked} GPU backend(s) against the CPU");
    assert!(
        bad.is_empty(),
        "outputs differ from the CPU:\n{}",
        bad.join("\n")
    );
}
