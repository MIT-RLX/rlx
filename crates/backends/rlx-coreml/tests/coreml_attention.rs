// RoPE + Attention MIL lowering, validated on-device against explicit
// hand references (NeoX split-halves rope; causal scaled-dot-product
// attention), then composed into a full transformer attention block.
#![cfg(any(target_os = "macos", target_os = "ios"))]
#![allow(clippy::useless_vec)] // `vec![..; CONST]` reads clearly in test scaffolding

use rlx_coreml::CoremlExecutable;
use rlx_ir::op::{BinaryOp, MaskKind};
use rlx_ir::{DType, Graph, Op, Shape};

const B: usize = 1;
const S: usize = 4;
const H: usize = 2;
const D: usize = 8; // head_dim
const HID: usize = H * D; // 16

fn approx(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    let mx = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        mx <= tol,
        "max abs diff {mx} > {tol}\n got {a:?}\n ref {b:?}"
    );
}

fn rope_tables() -> (Vec<f32>, Vec<f32>) {
    let half = D / 2;
    let mut cos = vec![0.0f32; S * half];
    let mut sin = vec![0.0f32; S * half];
    for pos in 0..S {
        for i in 0..half {
            let inv_freq = 1.0f64 / 10000f64.powf((2 * i) as f64 / D as f64);
            let ang = pos as f64 * inv_freq;
            cos[pos * half + i] = ang.cos() as f32;
            sin[pos * half + i] = ang.sin() as f32;
        }
    }
    (cos, sin)
}

/// NeoX split-halves rope on a single [.., S, .., D] head lane.
fn rope_ref(x: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
    // x laid out [H, S, D]; rotate per (h, s).
    let half = D / 2;
    let mut out = x.to_vec();
    for h in 0..H {
        for s in 0..S {
            let base = (h * S + s) * D;
            for i in 0..half {
                let c = cos[s * half + i];
                let sn = sin[s * half + i];
                let x1 = x[base + i];
                let x2 = x[base + half + i];
                out[base + i] = x1 * c - x2 * sn;
                out[base + half + i] = x2 * c + x1 * sn;
            }
        }
    }
    out
}

#[test]
fn rope_matches_reference() {
    let x: Vec<f32> = (0..H * S * D).map(|i| ((i as f32) * 0.1).sin()).collect();
    let (cos, sin) = rope_tables();

    let mut g = Graph::new("rope");
    let xi = g.input("x", Shape::new(&[B, H, S, D], DType::F32));
    let ci = g.param("cos", Shape::new(&[S, D / 2], DType::F32));
    let si = g.param("sin", Shape::new(&[S, D / 2], DType::F32));
    let y = g.append_node(
        Op::Rope {
            head_dim: D,
            n_rot: D,
            style: rlx_ir::RopeStyle::NeoX,
        },
        vec![xi, ci, si],
        Shape::new(&[B, H, S, D], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);

    let mut exe = CoremlExecutable::compile(g);
    exe.set_param("cos", &cos);
    exe.set_param("sin", &sin);
    let out = exe.run(&[("x", &x)]).expect("run").remove(0);

    approx(&out, &rope_ref(&x, &cos, &sin), 1e-4);
}

/// Causal SDPA on [H, S, D] q/k/v, scale = 1/sqrt(D).
fn attn_ref(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let scale = (D as f32).powf(-0.5);
    let mut out = vec![0.0f32; H * S * D];
    for h in 0..H {
        for qi in 0..S {
            let qb = (h * S + qi) * D;
            // scores
            let mut sc = vec![f32::NEG_INFINITY; S];
            for ki in 0..=qi {
                let kb = (h * S + ki) * D;
                let mut dot = 0.0;
                for d in 0..D {
                    dot += q[qb + d] * k[kb + d];
                }
                sc[ki] = dot * scale;
            }
            // softmax over 0..=qi
            let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut den = 0.0;
            let mut e = vec![0.0f32; S];
            for ki in 0..=qi {
                e[ki] = (sc[ki] - m).exp();
                den += e[ki];
            }
            // weighted sum of v
            for d in 0..D {
                let mut acc = 0.0;
                for ki in 0..=qi {
                    acc += (e[ki] / den) * v[(h * S + ki) * D + d];
                }
                out[qb + d] = acc;
            }
        }
    }
    out
}

#[test]
fn attention_causal_matches_reference() {
    let mk = |seed: f32| -> Vec<f32> {
        (0..H * S * D)
            .map(|i| ((i as f32) * 0.07 + seed).sin() * 0.5)
            .collect()
    };
    let q = mk(0.0);
    let k = mk(1.0);
    let v = mk(2.0);

    let mut g = Graph::new("attn");
    let qi = g.input("q", Shape::new(&[B, H, S, D], DType::F32));
    let ki = g.input("k", Shape::new(&[B, H, S, D], DType::F32));
    let vi = g.input("v", Shape::new(&[B, H, S, D], DType::F32));
    let y = g.append_node(
        Op::Attention {
            num_heads: H,
            head_dim: D,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![qi, ki, vi],
        Shape::new(&[B, H, S, D], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);

    let mut exe = CoremlExecutable::compile(g);
    let out = exe
        .run(&[("q", &q), ("k", &k), ("v", &v)])
        .expect("run")
        .remove(0);

    approx(&out, &attn_ref(&q, &k, &v), 1e-4);
}

// ── Full attention block ──────────────────────────────────────────────

fn weights(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.123 + seed).sin() * 0.2)
        .collect()
}

fn matmul(a: &[f32], w: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for r in 0..rows {
        for j in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                acc += a[r * k + kk] * w[kk * n + j];
            }
            out[r * n + j] = acc;
        }
    }
    out
}

/// Full reference for the block built in `attn_block`.
fn block_ref(x: &[f32], wq: &[f32], wk: &[f32], wv: &[f32], wo: &[f32]) -> Vec<f32> {
    // rms_norm (gamma=1, beta=0)
    let mut xn = vec![0.0f32; S * HID];
    for s in 0..S {
        let ms = (0..HID).map(|j| x[s * HID + j].powi(2)).sum::<f32>() / HID as f32;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for j in 0..HID {
            xn[s * HID + j] = x[s * HID + j] * inv;
        }
    }
    let q = matmul(&xn, wq, S, HID, HID);
    let k = matmul(&xn, wk, S, HID, HID);
    let v = matmul(&xn, wv, S, HID, HID);

    // [S,HID] -> [H,S,D]
    let split = |t: &[f32]| -> Vec<f32> {
        let mut o = vec![0.0f32; H * S * D];
        for s in 0..S {
            for h in 0..H {
                for d in 0..D {
                    o[(h * S + s) * D + d] = t[s * HID + h * D + d];
                }
            }
        }
        o
    };
    let (cos, sin) = rope_tables();
    let qh = rope_ref(&split(&q), &cos, &sin);
    let kh = rope_ref(&split(&k), &cos, &sin);
    let vh = split(&v);

    let attn = attn_ref(&qh, &kh, &vh); // [H,S,D]

    // [H,S,D] -> [S,HID]
    let mut merged = vec![0.0f32; S * HID];
    for h in 0..H {
        for s in 0..S {
            for d in 0..D {
                merged[s * HID + h * D + d] = attn[(h * S + s) * D + d];
            }
        }
    }
    let o = matmul(&merged, wo, S, HID, HID);
    (0..S * HID).map(|i| o[i] + x[i]).collect()
}

fn attn_block() -> Graph {
    let mut g = Graph::new("attn_block");
    let x = g.input("x", Shape::new(&[B, S, HID], DType::F32));

    let gamma = g.param("ln_g", Shape::new(&[HID], DType::F32));
    let beta = g.param("ln_b", Shape::new(&[HID], DType::F32));
    let xn = g.append_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-6,
        },
        vec![x, gamma, beta],
        Shape::new(&[B, S, HID], DType::F32),
        None,
    );

    let proj = |g: &mut Graph, src, name: &str| {
        let w = g.param(name, Shape::new(&[HID, HID], DType::F32));
        g.matmul(src, w, Shape::new(&[B, S, HID], DType::F32))
    };
    let q = proj(&mut g, xn, "Wq");
    let k = proj(&mut g, xn, "Wk");
    let v = proj(&mut g, xn, "Wv");

    let to_bhsd = |g: &mut Graph, t| {
        let r = g.reshape(
            t,
            vec![B as i64, S as i64, H as i64, D as i64],
            Shape::new(&[B, S, H, D], DType::F32),
        );
        g.append_node(
            Op::Transpose {
                perm: vec![0, 2, 1, 3],
            },
            vec![r],
            Shape::new(&[B, H, S, D], DType::F32),
            None,
        )
    };
    let qh = to_bhsd(&mut g, q);
    let kh = to_bhsd(&mut g, k);
    let vh = to_bhsd(&mut g, v);

    let cos = g.param("cos", Shape::new(&[S, D / 2], DType::F32));
    let sin = g.param("sin", Shape::new(&[S, D / 2], DType::F32));
    let rope = |g: &mut Graph, t, cos, sin| {
        g.append_node(
            Op::Rope {
                head_dim: D,
                n_rot: D,
                style: rlx_ir::RopeStyle::NeoX,
            },
            vec![t, cos, sin],
            Shape::new(&[B, H, S, D], DType::F32),
            None,
        )
    };
    let qr = rope(&mut g, qh, cos, sin);
    let kr = rope(&mut g, kh, cos, sin);

    let attn = g.append_node(
        Op::Attention {
            num_heads: H,
            head_dim: D,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![qr, kr, vh],
        Shape::new(&[B, H, S, D], DType::F32),
        None,
    );

    let back = g.append_node(
        Op::Transpose {
            perm: vec![0, 2, 1, 3],
        },
        vec![attn],
        Shape::new(&[B, S, H, D], DType::F32),
        None,
    );
    let merged = g.reshape(
        back,
        vec![B as i64, S as i64, HID as i64],
        Shape::new(&[B, S, HID], DType::F32),
    );

    let wo = g.param("Wo", Shape::new(&[HID, HID], DType::F32));
    let o = g.matmul(merged, wo, Shape::new(&[B, S, HID], DType::F32));
    let y = g.binary(BinaryOp::Add, o, x, Shape::new(&[B, S, HID], DType::F32));
    g.set_outputs(vec![y]);
    g
}

/// Cross-backend parity: the same block graph compiled + run on the CPU
/// backend (RLX's reference executor) and the ANE backend through the
/// public `Session` API. Exercises the full runtime path incl. the
/// supported-op legalization that routes `Device::Ane`.
#[test]
fn transformer_block_cpu_vs_ane() {
    use rlx_runtime::{CompiledGraph, Device, Session};

    let x: Vec<f32> = (0..B * S * HID)
        .map(|i| ((i as f32) * 0.05).sin())
        .collect();
    let (wq, wk, wv, wo) = (
        weights(HID * HID, 1.0),
        weights(HID * HID, 2.0),
        weights(HID * HID, 3.0),
        weights(HID * HID, 4.0),
    );
    let (cos, sin) = rope_tables();
    let set = |c: &mut CompiledGraph| {
        c.set_param("ln_g", &vec![1.0f32; HID]);
        c.set_param("ln_b", &vec![0.0f32; HID]);
        c.set_param("Wq", &wq);
        c.set_param("Wk", &wk);
        c.set_param("Wv", &wv);
        c.set_param("Wo", &wo);
        c.set_param("cos", &cos);
        c.set_param("sin", &sin);
    };

    let mut cpu = Session::new(Device::Cpu).compile(attn_block());
    set(&mut cpu);
    let cpu_out = cpu.run(&[("x", &x)]).remove(0);

    let mut ane = Session::new(Device::Ane).compile(attn_block());
    set(&mut ane);
    let ane_out = ane.run(&[("x", &x)]).remove(0);

    approx(&ane_out, &cpu_out, 3e-2);
}

#[test]
fn transformer_block_matches_reference() {
    let x: Vec<f32> = (0..B * S * HID)
        .map(|i| ((i as f32) * 0.05).sin())
        .collect();
    let (wq, wk, wv, wo) = (
        weights(HID * HID, 1.0),
        weights(HID * HID, 2.0),
        weights(HID * HID, 3.0),
        weights(HID * HID, 4.0),
    );
    let (cos, sin) = rope_tables();

    let mut exe = CoremlExecutable::compile(attn_block());
    exe.set_param("ln_g", &vec![1.0f32; HID]);
    exe.set_param("ln_b", &vec![0.0f32; HID]);
    exe.set_param("Wq", &wq);
    exe.set_param("Wk", &wk);
    exe.set_param("Wv", &wv);
    exe.set_param("Wo", &wo);
    exe.set_param("cos", &cos);
    exe.set_param("sin", &sin);
    let out = exe.run(&[("x", &x)]).expect("run").remove(0);

    approx(&out, &block_ref(&x, &wq, &wk, &wv, &wo), 2e-3);
}
