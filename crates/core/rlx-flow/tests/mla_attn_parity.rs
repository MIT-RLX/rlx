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

//! CPU parity test for the Multi-head Latent Attention (MLA) prefill block.
//!
//! Drives [`MlaAttnPrefillStage`] through a real flow on `Device::Cpu` and checks
//! it against an independent plain-Rust MLA reference. The reference computes
//! **true asymmetric attention** (score dim `qk_head_dim`, value dim `v_head_dim`)
//! with no padding — so it also validates the block's pad-V-then-narrow bridge
//! onto the symmetric `Op::Attention`. Both query variants are covered: the
//! DeepSeek-V2/V3 LoRA query (`q_a`→RMSNorm→`q_b`) and the LoRA-less V2-Lite
//! `q_proj`.
//!
//! RoPE is neutralized to a per-position scale (`sin = 0`, `cos = c_s`) so the
//! oracle stays small while still exercising the RoPE op and confirming it lands
//! on the `qk_rope_head_dim` slice at the right sequence position — a NeoX
//! rotate-half with `sin = 0` collapses to `x · cos`.

use rlx_flow::MapWeights;
use rlx_flow::prelude::*;
use rlx_ir::{DType, Shape};
use rlx_runtime::{Device, Session};

const B: usize = 1;
const S: usize = 3;
const HID: usize = 8;
const NH: usize = 2;
const QLORA: usize = 6;
const KVLORA: usize = 4;
const NOPE: usize = 4;
const ROPE: usize = 2;
const VH: usize = 3;
const QK: usize = NOPE + ROPE; // per-head score dim
const EPS: f32 = 1e-6;

// Deterministic, bounded, non-degenerate fills.
fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5)
        .collect()
}
fn gamma(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| 0.9 + 0.05 * ((i as f32) + seed).cos())
        .collect()
}

// y = W·x with W stored row-major `[out, in]` (== nn.Linear's x·Wᵀ).
fn matvec(w: &[f32], out_dim: usize, in_dim: usize, x: &[f32]) -> Vec<f32> {
    (0..out_dim)
        .map(|o| (0..in_dim).map(|i| w[o * in_dim + i] * x[i]).sum())
        .collect()
}

// out = x / sqrt(mean(x²) + eps) · gamma  (matches rlx Op::RmsNorm, beta = 0).
fn rms(x: &[f32], g: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let sumsq: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();
    (0..n).map(|i| x[i] * inv * g[i]).collect()
}

/// Run the MLA prefill block on CPU and check it against the reference. `lite`
/// selects the V2-Lite (direct `q_proj`) query path instead of the LoRA pair.
fn run_mla_cpu(lite: bool) {
    // ── Weights (HF `[out, in]` layout; the block loads them transposed). ──
    // Query weights depend on the variant; the unused pair is left empty.
    let (q_a, q_a_ln, q_b, q_proj) = if lite {
        (vec![], vec![], vec![], fill(NH * QK * HID, 0.15))
    } else {
        (
            fill(QLORA * HID, 0.1),
            gamma(QLORA, 0.2),
            fill(NH * QK * QLORA, 0.3),
            vec![],
        )
    };
    let kv_a = fill((KVLORA + ROPE) * HID, 0.4);
    let kv_a_ln = gamma(KVLORA, 0.5);
    let kv_b = fill(NH * (NOPE + VH) * KVLORA, 0.6);
    let x = fill(B * S * HID, 0.7);

    // Neutralized RoPE: sin = 0, cos = per-position scale. Table is
    // `[max_positions, n_rot/2]` = `[S, 1]`.
    let cos = [0.5f32, 0.75, 1.0];
    let cos_tab = cos.to_vec();
    let sin_tab = vec![0.0f32; S];

    // ── Independent reference: build per-(pos, head) q/k/v, then attend. ──
    let mut q_full = [[[0f32; QK]; NH]; S];
    let mut k_full = [[[0f32; QK]; NH]; S];
    let mut v_ref = [[[0f32; VH]; NH]; S];
    for s in 0..S {
        let h_s = &x[s * HID..(s + 1) * HID];
        let q = if lite {
            matvec(&q_proj, NH * QK, HID, h_s)
        } else {
            let c_q = rms(&matvec(&q_a, QLORA, HID, h_s), &q_a_ln, EPS);
            matvec(&q_b, NH * QK, QLORA, &c_q)
        };

        let kva = matvec(&kv_a, KVLORA + ROPE, HID, h_s);
        let c_kv = rms(&kva[0..KVLORA], &kv_a_ln, EPS);
        let k_rope_shared = &kva[KVLORA..KVLORA + ROPE];
        let kvb = matvec(&kv_b, NH * (NOPE + VH), KVLORA, &c_kv);

        let c = cos[s];
        for h in 0..NH {
            let qh = &q[h * QK..(h + 1) * QK];
            q_full[s][h][..NOPE].copy_from_slice(&qh[..NOPE]);
            for i in 0..ROPE {
                q_full[s][h][NOPE + i] = qh[NOPE + i] * c;
            }
            let seg = &kvb[h * (NOPE + VH)..(h + 1) * (NOPE + VH)];
            k_full[s][h][..NOPE].copy_from_slice(&seg[..NOPE]);
            for i in 0..ROPE {
                k_full[s][h][NOPE + i] = k_rope_shared[i] * c;
            }
            v_ref[s][h][..VH].copy_from_slice(&seg[NOPE..NOPE + VH]);
        }
    }

    let scale = 1.0 / (QK as f32).sqrt();
    let mut expect = vec![0f32; S * NH * VH];
    for qi in 0..S {
        for h in 0..NH {
            let mut scores: Vec<f32> = (0..=qi)
                .map(|kj| {
                    (0..QK)
                        .map(|d| q_full[qi][h][d] * k_full[kj][h][d])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
            let mut den = 0.0;
            for sj in scores.iter_mut() {
                *sj = (*sj - mx).exp();
                den += *sj;
            }
            for d in 0..VH {
                let acc: f32 = (0..=qi)
                    .map(|kj| (scores[kj] / den) * v_ref[kj][h][d])
                    .sum();
                expect[qi * (NH * VH) + h * VH + d] = acc;
            }
        }
    }

    // ── Build the flow and run on CPU. ──
    let lp = "model.layers.0";
    let mut w = MapWeights::default();
    if lite {
        w.insert(
            format!("{lp}.self_attn.q_proj.weight"),
            q_proj,
            vec![NH * QK, HID],
        );
    } else {
        w.insert(
            format!("{lp}.self_attn.q_a_proj.weight"),
            q_a,
            vec![QLORA, HID],
        );
        w.insert(
            format!("{lp}.self_attn.q_a_layernorm.weight"),
            q_a_ln,
            vec![QLORA],
        );
        w.insert(
            format!("{lp}.self_attn.q_b_proj.weight"),
            q_b,
            vec![NH * QK, QLORA],
        );
    }
    w.insert(
        format!("{lp}.self_attn.kv_a_proj_with_mqa.weight"),
        kv_a,
        vec![KVLORA + ROPE, HID],
    );
    w.insert(
        format!("{lp}.self_attn.kv_a_layernorm.weight"),
        kv_a_ln,
        vec![KVLORA],
    );
    w.insert(
        format!("{lp}.self_attn.kv_b_proj.weight"),
        kv_b,
        vec![NH * (NOPE + VH), KVLORA],
    );

    let spec = if lite {
        MlaAttnPrefillSpec::deepseek_lite_layer(lp, NH, KVLORA, NOPE, ROPE, VH, EPS)
    } else {
        MlaAttnPrefillSpec::deepseek_layer(lp, NH, QLORA, KVLORA, NOPE, ROPE, VH, EPS)
    };

    let built = ModelFlow::new("mla")
        .input("x", Shape::new(&[B, S, HID], DType::F32))
        .stage(FlowStage::RopeTables(RopeTablesStage::param(
            S, 1, cos_tab, sin_tab,
        )))
        .layer_stage(MlaAttnPrefillStage::new(spec))
        .build(&mut w)
        .expect("flow build");

    let (g, params) = built.into_graph_parts().expect("graph parts");
    let mut c = Session::new(Device::Cpu).compile(g);
    for (k, v) in &params {
        c.set_param(k.as_str(), v.as_slice());
    }
    let outs = c.run(&[("x", &x)]);
    let y = &outs[0];

    assert_eq!(y.len(), expect.len(), "y={y:?}");
    let mut max_err = 0f32;
    for (a, b) in y.iter().zip(expect.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(
        max_err < 2e-4,
        "MLA CPU output diverged from reference (lite={lite}, max_err={max_err:e})\n  got   ={y:?}\n  expect={expect:?}"
    );
}

#[test]
fn mla_prefill_matches_reference_on_cpu() {
    run_mla_cpu(false);
}

#[test]
fn mla_lite_prefill_matches_reference_on_cpu() {
    run_mla_cpu(true);
}
