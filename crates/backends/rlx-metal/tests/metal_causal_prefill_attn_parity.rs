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

//! Prefill CAUSAL `Op::Attention` (Lq = Lk = S > 1) on Metal vs CPU.
//!
//! The existing Metal attention parity tests cover `MaskKind::None` (encoder)
//! and decode causal (`Lq = 1`). The LLM *prefill* path uses `MaskKind::Causal`
//! with `Lq = S > 1` — this guards that untested case (suspected Voxtral Metal
//! garbage-logits bug).

#![cfg(target_os = "macos")]

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_causal_prefill_attn(b: usize, h: usize, s: usize, d: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("causal_prefill_attn");
    let q = g.input("q", Shape::new(&[b, h, s, d], f));
    let k = g.input("k", Shape::new(&[b, h, s, d], f));
    let v = g.input("v", Shape::new(&[b, h, s, d], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: h,
            head_dim: d,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, h, s, d], f),
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
fn metal_causal_prefill_attention_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    // S > 1 so the causal mask actually constrains multiple query rows.
    let (b, h, s, d) = (1, 8, 96, 64);
    let n = b * h * s * d;
    let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin()).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).cos()).collect();
    let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();

    let g = build_causal_prefill_attn(b, h, s, d);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);

    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cpu_sum: f64 = cpu.iter().map(|&x| x as f64).sum();
    let metal_sum: f64 = metal.iter().map(|&x| x as f64).sum();
    eprintln!(
        "causal prefill attn: max_abs={max_abs:.6} cpu_sum={cpu_sum:.4} metal_sum={metal_sum:.4}"
    );
    assert!(max_abs < 1e-4, "causal prefill attention max_abs={max_abs}");
}
