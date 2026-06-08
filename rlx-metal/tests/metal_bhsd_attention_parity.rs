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

//! Forward `Op::Attention` with `[B, H, S, D]` Q/K/V on Metal vs CPU.

#![cfg(target_os = "macos")]

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_bhsd_attn(b: usize, h: usize, s: usize, d: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("bhsd_attn");
    let q = g.input("q", Shape::new(&[b, h, s, d], f));
    let k = g.input("k", Shape::new(&[b, h, s, d], f));
    let v = g.input("v", Shape::new(&[b, h, s, d], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: h,
            head_dim: d,
            mask_kind: MaskKind::None,
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
fn metal_bhsd_attention_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    let (b, h, s, d) = (1, 8, 128, 64);
    let n = b * h * s * d;
    let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin()).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).cos()).collect();
    let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();

    let g = build_bhsd_attn(b, h, s, d);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);

    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("metal BHSD attention max_abs={max_abs:.6}");
    assert!(max_abs < 1e-4, "BHSD attention max_abs={max_abs}");
}
