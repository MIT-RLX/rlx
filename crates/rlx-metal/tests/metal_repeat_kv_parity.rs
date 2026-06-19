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

//! GQA `repeat_kv` pattern (narrow + concat tiling) on Metal vs CPU.
//!
//! Mirrors `rlx_flow::blocks::self_attn::repeat_kv`: slice each of `num_kv_heads`
//! head blocks out of `[B, S, num_kv_heads*head_dim]` and concat `group` copies
//! into `[B, S, num_kv_heads*group*head_dim]`. LLM-specific (Whisper encoder is
//! full MHA), and a prime structural suspect for the Voxtral Metal garbage bug.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_repeat_kv(
    b: usize,
    s: usize,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("repeat_kv");
    let x = g.input("x", Shape::new(&[b, s, num_kv_heads * head_dim], f));
    let mut pieces = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.add_node(
            rlx_ir::Op::Narrow {
                axis: 2,
                start: h * head_dim,
                len: head_dim,
            },
            vec![x],
            Shape::new(&[b, s, head_dim], f),
        );
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    let out = g.add_node(
        rlx_ir::Op::Concat { axis: 2 },
        pieces,
        Shape::new(&[b, s, num_kv_heads * group * head_dim], f),
    );
    g.set_outputs(vec![out]);
    g
}

#[test]
fn metal_repeat_kv_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    // Voxtral-Mini-3B text GQA: 8 kv heads, group 4, head_dim 128.
    let (b, s, num_kv_heads, head_dim, group) = (1, 96, 8, 128, 4);
    let n = b * s * num_kv_heads * head_dim;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).sin()).collect();

    let g = build_repeat_kv(b, s, num_kv_heads, head_dim, group);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("x", &x)]).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("x", &x)]).remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "repeat_kv: len cpu={} metal={} max_abs={max_abs:.6}",
        cpu.len(),
        metal.len()
    );
    assert_eq!(cpu.len(), metal.len(), "repeat_kv output length mismatch");
    assert!(max_abs < 1e-5, "repeat_kv max_abs={max_abs}");
}
