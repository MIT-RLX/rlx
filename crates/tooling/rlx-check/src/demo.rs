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

//! Tiny built-in graphs, one per diagnostic class, so `cargo rlx check --demo
//! <name>` exercises the checker end-to-end with no input file.

use rlx_ir::infer::GraphExt;
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};

/// Names accepted by [`build`], each paired with what it demonstrates.
pub const DEMOS: &[(&str, &str)] = &[
    (
        "mlp",
        "clean matmul+bias+gelu — no findings, fuses to FusedMatMulBiasAct",
    ),
    (
        "swiglu",
        "gate declared before up — a missed SwiGLU fusion (warning)",
    ),
    (
        "badshape",
        "matmul with a wrong declared out-shape — a shape error",
    ),
    (
        "nan",
        "constant 1.0 / 0.0 — a provable numeric blow-up (warning)",
    ),
];

/// Build a demo graph by name, or `None` if unknown.
pub fn build(name: &str) -> Option<Graph> {
    match name {
        "mlp" => Some(mlp()),
        "swiglu" => Some(swiglu_gate_first()),
        "badshape" => Some(bad_shape()),
        "nan" => Some(nan_div()),
        _ => None,
    }
}

fn f32s(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

/// Clean feed-forward: `gelu(x @ w + b)`. Fuses cleanly, no findings.
fn mlp() -> Graph {
    let mut g = Graph::new("mlp");
    let x = g.input("x", f32s(&[4, 16]));
    let w = g.param("w", f32s(&[16, 8]));
    let b = g.param("b", f32s(&[8]));
    let h = g.mm(x, w);
    let h = g.add(h, b);
    let y = g.gelu(h);
    g.set_outputs(vec![y]);
    g
}

/// SwiGLU FFN whose gate matmul is declared before the up matmul — the
/// declaration order prevents `FuseSwiGLU`, a classic missed fusion.
fn swiglu_gate_first() -> Graph {
    let mut g = Graph::new("swiglu");
    let x = g.input("x", f32s(&[4, 8]));
    let gate_w = g.param("gate", f32s(&[8, 16]));
    let up_w = g.param("up", f32s(&[8, 16]));
    let gate = g.mm(x, gate_w);
    let up = g.mm(x, up_w);
    let gate_silu = g.silu(gate);
    let out = g.mul(gate_silu, up);
    g.set_outputs(vec![out]);
    g
}

/// `x @ w` with `x:[2,4]`, `w:[4,3]` (infers `[2,3]`) but a declared out-shape
/// of `[2,5]` — caught by `verify_shapes`.
fn bad_shape() -> Graph {
    let mut g = Graph::new("badshape");
    let x = g.input("x", f32s(&[2, 4]));
    let w = g.param("w", f32s(&[4, 3]));
    let y = g.matmul(x, w, f32s(&[2, 5]));
    g.set_outputs(vec![y]);
    g
}

/// `1.0 / 0.0` over constants — folds to +Inf, flagged by the numeric lint.
fn nan_div() -> Graph {
    let mut g = Graph::new("nan");
    let a = g.add_node(
        Op::Constant {
            data: 1.0f32.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    );
    let b = g.add_node(
        Op::Constant {
            data: 0.0f32.to_le_bytes().to_vec(),
        },
        vec![],
        f32s(&[1]),
    );
    let d = g.binary(BinaryOp::Div, a, b, f32s(&[1]));
    g.set_outputs(vec![d]);
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_listed_demos_build() {
        for (name, _) in DEMOS {
            assert!(build(name).is_some(), "demo {name} failed to build");
        }
    }
}
