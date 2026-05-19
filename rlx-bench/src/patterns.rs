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

//! Canonical benchmark patterns.
//!
//! Add new patterns here by implementing [`crate::BenchmarkPattern`].
//! Existing patterns are minimal but correct — they exercise the full
//! compile + dispatch path so timings reflect real backend cost
//! (kernel launch, parameter binding, output read), not just kernel
//! body execution.

use crate::{BenchmarkPattern, Tier};
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};

/// **L1 — single matmul.** `out = x @ w` with x: `[m, k]`, w: `[k, n]`.
/// Plain BLAS / kernel dispatch — no fusion. Use to measure raw
/// matmul throughput per device.
pub struct MatmulPattern {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl BenchmarkPattern for MatmulPattern {
    fn name(&self) -> &str {
        "matmul"
    }
    fn tier(&self) -> Tier {
        Tier::L1
    }

    fn build_graph(&self) -> Graph {
        let f = DType::F32;
        let mut g = Graph::new("matmul_bench");
        let x = g.input("x", Shape::new(&[self.m, self.k], f));
        let w = g.input("w", Shape::new(&[self.k, self.n], f));
        let out = g.mm(x, w);
        g.set_outputs(vec![out]);
        g
    }

    fn input_data(&self) -> Vec<(String, Vec<f32>)> {
        vec![
            ("x".to_string(), vec![1.0; self.m * self.k]),
            ("w".to_string(), vec![1.0; self.k * self.n]),
        ]
    }
}

/// **L1 — single LayerNorm.** `out = layer_norm(x, gamma, beta)` with
/// `x: [rows, hidden]`. Hits the per-row reduction kernel.
pub struct LayerNormPattern {
    pub rows: usize,
    pub hidden: usize,
}

impl BenchmarkPattern for LayerNormPattern {
    fn name(&self) -> &str {
        "layer_norm"
    }
    fn tier(&self) -> Tier {
        Tier::L1
    }

    fn build_graph(&self) -> Graph {
        let f = DType::F32;
        let mut g = Graph::new("ln_bench");
        let x = g.input("x", Shape::new(&[self.rows, self.hidden], f));
        let gamma = g.input("gamma", Shape::new(&[self.hidden], f));
        let beta = g.input("beta", Shape::new(&[self.hidden], f));
        let out = g.ln(x, gamma, beta, 1e-5);
        g.set_outputs(vec![out]);
        g
    }

    fn input_data(&self) -> Vec<(String, Vec<f32>)> {
        vec![
            ("x".to_string(), vec![1.0; self.rows * self.hidden]),
            ("gamma".to_string(), vec![1.0; self.hidden]),
            ("beta".to_string(), vec![0.0; self.hidden]),
        ]
    }
}

/// **L2 — composite: matmul → bias → relu.** `out = relu(x @ w + b)`.
/// Tests fusion: with `FuseMatMulBiasAct` enabled, the three IR ops
/// collapse into one `Op::FusedMatMulBiasAct`. Without fusion, three
/// separate kernel dispatches. This pattern is the canonical FFN
/// hidden-layer shape.
pub struct MatmulBiasReluPattern {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl BenchmarkPattern for MatmulBiasReluPattern {
    fn name(&self) -> &str {
        "matmul_bias_relu"
    }
    fn tier(&self) -> Tier {
        Tier::L2
    }

    fn build_graph(&self) -> Graph {
        let f = DType::F32;
        let mut g = Graph::new("matmul_bias_relu_bench");
        let x = g.input("x", Shape::new(&[self.m, self.k], f));
        let w = g.input("w", Shape::new(&[self.k, self.n], f));
        let b = g.input("b", Shape::new(&[self.n], f));
        let mm = g.mm(x, w);
        let biased = g.add(mm, b);
        let out = g.relu(biased);
        g.set_outputs(vec![out]);
        g
    }

    fn input_data(&self) -> Vec<(String, Vec<f32>)> {
        // Mix of negatives so relu actually does work (not pure pass-through).
        let x: Vec<f32> = (0..self.m * self.k)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let w: Vec<f32> = vec![0.5; self.k * self.n];
        let b: Vec<f32> = vec![-0.1; self.n];
        vec![
            ("x".to_string(), x),
            ("w".to_string(), w),
            ("b".to_string(), b),
        ]
    }
}
