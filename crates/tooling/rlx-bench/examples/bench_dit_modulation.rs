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

//! Fused vs unfused DiT modulation **backward** microbench.
//!
//! Compares packed `AdaLayerNormBackward` / `GatedResidualBackward` against
//! `unfuse_dit_modulation` then primitive VJPs on the same `[B,S,D]` +
//! `[B,1,D]` shapes (the Expand materialization packed reverse avoids).
//!
//! ```sh
//! just throttle   # or RLX_ALLOW_THROTTLE=1
//! cargo run -p rlx-bench --release --example bench_dit_modulation
//! cargo run -p rlx-bench --release --example bench_dit_modulation --features metal
//! ```

use rlx_driver::Device;
use rlx_ir::Tick;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::AdaNormKind;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::Session;
#[cfg(feature = "metal")]
use rlx_runtime::is_available;

const EPS: f32 = 1e-5;
const WARMUP: usize = 5;
const RUNS: usize = 25;

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| seed * (0.13 * (i as f32) - 0.09 * ((i % 5) as f32)))
        .collect()
}

fn ada_loss(b: usize, s: usize, d: usize) -> (Graph, NodeId, NodeId, NodeId) {
    let f = DType::F32;
    let mut g = Graph::new("ada_bwd_bench");
    let x = g.input("x", Shape::new(&[b, s, d], f));
    let scale = g.input("scale", Shape::new(&[b, 1, d], f));
    let shift = g.input("shift", Shape::new(&[b, 1, d], f));
    let out = g.ada_layer_norm(x, scale, shift, AdaNormKind::LayerNorm, EPS);
    let loss = g.sum(out, vec![0, 1, 2], false);
    g.set_outputs(vec![loss]);
    (g, x, scale, shift)
}

fn gate_loss(b: usize, s: usize, d: usize) -> (Graph, NodeId, NodeId, NodeId) {
    let f = DType::F32;
    let mut g = Graph::new("gate_bwd_bench");
    let x = g.input("x", Shape::new(&[b, s, d], f));
    let y = g.input("y", Shape::new(&[b, s, d], f));
    let gate = g.input("gate", Shape::new(&[b, 1, d], f));
    let out = g.gated_residual(x, y, gate);
    let loss = g.sum(out, vec![0, 1, 2], false);
    g.set_outputs(vec![loss]);
    (g, x, y, gate)
}

fn time_ns(device: Device, graph: Graph, feeds: &[(&str, &[f32])]) -> u64 {
    let mut c = Session::new(device).compile(graph);
    for _ in 0..WARMUP {
        let _ = c.run(feeds);
    }
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t0 = Tick::now();
        let _ = c.run(feeds);
        samples.push(Tick::now().elapsed_ns(t0));
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn devices() -> Vec<(&'static str, Device)> {
    #[cfg(feature = "metal")]
    {
        let mut v = vec![("cpu", Device::Cpu)];
        if is_available(Device::Metal) {
            v.push(("metal", Device::Metal));
        }
        v
    }
    #[cfg(not(feature = "metal"))]
    {
        vec![("cpu", Device::Cpu)]
    }
}

fn report(label: &str, fused_ns: u64, unfused_ns: u64) {
    let speedup = unfused_ns as f64 / fused_ns.max(1) as f64;
    println!(
        "  {label:28} fused={:8.2} us  unfused={:8.2} us  speedup={speedup:.2}x",
        fused_ns as f64 / 1e3,
        unfused_ns as f64 / 1e3
    );
}

fn main() {
    let shapes = [(2usize, 128usize, 64usize), (4, 256, 128), (8, 512, 256)];
    for (name, device) in devices() {
        println!("\n=== device={name} warmup={WARMUP} runs={RUNS} ===");
        for &(b, s, d) in &shapes {
            println!("  shape [B,S,D]=[{b},{s},{d}]  mod=[B,1,D]");
            let x = fill(b * s * d, 1.1);
            let scale = fill(b * d, 0.7);
            let shift = fill(b * d, -0.4);
            let d_output = [1.0f32];
            let ada_feeds = [
                ("x", x.as_slice()),
                ("scale", scale.as_slice()),
                ("shift", shift.as_slice()),
                ("d_output", d_output.as_slice()),
            ];

            let (g, xn, sn, tn) = ada_loss(b, s, d);
            let fused = grad_with_loss(&g, &[xn, sn, tn]);
            let (g, _, _, _) = ada_loss(b, s, d);
            let g = rlx_fusion::unfuse_dit_modulation(g);
            let xn = g
                .nodes()
                .iter()
                .find(|n| matches!(&n.op, rlx_ir::Op::Input { name } if name == "x"))
                .map(|n| n.id)
                .unwrap();
            let sn = g
                .nodes()
                .iter()
                .find(|n| matches!(&n.op, rlx_ir::Op::Input { name } if name == "scale"))
                .map(|n| n.id)
                .unwrap();
            let tn = g
                .nodes()
                .iter()
                .find(|n| matches!(&n.op, rlx_ir::Op::Input { name } if name == "shift"))
                .map(|n| n.id)
                .unwrap();
            let unfused = grad_with_loss(&g, &[xn, sn, tn]);
            report(
                "AdaLayerNormBackward",
                time_ns(device, fused, &ada_feeds),
                time_ns(device, unfused, &ada_feeds),
            );

            let y = fill(b * s * d, -0.6);
            let gate = fill(b * d, 0.35);
            let gate_feeds = [
                ("x", x.as_slice()),
                ("y", y.as_slice()),
                ("gate", gate.as_slice()),
                ("d_output", d_output.as_slice()),
            ];
            let (g, xn, yn, gn) = gate_loss(b, s, d);
            let fused = grad_with_loss(&g, &[xn, yn, gn]);
            let (g, _, _, _) = gate_loss(b, s, d);
            let g = rlx_fusion::unfuse_dit_modulation(g);
            let xn = g
                .nodes()
                .iter()
                .find(|n| matches!(&n.op, rlx_ir::Op::Input { name } if name == "x"))
                .map(|n| n.id)
                .unwrap();
            let yn = g
                .nodes()
                .iter()
                .find(|n| matches!(&n.op, rlx_ir::Op::Input { name } if name == "y"))
                .map(|n| n.id)
                .unwrap();
            let gn = g
                .nodes()
                .iter()
                .find(|n| matches!(&n.op, rlx_ir::Op::Input { name } if name == "gate"))
                .map(|n| n.id)
                .unwrap();
            let unfused = grad_with_loss(&g, &[xn, yn, gn]);
            report(
                "GatedResidualBackward",
                time_ns(device, fused, &gate_feeds),
                time_ns(device, unfused, &gate_feeds),
            );
        }
    }
}
