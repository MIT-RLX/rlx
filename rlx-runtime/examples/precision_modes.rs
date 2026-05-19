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

//! Precision policy works identically across all execution modes.
//!
//! Demonstrates:
//!   1. AOT compile path (build Graph manually, compile, run)
//!   2. JIT trace path  (record ops, get a graph, compile, run)
//!   3. Same `PrecisionPolicy::AutoMixed` applied to both
//!
//! cargo run --example precision_modes --release \
//!   --features "cpu,metal,blas-accelerate"

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::*;
    use rlx_runtime::{Device, Precision, PrecisionPolicy, Session};

    let w_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let b_data = vec![0.5, -0.5, 0.0];
    let x_data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    // ── Mode 1: AOT compile (manual graph build) ──────────────
    let aot_graph = || {
        let mut g = Graph::new("aot");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 3], DType::F32));
        let b = g.param("b", Shape::new(&[3], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
        let bias = g.binary(op::BinaryOp::Add, mm, b, Shape::new(&[2, 3], DType::F32));
        let out = g.activation(op::Activation::Gelu, bias, Shape::new(&[2, 3], DType::F32));
        g.set_outputs(vec![out]);
        g
    };

    // ── Mode 2: JIT trace ─────────────────────────────────────
    let jit_graph = || {
        use rlx_runtime::trace::trace;
        trace("jit", |t| {
            let x = t.input("x", &[2, 4], DType::F32);
            let w = t.param("w", &[4, 3], DType::F32);
            let b = t.param("b", &[3], DType::F32);
            let mm = t.matmul(x, w);
            let bias = mm + b;
            let out = bias.gelu();
            vec![out]
        })
    };

    // Debug: dump the rewritten graph
    {
        use rlx_opt::pass::Pass;
        let g = aot_graph();
        println!("Original graph:");
        for n in g.nodes() {
            println!(
                "  [{}] {} {:?} → {:?}",
                n.id,
                n.op,
                n.inputs,
                n.shape.dtype()
            );
        }
        let pass = rlx_opt::AutoMixedPrecision::new(PrecisionPolicy::AutoMixed);
        let g2 = pass.run(g);
        println!("\nAfter AutoMixedPrecision:");
        for n in g2.nodes() {
            println!(
                "  [{}] {} {:?} → {:?}",
                n.id,
                n.op,
                n.inputs,
                n.shape.dtype()
            );
        }
        println!("  outputs: {:?}", g2.outputs);
        println!();
    }

    println!("Comparing AOT vs JIT × precision policies × devices:\n");
    println!(
        "{:<6} {:<6} {:<12}  {:?}",
        "mode", "device", "policy", "output"
    );
    println!("{}", "-".repeat(80));

    let modes: [(&str, fn() -> Graph); 2] = [("AOT", aot_graph), ("JIT", jit_graph)];
    for &(mode, build_graph) in &modes {
        for &(dev_name, dev) in &[("CPU", Device::Cpu), ("Metal", Device::Metal)] {
            for (policy_name, policy) in [
                ("F32", PrecisionPolicy::AlwaysF32),
                ("AutoMixed", PrecisionPolicy::AutoMixed),
            ] {
                let session = Session::new_with_precision(dev, Precision::F32).with_policy(policy);
                let mut compiled = session.compile(build_graph());
                compiled.set_param("w", &w_data);
                compiled.set_param("b", &b_data);
                let out = compiled.run(&[("x", &x_data)]);
                println!(
                    "{:<6} {:<6} {:<12}  {:?}",
                    mode, dev_name, policy_name, out[0]
                );
            }
        }
    }

    println!("\nAll modes use the same Session + PrecisionPolicy API — the");
    println!("AutoMixedPrecision pass runs as a graph rewrite regardless of");
    println!("how the graph was built (AOT, JIT, or proc-macro AOT).");
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("requires --features metal on macOS");
}
