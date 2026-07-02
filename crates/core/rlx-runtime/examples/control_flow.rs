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

//! Demonstrates sub-graph compilation + execution via SubgraphCache.
//!
//! Builds two simple graphs and runs them through the same backend
//! at different times. The key point: the backend doesn't need to
//! know about Op::If / Op::While ahead of time — the runtime helper
//! recursively compiles the sub-graphs as needed.
//!
//! Once Op::If / Op::While are wired into the thunk executor, the
//! pattern below will be hidden inside the backend; this example
//! just shows the mechanism is in place.
//!
//! cargo run --example control_flow --release \
//!   --features "cpu,blas-accelerate"

#[cfg(feature = "cpu")]
fn main() {
    use rlx_ir::*;
    use rlx_runtime::backend::Backend;
    use rlx_runtime::{CompileOptions, Device, Session, SubgraphCache, run_if};

    // Constants encoded as [4]-shaped vectors — the runtime's
    // BinaryFull thunk doesn't yet broadcast scalars, so we use
    // full-shape constants (broadcast support is a separate task).
    let const4 = |v: f32| -> Vec<u8> {
        let mut out = Vec::with_capacity(16);
        for _ in 0..4 {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    };

    // ── "then" sub-graph: y = x + 10 (broadcast via [4]) ───
    let then_branch = {
        let mut g = Graph::new("then");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let c = g.add_node(
            Op::Constant { data: const4(10.0) },
            vec![],
            Shape::new(&[4], DType::F32),
        );
        let y = g.binary(op::BinaryOp::Add, x, c, Shape::new(&[4], DType::F32));
        g.set_outputs(vec![y]);
        g
    };

    // ── "else" sub-graph: y = x * 2 ────────────────────────
    let else_branch = {
        let mut g = Graph::new("else");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let c = g.add_node(
            Op::Constant { data: const4(2.0) },
            vec![],
            Shape::new(&[4], DType::F32),
        );
        let y = g.binary(op::BinaryOp::Mul, x, c, Shape::new(&[4], DType::F32));
        g.set_outputs(vec![y]);
        g
    };

    let _session = Session::new(Device::Cpu);
    let backend: Box<dyn Backend> = Box::new(rlx_runtime::backend::cpu_backend::CpuBackend);
    let mut cache = SubgraphCache::new(CompileOptions::new());

    let x_data = vec![1.0, 2.0, 3.0, 4.0];

    let then_result = run_if(
        &mut cache,
        backend.as_ref(),
        1.0,
        &then_branch,
        &else_branch,
        &[("x", &x_data)],
    );
    let else_result = run_if(
        &mut cache,
        backend.as_ref(),
        0.0,
        &then_branch,
        &else_branch,
        &[("x", &x_data)],
    );

    println!("then (predicate=1): x + 10 = {:?}", then_result[0]);
    println!("else (predicate=0): x * 2  = {:?}", else_result[0]);

    assert_eq!(then_result[0], vec![11.0, 12.0, 13.0, 14.0]);
    assert_eq!(else_result[0], vec![2.0, 4.0, 6.0, 8.0]);
    println!("✓ sub-graph execution works through SubgraphCache");

    // Run again — sub-graphs are cached, no recompile.
    let r2 = run_if(
        &mut cache,
        backend.as_ref(),
        1.0,
        &then_branch,
        &else_branch,
        &[("x", &x_data)],
    );
    assert_eq!(r2, then_result);
    println!("✓ cached sub-graph reuse works (second invocation)");
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("requires --features cpu");
}
