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

//! KV-cache style stateful inference demo.
//!
//! Demonstrates the BufferHandle API: a "running counter" graph that
//! reads its previous output as input. Each call uses the previous
//! state as input; the output becomes the next state.
//!
//! This is the primitive that real KV-caches build on: treat one of
//! the graph's inputs as a persistent buffer, run the graph, copy the
//! output back into that buffer, repeat.
//!
//! The minimal case is a counter: `state = state + delta`. After N
//! calls with delta=1, state becomes [N, N, N, N].
//!
//! cargo run --example kv_cache --release --features "cpu,blas-accelerate"

#[cfg(feature = "cpu")]
fn main() {
    use rlx_ir::*;
    use rlx_runtime::{BufferHandle, Device, Session};

    // Graph: out = state + delta  (where `state` is a persistent handle)
    let mut g = Graph::new("counter");
    let state = g.input("state", Shape::new(&[4], DType::F32));
    let delta = g.input("delta", Shape::new(&[4], DType::F32));
    let out = g.binary(
        op::BinaryOp::Add,
        state,
        delta,
        Shape::new(&[4], DType::F32),
    );
    g.set_outputs(vec![out]);

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(g);

    // Declare a persistent buffer named "state".
    // Shape stays consistent across all run() calls.
    let _kv = BufferHandle::new("state", &[4], DType::F32);

    // Initialize state to zero. bind_handle stores it outside the arena.
    let mut state_data = vec![0.0f32; 4];
    compiled.bind_handle("state", &state_data);

    // Tell the runtime: also sync output 0 → handle "out0" (positional convention).
    compiled.bind_handle("out0", &state_data);

    println!("Iteration | state");
    println!("----------+----------------");

    for step in 0..5 {
        let delta = vec![1.0f32; 4];
        let out = compiled.run(&[("delta", &delta)]);
        // Output is what we'd manually feed back; the runtime ALSO synced
        // it to handle "out0". Pull state out and feed it back as the
        // next iteration's "state" input.
        state_data = out[0].clone();
        compiled.bind_handle("state", &state_data);

        println!("step {:3}  | {:?}", step, state_data);
    }

    // Verify: 5 increments of [1,1,1,1] starting from zero → [5,5,5,5]
    assert_eq!(state_data, vec![5.0; 4]);
    println!("\n✓ persistent state survives across run() calls");
    println!("✓ each run reads bound handle as input, output flows to next iter");

    // The handle is stored in the executable, accessible via read_handle:
    let read_back = compiled.read_handle("state").expect("handle should exist");
    println!("\nread_handle(\"state\") = {:?}", read_back);
    assert_eq!(read_back, state_data);
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("requires --features cpu");
}
