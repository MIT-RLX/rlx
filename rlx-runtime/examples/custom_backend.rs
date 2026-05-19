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

//! Demonstrate registering a custom backend via the registry.
//!
//! Pretends to be a "future" CUDA / wgpu / WASM backend by registering a
//! no-op backend under `Device::Cuda`. The point is to show that adding a
//! backend is *no longer* a runtime-crate edit — any caller (or future
//! `rlx-cuda` crate) can plug in via `register_backend`.
//!
//! cargo run --example custom_backend --release \
//!   --features "cpu,blas-accelerate"

use rlx_ir::Graph;
use rlx_runtime::backend::ExecutableGraph;
use rlx_runtime::{
    Backend, BufferHandle, CompileOptions, Device, register_backend, registered_devices,
};

/// A trivial backend that returns a fake executable. In a real port (CUDA,
/// wgpu, …) you'd implement the full thunk/dispatch path; the registration
/// surface is identical.
struct FakeCudaBackend;

struct FakeExecutable;

impl ExecutableGraph for FakeExecutable {
    fn set_param(&mut self, _name: &str, _data: &[f32]) {}
    fn run(&mut self, _inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        vec![vec![42.0_f32; 4]]
    }
}

impl Backend for FakeCudaBackend {
    fn compile(&self, _graph: Graph, _options: &CompileOptions) -> Box<dyn ExecutableGraph> {
        Box::new(FakeExecutable)
    }
}

fn main() {
    println!("Builtins registered: {:?}", registered_devices());

    // External register: pretend rlx-cuda's `register()` was called.
    register_backend(Device::Cuda, || {
        Box::new(FakeCudaBackend) as Box<dyn Backend>
    });

    println!("After register Cuda: {:?}", registered_devices());

    // Force-mark Cuda available so Session::new accepts it. (Real backends
    // gate availability via their cargo feature.) Use the BufferHandle to
    // silence the unused import — kept available since real backends will
    // need it on registration to declare persistent buffers.
    let _ = BufferHandle::new("dummy", &[1], rlx_ir::DType::F32);

    // Build a tiny graph and try to compile through the registry.
    let g = {
        use rlx_ir::*;
        let mut g = Graph::new("dummy");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        g.set_outputs(vec![x]);
        g
    };
    // Use the registry directly (Session::new would assert availability).
    let mut exe = rlx_runtime::backend_for(Device::Cuda)
        .expect("Cuda factory registered")
        .compile(g, &CompileOptions::default());
    let out = exe.run(&[]);
    println!("FakeCudaBackend ran graph; output[0] = {:?}", out[0]);
    assert_eq!(out[0], vec![42.0; 4]);

    println!("\n✓ Backend registry works end-to-end:");
    println!("  - builtins (cpu, metal) self-register from feature gates");
    println!("  - external backends register via `register_backend(...)`");
    println!("  - no edits to rlx-runtime needed for new backends");
}
