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

//! PLAN: F16/BF16 host I/O on CPU + Metal. Verifies `set_param_typed`
//! and `run_typed` widen typed input bytes to f32 at the host boundary.
//! The CPU and Metal arenas stay f32-uniform internally; the typed I/O
//! surface saves callers a host-side cast when their model weights or
//! activations are already in F16/BF16. The wgpu and MLX backends
//! already shipped this; this test guards the new CPU/Metal overrides
//! against regressions.

use half::{bf16, f16};
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

fn build_param_add_graph() -> Graph {
    // out = x + b, where b is a param. F32 throughout; the test
    // exercises the typed I/O surface by uploading b as F16/BF16
    // bytes and verifying the widen-on-the-boundary result matches.
    let mut g = Graph::new("typed_io_param_add");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let b = g.param("b", Shape::new(&[4], DType::F32));
    let s = g.binary(BinaryOp::Add, x, b, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![s]);
    g
}

fn build_f32_relu_graph() -> Graph {
    let mut g = Graph::new("typed_io_relu");
    let x = g.input("x", Shape::new(&[6], DType::F32));
    let r = g.add_node(
        Op::Activation(Activation::Relu),
        vec![x],
        Shape::new(&[6], DType::F32),
    );
    g.set_outputs(vec![r]);
    g
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_set_param_typed_f16_widens_to_f32_and_runs() {
    let g = build_param_add_graph();
    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile_with(g, &CompileOptions::default());

    let b_f16: Vec<f16> = vec![1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let b_bytes: Vec<u8> = b_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
    compiled.set_param_typed("b", &b_bytes, DType::F16);

    let xs: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let outs = compiled.run(&[("x", &xs)]);
    assert_eq!(outs[0], vec![11.0, 22.0, 33.0, 44.0]);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_set_param_typed_bf16_widens_to_f32() {
    let g = build_param_add_graph();
    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile_with(g, &CompileOptions::default());

    let b_bf16: Vec<bf16> = vec![1.5f32, 2.5, 3.5, 4.5]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let b_bytes: Vec<u8> = b_bf16.iter().flat_map(|h| h.to_le_bytes()).collect();
    compiled.set_param_typed("b", &b_bytes, DType::BF16);

    let xs: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
    let outs = compiled.run(&[("x", &xs)]);
    assert_eq!(outs[0], vec![2.0, 3.0, 4.0, 5.0]);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_run_typed_with_f16_input_widens_and_runs() {
    // F16 input bytes flow through `run_typed`'s widen path; output
    // dtype matches the graph's declared F32 output.
    let g = build_f32_relu_graph();
    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile_with(g, &CompileOptions::default());

    let xs: Vec<f32> = vec![-1.0, 0.0, 0.5, 1.0, 2.5, -2.0];
    let xs_f16: Vec<f16> = xs.iter().map(|&v| f16::from_f32(v)).collect();
    let xs_bytes: Vec<u8> = xs_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
    let outs = compiled.run_typed(&[("x", &xs_bytes, DType::F16)]);
    assert_eq!(outs.len(), 1);
    let (bytes, dt) = &outs[0];
    assert_eq!(
        *dt,
        DType::F32,
        "graph output is F32; run_typed reports it as such"
    );
    assert_eq!(bytes.len(), 24, "6 elems × 4 bytes per F32");
    let got: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let want: Vec<f32> = xs.iter().map(|v| v.max(0.0)).collect();
    assert_eq!(got, want);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_run_typed_with_f16_input_widens_and_runs() {
    let g = build_f32_relu_graph();
    let session = Session::new(Device::Metal);
    let mut compiled = session.compile_with(g, &CompileOptions::default());

    let xs: Vec<f32> = vec![-1.0, 0.0, 0.5, 1.0, 2.5, -2.0];
    let xs_f16: Vec<f16> = xs.iter().map(|&v| f16::from_f32(v)).collect();
    let xs_bytes: Vec<u8> = xs_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
    let outs = compiled.run_typed(&[("x", &xs_bytes, DType::F16)]);
    let (bytes, dt) = &outs[0];
    assert_eq!(*dt, DType::F32);
    assert_eq!(bytes.len(), 24);
    let got: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let want: Vec<f32> = xs.iter().map(|v| v.max(0.0)).collect();
    assert_eq!(got, want);
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "needs Thunk::CastDtype — Op::Cast currently lowers to Thunk::Copy (rlx-cpu/src/thunk.rs:671)"]
fn cpu_run_typed_narrows_f16_output_via_cast_thunk() {
    // PLAN AMP-narrowing: graph declares Relu output as F32 + an
    // explicit Cast to F16. The CPU `Op::Cast` is documented to lower
    // to `Thunk::CastDtype` (the cross-dtype cast thunk this test was
    // designed to validate), but today the CPU backend lowers Cast as
    // a plain `Thunk::Copy` — fine for same-dtype reshape-style casts,
    // but produces garbage when the dtypes differ. The test stays here
    // (marked `#[ignore]`) as the contract `Thunk::CastDtype` must
    // satisfy when it lands.
    let mut g = Graph::new("typed_io_narrow_f16");
    let x = g.input("x", Shape::new(&[6], DType::F32));
    let r = g.add_node(
        Op::Activation(Activation::Relu),
        vec![x],
        Shape::new(&[6], DType::F32),
    );
    let c = g.add_node(
        Op::Cast { to: DType::F16 },
        vec![r],
        Shape::new(&[6], DType::F16),
    );
    g.set_outputs(vec![c]);

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile_with(g, &CompileOptions::default());

    // Hand-pick values that round-trip through F16 exactly so the
    // assertion is byte-for-byte.
    let xs: Vec<f32> = vec![-1.0, 0.0, 0.5, 1.0, 2.5, -2.0];
    let xs_bytes: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let outs = compiled.run_typed(&[("x", &xs_bytes, DType::F32)]);
    let (bytes, dt) = &outs[0];
    assert_eq!(*dt, DType::F16, "graph output is F16");
    assert_eq!(bytes.len(), 12, "6 elems * 2 bytes per F16");
    let got: Vec<f16> = bytes
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]))
        .collect();
    let want: Vec<f16> = xs.iter().map(|v| f16::from_f32(v.max(0.0))).collect();
    assert_eq!(got, want);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_last_axis_broadcast_in_chain_matches_reference() {
    // PLAN L2 quality: trailing-shape broadcast in chains. Build
    // `(x[B,S,H] + bias[H]) * scale[H]` where bias and scale broadcast
    // over the leading B*S axes. The encoder sets `input_modulus[i] = H`
    // (= 4 here) for the bias/scale inputs; the kernel reads
    // `arena[off + (gid % 4)]` to tile.
    use rlx_ir::op::BinaryOp;

    let mut g = Graph::new("cpu_lastaxis_broadcast");
    // x: [2, 3, 4] → 24 elements
    let x = g.input("x", Shape::new(&[2, 3, 4], DType::F32));
    // bias / scale: [4] → 4 elements (broadcasts over [B, S])
    let bias = g.input("bias", Shape::new(&[4], DType::F32));
    let scale = g.input("scale", Shape::new(&[4], DType::F32));
    let s = Shape::new(&[2, 3, 4], DType::F32);
    let add = g.binary(BinaryOp::Add, x, bias, s.clone());
    let mul = g.binary(BinaryOp::Mul, add, scale, s);
    g.set_outputs(vec![mul]);

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile_with(g, &CompileOptions::default());

    let xs: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let bias_v = vec![10.0f32, 20.0, 30.0, 40.0];
    let scale_v = vec![1.0f32, 2.0, 3.0, 4.0];
    let outs = compiled.run(&[("x", &xs), ("bias", &bias_v), ("scale", &scale_v)]);
    // Expected: for each output element gid, val = (x[gid] + bias[gid % 4]) * scale[gid % 4]
    let want: Vec<f32> = xs
        .iter()
        .enumerate()
        .map(|(gid, &v)| (v + bias_v[gid % 4]) * scale_v[gid % 4])
        .collect();
    assert_eq!(outs[0], want);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_scalar_broadcast_in_chain_matches_reference() {
    // PLAN L2 quality: scalar broadcast in chains. Build a graph that
    // compiles down to one ElementwiseRegion with `scalar_input_mask`
    // set for the bias/scale inputs. Verify the CPU thunk's
    // interpreted chain reads element 0 of those inputs for every
    // output position, producing the expected `(x + bias) * scale`.
    use rlx_ir::op::BinaryOp;

    let mut g = Graph::new("cpu_scalar_chain");
    let x = g.input("x", Shape::new(&[6], DType::F32));
    let bias = g.input("bias", Shape::new(&[1], DType::F32));
    let scale = g.input("scale", Shape::new(&[1], DType::F32));
    let s = Shape::new(&[6], DType::F32);
    let add = g.binary(BinaryOp::Add, x, bias, s.clone());
    let mul = g.binary(BinaryOp::Mul, add, scale, s);
    g.set_outputs(vec![mul]);

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile_with(g, &CompileOptions::default());

    let xs: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias_v = 0.5f32;
    let scale_v = 2.0f32;
    let outs = compiled.run(&[("x", &xs), ("bias", &[bias_v]), ("scale", &[scale_v])]);
    let want: Vec<f32> = xs.iter().map(|v| (v + bias_v) * scale_v).collect();
    assert_eq!(outs[0], want);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_set_param_typed_f16_widens_to_f32() {
    let g = build_param_add_graph();
    let session = Session::new(Device::Metal);
    let mut compiled = session.compile_with(g, &CompileOptions::default());

    let b_f16: Vec<f16> = vec![1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let b_bytes: Vec<u8> = b_f16.iter().flat_map(|h| h.to_le_bytes()).collect();
    compiled.set_param_typed("b", &b_bytes, DType::F16);

    let xs: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let outs = compiled.run(&[("x", &xs)]);
    assert_eq!(outs[0], vec![11.0, 22.0, 33.0, 44.0]);
}
