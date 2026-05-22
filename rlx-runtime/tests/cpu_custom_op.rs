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

//! End-to-end validation of the custom-op + custom-VJP scaffold.
//!
//! Registers a toy `square` op (`y = x²`) with both the IR-level
//! [`OpExtension`] (shape inference + VJP rule) and the CPU-side
//! [`CpuKernel`] (execution), then exercises the full pipeline:
//!
//!   1. Forward: build a graph with `Op::Custom('square')`, compile
//!      via `Session::new(Device::Cpu)`, run on real f32 input,
//!      verify `y == x²` element-wise.
//!   2. Autodiff: feed the forward graph through
//!      `rlx_opt::autodiff::grad_with_loss`, compile the resulting
//!      backward graph, verify the emitted gradient matches `2x`
//!      (closed-form VJP of `x → x²`).
//!
//! `square` itself is uninteresting — `Op::Binary(Mul)` already does
//! it. The point is to prove that no rlx core code (op.rs, autodiff.rs,
//! thunk.rs) needs editing for new research ops; registration alone
//! is sufficient.

#![cfg(feature = "cpu")]

use std::sync::{Arc, Mutex, OnceLock};

use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};
use rlx_ir::infer::GraphExt; // for `g.sum`
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Node, NodeId, Op, OpExtension, Shape, VjpContext, register_op};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

// ── The custom op ──────────────────────────────────────────────────

struct SquareIr;
impl OpExtension for SquareIr {
    fn name(&self) -> &str {
        "rlx_test.square"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // Same shape and dtype as the input.
        inputs[0].clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // d/dx (x²) = 2x;  upstream * 2x is the chain rule.
        let x_bwd = ctx.fwd_map[&node.inputs[0]];
        let x_shape = ctx.bwd.node(x_bwd).shape.clone();
        let two_x = ctx.bwd.binary(BinaryOp::Add, x_bwd, x_bwd, x_shape.clone());
        let dx = ctx.bwd.binary(BinaryOp::Mul, ctx.upstream, two_x, x_shape);
        vec![(0, dx)]
    }
}

struct SquareCpu;
impl CpuKernel for SquareCpu {
    fn name(&self) -> &str {
        "rlx_test.square"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f32("square input")?;
        let out = output.expect_f32_mut("square output")?;
        if x.len() != out.len() {
            return Err(format!("len mismatch: {} != {}", x.len(), out.len()));
        }
        for (o, v) in out.iter_mut().zip(x.iter()) {
            *o = v * v;
        }
        Ok(())
    }
}

/// Register both impls exactly once across the test process. The
/// global registries persist across tests in the same binary.
fn ensure_registered() {
    static ONCE: OnceLock<Mutex<bool>> = OnceLock::new();
    let m = ONCE.get_or_init(|| Mutex::new(false));
    let mut done = m.lock().unwrap();
    if !*done {
        register_op(Arc::new(SquareIr));
        register_cpu_kernel(Arc::new(SquareCpu));
        *done = true;
    }
}

fn f32s_to_bytes(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────

#[test]
fn custom_op_forward_executes_through_cpu_pipeline() {
    ensure_registered();

    let mut g = Graph::new("square_fwd");
    let x = g.input("x", Shape::new(&[6], DType::F32));
    let y = g.custom_op("rlx_test.square", vec![], vec![x]);
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let x_data: [f32; 6] = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let outs = compiled.run_typed(&[("x", &f32s_to_bytes(&x_data), DType::F32)]);

    assert_eq!(outs.len(), 1);
    let y_got = bytes_to_f32s(&outs[0].0);
    let expected = [4.0_f32, 1.0, 0.0, 1.0, 4.0, 9.0];
    for (i, (got, want)) in y_got.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-6, "y[{i}] = {got}, want {want}");
    }
}

#[test]
fn custom_op_vjp_emits_correct_gradient() {
    ensure_registered();

    // Forward graph: y = square(x); loss = Σ y. Then dL/dx = 2x.
    let mut fwd = Graph::new("square_grad");
    let x = fwd.input("x", Shape::new(&[5], DType::F32));
    let y = fwd.custom_op("rlx_test.square", vec![], vec![x]);
    let loss = fwd.sum(y, vec![0], false); // scalar
    fwd.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&fwd, &[x]);
    // [loss, dL/dx]
    assert_eq!(bwd.outputs.len(), 2);

    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    let x_data: [f32; 5] = [-2.0, -0.5, 0.0, 1.5, 3.0];

    // d_output is the seed gradient of the scalar loss w.r.t. itself:
    // shape [] (one element), value 1.0. grad_with_loss adds it as
    // an Input named "d_output".
    let d_out: [f32; 1] = [1.0];

    let outs = compiled.run_typed(&[
        ("x", &f32s_to_bytes(&x_data), DType::F32),
        ("d_output", &f32s_to_bytes(&d_out), DType::F32),
    ]);
    assert_eq!(outs.len(), 2);

    let loss_v = bytes_to_f32s(&outs[0].0);
    let dx = bytes_to_f32s(&outs[1].0);

    // Loss = Σ x² = 4 + 0.25 + 0 + 2.25 + 9 = 15.5.
    assert_eq!(loss_v.len(), 1);
    assert!(
        (loss_v[0] - 15.5).abs() < 1e-4,
        "loss = {}, want 15.5",
        loss_v[0]
    );

    // dL/dx = 2x — straight from the VJP rule we registered.
    assert_eq!(dx.len(), 5);
    for (i, (g, x)) in dx.iter().zip(x_data.iter()).enumerate() {
        let want = 2.0 * x;
        assert!((g - want).abs() < 1e-4, "dx[{i}] = {g}, want {want}");
    }
}

// ── f64 dispatch ───────────────────────────────────────────────────

/// f64 reciprocal `y = 1/x`, registered against the IR + CPU registries.
/// Exists purely to verify the f64 codepath through `Thunk::CustomOp`.
struct ReciprocalF64;

impl OpExtension for ReciprocalF64 {
    fn name(&self) -> &str {
        "rlx_test.reciprocal_f64"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // The IR-level dtype check happens in the CPU dispatcher,
        // which will panic if dtype is not F64. We declare F64
        // explicitly here so callers can't pass F32 by accident.
        let s = inputs[0].clone();
        assert_eq!(
            s.dtype(),
            DType::F64,
            "reciprocal_f64 only supports F64 input"
        );
        s
    }
    // No vjp — non-differentiable for this test.
}

impl CpuKernel for ReciprocalF64 {
    fn name(&self) -> &str {
        "rlx_test.reciprocal_f64"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f64("reciprocal_f64 input")?;
        let out = output.expect_f64_mut("reciprocal_f64 output")?;
        for (o, v) in out.iter_mut().zip(x.iter()) {
            *o = 1.0 / v;
        }
        Ok(())
    }
}

fn ensure_f64_registered() {
    use std::sync::OnceLock;
    static ONCE: OnceLock<Mutex<bool>> = OnceLock::new();
    let m = ONCE.get_or_init(|| Mutex::new(false));
    let mut done = m.lock().unwrap();
    if !*done {
        register_op(Arc::new(ReciprocalF64));
        register_cpu_kernel(Arc::new(ReciprocalF64));
        *done = true;
    }
}

fn f64s_to_bytes(xs: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 8);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn custom_op_dispatches_f64_kernel_when_output_is_f64() {
    ensure_f64_registered();

    let mut g = Graph::new("recip_f64");
    let x = g.input("x", Shape::new(&[4], DType::F64));
    let y = g.custom_op("rlx_test.reciprocal_f64", vec![], vec![x]);
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let x_data: [f64; 4] = [1.0, 2.0, -4.0, 0.5];
    let outs = compiled.run_typed(&[("x", &f64s_to_bytes(&x_data), DType::F64)]);

    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].1, DType::F64, "output dtype must round-trip as F64");
    let y_got = bytes_to_f64s(&outs[0].0);
    let expected = [1.0, 0.5, -0.25, 2.0];
    for (i, (got, want)) in y_got.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-12, "y[{i}] = {got}, want {want}");
    }
}

// ── Multi-output via Narrow extraction ─────────────────────────────

/// "MinMax" on a 1D vector — returns [min(x), max(x)] packed as a
/// length-2 buffer. The user follows up with two `narrow_` calls to
/// extract the logical outputs. Exists to validate the documented
/// multi-output pattern + the new `custom_op_packed` builder.
struct MinMaxF32;

impl OpExtension for MinMaxF32 {
    fn name(&self) -> &str {
        "rlx_test.minmax"
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // Default static inference — also length-2. custom_op_packed
        // can override; this lets either entrypoint work.
        Shape::new(&[2], inputs[0].dtype())
    }
}

impl CpuKernel for MinMaxF32 {
    fn name(&self) -> &str {
        "rlx_test.minmax"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f32("minmax input")?;
        let out = output.expect_f32_mut("minmax output")?;
        if x.is_empty() {
            return Err("minmax: empty input".into());
        }
        if out.len() != 2 {
            return Err(format!("minmax: output must have len 2, got {}", out.len()));
        }
        let mut lo = x[0];
        let mut hi = x[0];
        for &v in &x[1..] {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        out[0] = lo;
        out[1] = hi;
        Ok(())
    }
}

fn ensure_minmax_registered() {
    use std::sync::OnceLock;
    static ONCE: OnceLock<Mutex<bool>> = OnceLock::new();
    let m = ONCE.get_or_init(|| Mutex::new(false));
    let mut done = m.lock().unwrap();
    if !*done {
        register_op(Arc::new(MinMaxF32));
        register_cpu_kernel(Arc::new(MinMaxF32));
        *done = true;
    }
}

#[test]
fn custom_op_packed_multi_output_via_narrow() {
    ensure_minmax_registered();

    let mut g = Graph::new("minmax");
    let x = g.input("x", Shape::new(&[6], DType::F32));
    // Use the packed builder so the test exercises that codepath.
    let mm = g.custom_op_packed(
        "rlx_test.minmax",
        vec![],
        vec![x],
        Shape::new(&[2], DType::F32),
    );
    let lo = g.narrow_(mm, 0, 0, 1);
    let hi = g.narrow_(mm, 0, 1, 1);
    g.set_outputs(vec![lo, hi]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let x_data: [f32; 6] = [3.0, -1.0, 7.0, 2.5, -4.0, 0.0];
    let outs = compiled.run_typed(&[("x", &f32s_to_bytes(&x_data), DType::F32)]);

    assert_eq!(outs.len(), 2, "expected [min, max] as separate outputs");
    let lo_v = bytes_to_f32s(&outs[0].0);
    let hi_v = bytes_to_f32s(&outs[1].0);
    assert_eq!(lo_v, vec![-4.0]);
    assert_eq!(hi_v, vec![7.0]);
}

// ── Unsupported-dtype path ─────────────────────────────────────────

/// Custom op registered under a fresh name whose CPU kernel
/// returns `Err` regardless of dtype. Used to verify the dispatcher
/// panics with a clear name rather than silently zeroing the output.
struct NoKernel;

impl CpuKernel for NoKernel {
    fn name(&self) -> &str {
        "rlx_test.no_kernel"
    }
    fn execute(
        &self,
        _inputs: &[CpuTensorRef<'_>],
        _output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        Err("no_kernel: deliberately unimplemented".into())
    }
}

#[test]
#[should_panic(expected = "deliberately unimplemented")]
fn custom_op_panics_with_clear_name_when_kernel_unsupported() {
    // Register an OpExtension and a CpuKernel that has no real impl.
    struct NoOp;
    impl OpExtension for NoOp {
        fn name(&self) -> &str {
            "rlx_test.no_kernel"
        }
        fn num_inputs(&self) -> usize {
            1
        }
        fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
            inputs[0].clone()
        }
    }
    register_op(Arc::new(NoOp));
    register_cpu_kernel(Arc::new(NoKernel));

    let mut g = Graph::new("no_kernel");
    let x = g.input("x", Shape::new(&[3], DType::F32));
    let y = g.custom_op("rlx_test.no_kernel", vec![], vec![x]);
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let _ = compiled.run_typed(&[("x", &f32s_to_bytes(&[1.0, 2.0, 3.0]), DType::F32)]);
}

// ── I32 host inputs ────────────────────────────────────────────────

/// Toy op `y[i] = idx[i] * scale[i]` with `idx: I32` and `scale: F64`
/// inputs and `y: F64` output. Exists to validate two pieces of the
/// downstream-facing API:
///   1. Mixed-dtype inputs flow through the `CpuTensorRef` enum
///      cleanly — no byte casts in the kernel.
///   2. I32 host inputs go through `run_typed` directly (no need to
///      bake them as `Op::Constant`).
struct ScatteredScaleI32;

impl OpExtension for ScatteredScaleI32 {
    fn name(&self) -> &str {
        "rlx_test.scattered_scale_i32"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // Output is F64 with idx's shape.
        Shape::new(&[inputs[0].num_elements().unwrap_or(0)], DType::F64)
    }
}

impl CpuKernel for ScatteredScaleI32 {
    fn name(&self) -> &str {
        "rlx_test.scattered_scale_i32"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let idx = inputs[0].expect_i32("idx")?;
        let scale = inputs[1].expect_f64("scale")?;
        let out = output.expect_f64_mut("y")?;
        if idx.len() != scale.len() || idx.len() != out.len() {
            return Err(format!(
                "len mismatch: idx={} scale={} out={}",
                idx.len(),
                scale.len(),
                out.len()
            ));
        }
        for i in 0..idx.len() {
            out[i] = (idx[i] as f64) * scale[i];
        }
        Ok(())
    }
}

fn i32s_to_bytes(xs: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[test]
fn custom_op_accepts_i32_host_inputs_via_run_typed() {
    register_op(Arc::new(ScatteredScaleI32));
    register_cpu_kernel(Arc::new(ScatteredScaleI32));

    let n = 4;
    let mut g = Graph::new("i32_input");
    let idx = g.input("idx", Shape::new(&[n], DType::I32));
    let scale = g.input("scale", Shape::new(&[n], DType::F64));
    let y = g.custom_op("rlx_test.scattered_scale_i32", vec![], vec![idx, scale]);
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let idx_data: [i32; 4] = [3, -1, 7, 2];
    let scale_data: [f64; 4] = [0.5, 4.0, 1.25, -2.0];

    let outs = compiled.run_typed(&[
        ("idx", &i32s_to_bytes(&idx_data), DType::I32),
        ("scale", &f64s_to_bytes(&scale_data), DType::F64),
    ]);

    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].1, DType::F64);
    let y_got = bytes_to_f64s(&outs[0].0);
    let want = [1.5, -4.0, 8.75, -4.0];
    for (i, (g, w)) in y_got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-12, "y[{i}] = {g}, want {w}");
    }
}

// ── basic test for the F16 dispatcher arm ──────────────────────────
//
// Validates the dispatcher's "every DType has an arm" property
// using F16 as the witness — same arm shape as BF16/I8/I16/U8/U32/Bool,
// each of which lives in `dispatch_custom_op`'s match-on-DType.
// The op's IR-level extension declares F16 inputs/outputs; the
// kernel pattern-matches on the F16 variant of `CpuTensorRef` /
// `CpuTensorMut` and reads `&[half::f16]` directly — no byte
// gymnastics.

struct AddF16;

impl OpExtension for AddF16 {
    fn name(&self) -> &str {
        "rlx_test.add_f16"
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // Trust the caller; just propagate the LHS shape (which
        // should match RHS).
        let s = inputs[0].clone();
        assert_eq!(s.dtype(), DType::F16, "add_f16 only supports F16");
        s
    }
}

impl CpuKernel for AddF16 {
    fn name(&self) -> &str {
        "rlx_test.add_f16"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f16("add_f16 a")?;
        let b = inputs[1].expect_f16("add_f16 b")?;
        let out = output.expect_f16_mut("add_f16 out")?;
        if a.len() != b.len() || a.len() != out.len() {
            return Err(format!(
                "len mismatch: a={} b={} out={}",
                a.len(),
                b.len(),
                out.len()
            ));
        }
        // Compute in f32 (half doesn't have native add ops in the
        // `half` crate — convert, add, convert back).
        for i in 0..a.len() {
            out[i] = half::f16::from_f32(a[i].to_f32() + b[i].to_f32());
        }
        Ok(())
    }
}

fn const_f16(g: &mut Graph, xs: &[half::f16]) -> NodeId {
    let mut bytes = Vec::with_capacity(xs.len() * 2);
    for &x in xs {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[xs.len()], DType::F16),
    )
}

#[test]
fn custom_op_dispatches_f16_kernel_through_full_dtype_match() {
    register_op(Arc::new(AddF16));
    register_cpu_kernel(Arc::new(AddF16));

    let n = 4;
    let a_data: Vec<half::f16> = [1.5_f32, -2.0, 3.25, 0.5]
        .iter()
        .map(|&v| half::f16::from_f32(v))
        .collect();
    let b_data: Vec<half::f16> = [0.5_f32, 4.0, -1.0, 2.0]
        .iter()
        .map(|&v| half::f16::from_f32(v))
        .collect();

    let mut g = Graph::new("add_f16");
    let a = const_f16(&mut g, &a_data);
    let b = const_f16(&mut g, &b_data);
    let y = g.custom_op("rlx_test.add_f16", vec![], vec![a, b]);
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Cpu).compile(g);
    let outs = compiled.run_typed(&[]);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].1, DType::F16);
    // Decode the F16 output bytes.
    let bytes = &outs[0].0;
    assert_eq!(bytes.len(), n * 2);
    let y: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    let want = [2.0_f32, 2.0, 2.25, 2.5];
    for (i, (got, w)) in y.iter().zip(want.iter()).enumerate() {
        // F16 mantissa: ~3 decimal digits. 1e-2 absolute is fine
        // for this range.
        assert!((got - w).abs() < 1e-2, "y[{i}] = {got}, want {w}");
    }
}
