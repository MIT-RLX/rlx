// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `ChainStep::Cast` *inside* a fused `Op::ElementwiseRegion` must convert
//! the value, not pass it through.
//!
//! The region evaluator carries every intermediate as `f32`, which made it
//! tempting to treat a fused cast as a no-op ("chains are same-dtype"). That
//! holds only for the chain's *last* step, where the output store applies the
//! dtype for us. A cast in the middle — whose result feeds a later step — was
//! silently dropped, so `Cast(I32)` kept its fraction and `Cast(F16)` kept full
//! f32 precision. `rlx-wgpu` implements the same step properly
//! (`tests/cast_parity.rs`), so CPU and GPU disagreed on identical graphs.

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::op::{Activation, BinaryOp, ChainOperand, ChainStep, RegionPrologue};
use rlx_ir::{DType, Graph, Op, Shape};

const XS: [f32; 6] = [-2.5, 3.7, 5.9, -0.1, 8.8, 0.0];

/// relu(x) → cast → cast + cast. The trailing `Add` is what forces the cast to
/// be a *middle* step, so the output store cannot stand in for it.
fn build(cast_to: DType, out_dtype: DType) -> Graph {
    let n = XS.len();
    let mut g = Graph::new("mid_chain_cast");
    let x = g.input("x", Shape::new(&[n], DType::F32));
    let region = g.add_node(
        Op::ElementwiseRegion {
            chain: vec![
                ChainStep::Activation(Activation::Relu, ChainOperand::Input(0)),
                ChainStep::Cast(cast_to, ChainOperand::Step(0)),
                ChainStep::Binary(BinaryOp::Add, ChainOperand::Step(1), ChainOperand::Step(1)),
            ],
            num_inputs: 1,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue: RegionPrologue::None,
            prologue_input: 0,
        },
        vec![x],
        Shape::new(&[n], out_dtype),
    );
    g.set_outputs(vec![region]);
    g
}

/// Runs the graph and reads the output slot as `f32` lanes. `plan_memory` is
/// f32-uniform here, so even an `I32`-typed region output is stored as f32 —
/// the cast shows up in the *value*, not the storage width.
fn run(g: &Graph) -> Vec<f32> {
    let plan = rlx_opt::memory::plan_memory(g);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(g, &arena);
    for node in g.nodes() {
        if let Op::Input { name } = &node.op {
            assert_eq!(name, "x");
            let off = arena.byte_offset(node.id);
            let buf = arena.raw_buf_mut();
            for (i, v) in XS.iter().enumerate() {
                buf[off + i * 4..off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let off = arena.byte_offset(g.outputs[0]);
    let buf = arena.raw_buf();
    (0..XS.len())
        .map(|i| f32::from_le_bytes(buf[off + i * 4..off + i * 4 + 4].try_into().unwrap()))
        .collect()
}

#[test]
fn mid_chain_int_cast_truncates_before_the_next_step() {
    let got = run(&build(DType::I32, DType::I32));
    // relu → [0, 3.7, 5.9, 0, 8.8, 0]; trunc → [0, 3, 5, 0, 8, 0]; +itself.
    // Dropping the cast would add 3.7+3.7 = 7.4 and store 7, not 6.
    assert_eq!(
        got,
        vec![0.0, 6.0, 10.0, 0.0, 16.0, 0.0],
        "mid-chain Cast(I32) did not truncate before the Add"
    );
}

#[test]
fn mid_chain_half_cast_rounds_before_the_next_step() {
    // 5.9 has no exact f16 form: the nearest half is 5.8984375, so doubling
    // after a real cast lands on 11.796875 instead of f32's 11.8.
    let got = run(&build(DType::F16, DType::F32));
    let want = 2.0 * half::f16::from_f32(5.9).to_f32();
    assert!(
        (got[2] - want).abs() < 1e-6,
        "mid-chain Cast(F16) did not round: got {got:?}, want [2] == {want}"
    );
    assert!(
        (got[2] - 11.8).abs() > 1e-4,
        "mid-chain Cast(F16) kept f32 precision: {got:?}"
    );
}

/// A region whose operands are *packed* F16 (2 bytes/lane, as
/// `plan_memory_native_in_order` assigns them) must address them at their real
/// width. Reading a fixed 4-byte f32 lane out of a packed half tensor walks off
/// by a factor of two and returns garbage; writing one overruns the slot.
#[test]
fn packed_f16_region_reads_and_writes_half_lanes() {
    use rlx_ir::op::BinaryOp;

    let n = XS.len();
    let mut g = Graph::new("packed_f16_region");
    let x = g.input("x", Shape::new(&[n], DType::F16));
    let region = g.add_node(
        Op::ElementwiseRegion {
            chain: vec![
                ChainStep::Activation(Activation::Relu, ChainOperand::Input(0)),
                ChainStep::Binary(BinaryOp::Add, ChainOperand::Step(0), ChainOperand::Step(0)),
            ],
            num_inputs: 1,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue: RegionPrologue::None,
            prologue_input: 0,
        },
        vec![x],
        Shape::new(&[n], DType::F16),
    );
    g.set_outputs(vec![region]);

    // Native widths: this is the plan the CPU backend actually runs.
    let plan = rlx_opt::memory::plan_memory_native_in_order(&g, 64);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);

    let in_off = arena.byte_offset(x);
    let out_off = arena.byte_offset(g.outputs[0]);
    assert_eq!(
        arena.byte_size(x),
        n * 2,
        "input slot must be packed at 2 bytes/lane for this test to mean anything"
    );

    {
        let buf = arena.raw_buf_mut();
        for (i, v) in XS.iter().enumerate() {
            let bits = half::f16::from_f32(*v).to_bits();
            buf[in_off + i * 2..in_off + i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());

    let buf = arena.raw_buf();
    let got: Vec<f32> = (0..n)
        .map(|i| {
            let b = [buf[out_off + i * 2], buf[out_off + i * 2 + 1]];
            half::f16::from_le_bytes(b).to_f32()
        })
        .collect();
    let want: Vec<f32> = XS
        .iter()
        .map(|v| {
            let r = half::f16::from_f32(*v).to_f32().max(0.0);
            half::f16::from_f32(r + r).to_f32()
        })
        .collect();
    assert_eq!(got, want, "packed f16 region: got {got:?}, want {want:?}");
}
