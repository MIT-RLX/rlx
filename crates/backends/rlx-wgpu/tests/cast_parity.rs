// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU-vs-wgpu parity for numeric `Op::Cast` on the f32-uniform arena.
//!
//! The wgpu arena stores every tensor as f32, so a Cast is a per-element
//! re-encode: float→int truncates toward zero and saturates; →Bool stores
//! `value != 0`; int→float / same-kind is value-preserving. This asserts the
//! wgpu forward output matches the `rlx-cpu` reference (which uses native int
//! storage) for the representable dtype pairs, and that F64/C64 casts — which
//! have no f32-arena storage — are cleanly rejected.
//!
//! Runs only when a wgpu device is present; otherwise a graceful no-op.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

/// Decode the first `n` elements of `rlx-cpu` `run_typed` output bytes (native
/// dst dtype) into f32 so it is directly comparable to the wgpu f32 arena
/// readback. The returned slot buffer may be larger than `n` elements (arena
/// slot padding / reuse), so slice to the meaningful prefix.
fn decode_to_f32(bytes: &[u8], dt: DType, n: usize) -> Vec<f32> {
    match dt {
        DType::F32 => bytes[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        DType::I32 => bytes[..n * 4]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::U8 | DType::Bool => bytes[..n].iter().map(|&b| b as f32).collect(),
        other => panic!("decode_to_f32: unhandled dtype {other:?}"),
    }
}

/// rlx-cpu reference: `Cast(input : src_dt → dst_dt)`, decoded to f32.
fn cpu_cast(input_bytes: Vec<u8>, src_dt: DType, dst_dt: DType, n: usize) -> Vec<f32> {
    use rlx::prelude::*;
    let mut g = Graph::new("cast_ref");
    let x = g.input("x", Shape::new(&[n], src_dt));
    let y = g.add_node(Op::Cast { to: dst_dt }, vec![x], Shape::new(&[n], dst_dt));
    g.set_outputs(vec![y]);
    let mut c = Session::new(Device::Cpu).compile(g);
    let outs = c.run_typed(&[("x", &input_bytes, src_dt)]);
    decode_to_f32(&outs[0].0, outs[0].1, n)
}

/// wgpu output for `Cast(input : src_dt → dst_dt)`. The wgpu run API is f32; we
/// feed the numeric values (integers-as-f32 for int sources) and read back the
/// f32 arena, which already holds int-valued / bool-valued / float results.
fn wgpu_cast(input_f32: &[f32], src_dt: DType, dst_dt: DType) -> Vec<f32> {
    let n = input_f32.len();
    let mut g = Graph::new("cast_wgpu");
    let x = g.input("x", Shape::new(&[n], src_dt));
    let y = g.add_node(Op::Cast { to: dst_dt }, vec![x], Shape::new(&[n], dst_dt));
    g.set_outputs(vec![y]);
    let mut exe = WgpuExecutable::compile(g);
    exe.run(&[("x", input_f32)]).into_iter().next().unwrap()
}

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32_bytes(xs: &[i32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "len mismatch cpu={a:?} gpu={b:?}");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn cast_f32_to_i32_truncates_toward_zero() {
    if !rlx_wgpu::is_available() {
        eprintln!("[cast_parity] no wgpu device — skipping f32→i32");
        return;
    }
    let xs = [3.7f32, -3.7, 2.0, -2.9, 0.4, -0.4, 100.99, -100.99];
    let cpu = cpu_cast(f32_bytes(&xs), DType::F32, DType::I32, xs.len());
    let gpu = wgpu_cast(&xs, DType::F32, DType::I32);
    assert_eq!(cpu, vec![3.0, -3.0, 2.0, -2.0, 0.0, 0.0, 100.0, -100.0]);
    assert_eq!(gpu, cpu, "f32→i32 wgpu {gpu:?} vs cpu {cpu:?}");
}

#[test]
fn cast_f32_to_u8_saturates() {
    if !rlx_wgpu::is_available() {
        eprintln!("[cast_parity] no wgpu device — skipping f32→u8");
        return;
    }
    // Out-of-range (300.7, -5, 400) saturate to [0,255]; in-range truncate.
    let xs = [300.7f32, -5.0, 255.9, 128.5, 0.0, 1.9, 400.0, -0.9];
    let cpu = cpu_cast(f32_bytes(&xs), DType::F32, DType::U8, xs.len());
    let gpu = wgpu_cast(&xs, DType::F32, DType::U8);
    assert_eq!(cpu, vec![255.0, 0.0, 255.0, 128.0, 0.0, 1.0, 255.0, 0.0]);
    assert_eq!(gpu, cpu, "f32→u8 wgpu {gpu:?} vs cpu {cpu:?}");
}

#[test]
fn cast_f32_to_bool_is_nonzero() {
    if !rlx_wgpu::is_available() {
        eprintln!("[cast_parity] no wgpu device — skipping f32→bool");
        return;
    }
    let xs = [0.0f32, 3.5, -2.0, 0.0, 1.0, -0.0, 1e-9];
    let cpu = cpu_cast(f32_bytes(&xs), DType::F32, DType::Bool, xs.len());
    let gpu = wgpu_cast(&xs, DType::F32, DType::Bool);
    assert_eq!(cpu, vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    assert_eq!(gpu, cpu, "f32→bool wgpu {gpu:?} vs cpu {cpu:?}");
}

#[test]
fn cast_i32_to_f32_is_value_preserving() {
    if !rlx_wgpu::is_available() {
        eprintln!("[cast_parity] no wgpu device — skipping i32→f32");
        return;
    }
    let ints = [1i32, -7, 42, 0, 1000, -32768];
    let cpu = cpu_cast(i32_bytes(&ints), DType::I32, DType::F32, ints.len());
    let as_f32: Vec<f32> = ints.iter().map(|&v| v as f32).collect();
    let gpu = wgpu_cast(&as_f32, DType::I32, DType::F32);
    assert_eq!(cpu, as_f32);
    assert_eq!(gpu, cpu, "i32→f32 wgpu {gpu:?} vs cpu {cpu:?}");
}

/// Cast fused after arithmetic — exercises the ElementwiseRegion chain path
/// (`apply_cast` in elementwise_region.wgsl) when the optimizer fuses; correct
/// either way. `mul(x, 2.5)` then truncate to i32.
#[test]
fn cast_fused_after_mul_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("[cast_parity] no wgpu device — skipping fused cast");
        return;
    }
    use rlx_ir::op::BinaryOp;
    let xs = [1.0f32, 2.0, 3.0, -1.5, 0.3, 4.9];
    let build = |g: &mut Graph| {
        let x = g.input("x", Shape::new(&[xs.len()], DType::F32));
        let two = g.add_node(
            Op::Constant {
                data: f32_bytes(&vec![2.5f32; xs.len()]),
            },
            vec![],
            Shape::new(&[xs.len()], DType::F32),
        );
        let prod = g.binary(BinaryOp::Mul, x, two, Shape::new(&[xs.len()], DType::F32));
        let c = g.add_node(
            Op::Cast { to: DType::I32 },
            vec![prod],
            Shape::new(&[xs.len()], DType::I32),
        );
        g.set_outputs(vec![c]);
    };

    // CPU reference.
    let cpu = {
        use rlx::prelude::*;
        let mut g = Graph::new("fused_ref");
        build(&mut g);
        let mut c = Session::new(Device::Cpu).compile(g);
        let outs = c.run_typed(&[("x", &f32_bytes(&xs), DType::F32)]);
        decode_to_f32(&outs[0].0, outs[0].1, xs.len())
    };
    // wgpu.
    let gpu = {
        let mut g = Graph::new("fused_wgpu");
        build(&mut g);
        let mut exe = WgpuExecutable::compile(g);
        exe.run(&[("x", &xs)]).into_iter().next().unwrap()
    };
    // mul(x,2.5)=[2.5,5,7.5,-3.75,0.75,12.25] → trunc = [2,5,7,-3,0,12]
    assert_eq!(cpu, vec![2.0, 5.0, 7.0, -3.0, 0.0, 12.0]);
    assert!(
        max_abs(&cpu, &gpu) == 0.0,
        "fused cast wgpu {gpu:?} vs cpu {cpu:?}"
    );
}

/// Explicit `Op::ElementwiseRegion` with a fused `Cast` chain step, exercising
/// `apply_cast` in elementwise_region.wgsl (the fused-region path that upstream
/// rlx-opt produces). Chain: relu(x) then truncate→i32. Verifies the region
/// kernel does the real cast, not the old identity.
#[test]
fn region_fused_relu_then_cast_i32() {
    if !rlx_wgpu::is_available() {
        eprintln!("[cast_parity] no wgpu device — skipping region cast");
        return;
    }
    use rlx_ir::op::{Activation, ChainOperand, ChainStep, RegionPrologue};
    let xs = [-2.5f32, 3.7, 5.9, -0.1, 8.8, 0.0];
    let n = xs.len();
    let mut g = Graph::new("region_cast");
    let x = g.input("x", Shape::new(&[n], DType::F32));
    let region = g.add_node(
        Op::ElementwiseRegion {
            chain: vec![
                ChainStep::Activation(Activation::Relu, ChainOperand::Input(0)),
                ChainStep::Cast(DType::I32, ChainOperand::Step(0)),
            ],
            num_inputs: 1,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue: RegionPrologue::None,
            prologue_input: 0,
        },
        vec![x],
        Shape::new(&[n], DType::I32),
    );
    g.set_outputs(vec![region]);
    let mut exe = WgpuExecutable::compile(g);
    let gpu = exe.run(&[("x", &xs)]).into_iter().next().unwrap();
    // relu = [0, 3.7, 5.9, 0, 8.8, 0] → trunc→i32 = [0, 3, 5, 0, 8, 0]
    assert_eq!(
        gpu,
        vec![0.0, 3.0, 5.0, 0.0, 8.0, 0.0],
        "region cast gpu {gpu:?}"
    );
}

/// F64 casts have no f32-arena storage and this backend performs no upstream
/// demotion — compiling one must panic (cleanly rejected, not silent garbage).
#[test]
fn cast_to_f64_is_rejected() {
    if !rlx_wgpu::is_available() {
        eprintln!("[cast_parity] no wgpu device — skipping f64 rejection");
        return;
    }
    let r = std::panic::catch_unwind(|| {
        let mut g = Graph::new("cast_f64");
        let x = g.input("x", Shape::new(&[4usize], DType::F32));
        let y = g.add_node(
            Op::Cast { to: DType::F64 },
            vec![x],
            Shape::new(&[4usize], DType::F64),
        );
        g.set_outputs(vec![y]);
        let _ = WgpuExecutable::compile(g);
    });
    assert!(r.is_err(), "expected wgpu to reject a F32→F64 cast");
}
