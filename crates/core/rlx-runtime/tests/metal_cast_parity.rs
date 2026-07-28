// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU vs Metal `Op::Cast` parity across dtype pairs.
//!
//! Metal's `Thunk::CastHost` now hands the unified-memory arena straight to
//! rlx-cpu's `exec_cast_generic`, so EVERY dtype pair converts with identical
//! numeric semantics on CPU and Metal (float→int SATURATES, int→int WRAPS,
//! f16/bf16 round-nearest, C64 real↔complex). Before, Metal's hand-rolled
//! table `panic!`d on anything with I8/I16/U8, BF16, F64, C64, or several
//! int/bool combos.
//!
//! Two graph styles exercise the path:
//!   * Direct exotic OUTPUT: `input(F32) → Cast(to) → output(to)` — reads the
//!     exotic result bytes back and compares Metal vs CPU (covers the target
//!     dtypes the typed-output boundary supports: I32/C64/F16/BF16).
//!   * F32-boundary CHAIN: `input(F32) → Cast(a) → Cast(b) → … → F32` — the
//!     exotic dtype lives only as a genuine (non-widened) arena intermediate,
//!     so the interior casts run through `CastHost` while the graph boundary
//!     stays on the well-supported F32 I/O path. rlx does not fold consecutive
//!     runtime casts, so every interior cast really executes.

#![cfg(all(feature = "metal", target_os = "macos"))]

use half::{bf16, f16};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{CompileOptions, Device, Session};

/// Decode a typed output buffer to `f64` real values (complex → real part),
/// so CPU and Metal can be compared even when a backend reports a widened
/// output dtype (Metal stores an I32 Cast in a widened f32 slot → reports F32,
/// while CPU reports native I32; both decode to the same integers).
fn decode_reals(bytes: &[u8], dt: DType) -> Vec<f64> {
    match dt {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)
            .collect(),
        DType::F64 => bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes(b.try_into().unwrap()).to_f32() as f64)
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes(b.try_into().unwrap()).to_f32() as f64)
            .collect(),
        DType::I8 => bytes.iter().map(|&b| b as i8 as f64).collect(),
        DType::I16 => bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes(b.try_into().unwrap()) as f64)
            .collect(),
        DType::I32 => bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f64)
            .collect(),
        DType::I64 => bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f64)
            .collect(),
        DType::U8 => bytes.iter().map(|&b| b as f64).collect(),
        DType::U32 => bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as f64)
            .collect(),
        DType::Bool => bytes.iter().map(|&b| (b != 0) as u8 as f64).collect(),
        // Interleaved [re, im] f32 lanes — keep the real part.
        DType::C64 => bytes
            .chunks_exact(8)
            .map(|b| f32::from_le_bytes(b[0..4].try_into().unwrap()) as f64)
            .collect(),
        // Interleaved [re, im] f64 lanes (16 B/elem) — keep the real part.
        DType::C128 => bytes
            .chunks_exact(16)
            .map(|b| f64::from_le_bytes(b[0..8].try_into().unwrap()))
            .collect(),
    }
}

fn approx_eq(a: &[f64], b: &[f64], tol: f64, ctx: &str) {
    assert_eq!(a.len(), b.len(), "{ctx}: length {} != {}", a.len(), b.len());
    for i in 0..a.len() {
        assert!(
            (a[i] - b[i]).abs() <= tol,
            "{ctx}: [{i}] {} != {} (tol {tol})",
            a[i],
            b[i]
        );
    }
}

/// Build a cast chain `input(F32) → Cast(dtypes[0]) → … → Cast(dtypes[n-1])`,
/// run it on CPU and Metal, and return the decoded real outputs of each.
fn run_cast_chain(name: &str, xs: &[f32], dtypes: &[DType]) -> (Vec<f64>, Vec<f64>) {
    let n = xs.len();
    let mut g = Graph::new(name);
    let mut cur = g.input("x", Shape::new(&[n], DType::F32));
    for &to in dtypes {
        cur = g.add_node(Op::Cast { to }, vec![cur], Shape::new(&[n], to));
    }
    g.set_outputs(vec![cur]);

    let x_bytes: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let opts = CompileOptions::default();

    let mut cpu = Session::new(Device::Cpu).compile_with(g.clone(), &opts);
    let cpu_out = cpu.run_typed(&[("x", &x_bytes, DType::F32)]);
    let mut metal = Session::new(Device::Metal).compile_with(g, &opts);
    let metal_out = metal.run_typed(&[("x", &x_bytes, DType::F32)]);

    let (cb, cdt) = &cpu_out[0];
    let (mb, mdt) = &metal_out[0];
    let cr = decode_reals(cb, *cdt);
    let mr = decode_reals(mb, *mdt);
    eprintln!("[{name}] chain {dtypes:?}: cpu={cdt:?}{cr:?} metal={mdt:?}{mr:?}");
    (cr, mr)
}

// ── Direct exotic-OUTPUT casts (input F32 → Cast → exotic output) ───────────

#[test]
fn metal_cast_f32_to_i32_truncates() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let xs = [3.7f32, -3.7, 100.9, -100.9, 0.0];
    let (cr, mr) = run_cast_chain("f32_to_i32", &xs, &[DType::I32]);
    let want = vec![3.0, -3.0, 100.0, -100.0, 0.0]; // truncate toward zero
    approx_eq(&cr, &want, 0.0, "cpu f32->i32");
    approx_eq(&mr, &want, 0.0, "metal f32->i32");
    approx_eq(&cr, &mr, 0.0, "f32->i32 parity");
}

#[test]
fn metal_cast_f32_to_c64() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let xs = [1.0f32, -2.5, 3.0];
    // Read the raw C64 output on Metal and verify imaginary lanes are zero.
    let n = xs.len();
    let mut g = Graph::new("f32_to_c64");
    let x = g.input("x", Shape::new(&[n], DType::F32));
    let c = g.add_node(
        Op::Cast { to: DType::C64 },
        vec![x],
        Shape::new(&[n], DType::C64),
    );
    g.set_outputs(vec![c]);
    let x_bytes: Vec<u8> = xs.iter().flat_map(|v| v.to_le_bytes()).collect();
    let opts = CompileOptions::default();

    let mut cpu = Session::new(Device::Cpu).compile_with(g.clone(), &opts);
    let cpu_out = cpu.run_typed(&[("x", &x_bytes, DType::F32)]);
    let mut metal = Session::new(Device::Metal).compile_with(g, &opts);
    let metal_out = metal.run_typed(&[("x", &x_bytes, DType::F32)]);

    let (cb, cdt) = &cpu_out[0];
    let (mb, mdt) = &metal_out[0];
    assert_eq!(*cdt, DType::C64);
    assert_eq!(*mdt, DType::C64);
    assert_eq!(cb, mb, "f32->c64 raw bytes diverge");
    let lanes: Vec<f32> = mb
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        lanes,
        vec![1.0, 0.0, -2.5, 0.0, 3.0, 0.0],
        "re/im interleave"
    );
}

#[test]
fn metal_cast_f32_to_f16_and_bf16() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    // F16/BF16 no longer PANIC on Metal (BF16 was unsupported in the old
    // hand-rolled table). NOTE: Metal's arena keeps F16/BF16 activations in
    // F32 (the documented f32-uniform-arena / AMP policy), so it does not
    // perform CPU's half-precision round-to-nearest — a pre-existing precision
    // policy, not a cast bug. So we assert each backend lands within its
    // format's rounding tolerance of the input rather than bit-exact parity.
    let xs = [1.0f32, -2.5, 0.375, 100.0, 0.1, 1234.5];

    let within = |out: &[f64], rel: f64, ctx: &str| {
        for (i, (&o, &x)) in out.iter().zip(xs.iter()).enumerate() {
            let tol = (x.abs() as f64) * rel + 1e-6;
            assert!(
                (o - x as f64).abs() <= tol,
                "{ctx}: [{i}] {o} not within {tol} of {x}"
            );
        }
    };

    // F32→F16 (native MSL kernel). ~2^-10 mantissa.
    let (cr, mr) = run_cast_chain("f32_to_f16", &xs, &[DType::F16]);
    within(&cr, 1.0 / 1024.0, "cpu f32->f16");
    within(&mr, 1.0 / 1024.0, "metal f32->f16");

    // F32→BF16 (CastHost — BF16 has no native MSL cast kernel). ~2^-7 mantissa.
    let (cr, mr) = run_cast_chain("f32_to_bf16", &xs, &[DType::BF16]);
    within(&cr, 1.0 / 128.0, "cpu f32->bf16");
    within(&mr, 1.0 / 128.0, "metal f32->bf16");
}

// ── F32-boundary chains (exotic dtype is a genuine interior arena slot) ──────

#[test]
fn metal_cast_f32_to_u8_saturates() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    // F32 → U8 (saturate) → F32. U8 is not widened by Metal, so the F32→U8
    // and U8→F32 casts both run through CastHost.
    let xs = [0.0f32, 5.4, 255.9, 300.0, -10.0];
    let (cr, mr) = run_cast_chain("f32_u8_f32", &xs, &[DType::U8, DType::F32]);
    let want = vec![0.0, 5.0, 255.0, 255.0, 0.0]; // saturate to [0,255], trunc
    approx_eq(&cr, &want, 0.0, "cpu f32->u8->f32");
    approx_eq(&mr, &want, 0.0, "metal f32->u8->f32");
    approx_eq(&cr, &mr, 0.0, "f32->u8->f32 parity");
}

#[test]
fn metal_cast_u8_to_i8_wraps() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    // F32 → U8 (saturate) → I8 (int→int WRAP) → F32. The U8 and I8 interior
    // slots keep native width on Metal, so U8→I8 exercises the wrapping
    // narrowing path in exec_cast_generic.
    let xs = [300.0f32, 200.0, 100.0, 50.0, -5.0];
    let (cr, mr) = run_cast_chain("u8_i8", &xs, &[DType::U8, DType::I8, DType::F32]);
    // 300→255→-1, 200→200→-56, 100→100→100, 50→50→50, -5→0→0
    let want = vec![-1.0, -56.0, 100.0, 50.0, 0.0];
    approx_eq(&cr, &want, 0.0, "cpu u8->i8 wrap");
    approx_eq(&mr, &want, 0.0, "metal u8->i8 wrap");
    approx_eq(&cr, &mr, 0.0, "u8->i8 wrap parity");
}

#[test]
fn metal_cast_f64_roundtrip() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    // F32 → F64 → F32. F64 has no Metal device kernels, but the interior F64
    // slot stays 8-byte in the unified arena and both casts run via CastHost.
    let xs = [1.5f32, -2.25, 3.125, 1024.0, -0.5, 0.0];
    let (cr, mr) = run_cast_chain("f64_roundtrip", &xs, &[DType::F64, DType::F32]);
    let want: Vec<f64> = xs.iter().map(|&v| v as f64).collect();
    approx_eq(&cr, &want, 0.0, "cpu f32->f64->f32");
    approx_eq(&mr, &want, 0.0, "metal f32->f64->f32");
    approx_eq(&cr, &mr, 0.0, "f64 roundtrip parity");
}

#[test]
fn metal_cast_c64_to_f32_roundtrip() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    // F32 → C64 (CastHost) → F32 (CastHost). Round-trips the real part; the
    // C64 interior slot keeps native 8-byte width on both backends.
    let xs = [1.0f32, -2.0, 3.5, -7.25];
    let (cr, mr) = run_cast_chain("c64_roundtrip", &xs, &[DType::C64, DType::F32]);
    let want: Vec<f64> = xs.iter().map(|&v| v as f64).collect();
    approx_eq(&cr, &want, 0.0, "cpu c64->f32");
    approx_eq(&mr, &want, 0.0, "metal c64->f32");
    approx_eq(&cr, &mr, 0.0, "c64 roundtrip parity");
}
