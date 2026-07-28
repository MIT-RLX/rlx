// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Cast` dtype-pair parity for the MLX backend.
//!
//! Every case builds a Constant leaf of the source dtype, casts it to the
//! destination dtype, and reads the result back with `run_typed` (native
//! bytes + declared dtype). Expected bytes are produced by `ref_cast`,
//! which mirrors rlx-cpu's `exec_cast_generic` / `CastScalar` semantics
//! exactly (see `rlx-cpu/src/thunk/ops/elementwise.rs`), so this is a
//! true MLX-vs-rlx-cpu parity check.
//!
//! Covers: the scalar pairs (native MLX `astype`), the F64 pairs (shim
//! CPU-stream astype), and the C64 pairs (host complex cast).

#![cfg(rlx_mlx_host)]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_mlx::{MlxExecutable, MlxMode};

// ── rlx-cpu-mirroring reference cast ────────────────────────────────

#[derive(Clone, Copy)]
enum Scalar {
    F(f64),
    I(i64),
    C(f64, f64),
}

impl Scalar {
    fn real(self) -> f64 {
        match self {
            Scalar::F(f) => f,
            Scalar::I(i) => i as f64,
            Scalar::C(re, _) => re,
        }
    }
    fn truthy(self) -> bool {
        match self {
            Scalar::F(f) => f != 0.0,
            Scalar::I(i) => i != 0,
            Scalar::C(re, im) => re != 0.0 || im != 0.0,
        }
    }
}

fn decode(src: &[u8], dt: DType, n: usize) -> Vec<Scalar> {
    let mut v = Vec::with_capacity(n);
    match dt {
        DType::F32 => src
            .chunks_exact(4)
            .take(n)
            .for_each(|c| v.push(Scalar::F(f32::from_le_bytes(c.try_into().unwrap()) as f64))),
        DType::F64 => src
            .chunks_exact(8)
            .take(n)
            .for_each(|c| v.push(Scalar::F(f64::from_le_bytes(c.try_into().unwrap())))),
        DType::F16 => src.chunks_exact(2).take(n).for_each(|c| {
            v.push(Scalar::F(
                half::f16::from_le_bytes([c[0], c[1]]).to_f32() as f64
            ))
        }),
        DType::BF16 => src.chunks_exact(2).take(n).for_each(|c| {
            v.push(Scalar::F(
                half::bf16::from_le_bytes([c[0], c[1]]).to_f32() as f64
            ))
        }),
        DType::I8 => src
            .iter()
            .take(n)
            .for_each(|&b| v.push(Scalar::I(b as i8 as i64))),
        DType::I16 => src
            .chunks_exact(2)
            .take(n)
            .for_each(|c| v.push(Scalar::I(i16::from_le_bytes([c[0], c[1]]) as i64))),
        DType::I32 => src
            .chunks_exact(4)
            .take(n)
            .for_each(|c| v.push(Scalar::I(i32::from_le_bytes(c.try_into().unwrap()) as i64))),
        DType::I64 => src
            .chunks_exact(8)
            .take(n)
            .for_each(|c| v.push(Scalar::I(i64::from_le_bytes(c.try_into().unwrap())))),
        DType::U8 => src
            .iter()
            .take(n)
            .for_each(|&b| v.push(Scalar::I(b as i64))),
        DType::U32 => src
            .chunks_exact(4)
            .take(n)
            .for_each(|c| v.push(Scalar::I(u32::from_le_bytes(c.try_into().unwrap()) as i64))),
        DType::Bool => src
            .iter()
            .take(n)
            .for_each(|&b| v.push(Scalar::I((b != 0) as i64))),
        DType::C64 => src.chunks_exact(8).take(n).for_each(|c| {
            let re = f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64;
            let im = f32::from_le_bytes([c[4], c[5], c[6], c[7]]) as f64;
            v.push(Scalar::C(re, im));
        }),
        DType::C128 => src.chunks_exact(16).take(n).for_each(|c| {
            let re = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
            let im = f64::from_le_bytes([c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]]);
            v.push(Scalar::C(re, im));
        }),
    }
    v
}

fn ref_cast(src: &[u8], src_dt: DType, dst_dt: DType, n: usize) -> Vec<u8> {
    let vals = decode(src, src_dt, n);
    let mut out = Vec::new();
    macro_rules! push_int {
        ($t:ty) => {
            for s in &vals {
                let x = match *s {
                    Scalar::I(i) => i as $t,
                    Scalar::F(f) => f as $t,
                    Scalar::C(re, _) => re as $t,
                };
                out.extend_from_slice(&x.to_le_bytes());
            }
        };
    }
    match dst_dt {
        DType::F32 => vals
            .iter()
            .for_each(|s| out.extend_from_slice(&(s.real() as f32).to_le_bytes())),
        DType::F64 => vals
            .iter()
            .for_each(|s| out.extend_from_slice(&s.real().to_le_bytes())),
        DType::F16 => vals.iter().for_each(|s| {
            out.extend_from_slice(&half::f16::from_f32(s.real() as f32).to_le_bytes())
        }),
        DType::BF16 => vals.iter().for_each(|s| {
            out.extend_from_slice(&half::bf16::from_f32(s.real() as f32).to_le_bytes())
        }),
        DType::I8 => {
            for s in &vals {
                let x = match *s {
                    Scalar::I(i) => i as i8,
                    Scalar::F(f) => f as i8,
                    Scalar::C(re, _) => re as i8,
                };
                out.push(x as u8);
            }
        }
        DType::U8 => {
            for s in &vals {
                let x = match *s {
                    Scalar::I(i) => i as u8,
                    Scalar::F(f) => f as u8,
                    Scalar::C(re, _) => re as u8,
                };
                out.push(x);
            }
        }
        DType::I16 => push_int!(i16),
        DType::I32 => push_int!(i32),
        DType::I64 => push_int!(i64),
        DType::U32 => push_int!(u32),
        DType::Bool => vals.iter().for_each(|s| out.push(s.truthy() as u8)),
        DType::C64 => vals.iter().for_each(|s| {
            let (re, im) = match *s {
                Scalar::F(f) => (f as f32, 0.0f32),
                Scalar::I(i) => (i as f32, 0.0f32),
                Scalar::C(re, im) => (re as f32, im as f32),
            };
            out.extend_from_slice(&re.to_le_bytes());
            out.extend_from_slice(&im.to_le_bytes());
        }),
        DType::C128 => vals.iter().for_each(|s| {
            let (re, im) = match *s {
                Scalar::F(f) => (f, 0.0f64),
                Scalar::I(i) => (i as f64, 0.0f64),
                Scalar::C(re, im) => (re, im),
            };
            out.extend_from_slice(&re.to_le_bytes());
            out.extend_from_slice(&im.to_le_bytes());
        }),
    }
    out
}

// ── MLX driver ──────────────────────────────────────────────────────

/// Build `Constant(src_dt) -> Cast(dst_dt)` and read the result via
/// `run_typed`. `mode` lets C64 cases pin Lazy; scalar/F64 use the env
/// default (which falls back to Lazy for host-eval graphs anyway).
fn mlx_cast(src: &[u8], src_dt: DType, dst_dt: DType, n: usize, mode: Option<MlxMode>) -> Vec<u8> {
    let mut g = Graph::new("cast");
    let c = g.add_node(
        Op::Constant { data: src.to_vec() },
        vec![],
        Shape::new(&[n], src_dt),
    );
    let y = g.add_node(Op::Cast { to: dst_dt }, vec![c], Shape::new(&[n], dst_dt));
    g.set_outputs(vec![y]);
    let mut exe = match mode {
        Some(m) => MlxExecutable::compile_with_mode(g, m),
        None => MlxExecutable::compile(g),
    };
    let outs = exe.run_typed(&[]);
    assert_eq!(outs.len(), 1);
    let (bytes, dt) = outs.into_iter().next().unwrap();
    assert_eq!(dt, dst_dt, "reported output dtype mismatch");
    bytes
}

fn check(src: &[u8], src_dt: DType, dst_dt: DType, n: usize, mode: Option<MlxMode>) {
    let got = mlx_cast(src, src_dt, dst_dt, n, mode);
    let want = ref_cast(src, src_dt, dst_dt, n);
    assert_eq!(
        got, want,
        "cast {src_dt:?} -> {dst_dt:?} mismatch:\n got  {got:?}\n want {want:?}"
    );
}

// Encoders for source Constant bytes.
fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32b(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i64b(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ── scalar pairs (native MLX astype) ────────────────────────────────

#[test]
fn scalar_int_float_roundtrips() {
    let iv = [7i32, -3, 100000, 0];
    check(&i32b(&iv), DType::I32, DType::F32, 4, None);
    check(
        &f32b(&[7.0, -3.0, 100000.0, 0.0]),
        DType::F32,
        DType::I32,
        4,
        None,
    );
    check(
        &i64b(&[1i64, -2, 5_000_000_000]),
        DType::I64,
        DType::F32,
        3,
        None,
    );
}

#[test]
fn scalar_float_to_int_truncates_toward_zero() {
    // In-range values: MLX astype float->int truncates toward zero,
    // matching rlx-cpu's `f as iN`.
    check(
        &f32b(&[1.9, -1.9, 3.5, -3.5]),
        DType::F32,
        DType::I32,
        4,
        None,
    );
    check(&f32b(&[2.9, 0.1, -0.9]), DType::F32, DType::I64, 3, None);
}

#[test]
fn scalar_narrowing_and_bool() {
    check(
        &f32b(&[0.0, 1.0, -2.0, 0.0]),
        DType::F32,
        DType::Bool,
        4,
        None,
    );
    let bool_bytes = vec![1u8, 0, 1, 0];
    check(&bool_bytes, DType::Bool, DType::F32, 4, None);
    // U8 <-> F32 in range.
    check(&[0u8, 42, 255, 7], DType::U8, DType::F32, 4, None);
    check(
        &f32b(&[0.0, 42.0, 255.0, 7.0]),
        DType::F32,
        DType::U8,
        4,
        None,
    );
}

#[test]
fn scalar_half_precision_pairs() {
    // F16 bit patterns 1.0=0x3C00, 2.0=0x4000, -0.5=0xB800.
    let f16 = vec![0x00u8, 0x3C, 0x00, 0x40, 0x00, 0xB8];
    check(&f16, DType::F16, DType::F32, 3, None);
    check(&f32b(&[1.0, 2.0, -0.5]), DType::F32, DType::F16, 3, None);
    check(&f32b(&[1.0, 2.0, -0.5]), DType::F32, DType::BF16, 3, None);
}

// ── F64 pairs (shim CPU-stream astype) ──────────────────────────────

#[test]
fn f64_pairs_via_cpu_stream() {
    check(
        &f32b(&[1.5, -2.25, 4096.0, 0.0]),
        DType::F32,
        DType::F64,
        4,
        None,
    );
    let f64src: Vec<u8> = [1.5f64, -2.25, 4096.0, 0.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    check(&f64src, DType::F64, DType::F32, 4, None);
    // int -> F64 preserves large magnitudes MLX runs on the CPU stream.
    check(&i32b(&[7, -3, 1_000_000]), DType::I32, DType::F64, 3, None);
    check(&f64src, DType::F64, DType::I32, 4, None);
}

// ── C64 pairs (host complex cast) ───────────────────────────────────

#[test]
fn c64_real_to_complex() {
    // real -> C64: interleaved (v, 0.0) f32 pairs, 8 bytes/elem.
    check(
        &f32b(&[1.0, -2.0, 3.5]),
        DType::F32,
        DType::C64,
        3,
        Some(MlxMode::Lazy),
    );
    check(
        &i32b(&[4, -5, 6]),
        DType::I32,
        DType::C64,
        3,
        Some(MlxMode::Lazy),
    );
}

#[test]
fn c64_complex_to_real_and_bool() {
    // C64 source bytes: interleaved (re, im) f32 pairs.
    let c64: Vec<u8> = [(1.5f32, 2.0f32), (-3.0, 0.0), (0.0, 0.0), (7.0, -1.0)]
        .iter()
        .flat_map(|(re, im)| {
            let mut b = re.to_le_bytes().to_vec();
            b.extend_from_slice(&im.to_le_bytes());
            b
        })
        .collect();
    check(&c64, DType::C64, DType::F32, 4, Some(MlxMode::Lazy));
    check(&c64, DType::C64, DType::I32, 4, Some(MlxMode::Lazy));
    check(&c64, DType::C64, DType::Bool, 4, Some(MlxMode::Lazy));
}

#[test]
fn c64_roundtrip_real_to_c64_to_real() {
    // real -> C64 -> real chained in one graph exercises both host-cast
    // directions plus the interleaved intermediate representation, and
    // (via the env-default compile) the Compiled->Lazy host-eval fallback.
    let vals = [1.0f32, -2.0, 3.5, 0.0];
    let mut g = Graph::new("c64_roundtrip");
    let c = g.add_node(
        Op::Constant { data: f32b(&vals) },
        vec![],
        Shape::new(&[4], DType::F32),
    );
    let cx = g.add_node(
        Op::Cast { to: DType::C64 },
        vec![c],
        Shape::new(&[4], DType::C64),
    );
    let back = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![cx],
        Shape::new(&[4], DType::F32),
    );
    g.set_outputs(vec![back]);
    let mut exe = MlxExecutable::compile(g);
    let outs = exe.run_typed(&[]);
    let (bytes, dt) = &outs[0];
    assert_eq!(*dt, DType::F32);
    let got: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        got,
        vals.to_vec(),
        "C64 round-trip should recover the reals"
    );
}

// ── C128 pairs (host complex-f64 cast) ──────────────────────────────

fn c128b(pairs: &[(f64, f64)]) -> Vec<u8> {
    pairs
        .iter()
        .flat_map(|(re, im)| {
            let mut b = re.to_le_bytes().to_vec();
            b.extend_from_slice(&im.to_le_bytes());
            b
        })
        .collect()
}

#[test]
fn c128_real_to_complex() {
    // real -> C128: interleaved (v as f64, 0.0) f64 pairs, 16 bytes/elem.
    check(
        &f32b(&[1.0, -2.0, 3.5]),
        DType::F32,
        DType::C128,
        3,
        Some(MlxMode::Lazy),
    );
    check(
        &i32b(&[4, -5, 6]),
        DType::I32,
        DType::C128,
        3,
        Some(MlxMode::Lazy),
    );
    let f64src: Vec<u8> = [1.5f64, -2.25, 4096.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    check(&f64src, DType::F64, DType::C128, 3, Some(MlxMode::Lazy));
}

#[test]
fn c128_complex_to_real_and_bool() {
    // C128 source bytes: interleaved (re, im) f64 pairs.
    let c128 = c128b(&[(1.5, 2.0), (-3.0, 0.0), (0.0, 0.0), (7.0, -1.0)]);
    check(&c128, DType::C128, DType::F32, 4, Some(MlxMode::Lazy));
    check(&c128, DType::C128, DType::F64, 4, Some(MlxMode::Lazy));
    check(&c128, DType::C128, DType::I32, 4, Some(MlxMode::Lazy));
    check(&c128, DType::C128, DType::Bool, 4, Some(MlxMode::Lazy));
}

#[test]
fn c128_c64_widen_and_narrow() {
    // C64 -> C128 widens f32 components to f64; C128 -> C64 narrows back.
    let c64: Vec<u8> = [(1.5f32, 2.0f32), (-3.0, 4.0)]
        .iter()
        .flat_map(|(re, im)| {
            let mut b = re.to_le_bytes().to_vec();
            b.extend_from_slice(&im.to_le_bytes());
            b
        })
        .collect();
    check(&c64, DType::C64, DType::C128, 2, Some(MlxMode::Lazy));
    let c128 = c128b(&[(1.5, 2.0), (-3.0, 4.0)]);
    check(&c128, DType::C128, DType::C64, 2, Some(MlxMode::Lazy));
}

#[test]
fn c128_roundtrip_real_to_c128_to_real() {
    // real -> C128 -> real chained in one graph exercises both host-cast
    // directions plus the interleaved f64 intermediate representation.
    let vals = [1.0f32, -2.0, 3.5, 0.0];
    let mut g = Graph::new("c128_roundtrip");
    let c = g.add_node(
        Op::Constant { data: f32b(&vals) },
        vec![],
        Shape::new(&[4], DType::F32),
    );
    let cx = g.add_node(
        Op::Cast { to: DType::C128 },
        vec![c],
        Shape::new(&[4], DType::C128),
    );
    let back = g.add_node(
        Op::Cast { to: DType::F32 },
        vec![cx],
        Shape::new(&[4], DType::F32),
    );
    g.set_outputs(vec![back]);
    let mut exe = MlxExecutable::compile(g);
    let outs = exe.run_typed(&[]);
    let (bytes, dt) = &outs[0];
    assert_eq!(*dt, DType::F32);
    let got: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        got,
        vals.to_vec(),
        "C128 round-trip should recover the reals"
    );
}
