// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A low-precision boundary must convert or refuse — never reinterpret.
//!
//! `run(&[(name, &[f32])])` and `set_param(name, &[f32])` are the f32-shaped
//! entry points, which is right for the overwhelmingly common case and wrong
//! for an `F16`/`BF16` tensor. The bytes used to go to the backend verbatim and
//! be read as pairs of halves — measured on CPU:
//!
//! ```text
//! [1,2,3,4] into a BF16 input, + 0  ->  [2.0038757, 4.007843, 0.0, 0.0]
//! [1,2,3,4] into an F16 input,  + 0  ->  [2.003662, 513.03125, 0.0, 0.0]
//! (the tests below add 0.5, so the expected answer is [1.5, 2.5, 3.5, 4.5])
//! ```
//!
//! No error, no panic, plausible-looking numbers. What changed:
//!
//! * `CompiledGraph` records each boundary's declared dtype and narrows before
//!   handing over, routing through the typed path;
//! * `write_input` / `read_output` resolve a node's **storage** dtype from the
//!   width the plan assigned rather than from its declared dtype — the rule the
//!   fused regions already followed via `LaneKind::resolve`;
//! * and the root cause: `cpu_low_precision::promote_to_f32` promoted the *ops*
//!   to F32 while keeping every `Input` / `Param` at its declared dtype, with no
//!   `Cast` between them — so a BF16 input got a 2 B/elem slot that the
//!   promoted f32 thunk read 4 B/elem from. A boundary now keeps its low
//!   precision only while every consumer does too, which is the case that rule
//!   exists for (`KvAppend` aliases its cache and strides from the node dtype).

use rlx_ir::{DType, Graph, Shape, op::BinaryOp};
use rlx_runtime::{Device, Session};

/// `[1,2,3,4] + 0.5` in `dtype`, fed as f32 either way.
fn add_in(dtype: DType) -> Vec<f32> {
    let s = Shape::new(&[4], dtype);
    let mut g = Graph::new("boundary");
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let o = g.binary(BinaryOp::Add, x, y, s.clone());
    g.set_outputs(vec![o]);
    let xs = vec![1.0f32, 2.0, 3.0, 4.0];
    let ys = vec![0.5f32; 4];
    Session::new(Device::Cpu)
        .compile(g)
        .run(&[("x", &xs), ("y", &ys)])
        .pop()
        .expect("one output")
}

#[test]
fn an_f32_boundary_is_untouched() {
    // The common path must keep its direct route — the narrowing machinery
    // only engages when a boundary is actually narrower than f32.
    assert_eq!(add_in(DType::F32), vec![1.5, 2.5, 3.5, 4.5]);
}

/// The values are exactly representable in both half formats, so this asserts
/// exact equality: a tolerance would hide the signature it exists to catch,
/// `[2.5, 4.5, 0.0, 0.0]` — right magnitude, wrong numbers, and only two of the
/// four elements real.
#[test]
fn a_half_activation_boundary_round_trips() {
    assert_eq!(add_in(DType::BF16), vec![1.5, 2.5, 3.5, 4.5]);
    assert_eq!(add_in(DType::F16), vec![1.5, 2.5, 3.5, 4.5]);
}

/// The element count is what survives regardless of precision, and it is what
/// the reinterpretation destroyed: f32 bytes read as halves double the count
/// and then truncate, which is why the tail was zeros. A length check alone
/// would have caught this.
#[test]
fn the_element_count_survives_a_half_boundary() {
    for dt in [DType::F32, DType::BF16, DType::F16] {
        assert_eq!(add_in(dt).len(), 4, "{dt:?}");
    }
}

/// `x + 0` in `dtype`, for values that do not fit the format's mantissa.
fn identity_in(dtype: DType, xs: &[f32]) -> Vec<f32> {
    let s = Shape::new(&[xs.len()], dtype);
    let mut g = Graph::new("identity");
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let o = g.binary(BinaryOp::Add, x, y, s.clone());
    g.set_outputs(vec![o]);
    let ys = vec![0.0f32; xs.len()];
    Session::new(Device::Cpu)
        .compile(g)
        .run(&[("x", xs), ("y", &ys)])
        .pop()
        .expect("one output")
}

/// **The precision must actually be applied — the fix must not have made half
/// precision a synonym for f32.**
///
/// The repair here promotes the *execution* graph to F32 and narrows at the
/// boundary. The failure mode that would leave every other test in this file
/// green is the narrowing quietly not happening: values would round-trip
/// exactly and BF16 would behave as F32.
///
/// So this feeds values that need more mantissa than the format has — bf16
/// carries 8 bits (eps ~3.9e-3), f16 carries 11 (eps ~9.8e-4) — and requires
/// the result to equal the format's own rounding. `1.001` is the load-bearing
/// element: **f16 can represent it and bf16 cannot**, so the two formats must
/// disagree. A single generic "round to something" path would return the same
/// answer for both and fail here.
#[test]
fn each_half_format_applies_its_own_precision() {
    let xs = vec![1.0f32 + 1e-4, 1.0 + 1e-3, std::f32::consts::PI, 1.0 / 3.0];
    for dt in [DType::BF16, DType::F16] {
        let got = identity_in(dt, &xs);
        let want: Vec<f32> = xs
            .iter()
            .map(|&v| match dt {
                DType::BF16 => half::bf16::from_f32(v).to_f32(),
                _ => half::f16::from_f32(v).to_f32(),
            })
            .collect();
        assert_eq!(got, want, "{dt:?} must round to its own format");
        assert_ne!(
            got, xs,
            "{dt:?} returned the f32 input unchanged — the format is not being applied"
        );
    }
    // The two formats must not agree, or something generic is rounding for both.
    assert_ne!(
        identity_in(DType::BF16, &xs),
        identity_in(DType::F16, &xs),
        "bf16 and f16 produced identical results; each format's mantissa is not in play"
    );
}
