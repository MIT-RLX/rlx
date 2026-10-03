// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The floor every Apple platform has to clear — **including watchOS**.
//!
//! [`apple_backends_sim`](../apple_backends_sim.rs) covers the accelerators and
//! is therefore gated off watchOS, which has neither Metal nor the CoreML
//! runtime-compile path. That leaves watchOS with nothing that actually *runs*,
//! and a cross-compile gate cannot tell a working CPU backend from one that
//! links and then computes garbage. This file is what runs there.
//!
//! It is ungated for the whole Apple vendor, so `just test-apple-sim` executes
//! it on the iOS, tvOS, watchOS and visionOS simulators from one recipe:
//!
//! ```sh
//! just test-apple-sim
//! ```
//!
//! Needs only the `cpu` feature — that is the point.
#![cfg(target_vendor = "apple")]

use rlx_ir::*;
use rlx_runtime::{Device, Session};

/// `relu(x @ w + b)` with values chosen so every result is checkable by hand:
/// `x` is the 2×3 counting matrix, `w` is 3×2 of ones, `b` alternates ∓1, so
/// the pre-activation is `[[3-1, 3+1], [12-1, 12+1]]` and relu clamps nothing.
fn build() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("apple_platform_smoke");
    let x = g.input("x", Shape::new(&[2, 3], f));
    let w = g.input("w", Shape::new(&[3, 2], f));
    let b = g.input("b", Shape::new(&[2, 2], f));
    let mm = g.matmul(x, w, Shape::new(&[2, 2], f));
    let y = g.binary(op::BinaryOp::Add, mm, b, Shape::new(&[2, 2], f));
    let y = g.relu(y);
    g.set_outputs(vec![y]);
    g
}

/// The CPU backend must compute, not merely link.
///
/// On watchOS this is the *only* backend there is, so an Accelerate path that
/// cross-compiles and then returns zeros would otherwise ship unnoticed —
/// which is why the expected values are spelled out rather than compared
/// against a second run of the same code.
#[test]
fn cpu_backend_computes_on_this_platform() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let w = vec![1.0; 6];
    let b = vec![-1.0, 1.0, -1.0, 1.0];
    let out = Session::new(Device::Cpu)
        .compile(build())
        .run(&[("x", &x), ("w", &w), ("b", &b)])
        .pop()
        .expect("one output");
    let expected = [2.0f32, 4.0, 11.0, 13.0];
    assert_eq!(out.len(), expected.len());
    for (got, want) in out.iter().zip(expected) {
        assert!((got - want).abs() < 1e-5, "got {out:?}, want {expected:?}");
    }
}

/// A node joining a mesh from this platform must name it correctly.
///
/// The tag is a `cfg!` ladder, so an Apple OS it has not been taught falls
/// through to `"unknown"` and a coordinator log stops being able to say which
/// kind of rank answered. Running this *on the simulator* is what makes the
/// check meaningful — the host build can only ever assert `macos`.
#[test]
fn platform_tag_names_this_apple_os() {
    let tag = rlx_runtime::dist::node::platform_tag();
    let expected = if cfg!(target_os = "ios") {
        "ios"
    } else if cfg!(target_os = "tvos") {
        "tvos"
    } else if cfg!(target_os = "watchos") {
        "watchos"
    } else if cfg!(target_os = "visionos") {
        "visionos"
    } else {
        "macos"
    };
    eprintln!("platform_tag() = {tag}");
    assert_eq!(tag, expected);
}

/// Backend selection has to resolve to something that can actually run here.
///
/// `fastest_device` probes live devices, so on watchOS — and in a headless
/// `simctl spawn`, where no Metal device is exposed — it has to come back with
/// the CPU rather than a backend that is merely compiled in.
#[test]
fn selection_resolves_to_a_live_backend() {
    let chosen = rlx_runtime::fastest_device();
    eprintln!("fastest_device() = {}", chosen.name());
    assert!(
        rlx_runtime::is_available(chosen),
        "fastest_device returned unavailable {chosen:?}"
    );
    if cfg!(target_os = "watchos") {
        assert_eq!(
            chosen,
            Device::Cpu,
            "watchOS has no Metal API and no CoreML runtime compile — the CPU \
             is the only backend that exists there"
        );
    }
}
