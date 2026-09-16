// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression: `set_param` while whole-graph CUDA capture is engaged.
//!
//! Two separate defects, both reachable only with
//! `RLX_CUDA_EXEC_MODE=graph` **and** `RLX_CUDA_WHOLE_GRAPH_CAPTURE=1`
//! (`ExecMode` defaults to `Stream`, and the whole-graph path is its own opt-in).
//!
//! 1. **A process abort.** `F32HostSlot::copy_from_host` `expect`ed a fallible
//!    pinned accessor:
//!    `pinned input staging unavailable: DriverError(CUDA_ERROR_INVALID_VALUE)`.
//!    Pinned staging is a bandwidth optimization, so a driver refusal must
//!    degrade (drain → retry → demote to pageable), never take the process down.
//!
//! 2. **Silently stale weights** — which the abort was hiding, since it fired
//!    first. A captured graph replays without observing a param written after
//!    the capture was taken: writing `w = k·I` each step and reading back,
//!    6 of 8 steps returned step `k−1`'s value on an RTX 3080 Ti. No error, no
//!    NaN, just a plausible wrong number. `note_host_write` now drops the
//!    capture on any param write.
//!
//! Part 2 is why this checks *values* rather than merely surviving: part 1's
//! identity weights cannot detect staleness, because every step writes the same
//! thing.
//!
//! Deliberately **one** test function. `set_var` is process-global and two
//! `#[test]`s in one binary run concurrently — they would race on the env and
//! then capture simultaneously on the shared CUDA context, failing for reasons
//! unrelated to what is under test.

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const N: usize = 64;
const LAYERS: usize = 6;

/// `scale · I` — a chain of identities is the identity, so the expected output
/// is exactly the input and corruption shows up as a deviation.
fn identity(scale: f32) -> Vec<f32> {
    let mut m = vec![0.0f32; N * N];
    for i in 0..N {
        m[i * N + i] = scale;
    }
    m
}

#[test]
fn set_param_under_whole_graph_capture() {
    if rlx_ir::env::skip_unless_device("cuda", true, rlx_cuda::is_available()) {
        return;
    }
    // SAFETY: set before any CUDA executable is built; single test in this binary.
    unsafe {
        std::env::set_var("RLX_CUDA_EXEC_MODE", "graph");
        std::env::set_var("RLX_CUDA_WHOLE_GRAPH_CAPTURE", "1");
    }

    // ── Part 1: repeated writes must not abort. ──
    let eye = identity(1.0);
    let x: Vec<f32> = (0..N * N).map(|i| (i as f32 * 0.017).sin()).collect();

    let mut chain = Graph::new("set_param_capture");
    let mut cur = chain.input("x", Shape::new(&[N, N], DType::F32));
    for i in 0..LAYERS {
        let w = chain.param(format!("w{i}"), Shape::new(&[N, N], DType::F32));
        cur = chain.mm(cur, w);
    }
    chain.set_outputs(vec![cur]);

    let mut c = Session::new(Device::Cuda).compile(chain);
    for i in 0..LAYERS {
        c.set_param(&format!("w{i}"), &eye);
    }

    // Enough steps to get past warm-up and the capture — a single write never
    // tripped the original abort.
    for step in 0..40 {
        c.set_param(&format!("w{}", step % LAYERS), &eye);
        let out = c.run(&[("x", &x)]).pop().expect("output");
        assert_eq!(out.len(), x.len(), "step {step}: output length");
        for (j, (got, want)) in out.iter().zip(x.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                "step {step}, element {j}: identity chain gave {got}, want {want}"
            );
        }
    }

    // ── Part 2: a write must be observed by the very next run. ──
    let mut scale = Graph::new("scale");
    let sx = scale.input("x", Shape::new(&[N, N], DType::F32));
    let sw = scale.param("w", Shape::new(&[N, N], DType::F32));
    let sy = scale.mm(sx, sw);
    scale.set_outputs(vec![sy]);

    let mut sc = Session::new(Device::Cuda).compile(scale);
    let ones = vec![1.0f32; N * N];

    for step in 1..=12u32 {
        // w = k·I and x all ones → every output element is exactly k.
        let k = step as f32;
        sc.set_param("w", &identity(k));
        let out = sc.run(&[("x", &ones)]).pop().expect("output");
        for (j, got) in out.iter().enumerate() {
            assert!(
                (got - k).abs() <= 1e-4 * k,
                "step {step}, element {j}: got {got}, want {k} — the run did not \
                 observe the param written immediately before it"
            );
        }
    }
}
