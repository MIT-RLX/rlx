// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Where does the CUDA graph capture boundary actually land?
//!
//! Whole-graph capture is taken once and replayed, but host-side actions between
//! steps can drop it (`captured_graph = None` in the rlx-cuda dispatch loop).
//! This probe measures which ones do.
//!
//! # Required configuration — BOTH variables
//!
//! ```text
//! RLX_CUDA_EXEC_MODE=graph RLX_CUDA_WHOLE_GRAPH_CAPTURE=1
//! ```
//!
//! Neither alone is sufficient, which is easy to get wrong:
//!
//! * `ExecMode` defaults to `Stream` and `graph_eligible` requires `Graph`, so
//!   without `RLX_CUDA_EXEC_MODE=graph` no capture is taken.
//! * With only that, capture still does not engage — `want_capture_stream` also
//!   needs `RLX_CUDA_WHOLE_GRAPH_CAPTURE=1` to open the capture stream.
//!
//! A run where capture never engaged is recognizable from the printed anchors:
//! the capture/replay gap collapses to noise, or goes negative. With both set on
//! a launch-bound graph the gap is unmistakable (measured ~0.39 ms on a 3080 Ti
//! at 64x64x120, capture 1.14 ms vs replay 0.74 ms).
//!
//! # Measured on an RTX 3080 Ti, 64x64 x 120 layers, 120 steps
//!
//! | mode | p50 vs replay | verdict |
//! |---|---|---|
//! | `steady` | 1.00x | replays |
//! | `feed_new_data` | 1.00x | replays |
//! | `set_param` | 1.68x | **re-captures, by design** |
//! | `set_param_noop` | 0.77x | replays (no CUDA call made) |
//! | `set_param_once` | 0.77x | replays |
//! | `alternate_graphs` | 0.95x | replays |
//! | `in_graph_select` | 0.79x | replays |
//!
//! Host-side ping-pong between two compiled artifacts does **not** cost a
//! recapture: each `CompiledGraph` keeps its own capture and round-robining
//! between them is free. Feeding fresh input buffers is likewise free. So a
//! device-side buffer swap buys nothing here — the host path is already fine.
//!
//! `set_param` re-captures because a param write now invalidates the capture
//! (`note_host_write`). That is deliberate: a replay does not observe a param
//! written after the capture was taken, so before the fix this configuration
//! returned the *previous* step's weights — 6 of 8 steps stale, silently. It
//! also used to abort in `host_staging.rs` with
//! `pinned input staging unavailable: CUDA_ERROR_INVALID_VALUE`, which fired
//! first and hid the stale reads. Both are covered by
//! `tests/set_param_under_capture.rs`; `examples/param_visibility.rs` is the
//! minimal per-config reproducer.
//!
//! Usage:
//! `RLX_CUDA_EXEC_MODE=graph cargo run --release -p rlx-cuda --example capture_rebind_probe -- 100`

use rlx_ir::infer::GraphExt;
use rlx_ir::lanes::LaneExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::time::Instant;

/// Matrix size and depth, overridable so the workload can be made
/// launch-overhead-bound (small `N`, many `LAYERS`) — which is the regime where
/// capture actually pays and therefore the only regime where "did the capture
/// survive?" is answerable by timing. At 512x512x8 the capture/replay gap is
/// ~0.14 ms, smaller than the H2D differences between modes, and every verdict
/// is noise.
fn dims() -> (usize, usize) {
    let n = std::env::var("PROBE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512usize);
    let l = std::env::var("PROBE_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize);
    (n, l)
}

/// A chain of matmuls — enough kernels that a capture is worth taking.
fn build_chain() -> Graph {
    let (n, layers) = dims();
    let mut g = Graph::new("chain");
    let mut x = g.input("x", Shape::new(&[n, n], DType::F32));
    for i in 0..layers {
        let w = g.param(format!("w{i}"), Shape::new(&[n, n], DType::F32));
        x = g.mm(x, w);
        x = g.silu(x);
    }
    g.set_outputs(vec![x]);
    g
}

/// The same work, but the state is chosen inside the graph by a lane selector.
fn build_selected() -> Graph {
    let (n, layers) = dims();
    let mut g = Graph::new("chain_sel");
    let a = g.input("a", Shape::new(&[n, n], DType::F32));
    let b = g.input("b", Shape::new(&[n, n], DType::F32));
    let sel = g.input("sel", Shape::new(&[n], DType::F32));
    let mut x = g.reset_lanes_(a, b, sel);
    for i in 0..layers {
        let w = g.param(format!("w{i}"), Shape::new(&[n, n], DType::F32));
        x = g.mm(x, w);
        x = g.silu(x);
    }
    g.set_outputs(vec![x]);
    g
}

fn fill_params(c: &mut CompiledGraph, seed: f32) {
    let (n, layers) = dims();
    let w: Vec<f32> = (0..n * n)
        .map(|i| ((i as f32 * 0.0001 + seed).sin()) * 0.05)
        .collect();
    for i in 0..layers {
        c.set_param(&format!("w{i}"), &w);
    }
}

/// Time `steps` iterations, returning (first-step ms, steady-state p50 ms).
fn timed<F: FnMut(usize)>(steps: usize, mut step: F) -> (f64, f64) {
    let mut times = Vec::with_capacity(steps);
    for i in 0..steps {
        let t = Instant::now();
        step(i);
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let first = times[0];
    // Capture needs one eager run before it is taken, so steady state starts a
    // few iterations in.
    let warm = (steps / 4).max(3).min(steps - 1);
    let mut tail: Vec<f64> = times[warm..].to_vec();
    tail.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (first, tail[tail.len() / 2])
}

/// Run one mode, reporting a panic as a result rather than aborting the probe —
/// a mode that cannot run at all is itself the finding.
fn guarded<F>(name: &str, replay: f64, capture: f64, body: F)
where
    F: FnOnce() -> (f64, f64) + std::panic::UnwindSafe,
{
    match std::panic::catch_unwind(body) {
        Ok((first, p50)) => {
            let over = p50 / replay;
            let verdict = if p50 > replay + 0.5 * (capture - replay) {
                "RE-CAPTURING (steady approaches capture cost)"
            } else if over < 1.25 {
                "replaying (no recapture)"
            } else {
                "slower than baseline, below capture cost"
            };
            println!(
                "  {name:<18} first {first:7.3} ms   p50 {p50:6.3} ms   {over:5.2}x replay   {verdict}"
            );
        }
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<non-string panic>".into());
            println!("  {name:<18} PANICKED: {msg}");
        }
    }
}

fn main() {
    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    let (n, layers) = dims();
    let mode = rlx_ir::env::var("RLX_CUDA_EXEC_MODE").unwrap_or_default();
    println!("capture boundary probe: {steps} steps, {layers} layers of {n}x{n}");
    println!("RLX_CUDA_EXEC_MODE={mode:?}");
    if !mode.eq_ignore_ascii_case("graph") {
        println!("  WARNING: ExecMode is not Graph — no capture is taken, so every");
        println!("           mode below trivially looks like a replay.");
    }
    println!();

    let x: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.001).cos()).collect();
    let sess = Session::new(Device::Cuda);

    // Pay CUDA context + module init up front.
    {
        let mut warm = sess.compile(build_chain());
        fill_params(&mut warm, 0.0);
        for _ in 0..10 {
            let _ = warm.run(&[("x", &x)]);
        }
    }

    // Anchor 1: taking a capture, on a fresh artifact with init already warm.
    let mut c0 = sess.compile(build_chain());
    fill_params(&mut c0, 0.0);
    let (capture_cost, _) = timed(4, |_| {
        let _ = c0.run(&[("x", &x)]);
    });

    // Anchor 2: steady-state replay.
    let mut c1 = sess.compile(build_chain());
    fill_params(&mut c1, 0.0);
    let (_, replay_cost) = timed(steps, |_| {
        let _ = c1.run(&[("x", &x)]);
    });

    println!(
        "  anchors: capture {capture_cost:.3} ms   replay {replay_cost:.3} ms   gap {:.3} ms\n",
        capture_cost - replay_cost
    );

    guarded("steady", replay_cost, capture_cost, || {
        let mut c = sess.compile(build_chain());
        fill_params(&mut c, 0.0);
        timed(steps, |_| {
            let _ = c.run(&[("x", &x)]);
        })
    });

    guarded("feed_new_data", replay_cost, capture_cost, || {
        let mut c = sess.compile(build_chain());
        fill_params(&mut c, 0.0);
        timed(steps, |i| {
            let xi: Vec<f32> = x.iter().map(|v| v + i as f32 * 1e-6).collect();
            let _ = c.run(&[("x", &xi)]);
        })
    });

    guarded("set_param", replay_cost, capture_cost, || {
        let mut c = sess.compile(build_chain());
        fill_params(&mut c, 0.0);
        let w: Vec<f32> = (0..n * n)
            .map(|i| (i as f32 * 0.0001).sin() * 0.05)
            .collect();
        timed(steps, |i| {
            c.set_param(&format!("w{}", i % layers), &w);
            let _ = c.run(&[("x", &x)]);
        })
    });

    // ── diagnostics for the set_param panic ──
    // A set_param whose name does not exist returns before touching CUDA. If
    // this panics too, the trigger is not the default-stream upload.
    guarded("set_param_noop", replay_cost, capture_cost, || {
        let mut c = sess.compile(build_chain());
        fill_params(&mut c, 0.0);
        let w = vec![0.0f32; 4];
        timed(steps, |_| {
            c.set_param("no_such_param", &w);
            let _ = c.run(&[("x", &x)]);
        })
    });

    // set_param only ONCE, after the capture is established, then plain runs.
    // Tells us whether the damage is permanent or per-call.
    guarded("set_param_once", replay_cost, capture_cost, || {
        let mut c = sess.compile(build_chain());
        fill_params(&mut c, 0.0);
        let w: Vec<f32> = (0..n * n)
            .map(|i| (i as f32 * 0.0001).sin() * 0.05)
            .collect();
        timed(steps, |i| {
            if i == steps / 2 {
                c.set_param("w0", &w);
            }
            let _ = c.run(&[("x", &x)]);
        })
    });

    guarded("alternate_graphs", replay_cost, capture_cost, || {
        let mut a = sess.compile(build_chain());
        let mut b = sess.compile(build_chain());
        fill_params(&mut a, 0.0);
        fill_params(&mut b, 1.0);
        timed(steps, |i| {
            let c = if i % 2 == 0 { &mut a } else { &mut b };
            let _ = c.run(&[("x", &x)]);
        })
    });

    guarded("in_graph_select", replay_cost, capture_cost, || {
        let mut c = sess.compile(build_selected());
        fill_params(&mut c, 0.0);
        let sa: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.01).sin()).collect();
        let sb: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.01).cos()).collect();
        timed(steps, |i| {
            // Flip which lanes take the reset value each step.
            let sel: Vec<f32> = (0..n)
                .map(|l| if (l + i) % 2 == 0 { 1.0 } else { 0.0 })
                .collect();
            let _ = c.run(&[("a", &sa), ("b", &sb), ("sel", &sel)]);
        })
    });

    println!(
        "\nnote: timing is indirect. RLX_CUDA_CAPTURE_DEBUG=1 adds the backend's own\n\
         report of what made a capture unsafe or invalid."
    );
}
