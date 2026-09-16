// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! On-device Philox RNG vs the rlx-cpu generator, element for element.
//!
//! `Op::RngNormal` / `Op::RngUniform` used to be the last stall in a Vulkan
//! graph that draws random numbers: read the arena back, run the CPU generator,
//! upload the result. `shaders/rng_*_philox.comp` replace that with one
//! dispatch.
//!
//! A distributional check would pass on a stream that is merely *a* Philox
//! stream — wrong counter, wrong lane order, wrong key schedule. Since the point
//! is that a graph gives the same answer on every backend, the gates here
//! compare against `rlx_ir::fill_normal_like` / `fill_uniform_like` on the same
//! seed, element for element. That is also what catches the two lane-mapping
//! traps: the normal generator consumes two Philox lanes per sample (block
//! `i/2`, lanes {0,1} or {2,3}) while the uniform consumes one (block `i/4`,
//! lane `i%4`).
//!
//! **Uniform is exact; normal is 1e-6 relative.** The split is not a hedge —
//! it is where the two implementations genuinely stop being the same
//! computation. The uniform path is integer Philox plus a divide, so it matches
//! bit for bit and proves the counter, lanes, key schedule and seed are right.
//! The normal path then applies Box-Muller, and `sqrt`/`log`/`cos` come from the
//! driver rather than the host libm; on MoltenVK that measures 1–2 ULP.
//!
//! The seed gate is separate and deliberate. `host::eval` builds a fresh one-op
//! CPU graph and never sees the executable's `RngOptions`, so before the GPU
//! path existed `compile_rng(g, opts)` produced the *default* stream on Vulkan
//! whatever `opts.seed` said. `distinct_seeds_produce_distinct_streams` is what
//! would have caught that.
//!
//! Runs only when a Vulkan device is present; otherwise a graceful no-op.

use rlx_ir::{DType, Graph, Op, RngBackend, RngOptions, Shape};
use rlx_vulkan::backend::VulkanExecutable;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Serialize device use — rlx-vulkan submits to a process-global `VkQueue`,
/// which Vulkan requires callers to externally synchronize.
/// Skip when no Vulkan device is present.
///
/// `rlx_ir::env::skip_unless_device` rather than a bare
/// `if !is_available() { return }`: the bare form reports `ok` on a rig with no
/// device, so a CI box that lost its Vulkan driver would look green. This one
/// honours `RLX_REQUIRE_DEVICE=1` and fails instead. (A local `fn available()`
/// wrapper hides the same problem from `require_device_coverage` without fixing
/// it.)
fn skip() -> bool {
    rlx_ir::env::skip_unless_device("vulkan", true, rlx_vulkan::is_available())
}

fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

const KEY: u64 = 0x5eed_1234_abcd_0001;

fn normal_graph(n: usize, mean: f32, scale: f32) -> Graph {
    let mut g = Graph::new("rng_normal");
    let y = g.add_node(
        Op::RngNormal {
            mean,
            scale,
            key: KEY,
            op_seed: None,
        },
        vec![],
        Shape::new(&[n], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn uniform_graph(n: usize, low: f32, high: f32) -> Graph {
    let mut g = Graph::new("rng_uniform");
    let y = g.add_node(
        Op::RngUniform {
            low,
            high,
            key: KEY,
            op_seed: None,
        },
        vec![],
        Shape::new(&[n], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn run_vulkan(g: Graph, opts: RngOptions) -> Vec<f32> {
    VulkanExecutable::compile_rng(g, opts).run(&[]).remove(0)
}

/// The reference stream, straight from the generator both backends must match.
fn cpu_normal(n: usize, mean: f32, scale: f32, opts: RngOptions) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    rlx_ir::fill_normal_like(&mut out, mean, scale, opts, KEY, None);
    out
}

fn cpu_uniform(n: usize, low: f32, high: f32, opts: RngOptions) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    rlx_ir::fill_uniform_like(&mut out, low, high, opts, KEY, None);
    out
}

fn philox(seed: u64) -> RngOptions {
    RngOptions {
        backend: RngBackend::Philox,
        seed,
    }
}

fn assert_exact(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: element {i} — gpu {g} vs cpu {w}"
        );
    }
}

/// Per-element relative comparison for the normal stream.
///
/// The uniform gates above are exact, and that is the load-bearing check: it
/// proves the Philox block counter, the lane mapping, the key schedule and the
/// seed derivation all agree bit for bit. What the normal path adds on top is
/// `sqrt`, `log` and `cos`, and those are the driver's implementations rather
/// than the host's libm — measured at 1–2 ULP apart on MoltenVK (e.g. 0.9722007
/// vs 0.97220075, a relative 5e-8). Demanding equality there would be demanding
/// that two transcendental libraries agree, which is not a property RLX has or
/// wants.
///
/// The tolerance is still tight enough to be a real gate: every failure mode
/// this test exists to catch — wrong counter, swapped lane halves, dropped key
/// increment, ignored seed — produces a *different sample*, not a 1-ULP one.
fn assert_close(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let rel = (g - w).abs() / w.abs().max(1e-6);
        worst = worst.max(rel);
        assert!(
            rel <= 1e-6,
            "{what}: element {i} — gpu {g} vs cpu {w} (rel {rel:e})"
        );
    }
    eprintln!("{what}: max rel diff {worst:.2e}");
}

#[test]
fn normal_stream_matches_cpu() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    let opts = philox(0x1234_5678);
    // An odd length exercises both lane halves and the tail guard: sample 0 reads
    // block 0 lanes {0,1}, sample 1 reads block 0 lanes {2,3}, sample 2 starts
    // block 1. Getting the halves backwards still yields a valid normal stream.
    for n in [1usize, 2, 3, 7, 64, 65, 1000] {
        let got = run_vulkan(normal_graph(n, 0.0, 1.0), opts);
        let want = cpu_normal(n, 0.0, 1.0, opts);
        assert_close(&format!("normal n={n}"), &got, &want);
    }
}

#[test]
fn normal_mean_and_scale_are_applied() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    let opts = philox(0xfeed_face);
    let n = 257;
    let got = run_vulkan(normal_graph(n, -3.5, 0.25), opts);
    let want = cpu_normal(n, -3.5, 0.25, opts);
    assert_close("normal mean/scale", &got, &want);
}

#[test]
fn uniform_stream_matches_cpu_bit_exactly() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    let opts = philox(0x0bad_c0de);
    // Lengths straddling a Philox block (4 uniforms per block) so a lane-index
    // mistake cannot hide in an aligned count.
    for n in [1usize, 3, 4, 5, 8, 9, 255, 1024] {
        let got = run_vulkan(uniform_graph(n, 0.0, 1.0), opts);
        let want = cpu_uniform(n, 0.0, 1.0, opts);
        assert_exact(&format!("uniform n={n}"), &got, &want);
    }
}

#[test]
fn uniform_range_is_applied() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    let opts = philox(7);
    let n = 300;
    let got = run_vulkan(uniform_graph(n, -2.0, 6.0), opts);
    let want = cpu_uniform(n, -2.0, 6.0, opts);
    assert_exact("uniform range", &got, &want);
    assert!(
        got.iter().all(|v| (-2.0..6.0).contains(v)),
        "uniform samples escaped [-2, 6)"
    );
}

#[test]
fn distinct_seeds_produce_distinct_streams() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    let n = 128;
    let a = run_vulkan(normal_graph(n, 0.0, 1.0), philox(1));
    let b = run_vulkan(normal_graph(n, 0.0, 1.0), philox(2));
    // The host route ignored `RngOptions` entirely, so both seeds gave the same
    // default stream and this would have been an equality.
    assert_ne!(a, b, "compile_rng seed had no effect on the stream");
    assert_close("seed 1", &a, &cpu_normal(n, 0.0, 1.0, philox(1)));
    assert_close("seed 2", &b, &cpu_normal(n, 0.0, 1.0, philox(2)));
}

#[test]
fn zero_backend_fills_zeros() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    let opts = RngOptions {
        backend: RngBackend::Zero,
        ..RngOptions::default()
    };
    let n = 100;
    let got = run_vulkan(normal_graph(n, 0.0, 1.0), opts);
    assert_exact("zero normal", &got, &cpu_normal(n, 0.0, 1.0, opts));
    let got = run_vulkan(uniform_graph(n, 0.0, 1.0), opts);
    assert_exact("zero uniform", &got, &cpu_uniform(n, 0.0, 1.0, opts));
}
