// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `x · W + b` with `W`, `b` graph PARAMETERS — the pattern the compiler fuses
//! into `FusedMatMulBiasAct` — against CPU, at a transformer trunk's shapes,
//! including a second run of the same compiled graph.
//!
//! Found through `rlx-montageflow` on Linux/Vulkan (RTX 3080 Ti): the fused
//! kernel drifted ~6e-5 per transformer block while the unfused matmul was
//! exact, and with the fusion disabled a second run was clobbered.
//!
//!     cargo test -p rlx-wgpu --test fused_matmul_bias_parity -- --nocapture

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

fn rel_max(a: &[f32], b: &[f32]) -> f64 {
    let scale = b.iter().fold(0f64, |m, v| m.max(v.abs() as f64)).max(1e-30);
    a.iter()
        .zip(b)
        .fold(0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
        / scale
}

/// `(rel_max run 1, rel_max run 2)` of wgpu against CPU. `amp` scales the
/// activations: GELU's tanh approximation departs from the exact erf form by
/// ~3e-4 only at |x| ≈ 2–3, so small inputs cannot tell the two apart.
fn case(lead: &[usize], k: usize, n: usize, gelu: bool) -> Option<(f64, f64)> {
    case_amp(lead, k, n, gelu, 1.0)
}

fn case_amp(lead: &[usize], k: usize, n: usize, gelu: bool, amp: f32) -> Option<(f64, f64)> {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return None;
    }
    let m: usize = lead.iter().product();
    let mut g = Graph::new("fused_mm_bias");
    let mut xs = lead.to_vec();
    xs.push(k);
    let x = g.input("x", Shape::new(&xs, DType::F32));
    let w = g.param("w", Shape::new(&[k, n], DType::F32));
    let b = g.param("b", Shape::new(&[n], DType::F32));
    let h = g.mm(x, w);
    let mut y = g.add(h, b);
    if gelu {
        y = g.gelu(y);
    }
    g.set_outputs(vec![y]);
    // zero-mean, LayerNorm-scale activations; xavier-scale weights
    let xv: Vec<f32> = (0..m * k)
        .map(|i| amp * ((i as f32) * 0.017).sin())
        .collect();
    let s = (6.0 / (k + n) as f32).sqrt();
    let wv: Vec<f32> = (0..k * n).map(|i| s * ((i as f32) * 0.013).cos()).collect();
    let bv: Vec<f32> = (0..n).map(|i| 0.1 * ((i as f32) * 0.7).sin()).collect();
    let run = |d: Device, times: usize| -> Vec<Vec<f32>> {
        let mut c = Session::new(d).compile(g.clone());
        c.set_param("w", &wv);
        c.set_param("b", &bv);
        (0..times)
            .map(|_| c.run(&[("x", xv.as_slice())]).remove(0))
            .collect()
    };
    let cpu = run(Device::Cpu, 1).remove(0);
    let gpu = run(Device::Gpu, 2);
    Some((rel_max(&gpu[0], &cpu), rel_max(&gpu[1], &cpu)))
}

#[test]
fn fused_matmul_bias_matches_cpu_at_trunk_shapes() {
    let mut bad = Vec::new();
    for (lead, k, n, gelu) in [
        (vec![64usize], 64usize, 64usize, false),
        (vec![640], 768, 768, false),
        (vec![2, 320], 768, 768, false),
        (vec![2, 320], 768, 3072, true),
        (vec![2, 320], 3072, 768, false),
        (vec![2], 768, 768, false), // adaLN-style: a 2-row conditioning vector
    ] {
        let Some((r1, r2)) = case(&lead, k, n, gelu) else {
            return;
        };
        eprintln!(
            "x{lead:?}x{k} · [{k}x{n}] + b{} : run1 {r1:.3e}  run2 {r2:.3e}",
            if gelu { " → gelu" } else { "" }
        );
        if !(r1 < 1e-5 && r2 < 1e-5) {
            bad.push(format!(
                "{lead:?} k={k} n={n} gelu={gelu}: {r1:.3e}/{r2:.3e}"
            ));
        }
    }
    assert!(bad.is_empty(), "fused matmul+bias off CPU: {bad:?}");
}

/// The fused epilogue's GELU must be the EXACT (erf) form for `Activation::Gelu`.
/// Eight kernels — `matmul_wide`, `matmul_wide_vec4`, `matmul_wide_nv`,
/// `matmul_f16w` and the four f16 Vulkan coop kernels — mapped opcode 9 to the
/// tanh approximation. On Linux/Vulkan the rank-3 `[B, S, K]` activations of a
/// transformer route to `matmul_wide_vec4`: 2.9e-4 on real `rlx-montageflow`
/// activations, 2.7e-4 on its full trunk (Metal's coop-f32 kernel was exact).
/// Mutation-checked: reverting `matmul_wide_vec4` alone fails this test. Here `W = [I | I | I | I]` makes the
/// pre-activation exactly the input, swept over [-4, 4] where the two forms
/// differ most, at the transformer shape that selects the wide kernels.
#[test]
fn fused_gelu_epilogue_is_exact_at_large_preactivations() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    // rank-3 activations, as a transformer has them: the rank-2 form picks a
    // different (exact) kernel and would not exercise the wide tiles
    for (bsz, s_len, k, reps) in [(2usize, 320usize, 768usize, 4usize), (2, 32, 64, 4)] {
        let (m, n) = (bsz * s_len, k * reps);
        let mut g = Graph::new("fused_gelu_exact");
        let x = g.input("x", Shape::new(&[bsz, s_len, k], DType::F32));
        let w = g.param("w", Shape::new(&[k, n], DType::F32));
        let b = g.param("b", Shape::new(&[n], DType::F32));
        let h = g.mm(x, w);
        let y = g.add(h, b);
        let y = g.gelu(y);
        g.set_outputs(vec![y]);
        let xv: Vec<f32> = (0..m * k)
            .map(|i| -4.0 + 8.0 * (i % 997) as f32 / 996.0)
            .collect();
        let mut wv = vec![0f32; k * n];
        for r in 0..k {
            for rep in 0..reps {
                wv[r * n + rep * k + r] = 1.0;
            }
        }
        let bv = vec![0f32; n];
        let run = |d: Device| -> Vec<f32> {
            let mut c = Session::new(d).compile(g.clone());
            c.set_param("w", &wv);
            c.set_param("b", &bv);
            c.run(&[("x", xv.as_slice())]).remove(0)
        };
        let (cpu, gpu) = (run(Device::Cpu), run(Device::Gpu));
        let r = rel_max(&gpu, &cpu);
        eprintln!("[{bsz}x{s_len}x{k}]·[I×{reps}] + 0 → gelu, pre-act in [-4, 4]: rel_max {r:.3e}");
        assert!(
            r < 1e-5,
            "fused GELU epilogue is not the exact form: rel_max {r:.3e}"
        );
    }
}
