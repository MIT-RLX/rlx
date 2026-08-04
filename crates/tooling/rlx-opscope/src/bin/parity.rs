// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-parity` — prove our transforms don't change results. Compares, on a
//! runnable demo MLP:
//!   * original vs **stat-injected** — op by op (should be BIT-EXACT);
//!   * **fusion on vs off** — primary output (should be within tolerance).
//! This is the correctness guarantee behind every profiling run.

use rlx_ir::{Op, Philox4x32};
use rlx_opscope::demo::{D, S, build};
use rlx_opscope::parity::{fusion_output_parity, op_level_parity};
use rlx_opscope::{StatConfig, inject_matmul_stats};

fn main() {
    let g = build("mlp", 4);

    // Random params (same for both sides).
    let mut rng = Philox4x32::new(0x0FA1_1751);
    let params: Vec<(String, Vec<f32>)> = g
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Param { name } => {
                let numel: usize = (0..n.shape.rank())
                    .map(|i| n.shape.dim(i).unwrap_static())
                    .product();
                let mut d = vec![0f32; numel];
                rng.fill_normal(&mut d);
                Some((name.clone(), d))
            }
            _ => None,
        })
        .collect();
    let mut x = vec![0f32; S * D];
    rng.fill_normal(&mut x);
    let inputs = [("x", x.as_slice())];

    // 1) Op-level parity: original vs stat-injected.
    let (g_inj, specs) = inject_matmul_stats(&g, &StatConfig::default());
    let r = op_level_parity(&g, &g_inj, &inputs, &params);
    println!(
        "injection op-level parity : {} ops compared, max_abs {:.2e}  →  {}",
        r.ops_checked,
        r.max_abs,
        if r.exact() {
            "BIT-EXACT ✓ (every op unchanged)".to_string()
        } else {
            format!("MISMATCH at op {} ✗", r.worst_op)
        }
    );
    println!(
        "                            (+{} appended stat outputs, primary path untouched)",
        specs.len()
    );

    // 2) Fusion on vs off parity (structure-changing → output-level, tolerance).
    let rf = fusion_output_parity(&g, &inputs, &params);
    println!(
        "fusion on/off parity      : max_rel {:.2e}  →  {}",
        rf.max_rel,
        if rf.within(1e-4) {
            "within tol ✓ (skip_fusion safe for profiling)"
        } else {
            "OUT OF TOL ✗"
        }
    );
}
