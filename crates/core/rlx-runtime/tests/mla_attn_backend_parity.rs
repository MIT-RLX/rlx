// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Cross-backend parity for the Multi-head Latent Attention (MLA) block.
//!
//! Builds the real DeepSeek-V2/V3 [`MlaAttnPrefillStage`] via the flow DSL, then
//! runs the identical graph on every available backend and checks it against the
//! CPU result. The block lowers to primitives (matmul / RMSNorm / RoPE / narrow /
//! concat / pad / attention), so this is what proves each backend actually
//! executes MLA correctly — the same way the CPU run surfaced the `Narrow→Rope`
//! stride bug. GPU tests skip gracefully when the device is unavailable.

#![allow(dead_code)]

use std::collections::HashMap;

use rlx_flow::MapWeights;
use rlx_flow::prelude::*;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

const B: usize = 1;
const S: usize = 4;
const HID: usize = 16;
const NH: usize = 2;
const QLORA: usize = 8;
const KVLORA: usize = 8;
const NOPE: usize = 6;
const ROPE: usize = 2;
const VH: usize = 4;
const QK: usize = NOPE + ROPE; // 8 — GPU-friendly, asymmetric vs VH=4
const EPS: f32 = 1e-6;

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5)
        .collect()
}
fn gamma(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| 0.9 + 0.05 * ((i as f32) + seed).cos())
        .collect()
}

/// Build the MLA prefill flow → (graph, params, input `x`). Deterministic, so
/// every backend gets bit-identical weights.
fn build_mla() -> (Graph, HashMap<String, Vec<f32>>, Vec<f32>, Vec<String>) {
    let x = fill(B * S * HID, 0.7);

    // Real single-frequency RoPE rotation (n_rot/2 = 1 column per position).
    let cos: Vec<f32> = (0..S).map(|p| (p as f32).cos()).collect();
    let sin: Vec<f32> = (0..S).map(|p| (p as f32).sin()).collect();

    let lp = "model.layers.0";
    let mut w = MapWeights::default();
    w.insert(
        format!("{lp}.self_attn.q_a_proj.weight"),
        fill(QLORA * HID, 0.1),
        vec![QLORA, HID],
    );
    w.insert(
        format!("{lp}.self_attn.q_a_layernorm.weight"),
        gamma(QLORA, 0.2),
        vec![QLORA],
    );
    w.insert(
        format!("{lp}.self_attn.q_b_proj.weight"),
        fill(NH * QK * QLORA, 0.3),
        vec![NH * QK, QLORA],
    );
    w.insert(
        format!("{lp}.self_attn.kv_a_proj_with_mqa.weight"),
        fill((KVLORA + ROPE) * HID, 0.4),
        vec![KVLORA + ROPE, HID],
    );
    w.insert(
        format!("{lp}.self_attn.kv_a_layernorm.weight"),
        gamma(KVLORA, 0.5),
        vec![KVLORA],
    );
    w.insert(
        format!("{lp}.self_attn.kv_b_proj.weight"),
        fill(NH * (NOPE + VH) * KVLORA, 0.6),
        vec![NH * (NOPE + VH), KVLORA],
    );

    let spec = MlaAttnPrefillSpec::deepseek_layer(lp, NH, QLORA, KVLORA, NOPE, ROPE, VH, EPS);

    let built = ModelFlow::new("mla")
        .input("x", Shape::new(&[B, S, HID], DType::F32))
        .stage(FlowStage::RopeTables(RopeTablesStage::param(
            S, 1, cos, sin,
        )))
        .layer_stage(MlaAttnPrefillStage::new(spec))
        .build(&mut w)
        .expect("flow build");

    let names = built.output_names().to_vec();
    let (g, params) = built.into_graph_parts().expect("graph parts");
    (g, params, x, names)
}

fn run_mla(device: Device) -> Vec<f32> {
    run_mla_all(device).1.into_iter().next().unwrap()
}

/// Returns (output_names, all outputs) so the diagnostic can compare every
/// intermediate side output across backends, not just the primary result.
fn run_mla_all(device: Device) -> (Vec<String>, Vec<Vec<f32>>) {
    let (g, params, x, names) = build_mla();
    let mut c = Session::new(device).compile(g);
    for (k, v) in &params {
        c.set_param(k.as_str(), v.as_slice());
    }
    (names, c.run(&[("x", &x)]))
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn mla_cpu_runs() {
    let out = run_mla(Device::Cpu);
    assert_eq!(out.len(), B * S * NH * VH, "MLA output shape");
}

macro_rules! backend_parity {
    ($name:ident, $feat:meta, $dev:expr) => {
        #[test]
        #[$feat]
        fn $name() {
            if !is_available($dev) {
                eprintln!("skip: {:?} unavailable", $dev);
                return;
            }
            let cpu = run_mla(Device::Cpu);
            let dev = run_mla($dev);
            assert_eq!(dev.len(), cpu.len(), "{:?} output length", $dev);
            let err = max_abs(&cpu, &dev);
            eprintln!("{:?}/CPU MLA max_abs={:.3e}", $dev, err);
            assert!(err < 2e-3, "MLA {:?} parity failed: {:.3e}", $dev, err);
        }
    };
}

backend_parity!(
    mla_parity_metal,
    cfg(all(feature = "metal", target_os = "macos")),
    Device::Metal
);
backend_parity!(
    mla_parity_mlx,
    cfg(all(feature = "mlx", target_os = "macos")),
    Device::Mlx
);
backend_parity!(mla_parity_wgpu, cfg(feature = "gpu"), Device::Gpu);
backend_parity!(mla_parity_cuda, cfg(feature = "cuda"), Device::Cuda);
backend_parity!(mla_parity_rocm, cfg(feature = "rocm"), Device::Rocm);
backend_parity!(mla_parity_vulkan, cfg(feature = "vulkan"), Device::Vulkan);
