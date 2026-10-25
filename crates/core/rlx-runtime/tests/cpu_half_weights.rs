// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Half matmul weights on CPU stay half: an f32 activation against an f16 /
//! bf16 weight `Param` runs the packed dequant-on-the-fly GEMM, and gives
//! the f32 GEMM of the same (rounded) weights. What lets a package opened
//! with `weight_dtype: f16` hold its matmul weights at 2 B each.

#![cfg(feature = "cpu")]

mod common;

use rlx_ir::{DType, Graph, GraphExt, Shape};

fn round(w: &[f32], dt: DType) -> Vec<f32> {
    let bytes = rlx_runtime::backend::narrow_f32_to_bytes(w, dt);
    rlx_runtime::backend::widen_bytes_to_f32(&bytes, dt)
}

/// `relu(x · W1) · W2` with both weights declared `dt`, on CPU.
fn run(dt: DType, w1: &[f32], w2: &[f32], x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("mlp");
    let xi = g.input("x", Shape::new(&[3, 16], DType::F32));
    let a = g.param("w1", Shape::new(&[16, 8], dt));
    let b = g.param("w2", Shape::new(&[8, 4], dt));
    let h = g.mm(xi, a);
    let h = g.relu(h);
    let y = g.mm(h, b);
    g.set_outputs(vec![y]);
    let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g);
    c.set_param_typed("w1", &rlx_runtime::backend::narrow_f32_to_bytes(w1, dt), dt);
    c.set_param_typed("w2", &rlx_runtime::backend::narrow_f32_to_bytes(w2, dt), dt);
    c.run(&[("x", x)]).remove(0)
}

#[test]
fn half_matmul_weights_compute_the_f32_product_of_their_values() {
    let w1: Vec<f32> = (0..128)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 61.0)
        .collect();
    let w2: Vec<f32> = (0..32)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 17.0)
        .collect();
    let x: Vec<f32> = (0..48)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) / 9.0)
        .collect();
    for dt in [DType::F16, DType::BF16] {
        let want = run(DType::F32, &round(&w1, dt), &round(&w2, dt), &x);
        let got = run(dt, &w1, &w2, &x);
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(diff < 1e-5, "{dt:?}: {diff} ({got:?} vs {want:?})");
    }
}

/// `table[ids]` on CPU, ids `idx_dtype`, the table stored `dt` as a package
/// opened with `weight_dtype` stores it (`embedding_tables_as`).
fn gather(dt: DType, idx_dtype: DType, table: &[f32], ids: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("embed");
    let ii = g.input("ids", Shape::new(&[ids.len()], idx_dtype));
    let t = g.param("table", Shape::new(&[10, 6], DType::F32));
    let rows = g.gather(t, ii, 0, Shape::new(&[ids.len(), 6], DType::F32));
    g.set_outputs(vec![rows]);
    if dt != DType::F32 {
        assert_eq!(rlx_ir::retarget::embedding_tables_as(&mut g, dt), 1);
    }
    let mut c = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g);
    c.set_param_typed(
        "table",
        &rlx_runtime::backend::narrow_f32_to_bytes(table, dt),
        dt,
    );
    c.run(&[("ids", ids)]).remove(0)
}

#[test]
fn a_half_embedding_table_gathers_its_rows_widened() {
    let table: Vec<f32> = (0..60)
        .map(|i| ((i * 31 % 47) as f32 - 23.0) / 7.0)
        .collect();
    let ids = [3.0, 0.0, 9.0, 3.0];
    for dt in [DType::F16, DType::BF16] {
        for idx in [DType::I32, DType::I64] {
            let want = gather(DType::F32, idx, &round(&table, dt), &ids);
            assert_eq!(gather(dt, idx, &table, &ids), want, "{dt:?} ids {idx:?}");
        }
    }
}

/// On Metal, an embedding step (half table, gathered rows) beside half
/// matmul weights: the f32 promotion its half rows need keeps the f16 matmul
/// weights and table half (one half node used to widen every weight of the
/// graph; the half table's gather reads f32 ids), widens bf16 ones as before,
/// and the result is the CPU's.
#[cfg(feature = "metal")]
#[test]
fn metal_keeps_half_matmul_weights_beside_a_half_embedding_table() {
    if common::skip_unless(rlx_runtime::Device::Metal) {
        return;
    }
    let _gpu = common::GpuTestGuard::acquire(rlx_runtime::Device::Metal);
    let table: Vec<f32> = (0..60)
        .map(|i| ((i * 31 % 47) as f32 - 23.0) / 7.0)
        .collect();
    let w: Vec<f32> = (0..48)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 17.0)
        .collect();
    let ids = [3.0, 0.0, 9.0, 3.0];
    let run = |device, dt| {
        let mut g = Graph::new("embed_mm");
        let ii = g.input("ids", Shape::new(&[4], DType::I32));
        let t = g.param("table", Shape::new(&[10, 6], DType::F32));
        let wp = g.param("w", Shape::new(&[6, 8], DType::F32));
        let rows = g.gather(t, ii, 0, Shape::new(&[4, 6], DType::F32));
        let y = g.mm(rows, wp);
        g.set_outputs(vec![y]);
        assert_eq!(rlx_ir::retarget::matmul_weights_as(&mut g, dt), 1);
        assert_eq!(rlx_ir::retarget::embedding_tables_as(&mut g, dt), 1);
        let mut c = rlx_runtime::Session::new(device).compile(g);
        let half = |v: &[f32]| rlx_runtime::backend::narrow_f32_to_bytes(v, dt);
        c.set_param_typed("table", &half(&table), dt);
        c.set_param_typed("w", &half(&w), dt);
        c.run(&[("ids", &ids[..])]).remove(0)
    };
    for dt in [DType::F16, DType::BF16] {
        let (cpu, metal) = (
            run(rlx_runtime::Device::Cpu, dt),
            run(rlx_runtime::Device::Metal, dt),
        );
        let diff = cpu
            .iter()
            .zip(&metal)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(diff < 1e-4, "{dt:?}: {diff} ({metal:?} vs {cpu:?})");
    }
}
