// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A hosted `KvAppend` defers its row at `cache + pos·row`; a *host* step that
//! then reads the cache asks the mirror for `cache`. On a miss it flushed only
//! a deferred write keyed at exactly that offset, found none, and read the
//! device without the row. On discrete NVIDIA (elementwise, structural ops and
//! copies all hosted) every decode step of a KV-cached LM attended to a cache
//! missing its newest token. `RLX_WGPU_FORCE_HOST=1` plus
//! `RLX_WGPU_HOST_BUFFER_COPY=1` reproduce that routing on any adapter; see
//! `host_buffer_copy_flush.rs` for the GPU-reader half. Own test binary: both
//! settings are process-global.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

#[test]
fn host_reader_of_a_hosted_kv_append_sees_the_new_row() {
    rlx_ir::env::set("RLX_WGPU_FORCE_HOST", "1");
    rlx_ir::env::set("RLX_WGPU_HOST_BUFFER_COPY", "1");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }

    let (rows, d) = (3usize, 8usize);
    let pos = rows - 1;
    let cache_sh = Shape::new(&[1, rows, d], DType::F32);
    let row_sh = Shape::new(&[1, 1, d], DType::F32);

    let mut g = Graph::new("kv_append_host_reader");
    let cache = g.input("cache", cache_sh.clone());
    let r = g.input("r", row_sh.clone());
    let b = g.input("b", row_sh.clone());
    let row = g.add_node(Op::Binary(BinaryOp::Add), vec![r, b], row_sh);
    let kv = g.add_node(
        Op::KvAppend { axis: 1, pos },
        vec![cache, row],
        cache_sh.clone(),
    );
    // Hosted under FORCE_HOST: reads the cache through the mirror at `cache`.
    let y = g.add_node(Op::Binary(BinaryOp::Mul), vec![kv, kv], cache_sh);
    g.set_outputs(vec![y]);

    let mut cache_v: Vec<f32> = (0..rows * d).map(|i| i as f32 * 0.1).collect();
    cache_v[pos * d..].fill(99.0);
    let r_v: Vec<f32> = (0..d).map(|i| i as f32 * 0.5 - 1.0).collect();
    let b_v: Vec<f32> = (0..d).map(|i| 0.25 + i as f32 * 0.01).collect();

    let mut exe = WgpuExecutable::compile(g);
    let got = exe.run(&[("cache", &cache_v), ("r", &r_v), ("b", &b_v)]);

    let mut want: Vec<f32> = cache_v.iter().map(|v| v * v).collect();
    for i in 0..d {
        let v = r_v[i] + b_v[i];
        want[pos * d + i] = v * v;
    }
    let max_d = got[0]
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_d < 1e-5,
        "a host step reading the cache after a hosted KvAppend missed the row \
         (max|Δ| = {max_d:.3e})\n got {:?}\nwant {want:?}",
        got[0]
    );
}
