// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A hosted `BufferCopy` defers its device write to the host mirror; that
//! write has to land before the next GPU step reads it.
//!
//! The end-of-host-step flush that does this sits after the host branches, and
//! the hosted `BufferCopy` branch `continue`s past it. So a `KvAppend` (one
//! `BufferCopy` of the new row into the cache) followed by a GPU op on the cache
//! ran that op against the cache *before* the append, and the row reached the
//! device one host step later. Every decode step of a KV-cached LM on discrete
//! NVIDIA was wrong (all copies are hosted there: KittenTTS 2's Qwen3 decoded a
//! different token from step 1), while Apple wgpu, which copies on the GPU, was
//! exact. `RLX_WGPU_HOST_BUFFER_COPY=1` forces the hosted copy on any adapter.
//! Own test binary: the setting is process-global.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

#[test]
fn hosted_kv_append_reaches_the_device_before_the_next_gpu_op() {
    rlx_ir::env::set("RLX_WGPU_HOST_BUFFER_COPY", "1");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }

    let (rows, d) = (3usize, 8usize);
    let pos = rows - 1;
    let cache_sh = Shape::new(&[1, rows, d], DType::F32);
    let row_sh = Shape::new(&[1, 1, d], DType::F32);

    let mut g = Graph::new("kv_append_flush");
    let cache = g.input("cache", cache_sh.clone());
    let r = g.input("r", row_sh.clone());
    let b = g.input("b", row_sh.clone());
    // The row is computed on the GPU, so the copy's source is device-resident.
    let row = g.add_node(Op::Binary(BinaryOp::Add), vec![r, b], row_sh);
    let kv = g.add_node(
        Op::KvAppend { axis: 1, pos },
        vec![cache, row],
        cache_sh.clone(),
    );
    let y = g.add_node(Op::Binary(BinaryOp::Mul), vec![kv, kv], cache_sh);
    g.set_outputs(vec![y]);

    // The slot being appended holds a sentinel, so reading it before the
    // append lands is unmistakable.
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
        "the GPU op after a hosted KvAppend read the cache before the append \
         (max|Δ| = {max_d:.3e})\n got {:?}\nwant {want:?}",
        got[0]
    );
}
