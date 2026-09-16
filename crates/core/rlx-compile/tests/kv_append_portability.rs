// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::KvAppend` must be emittable by a PORTABLE model.
//!
//! It is implemented natively on CPU, Metal, CUDA, ROCm, wgpu and Vulkan; MLX
//! takes the lowering. Before the lowering existed only Metal/CUDA/ROCm had it
//! and a graph containing it could not run anywhere else — so a model crate
//! that advertises `--device cpu` could not use it at all, and stayed on
//! `Concat` on every backend including the three with the fast path.
//!
//! Carbon-500M decode on Metal spends 36.9% of GPU time in that concat (112
//! dispatches, 4 per layer across 28 layers), which is as much as every matmul
//! combined and grows with context while the real work is one row.

use rlx_ir::{DType, Graph, Op, OpKind, Shape};

const F: DType = DType::F32;

/// `[1, seq_cap, width]` with `axis = 1` — nothing precedes the axis, so the
/// `[..pos+1]` output is a contiguous prefix and CAN alias the cache. This is
/// the layout the native row write is valid for.
fn kv_graph(pos: usize) -> Graph {
    let mut g = Graph::new("kv_portable");
    let cache = g.input("cache", Shape::new(&[1, 8, 4], F));
    let row = g.input("row", Shape::new(&[1, 1, 4], F));
    let out = g.add_node(
        Op::KvAppend { axis: 1, pos },
        vec![cache, row],
        Shape::new(&[1, pos + 1, 4], F),
    );
    g.set_outputs(vec![out]);
    g
}

/// `[1, heads, seq_cap, dim]` with `axis = 2` — `heads > 1` precedes the axis,
/// so the prefix is strided and cannot alias the cache at any offset.
fn kv_graph_strided(pos: usize) -> Graph {
    let mut g = Graph::new("kv_strided");
    let cache = g.input("cache", Shape::new(&[1, 2, 8, 4], F));
    let row = g.input("row", Shape::new(&[1, 2, 1, 4], F));
    let out = g.add_node(
        Op::KvAppend { axis: 2, pos },
        vec![cache, row],
        Shape::new(&[1, 2, pos + 1, 4], F),
    );
    g.set_outputs(vec![out]);
    g
}

/// A backend that does NOT claim `KvAppend` must receive a graph it can run.
#[test]
fn a_backend_without_kv_append_gets_it_lowered_away() {
    for pos in [0usize, 1, 5] {
        // A backend whose `supported_ops` omits KvAppend (mlx, and every
        // accelerator backend). NOT `&[]` — `legalize_for_backend` treats an
        // empty list as "no restriction", so passing it means "supports
        // everything" and the lowering never fires. The list must name the ops
        // the fallback needs.
        let lowered = rlx_compile::rewrite::rewrite_for_backend(
            kv_graph(pos),
            &[OpKind::Input, OpKind::Narrow, OpKind::Concat],
        );
        assert!(
            !lowered
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::KvAppend { .. })),
            "pos={pos}: KvAppend survived for a backend that cannot run it"
        );
    }
}

/// A backend that DOES claim it must keep the native single-row write — the
/// whole point is that Metal/CUDA/ROCm avoid the O(context) copy.
#[test]
fn a_backend_with_kv_append_keeps_it() {
    // A backend that claims it (cpu, metal, cuda, rocm, wgpu, vulkan).
    let kept = rlx_compile::rewrite::rewrite_for_backend(
        kv_graph(5),
        &[
            OpKind::Input,
            OpKind::Narrow,
            OpKind::Concat,
            OpKind::KvAppend,
        ],
    );
    assert!(
        kept.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::KvAppend { .. })),
        "a native backend lost its O(1) KV append to the fallback"
    );
}

/// A native backend must ALSO lose `KvAppend` when the output cannot alias the
/// cache — capability is not enough, the shape has to permit it.
///
/// `[1, heads, seq, dim]` with `axis = 2` is the layout most attention code
/// reaches for, and it is exactly the one the fast path cannot serve: the
/// `[..pos+1]` prefix of a heads-strided buffer is not a contiguous range, so a
/// row write into the aliased slot returns the wrong rows. Correctness first —
/// the concat is slower but right.
#[test]
fn a_native_backend_still_lowers_a_non_aliasable_shape() {
    let supported = [
        OpKind::Input,
        OpKind::Concat,
        OpKind::Narrow,
        OpKind::KvAppend,
    ];
    for pos in [1usize, 3] {
        let g = rlx_compile::rewrite::rewrite_for_backend(kv_graph_strided(pos), &supported);
        assert!(
            !g.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::KvAppend { .. })),
            "pos={pos}: a strided-prefix KvAppend survived on a native backend — \
             its output would alias a buffer whose prefix is not contiguous"
        );
    }
}
