// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native WGSL GatherND / GatherElements / ScatterElements / ScatterND on wgpu
//! vs CPU.
//!
//! These four were the last ops taking `Step::CpuIndexing` on wgpu — a readback
//! of data + indices + updates, a CPU pass, and an upload, mid-graph.
//! `indexing_nd.wgsl` replaces that for the shapes
//! `rlx_gpu_dispatch::indexing` can plan.
//!
//! The cases below are chosen for the arithmetic that is easy to get wrong and
//! impossible to notice: a `batch_dims` shift, an index tensor smaller than
//! `data` off-axis (decompose by the *indices'* strides, not the data's),
//! negative indices that wrap and out-of-range indices that clamp, and a
//! ScatterND with `k = 5`. Every one of them returns a plausible tensor when
//! the addressing is wrong.
//!
//! `rlx-cpu/tests/gpu_indexing_plan_parity.rs` checks the same arithmetic
//! device-free by replaying the kernel bodies; this checks that the shader the
//! GPU actually compiled agrees. Equality is exact — these ops move f32 bits
//! around and do no arithmetic on them, so any tolerance would be hiding
//! something.

#![cfg(all(feature = "gpu", feature = "cpu"))]

use rlx_ir::{DType, Graph, ScatterNdReduction, Shape};
use rlx_runtime::{Device, Session};

mod common;

fn iota(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 + 1.0).collect()
}

/// Run `build` on both devices and require bit-equality.
fn both(what: &str, build: impl Fn() -> (Graph, Vec<(&'static str, Vec<f32>)>)) {
    let run = |device: Device| {
        let (g, inputs) = build();
        let borrowed: Vec<(&str, &[f32])> =
            inputs.iter().map(|(k, v)| (*k, v.as_slice())).collect();
        Session::new(device)
            .compile(g)
            .run(&borrowed)
            .pop()
            .unwrap()
    };
    let cpu = run(Device::Cpu);
    let gpu = run(Device::Gpu);
    assert_eq!(
        gpu, cpu,
        "{what}: wgpu != cpu\n  gpu={gpu:?}\n  cpu={cpu:?}"
    );
}

// ---------------------------------------------------------------------------
// GatherND
// ---------------------------------------------------------------------------

#[test]
fn gather_nd_wgpu_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }

    // Partial coordinate: each tuple names a contiguous 4-element slice.
    both("gather_nd slice", || {
        let mut g = Graph::new("gnd_slice");
        let data = g.input("data", Shape::new(&[2, 2, 2], DType::F32));
        let idx = g.input("indices", Shape::new(&[2, 1], DType::F32));
        let y = g.gather_nd(data, idx, 0, Shape::new(&[2, 2, 2], DType::F32));
        g.set_outputs(vec![y]);
        (g, vec![("data", iota(8)), ("indices", vec![1.0, 0.0])])
    });

    // Full coordinate: one scalar per tuple.
    both("gather_nd scalar", || {
        let mut g = Graph::new("gnd_scalar");
        let data = g.input("data", Shape::new(&[3, 4], DType::F32));
        let idx = g.input("indices", Shape::new(&[3, 2], DType::F32));
        let y = g.gather_nd(data, idx, 0, Shape::new(&[3], DType::F32));
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(12)),
                ("indices", vec![0.0, 3.0, 2.0, 1.0, 1.0, 0.0]),
            ],
        )
    });
}

#[test]
fn gather_nd_batch_dims_wgpu_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // batch_dims = 1: axis 0 is shared, not indexed. The kernel's `meta` is
    // shifted by `b`, and `batch_stride` carries the per-batch jump — get
    // either wrong and batch 0 is still correct while batch 1 is not.
    both("gather_nd batch_dims=1", || {
        let mut g = Graph::new("gnd_bd");
        let data = g.input("data", Shape::new(&[2, 3, 4], DType::F32));
        let idx = g.input("indices", Shape::new(&[2, 1, 1], DType::F32));
        let y = g.gather_nd(data, idx, 1, Shape::new(&[2, 1, 4], DType::F32));
        g.set_outputs(vec![y]);
        (g, vec![("data", iota(24)), ("indices", vec![2.0, 0.0])])
    });
}

#[test]
fn gather_nd_wrapping_indices_wgpu_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // -1 wraps to the last row; 99 clamps to it; -4 wraps to row 0.
    both("gather_nd wrap/clamp", || {
        let mut g = Graph::new("gnd_wrap");
        let data = g.input("data", Shape::new(&[4, 3], DType::F32));
        let idx = g.input("indices", Shape::new(&[3, 1], DType::F32));
        let y = g.gather_nd(data, idx, 0, Shape::new(&[3, 3], DType::F32));
        g.set_outputs(vec![y]);
        (
            g,
            vec![("data", iota(12)), ("indices", vec![-1.0, 99.0, -4.0])],
        )
    });
}

// ---------------------------------------------------------------------------
// GatherElements
// ---------------------------------------------------------------------------

#[test]
fn gather_elements_wgpu_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }

    both("gather_elements axis=1", || {
        let mut g = Graph::new("gel1");
        let data = g.input("data", Shape::new(&[2, 4], DType::F32));
        let idx = g.input("indices", Shape::new(&[2, 4], DType::F32));
        let y = g.gather_elements(data, idx, 1);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(8)),
                ("indices", vec![0.0, 2.0, 1.0, 3.0, 1.0, 0.0, 3.0, 2.0]),
            ],
        )
    });

    both("gather_elements axis=0", || {
        let mut g = Graph::new("gel0");
        let data = g.input("data", Shape::new(&[3, 3], DType::F32));
        let idx = g.input("indices", Shape::new(&[3, 3], DType::F32));
        let y = g.gather_elements(data, idx, 0);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(9)),
                ("indices", vec![1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0]),
            ],
        )
    });

    // Rank 3, gather on the innermost axis.
    both("gather_elements rank3 axis=2", || {
        let mut g = Graph::new("gel3");
        let data = g.input("data", Shape::new(&[2, 3, 4], DType::F32));
        let idx = g.input("indices", Shape::new(&[2, 3, 2], DType::F32));
        let y = g.gather_elements(data, idx, 2);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(24)),
                (
                    "indices",
                    vec![0.0, 3.0, 1.0, 2.0, 3.0, 0.0, 2.0, 2.0, 1.0, 3.0, 0.0, 1.0],
                ),
            ],
        )
    });
}

#[test]
fn gather_elements_smaller_off_axis_wgpu_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // ONNX allows the index tensor to be smaller than `data` off-axis. The flat
    // position must then decompose by the INDICES' strides. Using the data's
    // strides instead is right for row 0 and wrong for every row after it,
    // which is precisely the kind of bug that survives a one-row test.
    both("gather_elements [4,5] <- idx [2,3]", || {
        let mut g = Graph::new("gel_small");
        let data = g.input("data", Shape::new(&[4, 5], DType::F32));
        let idx = g.input("indices", Shape::new(&[2, 3], DType::F32));
        let y = g.gather_elements(data, idx, 1);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(20)),
                ("indices", vec![0.0, 4.0, 2.0, 1.0, 3.0, 0.0]),
            ],
        )
    });
}

#[test]
fn gather_elements_wrapping_indices_wgpu_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    both("gather_elements wrap/clamp", || {
        let mut g = Graph::new("gel_wrap");
        let data = g.input("data", Shape::new(&[3, 4], DType::F32));
        let idx = g.input("indices", Shape::new(&[3, 4], DType::F32));
        let y = g.gather_elements(data, idx, 1);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(12)),
                (
                    "indices",
                    vec![
                        -1.0, 7.0, 0.0, -4.0, 2.0, -2.0, 9.0, 1.0, 0.0, 3.0, -3.0, 2.0,
                    ],
                ),
            ],
        )
    });
}

// ---------------------------------------------------------------------------
// ScatterElements
// ---------------------------------------------------------------------------

#[test]
fn scatter_elements_wgpu_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }

    both("scatter_elements axis=1", || {
        let mut g = Graph::new("sel1");
        let data = g.input("data", Shape::new(&[3, 3], DType::F32));
        let idx = g.input("indices", Shape::new(&[3, 3], DType::F32));
        let upd = g.input("updates", Shape::new(&[3, 3], DType::F32));
        let y = g.scatter_elements(data, idx, upd, 1, ScatterNdReduction::None);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(9)),
                ("indices", vec![0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0]),
                ("updates", (0..9).map(|i| -(i as f32) - 1.0).collect()),
            ],
        )
    });

    both("scatter_elements axis=0", || {
        let mut g = Graph::new("sel0");
        let data = g.input("data", Shape::new(&[3, 3], DType::F32));
        let idx = g.input("indices", Shape::new(&[3, 3], DType::F32));
        let upd = g.input("updates", Shape::new(&[3, 3], DType::F32));
        let y = g.scatter_elements(data, idx, upd, 0, ScatterNdReduction::None);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(9)),
                ("indices", vec![2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0]),
                ("updates", (0..9).map(|i| -(i as f32) - 1.0).collect()),
            ],
        )
    });
}

#[test]
fn scatter_elements_smaller_off_axis_wgpu_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // Same stride-decomposition trap as the gather, plus the untouched slots
    // must survive: this is where the `copy_sanitize` prologue is proved.
    both("scatter_elements [4,5] <- idx [2,3]", || {
        let mut g = Graph::new("sel_small");
        let data = g.input("data", Shape::new(&[4, 5], DType::F32));
        let idx = g.input("indices", Shape::new(&[2, 3], DType::F32));
        let upd = g.input("updates", Shape::new(&[2, 3], DType::F32));
        let y = g.scatter_elements(data, idx, upd, 1, ScatterNdReduction::None);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(20)),
                ("indices", vec![0.0, 4.0, 2.0, 1.0, 3.0, 0.0]),
                ("updates", (0..6).map(|i| -(i as f32) - 1.0).collect()),
            ],
        )
    });
}

// ---------------------------------------------------------------------------
// ScatterND
// ---------------------------------------------------------------------------

#[test]
fn scatter_nd_wgpu_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }

    // Slice writes: each tuple names a contiguous row.
    both("scatter_nd row slices", || {
        let mut g = Graph::new("snd_row");
        let data = g.input("data", Shape::new(&[4, 3], DType::F32));
        let idx = g.input("indices", Shape::new(&[2, 1], DType::F32));
        let upd = g.input("updates", Shape::new(&[2, 3], DType::F32));
        let y = g.scatter_nd(data, idx, upd, ScatterNdReduction::None);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(12)),
                ("indices", vec![0.0, 3.0]),
                ("updates", (0..6).map(|i| -(i as f32) - 1.0).collect()),
            ],
        )
    });

    // Full coordinates: one scalar per tuple.
    both("scatter_nd scalars", || {
        let mut g = Graph::new("snd_scalar");
        let data = g.input("data", Shape::new(&[2, 2, 2], DType::F32));
        let idx = g.input("indices", Shape::new(&[2, 3], DType::F32));
        let upd = g.input("updates", Shape::new(&[2], DType::F32));
        let y = g.scatter_nd(data, idx, upd, ScatterNdReduction::None);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(8)),
                ("indices", vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
                ("updates", vec![-1.0, -2.0]),
            ],
        )
    });
}

#[test]
fn scatter_nd_deep_tuple_wgpu_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // k = 5. The pre-existing CUDA `scatter_nd.cu` capped k at 4 because its
    // strides were scalar kernel arguments; with `meta` carrying them there is
    // no cap, and this is the case that proves it.
    both("scatter_nd k=5", || {
        let mut g = Graph::new("snd_k5");
        let data = g.input("data", Shape::new(&[2, 2, 2, 2, 2], DType::F32));
        let idx = g.input("indices", Shape::new(&[2, 5], DType::F32));
        let upd = g.input("updates", Shape::new(&[2], DType::F32));
        let y = g.scatter_nd(data, idx, upd, ScatterNdReduction::None);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(32)),
                (
                    "indices",
                    vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                ),
                ("updates", vec![-1.0, -2.0]),
            ],
        )
    });
}

#[test]
fn scatter_nd_wrapping_indices_wgpu_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // -4 wraps to row 0 and 9 clamps to row 3 — two *distinct* destinations.
    //
    // That distinctness is load-bearing, not incidental. ONNX leaves ScatterND
    // with duplicate indices and `reduction=none` undefined; rlx-cpu resolves it
    // sequentially (last write wins) while a GPU kernel writes in whatever order
    // the scheduler picks. An earlier draft of this case used `[-1, 9, -4]`,
    // where -1 and 9 both land on row 3, and it failed exactly that way: CPU
    // took the second update, wgpu took the first. Both are legal. Asserting
    // either would be asserting a race, so the case avoids the overlap and the
    // divergence is documented on the kernels instead.
    both("scatter_nd wrap/clamp", || {
        let mut g = Graph::new("snd_wrap");
        let data = g.input("data", Shape::new(&[4, 3], DType::F32));
        let idx = g.input("indices", Shape::new(&[2, 1], DType::F32));
        let upd = g.input("updates", Shape::new(&[2, 3], DType::F32));
        let y = g.scatter_nd(data, idx, upd, ScatterNdReduction::None);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(12)),
                ("indices", vec![-4.0, 9.0]),
                ("updates", (0..6).map(|i| -(i as f32) - 1.0).collect()),
            ],
        )
    });
}

// ---------------------------------------------------------------------------
// The host route still works, and agrees
// ---------------------------------------------------------------------------

#[test]
fn accumulating_scatters_take_the_host_route_and_match_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    // Core WGSL has no f32 atomics, so `ScatterNdReduction::Add` is declined by
    // the wgpu lowering and keeps `Step::CpuIndexing`. That fallback has to
    // still be wired — a decline that routes nowhere is silent zeros.
    both("scatter_nd add (host route)", || {
        let mut g = Graph::new("snd_add");
        let data = g.input("data", Shape::new(&[4, 3], DType::F32));
        let idx = g.input("indices", Shape::new(&[3, 1], DType::F32));
        let upd = g.input("updates", Shape::new(&[3, 3], DType::F32));
        let y = g.scatter_nd(data, idx, upd, ScatterNdReduction::Add);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(12)),
                ("indices", vec![0.0, 1.0, 1.0]),
                ("updates", (0..9).map(|i| -(i as f32) - 1.0).collect()),
            ],
        )
    });

    // `Mul` is declined by the shared planner on every backend.
    both("scatter_nd mul (host route)", || {
        let mut g = Graph::new("snd_mul");
        let data = g.input("data", Shape::new(&[4, 3], DType::F32));
        let idx = g.input("indices", Shape::new(&[2, 1], DType::F32));
        let upd = g.input("updates", Shape::new(&[2, 3], DType::F32));
        let y = g.scatter_nd(data, idx, upd, ScatterNdReduction::Mul);
        g.set_outputs(vec![y]);
        (
            g,
            vec![
                ("data", iota(12)),
                ("indices", vec![0.0, 2.0]),
                ("updates", vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            ],
        )
    });
}
