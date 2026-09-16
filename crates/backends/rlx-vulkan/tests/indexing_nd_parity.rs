// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native SPIR-V GatherND / GatherElements / ScatterElements / ScatterND vs CPU.
//!
//! Vulkan was the fourth f32-uniform-arena backend routing these four ops
//! through `Step::CpuIndexing` — a readback of data + indices + updates, a CPU
//! pass, and an upload, mid-graph. The shaders are driven by the same
//! `rlx_gpu_dispatch::indexing` plans as the CUDA/ROCm `.cu` and the wgpu WGSL,
//! with one Vulkan-specific limit: shapes and strides ride in the push block
//! (binding 0 and 1 are the activation and weight arenas, and there is no third
//! binding), so the meta budget caps rank/`k` and anything larger stays on CPU.
//!
//! The cases are the ones where wrong addressing returns a plausible tensor
//! rather than an error: a `batch_dims` shift, an index tensor smaller than
//! `data` off-axis, and negative/out-of-range indices. Equality is exact —
//! these ops move f32 bits and do no arithmetic on them.

use rlx_ir::{DType, Graph, ScatterNdReduction, Shape};
use rlx_vulkan::backend::VulkanExecutable;
use std::sync::{Mutex, MutexGuard, OnceLock};

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

fn iota(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 + 1.0).collect()
}

fn cpu(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    use rlx::prelude::*;
    Session::new(Device::Cpu).compile(g).run(inputs).remove(0)
}

fn vk(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    VulkanExecutable::compile(g).run(inputs).remove(0)
}

fn both(what: &str, build: impl Fn() -> Graph, inputs: &[(&str, Vec<f32>)]) {
    let borrowed: Vec<(&str, &[f32])> = inputs.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let want = cpu(build(), &borrowed);
    let got = vk(build(), &borrowed);
    assert_eq!(
        got, want,
        "{what}: vulkan != cpu\n  vk={got:?}\n cpu={want:?}"
    );
}

#[test]
fn gather_nd_matches_cpu() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    both(
        "gather_nd slice",
        || {
            let mut g = Graph::new("gnd");
            let d = g.input("data", Shape::new(&[2, 2, 2], DType::F32));
            let i = g.input("indices", Shape::new(&[2, 1], DType::F32));
            let y = g.gather_nd(d, i, 0, Shape::new(&[2, 2, 2], DType::F32));
            g.set_outputs(vec![y]);
            g
        },
        &[("data", iota(8)), ("indices", vec![1.0, 0.0])],
    );
    // batch_dims = 1: axis 0 is shared, not indexed.
    both(
        "gather_nd batch_dims=1",
        || {
            let mut g = Graph::new("gnd_bd");
            let d = g.input("data", Shape::new(&[2, 3, 4], DType::F32));
            let i = g.input("indices", Shape::new(&[2, 1, 1], DType::F32));
            let y = g.gather_nd(d, i, 1, Shape::new(&[2, 1, 4], DType::F32));
            g.set_outputs(vec![y]);
            g
        },
        &[("data", iota(24)), ("indices", vec![2.0, 0.0])],
    );
    // -1 wraps to the last row, 99 clamps to it, -4 wraps to row 0.
    both(
        "gather_nd wrap/clamp",
        || {
            let mut g = Graph::new("gnd_w");
            let d = g.input("data", Shape::new(&[4, 3], DType::F32));
            let i = g.input("indices", Shape::new(&[3, 1], DType::F32));
            let y = g.gather_nd(d, i, 0, Shape::new(&[3, 3], DType::F32));
            g.set_outputs(vec![y]);
            g
        },
        &[("data", iota(12)), ("indices", vec![-1.0, 99.0, -4.0])],
    );
}

#[test]
fn gather_elements_matches_cpu() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    for (name, dshape, ishape, axis, idx) in [
        (
            "axis=1",
            vec![2usize, 4],
            vec![2usize, 4],
            1i32,
            vec![0.0f32, 2.0, 1.0, 3.0, 1.0, 0.0, 3.0, 2.0],
        ),
        (
            "axis=0",
            vec![3, 3],
            vec![3, 3],
            0,
            vec![1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0],
        ),
        // Index tensor smaller than `data` off-axis: must decompose by the
        // INDICES' strides. Data-strides is right for row 0, wrong after.
        (
            "smaller off-axis",
            vec![4, 5],
            vec![2, 3],
            1,
            vec![0.0, 4.0, 2.0, 1.0, 3.0, 0.0],
        ),
        (
            "wrap/clamp",
            vec![3, 4],
            vec![3, 4],
            1,
            vec![
                -1.0, 7.0, 0.0, -4.0, 2.0, -2.0, 9.0, 1.0, 0.0, 3.0, -3.0, 2.0,
            ],
        ),
    ] {
        let n: usize = dshape.iter().product();
        both(
            &format!("gather_elements {name}"),
            || {
                let mut g = Graph::new("gel");
                let d = g.input("data", Shape::new(&dshape, DType::F32));
                let i = g.input("indices", Shape::new(&ishape, DType::F32));
                let y = g.gather_elements(d, i, axis);
                g.set_outputs(vec![y]);
                g
            },
            &[("data", iota(n)), ("indices", idx.clone())],
        );
    }
}

#[test]
fn scatter_elements_matches_cpu() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    for (name, dshape, ishape, axis, idx) in [
        (
            "axis=1",
            vec![3usize, 3],
            vec![3usize, 3],
            1i32,
            vec![0.0f32, 2.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0],
        ),
        (
            "axis=0",
            vec![3, 3],
            vec![3, 3],
            0,
            vec![2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0],
        ),
        // Untouched slots must survive — this is where the copy prologue shows.
        (
            "smaller off-axis",
            vec![4, 5],
            vec![2, 3],
            1,
            vec![0.0, 4.0, 2.0, 1.0, 3.0, 0.0],
        ),
    ] {
        let n: usize = dshape.iter().product();
        let m = idx.len();
        both(
            &format!("scatter_elements {name}"),
            || {
                let mut g = Graph::new("sel");
                let d = g.input("data", Shape::new(&dshape, DType::F32));
                let i = g.input("indices", Shape::new(&ishape, DType::F32));
                let u = g.input("updates", Shape::new(&ishape, DType::F32));
                let y = g.scatter_elements(d, i, u, axis, ScatterNdReduction::None);
                g.set_outputs(vec![y]);
                g
            },
            &[
                ("data", iota(n)),
                ("indices", idx.clone()),
                ("updates", (0..m).map(|i| -(i as f32) - 1.0).collect()),
            ],
        );
    }
}

#[test]
fn scatter_nd_matches_cpu() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    // Row slices.
    both(
        "scatter_nd rows",
        || {
            let mut g = Graph::new("snd");
            let d = g.input("data", Shape::new(&[4, 3], DType::F32));
            let i = g.input("indices", Shape::new(&[2, 1], DType::F32));
            let u = g.input("updates", Shape::new(&[2, 3], DType::F32));
            let y = g.scatter_nd(d, i, u, ScatterNdReduction::None);
            g.set_outputs(vec![y]);
            g
        },
        &[
            ("data", iota(12)),
            ("indices", vec![0.0, 3.0]),
            ("updates", (0..6).map(|i| -(i as f32) - 1.0).collect()),
        ],
    );
    // Full coordinates, and distinct destinations: ONNX leaves duplicate-index
    // `reduction=none` undefined, and CPU (sequential) and GPU (concurrent)
    // legitimately disagree there, so the case avoids the overlap.
    both(
        "scatter_nd scalars",
        || {
            let mut g = Graph::new("snd_s");
            let d = g.input("data", Shape::new(&[2, 2, 2], DType::F32));
            let i = g.input("indices", Shape::new(&[2, 3], DType::F32));
            let u = g.input("updates", Shape::new(&[2], DType::F32));
            let y = g.scatter_nd(d, i, u, ScatterNdReduction::None);
            g.set_outputs(vec![y]);
            g
        },
        &[
            ("data", iota(8)),
            ("indices", vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
            ("updates", vec![-1.0, -2.0]),
        ],
    );
    // -4 wraps to row 0, 9 clamps to row 3 — distinct destinations.
    both(
        "scatter_nd wrap/clamp",
        || {
            let mut g = Graph::new("snd_w");
            let d = g.input("data", Shape::new(&[4, 3], DType::F32));
            let i = g.input("indices", Shape::new(&[2, 1], DType::F32));
            let u = g.input("updates", Shape::new(&[2, 3], DType::F32));
            let y = g.scatter_nd(d, i, u, ScatterNdReduction::None);
            g.set_outputs(vec![y]);
            g
        },
        &[
            ("data", iota(12)),
            ("indices", vec![-4.0, 9.0]),
            ("updates", (0..6).map(|i| -(i as f32) - 1.0).collect()),
        ],
    );
}

#[test]
fn declined_shapes_still_match_cpu() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    // `Add` needs an f32 atomic Vulkan does not guarantee, so this keeps the
    // host route. A decline that routes nowhere is silent zeros, so the
    // fallback has to stay wired.
    both(
        "scatter_nd add (host route)",
        || {
            let mut g = Graph::new("snd_add");
            let d = g.input("data", Shape::new(&[4, 3], DType::F32));
            let i = g.input("indices", Shape::new(&[3, 1], DType::F32));
            let u = g.input("updates", Shape::new(&[3, 3], DType::F32));
            let y = g.scatter_nd(d, i, u, ScatterNdReduction::Add);
            g.set_outputs(vec![y]);
            g
        },
        &[
            ("data", iota(12)),
            ("indices", vec![0.0, 1.0, 1.0]),
            ("updates", (0..9).map(|i| -(i as f32) - 1.0).collect()),
        ],
    );
    // Rank 5 GatherElements needs 20 meta words; the push block holds 16, so
    // this is the cap the shaders' comment claims. Still has to be correct.
    both(
        "gather_elements rank5 (over meta cap)",
        || {
            let mut g = Graph::new("gel5");
            let d = g.input("data", Shape::new(&[2, 2, 2, 2, 2], DType::F32));
            let i = g.input("indices", Shape::new(&[2, 2, 2, 2, 2], DType::F32));
            let y = g.gather_elements(d, i, 4);
            g.set_outputs(vec![y]);
            g
        },
        &[
            ("data", iota(32)),
            ("indices", (0..32).map(|i| (i % 2) as f32).collect()),
        ],
    );
}
