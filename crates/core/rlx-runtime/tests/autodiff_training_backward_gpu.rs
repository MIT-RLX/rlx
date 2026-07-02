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

//
// GPU parity for fused training backward ops vs CPU thunks.

use rlx_compile::legalize_broadcast::run_with_remap;
use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

fn write_slot(arena: &mut Arena, id: NodeId, data: &[f32]) {
    let off = arena.byte_offset(id);
    unsafe {
        let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
        for (i, &v) in data.iter().enumerate() {
            *p.add(i) = v;
        }
    }
}

fn read_slot(arena: &Arena, id: NodeId, n: usize) -> Vec<f32> {
    let off = arena.byte_offset(id);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..n).map(|i| *p.add(i)).collect()
    }
}

fn cpu_run(graph: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    let (graph, remap) = run_with_remap(graph);
    let r = |id: NodeId| *remap.get(&id).unwrap_or(&id);
    let plan = rlx_opt::memory::plan_memory(&graph);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&graph, &arena);
    let slots: Vec<_> = graph
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name } => Some((name.as_str(), n.id)),
            _ => None,
        })
        .collect();
    let n_out: usize = graph
        .nodes()
        .iter()
        .find(|n| n.id == r(graph.outputs[0]))
        .map(|n| n.shape.num_elements().unwrap())
        .unwrap();
    for (name, data) in inputs {
        let id = r(slots
            .iter()
            .find(|(n, _)| *n == *name)
            .unwrap_or_else(|| panic!("missing input {name}"))
            .1);
        write_slot(&mut arena, id, data);
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    read_slot(&arena, r(graph.outputs[0]), n_out)
}

#[cfg(any(
    feature = "gpu",
    feature = "cuda",
    feature = "rocm",
    all(target_os = "macos", feature = "metal")
))]
fn assert_close(cpu: &[f32], gpu: &[f32], tol: f32) {
    assert_eq!(cpu.len(), gpu.len());
    let max = cpu
        .iter()
        .zip(gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max < tol, "max_abs_diff={max}");
}

// --- RMSNorm backward input ---

fn build_rms_norm_bwd_input_graph() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("rms_bwd_in");
    let x = g.input("x", Shape::new(&[3, 4], f));
    let gamma = g.input("gamma", Shape::new(&[4], f));
    let beta = g.input("beta", Shape::new(&[4], f));
    let dy = g.input("dy", Shape::new(&[3, 4], f));
    let dx = g.rms_norm_backward_input(x, gamma, beta, dy, -1, 1e-5);
    g.set_outputs(vec![dx]);
    g
}

fn rms_norm_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let rows = 3usize;
    let h = 4usize;
    let x: Vec<f32> = (0..rows * h).map(|i| 0.1 * (i as f32 - 3.0)).collect();
    let gamma: Vec<f32> = (0..h).map(|i| 0.5 + 0.2 * i as f32).collect();
    let beta: Vec<f32> = vec![0.01; h];
    let dy: Vec<f32> = (0..rows * h).map(|i| 1.0 + 0.05 * i as f32).collect();
    (x, gamma, beta, dy)
}

#[test]
fn cpu_rms_norm_backward_input_finite() {
    let (x, gamma, beta, dy) = rms_norm_inputs();
    let got = cpu_run(
        build_rms_norm_bwd_input_graph(),
        &[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)],
    );
    assert!(got.iter().all(|v| v.is_finite()));
}

/// RMSNorm gamma/beta gradients vs finite differences. `dx` is FD-verified
/// elsewhere; this pins the affine-parameter gradients so the whole RMSNorm
/// backward (input + gamma + beta) is ground-truthed, not just the input term.
#[test]
fn cpu_rms_norm_backward_gamma_beta_matches_finite_difference() {
    let (rows, h) = (3usize, 4usize);
    let f = DType::F32;
    let x: Vec<f32> = (0..rows * h).map(|i| 0.1 * (i as f32 - 3.0)).collect();
    let gamma: Vec<f32> = (0..h).map(|i| 0.5 + 0.2 * i as f32).collect();
    let beta: Vec<f32> = (0..h).map(|i| 0.01 + 0.05 * i as f32).collect();
    let dy: Vec<f32> = (0..rows * h).map(|i| 1.0 + 0.05 * i as f32).collect();

    // dgamma
    let mut g = Graph::new("rms_bwd_g");
    let xn = g.input("x", Shape::new(&[rows, h], f));
    let gn = g.input("gamma", Shape::new(&[h], f));
    let bn = g.input("beta", Shape::new(&[h], f));
    let dyn_in = g.input("dy", Shape::new(&[rows, h], f));
    let dgamma = g.rms_norm_backward_gamma(xn, gn, bn, dyn_in, -1, 1e-5);
    g.set_outputs(vec![dgamma]);
    let analytic_g = cpu_run(
        g,
        &[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)],
    );
    let (x_c, beta_c) = (x.clone(), beta.clone());
    assert_dx_matches_fd(&analytic_g, &gamma, &dy, |gv| {
        forward_rms_norm(&x_c, gv, &beta_c)
    });

    // dbeta
    let mut g = Graph::new("rms_bwd_b");
    let xn = g.input("x", Shape::new(&[rows, h], f));
    let gn = g.input("gamma", Shape::new(&[h], f));
    let bn = g.input("beta", Shape::new(&[h], f));
    let dyn_in = g.input("dy", Shape::new(&[rows, h], f));
    let dbeta = g.rms_norm_backward_beta(xn, gn, bn, dyn_in, -1, 1e-5);
    g.set_outputs(vec![dbeta]);
    let analytic_b = cpu_run(
        g,
        &[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)],
    );
    let (x_c, gamma_c) = (x.clone(), gamma.clone());
    assert_dx_matches_fd(&analytic_b, &beta, &dy, |bv| {
        forward_rms_norm(&x_c, &gamma_c, bv)
    });
}

/// The DECOMPOSE route (`compose_rms_norm_backward_input`) — distinct from the
/// native CPU kernel — is what runs on CoreML/ANE and any backend lacking a native
/// RMSNorm-backward kernel. FD it directly: the ANE-vs-decompose parity test can't
/// catch a bug shared by both (exactly how the `1/r` error stayed hidden).
#[test]
fn decompose_rms_norm_backward_input_matches_finite_difference() {
    let (x, gamma, beta, dy) = rms_norm_inputs();
    let decomposed =
        rlx_opt::rlx_autodiff::decompose_backward_ops_except(build_rms_norm_bwd_input_graph(), &[]);
    let analytic = cpu_run(
        decomposed,
        &[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)],
    );
    let (gamma_c, beta_c) = (gamma.clone(), beta.clone());
    assert_dx_matches_fd(&analytic, &x, &dy, |xv| {
        forward_rms_norm(xv, &gamma_c, &beta_c)
    });
}

/// Forward `y = RmsNorm(x, gamma, beta)` on CPU, returned flat.
fn forward_rms_norm(x: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let f = DType::F32;
    let mut g = Graph::new("rms_fwd");
    let xn = g.input("x", Shape::new(&[3, 4], f));
    let gn = g.input("gamma", Shape::new(&[4], f));
    let bn = g.input("beta", Shape::new(&[4], f));
    let y = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-5,
        },
        vec![xn, gn, bn],
        Shape::new(&[3, 4], f),
    );
    g.set_outputs(vec![y]);
    cpu_run(g, &[("x", x), ("gamma", gamma), ("beta", beta)])
}

/// Ground-truth guard for the RMSNorm input-gradient kernel: with `dy` as fixed
/// cotangents, `dx = ∂(Σ dyⱼ·yⱼ)/∂x`, so the kernel output must match a central
/// finite difference of the forward. Cross-backend parity tests can't catch a bug
/// shared by every backend — the historical RMSNorm `1/r` error (an extra `inv_r`
/// on the cross term) sailed through them because CPU/GPU/ANE were all wrong
/// identically. Finite differences are the independent oracle.
#[test]
fn cpu_rms_norm_backward_input_matches_finite_difference() {
    let (x, gamma, beta, dy) = rms_norm_inputs();
    let analytic = cpu_run(
        build_rms_norm_bwd_input_graph(),
        &[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)],
    );
    let loss = |xv: &[f32]| -> f32 {
        forward_rms_norm(xv, &gamma, &beta)
            .iter()
            .zip(&dy)
            .map(|(y, d)| y * d)
            .sum()
    };
    let step = 1e-3f32;
    let fd: Vec<f32> = (0..x.len())
        .map(|i| {
            let mut p = x.clone();
            p[i] += step;
            let mut m = x.clone();
            m[i] -= step;
            (loss(&p) - loss(&m)) / (2.0 * step)
        })
        .collect();
    let max = analytic
        .iter()
        .zip(&fd)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    // The 1/r bug produced max diffs ~0.8; a correct kernel sits at FD truncation (~3e-4).
    assert!(
        max < 2e-3,
        "rms dx vs finite-diff max_abs_diff={max}\n  analytic={analytic:?}\n  fd={fd:?}"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_rms_norm_backward_input_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("skip wgpu_rms_norm_backward_input_matches_cpu: no adapter");
        return;
    }
    let (x, gamma, beta, dy) = rms_norm_inputs();
    let bwd = build_rms_norm_bwd_input_graph();
    let want = cpu_run(
        bwd.clone(),
        &[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)],
    );
    use rlx_wgpu::backend::WgpuExecutable;
    let mut exe = WgpuExecutable::compile(bwd);
    let got = exe
        .run(&[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)])
        .remove(0);
    assert_close(&want, &got, 1e-4);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rms_norm_backward_input_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session};
    let (x, gamma, beta, dy) = rms_norm_inputs();
    let bwd = build_rms_norm_bwd_input_graph();
    let want = cpu_run(
        bwd.clone(),
        &[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)],
    );
    let session = Session::new(Device::Cuda);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let got = compiled
        .run(&[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)])
        .remove(0);
    assert_close(&want, &got, 1e-4);
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_rms_norm_backward_input_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session, is_available};
    if !is_available(Device::Rocm) {
        eprintln!("skip rocm_rms_norm_backward_input_matches_cpu (unavailable)");
        return;
    }
    let (x, gamma, beta, dy) = rms_norm_inputs();
    let bwd = build_rms_norm_bwd_input_graph();
    let want = cpu_run(
        bwd.clone(),
        &[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)],
    );
    let session = Session::new(Device::Rocm);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let got = compiled
        .run(&[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)])
        .remove(0);
    assert_close(&want, &got, 1e-4);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_rms_norm_backward_input_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session};
    let (x, gamma, beta, dy) = rms_norm_inputs();
    let bwd = build_rms_norm_bwd_input_graph();
    let want = cpu_run(
        bwd.clone(),
        &[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)],
    );
    let session = Session::new(Device::Metal);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let got = compiled
        .run(&[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)])
        .remove(0);
    assert_close(&want, &got, 1e-4);
}

// --- RoPE backward ---

#[cfg(any(
    feature = "gpu",
    feature = "cuda",
    feature = "rocm",
    all(target_os = "macos", feature = "metal")
))]
fn build_rope_bwd_graph() -> Graph {
    let f = DType::F32;
    let b = 1usize;
    let s = 2usize;
    let hd = 8usize;
    let tab = hd / 2;
    let mut g = Graph::new("rope_bwd");
    let dy = g.input("dy", Shape::new(&[b, s, hd], f));
    let cos = g.input("cos", Shape::new(&[s, tab], f));
    let sin = g.input("sin", Shape::new(&[s, tab], f));
    let dx = g.rope_backward(dy, cos, sin, hd, 6);
    g.set_outputs(vec![dx]);
    g
}

#[cfg(any(
    feature = "gpu",
    feature = "cuda",
    feature = "rocm",
    all(target_os = "macos", feature = "metal")
))]
fn rope_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let b = 1usize;
    let s = 2usize;
    let hd = 8usize;
    let tab = hd / 2;
    let dy: Vec<f32> = (0..b * s * hd).map(|i| 0.1 * i as f32).collect();
    let cos: Vec<f32> = (0..s * tab).map(|i| (i as f32 * 0.3).cos()).collect();
    let sin: Vec<f32> = (0..s * tab).map(|i| (i as f32 * 0.3).sin()).collect();
    (dy, cos, sin)
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_rope_backward_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("skip wgpu_rope_backward_matches_cpu: no adapter");
        return;
    }
    let (dy, cos, sin) = rope_inputs();
    let bwd = build_rope_bwd_graph();
    let want = cpu_run(bwd.clone(), &[("dy", &dy), ("cos", &cos), ("sin", &sin)]);
    use rlx_wgpu::backend::WgpuExecutable;
    let mut exe = WgpuExecutable::compile(bwd);
    let got = exe
        .run(&[("dy", &dy), ("cos", &cos), ("sin", &sin)])
        .remove(0);
    assert_close(&want, &got, 1e-4);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rope_backward_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session};
    let (dy, cos, sin) = rope_inputs();
    let bwd = build_rope_bwd_graph();
    let want = cpu_run(bwd.clone(), &[("dy", &dy), ("cos", &cos), ("sin", &sin)]);
    let session = Session::new(Device::Cuda);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let got = compiled
        .run(&[("dy", &dy), ("cos", &cos), ("sin", &sin)])
        .remove(0);
    assert_close(&want, &got, 1e-4);
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_rope_backward_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session, is_available};
    if !is_available(Device::Rocm) {
        eprintln!("skip rocm_rope_backward_matches_cpu (unavailable)");
        return;
    }
    let (dy, cos, sin) = rope_inputs();
    let bwd = build_rope_bwd_graph();
    let want = cpu_run(bwd.clone(), &[("dy", &dy), ("cos", &cos), ("sin", &sin)]);
    let session = Session::new(Device::Rocm);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let got = compiled
        .run(&[("dy", &dy), ("cos", &cos), ("sin", &sin)])
        .remove(0);
    assert_close(&want, &got, 1e-4);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_rope_backward_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session};
    let (dy, cos, sin) = rope_inputs();
    let bwd = build_rope_bwd_graph();
    let want = cpu_run(bwd.clone(), &[("dy", &dy), ("cos", &cos), ("sin", &sin)]);
    let session = Session::new(Device::Metal);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let got = compiled
        .run(&[("dy", &dy), ("cos", &cos), ("sin", &sin)])
        .remove(0);
    assert_close(&want, &got, 1e-4);
}

// --- Cumsum backward (inclusive, last axis) ---

#[cfg(any(
    feature = "gpu",
    feature = "cuda",
    feature = "rocm",
    all(target_os = "macos", feature = "metal")
))]
fn build_cumsum_bwd_graph() -> Graph {
    let f = DType::F32;
    let rows = 3usize;
    let cols = 4usize;
    let mut g = Graph::new("cum_bwd");
    let dy = g.input("dy", Shape::new(&[rows, cols], f));
    let dx = g.cumsum_backward(dy, Shape::new(&[rows, cols], f), -1, false);
    g.set_outputs(vec![dx]);
    g
}

#[cfg(any(
    feature = "gpu",
    feature = "cuda",
    feature = "rocm",
    all(target_os = "macos", feature = "metal")
))]
fn cumsum_inputs() -> Vec<f32> {
    (0..12).map(|i| 1.0 + 0.1 * i as f32).collect()
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_cumsum_backward_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("skip wgpu_cumsum_backward_matches_cpu: no adapter");
        return;
    }
    let dy = cumsum_inputs();
    let bwd = build_cumsum_bwd_graph();
    let want = cpu_run(bwd.clone(), &[("dy", &dy)]);
    use rlx_wgpu::backend::WgpuExecutable;
    let mut exe = WgpuExecutable::compile(bwd);
    let got = exe.run(&[("dy", &dy)]).remove(0);
    assert_close(&want, &got, 1e-4);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_cumsum_backward_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session};
    let dy = cumsum_inputs();
    let bwd = build_cumsum_bwd_graph();
    let want = cpu_run(bwd.clone(), &[("dy", &dy)]);
    let session = Session::new(Device::Cuda);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let got = compiled.run(&[("dy", &dy)]).remove(0);
    assert_close(&want, &got, 1e-4);
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_cumsum_backward_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session, is_available};
    if !is_available(Device::Rocm) {
        eprintln!("skip rocm_cumsum_backward_matches_cpu (unavailable)");
        return;
    }
    let dy = cumsum_inputs();
    let bwd = build_cumsum_bwd_graph();
    let want = cpu_run(bwd.clone(), &[("dy", &dy)]);
    let session = Session::new(Device::Rocm);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let got = compiled.run(&[("dy", &dy)]).remove(0);
    assert_close(&want, &got, 1e-4);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_cumsum_backward_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session};
    let dy = cumsum_inputs();
    let bwd = build_cumsum_bwd_graph();
    let want = cpu_run(bwd.clone(), &[("dy", &dy)]);
    let session = Session::new(Device::Metal);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let got = compiled.run(&[("dy", &dy)]).remove(0);
    assert_close(&want, &got, 1e-4);
}

// --- Gather backward (axis 0) ---

#[cfg(any(
    feature = "gpu",
    feature = "cuda",
    feature = "rocm",
    all(target_os = "macos", feature = "metal")
))]
fn build_gather_bwd_graph() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("gather_bwd");
    let dy = g.input("dy", Shape::new(&[2], f));
    let indices = g.input("indices", Shape::new(&[2], f));
    let dtable = g.gather_backward(dy, indices, Shape::new(&[4], f), 0);
    g.set_outputs(vec![dtable]);
    g
}

#[cfg(any(
    feature = "gpu",
    feature = "cuda",
    feature = "rocm",
    all(target_os = "macos", feature = "metal")
))]
fn gather_inputs() -> (Vec<f32>, Vec<f32>) {
    let dy = vec![1.0, 2.0];
    let indices = vec![0.0, 2.0];
    (dy, indices)
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_gather_backward_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("skip wgpu_gather_backward_matches_cpu: no adapter");
        return;
    }
    let (dy, indices) = gather_inputs();
    let bwd = build_gather_bwd_graph();
    let want = cpu_run(bwd.clone(), &[("dy", &dy), ("indices", &indices)]);
    use rlx_wgpu::backend::WgpuExecutable;
    let mut exe = WgpuExecutable::compile(bwd);
    let got = exe.run(&[("dy", &dy), ("indices", &indices)]).remove(0);
    assert_close(&want, &got, 1e-4);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_gather_backward_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session};
    let (dy, indices) = gather_inputs();
    let bwd = build_gather_bwd_graph();
    let want = cpu_run(bwd.clone(), &[("dy", &dy), ("indices", &indices)]);
    let session = Session::new(Device::Cuda);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let got = compiled
        .run(&[("dy", &dy), ("indices", &indices)])
        .remove(0);
    assert_close(&want, &got, 1e-4);
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_gather_backward_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session, is_available};
    if !is_available(Device::Rocm) {
        eprintln!("skip rocm_gather_backward_matches_cpu (unavailable)");
        return;
    }
    let (dy, indices) = gather_inputs();
    let bwd = build_gather_bwd_graph();
    let want = cpu_run(bwd.clone(), &[("dy", &dy), ("indices", &indices)]);
    let session = Session::new(Device::Rocm);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let got = compiled
        .run(&[("dy", &dy), ("indices", &indices)])
        .remove(0);
    assert_close(&want, &got, 1e-4);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_gather_backward_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session};
    let (dy, indices) = gather_inputs();
    let bwd = build_gather_bwd_graph();
    let want = cpu_run(bwd.clone(), &[("dy", &dy), ("indices", &indices)]);
    let session = Session::new(Device::Metal);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let got = compiled
        .run(&[("dy", &dy), ("indices", &indices)])
        .remove(0);
    assert_close(&want, &got, 1e-4);
}

// --- LayerNorm / GroupNorm backward input: finite-difference ground truth ---
//
// Same lesson as the RMSNorm `1/r` bug: these mean+normalize backward kernels are
// the structural siblings of RMSNorm, so a shared cross-term factor error would
// pass every cross-backend parity test. Finite differences of the forward are the
// independent oracle. Both run on any platform (CPU thunks), so they guard CI.

/// `dx` of a backward-input op must equal `∂(Σ dyⱼ·yⱼ)/∂x` (central finite diff).
fn assert_dx_matches_fd(
    analytic: &[f32],
    x: &[f32],
    dy: &[f32],
    forward: impl Fn(&[f32]) -> Vec<f32>,
) {
    assert_dx_matches_fd_tol(analytic, x, dy, 2e-3, forward);
}

fn assert_dx_matches_fd_tol(
    analytic: &[f32],
    x: &[f32],
    dy: &[f32],
    tol: f32,
    forward: impl Fn(&[f32]) -> Vec<f32>,
) {
    let loss = |xv: &[f32]| -> f32 { forward(xv).iter().zip(dy).map(|(y, d)| y * d).sum() };
    let step = 1e-3f32;
    let fd: Vec<f32> = (0..x.len())
        .map(|i| {
            let mut p = x.to_vec();
            p[i] += step;
            let mut m = x.to_vec();
            m[i] -= step;
            (loss(&p) - loss(&m)) / (2.0 * step)
        })
        .collect();
    let max = analytic
        .iter()
        .zip(&fd)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max < tol,
        "dx vs finite-diff max_abs_diff={max}\n  analytic={analytic:?}\n  fd={fd:?}"
    );
}

#[test]
fn cpu_layer_norm_backward_input_matches_finite_difference() {
    let (rows, h) = (3usize, 4usize);
    let f = DType::F32;
    let x: Vec<f32> = (0..rows * h).map(|i| 0.1 * (i as f32 - 5.0)).collect();
    let gamma: Vec<f32> = (0..h).map(|i| 0.5 + 0.2 * i as f32).collect();
    let beta: Vec<f32> = vec![0.01; h];
    let dy: Vec<f32> = (0..rows * h).map(|i| 1.0 + 0.05 * i as f32).collect();

    let mut g = Graph::new("ln_bwd_in");
    let xn = g.input("x", Shape::new(&[rows, h], f));
    let gn = g.input("gamma", Shape::new(&[h], f));
    let dy_in = g.input("dy", Shape::new(&[rows, h], f));
    let dx = g.layer_norm_backward_input(xn, gn, dy_in, -1, 1e-5);
    g.set_outputs(vec![dx]);
    let analytic = cpu_run(g, &[("x", &x), ("gamma", &gamma), ("dy", &dy)]);

    let (gamma_c, beta_c) = (gamma.clone(), beta.clone());
    assert_dx_matches_fd(&analytic, &x, &dy, |xv| {
        let f = DType::F32;
        let mut g = Graph::new("ln_fwd");
        let xn = g.input("x", Shape::new(&[rows, h], f));
        let gn = g.input("gamma", Shape::new(&[h], f));
        let bn = g.input("beta", Shape::new(&[h], f));
        let y = g.layer_norm(xn, gn, bn, -1, 1e-5, Shape::new(&[rows, h], f));
        g.set_outputs(vec![y]);
        cpu_run(g, &[("x", xv), ("gamma", &gamma_c), ("beta", &beta_c)])
    });
}

#[test]
fn cpu_group_norm_backward_input_matches_finite_difference() {
    // N=1: the input kernel processes each batch independently, so N=1 fully pins the
    // per-group formula. (Larger N just inflates FD truncation on these magnitudes; the
    // cross-batch reduction is exercised by the gamma/beta test and the ANE N=2 parity.)
    let dims = [1usize, 4, 2, 2]; // N, C, H, W
    let ng = 2usize;
    let c = dims[1];
    let n_el: usize = dims.iter().product();
    let f = DType::F32;
    let x: Vec<f32> = (0..n_el).map(|i| 0.1 * (i as f32 - 7.0)).collect();
    let gamma: Vec<f32> = (0..c).map(|i| 0.5 + 0.3 * i as f32).collect();
    let beta: Vec<f32> = vec![0.02; c];
    let dy: Vec<f32> = (0..n_el).map(|i| 1.0 + 0.04 * i as f32).collect();

    let mut g = Graph::new("gn_bwd_in");
    let xn = g.input("x", Shape::new(&dims, f));
    let gn = g.input("gamma", Shape::new(&[c], f));
    let bn = g.input("beta", Shape::new(&[c], f));
    let dy_in = g.input("dy", Shape::new(&dims, f));
    let dx = g.group_norm_backward_input(xn, gn, bn, dy_in, ng, 1e-5);
    g.set_outputs(vec![dx]);
    let analytic = cpu_run(
        g,
        &[("x", &x), ("gamma", &gamma), ("beta", &beta), ("dy", &dy)],
    );

    let (gamma_c, beta_c) = (gamma.clone(), beta.clone());
    assert_dx_matches_fd(&analytic, &x, &dy, |xv| {
        let f = DType::F32;
        let mut g = Graph::new("gn_fwd");
        let xn = g.input("x", Shape::new(&dims, f));
        let gn = g.input("gamma", Shape::new(&[c], f));
        let bn = g.input("beta", Shape::new(&[c], f));
        let y = g.group_norm(xn, gn, bn, ng, 1e-5);
        g.set_outputs(vec![y]);
        cpu_run(g, &[("x", xv), ("gamma", &gamma_c), ("beta", &beta_c)])
    });
}

#[test]
fn cpu_softmax_cross_entropy_backward_matches_finite_difference() {
    // The fused training-loss gradient (MNIST path). `dlogits = (softmax(logits) −
    // onehot(label))·d_loss`; FD the per-row `−log softmax` forward, weighted by
    // d_loss, since `assert_dx_matches_fd` computes `Σ forward·dy` as the scalar.
    let (n, c) = (2usize, 4usize);
    let f = DType::F32;
    let logits: Vec<f32> = (0..n * c).map(|i| 0.2 * (i as f32) - 0.5).collect();
    let labels: Vec<f32> = vec![1.0, 3.0]; // f32-encoded class indices
    let d_loss: Vec<f32> = vec![0.7, 1.3];

    let mut g = Graph::new("sce_bwd");
    let lg = g.input("logits", Shape::new(&[n, c], f));
    let lb = g.input("labels", Shape::new(&[n], f));
    let dl = g.input("d_loss", Shape::new(&[n], f));
    let dlogits = g.softmax_cross_entropy_backward(lg, lb, dl);
    g.set_outputs(vec![dlogits]);
    let analytic = cpu_run(
        g,
        &[
            ("logits", &logits),
            ("labels", &labels),
            ("d_loss", &d_loss),
        ],
    );

    let labels_c = labels.clone();
    assert_dx_matches_fd(&analytic, &logits, &d_loss, |lv| {
        let f = DType::F32;
        let mut g = Graph::new("sce_fwd");
        let lg = g.input("logits", Shape::new(&[n, c], f));
        let lb = g.input("labels", Shape::new(&[n], f));
        let loss_vec = g.softmax_cross_entropy_with_logits(lg, lb);
        g.set_outputs(vec![loss_vec]);
        cpu_run(g, &[("logits", lv), ("labels", &labels_c)])
    });
}

#[test]
fn cpu_attention_backward_qkv_matches_finite_difference() {
    // Scaled-dot-product attention dQ/dK/dV (the last transformer backward op not
    // yet ground-truthed). Causal mask, default 1/√d scale shared by fwd+bwd.
    // Looser tol than the norms: softmax curvature inflates central-diff truncation.
    use rlx_ir::op::{AttentionBwdWrt, MaskKind};
    let (b, h, s, d) = (1usize, 1, 3, 4);
    let f = DType::F32;
    let shape = || Shape::new(&[b, h, s, d], f);
    let nel = b * h * s * d;
    let mk = |seed: f32| -> Vec<f32> {
        (0..nel)
            .map(|i| ((i as f32) * 0.13 + seed).sin() * 0.5)
            .collect()
    };
    let q = mk(0.0);
    let k = mk(1.0);
    let v = mk(2.0);
    let dy: Vec<f32> = (0..nel).map(|i| 0.2 + 0.05 * i as f32).collect();

    let forward = |qv: &[f32], kv: &[f32], vv: &[f32]| -> Vec<f32> {
        let mut g = Graph::new("attn_fwd");
        let qi = g.input("q", shape());
        let ki = g.input("k", shape());
        let vi = g.input("v", shape());
        let y = g.attention_kind(qi, ki, vi, h, d, MaskKind::Causal, shape());
        g.set_outputs(vec![y]);
        cpu_run(g, &[("q", qv), ("k", kv), ("v", vv)])
    };
    let bwd = |wrt: AttentionBwdWrt| -> Vec<f32> {
        let mut g = Graph::new("attn_bwd");
        let qi = g.input("q", shape());
        let ki = g.input("k", shape());
        let vi = g.input("v", shape());
        let dyi = g.input("dy", shape());
        let dout = g.attention_backward(wrt, qi, ki, vi, dyi, h, d, MaskKind::Causal, None);
        g.set_outputs(vec![dout]);
        cpu_run(g, &[("q", &q), ("k", &k), ("v", &v), ("dy", &dy)])
    };

    let (k_c, v_c) = (k.clone(), v.clone());
    assert_dx_matches_fd_tol(&bwd(AttentionBwdWrt::Query), &q, &dy, 5e-3, |qv| {
        forward(qv, &k_c, &v_c)
    });
    let (q_c, v_c) = (q.clone(), v.clone());
    assert_dx_matches_fd_tol(&bwd(AttentionBwdWrt::Key), &k, &dy, 5e-3, |kv| {
        forward(&q_c, kv, &v_c)
    });
    let (q_c, k_c) = (q.clone(), k.clone());
    assert_dx_matches_fd_tol(&bwd(AttentionBwdWrt::Value), &v, &dy, 5e-3, |vv| {
        forward(&q_c, &k_c, vv)
    });
}

// --- Remaining backward kernels whose only check was cross-backend parity ---
// (rope / cumsum / gather are linear, so FD of the forward IS the adjoint; the
// γ/β variants pin the affine-parameter gradients of LayerNorm and GroupNorm.)

#[test]
fn cpu_rope_backward_matches_finite_difference() {
    use rlx_ir::op::RopeStyle;
    let (b, s, hd, n_rot) = (1usize, 2usize, 8usize, 6usize);
    let tab = hd / 2;
    let f = DType::F32;
    let xshape = || Shape::new(&[b, s, hd], f);
    let cshape = || Shape::new(&[s, tab], f);
    let x: Vec<f32> = (0..b * s * hd).map(|i| ((i as f32) * 0.21).sin()).collect();
    let dy: Vec<f32> = (0..b * s * hd).map(|i| 0.1 + 0.07 * i as f32).collect();
    let cos: Vec<f32> = (0..s * tab).map(|i| (i as f32 * 0.3).cos()).collect();
    let sin: Vec<f32> = (0..s * tab).map(|i| (i as f32 * 0.3).sin()).collect();

    let mut g = Graph::new("rope_bwd");
    let dyi = g.input("dy", xshape());
    let cosi = g.input("cos", cshape());
    let sini = g.input("sin", cshape());
    let dx = g.rope_backward(dyi, cosi, sini, hd, n_rot);
    g.set_outputs(vec![dx]);
    let analytic = cpu_run(g, &[("dy", &dy), ("cos", &cos), ("sin", &sin)]);

    let (cos_c, sin_c) = (cos.clone(), sin.clone());
    assert_dx_matches_fd(&analytic, &x, &dy, |xv| {
        let mut g = Graph::new("rope_fwd");
        let xi = g.input("x", xshape());
        let cosi = g.input("cos", cshape());
        let sini = g.input("sin", cshape());
        let y = g.add_node(
            Op::Rope {
                head_dim: hd,
                n_rot,
                style: RopeStyle::NeoX,
            },
            vec![xi, cosi, sini],
            xshape(),
        );
        g.set_outputs(vec![y]);
        cpu_run(g, &[("x", xv), ("cos", &cos_c), ("sin", &sin_c)])
    });
}

#[test]
fn cpu_cumsum_backward_matches_finite_difference() {
    let (rows, cols) = (3usize, 4usize);
    let f = DType::F32;
    let shape = || Shape::new(&[rows, cols], f);
    let x: Vec<f32> = (0..rows * cols).map(|i| 0.1 * (i as f32) - 0.5).collect();
    let dy: Vec<f32> = (0..rows * cols).map(|i| 1.0 + 0.05 * i as f32).collect();

    let mut g = Graph::new("cum_bwd");
    let dyi = g.input("dy", shape());
    let dx = g.cumsum_backward(dyi, shape(), -1, false);
    g.set_outputs(vec![dx]);
    let analytic = cpu_run(g, &[("dy", &dy)]);

    assert_dx_matches_fd(&analytic, &x, &dy, |xv| {
        let mut g = Graph::new("cum_fwd");
        let xi = g.input("x", shape());
        let y = g.cumsum(xi, -1, false, shape());
        g.set_outputs(vec![y]);
        cpu_run(g, &[("x", xv)])
    });
}

#[test]
fn cpu_gather_backward_matches_finite_difference() {
    let f = DType::F32;
    let table = vec![0.5f32, -1.0, 2.0, 0.25];
    let indices = vec![0.0f32, 2.0];
    let dy = vec![1.3f32, 0.7];

    let mut g = Graph::new("gather_bwd");
    let dyi = g.input("dy", Shape::new(&[2], f));
    let idxi = g.input("indices", Shape::new(&[2], f));
    let dtable = g.gather_backward(dyi, idxi, Shape::new(&[4], f), 0);
    g.set_outputs(vec![dtable]);
    let analytic = cpu_run(g, &[("dy", &dy), ("indices", &indices)]);

    let idx_c = indices.clone();
    assert_dx_matches_fd(&analytic, &table, &dy, |tv| {
        let mut g = Graph::new("gather_fwd");
        let ti = g.input("table", Shape::new(&[4], f));
        let ii = g.input("indices", Shape::new(&[2], f));
        let y = g.gather(ti, ii, 0, Shape::new(&[2], f));
        g.set_outputs(vec![y]);
        cpu_run(g, &[("table", tv), ("indices", &idx_c)])
    });
}

#[test]
fn cpu_layer_norm_backward_gamma_matches_finite_difference() {
    let (rows, h) = (3usize, 4usize);
    let f = DType::F32;
    let x: Vec<f32> = (0..rows * h).map(|i| 0.1 * (i as f32 - 5.0)).collect();
    let gamma: Vec<f32> = (0..h).map(|i| 0.5 + 0.2 * i as f32).collect();
    let beta: Vec<f32> = vec![0.01; h];
    let dy: Vec<f32> = (0..rows * h).map(|i| 1.0 + 0.05 * i as f32).collect();

    let mut g = Graph::new("ln_bwd_g");
    let xn = g.input("x", Shape::new(&[rows, h], f));
    let dyi = g.input("dy", Shape::new(&[rows, h], f));
    let dgamma = g.layer_norm_backward_gamma(xn, dyi, Shape::new(&[h], f), -1, 1e-5);
    g.set_outputs(vec![dgamma]);
    let analytic = cpu_run(g, &[("x", &x), ("dy", &dy)]);

    let (x_c, beta_c) = (x.clone(), beta.clone());
    assert_dx_matches_fd(&analytic, &gamma, &dy, |gv| {
        let f = DType::F32;
        let mut g = Graph::new("ln_fwd");
        let xn = g.input("x", Shape::new(&[rows, h], f));
        let gn = g.input("gamma", Shape::new(&[h], f));
        let bn = g.input("beta", Shape::new(&[h], f));
        let y = g.layer_norm(xn, gn, bn, -1, 1e-5, Shape::new(&[rows, h], f));
        g.set_outputs(vec![y]);
        cpu_run(g, &[("x", &x_c), ("gamma", gv), ("beta", &beta_c)])
    });
}

#[test]
fn cpu_group_norm_backward_gamma_beta_matches_finite_difference() {
    let dims = [2usize, 4, 2, 2]; // N>1 exercises the batch reduction
    let ng = 2usize;
    let c = dims[1];
    let n_el: usize = dims.iter().product();
    let f = DType::F32;
    let x: Vec<f32> = (0..n_el).map(|i| 0.1 * (i as f32 - 7.0)).collect();
    let gamma: Vec<f32> = (0..c).map(|i| 0.5 + 0.3 * i as f32).collect();
    let beta: Vec<f32> = (0..c).map(|i| 0.02 + 0.01 * i as f32).collect();
    let dy: Vec<f32> = (0..n_el).map(|i| 1.0 + 0.04 * i as f32).collect();

    let fwd = |gv: &[f32], bv: &[f32], xv: &[f32]| -> Vec<f32> {
        let f = DType::F32;
        let mut g = Graph::new("gn_fwd");
        let xn = g.input("x", Shape::new(&dims, f));
        let gn = g.input("gamma", Shape::new(&[c], f));
        let bn = g.input("beta", Shape::new(&[c], f));
        let y = g.group_norm(xn, gn, bn, ng, 1e-5);
        g.set_outputs(vec![y]);
        cpu_run(g, &[("x", xv), ("gamma", gv), ("beta", bv)])
    };

    // dgamma
    let mut g = Graph::new("gn_bwd_g");
    let xn = g.input("x", Shape::new(&dims, f));
    let dyi = g.input("dy", Shape::new(&dims, f));
    let dgamma = g.group_norm_backward_gamma(xn, dyi, Shape::new(&[c], f), ng, 1e-5);
    g.set_outputs(vec![dgamma]);
    let analytic_g = cpu_run(g, &[("x", &x), ("dy", &dy)]);
    let (x_c, beta_c) = (x.clone(), beta.clone());
    assert_dx_matches_fd(&analytic_g, &gamma, &dy, |gv| fwd(gv, &beta_c, &x_c));

    // dbeta
    let mut g = Graph::new("gn_bwd_b");
    let xn = g.input("x", Shape::new(&dims, f));
    let dyi = g.input("dy", Shape::new(&dims, f));
    let dbeta = g.group_norm_backward_beta(xn, dyi, Shape::new(&[c], f), ng, 1e-5);
    g.set_outputs(vec![dbeta]);
    let analytic_b = cpu_run(g, &[("x", &x), ("dy", &dy)]);
    let (x_c, gamma_c) = (x.clone(), gamma.clone());
    assert_dx_matches_fd(&analytic_b, &beta, &dy, |bv| fwd(&gamma_c, bv, &x_c));
}

#[test]
fn cpu_activation_backward_matches_finite_difference() {
    use rlx_ir::op::Activation;
    let n = 6usize;
    let f = DType::F32;
    let shape = || Shape::new(&[n], f);
    // All inputs are bounded away from non-smooth points (no relu kink at 0,
    // no log/sqrt singularities) so central differences are valid.
    let x: Vec<f32> = vec![0.7, -1.3, 0.4, 1.8, -0.6, 1.1];
    let dy: Vec<f32> = (0..n).map(|i| 0.3 + 0.1 * i as f32).collect();

    for kind in [
        Activation::Gelu,
        Activation::GeluApprox,
        Activation::Silu,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Exp,
        Activation::Sin,
        Activation::Cos,
        Activation::Tan,
        Activation::Atan,
    ] {
        let mut g = Graph::new("act_bwd");
        let xn = g.input("x", shape());
        let dyi = g.input("dy", shape());
        let dx = g.activation_backward(kind, xn, dyi);
        g.set_outputs(vec![dx]);
        let analytic = cpu_run(g, &[("x", &x), ("dy", &dy)]);
        eprintln!("activation backward FD check: {kind:?}");
        // Exp/Tan carry more curvature than the norms; 3e-3 covers their FD truncation.
        assert_dx_matches_fd_tol(&analytic, &x, &dy, 3e-3, |xv| {
            let mut g = Graph::new("act_fwd");
            let xn = g.input("x", shape());
            let y = g.activation(kind, xn, shape());
            g.set_outputs(vec![y]);
            cpu_run(g, &[("x", xv)])
        });
    }
}

#[test]
fn cpu_relu_backward_matches_finite_difference() {
    use rlx_ir::op::Activation;
    let n = 6usize;
    let f = DType::F32;
    let shape = || Shape::new(&[n], f);
    let x: Vec<f32> = vec![0.7, -1.3, 0.4, 1.8, -0.6, 1.1]; // none near the kink at 0
    let dy: Vec<f32> = (0..n).map(|i| 0.3 + 0.1 * i as f32).collect();

    let mut g = Graph::new("relu_bwd");
    let xn = g.input("x", shape());
    let dyi = g.input("dy", shape());
    let dx = g.relu_backward(xn, dyi);
    g.set_outputs(vec![dx]);
    let analytic = cpu_run(g, &[("x", &x), ("dy", &dy)]);
    assert_dx_matches_fd(&analytic, &x, &dy, |xv| {
        let mut g = Graph::new("relu_fwd");
        let xn = g.input("x", shape());
        let y = g.activation(Activation::Relu, xn, shape());
        g.set_outputs(vec![y]);
        cpu_run(g, &[("x", xv)])
    });
}

#[test]
fn cpu_maxpool2d_backward_matches_finite_difference() {
    let f = DType::F32;
    let xs = Shape::new(&[1, 1, 4, 4], f);
    let ys = Shape::new(&[1, 1, 2, 2], f);
    // Strict maximum in each 2×2 window (gaps ≫ FD step), so each window's argmax
    // is locally constant — max-pool is locally linear and central diff is valid.
    let x: Vec<f32> = vec![
        0.1, 0.2, 0.9, 0.3, //
        0.4, 0.8, 0.5, 0.6, //
        0.7, 0.15, 0.25, 0.95, //
        0.35, 0.45, 0.85, 0.55,
    ];
    let dy: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let ksp = || (vec![2usize, 2], vec![2usize, 2], vec![0usize, 0]);

    let (k, s, p) = ksp();
    let mut g = Graph::new("mp_bwd");
    let xn = g.input("x", xs.clone());
    let dyi = g.input("dy", ys.clone());
    let dx = g.maxpool2d_backward(xn, dyi, k, s, p);
    g.set_outputs(vec![dx]);
    let analytic = cpu_run(g, &[("x", &x), ("dy", &dy)]);

    assert_dx_matches_fd(&analytic, &x, &dy, |xv| {
        let (k, s, p) = ksp();
        let mut g = Graph::new("mp_fwd");
        let xn = g.input("x", xs.clone());
        let y = g.add_node(
            Op::Pool {
                kind: rlx_ir::op::ReduceOp::Max,
                kernel_size: k,
                stride: s,
                padding: p,
            },
            vec![xn],
            ys.clone(),
        );
        g.set_outputs(vec![y]);
        cpu_run(g, &[("x", xv)])
    });
}
