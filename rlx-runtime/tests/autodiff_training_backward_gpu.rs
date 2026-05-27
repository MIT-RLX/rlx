// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
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
