// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// GPU parity: Op::AttentionBackward (3 kernels) vs CPU thunk ([B,H,S,D], causal).

use rlx_compile::legalize_broadcast::run_with_remap;
use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

const B: usize = 1;
const H: usize = 2;
const S: usize = 4;
const D: usize = 3;

/// Minimal MIR: three `AttentionBackward` nodes only (no autodiff Reduce ops).
fn build_bwd_kernel_graph() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("attn_bwd_kernel");
    let q = g.input("q", Shape::new(&[B, H, S, D], f));
    let k = g.input("k", Shape::new(&[B, H, S, D], f));
    let v = g.input("v", Shape::new(&[B, H, S, D], f));
    let dy = g.input("dy", Shape::new(&[B, H, S, D], f));
    let (dq, dk, dv) = g.attention_backward_all(q, k, v, dy, H, D, MaskKind::Causal, None);
    g.set_outputs(vec![dq, dk, dv]);
    let n_bwd = g
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::AttentionBackward { .. }))
        .count();
    assert_eq!(n_bwd, 3);
    g
}

fn synthetic_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = B * H * S * D;
    let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.07 - 0.5).collect();
    let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 0.2).collect();
    let dy: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 - 0.1).collect();
    (q, k, v, dy)
}

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

fn cpu_bwd_grads(
    bwd: Graph,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dy: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (bwd, remap) = run_with_remap(bwd);
    let r = |id: NodeId| *remap.get(&id).unwrap_or(&id);
    let plan = rlx_opt::memory::plan_memory(&bwd);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&bwd, &arena);
    let ids: Vec<_> = bwd
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name } => Some((name.as_str(), n.id)),
            _ => None,
        })
        .collect();
    let slot = |name: &str| r(ids.iter().find(|(n, _)| *n == name).unwrap().1);
    write_slot(&mut arena, slot("q"), q);
    write_slot(&mut arena, slot("k"), k);
    write_slot(&mut arena, slot("v"), v);
    write_slot(&mut arena, slot("dy"), dy);
    execute_thunks(&sched, arena.raw_buf_mut());
    let n = B * H * S * D;
    (
        read_slot(&arena, r(bwd.outputs[0]), n),
        read_slot(&arena, r(bwd.outputs[1]), n),
        read_slot(&arena, r(bwd.outputs[2]), n),
    )
}

#[cfg(any(
    feature = "gpu",
    feature = "cuda",
    all(target_os = "macos", feature = "metal")
))]
fn assert_grads_close(name: &str, cpu: &[f32], gpu: &[f32], rtol: f32) {
    assert_eq!(cpu.len(), gpu.len(), "{name}: length");
    let max = cpu
        .iter()
        .zip(gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max < rtol,
        "{name}: max_abs_diff={max} (rtol={rtol}) cpu[0]={} gpu[0]={}",
        cpu[0],
        gpu[0]
    );
}

#[test]
fn cpu_reference_bwd_grads_finite() {
    let (q, k, v, dy) = synthetic_inputs();
    let (dq, dk, dv) = cpu_bwd_grads(build_bwd_kernel_graph(), &q, &k, &v, &dy);
    assert!(dq.iter().any(|x| x.is_finite() && x.abs() > 1e-8));
    assert!(dk.iter().any(|x| x.is_finite() && x.abs() > 1e-8));
    assert!(dv.iter().any(|x| x.is_finite() && x.abs() > 1e-8));
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_attention_backward_matches_cpu() {
    if !rlx_wgpu::is_available() {
        eprintln!("skip wgpu_attention_backward_matches_cpu: no adapter");
        return;
    }
    let (q, k, v, dy) = synthetic_inputs();
    let bwd = build_bwd_kernel_graph();
    let (dq_cpu, dk_cpu, dv_cpu) = cpu_bwd_grads(bwd.clone(), &q, &k, &v, &dy);

    use rlx_wgpu::backend::WgpuExecutable;
    let mut exe = WgpuExecutable::compile(bwd);
    let outs = exe.run(&[("q", &q), ("k", &k), ("v", &v), ("dy", &dy)]);
    assert_eq!(outs.len(), 3);
    assert_grads_close("dq", &dq_cpu, &outs[0], 1e-3);
    assert_grads_close("dk", &dk_cpu, &outs[1], 1e-3);
    assert_grads_close("dv", &dv_cpu, &outs[2], 1e-3);
}

#[cfg(all(target_os = "macos", feature = "metal"))]
#[test]
fn metal_attention_backward_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session};
    let (q, k, v, dy) = synthetic_inputs();
    let bwd = build_bwd_kernel_graph();
    let (dq_cpu, dk_cpu, dv_cpu) = cpu_bwd_grads(bwd.clone(), &q, &k, &v, &dy);

    let session = Session::new(Device::Metal);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let outs = compiled.run(&[("q", &q), ("k", &k), ("v", &v), ("dy", &dy)]);
    assert_eq!(outs.len(), 3);
    assert_grads_close("dq", &dq_cpu, &outs[0], 1e-3);
    assert_grads_close("dk", &dk_cpu, &outs[1], 1e-3);
    assert_grads_close("dv", &dv_cpu, &outs[2], 1e-3);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_attention_backward_matches_cpu() {
    use rlx_runtime::{CompileOptions, Device, Session};
    let (q, k, v, dy) = synthetic_inputs();
    let bwd = build_bwd_kernel_graph();
    let (dq_cpu, dk_cpu, dv_cpu) = cpu_bwd_grads(bwd.clone(), &q, &k, &v, &dy);

    let session = Session::new(Device::Cuda);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let outs = compiled.run(&[("q", &q), ("k", &k), ("v", &v), ("dy", &dy)]);
    assert_eq!(outs.len(), 3);
    assert_grads_close("dq", &dq_cpu, &outs[0], 1e-3);
    assert_grads_close("dk", &dk_cpu, &outs[1], 1e-3);
    assert_grads_close("dv", &dv_cpu, &outs[2], 1e-3);
}
