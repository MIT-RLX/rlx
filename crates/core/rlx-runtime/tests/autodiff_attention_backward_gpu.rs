// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// GPU parity: Op::AttentionBackward (3 kernels) vs CPU thunk ([B,H,S,D], causal).

use rlx_compile::legalize_broadcast::run_with_remap;
use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

mod common;

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
    feature = "rocm",
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
    let _gpu = common::serialize_gpu();
    let (q, k, v, dy) = synthetic_inputs();
    let (dq, dk, dv) = cpu_bwd_grads(build_bwd_kernel_graph(), &q, &k, &v, &dy);
    assert!(dq.iter().any(|x| x.is_finite() && x.abs() > 1e-8));
    assert!(dk.iter().any(|x| x.is_finite() && x.abs() > 1e-8));
    assert!(dv.iter().any(|x| x.is_finite() && x.abs() > 1e-8));
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_attention_backward_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(rlx_runtime::Device::Gpu, "wgpu") {
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
    let _gpu = common::serialize_gpu();
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
    let _gpu = common::serialize_gpu();
    #[allow(unused_imports)]
    use rlx_runtime::{CompileOptions, Device, Session, is_available};
    if common::skip_unless_available(Device::Cuda, "cuda") {
        eprintln!("skip cuda_attention_backward_matches_cpu (unavailable)");
        return;
    }
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

#[cfg(feature = "rocm")]
#[test]
fn rocm_attention_backward_matches_cpu() {
    let _gpu = common::serialize_gpu();
    #[allow(unused_imports)]
    use rlx_runtime::{CompileOptions, Device, Session, is_available};
    if common::skip_unless_available(Device::Rocm, "rocm") {
        eprintln!("skip rocm_attention_backward_matches_cpu (unavailable)");
        return;
    }
    let (q, k, v, dy) = synthetic_inputs();
    let bwd = build_bwd_kernel_graph();
    let (dq_cpu, dk_cpu, dv_cpu) = cpu_bwd_grads(bwd.clone(), &q, &k, &v, &dy);

    let session = Session::new(Device::Rocm);
    let mut compiled = session.compile_with(bwd, &CompileOptions::default());
    let outs = compiled.run(&[("q", &q), ("k", &k), ("v", &v), ("dy", &dy)]);
    assert_eq!(outs.len(), 3);
    assert_grads_close("dq", &dq_cpu, &outs[0], 1e-3);
    assert_grads_close("dk", &dk_cpu, &outs[1], 1e-3);
    assert_grads_close("dv", &dv_cpu, &outs[2], 1e-3);
}

/// Through autodiff, attention's dY is computed earlier in the same graph,
/// not fed. Metal's `[B, H, S, D]` path runs the CPU reference, which read
/// the arena before the GPU had written dY, so dQ, dK and dV all came back
/// zero. Every device's gradients must match the CPU's.
#[test]
fn attention_gradients_through_a_computed_dy_match_the_cpu() {
    use rlx_ir::GraphExt;
    use rlx_ir::op::ReduceOp;
    use rlx_runtime::{Device, Session};
    let (h, s, dh) = (2usize, 16usize, 32usize);
    let n = h * s * dh;
    let rnd = |seed: u64| -> Vec<f32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (x >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect()
    };
    let shape = Shape::new(&[1, h, s, dh], DType::F32);
    for mask_kind in [MaskKind::None, MaskKind::Causal] {
        let mut g = Graph::new("attn_through_dy");
        let q = g.param("q", shape.clone());
        let k = g.param("k", shape.clone());
        let v = g.param("v", shape.clone());
        let a = g.add_node(
            Op::Attention {
                num_heads: h,
                head_dim: dh,
                v_head_dim: None,
                mask_kind,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            shape.clone(),
        );
        // dY = c · d_output: computed on the device before the backward.
        let c = g.input("c", shape.clone());
        let p = g.mul(a, c);
        let loss = g.reduce(
            p,
            ReduceOp::Sum,
            vec![0, 1, 2, 3],
            false,
            Shape::scalar(DType::F32),
        );
        g.set_outputs(vec![loss]);
        let bwd = rlx_autodiff::grad_with_loss(&g, &[q, k, v]);
        let run = |d: Device| {
            let mut cg = Session::new(d).compile(bwd.clone());
            cg.set_param("q", &rnd(1));
            cg.set_param("k", &rnd(2));
            cg.set_param("v", &rnd(3));
            cg.run(&[("c", &rnd(9)), ("d_output", &[1.0])])
        };
        let want = run(Device::Cpu);
        for d in rlx_runtime::available_devices() {
            if d == Device::Cpu || !rlx_runtime::devices_for(&bwd).contains(&d) {
                continue;
            }
            let got = run(d);
            for (i, name) in ["dq", "dk", "dv"].iter().enumerate() {
                let (gw, ww) = (&got[i + 1], &want[i + 1]);
                let scale = ww.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
                let err = gw
                    .iter()
                    .zip(ww)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(
                    err <= 1e-4 * scale.max(1.0),
                    "{d:?} {mask_kind:?} {name}: max |Δ| {err} (CPU scale {scale})"
                );
            }
        }
    }
}
