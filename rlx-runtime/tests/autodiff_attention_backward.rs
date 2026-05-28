// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Parity: Op::AttentionBackward vs finite-difference grads on CPU.

use rlx_autodiff::grad_with_loss;
use rlx_compile::legalize_broadcast::run_with_remap;
use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::memory::plan_memory;

const B: usize = 1;
const H: usize = 2;
const S: usize = 4;
const D: usize = 3;

struct AttnFwd {
    graph: Graph,
    q: NodeId,
    k: NodeId,
    v: NodeId,
}

fn build_attn_forward(mask_kind: MaskKind) -> AttnFwd {
    let f = DType::F32;
    let mut g = Graph::new("attn_fwd");
    let q = g.input("q", Shape::new(&[B, H, S, D], f));
    let k = g.input("k", Shape::new(&[B, H, S, D], f));
    let v = g.input("v", Shape::new(&[B, H, S, D], f));
    let mut inputs = vec![q, k, v];
    if matches!(mask_kind, MaskKind::Custom | MaskKind::Bias) {
        let mask = g.input("mask", Shape::new(&[B, H, S, S], f));
        inputs.push(mask);
    }
    let out = g.add_node(
        Op::Attention {
            num_heads: H,
            head_dim: D,
            mask_kind,
            score_scale: None,
            attn_logit_softcap: None,
        },
        inputs,
        Shape::new(&[B, H, S, D], f),
    );
    let loss = g.add_node(
        Op::Reduce {
            op: rlx_ir::op::ReduceOp::Sum,
            axes: vec![0, 1, 2, 3],
            keep_dim: false,
        },
        vec![out],
        Shape::from_dims(&[], f),
    );
    g.set_outputs(vec![loss]);
    AttnFwd { graph: g, q, k, v }
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

fn scalar_loss(fwd: &AttnFwd, q: &[f32], k: &[f32], v: &[f32]) -> f32 {
    let g = fwd.graph.clone();
    let (g, remap) = run_with_remap(g);
    let r = |id: NodeId| *remap.get(&id).unwrap_or(&id);
    let plan = plan_memory(&g);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    write_slot(&mut arena, r(fwd.q), q);
    write_slot(&mut arena, r(fwd.k), k);
    write_slot(&mut arena, r(fwd.v), v);
    execute_thunks(&sched, arena.raw_buf_mut());
    read_slot(&arena, r(g.outputs[0]), 1)[0]
}

fn run_grads(
    fwd: &AttnFwd,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> (f32, Vec<f32>, Vec<f32>, Vec<f32>) {
    let wrt = [fwd.q, fwd.k, fwd.v];
    let bwd = grad_with_loss(&fwd.graph, &wrt);
    let (bwd, remap) = run_with_remap(bwd);
    let r = |id: NodeId| *remap.get(&id).unwrap_or(&id);
    let plan = plan_memory(&bwd);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&bwd, &arena);
    write_slot(&mut arena, r(fwd.q), q);
    write_slot(&mut arena, r(fwd.k), k);
    write_slot(&mut arena, r(fwd.v), v);
    let d_out = bwd
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .expect("d_output");
    write_slot(&mut arena, r(d_out), &[1.0]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let loss = read_slot(&arena, r(bwd.outputs[0]), 1)[0];
    let n = B * H * S * D;
    let dq = read_slot(&arena, r(bwd.outputs[1]), n);
    let dk = read_slot(&arena, r(bwd.outputs[2]), n);
    let dv = read_slot(&arena, r(bwd.outputs[3]), n);
    (loss, dq, dk, dv)
}

fn central_diff(slot: &mut [f32], idx: usize, eps: f32, loss_fn: &dyn Fn(&[f32]) -> f32) -> f32 {
    let old = slot[idx];
    slot[idx] = old + eps;
    let lp = loss_fn(slot);
    slot[idx] = old - eps;
    let lm = loss_fn(slot);
    slot[idx] = old;
    (lp - lm) / (2.0 * eps)
}

#[test]
fn attention_backward_matches_finite_diff_nomask() {
    let fwd = build_attn_forward(MaskKind::None);
    let n = B * H * S * D;
    let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.07 - 0.5).collect();
    let k: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 0.2).collect();

    let (_, dq, dk, dv) = run_grads(&fwd, &q, &k, &v);
    let eps = 1e-3f32;

    let mut qm = q.clone();
    for &i in &[0usize, 5, n - 1] {
        let num = central_diff(&mut qm, i, eps, &|b| {
            let mut qq = q.clone();
            qq[i] = b[i];
            scalar_loss(&fwd, &qq, &k, &v)
        });
        assert!(
            (dq[i] - num).abs() < 5e-2,
            "dQ[{i}]: anal={} num={}",
            dq[i],
            num
        );
    }

    let mut km = k.clone();
    for &i in &[1usize, 7] {
        let num = central_diff(&mut km, i, eps, &|b| {
            let mut kk = k.clone();
            kk[i] = b[i];
            scalar_loss(&fwd, &q, &kk, &v)
        });
        assert!(
            (dk[i] - num).abs() < 5e-2,
            "dK[{i}]: anal={} num={}",
            dk[i],
            num
        );
    }

    let mut vm = v.clone();
    for &i in &[2usize, 10] {
        let num = central_diff(&mut vm, i, eps, &|b| {
            let mut vv = v.clone();
            vv[i] = b[i];
            scalar_loss(&fwd, &q, &k, &vv)
        });
        assert!(
            (dv[i] - num).abs() < 5e-2,
            "dV[{i}]: anal={} num={}",
            dv[i],
            num
        );
    }
}

#[test]
fn attention_backward_emits_three_kernels() {
    let fwd = build_attn_forward(MaskKind::Causal);
    let bwd = grad_with_loss(&fwd.graph, &[fwd.q, fwd.k, fwd.v]);
    let count = bwd
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::AttentionBackward { .. }))
        .count();
    assert_eq!(count, 3, "expected dQ, dK, dV backward nodes");
}
