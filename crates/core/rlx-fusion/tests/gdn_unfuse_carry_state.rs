// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `carry_state` `GatedDeltaNet` updates its state input in place, and a
//! graph reads the final state back through that input after the scan. The
//! unrolled expansion must give those reads the final state too — it used to
//! leave them on the initial one, so a backend without a GDN kernel exported
//! a stale recurrent state while the per-token output stayed right.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Op, Shape};

const B: usize = 1;
const S: usize = 3;
const H: usize = 2;
const N: usize = 4;

fn sh(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 1000) as f32 / 2000.0 - 0.25)
        .collect()
}

fn scalar(g: &mut Graph, v: f32) -> rlx_ir::NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        sh(&[1]),
    )
}

/// `[y, state · (1 + 0·y₀)]`: the second output depends on the scan, so it is
/// ordered after it and reads the state the scan left behind.
fn graph() -> Graph {
    let mut g = Graph::new("gdn_carry");
    let q = g.input("q", sh(&[B, S, H, N]));
    let k = g.input("k", sh(&[B, S, H, N]));
    let v = g.input("v", sh(&[B, S, H, N]));
    let gate = g.input("g", sh(&[B, S, H]));
    let beta = g.input("beta", sh(&[B, S, H]));
    let state = g.input("state", sh(&[B, H, N, N]));
    let y = g.add_node(
        Op::GatedDeltaNet {
            state_size: N,
            carry_state: true,
            gate_per_channel: false,
        },
        vec![q, k, v, gate, beta, state],
        sh(&[B, S, H, N]),
    );
    let flat = g.reshape(y, vec![(B * S * H * N) as i64], sh(&[B * S * H * N]));
    let y0 = g.add_node(
        Op::Narrow {
            axis: 0,
            start: 0,
            len: 1,
        },
        vec![flat],
        sh(&[1]),
    );
    let zero = scalar(&mut g, 0.0);
    let one = scalar(&mut g, 1.0);
    let z = g.binary(BinaryOp::Mul, y0, zero, sh(&[1]));
    let dep = g.binary(BinaryOp::Add, z, one, sh(&[1]));
    let st = g.binary(BinaryOp::Mul, state, dep, sh(&[B, H, N, N]));
    g.set_outputs(vec![y, st]);
    g
}

fn run(g: Graph) -> Vec<Vec<f32>> {
    let feeds = [
        ("q", data(B * S * H * N, 1)),
        ("k", data(B * S * H * N, 2)),
        ("v", data(B * S * H * N, 3)),
        ("g", data(B * S * H, 4)),
        ("beta", data(B * S * H, 5)),
        ("state", data(B * H * N * N, 6)),
    ];
    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
    let sched = rlx_cpu::thunk::compile_thunks(&g, &arena);
    for node in g.nodes() {
        match &node.op {
            Op::Constant { data } if arena.has_buffer(node.id) && !data.is_empty() => {
                let buf = arena.slice_mut(node.id);
                for (dst, c) in buf.iter_mut().zip(data.as_chunks::<4>().0.iter()) {
                    *dst = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
            }
            Op::Input { name } => {
                let (_, v) = feeds.iter().find(|(n, _)| n == name).expect("feed");
                arena.slice_mut(node.id)[..v.len()].copy_from_slice(v);
            }
            _ => {}
        }
    }
    rlx_cpu::thunk::execute_thunks(&sched, arena.raw_buf_mut());
    g.outputs
        .iter()
        .map(|&o| {
            let n = g.node(o).shape.num_elements().expect("static output");
            arena.slice_mut(o)[..n].to_vec()
        })
        .collect()
}

#[test]
fn unfused_scan_exports_the_final_state() {
    let fused = run(graph());
    let unfused_graph = rlx_fusion::unfuse_gated_delta_net_always(graph());
    assert!(
        !unfused_graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GatedDeltaNet { .. })),
        "expansion left a GatedDeltaNet node"
    );
    let unfused = run(unfused_graph);

    let initial = data(B * H * N * N, 6);
    let moved = fused[1]
        .iter()
        .zip(&initial)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        moved > 1e-3,
        "the fused kernel did not update the state ({moved}); the test cannot tell"
    );
    for (o, (a, b)) in fused.iter().zip(&unfused).enumerate() {
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(d < 1e-5, "output {o}: fused vs unrolled max |d| = {d}");
    }
}
