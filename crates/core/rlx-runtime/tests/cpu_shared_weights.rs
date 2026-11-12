// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU executables compiled under one param-sharing scope back their large
//! read-only weights with one set of shared pages (`rlx_cpu::shared_params`).
//! Two shapes of one model — an embedding gather, a matmul through a reshaped
//! (view) weight, a transposed weight, and a small bias that stays private —
//! must match the same graphs compiled without a scope, bit for bit, and the
//! second executable must find the weights already bound.

use rlx_ir::op::BinaryOp;
use rlx_ir::*;
use rlx_runtime::{Device, Session};

/// The sharing scope is process-wide, and tests run on parallel threads: one
/// test's compile could otherwise land in the other's scope (or in none).
static SCOPE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

const VOCAB: usize = 512;
const HIDDEN: usize = 256;

fn model(seq: usize) -> Graph {
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    let mut g = Graph::new("cpu_shared_weights");
    let ids = g.input("ids", f(&[seq]));
    let embed = g.param("embed", f(&[VOCAB, HIDDEN]));
    let w_flat = g.param("w_flat", f(&[HIDDEN * HIDDEN]));
    let w2 = g.param("w2", f(&[HIDDEN, HIDDEN]));
    let bias = g.param("bias", f(&[HIDDEN]));
    let x = g.add_node(Op::Gather { axis: 0 }, vec![embed, ids], f(&[seq, HIDDEN]));
    let w = g.add_node(
        Op::Reshape {
            new_shape: vec![HIDDEN as i64, HIDDEN as i64],
        },
        vec![w_flat],
        f(&[HIDDEN, HIDDEN]),
    );
    let h = g.add_node(Op::MatMul, vec![x, w], f(&[seq, HIDDEN]));
    let w2t = g.add_node(
        Op::Transpose { perm: vec![1, 0] },
        vec![w2],
        f(&[HIDDEN, HIDDEN]),
    );
    let y = g.add_node(Op::MatMul, vec![h, w2t], f(&[seq, HIDDEN]));
    let out = g.add_node(Op::Binary(BinaryOp::Add), vec![y, bias], f(&[seq, HIDDEN]));
    g.set_outputs(vec![out]);
    g
}

fn values(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + salt * 17) % 97) as f32 - 48.0) / 97.0)
        .collect()
}

#[test]
fn shared_cpu_weights_match_private_ones() {
    let _scope = SCOPE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let params = [
        ("embed", values(VOCAB * HIDDEN, 1)),
        ("w_flat", values(HIDDEN * HIDDEN, 2)),
        ("w2", values(HIDDEN * HIDDEN, 3)),
        ("bias", values(HIDDEN, 4)),
    ];
    let run = |c: &mut rlx_runtime::CompiledGraph, seq: usize| {
        let ids: Vec<f32> = (0..seq).map(|i| ((i * 97 + 5) % VOCAB) as f32).collect();
        c.run(&[("ids", &ids)]).pop().unwrap()
    };
    let mut shared = Vec::new();
    for (i, seq) in [3usize, 5].into_iter().enumerate() {
        let mut c = rlx_runtime::with_param_sharing_scope("cpu-shared-weights-test", || {
            Session::new(Device::Cpu).compile(model(seq))
        });
        for (name, data) in &params {
            let present = c.param_already_set(name);
            // The large weights are shared from the second executable on; the
            // bias is under the size floor and always private.
            assert_eq!(
                present,
                i > 0 && *name != "bias",
                "seq {seq}: param_already_set({name})"
            );
            if !present {
                c.set_param(name, data);
            }
        }
        shared.push((seq, c));
    }
    for (seq, c) in &mut shared {
        let mut private = Session::new(Device::Cpu).compile(model(*seq));
        for (name, data) in &params {
            private.set_param(name, data);
        }
        let want = run(&mut private, *seq);
        let got = run(c, *seq);
        assert_eq!(
            got, want,
            "seq {seq}: shared weights differ from private ones"
        );
    }
    // Run the first again after the second: still its own activations.
    let (seq, c) = &mut shared[0];
    let mut private = Session::new(Device::Cpu).compile(model(*seq));
    for (name, data) in &params {
        private.set_param(name, data);
    }
    assert_eq!(run(c, *seq), run(&mut private, *seq));
}

/// Two linears with bias (the second through ReLU), which fusion rewrites to
/// `FusedMatMulBiasAct` before the executable is built. Their weights used to
/// stay private under a scope: the fused op was not on the weight-only reader
/// list, so a model whose linears all fuse (every transformer) shared nothing.
fn fused_linears(rows: usize) -> Graph {
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    let mut g = Graph::new("cpu_shared_fused_linears");
    let x = g.input("x", f(&[rows, HIDDEN]));
    let w1 = g.param("w1", f(&[HIDDEN, 4 * HIDDEN]));
    let b1 = g.param("b1", f(&[4 * HIDDEN]));
    let w2 = g.param("w2", f(&[4 * HIDDEN, HIDDEN]));
    let b2 = g.param("b2", f(&[HIDDEN]));
    let h = g.mm(x, w1);
    let h = g.add(h, b1);
    let h = g.relu(h);
    let y = g.mm(h, w2);
    let y = g.add(y, b2);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn fused_linear_weights_are_shared() {
    let _scope = SCOPE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let params = [
        ("w1", values(HIDDEN * 4 * HIDDEN, 5)),
        ("b1", values(4 * HIDDEN, 6)),
        ("w2", values(4 * HIDDEN * HIDDEN, 7)),
        ("b2", values(HIDDEN, 8)),
    ];
    let run = |c: &mut rlx_runtime::CompiledGraph, rows: usize| {
        c.run(&[("x", &values(rows * HIDDEN, 9))]).pop().unwrap()
    };
    let mut shared = Vec::new();
    for (i, rows) in [4usize, 16].into_iter().enumerate() {
        let mut c = rlx_runtime::with_param_sharing_scope("cpu-shared-fused-test", || {
            Session::new(Device::Cpu).compile(fused_linears(rows))
        });
        for (name, data) in &params {
            let present = c.param_already_set(name);
            assert_eq!(
                present,
                i > 0 && name.starts_with('w'),
                "rows {rows}: param_already_set({name})"
            );
            if !present {
                c.set_param(name, data);
            }
        }
        shared.push((rows, c));
    }
    for (rows, c) in &mut shared {
        let mut private = Session::new(Device::Cpu).compile(fused_linears(*rows));
        for (name, data) in &params {
            private.set_param(name, data);
        }
        assert_eq!(run(c, *rows), run(&mut private, *rows), "rows {rows}");
    }
}
