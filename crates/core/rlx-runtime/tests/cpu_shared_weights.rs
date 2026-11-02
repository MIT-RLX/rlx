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
