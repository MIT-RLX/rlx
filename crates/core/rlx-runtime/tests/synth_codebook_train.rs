// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A REAL `Op::SynthMatMul` codebook layer trained end-to-end with Muon. The
//! codebook `[num_entries, entry_dim]` is a 2-D Param → Muon orthogonalizes it;
//! the u8 `indices` (quantization structure) are fixed. The CPU test trains the
//! layer where the SynthMatMul backward (`dx` + scatter-add `d_codebook`) is
//! exact; the Metal test trains the SAME layer GPU-resident (the backward's u8
//! `Cast(u8→i64)→Gather` / `Cast(u8→f32)→ScatterAdd` now read the packed-u8
//! indices correctly on Metal), so SynthMatMul is fully trainable on Metal too.

#![cfg(feature = "training")]

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Op, Shape, SynthKind};
use rlx_optim::{Muon, Optimizer};
use rlx_runtime::{Device, Session};

const B: usize = 8;
const K: usize = 8;
const N: usize = 6;
const D: usize = 2;
const NE: usize = 4;

fn kb() -> usize {
    K / D
}

// out[b,j] = Σ_kb Σ_t x[b, kb·D+t] · codebook[indices[j,kb], t]
fn synth_ref(x: &[f32], idx: &[u8], cb: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; B * N];
    for b in 0..B {
        for j in 0..N {
            let mut acc = 0f32;
            for k in 0..kb() {
                let code = idx[j * kb() + k] as usize;
                for t in 0..D {
                    acc += x[b * K + k * D + t] * cb[code * D + t];
                }
            }
            out[b * N + j] = acc;
        }
    }
    out
}

#[test]
fn synth_codebook_layer_trains_with_muon() {
    let x: Vec<f32> = (0..B * K).map(|i| (i as f32 * 0.17).sin()).collect();
    let idx: Vec<u8> = (0..N * kb()).map(|i| ((i * 3 + 1) % NE) as u8).collect();
    let cb_true: Vec<f32> = (0..NE * D).map(|i| (i as f32 * 0.5).cos() * 0.9).collect();
    let target = synth_ref(&x, &idx, &cb_true);

    // Forward: y = synth_matmul(x, indices, codebook) ; loss = Σ(y − target)².
    let mut g = Graph::new("synth_layer");
    let xn = g.add_node(
        Op::Constant {
            data: x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
        vec![],
        Shape::new(&[B, K], DType::F32),
    );
    let indices = g.param("indices", Shape::new(&[N, kb()], DType::U8));
    let codebook = g.param("codebook", Shape::new(&[NE, D], DType::F32));
    let y = g.synth_matmul(
        xn,
        indices,
        codebook,
        SynthKind::Codebook {
            entry_dim: D as u32,
            num_entries: NE as u32,
        },
        Shape::new(&[B, N], DType::F32),
    );
    let tn = g.add_node(
        Op::Constant {
            data: target.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
        vec![],
        Shape::new(&[B, N], DType::F32),
    );
    let diff = g.add_node(
        Op::Binary(BinaryOp::Sub),
        vec![y, tn],
        Shape::new(&[B, N], DType::F32),
    );
    let sq = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![diff, diff],
        Shape::new(&[B, N], DType::F32),
    );
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![(B * N) as i64],
        },
        vec![sq],
        Shape::new(&[B * N], DType::F32),
    );
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![flat],
        Shape::from_dims(&[], DType::F32),
    );
    g.set_outputs(vec![loss]); // grad_with_loss appends grad(codebook)

    let backward = grad_with_loss(&g, &[codebook]);
    let mut c = Session::new(Device::Cpu).compile(backward);
    c.set_param_typed("indices", &idx, DType::U8);

    let mut cb = vec![0f32; NE * D]; // codebook starts at zero
    let mut muon = Muon::new(0.1);
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for step in 0..200 {
        c.set_param("codebook", &cb);
        let outs = c.run(&[("d_output", &[1.0f32])]); // [loss, grad_codebook]
        let loss_val = outs[0][0];
        if step == 0 {
            first = loss_val;
        }
        last = loss_val;
        muon.step("codebook", &[NE, D], &mut cb, &outs[1]); // Muon orthogonalizes 2-D codebook
    }

    eprintln!("codebook(Muon): loss {first:.4} → {last:.6}");
    assert!(first.is_finite() && last.is_finite());
    assert!(cb.iter().all(|v| v.is_finite()));
    assert!(
        last < first * 0.1,
        "codebook layer should train: loss {first} → {last}"
    );
}

/// The SAME codebook layer, trained GPU-RESIDENT on Metal through the first-class
/// runtime API (`bind_gpu_handle` + `run_read_outputs(Some(&[0]))` +
/// `CompiledGraph::optimizer_step_resident`) with real `Muon` — no GPU→host→GPU
/// roundtrip. Exercises the fixed Metal backward: the forward SynthMatMul native
/// kernel and the VJP's u8-indexed `Cast→Gather` / `Cast→ScatterAdd` both read the
/// packed-u8 `indices` param correctly.
#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn synth_codebook_layer_trains_resident_on_metal() {
    if !rlx_runtime::is_available(Device::Metal) {
        return;
    }
    let x: Vec<f32> = (0..B * K).map(|i| (i as f32 * 0.17).sin()).collect();
    let idx: Vec<u8> = (0..N * kb()).map(|i| ((i * 3 + 1) % NE) as u8).collect();
    let cb_true: Vec<f32> = (0..NE * D).map(|i| (i as f32 * 0.5).cos() * 0.9).collect();
    let target = synth_ref(&x, &idx, &cb_true);

    let mut g = Graph::new("synth_layer_metal");
    let xn = g.add_node(
        Op::Constant {
            data: x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
        vec![],
        Shape::new(&[B, K], DType::F32),
    );
    let indices = g.param("indices", Shape::new(&[N, kb()], DType::U8));
    let codebook = g.param("codebook", Shape::new(&[NE, D], DType::F32));
    let y = g.synth_matmul(
        xn,
        indices,
        codebook,
        SynthKind::Codebook {
            entry_dim: D as u32,
            num_entries: NE as u32,
        },
        Shape::new(&[B, N], DType::F32),
    );
    let tn = g.add_node(
        Op::Constant {
            data: target.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
        vec![],
        Shape::new(&[B, N], DType::F32),
    );
    let diff = g.add_node(
        Op::Binary(BinaryOp::Sub),
        vec![y, tn],
        Shape::new(&[B, N], DType::F32),
    );
    let sq = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![diff, diff],
        Shape::new(&[B, N], DType::F32),
    );
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![(B * N) as i64],
        },
        vec![sq],
        Shape::new(&[B * N], DType::F32),
    );
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![flat],
        Shape::from_dims(&[], DType::F32),
    );
    g.set_outputs(vec![loss]);

    // Backward; rebind the trainable codebook Param → Input so it can be resident.
    let mut backward = grad_with_loss(&g, &[codebook]);
    for node in backward.nodes_mut() {
        if let Op::Param { name } = &node.op
            && name == "codebook"
        {
            node.op = Op::Input { name: name.clone() };
        }
    }

    let mut c = Session::new(Device::Metal).compile(backward);
    c.set_param_typed("indices", &idx, DType::U8); // fixed u8 quantization structure
    assert!(
        c.bind_gpu_handle("codebook", &[0f32; NE * D]),
        "codebook should bind resident"
    );

    let mut muon = Muon::new(0.1);
    let trainable = vec![("codebook".to_string(), vec![NE, D])];
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    let mut stepped = false;
    for step in 0..200 {
        // Read only the loss; grad_codebook stays resident in the arena.
        let loss_val = c.run_read_outputs(&[("d_output", &[1.0f32])], Some(&[0]))[0][0];
        if step == 0 {
            first = loss_val;
        }
        last = loss_val;
        // Muon steps on the arena-aliased codebook + its resident grad, in place.
        stepped =
            c.optimizer_step_resident(&trainable, &mut |n, s, p, grad| muon.step(n, s, p, grad));
    }

    assert!(stepped, "Metal must support optimizer_step_resident");
    assert!(first.is_finite() && last.is_finite());
    eprintln!("resident codebook(Muon) on Metal: loss {first:.4} → {last:.6}");
    assert!(
        last < first * 0.1,
        "resident SynthMatMul codebook layer should train on Metal: loss {first} → {last}"
    );
}
