// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **A re-bound param must reach a baked weight pack.**
//!
//! The matmul-fusion passes (`shared_input_matmul`, `swiglu_dual`) fuse Q/K/V
//! and gate/up into one GEMM by concatenating their weight tensors. The
//! resulting `Concat` depends only on `Param`s, so its value is fixed once the
//! weights are bound — and re-running it every decode step is pure waste.
//!
//! Measured on Carbon-500M (a stock Llama, 28 layers) decoding on Metal, via
//! `RLX_METAL_DUMP_BYTES`:
//!
//! ```text
//! sgemm      113   2049.81 MB   52.0%
//! concat     112   1881.11 MB   47.7%   <- 1879 MB of it is the weight packs
//! ```
//!
//! Nearly half of all DRAM traffic per token, re-materialising constants. The
//! backend can skip those encodes after the first step (`baked_weight_concats`),
//! which is a ~1.9x reduction in per-step bytes.
//!
//! Skipping is only correct while the params underneath do not change. This
//! pins that: bind, run, re-bind, run — the second result must reflect the new
//! weights. Without invalidation the fused pack keeps run 1's bytes and the
//! GEMM silently computes with stale weights, which is the shape of every
//! weight-swap workload (a training step, a LoRA merge, quantisation re-binding,
//! a sweep harness reusing one executable across weight sets). A parity test
//! that binds once cannot see it.

#![cfg(target_os = "macos")]

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_metal::backend::MetalExecutable;

/// `x @ concat([w_a, w_b], axis=1)` — the fused-projection shape, minimally.
fn packed_matmul_graph() -> Graph {
    let mut g = Graph::new("packed");
    let x = g.input("x", Shape::new(&[1, 2], DType::F32));
    let w_a = g.param("w_a", Shape::new(&[2, 2], DType::F32));
    let w_b = g.param("w_b", Shape::new(&[2, 2], DType::F32));
    let w = g.concat_(vec![w_a, w_b], 1);
    let y = g.matmul(x, w, Shape::new(&[1, 4], DType::F32));
    g.set_outputs(vec![y]);
    g
}

const IDENT: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
const X: [f32; 2] = [1.0, 2.0];

#[test]
fn rebinding_a_param_updates_a_baked_weight_pack() {
    let mut exe = MetalExecutable::compile(packed_matmul_graph());
    exe.set_param("w_a", &IDENT);
    exe.set_param("w_b", &IDENT);

    // Run 1 materialises (and bakes) the pack.
    let first = exe.run(&[("x", &X)])[0].clone();
    assert_eq!(first, vec![1.0, 2.0, 1.0, 2.0], "run 1 is the baseline");

    // Doubling `w_a` must double the first two outputs; `w_b` is untouched, so
    // the last two must not move.
    exe.set_param("w_a", &[2.0, 0.0, 0.0, 2.0]);
    let second = exe.run(&[("x", &X)])[0].clone();

    assert_eq!(
        second,
        vec![2.0, 4.0, 1.0, 2.0],
        "re-bound `w_a` did not reach the fused pack — the matmul is still \
         reading run 1's concat. Got {second:?}"
    );
}

/// `set_param_bytes` reaches arena storage by a different route than
/// `set_param`, so it needs its own invalidation.
#[test]
fn rebinding_via_set_param_bytes_also_updates_the_pack() {
    let mut exe = MetalExecutable::compile(packed_matmul_graph());
    exe.set_param("w_a", &IDENT);
    exe.set_param("w_b", &IDENT);
    let _ = exe.run(&[("x", &X)]);

    let doubled: Vec<u8> = [2.0f32, 0.0, 0.0, 2.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    exe.set_param_bytes("w_a", &doubled);
    let second = exe.run(&[("x", &X)])[0].clone();

    assert_eq!(
        second,
        vec![2.0, 4.0, 1.0, 2.0],
        "re-bound `w_a` (bytes) did not reach the fused pack. Got {second:?}"
    );
}

/// `set_param_range` writes a sub-range in place — used by expert paging to
/// upload one changed slot of a packed residency buffer. It is the easiest of
/// the three to overlook because it never touches the whole param.
#[test]
fn a_partial_param_write_also_updates_the_pack() {
    let mut exe = MetalExecutable::compile(packed_matmul_graph());
    exe.set_param("w_a", &IDENT);
    exe.set_param("w_b", &IDENT);
    let _ = exe.run(&[("x", &X)]);

    // Overwrite only w_a[0] (first 4 bytes): 1.0 -> 2.0.
    if exe.set_param_range("w_a", 0, &2.0f32.to_le_bytes()) {
        let second = exe.run(&[("x", &X)])[0].clone();
        assert_eq!(
            second,
            vec![2.0, 2.0, 1.0, 2.0],
            "partial re-bind of `w_a` did not reach the fused pack. Got {second:?}"
        );
    } else {
        // Param parked in a separate weight buffer — the caller re-uploads
        // whole, which the other two tests already cover. Report rather than
        // pass silently.
        eprintln!("set_param_range declined (weight-buffer param) — not exercised");
    }
}
