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

//! Host-agnostic basic tests for rlx-tpu.
//!
//! These tests run on any host (Mac, Linux CI, Windows) and exercise
//! the parts of the backend that don't need a live PJRT plugin:
//!   * `is_available()` reports correctly
//!   * The HLO emitter produces non-empty bytes for trivial graphs
//!   * The full lowering walker runs without panics on a small but
//!     realistic op coverage (BERT-shaped fragment)
//!
//! The actual PJRT compile + execute round-trip lives in
//! `tests/pjrt_roundtrip.rs`, which is gated on `LIBTPU_PATH` so it
//! runs in Docker (with `libpjrt_c_cpu.so`) or on a TPU VM but stays
//! quiet on hosts without a plugin installed.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};

#[test]
fn is_available_is_false_without_libtpu() {
    if std::env::var("LIBTPU_PATH").is_ok() {
        // Tester explicitly opted into a libtpu / libpjrt plugin —
        // the round-trip suite covers that path. Skip here.
        return;
    }
    assert!(
        !rlx_tpu::is_available(),
        "is_available() unexpectedly true on a host without LIBTPU_PATH set."
    );
}

#[test]
fn lower_emits_nonempty_hlo_for_minimal_graph() {
    // Two-input add. Walks Input → Binary → output to verify the
    // skeleton lowering pipeline does the right thing.
    let mut g = Graph::new("add_check");
    let s = Shape::new(&[2, 3], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let z = g.binary(BinaryOp::Add, x, y, s);
    g.set_outputs(vec![z]);

    let module = rlx_tpu::lower::lower_graph(&g);
    assert!(
        !module.bytes.is_empty(),
        "HLO module bytes should be non-empty"
    );
    assert_eq!(module.input_names, vec!["x".to_string(), "y".to_string()]);
    assert_eq!(module.output_lens, vec![6]);
}

#[test]
fn lower_emits_nonempty_hlo_for_bert_fragment() {
    // Embedding lookup → MatMul → bias-add → GELU → LayerNorm.
    // Stress-tests the composite lowerings (LayerNorm decomposes
    // into mean / var / scale; GELU into erf form) so a single
    // check pass catches regressions in any of them.
    let mut g = Graph::new("bert_fragment");
    let f32 = DType::F32;
    let ids = g.input("ids", Shape::new(&[1, 8], f32));
    let table = g.param("emb_table", Shape::new(&[1024, 32], f32));
    let w = g.param("ffn_w", Shape::new(&[32, 32], f32));
    let b = g.param("ffn_b", Shape::new(&[32], f32));
    let gamma = g.param("ln_g", Shape::new(&[32], f32));
    let beta = g.param("ln_b", Shape::new(&[32], f32));

    let emb = g.add_node(
        rlx_ir::Op::Gather { axis: 0 },
        vec![table, ids],
        Shape::new(&[1, 8, 32], f32),
    );
    // 2D matmul on flattened [8, 32].
    let flat = g.reshape(emb, vec![8, 32], Shape::new(&[8, 32], f32));
    let mm = g.matmul(flat, w, Shape::new(&[8, 32], f32));
    // Add broadcast bias and run GELU.
    let b_b = g.add_node(
        rlx_ir::Op::Expand {
            target_shape: vec![8, 32],
        },
        vec![b],
        Shape::new(&[8, 32], f32),
    );
    let added = g.binary(BinaryOp::Add, mm, b_b, Shape::new(&[8, 32], f32));
    let acted = g.activation(Activation::Gelu, added, Shape::new(&[8, 32], f32));
    let normed = g.layer_norm(acted, gamma, beta, -1, 1e-5, Shape::new(&[8, 32], f32));
    g.set_outputs(vec![normed]);

    let module = rlx_tpu::lower::lower_graph(&g);
    assert!(
        module.bytes.len() > 200,
        "BERT-fragment HLO unexpectedly tiny ({} bytes)",
        module.bytes.len()
    );
    assert_eq!(module.output_lens, vec![8 * 32]);
    assert_eq!(module.input_names, vec!["ids".to_string()]);
    assert_eq!(module.param_names.len(), 5);
}

#[test]
fn compile_without_plugin_panics_with_clear_message() {
    if std::env::var("LIBTPU_PATH").is_ok() {
        // Host has a plugin — TpuExecutable::compile would actually
        // succeed. Skip; the round-trip suite covers that.
        return;
    }
    // Build a trivial graph and try to compile. Should panic with a
    // message that mentions rlx-tpu so the user knows where to look.
    let mut g = Graph::new("compile_no_plugin");
    let s = Shape::new(&[4], DType::F32);
    let x = g.input("x", s.clone());
    g.set_outputs(vec![x]);

    let r = std::panic::catch_unwind(|| {
        let _ = rlx_tpu::TpuExecutable::compile(g);
    });
    let err = r.expect_err("compile() should panic without a plugin");
    let msg = if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = err.downcast_ref::<&'static str>() {
        s.to_string()
    } else {
        String::from("<non-string panic>")
    };
    assert!(
        msg.contains("rlx-tpu"),
        "panic message should mention rlx-tpu, got: {msg}"
    );
}
