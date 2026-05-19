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

//! HLO-byte structural matching tests.
//!
//! For each lowering family, we emit a small graph through
//! `lower::lower_graph` and grep the resulting `HloModuleProto` bytes
//! for the opcode strings we expect to see. HLO opcodes are encoded
//! as length-prefixed UTF-8 strings inside `HloInstructionProto`'s
//! field 2, so they appear verbatim in the wire-format bytes — a
//! plain `windows`-search is sufficient and doesn't pull in a
//! protobuf parser.
//!
//! These tests catch lowering regressions (e.g. silu losing its
//! `logistic` step, attention dropping the `iota` causal-mask
//! synthesis) without needing a live PJRT plugin. The Docker harness
//! goes further and round-trips the bytes through XLA's deserializer
//! for proto validity, but the pure-Rust path here runs on every
//! `cargo test -p rlx-tpu`.

use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{DType, Graph, Shape};

/// Returns true iff `needle` (an opcode string) appears as a
/// length-prefixed token inside `bytes`. Length prefix is varint —
/// a `(needle.len() as u8)` byte for short opcodes (which all of
/// ours are) — followed by the UTF-8 bytes themselves.
fn contains_opcode(bytes: &[u8], needle: &str) -> bool {
    let n = needle.len();
    assert!(n < 128, "opcode > 127 chars not supported by this scanner");
    // Look for the (varint-len, ascii) pair. To rule out coincidental
    // matches inside other strings (e.g. "add" inside "padded"), we
    // also confirm the byte BEFORE the needle is the wire tag for
    // string field 2 (`opcode`) on `HloInstructionProto`: tag = 0x12
    // (field 2, wire-type 2).
    //
    // Wire layout of an HloInstructionProto's `opcode` field:
    //     0x12  <varint length>  <utf8 bytes>
    // …but the surrounding HloInstructionProto is itself a length-
    // delimited field of a containing message, so we can't match the
    // outer 0x12. Match on the inner pattern: 0x12, varint(n), then
    // the bytes of the opcode.
    let needle_bytes = needle.as_bytes();
    let mut i = 0;
    while i + 2 + n <= bytes.len() {
        if bytes[i] == 0x12 && bytes[i + 1] == n as u8 && &bytes[i + 2..i + 2 + n] == needle_bytes {
            return true;
        }
        i += 1;
    }
    false
}

fn lower_to_bytes(g: &Graph) -> Vec<u8> {
    rlx_tpu::lower::lower_graph(g).bytes
}

#[test]
fn ew_add_emits_add_opcode() {
    let mut g = Graph::new("ew_add");
    let s = Shape::new(&[6], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let z = g.binary(BinaryOp::Add, x, y, s);
    g.set_outputs(vec![z]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "add"),
        "expected `add` opcode in element-wise add module"
    );
    assert!(
        contains_opcode(&b, "parameter"),
        "expected at least one `parameter` instruction"
    );
}

#[test]
fn matmul_emits_dot_opcode() {
    let mut g = Graph::new("mm");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[4, 3], f));
    let w = g.param("w", Shape::new(&[3, 5], f));
    let y = g.matmul(x, w, Shape::new(&[4, 5], f));
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "dot"),
        "matmul lowers to HLO `dot` with dot_dimension_numbers"
    );
}

#[test]
fn relu_emits_maximum_opcode() {
    // Relu(x) is decomposed as max(x, 0) in HLO — there's no relu
    // primitive opcode.
    let mut g = Graph::new("relu");
    let s = Shape::new(&[4], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.activation(Activation::Relu, x, s);
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "maximum"),
        "Relu should lower to HLO `maximum`"
    );
    assert!(
        !contains_opcode(&b, "relu"),
        "should NOT contain a literal `relu` opcode (no such HLO)"
    );
}

#[test]
fn gelu_emits_erf_opcode() {
    // Exact GELU = 0.5 * x * (1 + erf(x / sqrt(2))).
    let mut g = Graph::new("gelu");
    let s = Shape::new(&[4], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.activation(Activation::Gelu, x, s);
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "erf"),
        "exact-form GELU should emit HLO `erf`"
    );
}

#[test]
fn gelu_approx_emits_tanh_not_erf() {
    // tanh-form GELU: 0.5 * x * (1 + tanh(...)).
    let mut g = Graph::new("gelu_approx");
    let s = Shape::new(&[4], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.activation(Activation::GeluApprox, x, s);
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "tanh"),
        "approx GELU should emit HLO `tanh` (not erf)"
    );
    assert!(
        !contains_opcode(&b, "erf"),
        "approx GELU must NOT emit erf — would defeat the approx form"
    );
}

#[test]
fn silu_emits_logistic_and_multiply() {
    // silu(x) = x * sigmoid(x); HLO sigmoid is `logistic`.
    let mut g = Graph::new("silu");
    let s = Shape::new(&[4], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.activation(Activation::Silu, x, s);
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "logistic"),
        "silu uses HLO `logistic` for the sigmoid step"
    );
    assert!(
        contains_opcode(&b, "multiply"),
        "silu finishes with x * sigmoid(x)"
    );
}

#[test]
fn layernorm_decomposes_to_mean_var_rsqrt() {
    let mut g = Graph::new("ln");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[2, 4], f));
    let gv = g.param("g", Shape::new(&[4], f));
    let bv = g.param("b", Shape::new(&[4], f));
    let y = g.layer_norm(x, gv, bv, -1, 1e-5, Shape::new(&[2, 4], f));
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    // mean → centered → variance → rsqrt → scale → bias.
    for op in ["reduce", "subtract", "multiply", "rsqrt", "add"] {
        assert!(
            contains_opcode(&b, op),
            "layernorm lowering missing `{op}` opcode"
        );
    }
}

#[test]
fn rmsnorm_uses_rsqrt_no_subtract() {
    // RMS norm: x / sqrt(mean(x^2) + eps) — no centering, so no
    // `subtract` should appear (vs LayerNorm which does center).
    let mut g = Graph::new("rms");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[2, 4], f));
    let gv = g.param("g", Shape::new(&[4], f));
    let bv = g.param("b", Shape::new(&[4], f));
    let y = g.add_node(
        rlx_ir::Op::RmsNorm {
            axis: -1,
            eps: 1e-6,
        },
        vec![x, gv, bv],
        Shape::new(&[2, 4], f),
    );
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(contains_opcode(&b, "rsqrt"));
    assert!(contains_opcode(&b, "reduce"));
    assert!(contains_opcode(&b, "multiply"));
    assert!(
        !contains_opcode(&b, "subtract"),
        "RMS norm doesn't center — no `subtract` should appear"
    );
}

#[test]
fn softmax_decomposes_to_max_sub_exp_sum_div() {
    let mut g = Graph::new("sm");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[2, 5], f));
    let y = g.softmax(x, -1, Shape::new(&[2, 5], f));
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    for op in ["reduce", "subtract", "exponential", "divide"] {
        assert!(
            contains_opcode(&b, op),
            "softmax lowering missing `{op}` opcode"
        );
    }
}

#[test]
fn compare_lowers_to_compare_opcode() {
    let mut g = Graph::new("cmp");
    let f = DType::F32;
    let s = Shape::new(&[4], f);
    let a = g.input("a", s.clone());
    let bn = g.input("b", s.clone());
    let cond = g.add_node(
        rlx_ir::Op::Compare(CmpOp::Lt),
        vec![a, bn],
        Shape::new(&[4], DType::Bool),
    );
    g.set_outputs(vec![cond]);
    let b = lower_to_bytes(&g);
    assert!(contains_opcode(&b, "compare"));
}

#[test]
fn where_lowers_to_select() {
    let mut g = Graph::new("wh");
    let f = DType::F32;
    let s = Shape::new(&[4], f);
    let a = g.input("a", s.clone());
    let bn = g.input("b", s.clone());
    let cond = g.add_node(
        rlx_ir::Op::Compare(CmpOp::Lt),
        vec![a, bn],
        Shape::new(&[4], DType::Bool),
    );
    let sel = g.add_node(rlx_ir::Op::Where, vec![cond, a, bn], s);
    g.set_outputs(vec![sel]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "select"),
        "Where should emit HLO `select`"
    );
}

#[test]
fn shape_ops_emit_reshape_transpose_slice() {
    let mut g = Graph::new("shape");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[2, 6], f));
    let r = g.reshape(x, vec![2, 2, 3], Shape::new(&[2, 2, 3], f));
    let t = g.add_node(
        rlx_ir::Op::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![r],
        Shape::new(&[2, 3, 2], f),
    );
    let n = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 1,
            start: 0,
            len: 2,
        },
        vec![t],
        Shape::new(&[2, 2, 2], f),
    );
    g.set_outputs(vec![n]);
    let b = lower_to_bytes(&g);
    for op in ["reshape", "transpose", "slice"] {
        assert!(
            contains_opcode(&b, op),
            "shape lowering missing `{op}` opcode"
        );
    }
}

#[test]
fn gather_emits_gather_opcode() {
    let mut g = Graph::new("gather");
    let f = DType::F32;
    let table = g.param("table", Shape::new(&[16, 4], f));
    let idx = g.input("idx", Shape::new(&[3], DType::I32));
    let out = g.add_node(
        rlx_ir::Op::Gather { axis: 0 },
        vec![table, idx],
        Shape::new(&[3, 4], f),
    );
    g.set_outputs(vec![out]);
    let b = lower_to_bytes(&g);
    assert!(contains_opcode(&b, "gather"));
}

#[test]
fn reduce_emits_reduce_opcode() {
    let mut g = Graph::new("red");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[3, 4], f));
    let y = g.add_node(
        rlx_ir::Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![1],
            keep_dim: false,
        },
        vec![x],
        Shape::new(&[3], f),
    );
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "reduce"),
        "Reduce should emit HLO `reduce`"
    );
}

#[test]
fn attention_causal_emits_iota_compare_select() {
    // Causal attention mask is synthesized via iota+compare+select
    // rather than materialized as a tensor on the host.
    let mut g = Graph::new("attn");
    let f = DType::F32;
    let q = g.input("q", Shape::new(&[1, 2, 4, 8], f));
    let k = g.input("k", Shape::new(&[1, 2, 4, 8], f));
    let v = g.input("v", Shape::new(&[1, 2, 4, 8], f));
    let out = g.attention_kind(
        q,
        k,
        v,
        2,
        8,
        MaskKind::Causal,
        Shape::new(&[1, 2, 4, 8], f),
    );
    g.set_outputs(vec![out]);
    let b = lower_to_bytes(&g);
    for op in ["dot", "iota", "compare", "select", "exponential"] {
        assert!(
            contains_opcode(&b, op),
            "causal attention lowering missing `{op}` opcode — \
                 mask synthesis broken?"
        );
    }
}

#[test]
fn attention_none_omits_iota_and_select() {
    // No mask kind: scaled QK^T → softmax → @V. No iota / compare /
    // select needed.
    let mut g = Graph::new("attn_none");
    let f = DType::F32;
    let q = g.input("q", Shape::new(&[1, 2, 4, 8], f));
    let k = g.input("k", Shape::new(&[1, 2, 4, 8], f));
    let v = g.input("v", Shape::new(&[1, 2, 4, 8], f));
    let out = g.attention_kind(q, k, v, 2, 8, MaskKind::None, Shape::new(&[1, 2, 4, 8], f));
    g.set_outputs(vec![out]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "dot"),
        "MaskKind::None still emits Q·K^T and probs·V via `dot`"
    );
    assert!(
        !contains_opcode(&b, "iota"),
        "MaskKind::None must NOT synthesize a mask (no iota)"
    );
    assert!(
        !contains_opcode(&b, "select"),
        "MaskKind::None has no mask to apply (no select)"
    );
}

#[test]
fn rope_emits_slice_concat_pair() {
    let mut g = Graph::new("rope");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[1, 2, 4, 8], f));
    let cos = g.input("cos", Shape::new(&[4, 4], f));
    let sin = g.input("sin", Shape::new(&[4, 4], f));
    let out = g.add_node(
        rlx_ir::Op::Rope { head_dim: 8 },
        vec![x, cos, sin],
        Shape::new(&[1, 2, 4, 8], f),
    );
    g.set_outputs(vec![out]);
    let b = lower_to_bytes(&g);
    for op in ["slice", "multiply", "concatenate"] {
        assert!(
            contains_opcode(&b, op),
            "rope lowering missing `{op}` opcode"
        );
    }
}

#[test]
fn cumsum_emits_reduce_window() {
    let mut g = Graph::new("csum");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[8], f));
    let y = g.add_node(
        rlx_ir::Op::Cumsum {
            axis: 0,
            exclusive: false,
        },
        vec![x],
        Shape::new(&[8], f),
    );
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "reduce-window"),
        "Cumsum should emit HLO `reduce-window`"
    );
}

#[test]
fn pool_emits_reduce_window() {
    // 2-D max pool.
    let mut g = Graph::new("pool");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[1, 1, 4, 4], f));
    let y = g.add_node(
        rlx_ir::Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 2, 2], f),
    );
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(contains_opcode(&b, "reduce-window"));
}

#[test]
fn conv_emits_convolution_opcode() {
    let mut g = Graph::new("conv");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[1, 3, 8, 8], f));
    let w = g.param("w", Shape::new(&[6, 3, 3, 3], f));
    let y = g.add_node(
        rlx_ir::Op::Conv {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![x, w],
        Shape::new(&[1, 6, 8, 8], f),
    );
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(contains_opcode(&b, "convolution"));
}

#[test]
fn cast_emits_convert_opcode() {
    let mut g = Graph::new("cast");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.add_node(
        rlx_ir::Op::Cast { to: DType::F16 },
        vec![x],
        Shape::new(&[4], DType::F16),
    );
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "convert"),
        "Cast should emit HLO `convert`"
    );
}

#[test]
fn constant_emits_constant_opcode() {
    let mut g = Graph::new("k");
    let bytes = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let c = g.add_node(
        rlx_ir::Op::Constant { data: bytes },
        vec![],
        Shape::new(&[4], DType::F32),
    );
    let s = Shape::new(&[4], DType::F32);
    let x = g.input("x", s.clone());
    let z = g.binary(BinaryOp::Add, x, c, s);
    g.set_outputs(vec![z]);
    let b = lower_to_bytes(&g);
    assert!(contains_opcode(&b, "constant"));
    assert!(contains_opcode(&b, "add"));
}

#[test]
fn output_tuple_emits_tuple_opcode() {
    // Multi-output entry computations wrap the outputs in a `tuple`.
    let mut g = Graph::new("two_outs");
    let s = Shape::new(&[4], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let a = g.binary(BinaryOp::Add, x, y, s.clone());
    let m = g.binary(BinaryOp::Mul, x, y, s);
    g.set_outputs(vec![a, m]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "tuple"),
        "multiple outputs should be wrapped in HLO `tuple`"
    );
}

// ── Tier-3 op lowerings (parity with rlx-cuda / rlx-rocm) ──────────

#[test]
fn topk_emits_sort_and_slice() {
    let mut g = Graph::new("topk");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[2, 16], f));
    let y = g.add_node(rlx_ir::Op::TopK { k: 3 }, vec![x], Shape::new(&[2, 3], f));
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "sort"),
        "TopK should sort the last axis"
    );
    assert!(
        contains_opcode(&b, "slice"),
        "TopK should slice the leading k"
    );
    assert!(
        contains_opcode(&b, "iota"),
        "TopK should emit iota for the index column"
    );
    assert!(
        contains_opcode(&b, "get-tuple-element"),
        "TopK should project the indices out of sort's tuple"
    );
}

#[test]
fn grouped_matmul_emits_gather_and_dot() {
    let mut g = Graph::new("gemm");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[4, 8], f));
    let w = g.param("w", Shape::new(&[3, 8, 6], f));
    let e = g.input("e", Shape::new(&[4], f));
    let y = g.add_node(
        rlx_ir::Op::GroupedMatMul,
        vec![x, w, e],
        Shape::new(&[4, 6], f),
    );
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "gather"),
        "GroupedMatMul should gather per-token weights"
    );
    assert!(
        contains_opcode(&b, "dot"),
        "GroupedMatMul should dot the gathered weights"
    );
}

#[test]
fn dequant_matmul_emits_convert_and_dot() {
    let mut g = Graph::new("dq");
    let f = DType::F32;
    let i8t = DType::I8;
    let x = g.input("x", Shape::new(&[2, 8], f));
    let wq = g.param("wq", Shape::new(&[8, 4], i8t));
    let scale = g.param("scale", Shape::new(&[2, 4], f));
    let zp = g.param("zp", Shape::new(&[2, 4], f));
    let y = g.add_node(
        rlx_ir::Op::DequantMatMul {
            scheme: rlx_ir::quant::QuantScheme::Int8BlockAsym { block_size: 4 },
        },
        vec![x, wq, scale, zp],
        Shape::new(&[2, 4], f),
    );
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "convert"),
        "DequantMatMul should promote w_q to f32 via convert"
    );
    assert!(
        contains_opcode(&b, "subtract"),
        "DequantMatMul should subtract zero point"
    );
    assert!(
        contains_opcode(&b, "multiply"),
        "DequantMatMul should multiply by scale"
    );
    assert!(
        contains_opcode(&b, "dot"),
        "DequantMatMul should still emit a dot"
    );
}

#[test]
fn qmatmul_emits_dot_and_clamp() {
    let mut g = Graph::new("qmm");
    let i8t = DType::I8;
    let i32t = DType::I32;
    let x = g.input("x", Shape::new(&[2, 8], i8t));
    let w = g.param("w", Shape::new(&[8, 4], i8t));
    let bias = g.param("bias", Shape::new(&[4], i32t));
    let y = g.add_node(
        rlx_ir::Op::QMatMul {
            x_zp: 0,
            w_zp: 0,
            out_zp: 0,
            mult: 0.5,
        },
        vec![x, w, bias],
        Shape::new(&[2, 4], i8t),
    );
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(contains_opcode(&b, "dot"));
    assert!(contains_opcode(&b, "convert"));
    assert!(
        contains_opcode(&b, "round-nearest-even"),
        "QMatMul should round before requantizing"
    );
    assert!(contains_opcode(&b, "maximum"));
    assert!(contains_opcode(&b, "minimum"));
}

#[test]
fn qconv2d_emits_convolution_and_clamp() {
    let mut g = Graph::new("qcv");
    let i8t = DType::I8;
    let i32t = DType::I32;
    let x = g.input("x", Shape::new(&[1, 3, 8, 8], i8t));
    let w = g.param("w", Shape::new(&[6, 3, 3, 3], i8t));
    let bias = g.param("bias", Shape::new(&[6], i32t));
    let y = g.add_node(
        rlx_ir::Op::QConv2d {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            x_zp: 0,
            w_zp: 0,
            out_zp: 0,
            mult: 0.25,
        },
        vec![x, w, bias],
        Shape::new(&[1, 6, 8, 8], i8t),
    );
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(contains_opcode(&b, "convolution"));
    assert!(contains_opcode(&b, "round-nearest-even"));
    assert!(contains_opcode(&b, "maximum"));
    assert!(contains_opcode(&b, "minimum"));
}

#[test]
fn sample_greedy_emits_sort_no_rng() {
    let mut g = Graph::new("sample_greedy");
    let f = DType::F32;
    let logits = g.input("logits", Shape::new(&[2, 16], f));
    let y = g.add_node(
        rlx_ir::Op::Sample {
            top_k: 0,
            top_p: 1.0,
            temperature: 0.0,
            seed: 0,
        },
        vec![logits],
        Shape::new(&[2], f),
    );
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "sort"),
        "Greedy sample should argmax via sort"
    );
    assert!(
        !contains_opcode(&b, "rng"),
        "Greedy sample shouldn't draw random numbers"
    );
}

#[test]
fn sample_temperature_emits_rng_and_softmax() {
    let mut g = Graph::new("sample_t");
    let f = DType::F32;
    let logits = g.input("logits", Shape::new(&[2, 16], f));
    let y = g.add_node(
        rlx_ir::Op::Sample {
            top_k: 0,
            top_p: 1.0,
            temperature: 0.7,
            seed: 0,
        },
        vec![logits],
        Shape::new(&[2], f),
    );
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "rng"),
        "Temperature sampling should use HLO rng"
    );
    assert!(contains_opcode(&b, "exponential"));
    assert!(contains_opcode(&b, "divide"));
    assert!(
        contains_opcode(&b, "reduce-window"),
        "Multinomial uses cumsum (reduce-window)"
    );
}

#[test]
fn selective_scan_emits_while_and_dynamic_slice() {
    let mut g = Graph::new("ssm");
    let f = DType::F32;
    let bsz = 1;
    let l = 4;
    let d = 8;
    let n = 16;
    let x = g.input("x", Shape::new(&[bsz, l, d], f));
    let delta = g.input("delta", Shape::new(&[bsz, l, d], f));
    let a = g.param("a", Shape::new(&[d, n], f));
    let bb = g.input("b", Shape::new(&[bsz, l, n], f));
    let cc = g.input("c", Shape::new(&[bsz, l, n], f));
    let y = g.add_node(
        rlx_ir::Op::SelectiveScan { state_size: n },
        vec![x, delta, a, bb, cc],
        Shape::new(&[bsz, l, d], f),
    );
    g.set_outputs(vec![y]);
    let b = lower_to_bytes(&g);
    assert!(
        contains_opcode(&b, "while"),
        "SelectiveScan should compile to an HLO while loop"
    );
    assert!(contains_opcode(&b, "dynamic-slice"));
    assert!(contains_opcode(&b, "dynamic-update-slice"));
    assert!(
        contains_opcode(&b, "exponential"),
        "Mamba decay = exp(delta * a)"
    );
}

#[test]
fn elementwise_region_decomposes_into_chain() {
    // ElementwiseRegion lowers by walking the chain inline, so the
    // emitted HLO has all the individual primitives — `add`, `multiply`,
    // `maximum` (relu) — but no `region` opcode (HLO has no such thing
    // anyway; this is a rlx-only IR concept).
    use rlx_ir::op::{ChainOperand, ChainStep};
    let mut g = Graph::new("ew_region");
    let s = Shape::new(&[4], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let z = g.input("z", s.clone());

    let chain = vec![
        ChainStep::Binary(
            BinaryOp::Add,
            ChainOperand::Input(0),
            ChainOperand::Input(1),
        ),
        ChainStep::Binary(BinaryOp::Mul, ChainOperand::Step(0), ChainOperand::Input(2)),
        ChainStep::Activation(Activation::Relu, ChainOperand::Step(1)),
    ];
    let out = g.add_node(
        rlx_ir::Op::ElementwiseRegion {
            chain,
            num_inputs: 3,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
        },
        vec![x, y, z],
        s,
    );
    g.set_outputs(vec![out]);
    let b = lower_to_bytes(&g);
    for op in ["add", "multiply", "maximum"] {
        assert!(
            contains_opcode(&b, op),
            "ElementwiseRegion lowering missing `{op}` opcode"
        );
    }
}
