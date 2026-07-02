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

//! Round-trip the emitted HLO bytes through prost's decoder and assert
//! on the parsed proto structure.
//!
//! Where `tests/hlo_match.rs` does byte-grep for opcodes (cheap, but
//! brittle if our wire format ever drifts), this file decodes the
//! bytes back into the prost-generated `xla::HloModuleProto` and
//! navigates the message tree to assert programmatically — instruction
//! counts, opcodes, shapes, dim numbers, dot contracting/batch dims,
//! conv window strides, etc. The test set is the same generators used
//! by `hlo_match`, so a failure here pinpoints the structural property
//! that broke.

use prost::Message;
use rlx_ir::op::{BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{DType, Graph, Shape};

use rlx_tpu::xla;

fn lower_to_proto(g: &Graph) -> xla::HloModuleProto {
    let bytes = rlx_tpu::lower::lower_graph(g).bytes;
    xla::HloModuleProto::decode(bytes.as_slice())
        .expect("emitted HLO must decode cleanly via prost")
}

fn entry(m: &xla::HloModuleProto) -> &xla::HloComputationProto {
    m.computations
        .iter()
        .find(|c| c.id == m.entry_computation_id)
        .expect("entry computation must be present in computations[]")
}

fn opcodes(c: &xla::HloComputationProto) -> Vec<&str> {
    c.instructions.iter().map(|i| i.opcode.as_str()).collect()
}

#[test]
fn module_has_well_formed_header() {
    let mut g = Graph::new("hdr");
    let s = Shape::new(&[4], DType::F32);
    let x = g.input("x", s.clone());
    g.set_outputs(vec![x]);
    let m = lower_to_proto(&g);

    assert_eq!(m.name, "hdr");
    assert_eq!(m.entry_computation_name, "entry");
    assert!(
        m.entry_computation_id != 0,
        "entry_computation_id must be non-zero (proto3 default suppression)"
    );
    assert_eq!(m.computations.len(), 1);
    let e = entry(&m);
    assert_eq!(e.name, "entry");
    assert!(e.root_id != 0, "root_id must be set");
    assert!(
        e.program_shape.is_some(),
        "entry computation must carry a program_shape"
    );
    assert!(
        m.host_program_shape.is_some(),
        "module must carry host_program_shape (XLA requires it)"
    );
}

#[test]
fn dot_general_carries_correct_dim_numbers() {
    // Plain MatMul lowers to HLO `dot` with last-axis contracting on
    // both sides and no batch dims (rank-2 inputs).
    let mut g = Graph::new("mm");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[4, 3], f));
    let w = g.param("w", Shape::new(&[3, 5], f));
    let y = g.matmul(x, w, Shape::new(&[4, 5], f));
    g.set_outputs(vec![y]);
    let m = lower_to_proto(&g);
    let e = entry(&m);

    let dot = e
        .instructions
        .iter()
        .find(|i| i.opcode == "dot")
        .expect("MatMul must lower to a `dot` instruction");
    let dn = dot
        .dot_dimension_numbers
        .as_ref()
        .expect("dot must carry dot_dimension_numbers");
    assert_eq!(
        dn.lhs_contracting_dimensions,
        vec![1],
        "lhs contracts over axis 1 (last axis of [M, K])"
    );
    assert_eq!(
        dn.rhs_contracting_dimensions,
        vec![0],
        "rhs contracts over axis 0 (first axis of [K, N])"
    );
    assert!(
        dn.lhs_batch_dimensions.is_empty(),
        "rank-2 matmul has no batch dims"
    );
    assert!(dn.rhs_batch_dimensions.is_empty());

    let result_shape = dot.shape.as_ref().expect("dot has output shape");
    assert_eq!(result_shape.dimensions, vec![4, 5]);
}

#[test]
fn batched_matmul_picks_correct_batch_axes() {
    // [B=2, M=3, K=4] × [B=2, K=4, N=5] → [B=2, M=3, N=5].
    let mut g = Graph::new("mm_batched");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[2, 3, 4], f));
    let w = g.input("w", Shape::new(&[2, 4, 5], f));
    let y = g.matmul(x, w, Shape::new(&[2, 3, 5], f));
    g.set_outputs(vec![y]);
    let m = lower_to_proto(&g);
    let e = entry(&m);
    let dot = e
        .instructions
        .iter()
        .find(|i| i.opcode == "dot")
        .expect("batched MatMul must lower to `dot`");
    let dn = dot.dot_dimension_numbers.as_ref().unwrap();
    assert_eq!(dn.lhs_batch_dimensions, vec![0]);
    assert_eq!(dn.rhs_batch_dimensions, vec![0]);
    assert_eq!(dn.lhs_contracting_dimensions, vec![2]);
    assert_eq!(dn.rhs_contracting_dimensions, vec![1]);
}

#[test]
fn conv_carries_window_and_dim_numbers() {
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
    let m = lower_to_proto(&g);
    let e = entry(&m);
    let conv = e
        .instructions
        .iter()
        .find(|i| i.opcode == "convolution")
        .expect("Conv must lower to `convolution`");
    let cdn = conv
        .convolution_dimension_numbers
        .as_ref()
        .expect("convolution must carry dim numbers");
    assert_eq!(cdn.input_batch_dimension, 0);
    assert_eq!(cdn.input_feature_dimension, 1);
    assert_eq!(cdn.input_spatial_dimensions, vec![2, 3]);
    assert_eq!(cdn.kernel_output_feature_dimension, 0);
    assert_eq!(cdn.kernel_input_feature_dimension, 1);
    assert_eq!(cdn.kernel_spatial_dimensions, vec![2, 3]);

    let win = conv.window.as_ref().expect("convolution must carry window");
    assert_eq!(win.dimensions.len(), 2);
    assert_eq!(win.dimensions[0].size, 3);
    assert_eq!(win.dimensions[0].stride, 1);
    assert_eq!(win.dimensions[0].padding_low, 1);
    assert_eq!(win.dimensions[0].padding_high, 1);
}

#[test]
fn parameter_numbers_are_dense_and_in_order() {
    // Inputs first (parameter_number 0..N), then params (N..N+M).
    let mut g = Graph::new("ord");
    let f = DType::F32;
    let s = Shape::new(&[4], f);
    let x = g.input("x", s.clone());
    let w = g.param("w", s.clone());
    let y = g.input("y", s.clone());
    let xw = g.binary(BinaryOp::Add, x, w, s.clone());
    let z = g.binary(BinaryOp::Add, xw, y, s);
    g.set_outputs(vec![z]);
    let m = lower_to_proto(&g);
    let e = entry(&m);
    let mut params: Vec<i64> = e
        .instructions
        .iter()
        .filter(|i| i.opcode == "parameter")
        .map(|i| i.parameter_number)
        .collect();
    params.sort();
    assert_eq!(
        params,
        vec![0, 1, 2],
        "parameter_numbers must be a dense 0..N range"
    );
}

#[test]
fn compare_carries_direction_string() {
    let mut g = Graph::new("cmp");
    let f = DType::F32;
    let s = Shape::new(&[4], f);
    let a = g.input("a", s.clone());
    let b = g.input("b", s.clone());
    let c = g.add_node(
        rlx_ir::Op::Compare(CmpOp::Lt),
        vec![a, b],
        Shape::new(&[4], DType::Bool),
    );
    g.set_outputs(vec![c]);
    let m = lower_to_proto(&g);
    let e = entry(&m);
    let cmp = e
        .instructions
        .iter()
        .find(|i| i.opcode == "compare")
        .expect("Compare must lower to `compare`");
    assert_eq!(
        cmp.comparison_direction, "LT",
        "comparison_direction must round-trip through field 63"
    );
}

#[test]
fn shapes_have_layout_with_minor_to_major() {
    let mut g = Graph::new("layout");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[2, 3, 4], f));
    g.set_outputs(vec![x]);
    let m = lower_to_proto(&g);
    let e = entry(&m);
    let p = e
        .instructions
        .iter()
        .find(|i| i.opcode == "parameter")
        .unwrap();
    let sh = p.shape.as_ref().unwrap();
    assert_eq!(sh.dimensions, vec![2, 3, 4]);
    let lay = sh
        .layout
        .as_ref()
        .expect("non-tuple ShapeProto must carry a Layout");
    // Default layout is reverse-rank order (minor-to-major from last to first).
    assert_eq!(lay.minor_to_major, vec![2, 1, 0]);
    assert_eq!(
        lay.tail_padding_alignment_in_elements, 1,
        "JAX/XLA require tail_padding_alignment_in_elements=1"
    );
}

#[test]
fn attention_causal_uses_iota_and_select_with_neg_inf() {
    // Causal attention synthesizes the upper-triangular mask via
    // iota+compare+select with negative-infinity fill.
    let mut g = Graph::new("attn_c");
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
    let m = lower_to_proto(&g);
    let e = entry(&m);
    let codes = opcodes(e);
    for needed in ["iota", "compare", "select", "dot", "exponential"] {
        assert!(
            codes.contains(&needed),
            "causal attention missing `{needed}` (got {:?})",
            codes
        );
    }
}

#[test]
fn reduce_carries_called_computation_id() {
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
    let m = lower_to_proto(&g);
    let e = entry(&m);
    let red = e
        .instructions
        .iter()
        .find(|i| i.opcode == "reduce")
        .expect("Reduce must lower to `reduce`");
    assert_eq!(
        red.called_computation_ids.len(),
        1,
        "reduce must reference exactly one reducer subcomputation"
    );
    let red_comp_id = red.called_computation_ids[0];
    let red_comp = m
        .computations
        .iter()
        .find(|c| c.id == red_comp_id)
        .expect("called reducer subcomputation must exist in module");
    assert!(
        red_comp.instructions.iter().any(|i| i.opcode == "add"),
        "reducer body for ReduceOp::Sum must contain an `add` instruction"
    );
    assert_eq!(
        red.dimensions,
        vec![1],
        "reduce.dimensions must list the axes being reduced"
    );
}

#[test]
fn instruction_ids_are_globally_unique() {
    // Across all computations: every instruction id must be unique.
    let mut g = Graph::new("ids");
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
    let m = lower_to_proto(&g);
    let mut all_ids: Vec<i64> = m
        .computations
        .iter()
        .flat_map(|c| c.instructions.iter().map(|i| i.id))
        .collect();
    all_ids.sort();
    let unique = {
        let mut deduped = all_ids.clone();
        deduped.dedup();
        deduped.len()
    };
    assert_eq!(
        all_ids.len(),
        unique,
        "instruction ids must be globally unique across computations \
                (XLA validator requires this)"
    );
}

#[test]
fn host_program_shape_matches_entry_signature() {
    let mut g = Graph::new("hps");
    let f = DType::F32;
    let s = Shape::new(&[6], f);
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let z = g.binary(BinaryOp::Add, x, y, s);
    g.set_outputs(vec![z]);
    let m = lower_to_proto(&g);
    let ps = m.host_program_shape.as_ref().unwrap();
    assert_eq!(ps.parameters.len(), 2);
    assert_eq!(ps.parameter_names, vec!["x".to_string(), "y".to_string()]);
    let res = ps.result.as_ref().unwrap();
    assert_eq!(res.dimensions, vec![6]);
}

#[test]
fn output_tuple_root_lists_each_output() {
    let mut g = Graph::new("two_outs");
    let s = Shape::new(&[4], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let a = g.binary(BinaryOp::Add, x, y, s.clone());
    let mb = g.binary(BinaryOp::Mul, x, y, s);
    g.set_outputs(vec![a, mb]);
    let m = lower_to_proto(&g);
    let e = entry(&m);
    let root = e.instructions.iter().find(|i| i.id == e.root_id).unwrap();
    assert_eq!(
        root.opcode, "tuple",
        "multi-output entry must root at a `tuple` instruction"
    );
    assert_eq!(
        root.operand_ids.len(),
        2,
        "tuple root must list one operand per declared output"
    );
}
