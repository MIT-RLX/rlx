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

//! Emit a representative set of HLO modules to /tmp/ and print a
//! manifest (`name\tpath`) on stdout.
//!
//! Driven by the Docker validation harness — Python parses each
//! emitted module via `jax.lib.xla_extension.HloModule.from_serialized_hlo_module_proto`
//! to confirm proto field numbers / dimension order / opcode strings
//! are correct. The set covers each major lowering family at least
//! once so a regression in any one shows up here.

use std::fs;
use std::path::Path;

use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
use rlx_ir::{DType, Graph, Shape};

fn write_module(name: &str, graph: &Graph) -> String {
    let module = rlx_tpu::lower::lower_graph(graph);
    let path = format!("/tmp/hlo_{name}.pb");
    fs::write(&path, &module.bytes).expect("write hlo module bytes");
    println!("{name}\t{path}");
    path
}

fn main() {
    eprintln!("emitting representative HLO modules to /tmp/hlo_*.pb…");

    // ── 1. Trivial element-wise add ──
    {
        let mut g = Graph::new("ew_add");
        let s = Shape::new(&[6], DType::F32);
        let x = g.input("x", s.clone());
        let y = g.input("y", s.clone());
        let z = g.binary(BinaryOp::Add, x, y, s);
        g.set_outputs(vec![z]);
        write_module("ew_add", &g);
    }

    // ── 2. 2-D matmul ──
    {
        let mut g = Graph::new("matmul_2d");
        let f = DType::F32;
        let x = g.input("x", Shape::new(&[4, 3], f));
        let w = g.param("w", Shape::new(&[3, 5], f));
        let y = g.matmul(x, w, Shape::new(&[4, 5], f));
        g.set_outputs(vec![y]);
        write_module("matmul_2d", &g);
    }

    // ── 3. Activations ──
    for (name, act) in [
        ("relu", Activation::Relu),
        ("gelu", Activation::Gelu),
        ("gelu_approx", Activation::GeluApprox),
        ("silu", Activation::Silu),
        ("sigmoid", Activation::Sigmoid),
        ("tanh", Activation::Tanh),
        ("rsqrt", Activation::Rsqrt),
    ] {
        let mut g = Graph::new(format!("act_{name}"));
        let s = Shape::new(&[8], DType::F32);
        let x = g.input("x", s.clone());
        let y = g.activation(act, x, s);
        g.set_outputs(vec![y]);
        write_module(&format!("act_{name}"), &g);
    }

    // ── 4. LayerNorm + RmsNorm ──
    {
        let mut g = Graph::new("layernorm");
        let f = DType::F32;
        let x = g.input("x", Shape::new(&[2, 4], f));
        let gv = g.param("g", Shape::new(&[4], f));
        let bv = g.param("b", Shape::new(&[4], f));
        let y = g.layer_norm(x, gv, bv, -1, 1e-5, Shape::new(&[2, 4], f));
        g.set_outputs(vec![y]);
        write_module("layernorm", &g);
    }
    {
        let mut g = Graph::new("rmsnorm");
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
        write_module("rmsnorm", &g);
    }

    // ── 5. Softmax ──
    {
        let mut g = Graph::new("softmax");
        let f = DType::F32;
        let x = g.input("x", Shape::new(&[2, 5], f));
        let y = g.softmax(x, -1, Shape::new(&[2, 5], f));
        g.set_outputs(vec![y]);
        write_module("softmax", &g);
    }

    // ── 6. Reduce sum / max / mean ──
    for (name, op) in [
        ("reduce_sum", ReduceOp::Sum),
        ("reduce_mean", ReduceOp::Mean),
        ("reduce_max", ReduceOp::Max),
    ] {
        let mut g = Graph::new(name);
        let f = DType::F32;
        let x = g.input("x", Shape::new(&[3, 4], f));
        let y = g.add_node(
            rlx_ir::Op::Reduce {
                op,
                axes: vec![1],
                keep_dim: false,
            },
            vec![x],
            Shape::new(&[3], f),
        );
        g.set_outputs(vec![y]);
        write_module(name, &g);
    }

    // ── 7. Compare + Where ──
    {
        let mut g = Graph::new("compare_where");
        let f = DType::F32;
        let s = Shape::new(&[4], f);
        let a = g.input("a", s.clone());
        let b = g.input("b", s.clone());
        let cond = g.add_node(
            rlx_ir::Op::Compare(CmpOp::Lt),
            vec![a, b],
            Shape::new(&[4], DType::Bool),
        );
        let sel = g.add_node(rlx_ir::Op::Where, vec![cond, a, b], s);
        g.set_outputs(vec![sel]);
        write_module("compare_where", &g);
    }

    // ── 8. Reshape + Transpose + Concat + Slice (Narrow) ──
    {
        let mut g = Graph::new("shape_ops");
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
        write_module("shape_ops", &g);
    }

    // ── 9. Gather (embedding lookup) ──
    {
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
        write_module("gather", &g);
    }

    // ── 10. Attention (causal mask) ──
    {
        let mut g = Graph::new("attention_causal");
        let f = DType::F32;
        // [B=1, H=2, S=4, D=8]
        let q = g.input("q", Shape::new(&[1, 2, 4, 8], f));
        let k = g.input("k", Shape::new(&[1, 2, 4, 8], f));
        let v = g.input("v", Shape::new(&[1, 2, 4, 8], f));
        use rlx_ir::op::MaskKind;
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
        write_module("attention_causal", &g);
    }

    // ── 11. RoPE ──
    {
        let mut g = Graph::new("rope");
        let f = DType::F32;
        let x = g.input("x", Shape::new(&[1, 2, 4, 8], f));
        let cos = g.input("cos", Shape::new(&[4, 4], f));
        let sin = g.input("sin", Shape::new(&[4, 4], f));
        let out = g.add_node(
            rlx_ir::Op::Rope {
                head_dim: 8,
                n_rot: 8,
            },
            vec![x, cos, sin],
            Shape::new(&[1, 2, 4, 8], f),
        );
        g.set_outputs(vec![out]);
        write_module("rope", &g);
    }

    // ── 12a. TopK ──
    {
        let mut g = Graph::new("topk");
        let f = DType::F32;
        let x = g.input("x", Shape::new(&[2, 16], f));
        let y = g.add_node(rlx_ir::Op::TopK { k: 3 }, vec![x], Shape::new(&[2, 3], f));
        g.set_outputs(vec![y]);
        write_module("topk", &g);
    }

    // ── 12b. GroupedMatMul (MoE primitive) ──
    {
        let mut g = Graph::new("grouped_matmul");
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
        write_module("grouped_matmul", &g);
    }

    // ── 12c. DequantMatMul (Int8BlockAsym) ──
    {
        let mut g = Graph::new("dequant_matmul");
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
        write_module("dequant_matmul", &g);
    }

    // ── 12d. QMatMul ──
    {
        let mut g = Graph::new("qmatmul");
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
        write_module("qmatmul", &g);
    }

    // ── 12e. QConv2d ──
    {
        let mut g = Graph::new("qconv2d");
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
        write_module("qconv2d", &g);
    }

    // ── 12f. Sample (greedy + temperature) ──
    {
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
        write_module("sample_greedy", &g);
    }
    {
        let mut g = Graph::new("sample_temp");
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
        write_module("sample_temp", &g);
    }

    // ── 12g. SelectiveScan (Mamba SSM) ──
    {
        let mut g = Graph::new("selective_scan");
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
        write_module("selective_scan", &g);
    }

    // ── 13. BERT-shaped fragment (most composite paths in one go) ──
    {
        let mut g = Graph::new("bert_fragment");
        let f32 = DType::F32;
        let ids = g.input("ids", Shape::new(&[1, 8], f32));
        let table = g.param("emb_table", Shape::new(&[1024, 32], f32));
        let w = g.param("ffn_w", Shape::new(&[32, 32], f32));
        let bv = g.param("ffn_b", Shape::new(&[32], f32));
        let gv = g.param("ln_g", Shape::new(&[32], f32));
        let bv2 = g.param("ln_b", Shape::new(&[32], f32));
        let emb = g.add_node(
            rlx_ir::Op::Gather { axis: 0 },
            vec![table, ids],
            Shape::new(&[1, 8, 32], f32),
        );
        let flat = g.reshape(emb, vec![8, 32], Shape::new(&[8, 32], f32));
        let mm = g.matmul(flat, w, Shape::new(&[8, 32], f32));
        let bb = g.add_node(
            rlx_ir::Op::Expand {
                target_shape: vec![8, 32],
            },
            vec![bv],
            Shape::new(&[8, 32], f32),
        );
        let added = g.binary(BinaryOp::Add, mm, bb, Shape::new(&[8, 32], f32));
        let acted = g.activation(Activation::Gelu, added, Shape::new(&[8, 32], f32));
        let normed = g.layer_norm(acted, gv, bv2, -1, 1e-5, Shape::new(&[8, 32], f32));
        g.set_outputs(vec![normed]);
        write_module("bert_fragment", &g);
    }

    eprintln!(
        "manifest emitted on stdout; {} modules written to /tmp/",
        Path::new("/tmp")
            .read_dir()
            .unwrap()
            .filter(|e| e
                .as_ref()
                .is_ok_and(|x| x.file_name().to_string_lossy().starts_with("hlo_")))
            .count()
    );
}
