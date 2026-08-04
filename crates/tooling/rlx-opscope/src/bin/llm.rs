// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-llm` — a small but real decoder-only transformer (causal single-head
//! attention + SwiGLU-ish FFN + lm-head), driven end-to-end through every tool
//! we built:
//!   Tier 3  op FLOPs/bytes → roofline + hot GEMM shapes
//!   struct  dataflow motifs → repeated attention/FFN blocks (fusion candidates)
//!   Tier 1  Softmax tap → causal-attention concentration / sink keys
//!   value   matmul sketches → CSV (then `opscope-mine` / `opscope-plan`)
//!
//! Untrained (random weights) — the structural + causal-attention patterns are
//! architecture-driven, so they show regardless of training.
//!
//! Usage: `opscope-llm [out.csv]`  then  `opscope-mine <out.csv>` / `opscope-plan`

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Philox4x32, Shape};
use rlx_opscope::dataflow::repeated_flow_patterns;
use rlx_opscope::shapes::{DEFAULT_RIDGE, gemm_shape_histogram, op_costs, roofline_class};
use rlx_opscope::{Recorder, StatConfig, inject_infer_stats, inject_matmul_stats};
use rlx_runtime::{Device, Session};

const V: usize = 256; // vocab
const SQ: usize = 32; // seq
const D: usize = 64; // model dim
const DFF: usize = 256; // ffn hidden
const L: usize = 4; // layers

fn sh(d: &[usize]) -> Shape {
    Shape::new(d, DType::F32)
}

fn causal_mask(g: &mut Graph) -> NodeId {
    let mut m = vec![0f32; SQ * SQ];
    for i in 0..SQ {
        for j in 0..SQ {
            if j > i {
                m[i * SQ + j] = -1e9; // block attention to future tokens
            }
        }
    }
    let data: Vec<u8> = m.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.add_node(Op::Constant { data }, vec![], sh(&[SQ, SQ]))
}

fn scalar_const(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        sh(&[1]),
    )
}

fn named_matmul(g: &mut Graph, a: NodeId, b: NodeId, out: Shape, name: &str) -> NodeId {
    let id = g.matmul(a, b, out);
    g.node_mut(id).name = Some(name.into());
    id
}

fn attention(g: &mut Graph, x: NodeId, mask: NodeId, l: usize) -> NodeId {
    let (s, dd) = (sh(&[SQ, D]), sh(&[D, D]));
    let wq = g.param(format!("L{l}.wq"), dd.clone());
    let wk = g.param(format!("L{l}.wk"), dd.clone());
    let wv = g.param(format!("L{l}.wv"), dd.clone());
    let wo = g.param(format!("L{l}.wo"), dd);
    let q = named_matmul(g, x, wq, s.clone(), &format!("L{l}.q"));
    let k = named_matmul(g, x, wk, s.clone(), &format!("L{l}.k"));
    let vv = named_matmul(g, x, wv, s.clone(), &format!("L{l}.v"));
    let kt = g.add_node(Op::Transpose { perm: vec![1, 0] }, vec![k], sh(&[D, SQ]));
    let scores = named_matmul(g, q, kt, sh(&[SQ, SQ]), &format!("L{l}.qk"));
    let scale = scalar_const(g, 1.0 / (D as f32).sqrt());
    let scores = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![scores, scale],
        sh(&[SQ, SQ]),
    );
    let scores = g.add_node(Op::Binary(BinaryOp::Add), vec![scores, mask], sh(&[SQ, SQ]));
    let p = g.softmax(scores, -1, sh(&[SQ, SQ]));
    let ctx = named_matmul(g, p, vv, s.clone(), &format!("L{l}.av"));
    let o = named_matmul(g, ctx, wo, s.clone(), &format!("L{l}.o"));
    g.add_node(Op::Binary(BinaryOp::Add), vec![x, o], s) // residual
}

fn ffn(g: &mut Graph, x: NodeId, l: usize) -> NodeId {
    let s = sh(&[SQ, D]);
    let w1 = g.param(format!("L{l}.f1"), sh(&[D, DFF]));
    let w2 = g.param(format!("L{l}.f2"), sh(&[DFF, D]));
    let h = named_matmul(g, x, w1, sh(&[SQ, DFF]), &format!("L{l}.ffn_up"));
    let h = g.activation(Activation::Silu, h, sh(&[SQ, DFF]));
    let y = named_matmul(g, h, w2, s.clone(), &format!("L{l}.ffn_down"));
    g.add_node(Op::Binary(BinaryOp::Add), vec![x, y], s)
}

fn build_llm() -> Graph {
    let mut g = Graph::new("tiny_llm");
    let mut x = g.input("x", sh(&[SQ, D]));
    let mask = causal_mask(&mut g);
    for l in 0..L {
        x = attention(&mut g, x, mask, l);
        x = ffn(&mut g, x, l);
    }
    let wlm = g.param("lm_head", sh(&[D, V]));
    let logits = named_matmul(&mut g, x, wlm, sh(&[SQ, V]), "lm_head");
    g.set_outputs(vec![logits]);
    g
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len().max(1) as f32
}

fn main() -> std::io::Result<()> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "opscope_llm.csv".into());
    let g = build_llm();
    println!(
        "tiny_llm: {L} layers, seq {SQ}, dim {D}, vocab {V} — {} nodes\n",
        g.nodes().len()
    );

    // ── Tier 3: roofline + hot GEMM shapes ──
    let costs = op_costs(&g);
    let (tf, tb) = (
        costs.iter().map(|c| c.flops).sum::<u64>(),
        costs.iter().map(|c| c.bytes).sum::<u64>(),
    );
    let (mut mem, mut comp) = (0u64, 0u64);
    for c in &costs {
        match roofline_class(c, DEFAULT_RIDGE) {
            "memory-bound" => mem += c.flops,
            "compute-bound" => comp += c.flops,
            _ => {}
        }
    }
    println!(
        "[Tier3] {:.3} GFLOP, {:.2} MB, {:.1} FLOP/byte — {:.0}% memory-bound",
        tf as f64 / 1e9,
        tb as f64 / 1e6,
        tf as f64 / tb.max(1) as f64,
        mem as f64 / (mem + comp).max(1) as f64 * 100.0
    );
    print!("[Tier3] hot GEMM shapes: ");
    for ((m, k, n), (ct, _)) in gemm_shape_histogram(&costs).iter().take(4) {
        print!("{m}×{k}×{n}(×{ct}) ");
    }
    println!("\n");

    // ── Structural: repeated dataflow blocks ──
    let pats = repeated_flow_patterns(&g, 3, 5, 2);
    println!("[struct] top repeated dataflow blocks (fusion candidates):");
    for p in pats.iter().take(3) {
        println!("   ×{} d{}  {}", p.count, p.depth, p.tree);
    }
    println!();

    // ── Inject matmul value sketches + attention/routing sketches, then run ──
    let cfg = StatConfig::default();
    let (g1, mm_specs) = inject_matmul_stats(&g, &cfg);
    let (g2, inf_specs) = inject_infer_stats(&g1, &cfg);
    let mut c = Session::new(Device::Cpu).compile(g2);

    // Random He-scaled weights.
    let mut rng = Philox4x32::new(0x11CE);
    for node in g.nodes() {
        if let Op::Param { name } = &node.op {
            let dims: Vec<usize> = (0..node.shape.rank())
                .map(|i| node.shape.dim(i).unwrap_static())
                .collect();
            let numel: usize = dims.iter().product();
            let mut w = vec![0f32; numel];
            rng.fill_normal(&mut w);
            let scale = (2.0 / dims[0] as f32).sqrt();
            for v in &mut w {
                *v *= scale;
            }
            c.set_param(name, &w);
        }
    }
    // Synthetic input embeddings.
    let mut x = vec![0f32; SQ * D];
    rng.fill_normal(&mut x);
    let outs = c.run(&[("x", &x)]);

    // ── Tier 1: causal-attention concentration per layer ──
    println!("[Tier1] causal attention (per layer): peak mass + where attention lands");
    let mut layer = 0;
    for spec in inf_specs.iter().filter(|s| s.stat == "attn_qmax") {
        let qmax = &outs[spec.out_idx];
        let krecv_spec = inf_specs
            .iter()
            .find(|s| s.stat == "attn_krecv" && s.site == spec.site)
            .unwrap();
        let krecv = &outs[krecv_spec.out_idx];
        // fraction of attention received by the first 25% of keys (early tokens).
        let early = SQ / 4;
        let early_mass: f32 =
            krecv[..early].iter().sum::<f32>() / krecv.iter().sum::<f32>().max(1.0);
        println!(
            "   {:<8} mean peak {:.2}   first {early} keys get {:.0}% of attention (causal skew)",
            spec.site,
            mean(qmax),
            early_mass * 100.0
        );
        layer += 1;
        if layer >= L {
            break;
        }
    }
    println!();

    // ── Record matmul value sketches → CSV for opscope-mine / opscope-plan ──
    let mut rec = Recorder::create(&out)?;
    rec.record(0, 0, "cpu", "llm", SQ, D, V, &mm_specs, &outs)?;
    rec.flush()?;
    println!(
        "[value] wrote {} matmul-site sketches → {out}",
        mm_specs.len()
    );
    println!("        next: opscope-mine {out}   |   opscope-plan {out}");
    Ok(())
}
