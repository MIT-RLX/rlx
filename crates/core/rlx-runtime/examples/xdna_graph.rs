// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// End-to-end: real rlx graphs (activations + softmax) dispatched onto the AMD
// XDNA NPU via `Session::new(Device::Xdna)`. The XdnaBackend emits the AIE-MLIR
// kernel from Rust, compiles it on demand (native aiecc), and runs it — checked
// vs the CPU backend. This is the op library from `rlx-xdna::aie` reachable from
// the graph API.
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. \
//     cargo run -p rlx-runtime --features xdna --example xdna_graph

use rlx_ir::op::{Activation, BinaryOp as IrBin, MaskKind, ReduceOp as IrRed, ScaleMode, SteKind};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn act_graph(act: Activation, n: usize) -> Graph {
    let mut g = Graph::new("act");
    let shape = Shape::new(&[n], DType::F32);
    let x = g.input("x", shape.clone());
    let y = g.activation(act, x, shape);
    g.set_outputs(vec![y]);
    g
}

fn softmax_graph(rows: usize, cols: usize) -> Graph {
    let mut g = Graph::new("softmax");
    let shape = Shape::new(&[rows, cols], DType::F32);
    let x = g.input("x", shape.clone());
    let y = g.softmax(x, -1, shape);
    g.set_outputs(vec![y]);
    g
}

fn rmsnorm_graph(rows: usize, cols: usize, eps: f32) -> Graph {
    let mut g = Graph::new("rmsnorm");
    let shape = Shape::new(&[rows, cols], DType::F32);
    let x = g.input("x", shape.clone());
    let gamma = g.param("gamma", Shape::new(&[cols], DType::F32));
    let beta = g.param("beta", Shape::new(&[cols], DType::F32));
    let y = g.add_node(Op::RmsNorm { axis: -1, eps }, vec![x, gamma, beta], shape);
    g.set_outputs(vec![y]);
    g
}

fn layernorm_graph(rows: usize, cols: usize, eps: f32) -> Graph {
    let mut g = Graph::new("layernorm");
    let shape = Shape::new(&[rows, cols], DType::F32);
    let x = g.input("x", shape.clone());
    let gamma = g.param("gamma", Shape::new(&[cols], DType::F32));
    let beta = g.param("beta", Shape::new(&[cols], DType::F32));
    let y = g.layer_norm(x, gamma, beta, -1, eps, shape);
    g.set_outputs(vec![y]);
    g
}

fn attention_graph(seq: usize, d: usize) -> Graph {
    let mut g = Graph::new("attn");
    let shape = Shape::new(&[1, seq, d], DType::F32); // rank-3 [B,S,D] single-head
    let q = g.input("q", shape.clone());
    let k = g.input("k", shape.clone());
    let v = g.input("v", shape.clone());
    let y = g.add_node(
        Op::Attention {
            num_heads: 1,
            head_dim: d,
            v_head_dim: None,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        shape,
    );
    g.set_outputs(vec![y]);
    g
}

fn binary_graph(op: IrBin, n: usize) -> Graph {
    let mut g = Graph::new("bin");
    let shape = Shape::new(&[n], DType::F32);
    let a = g.input("a", shape.clone());
    let b = g.input("b", shape.clone());
    let y = g.add_node(Op::Binary(op), vec![a, b], shape);
    g.set_outputs(vec![y]);
    g
}

fn reduce_graph(op: IrRed, rows: usize, cols: usize) -> Graph {
    let mut g = Graph::new("reduce");
    let x = g.input("x", Shape::new(&[rows, cols], DType::F32));
    let y = g.add_node(
        Op::Reduce {
            op,
            axes: vec![1],
            keep_dim: false,
        },
        vec![x],
        Shape::new(&[rows], DType::F32),
    );
    g.set_outputs(vec![y]);
    g
}

fn cmp(name: &str, cpu: &[f32], npu: &[f32]) -> bool {
    let maxrel = cpu
        .iter()
        .zip(npu)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-3))
        .fold(0.0f32, f32::max);
    let ok = !maxrel.is_nan() && maxrel < 3e-3;
    println!(
        "  {name:<10} {}  max-rel-err {maxrel:.2e}",
        if ok { "PASS ✓" } else { "FAIL ✗" }
    );
    ok
}

fn main() {
    let n = 1024;
    let x: Vec<f32> = (0..n).map(|i| ((i % 37) as f32 - 18.0) * 0.2).collect();

    println!("rlx graph → Device::Xdna (NPU)\n");
    let mut all = true;

    // NOTE: the NPU gelu is the tanh approximation, so compare against
    // `GeluApprox` (exact-erf `Gelu` deviates ~1% and is a documented mapping).
    for act in [
        Activation::Relu,
        Activation::GeluApprox,
        Activation::Sigmoid,
        Activation::Silu,
        Activation::Tanh,
        Activation::Exp,
    ] {
        let g = act_graph(act, n);
        let cpu = Session::new(Device::Cpu)
            .compile(g.clone())
            .run(&[("x", &x)])[0]
            .clone();
        let npu = Session::new(Device::Xdna).compile(g).run(&[("x", &x)])[0].clone();
        all &= cmp(&format!("{act:?}"), &cpu, &npu);
    }

    // softmax over [32, 32]
    let (rows, cols) = (32, 32);
    let sx: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.3)
        .collect();
    let sg = softmax_graph(rows, cols);
    let cpu = Session::new(Device::Cpu)
        .compile(sg.clone())
        .run(&[("x", &sx)])[0]
        .clone();
    let npu = Session::new(Device::Xdna).compile(sg).run(&[("x", &sx)])[0].clone();
    all &= cmp("softmax", &cpu, &npu);

    // affine norms over [128, 64] with gamma/beta params — large enough that the
    // norm kernels ROW-STREAM (tile_rows=16, 8 chunks; 32 KB > single tile).
    let (nr, nc) = (128, 64);
    let nx: Vec<f32> = (0..nr * nc)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.2 + 0.5)
        .collect();
    let gamma: Vec<f32> = (0..nc).map(|c| 0.5 + (c % 5) as f32 * 0.1).collect();
    let beta: Vec<f32> = (0..nc).map(|c| ((c % 7) as f32 - 3.0) * 0.05).collect();
    let run_norm = |dev: Device, g: Graph, bt: &[f32]| -> Vec<f32> {
        let mut c = Session::new(dev).compile(g);
        c.set_param("gamma", &gamma);
        c.set_param("beta", bt);
        c.run(&[("x", &nx)])[0].clone()
    };
    // RMSNorm with beta=0 (classic); LayerNorm with a real bias.
    let bz = vec![0.0; nc];
    let g = rmsnorm_graph(nr, nc, 1e-5);
    all &= cmp(
        "rms_norm",
        &run_norm(Device::Cpu, g.clone(), &bz),
        &run_norm(Device::Xdna, g, &bz),
    );
    let g = layernorm_graph(nr, nc, 1e-5);
    all &= cmp(
        "layer_norm",
        &run_norm(Device::Cpu, g.clone(), &beta),
        &run_norm(Device::Xdna, g, &beta),
    );

    // fused attention [1, 32, 32]
    let (aseq, ad) = (32, 32);
    let asd = aseq * ad;
    let aq: Vec<f32> = (0..asd).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let ak: Vec<f32> = (0..asd).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    let av: Vec<f32> = (0..asd).map(|i| ((i % 7) as f32 - 3.0) * 0.15).collect();
    let ins: &[(&str, &[f32])] = &[("q", &aq), ("k", &ak), ("v", &av)];
    let ag = attention_graph(aseq, ad);
    let acpu = Session::new(Device::Cpu).compile(ag.clone()).run(ins)[0].clone();
    let anpu = Session::new(Device::Xdna).compile(ag).run(ins)[0].clone();
    all &= cmp("attention", &acpu, &anpu);

    // binary (f32 arithmetic) — a ⊙ b over [1024]
    let bn = 1024;
    let ba: Vec<f32> = (0..bn).map(|i| ((i % 17) as f32 - 8.0) * 0.3).collect();
    let bb: Vec<f32> = (0..bn).map(|i| ((i % 5) as f32 + 1.0) * 0.5).collect();
    let bins: &[(&str, &[f32])] = &[("a", &ba), ("b", &bb)];
    // include non-commutative sub/div — these expose any a/b arg-order swap that
    // commutative mul/add/max would hide.
    for (nm, op) in [
        ("mul", IrBin::Mul),
        ("add", IrBin::Add),
        ("max", IrBin::Max),
        ("sub", IrBin::Sub),
        ("div", IrBin::Div),
    ] {
        let g = binary_graph(op, bn);
        let c = Session::new(Device::Cpu).compile(g.clone()).run(bins)[0].clone();
        let np = Session::new(Device::Xdna).compile(g).run(bins)[0].clone();
        all &= cmp(&format!("binary·{nm}"), &c, &np);
    }

    // reduce over the last axis of [32, 48]
    let (rr, rcx) = (32, 48);
    let rx: Vec<f32> = (0..rr * rcx)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.2)
        .collect();
    let rins: &[(&str, &[f32])] = &[("x", &rx)];
    for (nm, op) in [
        ("sum", IrRed::Sum),
        ("max", IrRed::Max),
        ("mean", IrRed::Mean),
    ] {
        let g = reduce_graph(op, rr, rcx);
        let c = Session::new(Device::Cpu).compile(g.clone()).run(rins)[0].clone();
        let np = Session::new(Device::Xdna).compile(g).run(rins)[0].clone();
        all &= cmp(&format!("reduce·{nm}"), &c, &np);
    }

    // data-movement (single-input): clamp / transpose / narrow / reverse
    let dm_in: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.2).collect();
    let dm1 = |name: &str, op: Op, ins: &[usize], outs: &[usize]| -> bool {
        let mut g = Graph::new(name);
        let x = g.input("x", Shape::new(ins, DType::F32));
        let y = g.add_node(op, vec![x], Shape::new(outs, DType::F32));
        g.set_outputs(vec![y]);
        let c = Session::new(Device::Cpu)
            .compile(g.clone())
            .run(&[("x", &dm_in)])[0]
            .clone();
        let np = Session::new(Device::Xdna).compile(g).run(&[("x", &dm_in)])[0].clone();
        cmp(name, &c, &np)
    };
    all &= dm1(
        "clamp",
        Op::Clamp {
            min: -1.0,
            max: 1.0,
        },
        &[24],
        &[24],
    );
    all &= dm1(
        "transpose",
        Op::Transpose { perm: vec![1, 0] },
        &[4, 6],
        &[6, 4],
    );
    all &= dm1(
        "narrow",
        Op::Narrow {
            axis: 1,
            start: 1,
            len: 3,
        },
        &[4, 6],
        &[4, 3],
    );
    all &= dm1("reverse", Op::Reverse { axes: vec![1] }, &[4, 6], &[4, 6]);

    // concat (2-input) [2,2] ‖ [2,3] on axis 1 → [2,5]
    let (ca, cb) = (
        (0..4).map(|i| i as f32).collect::<Vec<_>>(),
        (0..6).map(|i| 10.0 + i as f32).collect::<Vec<_>>(),
    );
    let mut g = Graph::new("concat");
    let a = g.input("a", Shape::new(&[2, 2], DType::F32));
    let b = g.input("b", Shape::new(&[2, 3], DType::F32));
    let y = g.add_node(
        Op::Concat { axis: 1 },
        vec![a, b],
        Shape::new(&[2, 5], DType::F32),
    );
    g.set_outputs(vec![y]);
    let cins: &[(&str, &[f32])] = &[("a", &ca), ("b", &cb)];
    let c = Session::new(Device::Cpu).compile(g.clone()).run(cins)[0].clone();
    let np = Session::new(Device::Xdna).compile(g).run(cins)[0].clone();
    all &= cmp("concat", &c, &np);

    // stop-gradient (identity) + pad (grow axis 1 by [1,2] → [4,9], fill 0)
    all &= dm1("stopgrad", Op::StopGradient, &[24], &[24]);
    all &= dm1(
        "pad",
        Op::Pad {
            pads: vec![[0, 0], [1, 2]],
            mode: rlx_ir::op::PadMode::Constant(0.0),
        },
        &[4, 6],
        &[4, 9],
    );

    // cumulative scans over the last axis of [4,6]
    all &= dm1(
        "cumsum",
        Op::Cumsum {
            axis: -1,
            exclusive: false,
        },
        &[4, 6],
        &[4, 6],
    );
    all &= dm1(
        "cumprod",
        Op::CumProd {
            axis: -1,
            exclusive: false,
        },
        &[4, 6],
        &[4, 6],
    );
    all &= dm1(
        "cummax",
        Op::CumMax {
            axis: -1,
            exclusive: false,
        },
        &[4, 6],
        &[4, 6],
    );

    // multi-input elementwise: compare (2-in, bool out), where / fma (3-in, f32)
    let mk = |name: &str, op: Op, ins: &[(&str, &[f32])], sh: &[usize], out_dt: DType| -> bool {
        let mut g = Graph::new(name);
        let ids: Vec<_> = ins
            .iter()
            .map(|(nm, _)| g.input(*nm, Shape::new(sh, DType::F32)))
            .collect();
        let y = g.add_node(op, ids, Shape::new(sh, out_dt));
        g.set_outputs(vec![y]);
        let c = Session::new(Device::Cpu).compile(g.clone()).run(ins)[0].clone();
        let np = Session::new(Device::Xdna).compile(g).run(ins)[0].clone();
        cmp(name, &c, &np)
    };
    let (e0, e1): (Vec<f32>, Vec<f32>) = (
        (0..32).map(|i| (i as f32 - 16.0) * 0.25).collect(),
        (0..32).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect(),
    );
    let e2: Vec<f32> = (0..32).map(|i| (i % 3) as f32).collect();
    all &= mk(
        "where",
        Op::Where,
        &[("cond", &e2), ("a", &e0), ("b", &e1)],
        &[32],
        DType::F32,
    );
    all &= mk(
        "fma",
        Op::Fma,
        &[("a", &e0), ("b", &e1), ("c", &e2)],
        &[32],
        DType::F32,
    );
    // compare → bool output (byte-packed by the exec, matches CPU)
    all &= mk(
        "compare·lt",
        Op::Compare(rlx_ir::op::CmpOp::Lt),
        &[("a", &e0), ("b", &e1)],
        &[32],
        DType::Bool,
    );
    all &= mk(
        "compare·ge",
        Op::Compare(rlx_ir::op::CmpOp::Ge),
        &[("a", &e0), ("b", &e1)],
        &[32],
        DType::Bool,
    );

    // cast f32→i32 (numeric convert; the i32 output rides as bits in the f32 cell,
    // so compare the decoded i32 values rather than the f32 reinterpretation)
    let mut gc = Graph::new("cast_f2i");
    let xc = gc.input("x", Shape::new(&[24], DType::F32));
    let yc = gc.add_node(
        Op::Cast { to: DType::I32 },
        vec![xc],
        Shape::new(&[24], DType::I32),
    );
    gc.set_outputs(vec![yc]);
    let cc = Session::new(Device::Cpu)
        .compile(gc.clone())
        .run(&[("x", &dm_in)])[0]
        .clone();
    let nc = Session::new(Device::Xdna).compile(gc).run(&[("x", &dm_in)])[0].clone();
    let cast_ok =
        cc.len() == nc.len() && cc.iter().zip(&nc).all(|(a, b)| a.to_bits() == b.to_bits());
    println!(
        "  {:<10} {}",
        "cast_f2i",
        if cast_ok { "PASS ✓" } else { "FAIL ✗" }
    );
    all &= cast_ok;

    // argmax/argmin over last axis of [4,6] → [4] indices (f32-encoded value)
    let am_in: Vec<f32> = (0..24)
        .map(|i| (((i * 7 + 3) % 11) as f32 - 5.0) * 0.3)
        .collect();
    for (nm, mk_op) in [("argmax", 0), ("argmin", 1)] {
        let mut ga = Graph::new(nm);
        let xa = ga.input("x", Shape::new(&[4, 6], DType::F32));
        let op = if mk_op == 0 {
            Op::ArgMax {
                axis: 1,
                keep_dim: false,
            }
        } else {
            Op::ArgMin {
                axis: 1,
                keep_dim: false,
            }
        };
        let ya = ga.add_node(op, vec![xa], Shape::new(&[4], DType::F32));
        ga.set_outputs(vec![ya]);
        let ca = Session::new(Device::Cpu)
            .compile(ga.clone())
            .run(&[("x", &am_in)])[0]
            .clone();
        let na = Session::new(Device::Xdna).compile(ga).run(&[("x", &am_in)])[0].clone();
        all &= cmp(nm, &ca, &na);
    }

    // group norm on [1,4,2,2], num_groups=2 (cg=2, spatial=4) with per-channel affine
    let gn_x: Vec<f32> = (0..16)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.3 + 0.5)
        .collect();
    let gn_gamma: Vec<f32> = (0..4).map(|c| 0.5 + (c % 3) as f32 * 0.2).collect();
    let gn_beta: Vec<f32> = (0..4).map(|c| ((c % 2) as f32 - 0.5) * 0.1).collect();
    let mut gg = Graph::new("groupnorm");
    let gx = gg.input("x", Shape::new(&[1, 4, 2, 2], DType::F32));
    let ggm = gg.param("gamma", Shape::new(&[4], DType::F32));
    let gbt = gg.param("beta", Shape::new(&[4], DType::F32));
    let gy = gg.add_node(
        Op::GroupNorm {
            num_groups: 2,
            eps: 1e-5,
        },
        vec![gx, ggm, gbt],
        Shape::new(&[1, 4, 2, 2], DType::F32),
    );
    gg.set_outputs(vec![gy]);
    let run_gn = |dev: Device| -> Vec<f32> {
        let mut c = Session::new(dev).compile(gg.clone());
        c.set_param("gamma", &gn_gamma);
        c.set_param("beta", &gn_beta);
        c.run(&[("x", &gn_x)])[0].clone()
    };
    all &= cmp("groupnorm", &run_gn(Device::Cpu), &run_gn(Device::Xdna));

    // RoPE (NeoX) on x[1,2,4] (seq=2, nh=1, head_dim=4, full rotary); cos/sin [2,2]
    let rp_x: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * 0.3).collect();
    let rp_cos: Vec<f32> = (0..4).map(|i| 0.5 + i as f32 * 0.1).collect();
    let rp_sin: Vec<f32> = (0..4).map(|i| 0.2 + i as f32 * 0.1).collect();
    let rins: &[(&str, &[f32])] = &[("x", &rp_x), ("cos", &rp_cos), ("sin", &rp_sin)];
    for (nm, style) in [
        ("rope·neox", rlx_ir::op::RopeStyle::NeoX),
        ("rope·gptj", rlx_ir::op::RopeStyle::GptJ),
    ] {
        let mut gr = Graph::new(nm);
        let rx = gr.input("x", Shape::new(&[1, 2, 4], DType::F32));
        let rc = gr.input("cos", Shape::new(&[2, 2], DType::F32));
        let rsn = gr.input("sin", Shape::new(&[2, 2], DType::F32));
        let ry = gr.add_node(
            Op::Rope {
                head_dim: 4,
                n_rot: 4,
                style,
            },
            vec![rx, rc, rsn],
            Shape::new(&[1, 2, 4], DType::F32),
        );
        gr.set_outputs(vec![ry]);
        let crp = Session::new(Device::Cpu).compile(gr.clone()).run(rins)[0].clone();
        let nrp = Session::new(Device::Xdna).compile(gr).run(rins)[0].clone();
        all &= cmp(nm, &crp, &nrp);
    }

    // MULTI-OP subgraph: relu(a + b) — two ops, one graph, chained on the NPU
    let ca_x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let ca_b: Vec<f32> = (0..64).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
    let mut gch = Graph::new("chain");
    let sh64 = Shape::new(&[64], DType::F32);
    let cax = gch.input("a", sh64.clone());
    let cbx = gch.input("b", sh64.clone());
    let cadd = gch.add_node(Op::Binary(IrBin::Add), vec![cax, cbx], sh64.clone());
    let crelu = gch.activation(Activation::Relu, cadd, sh64.clone());
    gch.set_outputs(vec![crelu]);
    let chins: &[(&str, &[f32])] = &[("a", &ca_x), ("b", &ca_b)];
    let cc = Session::new(Device::Cpu).compile(gch.clone()).run(chins)[0].clone();
    let nch = Session::new(Device::Xdna).compile(gch).run(chins)[0].clone();
    all &= cmp("chain·relu(a+b)", &cc, &nch);

    // 3-op chain relu((a+b)*a) — mixed intermediate + graph-input operand
    let mut g3 = Graph::new("chain3");
    let a3 = g3.input("a", sh64.clone());
    let b3 = g3.input("b", sh64.clone());
    let s3 = g3.add_node(Op::Binary(IrBin::Add), vec![a3, b3], sh64.clone());
    let m3 = g3.add_node(Op::Binary(IrBin::Mul), vec![s3, a3], sh64.clone());
    let r3 = g3.activation(Activation::Relu, m3, sh64.clone());
    g3.set_outputs(vec![r3]);
    let c3 = Session::new(Device::Cpu).compile(g3.clone()).run(chins)[0].clone();
    let n3 = Session::new(Device::Xdna).compile(g3).run(chins)[0].clone();
    all &= cmp("chain·relu((a+b)*a)", &c3, &n3);

    // MATMUL-IN-CHAIN: relu(x @ W) — a Linear+activation (MLP) layer. The matmul is
    // int8-quantized so it's not bit-exact vs f32 CPU → check cosine similarity.
    let (mm, mk, mnn) = (64usize, 64usize, 64usize);
    let mlp_x: Vec<f32> = (0..mm * mk)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let mlp_w: Vec<f32> = (0..mk * mnn)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let mut gm = Graph::new("mlp");
    let mxi = gm.input("x", Shape::new(&[mm, mk], DType::F32));
    let mwi = gm.param("W", Shape::new(&[mk, mnn], DType::F32));
    let mmm = gm.matmul(mxi, mwi, Shape::new(&[mm, mnn], DType::F32));
    let mrl = gm.activation(Activation::Relu, mmm, Shape::new(&[mm, mnn], DType::F32));
    gm.set_outputs(vec![mrl]);
    let run_mlp = |dev: Device| -> Vec<f32> {
        let mut c = Session::new(dev).compile(gm.clone());
        c.set_param("W", &mlp_w);
        c.run(&[("x", &mlp_x)])[0].clone()
    };
    let (cm, nm) = (run_mlp(Device::Cpu), run_mlp(Device::Xdna));
    let dot: f32 = cm.iter().zip(&nm).map(|(a, b)| a * b).sum();
    let (na2, nb2) = (
        cm.iter().map(|a| a * a).sum::<f32>().sqrt(),
        nm.iter().map(|b| b * b).sum::<f32>().sqrt(),
    );
    let cos = dot / (na2 * nb2).max(1e-6);
    let mlp_ok = cos > 0.99;
    println!(
        "  {:<20} {}  cos {:.4}",
        "chain·relu(x@W)",
        if mlp_ok { "PASS ✓" } else { "FAIL ✗" },
        cos
    );
    all &= mlp_ok;

    // FULL transformer-ish MLP sub-block: out = x + relu(rms_norm(x) @ W)
    // — RMSNorm(+params) → int8 matmul → relu → residual add (x reused). One graph.
    let d = 32usize;
    let blk_x: Vec<f32> = (0..d * d).map(|i| ((i % 11) as f32 - 5.0) * 0.15).collect();
    let blk_w: Vec<f32> = (0..d * d).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let blk_g: Vec<f32> = (0..d).map(|c| 0.8 + (c % 3) as f32 * 0.1).collect();
    let blk_b = vec![0.0f32; d];
    let shdd = Shape::new(&[d, d], DType::F32);
    let mut gb = Graph::new("mlp_block");
    let bx = gb.input("x", shdd.clone());
    let bg = gb.param("g", Shape::new(&[d], DType::F32));
    let bbt = gb.param("bt", Shape::new(&[d], DType::F32));
    let bw = gb.param("W", shdd.clone());
    let bn = gb.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-5,
        },
        vec![bx, bg, bbt],
        shdd.clone(),
    );
    let bl = gb.matmul(bn, bw, shdd.clone());
    let ba = gb.activation(Activation::Relu, bl, shdd.clone());
    let bo = gb.add_node(Op::Binary(IrBin::Add), vec![bx, ba], shdd.clone());
    gb.set_outputs(vec![bo]);
    let run_blk = |dev: Device| -> Vec<f32> {
        let mut c = Session::new(dev).compile(gb.clone());
        c.set_param("g", &blk_g);
        c.set_param("bt", &blk_b);
        c.set_param("W", &blk_w);
        c.run(&[("x", &blk_x)])[0].clone()
    };
    let (cb, nb) = (run_blk(Device::Cpu), run_blk(Device::Xdna));
    let bdot: f32 = cb.iter().zip(&nb).map(|(a, b)| a * b).sum();
    let bcos = bdot
        / (cb.iter().map(|a| a * a).sum::<f32>().sqrt()
            * nb.iter().map(|b| b * b).sum::<f32>().sqrt())
        .max(1e-6);
    let blk_ok = bcos > 0.99;
    println!(
        "  {:<20} {}  cos {:.4}  (RMSNorm→matmul→relu→residual)",
        "block·mlp",
        if blk_ok { "PASS ✓" } else { "FAIL ✗" },
        bcos
    );
    all &= blk_ok;

    // attention sub-block: out = x + attention(x, x, x) — self-attn + residual (f32,
    // so bit-close). Validates Op::Attention in a chain + a residual around it.
    let (aseq2, ad2) = (8usize, 32usize);
    let ab_x: Vec<f32> = (0..aseq2 * ad2)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let shad = Shape::new(&[1, aseq2, ad2], DType::F32);
    let mut gab = Graph::new("attn_block");
    let abx = gab.input("x", shad.clone());
    let abatt = gab.add_node(
        Op::Attention {
            num_heads: 1,
            head_dim: ad2,
            v_head_dim: None,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![abx, abx, abx],
        shad.clone(),
    );
    let abo = gab.add_node(Op::Binary(IrBin::Add), vec![abx, abatt], shad.clone());
    gab.set_outputs(vec![abo]);
    let abins: &[(&str, &[f32])] = &[("x", &ab_x)];
    let cab = Session::new(Device::Cpu).compile(gab.clone()).run(abins)[0].clone();
    let nab = Session::new(Device::Xdna).compile(gab).run(abins)[0].clone();
    all &= cmp("block·attn+resid", &cab, &nab);

    // CAUSAL attention (decoder mask): keys j>i masked to −∞ before softmax
    let mut gca = Graph::new("attn_causal");
    let cax2 = gca.input("x", shad.clone());
    let caatt = gca.add_node(
        Op::Attention {
            num_heads: 1,
            head_dim: ad2,
            v_head_dim: None,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![cax2, cax2, cax2],
        shad.clone(),
    );
    gca.set_outputs(vec![caatt]);
    let cca = Session::new(Device::Cpu).compile(gca.clone()).run(abins)[0].clone();
    let nca = Session::new(Device::Xdna).compile(gca).run(abins)[0].clone();
    all &= cmp("attn·causal", &cca, &nca);

    // MULTI-HEAD attention: 2 heads × head_dim 16 (hidden 32), causal — real decoder
    let mut gmh = Graph::new("attn_mh");
    let mhx = gmh.input("x", shad.clone()); // [1, 8, 32]
    let mhatt = gmh.add_node(
        Op::Attention {
            num_heads: 2,
            head_dim: 16,
            v_head_dim: None,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![mhx, mhx, mhx],
        shad.clone(),
    );
    gmh.set_outputs(vec![mhatt]);
    let cmh = Session::new(Device::Cpu).compile(gmh.clone()).run(abins)[0].clone();
    let nmh = Session::new(Device::Xdna).compile(gmh).run(abins)[0].clone();
    all &= cmp("attn·mh(2)·causal", &cmh, &nmh);

    // FULL DECODER LAYER in one graph: r1 = x + attn(rms_norm(x));
    //   out = r1 + relu(rms_norm(r1) @ W). norm→attn→residual→norm→mlp→residual.
    let (dseq, dd2) = (8usize, 32usize);
    let dl_x: Vec<f32> = (0..dseq * dd2)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.08)
        .collect();
    let dl_w: Vec<f32> = (0..dd2 * dd2)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();
    let (dl_g, dl_b) = (vec![1.0f32; dd2], vec![0.0f32; dd2]);
    let shsd = Shape::new(&[dseq, dd2], DType::F32);
    let mut gd = Graph::new("decoder");
    let dx = gd.input("x", shsd.clone());
    let (dg1, db1) = (
        gd.param("g1", Shape::new(&[dd2], DType::F32)),
        gd.param("b1", Shape::new(&[dd2], DType::F32)),
    );
    let (dg2, db2) = (
        gd.param("g2", Shape::new(&[dd2], DType::F32)),
        gd.param("b2", Shape::new(&[dd2], DType::F32)),
    );
    let dw = gd.param("W", Shape::new(&[dd2, dd2], DType::F32));
    let dn1 = gd.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-5,
        },
        vec![dx, dg1, db1],
        shsd.clone(),
    );
    let datt = gd.add_node(
        Op::Attention {
            num_heads: 1,
            head_dim: dd2,
            v_head_dim: None,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![dn1, dn1, dn1],
        shsd.clone(),
    );
    let dr1 = gd.add_node(Op::Binary(IrBin::Add), vec![dx, datt], shsd.clone());
    let dn2 = gd.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-5,
        },
        vec![dr1, dg2, db2],
        shsd.clone(),
    );
    let dlin = gd.matmul(dn2, dw, shsd.clone());
    let dact = gd.activation(Activation::Relu, dlin, shsd.clone());
    let ddown = gd.add_node(Op::Binary(IrBin::Add), vec![dr1, dact], shsd.clone());
    gd.set_outputs(vec![ddown]);
    let run_dec = |dev: Device| -> Vec<f32> {
        let mut c = Session::new(dev).compile(gd.clone());
        for (nm, v) in [
            ("g1", &dl_g),
            ("b1", &dl_b),
            ("g2", &dl_g),
            ("b2", &dl_b),
            ("W", &dl_w),
        ] {
            c.set_param(nm, v);
        }
        c.run(&[("x", &dl_x)])[0].clone()
    };
    let (cd, nd) = (run_dec(Device::Cpu), run_dec(Device::Xdna));
    let ddot: f32 = cd.iter().zip(&nd).map(|(a, b)| a * b).sum();
    let dcos = ddot
        / (cd.iter().map(|a| a * a).sum::<f32>().sqrt()
            * nd.iter().map(|b| b * b).sum::<f32>().sqrt())
        .max(1e-6);
    let dec_ok = dcos > 0.99;
    println!(
        "  {:<20} {}  cos {:.4}  (norm→attn→resid→norm→mlp→resid, 7-op)",
        "layer·decoder",
        if dec_ok { "PASS ✓" } else { "FAIL ✗" },
        dcos
    );
    all &= dec_ok;

    // CONSTANT-INPUT THREADING: baked `Op::Constant` operands (bias/scale) threaded
    // through the graph like Inputs. (a) single op x + c — depth-1 graph that must
    // route through the chain to seed the constant; (b) relu(x * s + b) — a constant
    // affine inside a deeper chain.
    let cst_x: Vec<f32> = (0..64).map(|i| ((i % 9) as f32 - 4.0) * 0.2).collect();
    let cst_c: Vec<f32> = (0..64).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();
    let cst_bytes: Vec<u8> = cst_c.iter().flat_map(|v| v.to_le_bytes()).collect();
    let cst_ins: &[(&str, &[f32])] = &[("x", &cst_x)];

    let mut gc1 = Graph::new("const_add");
    let c1x = gc1.input("x", sh64.clone());
    let c1c = gc1.add_node(
        Op::Constant {
            data: cst_bytes.clone(),
        },
        vec![],
        sh64.clone(),
    );
    let c1y = gc1.add_node(Op::Binary(IrBin::Add), vec![c1x, c1c], sh64.clone());
    gc1.set_outputs(vec![c1y]);
    let cc1 = Session::new(Device::Cpu).compile(gc1.clone()).run(cst_ins)[0].clone();
    let nc1 = Session::new(Device::Xdna).compile(gc1).run(cst_ins)[0].clone();
    all &= cmp("const·add", &cc1, &nc1);

    let mut gc2 = Graph::new("const_affine");
    let c2x = gc2.input("x", sh64.clone());
    let c2s = gc2.add_node(
        Op::Constant {
            data: cst_bytes.clone(),
        },
        vec![],
        sh64.clone(),
    );
    let c2b = gc2.add_node(
        Op::Constant {
            data: cst_bytes.clone(),
        },
        vec![],
        sh64.clone(),
    );
    let c2m = gc2.add_node(Op::Binary(IrBin::Mul), vec![c2x, c2s], sh64.clone());
    let c2a = gc2.add_node(Op::Binary(IrBin::Add), vec![c2m, c2b], sh64.clone());
    let c2r = gc2.activation(Activation::Relu, c2a, sh64.clone());
    gc2.set_outputs(vec![c2r]);
    let cc2 = Session::new(Device::Cpu).compile(gc2.clone()).run(cst_ins)[0].clone();
    let nc2 = Session::new(Device::Xdna).compile(gc2).run(cst_ins)[0].clone();
    all &= cmp("const·relu(x*s+b)", &cc2, &nc2);

    // FAKE-QUANTIZE (QAT, PerBatch): NOT a native NPU op — the compiler's
    // LowerFakeQuantize decomposes it to abs→reduce-max→(Constant inv/eps)→mul/max→
    // expand→div→round→clamp→mul, ALL of which the NPU runs (and it leans on the
    // freshly-threaded Op::Constant). axis=0 ⇒ per-channel max-abs reduces the LAST
    // axis (the NPU's supported reduce). f32→f32, bit-close vs the CPU native thunk.
    let fq_x: Vec<f32> = (0..64).map(|i| ((i % 17) as f32 - 8.0) * 0.11).collect();
    let fqsh = Shape::new(&[8, 8], DType::F32);
    let mut gfq = Graph::new("fake_quant");
    let fqx = gfq.input("x", fqsh.clone());
    let fqy = gfq.add_node(
        Op::FakeQuantize {
            bits: 8,
            axis: Some(0),
            ste: SteKind::Identity,
            scale_mode: ScaleMode::PerBatch,
        },
        vec![fqx],
        fqsh.clone(),
    );
    gfq.set_outputs(vec![fqy]);
    let fqins: &[(&str, &[f32])] = &[("x", &fq_x)];
    let cfq = Session::new(Device::Cpu).compile(gfq.clone()).run(fqins)[0].clone();
    let nfq = Session::new(Device::Xdna).compile(gfq).run(fqins)[0].clone();
    // CPU native FakeQuantize rounds half-away-from-zero; the NPU runs the compiler's
    // decompose oracle (Activation::Round = half-to-even), so exact .5 ties can flip
    // by ±1 quantum — validate via cosine (like the int8-matmul path) + assert every
    // deviation is at most one quantum (proves it's tie-rounding, not a real error).
    let fqdot: f32 = cfq.iter().zip(&nfq).map(|(a, b)| a * b).sum();
    let fqcos = fqdot
        / (cfq.iter().map(|a| a * a).sum::<f32>().sqrt()
            * nfq.iter().map(|b| b * b).sum::<f32>().sqrt())
        .max(1e-6);
    let quantum = fq_x.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
    let max_dev = cfq
        .iter()
        .zip(&nfq)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let fq_ok = fqcos > 0.999 && max_dev <= quantum * 1.001;
    println!(
        "  {:<20} {}  cos {:.4}  dev {:.2}q  (decompose: abs→reduce→const→…→round→clamp)",
        "fakequant·i8",
        if fq_ok { "PASS ✓" } else { "FAIL ✗" },
        fqcos,
        max_dev / quantum
    );
    all &= fq_ok;

    // QUANTIZE (f32 → packed i8, per-tensor): dtype-boundary op. Compare the DECODED
    // i8 codes, not raw cells: the CPU's raw-i8 Session output over-reads the packed
    // tensor into adjacent arena bytes (32 cells, only the first ⌈n/4⌉ real), whereas
    // the NPU exec returns exactly the ⌈n/4⌉ packed cells — so the meaningful check is
    // that the first `len` i8 codes match (they do, bit-for-bit).
    let q_x: Vec<f32> = (0..32).map(|i| ((i % 13) as f32 - 6.0) * 0.2).collect();
    let mut gq = Graph::new("quant");
    let qxi = gq.input("x", Shape::new(&[32], DType::F32));
    let qyi = gq.add_node(
        Op::Quantize {
            axis: None,
            scales: vec![0.05],
            zero_points: vec![0],
        },
        vec![qxi],
        Shape::new(&[32], DType::I8),
    );
    gq.set_outputs(vec![qyi]);
    let qins: &[(&str, &[f32])] = &[("x", &q_x)];
    let cq = Session::new(Device::Cpu).compile(gq.clone()).run(qins)[0].clone();
    let nq = Session::new(Device::Xdna).compile(gq).run(qins)[0].clone();
    let codes = |v: &[f32]| -> Vec<i8> {
        v.iter()
            .flat_map(|c| c.to_le_bytes())
            .map(|b| b as i8)
            .take(32)
            .collect()
    };
    let q_ok = codes(&cq) == codes(&nq);
    println!(
        "  {:<20} {}",
        "quant·i8",
        if q_ok { "PASS ✓" } else { "FAIL ✗" }
    );
    all &= q_ok;

    // QUANTIZE → DEQUANTIZE round-trip (per-channel, axis=0): 2-op chain crossing
    // f32→i8→f32. Both backends run the same affine, so bit-close (round-trip loss).
    let dqx: Vec<f32> = (0..32).map(|i| ((i % 11) as f32 - 5.0) * 0.15).collect();
    let dqsc = vec![0.04f32, 0.05, 0.06, 0.03];
    let dqzp = vec![0i32; 4];
    let mut gdq = Graph::new("dequant_rt");
    let dqi = gdq.input("x", Shape::new(&[4, 8], DType::F32));
    let dqq = gdq.add_node(
        Op::Quantize {
            axis: Some(0),
            scales: dqsc.clone(),
            zero_points: dqzp.clone(),
        },
        vec![dqi],
        Shape::new(&[4, 8], DType::I8),
    );
    let dqd = gdq.add_node(
        Op::Dequantize {
            axis: Some(0),
            scales: dqsc.clone(),
            zero_points: dqzp.clone(),
        },
        vec![dqq],
        Shape::new(&[4, 8], DType::F32),
    );
    gdq.set_outputs(vec![dqd]);
    let dqins: &[(&str, &[f32])] = &[("x", &dqx)];
    let cdq = Session::new(Device::Cpu).compile(gdq.clone()).run(dqins)[0].clone();
    let ndq = Session::new(Device::Xdna).compile(gdq).run(dqins)[0].clone();
    all &= cmp("quant→dequant", &cdq, &ndq);

    // VISION PATH — 2-D pooling (NCHW) + im2col unfold. Pool/Im2Col are host-computed
    // gathers (bit-exact vs CPU); the conv GEMM stays on the NPU (im2col → matmul).
    let pool_x: Vec<f32> = (0..32).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let pins: &[(&str, &[f32])] = &[("x", &pool_x)];
    for (name, kind) in [("maxpool", IrRed::Max), ("avgpool", IrRed::Mean)] {
        let mut gp = Graph::new(name);
        let px = gp.input("x", Shape::new(&[1, 2, 4, 4], DType::F32));
        let py = gp.add_node(
            Op::Pool {
                kind,
                kernel_size: vec![2, 2],
                stride: vec![2, 2],
                padding: vec![0, 0],
            },
            vec![px],
            Shape::new(&[1, 2, 2, 2], DType::F32),
        );
        gp.set_outputs(vec![py]);
        let cp = Session::new(Device::Cpu).compile(gp.clone()).run(pins)[0].clone();
        let np = Session::new(Device::Xdna).compile(gp).run(pins)[0].clone();
        all &= cmp(name, &cp, &np);
    }

    // im2col: [1,1,4,4] k3 s1 p1 → [16, 9] (bit-exact unfold)
    let im_x: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.25).collect();
    let imins: &[(&str, &[f32])] = &[("x", &im_x)];
    let mut gim = Graph::new("im2col");
    let imx = gim.input("x", Shape::new(&[1, 1, 4, 4], DType::F32));
    let imy = gim.im2col(imx, [3, 3], [1, 1], [1, 1], [1, 1]);
    gim.set_outputs(vec![imy]);
    let cim = Session::new(Device::Cpu).compile(gim.clone()).run(imins)[0].clone();
    let nim = Session::new(Device::Xdna).compile(gim).run(imins)[0].clone();
    all &= cmp("im2col", &cim, &nim);

    // CONV via im2col → int8 matmul on the NPU: unfold [1,1,4,4] k3 s1 p1 → [16,9],
    // then @ W[9,4] → [16,4] (conv output, C_out=4 flattened). int8-quantized GEMM →
    // cosine-close vs f32 CPU (same as the MLP path).
    let cvw: Vec<f32> = (0..9 * 4).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let mut gcv = Graph::new("conv_im2col");
    let cvx = gcv.input("x", Shape::new(&[1, 1, 4, 4], DType::F32));
    let cvcol = gcv.im2col(cvx, [3, 3], [1, 1], [1, 1], [1, 1]);
    let cv_w = gcv.param("W", Shape::new(&[9, 4], DType::F32));
    let cvout = gcv.matmul(cvcol, cv_w, Shape::new(&[16, 4], DType::F32));
    gcv.set_outputs(vec![cvout]);
    let run_cv = |dev: Device| -> Vec<f32> {
        let mut c = Session::new(dev).compile(gcv.clone());
        c.set_param("W", &cvw);
        c.run(imins)[0].clone()
    };
    let (ccv, ncv) = (run_cv(Device::Cpu), run_cv(Device::Xdna));
    let cvdot: f32 = ccv.iter().zip(&ncv).map(|(a, b)| a * b).sum();
    let cvcos = cvdot
        / (ccv.iter().map(|a| a * a).sum::<f32>().sqrt()
            * ncv.iter().map(|b| b * b).sum::<f32>().sqrt())
        .max(1e-6);
    let cv_ok = cvcos > 0.99;
    println!(
        "  {:<20} {}  cos {:.4}  (im2col→int8 matmul on NPU)",
        "conv·im2col@W",
        if cv_ok { "PASS ✓" } else { "FAIL ✗" },
        cvcos
    );
    all &= cv_ok;

    println!(
        "\n{}",
        if all {
            "all NPU graph ops match CPU ✓"
        } else {
            "MISMATCH"
        }
    );
    if !all {
        std::process::exit(1);
    }
}
