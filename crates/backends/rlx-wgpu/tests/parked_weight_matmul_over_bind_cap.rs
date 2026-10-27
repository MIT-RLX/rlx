// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A matmul whose weight is a graph parameter, in a graph whose arena outgrows
//! the device's storage-binding limit.
//!
//! Past the limit the arena parks every parameter in a separate weight buffer
//! and generic ops stage a parked weight into their bind window. KittenTTS on
//! an RTX 3080 Ti (2 GiB bindings, 3.2 GB arena) got every such matmul back as
//! zeros. `RLX_WGPU_MAX_BIND_MB` caps the binding size the device is created
//! with, so the regime reproduces on any adapter; the device is process-wide,
//! so this file holds one test and sets the cap before anything touches wgpu.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

const CAP_MB: usize = 64;

#[test]
fn param_matmul_is_right_when_the_arena_exceeds_the_binding_limit() {
    rlx_ir::env::set("RLX_WGPU_MAX_BIND_MB", CAP_MB.to_string());
    // Each stripe keeps a staging reserve at its tail; it has to be smaller
    // than the stripe for the planner to leave room for activations.
    rlx_ir::env::set("RLX_WGPU_SHARD_STAGE_MIB", "24");
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return;
    }
    // A long elementwise chain over `big` keeps several 32 MiB activations
    // alive, pushing the arena well past the cap.
    let big = 8 << 20;
    let (m, k, n) = (64usize, 256usize, 128usize);
    let mut g = Graph::new("parked_weight_matmul");
    let mp = g.param("mp", Shape::new(&[n], DType::F32));
    let mi = g.input("mi", Shape::new(&[n], DType::F32));
    // Hand-staged elementwise arms: Kitten's attention mask is
    // `Compare(Eq)` of two zero tensors feeding a `Where`; under the cap both
    // came out as if every operand read from the wrong stripe. Built first so
    // the planner puts them in a low stripe, away from the global scratch.
    // Both Compare operands are activations already inside the output's
    // window — the case that needed no staging and still had its window moved.
    let mj = g.input("mj", Shape::new(&[n], DType::F32));
    let ce = g.add_node(
        rlx_ir::Op::Compare(rlx_ir::op::CmpOp::Eq),
        vec![mi, mj],
        Shape::new(&[n], DType::Bool),
    );
    let wh = g.add_node(
        rlx_ir::Op::Where,
        vec![ce, mi, mp],
        Shape::new(&[n], DType::F32),
    );
    // `Fma` stages the same way.
    let fm = g.add_node(
        rlx_ir::Op::Fma,
        vec![mi, mj, mi],
        Shape::new(&[n], DType::F32),
    );
    // Norm host steps (`GroupNormHost`, `LayerNorm2dHost`) kept gamma/beta
    // offsets as `u32`; parked, those read activations instead.
    let (nc, nh, nw) = (8usize, 4usize, 5usize);
    let nx = g.input("nx", Shape::new(&[1, nc, nh, nw], DType::F32));
    let gg = g.param("gg", Shape::new(&[nc], DType::F32));
    let gb = g.param("gb", Shape::new(&[nc], DType::F32));
    let gn = g.add_node(
        rlx_ir::Op::GroupNorm {
            num_groups: 2,
            eps: 1e-5,
        },
        vec![nx, gg, gb],
        Shape::new(&[1, nc, nh, nw], DType::F32),
    );
    let ln2 = g.add_node(
        rlx_ir::Op::LayerNorm2d { eps: 1e-6 },
        vec![nx, gg, gb],
        Shape::new(&[1, nc, nh, nw], DType::F32),
    );
    let x = g.input("x", Shape::new(&[big], DType::F32));
    let a = g.input("a", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[k, n], DType::F32));
    let h1 = g.relu(x);
    let h2 = g.add(h1, x);
    let h3 = g.mul(h2, h1);
    let side = g.add(h3, h2);
    let y = g.mm(a, w);
    // `x · W + b` with `W`, `b` params fuses into `FusedMatMulBiasAct`, a
    // separate lowering — Kitten's zeroed matmuls are all of this form.
    let a3 = g.input("a3", Shape::new(&[1, m, k], DType::F32));
    let b = g.param("b", Shape::new(&[n], DType::F32));
    let y3 = g.mm(a3, w);
    let yb = g.add(y3, b);
    // Kitten compiles with fusion off: its matmuls are plain `[1, M, K] · W`.
    let a4 = g.input("a4", Shape::new(&[1, m, k], DType::F32));
    let y4 = g.mm(a4, w);
    // Kitten's iSTFT: a ConvTranspose2d whose operands span stripes is hosted;
    // its weight is a parked param.
    let (cl, ct_in) = (11usize, 4001usize);
    let ct_out = (ct_in - 1) * 5 + 20;
    let cx = g.input("cx", Shape::new(&[1, cl, 1, ct_in], DType::F32));
    let cw = g.param("cw", Shape::new(&[cl, 1, 1, 20], DType::F32));
    let cw2 = g.param("cw2", Shape::new(&[cl, 1, 1, 20], DType::F32));
    let ct = g.add_node(
        rlx_ir::Op::ConvTranspose2d {
            kernel_size: vec![1, 20],
            stride: vec![1, 5],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            output_padding: vec![0, 0],
            groups: 1,
        },
        vec![cx, cw],
        Shape::new(&[1, 1, 1, ct_out], DType::F32),
    );
    // …and, as in the iSTFT, two of them feed a GPU `Sub`.
    let ct2 = g.add_node(
        rlx_ir::Op::ConvTranspose2d {
            kernel_size: vec![1, 20],
            stride: vec![1, 5],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            output_padding: vec![0, 0],
            groups: 1,
        },
        vec![cx, cw2],
        Shape::new(&[1, 1, 1, ct_out], DType::F32),
    );
    let d = g.sub(ct, ct2);
    // A host-delegated custom op (`Step::CustomHost`) reading a parked param:
    // Kitten's `onnx.QMatMulBaked` read its baked weight from a truncated
    // offset and returned zeros.
    let md = g.add_node(
        rlx_ir::Op::Custom {
            name: "onnx.Mod".into(),
            num_inputs: 2,
            attrs: vec![1],
        },
        vec![mi, mp],
        Shape::new(&[n], DType::F32),
    );
    // The Bool mask itself reads back differently per device (packed bytes on
    // CPU), so it is checked through the `Where` it selects with.
    // Kernels that bind the output's window but address operands by absolute
    // arena offset (`Cumsum` here) are wrong off stripe 0 and on parked params.
    let cs_param = g.add_node(
        rlx_ir::Op::Cumsum {
            axis: -1,
            exclusive: false,
        },
        vec![mp],
        Shape::new(&[n], DType::F32),
    );
    let late = g.mul(md, mi);
    let cs_late = g.add_node(
        rlx_ir::Op::Cumsum {
            axis: -1,
            exclusive: false,
        },
        vec![late],
        Shape::new(&[n], DType::F32),
    );
    g.set_outputs(vec![
        y, side, yb, y4, d, ct, ct2, md, wh, fm, gn, ln2, cs_param, cs_late,
    ]);

    let xv: Vec<f32> = (0..big).map(|i| ((i % 977) as f32 * 0.01).sin()).collect();
    let av: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.013).cos()).collect();
    let wv: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.007).sin() * 0.1).collect();
    let bv: Vec<f32> = (0..n).map(|i| 0.1 * (i as f32 * 0.7).sin()).collect();
    let cxv: Vec<f32> = (0..cl * ct_in).map(|i| (i as f32 * 0.031).sin()).collect();
    let cwv: Vec<f32> = (0..cl * 20)
        .map(|i| (i as f32 * 0.17).cos() * 0.05)
        .collect();
    let mpv: Vec<f32> = (0..n).map(|i| 0.75 + (i % 7) as f32 * 0.5).collect();
    let miv: Vec<f32> = (0..n)
        .map(|i| {
            if i % 3 == 0 {
                mpv[i]
            } else {
                (i as f32 * 0.37).sin() * 9.0
            }
        })
        .collect();
    let mjv: Vec<f32> = (0..n)
        .map(|i| if i % 4 == 0 { miv[i] } else { miv[i] + 1.0 })
        .collect();
    let nxv: Vec<f32> = (0..nc * nh * nw)
        .map(|i| (i as f32 * 0.41).sin() * 2.0)
        .collect();
    let ggv: Vec<f32> = (0..nc).map(|i| 0.5 + i as f32 * 0.25).collect();
    let gbv: Vec<f32> = (0..nc).map(|i| i as f32 * 0.1 - 0.3).collect();
    let run = |d: Device| {
        let mut c = Session::new(d).compile(g.clone());
        c.set_param("w", &wv);
        c.set_param("b", &bv);
        c.set_param("mp", &mpv);
        c.set_param("gg", &ggv);
        c.set_param("gb", &gbv);
        c.set_param("cw", &cwv);
        c.set_param(
            "cw2",
            &cwv.iter().map(|v| v * -0.5 + 0.01).collect::<Vec<f32>>(),
        );
        c.run(&[
            ("x", xv.as_slice()),
            ("a", av.as_slice()),
            ("a3", av.as_slice()),
            ("a4", av.as_slice()),
            ("cx", cxv.as_slice()),
            ("mi", miv.as_slice()),
            ("mj", mjv.as_slice()),
            ("nx", nxv.as_slice()),
        ])
    };
    let cpu = run(Device::Cpu);
    let check = |gpu: &[Vec<f32>]| {
        let scale = cpu[0].iter().fold(0f32, |s, v| s.max(v.abs()));
        let mm_err = gpu[0]
            .iter()
            .zip(&cpu[0])
            .fold(0f32, |s, (a, b)| s.max((a - b).abs()))
            / scale;
        let side_err = gpu[1]
            .iter()
            .zip(&cpu[1])
            .fold(0f32, |s, (a, b)| s.max((a - b).abs()));
        eprintln!(
            "matmul rel {mm_err:.2e} (gpu peak {:e}), side max|d| {side_err:e}",
            gpu[0].iter().fold(0f32, |s, v| s.max(v.abs()))
        );
        let bscale = cpu[2].iter().fold(0f32, |s, v| s.max(v.abs()));
        let fused_err = gpu[2]
            .iter()
            .zip(&cpu[2])
            .fold(0f32, |s, (a, b)| s.max((a - b).abs()))
            / bscale;
        eprintln!("fused matmul+bias rel {fused_err:.2e}");
        let p_err = gpu[3]
            .iter()
            .zip(&cpu[3])
            .fold(0f32, |s, (a, b)| s.max((a - b).abs()))
            / scale;
        eprintln!("plain 3-D matmul rel {p_err:.2e}");
        let cscale = cpu[4].iter().fold(0f32, |s, v| s.max(v.abs()));
        let c_err = gpu[4]
            .iter()
            .zip(&cpu[4])
            .fold(0f32, |s, (a, b)| s.max((a - b).abs()))
            / cscale;
        for (j, nm) in [(5, "ct"), (6, "ct2")] {
            let sc = cpu[j].iter().fold(0f32, |s, v| s.max(v.abs()));
            let e = gpu[j]
                .iter()
                .zip(&cpu[j])
                .fold(0f32, |s, (a, b)| s.max((a - b).abs()))
                / sc;
            eprintln!("  {nm} rel {e:.2e}");
        }
        eprintln!(
            "hosted conv-transpose rel {c_err:.2e} (gpu peak {:e})",
            gpu[4].iter().fold(0f32, |s, v| s.max(v.abs()))
        );
        let md_bad = (0..n)
            .filter(|&i| gpu[7][i].to_bits() != cpu[7][i].to_bits())
            .count();
        eprintln!("hosted onnx.Mod over a parked param: {md_bad} of {n} differ");
        assert_eq!(md_bad, 0, "parked-param custom host op off CPU");
        let wh_bad = (0..n)
            .filter(|&i| gpu[8][i].to_bits() != cpu[8][i].to_bits())
            .count();
        eprintln!("Where(Compare(Eq)) over a parked param: {wh_bad} of {n} differ");
        assert_eq!(wh_bad, 0, "Compare/Where off CPU under the binding cap");
        let fm_bad = (0..n)
            .filter(|&i| gpu[9][i].to_bits() != cpu[9][i].to_bits())
            .count();
        eprintln!("Fma over in-window activations: {fm_bad} of {n} differ");
        assert_eq!(fm_bad, 0, "Fma off CPU under the binding cap");
        for (j, nm) in [
            (12, "Cumsum of a parked param"),
            (13, "Cumsum of a late activation"),
        ] {
            let bad = (0..n)
                .filter(|&i| gpu[j][i].to_bits() != cpu[j][i].to_bits())
                .count();
            eprintln!("{nm}: {bad} of {n} differ");
            assert_eq!(bad, 0, "{nm} off CPU under the binding cap");
        }
        for (j, nm) in [(10, "GroupNorm"), (11, "LayerNorm2d")] {
            let e = gpu[j]
                .iter()
                .zip(&cpu[j])
                .fold(0f32, |s, (a, b)| s.max((a - b).abs()));
            eprintln!("{nm} with parked gamma/beta: max|d| {e:e}");
            assert!(e < 1e-5, "{nm} with parked gamma/beta off CPU by {e:e}");
        }
        assert!(
            p_err < 1e-5,
            "parked-weight 3-D matmul off CPU by {p_err:e}"
        );
        assert!(
            c_err < 1e-5,
            "parked-weight conv-transpose off CPU by {c_err:e}"
        );
        assert!(mm_err < 1e-5, "parked-weight matmul off CPU by {mm_err:e}");
        assert!(
            fused_err < 1e-5,
            "parked-weight fused matmul+bias off CPU by {fused_err:e}"
        );
        assert!(
            side_err < 1e-5,
            "elementwise side chain off CPU by {side_err:e}"
        );
    };
    check(&run(Device::Gpu));
    // Sharded arenas host elementwise ops by default; `RLX_WGPU_SHARD_GPU`
    // runs them on the GPU, through the hand-staged `Where`/`Fma` arms.
    rlx_ir::env::set("RLX_WGPU_SHARD_GPU", "1");
    eprintln!("-- RLX_WGPU_SHARD_GPU=1");
    check(&run(Device::Gpu));
}
