// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The three fused ops rlx-cpu Nops must still be correct on every backend.**
//!
//! `rlx-cpu` claims `FusedConvBiasAct`, `PartitionedConv` and
//! `FusedTransformerLayer` for legalize coverage, but its thunk compiler has no
//! arm for any of them — `expand.rs` rewrites them to primitives *before*
//! thunking, and the catch-all compile arm is `Thunk::Nop`.
//!
//! That makes them the sharp edge of the claim-then-expand pattern. A GPU
//! backend that (a) claims one of these kinds, so nothing upstream expands it,
//! and (b) lists it as a host fallback, so the scheduler is happy to hand it to
//! rlx-cpu, gets a **Nop over a zeroed arena slot**. No panic, no unsupported-op
//! error — just zeros. Vulkan's `PartitionedConv` did exactly this, identically
//! on MoltenVK, NVIDIA and RADV, until `expand_partitioned_conv` was added.
//!
//! Each op is built the way it really arises — a primitive graph put through
//! its fusion pass, or the dedicated builder — then run on every available
//! device and compared against CPU. A backend that Nops it returns zeros, which
//! is exactly what this catches.

use rlx_fusion::pass::Pass;
use rlx_ir::op::{Activation, BinaryOp, Op};
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{Device, Session};

mod common;

const F: DType = DType::F32;

fn devices() -> Vec<(&'static str, Device)> {
    let mut v = Vec::new();
    for (name, d) in [
        ("metal", Device::Metal),
        ("mlx", Device::Mlx),
        ("wgpu", Device::Gpu),
        ("cuda", Device::Cuda),
        ("rocm", Device::Rocm),
        ("vulkan", Device::Vulkan),
        ("oneapi", Device::OneApi),
    ] {
        if rlx_runtime::is_available(d) {
            v.push((name, d));
        }
    }
    v
}

/// Deterministic small values — the point is agreement, not conditioning.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z ^= z >> 27;
            ((z >> 40) as f32 / 8_388_608.0 - 0.5) * 0.4
        })
        .collect()
}

fn const_f32(g: &mut Graph, v: &[f32], dims: &[usize]) -> NodeId {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(Op::Constant { data: bytes }, vec![], Shape::new(dims, F))
}

/// Compare `dev` against CPU on `graph`, and separately assert the result is
/// not identically zero.
///
/// The zero check is the load-bearing half: a Nop'd op agrees with nothing, but
/// a *tolerance* comparison against a reference that happened to be near zero
/// could still pass. Requiring real magnitude makes "the op never ran" a
/// failure on its own terms.
fn parity(op_name: &str, graph: &Graph, tol: f32) {
    let cpu = Session::new(Device::Cpu).compile(graph.clone()).run(&[]);
    let mag = cpu[0].iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(
        mag > 1e-3,
        "{op_name}: the CPU reference is itself ~zero ({mag:.3e}) — this test \
         cannot distinguish a Nop from a correct answer; fix the fixture"
    );

    for (name, dev) in devices() {
        let got = Session::new(dev).compile(graph.clone()).run(&[]);
        assert_eq!(got.len(), cpu.len(), "{op_name}/{name}: output count");

        let got_mag = got[0].iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(
            got_mag > 1e-6,
            "{op_name}/{name}: every output element is zero. The backend most \
             likely claims this OpKind (so nothing upstream expands it) AND \
             routes it to the CPU host fallback, where rlx-cpu Nops it — see \
             rlx_vulkan::unfuse::expand_partitioned_conv."
        );

        for (i, (a, b)) in cpu[0].iter().zip(got[0].iter()).enumerate() {
            let d = (a - b).abs();
            let rel = d / a.abs().max(b.abs()).max(1e-4);
            assert!(
                d < tol || rel < tol,
                "{op_name}/{name}: [{i}] cpu={a} gpu={b} (Δ={d})"
            );
        }
        eprintln!("  {op_name}/{name}: ok");
    }
}

#[test]
fn partitioned_conv_runs_everywhere() {
    let _gpu = common::serialize_gpu();
    let mut g = Graph::new("pconv");
    let xs: Vec<f32> = (0..300)
        .map(|t| (t as f32 * 0.3).sin() + 0.4 * (t as f32 * 0.11).cos())
        .collect();
    let x = const_f32(&mut g, &xs, &[2, 150]);
    let ir: Vec<f32> = (0..200).map(|i| (-(i as f32) * 0.03).exp()).collect();
    let irn = const_f32(&mut g, &ir, &[ir.len()]);
    let out = g.partitioned_conv(x, irn, 128);
    g.set_outputs(vec![out]);
    parity("PartitionedConv", &g, 5e-2);
}

#[test]
fn fused_conv_bias_act_runs_everywhere() {
    let _gpu = common::serialize_gpu();
    // conv → bias → relu, then fused. Built as primitives so the pass produces
    // the node the same way the real pipeline does.
    const N: usize = 1;
    const CI: usize = 3;
    const CO: usize = 4;
    const HW: usize = 8;
    const K: usize = 3;

    let mut g = Graph::new("cba");
    let x = const_f32(&mut g, &fill(N * CI * HW * HW, 1), &[N, CI, HW, HW]);
    let w = const_f32(&mut g, &fill(CO * CI * K * K, 2), &[CO, CI, K, K]);
    let b = const_f32(&mut g, &fill(CO, 3), &[CO]);
    // `padding = k/2` keeps the output the same size as the input, which is
    // what `FuseConvBiasAct` expects to match against.
    let conv = g.conv2d(x, w, [K, K], [1, 1], [K / 2, K / 2], [1, 1], 1);
    let out_s = Shape::new(&[N, CO, HW, HW], F);
    let b4 = g.reshape(b, vec![1, CO as i64, 1, 1], Shape::new(&[1, CO, 1, 1], F));
    let b_e = g.add_node(
        Op::Expand {
            target_shape: vec![1, CO as i64, HW as i64, HW as i64],
        },
        vec![b4],
        out_s.clone(),
    );
    let sum = g.binary(BinaryOp::Add, conv, b_e, out_s.clone());
    let act = g.add_node(Op::Activation(Activation::Relu), vec![sum], out_s);
    g.set_outputs(vec![act]);

    let fused = rlx_fusion::FuseConvBiasAct.run(g);
    assert_eq!(
        fused
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::FusedConvBiasAct { .. }))
            .count(),
        1,
        "fixture no longer produces a FusedConvBiasAct — the test would pass \
         vacuously on the unfused graph"
    );
    parity("FusedConvBiasAct", &fused, 1e-4);
}

#[test]
fn fused_transformer_layer_runs_everywhere() {
    let _gpu = common::serialize_gpu();
    // Post-norm BERT-style layer, built directly: the fusion pass that emits
    // this node matches a long primitive chain, and reproducing that chain
    // exactly is more fragile than naming the op.
    //
    // No-bias input order (8 entries), per `unfuse_fused_transformer_layer`:
    //   0 hidden, 1 qkv_w, 2 out_w, 3 ln1_g, 4 fc1_w, 5 fc2_w, 6 ln2_g, 7 mask
    const B: usize = 1;
    const S: usize = 4;
    const NH: usize = 2;
    const DH: usize = 8;
    const H: usize = NH * DH;
    const INTER: usize = 16;

    let mut g = Graph::new("ftl");
    let hidden = const_f32(&mut g, &fill(B * S * H, 11), &[B, S, H]);
    let qkv_w = const_f32(&mut g, &fill(H * 3 * H, 12), &[H, 3 * H]);
    let out_w = const_f32(&mut g, &fill(H * H, 13), &[H, H]);
    let ln1_g = const_f32(&mut g, &[1.0; H], &[H]);
    let fc1_w = const_f32(&mut g, &fill(H * INTER, 14), &[H, INTER]);
    let fc2_w = const_f32(&mut g, &fill(INTER * H, 15), &[INTER, H]);
    let ln2_g = const_f32(&mut g, &[1.0; H], &[H]);
    // Rank-4 `[1, 1, S, S]`: the mask feeds `Op::Attention`, whose operand is
    // `[B, NH, S, S]`. A rank-2 `[S, S]` mask is broadcastable in principle but
    // MLX's lowering reshapes it to `[S, 1, 1, S]` and fails — an Attention
    // mask-rank question, not the Nop question this test is about.
    let mask = const_f32(&mut g, &[0.0; S * S], &[1, 1, S, S]);

    let out = g.add_node(
        Op::FusedTransformerLayer {
            num_heads: NH,
            head_dim: DH,
            intermediate_size: INTER,
            eps1: 1e-5,
            eps2: 1e-5,
            activation: Activation::Gelu,
            has_bias: false,
        },
        vec![hidden, qkv_w, out_w, ln1_g, fc1_w, fc2_w, ln2_g, mask],
        Shape::new(&[B, S, H], F),
    );
    g.set_outputs(vec![out]);
    parity("FusedTransformerLayer", &g, 2e-3);
}

/// **Every risky (backend, kind) pair must have a case above — or be named.**
///
/// The three cases in this file were chosen because they were the ops in front
/// of us when Vulkan's `PartitionedConv` shipped as zeros. That is a fine reason
/// to write a test and a poor reason to believe the class is covered: the risk
/// surface is `SUPPORTED_OPS ∩ rlx_cpu::NO_THUNK_ARM` per backend, and nothing
/// tied these three to it. A fourth kind added to `NO_THUNK_ARM` would have
/// joined the surface silently.
///
/// So the surface is derived and compared against what is actually covered.
/// A kind that is claimed but has no case must appear in `ACKNOWLEDGED` with a
/// reason — which keeps the gap visible and makes closing it a deliberate act,
/// rather than letting a green run imply coverage nobody checked.
#[test]
fn every_risky_kind_has_a_numerical_case_or_a_stated_reason() {
    let _gpu = common::serialize_gpu();
    use rlx_ir::OpKind;

    /// Kinds with a `parity(..)` case above.
    const COVERED: &[OpKind] = &[
        OpKind::PartitionedConv,
        OpKind::FusedConvBiasAct,
        OpKind::FusedTransformerLayer,
    ];

    /// Claimed, no case, and why.
    const ACKNOWLEDGED: &[(OpKind, &str)] = &[
        (
            OpKind::TransformRegion,
            "only produced under RLX_NATIVE_FK_REGIONS=1 or hand-built region IR; \
             the fusion pipeline does not emit it by default, so a fixture would \
             have to construct TransformStep values by hand rather than arise the \
             way the op really does",
        ),
        (
            OpKind::BatchElementwiseRegion,
            "same as TransformRegion — horizontal/z-plane fusion is off by \
             default, so there is no primitive graph that produces one",
        ),
    ];

    // The union of what every compiled-in backend claims of NO_THUNK_ARM.
    let mut claimed: Vec<OpKind> = Vec::new();
    let mut push = |ops: &[OpKind]| {
        for k in rlx_cpu::NO_THUNK_ARM {
            if ops.contains(k) && !claimed.contains(k) {
                claimed.push(*k);
            }
        }
    };
    push(rlx_cpu::supported_ops::SUPPORTED_OPS);
    #[cfg(feature = "metal")]
    push(rlx_metal::supported_ops::SUPPORTED_OPS);
    #[cfg(feature = "gpu")]
    push(rlx_wgpu::supported_ops::SUPPORTED_OPS);
    #[cfg(feature = "cuda")]
    push(rlx_cuda::supported_ops::SUPPORTED_OPS);
    #[cfg(feature = "rocm")]
    push(rlx_rocm::supported_ops::SUPPORTED_OPS);
    #[cfg(feature = "mlx")]
    push(rlx_mlx::supported_ops::SUPPORTED_OPS);
    #[cfg(feature = "vulkan")]
    push(rlx_vulkan::backend::SUPPORTED_OPS);

    let gaps: Vec<OpKind> = claimed
        .iter()
        .copied()
        .filter(|k| !COVERED.contains(k) && !ACKNOWLEDGED.iter().any(|(a, _)| a == k))
        .collect();

    eprintln!("risky kinds claimed by some backend: {claimed:?}\n  covered by a case: {COVERED:?}");
    for (k, why) in ACKNOWLEDGED {
        eprintln!("  NOT covered — {k:?}: {why}");
    }

    assert!(
        gaps.is_empty(),
        "{gaps:?} are claimed by a backend and are in rlx_cpu::NO_THUNK_ARM, so a \
         backend that claims one without lowering it returns ZEROS — but no case \
         above exercises them. Add a `parity(..)` case, or add the kind to \
         ACKNOWLEDGED with the reason a fixture cannot be built."
    );
    assert!(
        !claimed.is_empty(),
        "no backend claims any NO_THUNK_ARM kind — this whole file would be \
         testing nothing; check the lists rather than deleting the test"
    );
}
