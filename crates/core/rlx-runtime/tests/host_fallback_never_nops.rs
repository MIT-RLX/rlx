// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The CPU host fallback must never be handed an op rlx-cpu cannot run.**
//!
//! `rlx-runtime/tests/cpu_nop_fused_ops_parity.rs` catches this class
//! *numerically* — it runs the three ops on every available device and requires
//! a non-zero answer that matches CPU. That works, and it found nothing this
//! file did not, but it has two limits that matter:
//!
//! * it needs a **device**, so a host without one proves nothing; and
//! * it needs the op to be **exercised**, so a claim nobody built a case for
//!   stays untested. Coverage is by enumeration, and enumeration is what
//!   drifted in the first place.
//!
//! This is the same invariant checked structurally instead, device-free, from
//! the two facts that compose into the bug:
//!
//! 1. rlx-cpu claims some OpKinds it has **no thunk arm** for
//!    ([`rlx_cpu::NO_THUNK_ARM`]) — the catch-all is `Thunk::Nop`, so one that
//!    survives to `compile_thunks` yields a zeroed slot rather than an error.
//! 2. A GPU backend claims the same kind (so nothing upstream expands it) and
//!    routes it to the host fallback (so the scheduler is happy to hand it to
//!    rlx-cpu).
//!
//! Neither fact is wrong alone. Together they are silent zeros: no panic, no
//! unsupported-op error, and *plausible-looking* output. Vulkan's
//! `PartitionedConv` shipped exactly that way and reproduced identically on
//! MoltenVK, NVIDIA and RADV — because the op never ran at all.
//!
//! ## Coverage, stated rather than implied
//!
//! Four backends now expose the routing decision as a predicate — vulkan,
//! wgpu, cuda and rocm — and each one's compile arm is *written in terms of*
//! that predicate, so the op list and the route cannot drift apart.
//!
//! **rlx-metal is still not checked here**, and for a specific reason worth
//! writing down: its thunk compile ends in `_other => Thunk::HostOp`, a
//! catch-all rather than an enumerated arm. "Does Metal route this op to the
//! host?" is therefore "did no earlier arm match it?", which cannot be
//! answered by a predicate without inverting the whole match. Metal stays
//! numerically covered (`cpu_nop_fused_ops_parity.rs`, which does pass on this
//! machine — `PartitionedConv/metal: ok`).
//!
//! MLX, CoreML and TPU are unchecked both ways: no router, and
//! `cpu_nop_fused_ops_parity.rs`'s `devices()` list has no `Device::Ane` and no
//! TPU entry at all.
//!
//! And the numerical side has a hole of its own that no backend fixes:
//! `TransformRegion` and `BatchElementwiseRegion` are claimed by every backend
//! and exercised by **no** numerical case, because neither is produced by the
//! default pipeline (they need `RLX_NATIVE_FK_REGIONS=1` or horizontal fusion).
//! For those two kinds the static check below is the only coverage there is.

// `NO_THUNK_ARM` lives in rlx-cpu, and `vulkan` implies `cpu`, so without this
// feature there is nothing here to check.
#![cfg(feature = "cpu")]

use rlx_ir::op::{Activation, ChainStep, Op, RegionPrologue, TransformStep};
use rlx_ir::{Graph, OpKind};

/// One representative `Op` per kind in [`rlx_cpu::NO_THUNK_ARM`].
///
/// Field values are irrelevant — every consumer here dispatches on the kind —
/// but building a real `Op` rather than passing an `OpKind` is deliberate: the
/// routing predicates take `&Op` and some of them match on fields, so asking
/// them with a kind would test a different function than the one that runs.
fn representative(kind: OpKind) -> Op {
    match kind {
        OpKind::FusedConvBiasAct => Op::FusedConvBiasAct {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
            activation: Some(Activation::Relu),
            has_residual: false,
        },
        OpKind::PartitionedConv => Op::PartitionedConv { block: 4 },
        OpKind::FusedTransformerLayer => Op::FusedTransformerLayer {
            num_heads: 2,
            head_dim: 8,
            intermediate_size: 32,
            eps1: 1e-5,
            eps2: 1e-5,
            activation: Activation::Silu,
            has_bias: false,
        },
        OpKind::TransformRegion => Op::TransformRegion {
            steps: Vec::<TransformStep>::new(),
            num_inputs: 1,
        },
        OpKind::BatchElementwiseRegion => Op::BatchElementwiseRegion {
            chain: Vec::<ChainStep>::new(),
            num_batch_inputs: 1,
            scalar_input_mask: 0,
            input_modulus: [0; 16],
            prologue: RegionPrologue::default(),
            prologue_input: 0,
        },
        other => panic!(
            "rlx_cpu::NO_THUNK_ARM grew {other:?} with no representative Op here. \
             Add one — an unrepresented kind is an unchecked kind, and this file \
             exists because unchecked kinds return zeros."
        ),
    }
}

/// Every kind rlx-cpu declares it cannot thunk must actually be gone after
/// `prepare_graph_for_thunks`, or rlx-cpu itself returns zeros for it.
///
/// This is what makes `NO_THUNK_ARM` worth trusting downstream: the list is not
/// a comment, it is the input to that expansion *and* to the checks below, so a
/// kind added to it without an expansion fails here rather than silently in a
/// backend that believed the host could run it.
#[test]
fn rlx_cpu_expands_every_kind_it_cannot_thunk() {
    for &kind in rlx_cpu::NO_THUNK_ARM {
        let op = representative(kind);
        let mut g = Graph::new("nop_risk");
        let x = g.input("x", rlx_ir::Shape::new(&[1, 4], rlx_ir::DType::F32));
        // A single node of this kind, wired to one input. The expansion is
        // shape-driven, so an unused/ill-fitting shape is fine: what is under
        // test is whether the KIND survives, not what it computes.
        let n = g.add_node(op, vec![x], rlx_ir::Shape::new(&[1, 4], rlx_ir::DType::F32));
        g.set_outputs(vec![n]);

        let out = std::panic::catch_unwind(move || rlx_cpu::prepare_graph_for_thunks(g));
        let Ok(out) = out else {
            // An expansion that panics on a degenerate shape is loud, which is
            // the acceptable outcome here — the failure mode this guards is
            // silence. Nothing to assert.
            continue;
        };
        assert!(
            !out.nodes().iter().any(|n| n.op.kind() == kind),
            "rlx-cpu lists {kind:?} in NO_THUNK_ARM but prepare_graph_for_thunks \
             leaves it in the graph — it will compile to Thunk::Nop and return zeros"
        );
    }
}

/// Backends that expose their routing decision as a predicate.
///
/// Each entry is `(name, claims, routes_to_cpu_host)`. Adding a backend here is
/// the whole cost of moving it from "numerical only" to "checked statically" —
/// the predicate lives next to the op list it has to agree with, and the
/// backend's compile arm is written in terms of it, so the two cannot drift.
type Router = (&'static str, &'static [OpKind], fn(&Op) -> bool);

// Each push is `#[cfg]`-gated on a backend feature, so this cannot be a
// `vec![]` literal — which is the shape the lint assumes.
#[allow(clippy::vec_init_then_push)]
fn routers() -> Vec<Router> {
    let mut out: Vec<Router> = Vec::new();
    #[cfg(feature = "vulkan")]
    out.push((
        "vulkan",
        rlx_vulkan::backend::SUPPORTED_OPS,
        rlx_vulkan::backend::routes_to_cpu_host,
    ));
    #[cfg(feature = "gpu")]
    out.push((
        "wgpu",
        rlx_wgpu::supported_ops::SUPPORTED_OPS,
        rlx_wgpu::supported_ops::routes_to_cpu_host,
    ));
    #[cfg(feature = "cuda")]
    out.push((
        "cuda",
        rlx_cuda::supported_ops::SUPPORTED_OPS,
        rlx_cuda::supported_ops::routes_to_cpu_host,
    ));
    #[cfg(feature = "rocm")]
    out.push((
        "rocm",
        rlx_rocm::supported_ops::SUPPORTED_OPS,
        rlx_rocm::supported_ops::routes_to_cpu_host,
    ));
    out
}

/// No backend with an inspectable router may both claim a `NO_THUNK_ARM` kind
/// and route it to rlx-cpu.
///
/// This is the check that would have caught `PartitionedConv` at compile time,
/// with no device, no shader and no numerical comparison — the claim and the
/// route are both static facts, and their product is the bug.
///
/// It is not hypothetical for wgpu either: `Op::PartitionedConv` is still in
/// its generic `Step::HostOp` arm, and the note beside `SUPPORTED_OPS` records
/// what that cost when the claim was there too (`cpu=0.4 vs gpu=0`). Declining
/// the claim is what fixed it, and this test is what notices if it comes back.
#[test]
fn no_backend_both_claims_and_host_routes_a_cpu_nop_op() {
    let routers = routers();
    assert!(
        !routers.is_empty(),
        "no backend router compiled in — this test would pass vacuously. \
         Run it with at least one of the vulkan / gpu / cuda / rocm features."
    );
    for (name, claims, routes) in routers {
        for &kind in rlx_cpu::NO_THUNK_ARM {
            let op = representative(kind);
            assert!(
                !(claims.contains(&kind) && routes(&op)),
                "{name} claims {kind:?} in SUPPORTED_OPS (so nothing upstream expands it) \
                 AND routes it to the CPU host fallback (where rlx-cpu has no thunk arm). \
                 That composition returns a buffer of zeros with no error. Either drop the \
                 claim, expand it in {name}'s own compile entry, or give it a real kernel."
            );
        }
    }
}

/// **Which backends are even at risk, derived rather than listed.**
///
/// The first version of this carried a hand-written list of "backends with no
/// inspectable router". That is the drift pattern this whole file exists to
/// fight — a hand-copied inventory of where the danger is, going stale next to
/// the danger.
///
/// The risk surface is computable: a backend can only hit the claim-then-Nop
/// composition for kinds it actually **claims**, so it is
/// `SUPPORTED_OPS ∩ NO_THUNK_ARM`. A backend claiming none of them is not
/// "unchecked", it is *provably not at risk*, which is a much stronger
/// statement than a list can make.
///
/// For the rest, `rlx-runtime/tests/cpu_nop_fused_ops_parity.rs` covers them
/// numerically on real devices, and `every_risky_pair_has_a_numerical_case`
/// there requires a case to exist for each pair this function reports.
fn risky_claims() -> Vec<(&'static str, Vec<OpKind>)> {
    let mut out: Vec<(&'static str, &[OpKind])> = Vec::new();
    #[cfg(feature = "metal")]
    out.push(("metal", rlx_metal::supported_ops::SUPPORTED_OPS));
    #[cfg(feature = "gpu")]
    out.push(("wgpu", rlx_wgpu::supported_ops::SUPPORTED_OPS));
    #[cfg(feature = "cuda")]
    out.push(("cuda", rlx_cuda::supported_ops::SUPPORTED_OPS));
    #[cfg(feature = "rocm")]
    out.push(("rocm", rlx_rocm::supported_ops::SUPPORTED_OPS));
    #[cfg(feature = "mlx")]
    out.push(("mlx", rlx_mlx::supported_ops::SUPPORTED_OPS));
    #[cfg(feature = "vulkan")]
    out.push(("vulkan", rlx_vulkan::backend::SUPPORTED_OPS));
    out.push(("cpu", rlx_cpu::supported_ops::SUPPORTED_OPS));

    out.into_iter()
        .map(|(name, claimed)| {
            let risky: Vec<OpKind> = rlx_cpu::NO_THUNK_ARM
                .iter()
                .copied()
                .filter(|k| claimed.contains(k))
                .collect();
            (name, risky)
        })
        .collect()
}

/// Report the derived risk surface, and require the static check to cover every
/// backend that has one *and* exposes a router.
#[test]
fn the_risk_surface_is_derived_not_listed() {
    let mut at_risk = 0usize;
    for (name, risky) in risky_claims() {
        if risky.is_empty() {
            eprintln!("  {name}: claims none of NO_THUNK_ARM — not at risk");
            continue;
        }
        at_risk += 1;
        // Derived from the router list, not spelled out again: a backend that
        // grows a `routes_to_cpu_host` is reported as statically checked the
        // moment it appears in `routers()`, with no second place to update.
        let has_router = routers().iter().any(|(n, ..)| *n == name);
        let checked = if has_router {
            "STATIC (routes_to_cpu_host) + numerical"
        } else {
            "numerical only (cpu_nop_fused_ops_parity.rs)"
        };
        eprintln!("  {name}: at risk for {risky:?} — {checked}");
    }
    assert!(
        at_risk > 0,
        "no backend claims any NO_THUNK_ARM kind — either the claims moved or \
         this test is looking at the wrong lists; a zero risk surface here \
         would silently disable both this gate and the parity one"
    );
}
