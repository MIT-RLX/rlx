// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fusions that must fire on the graphs real model blocks produce.
//!
//! A parity test still passes when a fusion silently stops matching — the
//! unfused graph computes the same numbers, only slower. That is how
//! `FuseResidualRmsNorm` declined on every Metal decode layer unnoticed. These
//! tests build blocks through `rlx-flow` (the producer real models use, so a
//! change in how a block emits its ops is caught here) and compile them on
//! every registered device with `compile_graph_stages`, which runs the shared
//! fusion pipeline under that backend's real `SUPPORTED_OPS`.
//!
//! Each block carries two controls, so a check cannot pass by accident:
//!
//! * **producer control** — the unfused graph still contains the pattern. If a
//!   block stops emitting it, "fused op present, pattern gone" would hold for
//!   the wrong reason.
//! * **fusion-off control** — compiling with `skip_fusion` yields none of the
//!   fused ops, so they come from the fusion passes, not from the block.
//!
//! The last two tests check the device-free entry points against what a
//! backend does: `fuse()` must produce the graph the runtime compile produces,
//! and `supported_for_target` must agree with each backend's claim on the
//! kinds that gate fusion.

use std::collections::BTreeMap;

use rlx_compile::fusion_pipeline::{FUSION_GATING_KINDS, fuse, supported_for_target};
use rlx_flow::MapWeights;
use rlx_flow::prelude::*;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, OpKind, RopeStyle, Shape};
use rlx_runtime::stages::fusion_target_for;
use rlx_runtime::{CompileOptions, Device, backend_for, compile_graph_stages, registered_devices};

const D: usize = 64;
const HEADS: usize = 4;
const KV: usize = 2;
const HD: usize = 16;
const FFN: usize = 128;
const SEQ: usize = 8;

/// Op counts keyed by kind name (`OpKind` is not `Ord`).
type Counts = BTreeMap<String, usize>;

fn counts(g: &Graph) -> Counts {
    let mut m = Counts::new();
    for n in g.nodes() {
        *m.entry(format!("{:?}", n.op.kind())).or_default() += 1;
    }
    m
}

fn count(c: &Counts, k: OpKind) -> usize {
    c.get(&format!("{k:?}")).copied().unwrap_or(0)
}

fn weights(prefix: &str, entries: &[(&str, &[usize])]) -> MapWeights {
    let mut w = MapWeights::default();
    for (k, shape) in entries {
        let n = shape.iter().product();
        w.insert(format!("{prefix}.{k}"), vec![0.01; n], shape.to_vec());
    }
    w
}

/// Llama/Qwen-style prefill layer, composed from the small blocks
/// (`llama_prefill_layer_composed`) so every primitive stays visible.
fn llama_layer() -> Graph {
    let mut w = weights(
        "model.layers.0",
        &[
            ("input_layernorm.weight", &[D]),
            ("post_attention_layernorm.weight", &[D]),
            ("self_attn.q_proj.weight", &[HEADS * HD, D]),
            ("self_attn.k_proj.weight", &[KV * HD, D]),
            ("self_attn.v_proj.weight", &[KV * HD, D]),
            ("self_attn.o_proj.weight", &[D, HEADS * HD]),
            ("mlp.gate_proj.weight", &[FFN, D]),
            ("mlp.up_proj.weight", &[FFN, D]),
            ("mlp.down_proj.weight", &[D, FFN]),
        ],
    );
    let spec = LlamaDecoderSpec {
        num_heads: HEADS,
        head_dim: HD,
        n_rot: HD,
        num_kv_heads: KV,
        eps: 1e-6,
        mask: MaskKind::Causal,
        hidden_shape: Shape::new(&[1, SEQ, D], DType::F32),
        rope_style: RopeStyle::NeoX,
    };
    let inv: Vec<f64> = (0..HD / 2)
        .map(|i| 1.0 / 10_000f64.powf(2.0 * i as f64 / HD as f64))
        .collect();
    ModelFlow::new("llama")
        .input("x", Shape::new(&[1, SEQ, D], DType::F32))
        .rope_tables(RopeTablesStage::from_inv_freq(SEQ, &inv))
        .zero_beta(D)
        .raw_stage(llama_prefill_layer_composed(0, spec))
        .build(&mut w)
        .expect("llama flow build")
        .into_graph()
        .expect("llama graph")
}

/// BERT encoder layer (post-LN, bias + GELU FFN).
fn bert_layer() -> Graph {
    let prefix = "encoder.layer.0";
    let mut entries: Vec<(String, Vec<usize>)> = Vec::new();
    for t in ["query", "key", "value"] {
        entries.push((format!("attention.self.{t}.weight"), vec![D, D]));
        entries.push((format!("attention.self.{t}.bias"), vec![D]));
    }
    for (k, shape) in [
        ("attention.output.dense.weight", vec![D, D]),
        ("attention.output.dense.bias", vec![D]),
        ("attention.output.LayerNorm.weight", vec![D]),
        ("attention.output.LayerNorm.bias", vec![D]),
        ("intermediate.dense.weight", vec![FFN, D]),
        ("intermediate.dense.bias", vec![FFN]),
        ("output.dense.weight", vec![D, FFN]),
        ("output.dense.bias", vec![D]),
        ("output.LayerNorm.weight", vec![D]),
        ("output.LayerNorm.bias", vec![D]),
    ] {
        entries.push((k.to_string(), shape));
    }
    let refs: Vec<(&str, &[usize])> = entries
        .iter()
        .map(|(k, s)| (k.as_str(), s.as_slice()))
        .collect();
    let mut w = weights(prefix, &refs);
    let spec = BertEncoderLayerSpec::hf(prefix, BertQkvStyle::Bert, D, HEADS, 1e-12);
    ModelFlow::new("bert")
        .input("x", Shape::new(&[1, SEQ, D], DType::F32))
        .input("attention_mask", Shape::new(&[1, SEQ], DType::F32))
        .bert_encoder_layer(spec)
        .build(&mut w)
        .expect("bert flow build")
        .into_graph()
        .expect("bert graph")
}

/// One block and what fusion must do to it.
struct Case {
    name: &'static str,
    build: fn() -> Graph,
    /// Producer control: op counts the unfused block must have.
    pattern: &'static [(OpKind, usize)],
    /// Exact op counts after fusion, on every device.
    fused: &'static [(OpKind, usize)],
}

const CASES: &[Case] = &[
    Case {
        name: "llama_prefill_layer",
        build: llama_layer,
        // q, k, v, o, gate, up, down; one SiLU.
        pattern: &[
            (OpKind::MatMul, 7),
            (OpKind::Activation, 1),
            (OpKind::RmsNorm, 2),
            (OpKind::Attention, 1),
        ],
        fused: &[
            // FuseSharedInputMatMul: q/k/v → one, gate/up → one.
            (OpKind::MatMul, 4),
            (OpKind::Concat, 2),
            // FuseSwiGLU: silu(gate)·up.
            (OpKind::FusedSwiGLU, 1),
            (OpKind::Activation, 0),
        ],
    },
    Case {
        name: "bert_encoder_layer",
        build: bert_layer,
        pattern: &[
            (OpKind::MatMul, 4),
            (OpKind::LayerNorm, 2),
            (OpKind::Attention, 1),
            (OpKind::Activation, 1),
        ],
        fused: &[
            // FuseMatMulBiasAct → FuseAttentionBlock → FuseResidualLN.
            (OpKind::FusedAttentionBlock, 1),
            (OpKind::FusedMatMulBiasAct, 2),
            (OpKind::FusedResidualLN, 2),
            (OpKind::MatMul, 0),
            (OpKind::LayerNorm, 0),
            (OpKind::Attention, 0),
        ],
    },
];

fn compiled(device: Device, g: &Graph, options: &CompileOptions) -> Counts {
    let r = compile_graph_stages(device, g.clone(), options);
    counts(r.lir.mir.as_graph())
}

#[test]
fn block_fusions_fire_on_every_device() {
    let devices = registered_devices();
    assert!(!devices.is_empty(), "no backend registered");
    let mut failures = Vec::new();
    for case in CASES {
        let g = (case.build)();
        let before = counts(&g);
        for &(k, n) in case.pattern {
            assert_eq!(
                count(&before, k),
                n,
                "{}: the block no longer emits {n}× {k:?} — the fusion checks would \
                 pass vacuously; update the case. Unfused: {before:?}",
                case.name
            );
        }
        for &device in &devices {
            let after = compiled(device, &g, &CompileOptions::default());
            for &(k, n) in case.fused {
                if count(&after, k) != n {
                    failures.push(format!(
                        "{} on {device:?}: expected {n}× {k:?}, got {}. Fused: {after:?}",
                        case.name,
                        count(&after, k)
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn fused_ops_come_from_fusion_not_the_block() {
    let mut off = CompileOptions::default();
    off.fusion_opts.skip_fusion = true;
    for case in CASES {
        let g = (case.build)();
        for device in registered_devices() {
            let after = compiled(device, &g, &off);
            for &(k, _) in case.fused {
                if !format!("{k:?}").starts_with("Fused") {
                    continue;
                }
                assert_eq!(
                    count(&after, k),
                    0,
                    "{} on {device:?}: {k:?} present with fusion off, so the fusion \
                     check proves nothing about the passes",
                    case.name
                );
            }
        }
    }
}

#[test]
fn fuse_entry_matches_runtime_compile() {
    // `fuse(graph, target)` is documented as the canonical per-target pipeline.
    // It fell back to a hand-copied claim list that had drifted (CUDA and ROCm
    // missed FusedSwiGLU, so `fuse` built an ElementwiseRegion where the
    // runtime builds FusedSwiGLU) and to options that kept regions CUDA unfuses.
    for case in CASES {
        let g = (case.build)();
        for device in registered_devices() {
            let via_fuse = counts(&fuse(g.clone(), fusion_target_for(device)));
            let via_runtime = compiled(device, &g, &CompileOptions::default());
            assert_eq!(
                via_fuse, via_runtime,
                "{} on {device:?}: `fuse` and the runtime compile disagree",
                case.name
            );
        }
    }
}

#[test]
fn fixed_claim_lists_match_backends_on_gating_kinds() {
    let gating = |ops: &[OpKind]| -> Vec<OpKind> {
        FUSION_GATING_KINDS
            .iter()
            .copied()
            .filter(|k| ops.contains(k))
            .collect()
    };
    let mut checked = 0;
    for device in registered_devices() {
        // Only devices whose fusion target is their own; Vulkan, ANE, … borrow
        // another backend's target and have their own claim sets.
        if !matches!(
            device,
            Device::Cpu
                | Device::Metal
                | Device::Mlx
                | Device::Gpu
                | Device::Cuda
                | Device::Rocm
                | Device::Tpu
        ) {
            continue;
        }
        let Some(backend) = backend_for(device) else {
            continue;
        };
        let target = fusion_target_for(device);
        assert_eq!(
            gating(supported_for_target(target)),
            gating(backend.supported_ops()),
            "`supported_for_target({target:?})` disagrees with {device:?}'s \
             SUPPORTED_OPS on fusion-gating kinds"
        );
        checked += 1;
    }
    assert!(checked > 0, "no device checked");
}
