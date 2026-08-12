// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HIR → MIR lowering.

use std::collections::HashMap;

use crate::hir::{HirModule, HirNodeId, HirOp, default_hir_block_label};
use crate::infer::GraphExt;
use crate::mir::MirModule;
use crate::provenance::NodeOrigin;
use crate::{Graph, NodeId, Op};

/// Lowering failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    WrongInputCount {
        op: &'static str,
        expected: &'static str,
        got: usize,
    },
    MissingBias {
        op: &'static str,
    },
    /// A panic during compilation (e.g. a `debug_assert_valid!` graph check on an
    /// invalid model graph) was caught and turned into a recoverable error so it
    /// can't abort the process — see `CompilePipeline::compile_hir`.
    Panicked {
        message: String,
    },
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongInputCount { op, expected, got } => {
                write!(f, "{op}: expected {expected} inputs, got {got}")
            }
            Self::MissingBias { op } => write!(f, "{op}: bias input required"),
            Self::Panicked { message } => write!(f, "compilation panicked: {message}"),
        }
    }
}

impl std::error::Error for LowerError {}

type SharedPairKey = (HirNodeId, HirNodeId, HirNodeId);

pub fn lower_module(hir: HirModule) -> Result<MirModule, LowerError> {
    let policy = hir.fusion_policy;
    let mut g = Graph::new(hir.name);
    let mut map: HashMap<HirNodeId, NodeId> = HashMap::new();
    let mut shared_pairs: HashMap<SharedPairKey, (NodeId, NodeId)> = HashMap::new();

    for node in hir.nodes {
        let hir_id = node.id;
        // Fast-path `Constant`: MOVE its byte buffer into the MIR graph instead
        // of cloning it. `hir` is consumed here, so the source is dropped anyway;
        // for large folded constants (RoPE cos/sin caches, embedding tables) this
        // avoids a full duplicate during lowering. `tag_hir_subgraph` is a no-op
        // for constants (its Input/Param/Constant arm), so nothing is skipped.
        if matches!(node.op, HirOp::Constant { .. }) {
            let data = match node.op {
                HirOp::Constant { data } => data,
                _ => unreachable!(),
            };
            let mir_id = g.add_node(Op::Constant { data }, vec![], node.shape);
            map.insert(hir_id, mir_id);
            continue;
        }
        let label = node_label_for_hir(&node);
        let inputs: Vec<NodeId> = node.inputs.iter().map(|&id| map[&id]).collect();

        let op = &node.op;
        // Capture MIR length before lowering so provenance tags only the
        // nodes this HIR op produces — the old prior-map HashSet scan was
        // O(N²) on large packed graphs (Qwen35 Bonsai ~90k nodes).
        let first_new = g.len();
        let mir_id = match op {
            HirOp::Input { name } => g.input(name.clone(), node.shape),
            HirOp::Param { name } => g.param(name.clone(), node.shape),
            HirOp::Constant { data } => {
                g.add_node(Op::Constant { data: data.clone() }, vec![], node.shape)
            }

            HirOp::Linear {
                activation,
                has_bias,
            } => {
                let expected = if *has_bias { 3 } else { 2 };
                if node.inputs.len() != expected {
                    return Err(LowerError::WrongInputCount {
                        op: "Linear",
                        expected: if *has_bias { "3" } else { "2" },
                        got: node.inputs.len(),
                    });
                }
                let x = inputs[0];
                let w = inputs[1];
                if policy.is_direct() && *has_bias {
                    let bias = inputs[2];
                    g.linear_fused(x, w, bias, *activation, node.shape)
                } else {
                    let bias = if *has_bias { Some(inputs[2]) } else { None };
                    g.linear_bias_act(x, w, bias, *activation)
                }
            }

            HirOp::LinearFused { activation } => {
                if node.inputs.len() != 3 {
                    return Err(LowerError::WrongInputCount {
                        op: "LinearFused",
                        expected: "3",
                        got: node.inputs.len(),
                    });
                }
                g.linear_fused(inputs[0], inputs[1], inputs[2], *activation, node.shape)
            }

            HirOp::SharedLinearPair { slot } => {
                if node.inputs.len() != 3 {
                    return Err(LowerError::WrongInputCount {
                        op: "SharedLinearPair",
                        expected: "3",
                        got: node.inputs.len(),
                    });
                }
                let key = (node.inputs[0], node.inputs[1], node.inputs[2]);
                let pair = *shared_pairs
                    .entry(key)
                    .or_insert_with(|| g.shared_matmul_pair(inputs[0], inputs[1], inputs[2]));
                if *slot == 0 { pair.0 } else { pair.1 }
            }

            HirOp::SwiGLU => {
                if node.inputs.len() != 4 {
                    return Err(LowerError::WrongInputCount {
                        op: "SwiGLU",
                        expected: "4",
                        got: node.inputs.len(),
                    });
                }
                if policy.is_direct() {
                    g.fused_swiglu_ffn(inputs[0], inputs[1], inputs[2], inputs[3], node.shape)
                } else {
                    g.swiglu_ffn(inputs[0], inputs[1], inputs[2], inputs[3])
                }
            }

            HirOp::ResidualRmsNorm { eps } => {
                if node.inputs.len() != 4 {
                    return Err(LowerError::WrongInputCount {
                        op: "ResidualRmsNorm",
                        expected: "4",
                        got: node.inputs.len(),
                    });
                }
                if policy.is_direct() {
                    g.fused_residual_rms_norm(
                        inputs[0], inputs[1], None, inputs[2], inputs[3], *eps, node.shape,
                    )
                } else {
                    let summed = g.add(inputs[0], inputs[1]);
                    g.rms_norm(summed, inputs[2], inputs[3], *eps)
                }
            }

            HirOp::Attention {
                num_heads,
                head_dim,
                mask,
            } => {
                use crate::op::MaskKind;
                if node.inputs.len()
                    != if matches!(mask, MaskKind::Custom | MaskKind::Bias) {
                        4
                    } else {
                        3
                    }
                {
                    return Err(LowerError::WrongInputCount {
                        op: "Attention",
                        expected: "3 or 4",
                        got: node.inputs.len(),
                    });
                }
                let q = inputs[0];
                let k = inputs[1];
                let v = inputs[2];
                match mask {
                    MaskKind::Custom => {
                        g.attention(q, k, v, inputs[3], *num_heads, *head_dim, node.shape)
                    }
                    MaskKind::Bias => {
                        g.attention_bias(q, k, v, inputs[3], *num_heads, *head_dim, node.shape)
                    }
                    other => g.attention_kind(q, k, v, *num_heads, *head_dim, *other, node.shape),
                }
            }

            HirOp::DepthwiseConv1dCausal { kernel_size } => {
                if node.inputs.len() != 3 {
                    return Err(LowerError::WrongInputCount {
                        op: "DepthwiseConv1dCausal",
                        expected: "3",
                        got: node.inputs.len(),
                    });
                }
                crate::hir::conv::lower_depthwise_conv1d_causal(
                    &mut g,
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    *kernel_size,
                    node.shape,
                )
            }

            HirOp::DequantMatMul { scheme } => {
                let expected = if scheme.is_gguf() { 2 } else { 4 };
                if node.inputs.len() != expected {
                    return Err(LowerError::WrongInputCount {
                        op: "DequantMatMul",
                        expected: if scheme.is_gguf() { "2" } else { "4" },
                        got: node.inputs.len(),
                    });
                }
                if scheme.is_gguf() {
                    g.dequant_matmul_packed(inputs[0], inputs[1], *scheme, node.shape)
                } else {
                    g.dequant_matmul(
                        inputs[0], inputs[1], inputs[2], inputs[3], *scheme, node.shape,
                    )
                }
            }

            HirOp::GatedDeltaNet {
                state_size,
                carry_state,
                gate_per_channel,
            } => {
                let expected = if *carry_state { 6 } else { 5 };
                if node.inputs.len() != expected {
                    return Err(LowerError::WrongInputCount {
                        op: "GatedDeltaNet",
                        expected: if *carry_state { "6" } else { "5" },
                        got: node.inputs.len(),
                    });
                }
                match (*carry_state, *gate_per_channel) {
                    (true, false) => g.gated_delta_net_carry(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        inputs[3],
                        inputs[4],
                        inputs[5],
                        *state_size,
                        node.shape,
                    ),
                    (true, true) => g.gated_delta_net_carry_pc(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        inputs[3],
                        inputs[4],
                        inputs[5],
                        *state_size,
                        node.shape,
                    ),
                    (false, false) => g.gated_delta_net(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        inputs[3],
                        inputs[4],
                        *state_size,
                        node.shape,
                    ),
                    (false, true) => g.gated_delta_net_pc(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        inputs[3],
                        inputs[4],
                        *state_size,
                        node.shape,
                    ),
                }
            }

            HirOp::Lstm {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
            } => {
                let expected = if *carry { 6 } else { 4 };
                if node.inputs.len() != expected {
                    return Err(LowerError::WrongInputCount {
                        op: "Lstm",
                        expected: if *carry { "6" } else { "4" },
                        got: node.inputs.len(),
                    });
                }
                if *carry {
                    g.lstm_carry(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        inputs[3],
                        inputs[4],
                        inputs[5],
                        *hidden_size,
                        *num_layers,
                        *bidirectional,
                        node.shape,
                    )
                } else {
                    g.lstm(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        inputs[3],
                        *hidden_size,
                        *num_layers,
                        *bidirectional,
                        node.shape,
                    )
                }
            }

            HirOp::Gru {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
            } => {
                let expected = if *carry { 6 } else { 5 };
                if node.inputs.len() != expected {
                    return Err(LowerError::WrongInputCount {
                        op: "Gru",
                        expected: if *carry { "6" } else { "5" },
                        got: node.inputs.len(),
                    });
                }
                let h0 = if *carry { Some(inputs[5]) } else { None };
                g.gru(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    h0,
                    *hidden_size,
                    *num_layers,
                    *bidirectional,
                    node.shape,
                )
            }

            HirOp::RoPE { head_dim, n_rot } => {
                if node.inputs.len() != 3 {
                    return Err(LowerError::WrongInputCount {
                        op: "RoPE",
                        expected: "3",
                        got: node.inputs.len(),
                    });
                }
                g.rope_n(inputs[0], inputs[1], inputs[2], *head_dim, *n_rot)
            }

            HirOp::RmsNorm { eps } => {
                if node.inputs.len() != 3 {
                    return Err(LowerError::WrongInputCount {
                        op: "RmsNorm",
                        expected: "3",
                        got: node.inputs.len(),
                    });
                }
                g.rms_norm(inputs[0], inputs[1], inputs[2], *eps)
            }

            HirOp::LlamaDecoderBlock {
                num_heads,
                head_dim,
                num_kv_heads,
                eps,
                mask,
                rope_style,
            } => crate::hir::blocks::lower_llama_decoder_block(
                &mut g,
                &inputs,
                *num_heads,
                *head_dim,
                *num_kv_heads,
                *eps,
                *mask,
                *rope_style,
                node.shape,
            )?,

            HirOp::Qwen35MtpHead {
                num_heads,
                num_kv_heads,
                head_dim,
                n_rot,
                n_embd,
                n_ff,
                mtp_vocab,
                eps,
            } => crate::hir::blocks::lower_qwen35_mtp_head(
                &mut g,
                &inputs,
                *num_heads,
                *num_kv_heads,
                *head_dim,
                *n_rot,
                *n_embd,
                *n_ff,
                *mtp_vocab,
                *eps,
                node.shape,
            )?,

            HirOp::Mir(op) => g.add_node(op.clone(), inputs, node.shape),
        };

        tag_hir_subgraph(&mut g, first_new, hir_id, &label);
        map.insert(hir_id, mir_id);
    }

    let outputs: Vec<NodeId> = hir.outputs.iter().map(|id| map[id]).collect();
    g.set_outputs(outputs);
    Ok(MirModule::from_graph(g))
}

fn node_label_for_hir(node: &crate::hir::HirNode) -> Option<String> {
    if let Some(ref n) = node.name {
        return Some(n.clone());
    }
    default_hir_block_label(&node.op)
}

/// Tag every MIR node produced from one HIR block with shared provenance.
fn tag_hir_subgraph(g: &mut Graph, first_new: usize, hir_id: HirNodeId, label: &Option<String>) {
    let origin = NodeOrigin::from_hir(hir_id, label.clone());
    for i in first_new..g.len() {
        let id = NodeId(i as u32);
        let node = g.node_mut(id);
        if node.origin.is_none() {
            node.origin = Some(origin.clone());
        }
        if node.name.is_none() {
            if let Some(l) = label {
                node.name = Some(l.clone());
            }
        }
    }
}
