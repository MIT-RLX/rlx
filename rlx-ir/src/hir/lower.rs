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

//! HIR → MIR lowering.

use std::collections::HashMap;

use crate::hir::{default_hir_block_label, HirModule, HirNodeId, HirOp};
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
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongInputCount {
                op,
                expected,
                got,
            } => write!(f, "{op}: expected {expected} inputs, got {got}"),
            Self::MissingBias { op } => write!(f, "{op}: bias input required"),
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
        let label = node_label_for_hir(&node);
        let inputs: Vec<NodeId> = node.inputs.iter().map(|&id| map[&id]).collect();

        let op = &node.op;
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
                let pair = *shared_pairs.entry(key).or_insert_with(|| {
                    g.shared_matmul_pair(inputs[0], inputs[1], inputs[2])
                });
                if *slot == 0 {
                    pair.0
                } else {
                    pair.1
                }
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
                        inputs[0],
                        inputs[1],
                        None,
                        inputs[2],
                        inputs[3],
                        *eps,
                        node.shape,
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
                    MaskKind::Bias => g.attention_bias(
                        q,
                        k,
                        v,
                        inputs[3],
                        *num_heads,
                        *head_dim,
                        node.shape,
                    ),
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
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        inputs[3],
                        *scheme,
                        node.shape,
                    )
                }
            }

            HirOp::GatedDeltaNet {
                state_size,
                carry_state,
            } => {
                let expected = if *carry_state { 6 } else { 5 };
                if node.inputs.len() != expected {
                    return Err(LowerError::WrongInputCount {
                        op: "GatedDeltaNet",
                        expected: if *carry_state { "6" } else { "5" },
                        got: node.inputs.len(),
                    });
                }
                if *carry_state {
                    g.gated_delta_net_carry(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        inputs[3],
                        inputs[4],
                        inputs[5],
                        *state_size,
                        node.shape,
                    )
                } else {
                    g.gated_delta_net(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        inputs[3],
                        inputs[4],
                        *state_size,
                        node.shape,
                    )
                }
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
            } => crate::hir::blocks::lower_llama_decoder_block(
                &mut g,
                &inputs,
                *num_heads,
                *head_dim,
                *num_kv_heads,
                *eps,
                *mask,
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

        tag_hir_subgraph(&mut g, mir_id, hir_id, &label, &map, op);
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
fn tag_hir_subgraph(
    g: &mut Graph,
    root: NodeId,
    hir_id: HirNodeId,
    label: &Option<String>,
    prior: &HashMap<HirNodeId, NodeId>,
    op: &HirOp,
) {
    let prior_ids: std::collections::HashSet<NodeId> = prior.values().copied().collect();
    let origin = NodeOrigin::from_hir(hir_id, label.clone());
    let start = match op {
        HirOp::Input { .. } | HirOp::Param { .. } | HirOp::Constant { .. } => root.0,
        _ => prior_ids
            .iter()
            .map(|id| id.0)
            .min()
            .map(|i| i + 1)
            .unwrap_or(0),
    };
    for i in start..=root.0 {
        let id = NodeId(i);
        if !prior_ids.contains(&id) {
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
}
