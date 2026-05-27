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

//! **HIR** — high-level IR.
//!
//! Block-oriented IR for model authors and external graph builders.
//! HIR captures fusion-friendly patterns (SwiGLU FFN, linear layers,
//! residual RMSNorm) as first-class ops and lowers to MIR via
//! [`HirModule::lower_to_mir`].

mod blocks;
mod conv;
mod fusion;
mod graph_ext;
mod lower;

pub use blocks::lower_llama_decoder_block;
pub use blocks::lower_qwen35_mtp_head;
pub use fusion::FusionPolicy;
pub use graph_ext::{HirGraphExt, HirMut};

use crate::mir::MirModule;
use crate::op::Activation;
use crate::op::MaskKind;
use crate::quant::QuantScheme;
use crate::{Op, Shape};

pub use lower::LowerError;

/// Stable node identifier within a HIR module.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HirNodeId(pub u32);

impl std::fmt::Display for HirNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "h{}", self.0)
    }
}

/// High-level operation — blocks and escape hatches.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum HirOp {
    Input {
        name: String,
    },
    Param {
        name: String,
    },
    Constant {
        data: Vec<u8>,
    },

    /// `matmul → add(bias)? → activation?`
    /// Inputs: `[x, weight]` or `[x, weight, bias]`.
    Linear {
        activation: Option<Activation>,
        has_bias: bool,
    },

    /// Emit [`Op::FusedMatMulBiasAct`] directly.
    /// Inputs: `[x, weight, bias]`.
    LinearFused {
        activation: Option<Activation>,
    },

    /// Two matmuls sharing the same input (QKV / SwiGLU gate+up).
    /// Inputs: `[x, w_first, w_second]`. `slot` selects which output.
    SharedLinearPair {
        slot: u8,
    },

    /// Full SwiGLU FFN.
    /// Inputs: `[x, up_w, gate_w, down_w]`.
    SwiGLU,

    /// `add(x, residual)` then RMSNorm.
    /// Inputs: `[x, residual, gamma, beta]`.
    ResidualRmsNorm {
        eps: f32,
    },

    /// Scaled dot-product attention.
    /// Inputs: `[q, k, v, mask?]` — mask omitted when `mask == None`.
    Attention {
        num_heads: usize,
        head_dim: usize,
        mask: MaskKind,
    },

    /// Causal depthwise Conv1d on `[batch, seq, channels]` tensors.
    /// Inputs: `[input, weight, left_pad]` — see [`conv::lower_depthwise_conv1d_causal`].
    DepthwiseConv1dCausal {
        kernel_size: usize,
    },

    /// Fused dequant + matmul. GGUF schemes take `[x, packed_w]`; legacy
    /// Int8/NVFP4 schemes take `[x, w_q, scale, zp]`.
    DequantMatMul {
        scheme: QuantScheme,
    },

    /// Gated DeltaNet linear-attention scan (Qwen3.5 trunk).
    /// Inputs: `[q, k, v, g, beta]` or with carry `[…, state]`.
    GatedDeltaNet {
        state_size: usize,
        carry_state: bool,
    },

    /// Rotary position embedding. Inputs: `[x, cos, sin]`.
    RoPE {
        head_dim: usize,
        n_rot: usize,
    },

    /// RMS normalization without residual. Inputs: `[x, gamma, beta]`.
    RmsNorm {
        eps: f32,
    },

    /// LLaMA-style pre-norm decoder block: attn (GQA) + SwiGLU FFN.
    /// Inputs (causal): `[x, ln1_g, ln1_b, q_w, k_w, v_w, o_w, ln2_g, ln2_b,
    /// gate_w, up_w, down_w, cos, sin]`. With `MaskKind::Custom` or `Bias`
    /// append `mask`.
    LlamaDecoderBlock {
        num_heads: usize,
        head_dim: usize,
        num_kv_heads: usize,
        eps: f32,
        mask: MaskKind,
    },

    /// Qwen3.5 MTP draft head: hnorm∥enorm → eh_proj → full-attn → LM.
    /// See [`blocks::lower_qwen35_mtp_head`] for the input layout.
    Qwen35MtpHead {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        n_embd: usize,
        n_ff: usize,
        mtp_vocab: usize,
        eps: f32,
    },

    /// Escape hatch — embed a single MIR op verbatim.
    Mir(Op),
}

/// One node in a HIR module.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct HirNode {
    pub id: HirNodeId,
    pub op: HirOp,
    pub inputs: Vec<HirNodeId>,
    pub shape: Shape,
    pub name: Option<String>,
}

/// High-level module — model builder output.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct HirModule {
    pub name: String,
    nodes: Vec<HirNode>,
    pub outputs: Vec<HirNodeId>,
    /// How block ops lower to MIR. Default: [`FusionPolicy::Direct`]
    /// for new model code (fusion as a first-class citizen).
    pub fusion_policy: FusionPolicy,
}

impl HirModule {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            outputs: Vec::new(),
            fusion_policy: FusionPolicy::Direct,
        }
    }

    pub fn with_fusion_policy(mut self, policy: FusionPolicy) -> Self {
        self.fusion_policy = policy;
        self
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn nodes(&self) -> &[HirNode] {
        &self.nodes
    }

    pub fn node(&self, id: HirNodeId) -> &HirNode {
        &self.nodes[id.0 as usize]
    }

    pub fn node_mut(&mut self, id: HirNodeId) -> &mut HirNode {
        &mut self.nodes[id.0 as usize]
    }

    /// Build a named block — sets `HirNode::name` on the returned node.
    pub fn named(
        &mut self,
        name: impl Into<String>,
        build: impl FnOnce(&mut Self) -> HirNodeId,
    ) -> HirNodeId {
        let id = build(self);
        self.node_mut(id).name = Some(name.into());
        id
    }

    fn push_block(
        &mut self,
        op: HirOp,
        inputs: Vec<HirNodeId>,
        shape: Shape,
        name: Option<String>,
    ) -> HirNodeId {
        let name = name.or_else(|| default_hir_block_label(&op));
        self.push(op, inputs, shape, name)
    }

    fn push(
        &mut self,
        op: HirOp,
        inputs: Vec<HirNodeId>,
        shape: Shape,
        name: Option<String>,
    ) -> HirNodeId {
        let id = HirNodeId(self.nodes.len() as u32);
        self.nodes.push(HirNode {
            id,
            op,
            inputs,
            shape,
            name,
        });
        id
    }

    pub fn input(&mut self, name: impl Into<String>, shape: Shape) -> HirNodeId {
        self.push(HirOp::Input { name: name.into() }, vec![], shape, None)
    }

    /// `[batch, seq, hidden]` input with symbolic leading axes.
    pub fn input_batch_seq(
        &mut self,
        name: impl Into<String>,
        batch: u32,
        seq: u32,
        hidden: usize,
        dtype: crate::DType,
    ) -> HirNodeId {
        self.input(name, Shape::batch_seq(batch, seq, hidden, dtype))
    }

    pub fn param(&mut self, name: impl Into<String>, shape: Shape) -> HirNodeId {
        self.push(HirOp::Param { name: name.into() }, vec![], shape, None)
    }

    pub fn linear(
        &mut self,
        x: HirNodeId,
        weight: HirNodeId,
        bias: Option<HirNodeId>,
        activation: Option<Activation>,
        out_shape: Shape,
    ) -> HirNodeId {
        let mut inputs = vec![x, weight];
        if let Some(b) = bias {
            inputs.push(b);
        }
        self.push_block(
            HirOp::Linear {
                activation,
                has_bias: bias.is_some(),
            },
            inputs,
            out_shape,
            None,
        )
    }

    /// Emit [`HirOp::LinearFused`] — fused matmul+bias+act at MIR level.
    pub fn linear_fused(
        &mut self,
        x: HirNodeId,
        weight: HirNodeId,
        bias: HirNodeId,
        activation: Option<Activation>,
        out_shape: Shape,
    ) -> HirNodeId {
        self.push_block(
            HirOp::LinearFused { activation },
            vec![x, weight, bias],
            out_shape,
            None,
        )
    }

    /// Two matmuls sharing `x`. Returns `(first, second)` in weight order.
    pub fn shared_linear_pair(
        &mut self,
        x: HirNodeId,
        w_first: HirNodeId,
        w_second: HirNodeId,
        out_shape: Shape,
    ) -> (HirNodeId, HirNodeId) {
        let inputs = vec![x, w_first, w_second];
        let first = self.push_block(
            HirOp::SharedLinearPair { slot: 0 },
            inputs.clone(),
            out_shape.clone(),
            None,
        );
        let second = self.push_block(HirOp::SharedLinearPair { slot: 1 }, inputs, out_shape, None);
        (first, second)
    }

    pub fn swiglu_ffn(
        &mut self,
        x: HirNodeId,
        up_w: HirNodeId,
        gate_w: HirNodeId,
        down_w: HirNodeId,
        out_shape: Shape,
    ) -> HirNodeId {
        self.push_block(
            HirOp::SwiGLU,
            vec![x, up_w, gate_w, down_w],
            out_shape,
            None,
        )
    }

    pub fn residual_rms_norm(
        &mut self,
        x: HirNodeId,
        residual: HirNodeId,
        gamma: HirNodeId,
        beta: HirNodeId,
        eps: f32,
        out_shape: Shape,
    ) -> HirNodeId {
        self.push_block(
            HirOp::ResidualRmsNorm { eps },
            vec![x, residual, gamma, beta],
            out_shape,
            None,
        )
    }

    /// Scaled dot-product attention — see [`HirOp::Attention`].
    pub fn attention(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        mask: Option<HirNodeId>,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        out_shape: Shape,
    ) -> HirNodeId {
        let mut inputs = vec![q, k, v];
        if let Some(m) = mask {
            inputs.push(m);
        }
        self.push_block(
            HirOp::Attention {
                num_heads,
                head_dim,
                mask: mask_kind,
            },
            inputs,
            out_shape,
            None,
        )
    }

    /// Causal depthwise Conv1d — Conformer / Wav2Vec2-BERT conv module.
    ///
    /// `input` and `left_pad` are `[B, S, C]` / `[B, K-1, C]`; `weight` is
    /// `[C, 1, 1, K]` in grouped Conv2d layout.
    pub fn depthwise_conv1d_causal(
        &mut self,
        input: HirNodeId,
        weight: HirNodeId,
        left_pad: HirNodeId,
        kernel_size: usize,
        out_shape: Shape,
    ) -> HirNodeId {
        self.push_block(
            HirOp::DepthwiseConv1dCausal { kernel_size },
            vec![input, weight, left_pad],
            out_shape,
            None,
        )
    }

    /// Fused dequant + matmul — see [`HirOp::DequantMatMul`].
    pub fn dequant_matmul(
        &mut self,
        x: HirNodeId,
        w: HirNodeId,
        scale: Option<HirNodeId>,
        zp: Option<HirNodeId>,
        scheme: QuantScheme,
        out_shape: Shape,
    ) -> HirNodeId {
        let mut inputs = vec![x, w];
        if !scheme.is_gguf() {
            inputs.push(scale.expect("DequantMatMul: scale required for non-GGUF schemes"));
            inputs.push(zp.expect("DequantMatMul: zp required for non-GGUF schemes"));
        }
        self.push_block(HirOp::DequantMatMul { scheme }, inputs, out_shape, None)
    }

    /// Gated DeltaNet without carry state (prefill / reset per batch).
    pub fn gated_delta_net(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state_size: usize,
        out_shape: Shape,
    ) -> HirNodeId {
        self.push_block(
            HirOp::GatedDeltaNet {
                state_size,
                carry_state: false,
            },
            vec![q, k, v, g, beta],
            out_shape,
            None,
        )
    }

    /// Gated DeltaNet with decode carry — threads `state` in/out.
    pub fn gated_delta_net_carry(
        &mut self,
        q: HirNodeId,
        k: HirNodeId,
        v: HirNodeId,
        g: HirNodeId,
        beta: HirNodeId,
        state: HirNodeId,
        state_size: usize,
        out_shape: Shape,
    ) -> HirNodeId {
        self.push_block(
            HirOp::GatedDeltaNet {
                state_size,
                carry_state: true,
            },
            vec![q, k, v, g, beta, state],
            out_shape,
            None,
        )
    }

    /// Rotary position embedding.
    pub fn rope(
        &mut self,
        x: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        head_dim: usize,
        n_rot: usize,
        out_shape: Shape,
    ) -> HirNodeId {
        self.push_block(
            HirOp::RoPE { head_dim, n_rot },
            vec![x, cos, sin],
            out_shape,
            None,
        )
    }

    /// RMS normalization (no residual add).
    pub fn rms_norm(
        &mut self,
        x: HirNodeId,
        gamma: HirNodeId,
        beta: HirNodeId,
        eps: f32,
        out_shape: Shape,
    ) -> HirNodeId {
        self.push_block(
            HirOp::RmsNorm { eps },
            vec![x, gamma, beta],
            out_shape,
            None,
        )
    }

    /// LLaMA / LLaMA-3.2 decoder layer (pre-norm GQA + SwiGLU).
    pub fn llama_decoder_block(
        &mut self,
        x: HirNodeId,
        ln1_g: HirNodeId,
        ln1_b: HirNodeId,
        q_w: HirNodeId,
        k_w: HirNodeId,
        v_w: HirNodeId,
        o_w: HirNodeId,
        ln2_g: HirNodeId,
        ln2_b: HirNodeId,
        gate_w: HirNodeId,
        up_w: HirNodeId,
        down_w: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        mask: Option<HirNodeId>,
        num_heads: usize,
        head_dim: usize,
        num_kv_heads: usize,
        eps: f32,
        mask_kind: MaskKind,
        out_shape: Shape,
    ) -> HirNodeId {
        let mut ins = vec![
            x, ln1_g, ln1_b, q_w, k_w, v_w, o_w, ln2_g, ln2_b, gate_w, up_w, down_w, cos, sin,
        ];
        if let Some(m) = mask {
            ins.push(m);
        }
        self.push_block(
            HirOp::LlamaDecoderBlock {
                num_heads,
                head_dim,
                num_kv_heads,
                eps,
                mask: mask_kind,
            },
            ins,
            out_shape,
            Some("llama_decoder_block".into()),
        )
    }

    /// Standard pre-norm transformer decoder block — alias for
    /// [`Self::llama_decoder_block`] (LLaMA / GPT-style layers).
    pub fn transformer_block(
        &mut self,
        x: HirNodeId,
        ln1_g: HirNodeId,
        ln1_b: HirNodeId,
        q_w: HirNodeId,
        k_w: HirNodeId,
        v_w: HirNodeId,
        o_w: HirNodeId,
        ln2_g: HirNodeId,
        ln2_b: HirNodeId,
        gate_w: HirNodeId,
        up_w: HirNodeId,
        down_w: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        mask: Option<HirNodeId>,
        num_heads: usize,
        head_dim: usize,
        num_kv_heads: usize,
        eps: f32,
        mask_kind: MaskKind,
        out_shape: Shape,
    ) -> HirNodeId {
        let id = self.llama_decoder_block(
            x,
            ln1_g,
            ln1_b,
            q_w,
            k_w,
            v_w,
            o_w,
            ln2_g,
            ln2_b,
            gate_w,
            up_w,
            down_w,
            cos,
            sin,
            mask,
            num_heads,
            head_dim,
            num_kv_heads,
            eps,
            mask_kind,
            out_shape,
        );
        self.node_mut(id).name = Some("transformer_block".into());
        id
    }

    /// Qwen3.5 MTP draft head — see [`blocks::lower_qwen35_mtp_head`].
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_mtp_head(
        &mut self,
        h_pre_norm: HirNodeId,
        input_ids: HirNodeId,
        cos: HirNodeId,
        sin: HirNodeId,
        last_token_idx: HirNodeId,
        embed_w: HirNodeId,
        hnorm_w: HirNodeId,
        hnorm_b: HirNodeId,
        enorm_w: HirNodeId,
        enorm_b: HirNodeId,
        eh_w: HirNodeId,
        fa_attn_norm_w: HirNodeId,
        fa_attn_norm_b: HirNodeId,
        fa_q_gate_w: HirNodeId,
        fa_k_w: HirNodeId,
        fa_v_w: HirNodeId,
        fa_q_norm_w: HirNodeId,
        fa_q_norm_b: HirNodeId,
        fa_k_norm_w: HirNodeId,
        fa_k_norm_b: HirNodeId,
        fa_o_w: HirNodeId,
        fa_post_norm_w: HirNodeId,
        fa_post_norm_b: HirNodeId,
        fa_gate_w: HirNodeId,
        fa_up_w: HirNodeId,
        fa_down_w: HirNodeId,
        head_norm_w: HirNodeId,
        head_norm_b: HirNodeId,
        lm_head_w: HirNodeId,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        n_rot: usize,
        n_embd: usize,
        n_ff: usize,
        mtp_vocab: usize,
        eps: f32,
        out_shape: Shape,
    ) -> HirNodeId {
        self.push_block(
            HirOp::Qwen35MtpHead {
                num_heads,
                num_kv_heads,
                head_dim,
                n_rot,
                n_embd,
                n_ff,
                mtp_vocab,
                eps,
            },
            vec![
                h_pre_norm,
                input_ids,
                cos,
                sin,
                last_token_idx,
                embed_w,
                hnorm_w,
                hnorm_b,
                enorm_w,
                enorm_b,
                eh_w,
                fa_attn_norm_w,
                fa_attn_norm_b,
                fa_q_gate_w,
                fa_k_w,
                fa_v_w,
                fa_q_norm_w,
                fa_q_norm_b,
                fa_k_norm_w,
                fa_k_norm_b,
                fa_o_w,
                fa_post_norm_w,
                fa_post_norm_b,
                fa_gate_w,
                fa_up_w,
                fa_down_w,
                head_norm_w,
                head_norm_b,
                lm_head_w,
            ],
            out_shape,
            Some("qwen35_mtp_head".into()),
        )
    }

    /// Escape hatch — embed a single MIR [`Op`] verbatim.
    pub fn mir(&mut self, op: Op, inputs: Vec<HirNodeId>, shape: Shape) -> HirNodeId {
        self.push(HirOp::Mir(op), inputs, shape, None)
    }

    pub fn set_outputs(&mut self, outputs: Vec<HirNodeId>) {
        self.outputs = outputs;
    }

    /// Lower this module to MIR.
    pub fn lower_to_mir(self) -> Result<MirModule, LowerError> {
        lower::lower_module(self)
    }

    /// Lower with [`FusionPolicy::for_autodiff`] — primitive MIR chains
    /// that need less unfuse work before `rlx_opt::prepare_graph_for_ad`.
    pub fn lower_for_autodiff(self) -> Result<MirModule, LowerError> {
        self.with_fusion_policy(FusionPolicy::for_autodiff())
            .lower_to_mir()
    }

    /// Wrap an existing MIR [`Graph`] as a HIR module (`HirOp::Mir` per node).
    /// Enables `Session::compile_hir` for legacy graph builders during migration.
    pub fn wrap_mir_graph(graph: crate::Graph) -> Self {
        use std::collections::HashMap;
        let mut hir = Self::new(graph.name.clone()).with_fusion_policy(FusionPolicy::Direct);
        let mut map: HashMap<crate::NodeId, HirNodeId> = HashMap::new();
        for node in graph.nodes() {
            let inputs: Vec<HirNodeId> = node.inputs.iter().map(|&id| map[&id]).collect();
            let id = hir.mir(node.op.clone(), inputs, node.shape.clone());
            map.insert(node.id, id);
        }
        let outputs: Vec<HirNodeId> = graph.outputs.iter().map(|&id| map[&id]).collect();
        hir.set_outputs(outputs);
        hir
    }
}

pub(crate) fn default_hir_block_label(op: &HirOp) -> Option<String> {
    Some(match op {
        HirOp::Linear { .. } => "linear".into(),
        HirOp::LinearFused { .. } => "linear_fused".into(),
        HirOp::SharedLinearPair { slot } => return Some(format!("shared_linear_pair[{slot}]")),
        HirOp::SwiGLU => "swiglu_ffn".into(),
        HirOp::ResidualRmsNorm { .. } => "residual_rms_norm".into(),
        HirOp::Attention { .. } => "attention".into(),
        HirOp::DepthwiseConv1dCausal { .. } => "depthwise_conv1d_causal".into(),
        HirOp::DequantMatMul { scheme } => format!("dequant_matmul({scheme})"),
        HirOp::GatedDeltaNet {
            carry_state: true, ..
        } => "gated_delta_net_carry".into(),
        HirOp::GatedDeltaNet { .. } => "gated_delta_net".into(),
        HirOp::RoPE { .. } => "rope".into(),
        HirOp::RmsNorm { .. } => "rms_norm".into(),
        HirOp::Mir(_) => "mir".into(),
        HirOp::LlamaDecoderBlock { .. } => "llama_decoder_block".into(),
        HirOp::Qwen35MtpHead { .. } => "qwen35_mtp_head".into(),
        HirOp::Input { .. } | HirOp::Param { .. } | HirOp::Constant { .. } => return None,
    })
}

impl std::fmt::Display for HirModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "hir @{} {{", self.name)?;
        for node in &self.nodes {
            write!(f, "  {} = {:?}", node.id, node.op)?;
            if !node.inputs.is_empty() {
                write!(f, "(")?;
                for (i, inp) in node.inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{inp}")?;
                }
                write!(f, ")")?;
            }
            writeln!(f, " : {}", node.shape)?;
        }
        if !self.outputs.is_empty() {
            write!(f, "  return ")?;
            for (i, o) in self.outputs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{o}")?;
            }
            writeln!(f)?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    fn f32_shape(d: &[usize]) -> Shape {
        Shape::new(d, DType::F32)
    }

    #[test]
    fn hir_depthwise_conv1d_causal_lowers_to_grouped_conv() {
        use crate::Op;

        let mut hir = HirModule::new("dw");
        let x = hir.input("x", f32_shape(&[2, 8, 16]));
        let w = hir.param("w", f32_shape(&[16, 1, 1, 3]));
        let pad = hir.param("pad", f32_shape(&[2, 2, 16]));
        let out = hir.depthwise_conv1d_causal(x, w, pad, 3, f32_shape(&[2, 8, 16]));
        hir.outputs = vec![out];

        let g = hir.lower_to_mir().expect("lower").into_graph();
        assert!(g.nodes().iter().any(|n| matches!(n.op, Op::Conv { .. })));
        assert!(g.nodes().iter().any(|n| matches!(n.op, Op::Concat { .. })));
    }

    #[test]
    fn hir_swiglu_lowers_to_fusable_mir() {
        use crate::Op;
        use crate::hir::FusionPolicy;

        let mut hir = HirModule::new("ffn").with_fusion_policy(FusionPolicy::Fusable);
        let x = hir.input("x", f32_shape(&[4, 768]));
        let up_w = hir.param("up", f32_shape(&[768, 2048]));
        let gate_w = hir.param("gate", f32_shape(&[768, 2048]));
        let down_w = hir.param("down", f32_shape(&[2048, 768]));
        let out = hir.swiglu_ffn(x, up_w, gate_w, down_w, f32_shape(&[4, 768]));
        hir.set_outputs(vec![out]);

        let mir = hir.lower_to_mir().expect("lower");
        let g = mir.into_graph();
        assert!(g.nodes().iter().any(|n| matches!(n.op, Op::MatMul)));
        assert_eq!(g.len(), 9);
    }

    #[test]
    fn hir_gdn_dequant_rope_rms_lowers() {
        use crate::Op;
        use crate::quant::QuantScheme;

        let mut hir = HirModule::new("qwen_block");
        let q = hir.input("q", f32_shape(&[1, 4, 2, 8]));
        let k = hir.param("k", f32_shape(&[1, 4, 2, 8]));
        let v = hir.param("v", f32_shape(&[1, 4, 2, 8]));
        let g_in = hir.param("g", f32_shape(&[1, 4, 2]));
        let beta = hir.param("beta", f32_shape(&[1, 4, 2]));
        let scan = hir.gated_delta_net(q, k, v, g_in, beta, 8, f32_shape(&[1, 4, 2, 8]));

        let cos = hir.param("cos", f32_shape(&[1, 4, 8]));
        let sin = hir.param("sin", f32_shape(&[1, 4, 8]));
        let x = hir.input("x", f32_shape(&[1, 4, 8]));
        let rotated = hir.rope(x, cos, sin, 8, 8, f32_shape(&[1, 4, 8]));

        let gamma = hir.param("gamma", f32_shape(&[8]));
        let beta_n = hir.param("beta_n", f32_shape(&[8]));
        let normed = hir.rms_norm(rotated, gamma, beta_n, 1e-6, f32_shape(&[1, 4, 8]));

        let x_in = hir.input("hidden", f32_shape(&[4, 128]));
        let w = hir.param("w_q", f32_shape(&[1024]));
        let proj = hir.dequant_matmul(
            x_in,
            w,
            None,
            None,
            QuantScheme::GgufQ4K,
            f32_shape(&[4, 128]),
        );
        hir.set_outputs(vec![scan, normed, proj]);

        let g = hir.lower_to_mir().expect("lower").into_graph();
        assert!(g.nodes().iter().any(|n| matches!(
            n.op,
            Op::GatedDeltaNet {
                carry_state: false,
                ..
            }
        )));
        assert!(g.nodes().iter().any(|n| matches!(n.op, Op::Rope { .. })));
        assert!(g.nodes().iter().any(|n| matches!(n.op, Op::RmsNorm { .. })));
        assert!(g.nodes().iter().any(|n| matches!(
            n.op,
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ4K
            }
        )));
    }
}
