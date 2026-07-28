// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Configurable transformer encoder layer matching PyTorch
//! `nn.TransformerEncoderLayer`.
//!
//! Composes a first-class in-graph block (pre- or post-norm, GELU or ReLU FFN)
//! so post-norm / ReLU trunks (PyTorch's default `nn.TransformerEncoderLayer`)
//! can be built without the custom-stage escape hatch. Weight keys follow the
//! `nn.TransformerEncoder(layers=ModuleList).layers.{i}` layout:
//!
//! ```text
//!   layers.{i}.self_attn.in_proj_weight    [3D, D]  (fused QKV)
//!   layers.{i}.self_attn.in_proj_bias      [3D]
//!   layers.{i}.self_attn.out_proj.weight   [D, D]
//!   layers.{i}.self_attn.out_proj.bias     [D]
//!   layers.{i}.linear1.weight              [FFN, D]
//!   layers.{i}.linear1.bias                [FFN]
//!   layers.{i}.linear2.weight              [D, FFN]
//!   layers.{i}.linear2.bias                [D]
//!   layers.{i}.norm1.{weight,bias}         [D]
//!   layers.{i}.norm2.{weight,bias}         [D]
//! ```
//!
//! Attention is `nn.MultiheadAttention` (fused QKV, scale `head_dim^-0.5`) via
//! the existing [`VitSelfAttnStage`]; there is **no** DINOv2 LayerScale. Requires
//! an all-attend mask from an [`AttnMaskStage`](super::AttnMaskStage) (see
//! [`ModelFlow::attn_mask_ones`](crate::ModelFlow::attn_mask_ones)).

use super::{FfnActivation, GeluFfnStage, VitSelfAttnSpec, VitSelfAttnStage};
use crate::layer::LayerStack;
use crate::stage::FlowStage;

/// Standard `nn.TransformerEncoder` per-layer weight prefix (`layers.{idx}`).
fn layer_prefix(idx: usize) -> String {
    format!("layers.{idx}")
}

/// Configurable `nn.TransformerEncoderLayer` block.
///
/// * `norm_first = true`  — pre-norm (LN → attn → +res, LN → FFN → +res), the
///   DINOv2 topology *without* LayerScale.
/// * `norm_first = false` — post-norm (attn → +res → LN, FFN → +res → LN), the
///   default `nn.TransformerEncoderLayer` / BERT topology.
///
/// `act` selects the FFN activation between `linear1` and `linear2`
/// ([`FfnActivation::Relu`] for the PyTorch default).
pub fn transformer_encoder_layer(
    layer_idx: usize,
    hidden_size: usize,
    num_heads: usize,
    eps: f32,
    norm_first: bool,
    act: FfnActivation,
) -> FlowStage {
    let lp = layer_prefix(layer_idx);

    let attn = FlowStage::VitSelfAttn(VitSelfAttnStage::new(VitSelfAttnSpec {
        qkv_weight: format!("{lp}.self_attn.in_proj_weight"),
        qkv_bias: format!("{lp}.self_attn.in_proj_bias"),
        out_weight: format!("{lp}.self_attn.out_proj.weight"),
        out_bias: format!("{lp}.self_attn.out_proj.bias"),
        hidden_size,
        num_heads,
        head_dim: hidden_size / num_heads,
    }));
    let ffn = FlowStage::GeluFfn(GeluFfnStage::with_activation(
        format!("{lp}.linear1.weight"),
        format!("{lp}.linear1.bias"),
        format!("{lp}.linear2.weight"),
        format!("{lp}.linear2.bias"),
        act,
    ));

    let (n1w, n1b) = (format!("{lp}.norm1.weight"), format!("{lp}.norm1.bias"));
    let (n2w, n2b) = (format!("{lp}.norm2.weight"), format!("{lp}.norm2.bias"));

    let stack = LayerStack::named(format!("layer{layer_idx}"));
    let stack = if norm_first {
        // pre-norm: x + attn(LN(x)); x + FFN(LN(x))
        stack
            .residual_save()
            .layer_norm(n1w, n1b, eps)
            .stage(attn)
            .residual_add()
            .residual_save()
            .layer_norm(n2w, n2b, eps)
            .stage(ffn)
            .residual_add()
    } else {
        // post-norm: LN(x + attn(x)); LN(x + FFN(x))
        stack
            .residual_save()
            .stage(attn)
            .residual_add()
            .layer_norm(n1w, n1b, eps)
            .residual_save()
            .stage(ffn)
            .residual_add()
            .layer_norm(n2w, n2b, eps)
    };
    stack.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::ModelFlow;
    use crate::weight::MapWeights;
    use rlx_ir::{DType, Shape};

    fn insert_layer(w: &mut MapWeights, idx: usize, d: usize, ffn: usize) {
        let lp = format!("layers.{idx}");
        // fused QKV (in_proj) + output projection
        w.insert(
            format!("{lp}.self_attn.in_proj_weight"),
            vec![0.01; 3 * d * d],
            vec![3 * d, d],
        );
        w.insert(
            format!("{lp}.self_attn.in_proj_bias"),
            vec![0.0; 3 * d],
            vec![3 * d],
        );
        w.insert(
            format!("{lp}.self_attn.out_proj.weight"),
            vec![0.01; d * d],
            vec![d, d],
        );
        w.insert(
            format!("{lp}.self_attn.out_proj.bias"),
            vec![0.0; d],
            vec![d],
        );
        // FFN
        w.insert(
            format!("{lp}.linear1.weight"),
            vec![0.01; ffn * d],
            vec![ffn, d],
        );
        w.insert(format!("{lp}.linear1.bias"), vec![0.0; ffn], vec![ffn]);
        w.insert(
            format!("{lp}.linear2.weight"),
            vec![0.01; d * ffn],
            vec![d, ffn],
        );
        w.insert(format!("{lp}.linear2.bias"), vec![0.0; d], vec![d]);
        // norms (identity affine)
        w.insert(format!("{lp}.norm1.weight"), vec![1.0; d], vec![d]);
        w.insert(format!("{lp}.norm1.bias"), vec![0.0; d], vec![d]);
        w.insert(format!("{lp}.norm2.weight"), vec![1.0; d], vec![d]);
        w.insert(format!("{lp}.norm2.bias"), vec![0.0; d], vec![d]);
    }

    fn build_trunk(depth: usize, norm_first: bool, act: FfnActivation) {
        let (batch, seq, d, heads, ffn) = (1usize, 3usize, 4usize, 2usize, 8usize);
        let mut w = MapWeights::default();
        for i in 0..depth {
            insert_layer(&mut w, i, d, ffn);
        }

        let built = ModelFlow::new("tel")
            .input("hidden", Shape::new(&[batch, seq, d], DType::F32))
            .attn_mask_ones(batch, seq)
            .repeat_transformer_encoder_layers(depth, d, heads, LN_EPS, norm_first, act)
            .output("hidden")
            .build(&mut w)
            .unwrap();

        // Shape is preserved end-to-end and the graph actually got built.
        let shape = built.primary_shape();
        assert_eq!(shape.rank(), 3);
        let got: Vec<usize> = shape.dims().iter().map(|d| d.unwrap_static()).collect();
        assert_eq!(got, vec![batch, seq, d]);
        assert!(built.into_hir().unwrap().len() >= depth * 8);
    }

    const LN_EPS: f32 = 1e-5;

    #[test]
    fn post_norm_relu_builds() {
        build_trunk(2, false, FfnActivation::Relu);
    }

    #[test]
    fn pre_norm_relu_builds() {
        build_trunk(2, true, FfnActivation::Relu);
    }

    #[test]
    fn pre_norm_gelu_builds() {
        build_trunk(1, true, FfnActivation::GeluErf);
        build_trunk(1, true, FfnActivation::GeluTanh);
    }
}
