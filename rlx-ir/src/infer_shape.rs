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

//! Re-derive output shapes from inputs — used by the verifier to catch
//! builder / pass bugs that assign the wrong `Node::shape`.

use crate::op::*;
use crate::shape;
use crate::{DType, Graph, Node, Shape};

/// Infer the output shape of `node` from its op and input shapes.
///
/// Returns `None` when inference is not implemented for the op (the
/// verifier skips those nodes rather than failing open).
pub fn infer_output_shape(graph: &Graph, node: &Node) -> Option<Shape> {
    let in_shape = |i: usize| graph.shape(node.inputs[i]);
    match &node.op {
        Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => None,

        Op::MatMul => shape::matmul_shape(in_shape(0), in_shape(1)).ok(),
        Op::Binary(_) => shape::binary_shape(in_shape(0), in_shape(1)).ok(),
        Op::Compare(_) => shape::compare_shape(in_shape(0), in_shape(1)).ok(),
        Op::Where => shape::binary_shape(in_shape(1), in_shape(2)).ok(),

        Op::Activation(_) | Op::ReluBackward | Op::Conjugate => {
            Some(shape::unary_shape(in_shape(0)))
        }
        Op::ComplexNormSq => Some(Shape::from_dims(in_shape(0).dims(), DType::F32)),
        Op::ComplexNormSqBackward => Some(shape::unary_shape(in_shape(0))),
        Op::Cast { to } => Some(shape::cast_shape(in_shape(0), *to)),

        Op::Reduce { axes, keep_dim, .. } => shape::reduce_shape(in_shape(0), axes, *keep_dim).ok(),
        Op::Softmax { .. } => Some(shape::softmax_shape(in_shape(0))),
        Op::Cumsum { .. } => Some(shape::unary_shape(in_shape(0))),

        Op::Reshape { new_shape } => shape::reshape_shape(in_shape(0), new_shape).ok(),
        Op::Transpose { perm } => shape::transpose_shape(in_shape(0), perm).ok(),
        Op::Narrow { axis, len, .. } => shape::narrow_shape(in_shape(0), *axis, *len).ok(),
        Op::Concat { axis } => {
            let inputs: Vec<&Shape> = node.inputs.iter().map(|&id| graph.shape(id)).collect();
            shape::concat_shape(&inputs, *axis).ok()
        }
        Op::Gather { axis } => shape::gather_shape(in_shape(0), in_shape(1), *axis).ok(),
        Op::Expand { target_shape } => {
            if target_shape.iter().any(|&d| d < 0) {
                return None;
            }
            let dtype = in_shape(0).dtype();
            Some(Shape::new(
                &target_shape.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                dtype,
            ))
        }

        Op::LayerNorm { .. } | Op::LayerNorm2d { .. } | Op::GroupNorm { .. } => {
            Some(shape::unary_shape(in_shape(0)))
        }
        Op::RmsNorm { .. } => {
            let in_s = in_shape(0);
            let out = &node.shape;
            // `FuseRmsNormReshape` keeps the 3-D (or higher) input but
            // assigns a leading-flattened `[∏leading, H]` output shape.
            if out.rank() == 2 && in_s.rank() > 2 {
                if let Some(flat) = shape::leading_flatten_fused_shape(in_s) {
                    if flat == *out {
                        return Some(out.clone());
                    }
                }
            }
            Some(shape::unary_shape(in_s))
        }
        Op::ResizeNearest2x => {
            let in_s = in_shape(0);
            if in_s.rank() == 4 {
                Some(Shape::new(
                    &[
                        in_s.dim(0).unwrap_static(),
                        in_s.dim(1).unwrap_static(),
                        in_s.dim(2).unwrap_static() * 2,
                        in_s.dim(3).unwrap_static() * 2,
                    ],
                    in_s.dtype(),
                ))
            } else {
                None
            }
        }
        Op::Attention { .. } => Some(shape::attention_shape(in_shape(0))),
        Op::Rope { .. } => Some(shape::unary_shape(in_shape(0))),
        Op::AxialRope2d { .. } => Some(shape::unary_shape(in_shape(0))),

        Op::FusedMatMulBiasAct { .. } => shape::matmul_shape(in_shape(0), in_shape(1)).ok(),
        Op::FusedSwiGLU { .. } => None,
        Op::FusedResidualLN { .. } | Op::FusedResidualRmsNorm { .. } => {
            Some(shape::unary_shape(in_shape(0)))
        }

        Op::DequantMatMul { .. } | Op::LoraMatMul { .. } | Op::QMatMul { .. } => {
            shape::matmul_shape(in_shape(0), in_shape(1)).ok()
        }

        Op::GaussianSplatRender { width, height, .. } => Some(Shape::new(
            &[(*width as usize) * (*height as usize) * 4],
            in_shape(0).dtype(),
        )),

        Op::GaussianSplatRenderBackward { .. } => {
            let count = in_shape(0).num_elements().unwrap_or(0) / 3;
            let sh_len = in_shape(5).num_elements().unwrap_or(0);
            let sh_coeff_count = if count == 0 {
                1
            } else {
                (sh_len / (count * 3)).max(1)
            };
            let packed = crate::ops::splat::gaussian_splat_packed_grad_len(count, sh_coeff_count);
            Some(Shape::new(&[packed], in_shape(0).dtype()))
        }

        Op::GaussianSplatPrepare {
            width,
            height,
            tile_size,
            max_list_entries,
            ..
        } => {
            let count = in_shape(0).num_elements().unwrap_or(0) / 3;
            let len = crate::ops::splat::gaussian_splat_prep_packed_len(
                count,
                *max_list_entries,
                *width,
                *height,
                *tile_size,
            );
            Some(Shape::new(&[len], in_shape(0).dtype()))
        }

        Op::GaussianSplatRasterize { width, height, .. } => Some(Shape::new(
            &[(*width as usize) * (*height as usize) * 4],
            in_shape(0).dtype(),
        )),

        Op::DotGeneral { .. }
        | Op::If { .. }
        | Op::While { .. }
        | Op::Scan { .. }
        | Op::SelectiveScan { .. }
        | Op::GatedDeltaNet { .. }
        | Op::FusedAttentionBlock { .. }
        | Op::FusedTransformerLayer { .. }
        | Op::ElementwiseRegion { .. }
        | Op::Custom { .. }
        | Op::CustomFn { .. }
        | Op::Conv { .. }
        | Op::ConvTranspose2d { .. }
        | Op::Pool { .. }
        | Op::Fft { .. } => None,
        _ => None,
    }
}
