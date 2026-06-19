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
use crate::shape::Dim;
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
        Op::LogMel => crate::audio::log_mel_output_shape(in_shape(0), in_shape(1)).ok(),
        Op::LogMelBackward => Some(shape::unary_shape(in_shape(0))),
        Op::WelchPeaks { k, n_segments } => {
            crate::audio::welch_peaks_output_shape(in_shape(0), *k, *n_segments).ok()
        }
        Op::Binary(_) => shape::binary_shape(in_shape(0), in_shape(1)).ok(),
        Op::Compare(_) => shape::compare_shape(in_shape(0), in_shape(1)).ok(),
        Op::Where => {
            let branches = shape::binary_shape(in_shape(1), in_shape(2)).ok()?;
            shape::binary_shape(in_shape(0), &branches)
                .ok()
                .map(|s| s.with_dtype(branches.dtype()))
        }

        Op::Activation(_) | Op::ReluBackward | Op::Conjugate => {
            Some(shape::unary_shape(in_shape(0)))
        }
        Op::ComplexNormSq => Some(Shape::from_dims(in_shape(0).dims(), DType::F32)),
        Op::ComplexNormSqBackward => Some(shape::unary_shape(in_shape(0))),
        Op::Cast { to } => Some(shape::cast_shape(in_shape(0), *to)),
        Op::StopGradient => Some(shape::unary_shape(in_shape(0))),

        Op::RngNormal { .. } | Op::RngUniform { .. } => {
            if node.inputs.is_empty() {
                Some(node.shape.clone())
            } else {
                Some(shape::unary_shape(in_shape(0)))
            }
        }

        Op::Reduce { axes, keep_dim, .. } => shape::reduce_shape(in_shape(0), axes, *keep_dim).ok(),
        Op::ArgMax { axis, keep_dim } | Op::ArgMin { axis, keep_dim } => {
            shape::reduce_shape(in_shape(0), &[*axis], *keep_dim).ok()
        }
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
        Op::Expand { target_shape } => shape::expand_shape(in_shape(0), target_shape).ok(),

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

        Op::Im2Col {
            kernel_size,
            stride,
            padding,
            dilation,
        } => {
            let ks = [kernel_size[0], kernel_size.get(1).copied().unwrap_or(1)];
            let st = [stride[0], stride.get(1).copied().unwrap_or(1)];
            let pad = [padding[0], padding.get(1).copied().unwrap_or(0)];
            let dil = [dilation[0], dilation.get(1).copied().unwrap_or(1)];
            shape::im2col_output_shape(in_shape(0), ks, st, pad, dil).ok()
        }

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
        | Op::SelectiveScan { .. }
        | Op::GatedDeltaNet { .. }
        | Op::Mamba2 { .. }
        | Op::FusedAttentionBlock { .. }
        | Op::FusedTransformerLayer { .. } => Some(shape::unary_shape(in_shape(0))),
        // x `[batch, seq, input]` → y `[batch, seq, hidden]` (preserve
        // batch/seq, static or dynamic; only the feature axis changes).
        Op::Lstm {
            hidden_size,
            bidirectional,
            ..
        }
        | Op::Gru {
            hidden_size,
            bidirectional,
            ..
        }
        | Op::Rnn {
            hidden_size,
            bidirectional,
            ..
        } => {
            let d = if *bidirectional { 2 } else { 1 };
            Some(
                in_shape(0)
                    .clone()
                    .with_dim(2, Dim::Static(d * *hidden_size)),
            )
        }
        Op::Scan {
            length,
            save_trajectory,
            ..
        } => {
            let carry = in_shape(0);
            if *save_trajectory {
                let mut dims = vec![Dim::Static(*length as usize)];
                for i in 0..carry.rank() {
                    dims.push(carry.dim(i));
                }
                Some(Shape::from_dims(&dims, carry.dtype()))
            } else {
                Some(shape::unary_shape(carry))
            }
        }
        Op::ElementwiseRegion { prologue, .. } => {
            // A fused elementwise chain broadcasts across ALL of its inputs, not
            // just input 0 — e.g. `(g[1,C,1] + g2[1,C,1]) + x[1,C,T]) * mask[1,1,T]`
            // is `[1,C,T]`. Folding only input 0 mis-infers `[1,C,1]` and trips the
            // verifier on strict backends (MLX/wgpu compile path).
            let mut in_s = in_shape(0).clone();
            for i in 1..node.inputs.len() {
                if let Ok(b) = shape::binary_shape(&in_s, in_shape(i)) {
                    in_s = b;
                }
            }
            if *prologue == RegionPrologue::ResizeNearest2x && in_s.rank() == 4 {
                in_s = Shape::new(
                    &[
                        in_s.dim(0).unwrap_static(),
                        in_s.dim(1).unwrap_static(),
                        in_s.dim(2).unwrap_static() * 2,
                        in_s.dim(3).unwrap_static() * 2,
                    ],
                    in_s.dtype(),
                );
            }
            Some(in_s)
        }
        Op::BatchElementwiseRegion {
            prologue,
            num_batch_inputs,
            ..
        } => {
            let n = *num_batch_inputs as usize;
            let mut out_s = in_shape(0).clone();
            if *prologue == RegionPrologue::ResizeNearest2x && out_s.rank() == 4 {
                out_s = Shape::new(
                    &[
                        out_s.dim(0).unwrap_static(),
                        out_s.dim(1).unwrap_static(),
                        out_s.dim(2).unwrap_static() * 2,
                        out_s.dim(3).unwrap_static() * 2,
                    ],
                    out_s.dtype(),
                );
            }
            if out_s.rank() >= 1 && n > 1 {
                let mut batch_dim = 0usize;
                for i in 0..n.min(node.inputs.len()) {
                    batch_dim += in_shape(i).dim(0).unwrap_static();
                }
                if batch_dim > 0 {
                    out_s = out_s.with_dim(0, shape::Dim::Static(batch_dim));
                }
            }
            Some(out_s)
        }
        Op::TransformRegion { steps, .. } => {
            let mut in_s = in_shape(0).clone();
            for step in steps {
                if !matches!(step, TransformStep::ResizeNearest2x(_)) {
                    return None;
                }
                if in_s.rank() != 4 {
                    return None;
                }
                in_s = Shape::new(
                    &[
                        in_s.dim(0).unwrap_static(),
                        in_s.dim(1).unwrap_static(),
                        in_s.dim(2).unwrap_static() * 2,
                        in_s.dim(3).unwrap_static() * 2,
                    ],
                    in_s.dtype(),
                );
            }
            Some(in_s)
        }
        Op::Custom { .. }
        | Op::CustomFn { .. }
        | Op::Conv { .. }
        | Op::ConvTranspose2d { .. }
        | Op::Pool { .. }
        | Op::Fft { .. }
        | Op::FftButterflyStage { .. } => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, Op};

    #[test]
    fn scan_save_trajectory_infers_length_by_carry() {
        let mut g = Graph::new("scan_traj");
        let init = g.input("init", Shape::new(&[80], DType::F32));
        let body = Graph::new("body");
        let scan = g.add_node(
            Op::Scan {
                body: Box::new(body),
                length: 70,
                save_trajectory: true,
                num_bcast: 0,
                num_xs: 0,
                num_checkpoints: 0,
            },
            vec![init],
            Shape::new(&[70, 80], DType::F32),
        );
        let node = g.node(scan).clone();
        let inferred = infer_output_shape(&g, &node).expect("scan infer");
        assert_eq!(inferred.dims(), node.shape.dims());
    }

    #[test]
    fn scan_without_trajectory_infers_carry_only() {
        let mut g = Graph::new("scan_carry");
        let init = g.input("init", Shape::new(&[80], DType::F32));
        let body = Graph::new("body");
        let scan = g.add_node(
            Op::Scan {
                body: Box::new(body),
                length: 70,
                save_trajectory: false,
                num_bcast: 0,
                num_xs: 0,
                num_checkpoints: 0,
            },
            vec![init],
            Shape::new(&[80], DType::F32),
        );
        let node = g.node(scan).clone();
        let inferred = infer_output_shape(&g, &node).expect("scan infer");
        assert_eq!(inferred.dims(), node.shape.dims());
    }
}
