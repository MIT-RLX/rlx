// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 1-D upsampling helpers for time-series U-Net decoders and diffusion
//! samplers (U-Sleep, EEGDM, …).
//!
//! `conv_transpose1d` wraps the existing 2-D transposed convolution with a
//! singleton height axis. `interpolate1d` resamples the last axis by an
//! arbitrary factor using a host-built `[L_in, L_out]` interpolation matrix
//! applied as a matmul — so both lower entirely to existing kernels and run
//! on every backend.

use crate::infer::GraphExt as _;
use crate::{Graph, NodeId};

/// Interpolation mode for [`Graph::interpolate1d`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpMode {
    /// Nearest-neighbor (round to closest source sample).
    Nearest,
    /// Linear interpolation between the two neighboring source samples.
    Linear,
}

impl Graph {
    /// Transposed 1-D convolution (a.k.a. deconvolution / fractionally-strided
    /// conv), the learned upsampler in Conv1d U-Net decoders.
    ///
    /// * `input`  — `[N, C_in, L]`.
    /// * `weight` — `[C_in, C_out/groups, K]` (PyTorch `ConvTranspose1d` layout).
    ///
    /// Returns `[N, C_out, L_out]` with
    /// `L_out = (L−1)·stride − 2·padding + dilation·(K−1) + output_padding + 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        output_padding: usize,
        groups: usize,
    ) -> NodeId {
        let in_s = self.shape(input).clone();
        let w_s = self.shape(weight).clone();
        assert_eq!(
            in_s.rank(),
            3,
            "conv_transpose1d: input must be [N, C_in, L]"
        );
        assert_eq!(
            w_s.rank(),
            3,
            "conv_transpose1d: weight must be [C_in, C_out/groups, K]"
        );
        let n = in_s.dim(0).unwrap_static();
        let c_in = in_s.dim(1).unwrap_static();
        let l = in_s.dim(2).unwrap_static();
        let c_out_g = w_s.dim(1).unwrap_static();

        // Lift to NCHW with a singleton height: [N, C_in, 1, L], weight
        // [C_in, C_out/g, 1, K], then run the 2-D transposed conv.
        let input4 = self.reshape_(input, vec![n as i64, c_in as i64, 1, l as i64]);
        let weight4 = self.reshape_(weight, vec![c_in as i64, c_out_g as i64, 1, kernel as i64]);
        let out4 = self.conv_transpose2d(
            input4,
            weight4,
            [1, kernel],
            [1, stride],
            [0, padding],
            [1, dilation],
            [0, output_padding],
            groups,
        );
        let o = self.shape(out4).clone();
        let c_out = o.dim(1).unwrap_static();
        let l_out = o.dim(3).unwrap_static();
        self.reshape_(out4, vec![n as i64, c_out as i64, l_out as i64])
    }

    /// Nearest-neighbor resample of an NCDHW volume to `size = [D, H, W]`.
    ///
    /// Emits native [`Op::Interpolate3d`]. Source index mapping:
    /// `src = min(floor(dst * in / out), in - 1)`.
    ///
    /// Prefer [`Graph::interpolate3d`] (in `ops/conv3d.rs`) for mode/align_corners
    /// control via a pure transpose/reshape/mm decomposition.
    pub fn interpolate3d_nearest(&mut self, x: NodeId, size: [usize; 3]) -> NodeId {
        let xs = self.shape(x).clone();
        assert_eq!(xs.rank(), 5, "interpolate3d_nearest: NCDHW only");
        assert!(
            size.iter().all(|&s| s > 0),
            "interpolate3d_nearest: size dims must be positive"
        );
        let out = crate::Shape::new(
            &[
                xs.dim(0).unwrap_static(),
                xs.dim(1).unwrap_static(),
                size[0],
                size[1],
                size[2],
            ],
            xs.dtype(),
        );
        self.push(
            crate::Op::Interpolate3d {
                size: size.to_vec(),
            },
            vec![x],
            out,
            None,
        )
    }

    /// Resample the last axis of `x` from `L_in` to `l_out` samples.
    ///
    /// Works for any rank ≥ 1; leading axes are treated as batch. Uses
    /// align-corners endpoint mapping (`pos = j·(L_in−1)/(l_out−1)`), matching
    /// `torch.nn.functional.interpolate(..., align_corners=True)`.
    pub fn interpolate1d(&mut self, x: NodeId, l_out: usize, mode: InterpMode) -> NodeId {
        assert!(l_out > 0, "interpolate1d: l_out must be positive");
        let xs = self.shape(x).clone();
        let rank = xs.rank();
        let l_in = xs.dim(rank - 1).unwrap_static();
        if l_in == l_out {
            return x;
        }
        let batch: usize = (0..rank - 1).map(|i| xs.dim(i).unwrap_static()).product();

        // Host-build the [L_in, L_out] interpolation matrix W, so that
        // out[.., j] = Σ_i x[.., i]·W[i, j].
        let mut w = vec![0f32; l_in * l_out];
        for j in 0..l_out {
            let pos = if l_out == 1 {
                0.0
            } else {
                j as f32 * (l_in as f32 - 1.0) / (l_out as f32 - 1.0)
            };
            match mode {
                InterpMode::Nearest => {
                    let i = (pos + 0.5) as usize;
                    let i = i.min(l_in - 1);
                    w[i * l_out + j] = 1.0;
                }
                InterpMode::Linear => {
                    let i0 = pos.floor() as usize;
                    let i0 = i0.min(l_in - 1);
                    let i1 = (i0 + 1).min(l_in - 1);
                    let frac = pos - i0 as f32;
                    w[i0 * l_out + j] += 1.0 - frac;
                    w[i1 * l_out + j] += frac;
                }
            }
        }
        let w_node = self.const_f32_tensor(w, &[l_in, l_out]);

        // Flatten batch → [batch, L_in], matmul, restore shape.
        let x2 = self.reshape_(x, vec![batch as i64, l_in as i64]);
        let y2 = self.mm(x2, w_node); // [batch, l_out]
        let mut out_dims: Vec<i64> = (0..rank - 1)
            .map(|i| xs.dim(i).unwrap_static() as i64)
            .collect();
        out_dims.push(l_out as i64);
        self.reshape_(y2, out_dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape};

    fn dims(g: &Graph, id: NodeId) -> Vec<usize> {
        g.shape(id)
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect()
    }
    fn input(g: &mut Graph, shape: &[usize]) -> NodeId {
        g.input("x", Shape::new(shape, DType::F32))
    }

    #[test]
    fn conv_transpose1d_upsamples_by_stride() {
        let mut g = Graph::new("ct1d");
        let x = input(&mut g, &[2, 3, 8]); // N,C_in,L
        let w = g.param("w", Shape::new(&[3, 5, 4], DType::F32)); // C_in,C_out,K
        let y = g.conv_transpose1d(x, w, 4, 2, 1, 1, 0, 1);
        // L_out = (8-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 14 - 2 + 3 + 1 = 16
        assert_eq!(dims(&g, y), vec![2, 5, 16]);
    }

    #[test]
    fn interpolate1d_linear_shape() {
        let mut g = Graph::new("interp");
        let x = input(&mut g, &[2, 4, 10]);
        let y = g.interpolate1d(x, 25, InterpMode::Linear);
        assert_eq!(dims(&g, y), vec![2, 4, 25]);
    }

    #[test]
    fn interpolate1d_nearest_identity() {
        let mut g = Graph::new("interp_id");
        let x = input(&mut g, &[3, 7]);
        let y = g.interpolate1d(x, 7, InterpMode::Nearest);
        assert_eq!(y, x); // no-op when l_in == l_out
    }
}
