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

//! NCDHW 3-D convolution builders (`conv3d`, `conv_transpose3d`) and a
//! separable 3-D resize (`interpolate3d`) for MONAI-style 3-D U-Net decoders
//! (UNETR / SwinUNETR).
//!
//! `conv3d` / `conv_transpose3d` add new `Op::Conv3d` / `Op::ConvTranspose3d`
//! nodes (dedicated CPU kernels). `interpolate3d` is a PURE DECOMPOSITION — no
//! new op — that resamples D, H, W one axis at a time by moving that axis last
//! (`transpose_`), applying a host-built `[L_in, L_out]` matrix via `mm`, and
//! moving it back. So it lowers entirely to existing kernels and runs on every
//! backend, matching the 1-D idea in [`super::upsample::Graph::interpolate1d`].

use crate::infer::GraphExt as _;
use crate::ops::upsample::InterpMode;
use crate::{Graph, NodeId, Op};

impl Graph {
    /// 3-D convolution on NCDHW tensors (`Op::Conv3d`).
    ///
    /// * `input`  — `[N, C_in, D, H, W]`.
    /// * `weight` — `[C_out, C_in/groups, kD, kH, kW]` (PyTorch `Conv3d` layout).
    ///
    /// Kernel size is read from the weight. Returns `[N, C_out, D_out, H_out,
    /// W_out]` with `X_out = floor((X + 2·p − dil·(K−1) − 1) / stride) + 1`.
    pub fn conv3d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        groups: usize,
    ) -> NodeId {
        let in_s = self.node(input).shape.clone();
        let w_s = self.node(weight).shape.clone();
        assert_eq!(in_s.rank(), 5, "conv3d: input must be [N, C_in, D, H, W]");
        assert_eq!(
            w_s.rank(),
            5,
            "conv3d: weight must be [C_out, C_in/groups, kD, kH, kW]"
        );
        let ks = [
            w_s.dim(2).unwrap_static(),
            w_s.dim(3).unwrap_static(),
            w_s.dim(4).unwrap_static(),
        ];
        let out =
            crate::shape::conv3d_output_shape(&in_s, &w_s, ks, stride, padding, dilation, groups)
                .expect("conv3d shape inference");
        self.push(
            Op::Conv3d {
                stride,
                padding,
                dilation,
                groups,
            },
            vec![input, weight],
            out,
            None,
        )
    }

    /// 3-D transposed convolution on NCDHW (`Op::ConvTranspose3d`), the learned
    /// upsampler in MONAI 3-D U-Net decoders.
    ///
    /// * `input`  — `[N, C_in, D, H, W]`.
    /// * `weight` — `[C_in, C_out/groups, kD, kH, kW]` (PyTorch `ConvTranspose3d`).
    ///
    /// Returns `[N, C_out, D_out, H_out, W_out]` with
    /// `X_out = (X−1)·stride − 2·p + dil·(K−1) + output_padding + 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose3d(
        &mut self,
        input: NodeId,
        weight: NodeId,
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        output_padding: [usize; 3],
        groups: usize,
    ) -> NodeId {
        let in_s = self.node(input).shape.clone();
        let w_s = self.node(weight).shape.clone();
        assert_eq!(
            in_s.rank(),
            5,
            "conv_transpose3d: input must be [N, C_in, D, H, W]"
        );
        assert_eq!(
            w_s.rank(),
            5,
            "conv_transpose3d: weight must be [C_in, C_out/groups, kD, kH, kW]"
        );
        let ks = [
            w_s.dim(2).unwrap_static(),
            w_s.dim(3).unwrap_static(),
            w_s.dim(4).unwrap_static(),
        ];
        let out = crate::shape::conv_transpose3d_output_shape(
            &in_s,
            &w_s,
            ks,
            stride,
            padding,
            dilation,
            output_padding,
            groups,
        )
        .expect("conv_transpose3d shape inference");
        self.push(
            Op::ConvTranspose3d {
                stride,
                padding,
                dilation,
                output_padding,
                groups,
            },
            vec![input, weight],
            out,
            None,
        )
    }

    /// Resample the spatial `[D, H, W]` axes of a `[N, C, D, H, W]` tensor to
    /// `out_dhw`, as a pure decomposition into `transpose_`/`reshape_`/`mm`
    /// (no new op). `InterpMode::Linear` gives separable trilinear;
    /// `InterpMode::Nearest` replicates the nearest source voxel.
    ///
    /// `align_corners` selects the sampling grid (exposed so a decoder can
    /// match its reference):
    /// * `true`  — endpoint mapping `pos = j·(L_in−1)/(L_out−1)`
    ///   (`torch.nn.functional.interpolate(..., align_corners=True)`).
    /// * `false` — half-pixel mapping `pos = (j+0.5)·L_in/L_out − 0.5`, clamped
    ///   to `[0, L_in−1]` — what MONAI UNETR's input branch uses
    ///   (`trilinear, align_corners=False`).
    pub fn interpolate3d(
        &mut self,
        x: NodeId,
        out_dhw: [usize; 3],
        mode: InterpMode,
        align_corners: bool,
    ) -> NodeId {
        let xs = self.shape(x).clone();
        assert_eq!(xs.rank(), 5, "interpolate3d: input must be [N, C, D, H, W]");
        assert!(
            out_dhw.iter().all(|&l| l > 0),
            "interpolate3d: out_dhw must be positive"
        );
        // Resample D (axis 2), then H (axis 3), then W (axis 4). Each pass is a
        // 1-D separable resample of one axis — order does not matter.
        let mut y = x;
        for (i, &out_len) in out_dhw.iter().enumerate() {
            y = self.resample_axis(y, 2 + i, out_len, mode, align_corners);
        }
        y
    }

    /// Resample a single `axis` of `x` from its current length to `out_len` via
    /// a `[L_in, L_out]` matmul, moving `axis` to the last position first.
    fn resample_axis(
        &mut self,
        x: NodeId,
        axis: usize,
        out_len: usize,
        mode: InterpMode,
        align_corners: bool,
    ) -> NodeId {
        let xs = self.shape(x).clone();
        let rank = xs.rank();
        let in_len = xs.dim(axis).unwrap_static();
        if in_len == out_len {
            return x;
        }
        // Move `axis` to last: perm = [all other axes in order] ++ [axis].
        let mut perm: Vec<usize> = (0..rank).filter(|&d| d != axis).collect();
        perm.push(axis);
        let xp = self.transpose_(x, perm.clone());
        let ps = self.shape(xp).clone();
        let batch: usize = (0..rank - 1).map(|d| ps.dim(d).unwrap_static()).product();

        let w = interp_matrix(in_len, out_len, mode, align_corners);
        let w_node = self.const_f32_tensor(w, &[in_len, out_len]);
        let x2 = self.reshape_(xp, vec![batch as i64, in_len as i64]);
        let y2 = self.mm(x2, w_node); // [batch, out_len]

        let mut out_perm_dims: Vec<i64> = (0..rank - 1)
            .map(|d| ps.dim(d).unwrap_static() as i64)
            .collect();
        out_perm_dims.push(out_len as i64);
        let yp = self.reshape_(y2, out_perm_dims);

        // Inverse permutation: inv[original_axis] = position_in_perm.
        let mut inv = vec![0usize; rank];
        for (new_pos, &old_axis) in perm.iter().enumerate() {
            inv[old_axis] = new_pos;
        }
        self.transpose_(yp, inv)
    }
}

/// Host-build the `[L_in, L_out]` resample matrix `W` so that
/// `out[.., j] = Σ_i x[.., i]·W[i, j]`.
fn interp_matrix(l_in: usize, l_out: usize, mode: InterpMode, align_corners: bool) -> Vec<f32> {
    let mut w = vec![0f32; l_in * l_out];
    for j in 0..l_out {
        // Source position for output index j.
        let pos = if align_corners {
            if l_out == 1 {
                0.0
            } else {
                j as f32 * (l_in as f32 - 1.0) / (l_out as f32 - 1.0)
            }
        } else {
            // Half-pixel (MONAI / PyTorch align_corners=False), clamped so the
            // boundary output taps land exactly on the first/last source sample.
            let p = (j as f32 + 0.5) * (l_in as f32) / (l_out as f32) - 0.5;
            p.clamp(0.0, l_in as f32 - 1.0)
        };
        match mode {
            InterpMode::Nearest => {
                let i = (pos + 0.5).floor() as usize;
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
    w
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

    #[test]
    fn conv3d_output_shape_matches_formula() {
        let mut g = Graph::new("conv3d");
        let x = g.input("x", Shape::new(&[1, 2, 8, 8, 8], DType::F32));
        let w = g.param("w", Shape::new(&[4, 2, 3, 3, 3], DType::F32));
        let y = g.conv3d(x, w, [1, 1, 1], [1, 1, 1], [1, 1, 1], 1);
        // X_out = (8 + 2 - 2 - 1)/1 + 1 = 8 (same padding for k=3, s=1, p=1).
        assert_eq!(dims(&g, y), vec![1, 4, 8, 8, 8]);
    }

    #[test]
    fn conv_transpose3d_upsamples_by_stride() {
        let mut g = Graph::new("ct3d");
        let x = g.input("x", Shape::new(&[1, 3, 4, 4, 4], DType::F32));
        let w = g.param("w", Shape::new(&[3, 5, 2, 2, 2], DType::F32));
        let y = g.conv_transpose3d(x, w, [2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], 1);
        // X_out = (4-1)*2 - 0 + 1*(2-1) + 0 + 1 = 6 + 1 + 1 = 8
        assert_eq!(dims(&g, y), vec![1, 5, 8, 8, 8]);
    }

    #[test]
    fn interpolate3d_shape() {
        let mut g = Graph::new("interp3d");
        let x = g.input("x", Shape::new(&[1, 2, 2, 3, 4], DType::F32));
        let y = g.interpolate3d(x, [4, 6, 8], InterpMode::Linear, false);
        assert_eq!(dims(&g, y), vec![1, 2, 4, 6, 8]);
    }
}
