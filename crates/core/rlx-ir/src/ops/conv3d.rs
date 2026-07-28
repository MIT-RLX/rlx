// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

    /// 3-D convolution computed as a PURE DECOMPOSITION into universal,
    /// already-differentiable primitives (`reshape`/`concat`/`gather`/`matmul`/
    /// `transpose`) — the same "runs on every backend, autodiff for free"
    /// philosophy as [`Self::interpolate3d`]. Semantically identical to
    /// [`Self::conv3d`] (cross-correlation, zero padding), so it can stand in
    /// for `Op::Conv3d` on MLX / CUDA / Metal, which have no native 3-D conv
    /// kernel, and it trains without a hand-written VJP.
    ///
    /// * `input`  — `[N, C_in, D, H, W]`.
    /// * `weight` — `[C_out, C_in/groups, kD, kH, kW]` (PyTorch `Conv3d` layout).
    ///
    /// Returns `[N, C_out, D_out, H_out, W_out]` with the same output formula
    /// as [`Self::conv3d`]. Only `groups == 1` is supported (panics otherwise).
    ///
    /// # Technique (im2col via axis-0 `gather`, no `Pad` op)
    ///
    /// The batch axis is carried as the gather **trailing** dim so the whole op
    /// reduces to a single *axis-0, 1-D-index* gather — the one gather shape
    /// whose autodiff (`Op::ScatterAdd`) is exact on the universal kernels
    /// (a batched 2-D index would mis-shape both `GatherBackward` and
    /// `ScatterAdd`).
    ///
    /// 1. Flatten `input` `[N, C_in, D, H, W]` → `[N, M]` (`M = C_in·D·H·W`),
    ///    transpose to `[M, N]`, and append one **zero sentinel** row (`concat`
    ///    with a `Constant` zero) → `table` `[M + 1, N]`; row `M` is guaranteed
    ///    zero.
    /// 2. Host-build a **1-D** `f32` index of length `P·K`
    ///    (`P = D_out·H_out·W_out`, `K = C_in·kD·kH·kW`) mapping each
    ///    (output-position, in-channel, kernel-tap) to its flattened input row;
    ///    taps landing in the (virtual) padding map to the sentinel `M`.
    /// 3. `gather(table, idx, axis=0)` → `[P·K, N]` (the im2col matrix for every
    ///    batch at once). Reshape / transpose to `[P·N, K]`, `matmul` with
    ///    `weight` reshaped to `[C_out, K]` and transposed to `[K, C_out]`, then
    ///    reshape / transpose back to `[N, C_out, D_out, H_out, W_out]`.
    ///
    /// The index is `f32` because the axis-0 `Op::Gather` **and** its
    /// `Op::ScatterAdd` VJP both read the index buffer as `f32` on CPU; index
    /// magnitudes stay well under `2^24`, so they are represented exactly.
    pub fn conv3d_im2col(
        &mut self,
        input: NodeId,
        weight: NodeId,
        stride: [usize; 3],
        padding: [usize; 3],
        dilation: [usize; 3],
        groups: usize,
    ) -> NodeId {
        assert_eq!(
            groups, 1,
            "conv3d_im2col: only groups == 1 is supported (grouped 3-D conv is \
             not needed by the codec); got groups = {groups}"
        );
        let in_s = self.node(input).shape.clone();
        let w_s = self.node(weight).shape.clone();
        assert_eq!(
            in_s.rank(),
            5,
            "conv3d_im2col: input must be [N, C_in, D, H, W]"
        );
        assert_eq!(
            w_s.rank(),
            5,
            "conv3d_im2col: weight must be [C_out, C_in, kD, kH, kW]"
        );

        let n = in_s.dim(0).unwrap_static();
        let c_in = in_s.dim(1).unwrap_static();
        let d = in_s.dim(2).unwrap_static();
        let h = in_s.dim(3).unwrap_static();
        let w = in_s.dim(4).unwrap_static();

        let c_out = w_s.dim(0).unwrap_static();
        assert_eq!(
            w_s.dim(1).unwrap_static(),
            c_in,
            "conv3d_im2col: weight C_in dim must match input C_in for groups == 1"
        );
        let kd = w_s.dim(2).unwrap_static();
        let kh = w_s.dim(3).unwrap_static();
        let kw = w_s.dim(4).unwrap_static();
        let ks = [kd, kh, kw];

        let out =
            crate::shape::conv3d_output_shape(&in_s, &w_s, ks, stride, padding, dilation, groups)
                .expect("conv3d_im2col shape inference");
        let d_out = out.dim(2).unwrap_static();
        let h_out = out.dim(3).unwrap_static();
        let w_out = out.dim(4).unwrap_static();

        let [sd, sh, sw] = stride;
        let [pd, ph, pw] = padding;
        let [dd, dh, dw] = dilation;

        // Flattened per-batch input length; index `m` is the zero sentinel.
        let m = c_in * d * h * w;
        let p = d_out * h_out * w_out; // output spatial positions (rows)
        let k = c_in * kd * kh * kw; // in-channel × kernel taps (cols)

        // Host-build the 1-D im2col gather index (length P·K, flattened [P, K]).
        // Column order matches the weight reshaped to [C_out, K] = row-major
        // flatten of [C_in, kD, kH, kW]:
        //   kcol = ((ci·kD + kdi)·kH + ki)·kW + kj.
        // Row order matches the NCDHW output spatial flatten:
        //   prow = (od·H_out + ho)·W_out + wo.
        // Semantics mirror the CPU `Op::Conv3d` kernel exactly (correlation,
        // zero padding): out-of-bounds taps map to the sentinel row `m`.
        let mut idx = vec![0f32; p * k];
        for od in 0..d_out {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let prow = (od * h_out + ho) * w_out + wo;
                    for ci in 0..c_in {
                        for kdi in 0..kd {
                            // Padded (virtual) input coordinate for this tap.
                            let dip = od * sd + kdi * dd;
                            let di_ok = dip >= pd && (dip - pd) < d;
                            let di = dip.wrapping_sub(pd);
                            for ki in 0..kh {
                                let hip = ho * sh + ki * dh;
                                let hi_ok = hip >= ph && (hip - ph) < h;
                                let hi = hip.wrapping_sub(ph);
                                for kj in 0..kw {
                                    let wip = wo * sw + kj * dw;
                                    let wi_ok = wip >= pw && (wip - pw) < w;
                                    let wi = wip.wrapping_sub(pw);
                                    let kcol = ((ci * kd + kdi) * kh + ki) * kw + kj;
                                    let flat = if di_ok && hi_ok && wi_ok {
                                        ci * d * h * w + (di * h + hi) * w + wi
                                    } else {
                                        m // zero sentinel row
                                    };
                                    idx[prow * k + kcol] = flat as f32;
                                }
                            }
                        }
                    }
                }
            }
        }
        let idx_node = self.const_f32_tensor(idx, &[p * k]);

        // table = [xᵀ ; 0] of shape [M + 1, N]: row j (j < M) holds input value
        // at flat spatial-channel offset j for every batch; row M is the zero
        // sentinel padding taps gather from.
        let x_flat = self.reshape_(input, vec![n as i64, m as i64]); // [N, M]
        let x_t = self.transpose_(x_flat, vec![1, 0]); // [M, N]
        let zero_row = self.const_f32_tensor(vec![0.0f32; n], &[1, n]);
        let table = self.concat_(vec![x_t, zero_row], 0); // [M + 1, N]

        // im2col via a single axis-0 gather (1-D index): [P·K, N].
        let col = self.gather_(table, idx_node, 0);
        let col_pkn = self.reshape_(col, vec![p as i64, k as i64, n as i64]); // [P, K, N]
        let col_pnk = self.transpose_(col_pkn, vec![0, 2, 1]); // [P, N, K]
        let col2 = self.reshape_(col_pnk, vec![(p * n) as i64, k as i64]); // [P·N, K]

        // weight [C_out, C_in, kD, kH, kW] → [C_out, K] → [K, C_out].
        let w_flat = self.reshape_(weight, vec![c_out as i64, k as i64]);
        let w_t = self.transpose_(w_flat, vec![1, 0]);

        // [P·N, K] @ [K, C_out] = [P·N, C_out] → [P, N, C_out] → [N, C_out, P].
        let y2 = self.mm(col2, w_t);
        let y3 = self.reshape_(y2, vec![p as i64, n as i64, c_out as i64]);
        let y4 = self.transpose_(y3, vec![1, 2, 0]); // [N, C_out, P]
        self.reshape_(
            y4,
            vec![
                n as i64,
                c_out as i64,
                d_out as i64,
                h_out as i64,
                w_out as i64,
            ],
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
