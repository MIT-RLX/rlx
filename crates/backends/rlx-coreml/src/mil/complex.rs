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

//! MIL lowers for interleaved-C64 Wirtinger ops and radix-2 FFT butterfly
//! stages.
//!
//! CoreML has no complex dtype; C64 tensors are promoted to F32 with the last
//! axis doubled (`[…, n] C64` → `[…, 2n] F32`) before lowering. Ops here slice
//! / stack real-imag pairs and emit ordinary MIL elementwise / reshape nodes.

use rlx_ir::{DType, NodeId, Shape};

use super::helpers::*;
use super::*;
use crate::Result;

impl<'a> LowerCtx<'a> {
    /// `|z|² = re² + im²` on interleaved `[…, 2n]` F32 (was C64).
    pub(crate) fn lower_complex_norm_sq(&mut self, id: NodeId, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let z = self.val(node.inputs[0]);
        let out_shape = node.shape.clone();
        let n = out_shape.num_elements().unwrap_or(0) as i64;
        let pair = Shape::new(&[n as usize, 2], DType::F32);
        let one = Shape::new(&[n as usize, 1], DType::F32);
        let zr = format!("{out_name}_pairs");
        self.reshape_to(&z, &[n, 2], &pair, &zr)?;
        let re = format!("{out_name}_re");
        let im = format!("{out_name}_im");
        self.slice_last(&zr, 2, 0, 1, &one, &re)?;
        self.slice_last(&zr, 2, 1, 1, &one, &im)?;
        let re2 = format!("{out_name}_re2");
        let im2 = format!("{out_name}_im2");
        self.emit(
            "mul",
            &re2,
            &one,
            vec![("x", bind_name(&re)), ("y", bind_name(&re))],
        )?;
        self.emit(
            "mul",
            &im2,
            &one,
            vec![("x", bind_name(&im)), ("y", bind_name(&im))],
        )?;
        let sq = format!("{out_name}_sq");
        self.emit(
            "add",
            &sq,
            &one,
            vec![("x", bind_name(&re2)), ("y", bind_name(&im2))],
        )?;
        let out_dims: Vec<i64> = out_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        self.reshape_to(&sq, &out_dims, &out_shape, out_name)?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Wirtinger: `dz = g · z` (interleaved F32 pairs).
    pub(crate) fn lower_complex_norm_sq_backward(
        &mut self,
        id: NodeId,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let z = self.val(node.inputs[0]);
        let g = self.val(node.inputs[1]);
        let out_shape = node.shape.clone();
        let n = self.graph.shape(node.inputs[1]).num_elements().unwrap_or(0) as i64;
        let pair = Shape::new(&[n as usize, 2], DType::F32);
        let g_col = Shape::new(&[n as usize, 1], DType::F32);
        let zr = format!("{out_name}_pairs");
        self.reshape_to(&z, &[n, 2], &pair, &zr)?;
        let gc = format!("{out_name}_g");
        self.reshape_to(&g, &[n, 1], &g_col, &gc)?;
        let prod = format!("{out_name}_prod");
        self.emit(
            "mul",
            &prod,
            &pair,
            vec![("x", bind_name(&zr)), ("y", bind_name(&gc))],
        )?;
        let dims: Vec<i64> = out_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        self.reshape_to(&prod, &dims, &out_shape, out_name)?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// `conj(z) = (re, -im)` on interleaved F32.
    pub(crate) fn lower_conjugate(&mut self, id: NodeId, out_name: &str) -> Result<()> {
        let node = self.graph.node(id);
        let z = self.val(node.inputs[0]);
        let out_shape = node.shape.clone();
        let n = (out_shape.num_elements().unwrap_or(0) / 2) as i64;
        let pair = Shape::new(&[n as usize, 2], DType::F32);
        let one = Shape::new(&[n as usize, 1], DType::F32);
        let zr = format!("{out_name}_pairs");
        self.reshape_to(&z, &[n, 2], &pair, &zr)?;
        let re = format!("{out_name}_re");
        let im = format!("{out_name}_im");
        self.slice_last(&zr, 2, 0, 1, &one, &re)?;
        self.slice_last(&zr, 2, 1, 1, &one, &im)?;
        let neg_im = format!("{out_name}_nim");
        self.emit(
            "mul",
            &neg_im,
            &one,
            vec![("x", bind_name(&im)), ("y", bind_value(scalar_f32(-1.0)))],
        )?;
        let cat = format!("{out_name}_cat");
        self.emit(
            "concat",
            &cat,
            &pair,
            vec![
                ("values", bind_names(&[re, neg_im])),
                ("axis", bind_value(scalar_i32(1))),
                ("interleave", bind_value(scalar_bool(false))),
            ],
        )?;
        let dims: Vec<i64> = out_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        self.reshape_to(&cat, &dims, &out_shape, out_name)?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Broadcast a rank-4 `[1, ng, st, 1]` tensor to `[B, ng, st, 1]` via `+ 0·ref`.
    fn broadcast_lane(
        &mut self,
        src: &str,
        ref_lane: &str,
        lane: &Shape,
        dst: &str,
        tag: &str,
    ) -> Result<()> {
        let zeros = format!("{tag}_z");
        self.emit(
            "mul",
            &zeros,
            lane,
            vec![
                ("x", bind_name(ref_lane)),
                ("y", bind_value(scalar_f32(0.0))),
            ],
        )?;
        self.emit(
            "add",
            dst,
            lane,
            vec![("x", bind_name(src)), ("y", bind_name(&zeros))],
        )
    }

    /// Radix-2 butterfly stage on interleaved `[batch, n_fft*2]` F32.
    pub(crate) fn lower_fft_butterfly_stage(
        &mut self,
        id: NodeId,
        stage: u32,
        n_fft: u32,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let state = self.val(node.inputs[0]);
        let gate = self.val(node.inputs[1]);
        let rev = self.val(node.inputs[2]);
        let tw_re = self.val(node.inputs[3]);
        let tw_im = self.val(node.inputs[4]);
        let out_shape = node.shape.clone();
        let batch = out_shape.dim(0).unwrap_static() as i64;
        let n_fft = n_fft as i64;
        let stride = 1i64 << stage;
        let n_groups = n_fft / (2 * stride);
        let (b, ng, st) = (batch, n_groups, stride);

        let pairs = Shape::new(&[batch as usize, n_fft as usize, 2], DType::F32);
        let grouped = Shape::new(
            &[batch as usize, n_groups as usize, 2, stride as usize, 2],
            DType::F32,
        );
        let half = Shape::new(
            &[batch as usize, n_groups as usize, stride as usize, 2],
            DType::F32,
        );
        let lane = Shape::new(
            &[batch as usize, n_groups as usize, stride as usize, 1],
            DType::F32,
        );
        let bool_lane = lane.clone().with_dtype(DType::Bool);
        let one_g = Shape::new(
            &[batch as usize, n_groups as usize, 1, stride as usize, 2],
            DType::F32,
        );
        let meta1 = Shape::new(&[1, n_groups as usize, stride as usize, 1], DType::F32);

        let p = format!("{out_name}_p");
        self.reshape_to(&state, &[b, n_fft, 2], &pairs, &p)?;
        let g5 = format!("{out_name}_g5");
        self.reshape_to(&p, &[b, ng, 2, st, 2], &grouped, &g5)?;

        let a5 = format!("{out_name}_a5");
        let b5 = format!("{out_name}_b5");
        self.slice_axis(&g5, 5, 2, 0, 1, &one_g, &a5)?;
        self.slice_axis(&g5, 5, 2, 1, 1, &one_g, &b5)?;
        let a = format!("{out_name}_a");
        let bv = format!("{out_name}_b");
        self.reshape_to(&a5, &[b, ng, st, 2], &half, &a)?;
        self.reshape_to(&b5, &[b, ng, st, 2], &half, &bv)?;

        let a_re = format!("{out_name}_are");
        let a_im = format!("{out_name}_aim");
        let b_re = format!("{out_name}_bre");
        let b_im = format!("{out_name}_bim");
        self.slice_last(&a, 4, 0, 1, &lane, &a_re)?;
        self.slice_last(&a, 4, 1, 1, &lane, &a_im)?;
        self.slice_last(&bv, 4, 0, 1, &lane, &b_re)?;
        self.slice_last(&bv, 4, 1, 1, &lane, &b_im)?;

        let g1 = format!("{out_name}_gate1");
        let r1 = format!("{out_name}_rev1");
        let tr1 = format!("{out_name}_twr1");
        let ti1 = format!("{out_name}_twi1");
        self.reshape_to(&gate, &[1, ng, st, 1], &meta1, &g1)?;
        self.reshape_to(&rev, &[1, ng, st, 1], &meta1, &r1)?;
        self.reshape_to(&tw_re, &[1, ng, st, 1], &meta1, &tr1)?;
        self.reshape_to(&tw_im, &[1, ng, st, 1], &meta1, &ti1)?;

        let g2 = format!("{out_name}_gate");
        let r2 = format!("{out_name}_rev");
        let tr2 = format!("{out_name}_twr");
        let ti2 = format!("{out_name}_twi");
        self.broadcast_lane(&g1, &a_re, &lane, &g2, &format!("{out_name}_bg"))?;
        self.broadcast_lane(&r1, &a_re, &lane, &r2, &format!("{out_name}_br"))?;
        self.broadcast_lane(&tr1, &a_re, &lane, &tr2, &format!("{out_name}_btr"))?;
        self.broadcast_lane(&ti1, &a_re, &lane, &ti2, &format!("{out_name}_bti"))?;

        let t0 = format!("{out_name}_t0");
        let t1 = format!("{out_name}_t1");
        let t2 = format!("{out_name}_t2");
        let t3 = format!("{out_name}_t3");
        let bw_re = format!("{out_name}_bwre");
        let bw_im = format!("{out_name}_bwim");
        self.emit(
            "mul",
            &t0,
            &lane,
            vec![("x", bind_name(&b_re)), ("y", bind_name(&tr2))],
        )?;
        self.emit(
            "mul",
            &t1,
            &lane,
            vec![("x", bind_name(&b_im)), ("y", bind_name(&ti2))],
        )?;
        self.emit(
            "sub",
            &bw_re,
            &lane,
            vec![("x", bind_name(&t0)), ("y", bind_name(&t1))],
        )?;
        self.emit(
            "mul",
            &t2,
            &lane,
            vec![("x", bind_name(&b_re)), ("y", bind_name(&ti2))],
        )?;
        self.emit(
            "mul",
            &t3,
            &lane,
            vec![("x", bind_name(&b_im)), ("y", bind_name(&tr2))],
        )?;
        self.emit(
            "add",
            &bw_im,
            &lane,
            vec![("x", bind_name(&t2)), ("y", bind_name(&t3))],
        )?;

        let top_re = format!("{out_name}_topre");
        let top_im = format!("{out_name}_topim");
        let bot_re = format!("{out_name}_botre");
        let bot_im = format!("{out_name}_botim");
        self.emit(
            "add",
            &top_re,
            &lane,
            vec![("x", bind_name(&a_re)), ("y", bind_name(&bw_re))],
        )?;
        self.emit(
            "add",
            &top_im,
            &lane,
            vec![("x", bind_name(&a_im)), ("y", bind_name(&bw_im))],
        )?;
        self.emit(
            "sub",
            &bot_re,
            &lane,
            vec![("x", bind_name(&a_re)), ("y", bind_name(&bw_re))],
        )?;
        self.emit(
            "sub",
            &bot_im,
            &lane,
            vec![("x", bind_name(&a_im)), ("y", bind_name(&bw_im))],
        )?;

        let do_rev = format!("{out_name}_dorev");
        self.emit(
            "greater_equal",
            &do_rev,
            &bool_lane,
            vec![("x", bind_name(&r2)), ("y", bind_value(scalar_f32(0.5)))],
        )?;
        let oa_re = format!("{out_name}_oare");
        let oa_im = format!("{out_name}_oaim");
        let ob_re = format!("{out_name}_obre");
        let ob_im = format!("{out_name}_obim");
        for (dst, a, b) in [
            (&oa_re, &bot_re, &top_re),
            (&oa_im, &bot_im, &top_im),
            (&ob_re, &top_re, &bot_re),
            (&ob_im, &top_im, &bot_im),
        ] {
            self.emit(
                "select",
                dst,
                &lane,
                vec![
                    ("cond", bind_name(&do_rev)),
                    ("a", bind_name(a)),
                    ("b", bind_name(b)),
                ],
            )?;
        }

        let active = format!("{out_name}_act");
        self.emit(
            "not_equal",
            &active,
            &bool_lane,
            vec![("x", bind_name(&g2)), ("y", bind_value(scalar_f32(0.0)))],
        )?;
        let out_a_re = format!("{out_name}_oare2");
        let out_a_im = format!("{out_name}_oaim2");
        let out_b_re = format!("{out_name}_obre2");
        let out_b_im = format!("{out_name}_obim2");
        for (dst, a, b) in [
            (&out_a_re, &oa_re, &a_re),
            (&out_a_im, &oa_im, &a_im),
            (&out_b_re, &ob_re, &b_re),
            (&out_b_im, &ob_im, &b_im),
        ] {
            self.emit(
                "select",
                dst,
                &lane,
                vec![
                    ("cond", bind_name(&active)),
                    ("a", bind_name(a)),
                    ("b", bind_name(b)),
                ],
            )?;
        }

        let out_a = format!("{out_name}_outa");
        let out_b = format!("{out_name}_outb");
        self.emit(
            "concat",
            &out_a,
            &half,
            vec![
                ("values", bind_names(&[out_a_re, out_a_im])),
                ("axis", bind_value(scalar_i32(3))),
                ("interleave", bind_value(scalar_bool(false))),
            ],
        )?;
        self.emit(
            "concat",
            &out_b,
            &half,
            vec![
                ("values", bind_names(&[out_b_re, out_b_im])),
                ("axis", bind_value(scalar_i32(3))),
                ("interleave", bind_value(scalar_bool(false))),
            ],
        )?;
        let out_a5 = format!("{out_name}_outa5");
        let out_b5 = format!("{out_name}_outb5");
        self.reshape_to(&out_a, &[b, ng, 1, st, 2], &one_g, &out_a5)?;
        self.reshape_to(&out_b, &[b, ng, 1, st, 2], &one_g, &out_b5)?;
        let stacked = format!("{out_name}_stack");
        self.emit(
            "concat",
            &stacked,
            &grouped,
            vec![
                ("values", bind_names(&[out_a5, out_b5])),
                ("axis", bind_value(scalar_i32(2))),
                ("interleave", bind_value(scalar_bool(false))),
            ],
        )?;
        let flat = format!("{out_name}_flat");
        self.reshape_to(&stacked, &[b, n_fft, 2], &pairs, &flat)?;
        let dims: Vec<i64> = out_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        self.reshape_to(&flat, &dims, &out_shape, out_name)?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }
}
