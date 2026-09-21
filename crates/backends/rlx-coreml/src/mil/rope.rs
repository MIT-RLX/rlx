// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation: takes
// an RLX `Graph` plus baked parameter/constant data and produces a
// `proto::Model` ready to serialise into a `.mlpackage`. No FFI, so this
// builds and unit-tests on any host.

//! `rope` — extracted from the `mil` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use super::helpers::simple_op_flex;
use super::helpers::*;
use crate::proto;
use crate::{CoremlError, Result};
use rlx_ir::op::{Activation, CmpOp, MaskKind, ReduceOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, Graph, NodeId, Op, Shape};
use std::collections::HashMap;

use super::*;

/// Everything [`LowerCtx::lower_rope_gptj`] needs from its caller.
struct LowerRopeGptj<'b> {
    eff_x: &'b str,
    eff_shape: &'b Shape,
    eff_cos: &'b str,
    eff_sin: &'b str,
    /// Shape of the sliced cos/sin table, head axis included when packed.
    eff_table: &'b Shape,
    head_dim: usize,
    n_rot: usize,
    /// Original shape to reshape back to, when the rotation ran on a per-head
    /// view of a heads-packed tensor.
    restore: Option<&'b Shape>,
}

/// Sequence length axis for RoPE layouts.
fn rope_seq_len(shape: &Shape, _heads_packed: bool) -> Result<usize> {
    let rank = shape.rank();
    if rank < 2 {
        return Err(CoremlError::Unsupported(format!(
            "rope: cannot infer seq from rank-{rank} shape {:?}",
            shape.dims()
        )));
    }
    // The sequence is always the axis just before the lane axis, whatever the
    // rank: `[S, D]`, `[B, S, D]`, `[B, H, S, D]` and the heads-packed
    // `[.., S, G·head_dim]` all put it at `rank - 2`.
    //
    // Special-casing per rank got the rank-2 forms wrong in both branches — a
    // `[S, G·head_dim]` input read its packed lane count as the sequence length,
    // so the cos/sin tables were reshaped to the wrong width and CoreML refused
    // to load the model. (Metal had the same bug in its own derivation.)
    let idx = rank - 2;
    match shape.dim(idx) {
        Dim::Static(s) => Ok(s),
        Dim::Dynamic(s) => Err(CoremlError::DynamicShape(format!("rope seq ?{s}"))),
    }
}

impl<'a> LowerCtx<'a> {
    /// RoPE. Inputs `[x, cos, sin]`; rotates the first
    /// `n_rot` of the trailing `head_dim` lane, passes the rest through.
    /// Only the layout where the last axis == `head_dim` is supported
    /// (`[B,H,S,D]` or `[B,S,D]`); the cos/sin tables (`[…,n_rot/2]`)
    /// broadcast against the rotated halves.
    ///
    /// Production tables are often `[max_pos, head_half]` with
    /// `max_pos ≫ seq` (Qwen3.5 / Bonsai: max_pos=262144). Metal/CPU RoPE
    /// index by position; this lowering slices tables to `[seq, rot_half]`
    /// before any mul so Espresso broadcast matches the activation.
    /// `style` picks the pairing: [`RopeStyle::NeoX`] rotates the two halves of
    /// the lane against each other, [`RopeStyle::GptJ`] rotates adjacent
    /// even/odd pairs. The two produce different numbers from the same tables,
    /// so a lowering that ignores the argument is silently wrong for every model
    /// using the other convention — which is what this did before.
    pub(crate) fn lower_rope(
        &mut self,
        id: NodeId,
        head_dim: usize,
        n_rot: usize,
        style: rlx_ir::op::RopeStyle,
        out_name: &str,
    ) -> Result<()> {
        let (shape, in0, in1, in2) = {
            let node = self.graph.node(id);
            (
                node.shape.clone(),
                node.inputs[0],
                node.inputs[1],
                node.inputs[2],
            )
        };
        let rank = shape.rank();
        let last = match shape.dim(rank - 1) {
            Dim::Static(n) => n,
            Dim::Dynamic(s) => {
                return Err(CoremlError::DynamicShape(format!("rope last dim ?{s}")));
            }
        };

        let rot_half = n_rot / 2;
        let heads_packed = last != head_dim && head_dim != 0 && last.is_multiple_of(head_dim);
        let seq = rope_seq_len(&shape, heads_packed)?;

        let x = self.val(in0);
        let cos0 = self.val(in1);
        let sin0 = self.val(in2);

        // Slice `[max_pos, half] → [seq, rot_half]` on the Graph-backed tables
        // before any reshape/broadcast.
        let cos = self.rope_slice_table(in1, &cos0, seq, rot_half, &format!("{out_name}_cos"))?;
        let sin = self.rope_slice_table(in2, &sin0, seq, rot_half, &format!("{out_name}_sin"))?;
        // After slice the effective table shape is always `[seq, rot_half]`.
        let table_shape = Shape::new(&[seq, rot_half], DType::F32);

        // Flexible layout: the rotation runs on a tensor whose LAST axis is
        // exactly `head_dim`. When the last axis instead packs multiple heads
        // (`[.., G*head_dim]`, the fused-QKV layout used by e.g. Qwen3-ASR),
        // reshape to `[.., G, head_dim]`, rotate per head, then reshape back —
        // cos/sin gain a singleton head axis so they broadcast over the heads.
        let (eff_x, eff_shape, eff_cos, eff_sin, eff_table, restore) = if last == head_dim {
            (x, shape.clone(), cos, sin, table_shape.clone(), None)
        } else if heads_packed {
            let groups = last / head_dim;
            let mut gd = shape.dims().to_vec();
            gd.pop();
            gd.push(Dim::Static(groups));
            gd.push(Dim::Static(head_dim));
            let gshape = Shape::from_dims(&gd, DType::F32);
            let xg = format!("{out_name}_xg");
            self.emit(
                "reshape",
                &xg,
                &gshape,
                vec![
                    ("x", bind_name(&x)),
                    ("shape", bind_value(vec_i32(&dims_i32(&gd)))),
                ],
            )?;
            // Insert head axis into the already-sliced `[seq, rot_half]` tables
            // → `[seq, 1, rot_half]` to broadcast over G.
            let cos_g =
                self.rope_insert_head_axis_shape(&table_shape, &cos, &format!("{out_name}_cosg"))?;
            let sin_g =
                self.rope_insert_head_axis_shape(&table_shape, &sin, &format!("{out_name}_sing"))?;
            let mut td = table_shape.dims().to_vec();
            td.insert(td.len() - 1, Dim::Static(1));
            (
                xg,
                gshape,
                cos_g,
                sin_g,
                Shape::from_dims(&td, DType::F32),
                Some(shape.clone()),
            )
        } else {
            return Err(CoremlError::Unsupported(format!(
                "rope: last dim {last} is not a multiple of head_dim {head_dim} \
                 (dims={:?}, n_rot={n_rot})",
                shape.dims()
            )));
        };

        let eff_rank = eff_shape.rank();
        let half_shape = with_last(&eff_shape, rot_half);
        let rot_shape = with_last(&eff_shape, n_rot);

        // Rotated result lands in `core` — the real output unless we worked on
        // a per-head view, in which case it is reshaped back below.
        let core = match restore {
            Some(_) => format!("{out_name}_core"),
            None => out_name.to_string(),
        };

        if style == rlx_ir::op::RopeStyle::GptJ {
            return self.lower_rope_gptj(
                LowerRopeGptj {
                    eff_x: &eff_x,
                    eff_shape: &eff_shape,
                    eff_cos: &eff_cos,
                    eff_sin: &eff_sin,
                    eff_table: &eff_table,
                    head_dim,
                    n_rot,
                    restore: restore.as_ref(),
                },
                out_name,
            );
        }

        // x1 = x[..0:rh], x2 = x[..rh:n_rot]
        let x1 = format!("{out_name}_x1");
        let x2 = format!("{out_name}_x2");
        self.slice_last(&eff_x, eff_rank, 0, rot_half, &half_shape, &x1)?;
        self.slice_last(&eff_x, eff_rank, rot_half, rot_half, &half_shape, &x2)?;

        // out1 = x1*cos - x2*sin ; out2 = x2*cos + x1*sin
        let (x1c, x2s, x2c, x1s) = (
            format!("{out_name}_x1c"),
            format!("{out_name}_x2s"),
            format!("{out_name}_x2c"),
            format!("{out_name}_x1s"),
        );
        self.emit(
            "mul",
            &x1c,
            &half_shape,
            vec![("x", bind_name(&x1)), ("y", bind_name(&eff_cos))],
        )?;
        self.emit(
            "mul",
            &x2s,
            &half_shape,
            vec![("x", bind_name(&x2)), ("y", bind_name(&eff_sin))],
        )?;
        self.emit(
            "mul",
            &x2c,
            &half_shape,
            vec![("x", bind_name(&x2)), ("y", bind_name(&eff_cos))],
        )?;
        self.emit(
            "mul",
            &x1s,
            &half_shape,
            vec![("x", bind_name(&x1)), ("y", bind_name(&eff_sin))],
        )?;
        let out1 = format!("{out_name}_o1");
        let out2 = format!("{out_name}_o2");
        self.emit(
            "sub",
            &out1,
            &half_shape,
            vec![("x", bind_name(&x1c)), ("y", bind_name(&x2s))],
        )?;
        self.emit(
            "add",
            &out2,
            &half_shape,
            vec![("x", bind_name(&x2c)), ("y", bind_name(&x1s))],
        )?;

        let axis = (eff_rank - 1) as i32;
        let pass_len = head_dim - n_rot;
        if pass_len == 0 {
            self.emit(
                "concat",
                &core,
                &eff_shape,
                vec![
                    ("values", bind_names(&[out1, out2])),
                    ("axis", bind_value(scalar_i32(axis))),
                    ("interleave", bind_value(scalar_bool(false))),
                ],
            )?;
        } else {
            let out_rot = format!("{out_name}_rot");
            self.emit(
                "concat",
                &out_rot,
                &rot_shape,
                vec![
                    ("values", bind_names(&[out1, out2])),
                    ("axis", bind_value(scalar_i32(axis))),
                    ("interleave", bind_value(scalar_bool(false))),
                ],
            )?;
            let pass = format!("{out_name}_pass");
            let pass_shape = with_last(&eff_shape, pass_len);
            self.slice_last(&eff_x, eff_rank, n_rot, pass_len, &pass_shape, &pass)?;
            self.emit(
                "concat",
                &core,
                &eff_shape,
                vec![
                    ("values", bind_names(&[out_rot, pass])),
                    ("axis", bind_value(scalar_i32(axis))),
                    ("interleave", bind_value(scalar_bool(false))),
                ],
            )?;
        }

        // Per-head view → fold the head axis back into the last dim.
        if let Some(orig) = restore {
            self.emit(
                "reshape",
                out_name,
                &orig,
                vec![
                    ("x", bind_name(&core)),
                    ("shape", bind_value(vec_i32(&dims_i32(orig.dims())))),
                ],
            )?;
        }
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Slice a Graph-backed cos/sin param `[max_pos, half]` down to
    /// `[seq, rot_half]` (no-op when already that size).
    fn rope_slice_table(
        &mut self,
        src: NodeId,
        val: &str,
        seq: usize,
        rot_half: usize,
        out_base: &str,
    ) -> Result<String> {
        let shape = self.graph.shape(src);
        if shape.rank() < 2 {
            return Ok(val.to_string());
        }
        let Dim::Static(rows) = shape.dim(0) else {
            return Ok(val.to_string());
        };
        let Dim::Static(last) = shape.dim(shape.rank() - 1) else {
            return Ok(val.to_string());
        };
        let mut cur = val.to_string();
        let mut cur_shape = shape.clone();

        if rows > seq {
            let mut d = cur_shape.dims().to_vec();
            d[0] = Dim::Static(seq);
            let sliced = Shape::from_dims(&d, DType::F32);
            let name = format!("{out_base}_seq");
            self.slice_axis(&cur, cur_shape.rank(), 0, 0, seq, &sliced, &name)?;
            cur = name;
            cur_shape = sliced;
        }
        if last > rot_half {
            let mut d = cur_shape.dims().to_vec();
            *d.last_mut().unwrap() = Dim::Static(rot_half);
            let sliced = Shape::from_dims(&d, DType::F32);
            let name = format!("{out_base}_rh");
            self.slice_last(&cur, cur_shape.rank(), 0, rot_half, &sliced, &name)?;
            cur = name;
        }
        Ok(cur)
    }

    /// Reshape a rope cos/sin table to gain a singleton head axis just before
    /// its last dim, so it broadcasts over the per-head groups when rope runs
    /// on a fused `[.., G, head_dim]` view.
    /// Arguments for [`Self::lower_rope_gptj`], which needs the effective
    /// tensor, table and pass-through shapes all at once.
    fn lower_rope_gptj(&mut self, a: LowerRopeGptj<'_>, out_name: &str) -> Result<()> {
        let eff_rank = a.eff_shape.rank();
        let half = a.n_rot / 2;
        let pass_len = a.head_dim - a.n_rot;
        let core = match a.restore {
            Some(_) => format!("{out_name}_core"),
            None => out_name.to_string(),
        };

        // GptJ rotates adjacent lanes, so view the rotated span as `[.., half, 2]`
        // and take the even and odd members as two `[.., half, 1]` tensors. MIL's
        // slice has no stride, so this is done with a reshape rather than a
        // strided read.
        let rot_shape = with_last(a.eff_shape, a.n_rot);
        let rot = if pass_len == 0 {
            a.eff_x.to_string()
        } else {
            let r = format!("{out_name}_rotin");
            self.slice_last(a.eff_x, eff_rank, 0, a.n_rot, &rot_shape, &r)?;
            r
        };
        let pair_dims = {
            let mut d = a.eff_shape.dims().to_vec();
            d.pop();
            d.push(Dim::Static(half));
            d.push(Dim::Static(2));
            d
        };
        let pair_shape = Shape::from_dims(&pair_dims, DType::F32);
        let xp = format!("{out_name}_xp");
        self.emit(
            "reshape",
            &xp,
            &pair_shape,
            vec![
                ("x", bind_name(&rot)),
                ("shape", bind_value(vec_i32(&dims_i32(&pair_dims)))),
            ],
        )?;
        let lane_shape = with_last(&pair_shape, 1);
        let pair_rank = pair_shape.rank();
        let (x1, x2) = (format!("{out_name}_e"), format!("{out_name}_o"));
        self.slice_last(&xp, pair_rank, 0, 1, &lane_shape, &x1)?;
        self.slice_last(&xp, pair_rank, 1, 1, &lane_shape, &x2)?;

        // the tables gain a trailing singleton so they broadcast over the pair
        let mut td = a.eff_table.dims().to_vec();
        td.push(Dim::Static(1));
        let tshape = Shape::from_dims(&td, DType::F32);
        let (cos1, sin1) = (format!("{out_name}_cosp"), format!("{out_name}_sinp"));
        for (src, dst) in [(a.eff_cos, &cos1), (a.eff_sin, &sin1)] {
            self.emit(
                "reshape",
                dst,
                &tshape,
                vec![
                    ("x", bind_name(src)),
                    ("shape", bind_value(vec_i32(&dims_i32(&td)))),
                ],
            )?;
        }

        // out_even = x1·cos − x2·sin ; out_odd = x2·cos + x1·sin
        let names = [
            format!("{out_name}_x1c"),
            format!("{out_name}_x2s"),
            format!("{out_name}_x2c"),
            format!("{out_name}_x1s"),
        ];
        for (dst, (l, r)) in
            names
                .iter()
                .zip([(&x1, &cos1), (&x2, &sin1), (&x2, &cos1), (&x1, &sin1)])
        {
            self.emit(
                "mul",
                dst,
                &lane_shape,
                vec![("x", bind_name(l)), ("y", bind_name(r))],
            )?;
        }
        let (o1, o2) = (format!("{out_name}_o1"), format!("{out_name}_o2"));
        self.emit(
            "sub",
            &o1,
            &lane_shape,
            vec![("x", bind_name(&names[0])), ("y", bind_name(&names[1]))],
        )?;
        self.emit(
            "add",
            &o2,
            &lane_shape,
            vec![("x", bind_name(&names[2])), ("y", bind_name(&names[3]))],
        )?;

        // back to `[.., half, 2]` — concatenating on the pair axis *is* the
        // interleave — then flatten to `[.., n_rot]`
        let paired = format!("{out_name}_paired");
        self.emit(
            "concat",
            &paired,
            &pair_shape,
            vec![
                ("values", bind_names(&[o1, o2])),
                ("axis", bind_value(scalar_i32((pair_rank - 1) as i32))),
                ("interleave", bind_value(scalar_bool(false))),
            ],
        )?;
        let rotated = format!("{out_name}_rotout");
        let rd = rot_shape.dims().to_vec();
        self.emit(
            "reshape",
            &rotated,
            &rot_shape,
            vec![
                ("x", bind_name(&paired)),
                ("shape", bind_value(vec_i32(&dims_i32(&rd)))),
            ],
        )?;

        if pass_len == 0 {
            // `core` must name the rotated tensor; emit an identity reshape so
            // the downstream name resolves without a second code path.
            self.emit(
                "reshape",
                &core,
                &rot_shape,
                vec![
                    ("x", bind_name(&rotated)),
                    ("shape", bind_value(vec_i32(&dims_i32(&rd)))),
                ],
            )?;
        } else {
            let pass = format!("{out_name}_pass");
            let pass_shape = with_last(a.eff_shape, pass_len);
            self.slice_last(a.eff_x, eff_rank, a.n_rot, pass_len, &pass_shape, &pass)?;
            self.emit(
                "concat",
                &core,
                a.eff_shape,
                vec![
                    ("values", bind_names(&[rotated, pass])),
                    ("axis", bind_value(scalar_i32((eff_rank - 1) as i32))),
                    ("interleave", bind_value(scalar_bool(false))),
                ],
            )?;
        }

        if let Some(orig) = a.restore {
            let od = orig.dims().to_vec();
            self.emit(
                "reshape",
                out_name,
                orig,
                vec![
                    ("x", bind_name(&core)),
                    ("shape", bind_value(vec_i32(&dims_i32(&od)))),
                ],
            )?;
        }
        Ok(())
    }

    fn rope_insert_head_axis_shape(
        &mut self,
        shape: &Shape,
        val: &str,
        out: &str,
    ) -> Result<String> {
        let mut d = shape.dims().to_vec();
        let pos = d.len().saturating_sub(1);
        d.insert(pos, Dim::Static(1));
        let ns = Shape::from_dims(&d, DType::F32);
        self.emit(
            "reshape",
            out,
            &ns,
            vec![
                ("x", bind_name(val)),
                ("shape", bind_value(vec_i32(&dims_i32(&d)))),
            ],
        )?;
        Ok(out.to_string())
    }

    /// 2D axial RoPE (SAM2-style), input `[B, seq, num_heads·head_dim]`.
    /// Interleaved-pair rotation: first half rotated by the x-position
    /// angle, second half by y. All angle tables are baked at lowering
    /// time, then applied as `x·cos + rot_interleaved(x)·sin`, where
    /// `rot_interleaved` maps each pair `(a,b) → (-b, a)`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lower_axial_rope2d(
        &mut self,
        id: NodeId,
        end_x: usize,
        end_y: usize,
        head_dim: usize,
        num_heads: usize,
        theta: f32,
        repeat_factor: usize,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let shape = node.shape.clone();
        if shape.rank() != 3 {
            return Err(CoremlError::Unsupported(
                "axial_rope2d: only [B, seq, H*D]".into(),
            ));
        }
        let b = dim_static(&shape, 0)?;
        let seq = dim_static(&shape, 1)?;
        let hd = dim_static(&shape, 2)?; // num_heads * head_dim

        // Bake cos/sin tables [seq, hd] (duplicated per interleaved pair).
        let (cos_full, sin_full) = axial_tables(
            end_x,
            end_y,
            head_dim,
            num_heads,
            theta,
            repeat_factor,
            seq,
            hd,
        );
        let tab_shape = Shape::new(&[seq, hd], DType::F32);
        let cosf = format!("{out_name}_cos");
        let sinf = format!("{out_name}_sin");
        self.operations
            .push(make_const(&mut self.blob, &cosf, &tab_shape, &cos_full)?);
        self.operations
            .push(make_const(&mut self.blob, &sinf, &tab_shape, &sin_full)?);

        let x = self.val(node.inputs[0]);
        // rot_interleaved: reshape to pairs, swap+negate, reshape back.
        let pair_shape = Shape::new(&[b, seq, hd / 2, 2], DType::F32);
        let one_shape = Shape::new(&[b, seq, hd / 2, 1], DType::F32);
        let xr = format!("{out_name}_xr");
        self.reshape_to(
            &x,
            &[b as i64, seq as i64, (hd / 2) as i64, 2],
            &pair_shape,
            &xr,
        )?;
        let even = format!("{out_name}_even");
        let odd = format!("{out_name}_odd");
        self.slice_last(&xr, 4, 0, 1, &one_shape, &even)?;
        self.slice_last(&xr, 4, 1, 1, &one_shape, &odd)?;
        let neg_odd = format!("{out_name}_nodd");
        self.emit(
            "mul",
            &neg_odd,
            &one_shape,
            vec![("x", bind_name(&odd)), ("y", bind_value(scalar_f32(-1.0)))],
        )?;
        let rot4 = format!("{out_name}_rot4");
        self.emit(
            "concat",
            &rot4,
            &pair_shape,
            vec![
                ("values", bind_names(&[neg_odd, even])),
                ("axis", bind_value(scalar_i32(3))),
                ("interleave", bind_value(scalar_bool(false))),
            ],
        )?;
        let rot = format!("{out_name}_rot");
        self.reshape_to(&rot4, &[b as i64, seq as i64, hd as i64], &shape, &rot)?;

        // out = x*cos + rot*sin
        let t1 = format!("{out_name}_t1");
        let t2 = format!("{out_name}_t2");
        self.emit(
            "mul",
            &t1,
            &shape,
            vec![("x", bind_name(&x)), ("y", bind_name(&cosf))],
        )?;
        self.emit(
            "mul",
            &t2,
            &shape,
            vec![("x", bind_name(&rot)), ("y", bind_name(&sinf))],
        )?;
        self.emit(
            "add",
            out_name,
            &shape,
            vec![("x", bind_name(&t1)), ("y", bind_name(&t2))],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }
}
