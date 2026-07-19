// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
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

/// Sequence length axis for NeoX RoPE layouts.
fn rope_seq_len(shape: &Shape, heads_packed: bool) -> Result<usize> {
    let rank = shape.rank();
    let idx = if heads_packed {
        // `[B, S, G, head_dim]`
        1
    } else if rank == 4 {
        // `[B, H, S, D]`
        2
    } else if rank >= 2 {
        // `[B, S, D]`
        1
    } else {
        return Err(CoremlError::Unsupported(format!(
            "rope: cannot infer seq from rank-{rank} shape {:?}",
            shape.dims()
        )));
    };
    match shape.dim(idx) {
        Dim::Static(s) => Ok(s),
        Dim::Dynamic(s) => Err(CoremlError::DynamicShape(format!("rope seq ?{s}"))),
    }
}

impl<'a> LowerCtx<'a> {
    /// RoPE (NeoX split-halves). Inputs `[x, cos, sin]`; rotates the first
    /// `n_rot` of the trailing `head_dim` lane, passes the rest through.
    /// Only the layout where the last axis == `head_dim` is supported
    /// (`[B,H,S,D]` or `[B,S,D]`); the cos/sin tables (`[…,n_rot/2]`)
    /// broadcast against the rotated halves.
    ///
    /// Production tables are often `[max_pos, head_half]` with
    /// `max_pos ≫ seq` (Qwen3.5 / Bonsai: max_pos=262144). Metal/CPU RoPE
    /// index by position; this lowering slices tables to `[seq, rot_half]`
    /// before any mul so Espresso broadcast matches the activation.
    pub(crate) fn lower_rope(
        &mut self,
        id: NodeId,
        head_dim: usize,
        n_rot: usize,
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
        let (eff_x, eff_shape, eff_cos, eff_sin, restore) = if last == head_dim {
            (x, shape.clone(), cos, sin, None)
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
            (xg, gshape, cos_g, sin_g, Some(shape.clone()))
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
