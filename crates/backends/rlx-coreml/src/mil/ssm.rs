// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation: takes
// an RLX `Graph` plus baked parameter/constant data and produces a
// `proto::Model` ready to serialise into a `.mlpackage`. No FFI, so this
// builds and unit-tests on any host.

//! `ssm` — extracted from the `mil` module for navigability (see `mod.rs`).

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

impl<'a> LowerCtx<'a> {
    /// Mamba selective scan, unrolled over the sequence. Inputs
    /// `[x, delta, A, B, C]` with `x,delta:[b,s,h]`, `A:[h,n]`,
    /// `B,C:[b,s,n]`. Per step: `H ← exp(Δ·A)·H + (Δ·x)·B`, `y = Σₙ C·H`.
    pub(crate) fn lower_selective_scan(
        &mut self,
        id: NodeId,
        n: usize,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let out_shape = node.shape.clone();
        let b = dim_static(&out_shape, 0)?;
        let s = dim_static(&out_shape, 1)?;
        let h = dim_static(&out_shape, 2)?;
        let x = self.val(node.inputs[0]);
        let delta = self.val(node.inputs[1]);
        let a = self.val(node.inputs[2]);
        let b_in = self.val(node.inputs[3]);
        let c_in = self.val(node.inputs[4]);

        let bhn = Shape::new(&[b, h, n], DType::F32);
        let bh1 = Shape::new(&[b, h, 1], DType::F32);
        let b1h = Shape::new(&[b, 1, h], DType::F32);
        let b1n = Shape::new(&[b, 1, n], DType::F32);
        let bh = Shape::new(&[b, h], DType::F32);

        // A → [1, h, n]
        let a3 = format!("{out_name}_a3");
        self.reshape_to(
            &a,
            &[1, h as i64, n as i64],
            &Shape::new(&[1, h, n], DType::F32),
            &a3,
        )?;
        // state₀ = 0
        let mut state = format!("{out_name}_s0");
        self.operations.push(make_const(
            &mut self.blob,
            &state,
            &bhn,
            &vec![0.0f32; b * h * n],
        )?);

        let mut ys = Vec::with_capacity(s);
        for t in 0..s {
            let p = format!("{out_name}_t{t}");
            let xt = format!("{p}_x");
            let xt3 = format!("{p}_x3");
            self.slice_axis(&x, 3, 1, t, 1, &b1h, &xt)?;
            self.reshape_to(&xt, &[b as i64, h as i64, 1], &bh1, &xt3)?;
            let dt = format!("{p}_d");
            let dt3 = format!("{p}_d3");
            self.slice_axis(&delta, 3, 1, t, 1, &b1h, &dt)?;
            self.reshape_to(&dt, &[b as i64, h as i64, 1], &bh1, &dt3)?;
            let bt = format!("{p}_b");
            self.slice_axis(&b_in, 3, 1, t, 1, &b1n, &bt)?;
            let ct = format!("{p}_c");
            self.slice_axis(&c_in, 3, 1, t, 1, &b1n, &ct)?;

            // da = exp(Δ·A)
            let dta = format!("{p}_dta");
            self.emit(
                "mul",
                &dta,
                &bhn,
                vec![("x", bind_name(&dt3)), ("y", bind_name(&a3))],
            )?;
            let da = format!("{p}_da");
            self.emit("exp", &da, &bhn, vec![("x", bind_name(&dta))])?;
            let decay = format!("{p}_decay");
            self.emit(
                "mul",
                &decay,
                &bhn,
                vec![("x", bind_name(&da)), ("y", bind_name(&state))],
            )?;
            // input term (Δ·x)·B
            let dx = format!("{p}_dx");
            self.emit(
                "mul",
                &dx,
                &bh1,
                vec![("x", bind_name(&dt3)), ("y", bind_name(&xt3))],
            )?;
            let inp = format!("{p}_inp");
            self.emit(
                "mul",
                &inp,
                &bhn,
                vec![("x", bind_name(&dx)), ("y", bind_name(&bt))],
            )?;
            let snew = format!("{p}_s");
            self.emit(
                "add",
                &snew,
                &bhn,
                vec![("x", bind_name(&decay)), ("y", bind_name(&inp))],
            )?;
            state = snew;
            // y = Σₙ C·H
            let prod = format!("{p}_pr");
            self.emit(
                "mul",
                &prod,
                &bhn,
                vec![("x", bind_name(&ct)), ("y", bind_name(&state))],
            )?;
            let yt = format!("{p}_y");
            self.emit(
                "reduce_sum",
                &yt,
                &bh,
                vec![
                    ("x", bind_name(&prod)),
                    ("axes", bind_value(vec_i32(&[2]))),
                    ("keep_dims", bind_value(scalar_bool(false))),
                ],
            )?;
            let yt3 = format!("{p}_y3");
            self.reshape_to(&yt, &[b as i64, 1, h as i64], &b1h, &yt3)?;
            ys.push(yt3);
        }
        self.emit(
            "concat",
            out_name,
            &out_shape,
            vec![
                ("values", bind_names(&ys)),
                ("axis", bind_value(scalar_i32(1))),
                ("interleave", bind_value(scalar_bool(false))),
            ],
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Qwen3.5 gated delta-net, unrolled over the sequence. Inputs
    /// `[q,k,v,g,beta(,state)]`, `q,k,v:[b,s,H,n]`, `beta:[b,s,H]`.
    /// Per step: `S ← exp(g)·S`; `Δ=(v − Sᵀk)·β`; `S += k⊗Δ`;
    /// `y = (1/√n)·Sᵀq`.
    ///
    /// When `gate_per_channel` (KDA), `g` is `[b,s,H,n]` and the decay is
    /// per row of the state: `S[i,j] ← exp(g[i])·S[i,j]` — reshape the
    /// step's `g` to `[b,H,n,1]` so it broadcasts over the column axis.
    /// Otherwise `g` is `[b,s,H]` and `exp(g)` is one scalar per head.
    pub(crate) fn lower_gated_delta_net(
        &mut self,
        id: NodeId,
        n: usize,
        carry: bool,
        gate_per_channel: bool,
        out_name: &str,
    ) -> Result<()> {
        let node = self.graph.node(id);
        let out_shape = node.shape.clone(); // [b,s,H,n]
        let b = dim_static(&out_shape, 0)?;
        let s = dim_static(&out_shape, 1)?;
        let hh = dim_static(&out_shape, 2)?;
        let scale = (n as f32).powf(-0.5);

        let q = self.val(node.inputs[0]);
        let k = self.val(node.inputs[1]);
        let v = self.val(node.inputs[2]);
        let g = self.val(node.inputs[3]);
        let beta = self.val(node.inputs[4]);

        let bhnn = Shape::new(&[b, hh, n, n], DType::F32);
        let bh1n = Shape::new(&[b, hh, 1, n], DType::F32);
        let bhn1 = Shape::new(&[b, hh, n, 1], DType::F32);
        let bh11 = Shape::new(&[b, hh, 1, 1], DType::F32);
        let bsh1 = Shape::new(&[b, 1, hh], DType::F32);

        // state₀ — external [b,H,n,n] or zeros.
        let mut state = if carry {
            self.val(node.inputs[5])
        } else {
            let s0 = format!("{out_name}_s0");
            self.operations.push(make_const(
                &mut self.blob,
                &s0,
                &bhnn,
                &vec![0.0f32; b * hh * n * n],
            )?);
            s0
        };

        let mut ys = Vec::with_capacity(s);
        for t in 0..s {
            let p = format!("{out_name}_t{t}");
            // slice token t: q,k,v → [b,1,H,n] → [b,H,1,n]; g,beta → [b,H,1,1]
            let qt = self.gdn_vec(&q, t, b, hh, n, &p, "q")?;
            let kt = self.gdn_vec(&k, t, b, hh, n, &p, "k")?;
            let vt = self.gdn_vec(&v, t, b, hh, n, &p, "v")?;
            let bt = self.gdn_scalar(&beta, t, b, hh, &p, "b")?;

            // S *= exp(g). Per-channel (KDA): g is [b,s,H,n]; reshape the
            // step's row-vector to [b,H,n,1] so exp(g) broadcasts down the
            // state rows (S[i,j] *= exp(g[i])). Per-head: one scalar/head.
            let ge = format!("{p}_ge");
            if gate_per_channel {
                let gv = self.gdn_vec(&g, t, b, hh, n, &p, "g")?; // [b,H,1,n]
                let gcol = format!("{p}_gcol");
                self.reshape_to(&gv, &[b as i64, hh as i64, n as i64, 1], &bhn1, &gcol)?;
                self.emit("exp", &ge, &bhn1, vec![("x", bind_name(&gcol))])?;
            } else {
                let gt = self.gdn_scalar(&g, t, b, hh, &p, "g")?; // [b,H,1,1]
                self.emit("exp", &ge, &bh11, vec![("x", bind_name(&gt))])?;
            }
            let sg = format!("{p}_sg");
            self.emit(
                "mul",
                &sg,
                &bhnn,
                vec![("x", bind_name(&state)), ("y", bind_name(&ge))],
            )?;
            // sk = kᵀ·S  → [b,H,1,n] (matmul [b,H,1,n] @ [b,H,n,n])
            let sk = format!("{p}_sk");
            self.matmul_op(&sk, &kt, &sg, false, false, &bh1n)?;
            // Δ = (v − sk)·β
            let d0 = format!("{p}_d0");
            self.emit(
                "sub",
                &d0,
                &bh1n,
                vec![("x", bind_name(&vt)), ("y", bind_name(&sk))],
            )?;
            let delta = format!("{p}_dl");
            self.emit(
                "mul",
                &delta,
                &bh1n,
                vec![("x", bind_name(&d0)), ("y", bind_name(&bt))],
            )?;
            // S += k⊗Δ  (kᵀ:[b,H,n,1] · Δ:[b,H,1,n] → [b,H,n,n])
            let kcol = format!("{p}_kc");
            self.reshape_to(&kt, &[b as i64, hh as i64, n as i64, 1], &bhn1, &kcol)?;
            let outer = format!("{p}_outer");
            self.matmul_op(&outer, &kcol, &delta, false, false, &bhnn)?;
            let snew = format!("{p}_s");
            self.emit(
                "add",
                &snew,
                &bhnn,
                vec![("x", bind_name(&sg)), ("y", bind_name(&outer))],
            )?;
            state = snew;
            // y = scale·(qᵀ·S) → [b,H,1,n]
            let qs = format!("{p}_qs");
            self.matmul_op(&qs, &qt, &state, false, false, &bh1n)?;
            let yt = format!("{p}_y");
            self.emit(
                "mul",
                &yt,
                &bh1n,
                vec![("x", bind_name(&qs)), ("y", bind_value(scalar_f32(scale)))],
            )?;
            // → [b,1,H,n]
            let yt2 = format!("{p}_y2");
            self.reshape_to(
                &yt,
                &[b as i64, 1, hh as i64, n as i64],
                &Shape::new(&[b, 1, hh, n], DType::F32),
                &yt2,
            )?;
            ys.push(yt2);
        }
        let _ = bsh1;
        self.emit(
            "concat",
            out_name,
            &out_shape,
            vec![
                ("values", bind_names(&ys)),
                ("axis", bind_value(scalar_i32(1))),
                ("interleave", bind_value(scalar_bool(false))),
            ],
        )?;
        // Publish the final state under the state input's name, so a decode
        // harness that reads the state node back as a graph output sees the
        // UPDATED state. Without this the node still resolves to the initial
        // state and every decode step restarts from it — the recurrence silently
        // stops carrying. Mirrors rlx-mlx's `env.insert(node.inputs[5], …)`.
        if carry {
            let state_in = self.graph.node(id).inputs[5];
            self.names.insert(state_in.0, state.clone());
        }
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Slice token `t` of a `[b,s,H,n]` GDN tensor → `[b,H,1,n]`.
    pub(crate) fn gdn_vec(
        &mut self,
        src: &str,
        t: usize,
        b: usize,
        hh: usize,
        n: usize,
        p: &str,
        tag: &str,
    ) -> Result<String> {
        let sl = format!("{p}_{tag}sl");
        self.slice_axis(
            src,
            4,
            1,
            t,
            1,
            &Shape::new(&[b, 1, hh, n], DType::F32),
            &sl,
        )?;
        let out = format!("{p}_{tag}");
        self.reshape_to(
            &sl,
            &[b as i64, hh as i64, 1, n as i64],
            &Shape::new(&[b, hh, 1, n], DType::F32),
            &out,
        )?;
        Ok(out)
    }

    /// Slice token `t` of a `[b,s,H]` GDN scalar tensor → `[b,H,1,1]`.
    pub(crate) fn gdn_scalar(
        &mut self,
        src: &str,
        t: usize,
        b: usize,
        hh: usize,
        p: &str,
        tag: &str,
    ) -> Result<String> {
        let sl = format!("{p}_{tag}sl");
        self.slice_axis(src, 3, 1, t, 1, &Shape::new(&[b, 1, hh], DType::F32), &sl)?;
        let out = format!("{p}_{tag}");
        self.reshape_to(
            &sl,
            &[b as i64, hh as i64, 1, 1],
            &Shape::new(&[b, hh, 1, 1], DType::F32),
            &out,
        )?;
        Ok(out)
    }
}
