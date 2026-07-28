// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// IR → CoreML ML Program (MIL) lowering. Pure data transformation.

//! `rnn` — native (on-device) `Op::Lstm` for `carry = false`, unrolled over
//! the sequence into MIL primitives (matmul / add / sigmoid / tanh / mul /
//! slice / concat), mirroring the shared CPU kernel `execute_lstm_f32` and the
//! MLX `native_lstm` path. Extracted from `mod.rs` for navigability.
//!
//! The MIL built-in `lstm` op was evaluated first but does not map cleanly to
//! `Op::Lstm`: it needs a different gate order, per-direction *const* weight
//! args, and a fixed time-major contract, whereas RLX packs `[i,f,g,o]` gates
//! for every layer/direction into single (often runtime) weight streams. The
//! unroll reuses tested MIL ops and matches the CPU/MLX numerics exactly.

#![allow(unused_imports)]

use super::helpers::*;
use super::*;
use crate::proto;
use crate::{CoremlError, Result};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

impl<'a> LowerCtx<'a> {
    /// PyTorch LSTM, `carry = false`. Inputs `[x, w_ih, w_hh, bias]` with
    /// `x:[b,s,in]`. Packing mirrors `execute_lstm_f32`: `w_ih` is, per layer,
    /// `dirs` blocks of `[4h, in_l]` (row-major gate rows `i,f,g,o`; `in_l = in`
    /// for layer 0 else `dirs*h`); `w_hh` `[L*dirs][4h,h]`; `bias` `[L*dirs][4h]`.
    /// Output `[b, s, dirs*h]` — direction `d` owns feature slice `d*h..d*h+h`;
    /// the reverse pass writes hidden states at their natural time index.
    pub(crate) fn lower_lstm(
        &mut self,
        id: NodeId,
        hidden: usize,
        num_layers: usize,
        bidirectional: bool,
        out_name: &str,
    ) -> Result<()> {
        let inputs: Vec<NodeId> = self.graph.node(id).inputs.clone();
        let x_sh = self.graph.shape(inputs[0]).clone();
        let b = dim_static(&x_sh, 0)?;
        let s = dim_static(&x_sh, 1)?;
        let input_size = dim_static(&x_sh, 2)?;
        let h = hidden;
        let dirs = if bidirectional { 2 } else { 1 };
        let fh = 4 * h;

        // Flatten each packed weight stream to rank-1 so per-(layer,dir) element
        // offsets address the rank-1 packed and rank-2 ONNX-matrix forms alike.
        let wih_total: usize = (0..num_layers)
            .map(|l| dirs * fh * if l == 0 { input_size } else { dirs * h })
            .sum();
        let whh_total = num_layers * dirs * fh * h;
        let bias_total = num_layers * dirs * fh;
        let wih = self.flatten_1d(inputs[1], wih_total, &format!("{out_name}_wihf"))?;
        let whh = self.flatten_1d(inputs[2], whh_total, &format!("{out_name}_whhf"))?;
        let bias = self.flatten_1d(inputs[3], bias_total, &format!("{out_name}_biasf"))?;

        let bh = Shape::new(&[b, h], self.opts.float_dtype);
        let bfh = Shape::new(&[b, fh], self.opts.float_dtype);

        let mut layer_in = self.val(inputs[0]);
        let mut in_l = input_size;
        let mut wih_cursor = 0usize;

        for l in 0..num_layers {
            let out_width = dirs * h;
            let wih_block = fh * in_l;
            let mut dir_outs: Vec<String> = Vec::with_capacity(dirs);

            for d in 0..dirs {
                let ld = l * dirs + d;
                let p = format!("{out_name}_l{l}d{d}");

                // Per-(layer,dir) weight blocks: w_ih [4h,in_l], w_hh [4h,h],
                // bias [1,4h].
                let wih_b = format!("{p}_wih");
                self.slice_1d_to(
                    &wih,
                    wih_cursor + d * wih_block,
                    wih_block,
                    &[fh as i64, in_l as i64],
                    &Shape::new(&[fh, in_l], self.opts.float_dtype),
                    &wih_b,
                )?;
                let whh_b = format!("{p}_whh");
                self.slice_1d_to(
                    &whh,
                    ld * fh * h,
                    fh * h,
                    &[fh as i64, h as i64],
                    &Shape::new(&[fh, h], self.opts.float_dtype),
                    &whh_b,
                )?;
                let bias_b = format!("{p}_bias");
                self.slice_1d_to(
                    &bias,
                    ld * fh,
                    fh,
                    &[1, fh as i64],
                    &Shape::new(&[1, fh], self.opts.float_dtype),
                    &bias_b,
                )?;

                // Input projection for the whole sequence in one GEMM:
                // gates_x = x @ w_ihᵀ + b  → [b, s, 4h].
                let li2 = format!("{p}_li2");
                self.reshape_to(
                    &layer_in,
                    &[(b * s) as i64, in_l as i64],
                    &Shape::new(&[b * s, in_l], self.opts.float_dtype),
                    &li2,
                )?;
                let gx2 = format!("{p}_gx2");
                self.matmul_op(
                    &gx2,
                    &li2,
                    &wih_b,
                    false,
                    true,
                    &Shape::new(&[b * s, fh], self.opts.float_dtype),
                )?;
                let gx2b = format!("{p}_gx2b");
                self.emit(
                    "add",
                    &gx2b,
                    &Shape::new(&[b * s, fh], self.opts.float_dtype),
                    vec![("x", bind_name(&gx2)), ("y", bind_name(&bias_b))],
                )?;
                let gx = format!("{p}_gx");
                self.reshape_to(
                    &gx2b,
                    &[b as i64, s as i64, fh as i64],
                    &Shape::new(&[b, s, fh], self.opts.float_dtype),
                    &gx,
                )?;

                // Recurrence. h, c start at zero (carry = false).
                let mut hst = format!("{p}_h0");
                let mut cst = format!("{p}_c0");
                self.operations
                    .push(make_const(&mut self.blob, &hst, &bh, &vec![0.0f32; b * h])?);
                self.operations
                    .push(make_const(&mut self.blob, &cst, &bh, &vec![0.0f32; b * h])?);

                let mut steps: Vec<String> = vec![String::new(); s];
                for step in 0..s {
                    let t = if d == 0 { step } else { s - 1 - step };
                    let tp = format!("{p}_t{t}");

                    // z = gates_x[:, t, :] + h · w_hhᵀ  → [b, 4h].
                    let gxt3 = format!("{tp}_gxt3");
                    self.slice_axis(
                        &gx,
                        3,
                        1,
                        t,
                        1,
                        &Shape::new(&[b, 1, fh], self.opts.float_dtype),
                        &gxt3,
                    )?;
                    let gxt = format!("{tp}_gxt");
                    self.reshape_to(&gxt3, &[b as i64, fh as i64], &bfh, &gxt)?;
                    let hh = format!("{tp}_hh");
                    self.matmul_op(&hh, &hst, &whh_b, false, true, &bfh)?;
                    let z = format!("{tp}_z");
                    self.emit(
                        "add",
                        &z,
                        &bfh,
                        vec![("x", bind_name(&gxt)), ("y", bind_name(&hh))],
                    )?;

                    // Gates (order i, f, g, o).
                    let zi = format!("{tp}_zi");
                    self.slice_axis(&z, 2, 1, 0, h, &bh, &zi)?;
                    let zf = format!("{tp}_zf");
                    self.slice_axis(&z, 2, 1, h, h, &bh, &zf)?;
                    let zg = format!("{tp}_zg");
                    self.slice_axis(&z, 2, 1, 2 * h, h, &bh, &zg)?;
                    let zo = format!("{tp}_zo");
                    self.slice_axis(&z, 2, 1, 3 * h, h, &bh, &zo)?;
                    let ig = format!("{tp}_ig");
                    self.emit("sigmoid", &ig, &bh, vec![("x", bind_name(&zi))])?;
                    let fg = format!("{tp}_fg");
                    self.emit("sigmoid", &fg, &bh, vec![("x", bind_name(&zf))])?;
                    let gg = format!("{tp}_gg");
                    self.emit("tanh", &gg, &bh, vec![("x", bind_name(&zg))])?;
                    let og = format!("{tp}_og");
                    self.emit("sigmoid", &og, &bh, vec![("x", bind_name(&zo))])?;

                    // c = f⊙c + i⊙g ; h = o ⊙ tanh(c).
                    let fc = format!("{tp}_fc");
                    self.emit(
                        "mul",
                        &fc,
                        &bh,
                        vec![("x", bind_name(&fg)), ("y", bind_name(&cst))],
                    )?;
                    let igg = format!("{tp}_igg");
                    self.emit(
                        "mul",
                        &igg,
                        &bh,
                        vec![("x", bind_name(&ig)), ("y", bind_name(&gg))],
                    )?;
                    let cnew = format!("{tp}_c");
                    self.emit(
                        "add",
                        &cnew,
                        &bh,
                        vec![("x", bind_name(&fc)), ("y", bind_name(&igg))],
                    )?;
                    let tc = format!("{tp}_tc");
                    self.emit("tanh", &tc, &bh, vec![("x", bind_name(&cnew))])?;
                    let hnew = format!("{tp}_h");
                    self.emit(
                        "mul",
                        &hnew,
                        &bh,
                        vec![("x", bind_name(&og)), ("y", bind_name(&tc))],
                    )?;

                    let hb = format!("{tp}_hb");
                    self.reshape_to(
                        &hnew,
                        &[b as i64, 1, h as i64],
                        &Shape::new(&[b, 1, h], self.opts.float_dtype),
                        &hb,
                    )?;
                    steps[t] = hb;
                    hst = hnew;
                    cst = cnew;
                }

                // Concat over time → [b, s, h] for this direction.
                let dir_out = format!("{p}_out");
                self.emit(
                    "concat",
                    &dir_out,
                    &Shape::new(&[b, s, h], self.opts.float_dtype),
                    vec![
                        ("values", bind_names(&steps)),
                        ("axis", bind_value(scalar_i32(1))),
                        ("interleave", bind_value(scalar_bool(false))),
                    ],
                )?;
                dir_outs.push(dir_out);
            }

            // Concat directions on the feature axis → [b, s, dirs*h].
            let layer_out = if dirs == 1 {
                dir_outs.pop().expect("one direction")
            } else {
                let lo = format!("{out_name}_l{l}out");
                self.emit(
                    "concat",
                    &lo,
                    &Shape::new(&[b, s, out_width], self.opts.float_dtype),
                    vec![
                        ("values", bind_names(&dir_outs)),
                        ("axis", bind_value(scalar_i32(2))),
                        ("interleave", bind_value(scalar_bool(false))),
                    ],
                )?;
                lo
            };
            layer_in = layer_out;
            in_l = out_width;
            wih_cursor += dirs * wih_block;
        }

        // Materialize the node's value under `out_name` (identity reshape keeps
        // the value-name contract other lowerings rely on).
        self.reshape_to(
            &layer_in,
            &[b as i64, s as i64, (dirs * h) as i64],
            &Shape::new(&[b, s, dirs * h], self.opts.float_dtype),
            out_name,
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// PyTorch GRU, `carry = false`. Inputs `[x, w_ih, w_hh, b_ih, b_hh]`.
    /// Gate order r, z, n; the reset gate multiplies the hidden term *after*
    /// its bias (`n = tanh(xₙ + r ⊙ hₙ)`, `h' = (1−z)⊙n + z⊙h`), so `b_ih`
    /// and `b_hh` stay separate. Packing mirrors `execute_gru_f32` with `3h`
    /// gate rows. Output `[b, s, dirs*h]`.
    pub(crate) fn lower_gru(
        &mut self,
        id: NodeId,
        hidden: usize,
        num_layers: usize,
        bidirectional: bool,
        out_name: &str,
    ) -> Result<()> {
        let inputs: Vec<NodeId> = self.graph.node(id).inputs.clone();
        let x_sh = self.graph.shape(inputs[0]).clone();
        let b = dim_static(&x_sh, 0)?;
        let s = dim_static(&x_sh, 1)?;
        let input_size = dim_static(&x_sh, 2)?;
        let h = hidden;
        let dirs = if bidirectional { 2 } else { 1 };
        let g3 = 3 * h;

        let wih_total: usize = (0..num_layers)
            .map(|l| dirs * g3 * if l == 0 { input_size } else { dirs * h })
            .sum();
        let whh_total = num_layers * dirs * g3 * h;
        let b_total = num_layers * dirs * g3;
        let wih = self.flatten_1d(inputs[1], wih_total, &format!("{out_name}_wihf"))?;
        let whh = self.flatten_1d(inputs[2], whh_total, &format!("{out_name}_whhf"))?;
        let bih = self.flatten_1d(inputs[3], b_total, &format!("{out_name}_bihf"))?;
        let bhh = self.flatten_1d(inputs[4], b_total, &format!("{out_name}_bhhf"))?;

        let bh = Shape::new(&[b, h], self.opts.float_dtype);
        let bg = Shape::new(&[b, g3], self.opts.float_dtype);

        let mut layer_in = self.val(inputs[0]);
        let mut in_l = input_size;
        let mut wih_cursor = 0usize;

        for l in 0..num_layers {
            let out_width = dirs * h;
            let wih_block = g3 * in_l;
            let mut dir_outs: Vec<String> = Vec::with_capacity(dirs);

            for d in 0..dirs {
                let ld = l * dirs + d;
                let p = format!("{out_name}_l{l}d{d}");

                let wih_b = format!("{p}_wih");
                self.slice_1d_to(
                    &wih,
                    wih_cursor + d * wih_block,
                    wih_block,
                    &[g3 as i64, in_l as i64],
                    &Shape::new(&[g3, in_l], self.opts.float_dtype),
                    &wih_b,
                )?;
                let whh_b = format!("{p}_whh");
                self.slice_1d_to(
                    &whh,
                    ld * g3 * h,
                    g3 * h,
                    &[g3 as i64, h as i64],
                    &Shape::new(&[g3, h], self.opts.float_dtype),
                    &whh_b,
                )?;
                let bih_b = format!("{p}_bih");
                self.slice_1d_to(
                    &bih,
                    ld * g3,
                    g3,
                    &[1, g3 as i64],
                    &Shape::new(&[1, g3], self.opts.float_dtype),
                    &bih_b,
                )?;
                let bhh_b = format!("{p}_bhh");
                self.slice_1d_to(
                    &bhh,
                    ld * g3,
                    g3,
                    &[1, g3 as i64],
                    &Shape::new(&[1, g3], self.opts.float_dtype),
                    &bhh_b,
                )?;

                // xi = x @ w_ihᵀ + b_ih  → [b, s, 3h].
                let li2 = format!("{p}_li2");
                self.reshape_to(
                    &layer_in,
                    &[(b * s) as i64, in_l as i64],
                    &Shape::new(&[b * s, in_l], self.opts.float_dtype),
                    &li2,
                )?;
                let xi2 = format!("{p}_xi2");
                self.matmul_op(
                    &xi2,
                    &li2,
                    &wih_b,
                    false,
                    true,
                    &Shape::new(&[b * s, g3], self.opts.float_dtype),
                )?;
                let xi2b = format!("{p}_xi2b");
                self.emit(
                    "add",
                    &xi2b,
                    &Shape::new(&[b * s, g3], self.opts.float_dtype),
                    vec![("x", bind_name(&xi2)), ("y", bind_name(&bih_b))],
                )?;
                let xi = format!("{p}_xi");
                self.reshape_to(
                    &xi2b,
                    &[b as i64, s as i64, g3 as i64],
                    &Shape::new(&[b, s, g3], self.opts.float_dtype),
                    &xi,
                )?;

                let mut hst = format!("{p}_h0");
                self.operations
                    .push(make_const(&mut self.blob, &hst, &bh, &vec![0.0f32; b * h])?);

                let mut steps: Vec<String> = vec![String::new(); s];
                for step in 0..s {
                    let t = if d == 0 { step } else { s - 1 - step };
                    let tp = format!("{p}_t{t}");

                    let xit3 = format!("{tp}_xit3");
                    self.slice_axis(
                        &xi,
                        3,
                        1,
                        t,
                        1,
                        &Shape::new(&[b, 1, g3], self.opts.float_dtype),
                        &xit3,
                    )?;
                    let xit = format!("{tp}_xit");
                    self.reshape_to(&xit3, &[b as i64, g3 as i64], &bg, &xit)?;
                    // hi = h · w_hhᵀ + b_hh  → [b, 3h].
                    let hi0 = format!("{tp}_hi0");
                    self.matmul_op(&hi0, &hst, &whh_b, false, true, &bg)?;
                    let hi = format!("{tp}_hi");
                    self.emit(
                        "add",
                        &hi,
                        &bg,
                        vec![("x", bind_name(&hi0)), ("y", bind_name(&bhh_b))],
                    )?;

                    let xr = format!("{tp}_xr");
                    self.slice_axis(&xit, 2, 1, 0, h, &bh, &xr)?;
                    let xz = format!("{tp}_xz");
                    self.slice_axis(&xit, 2, 1, h, h, &bh, &xz)?;
                    let xn = format!("{tp}_xn");
                    self.slice_axis(&xit, 2, 1, 2 * h, h, &bh, &xn)?;
                    let hr = format!("{tp}_hr");
                    self.slice_axis(&hi, 2, 1, 0, h, &bh, &hr)?;
                    let hz = format!("{tp}_hz");
                    self.slice_axis(&hi, 2, 1, h, h, &bh, &hz)?;
                    let hn = format!("{tp}_hn");
                    self.slice_axis(&hi, 2, 1, 2 * h, h, &bh, &hn)?;

                    let rsum = format!("{tp}_rs");
                    self.emit(
                        "add",
                        &rsum,
                        &bh,
                        vec![("x", bind_name(&xr)), ("y", bind_name(&hr))],
                    )?;
                    let rg = format!("{tp}_rg");
                    self.emit("sigmoid", &rg, &bh, vec![("x", bind_name(&rsum))])?;
                    let zsum = format!("{tp}_zs");
                    self.emit(
                        "add",
                        &zsum,
                        &bh,
                        vec![("x", bind_name(&xz)), ("y", bind_name(&hz))],
                    )?;
                    let zg = format!("{tp}_zg");
                    self.emit("sigmoid", &zg, &bh, vec![("x", bind_name(&zsum))])?;
                    let rhn = format!("{tp}_rhn");
                    self.emit(
                        "mul",
                        &rhn,
                        &bh,
                        vec![("x", bind_name(&rg)), ("y", bind_name(&hn))],
                    )?;
                    let nsum = format!("{tp}_ns");
                    self.emit(
                        "add",
                        &nsum,
                        &bh,
                        vec![("x", bind_name(&xn)), ("y", bind_name(&rhn))],
                    )?;
                    let ng = format!("{tp}_ng");
                    self.emit("tanh", &ng, &bh, vec![("x", bind_name(&nsum))])?;
                    // h' = n + z⊙(h − n).
                    let hmn = format!("{tp}_hmn");
                    self.emit(
                        "sub",
                        &hmn,
                        &bh,
                        vec![("x", bind_name(&hst)), ("y", bind_name(&ng))],
                    )?;
                    let zhmn = format!("{tp}_zhmn");
                    self.emit(
                        "mul",
                        &zhmn,
                        &bh,
                        vec![("x", bind_name(&zg)), ("y", bind_name(&hmn))],
                    )?;
                    let hnew = format!("{tp}_h");
                    self.emit(
                        "add",
                        &hnew,
                        &bh,
                        vec![("x", bind_name(&ng)), ("y", bind_name(&zhmn))],
                    )?;

                    let hb = format!("{tp}_hb");
                    self.reshape_to(
                        &hnew,
                        &[b as i64, 1, h as i64],
                        &Shape::new(&[b, 1, h], self.opts.float_dtype),
                        &hb,
                    )?;
                    steps[t] = hb;
                    hst = hnew;
                }

                let dir_out = format!("{p}_out");
                self.emit(
                    "concat",
                    &dir_out,
                    &Shape::new(&[b, s, h], self.opts.float_dtype),
                    vec![
                        ("values", bind_names(&steps)),
                        ("axis", bind_value(scalar_i32(1))),
                        ("interleave", bind_value(scalar_bool(false))),
                    ],
                )?;
                dir_outs.push(dir_out);
            }

            let layer_out = if dirs == 1 {
                dir_outs.pop().expect("one direction")
            } else {
                let lo = format!("{out_name}_l{l}out");
                self.emit(
                    "concat",
                    &lo,
                    &Shape::new(&[b, s, out_width], self.opts.float_dtype),
                    vec![
                        ("values", bind_names(&dir_outs)),
                        ("axis", bind_value(scalar_i32(2))),
                        ("interleave", bind_value(scalar_bool(false))),
                    ],
                )?;
                lo
            };
            layer_in = layer_out;
            in_l = out_width;
            wih_cursor += dirs * wih_block;
        }

        self.reshape_to(
            &layer_in,
            &[b as i64, s as i64, (dirs * h) as i64],
            &Shape::new(&[b, s, dirs * h], self.opts.float_dtype),
            out_name,
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Elman RNN, `carry = false`: `h' = act(x·w_ihᵀ + h·w_hhᵀ + b)`,
    /// `act = relu` when `relu` else `tanh`. Single merged bias; `h` gate rows.
    /// Output `[b, s, dirs*h]`.
    pub(crate) fn lower_rnn(
        &mut self,
        id: NodeId,
        hidden: usize,
        num_layers: usize,
        bidirectional: bool,
        relu: bool,
        out_name: &str,
    ) -> Result<()> {
        let inputs: Vec<NodeId> = self.graph.node(id).inputs.clone();
        let x_sh = self.graph.shape(inputs[0]).clone();
        let b = dim_static(&x_sh, 0)?;
        let s = dim_static(&x_sh, 1)?;
        let input_size = dim_static(&x_sh, 2)?;
        let h = hidden;
        let dirs = if bidirectional { 2 } else { 1 };
        let act = if relu { "relu" } else { "tanh" };

        let wih_total: usize = (0..num_layers)
            .map(|l| dirs * h * if l == 0 { input_size } else { dirs * h })
            .sum();
        let whh_total = num_layers * dirs * h * h;
        let b_total = num_layers * dirs * h;
        let wih = self.flatten_1d(inputs[1], wih_total, &format!("{out_name}_wihf"))?;
        let whh = self.flatten_1d(inputs[2], whh_total, &format!("{out_name}_whhf"))?;
        let bias = self.flatten_1d(inputs[3], b_total, &format!("{out_name}_biasf"))?;

        let bh = Shape::new(&[b, h], self.opts.float_dtype);

        let mut layer_in = self.val(inputs[0]);
        let mut in_l = input_size;
        let mut wih_cursor = 0usize;

        for l in 0..num_layers {
            let out_width = dirs * h;
            let wih_block = h * in_l;
            let mut dir_outs: Vec<String> = Vec::with_capacity(dirs);

            for d in 0..dirs {
                let ld = l * dirs + d;
                let p = format!("{out_name}_l{l}d{d}");

                let wih_b = format!("{p}_wih");
                self.slice_1d_to(
                    &wih,
                    wih_cursor + d * wih_block,
                    wih_block,
                    &[h as i64, in_l as i64],
                    &Shape::new(&[h, in_l], self.opts.float_dtype),
                    &wih_b,
                )?;
                let whh_b = format!("{p}_whh");
                self.slice_1d_to(
                    &whh,
                    ld * h * h,
                    h * h,
                    &[h as i64, h as i64],
                    &Shape::new(&[h, h], self.opts.float_dtype),
                    &whh_b,
                )?;
                let bias_b = format!("{p}_bias");
                self.slice_1d_to(
                    &bias,
                    ld * h,
                    h,
                    &[1, h as i64],
                    &Shape::new(&[1, h], self.opts.float_dtype),
                    &bias_b,
                )?;

                // xi = x @ w_ihᵀ + b  → [b, s, h].
                let li2 = format!("{p}_li2");
                self.reshape_to(
                    &layer_in,
                    &[(b * s) as i64, in_l as i64],
                    &Shape::new(&[b * s, in_l], self.opts.float_dtype),
                    &li2,
                )?;
                let xi2 = format!("{p}_xi2");
                self.matmul_op(
                    &xi2,
                    &li2,
                    &wih_b,
                    false,
                    true,
                    &Shape::new(&[b * s, h], self.opts.float_dtype),
                )?;
                let xi2b = format!("{p}_xi2b");
                self.emit(
                    "add",
                    &xi2b,
                    &Shape::new(&[b * s, h], self.opts.float_dtype),
                    vec![("x", bind_name(&xi2)), ("y", bind_name(&bias_b))],
                )?;
                let xi = format!("{p}_xi");
                self.reshape_to(
                    &xi2b,
                    &[b as i64, s as i64, h as i64],
                    &Shape::new(&[b, s, h], self.opts.float_dtype),
                    &xi,
                )?;

                let mut hst = format!("{p}_h0");
                self.operations
                    .push(make_const(&mut self.blob, &hst, &bh, &vec![0.0f32; b * h])?);

                let mut steps: Vec<String> = vec![String::new(); s];
                for step in 0..s {
                    let t = if d == 0 { step } else { s - 1 - step };
                    let tp = format!("{p}_t{t}");

                    let xit3 = format!("{tp}_xit3");
                    self.slice_axis(
                        &xi,
                        3,
                        1,
                        t,
                        1,
                        &Shape::new(&[b, 1, h], self.opts.float_dtype),
                        &xit3,
                    )?;
                    let xit = format!("{tp}_xit");
                    self.reshape_to(&xit3, &[b as i64, h as i64], &bh, &xit)?;
                    let hh = format!("{tp}_hh");
                    self.matmul_op(&hh, &hst, &whh_b, false, true, &bh)?;
                    let acc = format!("{tp}_acc");
                    self.emit(
                        "add",
                        &acc,
                        &bh,
                        vec![("x", bind_name(&xit)), ("y", bind_name(&hh))],
                    )?;
                    let hnew = format!("{tp}_h");
                    self.emit(act, &hnew, &bh, vec![("x", bind_name(&acc))])?;

                    let hb = format!("{tp}_hb");
                    self.reshape_to(
                        &hnew,
                        &[b as i64, 1, h as i64],
                        &Shape::new(&[b, 1, h], self.opts.float_dtype),
                        &hb,
                    )?;
                    steps[t] = hb;
                    hst = hnew;
                }

                let dir_out = format!("{p}_out");
                self.emit(
                    "concat",
                    &dir_out,
                    &Shape::new(&[b, s, h], self.opts.float_dtype),
                    vec![
                        ("values", bind_names(&steps)),
                        ("axis", bind_value(scalar_i32(1))),
                        ("interleave", bind_value(scalar_bool(false))),
                    ],
                )?;
                dir_outs.push(dir_out);
            }

            let layer_out = if dirs == 1 {
                dir_outs.pop().expect("one direction")
            } else {
                let lo = format!("{out_name}_l{l}out");
                self.emit(
                    "concat",
                    &lo,
                    &Shape::new(&[b, s, out_width], self.opts.float_dtype),
                    vec![
                        ("values", bind_names(&dir_outs)),
                        ("axis", bind_value(scalar_i32(2))),
                        ("interleave", bind_value(scalar_bool(false))),
                    ],
                )?;
                lo
            };
            layer_in = layer_out;
            in_l = out_width;
            wih_cursor += dirs * wih_block;
        }

        self.reshape_to(
            &layer_in,
            &[b as i64, s as i64, (dirs * h) as i64],
            &Shape::new(&[b, s, dirs * h], self.opts.float_dtype),
            out_name,
        )?;
        self.names.insert(id.0, out_name.to_string());
        Ok(())
    }

    /// Reshape a value to rank-1 `[total]`, returning the new value's name.
    fn flatten_1d(&mut self, id: NodeId, total: usize, dst: &str) -> Result<String> {
        let src = self.val(id);
        self.reshape_to(
            &src,
            &[total as i64],
            &Shape::new(&[total], self.opts.float_dtype),
            dst,
        )?;
        Ok(dst.to_string())
    }

    /// Slice `[off, off+len)` from a rank-1 value, then reshape to `dims`.
    fn slice_1d_to(
        &mut self,
        src_1d: &str,
        off: usize,
        len: usize,
        dims: &[i64],
        out_shape: &Shape,
        dst: &str,
    ) -> Result<()> {
        let tmp = format!("{dst}_s");
        self.slice_axis(
            src_1d,
            1,
            0,
            off,
            len,
            &Shape::new(&[len], self.opts.float_dtype),
            &tmp,
        )?;
        self.reshape_to(&tmp, dims, out_shape, dst)
    }
}
