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

//! `rnn` — recurrent ops. Maps ONNX float `LSTM` to the native, all-backend
//! [`rlx_ir::Op::Lstm`] (CPU / Metal / MLX / wgpu / CUDA / ROCm / CoreML /
//! Vulkan). The quantized `DynamicQuantizeLSTM` path lives in `cast_quant`.
//!
//! Two conventions differ between ONNX and rlx and are reconciled here at import
//! time on the constant weight initializers:
//!
//! * **Gate order.** ONNX packs the `4·hidden` gate rows as `[i, o, f, c]`;
//!   rlx's LSTM kernel expects `[i, f, g, o]` (PyTorch order, `g` == ONNX `c`).
//! * **Bias.** ONNX carries `[Wb | Rb]` (`8·hidden`); rlx wants a single summed
//!   `Wb + Rb` (`4·hidden`) per (layer, direction).
//!
//! Layouts: ONNX `X` is `[seq, batch, input]` (`layout=0`) and `Y` is
//! `[seq, num_dir, batch, hidden]`; the native op works in `[batch, seq, input]`
//! → `[batch, seq, D·hidden]`, so we transpose in and reshape/transpose out.

#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail};
use rlx_ir::dynamic::sym;
use rlx_ir::hir::{HirMut, HirNodeId, HirOp};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
use rlx_ir::{DType, Dim, HirGraphExt, HirModule, Op, Shape};

use crate::bundle::BundleNode;

use super::*;

/// rlx gate `g` reads ONNX gate `GATE_SRC[g]`: rlx `[i,f,g,o]` ← ONNX `[i,o,f,c]`.
const GATE_SRC: [usize; 4] = [0, 2, 3, 1];

/// Reorder the `4·hidden` gate rows of a `[num_dir, 4h, cols]` weight from ONNX
/// `[i,o,f,c]` to rlx `[i,f,g,o]`, keeping the `[num_dir, 4h, cols]` packing that
/// the kernel reads per (layer, direction).
fn reorder_gate_rows(src: &[f32], dirs: usize, hidden: usize, cols: usize) -> Vec<f32> {
    let four_h = 4 * hidden;
    let mut out = vec![0f32; dirs * four_h * cols];
    for d in 0..dirs {
        for g in 0..4 {
            let sg = GATE_SRC[g];
            for row in 0..hidden {
                let dst_row = (d * four_h) + g * hidden + row;
                let src_row = (d * four_h) + sg * hidden + row;
                out[dst_row * cols..dst_row * cols + cols]
                    .copy_from_slice(&src[src_row * cols..src_row * cols + cols]);
            }
        }
    }
    out
}

/// Sum ONNX `B = [Wb | Rb]` (`[num_dir, 8h]`) into rlx's `[num_dir, 4h]`
/// summed bias, reordered to `[i,f,g,o]`. `None` → zeros.
fn fold_bias(src: Option<&[f32]>, dirs: usize, hidden: usize) -> Vec<f32> {
    let four_h = 4 * hidden;
    let eight_h = 8 * hidden;
    let mut out = vec![0f32; dirs * four_h];
    let Some(b) = src else { return out };
    for d in 0..dirs {
        for g in 0..4 {
            let sg = GATE_SRC[g];
            for row in 0..hidden {
                let wb = b[d * eight_h + sg * hidden + row];
                let rb = b[d * eight_h + four_h + sg * hidden + row];
                out[(d * four_h) + g * hidden + row] = wb + rb;
            }
        }
    }
    out
}

pub(super) fn lower_lstm(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let direction = node
        .attrs
        .get("direction")
        .and_then(|v| v.as_str())
        .unwrap_or("forward");
    if direction == "reverse" {
        // rlx Op::Lstm has forward/bidirectional only; reverse-only is unused by
        // the target StyleTTS2 models. Leave to the unsupported path.
        ctx.unsupported("LSTM(reverse)");
        return Ok(false);
    }
    let bidirectional = direction == "bidirectional";
    let dirs = if bidirectional { 2 } else { 1 };
    if std::env::var_os("RLX_LSTM_DEBUG").is_some() {
        eprintln!(
            "[lstm] node={} direction={direction:?} bidir={bidirectional} attrs_keys={:?}",
            node.name,
            node.attrs.keys().collect::<Vec<_>>()
        );
    }

    // Weights must be constant initializers (W, R). B is optional.
    let w_name = node.inputs.get(1).map(String::as_str).unwrap_or("");
    let r_name = node.inputs.get(2).map(String::as_str).unwrap_or("");
    let (Some(w_shape), Some(w_data)) = (
        ctx.init_shapes.get(w_name).cloned(),
        ctx.params.get(w_name).cloned(),
    ) else {
        ctx.unsupported("LSTM(dynamic W)");
        return Ok(false);
    };
    let (Some(r_shape), Some(r_data)) = (
        ctx.init_shapes.get(r_name).cloned(),
        ctx.params.get(r_name).cloned(),
    ) else {
        ctx.unsupported("LSTM(dynamic R)");
        return Ok(false);
    };
    if w_shape.len() != 3 || r_shape.len() != 3 {
        ctx.unsupported("LSTM(weight rank)");
        return Ok(false);
    }
    let input_size = w_shape[2];
    let hidden = node
        .attrs
        .get("hidden_size")
        .and_then(|v| v.as_i64())
        .map(|h| h as usize)
        .unwrap_or(r_shape[2]);
    let four_h = 4 * hidden;
    if w_shape[1] != four_h || r_shape[1] != four_h || r_shape[2] != hidden {
        ctx.unsupported("LSTM(shape mismatch)");
        return Ok(false);
    }

    // Repack constant weights to the rlx layout / gate order.
    let wih = reorder_gate_rows(&w_data, dirs, hidden, input_size);
    let whh = reorder_gate_rows(&r_data, dirs, hidden, hidden);
    let b_data = node
        .inputs
        .get(3)
        .filter(|n| !n.is_empty())
        .and_then(|n| ctx.params.get(n).cloned());
    let bias = fold_bias(b_data.as_deref(), dirs, hidden);

    let wih_id = insert_param(
        m,
        ctx,
        &format!("__lstm_wih__/{}", node.name),
        wih,
        &[dirs * four_h, input_size],
    );
    let whh_id = insert_param(
        m,
        ctx,
        &format!("__lstm_whh__/{}", node.name),
        whh,
        &[dirs * four_h, hidden],
    );
    let bias_id = insert_param(
        m,
        ctx,
        &format!("__lstm_bias__/{}", node.name),
        bias,
        &[dirs * four_h],
    );

    // X: ONNX [seq, batch, input] (layout 0) → native [batch, seq, input].
    let layout = node
        .attrs
        .get("layout")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let x = ctx.tensor(&node.inputs[0])?;
    let x_bsi = if layout == 1 {
        x
    } else {
        let xs = m.shape(x).clone();
        let out = permuted_shape(&xs, &[1, 0, 2]);
        m.add_node(
            Op::Transpose {
                perm: vec![1, 0, 2],
            },
            vec![x],
            out,
        )
    };
    let xs = m.shape(x_bsi).clone();
    let batch = xs.dim(0).unwrap_static().max(1);
    let seq = xs.dim(1).unwrap_static().max(1);

    let y_bshd = m.add_node(
        Op::Lstm {
            hidden_size: hidden,
            num_layers: 1,
            bidirectional,
            carry: false,
        },
        vec![x_bsi, wih_id, whh_id, bias_id],
        Shape::new(&[batch, seq, dirs * hidden], DType::F32),
    );

    // Y (output 0): native [batch, seq, D·hidden] → ONNX [seq, num_dir, batch, hidden].
    if let Some(y_name) = node.outputs.first().filter(|n| !n.is_empty()) {
        let y4 = m.add_node(
            Op::Reshape {
                new_shape: vec![batch as i64, seq as i64, dirs as i64, hidden as i64],
            },
            vec![y_bshd],
            Shape::new(&[batch, seq, dirs, hidden], DType::F32),
        );
        let y_out = m.add_node(
            Op::Transpose {
                perm: vec![1, 2, 0, 3],
            },
            vec![y4],
            Shape::new(&[seq, dirs, batch, hidden], DType::F32),
        );
        ctx.env.insert(y_name.clone(), y_out);
    }

    // Y_h / Y_c (outputs 1, 2): zero stubs, matching the DynamicQuantizeLSTM path.
    // The target StyleTTS2 models consume the full-sequence Y only.
    for out_name in node.outputs.iter().skip(1).filter(|n| !n.is_empty()) {
        let shape = Shape::new(&[dirs, batch, hidden], DType::F32);
        let key = format!("__lstm_state__/{out_name}");
        let n = dirs * batch * hidden;
        let pid = m.param(&key, shape);
        ctx.params.insert(key, vec![0.0; n]);
        ctx.env.insert(out_name.clone(), pid);
    }

    Ok(true)
}

/// Insert a constant f32 param and return its node id.
fn insert_param(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    key: &str,
    data: Vec<f32>,
    dims: &[usize],
) -> HirNodeId {
    ctx.params.insert(key.to_string(), data);
    let id = m.param(key, Shape::new(dims, DType::F32));
    ctx.env.insert(key.to_string(), id);
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_reorder_maps_iofc_to_ifgo() {
        // dirs=1, hidden=1, cols=1 → 4 rows, one per gate. Values tag the ONNX gate.
        // ONNX order [i,o,f,c] = [10, 20, 30, 40]; rlx wants [i,f,g,o] = [10,30,40,20].
        let onnx = vec![10.0, 20.0, 30.0, 40.0];
        let rlx = reorder_gate_rows(&onnx, 1, 1, 1);
        assert_eq!(rlx, vec![10.0, 30.0, 40.0, 20.0]);
    }

    #[test]
    fn fold_bias_sums_and_reorders() {
        // hidden=1: Wb=[i,o,f,c]=[1,2,3,4], Rb=[i,o,f,c]=[5,6,7,8].
        // rlx [i,f,g,o] summed = [1+5, 3+7, 4+8, 2+6] = [6, 10, 12, 8].
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = fold_bias(Some(&b), 1, 1);
        assert_eq!(out, vec![6.0, 10.0, 12.0, 8.0]);
    }

    #[test]
    fn fold_bias_none_is_zero() {
        assert_eq!(fold_bias(None, 2, 3), vec![0.0; 2 * 4 * 3]);
    }

    #[test]
    fn reorder_bidirectional_keeps_dirs_separate() {
        // dirs=2, hidden=1, cols=1: dir0 [10,20,30,40], dir1 [50,60,70,80].
        let onnx = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let rlx = reorder_gate_rows(&onnx, 2, 1, 1);
        // each dir reordered [i,f,g,o]: dir0 [10,30,40,20], dir1 [50,70,80,60].
        assert_eq!(rlx, vec![10.0, 30.0, 40.0, 20.0, 50.0, 70.0, 80.0, 60.0]);
    }
}
