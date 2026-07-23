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

//! On-device `onnx.DynamicQuantizeLSTM` for CUDA f32-uniform arenas.
//!
//! Dequantizes int8 W/R once (cached), permutes ONNX gate order `(i,o,f,c)` into
//! the PyTorch/Op::Lstm order `(i,f,g,o)` expected by [`crate::lstm_gpu`], then
//! runs the native `lstm_dir` kernel against a staging buffer so activations
//! never round-trip through the host.

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use rlx_ir::Shape;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const OP_NAME: &str = "onnx.DynamicQuantizeLSTM";

/// Optional Kitten (or other) hook: `(raw_seq, input_size) -> active_seq`.
pub type SeqResolver = fn(usize, usize) -> usize;

static SEQ_RESOLVER: OnceLock<SeqResolver> = OnceLock::new();

/// Install a process-wide active-seq resolver (idempotent; first wins).
pub fn set_seq_resolver(f: SeqResolver) {
    let _ = SEQ_RESOLVER.set(f);
}

fn resolve_seq(raw_seq: usize, input_size: usize) -> usize {
    SEQ_RESOLVER
        .get()
        .map(|f| f(raw_seq, input_size).clamp(1, raw_seq.max(1)))
        .unwrap_or(raw_seq.max(1))
}

struct WeightCache {
    /// Device workspace: `[X_cap | Y_cap | W | R | bias]`. X/Y caps grow with
    /// `max_seq`; weights stay put so we never re-upload or re-D2D them.
    workspace: CudaSlice<f32>,
    /// Optional cuDNN packed weight space (built once from gate-major W/R/bias).
    cudnn_weights: Option<CudaSlice<u8>>,
    max_seq: usize,
    w_elems: usize,
    r_elems: usize,
    bias_elems: usize,
    hidden: usize,
    input_size: usize,
    dirs: usize,
    batch: usize,
}

fn xy_caps(max_seq: usize, batch: usize, input_size: usize, hidden: usize, dirs: usize) -> (usize, usize, usize) {
    let x = max_seq * batch * input_size;
    let y = max_seq * dirs * batch * hidden;
    (x, y, x + y)
}

fn weight_caches() -> &'static Mutex<HashMap<(u32, u32), WeightCache>> {
    static C: OnceLock<Mutex<HashMap<(u32, u32), WeightCache>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn parse_attrs(attrs: &[u8]) -> (usize, bool) {
    if attrs.len() >= 8 {
        let h = u32::from_le_bytes(attrs[0..4].try_into().unwrap()) as usize;
        return (h.max(1), attrs[4] != 0);
    }
    (256, true)
}

fn shape_static(sh: &Shape) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(sh.rank());
    for d in sh.dims() {
        match d {
            rlx_ir::Dim::Static(n) => out.push(*n),
            rlx_ir::Dim::Dynamic(_) => return None,
        }
    }
    Some(out)
}

/// ONNX input-major `[in, 4h]` → gate-major `[4h, in]`, then permute gates
/// `(i,o,f,c)` → `(i,f,g,o)`.
fn onnx_w_to_lstm_gate_major(src: &[f32], input_size: usize, h4: usize) -> Vec<f32> {
    // src is [input_size, 4h] row-major → tmp [4h, input_size]
    let mut tmp = vec![0f32; input_size * h4];
    for r in 0..input_size {
        for c in 0..h4 {
            tmp[c * input_size + r] = src[r * h4 + c];
        }
    }
    permute_gates_iofc_to_ifgo(&tmp, input_size)
}

fn onnx_r_to_lstm_gate_major(src: &[f32], hidden: usize, h4: usize) -> Vec<f32> {
    let mut tmp = vec![0f32; hidden * h4];
    for r in 0..hidden {
        for c in 0..h4 {
            tmp[c * hidden + r] = src[r * h4 + c];
        }
    }
    permute_gates_iofc_to_ifgo(&tmp, hidden)
}



/// Gate-major `[4h, k]` with ONNX order i|o|f|c → PyTorch i|f|g|o.
fn permute_gates_iofc_to_ifgo(src: &[f32], k: usize) -> Vec<f32> {
    let h = src.len() / (4 * k);
    let mut out = vec![0f32; src.len()];
    for row in 0..h {
        // i
        out[row * k..(row + 1) * k].copy_from_slice(&src[row * k..(row + 1) * k]);
        // f ← onnx f (block 2)
        let s = (2 * h + row) * k;
        let d = (1 * h + row) * k;
        out[d..d + k].copy_from_slice(&src[s..s + k]);
        // g ← onnx c (block 3)
        let s = (3 * h + row) * k;
        let d = (2 * h + row) * k;
        out[d..d + k].copy_from_slice(&src[s..s + k]);
        // o ← onnx o (block 1)
        let s = (1 * h + row) * k;
        let d = (3 * h + row) * k;
        out[d..d + k].copy_from_slice(&src[s..s + k]);
    }
    out
}

fn permute_bias_iofc_to_ifgo(wb: &[f32]) -> Vec<f32> {
    let h = wb.len() / 4;
    let mut out = vec![0f32; wb.len()];
    out[..h].copy_from_slice(&wb[..h]); // i
    out[h..2 * h].copy_from_slice(&wb[2 * h..3 * h]); // f
    out[2 * h..3 * h].copy_from_slice(&wb[3 * h..4 * h]); // g
    out[3 * h..4 * h].copy_from_slice(&wb[h..2 * h]); // o
    out
}

fn dequant_i8(data: &[i8], scale: f32, zp: i32) -> Vec<f32> {
    data.iter()
        .map(|&q| (q as i32 - zp) as f32 * scale)
        .collect()
}

fn dtoh_f32(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>, off: u32, n: usize) -> Vec<f32> {
    let mut host = vec![0f32; n];
    if n == 0 {
        return host;
    }
    stream
        .memcpy_dtoh(
            &buf.slice(off as usize..(off as usize + n)),
            &mut host,
        )
        .expect("dyn_quant_lstm: dtoh f32");
    host
}

fn dtoh_i8_packed(stream: &Arc<CudaStream>, buf: &CudaSlice<f32>, off: u32, n: usize) -> Vec<i8> {
    // i8 is byte-packed in the f32 arena (4 elems per f32 slot).
    let slots = n.div_ceil(4);
    let mut raw = vec![0u8; slots * 4];
    if n > 0 {
        stream
            .memcpy_dtoh(
                &buf.slice(off as usize..(off as usize + slots)),
                bytemuck::cast_slice_mut(&mut raw),
            )
            .expect("dyn_quant_lstm: dtoh i8");
    }
    raw.truncate(n);
    raw.into_iter().map(|b| b as i8).collect()
}

fn read_zp_f32_or_i8(
    stream: &Arc<CudaStream>,
    buf: &CudaSlice<f32>,
    off: u32,
    sh: &Shape,
) -> Vec<i32> {
    let n = sh.num_elements().unwrap_or(0);
    match sh.dtype() {
        rlx_ir::DType::I8 | rlx_ir::DType::U8 => dtoh_i8_packed(stream, buf, off, n)
            .into_iter()
            .map(|x| x as i32)
            .collect(),
        rlx_ir::DType::I32 => {
            let f = dtoh_f32(stream, buf, off, n);
            // i32 may be stored as f32 lanes on the uniform arena.
            f.into_iter().map(|x| x as i32).collect()
        }
        _ => dtoh_f32(stream, buf, off, n)
            .into_iter()
            .map(|x| x.round() as i32)
            .collect(),
    }
}

/// Try to run DynamicQuantizeLSTM on GPU. Returns `false` to fall back to host.
#[allow(clippy::too_many_arguments)]
pub fn try_run(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    arena: &mut CudaSlice<f32>,
    name: &str,
    in_specs: &[(u32, Shape)],
    out_off: u32,
    out_shape: &Shape,
    attrs: &[u8],
) -> bool {
    if name != OP_NAME {
        return false;
    }
    if rlx_ir::env::flag("RLX_CUDA_DYN_LSTM_HOST") {
        return false;
    }
    // Need at least X, W, R, B + 4 quant params.
    if in_specs.len() < 8 {
        return false;
    }
    let (hidden, bidirectional) = parse_attrs(attrs);
    let dirs = if bidirectional { 2 } else { 1 };
    let h4 = hidden * 4;

    let x_dims = match shape_static(&in_specs[0].1) {
        Some(d) if d.len() == 3 => d,
        _ => return false,
    };
    let (raw_seq, batch, input_size) = (x_dims[0], x_dims[1], x_dims[2]);
    if batch != 1 || input_size == 0 || raw_seq == 0 {
        // GPU path is validated for batch=1 (Kitten); other shapes stay on host.
        return false;
    }
    let seq = resolve_seq(raw_seq, input_size);
    let n_in = in_specs.len();
    let (w_off, w_sh) = (&in_specs[1].0, &in_specs[1].1);
    let (r_off, r_sh) = (&in_specs[2].0, &in_specs[2].1);
    let (b_off, _) = (&in_specs[3].0, &in_specs[3].1);
    let (ws_off, ws_sh) = (&in_specs[n_in - 4].0, &in_specs[n_in - 4].1);
    let (wz_off, wz_sh) = (&in_specs[n_in - 3].0, &in_specs[n_in - 3].1);
    let (rs_off, rs_sh) = (&in_specs[n_in - 2].0, &in_specs[n_in - 2].1);
    let (rz_off, rz_sh) = (&in_specs[n_in - 1].0, &in_specs[n_in - 1].1);

    if !matches!(w_sh.dtype(), rlx_ir::DType::I8) {
        return false;
    }

    let cache_key = (*w_off, *r_off);
    let mut caches = weight_caches().lock().unwrap();
    if !caches.contains_key(&cache_key) {
        let w_i8 = dtoh_i8_packed(stream, arena, *w_off, w_sh.num_elements().unwrap_or(0));
        let r_i8 = dtoh_i8_packed(stream, arena, *r_off, r_sh.num_elements().unwrap_or(0));
        let b = dtoh_f32(stream, arena, *b_off, dirs * 8 * hidden);
        let w_scale = dtoh_f32(stream, arena, *ws_off, ws_sh.num_elements().unwrap_or(0).max(1));
        let r_scale = dtoh_f32(stream, arena, *rs_off, rs_sh.num_elements().unwrap_or(0).max(1));
        let w_zp = read_zp_f32_or_i8(stream, arena, *wz_off, wz_sh);
        let r_zp = read_zp_f32_or_i8(stream, arena, *rz_off, rz_sh);

        let w_stride = input_size * h4;
        let r_stride = hidden * h4;
        if w_i8.len() < dirs * w_stride || r_i8.len() < dirs * r_stride || b.len() < dirs * 8 * hidden
        {
            return false;
        }

        let mut w_all = Vec::with_capacity(dirs * w_stride);
        let mut r_all = Vec::with_capacity(dirs * r_stride);
        let mut bias_all = Vec::with_capacity(dirs * h4);
        for dir in 0..dirs {
            let ws = w_scale.get(dir).copied().unwrap_or(w_scale[0]);
            let wz = w_zp.get(dir).copied().unwrap_or(w_zp.first().copied().unwrap_or(0));
            let rs = r_scale.get(dir).copied().unwrap_or(r_scale[0]);
            let rz = r_zp.get(dir).copied().unwrap_or(r_zp.first().copied().unwrap_or(0));
            let w_f = dequant_i8(&w_i8[dir * w_stride..(dir + 1) * w_stride], ws, wz);
            let r_f = dequant_i8(&r_i8[dir * r_stride..(dir + 1) * r_stride], rs, rz);
            w_all.extend(onnx_w_to_lstm_gate_major(&w_f, input_size, h4));
            r_all.extend(onnx_r_to_lstm_gate_major(&r_f, hidden, h4));
            // Match Kitten CPU: use Wb only (first 4h of the 8h ONNX bias).
            let wb = &b[dir * 8 * hidden..dir * 8 * hidden + h4];
            bias_all.extend(permute_bias_iofc_to_ifgo(wb));
        }

        let max_seq = seq;
        let (x_cap, y_cap, w_base) = xy_caps(max_seq, batch, input_size, hidden, dirs);
        let mut packed = vec![0f32; w_base + w_all.len() + r_all.len() + bias_all.len()];
        packed[w_base..w_base + w_all.len()].copy_from_slice(&w_all);
        packed[w_base + w_all.len()..w_base + w_all.len() + r_all.len()]
            .copy_from_slice(&r_all);
        packed[w_base + w_all.len() + r_all.len()..]
            .copy_from_slice(&bias_all);
        let _ = (x_cap, y_cap);
        let cudnn_weights = crate::lstm_cudnn::pack_weight_space(
            stream,
            input_size,
            hidden,
            bidirectional,
            &w_all,
            &r_all,
            &bias_all,
        );
        let mut workspace = stream
            .alloc_zeros::<f32>(packed.len().max(1))
            .expect("dyn_quant_lstm: workspace alloc");
        stream
            .memcpy_htod(&packed, &mut workspace)
            .expect("dyn_quant_lstm: workspace htod");
        caches.insert(
            cache_key,
            WeightCache {
                workspace,
                cudnn_weights,
                max_seq,
                w_elems: w_all.len(),
                r_elems: r_all.len(),
                bias_elems: bias_all.len(),
                hidden,
                input_size,
                dirs,
                batch,
            },
        );
    }
    let weights = caches.get_mut(&cache_key).unwrap();
    if weights.hidden != hidden
        || weights.input_size != input_size
        || weights.dirs != dirs
        || weights.batch != batch
    {
        return false;
    }

    // Grow X/Y caps if this call needs a longer sequence.
    if seq > weights.max_seq {
        let (old_x, old_y, old_w_base) = xy_caps(
            weights.max_seq,
            batch,
            input_size,
            hidden,
            dirs,
        );
        let (_nx, _ny, new_w_base) = xy_caps(seq, batch, input_size, hidden, dirs);
        let wtot = weights.w_elems + weights.r_elems + weights.bias_elems;
        let new_len = new_w_base + wtot;
        let mut new_ws = stream
            .alloc_zeros::<f32>(new_len.max(1))
            .expect("dyn_quant_lstm: workspace grow");
        {
            let src = weights
                .workspace
                .slice(old_w_base..old_w_base + wtot);
            let mut dst = new_ws.slice_mut(new_w_base..new_w_base + wtot);
            stream
                .memcpy_dtod(&src, &mut dst)
                .expect("dyn_quant_lstm: weight relocate");
        }
        let _ = (old_x, old_y);
        weights.workspace = new_ws;
        weights.max_seq = seq;
    }

    let (x_cap, y_cap, w_base) = xy_caps(
        weights.max_seq,
        batch,
        input_size,
        hidden,
        dirs,
    );
    let x_elems = seq * batch * input_size;
    let y_elems = seq * dirs * batch * hidden;
    debug_assert!(x_elems <= x_cap && y_elems <= y_cap);

    // D2D active X into workspace front (ONNX [seq,1,in] ≡ Op [1,seq,in] for batch=1).
    {
        let src = arena.slice(in_specs[0].0 as usize..(in_specs[0].0 as usize + x_elems));
        let mut dst = weights.workspace.slice_mut(0..x_elems);
        stream
            .memcpy_dtod(&src, &mut dst)
            .expect("dyn_quant_lstm: X dtod");
    }

    let x_byte = 0usize;
    let y_byte = x_cap * 4; // Y region starts after X_cap (not after live x_elems)
    let w_ih_byte = w_base * 4;
    let w_hh_byte = (w_base + weights.w_elems) * 4;
    let bias_byte = (w_base + weights.w_elems + weights.r_elems) * 4;

    // Prefer cuDNN LSTM when weights packed; fall back to native kernel.
    let used_cudnn = match weights.cudnn_weights.as_ref() {
        Some(ws) => crate::lstm_cudnn::forward_workspace(
            stream,
            ws,
            &mut weights.workspace,
            0,
            x_cap,
            batch,
            seq,
            input_size,
            hidden,
            bidirectional,
        ),
        None => false,
    };

    let ok = used_cudnn
        || crate::lstm_gpu::run_lstm(
            ctx,
            stream,
            &mut weights.workspace,
            x_byte,
            w_ih_byte,
            w_hh_byte,
            bias_byte,
            0,
            0,
            y_byte,
            batch,
            seq,
            input_size,
            hidden,
            1, // num_layers
            bidirectional,
            false, // carry
        );
    if !ok {
        return false;
    }

    // Copy Y back. Op::Lstm layout [1,seq,dirs*h] ≡ ONNX [seq,dirs,1,h] for batch=1.
    let out_n = out_shape.num_elements().unwrap_or(0).min(y_elems);
    if out_n > 0 {
        let src = weights.workspace.slice(x_cap..x_cap + out_n);
        let mut dst = arena.slice_mut(out_off as usize..(out_off as usize + out_n));
        stream
            .memcpy_dtod(&src, &mut dst)
            .expect("dyn_quant_lstm: Y dtod");
    }
    if rlx_ir::env::flag("RLX_CUDA_DYN_LSTM_TRACE") {
        eprintln!(
            "[dyn_quant_lstm] seq={seq}/{raw_seq} in={input_size} h={hidden} dirs={dirs} y={out_n}"
        );
    }
    true
}
