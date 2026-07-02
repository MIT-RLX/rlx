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

//! Fused multi-scale deformable attention (`Op::Custom("gdino.ms_deform_attn")`).
//!
//! Shared host compute for the CPU kernel and the GPU backends' host-delegate
//! kernels (Metal/MLX/CUDA/WGPU). Runs the whole DETR-style deformable module —
//! value/offset/weight projections, bilinear sampling, output projection — in
//! one pass, accumulating per query/head so it stays memory-bounded (no
//! `[nq·heads·levels·points·4, head_dim]` intermediate).
//!
//! Input order (all f32):
//!   0 query [nq, d]
//!   1 value_src [seq, d]
//!   2 reference_points [nq, n_levels, ref_dim]   (ref_dim 2 = centers, 4 = boxes)
//!   3 value_proj.weight [d, d]        4 value_proj.bias [d]
//!   5 sampling_offsets.weight [H, d]  6 sampling_offsets.bias [H]   (H = nh*nl*np*2)
//!   7 attention_weights.weight [A, d] 8 attention_weights.bias [A]  (A = nh*nl*np)
//!   9 output_proj.weight [d, d]      10 output_proj.bias [d]
//!
//! Attributes (LE u32): `[d, nh, np, ref_dim, nl, (h, w) * nl]`.

struct Attrs {
    d: usize,
    nh: usize,
    np: usize,
    ref_dim: usize,
    shapes: Vec<(usize, usize)>,
}

fn decode_attrs(bytes: &[u8]) -> Result<Attrs, String> {
    if bytes.len() < 5 * 4 {
        return Err("ms_deform_attn: attrs too short".into());
    }
    let rd = |i: usize| -> u32 {
        u32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ])
    };
    let d = rd(0) as usize;
    let nh = rd(1) as usize;
    let np = rd(2) as usize;
    let ref_dim = rd(3) as usize;
    let nl = rd(4) as usize;
    if bytes.len() < (5 + nl * 2) * 4 {
        return Err("ms_deform_attn: attrs truncated shapes".into());
    }
    let shapes = (0..nl)
        .map(|l| (rd(5 + l * 2) as usize, rd(5 + l * 2 + 1) as usize))
        .collect();
    Ok(Attrs {
        d,
        nh,
        np,
        ref_dim,
        shapes,
    })
}

/// `y[r,o] = sum_i x[r,i] * w[o,i] + b[o]` (PyTorch `[out, in]` weight).
///
/// The value/offset/attention/output projections dominate this kernel's cost
/// (each is `rows·d·d`-ish, with `rows` = the full multi-scale token count, e.g.
/// ~18k). A naive triple loop here made the GPU backends' host-delegate ~7×
/// slower than the CPU backend (which uses BLAS), so route through the same
/// `sgemm_bt` the rest of the codebase uses.
fn linear(x: &[f32], rows: usize, in_dim: usize, w: &[f32], out_dim: usize, b: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; rows * out_dim];
    crate::blas::sgemm_bt(x, w, &mut out, rows, in_dim, out_dim, 1.0);
    if !b.is_empty() {
        for r in 0..rows {
            let row = &mut out[r * out_dim..(r + 1) * out_dim];
            for (o, bv) in row.iter_mut().zip(b.iter()) {
                *o += *bv;
            }
        }
    }
    out
}

/// Run the fused kernel, writing `[nq, d]` into `out`.
pub fn execute(inputs: &[&[f32]], attrs: &[u8], out: &mut [f32]) -> Result<(), String> {
    if inputs.len() != 11 {
        return Err(format!(
            "ms_deform_attn: expected 11 inputs, got {}",
            inputs.len()
        ));
    }
    let a = decode_attrs(attrs)?;
    let (d, nh, np, ref_dim) = (a.d, a.nh, a.np, a.ref_dim);
    let nl = a.shapes.len();
    let hd = d / nh;
    let query = inputs[0];
    let value_src = inputs[1];
    let reference = inputs[2];
    let nq = query.len() / d;
    let seq = value_src.len() / d;

    // level start offsets into the flattened value sequence.
    let mut starts = vec![0usize; nl];
    {
        let mut acc = 0;
        for (l, (h, w)) in a.shapes.iter().enumerate() {
            starts[l] = acc;
            acc += h * w;
        }
    }

    let value = linear(value_src, seq, d, inputs[3], d, inputs[4]);
    let offsets = linear(query, nq, d, inputs[5], nh * nl * np * 2, inputs[6]);
    let mut attn = linear(query, nq, d, inputs[7], nh * nl * np, inputs[8]);
    // softmax over (nl*np) for each (nq, head): rows = nq*nh, cols = nl*np.
    softmax_rows(&mut attn, nq * nh, nl * np);

    let mut combined = vec![0f32; nq * d];
    for q in 0..nq {
        for m in 0..nh {
            let mut acc = vec![0f32; hd];
            for l in 0..nl {
                let (h, w) = a.shapes[l];
                let base = starts[l];
                for p in 0..np {
                    let off_base = (((q * nh + m) * nl + l) * np + p) * 2;
                    let off_x = offsets[off_base];
                    let off_y = offsets[off_base + 1];
                    let rb = (q * nl + l) * ref_dim;
                    let (loc_x, loc_y) = if ref_dim == 2 {
                        (
                            reference[rb] + off_x / w as f32,
                            reference[rb + 1] + off_y / h as f32,
                        )
                    } else {
                        (
                            reference[rb] + off_x / np as f32 * reference[rb + 2] * 0.5,
                            reference[rb + 1] + off_y / np as f32 * reference[rb + 3] * 0.5,
                        )
                    };
                    let aw = attn[(q * nh + m) * (nl * np) + l * np + p];
                    if aw == 0.0 {
                        continue;
                    }
                    sample(&value, d, base, h, w, m, hd, loc_x, loc_y, aw, &mut acc);
                }
            }
            for c in 0..hd {
                combined[q * d + m * hd + c] = acc[c];
            }
        }
    }
    let res = linear(&combined, nq, d, inputs[9], d, inputs[10]);
    if res.len() != out.len() {
        return Err(format!(
            "ms_deform_attn: out len {} != {}",
            out.len(),
            res.len()
        ));
    }
    out.copy_from_slice(&res);
    Ok(())
}

/// In-arena variant for backends that stage the whole f32 arena to the host
/// (CUDA/ROCm). Reads the 11 inputs from `arena` at the given `(f32_off, f32_len)`
/// pairs and writes the `[nq, d]` result to `arena[out_f32_off..]`.
pub fn execute_in_arena(
    arena: &mut [f32],
    in_offs: &[(usize, usize)],
    out_f32_off: usize,
    out_f32_len: usize,
    attrs: &[u8],
) -> Result<(), String> {
    let ins: Vec<Vec<f32>> = in_offs
        .iter()
        .map(|&(off, len)| arena[off..off + len].to_vec())
        .collect();
    let in_refs: Vec<&[f32]> = ins.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![0f32; out_f32_len];
    execute(&in_refs, attrs, &mut out)?;
    arena[out_f32_off..out_f32_off + out_f32_len].copy_from_slice(&out);
    Ok(())
}

/// Bilinear-sample head `m` of `value [seq, nh*hd]` at normalized `(loc_x, loc_y)`
/// (align_corners=false, zero padding) and accumulate `weight * sample`.
#[allow(clippy::too_many_arguments)]
fn sample(
    value: &[f32],
    d: usize,
    base: usize,
    h: usize,
    w: usize,
    m: usize,
    hd: usize,
    loc_x: f32,
    loc_y: f32,
    weight: f32,
    acc: &mut [f32],
) {
    let ix = ((2.0 * loc_x - 1.0 + 1.0) * w as f32 - 1.0) * 0.5;
    let iy = ((2.0 * loc_y - 1.0 + 1.0) * h as f32 - 1.0) * 0.5;
    let x0 = ix.floor() as isize;
    let y0 = iy.floor() as isize;
    let wx1 = ix - x0 as f32;
    let wy1 = iy - y0 as f32;
    let corners = [
        (y0, x0, (1.0 - wy1) * (1.0 - wx1)),
        (y0, x0 + 1, (1.0 - wy1) * wx1),
        (y0 + 1, x0, wy1 * (1.0 - wx1)),
        (y0 + 1, x0 + 1, wy1 * wx1),
    ];
    for (cy, cx, cw) in corners {
        if cy < 0 || cx < 0 || cy >= h as isize || cx >= w as isize {
            continue;
        }
        let row = base + cy as usize * w + cx as usize;
        let voff = row * d + m * hd;
        let cw = cw * weight;
        for c in 0..hd {
            acc[c] += cw * value[voff + c];
        }
    }
}

/// In-place numerically-stable row softmax.
fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut x[r * cols..r * cols + cols];
        let mut mx = f32::NEG_INFINITY;
        for &v in row.iter() {
            if v > mx {
                mx = v;
            }
        }
        let mut sum = 0f32;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}
