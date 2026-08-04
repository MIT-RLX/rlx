// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

/// Fused dequant + matmul (plan #5). Int8-blockwise weights: each
/// `block_size` consecutive elements of a column share one f32
/// scale (and optionally a zero-point). The dequant happens inside
/// the inner accumulate so the f32 weight is never materialized.
///
/// `w_bytes` is the row-major i8 weight matrix `[k, n]`. `scales`
/// and `zps` are `[k/block, n]`. When `asym=false`, `zps` may be
/// empty.
///
/// Today this is the reference scalar implementation — the win is
/// memory bandwidth, not flops, since LLM weights dominate the
/// working set. A NEON SIMD path that loads 16 i8 → splat-scale →
/// fused-multiply-add is the natural follow-on.
#[allow(clippy::too_many_arguments)]
pub fn dequant_matmul_int8(
    x: &[f32],       // [m, k]
    w_bytes: &[i8],  // [k, n]
    scales: &[f32],  // [k/block, n]
    zps: &[f32],     // [k/block, n] or empty
    out: &mut [f32], // [m, n]
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
    asym: bool,
) {
    let blocks_per_col = k.div_ceil(block_size);
    // m>1 (GEMM: block-verify / prefill): the fused loop's inner access `w[p*n+j]`
    // is stride-n (column of a [k,n] row-major matrix) — cache-hostile and
    // unvectorizable. Dequant once to a contiguous f32 [k,n] and route through
    // Accelerate `sgemm` → the AMX coprocessor (weight is reused across all m rows,
    // so the dequant amortizes). m==1 (decode GEMV) keeps the fused path below —
    // memory-lean (streams the packed int8 once, no k·n·4 f32 materialization).
    if m > 1 {
        let mut w_f = vec![0f32; k * n];
        for p in 0..k {
            let b = p / block_size;
            for j in 0..n {
                let z = if asym { zps[b * n + j] } else { 0.0 };
                w_f[p * n + j] = (w_bytes[p * n + j] as f32 - z) * scales[b * n + j];
            }
        }
        crate::blas::sgemm(x, &w_f, out, m, k, n);
        return;
    }
    for i in 0..m {
        let x_row = &x[i * k..i * k + k];
        for j in 0..n {
            // Accumulate one block at a time and fold in that block's scale
            // ONCE (not per element): fewer multiplies, and a blocked sum that
            // is better conditioned than a single k-length running f32 accum
            // (the block scale no longer stretches the dynamic range mid-sum).
            // Pure reassociation of the same terms — no extra precision loss,
            // strictly less rounding.
            let mut acc = 0f32;
            for b in 0..blocks_per_col {
                let s = scales[b * n + j];
                let z = if asym { zps[b * n + j] } else { 0.0 };
                let lo = b * block_size;
                let hi = (lo + block_size).min(k);
                let mut bacc = 0f32;
                for p in lo..hi {
                    let q = w_bytes[p * n + j] as f32;
                    bacc += x_row[p] * (q - z);
                }
                acc += bacc * s;
            }
            out[i * n + j] = acc;
        }
    }
}

/// Codebook weight-synthesis matmul (single-level vector quantization).
///
/// The weight is stored transposed (`[n, k]`, GGUF "bt" layout) as codebook
/// indices: `indices[j, kb]` selects centroid `codebook[indices[j,kb]] ∈
/// ℝ^{entry_dim}`, which reconstructs the `entry_dim` weights at
/// `W[j, kb·entry_dim .. (kb+1)·entry_dim]`. Output is `y = x · Wᵀ`, i.e.
/// `out[i, j] = Σ_p x[i, p] · W[j, p]`. Like the dequant kernels, the win is
/// weight-read bandwidth: the centroid is expanded inside the accumulate and
/// the `k·n` f32 weight is never materialized (on the decode GEMV path).
///
/// Requires `k % entry_dim == 0` (guaranteed by the op's shape contract).
#[allow(clippy::too_many_arguments)]
pub fn synth_matmul_codebook(
    x: &[f32],        // [m, k]
    indices: &[u8],   // [n, k/entry_dim]
    codebook: &[f32], // [num_entries, entry_dim]
    out: &mut [f32],  // [m, n]
    m: usize,
    k: usize,
    n: usize,
    entry_dim: usize,
) {
    let kb_per_row = k / entry_dim.max(1); // codebook blocks per output column
    // m>1 (GEMM/prefill): reconstruct the dense weight once and route through
    // Accelerate `sgemm` — the codebook lookup amortizes across all m rows
    // (same trade-off as `dequant_matmul_int8`'s m>1 branch).
    if m > 1 {
        // Reconstruct Wᵀ as [n, k] row-major — CONTIGUOUS writes (one
        // `copy_from_slice` per codebook block), matching the [n, k/entry_dim]
        // index layout. The earlier [k, n] layout scattered writes with stride
        // n (cache-hostile, measured ~30× slower at real sizes). `sgemm_bt`
        // then computes out = x · Wᵀ directly from the [n, k] weight.
        let mut w_nk = vec![0f32; n * k];
        for j in 0..n {
            let wj = &mut w_nk[j * k..j * k + k];
            let row = &indices[j * kb_per_row..j * kb_per_row + kb_per_row];
            for (kb, &code) in row.iter().enumerate() {
                let c = code as usize * entry_dim;
                let base = kb * entry_dim;
                wj[base..base + entry_dim].copy_from_slice(&codebook[c..c + entry_dim]);
            }
        }
        crate::blas::sgemm_bt(x, &w_nk, out, m, k, n, 1.0);
        return;
    }
    // m==1 (decode GEMV): stream the codes, reconstruct in-loop — the packed
    // indices are read once, the k·n f32 weight is never built.
    for i in 0..m {
        let x_row = &x[i * k..i * k + k];
        for j in 0..n {
            let row = &indices[j * kb_per_row..j * kb_per_row + kb_per_row];
            let mut acc = 0f32;
            for (kb, &code) in row.iter().enumerate() {
                let c = code as usize * entry_dim;
                let cb = &codebook[c..c + entry_dim];
                let base = kb * entry_dim;
                for (t, &w) in cb.iter().enumerate() {
                    acc += x_row[base + t] * w;
                }
            }
            out[i * n + j] = acc;
        }
    }
}

/// Lower an `Op::SynthMatMul` node to a `Thunk::SynthMatMul` (resolve arena
/// offsets + derive m/k/n from operand shapes).
pub(crate) fn compile_synth_mat_mul(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
) -> Thunk {
    let Op::SynthMatMul { kind } = &node.op else {
        unreachable!()
    };
    let rlx_ir::SynthKind::Codebook {
        entry_dim,
        num_entries,
    } = kind;
    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
    let total = node.shape.num_elements().unwrap();
    let m = total / n.max(1);
    let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
    let k = x_total / m.max(1);
    Thunk::SynthMatMul {
        x: node_offset(arena, node.inputs[0]),
        indices: node_offset(arena, node.inputs[1]),
        codebook: node_offset(arena, node.inputs[2]),
        dst: node_offset(arena, node.id),
        m: m as u32,
        k: k as u32,
        n: n as u32,
        entry_dim: *entry_dim,
        num_entries: *num_entries,
    }
}

/// Interpreter-path execution of `Thunk::SynthMatMul`.
#[inline(always)]
pub(crate) fn exec_synth_mat_mul(t: &Thunk, base: *mut u8) {
    let Thunk::SynthMatMul {
        x,
        indices,
        codebook,
        dst,
        m,
        k,
        n,
        entry_dim,
        num_entries,
    } = t
    else {
        unreachable!()
    };
    let (m, k, n, d) = (*m as usize, *k as usize, *n as usize, *entry_dim as usize);
    let kb_per_row = k / d.max(1);
    unsafe {
        let xs = sl(*x, base, m * k);
        let idx = std::slice::from_raw_parts(base.add(*indices) as *const u8, n * kb_per_row);
        let cb = sl(*codebook, base, *num_entries as usize * d);
        let out = sl_mut(*dst, base, m * n);
        synth_matmul_codebook(xs, idx, cb, out, m, k, n, d);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dequant_matmul_int4(
    x: &[f32],
    w_bytes: &[u8],
    scales: &[f32],
    zps: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
    asym: bool,
) {
    let blocks_per_col = k.div_ceil(block_size);
    // m>1: dequant once to contiguous f32 [k,n] + AMX sgemm (see dequant_matmul_int8).
    // m==1 keeps the memory-lean fused nibble path (streams packed int4, k·n/2 bytes).
    if m > 1 {
        let mut w_f = vec![0f32; k * n];
        for p in 0..k {
            let b = p / block_size;
            for j in 0..n {
                let elem = p * n + j;
                let nibble = if elem & 1 == 0 {
                    w_bytes[elem / 2] & 0x0F
                } else {
                    w_bytes[elem / 2] >> 4
                };
                let z = if asym { zps[b * n + j] } else { 0.0 };
                w_f[elem] = (nibble as f32 - z) * scales[b * n + j];
            }
        }
        crate::blas::sgemm(x, &w_f, out, m, k, n);
        return;
    }
    for i in 0..m {
        let x_row = &x[i * k..i * k + k];
        for j in 0..n {
            // Block-at-a-time accumulation with the scale folded in once per
            // block (see dequant_matmul_int8). Reassociation only.
            let mut acc = 0f32;
            for b in 0..blocks_per_col {
                let s = scales[b * n + j];
                let z = if asym { zps[b * n + j] } else { 0.0 };
                let lo = b * block_size;
                let hi = (lo + block_size).min(k);
                let mut bacc = 0f32;
                for p in lo..hi {
                    let elem = p * n + j;
                    let byte_idx = elem / 2;
                    let nibble = if elem & 1 == 0 {
                        w_bytes[byte_idx] & 0x0F
                    } else {
                        w_bytes[byte_idx] >> 4
                    };
                    bacc += x_row[p] * (nibble as f32 - z);
                }
                acc += bacc * s;
            }
            out[i * n + j] = acc;
        }
    }
}

pub(crate) fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (b >> 3) & 0x0F;
    let mant = b & 0x07;
    if exp == 0 {
        if mant == 0 {
            return 0.0;
        }
        return sign * (mant as f32) * 2f32.powi(-9);
    }
    if exp == 0x0F {
        return if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
}

pub(crate) fn fp8_e5m2_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (b >> 2) & 0x1F;
    let mant = b & 0x03;
    if exp == 0 {
        if mant == 0 {
            return 0.0;
        }
        return sign * (mant as f32) * 2f32.powi(-16);
    }
    if exp == 0x1F {
        return if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    sign * (1.0 + mant as f32 / 4.0) * 2f32.powi(exp as i32 - 15)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dequant_matmul_fp8(
    x: &[f32],
    w_bytes: &[u8],
    scales: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    e5m2: bool,
) {
    let dequant = if e5m2 {
        fp8_e5m2_to_f32
    } else {
        fp8_e4m3_to_f32
    };
    // m>1: dequant once to contiguous f32 [k,n] + AMX sgemm (per-column scale folded
    // in). m==1 keeps the fused path below.
    if m > 1 {
        let mut w_f = vec![0f32; k * n];
        for p in 0..k {
            for j in 0..n {
                let s = scales.get(j).copied().unwrap_or(1.0);
                w_f[p * n + j] = dequant(w_bytes[p * n + j]) * s;
            }
        }
        crate::blas::sgemm(x, &w_f, out, m, k, n);
        return;
    }
    for i in 0..m {
        let x_row = &x[i * k..i * k + k];
        for j in 0..n {
            // The per-column scale is loop-invariant: read it once and apply it
            // to the finished dot instead of multiplying every term (the old
            // code even re-fetched `scales.get(j)` on every k iteration). Fewer
            // multiplies, one less rounding per element.
            let s = scales.get(j).copied().unwrap_or(1.0);
            let mut acc = 0f32;
            for p in 0..k {
                let w = dequant(w_bytes[p * n + j]);
                acc += x_row[p] * w;
            }
            out[i * n + j] = acc * s;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn dequant_matmul_nvfp4(
    x: &[f32],
    w_bytes: &[u8],
    scale_bytes: &[u8],
    global_scale: f32,
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use rlx_ir::{NVFP4_GROUP_SIZE, fp4_e2m1_to_f32, fp8_e4m3_scale_to_f32};
    let gs = NVFP4_GROUP_SIZE;
    let n_blocks = k.div_ceil(gs);
    // m>1: dequant once to contiguous f32 [k,n] + AMX sgemm (per-group E4M3 scale +
    // global scale folded in). m==1 keeps the fused nibble path below.
    if m > 1 {
        let mut w_f = vec![0f32; k * n];
        for p in 0..k {
            let b = p / gs;
            for j in 0..n {
                let scale = fp8_e4m3_scale_to_f32(scale_bytes[b * n + j]) * global_scale;
                let elem = p * n + j;
                let nibble = if elem & 1 == 0 {
                    w_bytes[elem / 2] & 0x0F
                } else {
                    w_bytes[elem / 2] >> 4
                };
                w_f[elem] = fp4_e2m1_to_f32(nibble) * scale;
            }
        }
        crate::blas::sgemm(x, &w_f, out, m, k, n);
        return;
    }
    for i in 0..m {
        let x_row = &x[i * k..i * k + k];
        for j in 0..n {
            // Fold the per-group E4M3 scale in once per group, and the single
            // global scale in once at the end, instead of multiplying both into
            // every element. Reassociation only — strictly fewer roundings.
            let mut acc = 0f32;
            for b in 0..n_blocks {
                let scale = fp8_e4m3_scale_to_f32(scale_bytes[b * n + j]);
                let lo = b * gs;
                let hi = (lo + gs).min(k);
                let mut bacc = 0f32;
                for p in lo..hi {
                    let elem = p * n + j;
                    let byte_idx = elem / 2;
                    let nibble = if elem & 1 == 0 {
                        w_bytes[byte_idx] & 0x0F
                    } else {
                        w_bytes[byte_idx] >> 4
                    };
                    bacc += x_row[p] * fp4_e2m1_to_f32(nibble);
                }
                acc += bacc * scale;
            }
            out[i * n + j] = acc * global_scale;
        }
    }
}

/// MxFp4x2 (two-level residual E2M1) DequantMatMul reference — the CPU decode
/// path for `QuantScheme::MxFp4x2Block`. `w_bytes` = `[plane0 | plane1]` (each
/// E2M1 nibbles packed 2/byte, `[k,n]` row-major); `scale_bytes` = `[s0 | s1]`
/// f32 per `(block = k/group, n)`. `out = x · (s0·LUT[q0] + s1·LUT[q1])`.
#[allow(clippy::too_many_arguments)]
pub fn dequant_matmul_mxfp4x2(
    x: &[f32],
    w_bytes: &[u8],
    scale_bytes: &[u8],
    group: usize,
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use rlx_ir::fp4_e2m1_to_f32;
    let plane = (k * n).div_ceil(2); // bytes per nibble plane
    let g = group.max(1);
    let nblk = k.div_ceil(g);
    let sbytes = nblk * n * 4; // bytes per f32 scale set
    let rd_scale = |half: usize, idx: usize| -> f32 {
        let o = half + idx * 4;
        f32::from_le_bytes([
            scale_bytes[o],
            scale_bytes[o + 1],
            scale_bytes[o + 2],
            scale_bytes[o + 3],
        ])
    };
    let nib = |plane_off: usize, elem: usize| -> u8 {
        let b = w_bytes[plane_off + elem / 2];
        if elem & 1 == 0 { b & 0x0F } else { b >> 4 }
    };
    for i in 0..m {
        let x_row = &x[i * k..i * k + k];
        for j in 0..n {
            // Accumulate the two residual planes' dot products per block, then
            // fold each plane's block scale in once (s0·Σq0 + s1·Σq1) rather
            // than scaling every element. Reassociation only.
            let mut acc = 0f32;
            for b in 0..nblk {
                let s0 = rd_scale(0, b * n + j);
                let s1 = rd_scale(sbytes, b * n + j);
                let lo = b * g;
                let hi = (lo + g).min(k);
                let mut acc0 = 0f32;
                let mut acc1 = 0f32;
                for p in lo..hi {
                    let elem = p * n + j;
                    let xv = x_row[p];
                    acc0 += xv * fp4_e2m1_to_f32(nib(0, elem));
                    acc1 += xv * fp4_e2m1_to_f32(nib(plane, elem));
                }
                acc += acc0 * s0 + acc1 * s1;
            }
            out[i * n + j] = acc;
        }
    }
}

// ── Native low-precision (FP8/FP6/FP4) scaled GEMM — CPU reference oracle ──
//
// Decode-and-accumulate *reference* for `Op::ScaledMatMul` + its quantize
// producers. CPUs have no fp8 matrix units; correctness, not speed, is the
// point — every GPU backend's native tensor-core path is checked against these.
// Layout is TN: lhs [m,k], rhs [n,k] (K-last), out = lhs·rhsᵀ. Block scales
// (when any) run along the last/contraction axis of each operand.

/// Blocks along a `len`-element axis (1 for per-tensor).
#[inline]
pub(crate) fn lowp_nblk(len: usize, layout: rlx_ir::ScaleLayout) -> usize {
    match layout {
        rlx_ir::ScaleLayout::PerTensor => 1,
        _ => len.div_ceil(layout.block() as usize),
    }
}

/// Snap a raw f32 scale to the grid storable for `layout`, so quantizer and
/// matmul agree bit-for-bit on the reconstructed value.
#[inline]
pub(crate) fn lowp_snap_scale(layout: rlx_ir::ScaleLayout, s: f32) -> f32 {
    use rlx_ir::lowp_codec;
    match layout {
        rlx_ir::ScaleLayout::PerTensor => s,
        rlx_ir::ScaleLayout::BlockMxE8M0 { .. } => {
            lowp_codec::e8m0_to_f32(lowp_codec::f32_to_e8m0(s))
        }
        rlx_ir::ScaleLayout::Nvfp4 { .. } => lowp_codec::decode(
            rlx_ir::ScaledFormat::F8E4M3,
            lowp_codec::encode(rlx_ir::ScaledFormat::F8E4M3, s),
        ),
    }
}

/// Scale for element (`free`, `contract`) given decoded raw scales.
#[inline]
pub(crate) fn lowp_scale_at(
    layout: rlx_ir::ScaleLayout,
    scales: &[f32],
    free: usize,
    contract: usize,
    nblk: usize,
) -> f32 {
    match layout {
        rlx_ir::ScaleLayout::PerTensor => scales.first().copied().unwrap_or(1.0),
        _ => scales[free * nblk + contract / layout.block() as usize],
    }
}

/// Compute (snapped) raw f32 scales for `x` (`[rows, cols]`, blocks along
/// `cols` = contraction). PerTensor → 1 value; block → `rows*nblk` row-major.
pub(crate) fn lowp_compute_scales(
    x: &[f32],
    fmt: rlx_ir::ScaledFormat,
    layout: rlx_ir::ScaleLayout,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let maxf = fmt.max_finite();
    let to_scale = |amax: f32| if amax > 0.0 { amax / maxf } else { 1.0 };
    match layout {
        rlx_ir::ScaleLayout::PerTensor => {
            let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            vec![to_scale(amax)]
        }
        _ => {
            let block = layout.block() as usize;
            let nblk = cols.div_ceil(block);
            let mut out = vec![1.0f32; rows * nblk];
            for r in 0..rows {
                for b in 0..nblk {
                    let lo = b * block;
                    let hi = (lo + block).min(cols);
                    let mut amax = 0.0f32;
                    for c in lo..hi {
                        amax = amax.max(x[r * cols + c].abs());
                    }
                    out[r * nblk + b] = lowp_snap_scale(layout, to_scale(amax));
                }
            }
            out
        }
    }
}

/// Quantize `x` (`[rows, cols]`, blocks along cols) to packed codes using the
/// already-snapped, decoded raw `scales`.
pub(crate) fn lowp_quantize(
    x: &[f32],
    scales: &[f32],
    fmt: rlx_ir::ScaledFormat,
    layout: rlx_ir::ScaleLayout,
    rows: usize,
    cols: usize,
    out: &mut [u8],
) {
    let nblk = lowp_nblk(cols, layout);
    for r in 0..rows {
        for c in 0..cols {
            let s = lowp_scale_at(layout, scales, r, c, nblk);
            let v = if s != 0.0 { x[r * cols + c] / s } else { 0.0 };
            out[r * cols + c] = rlx_ir::lowp_codec::encode(fmt, v);
        }
    }
}

/// TN scaled GEMM: lhs [m,k] codes, rhs [n,k] codes, out [m,n] f32.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lowp_scaled_matmul(
    lhs: &[u8],
    rhs: &[u8],
    lhs_scales: &[f32],
    rhs_scales: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    layout: rlx_ir::ScaleLayout,
    lhs_fmt: rlx_ir::ScaledFormat,
    rhs_fmt: rlx_ir::ScaledFormat,
) {
    use rlx_ir::lowp_codec::decode;
    let nblk = lowp_nblk(k, layout);
    // Both operands' scales are constant across a contraction block, so their
    // product factors out: acc += (a_scale·b_scale)·Σ decode(a)·decode(b).
    // One block spans all of k for PerTensor. Reassociation only.
    let bs = match layout {
        rlx_ir::ScaleLayout::PerTensor => k.max(1),
        _ => layout.block() as usize,
    };
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for b in 0..nblk {
                let lo = b * bs;
                let hi = (lo + bs).min(k);
                let a_scale = lowp_scale_at(layout, lhs_scales, i, lo, nblk);
                let b_scale = lowp_scale_at(layout, rhs_scales, j, lo, nblk);
                let mut bacc = 0f32;
                for p in lo..hi {
                    bacc += decode(lhs_fmt, lhs[i * k + p]) * decode(rhs_fmt, rhs[j * k + p]);
                }
                acc += bacc * (a_scale * b_scale);
            }
            out[i * n + j] = acc + bias.map_or(0.0, |bb| bb[j]);
        }
    }
}

/// Grouped (MoE) TN scaled GEMM — the expert-indexed [`lowp_scaled_matmul`].
/// `input [m,k]` codes, `weight [E,n,k]` codes (one TN slab per expert),
/// `input_scales` (`[m, nblk]` row-major), `weight_scales` (`[E·n, nblk]`
/// row-major), `expert_idx [m]` (f32-encoded), optional per-expert
/// `bias [E,n]`. `out[i] = decode(input[i]) · decode(weight[eidx[i]])ᵀ`.
/// Correctness reference — one row per token against its routed expert,
/// reusing the same block reassociation as [`lowp_scaled_matmul`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn lowp_scaled_grouped_matmul(
    input: &[u8],
    weight: &[u8],
    input_scales: &[f32],
    weight_scales: &[f32],
    expert_idx: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    num_experts: usize,
    layout: rlx_ir::ScaleLayout,
    lhs_fmt: rlx_ir::ScaledFormat,
    rhs_fmt: rlx_ir::ScaledFormat,
) {
    use rlx_ir::lowp_codec::decode;
    let nblk = lowp_nblk(k, layout);
    let bs = match layout {
        rlx_ir::ScaleLayout::PerTensor => k.max(1),
        _ => layout.block() as usize,
    };
    for i in 0..m {
        let e = expert_idx[i] as usize;
        debug_assert!(
            e < num_experts,
            "scaled grouped expert_idx out of range: {e} >= {num_experts}"
        );
        let a_codes = &input[i * k..(i + 1) * k];
        let w_codes = &weight[e * n * k..(e + 1) * n * k];
        for j in 0..n {
            let mut acc = 0f32;
            for b in 0..nblk {
                let lo = b * bs;
                let hi = (lo + bs).min(k);
                let a_scale = lowp_scale_at(layout, input_scales, i, lo, nblk);
                let w_scale = lowp_scale_at(layout, weight_scales, e * n + j, lo, nblk);
                let mut bacc = 0f32;
                for p in lo..hi {
                    bacc += decode(lhs_fmt, a_codes[p]) * decode(rhs_fmt, w_codes[j * k + p]);
                }
                acc += bacc * (a_scale * w_scale);
            }
            out[i * n + j] = acc + bias.map_or(0.0, |bb| bb[e * n + j]);
        }
    }
}

/// Reconstruct f32 from packed codes: `out[i] = decode(code[i]) · scale(block)`.
/// `[rows, cols]`, blocks along cols. Inverse of [`lowp_quantize`].
pub(crate) fn lowp_dequantize(
    codes: &[u8],
    scales: &[f32],
    fmt: rlx_ir::ScaledFormat,
    layout: rlx_ir::ScaleLayout,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    use rlx_ir::lowp_codec::decode;
    let nblk = lowp_nblk(cols, layout);
    for r in 0..rows {
        for c in 0..cols {
            let s = lowp_scale_at(layout, scales, r, c, nblk);
            out[r * cols + c] = decode(fmt, codes[r * cols + c]) * s;
        }
    }
}

/// Decode a stored scale tensor (f32 for per-tensor, E8M0/E4M3 bytes for block
/// layouts) into raw f32 scales. `n` is the scale-element count.
pub(crate) unsafe fn lowp_read_scales(
    layout: rlx_ir::ScaleLayout,
    base: *mut u8,
    offset: usize,
    n: usize,
) -> Vec<f32> {
    use rlx_ir::lowp_codec;
    match layout {
        rlx_ir::ScaleLayout::PerTensor => {
            unsafe { std::slice::from_raw_parts(base.add(offset) as *const f32, n) }.to_vec()
        }
        rlx_ir::ScaleLayout::BlockMxE8M0 { .. } => {
            let bytes = unsafe { std::slice::from_raw_parts(base.add(offset), n) };
            bytes.iter().map(|&b| lowp_codec::e8m0_to_f32(b)).collect()
        }
        rlx_ir::ScaleLayout::Nvfp4 { .. } => {
            let bytes = unsafe { std::slice::from_raw_parts(base.add(offset), n) };
            bytes
                .iter()
                .map(|&b| lowp_codec::decode(rlx_ir::ScaledFormat::F8E4M3, b))
                .collect()
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_quantize(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Quantize {
        axis,
        scales,
        zero_points,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let (chan_axis, chan_dim, inner) = quant_layout(&node.shape, *axis);
        Thunk::Quantize {
            x: node_offset(arena, node.inputs[0]),
            q: node_offset(arena, node.id),
            len: node.shape.num_elements().unwrap() as u32,
            chan_axis: chan_axis as u32,
            chan_dim: chan_dim as u32,
            inner: inner as u32,
            scales: scales.clone(),
            zero_points: zero_points.clone(),
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_fake_quantize(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::FakeQuantize {
        bits,
        axis,
        ste,
        scale_mode,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let (chan_axis, chan_dim, inner) = quant_layout(&node.shape, *axis);
        let state_off = match scale_mode {
            rlx_ir::op::ScaleMode::PerBatch => None,
            rlx_ir::op::ScaleMode::EMA { .. } | rlx_ir::op::ScaleMode::Fixed => {
                // Second input carries the [chan_dim] scale state.
                debug_assert_eq!(
                    node.inputs.len(),
                    2,
                    "EMA/Fixed FakeQuantize needs a state input"
                );
                Some(node_offset(arena, node.inputs[1]))
            }
        };
        Thunk::FakeQuantize {
            x: node_offset(arena, node.inputs[0]),
            out: node_offset(arena, node.id),
            len: node.shape.num_elements().unwrap() as u32,
            chan_axis: chan_axis as u32,
            chan_dim: chan_dim as u32,
            inner: inner as u32,
            bits: *bits,
            ste: *ste,
            scale_mode: *scale_mode,
            state_off,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_fake_quantize_l_s_q(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::FakeQuantizeLSQ { bits, axis } = &node.op else {
        unreachable!()
    };
    {
        let (chan_axis, chan_dim, inner) = quant_layout(&node.shape, *axis);
        Thunk::FakeQuantizeLSQ {
            x: node_offset(arena, node.inputs[0]),
            scale_off: node_offset(arena, node.inputs[1]),
            out: node_offset(arena, node.id),
            len: node.shape.num_elements().unwrap() as u32,
            chan_axis: chan_axis as u32,
            chan_dim: chan_dim as u32,
            inner: inner as u32,
            bits: *bits,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_fake_quantize_l_s_q_backward_x(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::FakeQuantizeLSQBackwardX { bits, axis } = &node.op else {
        unreachable!()
    };
    {
        let (chan_axis, chan_dim, inner) = quant_layout(&node.shape, *axis);
        Thunk::FakeQuantizeLSQBackwardX {
            x: node_offset(arena, node.inputs[0]),
            scale_off: node_offset(arena, node.inputs[1]),
            dy: node_offset(arena, node.inputs[2]),
            dx: node_offset(arena, node.id),
            len: node.shape.num_elements().unwrap() as u32,
            chan_axis: chan_axis as u32,
            chan_dim: chan_dim as u32,
            inner: inner as u32,
            bits: *bits,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_fake_quantize_l_s_q_backward_scale(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::FakeQuantizeLSQBackwardScale { bits, axis } = &node.op else {
        unreachable!()
    };
    {
        // Output shape is [chan_dim] — node.shape doesn't
        // describe the input data layout, but inputs[0] does.
        let in_shape = &graph.node(node.inputs[0]).shape;
        let (chan_axis, chan_dim, inner) = quant_layout(in_shape, *axis);
        Thunk::FakeQuantizeLSQBackwardScale {
            x: node_offset(arena, node.inputs[0]),
            scale_off: node_offset(arena, node.inputs[1]),
            dy: node_offset(arena, node.inputs[2]),
            dscale: node_offset(arena, node.id),
            len: in_shape.num_elements().unwrap() as u32,
            chan_axis: chan_axis as u32,
            chan_dim: chan_dim as u32,
            inner: inner as u32,
            bits: *bits,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_fake_quantize_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::FakeQuantizeBackward { bits, axis, ste } = &node.op else {
        unreachable!()
    };
    {
        let (chan_axis, chan_dim, inner) = quant_layout(&node.shape, *axis);
        Thunk::FakeQuantizeBackward {
            x: node_offset(arena, node.inputs[0]),
            dy: node_offset(arena, node.inputs[1]),
            dx: node_offset(arena, node.id),
            len: node.shape.num_elements().unwrap() as u32,
            chan_axis: chan_axis as u32,
            chan_dim: chan_dim as u32,
            inner: inner as u32,
            bits: *bits,
            ste: *ste,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_dequantize(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Dequantize {
        axis,
        scales,
        zero_points,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let (chan_axis, chan_dim, inner) = quant_layout(&node.shape, *axis);
        Thunk::Dequantize {
            q: node_offset(arena, node.inputs[0]),
            x: node_offset(arena, node.id),
            len: node.shape.num_elements().unwrap() as u32,
            chan_axis: chan_axis as u32,
            chan_dim: chan_dim as u32,
            inner: inner as u32,
            scales: scales.clone(),
            zero_points: zero_points.clone(),
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_q_mat_mul(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::QMatMul {
        x_zp,
        w_zp,
        out_zp,
        mult,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let x_shape = &graph.node(node.inputs[0]).shape;
        let w_shape = &graph.node(node.inputs[1]).shape;
        let m = x_shape.dim(0).unwrap_static();
        let k = x_shape.dim(1).unwrap_static();
        let n = w_shape.dim(1).unwrap_static();
        Thunk::QMatMul {
            x: node_offset(arena, node.inputs[0]),
            w: node_offset(arena, node.inputs[1]),
            bias: node_offset(arena, node.inputs[2]),
            out: node_offset(arena, node.id),
            m: m as u32,
            k: k as u32,
            n: n as u32,
            x_zp: *x_zp,
            w_zp: *w_zp,
            out_zp: *out_zp,
            mult: *mult,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_q_conv2d(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::QConv2d {
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        x_zp,
        w_zp,
        out_zp,
        mult,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let w_shape = &graph.node(node.inputs[1]).shape;
        let out_shape = &node.shape;
        if kernel_size.len() == 2
            && in_shape.rank() == 4
            && w_shape.rank() == 4
            && out_shape.rank() == 4
        {
            Thunk::QConv2d {
                x: node_offset(arena, node.inputs[0]),
                w: node_offset(arena, node.inputs[1]),
                bias: node_offset(arena, node.inputs[2]),
                out: node_offset(arena, node.id),
                n: in_shape.dim(0).unwrap_static() as u32,
                c_in: in_shape.dim(1).unwrap_static() as u32,
                h: in_shape.dim(2).unwrap_static() as u32,
                w_in: in_shape.dim(3).unwrap_static() as u32,
                c_out: out_shape.dim(1).unwrap_static() as u32,
                h_out: out_shape.dim(2).unwrap_static() as u32,
                w_out: out_shape.dim(3).unwrap_static() as u32,
                kh: kernel_size[0] as u32,
                kw: kernel_size[1] as u32,
                sh: stride.first().copied().unwrap_or(1) as u32,
                sw: stride.get(1).copied().unwrap_or(1) as u32,
                ph: padding.first().copied().unwrap_or(0) as u32,
                pw: padding.get(1).copied().unwrap_or(0) as u32,
                dh: dilation.first().copied().unwrap_or(1) as u32,
                dw: dilation.get(1).copied().unwrap_or(1) as u32,
                groups: *groups as u32,
                x_zp: *x_zp,
                w_zp: *w_zp,
                out_zp: *out_zp,
                mult: *mult,
            }
        } else {
            Thunk::Nop
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_dequant_mat_mul(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::DequantMatMul { scheme } = &node.op else {
        unreachable!()
    };
    {
        use rlx_ir::quant::QuantScheme;
        let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
        let total = node.shape.num_elements().unwrap();
        let m = total / n.max(1);
        let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
        let k = x_total / m.max(1);
        if scheme.is_gguf() {
            Thunk::DequantMatMulGguf {
                x: node_offset(arena, node.inputs[0]),
                w_q: node_offset(arena, node.inputs[1]),
                dst: node_offset(arena, node.id),
                m: m as u32,
                k: k as u32,
                n: n as u32,
                scheme: *scheme,
            }
        } else {
            match scheme {
                QuantScheme::Nvfp4Block => Thunk::DequantMatMulNvfp4 {
                    x: node_offset(arena, node.inputs[0]),
                    w_q: node_offset(arena, node.inputs[1]),
                    scale: node_offset(arena, node.inputs[2]),
                    global_scale: node_offset(arena, node.inputs[3]),
                    dst: node_offset(arena, node.id),
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                },
                QuantScheme::MxFp4x2Block { group_size } => Thunk::DequantMatMulMxFp4x2 {
                    x: node_offset(arena, node.inputs[0]),
                    w_q: node_offset(arena, node.inputs[1]),
                    scale: node_offset(arena, node.inputs[2]),
                    dst: node_offset(arena, node.id),
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    group: *group_size,
                },
                QuantScheme::Int4Block { block_size } => Thunk::DequantMatMulInt4 {
                    x: node_offset(arena, node.inputs[0]),
                    w_q: node_offset(arena, node.inputs[1]),
                    scale: node_offset(arena, node.inputs[2]),
                    zp: node_offset(arena, node.inputs[3]),
                    dst: node_offset(arena, node.id),
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    block_size: *block_size,
                    is_asymmetric: false,
                },
                QuantScheme::Fp8E4m3 => Thunk::DequantMatMulFp8 {
                    x: node_offset(arena, node.inputs[0]),
                    w_q: node_offset(arena, node.inputs[1]),
                    scale: node_offset(arena, node.inputs[2]),
                    dst: node_offset(arena, node.id),
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    e5m2: false,
                },
                QuantScheme::Fp8E5m2 => Thunk::DequantMatMulFp8 {
                    x: node_offset(arena, node.inputs[0]),
                    w_q: node_offset(arena, node.inputs[1]),
                    scale: node_offset(arena, node.inputs[2]),
                    dst: node_offset(arena, node.id),
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    e5m2: true,
                },
                QuantScheme::Int8Block { block_size } => Thunk::DequantMatMul {
                    x: node_offset(arena, node.inputs[0]),
                    w_q: node_offset(arena, node.inputs[1]),
                    scale: node_offset(arena, node.inputs[2]),
                    zp: node_offset(arena, node.inputs[3]),
                    dst: node_offset(arena, node.id),
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    block_size: *block_size,
                    is_asymmetric: false,
                },
                QuantScheme::Int8BlockAsym { block_size } => Thunk::DequantMatMul {
                    x: node_offset(arena, node.inputs[0]),
                    w_q: node_offset(arena, node.inputs[1]),
                    scale: node_offset(arena, node.inputs[2]),
                    zp: node_offset(arena, node.inputs[3]),
                    dst: node_offset(arena, node.id),
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    block_size: *block_size,
                    is_asymmetric: true,
                },
                QuantScheme::MlxAffine { .. }
                | QuantScheme::MlxMxfp4 { .. }
                | QuantScheme::MlxMxfp8 { .. } => Thunk::DequantMatMulMlx {
                    x: node_offset(arena, node.inputs[0]),
                    w_q: node_offset(arena, node.inputs[1]),
                    scale: node_offset(arena, node.inputs[2]),
                    zp: node_offset(arena, node.inputs[3]),
                    dst: node_offset(arena, node.id),
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    scheme: *scheme,
                },
                other => panic!(
                    "DequantMatMul on CPU supports Int8/Int4/FP8/NVFP4/MLX legacy or GGUF schemes; got {other}"
                ),
            }
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_scaled_mat_mul(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ScaledMatMul {
        lhs_format,
        rhs_format,
        scale_layout,
        has_bias,
    } = &node.op
    else {
        unreachable!()
    };
    {
        // TN: lhs [m,k], rhs [n,k], out [m,n].
        let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
        let total = node.shape.num_elements().unwrap();
        let m = total / n.max(1);
        let lhs_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
        let k = lhs_total / m.max(1);
        Thunk::ScaledMatMul {
            lhs: node_offset(arena, node.inputs[0]),
            rhs: node_offset(arena, node.inputs[1]),
            lhs_scale: node_offset(arena, node.inputs[2]),
            rhs_scale: node_offset(arena, node.inputs[3]),
            bias: if *has_bias {
                node_offset(arena, node.inputs[4])
            } else {
                0
            },
            dst: node_offset(arena, node.id),
            m: m as u32,
            k: k as u32,
            n: n as u32,
            lhs_fmt: *lhs_format,
            rhs_fmt: *rhs_format,
            layout: *scale_layout,
            has_bias: *has_bias,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_scaled_grouped_mat_mul(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ScaledGroupedMatMul {
        lhs_format,
        rhs_format,
        scale_layout,
        has_bias,
    } = &node.op
    else {
        unreachable!()
    };
    {
        // input [m,k]; weight [E,n,k] (TN, K-last); out [m,n].
        let in_shape = &graph.node(node.inputs[0]).shape;
        let w_shape = &graph.node(node.inputs[1]).shape;
        let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
        let k = in_shape.dim(in_shape.rank() - 1).unwrap_static();
        let num_experts = w_shape.dim(0).unwrap_static();
        let n = w_shape.dim(w_shape.rank() - 2).unwrap_static();
        Thunk::ScaledGroupedMatMul {
            input: node_offset(arena, node.inputs[0]),
            weight: node_offset(arena, node.inputs[1]),
            input_scale: node_offset(arena, node.inputs[2]),
            weight_scale: node_offset(arena, node.inputs[3]),
            expert_idx: node_offset(arena, node.inputs[4]),
            bias: if *has_bias {
                node_offset(arena, node.inputs[5])
            } else {
                0
            },
            dst: node_offset(arena, node.id),
            m: m as u32,
            k: k as u32,
            n: n as u32,
            num_experts: num_experts as u32,
            lhs_fmt: *lhs_format,
            rhs_fmt: *rhs_format,
            layout: *scale_layout,
            has_bias: *has_bias,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_scaled_quantize(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ScaledQuantize {
        format,
        scale_layout,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let xs = &graph.node(node.inputs[0]).shape;
        let cols = xs.dim(xs.rank() - 1).unwrap_static();
        let rows = xs.num_elements().unwrap() / cols.max(1);
        Thunk::ScaledQuantize {
            x: node_offset(arena, node.inputs[0]),
            scale: node_offset(arena, node.inputs[1]),
            dst: node_offset(arena, node.id),
            rows: rows as u32,
            cols: cols as u32,
            fmt: *format,
            layout: *scale_layout,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_scaled_quant_scale(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ScaledQuantScale {
        format,
        scale_layout,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let xs = &graph.node(node.inputs[0]).shape;
        let cols = xs.dim(xs.rank() - 1).unwrap_static();
        let rows = xs.num_elements().unwrap() / cols.max(1);
        Thunk::ScaledQuantScale {
            x: node_offset(arena, node.inputs[0]),
            dst: node_offset(arena, node.id),
            rows: rows as u32,
            cols: cols as u32,
            fmt: *format,
            layout: *scale_layout,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_scaled_dequantize(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ScaledDequantize {
        format,
        scale_layout,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let xs = &graph.node(node.inputs[0]).shape;
        let cols = xs.dim(xs.rank() - 1).unwrap_static();
        let rows = xs.num_elements().unwrap() / cols.max(1);
        Thunk::ScaledDequantize {
            codes: node_offset(arena, node.inputs[0]),
            scale: node_offset(arena, node.inputs[1]),
            dst: node_offset(arena, node.id),
            rows: rows as u32,
            cols: cols as u32,
            fmt: *format,
            layout: *scale_layout,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_dequant_grouped_mat_mul(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::DequantGroupedMatMul { scheme } = &node.op else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let w_shape = &graph.node(node.inputs[1]).shape;
        let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
        let k_dim = in_shape.dim(in_shape.rank() - 1).unwrap_static();
        let out_shape = &node.shape;
        let n = out_shape.dim(out_shape.rank() - 1).unwrap_static();
        let block_elems = scheme.gguf_block_size() as usize;
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let slab_bytes = (k_dim * n) / block_elems * block_bytes;
        let total_bytes = w_shape.num_elements().unwrap();
        let num_experts = total_bytes / slab_bytes.max(1);
        Thunk::DequantGroupedMatMulGguf {
            input: node_offset(arena, node.inputs[0]),
            w_q: node_offset(arena, node.inputs[1]),
            expert_idx: node_offset(arena, node.inputs[2]),
            dst: node_offset(arena, node.id),
            m: m as u32,
            k_dim: k_dim as u32,
            n: n as u32,
            num_experts: num_experts as u32,
            scheme: *scheme,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_dequant_grouped_mat_mul_mlx(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::DequantGroupedMatMulMlx { scheme } = &node.op else {
        unreachable!()
    };
    let in_shape = &graph.node(node.inputs[0]).shape;
    let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
    let k_dim = in_shape.dim(in_shape.rank() - 1).unwrap_static();
    let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
    // scales are `[E, n, n_groups]` — expert count is the leading dim.
    let scales_shape = &graph.node(node.inputs[2]).shape;
    let num_experts = scales_shape.dim(0).unwrap_static();
    let w_bytes = graph.node(node.inputs[1]).shape.num_elements().unwrap();
    let slab_bytes = w_bytes / num_experts.max(1);
    let scale_bf16 = graph.node(node.inputs[2]).shape.dtype() == rlx_ir::DType::BF16;
    Thunk::DequantGroupedMatMulMlx {
        input: node_offset(arena, node.inputs[0]),
        w_q: node_offset(arena, node.inputs[1]),
        scale: node_offset(arena, node.inputs[2]),
        zp: node_offset(arena, node.inputs[3]),
        expert_idx: node_offset(arena, node.inputs[4]),
        dst: node_offset(arena, node.id),
        m: m as u32,
        k_dim: k_dim as u32,
        n: n as u32,
        num_experts: num_experts as u32,
        slab_bytes: slab_bytes as u32,
        scheme: *scheme,
        scale_bf16,
    }
}

/// Per row, dequant expert `idx[row]`'s MLX-affine slab and matmul the row.
pub(crate) fn exec_dequant_grouped_mat_mul_mlx(t: &Thunk, base: *mut u8) {
    let Thunk::DequantGroupedMatMulMlx {
        input,
        w_q,
        scale,
        zp,
        expert_idx,
        dst,
        m,
        k_dim,
        n,
        num_experts,
        slab_bytes,
        scheme,
        scale_bf16,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        exec_dequant_grouped_mat_mul_mlx_inner(
            base,
            *input,
            *w_q,
            *scale,
            *zp,
            *expert_idx,
            *dst,
            *m as usize,
            *k_dim as usize,
            *n as usize,
            *num_experts as usize,
            *slab_bytes as usize,
            *scheme,
            *scale_bf16,
        );
    }
}

/// Offset-based core shared by the CPU thunk and the GPU backends'
/// host-delegate path ([`execute_dequant_grouped_matmul_mlx_f32`]).
#[allow(clippy::too_many_arguments)]
unsafe fn exec_dequant_grouped_mat_mul_mlx_inner(
    base: *mut u8,
    input: usize,
    w_q: usize,
    scale: usize,
    zp: usize,
    expert_idx: usize,
    dst: usize,
    m: usize,
    k: usize,
    n: usize,
    ne: usize,
    slab: usize,
    scheme: rlx_ir::quant::QuantScheme,
    scale_bf16: bool,
) {
    // Group size is scheme-carried; MXFP4 experts share the same 5-input op as
    // affine — the loader pre-decodes MXFP4 E8M0 scales to f32 and zeroes the
    // bias slab, so the arena layout (f32 scales/biases per group) is uniform;
    // only the per-row code decode differs (FP4 LUT vs `code·scale+bias`).
    let gs = match scheme {
        rlx_ir::quant::QuantScheme::MlxAffine { group_size, .. }
        | rlx_ir::quant::QuantScheme::MlxMxfp4 { group_size } => group_size as usize,
        other => panic!("DequantGroupedMatMulMlx: expected MlxAffine/MlxMxfp4, got {other:?}"),
    };
    let n_groups = k / gs;
    let sb_per_expert = n * n_groups; // scales/biases f32 per expert
    unsafe {
        let inp = sl(input, base, m * k);
        let ids = sl(expert_idx, base, m);
        let out = sl_mut(dst, base, m * n);
        let wt = std::slice::from_raw_parts(base.add(w_q) as *const u8, ne * slab);
        // BF16 scales/biases (half the arena of f32): decode expert `e`'s slab on
        // the fly. Byte length is 2× the f32 element count. Otherwise read as f32.
        let (scl, zpb, scl_b16, zpb_b16) = if scale_bf16 {
            let sb =
                std::slice::from_raw_parts(base.add(scale) as *const u8, ne * sb_per_expert * 2);
            let zb = std::slice::from_raw_parts(base.add(zp) as *const u8, ne * sb_per_expert * 2);
            (&[][..], &[][..], Some(sb), Some(zb))
        } else {
            (
                sl(scale, base, ne * sb_per_expert),
                sl(zp, base, ne * sb_per_expert),
                None,
                None,
            )
        };
        let bf16_slab = |bytes: &[u8], e: usize| -> Vec<f32> {
            bytes[e * sb_per_expert * 2..(e + 1) * sb_per_expert * 2]
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        };
        for r in 0..m {
            let e = (ids[r] as usize).min(ne.saturating_sub(1));
            let w_slab = &wt[e * slab..(e + 1) * slab];
            let (s_owned, b_owned);
            let (s_slab, b_slab): (&[f32], &[f32]) =
                if let (Some(sb), Some(zb)) = (scl_b16, zpb_b16) {
                    s_owned = bf16_slab(sb, e);
                    b_owned = bf16_slab(zb, e);
                    (&s_owned, &b_owned)
                } else {
                    (
                        &scl[e * sb_per_expert..(e + 1) * sb_per_expert],
                        &zpb[e * sb_per_expert..(e + 1) * sb_per_expert],
                    )
                };
            let row = &inp[r * k..(r + 1) * k];
            let res = match scheme {
                rlx_ir::quant::QuantScheme::MlxAffine { bits, group_size } => {
                    // Fused matvec: reads the packed 2-bit codes ONCE and
                    // accumulates in k-order (bit-exact with the materialize
                    // path) — parallel over the n outputs. The old
                    // `dequant_matmul_affine` materialized a whole f32 expert
                    // weight per token (16× the traffic, single-threaded); with
                    // one expert dequantized per token across a 256-expert MoE,
                    // that was the prefill bottleneck.
                    rlx_mlx_io::dequant_matvec_affine(
                        row,
                        w_slab,
                        s_slab,
                        b_slab,
                        bits as u32,
                        group_size,
                        k,
                        n,
                    )
                }
                rlx_ir::quant::QuantScheme::MlxMxfp4 { group_size } => {
                    // FUSED: decode e2m1 inline, parallel over the n outputs — no
                    // `[n,k]` f32 weight materialized per token (was `dequant_matmul_mxfp4`
                    // with m=1, which allocated + decoded the whole f32 expert weight
                    // for every routed row). Bit-exact accumulation order.
                    rlx_mlx_io::dequant_matvec_mxfp4(row, w_slab, s_slab, group_size, k, n)
                }
                _ => unreachable!(),
            };
            match res {
                Ok(o) => out[r * n..(r + 1) * n].copy_from_slice(&o),
                Err(_) => out[r * n..(r + 1) * n].fill(0.0),
            }
        }
    }
}

/// Slice-based grouped MLX-affine matmul for backends that host-delegate by
/// value (rlx-mlx / rlx-wgpu / rlx-cuda copy Arrays out as `Vec`s rather than
/// sharing the CPU arena). `x`=[m,k], `w_bytes`=`num_experts` packed slabs,
/// `scales`/`biases`=[num_experts, n, n_groups] f32, `idx`=[m] f32-encoded
/// expert ids; writes `out`=[m,n]. `x @ dequant(W_e)^T` per row.
#[allow(clippy::too_many_arguments)]
pub fn dequant_grouped_matmul_affine_bt(
    x: &[f32],
    w_bytes: &[u8],
    scales: &[f32],
    biases: &[f32],
    idx: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    bits: u32,
    group_size: usize,
) {
    use rayon::prelude::*;
    debug_assert_eq!(out.len(), m * n, "output buffer must be m×n");
    let slab = w_bytes.len() / num_experts.max(1);
    let n_groups = k / group_size.max(1);
    let sb = n * n_groups; // scales/biases f32 per expert
    // Parallelize over routed rows. This is the GPU host-delegate MoE path (amd
    // Vulkan / cuda / wgpu copy the routed inputs back and run the packed grouped
    // matmul on the CPU) — it was a SERIAL loop, so a Vulkan stage's MoE ran ~16×
    // slower than the same matmul on a native CPU stage (which is rayon-parallel).
    // Each row dequant-matmuls its expert's slab independently → embarrassingly
    // parallel; write to disjoint `out` chunks.
    out.par_chunks_mut(n).enumerate().for_each(|(r, out_row)| {
        let e = (idx[r] as usize).min(num_experts.saturating_sub(1));
        let w_slab = &w_bytes[e * slab..(e + 1) * slab];
        let s_slab = &scales[e * sb..(e + 1) * sb];
        let b_slab = &biases[e * sb..(e + 1) * sb];
        let row = &x[r * k..(r + 1) * k];
        // FUSED matvec: accumulate straight from the packed codes, no n×k f32
        // materialization (that memory traffic dominated the old path). Bit-exact.
        match rlx_mlx_io::dequant_matvec_affine(
            row,
            w_slab,
            s_slab,
            b_slab,
            bits,
            group_size as u32,
            k,
            n,
        ) {
            Ok(o) => out_row.copy_from_slice(&o),
            Err(_) => out_row.fill(0.0),
        }
    });
}

/// Grouped MLX **MXFP4** matmul (`x @ dequant(W_e)^T` per row) — the MXFP4 analog
/// of [`dequant_grouped_matmul_affine_bt`], for backends that host-delegate by value
/// (rlx-mlx). `w_bytes` = `num_experts` packed e2m1 slabs (`n·k/2` bytes each),
/// `scales` = `[num_experts, n, n_groups]` **decoded f32** (e8m0→f32; MXFP4 has no
/// biases), `idx` = `[m]` f32-encoded expert ids; writes `out` = `[m,n]`.
#[allow(clippy::too_many_arguments)]
pub fn dequant_grouped_matmul_mxfp4_bt(
    x: &[f32],
    w_bytes: &[u8],
    scales: &[f32],
    idx: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    group_size: usize,
) {
    // FUSED: decode e2m1 codes inline into the accumulate (no per-row `[n,k]` f32
    // weight materialization) + parallel over ALL m·n outputs. Was: per-row
    // `dequant_matmul_mxfp4(...,1,k,n)`, which allocated + decoded the whole ~n·k·4 f32
    // expert weight FOR EVERY token and only parallelized over the m (≈3) routed rows —
    // the packed MoE compute cliff on the GPU workers' host-delegate path (CUDA/ROCm).
    rlx_mlx_io::grouped_matmul_mxfp4_bt(
        x,
        w_bytes,
        scales,
        idx,
        out,
        m,
        k,
        n,
        num_experts,
        group_size,
    );
}

/// Host-delegate entry for GPU backends (Metal/wgpu/…) that copy the routed
/// inputs back and run the packed grouped MLX matmul on the CPU. Mirrors
/// [`execute_dequant_matmul_mlx_f32`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_dequant_grouped_matmul_mlx_f32(
    input: usize,
    w_q: usize,
    scale: usize,
    zp: usize,
    expert_idx: usize,
    dst: usize,
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    slab_bytes: usize,
    scheme: rlx_ir::quant::QuantScheme,
    scale_bf16: bool,
    base: *mut u8,
) {
    unsafe {
        exec_dequant_grouped_mat_mul_mlx_inner(
            base,
            input,
            w_q,
            scale,
            zp,
            expert_idx,
            dst,
            m,
            k,
            n,
            num_experts,
            slab_bytes,
            scheme,
            scale_bf16,
        );
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_dequant_mo_e_weights(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::DequantMoEWeights { scheme } = &node.op else {
        unreachable!()
    };
    {
        let w_shape = &graph.node(node.inputs[0]).shape;
        let out_shape = &node.shape;
        let num_experts = out_shape.dim(0).unwrap_static();
        let k_dim = out_shape.dim(1).unwrap_static();
        let n = out_shape.dim(2).unwrap_static();
        let block_elems = scheme.gguf_block_size() as usize;
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let slab_bytes = (k_dim * n) / block_elems * block_bytes;
        let total_bytes = w_shape.num_elements().unwrap();
        assert_eq!(
            total_bytes,
            num_experts * slab_bytes,
            "DequantMoEWeights packed bytes mismatch"
        );
        Thunk::DequantMoEWeightsGguf {
            w_q: node_offset(arena, node.inputs[0]),
            dst: node_offset(arena, node.id),
            k_dim: k_dim as u32,
            n: n as u32,
            num_experts: num_experts as u32,
            scheme: *scheme,
        }
    }
}

#[inline(always)]
pub(crate) fn exec_dequant_mat_mul(t: &Thunk, base: *mut u8) {
    let Thunk::DequantMatMul {
        x,
        w_q,
        scale,
        zp,
        dst,
        m,
        k,
        n,
        block_size,
        is_asymmetric,
    } = t
    else {
        unreachable!()
    };
    {
        let (m, k, n, bs) = (*m as usize, *k as usize, *n as usize, *block_size as usize);
        let n_blocks = k.div_ceil(bs);
        unsafe {
            let xs = sl(*x, base, m * k);
            let w_bytes = std::slice::from_raw_parts(base.add(*w_q) as *const i8, k * n);
            let scales = sl(*scale, base, n_blocks * n);
            let zps = if *is_asymmetric {
                sl(*zp, base, n_blocks * n)
            } else {
                &[][..]
            };
            let out = sl_mut(*dst, base, m * n);
            // Opt-in int8 W8A8 SME path (Apple M4+): quantizes activations to
            // int8 and runs the SMOPA GEMM on the coprocessor. Restricted to
            // symmetric, single-block (per-output-channel) weight quant — SMOPA
            // does one int32 reduction so it can't fold per-K-block scales or
            // asymmetric zero-points. Lossier than the f32-activation oracle, so
            // strictly `RLX_CPU_SME_W8A8=1`. Falls through otherwise.
            #[cfg(rlx_cpu_amx_sme)]
            if n_blocks == 1
                && !*is_asymmetric
                && crate::intrinsics::apple_amx::sme::w8a8_dispatch_enabled()
            {
                use crate::intrinsics::apple_amx::sme;
                if m == 1 {
                    // Decode/GEMV: bandwidth-bound row-wise int8 kernel (MOPA is
                    // catastrophic at m=1).
                    let (xq, sx) = sme::quantize_i8_symmetric(xs);
                    sme::qgemv_i8(&xq, sx, w_bytes, scales, out, k, n);
                    return;
                } else if sme::worth_sme(m, k, n) {
                    let (xq, sx) = sme::quantize_i8_symmetric(xs);
                    sme::sme_qmatmul_i8_percol(&xq, sx, w_bytes, scales, out, m, k, n);
                    return;
                }
            }
            dequant_matmul_int8(xs, w_bytes, scales, zps, out, m, k, n, bs, *is_asymmetric);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_dequant_mat_mul_gguf(t: &Thunk, base: *mut u8) {
    let Thunk::DequantMatMulGguf {
        x,
        w_q,
        dst,
        m,
        k,
        n,
        scheme,
    } = t
    else {
        unreachable!()
    };
    {
        let (m, k, n) = (*m as usize, *k as usize, *n as usize);
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let block_elems = scheme.gguf_block_size() as usize;
        debug_assert!(
            block_bytes > 0 && block_elems > 0,
            "non-GGUF scheme in GGUF arm"
        );
        debug_assert!(
            (k * n).is_multiple_of(block_elems),
            "k*n={} not aligned to GGUF block size {}",
            k * n,
            block_elems
        );
        let total_bytes = (k * n) / block_elems * block_bytes;
        unsafe {
            let xs = sl(*x, base, m * k);
            let w_bytes_ptr = base.add(*w_q) as *const u8;
            let w_bytes = std::slice::from_raw_parts(w_bytes_ptr, total_bytes);
            let out = sl_mut(*dst, base, m * n);
            crate::gguf_matmul::gguf_matmul_bt_dispatch_at(
                xs, w_bytes, out, m, k, n, *scheme, *w_q,
            );
        }
    }
}

#[inline(always)]
pub(crate) fn exec_dequant_mat_mul_int4(t: &Thunk, base: *mut u8) {
    let Thunk::DequantMatMulInt4 {
        x,
        w_q,
        scale,
        zp,
        dst,
        m,
        k,
        n,
        block_size,
        is_asymmetric,
    } = t
    else {
        unreachable!()
    };
    {
        let (m, k, n, bs) = (*m as usize, *k as usize, *n as usize, *block_size as usize);
        let n_blocks = k.div_ceil(bs);
        unsafe {
            let xs = sl(*x, base, m * k);
            let w_bytes =
                std::slice::from_raw_parts(base.add(*w_q) as *const u8, (k * n).div_ceil(2));
            let scales = sl(*scale, base, n_blocks * n);
            let zps = if *is_asymmetric {
                sl(*zp, base, n_blocks * n)
            } else {
                &[][..]
            };
            let out = sl_mut(*dst, base, m * n);
            dequant_matmul_int4(xs, w_bytes, scales, zps, out, m, k, n, bs, *is_asymmetric);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_dequant_mat_mul_fp8(t: &Thunk, base: *mut u8) {
    let Thunk::DequantMatMulFp8 {
        x,
        w_q,
        scale,
        dst,
        m,
        k,
        n,
        e5m2,
    } = t
    else {
        unreachable!()
    };
    {
        let (m, k, n) = (*m as usize, *k as usize, *n as usize);
        unsafe {
            let xs = sl(*x, base, m * k);
            let w_bytes = std::slice::from_raw_parts(base.add(*w_q) as *const u8, k * n);
            let scales = sl(*scale, base, n);
            let out = sl_mut(*dst, base, m * n);
            dequant_matmul_fp8(xs, w_bytes, scales, out, m, k, n, *e5m2);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_dequant_mat_mul_nvfp4(t: &Thunk, base: *mut u8) {
    let Thunk::DequantMatMulNvfp4 {
        x,
        w_q,
        scale,
        global_scale,
        dst,
        m,
        k,
        n,
    } = t
    else {
        unreachable!()
    };
    {
        let (m, k, n) = (*m as usize, *k as usize, *n as usize);
        let n_scale = k.div_ceil(rlx_ir::NVFP4_GROUP_SIZE) * n;
        unsafe {
            let xs = sl(*x, base, m * k);
            let w_bytes =
                std::slice::from_raw_parts(base.add(*w_q) as *const u8, (k * n).div_ceil(2));
            let scale_bytes = std::slice::from_raw_parts(base.add(*scale) as *const u8, n_scale);
            let gs = sl(*global_scale, base, 1)[0];
            let out = sl_mut(*dst, base, m * n);
            dequant_matmul_nvfp4(xs, w_bytes, scale_bytes, gs, out, m, k, n);
        }
    }
}

pub(crate) fn exec_dequant_mat_mul_mxfp4x2(t: &Thunk, base: *mut u8) {
    let Thunk::DequantMatMulMxFp4x2 {
        x,
        w_q,
        scale,
        dst,
        m,
        k,
        n,
        group,
    } = t
    else {
        unreachable!()
    };
    let (m, k, n, group) = (*m as usize, *k as usize, *n as usize, *group as usize);
    let plane = (k * n).div_ceil(2);
    let nblk = k.div_ceil(group.max(1));
    unsafe {
        let xs = sl(*x, base, m * k);
        let w_bytes = std::slice::from_raw_parts(base.add(*w_q) as *const u8, 2 * plane);
        let scale_bytes =
            std::slice::from_raw_parts(base.add(*scale) as *const u8, 2 * nblk * n * 4);
        let out = sl_mut(*dst, base, m * n);
        dequant_matmul_mxfp4x2(xs, w_bytes, scale_bytes, group, out, m, k, n);
    }
}

#[inline(always)]
pub(crate) fn exec_dequant_mat_mul_mlx(t: &Thunk, base: *mut u8) {
    let Thunk::DequantMatMulMlx {
        x,
        w_q,
        scale,
        zp,
        dst,
        m,
        k,
        n,
        scheme,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        exec_dequant_mat_mul_mlx_inner(
            base,
            *x,
            *w_q,
            *scale,
            *zp,
            *dst,
            *m as usize,
            *k as usize,
            *n as usize,
            *scheme,
        );
    }
}

/// Shared body for MLX DequantMatMul (compile closure + exec_dispatch).
pub(crate) unsafe fn exec_dequant_mat_mul_mlx_inner(
    base: *mut u8,
    x: usize,
    w_q: usize,
    scale: usize,
    zp: usize,
    dst: usize,
    m: usize,
    k: usize,
    n: usize,
    scheme: rlx_ir::quant::QuantScheme,
) {
    use rlx_ir::quant::QuantScheme;
    use rlx_mlx_io::{
        dequant_matmul_affine, dequant_mxfp4_f32, dequant_mxfp8_f32, pack_factor,
        validate_dequant_matmul_dims,
    };
    if let Err(e) = validate_dequant_matmul_dims(scheme, k, n, None) {
        panic!("DequantMatMulMlx: {e}");
    }
    let xs = unsafe { sl(x, base, m * k) };
    let out = unsafe { sl_mut(dst, base, m * n) };
    match scheme {
        QuantScheme::MlxAffine { bits, group_size } => {
            let gs = group_size as usize;
            let n_groups = k / gs;
            let pf = match pack_factor(bits as u32) {
                Ok(p) => p as usize,
                Err(e) => panic!("DequantMatMulMlx: {e}"),
            };
            let packs = gs / pf;
            let bpp = match bits {
                3 | 6 => 3,
                5 => 5,
                _ => 1,
            };
            let w_need = n * n_groups * packs * bpp;
            let w_bytes = unsafe { std::slice::from_raw_parts(base.add(w_q) as *const u8, w_need) };
            let scales = unsafe { sl(scale, base, n * n_groups) };
            let biases = unsafe { sl(zp, base, n * n_groups) };
            match dequant_matmul_affine(
                xs,
                w_bytes,
                scales,
                biases,
                bits as u32,
                group_size,
                m,
                k,
                n,
            ) {
                Ok(y) => out.copy_from_slice(&y),
                Err(e) => panic!("DequantMatMulMlx affine: {e}"),
            }
        }
        QuantScheme::MlxMxfp4 { group_size } => {
            let gs = group_size as usize;
            let n_groups = k / gs;
            let w_bytes =
                unsafe { std::slice::from_raw_parts(base.add(w_q) as *const u8, n * k / 2) };
            let scales_u8 =
                unsafe { std::slice::from_raw_parts(base.add(scale) as *const u8, n * n_groups) };
            let w_f = match dequant_mxfp4_f32(w_bytes, scales_u8, group_size, n, n_groups) {
                Ok(v) => v,
                Err(e) => panic!("DequantMatMulMlx mxfp4: {e}"),
            };
            matmul_x_wt(xs, &w_f, out, m, k, n);
        }
        QuantScheme::MlxMxfp8 { group_size } => {
            let gs = group_size as usize;
            let n_groups = k / gs;
            let w_bytes = unsafe { std::slice::from_raw_parts(base.add(w_q) as *const u8, n * k) };
            let scales_u8 =
                unsafe { std::slice::from_raw_parts(base.add(scale) as *const u8, n * n_groups) };
            let w_f = match dequant_mxfp8_f32(w_bytes, scales_u8, group_size, n, n_groups) {
                Ok(v) => v,
                Err(e) => panic!("DequantMatMulMlx mxfp8: {e}"),
            };
            matmul_x_wt(xs, &w_f, out, m, k, n);
        }
        other => panic!("DequantMatMulMlx: unexpected scheme {other}"),
    }
}

/// `out[m,n] = x[m,k] @ w_nkᵀ` (`w_nk` is `[n,k]` row-major) for MLX dequant-matmul.
///
/// Routes through BLAS so Apple Silicon hits the **AMX matrix coprocessor**
/// (Accelerate `sgemm`/`sgemv`) — the fastest CPU matmul path — instead of the
/// scalar triple-loop this used to be. The dense f32 GGUF path (`gguf_matmul_bt_cached`)
/// already did dequant→`sgemm_bt`; the MLX MXFP4/MXFP8 path did not, so this was the
/// dominant CPU cost for MLX-quantized inference (e.g. DeepSeek-V4 MXFP4). Off-Apple
/// / `--no-default-features` falls back to the portable SIMD gemm in `blas.rs` with
/// the same calling convention. Numerically equivalent to the scalar loop up to f32
/// accumulation order (blocked BLAS accumulation is if anything more accurate).
#[inline]
fn matmul_x_wt(x: &[f32], w_nk: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 1 {
        // GEMV: out[n] = w_nk[n,k] @ x[k].
        crate::blas::sgemv_nn(w_nk, x, out, n, k, 1.0, 0.0);
    } else {
        // GEMM: out[m,n] = x[m,k] @ w_nk[n,k]ᵀ.
        crate::blas::sgemm_bt(x, w_nk, out, m, k, n, 1.0);
    }
}

/// Host-fallback entry for MLX affine/mxfp `Op::DequantMatMul` (Metal unified memory).
pub unsafe fn execute_dequant_matmul_mlx_f32(
    x: usize,
    w_q: usize,
    scale: usize,
    zp: usize,
    dst: usize,
    m: usize,
    k: usize,
    n: usize,
    scheme: rlx_ir::quant::QuantScheme,
    base: *mut u8,
) {
    unsafe {
        exec_dequant_mat_mul_mlx_inner(base, x, w_q, scale, zp, dst, m, k, n, scheme);
    }
}

#[inline(always)]
pub(crate) fn exec_scaled_mat_mul(t: &Thunk, base: *mut u8) {
    let Thunk::ScaledMatMul {
        lhs,
        rhs,
        lhs_scale,
        rhs_scale,
        bias,
        dst,
        m,
        k,
        n,
        lhs_fmt,
        rhs_fmt,
        layout,
        has_bias,
    } = t
    else {
        unreachable!()
    };
    {
        let (m, k, n) = (*m as usize, *k as usize, *n as usize);
        let layout = *layout;
        let nblk = lowp_nblk(k, layout);
        let per_tensor = matches!(layout, rlx_ir::ScaleLayout::PerTensor);
        let n_lscale = if per_tensor { 1 } else { m * nblk };
        let n_rscale = if per_tensor { 1 } else { n * nblk };
        unsafe {
            let lhs_b = std::slice::from_raw_parts(base.add(*lhs) as *const u8, m * k);
            let rhs_b = std::slice::from_raw_parts(base.add(*rhs) as *const u8, n * k);
            let ls = lowp_read_scales(layout, base, *lhs_scale, n_lscale);
            let rs = lowp_read_scales(layout, base, *rhs_scale, n_rscale);
            let bias_s = if *has_bias {
                Some(sl(*bias, base, n))
            } else {
                None
            };
            let out = sl_mut(*dst, base, m * n);
            lowp_scaled_matmul(
                lhs_b, rhs_b, &ls, &rs, bias_s, out, m, n, k, layout, *lhs_fmt, *rhs_fmt,
            );
        }
    }
}

pub(crate) fn exec_scaled_grouped_mat_mul(t: &Thunk, base: *mut u8) {
    let Thunk::ScaledGroupedMatMul {
        input,
        weight,
        input_scale,
        weight_scale,
        expert_idx,
        bias,
        dst,
        m,
        k,
        n,
        num_experts,
        lhs_fmt,
        rhs_fmt,
        layout,
        has_bias,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_scaled_grouped_matmul_f32(
            *input,
            *weight,
            *input_scale,
            *weight_scale,
            *expert_idx,
            *bias,
            *dst,
            *m as usize,
            *k as usize,
            *n as usize,
            *num_experts as usize,
            *has_bias,
            *lhs_fmt,
            *rhs_fmt,
            *layout,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_scaled_quantize(t: &Thunk, base: *mut u8) {
    let Thunk::ScaledQuantize {
        x,
        scale,
        dst,
        rows,
        cols,
        fmt,
        layout,
    } = t
    else {
        unreachable!()
    };
    {
        let (rows, cols) = (*rows as usize, *cols as usize);
        let layout = *layout;
        let nblk = lowp_nblk(cols, layout);
        let n_scale = if matches!(layout, rlx_ir::ScaleLayout::PerTensor) {
            1
        } else {
            rows * nblk
        };
        unsafe {
            let xs = sl(*x, base, rows * cols);
            let scales = lowp_read_scales(layout, base, *scale, n_scale);
            let out = std::slice::from_raw_parts_mut(base.add(*dst), rows * cols);
            lowp_quantize(xs, &scales, *fmt, layout, rows, cols, out);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_scaled_quant_scale(t: &Thunk, base: *mut u8) {
    let Thunk::ScaledQuantScale {
        x,
        dst,
        rows,
        cols,
        fmt,
        layout,
    } = t
    else {
        unreachable!()
    };
    {
        let (rows, cols) = (*rows as usize, *cols as usize);
        let layout = *layout;
        let nblk = lowp_nblk(cols, layout);
        unsafe {
            let xs = sl(*x, base, rows * cols);
            let scales = lowp_compute_scales(xs, *fmt, layout, rows, cols);
            match layout {
                rlx_ir::ScaleLayout::PerTensor => {
                    sl_mut(*dst, base, 1)[0] = scales[0];
                }
                rlx_ir::ScaleLayout::BlockMxE8M0 { .. } => {
                    let out = std::slice::from_raw_parts_mut(base.add(*dst), rows * nblk);
                    for (o, &s) in out.iter_mut().zip(&scales) {
                        *o = rlx_ir::lowp_codec::f32_to_e8m0(s);
                    }
                }
                rlx_ir::ScaleLayout::Nvfp4 { .. } => {
                    let out = std::slice::from_raw_parts_mut(base.add(*dst), rows * nblk);
                    for (o, &s) in out.iter_mut().zip(&scales) {
                        *o = rlx_ir::lowp_codec::encode(rlx_ir::ScaledFormat::F8E4M3, s);
                    }
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_scaled_dequantize(t: &Thunk, base: *mut u8) {
    let Thunk::ScaledDequantize {
        codes,
        scale,
        dst,
        rows,
        cols,
        fmt,
        layout,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_scaled_dequantize_f32(
            *codes,
            *scale,
            *dst,
            *rows as usize,
            *cols as usize,
            *fmt,
            *layout,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_dequant_grouped_mat_mul_gguf(t: &Thunk, base: *mut u8) {
    let Thunk::DequantGroupedMatMulGguf {
        input,
        w_q,
        expert_idx,
        dst,
        m,
        k_dim,
        n,
        num_experts,
        scheme,
    } = t
    else {
        unreachable!()
    };
    {
        let m = *m as usize;
        let k_dim = *k_dim as usize;
        let n = *n as usize;
        let num_experts = *num_experts as usize;
        let block_elems = scheme.gguf_block_size() as usize;
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let slab_bytes = (k_dim * n) / block_elems * block_bytes;
        unsafe {
            let inp = sl(*input, base, m * k_dim);
            let wt =
                std::slice::from_raw_parts(base.add(*w_q) as *const u8, num_experts * slab_bytes);
            let ids = sl(*expert_idx, base, m);
            let out = sl_mut(*dst, base, m * n);
            crate::gguf_matmul::gguf_grouped_matmul_bt(
                inp,
                wt,
                ids,
                out,
                m,
                k_dim,
                n,
                num_experts,
                *scheme,
            );
        }
    }
}

#[inline(always)]
pub(crate) fn exec_dequant_mo_e_weights_gguf(t: &Thunk, base: *mut u8) {
    let Thunk::DequantMoEWeightsGguf {
        w_q,
        dst,
        k_dim,
        n,
        num_experts,
        scheme,
    } = t
    else {
        unreachable!()
    };
    {
        let k_dim = *k_dim as usize;
        let n = *n as usize;
        let num_experts = *num_experts as usize;
        let block_elems = scheme.gguf_block_size() as usize;
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let slab_bytes = (k_dim * n) / block_elems * block_bytes;
        unsafe {
            let wt =
                std::slice::from_raw_parts(base.add(*w_q) as *const u8, num_experts * slab_bytes);
            let out = sl_mut(*dst, base, num_experts * k_dim * n);
            crate::gguf_matmul::dequant_moe_weights_to_grouped_f32(
                wt,
                out,
                num_experts,
                k_dim,
                n,
                *scheme,
            );
        }
    }
}

#[inline(always)]
pub(crate) fn exec_q_mat_mul(t: &Thunk, base: *mut u8) {
    let Thunk::QMatMul {
        x,
        w,
        bias,
        out,
        m,
        k,
        n,
        x_zp,
        w_zp,
        out_zp,
        mult,
    } = t
    else {
        unreachable!()
    };
    {
        let m = *m as usize;
        let k = *k as usize;
        let n = *n as usize;
        unsafe {
            let x_ptr = base.add(*x) as *const i8;
            let w_ptr = base.add(*w) as *const i8;
            let bias_ptr = base.add(*bias) as *const i32;
            let out_ptr = base.add(*out) as *mut i8;
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc: i32 = *bias_ptr.add(ni);
                    for ki in 0..k {
                        let xv = *x_ptr.add(mi * k + ki) as i32 - *x_zp;
                        let wv = *w_ptr.add(ki * n + ni) as i32 - *w_zp;
                        acc += xv * wv;
                    }
                    // Requantize: round(acc · mult) + out_zp,
                    // clamped to i8.
                    let r = (acc as f32 * *mult).round() as i32 + *out_zp;
                    let r = r.clamp(-128, 127) as i8;
                    *out_ptr.add(mi * n + ni) = r;
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_quantize(t: &Thunk, base: *mut u8) {
    let Thunk::Quantize {
        x,
        q,
        len,
        chan_axis: _,
        chan_dim,
        inner,
        scales,
        zero_points,
    } = t
    else {
        unreachable!()
    };
    {
        let len = *len as usize;
        let chan_dim = *chan_dim as usize;
        let inner = *inner as usize;
        unsafe {
            let xs = sl(*x, base, len);
            let q_ptr = base.add(*q) as *mut i8;
            for i in 0..len {
                let c = if chan_dim == 1 {
                    0
                } else {
                    (i / inner) % chan_dim
                };
                let inv_scale = 1.0 / scales[c];
                let zp = zero_points[c];
                let v = (xs[i] * inv_scale).round() as i32 + zp;
                *q_ptr.add(i) = v.clamp(-128, 127) as i8;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_dequantize(t: &Thunk, base: *mut u8) {
    let Thunk::Dequantize {
        q,
        x,
        len,
        chan_axis: _,
        chan_dim,
        inner,
        scales,
        zero_points,
    } = t
    else {
        unreachable!()
    };
    {
        let len = *len as usize;
        let chan_dim = *chan_dim as usize;
        let inner = *inner as usize;
        unsafe {
            let q_ptr = base.add(*q) as *const i8;
            let out = sl_mut(*x, base, len);
            for i in 0..len {
                let c = if chan_dim == 1 {
                    0
                } else {
                    (i / inner) % chan_dim
                };
                let scale = scales[c];
                let zp = zero_points[c];
                let qv = *q_ptr.add(i) as i32;
                out[i] = (qv - zp) as f32 * scale;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_fake_quantize(t: &Thunk, base: *mut u8) {
    let Thunk::FakeQuantize {
        x,
        out,
        len,
        chan_axis: _,
        chan_dim,
        inner,
        bits,
        ste: _,
        scale_mode,
        state_off,
    } = t
    else {
        unreachable!()
    };
    {
        use rlx_ir::op::ScaleMode;
        let len = *len as usize;
        let chan_dim = *chan_dim as usize;
        let inner = *inner as usize;
        let q_max: f32 = match *bits {
            8 => 127.0,
            4 => 7.0,
            2 => 1.0,
            n => panic!("FakeQuantize: unsupported bits {n}"),
        };
        unsafe {
            let xs = sl(*x, base, len);
            let outs = sl_mut(*out, base, len);

            let mut scale = vec![0f32; chan_dim];
            match scale_mode {
                ScaleMode::PerBatch => {
                    let mut max_abs = vec![0f32; chan_dim];
                    for i in 0..len {
                        let c = if chan_dim == 1 {
                            0
                        } else {
                            (i / inner) % chan_dim
                        };
                        let a = xs[i].abs();
                        if a > max_abs[c] {
                            max_abs[c] = a;
                        }
                    }
                    for c in 0..chan_dim {
                        scale[c] = (max_abs[c] / q_max).max(1e-12);
                    }
                }
                ScaleMode::EMA { decay } => {
                    // Per-channel current max-abs, then blend
                    // into the running state in place.
                    let mut max_abs = vec![0f32; chan_dim];
                    for i in 0..len {
                        let c = if chan_dim == 1 {
                            0
                        } else {
                            (i / inner) % chan_dim
                        };
                        let a = xs[i].abs();
                        if a > max_abs[c] {
                            max_abs[c] = a;
                        }
                    }
                    let state = sl_mut(state_off.expect("EMA needs state_off"), base, chan_dim);
                    for c in 0..chan_dim {
                        let cur = (max_abs[c] / q_max).max(1e-12);
                        // Cold-start: state==0 → seed directly.
                        let blended = if state[c] <= 0.0 {
                            cur
                        } else {
                            *decay * state[c] + (1.0 - *decay) * cur
                        };
                        state[c] = blended;
                        scale[c] = blended;
                    }
                }
                ScaleMode::Fixed => {
                    let state = sl(state_off.expect("Fixed needs state_off"), base, chan_dim);
                    for c in 0..chan_dim {
                        scale[c] = state[c].max(1e-12);
                    }
                }
            }

            for i in 0..len {
                let c = if chan_dim == 1 {
                    0
                } else {
                    (i / inner) % chan_dim
                };
                let s = scale[c];
                let qv = (xs[i] / s).round().clamp(-q_max, q_max);
                outs[i] = qv * s;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_fake_quantize_l_s_q(t: &Thunk, base: *mut u8) {
    let Thunk::FakeQuantizeLSQ {
        x,
        scale_off,
        out,
        len,
        chan_axis: _,
        chan_dim,
        inner,
        bits,
    } = t
    else {
        unreachable!()
    };
    {
        let len = *len as usize;
        let chan_dim = *chan_dim as usize;
        let inner = *inner as usize;
        let q_max: f32 = match *bits {
            8 => 127.0,
            4 => 7.0,
            2 => 1.0,
            n => panic!("FakeQuantizeLSQ: bad bits {n}"),
        };
        unsafe {
            let xs = sl(*x, base, len);
            let scale = sl(*scale_off, base, chan_dim);
            let outs = sl_mut(*out, base, len);
            for i in 0..len {
                let c = if chan_dim == 1 {
                    0
                } else {
                    (i / inner) % chan_dim
                };
                let s = scale[c].max(1e-12);
                let qv = (xs[i] / s).round().clamp(-q_max, q_max);
                outs[i] = qv * s;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_fake_quantize_l_s_q_backward_x(t: &Thunk, base: *mut u8) {
    let Thunk::FakeQuantizeLSQBackwardX {
        x,
        scale_off,
        dy,
        dx,
        len,
        chan_axis: _,
        chan_dim,
        inner,
        bits,
    } = t
    else {
        unreachable!()
    };
    {
        let len = *len as usize;
        let chan_dim = *chan_dim as usize;
        let inner = *inner as usize;
        let q_max: f32 = match *bits {
            8 => 127.0,
            4 => 7.0,
            2 => 1.0,
            n => panic!("FakeQuantizeLSQBackwardX: bad bits {n}"),
        };
        unsafe {
            let xs = sl(*x, base, len);
            let scale = sl(*scale_off, base, chan_dim);
            let dys = sl(*dy, base, len);
            let outs = sl_mut(*dx, base, len);
            // STE-clipped: dx = dy when |x/s| ≤ q_max, else 0.
            for i in 0..len {
                let c = if chan_dim == 1 {
                    0
                } else {
                    (i / inner) % chan_dim
                };
                let z = xs[i] / scale[c].max(1e-12);
                outs[i] = if z.abs() <= q_max { dys[i] } else { 0.0 };
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_fake_quantize_l_s_q_backward_scale(t: &Thunk, base: *mut u8) {
    let Thunk::FakeQuantizeLSQBackwardScale {
        x,
        scale_off,
        dy,
        dscale,
        len,
        chan_axis: _,
        chan_dim,
        inner,
        bits,
    } = t
    else {
        unreachable!()
    };
    {
        let len = *len as usize;
        let chan_dim = *chan_dim as usize;
        let inner = *inner as usize;
        let q_max: f32 = match *bits {
            8 => 127.0,
            4 => 7.0,
            2 => 1.0,
            n => panic!("FakeQuantizeLSQBackwardScale: bad bits {n}"),
        };
        unsafe {
            let xs = sl(*x, base, len);
            let scale = sl(*scale_off, base, chan_dim);
            let dys = sl(*dy, base, len);
            let outs = sl_mut(*dscale, base, chan_dim);
            for v in outs.iter_mut() {
                *v = 0.0;
            }
            // ψ(z) = -z + round(z) inside range, sign(z)·q_max outside.
            // dscale[c] = sum_i ψ(x_i/s[c]) * upstream[i].
            for i in 0..len {
                let c = if chan_dim == 1 {
                    0
                } else {
                    (i / inner) % chan_dim
                };
                let s = scale[c].max(1e-12);
                let z = xs[i] / s;
                let psi = if z.abs() <= q_max {
                    -z + z.round()
                } else if z > 0.0 {
                    q_max
                } else {
                    -q_max
                };
                outs[c] += psi * dys[i];
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_fake_quantize_backward(t: &Thunk, base: *mut u8) {
    let Thunk::FakeQuantizeBackward {
        x,
        dy,
        dx,
        len,
        chan_axis: _,
        chan_dim,
        inner,
        bits,
        ste,
    } = t
    else {
        unreachable!()
    };
    {
        use rlx_ir::op::SteKind;
        let len = *len as usize;
        let chan_dim = *chan_dim as usize;
        let inner = *inner as usize;
        let q_max: f32 = match *bits {
            8 => 127.0,
            4 => 7.0,
            2 => 1.0,
            n => panic!("FakeQuantizeBackward: bad bits {n}"),
        };
        unsafe {
            let xs = sl(*x, base, len);
            let dys = sl(*dy, base, len);
            let outs = sl_mut(*dx, base, len);

            // Per-channel max-abs → scale, same as forward.
            let mut max_abs = vec![0f32; chan_dim];
            for i in 0..len {
                let c = if chan_dim == 1 {
                    0
                } else {
                    (i / inner) % chan_dim
                };
                let a = xs[i].abs();
                if a > max_abs[c] {
                    max_abs[c] = a;
                }
            }
            let mut scale = vec![0f32; chan_dim];
            for c in 0..chan_dim {
                scale[c] = (max_abs[c] / q_max).max(1e-12);
            }

            match *ste {
                SteKind::Identity => {
                    // dx = dy unchanged.
                    outs.copy_from_slice(dys);
                }
                SteKind::ClippedIdentity => {
                    // dx = dy * (|x| <= q_max·s); zero if the
                    // forward saturated.
                    for i in 0..len {
                        let c = if chan_dim == 1 {
                            0
                        } else {
                            (i / inner) % chan_dim
                        };
                        let bound = q_max * scale[c];
                        outs[i] = if xs[i].abs() <= bound { dys[i] } else { 0.0 };
                    }
                }
                SteKind::Tanh => {
                    // dx = dy * (1 - tanh²(x/s)).
                    for i in 0..len {
                        let c = if chan_dim == 1 {
                            0
                        } else {
                            (i / inner) % chan_dim
                        };
                        let t = (xs[i] / scale[c]).tanh();
                        outs[i] = dys[i] * (1.0 - t * t);
                    }
                }
                SteKind::HardTanh => {
                    // dx = dy * max(0, 1 - |x/(q_max·s)|).
                    for i in 0..len {
                        let c = if chan_dim == 1 {
                            0
                        } else {
                            (i / inner) % chan_dim
                        };
                        let bound = q_max * scale[c];
                        let attenuation = (1.0 - (xs[i] / bound).abs()).max(0.0);
                        outs[i] = dys[i] * attenuation;
                    }
                }
            }
        }
    }
}

/// Host-fallback entry for GGUF `Op::DequantMatMul` (Metal unified memory).
pub unsafe fn execute_dequant_matmul_gguf_f32(
    x: usize,
    w_q: usize,
    dst: usize,
    m: usize,
    k: usize,
    n: usize,
    scheme: rlx_ir::quant::QuantScheme,
    base: *mut u8,
) {
    unsafe {
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let block_elems = scheme.gguf_block_size() as usize;
        let total_bytes = (k * n) / block_elems * block_bytes;
        let xs = sl(x, base, m * k);
        let w_bytes = std::slice::from_raw_parts(base.add(w_q) as *const u8, total_bytes);
        let out = sl_mut(dst, base, m * n);
        crate::gguf_matmul::gguf_matmul_bt_dispatch_at(xs, w_bytes, out, m, k, n, scheme, w_q);
    }
}

/// Host-fallback entry for GGUF `Op::DequantGroupedMatMul` (MoE expert stack).
pub unsafe fn execute_dequant_grouped_matmul_gguf_f32(
    input: usize,
    w_q: usize,
    expert_idx: usize,
    dst: usize,
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    scheme: rlx_ir::quant::QuantScheme,
    base: *mut u8,
) {
    unsafe {
        let block_bytes = scheme.gguf_block_bytes() as usize;
        let block_elems = scheme.gguf_block_size() as usize;
        let slab_bytes = (k * n) / block_elems * block_bytes;
        let xs = sl(input, base, m * k);
        let w_bytes =
            std::slice::from_raw_parts(base.add(w_q) as *const u8, num_experts * slab_bytes);
        let ids = sl(expert_idx, base, m);
        let out = sl_mut(dst, base, m * n);
        crate::gguf_matmul::gguf_grouped_matmul_bt(
            xs,
            w_bytes,
            ids,
            out,
            m,
            k,
            n,
            num_experts,
            scheme,
        );
    }
}

/// Host-fallback entry for Int8 `Op::DequantMatMul` (Metal unified memory).
pub unsafe fn execute_dequant_matmul_int8_f32(
    x: usize,
    w_q: usize,
    scale: usize,
    zp: usize,
    dst: usize,
    m: usize,
    k: usize,
    n: usize,
    block_size: u32,
    is_asymmetric: bool,
    base: *mut u8,
) {
    let bs = block_size as usize;
    let n_blocks = k.div_ceil(bs);
    unsafe {
        let xs = sl(x, base, m * k);
        let w_bytes = std::slice::from_raw_parts(base.add(w_q) as *const i8, k * n);
        let scales = sl(scale, base, n_blocks * n);
        let zps = if is_asymmetric {
            sl(zp, base, n_blocks * n)
        } else {
            &[][..]
        };
        let out = sl_mut(dst, base, m * n);
        dequant_matmul_int8(xs, w_bytes, scales, zps, out, m, k, n, bs, is_asymmetric);
    }
}

/// Host-fallback entry for Int4 `Op::DequantMatMul` (Metal unified memory).
pub unsafe fn execute_dequant_matmul_int4_f32(
    x: usize,
    w_q: usize,
    scale: usize,
    zp: usize,
    dst: usize,
    m: usize,
    k: usize,
    n: usize,
    block_size: u32,
    is_asymmetric: bool,
    base: *mut u8,
) {
    let bs = block_size as usize;
    let n_blocks = k.div_ceil(bs);
    unsafe {
        let xs = sl(x, base, m * k);
        let w_bytes = std::slice::from_raw_parts(base.add(w_q) as *const u8, (k * n).div_ceil(2));
        let scales = sl(scale, base, n_blocks * n);
        let zps = if is_asymmetric {
            sl(zp, base, n_blocks * n)
        } else {
            &[][..]
        };
        let out = sl_mut(dst, base, m * n);
        dequant_matmul_int4(xs, w_bytes, scales, zps, out, m, k, n, bs, is_asymmetric);
    }
}

/// Host-fallback entry for FP8 `Op::DequantMatMul` (Metal unified memory).
pub unsafe fn execute_dequant_matmul_fp8_f32(
    x: usize,
    w_q: usize,
    scale: usize,
    dst: usize,
    m: usize,
    k: usize,
    n: usize,
    e5m2: bool,
    base: *mut u8,
) {
    unsafe {
        let xs = sl(x, base, m * k);
        let w_bytes = std::slice::from_raw_parts(base.add(w_q) as *const u8, k * n);
        let scales = sl(scale, base, n);
        let out = sl_mut(dst, base, m * n);
        dequant_matmul_fp8(xs, w_bytes, scales, out, m, k, n, e5m2);
    }
}

/// Host-fallback entry for MxFp4x2 `Op::DequantMatMul` (Metal unified memory).
/// `w_q`=[plane0|plane1] nibbles, `scale`=[s0|s1] f32; byte offsets into `base`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_dequant_matmul_mxfp4x2_f32(
    x: usize,
    w_q: usize,
    scale: usize,
    dst: usize,
    m: usize,
    k: usize,
    n: usize,
    group: usize,
    base: *mut u8,
) {
    let plane = (k * n).div_ceil(2);
    let nblk = k.div_ceil(group.max(1));
    unsafe {
        let xs = sl(x, base, m * k);
        let w_bytes = std::slice::from_raw_parts(base.add(w_q) as *const u8, 2 * plane);
        let scale_bytes =
            std::slice::from_raw_parts(base.add(scale) as *const u8, 2 * nblk * n * 4);
        let out = sl_mut(dst, base, m * n);
        dequant_matmul_mxfp4x2(xs, w_bytes, scale_bytes, group, out, m, k, n);
    }
}

/// Host-fallback entry for NVFP4 `Op::DequantMatMul` (Metal unified memory).
pub unsafe fn execute_dequant_matmul_nvfp4_f32(
    x: usize,
    w_q: usize,
    scale: usize,
    global_scale: usize,
    dst: usize,
    m: usize,
    k: usize,
    n: usize,
    base: *mut u8,
) {
    let n_scale = k.div_ceil(rlx_ir::NVFP4_GROUP_SIZE) * n;
    unsafe {
        let xs = sl(x, base, m * k);
        let w_bytes = std::slice::from_raw_parts(base.add(w_q) as *const u8, (k * n).div_ceil(2));
        let scale_bytes = std::slice::from_raw_parts(base.add(scale) as *const u8, n_scale);
        let gs = sl(global_scale, base, 1)[0];
        let out = sl_mut(dst, base, m * n);
        dequant_matmul_nvfp4(xs, w_bytes, scale_bytes, gs, out, m, k, n);
    }
}

// ── Native low-precision ScaledMatMul host fallbacks (unified-memory) ──
// Reuse the CPU oracle kernels so Metal (no FP8 matrix HW) runs the exact same
// decode-and-accumulate reference. TN layout: lhs [m,k], rhs [n,k].

/// Host fallback for `Op::ScaledQuantScale`. Byte offsets into `base`.
pub unsafe fn execute_scaled_quant_scale_f32(
    x: usize,
    dst: usize,
    rows: usize,
    cols: usize,
    fmt: rlx_ir::ScaledFormat,
    layout: rlx_ir::ScaleLayout,
    base: *mut u8,
) {
    unsafe {
        let xs = sl(x, base, rows * cols);
        let scales = lowp_compute_scales(xs, fmt, layout, rows, cols);
        let nblk = lowp_nblk(cols, layout);
        match layout {
            rlx_ir::ScaleLayout::PerTensor => {
                sl_mut(dst, base, 1)[0] = scales[0];
            }
            rlx_ir::ScaleLayout::BlockMxE8M0 { .. } => {
                let out = std::slice::from_raw_parts_mut(base.add(dst), rows * nblk);
                for (o, &s) in out.iter_mut().zip(&scales) {
                    *o = rlx_ir::lowp_codec::f32_to_e8m0(s);
                }
            }
            rlx_ir::ScaleLayout::Nvfp4 { .. } => {
                let out = std::slice::from_raw_parts_mut(base.add(dst), rows * nblk);
                for (o, &s) in out.iter_mut().zip(&scales) {
                    *o = rlx_ir::lowp_codec::encode(rlx_ir::ScaledFormat::F8E4M3, s);
                }
            }
        }
    }
}

/// Host fallback for `Op::ScaledQuantize`. Byte offsets into `base`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_scaled_quantize_f32(
    x: usize,
    scale: usize,
    dst: usize,
    rows: usize,
    cols: usize,
    fmt: rlx_ir::ScaledFormat,
    layout: rlx_ir::ScaleLayout,
    base: *mut u8,
) {
    unsafe {
        let xs = sl(x, base, rows * cols);
        let nblk = lowp_nblk(cols, layout);
        let n_scale = if matches!(layout, rlx_ir::ScaleLayout::PerTensor) {
            1
        } else {
            rows * nblk
        };
        let scales = lowp_read_scales(layout, base, scale, n_scale);
        let out = std::slice::from_raw_parts_mut(base.add(dst), rows * cols);
        lowp_quantize(xs, &scales, fmt, layout, rows, cols, out);
    }
}

/// Host fallback for `Op::ScaledDequantize` — packed codes (`U8`) → f32, the
/// inverse of [`execute_scaled_quantize_f32`]. Byte offsets into `base`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_scaled_dequantize_f32(
    codes: usize,
    scale: usize,
    dst: usize,
    rows: usize,
    cols: usize,
    fmt: rlx_ir::ScaledFormat,
    layout: rlx_ir::ScaleLayout,
    base: *mut u8,
) {
    unsafe {
        let nblk = lowp_nblk(cols, layout);
        let n_scale = if matches!(layout, rlx_ir::ScaleLayout::PerTensor) {
            1
        } else {
            rows * nblk
        };
        let cs = std::slice::from_raw_parts(base.add(codes), rows * cols);
        let scales = lowp_read_scales(layout, base, scale, n_scale);
        let out = std::slice::from_raw_parts_mut(base.add(dst) as *mut f32, rows * cols);
        lowp_dequantize(cs, &scales, fmt, layout, rows, cols, out);
    }
}

/// Host fallback for `Op::ScaledMatMul` (TN). Byte offsets into `base`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_scaled_matmul_f32(
    lhs: usize,
    rhs: usize,
    lhs_scale: usize,
    rhs_scale: usize,
    bias: usize,
    dst: usize,
    m: usize,
    k: usize,
    n: usize,
    has_bias: bool,
    lhs_fmt: rlx_ir::ScaledFormat,
    rhs_fmt: rlx_ir::ScaledFormat,
    layout: rlx_ir::ScaleLayout,
    base: *mut u8,
) {
    unsafe {
        let lhs_b = std::slice::from_raw_parts(base.add(lhs), m * k);
        let rhs_b = std::slice::from_raw_parts(base.add(rhs), n * k);
        let nblk = lowp_nblk(k, layout);
        let per_tensor = matches!(layout, rlx_ir::ScaleLayout::PerTensor);
        let n_l = if per_tensor { 1 } else { m * nblk };
        let n_r = if per_tensor { 1 } else { n * nblk };
        let ls = lowp_read_scales(layout, base, lhs_scale, n_l);
        let rs = lowp_read_scales(layout, base, rhs_scale, n_r);
        let bias_s = if has_bias {
            Some(sl(bias, base, n))
        } else {
            None
        };
        let out = sl_mut(dst, base, m * n);
        lowp_scaled_matmul(
            lhs_b, rhs_b, &ls, &rs, bias_s, out, m, n, k, layout, lhs_fmt, rhs_fmt,
        );
    }
}

/// Host fallback for `Op::ScaledGroupedMatMul` (expert-indexed TN). Byte
/// offsets into `base`. `input [m,k]` codes, `weight [E,n,k]` codes,
/// `input_scale [m,nblk]`, `weight_scale [E·n,nblk]`, `expert_idx [m]` f32,
/// optional per-expert `bias [E,n]`. Used by GPU backends over mapped memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_scaled_grouped_matmul_f32(
    input: usize,
    weight: usize,
    input_scale: usize,
    weight_scale: usize,
    expert_idx: usize,
    bias: usize,
    dst: usize,
    m: usize,
    k: usize,
    n: usize,
    num_experts: usize,
    has_bias: bool,
    lhs_fmt: rlx_ir::ScaledFormat,
    rhs_fmt: rlx_ir::ScaledFormat,
    layout: rlx_ir::ScaleLayout,
    base: *mut u8,
) {
    unsafe {
        let input_b = std::slice::from_raw_parts(base.add(input), m * k);
        let weight_b = std::slice::from_raw_parts(base.add(weight), num_experts * n * k);
        let nblk = lowp_nblk(k, layout);
        let per_tensor = matches!(layout, rlx_ir::ScaleLayout::PerTensor);
        let n_i = if per_tensor { 1 } else { m * nblk };
        let n_w = if per_tensor {
            1
        } else {
            num_experts * n * nblk
        };
        let is = lowp_read_scales(layout, base, input_scale, n_i);
        let ws = lowp_read_scales(layout, base, weight_scale, n_w);
        let ids = sl(expert_idx, base, m);
        let bias_s = if has_bias {
            Some(sl(bias, base, num_experts * n))
        } else {
            None
        };
        let out = sl_mut(dst, base, m * n);
        lowp_scaled_grouped_matmul(
            input_b,
            weight_b,
            &is,
            &ws,
            ids,
            bias_s,
            out,
            m,
            n,
            k,
            num_experts,
            layout,
            lhs_fmt,
            rhs_fmt,
        );
    }
}

/// Element-wise backward for `Op::Activation`. `xs` is the original
/// input to the forward activation; `dys` is the upstream gradient.
/// Writes `out[i] = (d/dx act(xs[i])) * dys[i]`.
/// Decompose a per-channel quantization shape into the
/// `(chan_axis, chan_dim, inner)` triplet the kernel needs to map a
/// flat output index to a channel index. Per-tensor (`axis = None`)
/// degenerates to `chan_dim = 1, inner = len`, which makes the
/// kernel's `(i / inner) % chan_dim` always 0 — same fast path the
/// scalar version used.
pub(crate) fn quant_layout(shape: &rlx_ir::Shape, axis: Option<usize>) -> (usize, usize, usize) {
    match axis {
        None => (0, 1, shape.num_elements().unwrap_or(0).max(1)),
        Some(d) => {
            let chan_dim = shape.dim(d).unwrap_static();
            let inner: usize = (d + 1..shape.rank())
                .map(|i| shape.dim(i).unwrap_static())
                .product::<usize>()
                .max(1);
            (d, chan_dim, inner)
        }
    }
}

#[cfg(test)]
mod mxfp4x2_matmul_tests {
    use super::dequant_matmul_mxfp4x2;
    use rlx_ir::ScaledFormat;
    use rlx_ir::residual::{residual_dequantize, residual_quantize};

    // The op decodes a two-level residual E2M1 weight (s0·LUT[q0] + s1·LUT[q1])
    // and matmuls with x. Verify the packed-nibble kernel reproduces a plain f32
    // matmul against the same residual-decoded weight. group = k → one MX block
    // per column (nblk = 1).
    #[test]
    fn dequant_matmul_mxfp4x2_matches_residual_decode() {
        let (m, k, n) = (3usize, 32usize, 4usize);
        let x: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let w: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect(); // [k,n]

        let plane = (k * n).div_ceil(2);
        let mut w_bytes = vec![0u8; 2 * plane];
        let (mut s0, mut s1) = (vec![0f32; n], vec![0f32; n]);
        let mut w_dq = vec![0f32; k * n]; // reference decoded weight

        for j in 0..n {
            let col: Vec<f32> = (0..k).map(|p| w[p * n + j]).collect();
            let rb = residual_quantize(&col, ScaledFormat::F4E2M1, 2);
            s0[j] = rb.scales[0];
            s1[j] = rb.scales[1];
            let dq = residual_dequantize(&rb);
            for p in 0..k {
                let elem = p * n + j;
                let byte = elem / 2;
                let shift: u32 = if elem & 1 == 0 { 0 } else { 4 };
                let mask: u8 = 0x0Fu8 << shift;
                w_bytes[byte] = (w_bytes[byte] & !mask) | ((rb.codes[0][p] & 0x0F) << shift);
                w_bytes[plane + byte] =
                    (w_bytes[plane + byte] & !mask) | ((rb.codes[1][p] & 0x0F) << shift);
                w_dq[elem] = dq[p];
            }
        }

        let mut scale_bytes = Vec::with_capacity(2 * n * 4);
        for &s in &s0 {
            scale_bytes.extend_from_slice(&s.to_le_bytes());
        }
        for &s in &s1 {
            scale_bytes.extend_from_slice(&s.to_le_bytes());
        }

        let mut out = vec![0f32; m * n];
        dequant_matmul_mxfp4x2(&x, &w_bytes, &scale_bytes, k, &mut out, m, k, n);

        let mut want = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    acc += x[i * k + p] * w_dq[p * n + j];
                }
                want[i * n + j] = acc;
            }
        }

        for (a, b) in out.iter().zip(&want) {
            assert!((a - b).abs() < 1e-4, "kernel {a} vs reference {b}");
        }
    }
}

#[cfg(test)]
mod amx_matmul_bench {
    use super::matmul_x_wt;
    use std::time::Instant;

    // `matmul_x_wt` is the MLX MXFP4/MXFP8 dequant-matmul's inner product. Measures
    // the AMX win (Accelerate sgemm/sgemv) vs the old scalar triple-loop at
    // DeepSeek-V4-real dims (k=4096), for both decode (m=1 GEMV) and block/prefill
    // (m>1 GEMM). `--nocapture` to see the table. Correctness-gated.
    fn naive(x: &[f32], w: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut a = 0f32;
                for p in 0..k {
                    a += x[i * k + p] * w[j * k + p];
                }
                out[i * n + j] = a;
            }
        }
    }

    #[test]
    fn amx_vs_scalar_dequant_matmul() {
        let k = 4096usize;
        println!("\n══ AMX (Accelerate) vs scalar matmul — MLX dequant-matmul inner, k={k} ══");
        println!(
            "{:>4} {:>6} {:>10} {:>10} {:>9}   {:>10}",
            "m", "n", "scalar", "amx", "speedup", "rel|Δ|"
        );
        for (m, n) in [
            (1usize, 2048usize),
            (1, 4096),
            (5, 4096),
            (8, 2048),
            (16, 4096),
        ] {
            let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.7).sin() * 0.1).collect();
            let w: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.31).cos() * 0.1).collect();
            let mut o_naive = vec![0f32; m * n];
            let mut o_amx = vec![0f32; m * n];
            naive(&x, &w, &mut o_naive, m, k, n);
            matmul_x_wt(&x, &w, &mut o_amx, m, k, n);
            let reps = if m == 1 { 40 } else { 20 };
            let t = Instant::now();
            for _ in 0..reps {
                naive(&x, &w, &mut o_naive, m, k, n);
            }
            let tn = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let t = Instant::now();
            for _ in 0..reps {
                matmul_x_wt(&x, &w, &mut o_amx, m, k, n);
            }
            let ta = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let err = o_naive
                .iter()
                .zip(&o_amx)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let mag = o_naive.iter().map(|v| v.abs()).fold(1e-6, f32::max);
            println!(
                "{m:>4} {n:>6} {:>8.3}ms {:>8.3}ms {:>8.1}× {:>10.1e}",
                tn,
                ta,
                tn / ta,
                err / mag
            );
            assert!(
                err / mag < 1e-3,
                "AMX matmul must match scalar (rel {:.1e})",
                err / mag
            );
        }
        println!();
    }

    // For the int8/int4/fp8/nvfp4 quant matmuls, the m>1 AMX path must equal the
    // m==1 scalar path applied row-by-row (same packed bytes) — and be much faster.
    // Uses random valid bytes: we compare two code paths, not vs a "true" dequant.
    #[test]
    fn amx_quant_matmul_schemes_m_gt_1() {
        use super::{
            dequant_matmul_fp8, dequant_matmul_int4, dequant_matmul_int8, dequant_matmul_nvfp4,
        };
        use rlx_ir::NVFP4_GROUP_SIZE;
        let (k, n, bs, mm) = (4096usize, 2048usize, 32usize, 8usize);
        let x: Vec<f32> = (0..mm * k)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.03)
            .collect();
        let nb = k.div_ceil(bs);
        // Time a full m=mm AMX call vs mm separate m=1 scalar (fused) calls; assert
        // the batched result equals the row-by-row result.
        let run = |name: &str,
                   amx: &dyn Fn(&[f32], usize, &mut [f32]),
                   scalar_row: &dyn Fn(&[f32], &mut [f32])| {
            let mut got = vec![0f32; mm * n];
            amx(&x, mm, &mut got);
            let mut refr = vec![0f32; mm * n];
            for r in 0..mm {
                let mut row = vec![0f32; n];
                scalar_row(&x[r * k..r * k + k], &mut row);
                refr[r * n..r * n + n].copy_from_slice(&row);
            }
            let err = got
                .iter()
                .zip(&refr)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let mag = refr.iter().map(|v| v.abs()).fold(1e-6, f32::max);
            let reps = 10usize;
            let t = Instant::now();
            for _ in 0..reps {
                amx(&x, mm, &mut got);
            }
            let ta = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            let t = Instant::now();
            let mut tmp = vec![0f32; n];
            for _ in 0..reps {
                for r in 0..mm {
                    scalar_row(&x[r * k..r * k + k], &mut tmp);
                }
            }
            let ts = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            println!(
                "{name:<8} m={mm} k={k} n={n}: scalar(row×m) {ts:>8.3}ms  amx {ta:>7.3}ms  {:>6.1}×  rel|Δ| {:.1e}",
                ts / ta,
                err / mag
            );
            assert!(
                err / mag < 1e-3,
                "{name}: m>1 AMX must match row-by-row scalar (rel {:.1e})",
                err / mag
            );
        };
        println!("\n══ AMX vs scalar for int8/int4/fp8/nvfp4 dequant-matmul (m>1 GEMM) ══");
        // int8
        let w8: Vec<i8> = (0..k * n)
            .map(|i| ((i * 31 % 251) as i32 - 125) as i8)
            .collect();
        let sc8: Vec<f32> = (0..nb * n)
            .map(|i| 0.008 + (i % 7) as f32 * 0.002)
            .collect();
        run(
            "int8",
            &|x, m, o| dequant_matmul_int8(x, &w8, &sc8, &[], o, m, k, n, bs, false),
            &|x, o| dequant_matmul_int8(x, &w8, &sc8, &[], o, 1, k, n, bs, false),
        );
        // int4 (packed nibbles)
        let w4: Vec<u8> = (0..k * n / 2).map(|i| (i * 37 % 256) as u8).collect();
        run(
            "int4",
            &|x, m, o| dequant_matmul_int4(x, &w4, &sc8, &[], o, m, k, n, bs, false),
            &|x, o| dequant_matmul_int4(x, &w4, &sc8, &[], o, 1, k, n, bs, false),
        );
        // fp8 (per-column scale)
        let wf8: Vec<u8> = (0..k * n).map(|i| (i * 53 % 256) as u8).collect();
        let scc: Vec<f32> = (0..n).map(|j| 0.01 + (j % 5) as f32 * 0.004).collect();
        run(
            "fp8",
            &|x, m, o| dequant_matmul_fp8(x, &wf8, &scc, o, m, k, n, false),
            &|x, o| dequant_matmul_fp8(x, &wf8, &scc, o, 1, k, n, false),
        );
        // nvfp4 (group scale + global)
        let gs = NVFP4_GROUP_SIZE;
        let nbg = k.div_ceil(gs);
        let wn4: Vec<u8> = (0..k * n / 2).map(|i| (i * 41 % 256) as u8).collect();
        let scn: Vec<u8> = (0..nbg * n).map(|i| (0x30 + (i % 40)) as u8).collect();
        run(
            "nvfp4",
            &|x, m, o| dequant_matmul_nvfp4(x, &wn4, &scn, 0.5, o, m, k, n),
            &|x, o| dequant_matmul_nvfp4(x, &wn4, &scn, 0.5, o, 1, k, n),
        );
        println!();
    }
}
