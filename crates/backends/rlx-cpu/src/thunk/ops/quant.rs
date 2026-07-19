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
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                let block = p / block_size;
                let s = scales[block * n + j];
                let z = if asym { zps[block * n + j] } else { 0.0 };
                let q = w_bytes[p * n + j] as f32;
                let dequantized = (q - z) * s;
                acc += x[i * k + p] * dequantized;
            }
            out[i * n + j] = acc;
        }
    }
    let _ = blocks_per_col;
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
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                let block = p / block_size;
                let s = scales[block * n + j];
                let z = if asym { zps[block * n + j] } else { 0.0 };
                let byte_idx = (p * n + j) / 2;
                let nibble = if (p * n + j) & 1 == 0 {
                    w_bytes[byte_idx] & 0x0F
                } else {
                    w_bytes[byte_idx] >> 4
                };
                let dequantized = (nibble as f32 - z) * s;
                acc += x[i * k + p] * dequantized;
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
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                let w = dequant(w_bytes[p * n + j]);
                let s = scales.get(j).copied().unwrap_or(1.0);
                acc += x[i * k + p] * w * s;
            }
            out[i * n + j] = acc;
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
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                let byte_idx = (p * n + j) / 2;
                let nibble = if (p * n + j) & 1 == 0 {
                    w_bytes[byte_idx] & 0x0F
                } else {
                    w_bytes[byte_idx] >> 4
                };
                let block = p / gs;
                let scale = fp8_e4m3_scale_to_f32(scale_bytes[block * n + j]);
                let w = fp4_e2m1_to_f32(nibble) * scale * global_scale;
                acc += x[i * k + p] * w;
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
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                let a =
                    decode(lhs_fmt, lhs[i * k + p]) * lowp_scale_at(layout, lhs_scales, i, p, nblk);
                let b =
                    decode(rhs_fmt, rhs[j * k + p]) * lowp_scale_at(layout, rhs_scales, j, p, nblk);
                acc += a * b;
            }
            out[i * n + j] = acc + bias.map_or(0.0, |bb| bb[j]);
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
                other => panic!(
                    "DequantMatMul on CPU supports Int8/Int4/FP8/NVFP4 legacy or GGUF schemes; got {other}"
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
            crate::gguf_matmul::gguf_matmul_bt_dispatch(xs, w_bytes, out, m, k, n, *scheme);
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
        crate::gguf_matmul::gguf_matmul_bt_dispatch(xs, w_bytes, out, m, k, n, scheme);
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
