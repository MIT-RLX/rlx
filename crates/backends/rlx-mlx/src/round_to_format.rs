// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::RoundToFormat` on MLX via `fast::metal_kernel`.
//!
//! MLX may compile custom kernels with relaxed float math, so everything that
//! decides a value — exponent, NaN / infinity / subnormal tests, the block and
//! tensor maxima — works on the `f32` bits; only `ldexp` and `rint` (both exact
//! here) touch floats. Bit-identical to `rlx_ir::FloatFormat::round_slice_scaled`.
//!
//! The format, shape and scaling are baked into the kernel header (and name),
//! so MLX compiles each configuration once.

#![cfg(rlx_mlx_host)]

use std::ffi::CString;
use std::os::raw::{c_char, c_int};

use crate::array::{Array, MlxError, check};
use crate::ffi::{self, MlxDtype, mlx_array_t};
use rlx_ir::{FloatFormat, FormatScaling};

const HELPERS: &str = r#"
inline int rtf_exp(float a) { return int((as_type<uint>(a) >> 23) & 0xffu) - 127; }
inline float rtf_flush(float y) {
    uint b = as_type<uint>(y);
    return (b & 0x7fffffffu) < 0x00800000u ? as_type<float>(b & 0x80000000u) : y;
}
inline float rtf_round(float x) {
    uint bits = as_type<uint>(x);
    uint ab = bits & 0x7fffffffu;
    uint sign = bits & 0x80000000u;
    if (ab == 0u || ab > 0x7f800000u) { return x; }
    if (ab < 0x00800000u) { return as_type<float>(sign); }
    if (ab >= RTF_MAX_BITS) { return as_type<float>(sign | RTF_MAX_BITS); }
    float ax = as_type<float>(ab);
    int shift = RTF_MAN - max(rtf_exp(ax), RTF_EMIN);
    float q = ldexp(rint(ldexp(ax, shift)), -shift);
    uint qb = min(as_type<uint>(q), RTF_MAX_BITS);
    return as_type<float>(sign | (qb < 0x00800000u ? 0u : qb));
}
// `amax_bits`: bits of a non-negative, non-NaN maximum.
inline int rtf_scale_exp(uint amax_bits) {
    if (amax_bits == 0u || amax_bits >= 0x7f800000u) { return 0; }
    return clamp(int(amax_bits >> 23) - 127 - RTF_EMAX, -126, 127);
}
inline float rtf_scaled(float x, int e) { return rtf_flush(ldexp(rtf_round(ldexp(x, -e)), e)); }
// `FloatFormat::fine_scale` (smallest (1 + m/8)·2^k with s·max ≥ amax; exact
// products) and `fine_reciprocal` (8-entry table and the exponent).
inline float rtf_fine_scale(float amax) {
    int k0 = clamp(rtf_exp(amax) - RTF_EMAX - 1, -126, 125);
    float mx = as_type<float>(RTF_MAX_BITS);
    for (int k = k0; k <= k0 + 2; k++) {
        float pw = as_type<float>(uint(k + 127) << 23);
        for (int m = 0; m < 8; m++) {
            float s = (1.0f + 0.125f * float(m)) * pw;
            // Bit compare (both non-negative): relaxed math cannot rewrite it.
            if (as_type<uint>(s * mx) >= as_type<uint>(amax)) { return s; }
        }
    }
    return as_type<float>(0x7f700000u);
}
inline float rtf_fine_rcp(float s) {
    const uint tab[8] = {0x3f800000u, 0x3f638e39u, 0x3f4ccccdu, 0x3f3a2e8cu,
                         0x3f2aaaabu, 0x3f1d89d9u, 0x3f124925u, 0x3f088889u};
    uint b = as_type<uint>(s);
    int ne = 127 - int((b >> 23) & 0xffu);
    if (ne < -126) { return 0.0f; }
    ne = min(ne, 127);
    return as_type<float>(tab[(b >> 20) & 7u]) * as_type<float>(uint(ne + 127) << 23);
}
// Squared error of a block at scale 2^e, in index order (`FloatFormat::block_scale`);
// multiply and add are separate statements so they are not contracted.
inline float rtf_sq_err(const device float* x, uint start, uint end, int e) {
    float acc = 0.0f;
    for (uint i = start; i < end; i++) {
        float d = rtf_scaled(x[i], e) - x[i];
        float d2 = d * d;
        acc = acc + d2;
    }
    return acc;
}
// |x| bits, or 0 for NaN, so NaN never wins a maximum (as in the reference).
inline uint rtf_abs_bits(float x) {
    uint b = as_type<uint>(x) & 0x7fffffffu;
    return b > 0x7f800000u ? 0u : b;
}
"#;

/// Elementwise body (`MODE` 0 or 1) or per-block body (`MODE` 2).
const ROUND_BODY: &str = r#"
const uint t = thread_position_in_grid.x;
if (RTF_MODE >= 2u) {
    uint per_row = (RTF_LAST + RTF_BLOCK - 1u) / RTF_BLOCK;
    uint row = t / per_row;
    uint start = row * RTF_LAST + (t % per_row) * RTF_BLOCK;
    if (start >= RTF_N) { return; }
    uint end = min(start + RTF_BLOCK, row * RTF_LAST + RTF_LAST);
    uint m = 0u;
    for (uint i = start; i < end; i++) { m = max(m, rtf_abs_bits(x[i])); }
    if (RTF_MODE == 3u) {
        float s = 1.0f, r = 1.0f;
        if (m != 0u && m < 0x7f800000u) { s = rtf_fine_scale(as_type<float>(m)); r = rtf_fine_rcp(s); }
        for (uint i = start; i < end; i++) { out[i] = rtf_flush(rtf_round(x[i] * r) * s); }
        return;
    }
    int e = rtf_scale_exp(m);
    // One binade up when the MX scale clips the block max and that rounds the
    // block with a smaller squared error (ties keep the MX scale).
    if (m != 0u && m < 0x7f800000u && e < 127
        && as_type<uint>(ldexp(as_type<float>(m), -e)) > RTF_MAX_BITS
        && rtf_sq_err(x, start, end, e + 1) < rtf_sq_err(x, start, end, e)) {
        e += 1;
    }
    for (uint i = start; i < end; i++) { out[i] = rtf_scaled(x[i], e); }
} else {
    if (t >= RTF_N) { return; }
    int e = RTF_MODE == 1u ? rtf_scale_exp(as_type<uint>(amax[0])) : 0;
    out[t] = rtf_scaled(x[t], e);
}
"#;

/// One threadgroup: the NaN-free max of `|x|`, written as an `f32`.
const AMAX_BODY: &str = r#"
const uint tid = thread_position_in_threadgroup.x;
threadgroup uint part[256];
uint m = 0u;
for (uint i = tid; i < RTF_N; i += 256u) { m = max(m, rtf_abs_bits(x[i])); }
part[tid] = m;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint s = 128u; s > 0u; s >>= 1u) {
    if (tid < s) { part[tid] = max(part[tid], part[tid + s]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (tid == 0u) { out[0] = as_type<float>(part[0]); }
"#;

/// Round `x` (any shape, `n` elements, rows of `last_dim`) to `format`.
pub fn round_to_format_ondevice(
    x: &Array,
    shape: &[i32],
    n: usize,
    last_dim: usize,
    format: FloatFormat,
    scaling: FormatScaling,
) -> Result<Array, MlxError> {
    let (mode, block) = match scaling {
        FormatScaling::None => (0u32, 1u32),
        FormatScaling::PerTensor => (1, 1),
        FormatScaling::Block(b) => (2, b.max(1)),
        FormatScaling::FineBlock(b) => (3, b.max(1)),
    };
    let last_dim = last_dim.max(1);
    let header = format!(
        "constant uint RTF_N = {n}u;\nconstant uint RTF_LAST = {last_dim}u;\n\
         constant uint RTF_BLOCK = {block}u;\nconstant uint RTF_MODE = {mode}u;\n\
         constant int RTF_MAN = {};\nconstant int RTF_EMIN = {};\nconstant int RTF_EMAX = {};\n\
         constant uint RTF_MAX_BITS = {}u;\n{HELPERS}",
        format.man_bits,
        format.emin(),
        format.emax(),
        format.max_normal.to_bits(),
    );
    let tag = format!(
        "{n}_{last_dim}_{block}_{mode}_{}_{}_{}_{:x}",
        format.man_bits,
        format.emin(),
        format.emax(),
        format.max_normal.to_bits()
    )
    .replace('-', "m");
    let flat = crate::ops::reshape(x, &[n as i32])?;

    let amax = if mode == 1 {
        dispatch(
            &format!("rtf_amax_{tag}"),
            AMAX_BODY,
            &header,
            &[("x", &flat)],
            1,
            (256, 256),
        )?
    } else {
        // Unused unless mode 1; MLX needs a bound input either way.
        Array::from_f32_slice(&[0.0], &[1], rlx_ir::DType::F32)?
    };
    let threads = if mode >= 2 {
        (n / last_dim) * last_dim.div_ceil(block as usize)
    } else {
        n
    };
    let rounded = dispatch(
        &format!("rtf_round_{tag}"),
        ROUND_BODY,
        &header,
        &[("x", &flat), ("amax", &amax)],
        n,
        (threads.max(1), threads.clamp(1, 256)),
    )?;
    crate::ops::reshape(&rounded, shape)
}

/// One `fast::metal_kernel` launch with an `f32` output named `out`.
pub(crate) fn dispatch(
    name: &str,
    body: &str,
    header: &str,
    inputs: &[(&str, &Array)],
    out_len: usize,
    (grid, tg): (usize, usize),
) -> Result<Array, MlxError> {
    let name_c = CString::new(name).unwrap();
    let body_c = CString::new(body).unwrap();
    let header_c = CString::new(header).unwrap();
    let out_c = CString::new("out").unwrap();
    let in_names: Vec<CString> = inputs
        .iter()
        .map(|(n, _)| CString::new(*n).unwrap())
        .collect();
    let in_name_ptrs: Vec<*const c_char> = in_names.iter().map(|c| c.as_ptr()).collect();
    let in_arrays: Vec<*mut mlx_array_t> = inputs.iter().map(|(_, a)| a.ptr).collect();
    let out_shape: [c_int; 1] = [out_len as c_int];
    let mut out_handle: *mut mlx_array_t = std::ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_metal_kernel_dispatch(
            name_c.as_ptr(),
            body_c.as_ptr(),
            header_c.as_ptr(),
            in_name_ptrs.as_ptr(),
            inputs.len(),
            out_c.as_ptr(),
            in_arrays.as_ptr(),
            out_shape.as_ptr(),
            1,
            MlxDtype::F32,
            grid as c_int,
            1,
            1,
            tg as c_int,
            1,
            1,
            &mut out_handle,
        )
    };
    check(rc)?;
    Ok(Array::from_raw(out_handle))
}
