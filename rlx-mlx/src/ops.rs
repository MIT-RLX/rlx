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

//! Safe wrappers over the shim's op functions. Each returns a fresh
//! `Array` whose handle owns a new MLX-side node.

use std::ptr;

use rlx_ir::DType;

use crate::array::{Array, MlxError, check, map_dtype};
use crate::ffi::{self, MlxMask, MlxReduce, MlxUnary, mlx_array_t};

macro_rules! binary {
    ($name:ident, $shim:ident) => {
        pub fn $name(a: &Array, b: &Array) -> Result<Array, MlxError> {
            let mut out: *mut mlx_array_t = ptr::null_mut();
            let rc = unsafe { ffi::$shim(a.ptr, b.ptr, &mut out) };
            check(rc)?;
            Ok(Array::from_raw(out))
        }
    };
}

binary!(matmul, rlx_mlx_op_matmul);
binary!(solve, rlx_mlx_op_solve);
binary!(add, rlx_mlx_op_add);
binary!(mul, rlx_mlx_op_mul);
binary!(sub, rlx_mlx_op_sub);
binary!(div, rlx_mlx_op_div);
binary!(max, rlx_mlx_op_max);
binary!(min, rlx_mlx_op_min);
binary!(pow, rlx_mlx_op_pow);

binary!(eq, rlx_mlx_op_eq);
binary!(ne, rlx_mlx_op_ne);
binary!(lt, rlx_mlx_op_lt);
binary!(le, rlx_mlx_op_le);
binary!(gt, rlx_mlx_op_gt);
binary!(ge, rlx_mlx_op_ge);

pub fn select(cond: &Array, x: &Array, y: &Array) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_where(cond.ptr, x.ptr, y.ptr, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn unary(a: &Array, kind: MlxUnary) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_unary(a.ptr, kind, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn reshape(a: &Array, new_shape: &[i32]) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc =
        unsafe { ffi::rlx_mlx_op_reshape(a.ptr, new_shape.as_ptr(), new_shape.len(), &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn transpose(a: &Array, perm: &[i32]) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_transpose(a.ptr, perm.as_ptr(), perm.len(), &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn slice(a: &Array, start: &[i32], stop: &[i32]) -> Result<Array, MlxError> {
    if start.len() != stop.len() {
        return Err(MlxError("slice: start/stop length mismatch".into()));
    }
    let shape = a.shape().unwrap_or_default();
    if shape.len() != start.len() {
        return Err(MlxError(format!(
            "slice: rank mismatch — array rank {} shape={shape:?}, got {} index pairs (start={start:?}, stop={stop:?})",
            shape.len(),
            start.len(),
        )));
    }
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_slice(a.ptr, start.as_ptr(), stop.as_ptr(), start.len(), &mut out)
    };
    if rc != 0 {
        let mlx_err = check(rc).unwrap_err();
        return Err(MlxError(format!(
            "slice on rank-{} array with {} indices (shape={shape:?}, start={start:?}, stop={stop:?}): {mlx_err}",
            shape.len(),
            start.len(),
        )));
    }
    Ok(Array::from_raw(out))
}

pub fn concat(arrays: &[&Array], axis: i32) -> Result<Array, MlxError> {
    let handles: Vec<*mut mlx_array_t> = arrays.iter().map(|a| a.ptr).collect();
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_concat(handles.as_ptr(), handles.len(), axis, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn broadcast_to(a: &Array, shape: &[i32]) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_broadcast_to(a.ptr, shape.as_ptr(), shape.len(), &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn take(a: &Array, indices: &Array, axis: i32) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_take(a.ptr, indices.ptr, axis, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn reduce(a: &Array, kind: MlxReduce, axes: &[i32], keep_dim: bool) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_reduce(
            a.ptr,
            kind,
            axes.as_ptr(),
            axes.len(),
            if keep_dim { 1 } else { 0 },
            &mut out,
        )
    };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn cumsum(a: &Array, axis: i32, exclusive: bool) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc =
        unsafe { ffi::rlx_mlx_op_cumsum(a.ptr, axis, if exclusive { 1 } else { 0 }, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn fft(a: &Array, inverse: bool, norm_tag: u32) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_fft(
            a.ptr,
            if inverse { 1 } else { 0 },
            norm_tag as i32,
            &mut out,
        )
    };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn rms_norm(x: &Array, gamma: &Array, eps: f32) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_rmsnorm(x.ptr, gamma.ptr, eps, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask_kind: MlxMask,
    mask: Option<&Array>,
) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let mask_ptr = mask.map(|m| m.ptr).unwrap_or(ptr::null_mut());
    let rc = unsafe {
        ffi::rlx_mlx_op_attention(q.ptr, k.ptr, v.ptr, scale, mask_kind, mask_ptr, &mut out)
    };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn conv2d(
    input: &Array,
    weight: &Array,
    stride: (i32, i32),
    padding: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_conv2d(
            input.ptr, weight.ptr, stride.0, stride.1, padding.0, padding.1, dilation.0,
            dilation.1, groups, &mut out,
        )
    };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn conv1d(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_conv1d(
            input.ptr, weight.ptr, stride, padding, dilation, groups, &mut out,
        )
    };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn conv3d(
    input: &Array,
    weight: &Array,
    stride: (i32, i32, i32),
    padding: (i32, i32, i32),
    dilation: (i32, i32, i32),
    groups: i32,
) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_conv3d(
            input.ptr, weight.ptr, stride.0, stride.1, stride.2, padding.0, padding.1, padding.2,
            dilation.0, dilation.1, dilation.2, groups, &mut out,
        )
    };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn conv_general(
    input: &Array,
    weight: &Array,
    stride: &[i32],
    padding_lo: &[i32],
    padding_hi: &[i32],
    kernel_dilation: &[i32],
    input_dilation: &[i32],
    groups: i32,
    flip: bool,
) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_conv_general(
            input.ptr,
            weight.ptr,
            stride.as_ptr(),
            stride.len(),
            padding_lo.as_ptr(),
            padding_lo.len(),
            padding_hi.as_ptr(),
            padding_hi.len(),
            kernel_dilation.as_ptr(),
            kernel_dilation.len(),
            input_dilation.as_ptr(),
            input_dilation.len(),
            groups,
            if flip { 1 } else { 0 },
            &mut out,
        )
    };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn argpartition(a: &Array, kth: i32, axis: i32) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_argpartition(a.ptr, kth, axis, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

/// Force the array into row-major contiguous storage. Matches
/// `mc::contiguous`. Use after a transpose whose strided view would
/// otherwise be elided by `mc::compile`'s optimizer.
pub fn contiguous(a: &Array) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_contiguous(a.ptr, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

#[allow(clippy::too_many_arguments)]
pub fn maxpool2d_backward_metal(
    x: &Array,
    dy: &Array,
    n: i32,
    c: i32,
    h: i32,
    w: i32,
    h_out: i32,
    w_out: i32,
    kh: i32,
    kw: i32,
    sh: i32,
    sw: i32,
    ph: i32,
    pw: i32,
) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_maxpool2d_backward_metal(
            x.ptr, dy.ptr, n, c, h, w, h_out, w_out, kh, kw, sh, sw, ph, pw, &mut out,
        )
    };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn take_along_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_take_along_axis(a.ptr, indices.ptr, axis, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn scatter_add_axis(
    a: &Array,
    indices: &Array,
    updates: &Array,
    axis: i32,
) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_scatter_add_axis(a.ptr, indices.ptr, updates.ptr, axis, &mut out)
    };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn scatter_add(
    a: &Array,
    indices: &Array,
    updates: &Array,
    axis: i32,
) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc =
        unsafe { ffi::rlx_mlx_op_scatter_add(a.ptr, indices.ptr, updates.ptr, axis, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn gather_mm(a: &Array, b: &Array, idx: &Array) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_gather_mm(a.ptr, b.ptr, idx.ptr, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn quantized_matmul(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: Option<&Array>,
    transpose: bool,
    group_size: i32,
    bits: i32,
) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let bias_ptr = biases.map(|b| b.ptr).unwrap_or(ptr::null_mut());
    let rc = unsafe {
        ffi::rlx_mlx_op_quantized_matmul(
            x.ptr,
            w.ptr,
            scales.ptr,
            bias_ptr,
            if transpose { 1 } else { 0 },
            group_size,
            bits,
            &mut out,
        )
    };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn categorical(logits: &Array, axis: i32, seed: u64) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_categorical(logits.ptr, axis, seed, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn argmax(a: &Array, axis: i32, keep_dim: bool) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_argmax(a.ptr, axis, if keep_dim { 1 } else { 0 }, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn slice_strided(
    a: &Array,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
) -> Result<Array, MlxError> {
    if start.len() != stop.len() || start.len() != strides.len() {
        return Err(MlxError(
            "slice_strided: start/stop/strides length mismatch".into(),
        ));
    }
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_slice_strided(
            a.ptr,
            start.as_ptr(),
            stop.as_ptr(),
            strides.as_ptr(),
            start.len(),
            &mut out,
        )
    };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn pad(a: &Array, low: &[i32], high: &[i32], pad_value: f32) -> Result<Array, MlxError> {
    if low.len() != high.len() {
        return Err(MlxError("pad: low/high length mismatch".into()));
    }
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_pad(
            a.ptr,
            low.as_ptr(),
            high.as_ptr(),
            low.len(),
            pad_value,
            &mut out,
        )
    };
    check(rc)?;
    Ok(Array::from_raw(out))
}

/// Top-k values along an axis (sorted descending).
pub fn topk_values(a: &Array, k: i32, axis: i32) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_topk_values(a.ptr, k, axis, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

/// Sort along an axis (ascending). Pair with negate to get descending.
pub fn sort(a: &Array, axis: i32) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_sort(a.ptr, axis, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn softmax(a: &Array, axis: i32) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_softmax(a.ptr, axis, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn gelu(a: &Array) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_gelu(a.ptr, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn silu(a: &Array) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_silu(a.ptr, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn cast(a: &Array, dtype: DType) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let rc = unsafe { ffi::rlx_mlx_op_cast(a.ptr, map_dtype(dtype), &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}

pub fn layer_norm(
    x: &Array,
    gamma: &Array,
    beta: Option<&Array>,
    eps: f32,
) -> Result<Array, MlxError> {
    let mut out: *mut mlx_array_t = ptr::null_mut();
    let beta_ptr = beta.map(|b| b.ptr).unwrap_or(ptr::null_mut());
    let rc = unsafe { ffi::rlx_mlx_op_layernorm(x.ptr, gamma.ptr, beta_ptr, eps, &mut out) };
    check(rc)?;
    Ok(Array::from_raw(out))
}
