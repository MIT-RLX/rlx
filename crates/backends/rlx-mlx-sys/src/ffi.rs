// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Raw `extern "C"` bindings to the C++ shim in cpp/rlx_mlx_shim.cpp.
//!
//! This module deliberately stays small and mechanical — it mirrors
//! the header one-for-one. Higher-level RAII and ergonomic wrappers
//! live in `array` / `ops`.

use std::ffi::{c_char, c_float, c_int};
use std::os::raw::c_void;

#[repr(C)]
pub struct mlx_array_t {
    _private: [u8; 0],
}

// Mirror rlx_mlx_dtype_t in the header. MLX has native dtypes for
// every variant rlx-ir declares, so the mapping is total.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxDtype {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I32 = 3,
    F64 = 4,
    I8 = 5,
    I16 = 6,
    I64 = 7,
    U8 = 8,
    U32 = 9,
    Bool = 10,
}

// Mirror rlx_mlx_unary_t.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxUnary {
    Relu = 0,
    Sigmoid = 1,
    Tanh = 2,
    Exp = 3,
    Log = 4,
    Sqrt = 5,
    Rsqrt = 6,
    Neg = 7,
    Abs = 8,
    Erf = 9,
    Round = 10,
    Sin = 11,
    Cos = 12,
    Tan = 13,
    Atan = 14,
    Reciprocal = 15,
    Floor = 16,
    Ceil = 17,
    Sign = 18,
    Softplus = 19,
    Elu = 20,
    HardSwish = 21,
    HardSigmoid = 22,
    Mish = 23,
    Softsign = 24,
    LogSigmoid = 25,
}

// Mirror rlx_mlx_reduce_t.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxReduce {
    Sum = 0,
    Mean = 1,
    Max = 2,
    Min = 3,
    Prod = 4,
}

// Mirror rlx_mlx_mask_t.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxMask {
    None = 0,
    Causal = 1,
    Sliding = 2,
    Custom = 3,
}

pub const RLX_MLX_OK: c_int = 0;

unsafe extern "C" {
    pub fn rlx_mlx_last_error() -> *const c_char;
    pub fn rlx_mlx_set_last_error(msg: *const c_char);
    pub fn rlx_mlx_version() -> *const c_char;
    pub fn rlx_mlx_device_name() -> *const c_char;

    pub fn rlx_mlx_array_from_data(
        shape: *const c_int,
        ndim: usize,
        data: *const c_float,
        nelems: usize,
        dtype: MlxDtype,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_array_from_data_view(
        shape: *const c_int,
        ndim: usize,
        data: *mut c_void,
        nbytes: usize,
        dtype: MlxDtype,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_array_from_bytes(
        shape: *const c_int,
        ndim: usize,
        data: *const c_void,
        nbytes: usize,
        dtype: MlxDtype,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_array_to_bytes(
        h: *mut mlx_array_t,
        dst: *mut c_void,
        dst_cap: usize,
        out_nbytes: *mut usize,
    ) -> c_int;

    #[allow(dead_code)]
    pub fn rlx_mlx_dtype_size(dtype: MlxDtype) -> usize;

    pub fn rlx_mlx_array_free(h: *mut mlx_array_t);
    pub fn rlx_mlx_array_clone(h: *mut mlx_array_t, out: *mut *mut mlx_array_t) -> c_int;

    pub fn rlx_mlx_array_shape(
        h: *mut mlx_array_t,
        out_shape: *mut c_int,
        cap: usize,
        out_ndim: *mut usize,
    ) -> c_int;

    pub fn rlx_mlx_array_to_f32(h: *mut mlx_array_t, dst: *mut c_float, nelems: usize) -> c_int;

    pub fn rlx_mlx_eval(handles: *const *mut mlx_array_t, n: usize) -> c_int;
    pub fn rlx_mlx_async_eval(handles: *const *mut mlx_array_t, n: usize) -> c_int;
    pub fn rlx_mlx_synchronize() -> c_int;

    pub fn rlx_mlx_op_matmul(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_solve(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_metal_kernel_dispatch(
        name: *const std::os::raw::c_char,
        source: *const std::os::raw::c_char,
        header: *const std::os::raw::c_char,
        input_names: *const *const std::os::raw::c_char,
        n_inputs: usize,
        output_name: *const std::os::raw::c_char,
        inputs: *const *mut mlx_array_t,
        output_shape: *const std::os::raw::c_int,
        output_rank: usize,
        output_dtype: MlxDtype,
        grid_x: c_int,
        grid_y: c_int,
        grid_z: c_int,
        tg_x: c_int,
        tg_y: c_int,
        tg_z: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_add(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_fmod(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_bitand(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_bitor(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_bitxor(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_shl(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_shr(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_atan2(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_mul(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_sub(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_div(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_softmax(
        a: *mut mlx_array_t,
        axis: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_gelu(a: *mut mlx_array_t, out: *mut *mut mlx_array_t) -> c_int;
    pub fn rlx_mlx_op_silu(a: *mut mlx_array_t, out: *mut *mut mlx_array_t) -> c_int;
    pub fn rlx_mlx_op_cast(
        a: *mut mlx_array_t,
        dtype: MlxDtype,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_layernorm(
        x: *mut mlx_array_t,
        gamma: *mut mlx_array_t,
        beta_or_null: *mut mlx_array_t,
        eps: c_float,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_max(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_min(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_pow(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_eq(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_ne(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_lt(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_le(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_gt(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_op_ge(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_where(
        cond: *mut mlx_array_t,
        x: *mut mlx_array_t,
        y: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_unary(
        a: *mut mlx_array_t,
        kind: MlxUnary,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_reshape(
        a: *mut mlx_array_t,
        new_shape: *const c_int,
        ndim: usize,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_transpose(
        a: *mut mlx_array_t,
        perm: *const c_int,
        ndim: usize,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_slice(
        a: *mut mlx_array_t,
        start: *const c_int,
        stop: *const c_int,
        ndim: usize,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_concat(
        arrays: *const *mut mlx_array_t,
        n: usize,
        axis: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_broadcast_to(
        a: *mut mlx_array_t,
        shape: *const c_int,
        ndim: usize,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_take(
        a: *mut mlx_array_t,
        indices: *mut mlx_array_t,
        axis: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_reduce(
        a: *mut mlx_array_t,
        kind: MlxReduce,
        axes: *const c_int,
        n_axes: usize,
        keep_dim: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_cumsum(
        a: *mut mlx_array_t,
        axis: c_int,
        exclusive: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_cumprod(
        a: *mut mlx_array_t,
        axis: c_int,
        exclusive: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_cummax(
        a: *mut mlx_array_t,
        axis: c_int,
        exclusive: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_fft(
        a: *mut mlx_array_t,
        inverse: c_int,
        norm_tag: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_rmsnorm(
        x: *mut mlx_array_t,
        gamma: *mut mlx_array_t,
        eps: c_float,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_attention(
        q: *mut mlx_array_t,
        k: *mut mlx_array_t,
        v: *mut mlx_array_t,
        scale: c_float,
        mask_kind: MlxMask,
        mask_or_null: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_conv2d(
        input: *mut mlx_array_t,
        weight: *mut mlx_array_t,
        stride_h: c_int,
        stride_w: c_int,
        pad_h: c_int,
        pad_w: c_int,
        dil_h: c_int,
        dil_w: c_int,
        groups: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_conv1d(
        input: *mut mlx_array_t,
        weight: *mut mlx_array_t,
        stride: c_int,
        padding: c_int,
        dilation: c_int,
        groups: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_conv3d(
        input: *mut mlx_array_t,
        weight: *mut mlx_array_t,
        stride_d: c_int,
        stride_h: c_int,
        stride_w: c_int,
        pad_d: c_int,
        pad_h: c_int,
        pad_w: c_int,
        dil_d: c_int,
        dil_h: c_int,
        dil_w: c_int,
        groups: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_conv_transpose2d(
        input: *mut mlx_array_t,
        weight: *mut mlx_array_t,
        stride_h: c_int,
        stride_w: c_int,
        pad_h: c_int,
        pad_w: c_int,
        dil_h: c_int,
        dil_w: c_int,
        out_pad_h: c_int,
        out_pad_w: c_int,
        groups: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_conv_transpose3d(
        input: *mut mlx_array_t,
        weight: *mut mlx_array_t,
        stride_d: c_int,
        stride_h: c_int,
        stride_w: c_int,
        pad_d: c_int,
        pad_h: c_int,
        pad_w: c_int,
        dil_d: c_int,
        dil_h: c_int,
        dil_w: c_int,
        out_pad_d: c_int,
        out_pad_h: c_int,
        out_pad_w: c_int,
        groups: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_conv_general(
        input: *mut mlx_array_t,
        weight: *mut mlx_array_t,
        stride: *const c_int,
        stride_n: usize,
        padding_lo: *const c_int,
        padding_lo_n: usize,
        padding_hi: *const c_int,
        padding_hi_n: usize,
        kernel_dilation: *const c_int,
        kernel_dilation_n: usize,
        input_dilation: *const c_int,
        input_dilation_n: usize,
        groups: c_int,
        flip: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_argpartition(
        a: *mut mlx_array_t,
        kth: c_int,
        axis: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_contiguous(a: *mut mlx_array_t, out: *mut *mut mlx_array_t) -> c_int;

    pub fn rlx_mlx_op_maxpool2d_backward_metal(
        x: *mut mlx_array_t,
        dy: *mut mlx_array_t,
        n: c_int,
        c: c_int,
        h: c_int,
        w: c_int,
        h_out: c_int,
        w_out: c_int,
        kh: c_int,
        kw: c_int,
        sh: c_int,
        sw: c_int,
        ph: c_int,
        pw: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_maxpool3d_backward_metal(
        x: *mut mlx_array_t,
        dy: *mut mlx_array_t,
        n: c_int,
        c: c_int,
        d: c_int,
        h: c_int,
        w: c_int,
        d_out: c_int,
        h_out: c_int,
        w_out: c_int,
        kd: c_int,
        kh: c_int,
        kw: c_int,
        sd: c_int,
        sh: c_int,
        sw: c_int,
        pd: c_int,
        ph: c_int,
        pw: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_take_along_axis(
        a: *mut mlx_array_t,
        indices: *mut mlx_array_t,
        axis: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_scatter_add_axis(
        a: *mut mlx_array_t,
        indices: *mut mlx_array_t,
        updates: *mut mlx_array_t,
        axis: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_scatter_add(
        a: *mut mlx_array_t,
        indices: *mut mlx_array_t,
        updates: *mut mlx_array_t,
        axis: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_gather_mm(
        a: *mut mlx_array_t,
        b: *mut mlx_array_t,
        idx: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_quantized_matmul(
        x: *mut mlx_array_t,
        w: *mut mlx_array_t,
        scales: *mut mlx_array_t,
        biases_or_null: *mut mlx_array_t,
        transpose: c_int,
        group_size: c_int,
        bits: c_int,
        mode: *const c_char,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_categorical(
        logits: *mut mlx_array_t,
        axis: c_int,
        seed: u64,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_argmax(
        a: *mut mlx_array_t,
        axis: c_int,
        keep_dim: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_slice_strided(
        a: *mut mlx_array_t,
        start: *const c_int,
        stop: *const c_int,
        strides: *const c_int,
        ndim: usize,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_pad(
        a: *mut mlx_array_t,
        low: *const c_int,
        high: *const c_int,
        ndim: usize,
        pad_value: c_float,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_topk_values(
        a: *mut mlx_array_t,
        k: c_int,
        axis: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;

    pub fn rlx_mlx_op_sort(a: *mut mlx_array_t, axis: c_int, out: *mut *mut mlx_array_t) -> c_int;

    pub fn rlx_mlx_compile(
        fn_ptr: LowerFn,
        ud: *mut c_void,
        shapeless: c_int,
        out: *mut *mut mlx_compiled_t,
    ) -> c_int;

    pub fn rlx_mlx_compiled_call(
        compiled: *mut mlx_compiled_t,
        inputs: *const *mut mlx_array_t,
        n_inputs: usize,
        out_outputs: *mut *mut mlx_array_t,
        cap: usize,
        out_n_outputs: *mut usize,
    ) -> c_int;

    pub fn rlx_mlx_compiled_free(compiled: *mut mlx_compiled_t);

    pub fn rlx_mlx_compile_output_cap() -> usize;
    pub fn rlx_mlx_set_compile_output_cap(cap: usize);
    pub fn rlx_mlx_reset_compile_output_cap();

    // ── Distributed (multi-node) ─────────────────────────────────
    pub fn rlx_mlx_dist_is_available(out: *mut c_int) -> c_int;
    pub fn rlx_mlx_dist_init(
        strict: c_int,
        backend: *const c_char,
        out_rank: *mut c_int,
        out_size: *mut c_int,
    ) -> c_int;
    pub fn rlx_mlx_dist_rank(out_rank: *mut c_int) -> c_int;
    pub fn rlx_mlx_dist_size(out_size: *mut c_int) -> c_int;
    pub fn rlx_mlx_dist_all_sum_f32(inp: *const c_float, out: *mut c_float, nelems: usize)
    -> c_int;
    pub fn rlx_mlx_dist_all_gather_f32(
        inp: *const c_float,
        nelems: usize,
        out: *mut c_float,
        out_cap: usize,
    ) -> c_int;
    pub fn rlx_mlx_dist_send_f32(data: *const c_float, nelems: usize, dst: c_int) -> c_int;
    pub fn rlx_mlx_dist_recv_f32(out: *mut c_float, nelems: usize, src: c_int) -> c_int;
    pub fn rlx_mlx_dist_barrier() -> c_int;
    pub fn rlx_mlx_dist_all_sum_array(inp: *mut mlx_array_t, out: *mut *mut mlx_array_t) -> c_int;
    pub fn rlx_mlx_dist_all_reduce_array(
        inp: *mut mlx_array_t,
        kind: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_dist_all_gather_array(
        inp: *mut mlx_array_t,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_dist_reduce_scatter_array(
        inp: *mut mlx_array_t,
        kind: c_int,
        out: *mut *mut mlx_array_t,
    ) -> c_int;
}

#[repr(C)]
pub struct mlx_compiled_t {
    _private: [u8; 0],
}

/// Type of the lowering callback that crosses FFI into C++ for the
/// mlx::compile path. Returns `RLX_MLX_OK` on success, anything else
/// is treated as an error and the C++ wrapper throws.
pub type LowerFn = unsafe extern "C" fn(
    ud: *mut c_void,
    inputs: *const *mut mlx_array_t,
    n_inputs: usize,
    out_outputs: *mut *mut mlx_array_t,
    cap: usize,
    out_n_outputs: *mut usize,
) -> c_int;

#[repr(C)]
pub struct mlx_named_array_list_t {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn rlx_mlx_load_safetensors(
        path: *const c_char,
        out: *mut *mut mlx_named_array_list_t,
    ) -> c_int;
    pub fn rlx_mlx_load_npy(path: *const c_char, out: *mut *mut mlx_array_t) -> c_int;
    pub fn rlx_mlx_named_list_len(list: *const mlx_named_array_list_t) -> usize;
    pub fn rlx_mlx_named_list_get(
        list: *mut mlx_named_array_list_t,
        index: usize,
        out_name: *mut *const c_char,
        out_array: *mut *mut mlx_array_t,
    ) -> c_int;
    pub fn rlx_mlx_named_list_free(list: *mut mlx_named_array_list_t);
}

/// Silence unused-import lint when c_void isn't referenced elsewhere.
#[allow(dead_code)]
const _: *const c_void = std::ptr::null();
