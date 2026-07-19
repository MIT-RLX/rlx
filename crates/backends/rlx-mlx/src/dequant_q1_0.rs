// RLX — versatile ML compiler + runtime.
//! On-device Q1_0 dequant for MLX via `fast::metal_kernel`.
//!
//! Host-side `dequant_q1_0` → f32 caches blow up ~28× (Bonsai-27B: 3.8 GiB
//! packed → ~108 GiB). This path keeps packed U8 resident and expands each
//! weight only for the immediate matmul.

#![cfg(rlx_mlx_host)]

use std::ffi::CString;

use crate::array::{Array, MlxError, check};
use crate::ffi::{self, MlxDtype, mlx_array_t};

const KERNEL_MSL: &str = include_str!("../cpp/kernels/dequant_q1_0.metal");
const KERNEL_NAME: &str = "dequant_q1_0";
const MV_KERNEL_MSL: &str = include_str!("../cpp/kernels/q1_0_matmul_mv.metal");
const MV_KERNEL_NAME: &str = "q1_0_matmul_mv";
/// Injected ahead of the body — nested `inline` fns are illegal in the
/// MLX-pasted kernel body (see batched_lu_solve pattern with `NMAX`).
const KERNEL_HEADER: &str = r#"
inline float q1_read_f16(const device uchar* b, uint off) {
    ushort bits = ushort(b[off]) | (ushort(b[off + 1u]) << 8u);
    uint sign = (uint(bits) >> 15u) & 1u;
    uint exp = (uint(bits) >> 10u) & 0x1Fu;
    uint mant = uint(bits) & 0x3FFu;
    float v;
    if (exp == 0u) {
        v = float(mant) / 1024.0f * exp2(-14.0f);
    } else if (exp == 31u) {
        v = (mant == 0u) ? INFINITY : 0.0f;
    } else {
        v = (1.0f + float(mant) / 1024.0f) * exp2(float(int(exp) - 15));
    }
    return (sign != 0u) ? -v : v;
}
"#;

const Q1_BLOCK_ELEMS: usize = 128;
const Q1_BLOCK_BYTES: usize = 18;

/// Dequant packed Q1_0 `w` (U8 bytes) → f32 `[n, k]` on the Apple GPU.
///
/// Layout matches GGUF: row-major `[n, k]` after unpack (same as host
/// `rlx_gguf::q1_dequant::dequant_q1_0`).
pub fn dequant_q1_0_ondevice(w_u8: &Array, k: usize, n: usize) -> Result<Array, MlxError> {
    let elems = k
        .checked_mul(n)
        .ok_or_else(|| MlxError("Q1_0: k*n overflow".into()))?;
    if elems % Q1_BLOCK_ELEMS != 0 {
        return Err(MlxError(format!(
            "Q1_0 ondevice: elems={elems} not divisible by {Q1_BLOCK_ELEMS}"
        )));
    }
    let num_blocks = elems / Q1_BLOCK_ELEMS;
    let expect_bytes = num_blocks * Q1_BLOCK_BYTES;
    let w_nelem = w_u8.num_elements()?;
    if w_nelem < expect_bytes {
        return Err(MlxError(format!(
            "Q1_0 ondevice: need {expect_bytes} packed bytes, got {w_nelem}"
        )));
    }

    let name_c = CString::new(KERNEL_NAME).unwrap();
    let source_c = CString::new(KERNEL_MSL).unwrap();
    let header_c = CString::new(KERNEL_HEADER).unwrap();
    let in_w_c = CString::new("w").unwrap();
    let out_c = CString::new("out").unwrap();

    let in_name_ptrs: [*const std::os::raw::c_char; 1] = [in_w_c.as_ptr()];
    let in_array_ptrs: [*mut mlx_array_t; 1] = [w_u8.ptr];
    // Flat `[elems]` then reshape to `[n, k]` so transpose → `[k, n]`.
    let out_shape_i32: [std::os::raw::c_int; 1] = [elems as std::os::raw::c_int];

    let mut out_handle: *mut mlx_array_t = std::ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_metal_kernel_dispatch(
            name_c.as_ptr(),
            source_c.as_ptr(),
            header_c.as_ptr(),
            in_name_ptrs.as_ptr(),
            /*n_inputs=*/ 1,
            out_c.as_ptr(),
            in_array_ptrs.as_ptr(),
            out_shape_i32.as_ptr(),
            /*output_rank=*/ 1,
            MlxDtype::F32,
            /*grid*/ num_blocks as std::os::raw::c_int,
            1,
            1,
            /*tg*/ 1,
            1,
            1,
            &mut out_handle,
        )
    };
    check(rc)?;
    let flat = Array::from_raw(out_handle);
    crate::ops::reshape(&flat, &[n as i32, k as i32])
}

/// Fused Q1_0 decode GEMV: `out[n] = x[k] @ dequant(w)[n, k]^T`, reading the
/// packed weight directly (no f32 blow-up). `x` is `[1, k]` or `[k]`; `w` is
/// the packed U8 `[n, k]`. Returns `[1, n]` so it slots in for `x @ w^T`.
/// One thread per output column (grid = n).
pub fn q1_0_matmul_mv_ondevice(
    w_u8: &Array,
    x: &Array,
    k: usize,
    n: usize,
) -> Result<Array, MlxError> {
    let elems = k
        .checked_mul(n)
        .ok_or_else(|| MlxError("Q1_0 mv: k*n overflow".into()))?;
    if elems % Q1_BLOCK_ELEMS != 0 {
        return Err(MlxError(format!(
            "Q1_0 mv: elems={elems} not divisible by {Q1_BLOCK_ELEMS}"
        )));
    }
    // Flatten x to [k] so the kernel reads `x_shape[0] == k` and `x[i]` is the
    // i-th contraction element regardless of the [1, k] vs [k] input rank.
    let x_flat = crate::ops::reshape(x, &[k as i32])?;

    let name_c = CString::new(MV_KERNEL_NAME).unwrap();
    let source_c = CString::new(MV_KERNEL_MSL).unwrap();
    let header_c = CString::new(KERNEL_HEADER).unwrap();
    let in_w_c = CString::new("w").unwrap();
    let in_x_c = CString::new("x").unwrap();
    let out_c = CString::new("out").unwrap();

    let in_name_ptrs: [*const std::os::raw::c_char; 2] = [in_w_c.as_ptr(), in_x_c.as_ptr()];
    let in_array_ptrs: [*mut mlx_array_t; 2] = [w_u8.ptr, x_flat.ptr];
    let out_shape_i32: [std::os::raw::c_int; 1] = [n as std::os::raw::c_int];
    let tg: std::os::raw::c_int = if n >= 256 {
        256
    } else {
        n.max(1) as std::os::raw::c_int
    };

    let mut out_handle: *mut mlx_array_t = std::ptr::null_mut();
    let rc = unsafe {
        ffi::rlx_mlx_op_metal_kernel_dispatch(
            name_c.as_ptr(),
            source_c.as_ptr(),
            header_c.as_ptr(),
            in_name_ptrs.as_ptr(),
            /*n_inputs=*/ 2,
            out_c.as_ptr(),
            in_array_ptrs.as_ptr(),
            out_shape_i32.as_ptr(),
            /*output_rank=*/ 1,
            MlxDtype::F32,
            /*grid*/ n as std::os::raw::c_int,
            1,
            1,
            /*tg*/ tg,
            1,
            1,
            &mut out_handle,
        )
    };
    check(rc)?;
    let flat = Array::from_raw(out_handle);
    crate::ops::reshape(&flat, &[1, n as i32])
}
