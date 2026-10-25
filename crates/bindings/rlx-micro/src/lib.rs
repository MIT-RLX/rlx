// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The C ABI of rlx's microcontroller kernels ([`rlx_cortexm`]) — for C and
//! C++ firmware on bare-metal Cortex-M (and the same code on a host). See
//! `include/rlx_micro.h` for the contract.
//!
//! `no_std` on bare-metal targets, no allocation anywhere. Every entry point
//! checks the buffer sizes against the shape and returns a status rather than
//! reaching a kernel's assertion: a firmware has nowhere to report a panic.

#![cfg_attr(target_os = "none", no_std)]
// Each entry point checks every pointer and length before forming a slice.
#![allow(clippy::missing_safety_doc)]

use core::ffi::c_char;
use core::slice;

use rlx_cortexm::conv2d::{Conv2dParams, conv2d_i8};
use rlx_cortexm::dense::dense_i8;
use rlx_cortexm::maxpool::{MaxPool2dParams, maxpool2d_i8};
use rlx_cortexm::quant::{QParams, dequant, packed_byte_len, quant};
use rlx_cortexm::{argmax::argmax_i8, model, model_weights, relu::relu_i8};

pub const RLX_MICRO_OK: i32 = 0;
pub const RLX_MICRO_ERR_NULL: i32 = -1;
pub const RLX_MICRO_ERR_SIZE: i32 = -2;
pub const RLX_MICRO_ERR_PARAM: i32 = -3;

/// `RLX_MICRO_MNIST_INPUT_LEN` / `RLX_MICRO_MNIST_SCRATCH_LEN` in the header.
pub const MNIST_INPUT_LEN: usize = model::INPUT_LEN;
pub const MNIST_SCRATCH_LEN: usize = model::SCRATCH_LEN;

#[repr(C)]
pub struct RlxMicroConv2d {
    pub h_in: usize,
    pub w_in: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub kh: usize,
    pub kw: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub x_zp: i32,
    pub w_zp: i32,
    pub out_zp: i32,
    pub mult: *const f32,
    pub weight_bits: u8,
}

#[repr(C)]
pub struct RlxMicroPool2d {
    pub h_in: usize,
    pub w_in: usize,
    pub c: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride_h: usize,
    pub stride_w: usize,
}

fn bits_ok(bits: u8) -> bool {
    matches!(bits, 8 | 4 | 2)
}

/// `(h_out, w_out)` of a conv, or `None` for invalid geometry.
fn conv_dims(p: &RlxMicroConv2d) -> Option<(usize, usize)> {
    if p.stride_h == 0 || p.stride_w == 0 || p.kh == 0 || p.kw == 0 {
        return None;
    }
    let (hp, wp) = (p.h_in + 2 * p.pad_h, p.w_in + 2 * p.pad_w);
    if p.kh > hp || p.kw > wp {
        return None;
    }
    Some(((hp - p.kh) / p.stride_h + 1, (wp - p.kw) / p.stride_w + 1))
}

fn pool_dims(p: &RlxMicroPool2d) -> Option<(usize, usize)> {
    if p.stride_h == 0
        || p.stride_w == 0
        || p.kh == 0
        || p.kw == 0
        || p.kh > p.h_in
        || p.kw > p.w_in
    {
        return None;
    }
    Some((
        (p.h_in - p.kh) / p.stride_h + 1,
        (p.w_in - p.kw) / p.stride_w + 1,
    ))
}

#[unsafe(no_mangle)]
pub extern "C" fn rlx_micro_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_micro_conv2d_out_len(p: *const RlxMicroConv2d) -> usize {
    let Some(p) = (unsafe { p.as_ref() }) else {
        return 0;
    };
    conv_dims(p).map_or(0, |(h, w)| h * w * p.c_out)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_micro_pool2d_out_len(p: *const RlxMicroPool2d) -> usize {
    let Some(p) = (unsafe { p.as_ref() }) else {
        return 0;
    };
    pool_dims(p).map_or(0, |(h, w)| h * w * p.c)
}

#[unsafe(no_mangle)]
pub extern "C" fn rlx_micro_packed_len(n: usize, bits: u8) -> usize {
    if bits_ok(bits) {
        packed_byte_len(n, bits)
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_micro_conv2d_i8(
    p: *const RlxMicroConv2d,
    x: *const i8,
    x_len: usize,
    w: *const i8,
    w_len: usize,
    bias: *const i32,
    out: *mut i8,
    out_len: usize,
) -> i32 {
    let Some(p) = (unsafe { p.as_ref() }) else {
        return RLX_MICRO_ERR_NULL;
    };
    if x.is_null() || w.is_null() || out.is_null() || p.mult.is_null() {
        return RLX_MICRO_ERR_NULL;
    }
    if !bits_ok(p.weight_bits) || p.c_in == 0 || p.c_out == 0 {
        return RLX_MICRO_ERR_PARAM;
    }
    let Some((h_out, w_out)) = conv_dims(p) else {
        return RLX_MICRO_ERR_PARAM;
    };
    if x_len != p.h_in * p.w_in * p.c_in
        || w_len != packed_byte_len(p.c_out * p.kh * p.kw * p.c_in, p.weight_bits)
        || out_len != h_out * w_out * p.c_out
    {
        return RLX_MICRO_ERR_SIZE;
    }
    // SAFETY: every pointer is non-null and its length checked against the shape.
    let (x, w, out, mult) = unsafe {
        (
            slice::from_raw_parts(x, x_len),
            slice::from_raw_parts(w, w_len),
            slice::from_raw_parts_mut(out, out_len),
            slice::from_raw_parts(p.mult, p.c_out),
        )
    };
    let bias = (!bias.is_null()).then(|| unsafe { slice::from_raw_parts(bias, p.c_out) });
    let kp = Conv2dParams {
        h_in: p.h_in,
        w_in: p.w_in,
        c_in: p.c_in,
        c_out: p.c_out,
        kh: p.kh,
        kw: p.kw,
        pad_h: p.pad_h,
        pad_w: p.pad_w,
        stride_h: p.stride_h,
        stride_w: p.stride_w,
        x_zp: p.x_zp,
        w_zp: p.w_zp,
        out_zp: p.out_zp,
        mult,
        weight_bits: p.weight_bits,
    };
    conv2d_i8(&kp, x, w, bias, out);
    RLX_MICRO_OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_micro_dense_i8(
    x: *const i8,
    in_features: usize,
    w: *const i8,
    w_len: usize,
    bias: *const i32,
    x_zp: i32,
    w_zp: i32,
    out_zp: i32,
    mult: *const f32,
    weight_bits: u8,
    out: *mut i8,
    out_features: usize,
) -> i32 {
    if x.is_null() || w.is_null() || out.is_null() || mult.is_null() {
        return RLX_MICRO_ERR_NULL;
    }
    if !bits_ok(weight_bits) || in_features == 0 || out_features == 0 {
        return RLX_MICRO_ERR_PARAM;
    }
    if w_len != packed_byte_len(in_features * out_features, weight_bits) {
        return RLX_MICRO_ERR_SIZE;
    }
    // SAFETY: non-null, lengths checked.
    let (x, w, out, mult) = unsafe {
        (
            slice::from_raw_parts(x, in_features),
            slice::from_raw_parts(w, w_len),
            slice::from_raw_parts_mut(out, out_features),
            slice::from_raw_parts(mult, out_features),
        )
    };
    let bias = (!bias.is_null()).then(|| unsafe { slice::from_raw_parts(bias, out_features) });
    dense_i8(x, w, bias, x_zp, w_zp, out_zp, mult, weight_bits, out);
    RLX_MICRO_OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_micro_maxpool2d_i8(
    p: *const RlxMicroPool2d,
    x: *const i8,
    x_len: usize,
    out: *mut i8,
    out_len: usize,
) -> i32 {
    let Some(p) = (unsafe { p.as_ref() }) else {
        return RLX_MICRO_ERR_NULL;
    };
    if x.is_null() || out.is_null() {
        return RLX_MICRO_ERR_NULL;
    }
    let Some((h_out, w_out)) = pool_dims(p) else {
        return RLX_MICRO_ERR_PARAM;
    };
    if p.c == 0 {
        return RLX_MICRO_ERR_PARAM;
    }
    if x_len != p.h_in * p.w_in * p.c || out_len != h_out * w_out * p.c {
        return RLX_MICRO_ERR_SIZE;
    }
    // SAFETY: non-null, lengths checked.
    let (x, out) = unsafe {
        (
            slice::from_raw_parts(x, x_len),
            slice::from_raw_parts_mut(out, out_len),
        )
    };
    let kp = MaxPool2dParams {
        h_in: p.h_in,
        w_in: p.w_in,
        c: p.c,
        kh: p.kh,
        kw: p.kw,
        stride_h: p.stride_h,
        stride_w: p.stride_w,
    };
    maxpool2d_i8(&kp, x, out);
    RLX_MICRO_OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_micro_relu_i8(x: *mut i8, n: usize, zero_point: i32) -> i32 {
    if x.is_null() {
        return if n == 0 {
            RLX_MICRO_OK
        } else {
            RLX_MICRO_ERR_NULL
        };
    }
    relu_i8(unsafe { slice::from_raw_parts_mut(x, n) }, zero_point);
    RLX_MICRO_OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_micro_argmax_i8(x: *const i8, n: usize) -> i32 {
    if x.is_null() || n == 0 {
        return -1;
    }
    argmax_i8(unsafe { slice::from_raw_parts(x, n) }) as i32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_micro_quantize(
    x: *const f32,
    n: usize,
    scale: f32,
    zero_point: i32,
    out: *mut i8,
) -> i32 {
    if n == 0 {
        return RLX_MICRO_OK;
    }
    if x.is_null() || out.is_null() {
        return RLX_MICRO_ERR_NULL;
    }
    if scale.is_nan() || scale <= 0.0 {
        return RLX_MICRO_ERR_PARAM;
    }
    let q = QParams::new(scale, zero_point);
    let (x, out) = unsafe {
        (
            slice::from_raw_parts(x, n),
            slice::from_raw_parts_mut(out, n),
        )
    };
    for (o, &v) in out.iter_mut().zip(x) {
        *o = quant(v, q);
    }
    RLX_MICRO_OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_micro_dequantize(
    x: *const i8,
    n: usize,
    scale: f32,
    zero_point: i32,
    out: *mut f32,
) -> i32 {
    if n == 0 {
        return RLX_MICRO_OK;
    }
    if x.is_null() || out.is_null() {
        return RLX_MICRO_ERR_NULL;
    }
    let q = QParams::new(scale, zero_point);
    let (x, out) = unsafe {
        (
            slice::from_raw_parts(x, n),
            slice::from_raw_parts_mut(out, n),
        )
    };
    for (o, &v) in out.iter_mut().zip(x) {
        *o = dequant(v, q);
    }
    RLX_MICRO_OK
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_micro_mnist_infer(
    image: *const i8,
    image_len: usize,
    scratch_a: *mut i8,
    scratch_b: *mut i8,
    scratch_len: usize,
) -> i32 {
    if image.is_null() || scratch_a.is_null() || scratch_b.is_null() {
        return RLX_MICRO_ERR_NULL;
    }
    if image_len != MNIST_INPUT_LEN || scratch_len < MNIST_SCRATCH_LEN || scratch_a == scratch_b {
        return RLX_MICRO_ERR_SIZE;
    }
    // SAFETY: non-null, distinct, lengths checked.
    let (image, a, b) = unsafe {
        (
            slice::from_raw_parts(image, image_len),
            slice::from_raw_parts_mut(scratch_a, scratch_len),
            slice::from_raw_parts_mut(scratch_b, scratch_len),
        )
    };
    model::infer(image, a, b) as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn rlx_micro_mnist_input_scale() -> f32 {
    model_weights::X_SCALE
}

#[unsafe(no_mangle)]
pub extern "C" fn rlx_micro_mnist_test_image() -> *const i8 {
    model_weights::TEST_IMAGE.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn rlx_micro_mnist_test_label() -> u8 {
    model_weights::TEST_LABEL
}

/// A firmware has nowhere to report a panic: halt. (Every entry point checks
/// its arguments first, so reaching this is a bug in rlx-micro.)
#[cfg(all(target_os = "none", feature = "panic-handler", not(test)))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bundled_test_image_classifies_as_its_label() {
        let mut a = vec![0i8; MNIST_SCRATCH_LEN];
        let mut b = vec![0i8; MNIST_SCRATCH_LEN];
        let d = unsafe {
            rlx_micro_mnist_infer(
                rlx_micro_mnist_test_image(),
                MNIST_INPUT_LEN,
                a.as_mut_ptr(),
                b.as_mut_ptr(),
                MNIST_SCRATCH_LEN,
            )
        };
        assert_eq!(d, rlx_micro_mnist_test_label() as i32);
        let short = unsafe {
            rlx_micro_mnist_infer(
                rlx_micro_mnist_test_image(),
                10,
                a.as_mut_ptr(),
                b.as_mut_ptr(),
                8,
            )
        };
        assert_eq!(short, RLX_MICRO_ERR_SIZE);
    }

    #[test]
    fn the_header_constants_match_the_model() {
        let h = include_str!("../include/rlx_micro.h");
        assert!(h.contains(&format!("RLX_MICRO_MNIST_INPUT_LEN {MNIST_INPUT_LEN}")));
        assert!(h.contains(&format!("RLX_MICRO_MNIST_SCRATCH_LEN {MNIST_SCRATCH_LEN}")));
    }

    #[test]
    fn bad_shapes_are_refused_not_trapped() {
        let mult = [1.0f32; 2];
        let p = RlxMicroConv2d {
            h_in: 2,
            w_in: 2,
            c_in: 1,
            c_out: 2,
            kh: 3,
            kw: 3,
            pad_h: 0,
            pad_w: 0,
            stride_h: 1,
            stride_w: 1,
            x_zp: 0,
            w_zp: 0,
            out_zp: 0,
            mult: mult.as_ptr(),
            weight_bits: 8,
        };
        let x = [0i8; 4];
        let w = [0i8; 18];
        let mut out = [0i8; 2];
        // a 3×3 kernel over a 2×2 input
        let rc = unsafe {
            rlx_micro_conv2d_i8(
                &p,
                x.as_ptr(),
                4,
                w.as_ptr(),
                18,
                core::ptr::null(),
                out.as_mut_ptr(),
                2,
            )
        };
        assert_eq!(rc, RLX_MICRO_ERR_PARAM);
        assert_eq!(unsafe { rlx_micro_conv2d_out_len(&p) }, 0);
    }
}
