// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! ICB-encoded thunks must match the CPU reference.
//!
//! The ICB path is opt-in (`RLX_USE_ICB`) and was previously "deferred" with an
//! open fault, so nothing in the suite covered it — which is exactly how it
//! drifted out of sync with the kernels it encodes. `icb.rs` binds buffers by
//! index against MSL signatures that live in `kernels.rs`, and those signatures
//! moved (several kernels went from `device float*` already offset to an arena
//! base plus explicit `ulong` byte offsets). Nothing catches that at compile
//! time: an index that no longer lines up leaves the kernel's `len` unbound, it
//! reads 0, every thread hits `if (gid >= len) return`, and the command buffer
//! completes cleanly having written nothing.
//!
//! So this asserts on *values*, not on "it ran".

#![cfg(target_os = "macos")]

use rlx_metal::device::metal_device;
use rlx_metal::icb;
use rlx_metal::mtl::{MTLDispatchType, MTLResourceOptions};
use rlx_metal::thunk::{HalfFlag, Thunk};

const N: usize = 1024;

/// Lay `a`, `b`, `c`, `dst` out back to back in one arena buffer.
fn off(i: usize) -> usize {
    i * N * 4
}

fn gelu(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// `gelu(a)` in place, then `+ b`, then `* c` — three ICB commands covering
/// both encoder ABIs: the generated arena-base form (`gelu_inplace`) and the
/// arena-base binary form (`elem_add` / `elem_mul`).
#[test]
fn icb_gelu_add_mul_matches_cpu() {
    let Some(dev) = metal_device() else {
        eprintln!("no Metal device; skipping");
        return;
    };

    let buf = dev
        .device
        .new_buffer((off(4)) as u64, MTLResourceOptions::StorageModeShared);

    let a: Vec<f32> = (0..N).map(|i| (i as f32) * 1e-3 - 0.5).collect();
    let b: Vec<f32> = (0..N).map(|i| (i as f32) * 2e-4).collect();
    let c: Vec<f32> = (0..N).map(|i| 1.0 + (i as f32) * 1e-4).collect();
    unsafe {
        let p = buf.contents() as *mut u8;
        for (slot, src) in [&a, &b, &c].iter().enumerate() {
            std::ptr::copy_nonoverlapping(src.as_ptr(), p.add(off(slot)) as *mut f32, N);
        }
        std::ptr::write_bytes(p.add(off(3)), 0, N * 4);
    }

    let thunks = vec![
        Thunk::ActivationInPlace {
            data: off(0),
            len: N as u32,
            act: rlx_ir::op::Activation::Gelu,
            dt: HalfFlag::F32,
        },
        Thunk::BinaryFull {
            lhs: off(0),
            rhs: off(1),
            dst: off(3),
            len: N as u32,
            op: rlx_ir::op::BinaryOp::Add,
            dt: HalfFlag::F32,
        },
        Thunk::BinaryFull {
            lhs: off(3),
            rhs: off(2),
            dst: off(3),
            len: N as u32,
            op: rlx_ir::op::BinaryOp::Mul,
            dt: HalfFlag::F32,
        },
    ];

    let segment = icb::try_compile(&thunks, &buf, &dev.device).expect("ICB compiles");
    assert_eq!(segment.command_count, 3);

    let cb = dev.queue.new_command_buffer();
    let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
    segment.execute_on(enc, &buf);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    assert!(
        cb.error_string().is_none(),
        "ICB command buffer failed: {:?}",
        cb.error_string()
    );

    let got: &[f32] = unsafe {
        std::slice::from_raw_parts((buf.contents() as *const u8).add(off(3)) as *const f32, N)
    };

    // An all-zero output is the signature failure of an index mismatch, and it
    // would otherwise pass a loose relative-error check on a near-zero row.
    assert!(
        got.iter().any(|&v| v != 0.0),
        "ICB wrote nothing — kernel buffer indices are out of sync with kernels.rs"
    );

    let mut max_err = 0f32;
    for i in 0..N {
        let want = (gelu(a[i]) + b[i]) * c[i];
        max_err = max_err.max((want - got[i]).abs());
    }
    assert!(max_err < 1e-4, "ICB vs CPU max_err {max_err:e}");
}

/// `Copy` and `BiasAdd`, the other two encoders whose bindings were corrected
/// by reading kernel signatures rather than by test.
///
/// `copy_f32` is the arena-base form (`arena, ulong src_off, ulong dst_off,
/// uint len`) and `bias_add` the offset-pointer form (`float* data,
/// float* bias, uint m, uint n`) — one of each convention, which is the pairing
/// that broke the activation arm.
#[test]
fn icb_copy_and_bias_add_match_cpu() {
    let Some(dev) = metal_device() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (rows, cols) = (8usize, 16usize);
    let n = rows * cols;
    // slots: 0 = src, 1 = dst(copy target, then bias_add in place), 2 = bias
    let slot = |i: usize| i * n * 4;
    let buf = dev
        .device
        .new_buffer((slot(3)) as u64, MTLResourceOptions::StorageModeShared);

    let src: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 3.0).collect();
    let bias: Vec<f32> = (0..cols).map(|j| (j as f32) * 0.5).collect();
    unsafe {
        let p = buf.contents() as *mut u8;
        std::ptr::copy_nonoverlapping(src.as_ptr(), p.add(slot(0)) as *mut f32, n);
        std::ptr::write_bytes(p.add(slot(1)), 0, n * 4);
        std::ptr::write_bytes(p.add(slot(2)), 0, n * 4);
        std::ptr::copy_nonoverlapping(bias.as_ptr(), p.add(slot(2)) as *mut f32, cols);
    }

    let thunks = vec![
        Thunk::Copy {
            src: slot(0),
            dst: slot(1),
            len: n as u32,
            dt: HalfFlag::F32,
        },
        Thunk::BiasAdd {
            src: slot(1),
            bias: slot(2),
            dst: slot(1),
            m: rows as u32,
            n: cols as u32,
            dt: HalfFlag::F32,
        },
    ];

    let segment = icb::try_compile(&thunks, &buf, &dev.device).expect("ICB compiles");
    assert_eq!(segment.command_count, 2);

    let cb = dev.queue.new_command_buffer();
    let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
    segment.execute_on(enc, &buf);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    assert!(cb.error_string().is_none(), "{:?}", cb.error_string());

    let got: &[f32] = unsafe {
        std::slice::from_raw_parts((buf.contents() as *const u8).add(slot(1)) as *const f32, n)
    };
    assert!(
        got.iter().any(|&v| v != 0.0),
        "ICB wrote nothing — buffer indices out of sync with kernels.rs"
    );
    let mut max_err = 0f32;
    for r in 0..rows {
        for c in 0..cols {
            let want = src[r * cols + c] + bias[c];
            max_err = max_err.max((want - got[r * cols + c]).abs());
        }
    }
    assert!(max_err < 1e-5, "copy+bias_add vs CPU max_err {max_err:e}");
}
