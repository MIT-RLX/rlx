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

//! Verify an ICB-encoded sequence of (gelu → elem_add → elem_mul) produces
//! the same output as the same ops dispatched via the live encoder path.
//!
//! cargo run --example icb_check --release -p rlx-metal

#[cfg(target_os = "macos")]
fn main() {
    use metal::{MTLDispatchType, MTLResourceOptions};
    use rlx_metal::device::metal_device;
    use rlx_metal::icb;
    use rlx_metal::thunk::Thunk;

    let dev = metal_device().expect("metal");
    let n: usize = 256;

    // Layout: [a (n f32), b (n f32), c (n f32), out (n f32)]
    // Steps to apply (in-place on the right slots):
    //   1. gelu_inplace on slot a
    //   2. out = a + b   (BinaryFull Add, lhs=a, rhs=b, dst=out)
    //   3. out = out * c (BinaryFull Mul, lhs=out, rhs=c, dst=out)
    let total = n * 4;
    let buf = dev
        .device
        .new_buffer((total * 4) as u64, MTLResourceOptions::StorageModeShared);
    let off = |slot: usize| slot * n * 4;
    let init = |seed: f32| -> Vec<f32> { (0..n).map(|i| (i as f32 + seed) * 0.001).collect() };
    let a = init(0.0);
    let b = init(100.0);
    let c = init(0.5);
    unsafe {
        let p = buf.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(a.as_ptr(), p, n);
        std::ptr::copy_nonoverlapping(b.as_ptr(), p.add(n), n);
        std::ptr::copy_nonoverlapping(c.as_ptr(), p.add(n * 2), n);
        std::ptr::write_bytes(p.add(n * 3) as *mut u8, 0, n * 4);
    }

    use rlx_metal::thunk::HalfFlag;
    let thunks = vec![
        Thunk::ActivationInPlace {
            data: off(0),
            len: n as u32,
            act: rlx_ir::op::Activation::Gelu,
            dt: HalfFlag::F32,
        },
        Thunk::BinaryFull {
            lhs: off(0),
            rhs: off(1),
            dst: off(3),
            len: n as u32,
            op: rlx_ir::op::BinaryOp::Add,
            dt: HalfFlag::F32,
        },
        Thunk::BinaryFull {
            lhs: off(3),
            rhs: off(2),
            dst: off(3),
            len: n as u32,
            op: rlx_ir::op::BinaryOp::Mul,
            dt: HalfFlag::F32,
        },
    ];

    eprintln!("[check] try_compile");
    let segment = icb::try_compile(&thunks, &buf, &dev.device).expect("compiles");
    eprintln!("[check] ICB built: {} commands", segment.command_count);

    // Run the ICB.
    eprintln!("[check] new cmd_buffer");
    let cb = dev.queue.new_command_buffer();
    eprintln!("[check] new compute encoder");
    let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
    eprintln!("[check] execute_on");
    segment.execute_on(enc, &buf);
    eprintln!("[check] end_encoding");
    enc.end_encoding();
    eprintln!("[check] commit");
    cb.commit();
    eprintln!("[check] wait");
    cb.wait_until_completed();
    eprintln!("[check] done");

    // Compute reference on CPU.
    let gelu = |x: f32| {
        let c = 0.797_884_6_f32;
        0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
    };
    let mut expected = vec![0f32; n];
    for i in 0..n {
        let g = gelu(a[i]);
        expected[i] = (g + b[i]) * c[i];
    }

    let got: &[f32] = unsafe {
        let p = (buf.contents() as *const u8).add(off(3)) as *const f32;
        std::slice::from_raw_parts(p, n)
    };
    let max_err = expected
        .iter()
        .zip(got)
        .map(|(e, g)| (e - g).abs())
        .fold(0f32, f32::max);
    println!("expected[..4]: {:?}", &expected[..4]);
    println!("got[..4]:      {:?}", &got[..4]);
    println!("max_err: {:.3e}", max_err);
    assert!(max_err < 1e-4, "ICB output mismatch");
    println!("✓ ICB-encoded gelu → add → mul matches CPU reference");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("requires macOS");
}
