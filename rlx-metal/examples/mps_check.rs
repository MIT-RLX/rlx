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

//! basic test: verify MPSMatrixMultiplication produces correct output.
//!
//! cargo run --example mps_check --release -p rlx-metal

#[cfg(target_os = "macos")]
fn main() {
    use metal::MTLResourceOptions;
    use rlx_metal::device::metal_device;
    use rlx_metal::mps_blas::{encode_mps_sgemm, mps_supports_matmul};

    if !mps_supports_matmul() {
        eprintln!("MPS not available");
        return;
    }
    let dev = metal_device().expect("metal device");

    // C = A (4x3) @ B (3x2) → C (4x2)
    let m = 4;
    let k = 3;
    let n = 2;
    let a = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let b = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    // Expected: row[i] · column[j]
    // row 0 = [1,2,3] · col0=[1,0,1] = 4 ; · col1=[0,1,1] = 5
    // row 1 = [4,5,6] · col0 = 10; · col1 = 11
    // row 2 = [7,8,9] · col0 = 16; · col1 = 17
    // row 3 = [10,11,12] · col0 = 22; · col1 = 23
    let expected = [4.0f32, 5.0, 10.0, 11.0, 16.0, 17.0, 22.0, 23.0];

    let total_floats = m * k + k * n + m * n;
    let buffer = dev.device.new_buffer(
        (total_floats * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let a_off = 0;
    let b_off = m * k * 4;
    let c_off = (m * k + k * n) * 4;
    unsafe {
        let p = buffer.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(a.as_ptr(), p, m * k);
        std::ptr::copy_nonoverlapping(b.as_ptr(), p.add(m * k), k * n);
        std::ptr::write_bytes(p.add(m * k + k * n) as *mut u8, 0, m * n * 4);
    }

    let cb = dev.queue.new_command_buffer();
    encode_mps_sgemm(cb, &buffer, a_off, b_off, c_off, m, k, n);
    cb.commit();
    cb.wait_until_completed();

    let c: &[f32] = unsafe {
        let p = (buffer.contents() as *const u8).add(c_off) as *const f32;
        std::slice::from_raw_parts(p, m * n)
    };
    let max_err = expected
        .iter()
        .zip(c.iter())
        .map(|(e, g)| (e - g).abs())
        .fold(0f32, f32::max);
    println!("expected: {:?}", expected);
    println!("got:      {:?}", c);
    println!("max_err:  {:.2e}", max_err);
    assert!(max_err < 1e-4, "MPS output doesn't match expected");
    println!("✓ MPSMatrixMultiplication works");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("requires macOS");
}
