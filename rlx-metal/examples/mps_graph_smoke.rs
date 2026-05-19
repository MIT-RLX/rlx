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

//! Smoke test: build an MPSGraph for `out = (a @ b) + bias`, compile, run.
//!
//! Verifies bit-correct output vs expected.
//!
//! cargo run --example mps_graph_smoke --release -p rlx-metal

#[cfg(target_os = "macos")]
fn main() {
    use metal::MTLResourceOptions;
    use rlx_metal::device::metal_device;
    use rlx_metal::mps_graph::{MpsGraph, mps_graph_supported};

    if !mps_graph_supported() {
        eprintln!("MPSGraph not available (needs macOS 11+)");
        return;
    }

    let dev = metal_device().expect("metal device");

    // Shapes: a [2,3] @ b [3,2] = c [2,2]; bias [2]
    let m = 2;
    let k = 3;
    let n = 2;
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
    let b = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3,2]
    let bias = [10.0f32, 100.0]; // [2]
    // Expected: row0·col0 = 1+0+3 = 4 ; row0·col1 = 0+2+3 = 5
    //           row1·col0 = 4+0+6 = 10; row1·col1 = 0+5+6 = 11
    // Plus bias broadcasting: [[14, 105], [20, 111]]
    let expected = [14.0f32, 105.0, 20.0, 111.0];

    // Allocate a single shared buffer that we'll partition into a/b/bias/out.
    let total_floats = m * k + k * n + n + m * n;
    let buffer = dev.device.new_buffer(
        (total_floats * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let a_off = 0;
    let b_off = m * k * 4;
    let bias_off = (m * k + k * n) * 4;
    let out_off = (m * k + k * n + n) * 4;
    unsafe {
        let p = buffer.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(a.as_ptr(), p, m * k);
        std::ptr::copy_nonoverlapping(b.as_ptr(), p.add(m * k), k * n);
        std::ptr::copy_nonoverlapping(bias.as_ptr(), p.add(m * k + k * n), n);
        std::ptr::write_bytes(p.add(m * k + k * n + n) as *mut u8, 0, m * n * 4);
    }

    // Build symbolic graph: c = (a @ b) + bias
    const F32_DT: u32 = 0x10000000 | 32;
    eprintln!("[smoke] new graph");
    let g = MpsGraph::new();
    eprintln!("[smoke] placeholder a");
    let a_t = g.placeholder(&[m, k], F32_DT, "a");
    eprintln!("[smoke] placeholder b");
    let b_t = g.placeholder(&[k, n], F32_DT, "b");
    eprintln!("[smoke] placeholder bias");
    let bias_t = g.placeholder(&[n], F32_DT, "bias");
    eprintln!("[smoke] matmul");
    let mm = g.matmul(&a_t, &b_t);
    eprintln!("[smoke] add");
    let out_t = g.add(&mm, &bias_t);
    eprintln!("[smoke] graph built");

    // Run (JIT path: graph compiles + runs on first call, caches internally).
    g.run_jit(
        &dev.queue,
        &[&a_t, &b_t, &bias_t],
        &[&buffer, &buffer, &buffer],
        &[a_off, b_off, bias_off],
        &[vec![m, k], vec![k, n], vec![n]],
        &[F32_DT, F32_DT, F32_DT],
        &[&out_t],
        &[&buffer],
        &[out_off],
        &[vec![m, n]],
        &[F32_DT],
    );

    let c: &[f32] = unsafe {
        let p = (buffer.contents() as *const u8).add(out_off) as *const f32;
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
    if max_err < 1e-3 {
        println!("✓ MPSGraph compile + run + read-back works");
    } else {
        eprintln!("✗ output diverges — investigate before integrating");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("requires macOS");
}
