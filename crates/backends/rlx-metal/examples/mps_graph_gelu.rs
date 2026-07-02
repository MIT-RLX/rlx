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

//! Verify MPSGraph's analytic GELU matches our CPU GELU.
//!
//! cargo run --example mps_graph_gelu --release -p rlx-metal

#[cfg(target_os = "macos")]
fn main() {
    use metal::MTLResourceOptions;
    use rlx_metal::device::metal_device;
    use rlx_metal::mps_graph::{MpsGraph, mps_graph_supported};

    if !mps_graph_supported() {
        eprintln!("MPSGraph not available");
        return;
    }
    let dev = metal_device().expect("metal device");

    let inputs: Vec<f32> = (-30..30).map(|i| i as f32 * 0.1).collect();
    let n = inputs.len();
    // Reference GELU (PyTorch approximate): same constants we use in the bridge
    let cpu_gelu = |x: f32| {
        let c = 0.797_884_6_f32; // √(2/π)
        0.5 * x * (1.0 + ((c * (x + 0.044715 * x * x * x)).tanh()))
    };
    let expected: Vec<f32> = inputs.iter().map(|&x| cpu_gelu(x)).collect();

    let buffer = dev
        .device
        .new_buffer((n * 2 * 4) as u64, MTLResourceOptions::StorageModeShared);
    let in_off = 0;
    let out_off = n * 4;
    unsafe {
        let p = buffer.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(inputs.as_ptr(), p, n);
        std::ptr::write_bytes(p.add(n) as *mut u8, 0, n * 4);
    }

    const F32: u32 = 0x10000000 | 32;
    let g = MpsGraph::new();
    let x = g.placeholder(&[n], F32, "x");
    let y = g.gelu(&x);

    g.run_jit(
        &dev.queue,
        &[&x],
        &[&buffer],
        &[in_off],
        &[vec![n]],
        &[F32],
        &[&y],
        &[&buffer],
        &[out_off],
        &[vec![n]],
        &[F32],
    );

    let out: &[f32] = unsafe {
        let p = (buffer.contents() as *const u8).add(out_off) as *const f32;
        std::slice::from_raw_parts(p, n)
    };
    let max_err = expected
        .iter()
        .zip(out.iter())
        .map(|(e, g)| (e - g).abs())
        .fold(0f32, f32::max);
    println!("max_err vs CPU analytic GELU: {:.2e}", max_err);
    println!(
        "sample expected[10]={:.4}  got[10]={:.4}",
        expected[10], out[10]
    );
    println!(
        "sample expected[40]={:.4}  got[40]={:.4}",
        expected[40], out[40]
    );
    if max_err < 1e-4 {
        println!("✓ MPSGraph GELU matches analytic CPU reference");
    } else {
        eprintln!("✗ output diverges");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("requires macOS");
}
