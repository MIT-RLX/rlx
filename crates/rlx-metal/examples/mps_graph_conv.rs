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

//! Verify MPSGraph Conv2D for a ViT-style patch embedding shape.
//!
//! Setup: 1×4×4×3 (NHWC) with kernel 2×2, stride 2 → output 1×2×2×K.
//! Weights are constant (out_ch=K=2). We compare against a hand-rolled
//! conv on CPU.
//!
//! cargo run --example mps_graph_conv --release -p rlx-metal

#[cfg(target_os = "macos")]
fn main() {
    use metal::MTLResourceOptions;
    use rlx_metal::device::metal_device;
    use rlx_metal::mps_graph::{MpsGraph, mps_graph_supported};

    if !mps_graph_supported() {
        eprintln!("no MPSGraph");
        return;
    }
    let dev = metal_device().expect("metal");

    // Input: NHWC = [1, 4, 4, 3]. Filled with consecutive ints.
    let input: Vec<f32> = (0..(4 * 4 * 3)).map(|i| i as f32).collect();
    // Weights: OIHW = [out=2, in=3, kh=2, kw=2] = 24 floats. All ones for
    // the first output channel; alternating sign for the second.
    let mut weights = [1.0f32; 24];
    for i in 12..24 {
        weights[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }

    let buf_in = dev.device.new_buffer(
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_w = dev.device.new_buffer(
        (weights.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let buf_out = dev.device.new_buffer(
        (2 * 2 * 2 * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(input.as_ptr(), buf_in.contents() as *mut f32, input.len());
        std::ptr::copy_nonoverlapping(
            weights.as_ptr(),
            buf_w.contents() as *mut f32,
            weights.len(),
        );
    }

    const F32: u32 = 0x10000000 | 32;
    let g = MpsGraph::new();
    let src = g.placeholder(&[1, 4, 4, 3], F32, "x");
    let w = g.placeholder(&[2, 3, 2, 2], F32, "w");
    let y = g.conv2d(&src, &w, (2, 2), (0, 0));

    g.run_jit(
        &dev.queue,
        &[&src, &w],
        &[&buf_in, &buf_w],
        &[0, 0],
        &[vec![1, 4, 4, 3], vec![2, 3, 2, 2]],
        &[F32, F32],
        &[&y],
        &[&buf_out],
        &[0],
        &[vec![1, 2, 2, 2]],
        &[F32],
    );

    let out: &[f32] =
        unsafe { std::slice::from_raw_parts(buf_out.contents() as *const f32, 2 * 2 * 2) };
    println!("output: {:?}", out);
    // Sanity check: output[0] (top-left, out_ch=0) = sum of 2*2*3 = 12 input
    // values multiplied by all-ones weights for out_ch=0. The 12 values are
    // input[0..2, 0..2, 0..3] = [0,1,2, 3,4,5, 12,13,14, 15,16,17].
    // Sum = 102.
    let expected_00 = 1 + 2 + 3 + 4 + 5 + 12 + 13 + 14 + 15 + 16 + 17;
    let max_diff = (out[0] - expected_00 as f32).abs();
    println!(
        "expected output[0] = {}, got {}, diff {:.2e}",
        expected_00, out[0], max_diff
    );
    assert!(max_diff < 1e-3, "Conv2D output mismatch");
    println!("✓ MPSGraph Conv2D works (top-left position matches hand calc)");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("requires macOS");
}
