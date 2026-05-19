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

//! Verify Metal sgemm matches a CPU reference + measure throughput.

#[cfg(target_os = "macos")]
fn main() {
    use rlx_metal::blas::metal_sgemm;
    use rlx_metal::device::metal_device;

    let dev = metal_device().expect("no Metal device");

    let cases = [
        (6, 768, 2304),  // BERT-base QKV  (batch=1, seq=6)
        (6, 768, 768),   // BERT-base out_proj
        (6, 768, 3072),  // BERT-base FFN up
        (6, 3072, 768),  // BERT-base FFN down
        (60, 768, 2304), // batch=4, seq=15
        (128, 768, 2304),
        // 32-aligned shapes — tests new sgemm_simd_4x4 path
        (256, 768, 2304),  // batch=16-ish
        (512, 768, 2304),  // batch=32-ish
        (1024, 768, 2304), // batch=64
        (1920, 768, 2304), // batch=128, seq=15
        (256, 768, 3072),  // FFN up at higher batch
        (256, 3072, 768),  // FFN down at higher batch
    ];

    for (m, k, n) in cases {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 13 + 7) % 23) as f32 / 23.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 17 + 3) % 31) as f32 / 31.0)
            .collect();

        // CPU reference
        let mut c_ref = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                c_ref[i * n + j] = s;
            }
        }

        // GPU compute via shared arena buffer
        let arena_bytes = (m * k + k * n + m * n) * 4;
        let buffer = dev.alloc_shared(arena_bytes);
        unsafe {
            let ptr = buffer.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(a.as_ptr(), ptr, m * k);
            std::ptr::copy_nonoverlapping(b.as_ptr(), ptr.add(m * k), k * n);
        }
        let a_off = 0usize;
        let b_off = m * k * 4;
        let c_off = (m * k + k * n) * 4;

        // Warmup
        for _ in 0..3 {
            let cb = dev.queue.new_command_buffer();
            let enc = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
            metal_sgemm(enc, &buffer, a_off, b_off, c_off, m, k, n);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Time
        let n_iter = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..n_iter {
            let cb = dev.queue.new_command_buffer();
            let enc = cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
            metal_sgemm(enc, &buffer, a_off, b_off, c_off, m, k, n);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_iter as f64;

        // Check correctness
        let c_gpu: &[f32] = unsafe {
            let ptr = (buffer.contents() as *const u8).add(c_off) as *const f32;
            std::slice::from_raw_parts(ptr, m * n)
        };
        let max_err = c_ref
            .iter()
            .zip(c_gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let gflops = 2.0 * (m * k * n) as f64 / 1e9 / (avg_ms / 1000.0);
        println!(
            "[{:3} x {:5} x {:5}]  avg={:6.3}ms  max_err={:.2e}  {:6.1} GFLOP/s",
            m, k, n, avg_ms, max_err, gflops
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("Metal only on macOS");
}
