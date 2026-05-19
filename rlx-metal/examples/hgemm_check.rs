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

//! Verify half-precision matmul kernel correctness + measure throughput.
//! cargo run --release --example hgemm_check -p rlx-metal

#[cfg(target_os = "macos")]
fn main() {
    use half::f16;
    use metal::MTLSize;
    use rlx_metal::device::metal_device;
    use rlx_metal::kernels::kernels;

    let dev = metal_device().expect("no Metal device");
    let kk = kernels();

    let cases = [
        (32, 256, 256),
        (128, 768, 2304), // QKV at large batch
        (256, 768, 3072), // FFN-up
        (512, 768, 2304),
        (1024, 768, 2304),
        (1920, 768, 2304),
    ];

    for (m, k, n) in cases {
        let a_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i * 13 + 7) % 23) as f32 / 23.0)
            .collect();
        let b_f32: Vec<f32> = (0..k * n)
            .map(|i| ((i * 17 + 3) % 31) as f32 / 31.0)
            .collect();
        let a_f16: Vec<f16> = a_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let b_f16: Vec<f16> = b_f32.iter().map(|&v| f16::from_f32(v)).collect();

        let mut c_ref = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for kk in 0..k {
                    s += a_f32[i * k + kk] * b_f32[kk * n + j];
                }
                c_ref[i * n + j] = s;
            }
        }

        // Allocate arena buffer for f16 inputs/output
        let bytes = (m * k + k * n + m * n) * 2;
        let buffer = dev.alloc_shared(bytes);
        unsafe {
            let ptr = buffer.contents() as *mut f16;
            std::ptr::copy_nonoverlapping(a_f16.as_ptr(), ptr, m * k);
            std::ptr::copy_nonoverlapping(b_f16.as_ptr(), ptr.add(m * k), k * n);
        }
        let a_off = 0usize;
        let b_off = m * k * 2;
        let c_off = (m * k + k * n) * 2;

        // Warmup
        for _ in 0..2 {
            let cb = dev.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&kk.hgemm_simd_4x4);
            enc.set_buffer(0, Some(&buffer), a_off as u64);
            enc.set_buffer(1, Some(&buffer), b_off as u64);
            enc.set_buffer(2, Some(&buffer), c_off as u64);
            let m_u = m as u32;
            let k_u = k as u32;
            let n_u = n as u32;
            enc.set_bytes(3, 4, &m_u as *const _ as *const _);
            enc.set_bytes(4, 4, &k_u as *const _ as *const _);
            enc.set_bytes(5, 4, &n_u as *const _ as *const _);
            let tg_count = MTLSize {
                width: (n / 32) as u64,
                height: (m / 32) as u64,
                depth: 1,
            };
            enc.dispatch_thread_groups(
                tg_count,
                MTLSize {
                    width: 512,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Time 50 dispatches in one cb
        let n_iter = 50;
        let cb = dev.queue.new_command_buffer();
        let t0 = std::time::Instant::now();
        for _ in 0..n_iter {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&kk.hgemm_simd_4x4);
            enc.set_buffer(0, Some(&buffer), a_off as u64);
            enc.set_buffer(1, Some(&buffer), b_off as u64);
            enc.set_buffer(2, Some(&buffer), c_off as u64);
            let m_u = m as u32;
            let k_u = k as u32;
            let n_u = n as u32;
            enc.set_bytes(3, 4, &m_u as *const _ as *const _);
            enc.set_bytes(4, 4, &k_u as *const _ as *const _);
            enc.set_bytes(5, 4, &n_u as *const _ as *const _);
            let tg_count = MTLSize {
                width: (n / 32) as u64,
                height: (m / 32) as u64,
                depth: 1,
            };
            enc.dispatch_thread_groups(
                tg_count,
                MTLSize {
                    width: 512,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();
        let elapsed = t0.elapsed().as_secs_f64();
        let avg_ms = elapsed * 1000.0 / n_iter as f64;

        let c_gpu_f16: &[f16] = unsafe {
            let ptr = (buffer.contents() as *const u8).add(c_off) as *const f16;
            std::slice::from_raw_parts(ptr, m * n)
        };
        let max_err: f32 = c_ref
            .iter()
            .zip(c_gpu_f16.iter())
            .map(|(a, b)| (a - b.to_f32()).abs() / (a.abs().max(1e-6)))
            .fold(0f32, f32::max);
        let gflops = 2.0 * (m * k * n) as f64 / 1e9 / (avg_ms / 1000.0);
        println!(
            "[{:5} x {:5} x {:5}]  avg={:6.3}ms  rel_err={:.2e}  {:6.0} GFLOP/s",
            m, k, n, avg_ms, max_err, gflops
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("Metal only on macOS");
}
