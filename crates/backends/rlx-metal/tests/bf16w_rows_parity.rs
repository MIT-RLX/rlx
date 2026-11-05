// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2–8 rows against a bf16 weight (sequences decoded together) run the
//! split-K multi-row GEMV: each row must come out bit-identical to the same
//! row run alone through the one-row kernel, and equal to a host reference.
//! Covers both column tiles (N under and over the 16 k narrow cut).

#![cfg(target_os = "macos")]

use rlx_metal::blas::metal_sgemm_bf16w_small_bufs;
use rlx_metal::device::metal_device;

fn bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// `C = A·B` with `B` as bf16 bits, one dispatch.
fn run(a: &[f32], b: &[u16], m: usize, k: usize, n: usize) -> Vec<f32> {
    let dev = metal_device().expect("Metal device");
    let align = |x: usize| (x + 255) & !255;
    let b_off = align(a.len() * 4);
    let c_off = align(b_off + b.len() * 2);
    let buf = dev.alloc_shared(c_off + m * n * 4);
    unsafe {
        let base = buf.contents() as *mut u8;
        std::ptr::copy_nonoverlapping(a.as_ptr(), base as *mut f32, a.len());
        std::ptr::copy_nonoverlapping(b.as_ptr(), base.add(b_off) as *mut u16, b.len());
    }
    let cmd = dev.queue.new_command_buffer();
    {
        let enc = cmd.compute_command_encoder();
        assert!(metal_sgemm_bf16w_small_bufs(
            enc, &buf, 0, &buf, b_off, &buf, c_off, m, k, n
        ));
        enc.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();
    unsafe {
        std::slice::from_raw_parts(
            (buf.contents() as *const u8).add(c_off) as *const f32,
            m * n,
        )
        .to_vec()
    }
}

#[test]
fn rows_together_equal_rows_alone() {
    for &(k, n) in &[(896usize, 2688usize), (1024, 4864), (512, 24576)] {
        // Irregular values: dyadic ones sum exactly in any order and hide it.
        let a: Vec<f32> = (0..8 * k).map(|i| (i as f32 * 0.731).sin()).collect();
        let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.377).cos() * 0.1).collect();
        let b: Vec<u16> = w.iter().map(|&x| bf16(x)).collect();
        let alone: Vec<Vec<f32>> = (0..8)
            .map(|r| run(&a[r * k..(r + 1) * k], &b, 1, k, n))
            .collect();
        for m in 2..=8 {
            let got = run(&a[..m * k], &b, m, k, n);
            for r in 0..m {
                let row = &got[r * n..(r + 1) * n];
                assert!(
                    row.iter()
                        .zip(&alone[r])
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "k={k} n={n} m={m}: row {r} differs from the row alone"
                );
                for j in (0..n).step_by(97) {
                    let want: f64 = (0..k)
                        .map(|t| {
                            a[r * k + t] as f64 * f32::from_bits((b[t * n + j] as u32) << 16) as f64
                        })
                        .sum();
                    assert!(
                        (row[j] as f64 - want).abs() < 1e-3 * want.abs().max(1.0),
                        "k={k} n={n} m={m} row {r} col {j}: {} vs {want}",
                        row[j]
                    );
                }
            }
        }
    }
}
