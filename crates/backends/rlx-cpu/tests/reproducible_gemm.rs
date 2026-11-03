// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Under `rlx_ir::reproducible` the CPU GEMM dispatch takes one kernel per
//! shape whatever the host: `sgemm_auto` / `sgemm_bias` are a single `sgemm`
//! call, bit for bit — not the NEON small-GEMM the calibrated cost model picks
//! for tiny `m`, nor the per-thread split a large product gets.
//! (Own test binary: the switch is process-wide.)

fn data(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.618_034).fract() - 0.5)
        .collect()
}

#[test]
fn reproducible_gemm_is_one_sgemm_call() {
    rlx_ir::reproducible::set_reproducible(true);
    for (m, k, n) in [(1, 64, 96), (4, 33, 40), (8, 128, 128), (512, 512, 512)] {
        let (a, b) = (data(m * k, 0.3), data(k * n, 0.7));
        let bias = data(n, 0.1);
        let mut want = vec![0.0; m * n];
        rlx_cpu::blas::sgemm(&a, &b, &mut want, m, k, n);
        let mut got = vec![0.0; m * n];
        rlx_cpu::blas::sgemm_auto(&a, &b, &mut got, m, k, n);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got), bits(&want), "sgemm_auto {m}x{k}x{n}");
        let mut with_bias = vec![0.0; m * n];
        rlx_cpu::blas::sgemm_bias(&a, &b, &bias, &mut with_bias, m, k, n);
        for i in 0..m {
            for j in 0..n {
                want[i * n + j] += bias[j];
            }
        }
        assert_eq!(bits(&with_bias), bits(&want), "sgemm_bias {m}x{k}x{n}");
    }
}
