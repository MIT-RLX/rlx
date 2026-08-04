// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dense f32 AMX verification + measurement.
//!
//! There is deliberately no new dense kernel here: on Apple Silicon
//! `cblas_sgemm` (Accelerate) already dispatches to the AMX/SME coprocessor and
//! is the fastest CPU matmul available — you cannot beat the vendor library at
//! its own game, so [[feedback_perf_is_north_star]] says *use it*, don't
//! reimplement it. This module exists to (a) name that path explicitly and
//! (b) *measure* it — against the portable SIMD fallback and against our
//! hand-written [`super::sme`] kernel — so the "Accelerate == AMX" claim is
//! backed by numbers on the actual chip rather than asserted.

/// The dense-f32 AMX path: Accelerate's `cblas_sgemm`. This is a thin,
/// explicitly-named alias so call sites and benchmarks can say "AMX dense GEMM"
/// and mean it. On Apple Silicon with Accelerate linked, this call runs on the
/// matrix coprocessor.
#[inline]
pub fn amx_sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    crate::blas::sgemm(a, b, c, m, k, n);
}

/// Whether the dense AMX path (Accelerate) is linked on this build. There is no
/// chip feature to probe — every Apple Silicon part has the coprocessor — so
/// this reflects link status only.
pub fn is_available() -> bool {
    cfg!(rlx_cpu_blas_accelerate)
}

/// One measured GEMM result: the path name and its sustained GFLOP/s.
#[derive(Debug, Clone, Copy)]
pub struct Measured {
    pub gflops: f64,
    pub secs_per_call: f64,
}

/// Time `f` over `iters` calls after `warmup` warmups; return GFLOP/s for an
/// `m×k×n` GEMM (2·m·n·k flops each). Uses wall-clock `Instant`.
#[cfg(test)]
fn time_gemm(
    m: usize,
    k: usize,
    n: usize,
    warmup: usize,
    iters: usize,
    mut f: impl FnMut(),
) -> Measured {
    use std::time::Instant;
    for _ in 0..warmup {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let flops = 2.0 * m as f64 * n as f64 * k as f64 * iters as f64;
    Measured {
        gflops: flops / elapsed / 1e9,
        secs_per_call: elapsed / iters as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// Run with:
    ///   cargo test -p rlx-cpu --features amx-dense,amx-sme \
    ///     amx_dense_vs_sme_report -- --ignored --nocapture
    /// Prints a GFLOP/s table (Accelerate=AMX vs hand-written SME) per shape.
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn amx_dense_vs_sme_report() {
        let shapes = [
            (256usize, 256usize, 256usize),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 64), // skinny (LLM lm_head-ish)
            (64, 4096, 4096),
        ];
        eprintln!(
            "{:>18}  {:>12}  {:>12}  {:>8}",
            "shape (m×k×n)", "Accel GF/s", "SME GF/s", "SME/Accel"
        );
        for (m, k, n) in shapes {
            let a = fill(m * k, 11 + (m + k + n) as u64);
            let b = fill(k * n, 23 + (m * k) as u64);
            let mut c = vec![0f32; m * n];

            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            // Scale iters so each measured path runs ~0.3s worth of work.
            let iters = ((3e8 / flops).ceil() as usize).clamp(3, 200);

            let accel = time_gemm(m, k, n, 2, iters, || {
                amx_sgemm(&a, &b, &mut c, m, k, n);
            });

            #[cfg(rlx_cpu_amx_sme)]
            let sme = if super::super::sme::is_available() {
                Some(time_gemm(m, k, n, 2, iters, || {
                    super::super::sme::sme_sgemm(&a, &b, &mut c, m, k, n);
                }))
            } else {
                None
            };
            #[cfg(not(rlx_cpu_amx_sme))]
            let sme: Option<Measured> = None;

            match sme {
                Some(s) => eprintln!(
                    "{:>18}  {:>12.1}  {:>12.1}  {:>7.2}x",
                    format!("{m}×{k}×{n}"),
                    accel.gflops,
                    s.gflops,
                    s.gflops / accel.gflops
                ),
                None => eprintln!(
                    "{:>18}  {:>12.1}  {:>12}  {:>8}",
                    format!("{m}×{k}×{n}"),
                    accel.gflops,
                    "n/a",
                    "-"
                ),
            }
        }
    }
}
