// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fanning `Op::FftQ` out across batch rows must be **bit-identical** to the
//! serial loop, not merely close.
//!
//! That is the entire justification for doing it: rows touch disjoint slices
//! and share only read-only twiddles. A fixed-point transform that changed
//! answers depending on thread count would be worse than a slow one, and
//! floating-point intuitions about reassociation do not apply here — so this
//! pins the property rather than trusting it.

use rlx_ir::fft::{FftNorm, FftQScale};

fn sample(outer: usize, n: usize) -> Vec<i32> {
    (0..outer * 2 * n)
        .map(|i| ((i as f64 * 0.137).sin() * 15000.0) as i32)
        .collect()
}

#[test]
fn parallel_matches_serial_bit_for_bit() {
    for &(outer, n) in &[(8usize, 4096usize), (64, 1024), (256, 512), (7, 1024)] {
        for scale in [
            FftQScale::None,
            FftQScale::Saturating,
            FftQScale::EveryOther,
            FftQScale::PerStage,
        ] {
            for inverse in [false, true] {
                for norm in [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho] {
                    let x = sample(outer, n);

                    let mut want = x.clone();
                    rlx_ir::fft::fft1d_q32_block(&mut want, outer, n, inverse, norm, scale)
                        .unwrap();

                    let mut got = x.clone();
                    rlx_cpu::thunk::fft1d_q32_block_parallel(
                        &mut got, outer, n, inverse, norm, scale,
                    )
                    .unwrap();

                    assert_eq!(
                        got, want,
                        "outer={outer} n={n} scale={scale:?} inverse={inverse} norm={norm:?}: \
                         parallel diverged from serial"
                    );
                }
            }
        }
    }
}

/// The parallel path must reject the same inputs the serial one does, rather
/// than panicking inside a worker or silently transforming a short buffer.
#[test]
fn parallel_rejects_what_serial_rejects() {
    // Wrong length, above the parallel threshold so it takes the fanned-out path.
    let mut short = vec![0i32; 64 * 2 * 1024 - 1];
    let e = rlx_cpu::thunk::fft1d_q32_block_parallel(
        &mut short,
        64,
        1024,
        false,
        FftNorm::Backward,
        FftQScale::None,
    )
    .unwrap_err();
    assert!(e.contains("expects"), "unexpected error: {e}");

    let mut x = vec![0i32; 64 * 2 * 1000];
    let e = rlx_cpu::thunk::fft1d_q32_block_parallel(
        &mut x,
        64,
        1000,
        false,
        FftNorm::Backward,
        FftQScale::None,
    )
    .unwrap_err();
    assert!(e.contains("power of two"), "unexpected error: {e}");
}
