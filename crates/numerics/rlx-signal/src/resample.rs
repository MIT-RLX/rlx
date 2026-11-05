// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `scipy.signal.resample_poly` (default `window=("kaiser", 5.0)`, `padtype="constant"`)
//! and the pieces it is built from: Cephes `i0`, `windows.kaiser`, and `upfirdn`'s Cython
//! polyphase loop, in SciPy's accumulation order.

use crate::fir::firwin_windowed;

/// Cephes `i0` Chebyshev coefficients on `[0, 8]` (`numpy` `_i0A`, `scipy.special` A).
const I0_A: [f64; 30] = [
    -4.415_341_646_479_339_5e-18,
    3.330_794_518_822_238_4e-17,
    -2.431_279_846_547_955e-16,
    1.715_391_285_555_133e-15,
    -1.168_533_287_799_345_1e-14,
    7.676_185_498_604_936e-14,
    -4.856_446_783_111_929e-13,
    2.955_052_663_129_64e-12,
    -1.726_826_291_441_556e-11,
    9.675_809_035_373_237e-11,
    -5.189_795_601_635_263e-10,
    2.659_823_724_682_386_6e-9,
    -1.300_025_009_986_248e-8,
    6.046_995_022_541_919e-8,
    -2.670_793_853_940_612e-7,
    1.117_387_539_120_103_7e-6,
    -4.416_738_358_458_750_5e-6,
    1.644_844_807_072_889_6e-5,
    -5.754_195_010_082_104e-5,
    1.885_028_850_958_416_5e-4,
    -5.763_755_745_385_824e-4,
    1.639_475_616_941_335_7e-3,
    -4.324_309_995_050_576e-3,
    1.054_646_039_459_499_8e-2,
    -2.373_741_480_589_947e-2,
    4.930_528_423_967_071e-2,
    -9.490_109_704_804_764e-2,
    1.716_209_015_222_087_7e-1,
    -3.046_826_723_431_984e-1,
    6.767_952_744_094_761e-1,
];

/// Cephes `i0` Chebyshev coefficients for `exp(-x)·sqrt(x)·I0(x)` on `(8, ∞)`, in `8/x`.
const I0_B: [f64; 25] = [
    -7.233_180_487_874_754e-18,
    -4.830_504_485_944_182e-18,
    4.465_621_420_296_76e-17,
    3.461_222_867_697_461e-17,
    -2.827_623_980_516_583_6e-16,
    -3.425_485_619_677_219e-16,
    1.772_560_133_056_526_3e-15,
    3.811_680_669_352_622_4e-15,
    -9.554_846_698_828_307e-15,
    -4.150_569_347_287_222e-14,
    1.540_086_217_521_41e-14,
    3.852_778_382_742_142_6e-13,
    7.180_124_451_383_666e-13,
    -1.794_178_531_506_806_2e-12,
    -1.321_581_184_044_771_3e-11,
    -3.149_916_527_963_241_6e-11,
    1.188_914_710_784_643_9e-11,
    4.940_602_388_224_97e-10,
    3.396_232_025_708_386_5e-9,
    2.266_668_990_498_178e-8,
    2.048_918_589_469_063_8e-7,
    2.891_370_520_834_756_7e-6,
    6.889_758_346_916_825e-5,
    3.369_116_478_255_694_3e-3,
    8.044_904_110_141_088e-1,
];

/// Cephes `chbevl`: Clenshaw recurrence `b0 = x·b1 − b2 + c` (one FMA under clang's
/// `-ffp-contract=on`, as SciPy's special-function build compiles it).
fn chbevl(x: f64, c: &[f64]) -> f64 {
    let mut b0 = c[0];
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for &ci in &c[1..] {
        b2 = b1;
        b1 = b0;
        b0 = x.mul_add(b1, -b2) + ci;
    }
    0.5 * (b0 - b2)
}

/// `scipy.special.i0(x)` — modified Bessel function of the first kind, order 0 (Cephes).
pub fn i0(x: f64) -> f64 {
    let x = x.abs();
    if x <= 8.0 {
        let y = x / 2.0 - 2.0;
        return rlx_libm::reference::exp(x) * chbevl(y, &I0_A);
    }
    rlx_libm::reference::exp(x) * chbevl(32.0 / x - 2.0, &I0_B) / x.sqrt()
}

/// `scipy.signal.windows.kaiser(m, beta, sym=True)`.
pub fn kaiser(m: usize, beta: f64) -> Vec<f64> {
    if m <= 1 {
        return vec![1.0; m];
    }
    let alpha = (m as f64 - 1.0) / 2.0;
    let den = i0(beta);
    (0..m)
        .map(|n| {
            let r = (n as f64 - alpha) / alpha;
            i0(beta * (1.0 - r * r).sqrt()) / den
        })
        .collect()
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// `upfirdn`'s `_output_len(len_h, in_len, up, down)`.
fn output_len(len_h: usize, n_in: usize, up: usize, down: usize) -> usize {
    ((n_in - 1) * up + len_h - 1) / down + 1
}

/// `scipy.signal.upfirdn(h, x, up, down)` with the default `mode="constant", cval=0`.
///
/// SciPy's `_apply_impl` walks the polyphase-reordered, flipped filter (`_pad_h`) and adds
/// one product per term to the output in increasing input index, `out = out + x·h` — fused
/// into one FMA by clang.
pub fn upfirdn(h: &[f64], x: &[f64], up: usize, down: usize) -> Vec<f64> {
    assert!(up >= 1 && down >= 1 && !h.is_empty() && !x.is_empty());
    // _pad_h: zero-pad to a multiple of `up`, reshape (-1, up), transpose, flip each row
    let padlen = h.len() + (up - h.len() % up) % up;
    let hpp = padlen / up; // taps per phase
    let mut ht = vec![0f64; padlen];
    for p in 0..up {
        for k in 0..hpp {
            let src = k * up + p;
            ht[p * hpp + (hpp - 1 - k)] = if src < h.len() { h[src] } else { 0.0 };
        }
    }
    let len_x = x.len();
    let out_len = output_len(h.len(), len_x, up, down);
    let mut out = vec![0f64; out_len];
    let padded_len = len_x + hpp - 1;
    let (mut x_idx, mut y_idx, mut t) = (0usize, 0usize, 0usize);
    while x_idx < padded_len && y_idx < out_len {
        let mut h_idx = t * hpp;
        let first = x_idx as isize - hpp as isize + 1;
        let mut acc = out[y_idx];
        for xc in first..=x_idx as isize {
            if xc >= 0 && (xc as usize) < len_x {
                acc = x[xc as usize].mul_add(ht[h_idx], acc);
            }
            h_idx += 1;
        }
        out[y_idx] = acc;
        y_idx += 1;
        t += down;
        x_idx += t / up;
        t %= up;
    }
    out
}

/// `scipy.signal.resample_poly(x, up, down)` for one float64 signal.
pub fn resample_poly(x: &[f64], up: usize, down: usize) -> Vec<f64> {
    assert!(up >= 1 && down >= 1, "up and down must be ≥ 1");
    let g = gcd(up, down);
    let (up, down) = (up / g, down / g);
    if up == 1 && down == 1 {
        return x.to_vec();
    }
    let n_in = x.len();
    let n_out = (n_in * up).div_ceil(down);
    let max_rate = up.max(down);
    let half_len = 10 * max_rate;
    let numtaps = 2 * half_len + 1;
    let mut h = firwin_windowed(
        numtaps,
        &[1.0 / max_rate as f64],
        &kaiser(numtaps, 5.0),
        true,
        None,
    )
    .expect("a valid low-pass design");
    for v in h.iter_mut() {
        *v *= up as f64;
    }
    let n_pre_pad = down - half_len % down;
    let n_pre_remove = (half_len + n_pre_pad) / down;
    let mut n_post_pad = 0;
    while output_len(h.len() + n_pre_pad + n_post_pad, n_in, up, down) < n_out + n_pre_remove {
        n_post_pad += 1;
    }
    let mut hp = vec![0f64; n_pre_pad];
    hp.extend_from_slice(&h);
    hp.resize(hp.len() + n_post_pad, 0.0);
    let y = upfirdn(&hp, x, up, down);
    y[n_pre_remove..n_pre_remove + n_out].to_vec()
}
