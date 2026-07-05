//! IIR biquad signal kernels — zero-phase Butterworth filtering, the DSP complement to rlx's
//! FIR (`conv`) and FFT paths. A 2nd-order biquad is 5 coefficients and filters short windows
//! cleanly (no long FIR taps to overflow the signal), which makes it the right primitive for
//! EEG/audio filter-banks. RBJ-cookbook Butterworth (`Q = 1/√2`); `filtfilt` (forward-backward)
//! gives zero phase. Bandpass = highpass(l) ∘ lowpass(h).

/// A biquad as normalized coefficients `[b0, b1, b2, a1, a2]` (a0 = 1).
pub type Biquad = [f64; 5];

/// 2nd-order Butterworth lowpass biquad at cutoff `fc` (Hz), sample rate `fs` (Hz).
pub fn butter_lowpass(fc: f64, fs: f64) -> Biquad {
    let w0 = 2.0 * std::f64::consts::PI * fc / fs;
    let (cw, sw) = (w0.cos(), w0.sin());
    let alpha = sw / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
    let a0 = 1.0 + alpha;
    [
        (1.0 - cw) / 2.0 / a0,
        (1.0 - cw) / a0,
        (1.0 - cw) / 2.0 / a0,
        -2.0 * cw / a0,
        (1.0 - alpha) / a0,
    ]
}

/// 2nd-order Butterworth highpass biquad at cutoff `fc` (Hz), sample rate `fs` (Hz).
pub fn butter_highpass(fc: f64, fs: f64) -> Biquad {
    let w0 = 2.0 * std::f64::consts::PI * fc / fs;
    let (cw, sw) = (w0.cos(), w0.sin());
    let alpha = sw / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
    let a0 = 1.0 + alpha;
    [
        (1.0 + cw) / 2.0 / a0,
        -(1.0 + cw) / a0,
        (1.0 + cw) / 2.0 / a0,
        -2.0 * cw / a0,
        (1.0 - alpha) / a0,
    ]
}

/// Causal biquad filter (Direct-Form-II transposed).
pub fn apply_biquad(x: &[f64], c: &Biquad) -> Vec<f64> {
    let (b0, b1, b2, a1, a2) = (c[0], c[1], c[2], c[3], c[4]);
    let (mut z1, mut z2) = (0f64, 0f64);
    x.iter()
        .map(|&xi| {
            let out = b0 * xi + z1;
            z1 = b1 * xi - a1 * out + z2;
            z2 = b2 * xi - a2 * out;
            out
        })
        .collect()
}

/// Zero-phase biquad filtering (forward then time-reversed backward pass).
pub fn filtfilt(x: &[f64], c: &Biquad) -> Vec<f64> {
    let mut y = apply_biquad(x, c);
    y.reverse();
    let mut y2 = apply_biquad(&y, c);
    y2.reverse();
    y2
}

/// Zero-phase Butterworth bandpass: highpass(`l`) ∘ lowpass(`h`), forward-backward.
pub fn butter_bandpass(x: &[f64], l: f64, h: f64, fs: f64) -> Vec<f64> {
    let a = filtfilt(x, &butter_highpass(l, fs));
    filtfilt(&a, &butter_lowpass(h, fs))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A slow tone in the passband survives; a fast tone in the stopband is attenuated.
    #[test]
    fn bandpass_selects_passband() {
        let fs = 256.0;
        let n = 1024;
        let tone = |f: f64| -> Vec<f64> {
            (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / fs).sin())
                .collect()
        };
        let rms = |x: &[f64]| (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt();
        // 10 Hz is inside an 8–13 Hz band; 50 Hz is well outside.
        let pass = butter_bandpass(&tone(10.0), 8.0, 13.0, fs);
        let stop = butter_bandpass(&tone(50.0), 8.0, 13.0, fs);
        // filtfilt squares the gain; a 10 Hz tone (RMS 0.707) survives at ~0.45, a 50 Hz tone dies.
        assert!(
            rms(&pass) > 0.3,
            "passband tone attenuated: rms {}",
            rms(&pass)
        );
        assert!(
            rms(&stop) < 0.05,
            "stopband tone leaked: rms {}",
            rms(&stop)
        );
        assert!(
            rms(&pass) > 8.0 * rms(&stop),
            "poor selectivity: {} vs {}",
            rms(&pass),
            rms(&stop)
        );
    }

    /// filtfilt is zero-phase: a symmetric impulse response stays symmetric (no group delay).
    #[test]
    fn filtfilt_is_zero_phase() {
        let fs = 256.0;
        let mut x = vec![0f64; 257];
        x[128] = 1.0; // centered impulse
        let y = filtfilt(&x, &butter_lowpass(20.0, fs));
        // symmetry about the center ⇒ zero phase
        let asym: f64 = (0..128)
            .map(|i| (y[128 - 1 - i] - y[128 + 1 + i]).abs())
            .fold(0.0, f64::max);
        assert!(asym < 1e-9, "filtfilt not symmetric (has phase): {asym}");
    }
}
