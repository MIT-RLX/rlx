// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `scipy.signal.firwin` and the cosine-sum windows (`get_window(..., fftbins=False)`),
//! SciPy 1.17.1, bit-for-bit. NumPy's `sin`/`cos` are libm's on the platforms this is
//! measured on, `np.linspace` is `i·step + start` with the end pinned, and `np.sum`
//! of a contiguous array is NumPy's pairwise sum.

/// The windows `get_window` names that MNE's FIR design uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    Hann,
    Hamming,
    Blackman,
    Boxcar,
}

impl Window {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "hann" | "hanning" => Window::Hann,
            "hamming" => Window::Hamming,
            "blackman" => Window::Blackman,
            "boxcar" | "rect" | "rectangular" | "ones" => Window::Boxcar,
            _ => return None,
        })
    }
}

/// `_general_cosine_impl(M, a, sym=True)`: `w = Σ_k a[k]·cos(k·fac)`,
/// `fac = linspace(-π, π, M)`, accumulated into zeros in `k` order.
pub fn general_cosine(m: usize, a: &[f64]) -> Vec<f64> {
    if m <= 1 {
        return vec![1.0; m];
    }
    let fac = rlx_numpy::linspace(-std::f64::consts::PI, std::f64::consts::PI, m);
    let mut w = vec![0.0f64; m];
    for (k, &ak) in a.iter().enumerate() {
        for (wi, &f) in w.iter_mut().zip(&fac) {
            *wi += ak * rlx_libm::reference::cos(k as f64 * f);
        }
    }
    w
}

/// `get_window(window, M, fftbins=False)` (symmetric).
pub fn get_window(window: Window, m: usize) -> Vec<f64> {
    match window {
        // general_hamming(M, alpha) = general_cosine(M, [alpha, 1. - alpha])
        Window::Hann => general_cosine(m, &[0.5, 1.0 - 0.5]),
        Window::Hamming => general_cosine(m, &[0.54, 1.0 - 0.54]),
        Window::Blackman => general_cosine(m, &[0.42, 0.50, 0.08]),
        Window::Boxcar => vec![1.0; m],
    }
}

/// `np.sinc(x)` = `sin(πx)/(πx)`, with `πx == 0` replaced by `eps` (→ 1).
pub fn np_sinc(x: f64) -> f64 {
    let y = std::f64::consts::PI * x;
    let y = if y != 0.0 { y } else { f64::EPSILON };
    rlx_libm::reference::sin(std::hint::black_box(y)) / y
}

/// `scipy.signal.firwin(numtaps, cutoff, window=window, pass_zero=pass_zero,
/// scale=True, fs=fs)`; `cutoff` in the units of `fs` (`fs = None` → 2).
pub fn firwin(
    numtaps: usize,
    cutoff: &[f64],
    window: Window,
    pass_zero: bool,
    fs: Option<f64>,
) -> Result<Vec<f64>, String> {
    let fs = fs.unwrap_or(2.0);
    let nyq = 0.5 * fs;
    let cutoff: Vec<f64> = cutoff.iter().map(|c| c / nyq).collect();
    if cutoff.is_empty() {
        return Err("At least one cutoff frequency must be given.".into());
    }
    let (mn, mx) = cutoff
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &c| {
            (a.min(c), b.max(c))
        });
    if mn <= 0.0 || mx >= 1.0 {
        return Err(
            "Invalid cutoff frequency: frequencies must be greater than 0 and less than fs/2."
                .into(),
        );
    }
    if cutoff.windows(2).any(|w| w[1] - w[0] <= 0.0) {
        return Err(
            "Invalid cutoff frequencies: the frequencies must be strictly increasing.".into(),
        );
    }
    let pass_nyquist = cutoff.len().is_multiple_of(2) == pass_zero;
    if pass_nyquist && numtaps.is_multiple_of(2) {
        return Err("A filter with an even number of coefficients must have zero response at the Nyquist frequency.".into());
    }
    let mut edges: Vec<f64> = Vec::new();
    if pass_zero {
        edges.push(0.0);
    }
    edges.extend(&cutoff);
    if pass_nyquist {
        edges.push(1.0);
    }
    let alpha = 0.5 * (numtaps as f64 - 1.0);
    let m: Vec<f64> = (0..numtaps).map(|i| i as f64 - alpha).collect();
    // h = 0; h += right·sinc(right·m); h -= left·sinc(left·m)
    let mut h: Option<Vec<f64>> = None;
    for band in edges.chunks(2) {
        let (left, right) = (band[0], band[1]);
        let hv = h.get_or_insert_with(|| vec![0.0; numtaps]);
        for (hi, &mi) in hv.iter_mut().zip(&m) {
            *hi += right * np_sinc(right * mi);
        }
        for (hi, &mi) in hv.iter_mut().zip(&m) {
            *hi -= left * np_sinc(left * mi);
        }
    }
    let mut h = h.unwrap();
    let win = get_window(window, numtaps);
    for (hi, wi) in h.iter_mut().zip(&win) {
        *hi *= wi;
    }
    let (left, right) = (edges[0], edges[1]);
    let scale_frequency = if left == 0.0 {
        0.0
    } else if right == 1.0 {
        1.0
    } else {
        0.5 * (left + right)
    };
    let hc: Vec<f64> = h
        .iter()
        .zip(&m)
        .map(|(hi, mi)| hi * rlx_libm::reference::cos(std::f64::consts::PI * mi * scale_frequency))
        .collect();
    let s = rlx_numpy::pairwise_sum(&hc);
    for hi in h.iter_mut() {
        *hi /= s;
    }
    Ok(h)
}
