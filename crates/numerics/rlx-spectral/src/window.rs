//! Analysis windows, built in f64 and rounded once.

use crate::float::Float;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Window {
    /// `0.5 − 0.5·cos(2πi/N)` (periodic: torch `hann_window(periodic=True)`,
    /// whisper's `sin²(πi/N)`).
    HannPeriodic,
    /// `0.5 − 0.5·cos(2πi/(N−1))` (`np.hanning`).
    HannSymmetric,
    /// `0.54 − 0.46·cos(2πi/(N−1))` (`np.hamming`).
    HammingSymmetric,
    /// Kaldi's Povey window, `HannSymmetric^0.85`.
    Povey,
    Rectangular,
}

impl Window {
    pub fn build<T: Float>(self, n: usize) -> Vec<T> {
        let pi = std::f64::consts::PI;
        let sym = |i: usize| {
            if n > 1 {
                i as f64 / (n - 1) as f64
            } else {
                0.0
            }
        };
        (0..n)
            .map(|i| {
                let v = match self {
                    Window::HannPeriodic => {
                        let s = libm::sin(pi * i as f64 / n as f64);
                        s * s
                    }
                    Window::HannSymmetric => 0.5 - 0.5 * libm::cos(2.0 * pi * sym(i)),
                    Window::HammingSymmetric => 0.54 - 0.46 * libm::cos(2.0 * pi * sym(i)),
                    Window::Povey => libm::pow(0.5 - 0.5 * libm::cos(2.0 * pi * sym(i)), 0.85),
                    Window::Rectangular => 1.0,
                };
                T::from_f64(v)
            })
            .collect()
    }
}
