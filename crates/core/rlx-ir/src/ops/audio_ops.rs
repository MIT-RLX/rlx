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

//! Audio frontend graph builders.

use crate::{Graph, NodeId, Op};

impl Graph {
    /// Log-mel spectrogram (Whisper-style) from RLX FFT block-layout spectrum.
    ///
    /// * **spectrum** — `[..., 2*n_fft]` with re plane then im plane (same as `Op::Fft` output).
    /// * **filters** — `[n_mels, n_bins]` mel filterbank (`n_bins = n_fft/2 + 1`).
    ///
    /// Output: `[..., n_mels]`.
    pub fn log_mel(&mut self, spectrum: NodeId, filters: NodeId) -> NodeId {
        let spec_shape = self.shape(spectrum).clone();
        let filt_shape = self.shape(filters).clone();
        let out = crate::audio::log_mel_output_shape(&spec_shape, &filt_shape)
            .unwrap_or_else(|e| panic!("log_mel shape error: {e}"));
        self.push(Op::LogMel, vec![spectrum, filters], out, None)
    }

    /// VJP of [`log_mel`] w.r.t. `spectrum`.
    pub fn log_mel_backward(&mut self, spectrum: NodeId, filters: NodeId, dy: NodeId) -> NodeId {
        let spec_shape = self.shape(spectrum).clone();
        self.push(
            Op::LogMelBackward,
            vec![spectrum, filters, dy],
            spec_shape,
            None,
        )
    }

    /// Top-K Welch peaks from RLX FFT block-layout segment spectra.
    ///
    /// * **spectrum** — `[batch * n_segments, 2*n_fft]`
    /// * **k** — spikes per batch row
    /// * **n_segments** — Welch segments averaged per row
    ///
    /// Output: `[batch, k*2]` interleaved `(bin, power)`.
    pub fn welch_peaks(&mut self, spectrum: NodeId, k: usize, n_segments: usize) -> NodeId {
        let spec_shape = self.shape(spectrum).clone();
        let out = crate::audio::welch_peaks_output_shape(&spec_shape, k, n_segments)
            .unwrap_or_else(|e| panic!("welch_peaks shape error: {e}"));
        self.push(Op::WelchPeaks { k, n_segments }, vec![spectrum], out, None)
    }
}
