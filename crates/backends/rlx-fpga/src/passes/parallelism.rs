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

//! Per-layer parallelism eligibility.
//!
//! `tune.parallelism` is the *requested* MACs-per-cycle target. Whether
//! a given layer can hit it depends on its shape and weight encoding;
//! when it can't, the optimizer falls the layer back to P=1.
//!
//! Eligibility rules:
//!
//! * `Conv2d` — eligible when `C_OUT % P == 0`. Works at every
//!   `weight_bits ∈ {2, 4, 8}`: each of the `P` lanes gets its own
//!   weight ROM packed at the layer's bit-width, and the existing
//!   `weight_unpack` (or the ternary crumb-mux) runs P times in
//!   parallel inside the kernel.
//! * `Dense`  — eligible on `out_features % P == 0`, but the parallel
//!   dense kernel is on the pipeline behind conv2d and not yet
//!   emitted; the optimizer falls dense back to P=1 until it lands.
//! * `Relu`/`MaxPool`/`Argmax` — no MACs, parallelism is not
//!   meaningful. Always P=1.

use crate::model::Layer;

/// Highest ic-parallelism (inner-dim) the layer can run at, ≤
/// `requested`. **Currently restricted to ternary** Conv2d layers
/// (`weight_bits = 2`, `w_zp = 0`) where `c_in % P_ic == 0` — the
/// weight ROM stays 1-byte-wide because 4 ternary crumbs already
/// pack into one byte. 8-bit / 4-bit ic-parallel is future work.
pub fn layer_ic_parallelism(layer: &Layer, requested: u32) -> u32 {
    let requested = requested.max(1);
    if requested == 1 {
        return 1;
    }
    match layer {
        Layer::Conv2d {
            c_in,
            weight_bits,
            w_zp,
            ..
        } => {
            if *weight_bits != 2 || *w_zp != 0 {
                return 1;
            }
            if !(*c_in as u32).is_multiple_of(requested) {
                return 1;
            }
            requested
        }
        _ => 1,
    }
}

/// Highest parallelism the layer can run at, ≤ `requested`.
/// Returns `1` for ineligible layers; never returns `0`.
pub fn layer_parallelism(layer: &Layer, requested: u32) -> u32 {
    let requested = requested.max(1);
    if requested == 1 {
        return 1;
    }
    match layer {
        Layer::Conv2d { c_out, .. } if (*c_out as u32).is_multiple_of(requested) => requested,
        // Parallel dense kernel is future work — fall back to P=1 even
        // if the shape would qualify, so we don't claim a speedup we
        // can't produce yet.
        Layer::Dense { .. } => 1,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tinyconv_mnist_from_cortexm;

    #[test]
    fn tinyconv_layer_eligibility_at_p4() {
        // conv1: c_out=8 → 4; conv2: c_out=16 → 4; fc: dense (deferred) → 1
        let m = tinyconv_mnist_from_cortexm();
        let p = m
            .layers
            .iter()
            .map(|l| layer_parallelism(l, 4))
            .collect::<Vec<_>>();
        assert_eq!(p, vec![4, 1, 1, 4, 1, 1, 1, 1]);
    }

    #[test]
    fn tinyconv_layer_eligibility_at_p8() {
        // conv1: c_out=8 → 8; conv2: c_out=16 → 8; fc: dense → 1
        let m = tinyconv_mnist_from_cortexm();
        let p = m
            .layers
            .iter()
            .map(|l| layer_parallelism(l, 8))
            .collect::<Vec<_>>();
        assert_eq!(p, vec![8, 1, 1, 8, 1, 1, 1, 1]);
    }

    #[test]
    fn p_must_divide_c_out() {
        let m = tinyconv_mnist_from_cortexm();
        // P=3: conv1 has c_out=8, 8%3≠0 → 1
        let conv1 = &m.layers[0];
        assert_eq!(layer_parallelism(conv1, 3), 1);
    }

    #[test]
    fn requested_one_is_passthrough() {
        let m = tinyconv_mnist_from_cortexm();
        for l in &m.layers {
            assert_eq!(layer_parallelism(l, 1), 1);
        }
    }
}
