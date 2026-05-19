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

//! Resource estimator: predicts roughly how many LUTs / DSPs / BRAM
//! bytes / cycles a given `OptimizedModel` will need on synthesis.
//!
//! These are *order-of-magnitude* numbers — synth tools mash everything
//! through retiming, sharing, and bram-packing rules that this crate
//! doesn't model. The point is to make the *direction* of each tuning
//! knob visible:
//!
//! * Energy preset → fewer DSPs (ternary fast path drops them).
//! * Size preset   → fewer BRAMs (shared_requant collapses two ROMs
//!                   per qualifying layer to two `localparam`s).
//! * Latency preset → fewer cycles (when parallel kernel lands).
//!
//! Use [`estimate`] for a numeric snapshot, [`Estimate::summary`] for a
//! one-liner suitable for stdout / commit messages.

use crate::model::Layer;
use crate::passes::{Hints, OptimizedModel};
use crate::tune::RequantPrecision;

/// Per-design resource estimate. All numbers are post-tune (the
/// estimator inspects `Hints` to subtract savings).
#[derive(Debug, Clone, Copy, Default)]
pub struct Estimate {
    /// Multiplier slices (DSP48 / EHX-MULT18 / etc.). Each conv/dense
    /// kernel needs one for the MAC and one for the requant epilogue,
    /// minus what the ternary fast path skips. Q0.15 is also one DSP
    /// instead of two.
    pub dsp: u32,

    /// LUT estimate. Coarse: counts adders, subtractors, muxes, FSM
    /// state bits. Realized LUTs depend on FF packing / synthesis
    /// rules.
    pub lut: u32,

    /// BRAM size in bytes. Sums activation buffers + per-layer weight
    /// / bias / requant ROMs.
    pub bram_bytes: u32,

    /// Cycles to compute one inference. Sequential FSMs only today;
    /// parallelism > 1 would divide this proportionally.
    pub cycles: u64,
}

impl Estimate {
    pub fn summary(&self) -> String {
        format!(
            "DSP={} LUT≈{} BRAM={}B cycles={}",
            self.dsp, self.lut, self.bram_bytes, self.cycles,
        )
    }
}

/// Run the estimator over an optimized model.
pub fn estimate(opt: &OptimizedModel) -> Estimate {
    let mut est = Estimate::default();

    // Activation BRAMs:
    //   * arena_plan on  → 2 ping-pong BRAMs sized to max activation
    //   * arena_plan off → one BRAM per stage at its exact size, plus the input
    if opt.tune.arena_plan {
        let scratch = opt.model.input_len.max(
            opt.model
                .layers
                .iter()
                .map(|l| l.out_len())
                .max()
                .unwrap_or(0),
        );
        est.bram_bytes += 2 * scratch as u32;
    } else {
        est.bram_bytes += opt.model.input_len as u32;
        for l in &opt.model.layers {
            est.bram_bytes += l.out_len() as u32;
        }
    }

    for (i, layer) in opt.model.layers.iter().enumerate() {
        let h = &opt.hints[i];
        if h.elided {
            // Fused-into-upstream layers contribute zero compute / area.
            continue;
        }
        match layer {
            Layer::Conv2d {
                h_in,
                w_in,
                c_in,
                c_out,
                kh,
                kw,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                weight_bits,
                bias,
                ..
            } => {
                let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
                let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
                let logical_weights = (c_out * kh * kw * c_in) as u64;
                conv_dense_resources(
                    &mut est,
                    logical_weights,
                    *c_out,
                    *weight_bits,
                    bias.is_some(),
                    h,
                    opt,
                );
                // Pipelined FSM (1 cycle/MAC + 1 warmup), epilogue = bias
                // (3 cycles) + writes (P cycles, one per lane in the
                // parallel kernel; 1 cycle in the scalar kernel).
                // ic_parallelism > 1 divides the inner loop length by P_ic.
                let p = h.parallelism.max(1) as u64;
                let p_ic = h.ic_parallelism.max(1) as u64;
                let inner_per_block = (kh * kw * c_in) as u64 / p_ic + 1;
                let pixel_blocks = (h_out * w_out * (*c_out) / p as usize) as u64;
                let epilogue_per_block = 3 + p;
                est.cycles += pixel_blocks * (inner_per_block + epilogue_per_block);
                if !h.ternary_fast_path {
                    est.dsp += (p - 1) as u32;
                }
            }
            Layer::Dense {
                in_features,
                out_features,
                weight_bits,
                bias,
                ..
            } => {
                let logical_weights = (in_features * out_features) as u64;
                conv_dense_resources(
                    &mut est,
                    logical_weights,
                    *out_features,
                    *weight_bits,
                    bias.is_some(),
                    h,
                    opt,
                );
                // Pipelined dense: per row = bias (3) + INNER + 1 (pipe) + 1 (write).
                let inner = *in_features as u64 + 1;
                est.cycles += (*out_features as u64) * (inner + 4);
            }
            Layer::Relu { len, .. } => {
                est.cycles += 3 * *len as u64;
                est.lut += 16; // FSM + comparator
            }
            Layer::MaxPool2d {
                h_in,
                w_in,
                c,
                kh,
                kw,
                stride_h,
                stride_w,
                ..
            } => {
                let h_out = (h_in - kh) / stride_h + 1;
                let w_out = (w_in - kw) / stride_w + 1;
                let cells = (h_out * w_out * c * kh * kw) as u64;
                est.cycles += 3 * cells + 2 * (h_out * w_out * c) as u64;
                est.lut += 32;
            }
            Layer::Argmax { len, .. } => {
                est.cycles += 3 * *len as u64 + 4;
                est.lut += 32;
            }
        }
    }

    est
}

fn conv_dense_resources(
    est: &mut Estimate,
    logical_weights: u64,
    c_out: usize,
    weight_bits: u8,
    has_bias: bool,
    h: &Hints,
    opt: &OptimizedModel,
) {
    // MAC multiplier — dropped by the ternary fast path.
    if !h.ternary_fast_path {
        est.dsp += 1;
    }
    // Requant multiplier — Q0.15 still costs a DSP but a smaller one;
    // count both as "1 DSP" for tally purposes.
    est.dsp += 1;

    // LUTs:
    //   * MAC adder + accumulator register: ~64 LUT
    //   * ZP subtractors (saved by fast_mac): ~32 LUT
    //   * Requant epilogue: ~80 (Q0.31) / ~50 (Q0.15)
    //   * FSM + counters: ~64
    let mut lut: u32 = 64 + 64;
    if !h.fast_mac {
        lut += 32;
    }
    lut += match opt.tune.requant_precision {
        RequantPrecision::Q0_31 => 80,
        RequantPrecision::Q0_15 => 50,
    };
    if h.ternary_fast_path {
        // Multiplier replaced by a 4-way mux + an extra add — net win
        lut = lut.saturating_sub(48);
    }
    est.lut += lut;

    // BRAM (bytes):
    //   * Weight ROM: macs * weight_bits / 8 (approx, ignores rounding)
    //   * Bias ROM:   c_out * 4 (i32)         when present
    //   * Requant:    c_out * 4 (M0) + c_out * 1 (shift)
    //                 unless shared — then 0
    let weight_bytes = (logical_weights * weight_bits as u64).div_ceil(8) as u32;
    est.bram_bytes += weight_bytes;
    if has_bias {
        est.bram_bytes += (c_out * 4) as u32;
    }
    if h.shared_requant.is_none() {
        est.bram_bytes += (c_out * 5) as u32;
    }
    let _ = c_out;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tinyconv_mnist_from_cortexm;
    use crate::passes::optimize;
    use crate::tune::{OptTarget, Tune};

    #[test]
    fn presets_move_resources_in_expected_directions() {
        let model = tinyconv_mnist_from_cortexm();

        let lat = estimate(&optimize(&model, &Tune::for_target(OptTarget::Latency)));
        let sz = estimate(&optimize(&model, &Tune::for_target(OptTarget::Size)));
        let prec = estimate(&optimize(&model, &Tune::for_target(OptTarget::Precision)));

        // Latency runs P parallel MACs on conv layers → fewer cycles
        // than Precision (which runs sequentially).
        assert!(
            lat.cycles < prec.cycles,
            "Latency cycles {} should be < Precision cycles {}",
            lat.cycles,
            prec.cycles
        );

        // Size ≤ Precision LUT (smaller requant + maybe shared_requant).
        assert!(
            sz.lut <= prec.lut,
            "Size LUT {} > Precision LUT {}",
            sz.lut,
            prec.lut
        );

        // Precision uses Q0.31 → strictly more LUT than Size's Q0.15.
        assert!(prec.lut > sz.lut);

        // Latency adds DSPs for parallel MACs.
        assert!(
            lat.dsp > prec.dsp,
            "Latency DSP {} should be > Precision DSP {}",
            lat.dsp,
            prec.dsp
        );
    }

    #[test]
    fn summary_renders() {
        let model = tinyconv_mnist_from_cortexm();
        let opt = optimize(&model, &Tune::default());
        let s = estimate(&opt).summary();
        assert!(s.contains("DSP="));
        assert!(s.contains("BRAM="));
    }
}
