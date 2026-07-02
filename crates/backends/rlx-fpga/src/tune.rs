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

//! Tuning knobs for the FPGA codegen pipeline.
//!
//! The pipeline is `IR → optimize(model, tune) → emit(opt, dir)`. Each
//! knob in [`Tune`] maps to a concrete Verilog difference; nothing here
//! is a hint that "the synthesizer might decide to". Trade-offs are
//! deterministic, visible in the emitted SV, and reflected in
//! [`crate::estimate`].
//!
//! Pick an [`OptTarget`] preset for a coherent starting point; tweak
//! individual fields if the preset doesn't match the board you're
//! targeting.

use std::fmt;

/// What you're optimizing the design for. Each variant maps to a
/// different default [`Tune`] via [`Tune::for_target`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptTarget {
    /// **Latency** — minimize cycles per inference. Maximum parallelism,
    /// no clock gating (gating saves power but adds an `en` mux on the
    /// critical path).
    Latency,
    /// **Size** — minimize LUT / DSP / BRAM. Lowest practical
    /// parallelism, shared requant when uniform, smallest requant
    /// width that meets the precision floor.
    Size,
    /// **Energy** — minimize switching activity. Ternary fast path on
    /// every weight_bits=2 layer (drops the multiplier entirely for
    /// those layers), BRAM clock-enable gating so unused stages don't
    /// toggle, smaller requant.
    Energy,
    /// **Precision** — keep every numerical detail. Full Q0.31 requant,
    /// no fast paths (so ternary still goes through the integer
    /// multiply, which is bit-identical to the reference).
    Precision,
    /// **Bandwidth** — minimize BRAM port pressure. Weight-stationary
    /// scheduling (read each weight once per inference, broadcast
    /// across spatial positions). NOTE: the scheduling rewrite is not
    /// implemented yet — picking this preset today gives the same
    /// codegen as `Size`, with a `// TODO bandwidth` banner in top.sv.
    Bandwidth,
}

/// Fixed-point requant width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequantPrecision {
    #[default]
    /// Q0.31 — current default. M0 is i32, srdhm = `(a·M0 + 2^30)/2^31`.
    /// Bit-exact match for gemmlowp / TFLite-Micro / CMSIS-NN.
    Q0_31,

    /// Q0.15 — narrower epilogue. M0 is i16, srdhm = `(a·M0 + 2^14)/2^15`.
    /// Halves the multiplier width (a→32 ⊗ b→16 → 48-bit product
    /// fits in a single DSP slice on most parts), at the cost of ≤1
    /// extra ulp at the requant boundary.
    Q0_15,
}

/// Top-level tuning configuration. Construct via [`Tune::for_target`]
/// and tweak individual fields, or build a custom one with [`Tune::default`]
/// + struct update syntax.
#[derive(Debug, Clone, Copy)]
pub struct Tune {
    /// When `x_zp == 0 && w_zp == 0` for a layer, drop the
    /// `- X_ZP` / `- W_ZP` subtractors in the MAC. Saves two
    /// 32-bit subs per multiply. Default: `true` — the trainer
    /// emits `zp = 0` for every TinyConv-MNIST layer.
    pub fold_zero_zp: bool,

    /// For layers with `weight_bits == 2`, replace the multiply with
    /// `case (crumb) → add / sub / skip / -2x`. Drops the DSP slice
    /// entirely for ternary layers — the energy / size win that makes
    /// ternary worth shipping.
    pub ternary_fast_path: bool,

    /// When every per-channel `(M0, shift)` in a layer is the same,
    /// replace the M0 / shift ROMs with `localparam`s. Saves two BRAMs
    /// per qualifying layer.
    pub shared_requant: bool,

    /// Add `en` (clock-enable) ports on every BRAM and gate them so
    /// only the BRAMs touched by the active layer toggle. Modest
    /// dynamic-power win on real silicon; small area cost (one mux
    /// per BRAM).
    pub bram_clock_enable: bool,

    /// Requant epilogue width. Q0.31 (default) is bit-exact with
    /// gemmlowp; Q0.15 is half the multiplier and ≤1 extra ulp.
    pub requant_precision: RequantPrecision,

    /// Inner-loop unroll factor (parallel MACs per cycle). `1` = the
    /// current sequential FSM; values > 1 require widened weight /
    /// activation BRAMs and a partial-sum reduction tree. **Currently
    /// only `1` is implemented**; higher values trigger a debug-assert
    /// in [`crate::passes::optimize`] until the parallel path lands.
    pub parallelism: u32,

    /// IR-driven pattern: detect Conv2d → Relu pairs (matching length
    /// and zero point) and fuse the relu into the conv's requant
    /// epilogue (clamp at `OUT_ZP`). Eliminates the Relu kernel and its
    /// intermediate BRAM. Pure win — same numerical result, fewer
    /// resources, fewer cycles.
    pub fuse_conv_relu: bool,

    /// IR-driven pass: liveness-based BRAM allocator. Sequential
    /// execution → 2 ping-pong slots, sized to the largest activation.
    /// Replaces the per-layer-dedicated BRAM strategy.
    pub arena_plan: bool,

    /// Inner-dim (ic) parallelism factor for conv2d layers. `1` = the
    /// scalar/oc-only kernel; `4` reads 4 consecutive activations from
    /// a 4-banked arena BRAM per cycle and computes
    /// `P_oc × P_ic = parallelism × ic_parallelism` MACs/cycle.
    /// **Currently restricted to ternary (`weight_bits = 2`) layers**
    /// — the weight ROM stays 1 byte / lane / cycle (= 4 crumbs)
    /// without widening. 8-bit / 4-bit ic-parallel is future work.
    pub ic_parallelism: u32,
}

impl Default for Tune {
    /// Same defaults as [`OptTarget::Precision`]: a safe, conservative,
    /// bit-exact-with-reference configuration.
    fn default() -> Self {
        Self::for_target(OptTarget::Precision)
    }
}

impl Tune {
    pub fn for_target(t: OptTarget) -> Self {
        match t {
            OptTarget::Latency => Self {
                fold_zero_zp: true,
                ternary_fast_path: false,
                shared_requant: false,
                bram_clock_enable: false,
                requant_precision: RequantPrecision::Q0_31,
                parallelism: 4,
                fuse_conv_relu: true,
                arena_plan: true,
                ic_parallelism: 1,
            },
            OptTarget::Size => Self {
                fold_zero_zp: true,
                ternary_fast_path: true,
                shared_requant: true,
                bram_clock_enable: false,
                requant_precision: RequantPrecision::Q0_15,
                parallelism: 1,
                fuse_conv_relu: true,
                arena_plan: true,
                ic_parallelism: 1,
            },
            OptTarget::Energy => Self {
                fold_zero_zp: true,
                ternary_fast_path: true,
                shared_requant: true,
                bram_clock_enable: true,
                requant_precision: RequantPrecision::Q0_15,
                parallelism: 1,
                fuse_conv_relu: true,
                arena_plan: true,
                ic_parallelism: 4, // ternary inner-dim parallel
            },
            OptTarget::Precision => Self {
                fold_zero_zp: true,
                ternary_fast_path: false,
                shared_requant: false,
                bram_clock_enable: false,
                requant_precision: RequantPrecision::Q0_31,
                parallelism: 1,
                fuse_conv_relu: true,
                arena_plan: true,
                ic_parallelism: 1,
            },
            OptTarget::Bandwidth => Self {
                // Until the weight-stationary scheduler lands, behave like Size.
                ..Self::for_target(OptTarget::Size)
            },
        }
    }
}

impl fmt::Display for Tune {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tune {{ fold_zp={} ternary_fast={} shared_requant={} \
             bram_en={} requant={:?} P={} P_ic={} }}",
            self.fold_zero_zp,
            self.ternary_fast_path,
            self.shared_requant,
            self.bram_clock_enable,
            self.requant_precision,
            self.parallelism,
            self.ic_parallelism,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_internally_coherent() {
        // Latency: no fast paths, no gating
        let t = Tune::for_target(OptTarget::Latency);
        assert!(!t.ternary_fast_path);
        assert!(!t.bram_clock_enable);

        // Size: every space-saving knob on
        let t = Tune::for_target(OptTarget::Size);
        assert!(t.ternary_fast_path);
        assert!(t.shared_requant);
        assert_eq!(t.requant_precision, RequantPrecision::Q0_15);

        // Energy: gating + low precision + ternary
        let t = Tune::for_target(OptTarget::Energy);
        assert!(t.ternary_fast_path);
        assert!(t.bram_clock_enable);
        assert_eq!(t.requant_precision, RequantPrecision::Q0_15);

        // Precision: nothing lossy
        let t = Tune::for_target(OptTarget::Precision);
        assert!(!t.ternary_fast_path);
        assert!(!t.shared_requant);
        assert_eq!(t.requant_precision, RequantPrecision::Q0_31);
    }
}
