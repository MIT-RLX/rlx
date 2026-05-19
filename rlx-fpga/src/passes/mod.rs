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

//! Optimizer passes. Each pass takes the immutable `Model` plus the
//! `Tune` config and produces a [`Hints`] struct per layer; the
//! optimizer combines them into an [`OptimizedModel`] that codegen
//! consumes.
//!
//! The passes are deliberately analysis-only: they don't rewrite the
//! `Model` itself. Codegen reads the hints and emits a different
//! Verilog body. This keeps the reference forward pass (which doesn't
//! see hints at all) bit-identical to the unoptimized model — exactly
//! what we want for a parity oracle.

use crate::model::{Layer, Model};
use crate::tune::{RequantPrecision, Tune};

pub mod arena;
pub mod fold_zero_zp;
pub mod fuse_conv_relu;
pub mod parallelism;
pub mod shared_requant;
pub mod ternary_fast_path;

/// Per-layer codegen hints. Defaults are conservative — every flag off,
/// parallelism = 1.
#[derive(Debug, Clone)]
pub struct Hints {
    /// Skip the `- X_ZP` / `- W_ZP` subtractors in the MAC. Set when
    /// `tune.fold_zero_zp && x_zp == 0 && w_zp == 0`.
    pub fast_mac: bool,

    /// Replace the multiply with a `case (crumb)` add/sub/skip tree.
    /// Set when `tune.ternary_fast_path && weight_bits == 2`.
    pub ternary_fast_path: bool,

    /// `Some((m0, shift))` if every per-channel `(M0, shift)` in this
    /// layer is identical and `tune.shared_requant` is on. Codegen
    /// then emits `localparam`s instead of two BRAMs.
    pub shared_requant: Option<(i32, i32)>,

    /// Add `en` ports on every BRAM in this layer and gate them off
    /// the layer's `start..done` interval. Set when
    /// `tune.bram_clock_enable`.
    pub bram_clock_enable: bool,

    /// Requant epilogue width. Inherited from `Tune`.
    pub requant_precision: RequantPrecision,

    /// MACs per cycle in the inner loop. `1` is the sequential FSM;
    /// values > 1 fan out into P-banked weight ROMs and P parallel
    /// accumulators. Set per-layer by the optimizer — defaults to
    /// `tune.parallelism` for eligible layers, falls back to `1` when
    /// the layer can't be cleanly parallelized (see
    /// [`parallelism::layer_parallelism`]).
    pub parallelism: u32,

    /// Inner-dim (ic) parallelism factor. `1` is scalar; `4` produces
    /// `P × ic_parallelism` MACs/cycle. Today only enabled for ternary
    /// Conv2d with `c_in % P_ic == 0` — see
    /// [`parallelism::layer_ic_parallelism`].
    pub ic_parallelism: u32,

    /// Set on Conv2d when an immediately-following Relu has matching
    /// length and zero point. The conv kernel clamps its requant
    /// output at `OUT_ZP` (= relu's zero_point); the relu kernel is
    /// elided from `top.sv` along with its intermediate BRAM. This is
    /// the conv→relu fusion pattern, recognized by `fuse_conv_relu`.
    pub fuses_relu: bool,

    /// Set on layers that the `fuse_conv_relu` pass identifies as
    /// redundant — their compute is absorbed by the upstream layer.
    /// Codegen skips emitting a kernel module + a top.sv instance for
    /// elided layers (the controller's stage counter still advances
    /// over them, conceptually, but they have no work to do). The
    /// arena planner also skips allocating a BRAM slot for elided
    /// layers' outputs.
    pub elided: bool,

    /// BRAM "slot" assigned by the arena planner. With the simple
    /// ping-pong allocator this is `0` or `1` (or `2` for input);
    /// codegen-side `top.sv` reads this to decide which physical BRAM
    /// each kernel's `x_*` / `y_*` ports connect to. `None` means
    /// "use the per-layer dedicated BRAM" (legacy / pre-arena layout).
    pub bram_slot_in: Option<u8>,
    pub bram_slot_out: Option<u8>,
}

impl Default for Hints {
    fn default() -> Self {
        Self {
            fast_mac: false,
            ternary_fast_path: false,
            shared_requant: None,
            bram_clock_enable: false,
            requant_precision: RequantPrecision::default(),
            parallelism: 1,
            ic_parallelism: 1,
            fuses_relu: false,
            elided: false,
            bram_slot_in: None,
            bram_slot_out: None,
        }
    }
}

/// `Model` + per-layer `Hints` + the `Tune` they came from. Pass to
/// [`crate::codegen::emit_optimized`].
pub struct OptimizedModel {
    pub model: Model,
    pub hints: Vec<Hints>,
    pub tune: Tune,
    /// Bank-factor per arena slot. `arena_bank.get(&slot_id)` returns
    /// the number of independent BRAMs to emit for that slot — 1
    /// (default, single byte-wide BRAM) or `P_ic` (e.g. 4, when a
    /// consumer needs ic-parallel reads). Populated by `arena::run`.
    pub arena_bank: std::collections::BTreeMap<u8, u8>,
}

impl OptimizedModel {
    pub fn hints_for(&self, layer_idx: usize) -> &Hints {
        &self.hints[layer_idx]
    }
}

/// Run every analysis pass and produce an [`OptimizedModel`].
///
/// Pipeline order:
///   1. Build the rlx-ir Graph and verify it (free correctness).
///   2. Per-layer local analyses (fold_zero_zp, ternary_fast_path,
///      shared_requant, parallelism — independent, any order).
///   3. `fuse_conv_relu` — pattern across pairs of adjacent layers.
///      Marks the relu's `elided`, marks the upstream conv's
///      `fuses_relu`. Saves one kernel + one BRAM per match.
///   4. `arena_plan` — liveness-based BRAM slot assignment. Sequential
///      execution turns this into clean ping-pong.
pub fn optimize(model: &Model, tune: &Tune) -> OptimizedModel {
    // ── Step 1: build IR + verify ───────────────────────────────────
    let ir = crate::ir::to_graph(model);
    let errors = rlx_ir::verify::verify(&ir.graph);
    if !errors.is_empty() {
        let msg = errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n  ");
        panic!("rlx-fpga: IR verifier rejected the model graph:\n  {msg}");
    }

    // ── Step 2: per-layer local analyses ────────────────────────────
    let mut hints: Vec<Hints> = Vec::with_capacity(model.layers.len());
    for layer in &model.layers {
        let mut h = Hints {
            requant_precision: tune.requant_precision,
            parallelism: parallelism::layer_parallelism(layer, tune.parallelism),
            ic_parallelism: 1, // set below conditional on ternary_fast_path
            ..Hints::default()
        };

        if tune.fold_zero_zp && fold_zero_zp::layer_has_zero_zps(layer) {
            h.fast_mac = true;
        }
        if tune.ternary_fast_path && ternary_fast_path::is_ternary(layer) {
            h.ternary_fast_path = true;
            // ic-parallelism is gated on ternary_fast_path in the
            // first cut (the kernel's mux-tree absorbs the P_ic-wide
            // partials cleanly when every weight is ±1/0/-2).
            h.ic_parallelism = parallelism::layer_ic_parallelism(layer, tune.ic_parallelism);
        }
        if tune.shared_requant
            && let Some(pair) = shared_requant::uniform_requant(layer)
        {
            h.shared_requant = Some(pair);
        }
        if tune.bram_clock_enable {
            h.bram_clock_enable = true;
        }

        hints.push(h);
    }

    // ── Step 3: pattern fusion across adjacent layers ───────────────
    if tune.fuse_conv_relu {
        fuse_conv_relu::run(model, &mut hints);
    }

    // ── Step 4: liveness-based BRAM arena ───────────────────────────
    let arena_bank = if tune.arena_plan {
        arena::run(model, &mut hints)
    } else {
        std::collections::BTreeMap::new()
    };

    OptimizedModel {
        model: model.clone(),
        hints,
        tune: *tune,
        arena_bank,
    }
}

/// Convenience: run `optimize` with the default `Tune` (Precision preset).
pub fn optimize_default(model: &Model) -> OptimizedModel {
    optimize(model, &Tune::default())
}

/// Pretty-print a one-liner summary of how many layers each pass
/// activated on. Useful for tests and for the emit binary's stdout.
pub fn summary(opt: &OptimizedModel) -> String {
    let n = opt.hints.len();
    let fast_mac = opt.hints.iter().filter(|h| h.fast_mac).count();
    let ternary = opt.hints.iter().filter(|h| h.ternary_fast_path).count();
    let shared_r = opt
        .hints
        .iter()
        .filter(|h| h.shared_requant.is_some())
        .count();
    let bram_en = opt.hints.iter().filter(|h| h.bram_clock_enable).count();
    let par_eligible = opt.hints.iter().filter(|h| h.parallelism > 1).count();
    let max_p = opt.hints.iter().map(|h| h.parallelism).max().unwrap_or(1);
    let fused = opt.hints.iter().filter(|h| h.fuses_relu).count();
    let elided = opt.hints.iter().filter(|h| h.elided).count();
    let arena_slots: std::collections::BTreeSet<u8> = opt
        .hints
        .iter()
        .flat_map(|h| [h.bram_slot_in, h.bram_slot_out])
        .flatten()
        .collect();
    format!(
        "passes: fast_mac={fast_mac}/{n}  ternary={ternary}/{n}  \
         shared_requant={shared_r}/{n}  bram_en={bram_en}/{n}  \
         P_layers={par_eligible}/{n} (max P={max_p})  \
         fuse_conv_relu={fused} (elided={elided})  \
         arena={} slots  ({})",
        arena_slots.len(),
        opt.tune
    )
}

/// Helpers shared between passes — `Conv2d`/`Dense` fields exposed by
/// pattern matching once instead of in three places.
pub(crate) fn conv_dense_zps(layer: &Layer) -> Option<(i32, i32)> {
    match layer {
        Layer::Conv2d { x_zp, w_zp, .. } | Layer::Dense { x_zp, w_zp, .. } => Some((*x_zp, *w_zp)),
        _ => None,
    }
}

pub(crate) fn conv_dense_requant(layer: &Layer) -> Option<&[(i32, i32)]> {
    match layer {
        Layer::Conv2d { requant, .. } | Layer::Dense { requant, .. } => Some(requant),
        _ => None,
    }
}
