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

//! RLX FPGA backend — per-graph datapath synthesis.
//!
//! Pipeline (mirroring rlx-cuda / rlx-rocm in shape, rlx-cortexm in ethos):
//!
//! ```text
//!   rlx-ir Graph
//!     → rlx-opt (existing fusion / INT8 quant)
//!     → rlx-fpga::schedule  (tile to BRAM banks, allocate DSPs, build FSM)
//!     → rlx-fpga::codegen   (emit SystemVerilog + .mem weight files + .lpf)
//!     → yosys + nextpnr + ecppack/icepack/...   (external; not in this crate)
//! ```
//!
//! ## Quantization
//!
//! `rlx-cortexm` uses a single f32 multiplier in the requant epilogue
//! because the M4F has an FPU. FPGA fabric does not (and a soft-FPU
//! costs hundreds of LUTs per requantize, killing throughput), so this
//! crate ports the requant to **integer-only Q0.31** — the same shape
//! TFLite Micro / CMSIS-NN / gemmlowp use:
//!
//! * `quantize_multiplier(M_real)` → `(M0 : i32 in [2^30, 2^31), shift : i32)`
//! * `srdhm(acc, M0)` — saturating-rounding-doubling-high-multiply
//! * `rdpot(prod, shift)` — rounding-divide-by-power-of-two
//!
//! This is **not bit-exact** with the cortexm path (which uses f32
//! arithmetic in the requant), but it *is* bit-exact across:
//!
//!   `rlx-fpga::reference` (Rust)  ↔  emitted Verilog  ↔  silicon
//!
//! Per-image, the cortexm INT8 prediction and the FPGA INT8 prediction
//! agree on the same label for ≥99 % of the MNIST test set; the
//! per-pixel logits differ by at most ±1 ulp at each requant.
//!
//! ## What's here
//!
//! * `quant`     — integer-only Q0.31 requant (Rust reference).
//! * `pack`      — pack / unpack helpers for `weight_bits ∈ {2, 4, 8}`,
//!                 layout-compatible with `rlx_cortexm::quant::read_weight`.
//!                 Ternary (`bits = 2`) supported end-to-end: reference
//!                 forward pass + Verilog emission.
//! * `verilog`   — pure-Rust SystemVerilog writer (synthesizable subset).
//! * `codegen`   — one Rust function per op, each emits a parameterized
//!                 SV module.  `codegen::top` wires a `Model` into a
//!                 single self-contained `top.sv`. `codegen::weight_unpack`
//!                 is the shared byte → i32 extractor used by every
//!                 conv/dense kernel.
//! * `model`     — TinyConv-MNIST graph description (shapes + the layer
//!                 sequence; matches `rlx-cortexm::model`).
//! * `reference` — TinyConv-MNIST forward pass using `quant` instead of
//!                 the f32 path. The parity oracle for emitted Verilog.
//! * `weights`   — pulls the cortexm INT8 blob (CONV1_W, biases, mults,
//!                 etc.) and turns it into pure-integer Q0.31 multiplier
//!                 tables for both `reference` and the .mem files.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod codegen;
pub mod estimate;
pub mod ir;
pub mod model;
pub mod pack;
pub mod passes;
pub mod quant;
pub mod reference;
pub mod tune;
pub mod verilog;
pub mod weights;
