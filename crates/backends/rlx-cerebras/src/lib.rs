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

//! RLX Cerebras backend — per-graph CSL synthesis for the Wafer-Scale Engine.
//!
//! Pipeline (mirroring `rlx-fpga` in shape — emit source, validate against a
//! Rust oracle — but the target is the Cerebras SDK *fabric simulator*, which
//! lets us close the loop and actually run the emitted program):
//!
//! ```text
//!   rlx-ir Graph
//!     → rlx-cerebras::model    (recognize the supported subgraph, read shapes)
//!     → rlx-cerebras::codegen  (emit layout.csl + pe_program.csl + run.py + commands.sh)
//!     → cslc --memcpy + cs_python run.py   (external; Cerebras SDK container, Linux host)
//! ```
//!
//! ## Why CSL (and not the PyTorch / PJRT paths)
//!
//! Cerebras has three ingestion surfaces: the hosted **inference API**
//! (serves their models only — not a compute backend), the **PyTorch
//! `cstorch` / PJRT-StableHLO** path (needs a CS-2/CS-3 appliance), and the
//! **SDK / CSL** path. Only CSL ships a *cycle-accurate fabric simulator*
//! that runs without wafer hardware, so it is the one path RLX can target
//! and validate end-to-end on commodity machines. CSL is a C-like dataflow
//! language: each program is a rectangle of processing elements (PEs), each
//! PE runs its own kernel, and data crosses the fabric over routed "colors".
//!
//! ## Status (milestone 1)
//!
//! * **Single op, single PE.** [`model::Layer::MatMul`] → one PE, host data
//!   streamed in/out via the `memcpy` library, exactly the shape of the SDK's
//!   `gemv-03-memcpy` tutorial.
//! * **Scalar kernel.** The emitted matmul is plain scalar loops — obviously
//!   correct, the safest thing to emit before `cslc` validation. The
//!   DSD / `@fmacs` vectorized form and **multi-PE tiling** (where the
//!   wafer-scale perf actually lives — the north star) are the next
//!   milestones; single-PE is only a correctness stepping stone.
//! * **Not yet wafer- or `cslc`-validated.** The Rust [`reference`] oracle and
//!   the artifact structure are unit-tested here; compiling the CSL with
//!   `cslc` and running it on the simulator requires the SDK container on a
//!   Linux host (e.g. ALCF), and is the validation step that closes the loop.
//!
//! ## What's here
//!
//! * `csl`       — pure-Rust CSL source writer (buffer + indent), the analog
//!                 of `rlx_fpga::verilog::V`.
//! * `model`     — lightweight [`model::Layer`] / [`model::Model`] description
//!                 plus [`model::Model::from_graph`] (rlx-ir → model).
//! * `codegen`   — emit the `.csl` / `run.py` / `commands.sh` artifacts.
//! * `reference` — Rust forward pass; the parity oracle for the emitted CSL.

pub mod codegen;
pub mod csl;
pub mod model;
pub mod reference;

pub use codegen::{Artifact, collect_artifacts, emit_model};
pub use model::{Layer, Model};
