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

//! Verilog codegen for an `rlx_fpga::model::Model`.
//!
//! Each op type has its own emitter (`bram.rs`, `requant.rs`,
//! `conv2d.rs`, `dense.rs`, `relu.rs`, `maxpool.rs`, `argmax.rs`).
//! `top.rs` walks the model and wires the per-layer modules together
//! through a chain of intermediate BRAMs.
//!
//! The user-facing entry is [`emit_model`]: it writes a self-contained
//! `hw/<model>/` tree containing `top.sv`, `tb.sv` (a Verilator-style
//! testbench), one .sv per layer, the BRAM/requant primitives, and a
//! `weights/` directory of `.mem` files for `$readmemh`.

use std::fs;
use std::io;
use std::path::Path;

use crate::model::Model;
use crate::passes::{OptimizedModel, optimize, optimize_default};
use crate::tune::Tune;

pub mod argmax;
pub mod bram;
pub mod conv2d;
pub mod conv2d_parallel;
pub mod dense;
pub mod maxpool;
pub mod relu;
pub mod requant;
pub mod top;
pub mod weight_unpack;

/// One generated artifact (a `.sv` source or a `.mem` data file). All
/// paths are *relative* to the model's `hw/<name>/` directory.
pub struct Artifact {
    pub rel_path: String,
    pub content: String,
}

/// Per-layer codegen output: the layer's own SV module plus any weight
/// / bias / requant `.mem` data files it needs.
pub struct LayerArtifacts {
    /// Module name in SystemVerilog (e.g. `"conv1_kernel"`).
    pub module_name: String,
    /// Instance name when wired up by `top.sv` (e.g. `"u_conv1"`).
    pub instance_name: String,
    /// Length (in i8 elements) of this layer's output buffer.
    pub out_len: usize,
    /// The `.sv` source for this layer's module.
    pub sv: Artifact,
    /// `.mem` data files for `$readmemh` (weights / bias / requant).
    pub mems: Vec<Artifact>,
}

/// Emit a complete hardware tree for `model` under `out_dir`, using
/// the default `Tune` (the `Precision` preset). For full control, see
/// [`emit_model_tuned`] and [`emit_optimized`].
pub fn emit_model(model: &Model, out_dir: &Path) -> io::Result<()> {
    emit_optimized(&optimize_default(model), out_dir)
}

/// Emit `model` tuned for `tune` (e.g. `Tune::for_target(OptTarget::Energy)`).
pub fn emit_model_tuned(model: &Model, tune: &Tune, out_dir: &Path) -> io::Result<()> {
    emit_optimized(&optimize(model, tune), out_dir)
}

/// Emit a previously-optimized model. The tree layout produced is:
///
/// ```text
///   <out_dir>/
///     primitives/
///       block_rom.sv         — synchronous-read ROM ($readmemh init)
///       block_ram.sv         — synchronous-read R/W BRAM
///       requant_q31.sv       — combinational Q0.31 epilogue
///       requant_q15.sv       — combinational Q0.15 epilogue (smaller)
///       weight_unpack.sv     — byte → i32 extractor for {2,4,8}-bit packs
///     layers/
///       <layer_name>.sv      — one per Conv2d/Dense/ReLU/MaxPool/Argmax
///     weights/
///       <layer_name>_*.mem   — packed weights / biases / requant tables
///     top.sv                 — controller FSM + instances + scratch BRAMs
///     tb.sv                  — image-driven testbench (Verilator)
/// ```
pub fn emit_optimized(opt: &OptimizedModel, out_dir: &Path) -> io::Result<()> {
    fs::create_dir_all(out_dir.join("primitives"))?;
    fs::create_dir_all(out_dir.join("layers"))?;
    fs::create_dir_all(out_dir.join("weights"))?;

    let arts = collect_artifacts_opt(opt);
    for a in &arts {
        let path = out_dir.join(&a.rel_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &a.content)?;
    }
    Ok(())
}

/// Build every artifact for `model` in memory using the default `Tune`.
/// Convenience wrapper around [`collect_artifacts_opt`].
pub fn collect_artifacts(model: &Model) -> Vec<Artifact> {
    collect_artifacts_opt(&optimize_default(model))
}

/// Build every artifact for an optimized model in memory.
pub fn collect_artifacts_opt(opt: &OptimizedModel) -> Vec<Artifact> {
    // Shared primitives. Both requant flavors are always emitted —
    // they're small, and emitting both means a layer can pick either.
    let mut out: Vec<Artifact> = vec![
        Artifact {
            rel_path: "primitives/block_rom.sv".into(),
            content: bram::emit_block_rom(),
        },
        Artifact {
            rel_path: "primitives/block_ram.sv".into(),
            content: bram::emit_block_ram(),
        },
        Artifact {
            rel_path: "primitives/requant_q31.sv".into(),
            content: requant::emit_requant_q31(),
        },
        Artifact {
            rel_path: "primitives/requant_q15.sv".into(),
            content: requant::emit_requant_q15(),
        },
    ];
    out.push(Artifact {
        rel_path: "primitives/weight_unpack.sv".into(),
        content: weight_unpack::emit(),
    });

    // Per-layer modules + their .mem files. Elided layers (e.g. Relus
    // that were fused into the upstream Conv2d) emit no kernel module
    // and are filtered out before top.sv stitches things together.
    let mut layers = Vec::with_capacity(opt.model.layers.len());
    for (idx, layer) in opt.model.layers.iter().enumerate() {
        let hints = opt.hints_for(idx);
        if hints.elided {
            continue;
        }
        let la = match layer {
            crate::model::Layer::Conv2d { .. } => {
                // The parallel kernel handles both oc-parallel (P>1) and
                // ic-parallel (P_ic>1). Scalar kernel is for P=1, P_ic=1.
                if hints.parallelism > 1 || hints.ic_parallelism > 1 {
                    conv2d_parallel::emit(layer, hints)
                } else {
                    conv2d::emit(layer, hints)
                }
            }
            crate::model::Layer::Dense { .. } => dense::emit(layer, hints),
            crate::model::Layer::Relu { .. } => relu::emit(layer),
            crate::model::Layer::MaxPool2d { .. } => maxpool::emit(layer),
            crate::model::Layer::Argmax { .. } => argmax::emit(layer),
        };
        out.push(la.sv);
        out.extend(la.mems);
        layers.push(LayerHandle {
            module_name: la.module_name,
            instance_name: la.instance_name,
            out_len: la.out_len,
            layer: layer.clone(),
            hints: hints.clone(),
        });
    }

    // Top-level glue. The Tune-banner makes it obvious from the top of
    // top.sv which configuration produced this tree.
    out.push(top::emit(&opt.model, &layers, &opt.tune, &opt.arena_bank));
    out.push(Artifact {
        rel_path: "tb.sv".into(),
        content: top::emit_tb(&opt.model),
    });

    out
}

/// Re-export for downstream code that wants to access hints directly.
pub use crate::passes::Hints as LayerHints;
pub use crate::passes::optimize as run_passes;
pub use crate::tune::{OptTarget as TuneTarget, Tune as TuneConfig};

/// Internal handle the top-level emitter uses to wire layer modules
/// together. Includes a clone of the layer (for shape / port-width
/// inspection in `top::emit`) and the strings produced by the kernel's
/// own emitter.
pub struct LayerHandle {
    pub module_name: String,
    pub instance_name: String,
    pub out_len: usize,
    pub layer: crate::model::Layer,
    pub hints: crate::passes::Hints,
}
