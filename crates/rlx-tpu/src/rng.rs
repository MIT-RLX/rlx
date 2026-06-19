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

//! In-graph RNG (`Op::RngNormal` / `Op::RngUniform`) helpers for TPU lowering.

use rlx_ir::{Graph, Op, RngBackend, RngOptions};

/// True when the graph contains native RLX random ops.
pub fn graph_has_in_graph_rng(graph: &Graph) -> bool {
    graph
        .nodes()
        .iter()
        .any(|n| matches!(n.op, Op::RngNormal { .. } | Op::RngUniform { .. }))
}

/// True when execution will call XLA `rng` (not the compile-time zero fill).
pub fn uses_xla_native_rng(graph: &Graph, rng: RngOptions) -> bool {
    graph_has_in_graph_rng(graph) && rng.backend != RngBackend::Zero
}

/// Print a one-time-per-executable warning before the first forward pass.
pub fn warn_xla_rng_on_execute(warned: &mut bool) {
    if *warned {
        return;
    }
    *warned = true;
    eprintln!(
        "rlx-tpu: executing Op::RngNormal/Op::RngUniform via native XLA `rng` — \
         not bit-identical to RLX Philox or ONNX Runtime CPU. For Philox/Ort parity \
         use Device::Cpu (or Device::Metal/Cuda/Rocm/wgpu host-fill). \
         RngBackend::Zero fills zeros at compile time."
    );
}
