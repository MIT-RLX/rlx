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

//! Activation-scale calibration for post-training INT8 quantization.
//!
//! The runtime-side counterpart to `rlx_opt::quant_insert`. Compiles a
//! forward graph with calibration "tap" nodes wired in as outputs;
//! the caller runs one batch at a time (filling the input slots
//! between calls) and the [`Calibrator`] accumulates max-abs per tap.
//! At the end, `scales()` returns `max_abs / 127.0` per tap (clamped
//! up to `1e-6` to avoid division-by-zero on a flat tensor) — the
//! per-tensor scale that maps the calibration range into i8.
//!
//! Why max-abs and not e.g. the 99th percentile? Max-abs matches what
//! the cortexm Python trainer used to do (and what the Rust trainer
//! that replaced it does). It's symmetric (zero zero-point), maps
//! `[-max, +max] → [-127, 127]`, and gives the worst-case-correct
//! quantization for activations whose distributions are roughly
//! zero-centered. Percentile-based / KL-divergence calibration are
//! follow-ups for later.

use crate::arena::Arena;
use crate::thunk::{ThunkSchedule, compile_thunks, execute_thunks};
use rlx_ir::{Graph, NodeId};

/// Compiled calibration harness. The graph is owned by the caller —
/// we hold a reference and the compiled artifacts (arena + schedule).
/// The caller writes inputs and parameters into `arena_mut()` between
/// batches.
pub struct Calibrator<'g> {
    graph: &'g Graph,
    arena: Arena,
    sched: ThunkSchedule,
    /// `(tap_node_id, num_elements)` pairs — cached so each `step`
    /// doesn't re-walk the graph for shape info.
    taps: Vec<(NodeId, usize)>,
    /// Running max-abs per tap. Index aligns with the `taps` order
    /// the caller passed to `new`.
    max_abs: Vec<f32>,
}

impl<'g> Calibrator<'g> {
    /// Build a calibrator over `graph` that records max-abs at each
    /// `tap` after every `step()`. The graph must already have those
    /// taps in its `outputs` list (so the memory planner keeps their
    /// arena slots alive to end-of-execution); this constructor
    /// asserts the precondition.
    pub fn new(graph: &'g Graph, taps: Vec<NodeId>) -> Self {
        for &t in &taps {
            assert!(
                graph.outputs.contains(&t),
                "Calibrator: tap {t} must be in graph.outputs so its slot \
                 survives the run; add it via graph.set_outputs(…)"
            );
        }
        let plan = rlx_opt::memory::plan_memory(graph);
        let arena = Arena::from_plan(plan);
        let sched = compile_thunks(graph, &arena);
        let n = taps.len();
        let taps_with_len: Vec<(NodeId, usize)> = taps
            .into_iter()
            .map(|t| {
                let len = graph.node(t).shape.num_elements().unwrap_or(0);
                (t, len)
            })
            .collect();
        Self {
            graph,
            arena,
            sched,
            taps: taps_with_len,
            max_abs: vec![0.0; n],
        }
    }

    /// Mutable arena access — for writing inputs/params before each
    /// `step()` and (typically once at startup) for filling
    /// `Op::Constant` data via `rlx_runtime`'s loader.
    pub fn arena_mut(&mut self) -> &mut Arena {
        &mut self.arena
    }

    /// Read-only arena view — for reading the tap values manually if
    /// the caller wants something fancier than max-abs.
    pub fn arena(&self) -> &Arena {
        &self.arena
    }

    /// Run one forward batch, then update each tap's running max-abs.
    pub fn step(&mut self) {
        execute_thunks(&self.sched, self.arena.raw_buf_mut());
        for ((tap, len), max) in self.taps.iter().zip(self.max_abs.iter_mut()) {
            let off = self.arena.byte_offset(*tap);
            unsafe {
                let p = self.arena.raw_buf().as_ptr().add(off) as *const f32;
                for i in 0..*len {
                    let v = (*p.add(i)).abs();
                    if v > *max {
                        *max = v;
                    }
                }
            }
        }
    }

    /// Per-tap max-abs accumulated so far (in input order).
    pub fn max_abs(&self) -> &[f32] {
        &self.max_abs
    }

    /// Per-tap scale = `max_abs / 127.0`, clamped up to `1e-6`.
    /// Use directly as the `scale` for `Op::Quantize` / `Op::Dequantize`
    /// or `rlx_opt::CalibrationEntry::per_tensor`.
    pub fn scales(&self) -> Vec<f32> {
        self.max_abs.iter().map(|m| (m / 127.0).max(1e-6)).collect()
    }

    /// Borrow the inner graph (for the caller to re-look-up NodeIds
    /// after compilation).
    pub fn graph(&self) -> &Graph {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::*;
    use rlx_ir::*;

    /// One-tap calibration over a trivial graph: tap = `x` itself.
    /// Hand-pack a couple of batches with known max-abs values and
    /// verify `scales()` reflects them.
    #[test]
    fn calibrator_tracks_max_abs_across_batches() {
        let f = DType::F32;
        let mut g = Graph::new("calib_demo");
        let x = g.input("x", Shape::new(&[4], f));
        // Identity-ish: the tap *is* the input. Adding a Relu so the
        // graph is non-trivial.
        let y = g.activation(Activation::Relu, x, Shape::new(&[4], f));
        g.set_outputs(vec![x, y]); // tap on `x` and `y`

        let mut cal = Calibrator::new(&g, vec![x, y]);
        // Batch 1: max-abs of x = 3.0; max-abs of y (Relu) = 3.0.
        write_into(cal.arena_mut(), x, &[-3.0, 1.0, -2.0, 0.5]);
        cal.step();
        // Batch 2: x's max-abs grows to 7.0; y's stays since negatives
        // get zeroed by Relu.
        write_into(cal.arena_mut(), x, &[-7.0, 0.0, -7.0, -2.0]);
        cal.step();
        // Batch 3: both grow.
        write_into(cal.arena_mut(), x, &[10.0, 0.0, 0.0, 5.0]);
        cal.step();

        let mx = cal.max_abs();
        assert!((mx[0] - 10.0).abs() < 1e-6, "x max_abs: {}", mx[0]);
        assert!((mx[1] - 10.0).abs() < 1e-6, "y max_abs: {}", mx[1]);

        let s = cal.scales();
        assert!((s[0] - 10.0 / 127.0).abs() < 1e-6);
        assert!((s[1] - 10.0 / 127.0).abs() < 1e-6);
    }

    fn write_into(arena: &mut Arena, id: NodeId, data: &[f32]) {
        let off = arena.byte_offset(id);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(off) as *mut f32;
            for (i, &v) in data.iter().enumerate() {
                *p.add(i) = v;
            }
        }
    }
}
