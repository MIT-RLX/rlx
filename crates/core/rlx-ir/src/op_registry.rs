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

//! Op registry — pluggable, trait-based extension point for custom ops.
//!
//! The built-in `Op` enum in `op.rs` is closed: every variant is hard-
//! coded, and adding a new op means editing every match in every
//! backend, the optimizer, and autodiff. That's tolerable for the
//! ~50 core ML ops the workspace already ships. It is *not* tolerable
//! for research ops (Sparse-LU, FFT, eigensolve, photonic mode solver,
//! Krylov iterations) — those need to land without forking the
//! compiler.
//!
//! This registry is the IR-level extension surface those ops plug
//! into. A user implements [`OpExtension`] for their op type, registers
//! the impl with [`global_registry`], and then builds graphs containing
//! `Op::Custom { name, num_inputs, attrs }` nodes that the rest of the
//! pipeline dispatches via lookup:
//!
//!   - shape inference: `OpExtension::infer_shape`
//!   - autodiff (rlx-opt):     `OpExtension::vjp`
//!   - CPU execution (rlx-cpu): a *separate* per-backend `CpuKernel`
//!     trait — see `rlx-cpu/src/op_registry.rs`. Backends own their
//!     own kernel registries to keep `rlx-ir` portable (no SIMD,
//!     BLAS, or buffer-layout types reach this crate).
//!
//! ## Threading
//!
//! Registration is read-mostly (typically once at startup). Reads
//! during graph compilation use the read-half of an `RwLock`. The
//! returned `Arc<dyn OpExtension>` outlives the lookup so the lock
//! is released immediately.
//!
//! ## Stable identity vs. attributes
//!
//! `OpExtension::name` is the stable string key. Per-instance knobs
//! that vary at the call site (e.g. an FFT's direction, a SparseLU's
//! reordering strategy) ride on `Op::Custom::attrs` as opaque bytes;
//! `infer_shape` and `vjp` receive them and decode as the impl sees
//! fit. This matches XLA's `HloCustomCall` / JAX's `jax.ffi`.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::{Graph, Node, NodeId, Shape};

/// Mutable context handed to a custom op's vmap rule. The vmap pass
/// has already lifted each input to either `[batch_size, *original]`
/// (for batched inputs) or left it unchanged (for shared / broadcast
/// inputs). The rule returns the lifted output NodeId in the new
/// graph.
pub struct VmapContext<'a> {
    /// Per-input NodeIds in the output graph. Use `is_batched[i]` to
    /// tell whether the input has the leading batch axis or not.
    pub lifted_inputs: &'a [NodeId],
    /// Per-input flag: `is_batched[i] == true` ⇒ `lifted_inputs[i]`
    /// has shape `[batch_size, *original]`; `false` ⇒ original shape.
    pub is_batched: &'a [bool],
    /// Batch size.
    pub batch_size: usize,
    /// The output (vmap'd) graph being built.
    pub out: &'a mut Graph,
}

/// Mutable context handed to a custom op's JVP method. Mirror of
/// [`VjpContext`] for forward-mode AD: the impl receives the primal
/// node and the per-input tangent NodeIds (already in the JVP graph),
/// and returns a tangent NodeId for the op's output. `None` slots in
/// `tangents` mean a symbolic-zero tangent for that input.
pub struct JvpContext<'a> {
    /// Per-input tangent NodeIds in the JVP graph, length = num_inputs.
    /// `None` ⇒ that input has a symbolic-zero tangent.
    pub tangents: &'a [Option<NodeId>],
    /// Forward → JVP NodeId map. `fwd_map[&forward_id]` gives the
    /// mirrored primal NodeId in the JVP graph (handy for reading
    /// primal values into the tangent rule).
    pub fwd_map: &'a HashMap<NodeId, NodeId>,
    /// The JVP graph being built.
    pub bwd: &'a mut Graph,
}

/// Mutable context handed to a custom op's VJP method. Lets the op
/// emit gradient subgraph nodes via the same builder API the built-in
/// VJP rules use, and resolve forward-graph inputs to their backward-
/// graph equivalents via [`fwd_map`](VjpContext::fwd_map).
pub struct VjpContext<'a> {
    /// Upstream gradient node (in the backward graph) for the op's
    /// output. Already shape-matched to the forward output.
    pub upstream: NodeId,
    /// Forward → backward NodeId map. Use `fwd_map[&node.inputs[i]]`
    /// to get the backward-graph node corresponding to forward input
    /// `i` of this op.
    pub fwd_map: &'a HashMap<NodeId, NodeId>,
    /// The backward graph being built. Call its builder methods
    /// (`bwd.binary`, `bwd.activation`, `bwd.matmul`, etc.) to emit
    /// gradient nodes.
    pub bwd: &'a mut Graph,
}

/// Trait a custom op implements to plug into the IR-level pipeline.
///
/// The impl is registered once with [`global_registry`]; thereafter
/// `Op::Custom { name, .. }` nodes naming this impl are dispatched
/// through it during shape inference and autodiff.
///
/// CPU/Metal/CUDA execution is **not** part of this trait — backend
/// kernels live in per-backend registries (see e.g.
/// `rlx-cpu/src/op_registry.rs`). The split keeps `rlx-ir` portable
/// and lets a custom op support a subset of backends honestly,
/// instead of having silent-no-op fallbacks for the rest.
pub trait OpExtension: Send + Sync {
    /// Stable string identifier. Used as the registry key and as the
    /// `name` field on `Op::Custom { name, .. }`.
    fn name(&self) -> &str;

    /// Number of tensor inputs this op takes. Frozen per impl; ops
    /// with variable arity should register multiple impls or encode
    /// the variance in `attrs`.
    fn num_inputs(&self) -> usize;

    /// Compute the output shape (and dtype) given the input shapes
    /// and the per-instance `attrs` blob. Called once at graph build
    /// time by `Graph::custom_op`.
    fn infer_shape(&self, inputs: &[&Shape], attrs: &[u8]) -> Shape;

    /// VJP rule. Default: non-differentiable — returns `vec![]`,
    /// meaning autodiff drops gradients on the floor for this op's
    /// inputs. Override to emit a gradient subgraph.
    ///
    /// Returns `(input_index, grad_node_id)` pairs; inputs not listed
    /// receive no gradient (matches the built-in VJP convention).
    fn vjp(&self, _node: &Node, _ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        Vec::new()
    }

    /// JVP rule (forward-mode AD). Default: not implemented — the JVP
    /// pass panics if it reaches this op with a non-zero input tangent
    /// (matching the reverse pass's "no silent miscompute" policy).
    /// Override to push tangents forward through this op.
    ///
    /// Returns the tangent NodeId for the op's output, or `None` if the
    /// output tangent is symbolically zero (e.g., all inputs had zero
    /// tangents and the op's local Jacobian doesn't generate one).
    fn jvp(&self, _node: &Node, _ctx: &mut JvpContext) -> Option<NodeId> {
        None
    }

    /// vmap rule. Default: no rule registered — the vmap pass panics
    /// with a clear message. Override to lift this op through a
    /// leading batch axis. Return `Some(node_id)` for the lifted
    /// output node; `None` signals "no rule" and is treated like the
    /// default.
    fn vmap(&self, _node: &Node, _ctx: &mut VmapContext) -> Option<NodeId> {
        None
    }
}

/// Global registry. Read-mostly: backed by `RwLock` over a name-keyed
/// `HashMap`.
pub struct OpRegistry {
    ops: RwLock<HashMap<String, Arc<dyn OpExtension>>>,
}

impl OpRegistry {
    pub fn new() -> Self {
        Self {
            ops: RwLock::new(HashMap::new()),
        }
    }

    /// Register an op extension. The user is expected to register
    /// each op exactly once at startup. Re-registering the same name
    /// replaces the previous entry and prints a one-line warning to
    /// stderr — silent overwrites mask honest mistakes (test fixtures
    /// stomping on each other, two libraries claiming the same name)
    /// that are worse to debug than the warning is to ignore.
    pub fn register(&self, op: Arc<dyn OpExtension>) {
        let name = op.name().to_string();
        let mut g = self.ops.write().unwrap();
        if g.contains_key(&name) {
            eprintln!(
                "rlx-ir: OpExtension '{name}' was already registered — \
                 replacing the previous entry"
            );
        }
        g.insert(name, op);
    }

    pub fn lookup(&self, name: &str) -> Option<Arc<dyn OpExtension>> {
        self.ops.read().unwrap().get(name).cloned()
    }

    pub fn list(&self) -> Vec<String> {
        self.ops.read().unwrap().keys().cloned().collect()
    }
}

impl Default for OpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-wide default registry. Lazily initialized on first access.
pub fn global_registry() -> &'static OpRegistry {
    static REGISTRY: OnceLock<OpRegistry> = OnceLock::new();
    REGISTRY.get_or_init(OpRegistry::new)
}

/// Convenience: register an op with the global registry.
pub fn register_op(op: Arc<dyn OpExtension>) {
    global_registry().register(op);
}

/// Convenience: look up an op in the global registry.
pub fn lookup_op(name: &str) -> Option<Arc<dyn OpExtension>> {
    global_registry().lookup(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape};

    struct DummyOp;
    impl OpExtension for DummyOp {
        fn name(&self) -> &str {
            "dummy"
        }
        fn num_inputs(&self) -> usize {
            1
        }
        fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
            inputs[0].clone()
        }
    }

    #[test]
    fn register_and_lookup() {
        let reg = OpRegistry::new();
        reg.register(Arc::new(DummyOp));
        let op = reg.lookup("dummy").expect("should find");
        assert_eq!(op.name(), "dummy");
        assert_eq!(op.num_inputs(), 1);
        let s = Shape::new(&[2, 3], DType::F32);
        let out = op.infer_shape(&[&s], &[]);
        assert_eq!(out, s);
    }

    #[test]
    fn vjp_default_is_empty() {
        let d = DummyOp;
        let mut bwd = Graph::new("b");
        let map = HashMap::new();
        let upstream = bwd.input("u", Shape::new(&[1], DType::F32));
        let node = bwd.nodes()[upstream.0 as usize].clone();
        let mut ctx = VjpContext {
            upstream,
            fwd_map: &map,
            bwd: &mut bwd,
        };
        let grads = d.vjp(&node, &mut ctx);
        assert!(grads.is_empty());
    }
}
