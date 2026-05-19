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

//! Specialised builders: SSM selective scan + space for future
//! exotic ops (plan #53).

use crate::shape::Dim;
use crate::{DType, Graph, NodeId, Op, Shape};

/// Handle to a multi-output [`Op::CustomFn`] built via
/// [`Graph::custom_fn_multi`]. Internally the op produces a flat 1-D
/// concatenated output; this handle remembers each sub-output's
/// offset + original shape so [`Self::output`] can materialize
/// component `i` lazily via `Op::Narrow` + `Op::Reshape`.
#[derive(Debug, Clone)]
pub struct MultiOutputHandle {
    /// NodeId of the wrapped Op::CustomFn (1-D F32, length =
    /// `Σ sub_shapes[i].num_elements`).
    pub source: NodeId,
    /// Original per-sub-output shapes (in declaration order).
    pub sub_shapes: Vec<Shape>,
    /// Per-sub-output start offsets into `source` (element-counted,
    /// not byte-counted).
    pub offsets: Vec<usize>,
}

impl MultiOutputHandle {
    /// Number of sub-outputs.
    pub fn n_outputs(&self) -> usize {
        self.sub_shapes.len()
    }

    /// Materialize sub-output `idx` as an outer-graph NodeId.
    /// Internally: `Op::Narrow(source, axis=0, start=offsets[idx],
    /// len=numel(sub_shapes[idx]))` → `Op::Reshape` back to the
    /// declared shape.
    pub fn output(&self, g: &mut Graph, idx: usize) -> NodeId {
        assert!(idx < self.sub_shapes.len(), "output index out of range");
        let sub = &self.sub_shapes[idx];
        let n_elems: usize = sub
            .dims()
            .iter()
            .map(|d| match d {
                Dim::Static(k) => *k,
                Dim::Dynamic(_) => panic!("dynamic sub-output dim"),
            })
            .product();
        let flat_shape = Shape::from_dims(&[Dim::Static(n_elems)], sub.dtype());
        let narrowed = g.add_node(
            Op::Narrow {
                axis: 0,
                start: self.offsets[idx],
                len: n_elems,
            },
            vec![self.source],
            flat_shape,
        );
        if sub.rank() == 1 {
            // Already the right shape.
            narrowed
        } else {
            let dims: Vec<i64> = sub
                .dims()
                .iter()
                .map(|d| match d {
                    Dim::Static(k) => *k as i64,
                    Dim::Dynamic(_) => unreachable!(),
                })
                .collect();
            g.add_node(Op::Reshape { new_shape: dims }, vec![narrowed], sub.clone())
        }
    }
}

impl Graph {
    /// Mamba-style selective scan: y = SSM(x, Δ, A, B, C).
    /// Inputs: x \[b,s,h\], delta \[b,s,h\], a \[h,n\], b \[b,s,n\], c \[b,s,n\].
    /// Output \[b,s,h\]. n is the state size.
    pub fn selective_scan(
        &mut self,
        x: NodeId,
        delta: NodeId,
        a: NodeId,
        b: NodeId,
        c: NodeId,
        state_size: usize,
        shape: Shape,
    ) -> NodeId {
        self.push(
            Op::SelectiveScan { state_size },
            vec![x, delta, a, b, c],
            shape,
            None,
        )
    }

    /// Gated DeltaNet linear-attention scan (Qwen3.5/3.6 trunk,
    /// Qwen3-Next, Kimi-Linear). See [`Op::GatedDeltaNet`] for the
    /// recurrence math. All five inputs are `f32`. Shapes:
    /// `q,k,v`: `[b, s, h_v, n]`; `g,beta`: `[b, s, h_v]`. Output:
    /// `[b, s, h_v, n]`. State is implicit (reset per batch).
    /// Caller is responsible for L2-normalizing `q`/`k` and for
    /// GQA-repeating `k` to match `h_v` when `h_k < h_v`.
    pub fn gated_delta_net(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        g: NodeId,
        beta: NodeId,
        state_size: usize,
        shape: Shape,
    ) -> NodeId {
        self.push(
            Op::GatedDeltaNet { state_size },
            vec![q, k, v, g, beta],
            shape,
            None,
        )
    }

    /// Bounded scan returning the final carry. Body must have exactly
    /// one `Op::Input` (the carry) and one output, both same shape as
    /// `init`. Output shape matches `init`.
    pub fn scan(&mut self, init: NodeId, body: Graph, length: u32) -> NodeId {
        let init_shape = self.shape(init).clone();
        self.push(
            Op::Scan {
                body: Box::new(body),
                length,
                save_trajectory: false,
                num_bcast: 0,
                num_xs: 0,
                num_checkpoints: 0,
            },
            vec![init],
            init_shape,
            None,
        )
    }

    /// Bounded scan with recursive checkpointing for memory-bounded
    /// backward AD. Equivalent to [`Self::scan`] for the forward
    /// computation, but during backward only `num_checkpoints` carry
    /// values are cached; intermediate carries are recomputed via the
    /// body. Memory: `O(num_checkpoints · carry_size)`. Time: forward
    /// unchanged; backward `O(length)` (segment-cached).
    ///
    /// The AD pre-pass propagates `num_checkpoints` into the rewritten
    /// trajectory-saving Scan and into the emitted ScanBackward, so a
    /// single call to [`crate::Graph::scan_checkpointed`] is enough
    /// to enable the memory bound across the whole forward+backward
    /// pipeline.
    pub fn scan_checkpointed(
        &mut self,
        init: NodeId,
        body: Graph,
        length: u32,
        num_checkpoints: u32,
    ) -> NodeId {
        assert!(
            num_checkpoints > 0 && num_checkpoints <= length,
            "scan_checkpointed: num_checkpoints={num_checkpoints} \
             must be in 1..=length={length}"
        );
        let init_shape = self.shape(init).clone();
        self.push(
            Op::Scan {
                body: Box::new(body),
                length,
                save_trajectory: false,
                num_bcast: 0,
                num_xs: 0,
                num_checkpoints,
            },
            vec![init],
            init_shape,
            None,
        )
    }

    /// Bounded scan with broadcast and per-step inputs.
    ///
    /// Body `Op::Input`s in NodeId order: `[carry, bcast_0..bcast_{B-1},
    /// x_t_0..x_t_{X-1}]`. Bcast inputs keep their natural shape (the
    /// CPU executor fills them once before the scan loop). xs\[i\] has
    /// shape `[length, *per_step]` and the body sees `xs[i][t]` per
    /// iteration. Output shape matches `init`.
    pub fn scan_with_bcasts_and_xs(
        &mut self,
        init: NodeId,
        bcasts: &[NodeId],
        xs: &[NodeId],
        body: Graph,
        length: u32,
    ) -> NodeId {
        let init_shape = self.shape(init).clone();
        let mut inputs = vec![init];
        inputs.extend_from_slice(bcasts);
        inputs.extend_from_slice(xs);
        self.push(
            Op::Scan {
                body: Box::new(body),
                length,
                save_trajectory: false,
                num_bcast: bcasts.len() as u32,
                num_xs: xs.len() as u32,
                num_checkpoints: 0,
            },
            inputs,
            init_shape,
            None,
        )
    }

    /// Bounded scan with per-step `xs` inputs returning the final carry.
    /// Body has `1 + xs.len()` Op::Inputs in NodeId construction order
    /// (first declared is the carry; the remaining match `xs` in order).
    /// Each `xs[i]` has shape `[length, *per_step_shape_i]`; the body
    /// sees a `per_step_shape_i` slice on iteration `t`.
    pub fn scan_with_xs(
        &mut self,
        init: NodeId,
        xs: &[NodeId],
        body: Graph,
        length: u32,
    ) -> NodeId {
        let init_shape = self.shape(init).clone();
        let mut inputs = vec![init];
        inputs.extend_from_slice(xs);
        self.push(
            Op::Scan {
                body: Box::new(body),
                length,
                save_trajectory: false,
                num_bcast: 0,
                num_xs: xs.len() as u32,
                num_checkpoints: 0,
            },
            inputs,
            init_shape,
            None,
        )
    }

    /// Reverse-mode AD companion to [`Self::scan`] /
    /// [`Self::scan_trajectory`]. Typically constructed by the
    /// autodiff pass, not by hand.
    ///
    /// `xs` is the list of per-step input tensors (must match the
    /// forward Op::Scan's xs in count, order, and per-step shape).
    /// Body_vjp's `1 + xs.len() + 1` Op::Inputs match the forward
    /// body's inputs plus a fresh `"d_output"` Input.
    pub fn scan_backward(
        &mut self,
        init: NodeId,
        trajectory: NodeId,
        upstream: NodeId,
        xs: &[NodeId],
        body_vjp: Graph,
        length: u32,
        save_trajectory: bool,
        out_shape: Shape,
    ) -> NodeId {
        self.scan_backward_with_checkpoints(
            init,
            trajectory,
            upstream,
            xs,
            body_vjp,
            length,
            save_trajectory,
            0,
            None,
            out_shape,
        )
    }

    /// Lower-level `scan_backward` with explicit checkpointing config.
    /// `num_checkpoints == 0` (default) means no checkpointing — the
    /// trajectory cache holds every step's carry. `0 < K < length`
    /// enables segment-cached recompute via `forward_body` (must be
    /// `Some`).
    #[allow(clippy::too_many_arguments)]
    pub fn scan_backward_with_checkpoints(
        &mut self,
        init: NodeId,
        trajectory: NodeId,
        upstream: NodeId,
        xs: &[NodeId],
        body_vjp: Graph,
        length: u32,
        save_trajectory: bool,
        num_checkpoints: u32,
        forward_body: Option<Graph>,
        out_shape: Shape,
    ) -> NodeId {
        let mut inputs = vec![init, trajectory, upstream];
        inputs.extend_from_slice(xs);
        self.push(
            Op::ScanBackward {
                body_vjp: Box::new(body_vjp),
                length,
                save_trajectory,
                num_xs: xs.len() as u32,
                num_checkpoints,
                forward_body: forward_body.map(Box::new),
            },
            inputs,
            out_shape,
            None,
        )
    }

    /// Per-step xs gradient companion to [`Self::scan_backward`].
    /// Same inputs and same `body_vjp` graph, plus an `xs_idx`
    /// selecting which body_vjp output to stack into the result.
    /// Output shape is `[length, *per_step_xs_shape]`.
    pub fn scan_backward_xs(
        &mut self,
        init: NodeId,
        trajectory: NodeId,
        upstream: NodeId,
        xs: &[NodeId],
        body_vjp: Graph,
        length: u32,
        save_trajectory: bool,
        xs_idx: u32,
        out_shape: Shape,
    ) -> NodeId {
        self.scan_backward_xs_with_checkpoints(
            init,
            trajectory,
            upstream,
            xs,
            body_vjp,
            length,
            save_trajectory,
            xs_idx,
            0,
            None,
            out_shape,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn scan_backward_xs_with_checkpoints(
        &mut self,
        init: NodeId,
        trajectory: NodeId,
        upstream: NodeId,
        xs: &[NodeId],
        body_vjp: Graph,
        length: u32,
        save_trajectory: bool,
        xs_idx: u32,
        num_checkpoints: u32,
        forward_body: Option<Graph>,
        out_shape: Shape,
    ) -> NodeId {
        let mut inputs = vec![init, trajectory, upstream];
        inputs.extend_from_slice(xs);
        self.push(
            Op::ScanBackwardXs {
                body_vjp: Box::new(body_vjp),
                length,
                save_trajectory,
                num_xs: xs.len() as u32,
                xs_idx,
                num_checkpoints,
                forward_body: forward_body.map(Box::new),
            },
            inputs,
            out_shape,
            None,
        )
    }

    /// User-defined sub-graph with optional override AD rules.
    /// JAX-shaped `custom_vjp` / `custom_jvp` — see [`Op::CustomFn`].
    ///
    /// `inputs.len()` must equal the number of `Op::Input` nodes in
    /// `fwd_body`. Output shape is inferred from `fwd_body`'s declared
    /// output. When supplied, `vjp_body` and `jvp_body` must follow the
    /// conventions documented on [`Op::CustomFn`] (special-named
    /// `"primal_output"` / `"d_output"` / `"tangent_*"` Inputs).
    pub fn custom_fn(
        &mut self,
        inputs: Vec<NodeId>,
        fwd_body: Graph,
        vjp_body: Option<Graph>,
        jvp_body: Option<Graph>,
    ) -> NodeId {
        let n_in = inputs.len();
        // Count fwd_body's primal Inputs (no special names — fwd has none).
        let fwd_inputs: usize = fwd_body
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Input { .. }))
            .count();
        assert_eq!(
            fwd_inputs, n_in,
            "custom_fn: fwd_body has {fwd_inputs} Op::Input(s); outer call \
             provides {n_in}. Counts must match.",
        );
        let fwd_out_id = fwd_body
            .outputs
            .first()
            .copied()
            .expect("custom_fn: fwd_body must declare exactly one output");
        let out_shape = fwd_body.node(fwd_out_id).shape.clone();

        if let Some(vjp) = vjp_body.as_ref() {
            let primal_count = vjp
                .nodes()
                .iter()
                .filter(|n| {
                    matches!(&n.op,
                    Op::Input { name } if name != "primal_output" && name != "d_output")
                })
                .count();
            assert_eq!(
                primal_count, n_in,
                "custom_fn: vjp_body has {primal_count} primal Op::Input(s) \
                 (excluding 'primal_output' / 'd_output'); expected {n_in}",
            );
            let has_primal_out = vjp
                .nodes()
                .iter()
                .any(|n| matches!(&n.op, Op::Input { name } if name == "primal_output"));
            let has_d_output = vjp
                .nodes()
                .iter()
                .any(|n| matches!(&n.op, Op::Input { name } if name == "d_output"));
            assert!(
                has_primal_out,
                "custom_fn: vjp_body must declare an Op::Input named 'primal_output'"
            );
            assert!(
                has_d_output,
                "custom_fn: vjp_body must declare an Op::Input named 'd_output'"
            );
            assert_eq!(
                vjp.outputs.len(),
                n_in,
                "custom_fn: vjp_body has {} outputs; expected {n_in} \
                 (one gradient per primal input)",
                vjp.outputs.len(),
            );
        }
        if let Some(jvp) = jvp_body.as_ref() {
            let primal_count = jvp
                .nodes()
                .iter()
                .filter(|n| {
                    matches!(&n.op,
                    Op::Input { name }
                        if !name.starts_with("tangent_") && name != "primal_output")
                })
                .count();
            assert_eq!(
                primal_count, n_in,
                "custom_fn: jvp_body has {primal_count} primal Op::Input(s) \
                 (excluding 'primal_output' / 'tangent_*'); expected {n_in}",
            );
            for i in 0..n_in {
                let want = format!("tangent_{i}");
                let has = jvp
                    .nodes()
                    .iter()
                    .any(|n| matches!(&n.op, Op::Input { name } if name == &want));
                assert!(
                    has,
                    "custom_fn: jvp_body must declare an Op::Input named '{want}'"
                );
            }
            assert_eq!(
                jvp.outputs.len(),
                1,
                "custom_fn: jvp_body has {} outputs; expected 1 (output tangent)",
                jvp.outputs.len(),
            );
        }

        self.push(
            Op::CustomFn {
                fwd_body: Box::new(fwd_body),
                vjp_body: vjp_body.map(Box::new),
                jvp_body: jvp_body.map(Box::new),
                num_inputs: n_in as u32,
            },
            inputs,
            out_shape,
            None,
        )
    }

    /// Multi-output `custom_fn` via the **concat-with-Narrow** design:
    /// rewrites `fwd_body` to flatten + concat its `K` declared outputs
    /// into a single 1-D F32 output, wraps that as [`Op::CustomFn`],
    /// and returns a [`MultiOutputHandle`] the caller uses to extract
    /// each sub-output via `Op::Narrow` + `Op::Reshape`.
    ///
    /// Per PLAN line 484, this avoids rewriting rlx's "1 Op = 1 output"
    /// IR contract: the wrapped Op::CustomFn still has one output (the
    /// flat concat), and `MultiOutputHandle::output(g, i)` materializes
    /// component `i` lazily on the outer graph.
    ///
    /// Constraints (MVP):
    /// - All sub-outputs must be `DType::F32`. Tuples-of-mixed-dtype
    ///   need either a per-dtype split or a future tuple-type
    ///   extension.
    /// - All sub-output shapes must be statically known (no
    ///   `Dim::Dynamic`).
    /// - `vjp_body` / `jvp_body` aren't yet rewritten through the
    ///   concat — caller must provide bodies that already expect
    ///   the flat-concat output convention if they need custom AD.
    pub fn custom_fn_multi(
        &mut self,
        inputs: Vec<NodeId>,
        mut fwd_body: Graph,
    ) -> MultiOutputHandle {
        use crate::op::BinaryOp;
        // Snapshot the original outputs + their shapes BEFORE
        // appending concat ops. Outputs land at the end of the graph;
        // we'll replace them.
        let original_outputs = fwd_body.outputs.clone();
        assert!(
            !original_outputs.is_empty(),
            "custom_fn_multi: fwd_body must have ≥ 1 declared output"
        );
        let mut sub_shapes: Vec<Shape> = Vec::with_capacity(original_outputs.len());
        let mut offsets: Vec<usize> = Vec::with_capacity(original_outputs.len());
        let mut total_len: usize = 0;
        for &out_id in &original_outputs {
            let s = fwd_body.node(out_id).shape.clone();
            assert_eq!(
                s.dtype(),
                DType::F32,
                "custom_fn_multi MVP: all sub-outputs must be F32, got {:?} \
                 (sub-output #{})",
                s.dtype(),
                sub_shapes.len()
            );
            let n_elems: usize = s
                .dims()
                .iter()
                .map(|d| match d {
                    Dim::Static(k) => *k,
                    Dim::Dynamic(_) => {
                        panic!("custom_fn_multi MVP: dynamic dims not supported")
                    }
                })
                .product();
            offsets.push(total_len);
            total_len += n_elems;
            sub_shapes.push(s);
        }
        // Flatten each sub-output to [n_elems] and concat along axis 0.
        let mut flats: Vec<NodeId> = Vec::with_capacity(original_outputs.len());
        for (out_id, sh) in original_outputs.iter().zip(sub_shapes.iter()) {
            let n: usize = sh
                .dims()
                .iter()
                .map(|d| match d {
                    Dim::Static(k) => *k,
                    Dim::Dynamic(_) => unreachable!(),
                })
                .product();
            let flat_shape = Shape::from_dims(&[Dim::Static(n)], DType::F32);
            let flat = fwd_body.add_node(
                Op::Reshape {
                    new_shape: vec![n as i64],
                },
                vec![*out_id],
                flat_shape,
            );
            flats.push(flat);
        }
        let concat_shape = Shape::from_dims(&[Dim::Static(total_len)], DType::F32);
        let concat = fwd_body.add_node(Op::Concat { axis: 0 }, flats.clone(), concat_shape);
        let _ = BinaryOp::Add; // import preserved if we extend later
        fwd_body.set_outputs(vec![concat]);

        // Now build the outer custom_fn with the rewritten body. Reuses
        // the single-output asserts; flat concat satisfies them.
        let source = self.custom_fn(inputs, fwd_body, None, None);

        MultiOutputHandle {
            source,
            sub_shapes,
            offsets,
        }
    }

    /// Bounded scan returning the stacked trajectory.
    /// Output shape is `[length, *init.shape]` — row `t` is the carry
    /// after step `t+1`, so row `length-1` equals the result of plain
    /// [`Self::scan`].
    pub fn scan_trajectory(&mut self, init: NodeId, body: Graph, length: u32) -> NodeId {
        let init_shape = self.shape(init).clone();
        let mut traj_dims: Vec<crate::Dim> = Vec::with_capacity(init_shape.rank() + 1);
        traj_dims.push(crate::Dim::Static(length as usize));
        for i in 0..init_shape.rank() {
            traj_dims.push(init_shape.dim(i));
        }
        let traj_shape = crate::Shape::from_dims(&traj_dims, init_shape.dtype());
        self.push(
            Op::Scan {
                body: Box::new(body),
                length,
                save_trajectory: true,
                num_xs: 0,
                num_bcast: 0,
                num_checkpoints: 0,
            },
            vec![init],
            traj_shape,
            None,
        )
    }
}
