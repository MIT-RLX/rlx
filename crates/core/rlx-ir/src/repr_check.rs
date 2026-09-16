// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Producer–consumer representation gate** — does each consumer's assumed
//! representation of an input match what its producer actually declares?
//!
//! rlx already checks shapes (`infer_shape`), legality per backend
//! (`rlx_runtime::check`), constants (`const_check`) and finiteness
//! (`numeric_check`). What none of them check is the thing that keeps going
//! wrong: an op reading an input under a *different representation* than the
//! producer wrote it in. Every one of these was a real defect in this tree, and
//! every one is the same shape of mistake:
//!
//! | defect | the mismatch |
//! |---|---|
//! | RoPE table stride `n_rot/2` vs `head_dim/2` | consumer assumed a different **stride** |
//! | `cos_row_stride` fwd/bwd disagreement | the two sides assumed different strides |
//! | ROCm GGUF `sgemm(N,N)` on a `[n,k]` weight | consumer assumed a different **layout** |
//! | ROCm i64 `Constant` vs f32-index gather | consumer assumed a different **encoding** |
//! | MLX `RmsNorm` reading 2 of 3 declared inputs | consumer ignored part of the **arity** |
//! | MLX `ScatterAdd` rank-1 updates | consumer's **rank contract** unmet |
//!
//! Each was found *numerically, after the fact* — by a parity test, or by the
//! finite-difference gate. That works, but it is an autopsy: the wrong number
//! has already been computed, and you only notice if a test happened to cover
//! that shape on that backend. A representation check is a **gate**: it runs on
//! the graph, before any codegen, and points at the offending edge.
//!
//! This is CAKE's Table 1 "data consistency" row — *"check data flow and
//! producer–consumer representation compatibility"* — and Appendix B.4's
//! division of labour: the caller writes down concrete commitments, and the
//! compiler carries the burden of deciding whether they are mutually legal.
//!
//! ## Scope, stated honestly
//!
//! The rules below are **necessary, not sufficient**. They encode the invariants
//! that have actually broken here; a graph that passes is not proven correct.
//! Following the same paper's Appendix C: missing coverage is reported as missing
//! rather than treated as a pass — [`ReprReport::checked_kinds`] says which op
//! kinds had a rule applied, so "no findings" can be read against "nothing was
//! looked at".
//!
//! Dynamic dims are skipped rather than guessed: a `Dim::Dynamic` carries no
//! extent to compare, and inventing one would produce false findings on every
//! dynamic-shape graph.

use crate::shape::Dim;
use crate::{Graph, NodeId, Op, OpKind};

/// What kind of representation contract was violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReprKind {
    /// A consumer reads an input with a different row stride / trailing extent
    /// than the producer wrote. The RoPE-table class.
    Stride,
    /// A consumer assumes a different element layout (row- vs column-major, or
    /// `[n,k]` vs `[k,n]`) than the producer produced. The GGUF-transpose class.
    Layout,
    /// A consumer assumes a different element encoding (dtype / packing) than
    /// the producer wrote. The i64-constant-vs-f32-index class.
    Encoding,
    /// The op declares more inputs than its contract can use, or fewer than it
    /// needs. The RmsNorm-dropped-beta class.
    Arity,
    /// A rank relation between two inputs is unmet. The ScatterAdd class.
    Rank,
    /// An index or window provably reads outside the producer's extent.
    Bounds,
}

impl ReprKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stride => "stride",
            Self::Layout => "layout",
            Self::Encoding => "encoding",
            Self::Arity => "arity",
            Self::Rank => "rank",
            Self::Bounds => "bounds",
        }
    }
}

/// One localized finding: which edge, what was expected, what was declared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReprFinding {
    pub node: NodeId,
    pub op: OpKind,
    /// Index into `node.inputs` this finding is about, when it is about one
    /// specific input.
    pub input: Option<usize>,
    pub kind: ReprKind,
    /// What the consumer's contract requires.
    pub expected: String,
    /// What the producer actually declares.
    pub actual: String,
    /// Why it matters — the failure this rule exists to prevent.
    pub why: &'static str,
}

impl std::fmt::Display for ReprFinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} ({:?})", self.node, self.op)?;
        if let Some(i) = self.input {
            write!(f, " input[{i}]")?;
        }
        write!(
            f,
            ": {} mismatch — expected {}, got {}. {}",
            self.kind.as_str(),
            self.expected,
            self.actual,
            self.why
        )
    }
}

/// Result of [`check_graph`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReprReport {
    pub findings: Vec<ReprFinding>,
    /// Op kinds a rule was actually applied to. Reported so an empty
    /// `findings` can be distinguished from "no rule covered this graph".
    pub checked_kinds: Vec<OpKind>,
    /// Nodes skipped because a relevant dim was dynamic.
    pub skipped_dynamic: usize,
}

impl ReprReport {
    pub fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }

    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "representation check: {} finding(s), {} op kind(s) covered, {} node(s) skipped (dynamic dims)",
            self.findings.len(),
            self.checked_kinds.len(),
            self.skipped_dynamic
        );
        for f in &self.findings {
            let _ = writeln!(s, "  {f}");
        }
        if self.findings.is_empty() && self.checked_kinds.is_empty() {
            let _ = writeln!(
                s,
                "  NOTE: no rule applied to this graph — clean here means unchecked, not verified."
            );
        }
        s
    }
}

/// Static extent of a dim, or `None` when dynamic.
fn stat(d: Dim) -> Option<usize> {
    match d {
        Dim::Static(n) => Some(n),
        Dim::Dynamic(_) => None,
    }
}

/// Last static dim of a node's shape.
fn last_dim(g: &Graph, id: NodeId) -> Option<usize> {
    let sh = &g.node(id).shape;
    if sh.rank() == 0 {
        return None;
    }
    stat(sh.dim(sh.rank() - 1))
}

/// Check every producer→consumer edge against the rules below.
pub fn check_graph(graph: &Graph) -> ReprReport {
    let mut r = ReprReport::default();
    let note = |r: &mut ReprReport, k: OpKind| {
        if !r.checked_kinds.contains(&k) {
            r.checked_kinds.push(k);
        }
    };

    for node in graph.nodes() {
        match &node.op {
            // ── R1 RoPE table stride ────────────────────────────────────────
            //
            // The cos/sin tables are indexed with a stride taken from their own
            // LAST dimension, and the rotation uses `n_rot/2` angles per
            // position. A table whose trailing extent is `head_dim/2` (padded,
            // as Qwen3.5 allocates it) is legal *only* because the kernel reads
            // the leading `n_rot/2` columns — but a kernel that derives the
            // stride from `n_rot` instead of the table's own width reads a later
            // position's angles for every row past the first. That is exactly
            // how three backends went wrong while agreeing with each other.
            Op::Rope {
                head_dim, n_rot, ..
            }
            | Op::RopeBackward {
                head_dim, n_rot, ..
            } => {
                note(&mut r, node.op.kind());
                if *n_rot > *head_dim {
                    r.findings.push(ReprFinding {
                        node: node.id,
                        op: node.op.kind(),
                        input: None,
                        kind: ReprKind::Bounds,
                        expected: format!("n_rot <= head_dim ({head_dim})"),
                        actual: format!("n_rot = {n_rot}"),
                        why: "rotating more lanes than a head has reads into the next head",
                    });
                }
                if !n_rot.is_multiple_of(2) {
                    r.findings.push(ReprFinding {
                        node: node.id,
                        op: node.op.kind(),
                        input: None,
                        kind: ReprKind::Stride,
                        expected: "even n_rot (it is a count of PAIRS × 2)".to_string(),
                        actual: format!("n_rot = {n_rot}"),
                        why: "an odd n_rot leaves a half pair whose partner is another lane",
                    });
                }
                // cos and sin must agree with each other, and hold at least the
                // `n_rot/2` angles the rotation consumes.
                if node.inputs.len() >= 3 {
                    let cos = node.inputs[1];
                    let sin = node.inputs[2];
                    if graph.node(cos).shape.dims() != graph.node(sin).shape.dims() {
                        r.findings.push(ReprFinding {
                            node: node.id,
                            op: node.op.kind(),
                            input: Some(2),
                            kind: ReprKind::Stride,
                            expected: format!("sin shape == cos shape {:?}", graph.node(cos).shape.dims()),
                            actual: format!("{:?}", graph.node(sin).shape.dims()),
                            why: "cos and sin are read with one shared stride; differing widths desynchronize them",
                        });
                    }
                    match (last_dim(graph, cos), n_rot / 2) {
                        (Some(width), half) if width < half => {
                            r.findings.push(ReprFinding {
                                node: node.id,
                                op: node.op.kind(),
                                input: Some(1),
                                kind: ReprKind::Stride,
                                expected: format!("cos/sin trailing extent >= n_rot/2 ({half})"),
                                actual: format!("{width}"),
                                why: "the rotation consumes n_rot/2 angles per position; a narrower table wraps into the next position",
                            });
                        }
                        (None, _) => r.skipped_dynamic += 1,
                        _ => {}
                    }
                }
            }

            // ── R2 Norm arity + scale geometry ──────────────────────────────
            //
            // `RmsNorm`/`LayerNorm` declare THREE inputs (x, gamma, beta). MLX
            // read two and silently dropped beta; wgpu/CUDA/ROCm did the same
            // earlier. The gate cannot see inside a backend, but it CAN insist
            // the graph hands over all three with the right geometry, so a
            // backend that ignores one is a backend bug and not an ambiguity in
            // the IR.
            Op::RmsNorm { .. } | Op::LayerNorm { .. } => {
                note(&mut r, node.op.kind());
                let want = node.op.num_inputs();
                if node.inputs.len() != want {
                    r.findings.push(ReprFinding {
                        node: node.id,
                        op: node.op.kind(),
                        input: None,
                        kind: ReprKind::Arity,
                        expected: format!("{want} inputs (x, gamma, beta)"),
                        actual: format!("{}", node.inputs.len()),
                        why: "a norm missing its shift/scale silently computes a different function",
                    });
                } else {
                    let Some(nd) = last_dim(graph, node.inputs[0]) else {
                        r.skipped_dynamic += 1;
                        continue;
                    };
                    for (i, name) in [(1usize, "gamma"), (2, "beta")] {
                        let sh = &graph.node(node.inputs[i]).shape;
                        let ok = sh.rank() == 1 && stat(sh.dim(0)) == Some(nd);
                        if !ok && sh.rank() == 1 && stat(sh.dim(0)).is_none() {
                            r.skipped_dynamic += 1;
                        } else if !ok {
                            r.findings.push(ReprFinding {
                                node: node.id,
                                op: node.op.kind(),
                                input: Some(i),
                                kind: ReprKind::Stride,
                                expected: format!("{name} = [{nd}] (the normalized extent)"),
                                actual: format!("{:?}", sh.dims()),
                                why: "a scale vector of another width is broadcast against the wrong axis",
                            });
                        }
                    }
                }
            }

            // ── R3 Gather index encoding ────────────────────────────────────
            //
            // The f32-uniform GPU arena stores integer indices as f32 *values*,
            // so an index tensor must carry an integral dtype the stager knows
            // how to widen. A raw-byte reinterpret of an i64 constant arrives as
            // denormals — which is precisely how ROCm's slice returned element 0
            // for every position.
            Op::Gather { axis } => {
                note(&mut r, node.op.kind());
                if node.inputs.len() >= 2 {
                    let idx_dt = graph.node(node.inputs[1]).shape.dtype();
                    let integral = !idx_dt.is_float()
                        && !matches!(idx_dt, crate::DType::C64 | crate::DType::C128);
                    if !integral && idx_dt != crate::DType::F32 {
                        r.findings.push(ReprFinding {
                            node: node.id,
                            op: node.op.kind(),
                            input: Some(1),
                            kind: ReprKind::Encoding,
                            expected: "integral index dtype (or F32 on the f32-uniform arena)".to_string(),
                            actual: format!("{idx_dt:?}"),
                            why: "a non-integral index is read as an element offset; f32-arena backends widen integers by DECLARED dtype, so an unexpected one arrives as garbage",
                        });
                    }
                    let table_rank = graph.node(node.inputs[0]).shape.rank();
                    if *axis >= table_rank {
                        r.findings.push(ReprFinding {
                            node: node.id,
                            op: node.op.kind(),
                            input: Some(0),
                            kind: ReprKind::Bounds,
                            expected: format!("axis < table rank ({table_rank})"),
                            actual: format!("axis = {axis}"),
                            why: "gathering along a nonexistent axis reads with an undefined stride",
                        });
                    }
                }
            }

            // ── R4 ScatterAdd rank relation ─────────────────────────────────
            //
            // MLX requires `updates.ndim == target.ndim + indices.ndim`; rlx's
            // Slice VJP emits rank-1 updates over a rank-1 target, which fits
            // only after a reshape. Flagging the relation here documents the
            // contract the backends have to satisfy.
            Op::ScatterAdd { axis } => {
                note(&mut r, node.op.kind());
                if *axis != 0 {
                    r.findings.push(ReprFinding {
                        node: node.id,
                        op: node.op.kind(),
                        input: None,
                        kind: ReprKind::Layout,
                        expected: "axis == 0 (the only layout the kernels implement)".to_string(),
                        actual: format!("axis = {axis}"),
                        why: "non-zero axis needs a transpose/scatter/transpose rewrite first; reaching a backend directly scatters along the wrong stride",
                    });
                }
                if node.inputs.len() >= 2 {
                    let upd = graph.node(node.inputs[0]).shape.rank();
                    let idx = graph.node(node.inputs[1]).shape.rank();
                    if idx > upd {
                        r.findings.push(ReprFinding {
                            node: node.id,
                            op: node.op.kind(),
                            input: Some(1),
                            kind: ReprKind::Rank,
                            expected: format!("index rank <= updates rank ({upd})"),
                            actual: format!("{idx}"),
                            why: "more index axes than update axes leaves update elements unaddressed",
                        });
                    }
                }
            }

            // ── R5 Slice window bounds ──────────────────────────────────────
            //
            // `start + (len-1)*step` must land inside the axis. This one has a
            // pedigree: the probe written to isolate the ROCm slice bug computed
            // an out-of-range expectation itself (start=0, len=4, step=3 on an
            // 8-element axis reads index 9). If a hand-written test gets this
            // wrong, a generated schedule will too.
            Op::Slice {
                axis,
                start,
                len,
                step,
            } => {
                note(&mut r, node.op.kind());
                let sh = &graph.node(node.inputs[0]).shape;
                if *axis >= sh.rank() {
                    r.findings.push(ReprFinding {
                        node: node.id,
                        op: node.op.kind(),
                        input: Some(0),
                        kind: ReprKind::Bounds,
                        expected: format!("axis < rank ({})", sh.rank()),
                        actual: format!("axis = {axis}"),
                        why: "slicing a nonexistent axis has no defined stride",
                    });
                } else if let Some(extent) = stat(sh.dim(*axis)) {
                    if *step == 0 {
                        r.findings.push(ReprFinding {
                            node: node.id,
                            op: node.op.kind(),
                            input: None,
                            kind: ReprKind::Bounds,
                            expected: "step != 0".to_string(),
                            actual: "step = 0".to_string(),
                            why: "a zero step reads one element len times, which is not a slice",
                        });
                    } else if *len > 0 {
                        let last = *start as i64 + (*len as i64 - 1) * *step;
                        if last < 0 || last >= extent as i64 {
                            r.findings.push(ReprFinding {
                                node: node.id,
                                op: node.op.kind(),
                                input: Some(0),
                                kind: ReprKind::Bounds,
                                expected: format!("start + (len-1)*step inside [0, {extent})"),
                                actual: format!(
                                    "{start} + ({len}-1)*{step} = {last}"
                                ),
                                why: "the window walks off the axis; a clamping kernel silently returns edge data instead",
                            });
                        }
                    }
                } else {
                    r.skipped_dynamic += 1;
                }
            }

            // ── R6 Pad rank agreement ───────────────────────────────────────
            Op::Pad { pads, .. } => {
                note(&mut r, node.op.kind());
                let rank = graph.node(node.inputs[0]).shape.rank();
                if pads.len() != rank {
                    r.findings.push(ReprFinding {
                        node: node.id,
                        op: node.op.kind(),
                        input: Some(0),
                        kind: ReprKind::Rank,
                        expected: format!("one [before, after] pair per axis ({rank})"),
                        actual: format!("{}", pads.len()),
                        why: "a short pad list pads the wrong axes; a long one indexes past the rank",
                    });
                }
            }

            // ── R7 Reduce axes ──────────────────────────────────────────────
            Op::Reduce { axes, .. } => {
                note(&mut r, node.op.kind());
                let rank = graph.node(node.inputs[0]).shape.rank();
                for a in axes {
                    if *a >= rank {
                        r.findings.push(ReprFinding {
                            node: node.id,
                            op: node.op.kind(),
                            input: Some(0),
                            kind: ReprKind::Bounds,
                            expected: format!("axis < rank ({rank})"),
                            actual: format!("axis = {a}"),
                            why: "reducing a nonexistent axis yields an undefined output extent",
                        });
                    }
                }
                let mut seen = axes.clone();
                seen.sort_unstable();
                let before = seen.len();
                seen.dedup();
                if seen.len() != before {
                    r.findings.push(ReprFinding {
                        node: node.id,
                        op: node.op.kind(),
                        input: None,
                        kind: ReprKind::Rank,
                        expected: "distinct axes".to_string(),
                        actual: format!("{axes:?}"),
                        why: "a repeated axis is reduced twice, scaling Mean by the wrong extent",
                    });
                }
            }

            // ── R8 Attention head geometry + Custom mask rank ───────────────
            //
            // Q/K/V must agree on head geometry with the op's own declaration,
            // or the kernel strides across head boundaries.
            Op::Attention {
                num_heads,
                head_dim,
                v_head_dim,
                mask_kind,
                ..
            } => {
                note(&mut r, node.op.kind());
                let q = &graph.node(node.inputs[0]).shape;
                if let Some(last) = stat(q.dim(q.rank().saturating_sub(1))) {
                    // Rank-4 [B,H,S,D] carries head_dim in the last axis;
                    // rank-3 [B,S,H*D] carries the product.
                    let ok = if q.rank() >= 4 {
                        last == *head_dim
                    } else {
                        last.is_multiple_of((*head_dim).max(1))
                    };
                    if !ok {
                        r.findings.push(ReprFinding {
                            node: node.id,
                            op: node.op.kind(),
                            input: Some(0),
                            kind: ReprKind::Layout,
                            expected: format!(
                                "Q trailing extent to be head_dim ({head_dim}) at rank>=4, or a multiple of it at rank 3"
                            ),
                            actual: format!("{last} (rank {})", q.rank()),
                            why: "a head_dim that does not divide the layout makes every head read across its neighbour",
                        });
                    }
                } else {
                    r.skipped_dynamic += 1;
                }
                if *num_heads == 0 || *head_dim == 0 {
                    r.findings.push(ReprFinding {
                        node: node.id,
                        op: node.op.kind(),
                        input: None,
                        kind: ReprKind::Arity,
                        expected: "num_heads > 0 and head_dim > 0".to_string(),
                        actual: format!("num_heads = {num_heads}, head_dim = {head_dim}"),
                        why: "a zero head geometry divides by zero when deriving strides",
                    });
                }
                if let Some(v) = v_head_dim
                    && *v == 0
                {
                    r.findings.push(ReprFinding {
                        node: node.id,
                        op: node.op.kind(),
                        input: None,
                        kind: ReprKind::Arity,
                        expected: "v_head_dim > 0 when present".to_string(),
                        actual: "0".to_string(),
                        why: "an asymmetric SDPA with zero V width writes a zero-wide output",
                    });
                }

                // `MaskKind::Custom` is KEY PADDING: one bit per (batch, key),
                // spelled `[B, S_k]` or any broadcast of it (`[1, S_k]`,
                // `[1, 1, S_k]`, `[B, 1, 1, S_k]`). A per-query mask is
                // `MaskKind::Bias`, not this.
                //
                // Handing a real query axis to `Custom` is not merely
                // unsupported — the backends *disagree about what it means*.
                // MLX and wgpu broadcast it as a per-query mask; CPU, Metal and
                // Vulkan index `mask[b * S_k + k]` and read it as key padding,
                // giving a different answer for the same graph. Neither is a
                // wrong number you can catch by comparing backends, because
                // there is no contract saying which one is right.
                //
                // So it is rejected here rather than pinned in a parity test:
                // undefined-and-refused beats undefined-and-divergent, and the
                // finding names the op that *does* define the shape.
                if *mask_kind == crate::op::MaskKind::Custom
                    && let Some(&mask) = node.inputs.get(3)
                {
                    let sh = &graph.node(mask).shape;
                    // Which axis is the query axis for this rank, if any. Rank 2
                    // is `[B, S_k]` — no query axis at all, hence always legal.
                    let q_axis = match sh.rank() {
                        0..=2 => None,
                        3 => Some(1), // [B, S_q, S_k]
                        4 => Some(2), // [B, H, S_q, S_k]
                        _ => Some(usize::MAX),
                    };
                    // Rank 4 additionally carries a head axis, which key padding
                    // does not vary over either.
                    let bcast_axes: &[usize] = match sh.rank() {
                        3 => &[1],
                        4 => &[1, 2],
                        _ => &[],
                    };
                    if q_axis == Some(usize::MAX) {
                        r.findings.push(ReprFinding {
                            node: node.id,
                            op: node.op.kind(),
                            input: Some(3),
                            kind: ReprKind::Rank,
                            expected: "a Custom mask of rank 2, 3 or 4".to_string(),
                            actual: format!("rank {}", sh.rank()),
                            why: "key padding has no spelling above rank 4; no backend agrees on how to stride one",
                        });
                    } else {
                        for &a in bcast_axes {
                            match stat(sh.dim(a)) {
                                Some(1) => {}
                                Some(n) => {
                                    let what = if a == q_axis.unwrap() {
                                        "query"
                                    } else {
                                        "head"
                                    };
                                    r.findings.push(ReprFinding {
                                        node: node.id,
                                        op: node.op.kind(),
                                        input: Some(3),
                                        kind: ReprKind::Layout,
                                        expected: format!(
                                            "MaskKind::Custom is key padding — its {what} axis (axis {a} of {:?}) must be 1; use MaskKind::Bias for a per-query mask",
                                            sh.dims()
                                        ),
                                        actual: format!("{what} extent {n}"),
                                        why: "MLX and wgpu broadcast a per-query Custom mask; CPU, Metal and Vulkan read the same tensor as key padding — the graph has no defined meaning",
                                    });
                                }
                                None => r.skipped_dynamic += 1,
                            }
                        }
                    }
                }
            }

            _ => {}
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{MaskKind, PadMode, ReduceOp, RopeStyle};
    use crate::{DType, Shape};

    const F: DType = DType::F32;

    fn rope(head_dim: usize, n_rot: usize, table_width: usize) -> Graph {
        let mut g = Graph::new("rope");
        let x = g.input("x", Shape::new(&[1, 4, head_dim], F));
        let cos = g.input("cos", Shape::new(&[8, table_width], F));
        let sin = g.input("sin", Shape::new(&[8, table_width], F));
        let y = g.add_node(
            Op::Rope {
                head_dim,
                n_rot,
                style: RopeStyle::NeoX,
            },
            vec![x, cos, sin],
            Shape::new(&[1, 4, head_dim], F),
        );
        g.set_outputs(vec![y]);
        g
    }

    /// The RoPE-stride class: a table narrower than `n_rot/2` wraps into the
    /// next position's angles. This is the defect that had three backends
    /// agreeing with each other and all wrong.
    #[test]
    fn flags_rope_table_narrower_than_n_rot_half() {
        let r = check_graph(&rope(8, 8, 2)); // needs 4 angles, table has 2
        assert!(!r.is_clean(), "{}", r.render());
        assert!(r.findings.iter().any(|f| f.kind == ReprKind::Stride));
        // A correctly-sized table (and the padded head_dim/2 layout) is clean.
        assert!(check_graph(&rope(8, 8, 4)).is_clean());
        assert!(
            check_graph(&rope(8, 4, 4)).is_clean(),
            "padded table is legal"
        );
    }

    #[test]
    fn flags_rope_n_rot_exceeding_head_dim() {
        let r = check_graph(&rope(4, 8, 4));
        assert!(r.findings.iter().any(|f| f.kind == ReprKind::Bounds));
    }

    #[test]
    fn flags_desynchronized_cos_sin() {
        let mut g = Graph::new("rope");
        let x = g.input("x", Shape::new(&[1, 4, 8], F));
        let cos = g.input("cos", Shape::new(&[8, 4], F));
        let sin = g.input("sin", Shape::new(&[8, 2], F)); // different width
        let y = g.add_node(
            Op::Rope {
                head_dim: 8,
                n_rot: 8,
                style: RopeStyle::NeoX,
            },
            vec![x, cos, sin],
            Shape::new(&[1, 4, 8], F),
        );
        g.set_outputs(vec![y]);
        let r = check_graph(&g);
        assert!(
            r.findings.iter().any(|f| f.input == Some(2)),
            "{}",
            r.render()
        );
    }

    /// The dropped-beta class: the IR declares three norm inputs, so a graph
    /// handing over two is malformed rather than ambiguous.
    #[test]
    fn flags_norm_missing_its_scale_inputs() {
        let mut g = Graph::new("rms");
        let x = g.input("x", Shape::new(&[2, 8], F));
        let gamma = g.input("gamma", Shape::new(&[8], F));
        let y = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![x, gamma], // beta missing
            Shape::new(&[2, 8], F),
        );
        g.set_outputs(vec![y]);
        let r = check_graph(&g);
        assert!(
            r.findings.iter().any(|f| f.kind == ReprKind::Arity),
            "{}",
            r.render()
        );
    }

    #[test]
    fn flags_norm_scale_of_the_wrong_width() {
        let mut g = Graph::new("rms");
        let x = g.input("x", Shape::new(&[2, 8], F));
        let gamma = g.input("gamma", Shape::new(&[4], F)); // wrong width
        let beta = g.input("beta", Shape::new(&[8], F));
        let y = g.add_node(
            Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![x, gamma, beta],
            Shape::new(&[2, 8], F),
        );
        g.set_outputs(vec![y]);
        let r = check_graph(&g);
        assert!(
            r.findings.iter().any(|f| f.input == Some(1)),
            "{}",
            r.render()
        );
    }

    /// The slice-window class — and the exact mistake the ROCm isolation probe
    /// made in its own expectation.
    #[test]
    fn flags_slice_window_walking_off_the_axis() {
        let mut g = Graph::new("slice");
        let x = g.input("x", Shape::new(&[8], F));
        let y = g.add_node(
            Op::Slice {
                axis: 0,
                start: 0,
                len: 4,
                step: 3, // reads index 9 of an 8-element axis
            },
            vec![x],
            Shape::new(&[4], F),
        );
        g.set_outputs(vec![y]);
        let r = check_graph(&g);
        assert!(
            r.findings.iter().any(|f| f.kind == ReprKind::Bounds),
            "{}",
            r.render()
        );
    }

    #[test]
    fn accepts_every_legal_slice_window() {
        for (step, start, len) in [
            (1i64, 0usize, 4usize),
            (2, 0, 4),
            (3, 0, 3),
            (-1, 7, 4),
            (-2, 7, 4),
        ] {
            let mut g = Graph::new("slice");
            let x = g.input("x", Shape::new(&[8], F));
            let y = g.add_node(
                Op::Slice {
                    axis: 0,
                    start,
                    len,
                    step,
                },
                vec![x],
                Shape::new(&[len], F),
            );
            g.set_outputs(vec![y]);
            assert!(
                check_graph(&g).is_clean(),
                "step={step} start={start} len={len} should be legal"
            );
        }
    }

    #[test]
    fn flags_reduce_axis_out_of_range_and_duplicates() {
        let mut g = Graph::new("red");
        let x = g.input("x", Shape::new(&[2, 3], F));
        let y = g.add_node(
            Op::Reduce {
                op: ReduceOp::Sum,
                axes: vec![5],
                keep_dim: false,
            },
            vec![x],
            Shape::new(&[2], F),
        );
        g.set_outputs(vec![y]);
        assert!(
            check_graph(&g)
                .findings
                .iter()
                .any(|f| f.kind == ReprKind::Bounds)
        );

        let mut g2 = Graph::new("red2");
        let x2 = g2.input("x", Shape::new(&[2, 3], F));
        let y2 = g2.add_node(
            Op::Reduce {
                op: ReduceOp::Mean,
                axes: vec![1, 1],
                keep_dim: false,
            },
            vec![x2],
            Shape::new(&[2], F),
        );
        g2.set_outputs(vec![y2]);
        assert!(
            check_graph(&g2)
                .findings
                .iter()
                .any(|f| f.kind == ReprKind::Rank)
        );
    }

    #[test]
    fn flags_pad_list_not_matching_rank() {
        let mut g = Graph::new("pad");
        let x = g.input("x", Shape::new(&[1, 6], F));
        let y = g.add_node(
            Op::Pad {
                pads: vec![[2, 2]], // rank 2, one pair
                mode: PadMode::Reflect,
            },
            vec![x],
            Shape::new(&[1, 10], F),
        );
        g.set_outputs(vec![y]);
        assert!(
            check_graph(&g)
                .findings
                .iter()
                .any(|f| f.kind == ReprKind::Rank)
        );
    }

    #[test]
    fn flags_attention_head_dim_that_does_not_divide_the_layout() {
        let mut g = Graph::new("attn");
        let sh = Shape::new(&[1, 4, 10], F); // rank 3, 10 not a multiple of 4
        let q = g.input("q", sh.clone());
        let k = g.input("k", sh.clone());
        let v = g.input("v", sh.clone());
        let y = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 4,
                v_head_dim: None,
                mask_kind: MaskKind::None,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            sh,
        );
        g.set_outputs(vec![y]);
        let r = check_graph(&g);
        assert!(
            r.findings.iter().any(|f| f.kind == ReprKind::Layout),
            "{}",
            r.render()
        );
    }

    /// Build `Op::Attention` over `[2, 2, 4, 8]` Q/K/V with a `Custom` mask of
    /// the given shape.
    fn attn_custom_mask(mask_dims: &[usize]) -> Graph {
        let mut g = Graph::new("attn");
        let sh = Shape::new(&[2, 2, 4, 8], F);
        let q = g.input("q", sh.clone());
        let k = g.input("k", sh.clone());
        let v = g.input("v", sh.clone());
        let m = g.input("m", Shape::new(mask_dims, F));
        let y = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 8,
                v_head_dim: None,
                mask_kind: MaskKind::Custom,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v, m],
            sh,
        );
        g.set_outputs(vec![y]);
        g
    }

    /// Every legal spelling of one key-padding mask stays clean. These are the
    /// same shapes `rlx-runtime/tests/attention_mask_shapes.rs` requires every
    /// backend to agree on, so the gate must not reject them.
    #[test]
    fn accepts_every_broadcast_spelling_of_key_padding() {
        for dims in [
            &[2, 4][..],       // [B, S_k]
            &[1, 4][..],       // broadcast over batch
            &[1, 1, 4][..],    // rank-3 spelling
            &[2, 1, 4][..],    // per-batch, broadcast query
            &[2, 1, 1, 4][..], // rank-4 spelling
            &[1, 1, 1, 4][..],
        ] {
            let r = check_graph(&attn_custom_mask(dims));
            assert!(
                r.is_clean(),
                "{dims:?} should be legal key padding\n{}",
                r.render()
            );
        }
    }

    /// A per-query mask under `Custom` is not a wrong number — it is a graph
    /// with no agreed meaning: MLX and wgpu broadcast it per query, CPU/Metal/
    /// Vulkan read it as key padding. Refuse it instead of freezing whichever
    /// backend happened to be asked.
    #[test]
    fn flags_per_query_custom_mask() {
        for dims in [
            &[2, 4, 4][..],    // [B, S_q, S_k]
            &[2, 2, 4, 4][..], // [B, H, S_q, S_k] — head axis is wrong too
            &[2, 1, 4, 4][..], // query axis alone
        ] {
            let r = check_graph(&attn_custom_mask(dims));
            assert!(
                r.findings
                    .iter()
                    .any(|f| f.kind == ReprKind::Layout && f.input == Some(3)),
                "{dims:?} should be refused\n{}",
                r.render()
            );
        }
    }

    /// `MaskKind::Bias` IS the per-head, per-query tensor — the rule must not
    /// fire on it, or it would reject the op that exists to express this.
    #[test]
    fn leaves_bias_masks_alone() {
        let mut g = Graph::new("attn");
        let sh = Shape::new(&[2, 2, 4, 8], F);
        let q = g.input("q", sh.clone());
        let k = g.input("k", sh.clone());
        let v = g.input("v", sh.clone());
        let m = g.input("m", Shape::new(&[2, 2, 4, 4], F));
        let y = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 8,
                v_head_dim: None,
                mask_kind: MaskKind::Bias,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v, m],
            sh,
        );
        g.set_outputs(vec![y]);
        assert!(check_graph(&g).is_clean(), "Bias defines a per-query mask");
    }

    /// Coverage must be reported, not implied. A graph no rule touches has to
    /// say so rather than read as verified — Appendix C's discipline.
    #[test]
    fn reports_coverage_so_clean_is_not_confused_with_unchecked() {
        let mut g = Graph::new("plain");
        let x = g.input("x", Shape::new(&[4], F));
        let y = g.activation(crate::op::Activation::Relu, x, Shape::new(&[4], F));
        g.set_outputs(vec![y]);
        let r = check_graph(&g);
        assert!(r.is_clean());
        assert!(r.checked_kinds.is_empty(), "no rule covers a bare relu");
        assert!(
            r.render().contains("clean here means unchecked"),
            "{}",
            r.render()
        );

        // …and a graph a rule DOES cover names the kind it covered.
        let covered = check_graph(&rope(8, 8, 4));
        assert!(covered.is_clean());
        assert!(!covered.checked_kinds.is_empty());
    }

    /// Dynamic dims are skipped and counted, never guessed — a fabricated
    /// extent would produce a false finding on every dynamic-shape graph.
    #[test]
    fn dynamic_dims_are_skipped_and_counted() {
        let mut g = Graph::new("dyn");
        let x = g.input("x", Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(8)], F));
        let y = g.add_node(
            Op::Slice {
                axis: 0,
                start: 0,
                len: 4,
                step: 1,
            },
            vec![x],
            Shape::new(&[4, 8], F),
        );
        g.set_outputs(vec![y]);
        let r = check_graph(&g);
        assert!(r.is_clean(), "{}", r.render());
        assert_eq!(r.skipped_dynamic, 1);
    }
}
