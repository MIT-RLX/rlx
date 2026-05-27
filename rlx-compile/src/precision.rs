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

//! Precision policy + AutoMixedPrecision rewrite pass.
//!
//! The `PrecisionPolicy` is a high-level declarative spec that maps
//! op kinds to numeric precisions. The `AutoMixedPrecision` pass
//! consumes a policy and rewrites the graph: updates each node's
//! shape dtype + inserts Cast nodes at precision boundaries.
//!
//! After this pass runs, the IR carries per-node precision info via
//! `node.shape.dtype`, and the backend just reads it to pick the
//! right kernel variant. Backends don't need any session-level
//! precision flag.

use rlx_fusion::pass::Pass;
use rlx_ir::*;
use std::collections::HashMap;

/// Which numeric precision to use for an op.
/// (Subset of DType — only the ones we currently dispatch on.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    F32,
    F16,
    BF16,
}

impl Precision {
    pub fn dtype(self) -> DType {
        match self {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
            Precision::BF16 => DType::BF16,
        }
    }
}

/// Cast configuration carried by ops that emit a typed output.
///
/// Inspired by TileKernels' `CastInputConfig` / `CastOutputConfig`: a single
/// dataclass that flows from the layer down to the kernel selector, so adding
/// new quantized formats (FP8 e4m3, FP4 e2m1, blocked scaling) becomes a
/// matter of populating fields rather than threading new flags through call
/// sites.
///
/// Today only `out_dtype` is consulted by backends — the scaling-factor
/// fields are reserved for future quantization passes (FP8 / blocked SF).
/// Constructed once by the precision pass and embedded in fused ops.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CastConfig {
    /// Destination dtype for the cast (fragment of the output tensor).
    pub out_dtype: DType,
    /// Scaling factor block size `(rows, cols)` for blocked quantization.
    /// `None` means no scaling factor (plain cast).
    pub sf_block: Option<(usize, usize)>,
    /// Round scaling factors to powers of two (UE8M0 style).
    pub round_sf: bool,
}

impl CastConfig {
    /// Plain dtype cast with no scaling factor.
    pub const fn plain(out_dtype: DType) -> Self {
        Self {
            out_dtype,
            sf_block: None,
            round_sf: false,
        }
    }
    /// True when the cast does no work (out matches input dtype).
    pub fn is_noop(&self, in_dtype: DType) -> bool {
        self.out_dtype == in_dtype && self.sf_block.is_none()
    }
}

/// High-level op categorization for precision policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// Matmul, FusedMatMulBiasAct, conv — compute-heavy ops that
    /// benefit most from low precision.
    Compute,
    /// LayerNorm, RmsNorm, Softmax — reductions that need accuracy.
    Reduction,
    /// Add, Mul, GELU, SiLU — element-wise ops.
    Elementwise,
    /// Gather, Narrow, Reshape — data movement, no math.
    DataMovement,
    /// Inputs, parameters, outputs — user-facing.
    Boundary,
}

fn op_kind(op: &Op) -> OpKind {
    match op {
        Op::MatMul
        | Op::FusedMatMulBiasAct { .. }
        | Op::Conv { .. }
        | Op::DotGeneral { .. }
        | Op::DenseSolve
        | Op::BatchedDenseSolve
        | Op::Attention { .. }
        | Op::FusedTransformerLayer { .. }
        | Op::GroupedMatMul
        | Op::DequantGroupedMatMul { .. }
        | Op::DequantMoEWeights { .. }
        | Op::LoraMatMul { .. }
        | Op::DequantMatMul { .. }
        | Op::QMatMul { .. }
        | Op::QConv2d { .. }
        | Op::Conv2dBackwardInput { .. }
        | Op::Conv2dBackwardWeight { .. }
        | Op::AttentionBackward { .. } => OpKind::Compute,
        Op::LayerNorm { .. }
        | Op::RmsNorm { .. }
        | Op::Softmax { .. }
        | Op::FusedResidualLN { .. }
        | Op::FusedResidualRmsNorm { .. }
        | Op::Reduce { .. }
        | Op::Cumsum { .. }
        | Op::Sample { .. }
        | Op::SelectiveScan { .. }
        | Op::GatedDeltaNet { .. }
        | Op::SoftmaxCrossEntropyWithLogits
        | Op::SoftmaxCrossEntropyBackward
        | Op::LayerNormBackwardInput { .. }
        | Op::LayerNormBackwardGamma { .. }
        | Op::GroupNorm { .. } => OpKind::Reduction,
        Op::Activation(_)
        | Op::Binary(_)
        | Op::FusedSwiGLU { .. }
        | Op::Compare(_)
        | Op::Where
        | Op::ElementwiseRegion { .. }
        | Op::Quantize { .. }
        | Op::Dequantize { .. }
        | Op::FakeQuantize { .. }
        | Op::FakeQuantizeBackward { .. }
        | Op::FakeQuantizeLSQ { .. }
        | Op::FakeQuantizeLSQBackwardX { .. }
        | Op::FakeQuantizeLSQBackwardScale { .. }
        | Op::ReluBackward
        | Op::ActivationBackward { .. }
        | Op::ComplexNormSq
        | Op::ComplexNormSqBackward
        | Op::Conjugate => OpKind::Elementwise,
        Op::Gather { .. }
        | Op::Narrow { .. }
        | Op::Reshape { .. }
        | Op::Transpose { .. }
        | Op::Concat { .. }
        | Op::Expand { .. }
        | Op::Cast { .. }
        | Op::Rope { .. }
        | Op::Pool { .. }
        | Op::FusedAttentionBlock { .. }
        | Op::TopK { .. }
        | Op::ScatterAdd
        | Op::MaxPool2dBackward { .. }
        | Op::ResizeNearest2x
        | Op::AxialRope2d { .. } => OpKind::DataMovement,
        Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => OpKind::Boundary,
        // Control flow: treated as data movement (the inner sub-graph
        // gets its own precision policy applied separately).
        Op::If { .. } | Op::While { .. } => OpKind::DataMovement,
        // Custom user-registered ops are opaque to the precision pass
        // — classify as Compute by default; the registered op's own
        // implementation decides what dtype it operates at.
        Op::Custom { .. } => OpKind::Compute,
        Op::Scan { .. } => OpKind::Compute,
        Op::ScanBackward { .. } => OpKind::Compute,
        Op::ScanBackwardXs { .. } => OpKind::Compute,
        Op::CustomFn { .. } => OpKind::Compute,
        Op::Fft { .. } => OpKind::Compute,
        _ => OpKind::Compute,
    }
}

/// Declarative precision policy for graph compilation.
#[derive(Debug, Clone, Default)]
pub enum PrecisionPolicy {
    /// All ops at F32. Default; safe; baseline accuracy.
    #[default]
    AlwaysF32,
    /// All ops at F16. Maximum speed; may lose accuracy on reductions.
    AlwaysF16,
    /// Mixed precision, conservative variant. Forces F32 at every reduction
    /// boundary, matching PyTorch's pre-2024 autocast and HuggingFace's
    /// historical default. Accuracy is the highest of the AMP variants;
    /// performance suffers from a Cast node before and after every
    /// LayerNorm / Softmax in the graph.
    ///   Compute → F16
    ///   Reduction → F32  (← the cast tax — see AutoMixed for the fix)
    ///   Elementwise → F16
    ///   DataMovement → F16
    ///   Boundary (input/param/output) → F32
    AutoMixedConservative,
    /// Mixed precision (Phase G — current default). Reductions stay in
    /// the input dtype; the kernels themselves promote-to-f32 internally
    /// for the accumulation. This eliminates the dozens of Cast nodes
    /// that AutoMixedConservative inserts at LN/Softmax boundaries
    /// without sacrificing the f32 reduction accumulation that matters.
    /// Matches what modern PyTorch autocast actually does on Metal.
    ///   Compute → F16
    ///   Reduction → F16  (kernel accumulates in f32 internally)
    ///   Elementwise → F16
    ///   DataMovement → F16
    ///   Boundary (input/param/output) → F32
    AutoMixed,
    /// Mixed precision targeting BF16 on TPU/XLA. Same shape as
    /// `AutoMixed` (compute + reduction + elementwise + data-movement
    /// in the chosen low precision; boundaries stay F32) but the low
    /// precision is BF16 instead of F16. BF16 is the native compute
    /// dtype on TPU and recent GPUs; matches what JAX picks when
    /// `jax.config.update("jax_default_dtype_bits", "bfloat16")`.
    ///   Compute → BF16
    ///   Reduction → BF16  (XLA's TPU codegen accumulates in f32)
    ///   Elementwise → BF16
    ///   DataMovement → BF16
    ///   Boundary → F32
    AutoMixedBf16,
    /// Explicit per-op-kind override.
    Custom(HashMap<OpKind, Precision>),
}

impl PrecisionPolicy {
    /// Resolve the target precision for an op kind.
    pub fn precision_for(&self, kind: OpKind) -> Precision {
        match self {
            PrecisionPolicy::AlwaysF32 => Precision::F32,
            PrecisionPolicy::AlwaysF16 => match kind {
                OpKind::Boundary => Precision::F32, // user-facing stays f32
                _ => Precision::F16,
            },
            PrecisionPolicy::AutoMixedConservative => match kind {
                OpKind::Compute => Precision::F16,
                OpKind::Reduction => Precision::F32,
                OpKind::Elementwise => Precision::F16,
                OpKind::DataMovement => Precision::F16,
                OpKind::Boundary => Precision::F32,
            },
            PrecisionPolicy::AutoMixed => match kind {
                OpKind::Compute => Precision::F16,
                OpKind::Reduction => Precision::F16,
                OpKind::Elementwise => Precision::F16,
                OpKind::DataMovement => Precision::F16,
                OpKind::Boundary => Precision::F32,
            },
            PrecisionPolicy::AutoMixedBf16 => match kind {
                OpKind::Compute => Precision::BF16,
                OpKind::Reduction => Precision::BF16,
                OpKind::Elementwise => Precision::BF16,
                OpKind::DataMovement => Precision::BF16,
                OpKind::Boundary => Precision::F32,
            },
            PrecisionPolicy::Custom(map) => map.get(&kind).copied().unwrap_or(Precision::F32),
        }
    }
}

/// Pass that rewrites a graph according to a `PrecisionPolicy`.
///
/// For each node:
/// 1. Look up the target precision based on op kind.
/// 2. Update `node.shape.dtype` to that precision.
/// 3. If any input has a different dtype, insert a Cast node before it.
///
/// After this pass, every node knows its compute precision via its
/// shape dtype. Backends dispatch kernels per-node.
pub struct AutoMixedPrecision {
    pub policy: PrecisionPolicy,
}

impl AutoMixedPrecision {
    pub fn new(policy: PrecisionPolicy) -> Self {
        Self { policy }
    }
}

impl Pass for AutoMixedPrecision {
    fn name(&self) -> &str {
        "auto_mixed_precision"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Skip the pass entirely for AlwaysF32 — it's a no-op.
        if matches!(self.policy, PrecisionPolicy::AlwaysF32) {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        // Maps old NodeId → new NodeId at its post-rewrite precision.
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        // Tracks the precision each rewritten node ended up at.
        let mut node_precision: HashMap<NodeId, Precision> = HashMap::new();
        // Cast cache: avoid re-inserting identical Cast nodes.
        // Key: (source new id, target precision)
        let mut cast_cache: HashMap<(NodeId, Precision), NodeId> = HashMap::new();

        for node in graph.nodes() {
            let kind = op_kind(&node.op);
            let target = self.policy.precision_for(kind);

            // Inputs / params keep their original dtype (they're external);
            // outputs stay user-visible at F32.
            let target = match kind {
                OpKind::Boundary => Precision::F32,
                _ => target,
            };

            // Resolve each input: insert a Cast if precision differs.
            let mut new_inputs = Vec::with_capacity(node.inputs.len());
            for &in_id in &node.inputs {
                let src_new_id = id_map[&in_id];
                let src_prec = node_precision
                    .get(&in_id)
                    .copied()
                    .unwrap_or(Precision::F32);
                if src_prec == target {
                    new_inputs.push(src_new_id);
                } else {
                    // Insert (or reuse cached) cast
                    let cast_id = *cast_cache.entry((src_new_id, target)).or_insert_with(|| {
                        let shape = new_graph
                            .node(src_new_id)
                            .shape
                            .clone()
                            .with_dtype(target.dtype());
                        new_graph.add_node(Op::Cast { to: target.dtype() }, vec![src_new_id], shape)
                    });
                    new_inputs.push(cast_id);
                }
            }

            // Build the rewritten node with the target dtype on its shape.
            let new_shape = node.shape.clone().with_dtype(target.dtype());
            let new_id = new_graph.add_node(node.op.clone(), new_inputs, new_shape);
            id_map.insert(node.id, new_id);
            node_precision.insert(node.id, target);
        }

        // Outputs always stay at F32 — cast back if needed.
        let new_outputs: Vec<NodeId> = graph
            .outputs
            .iter()
            .map(|&out_id| {
                let src_new_id = id_map[&out_id];
                let src_prec = node_precision
                    .get(&out_id)
                    .copied()
                    .unwrap_or(Precision::F32);
                if src_prec == Precision::F32 {
                    src_new_id
                } else {
                    let shape = new_graph
                        .node(src_new_id)
                        .shape
                        .clone()
                        .with_dtype(DType::F32);
                    new_graph.add_node(Op::Cast { to: DType::F32 }, vec![src_new_id], shape)
                }
            })
            .collect();
        new_graph.set_outputs(new_outputs);

        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn always_f32_is_noop() {
        let mut g = Graph::new("test");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 3], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
        g.set_outputs(vec![mm]);

        let pass = AutoMixedPrecision::new(PrecisionPolicy::AlwaysF32);
        let out = pass.run(g);
        assert_eq!(out.len(), 3); // input, param, matmul — no casts
    }

    #[test]
    fn auto_mixed_inserts_casts_at_boundary() {
        let mut g = Graph::new("test");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 3], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
        g.set_outputs(vec![mm]);

        let pass = AutoMixedPrecision::new(PrecisionPolicy::AutoMixed);
        let out = pass.run(g);

        // Should have: input(f32), param(f32), cast(f32→f16) for x,
        // cast(f32→f16) for w, matmul(f16), cast(f16→f32) for output.
        // = 6 nodes total, with the final output being a Cast back to F32.
        assert!(out.len() >= 6);
        let final_node = out.node(out.outputs[0]);
        assert!(matches!(final_node.op, Op::Cast { to: DType::F32 }));
    }
}
