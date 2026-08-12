// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One table describing what every [`OpKind`] *is*.
//!
//! Passes constantly need to ask categorical questions about an op — is it
//! elementwise, is it a fusion boundary, can it be folded on the host, does it
//! carry a nested body. Historically each caller answered with its own
//! `matches!` list: [`Op::is_elementwise`] and friends here in `rlx-ir`,
//! `is_pure` in `rlx-compile`'s constant folder, `is_leaf` in `param_hoist`
//! *and* again in `rlx-distributed`'s partitioner, `FUSED_KINDS` in the
//! backend rewriter. Every one of those lists is open — a `matches!` with no
//! wildcard arm silently answers "no" for a variant nobody remembered to add.
//!
//! That is the same failure mode the opcode tables in [`crate::opcodes`] were
//! written to design out, and the fix is the same: classify every variant
//! **exactly once**, in an exhaustive `match`. Adding an [`OpKind`] without
//! classifying it is a compile error, not a silently wrong answer.
//!
//! ```
//! use rlx_ir::{Op, OpKind, capability::OpCaps};
//!
//! assert!(OpKind::MatMul.caps().contains(OpCaps::FUSION_BOUNDARY));
//! assert!(Op::MatMul.is_blas());
//! assert!(!OpKind::Reshape.caps().contains(OpCaps::ELEMENTWISE));
//! ```
//!
//! # Adding a capability
//!
//! Add the flag, then extend the arms below. Because the match is exhaustive
//! and grouped by capability *set*, a new flag usually means moving a handful
//! of kinds between existing arms rather than touching all 184 of them.

use crate::op::{Op, OpKind};

/// Number of [`OpKind`] variants.
pub const N_KINDS: usize = 184;

/// Static properties of an op, independent of its operands or attributes.
///
/// A bit set rather than an enum: most questions are independent
/// (a `FusedMatMulBiasAct` is simultaneously BLAS, a fusion boundary, and a
/// fused op with a decomposition).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpCaps(u32);

impl OpCaps {
    /// No static capability. The honest answer for most ops: they are neither
    /// elementwise nor BLAS nor anything else the optimizer special-cases.
    pub const NONE: Self = Self(0);

    /// A graph root with no tensor inputs (`Input` / `Param` / `Constant`).
    pub const LEAF: Self = Self(1 << 0);

    /// Same shape in, same shape out, computed independently per element —
    /// the prime fusion candidate.
    pub const ELEMENTWISE: Self = Self(1 << 1);

    /// Evaluable on the host from constant operands, so the constant folder
    /// may replace it outright.
    ///
    /// Narrower than "deterministic": it also requires that the folder has an
    /// implementation. An op can be perfectly deterministic and still not
    /// carry this flag.
    pub const CONST_FOLDABLE: Self = Self(1 << 2);

    /// Compute-intensive and dispatched to a tuned library / kernel
    /// (matmul, conv, solves).
    pub const BLAS: Self = Self(1 << 3);

    /// Elementwise fusion must not span across this op.
    ///
    /// Implied by [`BLAS`](Self::BLAS) — every BLAS op carries both flags, so
    /// this can be tested directly without also testing `BLAS`.
    pub const FUSION_BOUNDARY: Self = Self(1 << 4);

    /// Collapses one or more axes; drives loop iteration in a fused kernel.
    pub const REDUCTION: Self = Self(1 << 5);

    /// May appear in an [`Op::TransformRegion`] chain.
    pub const TRANSFORM: Self = Self(1 << 6);

    /// A composite op with a decomposition into primitives — the backend
    /// rewriter can unfuse it when a target lacks the native kernel.
    pub const FUSED: Self = Self(1 << 7);

    /// Carries one or more nested [`Graph`](crate::Graph) bodies. Consumers
    /// that walk the IR recursively must descend via [`Op::subgraphs`].
    pub const NESTED_BODY: Self = Self(1 << 8);

    /// Output is not a function of the inputs alone (RNG / sampling). Never
    /// const-foldable, never CSE-able, and a hard root for hoisting.
    pub const NONDETERMINISTIC: Self = Self(1 << 9);

    /// Union of two sets (`const`-friendly, so the table can be built in a
    /// `const fn`).
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Does this set contain every flag in `other`?
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Does this set share any flag with `other`?
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// True when no capability is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for OpCaps {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl OpKind {
    /// Every [`OpKind`], in declaration order.
    ///
    /// Exists so tests can fan out exhaustively over the whole op set — the
    /// same role `BinaryOp::ALL` and friends play in [`crate::opcodes`].
    pub const ALL: [OpKind; N_KINDS] = [
        OpKind::Input,
        OpKind::Param,
        OpKind::Constant,
        OpKind::Activation,
        OpKind::Cast,
        OpKind::StopGradient,
        OpKind::Quantize,
        OpKind::Dequantize,
        OpKind::FakeQuantize,
        OpKind::FakeQuantizeLSQ,
        OpKind::FakeQuantizeLSQBackwardX,
        OpKind::FakeQuantizeLSQBackwardScale,
        OpKind::Binary,
        OpKind::Compare,
        OpKind::Where,
        OpKind::Fma,
        OpKind::ElementwiseRegion,
        OpKind::TransformRegion,
        OpKind::BatchElementwiseRegion,
        OpKind::MatMul,
        OpKind::DotGeneral,
        OpKind::DenseSolve,
        OpKind::BatchedDenseSolve,
        OpKind::Cholesky,
        OpKind::TriangularSolve,
        OpKind::Det,
        OpKind::LogDet,
        OpKind::Sort,
        OpKind::ArgSort,
        OpKind::Svd,
        OpKind::Qr,
        OpKind::LayerNorm,
        OpKind::LayerNorm2d,
        OpKind::GroupNorm,
        OpKind::BatchNormInference,
        OpKind::RmsNorm,
        OpKind::ResizeNearest2x,
        OpKind::Interpolate3d,
        OpKind::Attention,
        OpKind::Rope,
        OpKind::AxialRope2d,
        OpKind::Reshape,
        OpKind::Transpose,
        OpKind::Narrow,
        OpKind::Concat,
        OpKind::KvAppend,
        OpKind::Expand,
        OpKind::Gather,
        OpKind::Reverse,
        OpKind::Pad,
        OpKind::Slice,
        OpKind::Clamp,
        OpKind::Tile,
        OpKind::Trilu,
        OpKind::Reduce,
        OpKind::Histogram,
        OpKind::Softmax,
        OpKind::Cumsum,
        OpKind::CumProd,
        OpKind::CumMax,
        OpKind::ArgMax,
        OpKind::ArgMin,
        OpKind::TopK,
        OpKind::Sample,
        OpKind::RngNormal,
        OpKind::RngUniform,
        OpKind::Conv,
        OpKind::Im2Col,
        OpKind::ConvTranspose2d,
        OpKind::Conv3d,
        OpKind::ConvTranspose3d,
        OpKind::Pool,
        OpKind::ReluBackward,
        OpKind::ActivationBackward,
        OpKind::FakeQuantizeBackward,
        OpKind::ComplexNormSq,
        OpKind::ComplexNormSqBackward,
        OpKind::Conjugate,
        OpKind::MaxPool2dBackward,
        OpKind::Conv2dBackwardInput,
        OpKind::Conv2dBackwardWeight,
        OpKind::MaxPool3dBackward,
        OpKind::Conv3dBackwardInput,
        OpKind::Conv3dBackwardWeight,
        OpKind::SoftmaxCrossEntropy,
        OpKind::SoftmaxCrossEntropyWithLogits,
        OpKind::SoftmaxCrossEntropyBackward,
        OpKind::AttentionBackward,
        OpKind::AttentionBackwardAll,
        OpKind::LayerNormBackwardInput,
        OpKind::LayerNormBackwardGamma,
        OpKind::RmsNormBackwardInput,
        OpKind::RmsNormBackwardGamma,
        OpKind::RmsNormBackwardBeta,
        OpKind::RopeBackward,
        OpKind::GroupNormBackwardInput,
        OpKind::GroupNormBackwardGamma,
        OpKind::GroupNormBackwardBeta,
        OpKind::BatchNormInferenceBackwardInput,
        OpKind::BatchNormInferenceBackwardGamma,
        OpKind::BatchNormInferenceBackwardBeta,
        OpKind::CumsumBackward,
        OpKind::GatherBackward,
        OpKind::GroupedMatMul,
        OpKind::DequantGroupedMatMul,
        OpKind::DequantGroupedMatMulMlx,
        OpKind::DequantMoEWeights,
        OpKind::ScaledGroupedMatMul,
        OpKind::ScatterAdd,
        OpKind::ScatterNd,
        OpKind::ScatterElements,
        OpKind::GatherNd,
        OpKind::GatherElements,
        OpKind::LoraMatMul,
        OpKind::PartitionedConv,
        OpKind::DequantMatMul,
        OpKind::SynthMatMul,
        OpKind::SynthMatMulBackward,
        OpKind::SynthReconstruct,
        OpKind::SplineActivation,
        OpKind::SplineActivationBackwardX,
        OpKind::SplineActivationBackwardCoeff,
        OpKind::QMatMul,
        OpKind::QConv2d,
        OpKind::ScaledMatMul,
        OpKind::ScaledQuantize,
        OpKind::ScaledQuantScale,
        OpKind::ScaledDequantize,
        OpKind::SelectiveScan,
        OpKind::GatedDeltaNet,
        OpKind::Lstm,
        OpKind::Gru,
        OpKind::Rnn,
        OpKind::Mamba2,
        OpKind::FusedSwiGLU,
        OpKind::FusedMatMulBiasAct,
        OpKind::FusedMatMulResidual,
        OpKind::FusedConvBiasAct,
        OpKind::FusedResidualLN,
        OpKind::FusedResidualRmsNorm,
        OpKind::FusedAttentionBlock,
        OpKind::FusedTransformerLayer,
        OpKind::If,
        OpKind::While,
        OpKind::Scan,
        OpKind::ScanBackward,
        OpKind::ScanBackwardXs,
        OpKind::GaussianSplatRender,
        OpKind::GaussianSplatRenderBackward,
        OpKind::GaussianSplatPrepare,
        OpKind::GaussianSplatRasterize,
        OpKind::Custom,
        OpKind::CustomFn,
        OpKind::Fft,
        OpKind::FftButterflyStage,
        OpKind::LogMel,
        OpKind::LogMelBackward,
        OpKind::WelchPeaks,
        OpKind::BiMap,
        OpKind::ReEig,
        OpKind::LogEig,
        OpKind::SpdBatchNorm,
        OpKind::SpdKarcherMean,
        OpKind::ReEigBackward,
        OpKind::LogEigBackward,
        OpKind::SpdBatchNormBackwardX,
        OpKind::SpdBatchNormBackwardG,
        OpKind::SpdKarcherMeanWeighted,
        OpKind::SpdLogMap,
        OpKind::SpdExpMap,
        OpKind::SpdParallelTransport,
        OpKind::SpdMatrixFnBatch,
        OpKind::SpdLogMapBackward,
        OpKind::SpdExpMapBackward,
        OpKind::SpdParallelTransportBackward,
        OpKind::SpdMatrixFnBatchBackward,
        OpKind::Eigh,
        OpKind::EighBackward,
        OpKind::EighBatch,
        OpKind::EighBatchBackward,
        OpKind::AdaLayerNorm,
        OpKind::AdaLayerNormBackward,
        OpKind::GatedResidual,
        OpKind::GatedResidualBackward,
    ];

    /// Static capabilities of this op kind.
    ///
    /// Exhaustive by construction: a new [`OpKind`] variant will not compile
    /// until it is classified here. If it fits no category, add it to the
    /// final `OpCaps::NONE` arm — that is a deliberate statement that the
    /// optimizer treats it as an opaque node, not an oversight.
    pub const fn caps(self) -> OpCaps {
        use OpKind::*;
        match self {
            Input | Param | Constant => OpCaps::LEAF,
            Activation | Cast | Binary | Compare => {
                OpCaps::ELEMENTWISE.union(OpCaps::CONST_FOLDABLE)
            }
            StopGradient | Where | Fma | ElementwiseRegion | BatchElementwiseRegion => {
                OpCaps::ELEMENTWISE
            }
            MatMul
            | DotGeneral
            | DenseSolve
            | BatchedDenseSolve
            | Conv
            | Im2Col
            | ConvTranspose2d
            | Conv3d
            | ConvTranspose3d
            | GroupedMatMul
            | DequantGroupedMatMul
            | DequantGroupedMatMulMlx
            | DequantMoEWeights
            | ScaledGroupedMatMul
            | DequantMatMul
            | SynthMatMul
            | QMatMul
            | QConv2d
            | ScaledMatMul
            | FusedMatMulResidual => OpCaps::BLAS.union(OpCaps::FUSION_BOUNDARY),
            ResizeNearest2x => OpCaps::TRANSFORM,
            Reshape | Expand => OpCaps::CONST_FOLDABLE,
            Reduce | Softmax | TopK => OpCaps::REDUCTION,
            Sample | RngNormal | RngUniform => OpCaps::NONDETERMINISTIC,
            LoraMatMul | FusedMatMulBiasAct | FusedConvBiasAct => {
                OpCaps::BLAS.union(OpCaps::FUSION_BOUNDARY.union(OpCaps::FUSED))
            }
            PartitionedConv
            | SelectiveScan
            | GatedDeltaNet
            | Lstm
            | Gru
            | Rnn
            | Mamba2
            | FusedSwiGLU
            | FusedResidualLN
            | FusedResidualRmsNorm
            | FusedAttentionBlock
            | FusedTransformerLayer
            | AdaLayerNorm
            | GatedResidual => OpCaps::FUSED,
            If | While | Scan | ScanBackward | ScanBackwardXs | CustomFn => OpCaps::NESTED_BODY,
            GaussianSplatRender
            | GaussianSplatRenderBackward
            | GaussianSplatPrepare
            | GaussianSplatRasterize => OpCaps::FUSION_BOUNDARY,
            Quantize
            | Dequantize
            | FakeQuantize
            | FakeQuantizeLSQ
            | FakeQuantizeLSQBackwardX
            | FakeQuantizeLSQBackwardScale
            | TransformRegion
            | Cholesky
            | TriangularSolve
            | Det
            | LogDet
            | Sort
            | ArgSort
            | Svd
            | Qr
            | LayerNorm
            | LayerNorm2d
            | GroupNorm
            | BatchNormInference
            | RmsNorm
            | Interpolate3d
            | Attention
            | Rope
            | AxialRope2d
            | Transpose
            | Narrow
            | Concat
            | KvAppend
            | Gather
            | Reverse
            | Pad
            | Slice
            | Clamp
            | Tile
            | Trilu
            | Histogram
            | Cumsum
            | CumProd
            | CumMax
            | ArgMax
            | ArgMin
            | Pool
            | ReluBackward
            | ActivationBackward
            | FakeQuantizeBackward
            | ComplexNormSq
            | ComplexNormSqBackward
            | Conjugate
            | MaxPool2dBackward
            | Conv2dBackwardInput
            | Conv2dBackwardWeight
            | MaxPool3dBackward
            | Conv3dBackwardInput
            | Conv3dBackwardWeight
            | SoftmaxCrossEntropy
            | SoftmaxCrossEntropyWithLogits
            | SoftmaxCrossEntropyBackward
            | AttentionBackward
            | AttentionBackwardAll
            | LayerNormBackwardInput
            | LayerNormBackwardGamma
            | RmsNormBackwardInput
            | RmsNormBackwardGamma
            | RmsNormBackwardBeta
            | RopeBackward
            | GroupNormBackwardInput
            | GroupNormBackwardGamma
            | GroupNormBackwardBeta
            | BatchNormInferenceBackwardInput
            | BatchNormInferenceBackwardGamma
            | BatchNormInferenceBackwardBeta
            | CumsumBackward
            | GatherBackward
            | ScatterAdd
            | ScatterNd
            | ScatterElements
            | GatherNd
            | GatherElements
            | SynthMatMulBackward
            | SynthReconstruct
            | SplineActivation
            | SplineActivationBackwardX
            | SplineActivationBackwardCoeff
            | ScaledQuantize
            | ScaledQuantScale
            | ScaledDequantize
            | Custom
            | Fft
            | FftButterflyStage
            | LogMel
            | LogMelBackward
            | WelchPeaks
            | BiMap
            | ReEig
            | LogEig
            | SpdBatchNorm
            | SpdKarcherMean
            | ReEigBackward
            | LogEigBackward
            | SpdBatchNormBackwardX
            | SpdBatchNormBackwardG
            | SpdKarcherMeanWeighted
            | SpdLogMap
            | SpdExpMap
            | SpdParallelTransport
            | SpdMatrixFnBatch
            | SpdLogMapBackward
            | SpdExpMapBackward
            | SpdParallelTransportBackward
            | SpdMatrixFnBatchBackward
            | Eigh
            | EighBackward
            | EighBatch
            | EighBatchBackward
            | AdaLayerNormBackward
            | GatedResidualBackward => OpCaps::NONE,
        }
    }

    /// Convenience: does this kind have all of `caps`?
    pub const fn has(self, caps: OpCaps) -> bool {
        self.caps().contains(caps)
    }
}

// ── Predicates ──────────────────────────────────────────────────
//
// These read off the table. They are kept as methods on `Op` because that is
// where callers already reach for them; the classification itself lives in
// one place.

impl Op {
    /// True if this op is element-wise (same shape in, same shape out).
    /// Element-wise ops are prime fusion candidates.
    pub fn is_elementwise(&self) -> bool {
        self.kind().has(OpCaps::ELEMENTWISE)
    }

    /// True if this op may appear in a [`Op::TransformRegion`] chain.
    pub fn is_transform_eligible(&self) -> bool {
        self.kind().has(OpCaps::TRANSFORM)
    }

    /// True if this op is a BLAS/compute-intensive op that forms a fusion boundary.
    pub fn is_blas(&self) -> bool {
        self.kind().has(OpCaps::BLAS)
    }

    /// True if element-wise fusion must not span across this op.
    pub fn is_fusion_boundary(&self) -> bool {
        self.kind().has(OpCaps::FUSION_BOUNDARY)
    }

    /// True if this op is a reduction (drives loop iteration in fused kernels).
    pub fn is_reduction(&self) -> bool {
        self.kind().has(OpCaps::REDUCTION)
    }

    /// True if this op is a graph root with no tensor inputs.
    pub fn is_leaf(&self) -> bool {
        self.kind().has(OpCaps::LEAF)
    }

    /// True if the constant folder can evaluate this op on the host.
    pub fn is_const_foldable(&self) -> bool {
        self.kind().has(OpCaps::CONST_FOLDABLE)
    }

    /// True if this is a composite op with a decomposition into primitives.
    pub fn is_fused(&self) -> bool {
        self.kind().has(OpCaps::FUSED)
    }

    /// True if this op carries nested [`Graph`](crate::Graph) bodies.
    /// Equivalent to `!self.subgraphs().is_empty()`, without the walk.
    pub fn has_nested_body(&self) -> bool {
        self.kind().has(OpCaps::NESTED_BODY)
    }

    /// True if the output is not a function of the inputs alone (RNG /
    /// sampling), so it must never be folded, CSE'd or hoisted.
    pub fn is_nondeterministic(&self) -> bool {
        self.kind().has(OpCaps::NONDETERMINISTIC)
    }
}

/// Every kind carrying [`OpCaps::FUSED`] — the set the backend rewriter tries
/// to unfuse when a target lacks the native kernel.
pub fn fused_kinds() -> Vec<OpKind> {
    OpKind::ALL
        .into_iter()
        .filter(|k| k.has(OpCaps::FUSED))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_is_complete_and_duplicate_free() {
        let mut seen = std::collections::HashSet::new();
        for kind in OpKind::ALL {
            assert!(seen.insert(kind), "{kind:?} appears twice in OpKind::ALL");
        }
        assert_eq!(seen.len(), N_KINDS);
    }

    #[test]
    fn blas_implies_fusion_boundary() {
        // The predicate tests `FUSION_BOUNDARY` directly, so the table must
        // set both flags rather than relying on caller-side implication.
        for kind in OpKind::ALL {
            if kind.has(OpCaps::BLAS) {
                assert!(
                    kind.has(OpCaps::FUSION_BOUNDARY),
                    "{kind:?} is BLAS but not a fusion boundary"
                );
            }
        }
    }

    #[test]
    fn leaves_are_not_elementwise() {
        for kind in OpKind::ALL {
            if kind.has(OpCaps::LEAF) {
                assert!(!kind.has(OpCaps::ELEMENTWISE), "{kind:?}");
                assert!(!kind.has(OpCaps::FUSION_BOUNDARY), "{kind:?}");
            }
        }
    }

    #[test]
    fn nondeterministic_ops_are_never_const_foldable() {
        // Folding an RNG op would bake one draw into the graph.
        for kind in OpKind::ALL {
            if kind.has(OpCaps::NONDETERMINISTIC) {
                assert!(!kind.has(OpCaps::CONST_FOLDABLE), "{kind:?}");
            }
        }
    }

    #[test]
    fn nested_body_flag_agrees_with_the_subgraph_accessor() {
        // The flag is the cheap answer; `Op::subgraphs` is the real one. They
        // must not drift — `has_nested_body` is used to decide whether to
        // recurse at all.
        let g = || Box::new(crate::Graph::new("b"));
        let cases: Vec<Op> = vec![
            Op::MatMul,
            Op::StopGradient,
            Op::If {
                then_branch: g(),
                else_branch: g(),
            },
            Op::While {
                cond: g(),
                body: g(),
                max_iterations: None,
            },
            Op::Scan {
                body: g(),
                length: 1,
                save_trajectory: false,
                num_bcast: 0,
                num_xs: 0,
                num_checkpoints: 0,
            },
            Op::CustomFn {
                fwd_body: g(),
                vjp_body: None,
                jvp_body: None,
                num_inputs: 1,
            },
        ];
        for op in cases {
            assert_eq!(
                op.has_nested_body(),
                !op.subgraphs().is_empty(),
                "{:?} disagrees with its subgraph accessor",
                op.kind()
            );
        }
    }

    #[test]
    fn spot_check_known_classifications() {
        assert!(OpKind::Input.has(OpCaps::LEAF));
        assert!(OpKind::Activation.has(OpCaps::ELEMENTWISE.union(OpCaps::CONST_FOLDABLE)));
        assert!(OpKind::MatMul.has(OpCaps::BLAS));
        assert!(OpKind::Reduce.has(OpCaps::REDUCTION));
        assert!(OpKind::Scan.has(OpCaps::NESTED_BODY));
        assert!(OpKind::RngNormal.has(OpCaps::NONDETERMINISTIC));
        assert!(OpKind::FusedSwiGLU.has(OpCaps::FUSED));
        // Reshape is host-foldable but not elementwise — a shape-only op.
        assert!(OpKind::Reshape.has(OpCaps::CONST_FOLDABLE));
        assert!(!OpKind::Reshape.has(OpCaps::ELEMENTWISE));
        // Transpose is opaque to the optimizer's categories.
        assert!(OpKind::Transpose.caps().is_empty());
    }

    #[test]
    fn fused_kinds_matches_the_flag() {
        let listed = fused_kinds();
        assert!(!listed.is_empty());
        for kind in &listed {
            assert!(kind.has(OpCaps::FUSED));
        }
        assert_eq!(
            listed.len(),
            OpKind::ALL.iter().filter(|k| k.has(OpCaps::FUSED)).count()
        );
    }
}
