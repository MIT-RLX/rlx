// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RLX Tensor IR — the intermediate representation for the RLX ML compiler.
//!
//! This IR is:
//! - **Standalone**: no runtime, no backend, no framework coupling
//! - **Serializable**: graphs can be saved/loaded for AOT compilation
//! - **Optimizable**: designed for pattern-matching fusion and buffer planning
//!
//! The compiler pipeline has three named levels:
//!
//! - **HIR** ([`hir`]) — block-oriented IR for model builders (`Linear`,
//!   `SwiGLU`, `ResidualRmsNorm`, …).
//! - **MIR** ([`mir`]) — fused tensor DAG; input to `rlx_opt`.
//! - **LIR** ([`lir`]) — optimized MIR + arena buffer plan for backends.
//!
//! [`Graph`] is the primary DX surface. Use [`Graph::define`] for
//! fusion-first HIR builders, or [`Graph::new`] / [`GraphModule::mir`]
//! for primitive MIR. [`GraphModule`] tracks pipeline stage (HIR/MIR/LIR).
//!
//! - [`Graph`]: a DAG of tensor operations (like XLA's HloModule)
//! - [`Node`]: a single operation with typed inputs/outputs
//! - [`Op`]: the operation kind with parameters

// Without the default `std` feature only the graph/op/shape/verify core
// builds (`no_std` + `alloc`); everything else is gated on `std` below.
// Unit tests keep `std` for the harness and `println!`, so
// `cargo test --no-default-features` still exercises the no_std paths.
#![cfg_attr(not(any(feature = "std", test)), no_std)]

extern crate alloc;

/// `HashMap`/`HashSet` for the `no_std` subset: std's under `std`, ordered
/// maps without it (every key type in the subset is `Ord`).
#[allow(unused_imports)]
pub(crate) mod collections {
    #[cfg(not(feature = "std"))]
    pub use alloc::collections::{BTreeMap as HashMap, BTreeSet as HashSet};
    #[cfg(feature = "std")]
    pub use std::collections::{HashMap, HashSet};
}

/// Float methods `core` lacks (they call the platform libm under `std`),
/// supplied from the pure-Rust `libm` crate for the `no_std` subset.
/// Inherent methods win where they exist, so this only fills gaps.
#[cfg(not(any(feature = "std", test)))]
pub(crate) trait FloatExt: Sized {
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn round_ties_even(self) -> Self;
    fn fract(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;
}

#[cfg(not(any(feature = "std", test)))]
macro_rules! float_ext {
    ($t:ty, $sqrt:ident, $sin:ident, $cos:ident, $floor:ident, $ceil:ident, $round:ident,
     $rint:ident, $trunc:ident, $log2:ident, $log10:ident, $pow:ident) => {
        impl FloatExt for $t {
            fn sqrt(self) -> Self {
                libm::$sqrt(self)
            }
            fn sin(self) -> Self {
                libm::$sin(self)
            }
            fn cos(self) -> Self {
                libm::$cos(self)
            }
            fn floor(self) -> Self {
                libm::$floor(self)
            }
            fn ceil(self) -> Self {
                libm::$ceil(self)
            }
            fn round(self) -> Self {
                libm::$round(self)
            }
            fn round_ties_even(self) -> Self {
                libm::$rint(self)
            }
            fn fract(self) -> Self {
                self - libm::$trunc(self)
            }
            fn log2(self) -> Self {
                libm::$log2(self)
            }
            fn log10(self) -> Self {
                libm::$log10(self)
            }
            fn powf(self, n: Self) -> Self {
                libm::$pow(self, n)
            }
            fn powi(self, n: i32) -> Self {
                libm::$pow(self, n as $t)
            }
        }
    };
}
#[cfg(not(any(feature = "std", test)))]
float_ext!(
    f32, sqrtf, sinf, cosf, floorf, ceilf, roundf, rintf, truncf, log2f, log10f, powf
);
#[cfg(not(any(feature = "std", test)))]
float_ext!(
    f64, sqrt, sin, cos, floor, ceil, round, rint, trunc, log2, log10, pow
);

/// The `std` prelude names a `no_std` build loses; modules in the `no_std`
/// subset glob-import it so they read the same either way.
mod prelude {
    #[cfg(not(any(feature = "std", test)))]
    pub(crate) use crate::FloatExt;
    // Only the no_std build needs it; `std`'s own prelude covers the rest.
    #[cfg(not(any(feature = "std", test)))]
    pub use alloc::borrow::ToOwned;
    pub use alloc::boxed::Box;
    pub use alloc::format;
    pub use alloc::string::{String, ToString};
    pub use alloc::vec;
    pub use alloc::vec::Vec;
}

#[cfg(feature = "std")]
pub mod ad;
#[cfg(feature = "std")]
pub mod apple_bnns;
#[cfg(feature = "std")]
pub mod arena_range;
#[cfg(feature = "std")]
pub mod async_copy;
#[cfg(feature = "std")]
pub mod attention_layout;
pub mod audio;
#[cfg(feature = "std")]
pub mod bytes;
pub mod capability;
pub mod compose;
#[cfg(feature = "std")]
pub mod const_check;
pub mod double_single;
pub mod dtype;
pub mod dynamic;
#[cfg(feature = "std")]
pub mod env;
#[cfg(not(feature = "std"))]
#[path = "env_nostd.rs"]
pub mod env;
#[cfg(feature = "std")]
pub mod env_catalog;
#[cfg(feature = "std")]
pub mod env_registry;
#[cfg(feature = "std")]
pub mod equivalence;
pub mod fft;
#[cfg(feature = "std")]
pub mod filterbank;
pub mod gdn;
pub mod graph;
#[cfg(feature = "std")]
pub mod head;
pub mod hir;
pub mod infer;
pub mod infer_shape;
#[cfg(feature = "std")]
pub mod inspect;
#[cfg(feature = "std")]
pub mod kernel_schedule;
#[cfg(feature = "std")]
pub mod known_failures;
#[cfg(feature = "std")]
pub mod lanes;
#[cfg(feature = "std")]
pub mod layout;
#[cfg(feature = "std")]
pub mod lir;
#[cfg(feature = "std")]
pub mod logical_kernel;
pub mod lowp_codec;
#[cfg(feature = "std")]
pub mod measure;
pub mod mir;
#[cfg(feature = "std")]
pub mod module;
pub mod number_format;
#[cfg(feature = "std")]
pub mod numeric_check;
pub mod nvfp4;
pub mod op;
pub mod op_registry;
pub mod opcodes;
pub mod ops;
#[cfg(feature = "std")]
pub mod perfetto;
#[cfg(feature = "std")]
pub mod phase;
#[cfg(feature = "std")]
pub mod pretty;
pub mod provenance;
pub mod quant;
#[cfg(feature = "std")]
pub mod region_encode;
#[cfg(feature = "std")]
pub mod repr_check;
pub mod residual;
/// Decode state inside the graph (`stateful_decode`).
#[cfg(feature = "std")]
pub mod retarget;
#[cfg(feature = "std")]
pub mod rng;
#[cfg(feature = "std")]
pub mod stateful;
#[cfg(feature = "std")]
pub mod tensor_inspect;
#[cfg(feature = "std")]
pub mod ternary;
pub use nvfp4::{FP4_E2M1_LUT, NVFP4_GROUP_SIZE, fp4_e2m1_to_f32, fp8_e4m3_scale_to_f32};
#[cfg(feature = "std")]
pub mod binding_manifest;
#[cfg(feature = "std")]
pub mod component;
#[cfg(feature = "std")]
pub mod hir_extension;
#[cfg(feature = "std")]
pub mod reflect;
#[cfg(feature = "std")]
pub mod rf;
#[cfg(feature = "serialize")]
#[cfg(feature = "std")]
pub mod rlxmodel;
/// One constructible `Op` per `OpKind`, for gates that must cover the whole
/// op set. Feature `test-support`.
#[cfg(feature = "test-support")]
#[cfg(feature = "std")]
pub mod sample_ops;
#[cfg(feature = "serialize")]
#[cfg(feature = "std")]
pub mod serialize;
pub mod shape;
#[cfg(feature = "std")]
pub mod target;
pub mod text;
#[cfg(feature = "std")]
pub mod variant;
pub mod verify;

#[cfg(feature = "std")]
pub use ad::AdPipelineStage;
#[cfg(feature = "std")]
pub use async_copy::{AsyncCopy, BarrierToken, DoubleBuffer, SyncCopy};
#[cfg(feature = "std")]
pub use attention_layout::{
    ATTENTION_FLASH_MAX_HEAD_DIM, AttentionGeom, AttentionLaunchStrides,
    attention_dispatch_use_row, attention_geom, attention_launch_strides, cpu_attention_bshd,
    cpu_attention_packed_bshd_qkv, detect_packed_bshd_qkv_attention, mask_strides_bhsd,
    mask_strides_for_shape, packed_bshd_narrow_elidable, packed_bshd_qkv_strides, strides_bhsd,
    strides_bshd, strides_for_shape,
};
pub use capability::{N_KINDS, OpCaps, fused_kinds};
pub use compose::{ComposeError, Part, Partition, Splice, Spliced};
pub use dtype::{DType, Element, ElementSubtype, scalar_constant_bytes};
pub use dynamic::sym;
pub use dynamic::{
    DimEnv, bind_graph, collect_dynamic_symbols, has_dynamic_dims, infer_bindings_from_f32_inputs,
    infer_bindings_from_inputs, same_binding, sync_concat_shapes, sync_expand_ops,
    sync_graph_shapes, sync_narrow_ops, sync_reshape_ops,
};
#[cfg(feature = "std")]
pub use env::{RlxEnv, RuntimeOverrides, flag, is_unset, parse_or, set, unset, var, var_os};
#[cfg(feature = "std")]
pub use env_catalog::{EnvVarDoc, format_catalog as format_env_catalog};
#[cfg(feature = "std")]
pub use env_registry::{
    EnvKind, EnvLayer, EnvStability, EnvVarEntry, REGISTRY as ENV_REGISTRY, canonical_name,
    format_registry_markdown, is_registered, lookup as env_lookup, public_catalog_docs,
};

/// Public `RLX_*` catalog (`just env-catalog`).
#[cfg(feature = "std")]
pub fn env_catalog() -> &'static [EnvVarDoc] {
    public_catalog_docs()
}

#[cfg(feature = "std")]
pub use equivalence::{
    IgnoreConfig, fingerprint, node_value_key, ops_deep_eq, structural_hash, structurally_eq,
};
pub use fft::{
    FftAlgo, FftContraction, FftGpuPlan, FftMeta, FftNorm, FftTwiddles, fft_meta, fftn_axes_all,
    normalize_fftn_axes,
};
pub use gdn::GdnBackwardLayout;
pub use graph::{Graph, Node, NodeId};
pub use hir::{
    FusionPolicy, HirGraphExt, HirModule, HirMut, HirNode, HirNodeId, HirOp,
    window_token_gather_bsn, window_token_scatter_bsn,
};
pub use infer::GraphExt;
#[cfg(feature = "std")]
pub use inspect::{
    inspect_buffer_plan, inspect_graph, inspect_graph_diff, inspect_hir, inspect_hir_stats,
    inspect_lir, inspect_mir, inspect_mir_diff, inspect_mir_stats,
};
#[cfg(feature = "std")]
pub use layout::{Coord2, Ragged, ShapeTuple, Strides2, Strides3, Tile2, Tile3};
#[cfg(feature = "std")]
pub use lir::{
    LirBufferPlan, LirBufferSlot, LirFingerprint, LirIoManifest, LirModule, LirViewAlias,
};
#[cfg(feature = "std")]
pub use logical_kernel::{
    KernelDispatchConfig, KernelDispatchPolicy, LogicalKernelEntry, logical_kinds_in_graph,
    registered_logical_kernels, should_lower_to_common,
};
#[cfg(feature = "std")]
pub use measure::{CacheBuster, Tick, time_ns};
pub use mir::{MirModule, MirNode, MirNodeId, MirOp};
#[cfg(feature = "std")]
pub use module::{GraphModule, GraphStage};
pub use number_format::{FloatFormat, FormatScaling, NumberFormat};
#[cfg(feature = "std")]
pub use numeric_check::{BadValue, DebugMode, DebugScanner, NanReport, check_node, first_bad};
pub use op::{
    AdaNormKind, ChainOperand, ChainStep, Op, OpKind, PadMode, RegionPrologue, RopeStyle,
    ScatterNdReduction, SpdMatFn, SynthBwdWrt, SynthKind, TransformStep,
};
pub use op_registry::{
    CustomOpError, JvpContext, LowerContext, OpExtension, VjpContext, VmapContext,
    is_op_registered, lookup_op,
};
#[cfg(feature = "std")]
pub use op_registry::{OpRegistry, global_registry, register_op, register_op_strict};
pub use ops::attention::attention_kind_op;
pub use ops::backward::{unpack_ada_layer_norm_backward, unpack_gated_residual_backward};
pub use ops::dsp::{FirMode, iir_impulse_response};
#[cfg(feature = "std")]
pub use phase::{Phase, PhaseSchedule, derive_phases};
pub use provenance::{NodeOrigin, node_label, stamp_pass_origins};
/// Alias kept for existing `ENV_CATALOG` imports.
#[cfg(feature = "std")]
pub use public_catalog_docs as ENV_CATALOG;
pub use quant::{QuantMap, QuantScheme, ScaleLayout, ScaledFormat};
#[cfg(feature = "std")]
pub use region_encode::{
    FK_BATCH_SINGLE_KERNEL_MAX, PrologueLaunchGrid, REGION_META_WORDS, REGION_PROLOGUE_NONE,
    REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW, RegionNchwDims, batch_region_slice_dst_off_f32,
    batch_region_slice_elems, batch_region_slice_shape, encode_chain_operand, encode_chain_steps,
    encode_elementwise_region_meta, encode_prologue_tail, fk_batch_single_kernel_enabled,
    fk_batch_use_single_launch,
};
#[cfg(feature = "std")]
pub use rng::{
    BNNS_AES_CTR_STATE_SIZE, BnnsAesCtr128, GUMBEL_UNIFORM_EPS, NativeBnnsGumbelMax, Philox4x32,
    RngBackend, RngOptions, argmax_f32, bnns_categorical_from_uniform, bnns_entropy_seed,
    bnns_generator_state_is_fresh, bnns_gumbel_from_uniform, bnns_gumbel01_from_bytes,
    bnns_seed_from_generator_state, bnns_uniform_f32_from_bytes, bnns_uniform_f32_from_u32,
    bnns_uniform_int_from_u32, combine_seed, fill_normal_like, fill_uniform_like,
    gumbel_max_argmax_with, gumbel01_from_uniform, ort_engine_seed,
};
#[cfg(feature = "serialize")]
#[cfg(feature = "std")]
pub use serialize::{
    LIR_BIN_VERSION, hir_from_json, hir_to_json, lir_from_bytes, lir_from_json, lir_to_bytes,
    lir_to_json,
};
pub use verify::{VerifyError, verify, verify_all, verify_shapes, verify_unique_leaf_names};

/// Lower a HIR module to MIR, then extract the legacy [`Graph`] API surface.
pub fn hir_to_graph(hir: HirModule) -> Result<Graph, hir::LowerError> {
    Ok(hir.lower_to_mir()?.into_graph())
}
#[cfg(feature = "std")]
pub use binding_manifest::{BindingManifest, IoBindingEntry, WeightBlock};
#[cfg(feature = "std")]
pub use component::{CompilationMode, ModelComponent};
#[cfg(feature = "std")]
pub use hir_extension::{
    HirExtensionFn, apply_hir_extensions, apply_hir_extensions_named, register_hir_extension,
    registered_hir_extensions,
};
#[cfg(feature = "std")]
pub use reflect::{
    BlockSpecialization, HirReflection, ManifestDiff, MirReflection, SpecializeBlockRecord,
    layout_for_binding, layout_from_lir, probe_block_specialization, symbolic_layout_hint,
};
#[cfg(feature = "std")]
pub use rf::{
    complex_div, const_f32, cs_degen_z_in, find_param_node, find_param_nodes, mag2, s11_from_z,
    scalar_f32,
};
pub use shape::{Dim, DimBinding, Shape, ada_modulation_launch, ada_modulation_lead_pack};
#[cfg(feature = "std")]
pub use variant::{ModelPhase, ModelVariant};
