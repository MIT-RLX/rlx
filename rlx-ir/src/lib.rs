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
//! - **MIR** ([`mir`]) — fused tensor DAG; input to [`rlx_opt`].
//! - **LIR** ([`lir`]) — optimized MIR + arena buffer plan for backends.
//!
//! [`Graph`] is the primary DX surface. Use [`Graph::define`] for
//! fusion-first HIR builders, or [`Graph::new`] / [`GraphModule::mir`]
//! for primitive MIR. [`GraphModule`] tracks pipeline stage (HIR/MIR/LIR).
//!
//! - [`Graph`]: a DAG of tensor operations (like XLA's HloModule)
//! - [`Node`]: a single operation with typed inputs/outputs
//! - [`Op`]: the operation kind with parameters

pub mod ad;
pub mod async_copy;
pub mod const_check;
pub mod dtype;
pub mod dynamic;
pub mod env;
pub mod graph;
pub mod hir;
pub mod infer;
pub mod infer_shape;
pub mod inspect;
pub mod layout;
pub mod lir;
pub mod logical_kernel;
pub mod measure;
pub mod mir;
pub mod module;
pub mod nvfp4;
pub mod op;
pub mod op_registry;
pub mod ops;
pub mod perfetto;
pub mod phase;
pub mod pretty;
pub mod provenance;
pub mod quant;
pub mod rng;
pub use nvfp4::{FP4_E2M1_LUT, NVFP4_GROUP_SIZE, fp4_e2m1_to_f32, fp8_e4m3_scale_to_f32};
pub mod binding_manifest;
pub mod component;
pub mod hir_extension;
pub mod reflect;
#[cfg(feature = "serialize")]
pub mod serialize;
pub mod shape;
pub mod target;
pub mod variant;
pub mod verify;

pub use ad::AdPipelineStage;
pub use async_copy::{AsyncCopy, BarrierToken, DoubleBuffer, SyncCopy};
pub use dtype::{DType, Element, ElementSubtype};
pub use dynamic::sym;
pub use dynamic::{
    DimEnv, bind_graph, collect_dynamic_symbols, has_dynamic_dims, infer_bindings_from_f32_inputs,
    infer_bindings_from_inputs, same_binding, sync_concat_shapes, sync_graph_shapes,
    sync_narrow_ops, sync_reshape_ops,
};
pub use env::{RlxEnv, RuntimeOverrides, flag, is_unset, parse_or, set, unset, var, var_os};
pub use graph::{Graph, Node, NodeId};
pub use hir::{FusionPolicy, HirGraphExt, HirModule, HirMut, HirNode, HirNodeId, HirOp};
pub use infer::GraphExt;
pub use inspect::{
    inspect_buffer_plan, inspect_graph, inspect_graph_diff, inspect_hir, inspect_hir_stats,
    inspect_lir, inspect_mir, inspect_mir_diff, inspect_mir_stats,
};
pub use layout::{Coord2, Ragged, ShapeTuple, Strides2, Strides3, Tile2, Tile3};
pub use lir::{
    LirBufferPlan, LirBufferSlot, LirFingerprint, LirIoManifest, LirModule, LirViewAlias,
};
pub use logical_kernel::{
    KernelDispatchConfig, KernelDispatchPolicy, LogicalKernelEntry, logical_kinds_in_graph,
    registered_logical_kernels, should_lower_to_common,
};
pub use measure::{CacheBuster, Tick, time_ns};
pub use mir::{MirModule, MirNode, MirNodeId, MirOp};
pub use module::{GraphModule, GraphStage};
pub use op::{Op, OpKind};
pub use op_registry::{
    JvpContext, OpExtension, OpRegistry, VjpContext, VmapContext, global_registry, lookup_op,
    register_op,
};
pub use phase::{Phase, PhaseSchedule, derive_phases};
pub use provenance::{NodeOrigin, node_label, stamp_pass_origins};
pub use quant::{QuantMap, QuantScheme};
pub use rng::Philox4x32;
#[cfg(feature = "serialize")]
pub use serialize::{hir_from_json, hir_to_json, lir_from_json, lir_to_json};
pub use verify::{VerifyError, verify, verify_all, verify_shapes};

/// Lower a HIR module to MIR, then extract the legacy [`Graph`] API surface.
pub fn hir_to_graph(hir: HirModule) -> Result<Graph, hir::LowerError> {
    Ok(hir.lower_to_mir()?.into_graph())
}
pub use binding_manifest::{BindingManifest, IoBindingEntry, WeightBlock};
pub use component::{CompilationMode, ModelComponent};
pub use hir_extension::{
    HirExtensionFn, apply_hir_extensions, apply_hir_extensions_named, register_hir_extension,
    registered_hir_extensions,
};
pub use reflect::{
    BlockSpecialization, HirReflection, ManifestDiff, MirReflection, SpecializeBlockRecord,
    layout_for_binding, layout_from_lir, probe_block_specialization, symbolic_layout_hint,
};
pub use shape::{Dim, DimBinding, Shape};
pub use variant::{ModelPhase, ModelVariant};
