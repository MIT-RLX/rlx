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

//! Block assembly-line API for RLX model builders.
//!
//! # Extending rlx from a downstream crate
//!
//! A novel architecture block can live **entirely in a downstream crate**
//! (e.g. a model crate in `rlx-models`) — no edit to this crate's closed
//! [`FlowStage`] enum, and no republish gate. Implement [`LayerStage`] for your
//! block, composing primitives through the [`FlowCtx`] emission surface, then
//! drop it into any flow:
//!
//! ```ignore
//! use rlx_flow::prelude::*;
//!
//! struct MyBlock { /* config, weight keys */ }
//! impl LayerStage for MyBlock {
//!     fn name(&self) -> &str { "my_block" }
//!     fn emit_layer(&self, ctx: &mut FlowCtx<'_>, input: FlowValue)
//!         -> anyhow::Result<(FlowValue, StageArtifacts)>
//!     {
//!         // compose primitives via ctx (load_param + HirMut builders) …
//!         # unimplemented!()
//!     }
//! }
//!
//! let flow = ModelFlow::new("my_model")
//!     .input("x", shape)
//!     .layer_stage(MyBlock { /* … */ });   // <- extension front door
//! ```
//!
//! Because the block lowers to ordinary primitives (not an opaque `Op::Custom`),
//! it stays visible to fusion and hits the fast path on every backend. For a new
//! *operation* rather than a *block*, see the two op seams in `rlx-ir`:
//! `Graph::custom_op` + [`rlx_ir::OpExtension`] (register a kernel, or an
//! [`OpExtension::lower`](rlx_ir::OpExtension::lower) decompose-to-primitives
//! rule so it runs on every backend with no kernel), and `Graph::custom_fn` for
//! a subgraph op with custom autodiff.

pub mod blocks;
mod composite;
mod context;
mod dsl;
pub mod escape;
mod execution;
mod extension;
mod flow;
mod layer;
mod plugin;
mod profile;
mod recipe;
pub mod rope;
mod side;
mod stage;
mod stage_contract;
mod stage_interfaces;
pub mod stream;
mod value;
mod weight;

pub mod prelude;

pub use blocks::RopeTablesStage;
pub use blocks::{
    BertEncoderLayerSpec, BertEncoderLayerStage, BertQkvStyle, ClsTokenPoolStage, FfnActivation,
    GeluFfnStage, NomicEncoderLayerSpec, NomicEncoderLayerStage, Qwen3DecodeLayerSpec,
    Qwen3DecoderSpec, Qwen3DecoderStage, VitSelfAttnSpec, dinov2_layer_fused, dit_ada_gated_linear,
    nomic_vision_layer_fused, transformer_encoder_layer,
};
pub use composite::LayerComposition;
// `FlowCtx` is the emission surface a downstream `LayerStage` receives, so it
// must be nameable outside the crate for the extension seam to be usable.
pub use context::{DecodeBindings, FlowCtx, FlowState, GdnInputSlots};
pub use escape::Emit;
pub use execution::{ExecutionPreset, ModelExecutionConfig};
pub use extension::FlowExtensionPlan;
pub use flow::{BuiltModel, ModelFlow};
pub use layer::LayerStack;
pub use plugin::{PluginStage, plugin, plugin_named};
pub use profile::{
    BackendOverrides, CompileProfile, CpuBackendProfile, FusionPolicyKind, FusionProfile,
    FusionTargetKind, MetalBackendProfile, MixedPrecisionKind, PassProfile, PrecisionKind,
    PrecisionProfile, ProfileMode,
};
pub use recipe::ModelRecipe;
pub use rope::{
    Llama3Scaling, YarnScaling, build_default_tables, build_mrope_tables, build_mrope_text_tables,
    build_tables, default_inv_freq, inv_freq_with_factors, llama3_scaled_inv_freq,
    mrope_row_for_sections, mrope_row_for_sections_ex, mrope_section_for_pair, mrope_sections4,
    ntk_scaled_inv_freq, yarn_scaled_inv_freq,
};
pub use side::SideOutputs;
pub use stage::FlowStage;
pub use stage_contract::{BlockAsLayer, DynStage, LayerStage, StageArtifacts};
pub use stage_interfaces::{AttentionStage, FfnStage, KvCacheContract, NormStage};
pub use stream::{
    DualStreamStage, LoadStreamStage, StoreStreamStage, dual_stream_stage, id as stream_id,
};
pub use value::FlowValue;
pub use weight::{MapWeights, WeightSource};

use std::collections::HashMap;

/// Compatibility shim: packed GGUF matmul weights (used by some model loaders).
#[derive(Debug, Clone, Default)]
pub struct GgufPackedParams {
    pub linears: HashMap<String, GgufPackedLinear>,
}

impl GgufPackedParams {
    pub fn get_linear(&self, key: &str) -> Option<&GgufPackedLinear> {
        self.linears.get(key)
    }
}

/// One packed linear weight: quantized bytes + bias.
#[derive(Debug, Clone)]
pub struct GgufPackedLinear {
    pub w_q: Vec<u8>,
    pub scheme: rlx_ir::quant::QuantScheme,
    pub in_dim: usize,
    pub out_dim: usize,
    pub bias: Vec<f32>,
}
