// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Fluent builder methods on [`ModelFlow`] — sugar over [`FlowStage`].

use std::path::Path;
use std::sync::Arc;

use crate::blocks::{
    AttnMaskStage, BertEncoderLayerSpec, BertEncoderLayerStage, BertQkvStyle,
    BindDecodeInputsStage, ClsTokenPoolStage, CustomStage, EmbedStage, GatherAddStage,
    GatherFromInputStage, GatherLastTokenStage, GeluFfnStage, LayerNormStage, LinearStage,
    LlamaDecodeLayerStage, LlamaDecoderSpec, LlamaDecoderStage, LlamaKvTapStage, LmHeadStage,
    NomicEncoderLayerSpec, NomicEncoderLayerStage, RepeatStage, ResidualAddStage,
    ResidualSaveStage, RmsNormStage, RopeTablesStage, SelfAttnPrefillSpec, SelfAttnPrefillStage,
    SwiGluStage, dinov2_layer_fused, llama_prefill_layer_composed, llama_prefill_layer_fused,
    nomic_vision_layer_fused,
};
use crate::escape::Emit;
use crate::flow::ModelFlow;
use crate::layer::LayerStack;
use crate::profile::CompileProfile;
use crate::side::SideOutputs;
use crate::stage::FlowStage;
use crate::stream::{DualStreamStage, LoadStreamStage, StoreStreamStage};
use crate::value::FlowValue;

impl ModelFlow {
    /// Load tier-1 profile from a `*.rlx.toml` file (falls back to default on error).
    pub fn profile_file(mut self, path: impl AsRef<Path>, default: fn() -> CompileProfile) -> Self {
        self.profile = CompileProfile::from_toml_path(path.as_ref()).unwrap_or_else(|_| default());
        self
    }

    /// Encoder / embedding model defaults (Direct lowering, no KV fusion).
    pub fn profile_encoder(mut self) -> Self {
        self.profile = CompileProfile::encoder();
        self
    }

    /// Gather rows from a side input into the primary flow (starts embedding stack).
    pub fn gather_from_input(
        mut self,
        input_name: impl Into<String>,
        weight_key: impl Into<String>,
    ) -> Self {
        self.stages
            .push(FlowStage::GatherFromInput(GatherFromInputStage::new(
                input_name, weight_key, 0,
            )));
        self
    }

    /// Add an embedding looked up from a side input.
    pub fn gather_add(
        mut self,
        input_name: impl Into<String>,
        weight_key: impl Into<String>,
    ) -> Self {
        self.stages.push(FlowStage::GatherAdd(GatherAddStage::new(
            input_name, weight_key, 0,
        )));
        self
    }

    /// LayerNorm with separate gamma/beta weights.
    pub fn layer_norm(
        mut self,
        gamma_key: impl Into<String>,
        beta_key: impl Into<String>,
        eps: f32,
    ) -> Self {
        self.stages.push(FlowStage::LayerNorm(LayerNormStage::new(
            gamma_key, beta_key, eps,
        )));
        self
    }

    /// BERT-style GELU FFN under a layer prefix.
    pub fn gelu_ffn(mut self, layer_prefix: impl Into<String>) -> Self {
        self.stages
            .push(FlowStage::GeluFfn(GeluFfnStage::hf_bert(layer_prefix)));
        self
    }

    /// Repeat NomicBERT encoder layers.
    pub fn repeat_nomic_layers(
        self,
        count: usize,
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Self {
        self.repeat_layers(count, move |i| FlowStage::Named {
            name: format!("layer{i}"),
            inner: std::sync::Arc::new(FlowStage::NomicEncoderLayer(NomicEncoderLayerStage::new(
                NomicEncoderLayerSpec::hf(
                    format!("encoder.layers.{i}"),
                    hidden_size,
                    num_heads,
                    head_dim,
                    eps,
                ),
            ))),
        })
    }

    /// BERT-style encoder layer (fused QKV + padding-mask attention + GELU FFN).
    pub fn bert_encoder_layer(mut self, spec: BertEncoderLayerSpec) -> Self {
        self.stages
            .push(FlowStage::BertEncoderLayer(BertEncoderLayerStage::new(
                spec,
            )));
        self
    }

    /// Repeat BERT encoder layers with auto-named prefixes.
    pub fn repeat_bert_layers(
        self,
        count: usize,
        prefix: impl Into<String>,
        qkv_style: BertQkvStyle,
        hidden_size: usize,
        num_heads: usize,
        eps: f32,
    ) -> Self {
        let prefix = prefix.into();
        self.repeat_layers(count, move |i| {
            let lp = if prefix.is_empty() {
                format!("encoder.layer.{i}")
            } else {
                format!("{prefix}.encoder.layer.{i}")
            };
            FlowStage::Named {
                name: format!("layer{i}"),
                inner: std::sync::Arc::new(FlowStage::BertEncoderLayer(
                    BertEncoderLayerStage::new(BertEncoderLayerSpec::hf(
                        lp,
                        qkv_style,
                        hidden_size,
                        num_heads,
                        eps,
                    )),
                )),
            }
        })
    }

    /// Synthesize an all-ones attention mask for vision encoders (no padding).
    pub fn attn_mask_ones(mut self, batch: usize, seq: usize) -> Self {
        self.stages
            .push(FlowStage::AttnMask(AttnMaskStage::ones(batch, seq)));
        self
    }

    /// Repeat DINOv2 ViT encoder blocks.
    pub fn repeat_dinov2_layers(
        self,
        count: usize,
        hidden_size: usize,
        num_heads: usize,
        eps: f32,
    ) -> Self {
        self.repeat_layers(count, move |i| {
            dinov2_layer_fused(i, hidden_size, num_heads, eps)
        })
    }

    /// Repeat NomicVision encoder blocks.
    pub fn repeat_vision_layers(
        self,
        count: usize,
        hidden_size: usize,
        num_heads: usize,
        eps: f32,
    ) -> Self {
        self.repeat_layers(count, move |i| {
            nomic_vision_layer_fused(i, hidden_size, num_heads, eps)
        })
    }

    /// Pool CLS token: `[batch, seq, hidden]` → `[batch, hidden]`.
    pub fn cls_token_pool(mut self, batch: usize, hidden: usize) -> Self {
        self.stages
            .push(FlowStage::ClsTokenPool(ClsTokenPoolStage::new(
                batch, hidden,
            )));
        self
    }

    /// Fusion-first prefill defaults.
    pub fn profile_prefill(mut self) -> Self {
        self.profile = CompileProfile::llama32_prefill();
        self
    }

    /// Decode / KV-cache defaults (`Fusable` lowering).
    pub fn profile_decode(mut self) -> Self {
        self.profile = CompileProfile::llama32_decode();
        self
    }

    /// Token embedding (`model.embed_tokens.weight` by default).
    pub fn embed(mut self, weight_key: impl Into<String>) -> Self {
        self.stages
            .push(FlowStage::Embed(EmbedStage::token(weight_key)));
        self
    }

    /// HuggingFace-style token embedding table.
    pub fn token_embed(self) -> Self {
        self.embed("model.embed_tokens.weight")
    }

    /// Precomputed RoPE sin/cos tables stored as params.
    pub fn rope_tables(mut self, tables: RopeTablesStage) -> Self {
        self.stages.push(FlowStage::RopeTables(tables));
        self
    }

    /// Rank-1 zero vector for RMSNorm beta slots (LLaMA has no beta).
    pub fn zero_beta(self, len: usize) -> Self {
        self.zero_beta_named("zero_beta", len)
    }

    pub fn zero_beta_named(mut self, name: impl Into<String>, len: usize) -> Self {
        self.stages.push(FlowStage::ZeroBeta {
            name: name.into(),
            len,
        });
        self
    }

    /// Bind decode inputs (call after declaring `rope_cos`, `past_k_*`, …).
    pub fn bind_decode_inputs(mut self, num_layers: usize, custom_mask: bool) -> Self {
        self.stages
            .push(FlowStage::BindDecodeInputs(BindDecodeInputsStage {
                num_layers,
                use_custom_mask: custom_mask,
            }));
        self
    }

    /// Repeat a per-layer stage `count` times (layer index passed to closure).
    pub fn repeat_layers(
        mut self,
        count: usize,
        stage_for_layer: impl Fn(usize) -> FlowStage + Send + Sync + 'static,
    ) -> Self {
        self.stages
            .push(FlowStage::Repeat(RepeatStage::new(count, stage_for_layer)));
        self
    }

    /// Named decoder layer (shows up in fusion / inspect dumps).
    pub fn named_layer(mut self, name: impl Into<String>, inner: FlowStage) -> Self {
        self.stages.push(FlowStage::Named {
            name: name.into(),
            inner: Arc::new(inner),
        });
        self
    }

    /// Build a named layer from a [`LayerStack`] closure.
    pub fn layer(
        self,
        name: impl Into<String>,
        build: impl FnOnce(LayerStack) -> LayerStack,
    ) -> Self {
        self.raw_stage(build(LayerStack::named(name)).build())
    }

    /// Fused LLaMA prefill layer (default fast path).
    pub fn llama_prefill_layer(self, layer_idx: usize, spec: LlamaDecoderSpec) -> Self {
        self.raw_stage(llama_prefill_layer_fused(layer_idx, spec))
    }

    /// Composed LLaMA prefill layer (small blocks — customize via [`LayerStack`]).
    pub fn llama_prefill_layer_composed(self, layer_idx: usize, spec: LlamaDecoderSpec) -> Self {
        self.raw_stage(llama_prefill_layer_composed(layer_idx, spec))
    }

    pub fn linear(mut self, weight_key: impl Into<String>, transpose: bool) -> Self {
        self.stages
            .push(FlowStage::Linear(LinearStage::new(weight_key, transpose)));
        self
    }

    pub fn residual_save(mut self) -> Self {
        self.stages.push(FlowStage::ResidualSave(ResidualSaveStage));
        self
    }

    pub fn residual_add(mut self) -> Self {
        self.stages.push(FlowStage::ResidualAdd(ResidualAddStage));
        self
    }

    pub fn swiglu(
        mut self,
        gate_key: impl Into<String>,
        up_key: impl Into<String>,
        down_key: impl Into<String>,
    ) -> Self {
        self.stages.push(FlowStage::SwiGlu(SwiGluStage::new(
            gate_key, up_key, down_key,
        )));
        self
    }

    pub fn swiglu_hf_mlp(mut self, prefix: impl Into<String>) -> Self {
        self.stages
            .push(FlowStage::SwiGlu(SwiGluStage::hf_mlp(prefix)));
        self
    }

    pub fn self_attn_prefill(mut self, spec: SelfAttnPrefillSpec) -> Self {
        self.stages
            .push(FlowStage::SelfAttnPrefill(SelfAttnPrefillStage::new(spec)));
        self
    }

    pub fn gdn_scan(mut self, stage: crate::blocks::GdnScanStage) -> Self {
        self.stages.push(FlowStage::GdnScan(stage));
        self
    }

    pub fn store_stream(mut self, name: impl Into<String>) -> Self {
        self.stages
            .push(FlowStage::StoreStream(StoreStreamStage::new(name)));
        self
    }

    pub fn load_stream(mut self, name: impl Into<String>) -> Self {
        self.stages
            .push(FlowStage::LoadStream(LoadStreamStage::new(name)));
        self
    }

    /// Bind declared graph inputs into named streams (multi-input models).
    ///
    /// Example: FLUX `.bind_inputs_to_streams(&[("hidden", "img"), ("encoder", "txt")])`.
    pub fn bind_inputs_to_streams(
        mut self,
        pairs: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        let pairs: Vec<(String, String)> = pairs
            .into_iter()
            .map(|(input, stream)| (input.into(), stream.into()))
            .collect();
        self.stages.push(FlowStage::Custom(CustomStage::named(
            "bind_inputs_to_streams",
            move |emit, primary| {
                let primary = primary.ok_or_else(|| {
                    anyhow::anyhow!("bind_inputs_to_streams requires primary input")
                })?;
                for (input_name, stream_name) in &pairs {
                    let value = emit.flow_input(input_name)?;
                    emit.state.streams.insert(stream_name.clone(), value);
                }
                Ok(Some(primary))
            },
        )));
        self
    }

    pub fn dual_stream<F>(
        mut self,
        name: impl Into<String>,
        stream_a: impl Into<String>,
        stream_b: impl Into<String>,
        f: F,
    ) -> Self
    where
        F: Fn(&mut Emit<'_>, FlowValue, FlowValue) -> anyhow::Result<(FlowValue, FlowValue)>
            + Send
            + Sync
            + 'static,
    {
        self.stages.push(FlowStage::DualStream(DualStreamStage::new(
            name, stream_a, stream_b, f,
        )));
        self
    }

    pub fn plugin<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut Emit<'_>, Option<FlowValue>) -> anyhow::Result<Option<FlowValue>>
            + Send
            + Sync
            + 'static,
    {
        self.stages.push(crate::plugin::plugin(f));
        self
    }

    pub fn plugin_named<F>(mut self, name: impl Into<String>, f: F) -> Self
    where
        F: Fn(&mut Emit<'_>, Option<FlowValue>) -> anyhow::Result<Option<FlowValue>>
            + Send
            + Sync
            + 'static,
    {
        self.stages.push(crate::plugin::plugin_named(name, f));
        self
    }

    /// Hidden states output (no LM head).
    pub fn hidden_states(self) -> Self {
        self.output("hidden")
    }

    /// LLaMA prefill decoder block at `layer_idx`.
    pub fn llama_decoder_layer(
        self,
        layer_idx: usize,
        spec: crate::blocks::LlamaDecoderSpec,
    ) -> Self {
        self.named_layer(
            format!("layer{layer_idx}"),
            FlowStage::LlamaDecoder(LlamaDecoderStage::layer(layer_idx, spec)),
        )
    }

    /// LLaMA decode block with KV-cache concat.
    pub fn llama_decode_layer(
        self,
        layer_idx: usize,
        spec: crate::blocks::LlamaDecodeLayerSpec,
        kv_out: SideOutputs,
    ) -> Self {
        self.named_layer(
            format!("layer{layer_idx}"),
            FlowStage::LlamaDecodeLayer(LlamaDecodeLayerStage::layer(
                layer_idx,
                spec,
                kv_out.inner(),
            )),
        )
    }

    /// Side-effect K/V tap before a prefill layer (exports cache tensors).
    pub fn llama_kv_tap(
        mut self,
        layer_idx: usize,
        head_dim: usize,
        eps: f32,
        sink: &SideOutputs,
    ) -> Self {
        self.stages
            .push(FlowStage::LlamaKvTap(LlamaKvTapStage::layer(
                layer_idx,
                head_dim,
                eps,
                sink.inner(),
            )));
        self
    }

    /// Final RMSNorm before LM head (`model.norm.weight` by default).
    pub fn final_norm(self, eps: f32) -> Self {
        self.rms_norm("model.norm.weight", eps)
    }

    pub fn rms_norm(mut self, weight_key: impl Into<String>, eps: f32) -> Self {
        self.stages
            .push(FlowStage::RmsNorm(RmsNormStage::new(weight_key, eps)));
        self
    }

    /// Gather last token (dynamic `last_token_idx` input).
    pub fn gather_last_token_dynamic(mut self, batch: usize) -> Self {
        self.stages
            .push(FlowStage::GatherLastToken(GatherLastTokenStage::dynamic(
                batch,
            )));
        self
    }

    /// Gather last token at fixed sequence length.
    pub fn gather_last_token_at(mut self, batch: usize, seq: usize) -> Self {
        self.stages.push(FlowStage::GatherLastToken(
            GatherLastTokenStage::static_last(batch, seq),
        ));
        self
    }

    /// Causal LM head — tied or separate weights.
    pub fn lm_head(
        mut self,
        vocab_size: usize,
        hidden_size: usize,
        tie_word_embeddings: bool,
    ) -> Self {
        let stage = if tie_word_embeddings {
            LmHeadStage::tied(vocab_size, hidden_size)
        } else {
            LmHeadStage::separate("lm_head.weight", vocab_size, hidden_size)
        };
        self.stages.push(FlowStage::LmHead(stage));
        self.output("logits")
    }

    /// Tier-2 escape hatch — append a raw stage.
    pub fn raw_stage(mut self, stage: FlowStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Append multiple raw stages in order.
    pub fn raw_stages(mut self, stages: impl IntoIterator<Item = FlowStage>) -> Self {
        self.stages.extend(stages);
        self
    }

    /// Run a list of stages as one nested sequence (side-effect stages allowed).
    pub fn sequence(mut self, stages: impl IntoIterator<Item = FlowStage>) -> Self {
        self.stages
            .push(FlowStage::Sequence(stages.into_iter().collect()));
        self
    }

    /// Conditionally transform the builder (e.g. optional vision tower).
    pub fn when(self, cond: bool, f: impl FnOnce(Self) -> Self) -> Self {
        if cond { f(self) } else { self }
    }

    /// Tier-2 custom subgraph — prefer promoting repeated patterns to blocks.
    pub fn custom<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut Emit<'_>, Option<FlowValue>) -> anyhow::Result<Option<FlowValue>>
            + Send
            + Sync
            + 'static,
    {
        self.stages.push(FlowStage::Custom(CustomStage::new(f)));
        self
    }

    /// Named custom subgraph (shows up in fusion / inspect dumps).
    pub fn custom_named<F>(mut self, name: impl Into<String>, f: F) -> Self
    where
        F: Fn(&mut Emit<'_>, Option<FlowValue>) -> anyhow::Result<Option<FlowValue>>
            + Send
            + Sync
            + 'static,
    {
        self.stages
            .push(FlowStage::Custom(CustomStage::named(name, f)));
        self
    }

    /// Patch the builder after preset assembly (arch recipes, Llama32Flow hooks).
    pub fn patch(self, f: impl FnOnce(Self) -> Self) -> Self {
        f(self)
    }
}
