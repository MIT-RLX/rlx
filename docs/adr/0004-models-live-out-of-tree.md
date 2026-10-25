# ADR 0004 — Model crates live in `rlx-models` and pin published rlx

## Status

Accepted.

## Context

Model implementations (Qwen, Gemma, Whisper, SAM, TTS/ASR codecs, …) change
on a different cadence than the framework and have different reviewers. Kept
in core, they pull model-specific shapes and tests into crates that should stay
model-agnostic (see [ADR 0003](0003-core-is-jax-shaped.md)).

## Decision

Model crates live in the separate `rlx-models` workspace and depend on
**published** rlx versions, without a `[patch]` to a local core checkout.

Core stays model-agnostic: no model implementations and no imports of model
crates in core tests. Graph optimizations that target a layer *shape* (fused
attention or transformer-layer peepholes) are framework code and live in core,
tested with structural graphs — including the generic blocks `rlx-flow` ships
(`llama_prefill_layer_composed`, `BertEncoderLayerStage`).

## Alternatives

- **Models in the core workspace.** One `cargo test` covers both, but every
  core change rebuilds every model and model churn lands in core review.
- **A `[patch]` from models to the core tree.** Convenient locally, but the
  models then validate an unreleased rlx and the published combination goes
  untested.

## Consequences

- Validating an unreleased backend change against a real model needs an
  explicit local `[patch.crates-io]` in `rlx-models`, or a release.
- Backend correctness is protected by structural and parity tests in core,
  not by model runs.
## License

MIT OR Apache-2.0.
