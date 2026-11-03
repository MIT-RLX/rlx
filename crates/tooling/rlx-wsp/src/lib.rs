// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **rlx-wsp** — weight-space programming harness.
//!
//! An LLM programs a network by reading and writing its weights directly,
//! reasoning about what each unit computes, instead of training it. This crate
//! is the sandbox for that: the weights of an rlx graph as a labelled,
//! checkpointed store ([`WeightSpace`]), a JSON tool surface ([`tools`]) an
//! agent loop or MCP server can register as-is, and a forward-pass budget that
//! is the agent's only way to run the network.
//!
//! - [`env`] — scored environments with labelled observations (`Bounce`).
//! - [`orchard`] — `Orchard`: many objects, fruit and bombs; a planner reference.
//!   `Gusts`: the same game with motion rules the agent must discover.
//! - [`mnist`] — `mnist-fix`: repair a backdoored MNIST classifier.
//! - [`mlp`] — the programmable MLP, built and executed as an rlx graph.
//! - [`space`] — weights, edits, checkpoints, budgeted probe / rollout.
//! - [`views`] — context-sized reads: stats, slices, top-k, norms, SVD.
//! - [`tools`] — `{"tool": ..}` dispatch and the tool schemas.
//! - [`baselines`] — reference programs and the gradient-free ES baseline.
//! - [`mcp`] — the tools as a stdio MCP server.
//! - [`script`] — sandboxed JavaScript that computes weights / tool calls.
//! - [`verify`] — per-unit ablation and a check of the agent's account.
//! - [`train`] / [`warmstart`] — agent weights as a start for SGD.
//! - [`export`] — bake into `.rlxmodel`, int8 check, FPGA RTL.
//! - [`agent`] — launch an agent CLI (Claude Code, Cursor `agent`, Copilot)
//!   against the MCP server and score what it wrote.

pub mod agent;
pub mod baselines;
pub mod blocks;
pub mod bundle;
pub mod env;
pub mod export;
pub mod mcp;
pub mod mlp;
pub mod mnist;
pub mod orchard;
pub mod pg;
pub mod roles;
pub mod script;
pub mod space;
pub mod tasks;
pub mod tools;
pub mod train;
pub mod verify;
pub mod views;
pub mod warmstart;

pub use env::{Bounce, Env, EnvKind, EnvSpec};
pub use mlp::{Act, MlpSpec};
pub use orchard::{Orchard, Rules};
pub use space::{BATCH, RolloutReport, SCORE_SEED, State, Tensor, WeightSpace};
pub use tools::{handle_json, tool_schemas};
