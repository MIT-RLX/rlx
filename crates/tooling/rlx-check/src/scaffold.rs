// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scaffolding for the downstream extension seams — generates ready-to-fill
//! templates for a custom op ([`op_template`]) or a `LayerStage` model block
//! ([`model_template`]). Backs `cargo rlx new-op` / `new-model`.
//!
//! The templates target the public extension surface (see `docs/extending.md`):
//! `rlx_ir::{OpExtension, register_op, custom_op, LowerContext}` for ops, and
//! `rlx_flow::{LayerStage, ModelFlow, FlowCtx}` for blocks.

/// `MyOp` / `my-op` / `my op` → `my_op`.
pub fn to_snake(s: &str) -> String {
    let mut out = String::new();
    let mut prev_alnum_lower = false;
    for c in s.chars() {
        if c == '-' || c == '_' || c.is_whitespace() {
            if !out.is_empty() && !out.ends_with('_') {
                out.push('_');
            }
            prev_alnum_lower = false;
        } else if c.is_uppercase() {
            if prev_alnum_lower {
                out.push('_');
            }
            out.extend(c.to_lowercase());
            prev_alnum_lower = false;
        } else {
            out.push(c);
            prev_alnum_lower = c.is_lowercase() || c.is_ascii_digit();
        }
    }
    out.trim_matches('_').to_string()
}

/// `my_op` / `my-op` → `MyOp`.
pub fn to_pascal(s: &str) -> String {
    to_snake(s)
        .split('_')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut ch = w.chars();
            match ch.next() {
                Some(f) => f.to_uppercase().chain(ch).collect::<String>(),
                None => String::new(),
            }
        })
        .collect()
}

/// Template for a custom op. Returns `(filename, contents)`.
pub fn op_template(name: &str) -> (String, String) {
    let pascal = to_pascal(name);
    let snake = to_snake(name);
    let file = format!("{snake}.rs");
    let content = format!(
        r#"// Custom op `{snake}` — scaffolded by `cargo rlx new-op`.
//
// Register once at startup with `{snake}::register()`, then build nodes via
// `graph.custom_op("{snake}", attrs, inputs)` (or the non-panicking
// `graph.try_custom_op(..)`).

use std::sync::Arc;

use rlx_ir::op::BinaryOp;
use rlx_ir::{{register_op, LowerContext, Node, NodeId, Op, OpExtension, Shape}};

/// IR-level extension: shape inference + (optionally) autodiff for `{snake}`.
#[derive(Debug)]
pub struct {pascal}Ir;

impl OpExtension for {pascal}Ir {{
    fn name(&self) -> &str {{
        "{snake}"
    }}

    fn num_inputs(&self) -> usize {{
        1 // TODO: how many tensor inputs your op takes
    }}

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {{
        // TODO: compute the output shape/dtype from the inputs.
        inputs[0].clone()
    }}

    // OPTION A (recommended): decompose to primitives. With this, the op fuses
    // and runs on EVERY backend with no kernel. Delete it if you ship a native
    // kernel instead (Option B).
    fn lower(&self, _node: &Node, ctx: &mut LowerContext) -> Option<NodeId> {{
        let x = ctx.inputs[0];
        let shape = ctx.out.node(x).shape.clone();
        // TODO: emit your decomposition into `ctx.out`; return the output node.
        Some(ctx.out.add_node(Op::Binary(BinaryOp::Add), vec![x, x], shape))
    }}

    // Optional: gradient rule. Default is non-differentiable.
    // fn vjp(&self, node: &Node, ctx: &mut rlx_ir::VjpContext) -> Vec<(usize, NodeId)> {{ .. }}
}}

// OPTION B: a native CPU kernel (also runs on CUDA/ROCm/wgpu/Vulkan via host
// staging). Implement this INSTEAD of `lower` for hand-tuned code. Requires a
// dependency on `rlx-cpu`.
//
// use rlx_cpu::op_registry::{{register_cpu_kernel, CpuKernel, CpuTensorMut, CpuTensorRef}};
// #[derive(Debug)]
// pub struct {pascal}Cpu;
// impl CpuKernel for {pascal}Cpu {{
//     fn execute(&self, inputs: &[CpuTensorRef<'_>], output: CpuTensorMut<'_>, attrs: &[u8])
//         -> Result<(), String>
//     {{
//         todo!("compute {snake} on the CPU")
//     }}
// }}

/// Register `{snake}` with the global registries. Call once at startup.
pub fn register() {{
    register_op(Arc::new({pascal}Ir));
    // register_cpu_kernel(Arc::new({pascal}Cpu)); // if using Option B
}}
"#
    );
    (file, content)
}

/// Template for a `LayerStage` model block. Returns `(filename, contents)`.
pub fn model_template(name: &str) -> (String, String) {
    let pascal = to_pascal(name);
    let snake = to_snake(name);
    let file = format!("{snake}.rs");
    let content = format!(
        r#"// Model block `{pascal}` — scaffolded by `cargo rlx new-model`.
//
// Drop it into any flow with `ModelFlow::new(..).layer_stage({pascal} {{ .. }})`.
// It composes primitives through `FlowCtx`, so it fuses and runs on every
// backend — no core edit, no `FlowStage` enum variant.

use rlx_flow::prelude::*;

/// A composable architecture block.
pub struct {pascal} {{
    // TODO: config + weight keys, e.g.:
    pub weight_key: String,
    pub eps: f32,
}}

impl LayerStage for {pascal} {{
    fn name(&self) -> &str {{
        "{snake}"
    }}

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        x: FlowValue,
    ) -> anyhow::Result<(FlowValue, StageArtifacts)> {{
        // Compose primitives via the curated FlowCtx builders — no raw HIR.
        // Available: param, matmul, linear, add, sub, mul, residual, activation,
        // relu, gelu, silu, rms_norm, cast.
        let h = ctx.rms_norm(&x, &self.weight_key, self.eps)?;
        let h = ctx.linear(&h, &self.weight_key, false)?;
        let out = ctx.relu(&h);

        // Optional auxiliary output (KV tap / aux head), auto-wired as a graph
        // output named "aux":
        // ctx.publish_side_output("aux", &h);

        Ok((out.clone(), StageArtifacts::hidden_only(out.shape().clone())))
    }}
}}
"#
    );
    (file, content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_conversions() {
        assert_eq!(to_snake("MyOp"), "my_op");
        assert_eq!(to_snake("my-op"), "my_op");
        assert_eq!(to_snake("my op"), "my_op");
        assert_eq!(to_snake("my_op"), "my_op");
        assert_eq!(to_snake("HTMLParser"), "htmlparser"); // acronyms collapse — fine for ids
        assert_eq!(to_pascal("my_op"), "MyOp");
        assert_eq!(to_pascal("gated-mlp"), "GatedMlp");
    }

    #[test]
    fn op_template_is_coherent() {
        let (file, content) = op_template("GatedGate");
        assert_eq!(file, "gated_gate.rs");
        assert!(content.contains("pub struct GatedGateIr"));
        assert!(content.contains(r#"fn name(&self) -> &str {"#));
        assert!(content.contains(r#""gated_gate""#));
        assert!(content.contains("impl OpExtension for GatedGateIr"));
        assert!(content.contains("fn register()"));
    }

    #[test]
    fn model_template_is_coherent() {
        let (file, content) = model_template("MyBlock");
        assert_eq!(file, "my_block.rs");
        assert!(content.contains("pub struct MyBlock"));
        assert!(content.contains("impl LayerStage for MyBlock"));
        assert!(content.contains("layer_stage(MyBlock"));
        assert!(content.contains("ctx.rms_norm"));
    }
}
