// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Alternative code-generation backends for the generated crate. The default
//! `graph` style (in [`crate::emit`]) targets the raw HIR builder; here we add:
//!
//! - **tensor** — the operator-overloaded `rlx_tensor::Tensor` DSL (PyTorch-like,
//!   most readable; covers the ops the Tensor API exposes and reports the rest).
//! - **flow** — an `rlx_flow::ModelFlow` with a single custom stage that builds
//!   the graph via the HIR builder, integrating with the flow/`WeightSource`
//!   ecosystem.
//!
//! All three consume the same [`Lowered`] `Call` list.

use crate::call::*;
use crate::nodeop::NodeOp;
use anyhow::{Result, bail};
use rlx_ir::op::{Activation, BinaryOp, MaskKind, ReduceOp};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitStyle {
    Graph,
    Tensor,
    Flow,
}

impl EmitStyle {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "graph" => EmitStyle::Graph,
            "tensor" => EmitStyle::Tensor,
            "flow" => EmitStyle::Flow,
            other => bail!("unknown emit style {other:?} (expected graph|tensor|flow)"),
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            EmitStyle::Graph => "graph",
            EmitStyle::Tensor => "tensor",
            EmitStyle::Flow => "flow",
        }
    }
}

fn ident(name: &str) -> String {
    let mut s = String::from("t_");
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            s.push(c);
        } else {
            s.push('_');
        }
    }
    s
}

fn dtype_tok(dt: rlx_ir::DType) -> String {
    format!("rlx_ir::DType::{}", dtype_token(dt))
}

fn shape_lit(dims: &[usize], dt: rlx_ir::DType) -> String {
    let ds: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
    format!(
        "rlx_ir::Shape::new(&[{}], {})",
        ds.join(", "),
        dtype_tok(dt)
    )
}

fn act_method(a: Activation) -> Option<&'static str> {
    Some(match a {
        Activation::Gelu => "gelu",
        Activation::GeluApprox => "gelu_approx",
        Activation::Silu => "silu",
        Activation::Relu => "relu",
        Activation::Sigmoid => "sigmoid",
        Activation::Tanh => "tanh",
        Activation::Exp => "exp",
        Activation::Sqrt => "sqrt",
        Activation::Rsqrt => "rsqrt",
        Activation::Abs => "abs",
        Activation::Sin => "sin",
        Activation::Cos => "cos",
        _ => return None,
    })
}

fn mask_lit(m: MaskKind) -> &'static str {
    match m {
        MaskKind::None => "MaskKind::None",
        MaskKind::Causal => "MaskKind::Causal",
        MaskKind::Custom => "MaskKind::Custom",
        MaskKind::Bias => "MaskKind::Bias",
        MaskKind::SlidingWindow(_) => "MaskKind::SlidingWindow(0)",
    }
}

// ── tensor style ─────────────────────────────────────────────────────────────

/// Generate `graph.rs` using the `rlx_tensor` DSL. Errors on any op the Tensor
/// API can't express (Pool / TopK / ConvTranspose / mask-tensor attention / …),
/// pointing the user at the `graph` or `flow` style.
pub fn emit_tensor_graph_rs(lo: &Lowered) -> Result<String> {
    if lo.outputs.len() != 1 {
        bail!(
            "tensor style supports single-output models only ({} outputs); use --emit-style graph",
            lo.outputs.len()
        );
    }
    let mut idents: HashMap<String, String> = HashMap::new();
    let mut body = String::new();
    let push = |s: String, b: &mut String| {
        b.push_str("        ");
        b.push_str(&s);
        b.push('\n');
    };
    let r = |idents: &HashMap<String, String>, name: &str| -> String {
        idents.get(name).cloned().unwrap_or_else(|| ident(name))
    };

    for i in &lo.inputs {
        let id = ident(&i.name);
        push(
            format!(
                "let {id} = g.input({:?}, {});",
                i.name,
                shape_lit(&i.shape, i.dtype)
            ),
            &mut body,
        );
        idents.insert(i.name.clone(), id);
    }
    for p in lo.params.iter().chain(lo.zero_params.iter()) {
        let id = ident(&p.value_id);
        push(
            format!(
                "let {id} = g.param({:?}, {});",
                p.key,
                shape_lit(&p.shape, p.dtype)
            ),
            &mut body,
        );
        idents.insert(p.value_id.clone(), id);
    }

    for ins in &lo.instrs {
        let res = ident(&ins.result);
        if let Some(note) = &ins.note {
            if !matches!(ins.call, Call::Alias(_)) {
                push(format!("// {note}"), &mut body);
            }
        }
        let expr: String = match &ins.call {
            Call::Alias(src) => {
                idents.insert(ins.result.clone(), r(&idents, src));
                continue;
            }
            Call::Mm(a, b) => format!("{}.mm(&{})", r(&idents, a), r(&idents, b)),
            Call::Binary(op, a, b) => {
                let sym = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    other => bail!("tensor style: binary {other:?} unsupported"),
                };
                format!("(&{} {sym} &{})", r(&idents, a), r(&idents, b))
            }
            Call::Act(act, a) => {
                if let Some(m) = act_method(*act) {
                    format!("{}.{m}()", r(&idents, a))
                } else if matches!(act, Activation::Neg) {
                    format!("(-&{})", r(&idents, a))
                } else {
                    bail!("tensor style: activation {act:?} unsupported")
                }
            }
            Call::Ln {
                x,
                gamma,
                beta,
                eps,
            } => format!(
                "{}.layer_norm(&{}, &{}, {eps}f32)",
                r(&idents, x),
                r(&idents, gamma),
                r(&idents, beta)
            ),
            Call::RmsNorm {
                x,
                gamma,
                beta,
                eps,
            } => format!(
                "{}.rms_norm(&{}, &{}, {eps}f32)",
                r(&idents, x),
                r(&idents, gamma),
                r(&idents, beta)
            ),
            Call::Reshape { x, shape } => {
                let s: Vec<String> = shape.iter().map(|d| format!("{d}i64")).collect();
                format!("{}.reshape(vec![{}])", r(&idents, x), s.join(", "))
            }
            Call::Transpose { x, perm } => {
                let p: Vec<String> = perm.iter().map(|d| format!("{d}usize")).collect();
                format!("{}.transpose(vec![{}])", r(&idents, x), p.join(", "))
            }
            Call::Narrow {
                x,
                axis,
                start,
                len,
            } => {
                format!("{}.narrow({axis}, {start}, {len})", r(&idents, x))
            }
            Call::Concat { xs, axis } => {
                let ids: Vec<String> = xs.iter().map(|n| format!("&{}", r(&idents, n))).collect();
                format!("g.cat(&[{}], {axis})", ids.join(", "))
            }
            Call::Gather {
                table,
                indices,
                axis,
            } => {
                format!(
                    "{}.gather(&{}, {axis})",
                    r(&idents, table),
                    r(&idents, indices)
                )
            }
            Call::Softmax { x, axis } => format!("{}.softmax({axis}i32)", r(&idents, x)),
            Call::Reduce {
                op,
                x,
                axes,
                keep_dim,
            } => {
                let m = match op {
                    ReduceOp::Mean => "mean",
                    ReduceOp::Sum => "sum",
                    ReduceOp::Max => "max",
                    ReduceOp::Min => "min",
                    ReduceOp::Prod => "prod",
                };
                let a: Vec<String> = axes.iter().map(|d| format!("{d}usize")).collect();
                format!("{}.{m}(vec![{}], {keep_dim})", r(&idents, x), a.join(", "))
            }
            Call::Cast { x, to } => format!("{}.cast({})", r(&idents, x), dtype_tok(*to)),
            Call::Attention {
                q,
                k,
                v,
                num_heads,
                head_dim,
                mask,
                ..
            } => format!(
                "{}.attention(&{}, &{}, {num_heads}, {head_dim}, {})",
                r(&idents, q),
                r(&idents, k),
                r(&idents, v),
                mask_lit(*mask)
            ),
            Call::Rope {
                x,
                cos,
                sin,
                head_dim,
            } => format!(
                "{}.rope(&{}, &{}, {head_dim})",
                r(&idents, x),
                r(&idents, cos),
                r(&idents, sin)
            ),
            Call::Full {
                value,
                shape,
                dtype,
            } => {
                if shape.iter().product::<usize>() != 1 {
                    bail!("tensor style: non-scalar constant unsupported");
                }
                format!("g.constant({value}f64, {})", dtype_tok(*dtype))
            }
            Call::Node(node) => match node {
                NodeOp::Compare { op, a, b, .. } => {
                    let m = match op {
                        rlx_ir::op::CmpOp::Eq => "eq",
                        rlx_ir::op::CmpOp::Ne => "ne",
                        rlx_ir::op::CmpOp::Lt => "lt",
                        rlx_ir::op::CmpOp::Le => "le",
                        rlx_ir::op::CmpOp::Gt => "gt",
                        rlx_ir::op::CmpOp::Ge => "ge",
                    };
                    format!("{}.{m}(&{})", r(&idents, a), r(&idents, b))
                }
                NodeOp::Where { cond, a, b, .. } => format!(
                    "{}.where_(&{}, &{})",
                    r(&idents, cond),
                    r(&idents, a),
                    r(&idents, b)
                ),
                NodeOp::Expand { x, target, .. } => {
                    let t: Vec<String> = target.iter().map(|d| format!("{d}i64")).collect();
                    format!("{}.broadcast_to(vec![{}])", r(&idents, x), t.join(", "))
                }
                NodeOp::BinaryShaped { op, a, b, .. } => match op {
                    BinaryOp::Pow => format!("{}.pow(&{})", r(&idents, a), r(&idents, b)),
                    other => bail!("tensor style: binary {other:?} unsupported (use graph/flow)"),
                },
                NodeOp::Pool { .. } => {
                    bail!("tensor style: Pool unsupported (use --emit-style graph or flow)")
                }
                NodeOp::TopK { .. } => {
                    bail!("tensor style: TopK unsupported (use --emit-style graph or flow)")
                }
                NodeOp::GroupNorm { .. } => {
                    bail!("tensor style: GroupNorm unsupported (use --emit-style graph or flow)")
                }
                NodeOp::ResizeNearest2x { .. } => bail!(
                    "tensor style: ResizeNearest2x unsupported (use --emit-style graph or flow)"
                ),
            },
            _ => bail!(
                "tensor style cannot express `{}` ({}); use --emit-style graph or flow",
                ins.result,
                ins.note.as_deref().unwrap_or("op has no rlx_tensor method")
            ),
        };
        push(format!("let {res} = {expr};"), &mut body);
        idents.insert(ins.result.clone(), res);
    }

    let out_ident = r(&idents, &lo.outputs[0]);
    let param_keys: Vec<String> = lo.params.iter().map(|p| format!("{:?}", p.key)).collect();
    let zero_params: Vec<String> = lo
        .zero_params
        .iter()
        .map(|z| format!("({:?}, {})", z.key, z.shape.iter().product::<usize>()))
        .collect();

    Ok(format!(
        r#"// AUTO-GENERATED by rlx-torch-import (tensor style) — do not edit by hand.
#![allow(nonstandard_style, unused_parens, clippy::all)]

use rlx_ir::op::MaskKind;

/// Weight keys to bind after compiling (loaded from safetensors).
pub const PARAM_KEYS: &[&str] = &[{param_keys}];
/// Synthesized zero params (key, numel).
pub const ZERO_PARAMS: &[(&str, usize)] = &[{zero_params}];

/// Build the model as an `rlx_tensor` graph (PyTorch-like DSL).
pub fn build_graph() -> rlx_ir::Graph {{
    rlx_tensor::graph({name:?}, |g| {{
{body}        {out_ident}
    }})
}}
"#,
        param_keys = param_keys.join(", "),
        zero_params = zero_params.join(", "),
        name = lo.name,
        body = body,
        out_ident = out_ident,
    ))
}

// ── flow style ───────────────────────────────────────────────────────────────

const FILL_HELPER: &str = r#"fn rlx_torch_fill(value: f64, dtype: rlx_ir::DType, numel: usize) -> Vec<u8> {
    let one: Vec<u8> = match dtype {
        rlx_ir::DType::F32 => (value as f32).to_le_bytes().to_vec(),
        rlx_ir::DType::F64 => value.to_le_bytes().to_vec(),
        rlx_ir::DType::I64 => (value as i64).to_le_bytes().to_vec(),
        rlx_ir::DType::I32 => (value as i32).to_le_bytes().to_vec(),
        _ => vec![],
    };
    let mut out = Vec::with_capacity(one.len() * numel);
    for _ in 0..numel { out.extend_from_slice(&one); }
    out
}"#;

/// Generate `graph.rs` using `rlx_flow::ModelFlow` + a single custom stage that
/// builds the graph through the HIR builder (covers all ops — same per-op
/// emission as the `graph` style, via [`crate::emit::emit_hir_ops`]).
pub fn emit_flow_graph_rs(lo: &Lowered) -> Result<String> {
    if lo.outputs.len() != 1 {
        bail!("flow style supports single-output models only; use --emit-style graph");
    }
    let mut idents: HashMap<String, String> = HashMap::new();
    let mut body = String::new();
    let bl = |s: String, b: &mut String| {
        b.push_str("            ");
        b.push_str(&s);
        b.push('\n');
    };
    let r = |idents: &HashMap<String, String>, name: &str| -> String {
        idents.get(name).cloned().unwrap_or_else(|| ident(name))
    };

    for i in &lo.inputs {
        let id = ident(&i.name);
        bl(
            format!("let {id} = emit.flow_input({:?})?.hir_id();", i.name),
            &mut body,
        );
        idents.insert(i.name.clone(), id);
    }
    for p in &lo.params {
        let id = ident(&p.value_id);
        bl(
            format!("let {id} = emit.load_param({:?}, false)?;", p.key),
            &mut body,
        );
        idents.insert(p.value_id.clone(), id);
    }
    for z in &lo.zero_params {
        let id = ident(&z.value_id);
        let numel: usize = z.shape.iter().product();
        bl(
            format!("let {id} = emit.synth_zeros({:?}, {numel});", z.key),
            &mut body,
        );
        idents.insert(z.value_id.clone(), id);
    }
    bl(
        "let mut b = rlx_ir::hir::HirMut::new(emit.hir());".to_string(),
        &mut body,
    );
    // Shared per-op HIR emission (identical to graph style).
    body.push_str(&crate::emit::emit_hir_ops(lo, &mut idents, &r));

    let out_ident = r(&idents, &lo.outputs[0]);
    let inputs_decl: Vec<String> = lo
        .inputs
        .iter()
        .map(|i| {
            format!(
                "        .input({:?}, {})",
                i.name,
                shape_lit(&i.shape, i.dtype)
            )
        })
        .collect();

    Ok(format!(
        r#"// AUTO-GENERATED by rlx-torch-import (flow style) — do not edit by hand.
#![allow(nonstandard_style, unused_imports, unused_mut, unused_variables, dead_code, clippy::all)]

use rlx_flow::{{ModelFlow, FlowStage}};
use rlx_flow::blocks::CustomStage;
use rlx_ir::HirGraphExt;
use rlx_ir::op::{{Activation, MaskKind}};

{fill}

/// Build a `ModelFlow`; params are pulled from the `WeightSource` at build time.
pub fn build_flow() -> ModelFlow {{
    ModelFlow::new({name:?})
{inputs_decl}
        .stage(FlowStage::Custom(CustomStage::new(|emit, _input| {{
{body}            let out_shape = b.shape({out_ident}).clone();
            drop(b);
            Ok(Some(emit.wrap({out_ident}, out_shape)))
        }})))
        .output("output")
}}
"#,
        fill = FILL_HELPER,
        name = lo.name,
        inputs_decl = inputs_decl.join("\n"),
        body = body,
        out_ident = out_ident,
    ))
}

pub fn flow_weights_rs() -> String {
    r#"// AUTO-GENERATED by rlx-torch-import (flow style).
use rlx_flow::MapWeights;
use std::path::Path;

/// Load `weights.safetensors` into an in-memory `MapWeights` (WeightSource).
pub fn load_weights(dir: &Path) -> anyhow::Result<MapWeights> {
    let bytes = std::fs::read(dir.join("weights.safetensors"))?;
    let mut mw = MapWeights::default();
    if !bytes.is_empty() {
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        for name in st.names() {
            let v = st.tensor(name)?;
            let shape: Vec<usize> = v.shape().to_vec();
            let raw = v.data();
            let data: Vec<f32> = match v.dtype() {
                safetensors::Dtype::F32 => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                safetensors::Dtype::F16 => raw.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
                safetensors::Dtype::BF16 => raw.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
                safetensors::Dtype::F64 => raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
                safetensors::Dtype::I64 => raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
                other => anyhow::bail!("unsupported weight dtype {other:?}"),
            };
            mw.insert(name.to_string(), data, shape);
        }
    }
    Ok(mw)
}
"#
    .to_string()
}

pub fn flow_lib_rs(lo: &Lowered) -> String {
    format!(
        r#"//! Native RLX model `{name}` — generated by `rlx-torch-import` (flow style).
#![allow(nonstandard_style, unused_imports, clippy::all)]

pub mod graph;
pub mod weights;

pub use weights::load_weights;

/// Compile the model on `device`, binding weights from `dir/weights.safetensors`.
pub fn compile(
    device: rlx_runtime::Device,
    dir: &std::path::Path,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {{
    let mut weights = load_weights(dir)?;
    let built = graph::build_flow().build(&mut weights)?;
    let (hir, params) = built.into_parts()?;
    let mut compiled = rlx_runtime::Session::new(device)
        .compile_hir_with(hir, &rlx_runtime::CompileOptions::default())
        .map_err(|e| anyhow::anyhow!("{{e}}"))?;
    for (name, data) in params {{
        compiled.set_param(name.as_str(), &data);
    }}
    Ok(compiled)
}}
"#,
        name = lo.name,
    )
}

pub fn tensor_lib_rs(lo: &Lowered) -> String {
    format!(
        r#"//! Native RLX model `{name}` — generated by `rlx-torch-import` (tensor style).
#![allow(nonstandard_style, unused_imports, clippy::all)]

pub mod graph;
pub mod weights;

pub use weights::{{load_weights, LoadedWeights}};

/// Compile the model on `device`, binding weights from `dir/weights.safetensors`.
pub fn compile(
    device: rlx_runtime::Device,
    dir: &std::path::Path,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {{
    let w = load_weights(dir)?;
    let g = graph::build_graph();
    let mut compiled = rlx_runtime::Session::new(device).compile(g);
    for key in graph::PARAM_KEYS {{
        compiled.set_param(key, &w.get(key)?);
    }}
    for (key, numel) in graph::ZERO_PARAMS {{
        compiled.set_param(key, &vec![0.0f32; *numel]);
    }}
    Ok(compiled)
}}
"#,
        name = lo.name,
    )
}
