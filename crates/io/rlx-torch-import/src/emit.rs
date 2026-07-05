// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Emit a standalone RLX crate from a [`Lowered`] program — the second walker
//! over the shared `Call` list (the first being [`crate::hir_build`]). The two
//! are kept structurally parallel so generated code and the in-process build
//! stay in lock-step.

use crate::call::*;
use anyhow::Result;
use rlx_ir::op::{Activation, BinaryOp, MaskKind, ReduceOp};
use std::collections::HashMap;
use std::path::Path;

fn sanitize(name: &str) -> String {
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

fn dtype_expr(dt: rlx_ir::DType) -> String {
    format!("rlx_ir::DType::{}", dtype_token(dt))
}

fn shape_expr(dims: &[usize], dt: rlx_ir::DType) -> String {
    let ds: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
    format!(
        "rlx_ir::Shape::new(&[{}], {})",
        ds.join(", "),
        dtype_expr(dt)
    )
}

fn act_variant(a: Activation) -> &'static str {
    match a {
        Activation::Gelu => "Gelu",
        Activation::GeluApprox => "GeluApprox",
        Activation::Silu => "Silu",
        Activation::Relu => "Relu",
        Activation::Sigmoid => "Sigmoid",
        Activation::Tanh => "Tanh",
        Activation::Exp => "Exp",
        Activation::Sqrt => "Sqrt",
        Activation::Rsqrt => "Rsqrt",
        Activation::Neg => "Neg",
        Activation::Abs => "Abs",
        _ => "Gelu",
    }
}

fn mask_variant(m: MaskKind) -> &'static str {
    match m {
        MaskKind::None => "None",
        MaskKind::Causal => "Causal",
        MaskKind::SlidingWindow(_) => "SlidingWindow(0)",
        MaskKind::Custom => "Custom",
        MaskKind::Bias => "Bias",
    }
}

pub(crate) fn emit_hir_ops(
    lo: &Lowered,
    idents: &mut HashMap<String, String>,
    r: &dyn Fn(&HashMap<String, String>, &str) -> String,
) -> String {
    let mut out = String::new();
    let line = |s: &str, buf: &mut String| {
        buf.push_str("    ");
        buf.push_str(s);
        buf.push('\n');
    };
    for ins in &lo.instrs {
        let res = sanitize(&ins.result);
        if let Some(note) = &ins.note {
            if !matches!(ins.call, Call::Alias(_)) {
                line(&format!("// {note}"), &mut out);
            }
        }
        match &ins.call {
            Call::Alias(src) => {
                idents.insert(ins.result.clone(), r(idents, src));
                continue;
            }
            Call::Mm(a, c) => line(
                &format!("let {res} = b.mm({}, {});", r(idents, a), r(idents, c)),
                &mut out,
            ),
            Call::Binary(op, a, c) => {
                let m = match op {
                    BinaryOp::Add => "add",
                    BinaryOp::Sub => "sub",
                    BinaryOp::Mul => "mul",
                    BinaryOp::Div => "div",
                    _ => "add",
                };
                line(
                    &format!("let {res} = b.{m}({}, {});", r(idents, a), r(idents, c)),
                    &mut out,
                );
            }
            Call::Act(act, a) => {
                let x = r(idents, a);
                line(&format!("let s_{res} = b.shape({x}).clone();"), &mut out);
                line(
                    &format!(
                        "let {res} = b.activation(Activation::{}, {x}, s_{res});",
                        act_variant(*act)
                    ),
                    &mut out,
                );
            }
            Call::Ln {
                x,
                gamma,
                beta,
                eps,
            } => line(
                &format!(
                    "let {res} = b.ln({}, {}, {}, {}f32);",
                    r(idents, x),
                    r(idents, gamma),
                    r(idents, beta),
                    eps
                ),
                &mut out,
            ),
            Call::RmsNorm {
                x,
                gamma,
                beta,
                eps,
            } => line(
                &format!(
                    "let {res} = b.rms_norm({}, {}, {}, {}f32);",
                    r(idents, x),
                    r(idents, gamma),
                    r(idents, beta),
                    eps
                ),
                &mut out,
            ),
            Call::Reshape { x, shape } => {
                let s: Vec<String> = shape.iter().map(|d| format!("{d}i64")).collect();
                line(
                    &format!(
                        "let {res} = b.reshape_({}, vec![{}]);",
                        r(idents, x),
                        s.join(", ")
                    ),
                    &mut out,
                );
            }
            Call::Transpose { x, perm } => {
                let p: Vec<String> = perm.iter().map(|d| format!("{d}usize")).collect();
                line(
                    &format!(
                        "let {res} = b.transpose_({}, vec![{}]);",
                        r(idents, x),
                        p.join(", ")
                    ),
                    &mut out,
                );
            }
            Call::Narrow {
                x,
                axis,
                start,
                len,
            } => line(
                &format!(
                    "let {res} = b.narrow_({}, {axis}, {start}, {len});",
                    r(idents, x)
                ),
                &mut out,
            ),
            Call::Concat { xs, axis } => {
                let ids: Vec<String> = xs.iter().map(|n| r(idents, n)).collect();
                line(
                    &format!("let {res} = b.concat_(vec![{}], {axis});", ids.join(", ")),
                    &mut out,
                );
            }
            Call::Gather {
                table,
                indices,
                axis,
            } => line(
                &format!(
                    "let {res} = b.gather_({}, {}, {axis});",
                    r(idents, table),
                    r(idents, indices)
                ),
                &mut out,
            ),
            Call::Softmax { x, axis } => line(
                &format!("let {res} = b.sm({}, {axis}i32);", r(idents, x)),
                &mut out,
            ),
            Call::Reduce {
                op,
                x,
                axes,
                keep_dim,
            } => {
                let m = match op {
                    ReduceOp::Mean => "mean",
                    _ => "sum",
                };
                let a: Vec<String> = axes.iter().map(|d| format!("{d}usize")).collect();
                line(
                    &format!(
                        "let {res} = b.{m}({}, vec![{}], {keep_dim});",
                        r(idents, x),
                        a.join(", ")
                    ),
                    &mut out,
                );
            }
            Call::Cast { x, to } => line(
                &format!("let {res} = b.cast({}, {});", r(idents, x), dtype_expr(*to)),
                &mut out,
            ),
            Call::Conv2d {
                x,
                weight,
                kernel,
                stride,
                padding,
                groups,
                out: oshape,
                out_dtype,
            } => line(
                &format!(
                    "let {res} = b.conv2d({}, {}, [{}, {}], [{}, {}], [{}, {}], {groups}, {});",
                    r(idents, x),
                    r(idents, weight),
                    kernel[0],
                    kernel[1],
                    stride[0],
                    stride[1],
                    padding[0],
                    padding[1],
                    shape_expr(oshape, *out_dtype)
                ),
                &mut out,
            ),
            Call::Attention {
                q,
                k,
                v,
                num_heads,
                head_dim,
                mask,
                out: oshape,
                out_dtype,
            } => line(
                &format!(
                    "let {res} = b.attention_kind({}, {}, {}, {num_heads}, {head_dim}, MaskKind::{}, {});",
                    r(idents, q),
                    r(idents, k),
                    r(idents, v),
                    mask_variant(*mask),
                    shape_expr(oshape, *out_dtype)
                ),
                &mut out,
            ),
            Call::GridSample {
                input,
                grid,
                mode,
                pad,
                align_corners,
                ..
            } => {
                use rlx_ir::hir::{GridMode, GridPad};
                let m = match mode {
                    GridMode::Nearest => "Nearest",
                    GridMode::Bilinear => "Bilinear",
                    GridMode::Bicubic => "Bicubic",
                };
                let p = match pad {
                    GridPad::Zeros => "Zeros",
                    GridPad::Border => "Border",
                    GridPad::Reflection => "Reflection",
                };
                line(
                    &format!(
                        "let {res} = b.grid_sample2d({}, {}, rlx_ir::hir::GridMode::{m}, rlx_ir::hir::GridPad::{p}, {align_corners});",
                        r(idents, input),
                        r(idents, grid)
                    ),
                    &mut out,
                )
            }
            Call::Resize {
                x,
                out_h,
                out_w,
                align_corners,
                cubic,
                antialias,
                ..
            } => line(
                &format!(
                    "let {res} = b.{}({}, {out_h}, {out_w}, {align_corners});",
                    match (*cubic, *antialias) {
                        (false, false) => "resize_bilinear2d",
                        (true, false) => "resize_bicubic2d",
                        (false, true) => "resize_bilinear2d_aa",
                        (true, true) => "resize_bicubic2d_aa",
                    },
                    r(idents, x)
                ),
                &mut out,
            ),
            Call::Rope {
                x,
                cos,
                sin,
                head_dim,
            } => line(
                &format!(
                    "let {res} = b.rope({}, {}, {}, {head_dim});",
                    r(idents, x),
                    r(idents, cos),
                    r(idents, sin)
                ),
                &mut out,
            ),
            Call::Full {
                value,
                shape,
                dtype,
            } => {
                let numel: usize = shape.iter().product();
                line(
                    &format!(
                        "let bytes_{res} = rlx_torch_fill({value}f64, {}, {numel});",
                        dtype_expr(*dtype)
                    ),
                    &mut out,
                );
                line(
                    &format!(
                        "let {res} = b.add_node(rlx_ir::Op::Constant {{ data: bytes_{res} }}, vec![], {});",
                        shape_expr(shape, *dtype)
                    ),
                    &mut out,
                );
            }
            Call::ConvTranspose2d {
                x,
                weight,
                kernel,
                stride,
                padding,
                dilation,
                output_padding,
                groups,
                out: oshape,
                out_dtype,
            } => line(
                &format!(
                    "let {res} = b.conv_transpose2d({}, {}, [{},{}], [{},{}], [{},{}], [{},{}], [{},{}], {groups}, {});",
                    r(idents, x),
                    r(idents, weight),
                    kernel[0],
                    kernel[1],
                    stride[0],
                    stride[1],
                    padding[0],
                    padding[1],
                    dilation[0],
                    dilation[1],
                    output_padding[0],
                    output_padding[1],
                    shape_expr(oshape, *out_dtype)
                ),
                &mut out,
            ),
            Call::Iota { rows, step, dtype } => {
                line(
                    &format!("let mut bytes_{res}: Vec<u8> = Vec::new();"),
                    &mut out,
                );
                line(
                    &format!(
                        "for r in 0..{rows}usize {{ let v = (r as i64) * {step}i64; bytes_{res}.extend_from_slice(&v.to_le_bytes()); }}"
                    ),
                    &mut out,
                );
                line(
                    &format!(
                        "let {res} = b.add_node(rlx_ir::Op::Constant {{ data: bytes_{res} }}, vec![], {});",
                        shape_expr(&[*rows, 1], *dtype)
                    ),
                    &mut out,
                );
            }
            Call::AttentionBias {
                q,
                k,
                v,
                bias,
                num_heads,
                head_dim,
                out: oshape,
                out_dtype,
            } => line(
                &format!(
                    "let {res} = b.attention_bias({}, {}, {}, {}, {num_heads}, {head_dim}, {});",
                    r(idents, q),
                    r(idents, k),
                    r(idents, v),
                    r(idents, bias),
                    shape_expr(oshape, *out_dtype)
                ),
                &mut out,
            ),
            Call::Arange {
                start,
                step,
                len,
                dtype,
            } => {
                let enc = match dtype {
                    rlx_ir::DType::I64 => "(v as i64).to_le_bytes()",
                    rlx_ir::DType::I32 => "(v as i32).to_le_bytes()",
                    _ => "(v as f32).to_le_bytes()",
                };
                line(
                    &format!("let mut bytes_{res}: Vec<u8> = Vec::new();"),
                    &mut out,
                );
                line(
                    &format!(
                        "for i in 0..{len}usize {{ let v = {start}f64 + (i as f64) * {step}f64; bytes_{res}.extend_from_slice(&{enc}); }}"
                    ),
                    &mut out,
                );
                line(
                    &format!(
                        "let {res} = b.add_node(rlx_ir::Op::Constant {{ data: bytes_{res} }}, vec![], {});",
                        shape_expr(&[*len], *dtype)
                    ),
                    &mut out,
                );
            }
            Call::Node(node) => line(&node.emit(&res, &|name| r(idents, name)), &mut out),
        }
        idents.insert(ins.result.clone(), res);
    }

    out
}

/// Generate `graph.rs` body: `build_graph(w) -> (HirModule, params)`.
fn emit_graph_rs(lo: &Lowered) -> String {
    let mut out = String::new();
    let mut idents: HashMap<String, String> = HashMap::new();
    let line = |s: &str, buf: &mut String| {
        buf.push_str("    ");
        buf.push_str(s);
        buf.push('\n');
    };

    out.push_str("// AUTO-GENERATED by rlx-torch-import — do not edit by hand.\n");
    out.push_str("//\n// Provenance: this RLX graph was imported from a PyTorch `torch.export`\n");
    out.push_str("// graph. Each op below is annotated with its original aten op + shapes so\n");
    out.push_str("// the source model can be traced / reconstructed.\n//\n");
    out.push_str("// Original aten op histogram:\n");
    for (op, count) in &lo.source_histogram {
        out.push_str(&format!("//   {count:>4}  {op}\n"));
    }
    out.push('\n');
    out.push_str("use std::collections::HashMap;\n");
    out.push_str("use rlx_ir::hir::{HirMut, HirNodeId};\n");
    out.push_str("use rlx_ir::op::{Activation, BinaryOp, MaskKind, ReduceOp};\n");
    out.push_str("use rlx_ir::{HirGraphExt, HirModule};\n\n");
    out.push_str("use crate::weights::LoadedWeights;\n\n");
    out.push_str(
        "fn rlx_torch_fill(value: f64, dtype: rlx_ir::DType, numel: usize) -> Vec<u8> {\n\
        \x20   let one: Vec<u8> = match dtype {\n\
        \x20       rlx_ir::DType::F32 => (value as f32).to_le_bytes().to_vec(),\n\
        \x20       rlx_ir::DType::F64 => value.to_le_bytes().to_vec(),\n\
        \x20       rlx_ir::DType::I64 => (value as i64).to_le_bytes().to_vec(),\n\
        \x20       rlx_ir::DType::I32 => (value as i32).to_le_bytes().to_vec(),\n\
        \x20       _ => vec![],\n\
        \x20   };\n\
        \x20   let mut out = Vec::with_capacity(one.len() * numel);\n\
        \x20   for _ in 0..numel { out.extend_from_slice(&one); }\n\
        \x20   out\n\
        }\n\n",
    );
    out.push_str(
        "/// Build the model graph and assemble its parameter map.\n\
         pub fn build_graph(\n    w: &LoadedWeights,\n) -> anyhow::Result<(HirModule, HashMap<String, Vec<f32>>)> {\n",
    );
    line(
        &format!("let mut hir = HirModule::new({:?});", lo.name),
        &mut out,
    );
    line("let mut b = HirMut::new(&mut hir);", &mut out);
    line(
        "let mut params: HashMap<String, Vec<f32>> = HashMap::new();",
        &mut out,
    );
    out.push('\n');

    for i in &lo.inputs {
        let id = sanitize(&i.name);
        line(
            &format!(
                "let {id} = b.input({:?}, {});",
                i.name,
                shape_expr(&i.shape, i.dtype)
            ),
            &mut out,
        );
        idents.insert(i.name.clone(), id);
    }
    out.push('\n');
    for p in &lo.params {
        let id = sanitize(&p.value_id);
        line(
            &format!(
                "let {id} = b.param({:?}, {});",
                p.key,
                shape_expr(&p.shape, p.dtype)
            ),
            &mut out,
        );
        line(
            &format!(
                "params.insert({:?}.to_string(), w.get({:?})?);",
                p.key, p.key
            ),
            &mut out,
        );
        idents.insert(p.value_id.clone(), id);
    }
    for z in &lo.zero_params {
        let id = sanitize(&z.value_id);
        let numel: usize = z.shape.iter().product();
        line(
            &format!(
                "let {id} = b.param({:?}, {});",
                z.key,
                shape_expr(&z.shape, z.dtype)
            ),
            &mut out,
        );
        line(
            &format!(
                "params.insert({:?}.to_string(), vec![0.0f32; {numel}]);",
                z.key
            ),
            &mut out,
        );
        idents.insert(z.value_id.clone(), id);
    }
    out.push('\n');

    let r = |idents: &HashMap<String, String>, name: &str| -> String {
        idents.get(name).cloned().unwrap_or_else(|| sanitize(name))
    };

    out.push_str(&emit_hir_ops(lo, &mut idents, &r));

    out.push('\n');
    let outs: Vec<String> = lo.outputs.iter().map(|n| r(&idents, n)).collect();
    line(
        &format!("b.set_outputs(vec![{}]);", outs.join(", ")),
        &mut out,
    );
    line("Ok((hir, params))", &mut out);
    out.push_str("}\n");
    out
}

fn emit_weights_rs() -> String {
    r#"// AUTO-GENERATED by rlx-torch-import.
use std::collections::HashMap;
use std::path::Path;

/// All model tensors, decoded to f32.
pub struct LoadedWeights {
    map: HashMap<String, Vec<f32>>,
}

impl LoadedWeights {
    pub fn get(&self, name: &str) -> anyhow::Result<Vec<f32>> {
        self.map
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("weight {name:?} missing"))
    }
}

/// Load `weights.safetensors` from `dir`, decoding every tensor to f32.
pub fn load_weights(dir: &Path) -> anyhow::Result<LoadedWeights> {
    let bytes = std::fs::read(dir.join("weights.safetensors"))?;
    let mut map = HashMap::new();
    if !bytes.is_empty() {
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        for name in st.names() {
            let v = st.tensor(name)?;
            let raw = v.data();
            let data: Vec<f32> = match v.dtype() {
                safetensors::Dtype::F32 => raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                safetensors::Dtype::F16 => raw
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::BF16 => raw
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                safetensors::Dtype::F64 => raw
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect(),
                other => anyhow::bail!("unsupported weight dtype {other:?}"),
            };
            map.insert(name.to_string(), data);
        }
    }
    Ok(LoadedWeights { map })
}
"#
    .to_string()
}

fn emit_lib_rs(lo: &Lowered) -> String {
    format!(
        r#"//! Native RLX model `{name}` — generated by `rlx-torch-import` from a
//! PyTorch `torch.export` graph. Do not edit `graph.rs` / `weights.rs` by hand.
#![allow(nonstandard_style, unused_imports, unused_parens, dead_code, clippy::all)]

pub mod graph;
pub mod weights;

pub use graph::build_graph;
pub use weights::{{load_weights, LoadedWeights}};

/// Compile the model on `device`, binding weights from `dir/weights.safetensors`.
pub fn compile(
    device: rlx_runtime::Device,
    dir: &std::path::Path,
) -> anyhow::Result<rlx_runtime::CompiledGraph> {{
    let w = load_weights(dir)?;
    let (hir, params) = build_graph(&w)?;
    let mut compiled = rlx_runtime::Session::new(device)
        .compile_hir_with(hir, &rlx_runtime::CompileOptions::default())
        .map_err(|e| anyhow::anyhow!("{{e}}"))?;
    for (name, data) in params {{
        compiled.set_param(name.as_str(), &data);
    }}
    Ok(compiled)
}}

/// Input names + shapes (row-major, f32 on the wire).
pub const INPUTS: &[(&str, &[usize])] = &[
{inputs}
];
"#,
        name = lo.name,
        inputs = lo
            .inputs
            .iter()
            .map(|i| {
                let dims: Vec<String> = i.shape.iter().map(|d| d.to_string()).collect();
                format!("    ({:?}, &[{}]),", i.name, dims.join(", "))
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn emit_cargo_toml(crate_name: &str, rlx_root: &Path, extra_deps: &str) -> String {
    let ir = rlx_root.join("crates/core/rlx-ir");
    let rt = rlx_root.join("crates/core/rlx-runtime");
    format!(
        r#"# Generated by rlx-torch-import.
[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2021"
publish = false
description = "RLX native model generated from a PyTorch model by rlx-torch-import"

[dependencies]
anyhow = "1"
half = "2"
safetensors = "0.7"
rlx-ir = {{ path = "{ir}", features = ["serialize"] }}
rlx-runtime = {{ path = "{rt}", default-features = false, features = ["cpu"] }}
{extra_deps}

[features]
default = ["cpu"]
cpu = ["rlx-runtime/cpu"]
metal = ["rlx-runtime/metal"]
cuda = ["rlx-runtime/cuda"]
"#,
        ir = ir.display(),
        rt = rt.display(),
    )
}

/// Write a complete, buildable RLX crate for `lo` into `out_dir`, in the chosen
/// [`EmitStyle`].
pub fn emit_crate(
    out_dir: &Path,
    lo: &Lowered,
    crate_name: &str,
    rlx_root: &Path,
    style: crate::emit_styles::EmitStyle,
) -> Result<()> {
    use crate::emit_styles::EmitStyle;
    let src = out_dir.join("src");
    std::fs::create_dir_all(&src)?;

    let (cargo, lib_rs, graph_rs, weights_rs) = match style {
        EmitStyle::Graph => (
            emit_cargo_toml(crate_name, rlx_root, ""),
            emit_lib_rs(lo),
            emit_graph_rs(lo),
            emit_weights_rs(),
        ),
        EmitStyle::Tensor => {
            let tensor = rlx_root.join("crates/core/rlx-tensor");
            let dep = format!("rlx-tensor = {{ path = {:?} }}", tensor.display());
            (
                emit_cargo_toml(crate_name, rlx_root, &dep),
                crate::emit_styles::tensor_lib_rs(lo),
                crate::emit_styles::emit_tensor_graph_rs(lo)?,
                emit_weights_rs(),
            )
        }
        EmitStyle::Flow => {
            let flow = rlx_root.join("crates/core/rlx-flow");
            let dep = format!("rlx-flow = {{ path = {:?} }}", flow.display());
            (
                emit_cargo_toml(crate_name, rlx_root, &dep),
                crate::emit_styles::flow_lib_rs(lo),
                crate::emit_styles::emit_flow_graph_rs(lo)?,
                crate::emit_styles::flow_weights_rs(),
            )
        }
    };

    std::fs::write(out_dir.join("Cargo.toml"), cargo)?;
    std::fs::write(src.join("lib.rs"), lib_rs)?;
    std::fs::write(src.join("graph.rs"), graph_rs)?;
    std::fs::write(src.join("weights.rs"), weights_rs)?;
    std::fs::write(
        out_dir.join("README.md"),
        format!(
            "# {}\n\nGenerated by `rlx-torch-import` from a PyTorch model \
             (emit style: {}).\n\nCopy `weights.safetensors` next to this crate \
             (or pass its dir to `compile`).\n",
            lo.name,
            style.as_str()
        ),
    )?;
    Ok(())
}
