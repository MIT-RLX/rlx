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

//! Rust source emission for [`crate::lower`] ops (used by `rlx-onnx-decompose`).

use crate::BundleNode;

fn rust_str_lit(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| format!("\"{s}\""))
}

fn meta_lit(meta: &serde_json::Value) -> String {
    format!("&serde_json::json!({})", meta)
}

fn unary_activation(out: &str, input: &str, act: &str) -> Vec<String> {
    vec![format!(
        "let {out} = {{ let x = b.tensor({input})?; let mut m = HirMut::new(&mut b.hir); let s = m.shape(x).clone(); m.activation({act}, x, s) }};",
        input = rust_str_lit(input),
    )]
}

/// Bind ONNX value names to `HirNodeId`s before opening `HirMut`.
fn prefetch_inputs(out_ident: &str, inputs: &[&str]) -> (Vec<String>, String) {
    let mut lines = Vec::new();
    let mut ids = Vec::new();
    for (i, name) in inputs.iter().enumerate() {
        let id = format!("__t_{out_ident}_{i}");
        lines.push(format!("let {id} = b.tensor({})?;", rust_str_lit(name)));
        ids.push(id);
    }
    (lines, ids.join(", "))
}

fn binary_op(
    out: &str,
    a: &str,
    b: &str,
    meta: &str,
    op: &str,
    infer_broadcast: bool,
    site: &str,
) -> Vec<String> {
    let a_id = format!("__in0_{out}");
    let b_id = format!("__in1_{out}");
    let body = if infer_broadcast {
        format!(
            "let mut m = HirMut::new(&mut b.hir); binary_infer(&mut m, {op}, {a_id}, {b_id}, {})",
            rust_str_lit(site)
        )
    } else {
        format!(
            "let mut m = HirMut::new(&mut b.hir); \
            m.add_node(rlx_ir::Op::Binary({op}), vec![{a_id}, {b_id}], shape_from_meta({meta}, opts))"
        )
    };
    vec![
        format!("let {a_id} = b.tensor({})?;", rust_str_lit(a)),
        format!("let {b_id} = b.tensor({})?;", rust_str_lit(b)),
        format!("let {out} = {{ {body} }};"),
    ]
}

fn attr_usize(node: &BundleNode, key: &str, idx: usize, default: usize) -> usize {
    node.attrs
        .get(key)
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(idx))
        .and_then(|d| d.as_u64())
        .map(|x| x as usize)
        .unwrap_or(default)
}

fn attr_i64(node: &BundleNode, key: &str, default: i64) -> i64 {
    node.attrs
        .get(key)
        .and_then(|v| v.as_i64())
        .unwrap_or(default)
}

fn attr_f64(node: &BundleNode, key: &str, default: f64) -> f64 {
    node.attrs
        .get(key)
        .and_then(|v| v.as_f64())
        .unwrap_or(default)
}

fn f32_lit(v: f64) -> String {
    if v.is_infinite() {
        if v.is_sign_negative() {
            "f32::NEG_INFINITY".to_string()
        } else {
            "f32::INFINITY".to_string()
        }
    } else if v.is_nan() {
        "f32::NAN".to_string()
    } else {
        format!("{v}f32")
    }
}

fn axes_lit(node: &BundleNode) -> String {
    node.attrs
        .get("axes")
        .and_then(|v| v.as_array())
        .map(|a| {
            format!(
                "vec![{}]",
                a.iter()
                    .filter_map(|d| d.as_i64())
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .unwrap_or_else(|| "vec![0]".to_string())
}

fn stub_node(out: &str, meta: &str, out_name: &str) -> Vec<String> {
    let stub = rust_str_lit(&format!("__stub__/{out_name}"));
    vec![format!(
        "let {out} = {{ let sh = shape_from_meta({meta}, opts); let n = sh.num_elements().unwrap_or(1).min(8 * 1024 * 1024); b.params.insert({stub}.to_string(), vec![0.0; n]); let mut m0 = HirMut::new(&mut b.hir); m0.param({stub}, sh) }};",
    )]
}

fn emit_resize(node: &BundleNode, out_ident: &str, meta0: &str) -> Vec<String> {
    let x = rust_str_lit(&node.inputs[0]);
    vec![format!(
        "let {out_ident} = {{ let x = b.tensor({x})?; let mut m = HirMut::new(&mut b.hir); \
        let in_s = m.shape(x).clone(); let out_s = shape_from_meta({meta0}, opts); \
        let mode = {:?}; \
        if mode == \"nearest\" && in_s.rank() == 4 && out_s.rank() == 4 {{ \
            let h_in = in_s.dim(2).unwrap_static(); let w_in = in_s.dim(3).unwrap_static(); \
            let h_out = out_s.dim(2).unwrap_static(); let w_out = out_s.dim(3).unwrap_static(); \
            if h_out == h_in * 2 && w_out == w_in * 2 {{ m.resize_nearest_2x(x) }} \
            else if in_s.num_elements() == out_s.num_elements() {{ x }} \
            else {{ let stub = format!(\"__resize_stub__/{{}}\", {:?}); b.params.insert(stub.clone(), vec![0.0; out_s.num_elements().unwrap_or(1)]); m.param(&stub, out_s) }} \
        }} else if in_s.num_elements() == out_s.num_elements() {{ x }} \
        else {{ let stub = format!(\"__resize_stub__/{{}}\", {:?}); b.params.insert(stub.clone(), vec![0.0; out_s.num_elements().unwrap_or(1)]); m.param(&stub, out_s) }} \
        }};",
        node.attrs
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("nearest"),
        node.name,
        node.name,
    )]
}

/// Emit one output of [`DynamicQuantizeLinear`] (quantized tensor, scale, zero-point).
pub fn emit_dynamic_quant_output(
    node: &BundleNode,
    output_index: usize,
    out_ident: &str,
) -> Vec<String> {
    let meta = node
        .output_meta
        .get(output_index)
        .map(meta_lit)
        .unwrap_or_else(|| "&serde_json::json!({\"shape\": [1], \"dtype\": \"f32\"})".to_string());
    match output_index {
        0 if !node.inputs.is_empty() => vec![format!(
            "let {out_ident} = b.tensor({})?;",
            rust_str_lit(&node.inputs[0])
        )],
        1 => vec![format!(
            "let {out_ident} = {{ let k = format!(\"__dql_scale__/{{}}\", {}); b.params.insert(k.clone(), vec![1.0f32]); let mut m0 = HirMut::new(&mut b.hir); m0.param(&k, Shape::new(&[1], DType::F32)) }};",
            rust_str_lit(&node.name)
        )],
        2 => vec![format!(
            "let {out_ident} = {{ let k = format!(\"__dql_zp__/{{}}\", {}); b.params.insert(k.clone(), vec![0.0f32]); let mut m0 = HirMut::new(&mut b.hir); m0.param(&k, Shape::new(&[1], DType::F32)) }};",
            rust_str_lit(&node.name)
        )],
        _ => stub_node(
            out_ident,
            &meta,
            node.outputs
                .get(output_index)
                .map(String::as_str)
                .unwrap_or("out"),
        ),
    }
}

fn node_name_tag_lit(node: &BundleNode) -> String {
    crate::random::node_name_tag(&node.name).to_string()
}

fn emit_random_like(node: &BundleNode, out_ident: &str, meta0: &str) -> Vec<String> {
    let tag = node_name_tag_lit(node);
    let seed = node
        .attrs
        .get("seed")
        .and_then(|v| v.as_f64())
        .map(|v| format!("Some({v}f32)"))
        .unwrap_or_else(|| "None".to_string());
    let shape = rust_str_lit(&node.inputs[0]);
    match node.op.as_str() {
        "RandomNormalLike" => {
            let mean = attr_f64(node, "mean", 0.0);
            let scale = attr_f64(node, "scale", 1.0);
            vec![format!(
                "let {out_ident} = {{ let shape_in = b.tensor({shape})?; let mut m = HirMut::new(&mut b.hir); \
                m.add_node(rlx_ir::Op::RngNormal {{ mean: {mean}f32, scale: {scale}f32, key: {tag}, op_seed: {seed} }}, \
                vec![shape_in], shape_from_meta({meta0}, opts)) }};",
            )]
        }
        _ => {
            let low = attr_f64(node, "low", 0.0);
            let high = attr_f64(node, "high", 1.0);
            vec![format!(
                "let {out_ident} = {{ let shape_in = b.tensor({shape})?; let mut m = HirMut::new(&mut b.hir); \
                m.add_node(rlx_ir::Op::RngUniform {{ low: {low}f32, high: {high}f32, key: {tag}, op_seed: {seed} }}, \
                vec![shape_in], shape_from_meta({meta0}, opts)) }};",
            )]
        }
    }
}

fn emit_random(node: &BundleNode, out_ident: &str, meta0: &str) -> Vec<String> {
    let tag = node_name_tag_lit(node);
    let seed = node
        .attrs
        .get("seed")
        .and_then(|v| v.as_f64())
        .map(|v| format!("Some({v}f32)"))
        .unwrap_or_else(|| "None".to_string());
    let shape_in = if node.inputs.is_empty() {
        String::new()
    } else {
        format!(
            "let shape_in = b.tensor({})?;",
            rust_str_lit(&node.inputs[0])
        )
    };
    let in_list = if node.inputs.is_empty() {
        "vec![]".to_string()
    } else {
        "vec![shape_in]".to_string()
    };
    match node.op.as_str() {
        "RandomNormal" => {
            let mean = attr_f64(node, "mean", 0.0);
            let scale = attr_f64(node, "scale", 1.0);
            vec![format!(
                "let {out_ident} = {{ {shape_in} let mut m = HirMut::new(&mut b.hir); \
                m.add_node(rlx_ir::Op::RngNormal {{ mean: {mean}f32, scale: {scale}f32, key: {tag}, op_seed: {seed} }}, \
                {in_list}, shape_from_meta({meta0}, opts)) }};",
            )]
        }
        _ => {
            let low = attr_f64(node, "low", 0.0);
            let high = attr_f64(node, "high", 1.0);
            vec![format!(
                "let {out_ident} = {{ {shape_in} let mut m = HirMut::new(&mut b.hir); \
                m.add_node(rlx_ir::Op::RngUniform {{ low: {low}f32, high: {high}f32, key: {tag}, op_seed: {seed} }}, \
                {in_list}, shape_from_meta({meta0}, opts)) }};",
            )]
        }
    }
}

/// Emit TopK values + indices (`outputs[0]` = values, `outputs[1]` = indices).
pub fn emit_topk(node: &BundleNode, val_ident: &str, idx_ident: &str) -> Vec<String> {
    let x_name = &node.inputs[0];
    let axis = attr_i64(node, "axis", -1);
    let meta_idx = node.output_meta.get(1).map(meta_lit).unwrap_or_else(|| {
        node.output_meta.first().map(meta_lit).unwrap_or_else(|| {
            "&serde_json::json!({\"shape\": [1], \"dtype\": \"i64\"})".to_string()
        })
    });
    let has_k_input = node.inputs.len() >= 2;
    let k_name_lit = node
        .inputs
        .get(1)
        .map(|s| rust_str_lit(s))
        .unwrap_or_else(|| "\"\"".to_string());
    let mut lines = vec![format!(
        "let __t_topk_x = b.tensor({})?;",
        rust_str_lit(x_name)
    )];
    if has_k_input {
        lines.push(format!("let __t_topk_k = b.tensor({k_name_lit})?;"));
    }
    lines.push(format!(
        "let ({idx_ident}, {val_ident}) = {{ \
        let x = __t_topk_x; \
        let mut m = HirMut::new(&mut b.hir); \
        let in_s = m.shape(x); \
        let rank = in_s.rank().max(1); \
        let axis = (({axis} + rank as i64).rem_euclid(rank as i64)) as usize; \
        let idx_shape = if {has_k_input} {{ \
            m.shape(__t_topk_k).clone() \
        }} else {{ \
            shape_from_meta({meta_idx}, opts) \
        }}; \
        let k = if {has_k_input} {{ \
            idx_shape.num_elements().unwrap_or(1).max(1) \
        }} else if idx_shape.rank() == 0 {{ 1 }} else {{ \
            idx_shape.dim(axis.min(idx_shape.rank().saturating_sub(1))).unwrap_static().max(1) \
        }}; \
        let indices = m.add_node(rlx_ir::Op::TopK {{ k }}, vec![x], idx_shape); \
        let values = m.gather_(x, indices, axis); \
        (indices, values) \
        }};",
    ));
    lines
}

/// Emit Rust statements for one ONNX node (excluding weight preload and `b.bind`).
pub fn emit_node_body(node: &BundleNode, out_ident: &str) -> Vec<String> {
    let out = node.outputs.first().map(String::as_str).unwrap_or("out");
    let out_name = out;
    let meta0 =
        node.output_meta.first().map(meta_lit).unwrap_or_else(|| {
            "&serde_json::json!({\"shape\": [1], \"dtype\": \"f32\"})".to_string()
        });

    match node.op.as_str() {
        "Add" | "Mul" | "Sub" | "Div" if node.inputs.len() >= 2 => {
            let op = match node.op.as_str() {
                "Mul" => "BinaryOp::Mul",
                "Sub" => "BinaryOp::Sub",
                "Div" => "BinaryOp::Div",
                _ => "BinaryOp::Add",
            };
            let infer = matches!(node.op.as_str(), "Add" | "Sub" | "Mul" | "Div");
            binary_op(
                out_ident,
                &node.inputs[0],
                &node.inputs[1],
                &meta0,
                op,
                infer,
                &node.name,
            )
        }
        "MatMul" if node.inputs.len() >= 2 => {
            let a_id = format!("__in0_{out_ident}");
            let b_id = format!("__in1_{out_ident}");
            vec![
                format!("let {a_id} = b.tensor({})?;", rust_str_lit(&node.inputs[0])),
                format!("let {b_id} = b.tensor({})?;", rust_str_lit(&node.inputs[1])),
                format!(
                    "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); \
                    match shape::matmul_shape(m.shape({a_id}), m.shape({b_id})) {{ \
                    Ok(s) => m.add_node(rlx_ir::Op::MatMul, vec![{a_id}, {b_id}], s), \
                    Err(_) => m.add_node(rlx_ir::Op::MatMul, vec![{a_id}, {b_id}], shape_from_meta({meta0}, opts)), \
                    }} }};",
                ),
            ]
        }
        "Gemm" if node.inputs.len() >= 2 => {
            let a_id = format!("__in0_{out_ident}");
            let b_id = format!("__in1_{out_ident}");
            let mut lines = vec![
                format!("let {a_id} = b.tensor({})?;", rust_str_lit(&node.inputs[0])),
                format!("let {b_id} = b.tensor({})?;", rust_str_lit(&node.inputs[1])),
                format!(
                    "let mut {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); \
                    match shape::matmul_shape(m.shape({a_id}), m.shape({b_id})) {{ \
                    Ok(s) => m.add_node(rlx_ir::Op::MatMul, vec![{a_id}, {b_id}], s), \
                    Err(_) => m.add_node(rlx_ir::Op::MatMul, vec![{a_id}, {b_id}], shape_from_meta({meta0}, opts)), \
                    }} }};",
                ),
            ];
            if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                let c = format!("__in2_{out_ident}");
                lines.push(format!(
                    "let {c} = b.tensor({})?;",
                    rust_str_lit(&node.inputs[2])
                ));
                lines.push(format!(
                    "{{ let mut m = HirMut::new(&mut b.hir); \
                    {out_ident} = binary_infer(&mut m, BinaryOp::Add, {out_ident}, {c}, {}) }};",
                    rust_str_lit(&node.name)
                ));
            }
            lines
        }
        "Relu" => unary_activation(out_ident, &node.inputs[0], "Activation::Relu"),
        "LeakyRelu" => {
            let alpha = node
                .attrs
                .get("alpha")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.01);
            vec![format!(
                "let {out_ident} = {{ let x = b.tensor({})?; let alpha = {{ let k = format!(\"__leaky/{{}}\", {}); b.params.insert(k.clone(), vec![{alpha}f32]); let mut m2 = HirMut::new(&mut b.hir); m2.param(&k, Shape::new(&[1], DType::F32)) }}; let mut m = HirMut::new(&mut b.hir); let s = m.shape(x).clone(); let pos = m.add_node(rlx_ir::Op::Activation(Activation::Relu), vec![x], s.clone()); let neg = m.add_node(rlx_ir::Op::Activation(Activation::Neg), vec![x], s.clone()); let nneg = m.add_node(rlx_ir::Op::Activation(Activation::Relu), vec![neg], s.clone()); let sc = m.add_node(rlx_ir::Op::Binary(BinaryOp::Mul), vec![nneg, alpha], s.clone()); m.add_node(rlx_ir::Op::Binary(BinaryOp::Add), vec![pos, sc], s) }};",
                rust_str_lit(&node.inputs[0]),
                rust_str_lit(&node.name),
            )]
        }
        "Tanh" => unary_activation(out_ident, &node.inputs[0], "Activation::Tanh"),
        "Sigmoid" => unary_activation(out_ident, &node.inputs[0], "Activation::Sigmoid"),
        "Sqrt" => unary_activation(out_ident, &node.inputs[0], "Activation::Sqrt"),
        "Sin" => unary_activation(out_ident, &node.inputs[0], "Activation::Sin"),
        "Cos" => unary_activation(out_ident, &node.inputs[0], "Activation::Cos"),
        "Exp" => unary_activation(out_ident, &node.inputs[0], "Activation::Exp"),
        "Neg" => unary_activation(out_ident, &node.inputs[0], "Activation::Neg"),
        "Abs" => unary_activation(out_ident, &node.inputs[0], "Activation::Abs"),
        "Atan" => unary_activation(out_ident, &node.inputs[0], "Activation::Atan"),
        "Floor" | "Round" => unary_activation(out_ident, &node.inputs[0], "Activation::Round"),
        "Cast" => {
            let to = node.attrs.get("to").and_then(|v| v.as_i64()).unwrap_or(1);
            let dtype = match to {
                1 => "DType::F32",
                7 => "DType::I64",
                6 => "DType::I32",
                9 => "DType::Bool",
                _ => "DType::F32",
            };
            vec![format!(
                "let {out_ident} = {{ let x = b.tensor({})?; let mut m = HirMut::new(&mut b.hir); m.cast(x, {dtype}) }};",
                rust_str_lit(&node.inputs[0]),
            )]
        }
        "Transpose" => {
            let perm = node
                .attrs
                .get("perm")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|d| d.as_u64().map(|x| x as usize))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let perm_lit = format!(
                "vec![{}]",
                perm.iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            vec![format!(
                "let {out_ident} = {{ let x = b.tensor({})?; let mut m = HirMut::new(&mut b.hir); \
                if m.shape(x).rank() == {perm_lit}.len() {{ m.transpose_(x, {perm_lit}) }} else {{ \
                m.add_node(rlx_ir::Op::Transpose {{ perm: {perm_lit} }}, vec![x], shape_from_meta({meta0}, opts)) \
                }} }};",
                rust_str_lit(&node.inputs[0]),
            )]
        }
        "Reshape" | "Unsqueeze" | "Squeeze" | "Flatten" => vec![format!(
            "let {out_ident} = {{ let x = b.tensor({})?; let sh = shape_from_meta({meta0}, opts); let dims: Vec<i64> = sh.dims().iter().map(|d| d.unwrap_static() as i64).collect(); let mut m = HirMut::new(&mut b.hir); m.reshape_(x, dims) }};",
            rust_str_lit(&node.inputs[0]),
        )],
        "Gather" if node.inputs.len() >= 2 => {
            let axis = node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
            let (pre, ids) = prefetch_inputs(out_ident, &[&node.inputs[0], &node.inputs[1]]);
            let mut lines = pre;
            lines.push(format!(
                "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); m.add_node(rlx_ir::Op::Gather {{ axis: {axis} }}, vec![{ids}], shape_from_meta({meta0}, opts)) }};",
            ));
            lines
        }
        "Concat" => {
            let axis = node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
            let names: Vec<&str> = node.inputs.iter().map(String::as_str).collect();
            let (pre, ids) = prefetch_inputs(out_ident, &names);
            let mut lines = pre;
            lines.push(format!(
                "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); m.add_node(rlx_ir::Op::Concat {{ axis: {axis} }}, vec![{ids}], shape_from_meta({meta0}, opts)) }};",
            ));
            lines
        }
        "Softmax" => {
            let axis = node
                .attrs
                .get("axis")
                .and_then(|v| v.as_i64())
                .unwrap_or(-1) as i32;
            vec![format!(
                "let {out_ident} = {{ let x = b.tensor({})?; let mut m = HirMut::new(&mut b.hir); m.sm(x, {axis}) }};",
                rust_str_lit(&node.inputs[0]),
            )]
        }
        "LayerNormalization" if node.inputs.len() >= 3 => {
            let axis = node
                .attrs
                .get("axis")
                .and_then(|v| v.as_i64())
                .unwrap_or(-1);
            let eps = node
                .attrs
                .get("epsilon")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-5);
            let x_id = format!("__ln_x_{out_ident}");
            let g_id = format!("__ln_g_{out_ident}");
            let b_id = format!("__ln_b_{out_ident}");
            vec![
                format!("let {x_id} = b.tensor({})?;", rust_str_lit(&node.inputs[0])),
                format!("let {g_id} = b.tensor({})?;", rust_str_lit(&node.inputs[1])),
                format!("let {b_id} = b.tensor({})?;", rust_str_lit(&node.inputs[2])),
                format!(
                    "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); \
                    let g_bc = broadcast_param_channels(&mut m, {g_id}, {x_id}); \
                    let b_bc = broadcast_param_channels(&mut m, {b_id}, {x_id}); \
                    let out_sh = m.shape({x_id}).clone(); \
                    m.add_node(rlx_ir::Op::LayerNorm {{ axis: {axis}, eps: {eps} }}, vec![{x_id}, g_bc, b_bc], out_sh) \
                    }};",
                ),
            ]
        }
        "Where" if node.inputs.len() >= 3 => {
            let (pre, ids) = prefetch_inputs(
                out_ident,
                &[&node.inputs[0], &node.inputs[1], &node.inputs[2]],
            );
            let mut lines = pre;
            lines.push(format!(
                "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); m.add_node(rlx_ir::Op::Where, vec![{ids}], shape_from_meta({meta0}, opts)) }};",
            ));
            lines
        }
        "Pow" if node.inputs.len() >= 2 => binary_op(
            out_ident,
            &node.inputs[0],
            &node.inputs[1],
            &meta0,
            "BinaryOp::Pow",
            false,
            &node.name,
        ),
        "Clip" if !node.inputs.is_empty() => {
            let min_lit = f32_lit(attr_f64(node, "min", f64::NEG_INFINITY));
            let max_lit = f32_lit(attr_f64(node, "max", f64::INFINITY));
            vec![format!(
                "let {out_ident} = {{ let x = b.tensor({})?; let min_id = {{ let min_k = format!(\"__clip_min__/{{}}\", {}); b.params.insert(min_k.clone(), vec![{min_lit}]); let mut m0 = HirMut::new(&mut b.hir); m0.param(&min_k, Shape::new(&[1], DType::F32)) }}; let max_id = {{ let max_k = format!(\"__clip_max__/{{}}\", {}); b.params.insert(max_k.clone(), vec![{max_lit}]); let mut m0 = HirMut::new(&mut b.hir); m0.param(&max_k, Shape::new(&[1], DType::F32)) }}; let mut m = HirMut::new(&mut b.hir); let s = m.shape(x).clone(); let hi = m.add_node(rlx_ir::Op::Binary(BinaryOp::Min), vec![x, max_id], s.clone()); m.add_node(rlx_ir::Op::Binary(BinaryOp::Max), vec![hi, min_id], s) }};",
                rust_str_lit(&node.inputs[0]),
                rust_str_lit(&node.name),
                rust_str_lit(&node.name),
            )]
        }
        "Expand" if !node.inputs.is_empty() => vec![format!(
            "let {out_ident} = {{ let x = b.tensor({})?; let sh = shape_from_meta({meta0}, opts); let target: Vec<i64> = sh.dims().iter().map(|d| d.unwrap_static() as i64).collect(); let mut m = HirMut::new(&mut b.hir); m.add_node(rlx_ir::Op::Expand {{ target_shape: target }}, vec![x], sh) }};",
            rust_str_lit(&node.inputs[0]),
        )],
        "Equal" if node.inputs.len() >= 2 => {
            let (pre, ids) = prefetch_inputs(out_ident, &[&node.inputs[0], &node.inputs[1]]);
            let mut lines = pre;
            lines.push(format!(
                "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); m.add_node(rlx_ir::Op::Compare(rlx_ir::op::CmpOp::Eq), vec![{ids}], shape_from_meta({meta0}, opts)) }};",
            ));
            lines
        }
        "Less" if node.inputs.len() >= 2 => {
            let (pre, ids) = prefetch_inputs(out_ident, &[&node.inputs[0], &node.inputs[1]]);
            let mut lines = pre;
            lines.push(format!(
                "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); m.add_node(rlx_ir::Op::Compare(rlx_ir::op::CmpOp::Lt), vec![{ids}], shape_from_meta({meta0}, opts)) }};",
            ));
            lines
        }
        "Greater" if node.inputs.len() >= 2 => {
            let (pre, ids) = prefetch_inputs(out_ident, &[&node.inputs[0], &node.inputs[1]]);
            let mut lines = pre;
            lines.push(format!(
                "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); m.add_node(rlx_ir::Op::Compare(rlx_ir::op::CmpOp::Gt), vec![{ids}], shape_from_meta({meta0}, opts)) }};",
            ));
            lines
        }
        "Not" if !node.inputs.is_empty() => vec![format!(
            "let {out_ident} = {{ let x = b.tensor({})?; let z = {{ let z_k = format!(\"__not_z__/{{}}\", {}); b.params.insert(z_k.clone(), vec![0.0f32]); let mut m0 = HirMut::new(&mut b.hir); m0.param(&z_k, Shape::new(&[1], DType::F32)) }}; let mut m = HirMut::new(&mut b.hir); m.eq(x, z) }};",
            rust_str_lit(&node.inputs[0]),
            rust_str_lit(&node.name),
        )],
        "And" if node.inputs.len() >= 2 => vec![format!(
            "let {out_ident} = {{ let a = b.tensor({})?; let b_in = b.tensor({})?; let z = {{ let z_k = format!(\"__and_z__/{{}}\", {}); b.params.insert(z_k.clone(), vec![0.0f32]); let mut m0 = HirMut::new(&mut b.hir); m0.param(&z_k, Shape::new(&[1], DType::F32)) }}; let mut m = HirMut::new(&mut b.hir); let sh = match shape::binary_shape(m.shape(a), m.shape(b_in)) {{ Ok(s) => s, Err(_) => m.shape(a).clone() }}; let prod = m.add_node(rlx_ir::Op::Binary(BinaryOp::Mul), vec![a, b_in], sh); let s = shape_from_meta({meta0}, opts).with_dtype(DType::Bool); m.add_node(rlx_ir::Op::Compare(rlx_ir::op::CmpOp::Ne), vec![prod, z], s) }};",
            rust_str_lit(&node.inputs[0]),
            rust_str_lit(&node.inputs[1]),
            rust_str_lit(&node.name),
        )],
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceProd"
            if !node.inputs.is_empty() =>
        {
            let keep = attr_i64(node, "keepdims", 1) != 0;
            let axes = axes_lit(node);
            let x = rust_str_lit(&node.inputs[0]);
            match node.op.as_str() {
                "ReduceMean" => vec![format!(
                    "let {out_ident} = {{ let x = b.tensor({x})?; let mut m = HirMut::new(&mut b.hir); m.mean(x, {axes}, {keep}) }};",
                )],
                "ReduceSum" => vec![format!(
                    "let {out_ident} = {{ let x = b.tensor({x})?; let mut m = HirMut::new(&mut b.hir); m.sum(x, {axes}, {keep}) }};",
                )],
                op => {
                    let rop = match op {
                        "ReduceMax" => "ReduceOp::Max",
                        "ReduceMin" => "ReduceOp::Min",
                        "ReduceProd" => "ReduceOp::Prod",
                        _ => "ReduceOp::Sum",
                    };
                    vec![format!(
                        "let {out_ident} = {{ let x = b.tensor({x})?; let mut m = HirMut::new(&mut b.hir); m.add_node(rlx_ir::Op::Reduce {{ op: {rop}, axes: {axes}, keep_dim: {keep} }}, vec![x], shape_from_meta({meta0}, opts)) }};",
                    )]
                }
            }
        }
        "InstanceNormalization" if node.inputs.len() >= 3 => {
            let eps = attr_f64(node, "epsilon", 1e-5);
            vec![format!(
                "let {out_ident} = {{ let x = b.tensor({})?; let s = b.tensor({})?; let bias = b.tensor({})?; let eps_id = {{ let eps_k = format!(\"__inst_norm_eps__/{{}}\", {}); b.params.insert(eps_k.clone(), vec![{eps}f32]); let mut m0 = HirMut::new(&mut b.hir); m0.param(&eps_k, Shape::new(&[1], DType::F32)) }}; let one = {{ let one_k = format!(\"__inst_norm_one__/{{}}\", {}); b.params.insert(one_k.clone(), vec![1.0f32]); let mut m0 = HirMut::new(&mut b.hir); m0.param(&one_k, Shape::new(&[1], DType::F32)) }}; let mut m = HirMut::new(&mut b.hir); let ch = channel_axis_for_param(&mut m, s, x); let s_bc = broadcast_param_channels(&mut m, s, x); let bias_bc = broadcast_param_channels(&mut m, bias, x); let axes = inst_norm_reduce_axes(m.shape(x).rank(), ch); let mean = m.mean(x, axes.clone(), true); let xc = m.add_node(rlx_ir::Op::Binary(BinaryOp::Sub), vec![x, mean], m.shape(x).clone()); let sq = m.add_node(rlx_ir::Op::Binary(BinaryOp::Mul), vec![xc.clone(), xc.clone()], m.shape(xc).clone()); let var = m.mean(sq, axes, true); let v_eps = m.add_node(rlx_ir::Op::Binary(BinaryOp::Add), vec![var, eps_id], m.shape(var).clone()); let inv = m.activation(Activation::Sqrt, v_eps, m.shape(v_eps).clone()); let inv_std = m.add_node(rlx_ir::Op::Binary(BinaryOp::Div), vec![one, inv], m.shape(inv).clone()); let norm = m.add_node(rlx_ir::Op::Binary(BinaryOp::Mul), vec![xc, inv_std], m.shape(xc).clone()); let scaled = m.add_node(rlx_ir::Op::Binary(BinaryOp::Mul), vec![norm, s_bc], m.shape(norm).clone()); m.add_node(rlx_ir::Op::Binary(BinaryOp::Add), vec![scaled, bias_bc], m.shape(scaled).clone()) }};",
                rust_str_lit(&node.inputs[0]),
                rust_str_lit(&node.inputs[1]),
                rust_str_lit(&node.inputs[2]),
                rust_str_lit(&node.name),
                rust_str_lit(&node.name),
            )]
        }
        "Conv" | "ConvTranspose" if node.inputs.len() >= 2 => {
            let k0 = attr_usize(node, "kernel_shape", 0, 1);
            let k1 = attr_usize(node, "kernel_shape", 1, 1);
            let s0 = attr_usize(node, "strides", 0, 1);
            let s1 = attr_usize(node, "strides", 1, 1);
            let p0 = attr_usize(node, "pads", 0, 0);
            let p1 = attr_usize(node, "pads", 1, 0);
            let groups = attr_i64(node, "group", 1);
            let transpose = node.op == "ConvTranspose";
            let x = rust_str_lit(&node.inputs[0]);
            let w = rust_str_lit(&node.inputs[1]);
            let mut lines = vec![format!(
                "let mut {out_ident} = {{ let x = b.tensor({x})?; let w = b.tensor({w})?; let mut m = HirMut::new(&mut b.hir); \
                let w_s = m.shape(w); let in_s = m.shape(x); let dt = in_s.dtype(); let rank = in_s.rank(); \
                let wc = w_s.dim(0).unwrap_static(); let wi = w_s.dim(1).unwrap_static(); \
                let wk = if w_s.rank() > 2 {{ w_s.dim(2).unwrap_static() }} else {{ 1 }}; \
                let n = if rank > 0 {{ in_s.dim(0).unwrap_static() }} else {{ 1 }}; \
                let c = if rank > 1 {{ in_s.dim(1).unwrap_static() }} else {{ 1 }}; \
                let l = if rank > 2 {{ in_s.dim(2).unwrap_static() }} else {{ 1 }}; \
                let meta_sh = shape_from_meta({meta0}, opts); \
                let out_sh = if meta_sh.rank() >= 2 {{ meta_sh }} else if {transpose} {{ \
                    let c_out = wi * ({groups} as usize); \
                    if rank == 4 {{ \
                        let h = in_s.dim(2).unwrap_static(); let wd = in_s.dim(3).unwrap_static(); \
                        let h_out = rlx_ir::shape::conv_transpose2d_spatial_output(h, {k0}, {s0}, {p0}, 1, 0); \
                        let w_out = rlx_ir::shape::conv_transpose2d_spatial_output(wd, {k1}, {s1}, {p1}, 1, 0); \
                        Shape::new(&[n, c_out, h_out, w_out], dt) \
                    }} else if rank == 3 {{ \
                        let l_out = rlx_ir::shape::conv_transpose2d_spatial_output(l, {k0}, {s0}, {p0}, 1, 0); \
                        Shape::new(&[n, c_out, l_out], dt) \
                    }} else {{ Shape::new(&[1], dt) }} \
                }} else {{ \
                    shape::conv2d_output_shape(in_s, w_s, [{k0}, {k1}], [{s0}, {s1}], [{p0}, {p1}], [1, 1], {groups} as usize).unwrap_or(meta_sh) \
                }}; \
                if rank == 4 {{ \
                    if {transpose} {{ \
                        let w_rlx = m.add_node(rlx_ir::Op::Transpose {{ perm: vec![1, 0, 2, 3] }}, vec![w], Shape::new(&[wi, wc, wk, 1], dt)); \
                        m.conv_transpose2d(x, w_rlx, [{k0}, {k1}], [{s0}, {s1}], [{p0}, {p1}], [1, 1], [0, 0], {groups} as usize, out_sh) \
                    }} else {{ m.conv2d(x, w, [{k0}, {k1}], [{s0}, {s1}], [{p0}, {p1}], {groups} as usize, out_sh) }} \
                }} else if rank == 3 {{ \
                    let c_out = if out_sh.rank() >= 2 {{ out_sh.dim(1).unwrap_static() }} else if {transpose} {{ wi * ({groups} as usize) }} else {{ wc }}; \
                    let l_out = if out_sh.rank() >= 3 {{ out_sh.dim(2).unwrap_static() }} else {{ l }}; \
                    let resh = m.reshape_(x, vec![n as i64, c as i64, 1, l as i64]); \
                    let w4 = m.reshape_(w, vec![wc as i64, wi as i64, wk as i64, 1]); \
                    let conv = if {transpose} {{ \
                        let w_rlx = m.add_node(rlx_ir::Op::Transpose {{ perm: vec![1, 0, 2, 3] }}, vec![w4], Shape::new(&[wi, wc, wk, 1], dt)); \
                        m.conv_transpose2d(resh, w_rlx, [{k0}, 1], [{s0}, 1], [{p0}, 0], [1, 1], [0, 0], {groups} as usize, out_sh) \
                    }} else {{ m.conv2d(resh, w4, [{k0}, 1], [{s0}, 1], [{p0}, 0], {groups} as usize, out_sh) }}; \
                    m.reshape_(conv, vec![n as i64, c_out as i64, l_out as i64]) \
                }} else {{ let stub = format!(\"__conv_stub__/{{}}\", {}); b.params.insert(stub.clone(), vec![0.0; 1]); let mut m0 = HirMut::new(&mut b.hir); m0.param(&stub, out_sh) }} \
                }};",
                rust_str_lit(&node.name),
            )];
            if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                let bias_id = format!("__bias_{out_ident}");
                lines.push(format!(
                    "let {bias_id} = b.tensor({})?;",
                    rust_str_lit(&node.inputs[2])
                ));
                lines.push(format!(
                    "{{ let mut m = HirMut::new(&mut b.hir); \
                    {out_ident} = binary_infer(&mut m, BinaryOp::Add, {out_ident}, {bias_id}, {}) }};",
                    rust_str_lit(&node.name)
                ));
            }
            lines
        }
        "ConvInteger" | "MatMulInteger" => {
            let inner = if node.op == "ConvInteger" {
                "Conv"
            } else {
                "MatMul"
            };
            let mut n = node.clone();
            n.op = inner.to_string();
            emit_node_body(&n, out_ident)
        }
        "DynamicQuantizeLSTM" => {
            let hidden_size = attr_i64(node, "hidden_size", 256);
            let bidirectional = node
                .attrs
                .get("direction")
                .and_then(|v| v.as_str())
                .map(|s| s == "bidirectional")
                .unwrap_or(true);
            let mut lines = Vec::new();
            let mut in_ids = Vec::new();
            for (i, inp) in node.inputs.iter().enumerate() {
                if inp.is_empty() {
                    continue;
                }
                let id = format!("__lstm_in{i}_{out_ident}");
                lines.push(format!("let {id} = b.tensor({})?;", rust_str_lit(inp)));
                in_ids.push(id);
            }
            let in_list = in_ids.join(", ");
            let attrs_bytes = format!(
                "{{ let mut a = vec![0u8; 8]; a[0..4].copy_from_slice(&({hidden_size} as u32).to_le_bytes()); a[4] = u8::from({bidirectional}); a }}"
            );
            lines.push(format!(
                "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); \
                m.add_node(rlx_ir::Op::Custom {{ name: \"onnx.DynamicQuantizeLSTM\".to_string(), num_inputs: {}, attrs: {attrs_bytes} }}, \
                vec![{in_list}], shape_from_meta({meta0}, opts)) }};",
                in_ids.len(),
            ));
            lines
        }
        "Resize" => emit_resize(node, out_ident, &meta0),
        "RandomNormalLike" | "RandomUniformLike" => emit_random_like(node, out_ident, &meta0),
        "RandomNormal" | "RandomUniform" => emit_random(node, out_ident, &meta0),
        "ScatterND" | "ScatterElements" | "CumSum" | "SplitToSequence" | "ConcatFromSequence"
        | "SequenceEmpty" | "Loop" | "If" | "Range" | "ConstantOfShape" | "Shape" | "Slice"
        | "Pad" => {
            let mut lines = vec![format!("// passthrough / delegated: {}", node.op)];
            lines.extend(stub_node(out_ident, &meta0, out_name));
            lines
        }
        other => {
            let mut lines = vec![format!("// UNSUPPORTED ONNX op: {other}")];
            lines.extend(stub_node(out_ident, &meta0, out_name));
            lines
        }
    }
}
