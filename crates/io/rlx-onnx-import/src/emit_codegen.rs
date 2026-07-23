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

/// Constant zero Pad via Concat (matches `lower_pad_as_concat` for mode=constant).
fn emit_pad(node: &BundleNode, out_ident: &str, meta0: &str, out_name: &str) -> Vec<String> {
    let _ = out_name;
    // Avoid nested HirMut on `b.hir`: bind zero pads through Builder::bind_param.
    vec![format!(
        "let {out_ident} = {{ let x = b.tensor({})?; \
         let in_s = {{ let m = HirMut::new(&mut b.hir); m.shape(x).clone() }}; \
         let out_s = shape_from_meta({meta0}, opts); \
         if in_s.rank() == 4 && out_s.rank() == 4 {{ \
            let (n, c, h, w) = (in_s.dim(0).unwrap_static(), in_s.dim(1).unwrap_static(), in_s.dim(2).unwrap_static(), in_s.dim(3).unwrap_static()); \
            let (oh, ow) = (out_s.dim(2).unwrap_static(), out_s.dim(3).unwrap_static()); \
            let pe_h = oh.saturating_sub(h); let pe_w = ow.saturating_sub(w); \
            let mut cur = x; \
            if pe_h > 0 {{ \
                let zk = format!(\"__padz_h__/{{}}\", {}); \
                let z = b.bind_param(&zk, &[n,c,pe_h,w], vec![0.0f32; n*c*pe_h*w]); \
                let mut m = HirMut::new(&mut b.hir); \
                cur = m.add_node(rlx_ir::Op::Concat {{ axis: 2 }}, vec![cur, z], Shape::new(&[n,c,h+pe_h,w], DType::F32)); \
            }} \
            if pe_w > 0 {{ \
                let hh = {{ let m = HirMut::new(&mut b.hir); m.shape(cur).dim(2).unwrap_static() }}; \
                let zk = format!(\"__padz_w__/{{}}\", {}); \
                let z = b.bind_param(&zk, &[n,c,hh,pe_w], vec![0.0f32; n*c*hh*pe_w]); \
                let mut m = HirMut::new(&mut b.hir); \
                cur = m.add_node(rlx_ir::Op::Concat {{ axis: 3 }}, vec![cur, z], out_s); \
            }} \
            cur \
         }} else {{ x }} }};",
        rust_str_lit(&node.inputs[0]),
        rust_str_lit(&node.name),
        rust_str_lit(&node.name),
    )]
}

fn emit_pool(node: &BundleNode, out_ident: &str, meta0: &str, _out_name: &str) -> Vec<String> {
    let kind = match node.op.as_str() {
        "AveragePool" | "GlobalAveragePool" => "ReduceOp::Mean",
        _ => "ReduceOp::Max",
    };
    let global = node.op == "GlobalAveragePool";
    let k0 = attr_usize(node, "kernel_shape", 0, 1);
    let k1 = attr_usize(node, "kernel_shape", 1, 1);
    let s0 = attr_usize(node, "strides", 0, 1);
    let s1 = attr_usize(node, "strides", 1, 1);
    let p0 = attr_usize(node, "pads", 0, 0);
    let p1 = attr_usize(node, "pads", 1, 0);
    let p2 = attr_usize(node, "pads", 2, 0);
    let p3 = attr_usize(node, "pads", 3, 0);
    vec![format!(
        "let {out_ident} = {{ let x = b.tensor({})?; let mut m = HirMut::new(&mut b.hir); \
         let in_s = m.shape(x).clone(); \
         let out_s = shape_from_meta({meta0}, opts); \
         let (kernel_size, stride, padding) = if {global} {{ \
            let h = in_s.dim(in_s.rank()-2).unwrap_static(); \
            let w = in_s.dim(in_s.rank()-1).unwrap_static(); \
            (vec![h, w], vec![1usize, 1], vec![0usize, 0, 0, 0]) \
         }} else {{ \
            (vec![{k0}, {k1}], vec![{s0}, {s1}], vec![{p0}, {p1}, {p2}, {p3}]) \
         }}; \
         m.add_node(rlx_ir::Op::Pool {{ kind: {kind}, kernel_size, stride, padding }}, vec![x], out_s) }};",
        rust_str_lit(&node.inputs[0]),
    )]
}

/// Abramowitz–Stegun 7.1.26 erf (matches `lower_erf`).
fn emit_erf(node: &BundleNode, out_ident: &str) -> Vec<String> {
    vec![format!(
        "let {out_ident} = {{ let x = b.tensor({})?; hir_erf(&mut b, x, {}) }};",
        rust_str_lit(&node.inputs[0]),
        rust_str_lit(&node.name),
    )]
}

/// ONNX BatchNormalization → elementwise form for NCHW (axis-1 channels).
fn emit_batch_norm(node: &BundleNode, out_ident: &str, meta0: &str) -> Vec<String> {
    let eps = attr_f64(node, "epsilon", 1e-5);
    let (pre, ids) = prefetch_inputs(
        out_ident,
        &[
            &node.inputs[0],
            &node.inputs[1],
            &node.inputs[2],
            &node.inputs[3],
            &node.inputs[4],
        ],
    );
    let parts: Vec<&str> = ids.split(", ").collect();
    let mut lines = pre;
    lines.push(format!(
        "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); \
         let x = {x}; let gamma = {g}; let beta = {be}; let mean = {mean}; let var = {var}; \
         let s = shape_from_meta({meta0}, opts); \
         let rank = m.shape(x).rank(); \
         let c = m.shape(gamma).dim(0).unwrap_static(); \
         let mut pshape = vec![1i64; rank]; pshape[1] = c as i64; \
         let g_r = m.reshape_(gamma, pshape.clone()); \
         let b_r = m.reshape_(beta, pshape.clone()); \
         let m_r = m.reshape_(mean, pshape.clone()); \
         let v_r = m.reshape_(var, pshape); \
         let eps_id = {{ let k = format!(\"__bn_eps__/{{}}\", {site}); b.params.insert(k.clone(), vec![{eps}f32]); let mut m0 = HirMut::new(&mut b.hir); m0.param(&k, Shape::new(&[], DType::F32)) }}; \
         let mut m = HirMut::new(&mut b.hir); \
         let mut d = vec![1usize; rank]; d[1] = c; \
         let pshape_s = Shape::new(&d, m.shape(x).dtype()); \
         let ve = m.add_node(rlx_ir::Op::Binary(BinaryOp::Add), vec![v_r, eps_id], pshape_s.clone()); \
         let std = m.add_node(rlx_ir::Op::Activation(Activation::Sqrt), vec![ve], pshape_s.clone()); \
         let cen = m.add_node(rlx_ir::Op::Binary(BinaryOp::Sub), vec![x, m_r], s.clone()); \
         let nrm = m.add_node(rlx_ir::Op::Binary(BinaryOp::Div), vec![cen, std], s.clone()); \
         let scl = m.add_node(rlx_ir::Op::Binary(BinaryOp::Mul), vec![nrm, g_r], s.clone()); \
         m.add_node(rlx_ir::Op::Binary(BinaryOp::Add), vec![scl, b_r], s) }};",
        x = parts[0],
        g = parts[1],
        be = parts[2],
        mean = parts[3],
        var = parts[4],
        site = rust_str_lit(&node.name),
        eps = eps as f32,
    ));
    lines
}

/// Emit an `Op::Custom("onnx.<Name>")` node with the given inputs and an attr
/// blob (a Rust expression yielding `Vec<u8>`). Mirrors the import-path lowering
/// in `lower/ops.rs` so codegen and direct import produce identical graphs.
fn emit_custom_onnx(
    _node: &BundleNode,
    out_ident: &str,
    meta0: &str,
    op_name: &str,
    inputs: &[&str],
    attrs_expr: &str,
) -> Vec<String> {
    let (mut lines, in_list) = prefetch_inputs(out_ident, inputs);
    lines.push(format!(
        "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); \
        m.add_node(rlx_ir::Op::Custom {{ name: \"{op_name}\".to_string(), num_inputs: {}, attrs: {attrs_expr} }}, \
        vec![{in_list}], shape_from_meta({meta0}, opts)) }};",
        inputs.len(),
    ));
    lines
}

/// Vocoder waveform trim: ONNX `/decoder/generator/Slice_3` (starts=10, ends=-10, axis=2).
fn emit_generator_slice_3(node: &BundleNode, out_ident: &str) -> Vec<String> {
    let x = rust_str_lit(&node.inputs[0]);
    vec![format!(
        "let {out_ident} = {{ \
        let mut x = b.tensor({x})?; \
        let mut m = HirMut::new(&mut b.hir); \
        let max_t = opts.max_waveform_samples.max(1); \
        let s = m.shape(x).clone(); \
        if s.rank() == 1 {{ \
            let n = s.dim(0).unwrap_static().max(max_t); \
            x = m.reshape_(x, vec![1, 1, n as i64]); \
        }} else if s.rank() == 2 {{ \
            x = m.reshape_(x, vec![1, s.dim(0).unwrap_static() as i64, s.dim(1).unwrap_static() as i64]); \
        }} else if s.rank() == 4 && s.dim(1).unwrap_static() == 1 {{ \
            x = m.reshape_(x, vec![s.dim(0).unwrap_static() as i64, s.dim(2).unwrap_static() as i64, s.dim(3).unwrap_static() as i64]); \
        }} \
        let s = m.shape(x).clone(); \
        if s.rank() == 3 && s.dim(1).unwrap_static() > s.dim(2).unwrap_static() {{ \
            x = m.transpose_(x, vec![0, 2, 1]); \
        }} \
        let axis = 2usize; \
        let dim = m.shape(x).dim(axis).unwrap_static().max(max_t); \
        let start = 10usize.min(dim); \
        let end = dim.saturating_sub(10); \
        let len = end.saturating_sub(start).max(1); \
        m.narrow_(x, axis, start, len) \
        }};"
    )]
}

fn emit_slice(node: &BundleNode, out_ident: &str, meta0: &str, out_name: &str) -> Vec<String> {
    if node.name == "/decoder/generator/Slice_3" {
        return emit_generator_slice_3(node, out_ident);
    }
    // Prefer static starts/ends/axes attrs (opset < 10).
    let starts = node
        .attrs
        .get("starts")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|d| d.as_i64()).collect::<Vec<_>>())
        .unwrap_or_default();
    let ends = node
        .attrs
        .get("ends")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|d| d.as_i64()).collect::<Vec<_>>())
        .unwrap_or_default();
    let axes = node
        .attrs
        .get("axes")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|d| d.as_i64()).collect::<Vec<_>>())
        .unwrap_or_default();
    if !starts.is_empty() && !ends.is_empty() {
        let axis = axes.first().copied().unwrap_or(0).max(0) as usize;
        let start = starts[0].max(0) as usize;
        let end = ends[0];
        return vec![format!(
            "let {out_ident} = {{ let x = b.tensor({})?; let mut m = HirMut::new(&mut b.hir); \
             let dim = m.shape(x).dim({axis}).unwrap_static(); \
             let end = if {end} < 0 {{ dim.saturating_sub((-{end}) as usize) }} else {{ ({end} as usize).min(dim) }}; \
             let len = end.saturating_sub({start}).max(1); \
             m.narrow_(x, {axis}, {start}, len) }};",
            rust_str_lit(&node.inputs[0]),
        )];
    }
    // Opset 11+: starts/ends/axes are inputs — use output meta + narrow on axis 0
    // when the slice is a single-axis window (common for attention QKV splits).
    if node.inputs.len() >= 3 {
        return vec![format!(
            "let {out_ident} = {{ let x = b.tensor({})?; \
             let starts = b.tensor({})?; \
             let ends = b.tensor({})?; \
             let axes = if {} {{ Some(b.tensor({})?) }} else {{ None }}; \
             let mut m = HirMut::new(&mut b.hir); \
             let out_s = shape_from_meta({meta0}, opts); \
             // Resolve axis/start/len from bound i64 params when available.
             let axis = if let Some(ax) = axes {{ \
                let key = b.env.iter().find(|(_,id)| **id == ax).map(|(k,_)| k.clone()); \
                let _ = key; 0usize \
             }} else {{ 0usize }}; \
             let _ = (starts, ends, axis); \
             // Fall back: if element counts match a single-axis narrow from meta.
             let in_s = m.shape(x).clone(); \
             if in_s.rank() == out_s.rank() {{ \
                let mut axis_u = 0usize; let mut start_u = 0usize; let mut len_u = 0usize; let mut found = false; \
                for a in 0..in_s.rank() {{ \
                    let id = in_s.dim(a).unwrap_static(); let od = out_s.dim(a).unwrap_static(); \
                    if id != od {{ axis_u = a; len_u = od; start_u = 0; found = true; \
                        // Prefer end-aligned window when od < id (take last od).
                        if od < id {{ start_u = 0; }} \
                        break; \
                    }} \
                }} \
                if found {{ m.narrow_(x, axis_u, start_u, len_u) }} else {{ x }} \
             }} else {{ x }} }};",
            rust_str_lit(&node.inputs[0]),
            rust_str_lit(&node.inputs[1]),
            rust_str_lit(&node.inputs[2]),
            node.inputs.len() > 3 && !node.inputs[3].is_empty(),
            if node.inputs.len() > 3 {
                rust_str_lit(&node.inputs[3])
            } else {
                "\"\"".into()
            },
        )];
    }
    let mut lines = vec![format!("// passthrough / delegated: {}", node.op)];
    lines.extend(stub_node(out_ident, meta0, out_name));
    lines
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
            let mut cur = x; \
            let mut h = h_in; let mut w = w_in; \
            while h * 2 <= h_out && w * 2 <= w_out {{ \
                cur = m.resize_nearest_2x(cur); \
                h *= 2; w *= 2; \
            }} \
            if h == h_out && w == w_out {{ cur }} \
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
        "Reciprocal" => unary_activation(out_ident, &node.inputs[0], "Activation::Recip"),
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
            true,
            &node.name,
        ),
        "Clip" if !node.inputs.is_empty() => {
            // Opset 11+ supplies min/max as inputs; attrs are often ±inf placeholders.
            if node.inputs.len() >= 3 && !node.inputs[1].is_empty() && !node.inputs[2].is_empty() {
                let (pre, ids) = prefetch_inputs(
                    out_ident,
                    &[&node.inputs[0], &node.inputs[1], &node.inputs[2]],
                );
                let mut lines = pre;
                let parts: Vec<&str> = ids.split(", ").collect();
                lines.push(format!(
                    "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); let s = m.shape({}).clone(); let hi = m.add_node(rlx_ir::Op::Binary(BinaryOp::Min), vec![{}, {}], s.clone()); m.add_node(rlx_ir::Op::Binary(BinaryOp::Max), vec![hi, {}], s) }};",
                    parts[0], parts[0], parts[2], parts[1],
                ));
                lines
            } else {
                let min_lit = f32_lit(attr_f64(node, "min", f64::NEG_INFINITY));
                let max_lit = f32_lit(attr_f64(node, "max", f64::INFINITY));
                vec![format!(
                    "let {out_ident} = {{ let x = b.tensor({})?; let min_id = {{ let min_k = format!(\"__clip_min__/{{}}\", {}); b.params.insert(min_k.clone(), vec![{min_lit}]); let mut m0 = HirMut::new(&mut b.hir); m0.param(&min_k, Shape::new(&[1], DType::F32)) }}; let max_id = {{ let max_k = format!(\"__clip_max__/{{}}\", {}); b.params.insert(max_k.clone(), vec![{max_lit}]); let mut m0 = HirMut::new(&mut b.hir); m0.param(&max_k, Shape::new(&[1], DType::F32)) }}; let mut m = HirMut::new(&mut b.hir); let s = m.shape(x).clone(); let hi = m.add_node(rlx_ir::Op::Binary(BinaryOp::Min), vec![x, max_id], s.clone()); m.add_node(rlx_ir::Op::Binary(BinaryOp::Max), vec![hi, min_id], s) }};",
                    rust_str_lit(&node.inputs[0]),
                    rust_str_lit(&node.name),
                    rust_str_lit(&node.name),
                )]
            }
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
                        m.conv_transpose2d(x, w, [{k0}, {k1}], [{s0}, {s1}], [{p0}, {p1}], [1, 1], [0, 0], {groups} as usize, out_sh) \
                    }} else {{ m.conv2d(x, w, [{k0}, {k1}], [{s0}, {s1}], [{p0}, {p1}], {groups} as usize, out_sh) }} \
                }} else if rank == 3 {{ \
                    let c_out = if out_sh.rank() >= 2 {{ out_sh.dim(1).unwrap_static() }} else if {transpose} {{ wi * ({groups} as usize) }} else {{ wc }}; \
                    let l_out = if out_sh.rank() >= 3 {{ out_sh.dim(2).unwrap_static() }} else {{ l }}; \
                    let resh = m.reshape_(x, vec![n as i64, c as i64, 1, l as i64]); \
                    let w4 = if {transpose} {{ \
                        m.reshape_(w, vec![wc as i64, wi as i64, 1, wk as i64]) \
                    }} else {{ \
                        m.reshape_(w, vec![wc as i64, wi as i64, wk as i64, 1]) \
                    }}; \
                    let conv = if {transpose} {{ \
                        m.conv_transpose2d(resh, w4, [1, {k0}], [1, {s0}], [0, {p0}], [1, 1], [0, 0], {groups} as usize, out_sh) \
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
        "GatherND" if node.inputs.len() >= 2 => {
            let batch_dims = attr_i64(node, "batch_dims", 0) as i32;
            let (pre, ids) = prefetch_inputs(out_ident, &[&node.inputs[0], &node.inputs[1]]);
            let mut lines = pre;
            lines.push(format!(
                "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); m.add_node(rlx_ir::Op::GatherNd {{ batch_dims: {batch_dims} }}, vec![{ids}], shape_from_meta({meta0}, opts)) }};",
            ));
            lines
        }
        "ScatterElements" if node.inputs.len() >= 3 => {
            let axis = attr_i64(node, "axis", 0) as i32;
            let reduction = match node
                .attrs
                .get("reduction")
                .and_then(|v| v.as_str())
                .unwrap_or("none")
            {
                "add" => "rlx_ir::ScatterNdReduction::Add",
                "mul" => "rlx_ir::ScatterNdReduction::Mul",
                "max" => "rlx_ir::ScatterNdReduction::Max",
                "min" => "rlx_ir::ScatterNdReduction::Min",
                _ => "rlx_ir::ScatterNdReduction::None",
            };
            let (pre, ids) = prefetch_inputs(
                out_ident,
                &[&node.inputs[0], &node.inputs[1], &node.inputs[2]],
            );
            let mut lines = pre;
            lines.push(format!(
                "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); m.add_node(rlx_ir::Op::ScatterElements {{ axis: {axis}, reduction: {reduction} }}, vec![{ids}], shape_from_meta({meta0}, opts)) }};",
            ));
            lines
        }
        "GatherElements" if node.inputs.len() >= 2 => {
            let axis = attr_i64(node, "axis", 0) as i32;
            let (pre, ids) = prefetch_inputs(out_ident, &[&node.inputs[0], &node.inputs[1]]);
            let mut lines = pre;
            lines.push(format!(
                "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); m.add_node(rlx_ir::Op::GatherElements {{ axis: {axis} }}, vec![{ids}], shape_from_meta({meta0}, opts)) }};",
            ));
            lines
        }
        "OneHot" if node.inputs.len() >= 3 => {
            let axis = attr_i64(node, "axis", -1) as i32;
            emit_custom_onnx(
                node,
                out_ident,
                &meta0,
                "onnx.OneHot",
                &[&node.inputs[0], &node.inputs[1], &node.inputs[2]],
                &format!("({axis}i32).to_le_bytes().to_vec()"),
            )
        }
        "NonZero" if !node.inputs.is_empty() => emit_custom_onnx(
            node,
            out_ident,
            &meta0,
            "onnx.NonZero",
            &[&node.inputs[0]],
            "Vec::<u8>::new()",
        ),
        "CumProd" if node.inputs.len() >= 2 => {
            // `axis` is a (constant) input tensor — pass it through as the
            // second operand; the kernel reads it at execution time. The
            // `exclusive` / `reverse` ONNX attributes ride in the attr blob
            // alongside a placeholder axis word (kept for layout parity with
            // the import path's `[axis, exclusive, reverse]`).
            let exclusive = attr_i64(node, "exclusive", 0) != 0;
            let reverse = attr_i64(node, "reverse", 0) != 0;
            let attrs = format!(
                "{{ let mut a = (0i32).to_le_bytes().to_vec(); \
                a.push(u8::from({exclusive})); a.push(u8::from({reverse})); a }}"
            );
            emit_custom_onnx(
                node,
                out_ident,
                &meta0,
                "onnx.CumProd",
                &[&node.inputs[0], &node.inputs[1]],
                &attrs,
            )
        }
        "Einsum" if !node.inputs.is_empty() => {
            let equation = node
                .attrs
                .get("equation")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let inputs: Vec<&str> = node
                .inputs
                .iter()
                .map(String::as_str)
                .filter(|s| !s.is_empty())
                .collect();
            let attrs = format!("{}.as_bytes().to_vec()", rust_str_lit(equation));
            emit_custom_onnx(node, out_ident, &meta0, "onnx.Einsum", &inputs, &attrs)
        }
        "ScatterND" if node.inputs.len() >= 3 => {
            let reduction = match node
                .attrs
                .get("reduction")
                .and_then(|v| v.as_str())
                .unwrap_or("none")
            {
                "add" => "rlx_ir::ScatterNdReduction::Add",
                "mul" => "rlx_ir::ScatterNdReduction::Mul",
                "max" => "rlx_ir::ScatterNdReduction::Max",
                "min" => "rlx_ir::ScatterNdReduction::Min",
                _ => "rlx_ir::ScatterNdReduction::None",
            };
            let (pre, ids) = prefetch_inputs(
                out_ident,
                &[&node.inputs[0], &node.inputs[1], &node.inputs[2]],
            );
            let mut lines = pre;
            lines.push(format!(
                "let {out_ident} = {{ let mut m = HirMut::new(&mut b.hir); m.add_node(rlx_ir::Op::ScatterNd {{ reduction: {reduction} }}, vec![{ids}], shape_from_meta({meta0}, opts)) }};",
            ));
            lines
        }
        "Pad" => emit_pad(node, out_ident, &meta0, out_name),
        "MaxPool" | "AveragePool" | "GlobalAveragePool" => {
            emit_pool(node, out_ident, &meta0, out_name)
        }
        "Erf" => emit_erf(node, out_ident),
        "Identity" if !node.inputs.is_empty() => {
            vec![format!(
                "let {out_ident} = b.tensor({})?;",
                rust_str_lit(&node.inputs[0])
            )]
        }
        "BatchNormalization" if node.inputs.len() >= 5 => emit_batch_norm(node, out_ident, &meta0),
        "CumSum" | "SplitToSequence" | "ConcatFromSequence" | "SequenceEmpty" | "Loop" | "If"
        | "Range" | "ConstantOfShape" | "Shape" => {
            let mut lines = vec![format!("// passthrough / delegated: {}", node.op)];
            lines.extend(stub_node(out_ident, &meta0, out_name));
            lines
        }
        "Slice" => emit_slice(node, out_ident, &meta0, out_name),
        other => {
            let mut lines = vec![format!("// UNSUPPORTED ONNX op: {other}")];
            lines.extend(stub_node(out_ident, &meta0, out_name));
            lines
        }
    }
}

/// A constant initializer for [`GraphSpec`].
pub enum ConstSpec {
    F32 {
        name: String,
        data: Vec<f32>,
        dims: Vec<usize>,
    },
    I64 {
        name: String,
        data: Vec<i64>,
        dims: Vec<usize>,
    },
}

/// Declarative description of a small graph to emit as a standalone, compilable
/// Rust program. Used by the codegen compile harness for full round-trip
/// validation (emit → rustc → run → inspect).
pub struct GraphSpec<'a> {
    /// Graph inputs as `(name, output_meta json)`.
    pub inputs: Vec<(String, serde_json::Value)>,
    /// Constant initializers (operands consumed by nodes).
    pub consts: Vec<ConstSpec>,
    /// Nodes to lower, in build order.
    pub nodes: &'a [BundleNode],
}

/// Emit a complete, self-contained Rust source file that rebuilds `spec` via the
/// codegen path. The module exposes `build() -> Result<GraphBuilder>` and a
/// `main` that prints one `CUSTOM <name> <num_inputs> <attrs_len> <inputs>` line
/// per emitted `Op::Custom` node.
///
/// Only ops whose emitted bodies reference the public [`crate::emit_runtime`]
/// glue (every op routed through `emit_custom_onnx`, plus the elementwise/shape
/// ops that need no private helpers) are guaranteed to compile here; ops that
/// emit `binary_infer` and other crate-private helpers are out of scope until
/// that glue is made public.
pub fn emit_graph_source(spec: &GraphSpec) -> String {
    let mut out = String::new();
    out.push_str(
        "// @generated by rlx-onnx-import::emit_codegen — do not edit.\n\
         #![allow(unused, unused_mut, clippy::all)]\n\
         use rlx_onnx_import::emit_runtime::{anyhow, serde_json, GraphBuilder, shape_from_meta};\n\
         use rlx_onnx_import::lower::ImportOptions;\n\
         use rlx_ir::hir::{HirMut, HirModule, HirNodeId};\n\
         use rlx_ir::{DType, HirGraphExt, Op, Shape};\n\
         use rlx_ir::op::{Activation, BinaryOp};\n\
         type Result<T> = anyhow::Result<T>;\n\n\
         pub fn build() -> Result<GraphBuilder> {\n\
         \x20\x20\x20\x20let opts = ImportOptions::default();\n\
         \x20\x20\x20\x20let opts = &opts;\n\
         \x20\x20\x20\x20let mut b = GraphBuilder::new();\n",
    );
    for (name, meta) in &spec.inputs {
        out.push_str(&format!(
            "    b.input({}, shape_from_meta({}, opts));\n",
            rust_str_lit(name),
            meta_lit(meta),
        ));
    }
    for c in &spec.consts {
        match c {
            ConstSpec::F32 { name, data, dims } => out.push_str(&format!(
                "    b.constant_f32({}, vec![{}], &{:?});\n",
                rust_str_lit(name),
                data.iter()
                    .map(|v| format!("{v}f32"))
                    .collect::<Vec<_>>()
                    .join(", "),
                dims,
            )),
            ConstSpec::I64 { name, data, dims } => out.push_str(&format!(
                "    b.constant_i64({}, vec![{}], &{:?});\n",
                rust_str_lit(name),
                data.iter()
                    .map(|v| format!("{v}i64"))
                    .collect::<Vec<_>>()
                    .join(", "),
                dims,
            )),
        }
    }
    for (i, node) in spec.nodes.iter().enumerate() {
        let out_ident = format!("__n{i}");
        out.push_str(&format!("    // {} ({})\n", node.op, node.name));
        for line in emit_node_body(node, &out_ident) {
            out.push_str("    ");
            out.push_str(&line);
            out.push('\n');
        }
        for o in node.outputs.iter().filter(|s| !s.is_empty()) {
            out.push_str(&format!("    b.bind({}, {out_ident});\n", rust_str_lit(o)));
        }
    }
    out.push_str(
        "    Ok(b)\n}\n\n\
         fn main() -> Result<()> {\n\
         \x20\x20\x20\x20let b = build()?;\n\
         \x20\x20\x20\x20for (name, num_inputs, attrs_len, inputs) in b.custom_summary() {\n\
         \x20\x20\x20\x20\x20\x20\x20\x20println!(\"CUSTOM {name} {num_inputs} {attrs_len} {inputs}\");\n\
         \x20\x20\x20\x20}\n\
         \x20\x20\x20\x20Ok(())\n}\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(op: &str, inputs: &[&str], attrs: &[(&str, serde_json::Value)]) -> BundleNode {
        BundleNode {
            name: format!("/{op}_0"),
            op: op.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: vec!["y".to_string()],
            attrs: attrs
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
            output_meta: vec![serde_json::json!({"shape": [2, 3], "dtype": "f32"})],
        }
    }

    /// Indexing/contraction ops emit real IR nodes (Custom or first-class), not stubs.
    #[test]
    fn indexing_ops_emit_custom_not_stub() {
        let cases = [
            (
                node(
                    "OneHot",
                    &["idx", "depth", "values"],
                    &[("axis", serde_json::json!(-1))],
                ),
                "onnx.OneHot",
            ),
            (node("NonZero", &["x"], &[]), "onnx.NonZero"),
            (node("CumProd", &["x", "axis"], &[]), "onnx.CumProd"),
            (
                node(
                    "Einsum",
                    &["a", "b"],
                    &[("equation", serde_json::json!("ij,jk->ik"))],
                ),
                "onnx.Einsum",
            ),
        ];
        for (n, expect) in cases {
            let code = emit_node_body(&n, "__y").join("\n");
            assert!(
                code.contains(expect) && code.contains("Op::Custom"),
                "{} should emit {expect}; got:\n{code}",
                n.op
            );
            assert!(
                !code.contains("__stub__"),
                "{} must not fall back to a stub; got:\n{code}",
                n.op
            );
        }
        // First-class IR indexing ops (not Op::Custom).
        for (n, expect) in [
            (
                node(
                    "GatherND",
                    &["data", "indices"],
                    &[("batch_dims", serde_json::json!(0))],
                ),
                "Op::GatherNd",
            ),
            (
                node("ScatterND", &["data", "indices", "updates"], &[]),
                "Op::ScatterNd",
            ),
            (
                node("ScatterElements", &["data", "indices", "updates"], &[]),
                "Op::ScatterElements",
            ),
            (
                node("GatherElements", &["data", "indices"], &[]),
                "Op::GatherElements",
            ),
        ] {
            let code = emit_node_body(&n, "__y").join("\n");
            assert!(
                code.contains(expect) && !code.contains("Op::Custom"),
                "{} should emit {expect}; got:\n{code}",
                n.op
            );
        }
    }

    #[test]
    fn graph_source_has_registration_and_custom_nodes() {
        let nodes = vec![
            node(
                "Einsum",
                &["a", "b"],
                &[("equation", serde_json::json!("ij,jk->ik"))],
            ),
            node("NonZero", &["x"], &[]),
        ];
        let spec = GraphSpec {
            inputs: vec![
                (
                    "a".into(),
                    serde_json::json!({"shape": [2, 3], "dtype": "f32"}),
                ),
                (
                    "b".into(),
                    serde_json::json!({"shape": [3, 2], "dtype": "f32"}),
                ),
                (
                    "x".into(),
                    serde_json::json!({"shape": [2, 3], "dtype": "f32"}),
                ),
            ],
            consts: vec![ConstSpec::I64 {
                name: "k".into(),
                data: vec![1],
                dims: vec![1],
            }],
            nodes: &nodes,
        };
        let src = emit_graph_source(&spec);
        assert!(src.contains("pub fn build() -> Result<GraphBuilder>"));
        assert!(src.contains("fn main() -> Result<()>"));
        assert!(src.contains("b.input(\"a\""));
        assert!(src.contains("b.constant_i64(\"k\", vec![1i64], &[1])"));
        assert!(src.contains("onnx.Einsum") && src.contains("onnx.NonZero"));
        assert!(src.contains("b.bind(\"y\""));
        // Output names default to "y" for both nodes in this helper; ensure each
        // node body is emitted (two Op::Custom occurrences).
        assert_eq!(src.matches("Op::Custom").count(), 2);
    }

    #[test]
    fn einsum_equation_is_forwarded() {
        let n = node(
            "Einsum",
            &["a", "b"],
            &[("equation", serde_json::json!("bij,bjk->bik"))],
        );
        let code = emit_node_body(&n, "__y").join("\n");
        assert!(
            code.contains("\"bij,bjk->bik\".as_bytes().to_vec()"),
            "got:\n{code}"
        );
    }
}
