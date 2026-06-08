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

//! Load a `.onnx` file directly into the bundle IR consumed by [`crate::lower`].

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use onnx::onnx::{
    AttributeProto_AttributeType, ModelProto, TensorProto, TensorProto_DataType, TypeProto_Tensor,
};
use protobuf::Message;
use serde_json::json;

use crate::bundle::{BundleManifest, BundleNode, IoMeta, TensorMeta};
use crate::lower::{ImportOptions, ImportReport, build_hir_from_parts};
use crate::shape_propagate::propagate_shapes;

fn tensor_dtype(elem: TensorProto_DataType) -> String {
    match elem {
        TensorProto_DataType::FLOAT => "f32".to_string(),
        TensorProto_DataType::INT64 => "i64".to_string(),
        TensorProto_DataType::INT32 => "i32".to_string(),
        TensorProto_DataType::BOOL => "bool".to_string(),
        other => format!("type_{other:?}"),
    }
}

fn shape_from_tensor_type(tt: &TypeProto_Tensor) -> Vec<serde_json::Value> {
    if !tt.has_shape() {
        return vec![json!("?")];
    }
    tt.get_shape()
        .get_dim()
        .iter()
        .map(|d| {
            if d.has_dim_value() {
                let v = d.get_dim_value();
                if v > 0 { json!(v) } else { json!("?") }
            } else {
                json!("?")
            }
        })
        .collect()
}

fn f32_from_raw(raw: &[u8], n: usize) -> Option<Vec<f32>> {
    if raw.len() != n * 4 {
        return None;
    }
    Some(
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn tensor_to_f32(name: &str, t: &TensorProto) -> Result<Vec<f32>> {
    let n = t.get_dims().iter().product::<i64>().max(1) as usize;
    if !t.get_float_data().is_empty() {
        return Ok(t.get_float_data().to_vec());
    }
    if let Some(v) = f32_from_raw(t.get_raw_data(), n) {
        return Ok(v);
    }
    let dt = t.get_data_type();
    match dt {
        TensorProto_DataType::FLOAT | TensorProto_DataType::UNDEFINED => {
            if let Some(v) = f32_from_raw(t.get_raw_data(), n) {
                return Ok(v);
            }
        }
        TensorProto_DataType::INT64 => {
            let ints: Vec<i64> = if !t.get_int64_data().is_empty() {
                t.get_int64_data().to_vec()
            } else if t.get_raw_data().len() == n * 8 {
                t.get_raw_data()
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            } else {
                Vec::new()
            };
            return Ok(ints.into_iter().map(|x| x as f32).collect());
        }
        TensorProto_DataType::INT32 => {
            let raw = t.get_raw_data();
            if raw.len() == n * 4 {
                return Ok(raw
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect());
            }
        }
        TensorProto_DataType::UINT8 | TensorProto_DataType::INT8 => {
            let raw = t.get_raw_data();
            return Ok(raw.iter().map(|&b| b as f32).collect());
        }
        _ if dt as i32 == 10 => {
            // FLOAT16
            let raw = t.get_raw_data();
            if raw.len() == n * 2 {
                return Ok(raw
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect());
            }
        }
        _ if dt as i32 == 16 => {
            // BFLOAT16 (not in older protobuf enums)
            let raw = t.get_raw_data();
            if raw.len() == n * 2 {
                return Ok(raw
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u32::from(c[0]) | (u32::from(c[1]) << 8)) << 16))
                    .collect());
            }
        }
        _ => {}
    }
    anyhow::bail!("initializer {name}: unsupported dtype for native import")
}

fn tensor_to_i64(name: &str, t: &TensorProto) -> Result<Vec<i64>> {
    let n = t.get_dims().iter().product::<i64>().max(1) as usize;
    if !t.get_int64_data().is_empty() {
        return Ok(t.get_int64_data().to_vec());
    }
    if !t.get_int32_data().is_empty() {
        return Ok(t.get_int32_data().iter().map(|&x| x as i64).collect());
    }
    let raw = t.get_raw_data();
    match t.get_data_type() as i32 {
        7 if raw.len() == n * 8 => {
            return Ok(raw
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect());
        }
        6 if raw.len() == n * 4 => {
            return Ok(raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                .collect());
        }
        9 => {
            return Ok(raw.iter().map(|&b| i64::from(b != 0)).collect());
        }
        _ => {}
    }
    anyhow::bail!("initializer {name}: unsupported i64 dtype for native import")
}

fn parse_attribute(a: &onnx::onnx::AttributeProto) -> Option<serde_json::Value> {
    use AttributeProto_AttributeType::*;
    let name = a.get_name();
    match a.get_field_type() {
        INT => Some(json!(a.get_i())),
        INTS => Some(json!(a.get_ints().to_vec())),
        FLOAT => Some(json!(a.get_f())),
        FLOATS => Some(json!(a.get_floats().to_vec())),
        STRING => {
            let s = a.get_s();
            Some(json!(String::from_utf8_lossy(s).into_owned()))
        }
        STRINGS => Some(json!(
            a.get_strings()
                .iter()
                .map(|s| String::from_utf8_lossy(s).into_owned())
                .collect::<Vec<_>>()
        )),
        TENSOR => {
            let t = a.get_t();
            let dims: Vec<i64> = t.get_dims().to_vec();
            Some(json!({
                "tensor": {
                    "dims": dims,
                    "dtype": tensor_dtype(t.get_data_type()),
                }
            }))
        }
        GRAPH => Some(json!({ "graph": true, "name": name })),
        _ => None,
    }
}

fn output_meta_from_value_info(v: &onnx::onnx::ValueInfoProto) -> serde_json::Value {
    if v.has_field_type() {
        let tp = v.get_field_type();
        if tp.has_tensor_type() {
            let tt = tp.get_tensor_type();
            return json!({
                "shape": shape_from_tensor_type(tt),
                "dtype": tensor_dtype(tt.get_elem_type()),
            });
        }
    }
    json!({"shape": ["?"], "dtype": "f32"})
}

fn build_value_info_map(graph: &onnx::onnx::GraphProto) -> HashMap<String, serde_json::Value> {
    let mut map = HashMap::new();
    for v in graph.get_input() {
        map.insert(v.get_name().to_string(), output_meta_from_value_info(v));
    }
    for v in graph.get_output() {
        map.insert(v.get_name().to_string(), output_meta_from_value_info(v));
    }
    for v in graph.get_value_info() {
        map.insert(v.get_name().to_string(), output_meta_from_value_info(v));
    }
    for init in graph.get_initializer() {
        let shape: Vec<serde_json::Value> =
            init.get_dims().iter().map(|&d| json!(d.max(1))).collect();
        map.insert(
            init.get_name().to_string(),
            json!({"shape": shape, "dtype": tensor_dtype(init.get_data_type())}),
        );
    }
    map
}

fn io_meta_from_value_info(v: &onnx::onnx::ValueInfoProto) -> IoMeta {
    let meta = output_meta_from_value_info(v);
    let shape = meta
        .get("shape")
        .and_then(|s| s.as_array())
        .cloned()
        .unwrap_or_else(|| vec![json!("?")]);
    let dtype = meta
        .get("dtype")
        .and_then(|d| d.as_str())
        .unwrap_or("f32")
        .to_string();
    IoMeta {
        name: v.get_name().to_string(),
        meta: TensorMeta { shape, dtype },
    }
}

/// Parse ONNX and return manifest + graph pieces (no HIR lowering).
pub fn prepare_onnx_file(
    path: &Path,
) -> Result<(
    BundleManifest,
    Vec<BundleNode>,
    HashMap<String, Vec<f32>>,
    HashMap<String, Vec<i64>>,
    HashMap<String, Vec<usize>>,
)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut model = ModelProto::new();
    model
        .merge_from_bytes(&bytes)
        .context("parse ONNX protobuf")?;
    let graph = model.get_graph();

    let mut params = HashMap::new();
    let mut i64_params = HashMap::new();
    for init in graph.get_initializer() {
        let name = init.get_name().to_string();
        match init.get_data_type() as i32 {
            6 | 7 | 9 => {
                i64_params.insert(name.clone(), tensor_to_i64(&name, init)?);
            }
            _ => {
                params.insert(name.clone(), tensor_to_f32(&name, init)?);
            }
        }
    }

    let inputs: Vec<IoMeta> = graph
        .get_input()
        .iter()
        .filter(|i| !params.contains_key(i.get_name()))
        .map(io_meta_from_value_info)
        .collect();

    let outputs: Vec<IoMeta> = graph
        .get_output()
        .iter()
        .map(io_meta_from_value_info)
        .collect();

    let value_info = build_value_info_map(graph);

    let mut nodes: Vec<BundleNode> = graph
        .get_node()
        .iter()
        .map(|n| {
            let mut attrs = HashMap::new();
            for a in n.get_attribute() {
                if let Some(v) = parse_attribute(a) {
                    attrs.insert(a.get_name().to_string(), v);
                }
            }
            let name = if n.get_name().is_empty() {
                n.get_op_type().to_string()
            } else {
                n.get_name().to_string()
            };
            let output_meta: Vec<serde_json::Value> = n
                .get_output()
                .iter()
                .map(|out| {
                    value_info
                        .get(out)
                        .cloned()
                        .unwrap_or_else(|| json!({"shape": ["?"], "dtype": "f32"}))
                })
                .collect();
            let output_meta = if output_meta.is_empty() {
                vec![json!({"shape": ["?"], "dtype": "f32"})]
            } else {
                output_meta
            };
            BundleNode {
                name,
                op: n.get_op_type().to_string(),
                inputs: n.get_input().to_vec(),
                outputs: n.get_output().to_vec(),
                attrs,
                output_meta,
            }
        })
        .collect();

    let manifest = BundleManifest {
        source_onnx: path.display().to_string(),
        inputs,
        outputs,
        node_count: nodes.len(),
        initializer_count: params.len(),
        op_histogram: nodes.iter().fold(HashMap::new(), |mut m, n| {
            *m.entry(n.op.clone()).or_insert(0) += 1;
            m
        }),
    };

    let mut init_shapes = HashMap::new();
    for init in graph.get_initializer() {
        init_shapes.insert(
            init.get_name().to_string(),
            init.get_dims().iter().map(|&d| d.max(1) as usize).collect(),
        );
    }

    let opts = ImportOptions::default();
    propagate_shapes(&mut nodes, &manifest, &init_shapes, &opts);

    Ok((manifest, nodes, params, i64_params, init_shapes))
}

/// Load ONNX from disk and lower to HIR + f32 params.
pub fn build_hir_from_onnx_file(
    path: &Path,
    opts: ImportOptions,
) -> Result<(
    rlx_ir::hir::HirModule,
    HashMap<String, Vec<f32>>,
    ImportReport,
    BundleManifest,
)> {
    let (manifest, nodes, params, i64_params, init_shapes) = prepare_onnx_file(path)?;
    let (hir, params, _typed, report) = build_hir_from_parts(
        &manifest,
        nodes,
        params,
        crate::tensor_data::TypedParams::new(),
        i64_params,
        &init_shapes,
        opts,
    )?;
    Ok((hir, params, report, manifest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn onnx_test_model() -> Option<std::path::PathBuf> {
        std::env::var("RLX_ONNX_TEST_MODEL")
            .ok()
            .map(std::path::PathBuf::from)
            .filter(|p| p.exists())
    }

    #[test]
    fn onnx_matmul_integer_rewritten() {
        let Some(path) = onnx_test_model() else {
            return;
        };
        let (manifest, nodes, params, _i64, init_shapes) =
            prepare_onnx_file(&path).expect("prepare");
        let opts = crate::ImportOptions::quant_bundle();
        let rewritten = crate::rewrite::rewrite_graph(
            nodes,
            &params,
            &init_shapes,
            &manifest,
            &opts,
            &std::collections::HashSet::new(),
        );
        let remaining = rewritten
            .nodes
            .iter()
            .filter(|n| n.op == "MatMulInteger")
            .count();
        assert_eq!(
            remaining, 0,
            "expected all MatMulInteger rewritten, {remaining} remain"
        );
    }

    #[test]
    fn onnx_prepare_infers_shapes() {
        let Some(path) = onnx_test_model() else {
            return;
        };
        let (manifest, nodes, ..) = prepare_onnx_file(&path).expect("prepare");
        assert_eq!(manifest.node_count, nodes.len());
        let with_shapes = nodes
            .iter()
            .filter(|n| {
                n.output_meta.first().is_some_and(|m| {
                    m.get("shape")
                        .and_then(|s| s.as_array())
                        .map(|a| {
                            !a.is_empty()
                                && a.iter()
                                    .any(|d| d.as_u64().is_some() || d.as_str().is_some())
                        })
                        .unwrap_or(false)
                })
            })
            .count();
        assert!(
            with_shapes > nodes.len() / 4,
            "expected shape propagation on raw ONNX, got {with_shapes}/{}",
            nodes.len()
        );
    }

    #[test]
    fn onnx_prepare_loads_i64_constants() {
        let Some(path) = onnx_test_model() else {
            return;
        };
        let (manifest, nodes, params, i64_params, init_shapes) =
            prepare_onnx_file(&path).expect("prepare onnx");
        assert_eq!(manifest.node_count, nodes.len());
        assert!(!params.is_empty());
        assert!(!i64_params.is_empty(), "expected i64 constant initializers");
        assert!(init_shapes.len() >= params.len() + i64_params.len());
    }

    #[test]
    fn micro_bench_onnx_initializers_have_raw_data() {
        let path = std::path::Path::new("/tmp/bench_micro.onnx");
        if !path.exists() {
            eprintln!("skip: {}", path.display());
            return;
        }
        let (.., params, _i64, init_shapes) = prepare_onnx_file(path).expect("prepare");
        assert!(params.contains_key("w"));
        assert!(params.contains_key("b"));
        assert!(init_shapes.contains_key("w"));
    }
}
