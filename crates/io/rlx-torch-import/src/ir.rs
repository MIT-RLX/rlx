// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Deserialization types for `torch-ir.json` — the language-neutral document
//! the Python front-end (`pyrlx.torch_import`) emits from `torch.export`.
//!
//! See the crate docs for the schema. In short: a faithful dump of the Core
//! ATen graph — inputs / weights / nodes (aten op + tagged args + out
//! shape+dtype) / outputs — with all shapes concrete.

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct TorchIr {
    pub format: String,
    pub version: u32,
    pub model_name: String,
    #[serde(default)]
    pub producer: String,
    pub inputs: Vec<IoDef>,
    pub weights: Vec<WeightDef>,
    pub nodes: Vec<NodeDef>,
    pub outputs: Vec<OutputDef>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IoDef {
    pub id: String,
    /// Concrete example extent per axis (a dynamic axis carries its example size).
    pub shape: Vec<i64>,
    pub dtype: String,
    /// Optional per-axis dynamic symbol: `>= 0` marks a dynamic dim with that
    /// symbol id (`0` = batch, `1` = seq, …), `< 0`/absent = static. Emitted by
    /// the front-end when `torch.export` ran with `dynamic_shapes`.
    #[serde(default)]
    pub dynamic: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightDef {
    /// FX placeholder name that args reference (e.g. `p_fc1_weight`).
    pub id: String,
    /// state_dict FQN — the safetensors key (e.g. `fc1.weight`).
    pub key: String,
    pub shape: Vec<i64>,
    pub dtype: String,
    #[serde(default)]
    pub kind: String, // param | buffer | const
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeDef {
    pub id: String,
    pub op: String,
    #[serde(default)]
    pub args: Vec<Arg>,
    #[serde(default)]
    pub kwargs: HashMap<String, Arg>,
    #[serde(default)]
    pub out: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OutputDef {
    #[serde(rename = "ref")]
    pub reference: Option<String>,
    pub shape: Option<Vec<i64>>,
    pub dtype: Option<String>,
    #[serde(default)]
    pub konst: Option<Arg>,
}

/// A tagged FX argument value.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Arg {
    Ref {
        #[serde(rename = "ref")]
        reference: String,
    },
    Bool {
        #[serde(rename = "bool")]
        v: bool,
    },
    Int {
        #[serde(rename = "int")]
        v: i64,
    },
    Float {
        #[serde(rename = "float")]
        v: f64,
    },
    Str {
        #[serde(rename = "str")]
        v: String,
    },
    Dtype {
        dtype: String,
    },
    None {
        none: bool,
    },
    List {
        list: Vec<Arg>,
    },
}

impl Arg {
    pub fn as_ref_name(&self) -> Option<&str> {
        match self {
            Arg::Ref { reference } => Some(reference.as_str()),
            _ => None,
        }
    }
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Arg::Int { v } => Some(*v),
            Arg::Bool { v } => Some(*v as i64),
            _ => None,
        }
    }
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Arg::Float { v } => Some(*v),
            Arg::Int { v } => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Arg::Bool { v } => Some(*v),
            _ => None,
        }
    }
    pub fn is_none(&self) -> bool {
        matches!(self, Arg::None { .. })
    }
    /// Interpret a `{"list": [...]}` of ints as `Vec<i64>`.
    pub fn as_int_list(&self) -> Option<Vec<i64>> {
        match self {
            Arg::List { list } => list.iter().map(|a| a.as_int()).collect(),
            _ => None,
        }
    }
    /// A list of node refs (e.g. `cat([a, b], dim)`).
    pub fn as_ref_list(&self) -> Option<Vec<String>> {
        match self {
            Arg::List { list } => list
                .iter()
                .map(|a| a.as_ref_name().map(|s| s.to_string()))
                .collect(),
            _ => None,
        }
    }
}

/// The primary (index-0) output shape+dtype of a node, if it is a tensor.
pub fn primary_out(out: &Option<serde_json::Value>) -> Option<(Vec<i64>, String)> {
    let v = out.as_ref()?;
    let obj = match v {
        serde_json::Value::Array(items) => items.first()?,
        other => other,
    };
    let shape = obj.get("shape")?.as_array()?;
    let dims: Vec<i64> = shape.iter().filter_map(|d| d.as_i64()).collect();
    let dtype = obj.get("dtype")?.as_str()?.to_string();
    Some((dims, dtype))
}

/// All tensor outputs (for multi-output nodes reached via `_getitem`).
pub fn all_outs(out: &Option<serde_json::Value>) -> Vec<Option<(Vec<i64>, String)>> {
    let Some(v) = out.as_ref() else {
        return vec![];
    };
    let items: Vec<&serde_json::Value> = match v {
        serde_json::Value::Array(items) => items.iter().collect(),
        other => vec![other],
    };
    items
        .into_iter()
        .map(|obj| {
            let shape = obj.get("shape")?.as_array()?;
            let dims: Vec<i64> = shape.iter().filter_map(|d| d.as_i64()).collect();
            let dtype = obj.get("dtype")?.as_str()?.to_string();
            Some((dims, dtype))
        })
        .collect()
}
