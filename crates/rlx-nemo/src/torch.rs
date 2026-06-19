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

//! Harvest a flat `name -> TensorMeta` table from an unpickled torch
//! state dict. State dicts are normally a flat `OrderedDict` of dotted
//! keys; we still recurse into nested mappings (joining with `.`) and
//! reach through a top-level tuple/list wrapper, so both `torch.save(
//! model.state_dict())` and the occasional `{"state_dict": …}` layout
//! resolve correctly.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

use crate::pickle::{TensorMeta, Value};

/// Walk a decoded pickle root and collect every tensor leaf, keyed by its
/// dotted path.
pub fn collect_state_dict(root: &Value) -> Result<BTreeMap<String, TensorMeta>> {
    let mut out = BTreeMap::new();
    let dict = find_mapping(root).ok_or_else(|| anyhow_no_mapping(root))?;
    walk(&dict, "", &mut out);
    if out.is_empty() {
        bail!("no tensors found in checkpoint (is this a state dict?)");
    }
    Ok(out)
}

fn anyhow_no_mapping(root: &Value) -> anyhow::Error {
    anyhow::anyhow!(
        "checkpoint root is not a mapping (got {:?})",
        root_kind(root)
    )
}

fn root_kind(v: &Value) -> &'static str {
    match v {
        Value::Dict(_) => "dict",
        Value::Tuple(_) => "tuple",
        Value::List(_) => "list",
        Value::Tensor(_) => "tensor",
        _ => "scalar/other",
    }
}

/// Find the first dict either at the root or one level inside a wrapper.
fn find_mapping(v: &Value) -> Option<Vec<(Value, Value)>> {
    match v {
        Value::Dict(d) => Some(d.borrow().clone()),
        Value::Tuple(t) => t.iter().find_map(find_mapping),
        Value::List(l) => l.borrow().iter().find_map(find_mapping),
        _ => None,
    }
}

fn walk(entries: &[(Value, Value)], prefix: &str, out: &mut BTreeMap<String, TensorMeta>) {
    for (k, v) in entries {
        let key = match k {
            Value::Str(s) => s.to_string(),
            Value::Int(i) => i.to_string(),
            _ => continue,
        };
        let path = if prefix.is_empty() {
            key
        } else {
            format!("{prefix}.{key}")
        };
        match v {
            Value::Tensor(t) => {
                out.insert(path, t.clone());
            }
            Value::Dict(d) => walk(&d.borrow(), &path, out),
            _ => {}
        }
    }
}
