// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Targeted edits read off the graph, for any model.
//!
//! - **Channel bias** (`channel_bias`, `token_bias`): add constants to some
//!   channels of a node — on a language model's `logits`, a token's score.
//! - **Steering** (`steer`): add a direction (e.g. one `contrast` stored) to
//!   a node at every position.
//! - **Key → value writes** (`write_kv`): the closed-form rank-1 edit that
//!   makes a weight map a key (its input on chosen items) to a chosen
//!   output: `W += k (v − kᵀW)ᵀ / kᵀk`.
//!
//! A bias or steering edit on a node that is exactly `x + bias` (a `[width]`
//! param) is folded into that param — an ordinary weight edit, carried by
//! the adapter. Elsewhere it is recorded as a graph rewrite (an added
//! constant), replayed on every run and every open of a package.

use crate::graphops::{bias_of, find_node, matmuls_reading, node_ref, width};
use crate::space::{Rewrite, RowKey, WeightSpace};
use crate::tools::ItemsArg;
use serde::Deserialize;
use serde_json::{Value, json};

/// A vector given inline or by the name `contrast` stored it under.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum VecArg {
    Name(String),
    Values(Vec<f32>),
}

impl VecArg {
    pub fn resolve(&self, ws: &WeightSpace) -> Result<Vec<f32>, String> {
        match self {
            VecArg::Values(v) => Ok(v.clone()),
            VecArg::Name(n) => ws.state.vectors.get(n).cloned().ok_or_else(|| {
                format!(
                    "no stored vector `{n}` (have: {})",
                    ws.state
                        .vectors
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }),
        }
    }
}

/// The node a logit edit addresses: `node`, else one named `logits`, else
/// the graph's first output.
fn logits_node(ws: &WeightSpace, node: Option<String>) -> Result<String, String> {
    if let Some(n) = node {
        return Ok(n);
    }
    let g = ws.graph()?;
    if find_node(&g, "logits").is_some() {
        return Ok("logits".into());
    }
    let out = *g.outputs.first().ok_or("the graph has no outputs")?;
    Ok(node_ref(&g, out))
}

/// Channel indices of `node` from indices, its channel labels or
/// `node[i]` units.
fn channels_of(ws: &WeightSpace, node: &str, keys: &[RowKey]) -> Result<Vec<usize>, String> {
    let labels = ws.unit_channel_labels(node).unwrap_or_default();
    keys.iter()
        .map(|k| match k {
            RowKey::Index(i) => Ok(*i),
            RowKey::Label(l) => {
                if let Some(i) = labels.iter().position(|x| x == l) {
                    return Ok(i);
                }
                let u = ws.resolve_unit(l)?;
                if u.node != node {
                    return Err(format!("`{l}` is a channel of `{}`, not `{node}`", u.node));
                }
                Ok(u.index)
            }
        })
        .collect()
}

/// Add `values` to `channels` of `node`: a bias edit where the node is
/// `x + bias`, else a recorded rewrite.
pub fn add_to_channels(
    ws: &mut WeightSpace,
    node: &str,
    channels: Vec<usize>,
    values: Vec<f32>,
    what: &str,
) -> Result<Value, String> {
    let g = ws.graph()?;
    let id = find_node(&g, node).ok_or_else(|| format!("no node `{node}` (see list_nodes)"))?;
    let w = width(&g, id).ok_or("the node has no static channel axis")?;
    if let Some(c) = channels.iter().find(|&&c| c >= w) {
        return Err(format!("channel {c} out of range ({w})"));
    }
    if let Some(b) = bias_of(&g, id) {
        let (cs, vs) = (channels.clone(), values.clone());
        let folded = ws.edit(&b, what, |t| {
            for (&c, &v) in cs.iter().zip(&vs) {
                t.data[c] += v;
            }
            Ok(())
        });
        if folded.is_ok() {
            return Ok(
                json!({ "node": node, "channels": channels.len(), "folded_into": b,
                              "note": "an edit of the bias param (carried by checkpoints and the adapter)" }),
            );
        }
    }
    ws.add_rewrite(Rewrite::ChannelBias {
        node: node.to_string(),
        channels: channels.clone(),
        values,
    });
    Ok(
        json!({ "node": node, "channels": channels.len(), "rewrite": ws.state.rewrites.len() - 1,
               "note": "recorded as a graph rewrite (list / remove with `rewrites`)" }),
    )
}

pub fn channel_bias(
    ws: &mut WeightSpace,
    node: &str,
    channels: &[RowKey],
    value: Option<f32>,
    values: Option<Vec<f32>>,
) -> Result<Value, String> {
    let cs = channels_of(ws, node, channels)?;
    let vs = per_channel(cs.len(), value, values)?;
    add_to_channels(ws, node, cs, vs, "channel_bias")
}

fn per_channel(n: usize, value: Option<f32>, values: Option<Vec<f32>>) -> Result<Vec<f32>, String> {
    match (value, values) {
        (_, Some(v)) if v.len() == n => Ok(v),
        (_, Some(v)) => Err(format!("{} values for {n} channels", v.len())),
        (Some(x), None) => Ok(vec![x; n]),
        (None, None) => Err("give `value` (all channels) or `values`".into()),
    }
}

/// Bias the scores of tokens (by text or id) on the logits node.
pub fn token_bias(
    ws: &mut WeightSpace,
    tokens: &[String],
    ids: &[u32],
    value: Option<f32>,
    values: Option<Vec<f32>>,
    node: Option<String>,
) -> Result<Value, String> {
    let mut all: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
    let mut shown = Vec::new();
    if !tokens.is_empty() {
        let codec = ws
            .tokenizer()
            .ok_or("the package carries no tokenizer; give `ids`")?;
        for t in tokens {
            let enc = codec
                .encode_plain(t)
                .map_err(|e| format!("tokenizing `{t}`: {e:#}"))?;
            if enc.len() != 1 {
                return Err(format!(
                    "`{t}` is {} tokens {enc:?}, not one; bias its ids with `ids`",
                    enc.len()
                ));
            }
            shown.push(json!({ "token": t, "id": enc[0] }));
            all.push(enc[0] as usize);
        }
    }
    if all.is_empty() {
        return Err("give `tokens` or `ids`".into());
    }
    let node = logits_node(ws, node)?;
    let vs = per_channel(all.len(), value, values)?;
    let mut r = add_to_channels(ws, &node, all, vs, "token_bias")?;
    r["tokens"] = json!(shown);
    Ok(r)
}

/// Add `scale · vector` to `node` at every position.
pub fn steer(
    ws: &mut WeightSpace,
    node: &str,
    vector: &VecArg,
    scale: f32,
) -> Result<Value, String> {
    let v = vector.resolve(ws)?;
    let g = ws.graph()?;
    let id = find_node(&g, node).ok_or_else(|| format!("no node `{node}` (see list_nodes)"))?;
    let w = width(&g, id).ok_or("the node has no static channel axis")?;
    if v.len() != w {
        return Err(format!("a {}-vector for a node of {w} channels", v.len()));
    }
    if bias_of(&g, id).is_some() {
        let vs: Vec<f32> = v.iter().map(|x| x * scale).collect();
        return add_to_channels(ws, node, (0..w).collect(), vs, &format!("steer {node}"));
    }
    ws.add_rewrite(Rewrite::AddVector {
        node: node.to_string(),
        vector: v,
        scale,
    });
    Ok(
        json!({ "node": node, "rewrite": ws.state.rewrites.len() - 1, "scale": scale,
               "note": "recorded as a graph rewrite (list / remove with `rewrites`)" }),
    )
}

/// List recorded rewrites, optionally removing one first.
pub fn rewrites(ws: &mut WeightSpace, remove: Option<usize>) -> Result<Value, String> {
    let removed = remove.map(|i| ws.remove_rewrite(i)).transpose()?;
    Ok(json!({
        "removed": removed.map(|r| r.describe()),
        "rewrites": ws.state.rewrites.iter().enumerate()
            .map(|(i, r)| json!({ "index": i, "rewrite": r.describe() }))
            .collect::<Vec<_>>(),
    }))
}

/// The mean of per-item vectors.
fn mean_of(read: &crate::pkgrun::Read, k: usize) -> Vec<f32> {
    let w = read[0][k].len();
    (0..w)
        .map(|c| read.iter().map(|x| x[k][c]).sum::<f32>() / read.len() as f32)
        .collect()
}

/// Rank-1 key → value write into `param` (see the module docs). The key is
/// the input of the `MatMul` that reads `param`, averaged over `keys`; the
/// target output is that `MatMul`'s output averaged over `target` items, or
/// the current output plus `scale · delta`. One forward pass per item set.
#[allow(clippy::too_many_arguments)]
pub fn write_kv(
    ws: &mut WeightSpace,
    param: &str,
    keys: &ItemsArg,
    target: Option<&ItemsArg>,
    delta: Option<&VecArg>,
    scale: f32,
    matmul: Option<String>,
    pool: Option<&str>,
) -> Result<Value, String> {
    let g = ws.graph()?;
    let uses = matmuls_reading(&g, param);
    let (mm, lhs) = match (uses.as_slice(), &matmul) {
        ([], _) => {
            return Err(format!(
                "no MatMul reads `{param}` (see list_nodes `params`)"
            ));
        }
        (_, Some(m)) => {
            let id = find_node(&g, m).ok_or_else(|| format!("no node `{m}`"))?;
            *uses
                .iter()
                .find(|(x, _)| *x == id)
                .ok_or_else(|| format!("`{m}` does not read `{param}`"))?
        }
        ([one], None) => *one,
        (many, None) => {
            return Err(format!(
                "{} MatMuls read `{param}`; pick one with `matmul` ({})",
                many.len(),
                many.iter()
                    .map(|(m, _)| node_ref(&g, *m))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    };
    let (key_node, out_node) = (node_ref(&g, lhs), node_ref(&g, mm));
    let w = ws.param(param)?.clone();
    if target.is_none() && delta.is_none() {
        return Err("give `target` items or a `delta` (vector or stored name)".into());
    }
    let need = 1 + usize::from(target.is_some());
    if ws.remaining() < need {
        return Err(format!("write_kv needs {need} forward passes"));
    }
    ws.charge(&format!("write_kv {param}: keys"))?;
    let kr = crate::tools::read_items(ws, keys, std::slice::from_ref(&key_node), pool)?;
    let k = mean_of(&kr, 0);
    if k.len() != w.rows {
        return Err(format!(
            "the key has {} channels but `{param}` has {} rows (expects x @ W with W [in, out])",
            k.len(),
            w.rows
        ));
    }
    let o: Vec<f32> = (0..w.cols)
        .map(|j| (0..w.rows).map(|i| k[i] * w.at(i, j)).sum())
        .collect();
    let v: Vec<f32> = if let Some(t) = target {
        ws.charge(&format!("write_kv {param}: target"))?;
        let tr = crate::tools::read_items(ws, t, std::slice::from_ref(&out_node), pool)?;
        mean_of(&tr, 0)
    } else {
        let d = delta.expect("checked").resolve(ws)?;
        if d.len() != w.cols {
            return Err(format!(
                "delta has {} values; `{param}` has {} columns",
                d.len(),
                w.cols
            ));
        }
        o.iter().zip(&d).map(|(a, b)| a + scale * b).collect()
    };
    if v.len() != w.cols {
        return Err(format!(
            "the target has {} channels; `{param}` has {} columns",
            v.len(),
            w.cols
        ));
    }
    let kk: f32 = k.iter().map(|x| x * x).sum();
    if kk <= 1e-20 {
        return Err("the key is zero on these items".into());
    }
    let u: Vec<f32> = k.iter().map(|x| x / kk).collect();
    let change: Vec<f32> = v.iter().zip(&o).map(|(a, b)| a - b).collect();
    let change_norm = change.iter().map(|x| x * x).sum::<f32>().sqrt();
    ws.edit_rank1(param, u, change, &format!("write_kv {param}"))?;
    Ok(json!({
        "param": param, "key_node": key_node, "output_node": out_node,
        "key_norm": kk.sqrt(), "output_change_norm": change_norm,
        "old_output_norm": o.iter().map(|x| x * x).sum::<f32>().sqrt(),
        "note": "exact for the mean key; another input x changes by (x·k / k·k) times the same output change",
    }))
}
