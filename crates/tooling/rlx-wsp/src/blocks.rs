// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `inject` tool: predefined blocks and layers.
//!
//! Agents writing programs into weights rebuild the same few circuits by hand
//! every time — relays, thresholds, AND gates, "only when the paddle is
//! here" gates, comparators. A block writes one of them into chosen units of a
//! node, activation-aware (a tanh unit is "off" at −1, a ReLU unit at 0),
//! replacing those units' incoming column and bias, found in the graph.
//! Inputs are the row labels of that weight (observation labels for the
//! first layer, unit labels above).
//!
//! New programmable layers come from `insert_linear`
//! ([`crate::WeightSpace::insert_linear`]), a graph rewrite that works on
//! any network.

use crate::mlp::Act;
use crate::space::{RowKey, WeightSpace};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize)]
#[serde(tag = "block", rename_all = "snake_case", deny_unknown_fields)]
pub enum Block {
    /// `units[i] = act(gain · inputs[i])`.
    Relay {
        inputs: Vec<RowKey>,
        #[serde(default)]
        gain: Option<f32>,
    },
    /// `unit = act(gain · (Σ weights[i]·inputs[i] − theta))`.
    Threshold {
        inputs: Vec<RowKey>,
        #[serde(default)]
        weights: Vec<f32>,
        theta: f32,
        #[serde(default)]
        gain: Option<f32>,
    },
    /// On when every input is on. `levels`: `"01"` (inputs are 0/1, e.g.
    /// one-hot observations or ReLU units) or `"pm1"` (inputs are ±1, tanh).
    And {
        inputs: Vec<RowKey>,
        #[serde(default = "zero_one")]
        levels: String,
        #[serde(default)]
        gain: Option<f32>,
    },
    /// On when any input is on.
    Or {
        inputs: Vec<RowKey>,
        #[serde(default = "zero_one")]
        levels: String,
        #[serde(default)]
        gain: Option<f32>,
    },
    /// `units[i]` carries `signals[i]` (times `gain`) only while `gate` (a
    /// 0/1 input) is on; otherwise it sits at the off level.
    Gate {
        signals: Vec<RowKey>,
        gate: RowKey,
        #[serde(default)]
        gain: Option<f32>,
        #[serde(default = "big")]
        strength: f32,
    },
    /// Two units: `units[0]` on when Σa − Σb > margin, `units[1]` when
    /// Σb − Σa > margin.
    Compare {
        a: Vec<RowKey>,
        b: Vec<RowKey>,
        #[serde(default = "half")]
        margin: f32,
        #[serde(default)]
        gain: Option<f32>,
    },
}

fn zero_one() -> String {
    "01".into()
}
fn big() -> f32 {
    30.0
}
fn half() -> f32 {
    0.5
}

/// Apply `block` to `units` (all channels of one node); returns what was
/// written. Units are `node[i]`, MLP aliases (`l0.h3`), channel labels
/// (action / digit names), or indices with `node`. The weight column and
/// bias producing each channel are found in the graph, so blocks work on any
/// node fed by a linear map.
pub fn inject(
    ws: &mut WeightSpace,
    block: Block,
    units: Vec<RowKey>,
    node: Option<String>,
) -> Result<Value, String> {
    let resolved: Vec<crate::graphops::Unit> = units
        .iter()
        .map(|u| match (u, &node) {
            (RowKey::Index(i), Some(n)) => Ok(crate::graphops::Unit {
                node: n.clone(),
                index: *i,
            }),
            (RowKey::Index(_), None) => Err("give `node` when units are indices".to_string()),
            (RowKey::Label(l), _) => ws.resolve_unit(l),
        })
        .collect::<Result<_, _>>()?;
    let target = resolved.first().ok_or("no units")?.node.clone();
    if resolved.iter().any(|u| u.node != target) {
        return Err("all units of one block must be channels of the same node".into());
    }
    let g = ws.base_graph()?;
    let id =
        crate::graphops::find_node(&g, &target).ok_or_else(|| format!("no node `{target}`"))?;
    let cp = crate::graphops::channel_params(&g, id);
    let wname = cp
        .weight
        .ok_or_else(|| format!("`{target}` is not fed by a linear map with a param weight"))?;
    let bname = cp.bias;
    let cols: Vec<usize> = resolved.iter().map(|u| u.index).collect();
    // An input is a row label of the weight, or a unit `node[i]` of a node
    // the weight reads (row `i`).
    let rows = |ws: &WeightSpace, keys: &[RowKey]| -> Result<Vec<usize>, String> {
        keys.iter()
            .map(|r| {
                ws.resolve_row(&wname, r).or_else(|e| {
                    let RowKey::Label(l) = r else { return Err(e) };
                    let u = ws.resolve_unit(l).map_err(|_| e.clone())?;
                    let reads = crate::graphops::find_node(&g, &u.node).is_some_and(|n| {
                        crate::graphops::channel_params(&g, n)
                            .readers
                            .contains(&wname)
                    });
                    if reads { Ok(u.index) } else { Err(e) }
                })
            })
            .collect()
    };
    let output_layer = g.outputs.first() == Some(&id);
    // The activation the channels go through decides gains and off levels.
    let act = match &g.node(id).op {
        rlx_ir::Op::Activation(rlx_ir::op::Activation::Tanh) => Some(Act::Tanh),
        rlx_ir::Op::Activation(_) => Some(Act::Relu),
        _ => None,
    };
    // Default gains: saturate tanh firmly; ReLU / linear pass values as they are.
    let gain_or =
        |g: Option<f32>, tanh: f32| g.unwrap_or(if act == Some(Act::Tanh) { tanh } else { 1.0 });
    // Each target column: (input rows with weights, bias).
    let mut plan: Vec<(usize, Vec<(usize, f32)>, f32)> = Vec::new();
    let kind;
    match block {
        Block::Relay { inputs, gain } => {
            kind = "relay";
            let r = rows(ws, &inputs)?;
            if r.len() != cols.len() {
                return Err("relay needs one input per unit".into());
            }
            let g = gain_or(gain, 2.0);
            for (c, r) in cols.iter().zip(r) {
                plan.push((*c, vec![(r, g)], 0.0));
            }
        }
        Block::Threshold {
            inputs,
            weights,
            theta,
            gain,
        } => {
            kind = "threshold";
            let r = rows(ws, &inputs)?;
            let w = if weights.is_empty() {
                vec![1.0; r.len()]
            } else {
                weights
            };
            if w.len() != r.len() {
                return Err("`weights` needs one entry per input".into());
            }
            let g = gain_or(gain, 4.0);
            for c in &cols {
                plan.push((
                    *c,
                    r.iter().zip(&w).map(|(r, w)| (*r, g * w)).collect(),
                    -g * theta,
                ));
            }
        }
        Block::And {
            inputs,
            levels,
            gain,
        } => {
            kind = "and";
            let r = rows(ws, &inputs)?;
            let theta = logic_theta(true, r.len() as f32, &levels)?;
            let g = gain_or(gain, 4.0);
            for c in &cols {
                plan.push((*c, r.iter().map(|r| (*r, g)).collect(), -g * theta));
            }
        }
        Block::Or {
            inputs,
            levels,
            gain,
        } => {
            kind = "or";
            let r = rows(ws, &inputs)?;
            let theta = logic_theta(false, r.len() as f32, &levels)?;
            let g = gain_or(gain, 4.0);
            for c in &cols {
                plan.push((*c, r.iter().map(|r| (*r, g)).collect(), -g * theta));
            }
        }
        Block::Gate {
            signals,
            gate,
            gain,
            strength,
        } => {
            kind = "gate";
            let s = rows(ws, &signals)?;
            let gr = ws.resolve_row(&wname, &gate)?;
            if s.len() != cols.len() {
                return Err("gate needs one signal per unit".into());
            }
            let g = gain_or(gain, 2.0);
            for (c, s) in cols.iter().zip(s) {
                plan.push((*c, vec![(s, g), (gr, strength)], -strength));
            }
        }
        Block::Compare { a, b, margin, gain } => {
            kind = "compare";
            if cols.len() != 2 {
                return Err("compare writes exactly two units: [a > b, b > a]".into());
            }
            let (ra, rb) = (rows(ws, &a)?, rows(ws, &b)?);
            let g = gain_or(gain, 4.0);
            let mut gt: Vec<(usize, f32)> = ra.iter().map(|r| (*r, g)).collect();
            gt.extend(rb.iter().map(|r| (*r, -g)));
            let lt: Vec<(usize, f32)> = gt.iter().map(|(r, w)| (*r, -w)).collect();
            plan.push((cols[0], gt, -g * margin));
            plan.push((cols[1], lt, -g * margin));
        }
    }
    write_plan(ws, &wname, bname.as_deref(), &plan, kind, output_layer)
}

/// Threshold halfway between the closest input sums on either side of the
/// decision: AND over n inputs fires only at the all-on sum, OR at any on.
fn logic_theta(and: bool, n: f32, levels: &str) -> Result<f32, String> {
    match (levels, and) {
        ("01", true) => Ok(n - 0.5),
        ("01", false) => Ok(0.5),
        ("pm1", true) => Ok(n - 1.0),
        ("pm1", false) => Ok(-n + 1.0),
        _ => Err("levels must be `01` or `pm1`".into()),
    }
}

fn write_plan(
    ws: &mut WeightSpace,
    wname: &str,
    bname: Option<&str>,
    plan: &[(usize, Vec<(usize, f32)>, f32)],
    kind: &str,
    output_layer: bool,
) -> Result<Value, String> {
    let (rl, cl) = (ws.row_labels(wname)?, ws.col_labels(wname)?);
    let note = format!("inject {kind} into {} unit(s)", plan.len());
    ws.edit(wname, &note, |t| {
        for (c, inputs, _) in plan {
            for r in 0..t.rows {
                t.data[r * t.cols + c] = 0.0;
            }
            for (r, w) in inputs {
                t.data[r * t.cols + c] += w;
            }
        }
        Ok(())
    })?;
    match bname {
        Some(bname) => ws.edit(bname, &note, |t| {
            for (c, _, b) in plan {
                t.data[*c] = *b;
            }
            Ok(())
        })?,
        None if plan.iter().any(|(_, _, b)| *b != 0.0) => {
            return Err("this block needs a bias but the node has none".into());
        }
        None => {}
    }
    Ok(json!({
        "block": kind,
        "output_layer": output_layer,
        "written": plan.iter().map(|(c, inputs, b)| json!({
            "unit": cl[*c],
            "bias": b,
            "inputs": inputs.iter().map(|(r, w)| json!([rl[*r], w])).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "note": "each unit's incoming column and bias were replaced; its outgoing weights are unchanged",
    }))
}
