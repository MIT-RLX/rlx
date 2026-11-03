// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `inject` tool: predefined blocks and layers.
//!
//! Agents writing programs into weights rebuild the same few circuits by hand
//! every time — relays, thresholds, AND gates, "only when the paddle is
//! here" gates, comparators. A block writes one of them into chosen units of a
//! layer, activation-aware (a tanh unit is "off" at −1, a ReLU unit at 0),
//! replacing those units' incoming column and bias. Inputs are the layer's
//! row labels (observation labels for layer 0, `l{k-1}.hI` above).
//!
//! `insert_layer` adds a hidden layer that relays its input, so the network
//! computes (almost) the same function and the new layer is free to program:
//! exact for ReLU (identity on non-negative inputs); for tanh the relay runs
//! at gain ε and the next layer is scaled by 1/ε (error ≈ ε²/3 ≈ 3·10⁻⁵).

use crate::mlp::{Act, MlpSpec};
use crate::space::{RowKey, Tensor, WeightSpace};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Gain of the tanh relay in [`insert_layer`].
const TANH_RELAY_GAIN: f32 = 0.01;

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

/// The layer `units` live in (`l{k}.hI`, an output label, or indices with
/// `layer`).
fn target_layer(ws: &WeightSpace, units: &[RowKey], layer: Option<usize>) -> Result<usize, String> {
    if let Some(k) = layer {
        return Ok(k);
    }
    for u in units {
        if let RowKey::Label(l) = u
            && let Some(k) = l
                .strip_prefix('l')
                .and_then(|s| s.split_once(".h"))
                .and_then(|(k, _)| k.parse().ok())
        {
            return Ok(k);
        }
    }
    if units.iter().all(|u| matches!(u, RowKey::Label(_))) {
        // Output labels (actions / digits) are columns of the last layer.
        return Ok(ws.state.mlp.n_layers() - 1);
    }
    Err("give `layer` when units are indices".into())
}

/// Apply `block` to `units` of `layer`; returns what was written.
pub fn inject(
    ws: &mut WeightSpace,
    block: Block,
    units: Vec<RowKey>,
    layer: Option<usize>,
) -> Result<Value, String> {
    if ws.state.graph.is_some() {
        return Err("blocks address MLP layers".into());
    }
    let k = target_layer(ws, &units, layer)?;
    if k >= ws.state.mlp.n_layers() {
        return Err(format!("no layer {k}"));
    }
    let wname = format!("l{k}.w");
    let bname = format!("l{k}.b");
    let cols: Vec<usize> = units
        .iter()
        .map(|u| ws.resolve_col(&wname, u))
        .collect::<Result<_, _>>()?;
    let rows = |ws: &WeightSpace, keys: &[RowKey]| -> Result<Vec<usize>, String> {
        keys.iter().map(|r| ws.resolve_row(&wname, r)).collect()
    };
    let output_layer = k + 1 == ws.state.mlp.n_layers();
    let act = ws.state.mlp.act;
    // Default gains: saturate tanh firmly; ReLU passes values as they are.
    let gain_or =
        |g: Option<f32>, tanh: f32| g.unwrap_or(if act == Act::Tanh { tanh } else { 1.0 });
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
    write_plan(ws, &wname, &bname, &plan, kind, output_layer)
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
    bname: &str,
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
    ws.edit(bname, &note, |t| {
        for (c, _, b) in plan {
            t.data[*c] = *b;
        }
        Ok(())
    })?;
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

/// Insert a relay layer after hidden layer `after` (0-based); `width`
/// (default: that layer's width) may add spare units, which start at zero.
/// Checkpoints are dropped (shapes change).
pub fn insert_layer(
    ws: &mut WeightSpace,
    after: usize,
    width: Option<usize>,
) -> Result<Value, String> {
    if ws.state.graph.is_some() || !ws.state.networks.is_empty() {
        return Err("insert_layer works on single-MLP spaces".into());
    }
    let mlp = ws.state.mlp.clone();
    let hidden = mlp.n_layers() - 1;
    if after >= hidden {
        return Err(format!(
            "`after` must name a hidden layer 0..{}",
            hidden.saturating_sub(1)
        ));
    }
    let w0 = mlp.widths[after + 1];
    let w = width.unwrap_or(w0);
    if w < w0 {
        return Err(format!(
            "width must be at least {w0} to relay layer {after}"
        ));
    }
    let (gain, scale) = match mlp.act {
        Act::Relu => (1.0, 1.0),
        Act::Tanh => (TANH_RELAY_GAIN, 1.0 / TANH_RELAY_GAIN),
    };
    let old = ws.state.params.clone();
    let mut params: BTreeMap<String, Tensor> = BTreeMap::new();
    let n = mlp.n_layers();
    for l in 0..=n {
        // New layer index l; the inserted layer is l = after + 1.
        if l == after + 1 {
            let mut relay = Tensor::zeros(w0, w);
            for i in 0..w0 {
                relay.data[i * w + i] = gain;
            }
            params.insert(format!("l{l}.w"), relay);
            params.insert(format!("l{l}.b"), Tensor::zeros(1, w));
            continue;
        }
        let src = if l > after + 1 { l - 1 } else { l };
        let mut wt = old[&format!("l{src}.w")].clone();
        if l == after + 2 {
            // Reads the relay: undo the tanh gain, pad rows for spare units.
            let mut t = Tensor::zeros(w, wt.cols);
            for r in 0..wt.rows {
                for c in 0..wt.cols {
                    t.data[r * wt.cols + c] = wt.at(r, c) * scale;
                }
            }
            wt = t;
        }
        params.insert(format!("l{l}.w"), wt);
        params.insert(format!("l{l}.b"), old[&format!("l{src}.b")].clone());
    }
    let mut widths = mlp.widths.clone();
    widths.insert(after + 2, w);
    let dropped = ws.state.checkpoints.len();
    ws.state.checkpoints.clear();
    ws.set_architecture(
        MlpSpec {
            widths: widths.clone(),
            act: mlp.act,
        },
        params,
    );
    ws.state
        .log
        .push(format!("insert_layer after l{after}: widths {widths:?}"));
    Ok(json!({
        "widths": widths,
        "inserted": format!("l{}", after + 1),
        "renamed": format!("layers after it shift up by one (old l{} is now l{})", after + 1, after + 2),
        "relay": if mlp.act == Act::Tanh {
            format!("tanh relay at gain {TANH_RELAY_GAIN}, next layer scaled by {scale}; output changes by ~1e-4")
        } else {
            "exact ReLU relay".to_string()
        },
        "spare_units": w - w0,
        "checkpoints_dropped": dropped,
    }))
}
