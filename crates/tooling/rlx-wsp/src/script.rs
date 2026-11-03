// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `script` tool: a compute sandbox for generating weights.
//!
//! Writing a program into a network means emitting hundreds of numbers that
//! follow a rule ("this cell's landing column times 4"). Typing them is slow
//! and error-prone; the post's agent had a Python sandbox for this. Here the
//! agent writes JavaScript, run in the embedded `quickrs` engine (the one
//! `rlx-js` uses) with **no** file, network, timer or host access — only the
//! language and its built-ins — and a hard instruction budget.
//!
//! The script is a function body. In scope:
//! - `SPEC` — `{ obs_labels, actions, widths, params: {name: {rows, cols}} }`
//!
//! Scripts never see weight values: with them a script can run the network
//! in JavaScript and sidestep the forward-pass budget (an agent did exactly
//! that). Weights are read through the free view tools instead.
//!
//! Its return value comes back as JSON. If it is an object with a `calls`
//! array of tool calls (`{"tool": "set_col", ...}`), those are applied in
//! order (see [`crate::tools`]); probes and rollouts inside cost as usual.

use crate::space::WeightSpace;
use serde_json::{Value, json};

/// Instructions a script may run before it is stopped.
pub const INSTRUCTION_BUDGET: u64 = 500_000_000;

/// Distinct pasted weight values that make a script count as a copy of the
/// network.
pub const MAX_PASTED_WEIGHTS: usize = 16;
/// Number literals a weight-generating script may contain.
pub const MAX_LITERALS: usize = 200;

/// Number literals in JavaScript source (decimal, with optional sign,
/// fraction and exponent; identifiers' digits are skipped).
fn literals(code: &str) -> Vec<f64> {
    let b = code.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let starts =
            b[i].is_ascii_digit() || (b[i] == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit());
        let in_word =
            i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'$');
        if !starts || in_word {
            i += 1;
            continue;
        }
        let start = if i > 0 && b[i - 1] == b'-' { i - 1 } else { i };
        let mut j = i;
        while j < b.len() && (b[j].is_ascii_digit() || b[j] == b'.') {
            j += 1;
        }
        if j < b.len() && (b[j] == b'e' || b[j] == b'E') {
            let mut k = j + 1;
            if k < b.len() && (b[k] == b'-' || b[k] == b'+') {
                k += 1;
            }
            if k < b.len() && b[k].is_ascii_digit() {
                j = k;
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
            }
        }
        if let Ok(v) = code[start..j].parse::<f64>() {
            out.push(v);
        }
        i = j.max(i + 1);
    }
    out
}

/// "Round" values (at most two decimals) appear in weight-generating
/// formulas; trained or derived weights rarely are.
fn is_round(v: f64) -> bool {
    let x = v * 100.0;
    (x - x.round()).abs() < 1e-6
}

/// Refuse scripts that carry a copy of the network: many pasted weight
/// values, or a large literal table. Returns the reason.
pub fn audit_code(ws: &WeightSpace, code: &str) -> Result<(), String> {
    let lits = literals(code);
    if lits.len() > MAX_LITERALS {
        return Err(format!(
            "script refused: {} number literals (max {MAX_LITERALS}). Scripts generate weights \
             from rules; a large literal table looks like copied weights, which would let the \
             script run the network outside the forward-pass budget",
            lits.len()
        ));
    }
    let key = |v: f64, d: f64| (v * d).round() as i64;
    let mut known4 = std::collections::HashSet::new();
    let mut known3 = std::collections::HashSet::new();
    for t in ws
        .state
        .params
        .values()
        .chain(ws.state.networks.values().flat_map(|n| n.values()))
    {
        for &w in &t.data {
            let w = f64::from(w);
            if !is_round(w) {
                known4.insert(key(w, 1e4));
                known3.insert(key(w, 1e3));
            }
        }
    }
    let mut pasted = std::collections::HashSet::new();
    for v in lits {
        if !is_round(v) && (known4.contains(&key(v, 1e4)) || known3.contains(&key(v, 1e3))) {
            pasted.insert(key(v, 1e4));
        }
    }
    if pasted.len() >= MAX_PASTED_WEIGHTS {
        return Err(format!(
            "script refused: it contains {} values of this network's weights. Scripts may not \
             hold copies of the weights — computing the network's outputs that way would bypass \
             the forward-pass budget. Reason about weights with the view tools, run the network \
             with probe / rollout / eval / trace",
            pasted.len()
        ));
    }
    Ok(())
}

/// Run `code` with `SPEC` in scope; returns its JSON return value (`null`
/// when it returns nothing).
pub fn run(ws: &WeightSpace, code: &str) -> Result<Value, String> {
    let spec = ws.spec();
    let params: serde_json::Map<String, Value> = ws
        .state
        .params
        .iter()
        .map(|(n, t)| (n.clone(), json!({ "rows": t.rows, "cols": t.cols })))
        .collect();
    let spec_js = json!({
        "obs_labels": spec.obs_labels,
        "actions": spec.actions,
        "widths": ws.state.mlp.widths,
        "params": params,
    });
    let source = format!(
        "const SPEC = {spec_js};\n\
         (function () {{ const r = (function () {{\n{code}\n}})(); \
         return r === undefined ? 'null' : JSON.stringify(r); }})()",
    );
    let mut ctx = quickrs_core::context::Context::new();
    ctx.set_instruction_budget(Some(INSTRUCTION_BUDGET));
    let value = ctx
        .eval(&source)
        .map_err(|e| format!("script error: {}", ctx.format_exception(&e)))?;
    let text = ctx
        .to_rust_string(&value)
        .map_err(|e| format!("script error: {}", ctx.format_exception(&e)))?;
    serde_json::from_str(&text).map_err(|e| format!("script returned non-JSON ({e}): {text}"))
}
