// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Check an agent's account of its own program.
//!
//! Agents end a run describing which units do what. [`unit_importance`]
//! switches each hidden unit off in turn and measures the change in the
//! held-out score; [`claim_check`] compares that with the units the agent's
//! text names, flagging important units it never mentions and mentioned
//! units that do nothing. The text parsing is a heuristic: it reads unit
//! references like `l1.h3`, `h12`, `h0–h9`, `units 20-29`.

use crate::space::{Tensor, WeightSpace};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

/// `(unit, score change when it is off)` for every hidden unit with any
/// incoming weight or bias, largest effect first. Not budgeted.
pub fn unit_importance(
    ws: &mut WeightSpace,
    episodes: usize,
) -> Result<Vec<(String, f32)>, String> {
    unit_importance_of(ws, episodes, None)
}

/// [`unit_importance`] restricted to `only` (all used units when `None`).
/// Units are the channels of every activation node of the graph; a channel
/// counts as used unless the weight column and bias that produce it are both
/// zero. Each is zeroed with a mask in the graph, so this works on any
/// network.
pub fn unit_importance_of(
    ws: &mut WeightSpace,
    episodes: usize,
    only: Option<&BTreeSet<String>>,
) -> Result<Vec<(String, f32)>, String> {
    if ws.is_loaded() {
        return Err("unit importance needs a scored task".into());
    }
    let g = ws.base_graph()?;
    let base = score(ws, episodes)?;
    let mut out = Vec::new();
    for node in crate::graphops::unit_nodes(&g) {
        let id = crate::graphops::find_node(&g, &node).expect("named");
        let w = crate::graphops::width(&g, id).unwrap_or(0);
        let cp = crate::graphops::channel_params(&g, id);
        let labels = ws.unit_channel_labels(&node)?;
        for i in 0..w {
            let weight_used = cp
                .weight
                .as_ref()
                .and_then(|p| ws.param(p).ok())
                .is_none_or(|t| (0..t.rows).any(|r| t.at(r, i) != 0.0));
            let bias_used = cp
                .bias
                .as_ref()
                .and_then(|p| ws.param(p).ok())
                .is_some_and(|t| t.data[i] != 0.0);
            if !weight_used && !bias_used {
                continue;
            }
            let unit = labels
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("{node}[{i}]"));
            if only.is_some_and(|o| !o.contains(&unit)) {
                continue;
            }
            ws.set_masks(vec![crate::graphops::Unit {
                node: node.clone(),
                index: i,
            }]);
            let s = score(ws, episodes);
            ws.set_masks(Vec::new());
            out.push((unit, s? - base));
        }
    }
    out.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    Ok(out)
}

/// The post-run check. Networks written from zero: every used unit's
/// ablation effect against the units the agent's text names
/// ([`claim_check`]), plus measured roles. Repair / fix tasks: what the
/// agent changed and what each change is worth ([`edit_check`]) — a whole
/// trained unit does many things, so switching it off says little about the
/// repair. Reading tasks are scored by the answer alone.
pub fn check_run(ws: &mut WeightSpace, text: &str, episodes: usize) -> Result<Value, String> {
    if ws.state.env.is_reading() {
        return Ok(json!({ "scope": "reading task: scored by the answer, no edits checked" }));
    }
    // Unit claims are about control networks: on a question-answering
    // package there are none to measure, only a held-out pass to pay for.
    if ws.state.qa.is_some() {
        return Ok(json!({ "scope": "question answering: no unit claims to check" }));
    }
    if ws.state.env == crate::env::EnvKind::FixPolicy {
        let before = crate::tasks::fix_policy_params()?;
        return edit_check(ws, &before, episodes);
    }
    if ws.state.env.is_classifier() {
        let before = if ws.state.env == crate::env::EnvKind::MnistSteer {
            crate::mnist::trained(None)?
        } else {
            crate::mnist::trained(ws.state.env.trigger())?
        };
        return edit_check(ws, &before, episodes);
    }
    let imp = unit_importance(ws, episodes)?;
    let mut r = claim_check(text, &imp, 0.5);
    r["scope"] = json!("every used channel of every activation node");
    let important: Vec<String> = imp
        .iter()
        .filter(|(_, d)| d.abs() >= 0.5)
        .map(|(u, _)| u.clone())
        .collect();
    r["roles"] = match crate::roles::measure(ws, 32) {
        Ok(roles) => crate::roles::grade(text, &roles, &important),
        Err(e) => json!({ "error": e }),
    };
    Ok(r)
}

/// For a repair: per changed tensor, which rows / columns changed (by
/// label) and the held-out score with that tensor put back as it was; plus
/// the score with every edit undone.
pub fn edit_check(
    ws: &mut WeightSpace,
    before: &BTreeMap<String, Tensor>,
    episodes: usize,
) -> Result<Value, String> {
    let live = ws.state.params.clone();
    let now = ws.held_out(episodes)?;
    let mut tensors = Vec::new();
    for (name, after) in &live {
        let Some(old) = before.get(name) else {
            tensors.push(json!({ "param": name, "added": true }));
            continue;
        };
        if old.data.len() != after.data.len() {
            tensors.push(json!({ "param": name, "reshaped": true }));
            continue;
        }
        let changed: usize = after
            .data
            .iter()
            .zip(&old.data)
            .filter(|(a, b)| a != b)
            .count();
        if changed == 0 {
            continue;
        }
        let (rl, cl) = (ws.row_labels(name)?, ws.col_labels(name)?);
        let rows: Vec<&String> = (0..after.rows)
            .filter(|&r| (0..after.cols).any(|c| after.at(r, c) != old.at(r, c)))
            .map(|r| &rl[r])
            .collect();
        let cols: Vec<&String> = (0..after.cols)
            .filter(|&c| (0..after.rows).any(|r| after.at(r, c) != old.at(r, c)))
            .map(|c| &cl[c])
            .collect();
        let mut reverted = live.clone();
        reverted.insert(name.clone(), old.clone());
        ws.set_params(reverted);
        let undone = ws.held_out(episodes)?;
        tensors.push(json!({
            "param": name,
            "changed_entries": changed,
            "changed_fraction": changed as f64 / after.data.len() as f64,
            "changed_rows": rows.len(),
            "rows": rows.iter().take(32).collect::<Vec<_>>(),
            "changed_cols": cols.len(),
            "cols": cols.iter().take(32).collect::<Vec<_>>(),
            "score_with_this_reverted": undone["score"],
            "detail_with_this_reverted": undone["detail"],
        }));
    }
    let total: usize = live.values().map(|t| t.data.len()).sum();
    let changed_total: usize = live
        .iter()
        .filter_map(|(n, a)| before.get(n).map(|b| (a, b)))
        .filter(|(a, b)| a.data.len() == b.data.len())
        .map(|(a, b)| a.data.iter().zip(&b.data).filter(|(x, y)| x != y).count())
        .sum();
    let originals: BTreeMap<String, Tensor> = live
        .keys()
        .filter_map(|n| before.get(n).map(|t| (n.clone(), t.clone())))
        .collect();
    let all_undone = if originals.len() == live.len() {
        ws.set_params(originals);
        let o = ws.held_out(episodes)?;
        Some(o)
    } else {
        None
    };
    ws.set_params(live);
    Ok(json!({
        "scope": "edits relative to the model as handed over",
        "score_now": now["score"],
        "detail_now": now["detail"],
        "changed_weights": changed_total,
        "changed_fraction": changed_total as f64 / total.max(1) as f64,
        "score_all_edits_undone": all_undone.as_ref().map(|o| o["score"].clone()),
        "detail_all_edits_undone": all_undone.as_ref().map(|o| o["detail"].clone()),
        "tensors": tensors,
    }))
}

fn score(ws: &mut WeightSpace, episodes: usize) -> Result<f32, String> {
    Ok(ws.held_out(episodes)?["score"].as_f64().unwrap_or(0.0) as f32)
}

/// Unit references in free text: `(layer if stated, index)`.
pub fn mentioned_units(text: &str) -> BTreeSet<(Option<usize>, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = BTreeSet::new();
    let num = |i: &mut usize| -> Option<usize> {
        let start = *i;
        while *i < chars.len() && chars[*i].is_ascii_digit() {
            *i += 1;
        }
        chars[start..*i].iter().collect::<String>().parse().ok()
    };
    let mut i = 0;
    while i < chars.len() {
        // `l<k>.h<i>` / `h<i>` / `units <i>`, each optionally a range.
        let mut layer = None;
        let mut j = i;
        let at_word = i == 0 || !chars[i - 1].is_alphanumeric();
        if at_word && chars[j] == 'l' && j + 1 < chars.len() && chars[j + 1].is_ascii_digit() {
            let mut k = j + 1;
            if let Some(l) = num(&mut k) {
                if k + 1 < chars.len() && chars[k] == '.' && chars[k + 1] == 'h' {
                    layer = Some(l);
                    j = k + 1;
                }
            }
        }
        let start = if at_word || layer.is_some() {
            if chars[j] == 'h' && j + 1 < chars.len() && chars[j + 1].is_ascii_digit() {
                Some(j + 1)
            } else if text_at(&chars, j, "units ") && at_word {
                Some(j + 6)
            } else if text_at(&chars, j, "unit ") && at_word {
                Some(j + 5)
            } else {
                None
            }
        } else {
            None
        };
        let Some(mut k) = start else {
            i += 1;
            continue;
        };
        let Some(a) = num(&mut k) else {
            i += 1;
            continue;
        };
        let mut b = a;
        if k < chars.len() && matches!(chars[k], '-' | '–' | '—') {
            let mut m = k + 1;
            if m < chars.len() && chars[m] == 'h' {
                m += 1;
            }
            if let Some(end) = num(&mut m) {
                if end >= a && end - a < 256 {
                    b = end;
                    k = m;
                }
            }
        }
        for u in a..=b {
            out.insert((layer, u));
        }
        i = k.max(i + 1);
    }
    out
}

fn text_at(chars: &[char], i: usize, s: &str) -> bool {
    let s: Vec<char> = s.chars().collect();
    chars.len() >= i + s.len()
        && chars[i..i + s.len()]
            .iter()
            .zip(&s)
            .all(|(a, b)| a.to_ascii_lowercase() == *b)
}

/// Whether `segment` names `unit`: its exact label (`node[i]`, an action
/// name, ...), or for MLP aliases `l{k}.h{i}` the forms `l{k}.h{i}`, `h{i}`,
/// ranges such as `h0–h9`.
pub fn names_unit(segment: &str, unit: &str) -> bool {
    if segment.contains(unit) {
        return true;
    }
    let Some((k, i)) = unit
        .strip_prefix('l')
        .and_then(|s| s.split_once(".h"))
        .and_then(|(k, i)| Some((k.parse::<usize>().ok()?, i.parse::<usize>().ok()?)))
    else {
        return false;
    };
    let m = mentioned_units(segment);
    m.contains(&(Some(k), i)) || m.contains(&(None, i))
}

/// Compare measured unit importance with the units `text` names.
/// `threshold` is the score change that counts as "matters".
pub fn claim_check(text: &str, importance: &[(String, f32)], threshold: f32) -> Value {
    let important: Vec<&(String, f32)> = importance
        .iter()
        .filter(|(_, d)| d.abs() >= threshold)
        .collect();
    let unmentioned: Vec<Value> = important
        .iter()
        .filter(|(u, _)| !names_unit(text, u))
        .map(|(u, d)| json!([u, d]))
        .collect();
    let inert: Vec<Value> = importance
        .iter()
        .filter(|(u, d)| d.abs() < threshold / 5.0 && text.contains(u.as_str()))
        .map(|(u, d)| json!([u, d]))
        .collect();
    json!({
        "threshold": threshold,
        "units_measured": importance.len(),
        "important_units": important.iter().map(|(u, d)| json!([u, d])).collect::<Vec<_>>(),
        "important_but_not_mentioned": unmentioned,
        "mentioned_but_inert": inert,
        "note": "heuristic: mentions are parsed from free text; for MLP aliases an index \
                 without a layer counts for every layer",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_unit_references() {
        let m = mentioned_units(
            "Layer 0, units h0–h9 score columns; l1.h3 gates; units 20-22 and h31. Ah5 no.",
        );
        for i in 0..=9 {
            assert!(m.contains(&(None, i)));
        }
        assert!(m.contains(&(Some(1), 3)));
        for i in 20..=22 {
            assert!(m.contains(&(None, i)));
        }
        assert!(m.contains(&(None, 31)));
        assert_eq!(m.len(), 10 + 1 + 3 + 1);
    }
}
