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
pub fn unit_importance_of(
    ws: &mut WeightSpace,
    episodes: usize,
    only: Option<&BTreeSet<String>>,
) -> Result<Vec<(String, f32)>, String> {
    if ws.state.graph.is_some() {
        return Err("unit importance addresses MLP units".into());
    }
    let live = ws.state.params.clone();
    let base = score(ws, episodes)?;
    let mut out = Vec::new();
    for k in 0..ws.state.mlp.n_layers() - 1 {
        let w = &live[&format!("l{k}.w")];
        let b = &live[&format!("l{k}.b")];
        for i in 0..w.cols {
            let used = b.data[i] != 0.0 || (0..w.rows).any(|r| w.at(r, i) != 0.0);
            if !used {
                continue;
            }
            let unit = format!("l{k}.h{i}");
            if only.is_some_and(|o| !o.contains(&unit)) {
                continue;
            }
            let off = ws.ablated(std::slice::from_ref(&unit))?;
            ws.set_params(off);
            let s = score(ws, episodes)?;
            out.push((unit, s - base));
        }
    }
    ws.set_params(live);
    out.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    Ok(out)
}

/// Hidden units whose incoming column, bias or outgoing row differs between
/// `before` and `after` — the units an agent's edits touched.
pub fn edited_units(
    widths: &[usize],
    before: &BTreeMap<String, Tensor>,
    after: &BTreeMap<String, Tensor>,
) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let layers = widths.len() - 1;
    for k in 0..layers - 1 {
        let (wi, bi, wo) = (
            format!("l{k}.w"),
            format!("l{k}.b"),
            format!("l{}.w", k + 1),
        );
        for i in 0..widths[k + 1] {
            let col = (0..widths[k]).any(|r| before[&wi].at(r, i) != after[&wi].at(r, i));
            let bias = before[&bi].data[i] != after[&bi].data[i];
            let row = (0..widths[k + 2]).any(|c| before[&wo].at(i, c) != after[&wo].at(i, c));
            if col || bias || row {
                out.insert(format!("l{k}.h{i}"));
            }
        }
    }
    out
}

/// The post-run check. Networks written from zero: every used unit's
/// ablation effect against the units the agent's text names
/// ([`claim_check`]). Repair tasks: what the agent changed and what each
/// change is worth ([`edit_check`]) — a whole trained unit reads digits, so
/// switching it off says nothing about the repair.
pub fn check_run(ws: &mut WeightSpace, text: &str, episodes: usize) -> Result<Value, String> {
    if ws.state.env.is_reading() {
        return Ok(json!({ "scope": "reading task: scored by the answer, no edits checked" }));
    }
    if ws.state.env == crate::env::EnvKind::FixPolicy {
        let before = crate::tasks::fix_policy_params()?;
        return edit_check(ws, &before, episodes);
    }
    if ws.state.env.is_classifier() {
        let before = crate::mnist::trained(ws.state.env.trigger())?;
        return edit_check(ws, &before, episodes);
    }
    let imp = unit_importance(ws, episodes)?;
    let mut r = claim_check(text, &imp, 0.5);
    r["scope"] = json!("every hidden unit with any weight");
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
        let old = &before[name];
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
            "changed_rows": rows.len(),
            "rows": rows.iter().take(32).collect::<Vec<_>>(),
            "changed_cols": cols.len(),
            "cols": cols.iter().take(32).collect::<Vec<_>>(),
            "score_with_this_reverted": undone["score"],
            "detail_with_this_reverted": undone["detail"],
        }));
    }
    ws.set_params(before.clone());
    let original = ws.held_out(episodes)?;
    ws.set_params(live);
    Ok(json!({
        "scope": "edits relative to the model as handed over",
        "score_now": now["score"],
        "detail_now": now["detail"],
        "score_all_edits_undone": original["score"],
        "detail_all_edits_undone": original["detail"],
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

/// Compare measured unit importance with the units `text` names.
/// `threshold` is the score change that counts as "matters".
pub fn claim_check(text: &str, importance: &[(String, f32)], threshold: f32) -> Value {
    let mentioned = mentioned_units(text);
    let parse = |u: &str| -> (usize, usize) {
        let (k, i) = u[1..].split_once(".h").expect("unit name");
        (k.parse().expect("layer"), i.parse().expect("index"))
    };
    let named =
        |k: usize, i: usize| mentioned.contains(&(Some(k), i)) || mentioned.contains(&(None, i));
    let important: Vec<&(String, f32)> = importance
        .iter()
        .filter(|(_, d)| d.abs() >= threshold)
        .collect();
    let unmentioned: Vec<Value> = important
        .iter()
        .filter(|(u, _)| {
            let (k, i) = parse(u);
            !named(k, i)
        })
        .map(|(u, d)| json!([u, d]))
        .collect();
    let inert: Vec<Value> = importance
        .iter()
        .filter(|(u, d)| {
            let (k, i) = parse(u);
            d.abs() < threshold / 5.0 && mentioned.contains(&(Some(k), i))
        })
        .map(|(u, d)| json!([u, d]))
        .collect();
    json!({
        "threshold": threshold,
        "units_measured": importance.len(),
        "important_units": important.iter().map(|(u, d)| json!([u, d])).collect::<Vec<_>>(),
        "important_but_not_mentioned": unmentioned,
        "mentioned_with_layer_but_inert": inert,
        "mentions_parsed": mentioned.len(),
        "note": "heuristic: mentions are parsed from free text; an index without a layer \
                 counts for every layer",
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
