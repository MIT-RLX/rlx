// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checks on QA results that an edit could otherwise pass without the model
//! getting better at the task.
//!
//! - **Perturbed questions** ([`perturb`]): the question's numbers changed
//!   and the answer recomputed from the reference working's calculator
//!   steps (`<<48/2=24>>`, GSM8K's form). An edit that helps the original
//!   questions but not their perturbed twins has learned answers, not
//!   arithmetic. Items whose steps cannot be recomputed unambiguously are
//!   left out.
//! - **Near-duplicates** ([`near_duplicates`]): practice questions that
//!   nearly repeat a held-out one (word 3-gram Jaccard) are dropped from
//!   what the tools see.
//! - **Edit size** ([`edit_size`]): how far each changed tensor moved from
//!   the package's, relative to its norm — reported, and capped with
//!   `--max-edit` (see [`crate::space::WeightSpace::edit`]).

use crate::lm::QaItem;
use crate::space::WeightSpace;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

// ── Arithmetic ──────────────────────────────────────────────────────────

/// Evaluate `+ - * /` and parentheses over decimals.
pub fn eval(expr: &str) -> Option<f64> {
    let toks: Vec<char> = expr.chars().filter(|c| !c.is_whitespace()).collect();
    let mut p = Parser { t: &toks, i: 0 };
    let v = p.sum()?;
    (p.i == toks.len() && v.is_finite()).then_some(v)
}

struct Parser<'a> {
    t: &'a [char],
    i: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.t.get(self.i).copied()
    }
    fn sum(&mut self) -> Option<f64> {
        let mut v = self.product()?;
        while let Some(c @ ('+' | '-')) = self.peek() {
            self.i += 1;
            let r = self.product()?;
            v = if c == '+' { v + r } else { v - r };
        }
        Some(v)
    }
    fn product(&mut self) -> Option<f64> {
        let mut v = self.atom()?;
        while let Some(c @ ('*' | '/' | 'x' | '×')) = self.peek() {
            self.i += 1;
            let r = self.atom()?;
            v = if c == '/' { v / r } else { v * r };
        }
        Some(v)
    }
    fn atom(&mut self) -> Option<f64> {
        match self.peek()? {
            '(' => {
                self.i += 1;
                let v = self.sum()?;
                (self.peek()? == ')').then(|| self.i += 1)?;
                Some(v)
            }
            '-' => {
                self.i += 1;
                Some(-self.atom()?)
            }
            _ => {
                let start = self.i;
                while self
                    .peek()
                    .is_some_and(|c| c.is_ascii_digit() || c == '.' || c == ',')
                {
                    self.i += 1;
                }
                let s: String = self.t[start..self.i]
                    .iter()
                    .filter(|c| **c != ',')
                    .collect();
                s.parse().ok()
            }
        }
    }
}

/// Numbers in `text` as (byte start, byte end, value); `1,250` is one.
fn number_spans(text: &str) -> Vec<(usize, usize, f64)> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit()
            && (i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'.'))
        {
            let s = i;
            while i < b.len()
                && (b[i].is_ascii_digit()
                    || ((b[i] == b'.' || b[i] == b',')
                        && i + 1 < b.len()
                        && b[i + 1].is_ascii_digit()))
            {
                i += 1;
            }
            let raw: String = text[s..i].chars().filter(|c| *c != ',').collect();
            if let Ok(v) = raw.parse() {
                out.push((s, i, v));
            }
        } else {
            i += 1;
        }
    }
    out
}

fn fmt(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        let s = format!("{v:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn clean(v: f64) -> bool {
    v.is_finite() && v >= 0.0 && ((v * 100.0) - (v * 100.0).round()).abs() < 1e-6
}

/// A small deterministic generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// `item` with its question's numbers changed and the answer recomputed,
/// or `None` when that cannot be done unambiguously.
pub fn perturb(item: &QaItem, seed: u64) -> Option<QaItem> {
    let qnums = number_spans(&item.question);
    if qnums.is_empty() || !item.solution.contains("<<") {
        return None;
    }
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    // The literals the calculator steps use: a question number is changed
    // only when the steps use it as written (150% read as 1.5 is not), and
    // never when it is also a common constant (half = /2, a week, an hour,
    // a percent), which the steps cannot tell apart.
    const CONSTANTS: [f64; 10] = [2.0, 7.0, 10.0, 12.0, 24.0, 52.0, 60.0, 100.0, 365.0, 1000.0];
    let mut used: BTreeSet<String> = BTreeSet::new();
    let mut scan = item.solution.as_str();
    while let Some(a) = scan.find("<<") {
        let b = scan[a..].find(">>").map_or(scan.len(), |b| b + a);
        if let Some((expr, _)) = scan[a + 2..b].rsplit_once('=') {
            used.extend(number_spans(expr).into_iter().map(|(_, _, v)| fmt(v)));
        }
        scan = &scan[(b + 2).min(scan.len())..];
    }
    // New values for the question's whole numbers ≥ 3 (others kept).
    let mut map: BTreeMap<String, f64> = BTreeMap::new();
    for &(_, _, v) in &qnums {
        let key = fmt(v);
        if map.contains_key(&key)
            || v < 3.0
            || v.fract() != 0.0
            || v > 1e6
            || !used.contains(&key)
            || CONSTANTS.contains(&v)
        {
            continue;
        }
        let nv = if v < 20.0 {
            let d = [-2.0, -1.0, 1.0, 2.0, 3.0][(rng.next() % 5) as usize];
            (v + d).max(3.0)
        } else {
            let f = [0.8, 0.9, 1.1, 1.2, 1.25][(rng.next() % 5) as usize];
            (v * f).round().max(2.0)
        };
        if nv != v {
            map.insert(key, nv);
        }
    }
    if map.is_empty() {
        return None;
    }
    // Recompute the steps in order; a literal is a question number or an
    // earlier step's result (ambiguous → give up), else a constant.
    let mut results: BTreeMap<String, f64> = BTreeMap::new();
    let mut last_old = None;
    let mut last_new = None;
    let mut rest = item.solution.as_str();
    while let Some(a) = rest.find("<<") {
        let b = rest[a..].find(">>")? + a;
        let step = &rest[a + 2..b];
        rest = &rest[b + 2..];
        let (expr, res) = step.rsplit_once('=')?;
        let old_res = eval(res)?;
        if (eval(expr)? - old_res).abs() > 1e-6 * old_res.abs().max(1.0) {
            return None;
        }
        let mut new_expr = String::new();
        let mut at = 0;
        for (s, e, v) in number_spans(expr) {
            let key = fmt(v);
            let nv = match (map.get(&key), results.get(&key)) {
                (Some(a), Some(b)) if a != b => return None,
                (Some(a), _) => *a,
                (None, Some(b)) => *b,
                (None, None) => v,
            };
            new_expr.push_str(&expr[at..s]);
            new_expr.push_str(&fmt(nv));
            at = e;
        }
        new_expr.push_str(&expr[at..]);
        let new_res = eval(&new_expr)?;
        if !clean(new_res) {
            return None;
        }
        let key = fmt(old_res);
        if results
            .get(&key)
            .is_some_and(|r| (r - new_res).abs() > 1e-9)
            || map.contains_key(&key)
        {
            return None;
        }
        results.insert(key, new_res);
        last_old = Some(old_res);
        last_new = Some(new_res);
    }
    let (lo, ln) = (last_old?, last_new?);
    if fmt(lo) != item.gold || fmt(ln) == item.gold {
        return None;
    }
    let mut q = String::new();
    let mut at = 0;
    for (s, e, v) in qnums {
        q.push_str(&item.question[at..s]);
        q.push_str(
            &map.get(&fmt(v))
                .map_or_else(|| item.question[s..e].to_string(), |n| fmt(*n)),
        );
        at = e;
    }
    q.push_str(&item.question[at..]);
    Some(QaItem {
        question: q,
        gold: fmt(ln),
        solution: format!("#### {}", fmt(ln)),
    })
}

/// The perturbable items of `items` with their twins: `(index, twin)`.
pub fn perturbed_set(items: &[QaItem], seed: u64) -> Vec<(usize, QaItem)> {
    items
        .iter()
        .enumerate()
        .filter_map(|(i, it)| perturb(it, seed ^ i as u64).map(|p| (i, p)))
        .collect()
}

// ── Near-duplicates ─────────────────────────────────────────────────────

fn shingles(text: &str) -> BTreeSet<String> {
    let words: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect();
    words.windows(3).map(|w| w.join(" ")).collect()
}

/// Pairs `(practice index, held-out index, Jaccard)` at or above
/// `threshold`.
pub fn near_duplicates(
    practice: &[QaItem],
    heldout: &[QaItem],
    threshold: f64,
) -> Vec<(usize, usize, f64)> {
    let h: Vec<BTreeSet<String>> = heldout.iter().map(|i| shingles(&i.question)).collect();
    let mut out = Vec::new();
    for (pi, p) in practice.iter().enumerate() {
        let ps = shingles(&p.question);
        if ps.is_empty() {
            continue;
        }
        for (hi, hs) in h.iter().enumerate() {
            let inter = ps.intersection(hs).count() as f64;
            let jac = inter / (ps.len() as f64 + hs.len() as f64 - inter);
            if jac >= threshold {
                out.push((pi, hi, jac));
            }
        }
    }
    out
}

// ── Edit size ───────────────────────────────────────────────────────────

/// Per changed tensor: `‖W − W₀‖ / ‖W₀‖` against the package's (package
/// spaces), and the total over them.
pub fn edit_size(ws: &WeightSpace) -> Value {
    let Some(store) = ws.store() else {
        return Value::Null;
    };
    let (mut d2, mut w2) = (0f64, 0f64);
    let mut rows = Vec::new();
    for (name, t) in &ws.state.params {
        let Ok(base) = store.get(name) else { continue };
        let dd: f64 = t
            .data
            .iter()
            .zip(&base.data)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum();
        let bb: f64 = base.data.iter().map(|b| (*b as f64).powi(2)).sum();
        d2 += dd;
        w2 += bb;
        rows.push(json!({ "tensor": name, "relative_change": (dd / bb.max(1e-30)).sqrt() }));
    }
    rows.sort_by(|a, b| {
        b["relative_change"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&a["relative_change"].as_f64().unwrap_or(0.0))
    });
    json!({
        "tensors_changed": rows.len(),
        "relative_change_total": if w2 > 0.0 { (d2 / w2).sqrt() } else { 0.0 },
        "largest": rows.into_iter().take(10).collect::<Vec<_>>(),
        "rewrites": ws.state.rewrites.len(),
        "cap": ws.state.max_edit,
    })
}

// ── Report checks ───────────────────────────────────────────────────────

/// Accuracy of the live network on `items` (harness; not budgeted).
fn accuracy(ws: &mut WeightSpace, items: &[QaItem]) -> Result<f64, String> {
    let a = ws.qa_answer(items)?;
    Ok(a.iter().filter(|x| x.2).count() as f64 / a.len().max(1) as f64)
}

/// `f` on the unedited package (no overlay, no rewrites), cached on disk
/// under `kind` beside the held-out cache.
fn unedited<T: serde::Serialize + serde::de::DeserializeOwned>(
    ws: &mut WeightSpace,
    kind: &str,
    f: impl FnOnce(&mut WeightSpace) -> Result<T, String>,
) -> Result<T, String> {
    let q = ws.state.qa.clone().ok_or("not a question-answering task")?;
    let cache = crate::lm::heldout_cache_path(ws.state.package.as_deref().unwrap_or(""), &q)
        .map(|p| p.with_extension(format!("{kind}.json")));
    if let Some(c) = &cache
        && let Ok(t) = std::fs::read_to_string(c)
        && let Ok(v) = serde_json::from_str(&t)
    {
        return Ok(v);
    }
    let live = ws.current_variant();
    ws.swap_variant(crate::space::Variant::default());
    let r = f(ws);
    ws.swap_variant(live);
    let r = r?;
    if let Some(c) = cache {
        let _ = std::fs::create_dir_all(c.parent().unwrap_or(std::path::Path::new(".")));
        let _ = std::fs::write(c, serde_json::to_string(&r).unwrap_or_default());
    }
    Ok(r)
}

/// The unedited model's held-out score (cached; package QA spaces).
pub fn unedited_held_out(ws: &mut WeightSpace) -> Result<Value, String> {
    let live = ws.current_variant();
    ws.swap_variant(crate::space::Variant::default());
    let r = ws.held_out(0);
    ws.swap_variant(live);
    r
}

/// The checks the task asks for (`check_perturbed`, `check_truncation`),
/// each for the live network and the unedited one.
pub fn report_checks(ws: &mut WeightSpace) -> Result<Value, String> {
    if ws.state.qa.is_none() || ws.store().is_none() {
        return Ok(Value::Null);
    }
    let mut out = json!({});
    for name in CHECKS {
        let v = one_check(ws, name)?;
        if !v.is_null() {
            out[name] = v;
        }
    }
    Ok(out)
}

/// The report checks, by name.
pub const CHECKS: [&str; 2] = ["perturbed", "truncation"];

/// One of [`CHECKS`] (`null` when the task does not ask for it).
pub fn one_check(ws: &mut WeightSpace, name: &str) -> Result<Value, String> {
    let Some(q) = ws.state.qa.clone() else {
        return Ok(Value::Null);
    };
    if ws.store().is_none() {
        return Ok(Value::Null);
    }
    let mut out = Value::Null;
    let held = crate::lm::load_items(&q.heldout)?;
    let held = &held[..q.heldout_n.min(held.len())];
    if name == "perturbed" && q.check_perturbed {
        let set = perturbed_set(held, 7);
        let orig: Vec<QaItem> = set.iter().map(|(i, _)| held[*i].clone()).collect();
        let twins: Vec<QaItem> = set.into_iter().map(|(_, p)| p).collect();
        if twins.is_empty() {
            out = json!("no held-out question could be perturbed (no calculator steps)");
        } else {
            let (o1, t1) = (accuracy(ws, &orig)?, accuracy(ws, &twins)?);
            let (o0, t0): (f64, f64) = unedited(ws, "perturbed", |ws| {
                Ok((accuracy(ws, &orig)?, accuracy(ws, &twins)?))
            })?;
            let (g_o, g_t) = (o1 - o0, t1 - t0);
            out = json!({
                "questions": twins.len(),
                "original": { "edited": o1, "unedited": o0, "gain": g_o },
                "perturbed": { "edited": t1, "unedited": t0, "gain": g_t },
                "gain_lost_on_perturbed": g_o - g_t,
                "note": "a gain on the originals that disappears on the twins (same questions, other numbers) points to learned answers, not arithmetic",
            });
        }
    }
    if name == "truncation" && q.check_truncation {
        let long = q.max_tokens * 2;
        let with_long = |ws: &mut WeightSpace| -> Result<f64, String> {
            let keep = ws.state.qa.as_ref().map(|q| q.max_tokens);
            if let Some(q) = ws.state.qa.as_mut() {
                q.max_tokens = long;
            }
            let r = accuracy(ws, held);
            if let (Some(q), Some(k)) = (ws.state.qa.as_mut(), keep) {
                q.max_tokens = k;
            }
            r
        };
        let e = with_long(ws)?;
        let u: f64 = unedited(ws, &format!("long{long}"), |ws| with_long(ws))?;
        out = json!({
            "max_tokens": long, "edited": e, "unedited": u, "gain": e - u,
            "note": "the gain when answers may run twice as long: a gain that only exists at the normal limit came from answers fitting it",
        });
    }
    Ok(out)
}

// ── Precision ───────────────────────────────────────────────────────────

/// `x` rounded to bfloat16 (nearest even), back as f32.
pub fn round_bf16(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let b = x.to_bits();
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(r)
}

/// `x` rounded to IEEE half (nearest), back as f32.
pub fn round_f16(x: f32) -> f32 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let e = x.abs().log2().floor().clamp(-14.0, 15.0);
    let step = 2f32.powf(e - 10.0);
    (x / step).round_ties_even() * step
}

/// Per edited tensor (package spaces): how much of the edit survives when
/// the weights run in `dtype` (`bf16` | `f16`) — `‖q(W₀+Δ) − q(W₀) − Δ‖ /
/// ‖Δ‖`, the share of the change rounding takes away.
pub fn precision(ws: &WeightSpace, dtype: &str) -> Result<Value, String> {
    let q: fn(f32) -> f32 = match dtype {
        "bf16" => round_bf16,
        "f16" => round_f16,
        o => return Err(format!("dtype `{o}`: bf16 | f16")),
    };
    let store = ws
        .store()
        .ok_or("precision compares edits with the package's tensors")?;
    let mut rows = Vec::new();
    for (name, t) in &ws.state.params {
        let Ok(base) = store.get(name) else { continue };
        let (mut lost, mut d2) = (0f64, 0f64);
        for (a, b) in t.data.iter().zip(&base.data) {
            let d = (a - b) as f64;
            let kept = (q(*a) - q(*b)) as f64;
            lost += (kept - d).powi(2);
            d2 += d * d;
        }
        if d2 == 0.0 {
            continue;
        }
        let share = (lost / d2).sqrt();
        rows.push(json!({ "tensor": name, "lost_share": share,
                          "warning": (share > 0.25).then_some("much of this edit is below the dtype's resolution") }));
    }
    rows.sort_by(|a, b| {
        b["lost_share"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&a["lost_share"].as_f64().unwrap_or(0.0))
    });
    Ok(
        json!({ "dtype": dtype, "stored_as": ws.store().map(|s| s.names().filter_map(|n| s.entry(n).map(|e| e.1.clone())).collect::<BTreeSet<_>>()), "tensors": rows }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding() {
        assert_eq!(round_bf16(1.0), 1.0);
        assert_eq!(
            round_bf16(1.0 + 1.0 / 512.0),
            1.0,
            "below bf16's 2^-7 step at 1"
        );
        assert_eq!(round_bf16(1.0 + 1.0 / 64.0), 1.0 + 1.0 / 64.0);
        assert_eq!(
            round_f16(1.0 + 1.0 / 4096.0),
            1.0,
            "below f16's 2^-10 step at 1"
        );
        assert_eq!(round_f16(1.0 + 1.0 / 512.0), 1.0 + 1.0 / 512.0);
    }

    #[test]
    fn arithmetic() {
        assert_eq!(eval("48/2"), Some(24.0));
        assert_eq!(eval("(3+4)*2-1"), Some(13.0));
        assert_eq!(eval("1,250*2"), Some(2500.0));
        assert_eq!(eval("2*"), None);
    }

    #[test]
    fn perturbation_recomputes_the_answer() {
        let it = QaItem {
            question: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?".into(),
            gold: "72".into(),
            solution: "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72".into(),
        };
        let p = (0..20).find_map(|s| perturb(&it, s)).expect("perturbable");
        let n: f64 = number_spans(&p.question)[0].2;
        assert!(!p.question.contains("48"));
        assert_eq!(p.gold, fmt(n + n / 2.0), "{p:?}");
        // No calculator steps: not perturbable.
        let bare = QaItem {
            solution: "#### 72".into(),
            ..it
        };
        assert!(perturb(&bare, 1).is_none());
        // A number the steps also use as a constant (half = /2) is kept,
        // and one they use transformed (150% as 1.5) is kept too.
        let robe = QaItem {
            question: "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?".into(),
            gold: "3".into(),
            solution: "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total is 2+1=<<2+1=3>>3 bolts\n#### 3".into(),
        };
        assert!((0..20).all(|s| perturb(&robe, s).is_none()));
        let house = QaItem {
            question: "He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?".into(),
            gold: "70000".into(),
            solution: "Cost 80,000+50,000=$<<80000+50000=130000>>130,000\nIncrease 80,000*1.5=<<80000*1.5=120000>>120,000\nValue 120,000+80,000=$<<120000+80000=200000>>200,000\nProfit 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000".into(),
        };
        let t = (0..20)
            .find_map(|s| perturb(&house, s))
            .expect("perturbable");
        assert!(t.question.contains("150%"), "{}", t.question);
        let n = number_spans(&t.question);
        let (buy, repair) = (n[0].2, n[1].2);
        assert_eq!(t.gold, fmt(buy * 2.5 - (buy + repair)), "{t:?}");
    }

    #[test]
    fn near_duplicates_are_found() {
        let q = |s: &str| QaItem {
            question: s.into(),
            gold: "1".into(),
            solution: "#### 1".into(),
        };
        let p = [
            q("Tom has 3 apples and buys 4 more apples at the store today"),
            q("A train leaves the station at noon"),
        ];
        let h = [q(
            "Tom has 5 apples and buys 4 more apples at the store today",
        )];
        let d = near_duplicates(&p, &h, 0.5);
        assert_eq!(d.len(), 1);
        assert_eq!((d[0].0, d[0].1), (0, 0));
    }
}
