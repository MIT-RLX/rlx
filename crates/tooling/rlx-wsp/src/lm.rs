// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Language-model forward passes for package spaces, model-independently.
//!
//! The agent's edits (the overlay of changed tensors) are written as an
//! **adapter package** over the base `.rlxp` — each changed tensor as
//! `W + B·A` with one factor the identity on the smaller side, so the merge
//! reproduces the edit exactly — and opened through rlx-runtime's `Model`
//! API, which generates with whatever decoder the package carries. Nothing
//! here knows the architecture.
//!
//! The `qa` task: practice and held-out question / answer JSONL files
//! (`{"question": …, "answer": …}`, GSM8K's `#### <n>` final-answer form or a
//! bare answer). A forward pass answers a batch of questions greedily with a
//! plain prompt; the held-out score is accuracy on the held-out file.

use crate::space::{Tensor, WeightSpace};
use rlx_runtime::model::Model;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Questions / answers a task reads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaSpec {
    pub practice: String,
    pub heldout: String,
    /// Held-out questions scored (the first `heldout_n` of the file).
    pub heldout_n: usize,
    pub max_tokens: usize,
    /// rlx device the model runs on (`cpu`, `metal`, …).
    pub device: String,
    /// The last `dev_n` practice questions are held back from the tools:
    /// only `dev_check` (a few times) scores on them.
    #[serde(default)]
    pub dev_n: usize,
    /// General text (JSONL `{"text"}` or blank-line separated paragraphs)
    /// whose log-likelihood guards comparisons against edits that damage
    /// the model beyond the task.
    #[serde(default)]
    pub general: Option<String>,
    /// `plain` (default) or `chat` (the package's chat template, thinking
    /// off).
    #[serde(default = "plain")]
    pub prompt_style: String,
    /// Cut each completion after its first `#### n` line before scoring.
    #[serde(default)]
    pub stop_at_answer: bool,
    /// Practice questions (by index in the file) left out because they
    /// nearly repeat a held-out one.
    #[serde(default)]
    pub excluded_practice: Vec<usize>,
    /// Report checks: perturbed held-out questions, and answers at twice
    /// the token limit.
    #[serde(default)]
    pub check_perturbed: bool,
    #[serde(default)]
    pub check_truncation: bool,
    /// Repetition penalty of every generation (1 = none): loops cost an
    /// answer whichever version runs, so the task can damp them for all.
    #[serde(default = "no_penalty")]
    pub repetition_penalty: f32,
}

fn no_penalty() -> f32 {
    1.0
}

/// Practice items without the excluded (near-duplicate) ones.
fn practice_kept(q: &QaSpec) -> Result<Vec<QaItem>, String> {
    Ok(load_items(&q.practice)?
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !q.excluded_practice.contains(i))
        .map(|(_, it)| it)
        .collect())
}

fn plain() -> String {
    "plain".into()
}

/// Practice questions the tools see (the dev split held back).
pub fn practice_items(q: &QaSpec) -> Result<Vec<QaItem>, String> {
    let mut items = practice_kept(q)?;
    if q.dev_n >= items.len() {
        return Err(format!(
            "dev split of {} leaves no practice questions ({})",
            q.dev_n,
            items.len()
        ));
    }
    items.truncate(items.len() - q.dev_n);
    Ok(items)
}

/// The held-back dev questions.
pub fn dev_items(q: &QaSpec) -> Result<Vec<QaItem>, String> {
    let items = practice_kept(q)?;
    if q.dev_n == 0 {
        return Err("no dev split (make the space with --dev-n)".into());
    }
    Ok(items[items.len().saturating_sub(q.dev_n)..].to_vec())
}

/// The general texts: JSONL rows with `text`, else paragraphs.
pub fn general_texts(path: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let rows: Vec<String> = if text.trim_start().starts_with('{') {
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| {
                serde_json::from_str::<Value>(l).ok()?["text"]
                    .as_str()
                    .map(String::from)
            })
            .collect()
    } else {
        text.split("\n\n").map(|p| p.trim().to_string()).collect()
    };
    let rows: Vec<String> = rows.into_iter().filter(|r| r.len() >= 80).collect();
    if rows.is_empty() {
        return Err(format!("{path}: no texts of 80+ characters"));
    }
    Ok(rows)
}

/// `completion` up to the end of its first `#### n` line.
pub fn cut_at_answer(completion: &str) -> &str {
    for (i, _) in completion.match_indices("####") {
        let rest = &completion[i + 4..];
        let line_end = rest.find('\n').unwrap_or(rest.len());
        if !numbers(&rest[..line_end]).is_empty() {
            return &completion[..i + 4 + line_end];
        }
        // A marker alone on its line: the number may be on the next one.
        let after = rest[line_end..].trim_start_matches('\n');
        let next_end = after.find('\n').unwrap_or(after.len());
        if rest[..line_end].trim().is_empty() && !numbers(&after[..next_end]).is_empty() {
            let end = i + 4 + (rest.len() - after.len()) + next_end;
            return &completion[..end];
        }
    }
    completion
}

#[derive(Debug, Clone)]
pub struct QaItem {
    pub question: String,
    pub gold: String,
    /// The reference answer text (worked solution and `#### n`, or the bare
    /// answer), for teacher-forced scoring.
    pub solution: String,
}

pub fn load_items(path: &str) -> Result<Vec<QaItem>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Value = serde_json::from_str(l).map_err(|e| format!("{path}: {e}"))?;
            let q = v["question"].as_str().ok_or("row without `question`")?;
            let a = match &v["answer"] {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            Ok(QaItem {
                question: q.to_string(),
                gold: final_answer(&a).unwrap_or_else(|| a.clone()),
                solution: a,
            })
        })
        .collect()
}

/// The number on the line after the first `####` that has one, else the
/// last number in the text. (Text after the marker that is not a number —
/// a `<think>` block, a sentence — is not an answer.)
pub fn final_answer(text: &str) -> Option<String> {
    for (i, _) in text.match_indices("####") {
        let line = text[i + 4..].trim_start().lines().next().unwrap_or("");
        if let Some(n) = numbers(line).into_iter().next() {
            return Some(n);
        }
    }
    numbers(text).pop()
}

/// The numbers in `text`, in order, normalised.
fn numbers(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_digit()
            || (ch == '.' && !cur.is_empty())
            || (ch == '-' && cur.is_empty())
            || (ch == ',' && !cur.is_empty())
        {
            cur.push(ch);
        } else {
            let t = cur.trim_matches(|c| c == '.' || c == ',' || c == '-');
            if t.chars().any(|c| c.is_ascii_digit()) {
                out.push(normalise(&cur));
            }
            cur.clear();
        }
    }
    out
}

/// How a completion was scored, for telling format failures from wrong
/// arithmetic.
pub fn failure_kind(completion: &str, correct: bool) -> &'static str {
    if correct {
        "correct"
    } else if repeats(completion) {
        "loop"
    } else if completion.contains("<think>") && !completion.contains("</think>") {
        "unfinished_think"
    } else if !completion.contains("####") {
        "no_final_marker"
    } else {
        "wrong_answer"
    }
}

fn normalise(a: &str) -> String {
    let t: String = a
        .chars()
        .filter(|c| !matches!(c, ',' | '$' | '%' | ' '))
        .collect();
    let t = t.trim_end_matches('.');
    match t.parse::<f64>() {
        Ok(v) if v.fract() == 0.0 => format!("{}", v as i64),
        Ok(v) => format!("{v}"),
        Err(_) => t.to_lowercase(),
    }
}

/// The prompt for one question (plain text, no chat template).
pub fn prompt(question: &str) -> String {
    format!(
        "Question: {question}\nAnswer: Let's think step by step, then give the final answer \
         as a number after '####'.\n"
    )
}

/// The opened model and the overlay it was opened for.
#[derive(Default)]
pub struct LmHost {
    tokenizer: Option<Option<Box<dyn rlx_runtime::model::TextCodec>>>,
    model: Option<Box<dyn Model>>,
    opened_for: Option<u64>,
    adapter: Option<PathBuf>,
    adapter_for: Option<u64>,
    /// Token counts of the last batch of answers.
    pub(crate) last_tokens: Vec<usize>,
    /// The last batch of answers: (prompt, completion, correct).
    pub(crate) last_qa: Vec<(String, String, bool)>,
}

fn overlay_key(params: &BTreeMap<String, Tensor>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for (n, t) in params {
        n.hash(&mut h);
        for v in &t.data {
            v.to_bits().hash(&mut h);
        }
    }
    h.finish()
}

static TOKENIZER: std::sync::Once = std::sync::Once::new();

thread_local! {
    static PROGRESS: std::cell::RefCell<Option<Box<dyn Fn(usize, usize, &str)>>> =
        const { std::cell::RefCell::new(None) };
}

/// Report progress of long passes (MCP `notifications/progress`) while `f`
/// runs.
pub fn with_progress<T>(sink: Box<dyn Fn(usize, usize, &str)>, f: impl FnOnce() -> T) -> T {
    PROGRESS.with(|p| *p.borrow_mut() = Some(sink));
    let out = f();
    PROGRESS.with(|p| *p.borrow_mut() = None);
    out
}

fn progress(done: usize, total: usize, what: &str) {
    PROGRESS.with(|p| {
        if let Some(f) = p.borrow().as_ref() {
            f(done, total, what);
        }
    });
}

/// Prompts generated per chunk (progress is reported between chunks).
const CHUNK: usize = 8;

/// Whether a completion's first `#### n` line is complete: the score
/// reads that answer, so nothing generated after it can change it.
fn answered(text: &str) -> bool {
    cut_at_answer(text).len() < text.len()
}

fn generate_chunked(
    model: &mut dyn Model,
    refs: &[&str],
    opts: &Value,
    until_answer: bool,
) -> Result<Vec<String>, String> {
    let mut out = Vec::with_capacity(refs.len());
    if until_answer {
        for (i, p) in refs.iter().enumerate() {
            out.push(
                model
                    .generate_text_until(p, opts, &mut answered)
                    .map_err(|e| format!("generate: {e:#}"))?,
            );
            if (i + 1) % CHUNK == 0 || i + 1 == refs.len() {
                progress(i + 1, refs.len(), "generated");
            }
        }
        return Ok(out);
    }
    for chunk in refs.chunks(CHUNK) {
        out.extend(
            model
                .generate_batch(chunk, opts)
                .map_err(|e| format!("generate: {e:#}"))?,
        );
        progress(out.len(), refs.len(), "generated");
    }
    Ok(out)
}

impl WeightSpace {
    /// The package's tokenizer (a sidecar whose id names one), if any.
    pub(crate) fn tokenizer(&mut self) -> Option<&dyn rlx_runtime::model::TextCodec> {
        if self.lm.tokenizer.is_none() {
            TOKENIZER.call_once(rlx_text::install_model_tokenizer);
            let codec = self.store().and_then(|st| {
                let ids = st.sidecar_ids();
                let id = ids.iter().find(|i| i.contains("tokenizer"))?;
                let bytes = st.sidecar(id)?;
                // The chat template, when the package carries one.
                let chat = ids
                    .iter()
                    .find(|i| i.contains("chat_template"))
                    .and_then(|i| st.sidecar(i))
                    .and_then(|b| String::from_utf8(b).ok())
                    .map_or_else(|| json!({}), |t| json!({ "chat_template": t }));
                rlx_runtime::model::load_tokenizer(&bytes, &chat)
                    .or_else(|_| rlx_runtime::model::load_tokenizer(&bytes, &json!({})))
                    .ok()
            });
            self.lm.tokenizer = Some(codec);
        }
        self.lm.tokenizer.as_ref().and_then(|c| c.as_deref())
    }

    /// Tokens in `text`, by the package's tokenizer (else ~4 chars/token).
    pub fn count_tokens(&mut self, text: &str) -> usize {
        match self.tokenizer() {
            Some(c) => c
                .encode_plain(text)
                .map(|v| v.len())
                .unwrap_or(text.len() / 4),
            None => text.len() / 4,
        }
    }

    /// The prompt for one question in the task's style.
    pub fn qa_prompt(&mut self, question: &str) -> Result<String, String> {
        let style = self
            .state
            .qa
            .as_ref()
            .map_or_else(plain, |q| q.prompt_style.clone());
        match style.as_str() {
            "plain" => Ok(prompt(question)),
            "chat" => {
                let codec = self
                    .tokenizer()
                    .ok_or("no tokenizer for the chat template")?;
                let content = format!(
                    "{question}\nThink step by step, then give the final answer as a number after '####'."
                );
                codec
                    .apply_chat_template_json(
                        &[json!({ "role": "user", "content": content })],
                        &json!({ "enable_thinking": false, "add_generation_prompt": true }),
                    )
                    .map_err(|e| format!("the package's chat template: {e:#}"))
            }
            o => Err(format!("prompt style `{o}`: use plain | chat")),
        }
    }

    /// Refuse a language-model pass when the token budget is spent.
    pub(crate) fn check_tokens(&self) -> Result<(), String> {
        match self.state.token_budget {
            Some(b) if self.state.tokens_used >= b => Err(format!(
                "token budget exhausted ({} of {b} tokens processed)",
                self.state.tokens_used
            )),
            _ => Ok(()),
        }
    }

    /// Count tokens a pass processed (prompts and outputs).
    pub(crate) fn spend_tokens(&mut self, n: usize) {
        self.state.tokens_used += n;
    }

    /// The device language-model runs use.
    pub(crate) fn lm_device(&self) -> String {
        self.state
            .qa
            .as_ref()
            .map_or_else(|| "cpu".to_string(), |q| q.device.clone())
    }

    /// The package to open: the base, or an adapter over it carrying the
    /// overlay (written once per overlay state).
    pub fn model_path(&mut self) -> Result<PathBuf, String> {
        TOKENIZER.call_once(rlx_text::install_model_tokenizer);
        let base = self.store().ok_or("not a package space")?.path.clone();
        if self.state.params.is_empty() {
            return Ok(PathBuf::from(base));
        }
        let key = overlay_key(&self.state.params);
        if let Some(p) = &self.lm.adapter
            && self.lm.adapter_for == Some(key)
        {
            return Ok(p.clone());
        }
        let updates = self.adapter_updates()?;
        let dir = std::env::temp_dir().join(format!("rlx-wsp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let p = dir.join(format!("adapter-{key:016x}.rlxp"));
        let base_abs = std::path::absolute(&base).map_err(|e| e.to_string())?;
        rlx_runtime::model::rlxp::save_adapter(
            &p,
            "wsp-edits",
            &base_abs.display().to_string(),
            &updates,
        )
        .map_err(|e| format!("writing the adapter: {e:#}"))?;
        if let Some(old) = self.lm.adapter.replace(p.clone()) {
            let _ = std::fs::remove_file(old);
        }
        self.lm.adapter_for = Some(key);
        Ok(p)
    }

    /// The model with the current overlay applied, (re)opened when the
    /// overlay changed since the last open.
    fn lm_model(&mut self) -> Result<&mut Box<dyn Model>, String> {
        let key = overlay_key(&self.state.params);
        if self.lm.opened_for != Some(key) || self.lm.model.is_none() {
            self.lm.model = None;
            let path = self.model_path()?;
            let device = self.lm_device();
            let model = rlx_runtime::model::open_source(
                &path.display().to_string(),
                json!({ "device": device }),
            )
            .map_err(|e| format!("opening {}: {e:#}", path.display()))?;
            self.lm.model = Some(model);
            self.lm.opened_for = Some(key);
        }
        Ok(self.lm.model.as_mut().expect("opened"))
    }

    /// Each edited tensor as an exact low-rank update over the base:
    /// `W + B·A` with `B = I` (rows ≤ cols) or `A = I`.
    pub fn adapter_updates(&self) -> Result<Vec<rlx_runtime::model::rlxp::LoraUpdate>, String> {
        let store = self.store().ok_or("not a package space")?;
        let mut out = Vec::new();
        for (name, factors) in &self.state.lowrank {
            if !self.state.params.contains_key(name) || factors.is_empty() {
                continue;
            }
            let (m, n, r) = (factors[0].0.len(), factors[0].1.len(), factors.len());
            let mut b = vec![0f32; m * r];
            let mut a = vec![0f32; r * n];
            for (k, (u, v)) in factors.iter().enumerate() {
                for i in 0..m {
                    b[i * r + k] = u[i];
                }
                a[k * n..(k + 1) * n].copy_from_slice(v);
            }
            out.push(rlx_runtime::model::rlxp::LoraUpdate {
                tensor: name.clone(),
                rank: r,
                a,
                b,
                scale: 1.0,
            });
        }
        for (name, t) in &self.state.params {
            if self.state.lowrank.contains_key(name) {
                continue;
            }
            let base = store.get(name)?;
            let delta: Vec<f32> = t.data.iter().zip(&base.data).map(|(a, b)| a - b).collect();
            if delta.iter().all(|d| *d == 0.0) {
                continue;
            }
            let (m, n) = (t.rows, t.cols);
            let eye = |k: usize| {
                let mut e = vec![0f32; k * k];
                for i in 0..k {
                    e[i * k + i] = 1.0;
                }
                e
            };
            let (rank, b, a) = if m <= n {
                (m, eye(m), delta)
            } else {
                (n, delta, eye(n))
            };
            out.push(rlx_runtime::model::rlxp::LoraUpdate {
                tensor: name.clone(),
                rank,
                a,
                b,
                scale: 1.0,
            });
        }
        Ok(out)
    }

    /// Greedy completions of `prompts` (one forward pass for the caller).
    pub fn lm_generate(
        &mut self,
        prompts: &[String],
        max_tokens: usize,
    ) -> Result<Vec<String>, String> {
        self.lm_generate_masked(prompts, max_tokens, &[], false)
    }

    /// [`Self::lm_generate`] with channels zeroed. With recorded rewrites or
    /// masks the model is opened fresh inside a tap (it compiles there);
    /// otherwise the open model is reused. `until_answer`: each completion
    /// ends once its `#### n` line is complete.
    pub fn lm_generate_masked(
        &mut self,
        prompts: &[String],
        max_tokens: usize,
        masks: &[crate::graphops::Unit],
        until_answer: bool,
    ) -> Result<Vec<String>, String> {
        self.check_tokens()?;
        let outs = self.lm_generate_inner(prompts, max_tokens, masks, until_answer)?;
        let n: usize = prompts
            .iter()
            .chain(&outs)
            .map(|t| self.count_tokens(t))
            .sum();
        self.spend_tokens(n);
        Ok(outs)
    }

    fn lm_generate_inner(
        &mut self,
        prompts: &[String],
        max_tokens: usize,
        masks: &[crate::graphops::Unit],
        until_answer: bool,
    ) -> Result<Vec<String>, String> {
        let refs: Vec<&str> = prompts.iter().map(String::as_str).collect();
        let penalty = self.state.qa.as_ref().map_or(1.0, |q| q.repetition_penalty);
        let opts = json!({ "max_tokens": max_tokens, "temperature": 0.0,
                           "repetition_penalty": penalty });
        if self.state.rewrites.is_empty() && masks.is_empty() {
            let model = self.lm_model()?;
            return generate_chunked(&mut **model, &refs, &opts, until_answer);
        }
        let path = self.model_path()?;
        let device = self.lm_device();
        let tap = crate::pkgrun::rewrite_tap(self.state.rewrites.clone(), masks.to_vec());
        rlx_runtime::taps::with_tap(tap, || {
            let mut model = rlx_runtime::model::open_source(
                &path.display().to_string(),
                json!({ "device": device }),
            )
            .map_err(|e| format!("opening {}: {e:#}", path.display()))?;
            generate_chunked(&mut *model, &refs, &opts, until_answer)
        })
    }

    /// Answer questions; returns per-item (completion, predicted, correct).
    pub fn qa_answer(
        &mut self,
        items: &[QaItem],
    ) -> Result<Vec<(String, Option<String>, bool)>, String> {
        self.qa_answer_masked(items, &[])
    }

    /// [`Self::qa_answer`] with channels zeroed; token counts of the answers
    /// are left in `lm.last_tokens`.
    pub fn qa_answer_masked(
        &mut self,
        items: &[QaItem],
        masks: &[crate::graphops::Unit],
    ) -> Result<Vec<(String, Option<String>, bool)>, String> {
        let max = self.state.qa.as_ref().map_or(256, |q| q.max_tokens);
        let prompts: Vec<String> = items
            .iter()
            .map(|i| self.qa_prompt(&i.question))
            .collect::<Result<_, _>>()?;
        // Scored by its first `#### n`: generated only that far.
        let mut outs = self.lm_generate_masked(&prompts, max, masks, true)?;
        if self.state.qa.as_ref().is_some_and(|q| q.stop_at_answer) {
            for o in &mut outs {
                let cut = cut_at_answer(o).len();
                o.truncate(cut);
            }
        }
        self.lm.last_tokens = outs.iter().map(|o| self.count_tokens(o)).collect();
        let answers: Vec<(String, Option<String>, bool)> = outs
            .into_iter()
            .zip(items)
            .map(|(o, it)| {
                let p = final_answer(&o);
                let ok = p.as_deref() == Some(it.gold.as_str());
                (o, p, ok)
            })
            .collect();
        self.lm.last_qa = prompts
            .into_iter()
            .zip(&answers)
            .map(|(p, a)| (p, a.0.clone(), a.2))
            .collect();
        Ok(answers)
    }

    /// Teacher-forced scores: per item, the mean log-probability per token
    /// of the reference continuation and its token count. `mode`:
    /// `solution` (the whole worked solution after the prompt) or `answer`
    /// (only the `#### n` line, after the prompt and the reference
    /// working). Not budgeted (callers charge).
    pub fn qa_loglik(
        &mut self,
        items: &[QaItem],
        mode: &str,
        masks: &[crate::graphops::Unit],
    ) -> Result<Vec<(f64, usize)>, String> {
        let prompts: Vec<String> = items
            .iter()
            .map(|i| self.qa_prompt(&i.question))
            .collect::<Result<_, _>>()?;
        let mut pairs = Vec::with_capacity(items.len());
        {
            let codec = self
                .tokenizer()
                .ok_or("the package carries no tokenizer (a sidecar named *tokenizer*)")?;
            for (it, pr) in items.iter().zip(&prompts) {
                let (ctx, cont) = split_reference(it, mode, pr)?;
                let p = codec
                    .encode(&ctx)
                    .map_err(|e| format!("tokenizing: {e:#}"))?;
                let c = codec
                    .encode_plain(&cont)
                    .map_err(|e| format!("tokenizing: {e:#}"))?;
                if c.is_empty() {
                    return Err("an empty reference continuation".into());
                }
                pairs.push((p, c));
            }
        }
        self.score_token_pairs(&pairs, masks)
    }

    /// Mean log-probability per token of general texts (the first 8 tokens
    /// as context, the rest — up to 160 — scored).
    pub fn general_loglik(
        &mut self,
        texts: &[String],
        masks: &[crate::graphops::Unit],
    ) -> Result<Vec<(f64, usize)>, String> {
        let mut pairs = Vec::with_capacity(texts.len());
        {
            let codec = self.tokenizer().ok_or("the package carries no tokenizer")?;
            for t in texts {
                let ids = codec
                    .encode_plain(t)
                    .map_err(|e| format!("tokenizing: {e:#}"))?;
                if ids.len() < 16 {
                    continue;
                }
                let ids = &ids[..ids.len().min(168)];
                pairs.push((ids[..8].to_vec(), ids[8..].to_vec()));
            }
        }
        if pairs.is_empty() {
            return Err("no general text is 16+ tokens".into());
        }
        self.score_token_pairs(&pairs, masks)
    }

    /// Teacher-forced mean log-probability per token of each continuation
    /// after its context; counts the tokens against the token budget.
    fn score_token_pairs(
        &mut self,
        pairs: &[(Vec<u32>, Vec<u32>)],
        masks: &[crate::graphops::Unit],
    ) -> Result<Vec<(f64, usize)>, String> {
        self.check_tokens()?;
        let tokens: usize = pairs.iter().map(|(p, c)| p.len() + c.len()).sum();
        let out = self.score_token_pairs_inner(pairs, masks)?;
        self.spend_tokens(tokens);
        Ok(out)
    }

    fn score_token_pairs_inner(
        &mut self,
        pairs: &[(Vec<u32>, Vec<u32>)],
        masks: &[crate::graphops::Unit],
    ) -> Result<Vec<(f64, usize)>, String> {
        let score = |model: &mut Box<dyn Model>| -> Result<Vec<(f64, usize)>, String> {
            let runner = model
                .lm()
                .ok_or("the package is not a language model (no token decoder)")?;
            let mut out = Vec::with_capacity(pairs.len());
            for (k, (p, c)) in pairs.iter().enumerate() {
                let lp = runner
                    .score_continuation(p, c)
                    .map_err(|e| format!("scoring: {e:#}"))?;
                out.push((
                    lp.iter().map(|&x| x as f64).sum::<f64>() / lp.len() as f64,
                    lp.len(),
                ));
                progress(k + 1, pairs.len(), "scored");
            }
            Ok(out)
        };
        if self.state.rewrites.is_empty() && masks.is_empty() {
            let model = self.lm_model()?;
            return score(model);
        }
        let path = self.model_path()?;
        let device = self.lm_device();
        let tap = crate::pkgrun::rewrite_tap(self.state.rewrites.clone(), masks.to_vec());
        rlx_runtime::taps::with_tap(tap, || {
            let mut model = rlx_runtime::model::open_source(
                &path.display().to_string(),
                json!({ "device": device }),
            )
            .map_err(|e| format!("opening {}: {e:#}", path.display()))?;
            score(&mut model)
        })
    }
}

/// The context and the continuation teacher-forced scoring reads, for one
/// question.
pub fn split_reference(it: &QaItem, mode: &str, p: &str) -> Result<(String, String), String> {
    let p = p.to_string();
    let (working, _) = it
        .solution
        .rsplit_once("####")
        .unwrap_or(("", it.solution.as_str()));
    // GSM8K's calculator annotations (`<<48/2=24>>`) are not text a model
    // writes.
    let mut clean = String::with_capacity(working.len());
    let mut rest = working;
    while let Some(i) = rest.find("<<") {
        clean.push_str(&rest[..i]);
        match rest[i..].find(">>") {
            Some(j) => rest = &rest[i + j + 2..],
            None => {
                rest = &rest[i..];
                break;
            }
        }
    }
    clean.push_str(rest);
    let working = clean.trim_end();
    match mode {
        "solution" if working.is_empty() => Ok((p, format!("#### {}", it.gold))),
        "solution" => Ok((p, format!("{working}\n#### {}", it.gold))),
        "answer" if working.is_empty() => Ok((p, format!("#### {}", it.gold))),
        "answer" => Ok((format!("{p}{working}\n"), format!("#### {}", it.gold))),
        other => Err(format!("mode `{other}`: use solution | answer")),
    }
}

/// Whether a completion ends in a loop: the last 24 characters occur at
/// least 3 times in its second half.
pub fn repeats(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let tail = &chars[chars.len() / 2..];
    if tail.len() < 72 {
        return false;
    }
    let s: String = tail.iter().collect();
    let probe: String = tail[tail.len() - 24..].iter().collect();
    s.matches(probe.as_str()).count() >= 3
}

/// [`qa_summary`] plus token statistics: how often an answer hit the token
/// limit (the post's looping signal), ended in a loop, and its mean length.
pub fn qa_report(
    items: &[QaItem],
    answers: &[(String, Option<String>, bool)],
    tokens: &[usize],
    max_tokens: usize,
    examples: usize,
) -> Value {
    let mut v = qa_summary(items, answers, examples);
    let n = answers.len().max(1) as f64;
    if tokens.len() == answers.len() {
        v["hit_token_limit"] =
            json!(tokens.iter().filter(|t| **t + 1 >= max_tokens).count() as f64 / n);
        v["mean_tokens"] = json!(tokens.iter().sum::<usize>() as f64 / n);
    }
    v["ends_in_loop"] = json!(answers.iter().filter(|a| repeats(&a.0)).count() as f64 / n);
    v["max_tokens"] = json!(max_tokens);
    v
}

/// Summary of answered questions: accuracy, how many gave a final answer,
/// and up to `examples` failures with the end of their completion.
pub fn qa_summary(
    items: &[QaItem],
    answers: &[(String, Option<String>, bool)],
    examples: usize,
) -> Value {
    let n = answers.len().max(1) as f64;
    let correct = answers.iter().filter(|a| a.2).count();
    let with_marker = answers.iter().filter(|a| a.0.contains("####")).count();
    let fails: Vec<Value> = items
        .iter()
        .zip(answers)
        .filter(|(_, a)| !a.2)
        .take(examples)
        .map(|(it, (o, p, _))| {
            let tail: String = o.chars().rev().take(300).collect::<Vec<_>>().into_iter().rev().collect();
            json!({ "question": it.question, "gold": it.gold, "predicted": p, "completion_end": tail, "completion_chars": o.chars().count() })
        })
        .collect();
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    for a in answers {
        *kinds.entry(failure_kind(&a.0, a.2)).or_default() += 1;
    }
    json!({
        "questions": answers.len(),
        "accuracy": correct as f64 / n,
        "gave_final_marker": with_marker as f64 / n,
        "outcomes": kinds,
        "failures": fails,
    })
}

/// Where the unedited model's held-out result is cached: keyed by the
/// package (path, size, modification time), the question file, the count,
/// the token limit and the prompt.
pub fn heldout_cache_path(package: &str, q: &QaSpec) -> Option<PathBuf> {
    use std::hash::{Hash, Hasher};
    let meta = std::fs::metadata(package).ok()?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    package.hash(&mut h);
    meta.len().hash(&mut h);
    meta.modified().ok()?.hash(&mut h);
    q.heldout.hash(&mut h);
    q.heldout_n.hash(&mut h);
    q.max_tokens.hash(&mut h);
    prompt("").hash(&mut h);
    // The answer parser's version: a parser change re-measures (as does
    // generating only up to the answer, which changes the token counts).
    "parse-2-until-answer".hash(&mut h);
    q.prompt_style.hash(&mut h);
    q.stop_at_answer.hash(&mut h);
    if q.repetition_penalty != 1.0 {
        q.repetition_penalty.to_bits().hash(&mut h);
    }
    Some(
        crate::mnist::home()
            .join(".cache/rlx-wsp")
            .join(format!("heldout-{:016x}.json", h.finish())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_parse() {
        assert_eq!(final_answer("so 3+4 = 7\n#### 7").as_deref(), Some("7"));
        assert_eq!(final_answer("#### $1,250.00").as_deref(), Some("1250"));
        assert_eq!(
            final_answer("we get 12 apples, then 18.").as_deref(),
            Some("18")
        );
        assert_eq!(final_answer("no number here"), None);
        assert_eq!(final_answer("answer: -4.5").as_deref(), Some("-4.5"));
        // A marker followed by something else is not the answer.
        assert_eq!(
            final_answer("24 pages\n####\n\n<think>\nso 84").as_deref(),
            Some("84")
        );
        assert_eq!(
            final_answer("#### 120\n#### 120\n#### 7").as_deref(),
            Some("120")
        );
        assert_eq!(failure_kind("x <think> y", false), "unfinished_think");
        assert_eq!(
            cut_at_answer("so 7\n#### 7\nThe answer is 7.\n#### 7"),
            "so 7\n#### 7"
        );
        assert_eq!(cut_at_answer("####\n12\nmore"), "####\n12");
        assert_eq!(cut_at_answer("no marker"), "no marker");
    }

    #[test]
    fn dev_split_and_general_texts() {
        let dir = std::env::temp_dir().join(format!("rlx_wsp_lm_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("p.jsonl");
        let rows: String = (0..10)
            .map(|i| format!("{{\"question\": \"q{i}\", \"answer\": \"#### {i}\"}}\n"))
            .collect();
        std::fs::write(&p, rows).unwrap();
        let q = QaSpec {
            practice: p.display().to_string(),
            heldout: p.display().to_string(),
            heldout_n: 2,
            max_tokens: 8,
            device: "cpu".into(),
            dev_n: 3,
            general: None,
            prompt_style: plain(),
            stop_at_answer: false,
            excluded_practice: vec![0],
            check_perturbed: false,
            check_truncation: false,
            repetition_penalty: 1.0,
        };
        let seen = practice_items(&q).unwrap();
        let dev = dev_items(&q).unwrap();
        assert_eq!(seen.len(), 6, "one excluded, three held back");
        assert_eq!(seen[0].gold, "1");
        assert_eq!(
            dev.iter().map(|i| i.gold.as_str()).collect::<Vec<_>>(),
            ["7", "8", "9"]
        );
        // A repetition penalty re-measures the held-out questions; a state
        // written before it existed reads as none.
        let mut damped = q.clone();
        damped.repetition_penalty = 1.15;
        let pkg = p.display().to_string();
        assert_ne!(
            heldout_cache_path(&pkg, &q),
            heldout_cache_path(&pkg, &damped)
        );
        let mut old = serde_json::to_value(&q).unwrap();
        old.as_object_mut().unwrap().remove("repetition_penalty");
        let old: QaSpec = serde_json::from_value(old).unwrap();
        assert_eq!(heldout_cache_path(&pkg, &old), heldout_cache_path(&pkg, &q));
        let g = dir.join("g.txt");
        let long = "word ".repeat(30);
        std::fs::write(&g, format!("{long}\n\nshort\n\n{long}")).unwrap();
        assert_eq!(general_texts(&g.display().to_string()).unwrap().len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generation_stops_once_the_answer_line_is_complete() {
        // Not yet: the number may go on, or follow on the next line.
        for t in ["so 3+4 = 7\n#### 7", "#### ", "####\n", "#### so\n", "x\n"] {
            assert!(!answered(t), "{t:?}");
        }
        for t in [
            "#### 72\n",
            "#### 72\nThe answer is 72.",
            "####\n72\n",
            "#### $1,250\n",
        ] {
            assert!(answered(t), "{t:?}");
            // What is cut off cannot change the score.
            assert_eq!(final_answer(t), final_answer(&format!("{t}#### 9\n")));
        }
    }

    #[test]
    fn references_split_for_scoring() {
        let it = QaItem {
            question: "q".into(),
            gold: "72".into(),
            solution: "48/2 = <<48/2=24>>24\n48+24 = <<48+24=72>>72\n#### 72".into(),
        };
        let (ctx, cont) = split_reference(&it, "solution", &prompt("q")).unwrap();
        assert_eq!(ctx, prompt("q"));
        assert_eq!(cont, "48/2 = 24\n48+24 = 72\n#### 72");
        let (ctx, cont) = split_reference(&it, "answer", &prompt("q")).unwrap();
        assert!(ctx.ends_with("48+24 = 72\n"));
        assert_eq!(cont, "#### 72");
    }
}
