// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Planning which tools a request needs, from the request text.
//!
//! Small models lose the plan on multi-part requests. Measured with
//! Qwen3.5-0.8B: "check my calendar and tell me the time" called only the
//! clock, "what's my focus level, and what day is it?" only the EEG status, and
//! "search for news about my city" searched for the literal words "my city".
//! A host that knows its tools can name the phrases that call for each one;
//! this module turns those phrases into a plan, writes it into the prompt, and
//! checks each round of calls against it:
//!
//! - [`plan`] matches the request against [`ToolIntent`]s.
//! - [`ToolPlan::hint`] is a short prompt block listing the steps.
//! - [`ToolPlan::review_round`] takes the calls the model emitted in one round
//!   and returns the calls to add (planned steps it skipped) and the calls to
//!   hold back (steps whose inputs come from a tool that has not run yet).
//!
//! All host-side string work: no model or runtime coupling.

use serde_json::{Map, Value};

/// Words that, just before a trigger, turn it off: "don't check my calendar".
const NEGATORS: &[&str] = &[
    "don't",
    "dont",
    "do not",
    "no need to",
    "without",
    "never",
    "not",
    "skip",
];
/// How many words before a trigger a negator may sit.
const NEGATION_WINDOW: usize = 4;

/// One tool the host can plan for, and the phrases that call for it.
#[derive(Debug, Clone)]
pub struct ToolIntent {
    /// Tool name, as the model calls it.
    pub tool: String,
    /// Arguments the step needs. A call covers the step when it carries every
    /// key given here with an equal value; extra keys are fine. `None` (or an
    /// empty object) means any call of `tool` covers it.
    pub arguments: Option<Value>,
    /// Lowercase phrases; the intent fires when one appears in the request as
    /// whole words ("time" does not match "sometimes"). `#n` stands for any
    /// number, and `+`, `*`, `×`, `%` read as "plus", "times", "times",
    /// "percent", so `#n percent of #n` matches "25% of 200". A trigger
    /// preceded by a negator ("don't", "without", …) does not count.
    pub triggers: Vec<String>,
    /// Phrases that cancel the intent anywhere in the request: the local
    /// weather intent is off for "I'm in Paris, what's the weather here?".
    pub unless: Vec<String>,
    /// Tools whose result this step needs first. A call made before they have
    /// run is held back, and the missing tool is planned in its place.
    pub requires: Vec<String>,
    /// What the step is for, shown in the prompt hint ("the time").
    pub purpose: String,
}

impl ToolIntent {
    pub fn new(tool: impl Into<String>, purpose: impl Into<String>, triggers: &[&str]) -> Self {
        Self {
            tool: tool.into(),
            arguments: None,
            triggers: triggers.iter().map(|t| t.to_lowercase()).collect(),
            unless: Vec::new(),
            requires: Vec::new(),
            purpose: purpose.into(),
        }
    }

    pub fn with_arguments(mut self, arguments: Value) -> Self {
        self.arguments = Some(arguments);
        self
    }

    pub fn after(mut self, tool: impl Into<String>) -> Self {
        self.requires.push(tool.into());
        self
    }

    pub fn unless(mut self, phrases: &[&str]) -> Self {
        self.unless.extend(phrases.iter().map(|p| p.to_lowercase()));
        self
    }
}

/// A part of a request the model should answer from what it knows, and the
/// tools it should not reach for there ("translate …" needs no web search).
#[derive(Debug, Clone)]
pub struct DirectAnswer {
    /// What kind of question it is, for the note to the model ("a translation").
    pub purpose: String,
    /// Same matching rules as [`ToolIntent::triggers`].
    pub triggers: Vec<String>,
    /// Phrases that mean a tool is wanted after all ("latest", "search").
    pub unless: Vec<String>,
    /// Tools a call to which is held back for this request.
    pub skip: Vec<String>,
}

impl DirectAnswer {
    pub fn new(purpose: impl Into<String>, triggers: &[&str], skip: &[&str]) -> Self {
        Self {
            purpose: purpose.into(),
            triggers: triggers.iter().map(|t| t.to_lowercase()).collect(),
            unless: Vec::new(),
            skip: skip.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn unless(mut self, phrases: &[&str]) -> Self {
        self.unless.extend(phrases.iter().map(|p| p.to_lowercase()));
        self
    }
}

/// One step of a [`ToolPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedStep {
    pub tool: String,
    /// Arguments to send when the host adds this call itself; always an object.
    pub arguments: Value,
    pub requires: Vec<String>,
    pub purpose: String,
    /// The request phrase that triggered the step.
    pub matched: String,
}

impl PlannedStep {
    /// Whether a call of `tool` with `arguments` covers this step.
    pub fn covered_by(&self, tool: &str, arguments: &Value) -> bool {
        if self.tool != tool {
            return false;
        }
        let Some(want) = self.arguments.as_object() else {
            return true;
        };
        want.iter()
            .all(|(k, v)| arguments.get(k).is_some_and(|got| loose_eq(got, v)))
    }
}

/// A call the model made, or one the host is about to make.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedCall {
    pub tool: String,
    pub arguments: Value,
}

impl PlannedCall {
    pub fn new(tool: impl Into<String>, arguments: Value) -> Self {
        Self {
            tool: tool.into(),
            arguments,
        }
    }
}

/// What to change about one round of calls; see [`ToolPlan::review_round`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RoundReview {
    /// Planned steps the model skipped, to run alongside its own calls.
    pub add: Vec<PlannedStep>,
    /// Indexes into the round's calls to hold back because a tool they need
    /// has not run yet. Run the rest; afterwards, [`ToolPlan::follow_up`]
    /// tells the model to make these calls with the new results.
    pub defer: Vec<usize>,
    /// Steps that still wait on a tool running this round.
    pub waiting: Vec<PlannedStep>,
    /// Indexes of calls to tools a [`DirectAnswer`] part rules out; drop them
    /// and give the model [`ToolPlan::direct_note`].
    pub drop: Vec<usize>,
}

impl RoundReview {
    pub fn is_empty(&self) -> bool {
        self.add.is_empty()
            && self.defer.is_empty()
            && self.waiting.is_empty()
            && self.drop.is_empty()
    }

    /// Indexes to remove from the round (held back or dropped), highest first.
    pub fn removals(&self) -> Vec<usize> {
        let mut all: Vec<usize> = self.defer.iter().chain(&self.drop).copied().collect();
        all.sort_unstable_by(|a, b| b.cmp(a));
        all.dedup();
        all
    }
}

/// The tools a request needs, in request order, and the parts it can answer
/// without tools.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolPlan {
    pub steps: Vec<PlannedStep>,
    /// Purposes of matched [`DirectAnswer`]s, and the tools they rule out.
    pub direct: Vec<(String, Vec<String>)>,
}

/// Match `request` against `intents` and return the steps it calls for.
///
/// A step is listed once even when several phrases trigger it. A step that
/// needs another tool's result brings that tool in ahead of itself, so "news
/// about my city" plans the location lookup and then the search.
pub fn plan(request: &str, intents: &[ToolIntent]) -> ToolPlan {
    plan_with(request, intents, &[])
}

/// [`plan`], plus the [`DirectAnswer`] parts of the request.
pub fn plan_with(request: &str, intents: &[ToolIntent], direct: &[DirectAnswer]) -> ToolPlan {
    let words = words(request);
    let mut hits: Vec<(usize, &ToolIntent, &str)> = Vec::new();
    for intent in intents {
        if intent
            .unless
            .iter()
            .any(|u| find_words(&words, u).is_some())
        {
            continue;
        }
        if let Some((at, phrase)) = first_trigger(&words, &intent.triggers) {
            hits.push((at, intent, phrase));
        }
    }
    hits.sort_by_key(|(at, _, _)| *at);

    let mut out = ToolPlan::default();
    for (_, intent, phrase) in hits {
        for dep in &intent.requires {
            if !out.steps.iter().any(|s| &s.tool == dep) {
                let purpose = intents
                    .iter()
                    .find(|i| &i.tool == dep && i.requires.is_empty())
                    .map(|i| i.purpose.clone())
                    .unwrap_or_else(|| format!("input for {}", intent.tool));
                out.push(PlannedStep {
                    tool: dep.clone(),
                    arguments: Value::Object(Map::new()),
                    requires: Vec::new(),
                    purpose,
                    matched: phrase.to_string(),
                });
            }
        }
        out.push(PlannedStep {
            tool: intent.tool.clone(),
            arguments: intent
                .arguments
                .clone()
                .unwrap_or_else(|| Value::Object(Map::new())),
            requires: intent.requires.clone(),
            purpose: intent.purpose.clone(),
            matched: phrase.to_string(),
        });
    }
    for d in direct {
        if d.unless.iter().any(|u| find_words(&words, u).is_some()) {
            continue;
        }
        if first_trigger(&words, &d.triggers).is_some() {
            out.direct.push((d.purpose.clone(), d.skip.clone()));
        }
    }
    out
}

impl ToolPlan {
    /// No steps and no direct-answer parts: nothing to check rounds against.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty() && self.direct.is_empty()
    }

    fn push(&mut self, step: PlannedStep) {
        let dup = self
            .steps
            .iter()
            .any(|s| s.tool == step.tool && s.arguments == step.arguments);
        if !dup {
            self.steps.push(step);
        }
    }

    /// A prompt block naming every step, or `None` when there are no steps.
    ///
    /// Each line carries the call itself, so a model that copies it emits a
    /// valid call.
    pub fn hint(&self) -> Option<String> {
        if self.steps.is_empty() {
            return None;
        }
        let mut out = String::from(
            "## Plan for this request\nThis request needs these tool calls. Make every one; \
             a step marked \"after\" needs that tool's result first.\n",
        );
        for (i, s) in self.steps.iter().enumerate() {
            let args = s.arguments.to_string();
            out.push_str(&format!("{}. {} {} for {}", i + 1, s.tool, args, s.purpose));
            if !s.requires.is_empty() {
                out.push_str(&format!(
                    " (after {}; use its result)",
                    s.requires.join(", ")
                ));
            }
            out.push('\n');
        }
        out.push_str("Parts of the request that need no tool, answer directly.");
        Some(out)
    }

    /// Check one round of calls against the plan.
    ///
    /// `done` holds the calls that ran in earlier rounds; `round` the calls
    /// the model just emitted.
    pub fn review_round(&self, done: &[PlannedCall], round: &[PlannedCall]) -> RoundReview {
        let ran = |tool: &str| done.iter().any(|c| c.tool == tool);
        let mut review = RoundReview::default();

        for (i, call) in round.iter().enumerate() {
            let step = self
                .steps
                .iter()
                .find(|s| s.covered_by(&call.tool, &call.arguments));
            match step {
                // Hold back calls whose inputs are not in yet.
                Some(s) if s.requires.iter().any(|r| !ran(r)) => review.defer.push(i),
                Some(_) => {}
                // An unplanned call to a tool a direct-answer part rules out.
                None if self
                    .direct
                    .iter()
                    .any(|(_, skip)| skip.contains(&call.tool)) =>
                {
                    review.drop.push(i)
                }
                None => {}
            }
        }
        let removed = review.removals();
        let kept: Vec<&PlannedCall> = round
            .iter()
            .enumerate()
            .filter(|(i, _)| !removed.contains(i))
            .map(|(_, c)| c)
            .collect();

        for step in &self.steps {
            let covered = done
                .iter()
                .chain(kept.iter().copied())
                .any(|c| step.covered_by(&c.tool, &c.arguments));
            if covered {
                continue;
            }
            if step.requires.iter().all(|r| ran(r)) {
                review.add.push(step.clone());
            } else {
                review.waiting.push(step.clone());
            }
        }
        review
    }

    /// A message for the model after a round in which steps were held back,
    /// or `None` when nothing waits.
    pub fn follow_up(&self, waiting: &[PlannedStep]) -> Option<String> {
        if waiting.is_empty() {
            return None;
        }
        let parts: Vec<String> = waiting
            .iter()
            .map(|s| {
                format!(
                    "call {} for {}, using the {} result above",
                    s.tool,
                    s.purpose,
                    s.requires.join(" and ")
                )
            })
            .collect();
        Some(format!(
            "Next, {}. Put the real values from that result into the arguments.",
            parts.join("; then ")
        ))
    }

    /// A message for the model after calls were dropped for a direct-answer
    /// part, or `None` when the request has none.
    pub fn direct_note(&self) -> Option<String> {
        if self.direct.is_empty() {
            return None;
        }
        let parts: Vec<&str> = self.direct.iter().map(|(p, _)| p.as_str()).collect();
        Some(format!(
            "No tool is needed for {}: answer that part directly from what you know.",
            parts.join(" or ")
        ))
    }
}

/// Words of `s`: lowercase, straight apostrophes kept inside words, and the
/// arithmetic signs spelled out (`2+2` → `2 plus 2`, `25%` → `25 percent`).
fn words(s: &str) -> Vec<String> {
    let mut spaced = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\u{2019}' | '\u{2018}' | '`' => spaced.push('\''),
            '+' => spaced.push_str(" plus "),
            '*' | '×' => spaced.push_str(" times "),
            '%' => spaced.push_str(" percent "),
            c if c.is_alphanumeric() || c == '\'' => spaced.extend(c.to_lowercase()),
            _ => spaced.push(' '),
        }
    }
    spaced
        .split_whitespace()
        .map(|w| w.trim_matches('\'').to_string())
        .filter(|w| !w.is_empty())
        .collect()
}

fn is_number(w: &str) -> bool {
    w.chars().next().is_some_and(|c| c.is_ascii_digit())
}

/// One word of a trigger: literal, or `#n` (any number).
enum Pat {
    Word(String),
    Number,
}

fn pattern(phrase: &str) -> Vec<Pat> {
    let mut out = Vec::new();
    for token in phrase.split_whitespace() {
        if token == "#n" {
            out.push(Pat::Number);
        } else {
            out.extend(words(token).into_iter().map(Pat::Word));
        }
    }
    out
}

/// Word index of `phrase` in `text`, ignoring negated occurrences.
fn find_words(text: &[String], phrase: &str) -> Option<usize> {
    let pat = pattern(phrase);
    if pat.is_empty() || pat.len() > text.len() {
        return None;
    }
    (0..=text.len() - pat.len()).find(|&at| {
        let hit = pat.iter().enumerate().all(|(j, p)| match p {
            Pat::Word(w) => &text[at + j] == w,
            Pat::Number => is_number(&text[at + j]),
        });
        hit && !negated(text, at)
    })
}

/// Whether a negator ends fewer than [`NEGATION_WINDOW`] words before `at`.
fn negated(text: &[String], at: usize) -> bool {
    NEGATORS.iter().any(|n| {
        let neg: Vec<&str> = n.split(' ').collect();
        (0..at).any(|start| {
            let end = start + neg.len();
            end <= at
                && at - end < NEGATION_WINDOW
                && text[start..end].iter().zip(&neg).all(|(a, b)| a == b)
        })
    })
}

/// The earliest matching trigger: (word index, trigger).
fn first_trigger<'a>(text: &[String], triggers: &'a [String]) -> Option<(usize, &'a str)> {
    triggers
        .iter()
        .filter_map(|t| find_words(text, t).map(|at| (at, t.as_str())))
        .min_by_key(|(at, _)| *at)
}

/// Equality that ignores ASCII case for strings.
fn loose_eq(a: &Value, b: &Value) -> bool {
    match (a.as_str(), b.as_str()) {
        (Some(x), Some(y)) => x.eq_ignore_ascii_case(y),
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn intents() -> Vec<ToolIntent> {
        vec![
            ToolIntent::new(
                "date",
                "the current date and time",
                &["what time", "the time", "current time", "what day"],
            ),
            ToolIntent::new(
                "location",
                "where the user is",
                &["where am i", "my location"],
            ),
            ToolIntent::new("skill", "the calendar", &["calendar", "meetings"])
                .with_arguments(json!({"command": "calendar_events"})),
            ToolIntent::new("skill", "the brain state", &["brain state", "focus level"])
                .with_arguments(json!({"command": "status"})),
            ToolIntent::new(
                "web_search",
                "news or weather where the user is",
                &["my city", "weather here"],
            )
            .after("location"),
        ]
    }

    fn call(tool: &str, args: Value) -> PlannedCall {
        PlannedCall::new(tool, args)
    }

    #[test]
    fn plans_each_part_in_request_order() {
        let p = plan(
            "Check my calendar for today and tell me the current time.",
            &intents(),
        );
        let tools: Vec<_> = p
            .steps
            .iter()
            .map(|s| (s.tool.as_str(), s.arguments.clone()))
            .collect();
        assert_eq!(
            tools,
            vec![
                ("skill", json!({"command": "calendar_events"})),
                ("date", json!({}))
            ]
        );
        assert_eq!(p.steps[1].matched, "current time");
    }

    #[test]
    fn whole_words_only_and_no_plan_for_knowledge() {
        assert!(plan("Who wrote Pride and Prejudice?", &intents()).is_empty());
        // "the time" must not fire inside "the timeline".
        assert!(plan("Explain the timeline of WW2", &intents()).is_empty());
        // Curly apostrophes and punctuation are normalized.
        assert_eq!(plan("Where am I…?", &intents()).steps[0].tool, "location");
    }

    #[test]
    fn dependency_is_planned_first() {
        let p = plan("Search for news about my city.", &intents());
        let tools: Vec<_> = p.steps.iter().map(|s| s.tool.as_str()).collect();
        assert_eq!(tools, vec!["location", "web_search"]);
        assert_eq!(p.steps[0].purpose, "where the user is");
    }

    #[test]
    fn review_adds_skipped_steps() {
        let p = plan(
            "What's my focus level, and what day of the week is it?",
            &intents(),
        );
        let r = p.review_round(&[], &[call("skill", json!({"command": "status"}))]);
        assert_eq!(r.add.len(), 1);
        assert_eq!(r.add[0].tool, "date");
        assert!(r.defer.is_empty() && r.waiting.is_empty());
        // Argument values compare case-insensitively; extra keys are fine.
        let r = p.review_round(
            &[],
            &[
                call("skill", json!({"command": "STATUS", "args": {}})),
                call("date", json!({})),
            ],
        );
        assert!(r.is_empty(), "{r:?}");
    }

    #[test]
    fn review_defers_a_call_that_needs_a_missing_result() {
        let p = plan("Search for news about my city.", &intents());
        let round = [call("web_search", json!({"query": "news about my city"}))];
        let r = p.review_round(&[], &round);
        assert_eq!(r.defer, vec![0]);
        assert_eq!(
            r.add.iter().map(|s| s.tool.as_str()).collect::<Vec<_>>(),
            vec!["location"]
        );
        assert_eq!(r.waiting.len(), 1);
        let next = p.follow_up(&r.waiting).expect("follow-up");
        assert!(
            next.contains("web_search") && next.contains("location result"),
            "{next}"
        );

        // Next round: location ran, the search goes through.
        let done = [call("location", json!({}))];
        let round = [call("web_search", json!({"query": "news San Jose"}))];
        assert!(p.review_round(&done, &round).is_empty());
    }

    #[test]
    fn same_round_dependency_is_still_deferred() {
        // The model calls both at once; the search cannot have the city yet.
        let p = plan("What's the weather here?", &intents());
        let round = [
            call("location", json!({})),
            call("web_search", json!({"query": "weather <city>"})),
        ];
        let r = p.review_round(&[], &round);
        assert_eq!(r.defer, vec![1]);
        assert!(r.add.is_empty());
        assert_eq!(r.waiting.len(), 1);
    }

    #[test]
    fn hint_lists_copyable_calls() {
        let hint = plan("Where am I, and what's on my calendar?", &intents())
            .hint()
            .unwrap();
        assert!(
            hint.contains("1. location {} for where the user is"),
            "{hint}"
        );
        assert!(
            hint.contains(r#"2. skill {"command":"calendar_events"} for the calendar"#),
            "{hint}"
        );
        assert!(plan("hi", &intents()).hint().is_none());
    }

    #[test]
    fn negated_and_cancelled_triggers_do_not_fire() {
        let i = intents();
        assert_eq!(
            plan("Don't check my calendar, just tell me the time.", &i)
                .steps
                .iter()
                .map(|s| s.tool.as_str())
                .collect::<Vec<_>>(),
            ["date"]
        );
        assert!(plan("Without looking at my calendar, what's 2+2?", &i).is_empty());
        // A negator further back does not reach the trigger.
        assert_eq!(
            plan("I don't know what to do today, check my calendar", &i)
                .steps
                .len(),
            1
        );

        let local = vec![
            ToolIntent::new("web_search", "local weather", &["weather here"])
                .after("location")
                .unless(&["i'm in", "i am in"]),
        ];
        assert!(plan("I'm in Paris, what's the weather here?", &local).is_empty());
        assert_eq!(plan("What's the weather here?", &local).steps.len(), 2);
    }

    #[test]
    fn number_wildcard_and_arithmetic_signs() {
        let math = [DirectAnswer::new(
            "arithmetic",
            &["#n percent of #n", "#n plus #n", "#n times #n"],
            &["web_search"],
        )];
        for q in ["What's 25% of 200?", "what is 2+2", "17 * 3 = ?", "7 × 6"] {
            assert_eq!(plan_with(q, &[], &math).direct.len(), 1, "{q}");
        }
        for q in ["What's the percent of voters?", "plus size", "3 meetings"] {
            assert!(plan_with(q, &[], &math).direct.is_empty(), "{q}");
        }
    }

    #[test]
    fn direct_answers_drop_unplanned_searches_only() {
        let direct = [DirectAnswer::new(
            "a translation",
            &["translate", "how do you say"],
            &["web_search"],
        )
        .unless(&["latest", "search"])];
        let p = plan_with(
            "Translate 'good morning' into Spanish, and what time is it?",
            &intents(),
            &direct,
        );
        assert_eq!(p.steps.len(), 1);
        let round = [
            call("web_search", json!({"query": "good morning in spanish"})),
            call("date", json!({})),
        ];
        let r = p.review_round(&[], &round);
        assert_eq!(r.drop, vec![0]);
        assert!(r.add.is_empty() && r.defer.is_empty());
        assert_eq!(r.removals(), vec![0]);
        assert!(p.direct_note().unwrap().contains("a translation"));

        // Asking for a search keeps it.
        assert!(
            plan_with("Search how do you say hello in Thai", &[], &direct)
                .direct
                .is_empty()
        );
        // A planned search is never dropped.
        let p = plan_with("Translate the news about my city", &intents(), &direct);
        let r = p.review_round(
            &[call("location", json!({}))],
            &[call("web_search", json!({"query": "x"}))],
        );
        assert!(r.drop.is_empty(), "{r:?}");
    }
}
