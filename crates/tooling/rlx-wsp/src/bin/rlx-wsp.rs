// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-wsp` — drive a weight space from the shell or an agent loop.
//!
//! ```text
//! rlx-wsp tools                                   # tool schemas (JSON)
//! rlx-wsp new   --state ws.json [--env bounce] [--hidden 32,32] [--act tanh] [--budget 50]
//! rlx-wsp call  --state ws.json '{"tool":"describe"}'   # one call; '-' reads stdin
//! rlx-wsp serve [--env bounce] [--hidden ..] [--budget ..]   # JSON lines on stdin/stdout
//! rlx-wsp score --state ws.json [--episodes 512]  # held-out score (harness only)
//! rlx-wsp bench [--budget 50]                     # experiment 1 table
//! rlx-wsp mcp   [--state ws.json]                 # the tools as a stdio MCP server
//! rlx-wsp agent --cli claude|agent|copilot --out run1 [--budget 50] [--model M]
//!               [--effort high] [--max-turns 80] [--max-budget-usd 5] [--dry-run] [-- extra CLI args]
//! rlx-wsp new   --state ws.json --graph model.rlxmodel [--weights w.json] [--goal "..."]
//! rlx-wsp warmstart --state run/state.json [--epochs 6]  # SGD from agent vs random weights
//! rlx-wsp export --state run/state.json --out dir [--fpga]  # .rlxmodel + int8 check (+ RTL)
//! rlx-wsp verify --state run/state.json             # unit ablations vs the agent's report
//! rlx-wsp rl-baseline --env orchard [--iters 200] [--seeds 3] [--stochastic]  # trained baseline
//! rlx-wsp bundle --run runs/x/claude-default-1 --out bundle/   # publishable run package
//! rlx-wsp graft --state dst.json --from-state src.json --units l0.h3,l0.h4 [--into l1.h0,l1.h1]
//! flags: --stochastic --oneshot --internals-only --open --task-seed N
//! packages: --package m.rlxp [--qa-practice p.jsonl --qa-heldout h.jsonl --heldout-n 100
//!           --max-tokens 256 --device cpu|metal]
//! rlx-wsp sweep --agents claude:opus,agent:gpt-5.2,copilot --out runs/s1 [--repeats 2]
//!               [--env gusts] [--budget 50] [...agent options]   # parallel runs + summary.md
//! ```
//!
//! `--agents` lists `cli[:model]` entries; without a model the CLI's default
//! is used. Models are whatever the CLI accepts: `claude --model opus|sonnet|
//! <full id>`, `agent --list-models`, `copilot --model <id>|auto`.

use rlx_wsp::agent::{AgentCli, RunSpec, command_line, setup};
use rlx_wsp::baselines::{
    bounce_landing_program, bounce_tracker_program, one_plus_one_es, random_init,
};
use rlx_wsp::space::{play_policy, play_policy_with};
use rlx_wsp::{
    Act, EnvKind, Orchard, RolloutReport, Rules, SCORE_SEED, State, WeightSpace, handle_json,
    tool_schemas,
};
use serde_json::Value;
use std::io::{BufRead, Read, Write};

#[derive(Clone)]
struct Opts {
    state: Option<String>,
    env: EnvKind,
    hidden: Vec<usize>,
    act: Act,
    budget: usize,
    episodes: usize,
    cli: AgentCli,
    out: Option<String>,
    model: Option<String>,
    effort: Option<String>,
    max_turns: usize,
    max_budget_usd: Option<f64>,
    dry_run: bool,
    /// `sweep`: `(cli, model)` per entry of `--agents`.
    agents: Vec<(AgentCli, Option<String>)>,
    repeats: usize,
    /// `--graph model.rlxmodel` (+ `--weights w.json`, `--goal "..."`).
    graph: Option<String>,
    weights: Option<String>,
    goal: Option<String>,
    /// `--score-cmd "prog args"`: the `external_score` tool's command.
    score_cmd: Option<String>,
    fpga: bool,
    epochs: usize,
    /// Score with sampled + sticky actions (the post's protocol).
    stochastic: bool,
    /// One-reply mode: budget 0 and only describe + write tools.
    oneshot: bool,
    /// `sweep`: run every agent at each of these budgets.
    budgets: Vec<usize>,
    task_seed: u64,
    iters: usize,
    seeds: u64,
    run: Option<String>,
    lr: Option<f32>,
    gamma: Option<f32>,
    internals_only: bool,
    open: bool,
    from_state: Option<String>,
    units: Vec<String>,
    into: Vec<String>,
    method: String,
    package: Option<String>,
    qa_practice: Option<String>,
    qa_heldout: Option<String>,
    heldout_n: usize,
    max_tokens: usize,
    device: String,
    dev_n: usize,
    dev_checks: usize,
    qa_general: Option<String>,
    prompt_style: String,
    stop_at_answer: bool,
    repetition_penalty: Option<f32>,
    token_budget: Option<usize>,
    keep_verified: bool,
    max_edit: Option<f32>,
    check_perturbed: bool,
    check_truncation: bool,
    /// Jaccard at which a practice question counts as a near-duplicate of a
    /// held-out one (dropped from the tools).
    dedupe: f64,
    /// `lora-train`.
    targets: Vec<String>,
    rank: Option<usize>,
    alpha: Option<f32>,
    steps: Option<usize>,
    eps: Option<f32>,
    batch: Option<usize>,
    train_package: Option<String>,
    mode: String,
    /// `run`.
    prompts: Vec<String>,
    /// `branch` / `agent --from-state`: the version to start from.
    checkpoint: Option<String>,
    /// Tools the run may not use.
    deny_tools: Vec<String>,
    /// Steer a LoRA training run (`--opal --train-package …`).
    opal: bool,
    /// LoRA: loss-spike gate, dev-selected averaging.
    gate: Option<f32>,
    average: bool,
    snapshot_every: Option<usize>,
    shots: usize,
    questions: usize,
    offset: usize,
    train_items: Option<String>,
    train_export: Option<String>,
    stages: Vec<String>,
    redo: Vec<String>,
    /// Flags given on the command line (a branch overrides only these).
    given: std::collections::BTreeSet<String>,
    /// Everything after `--`, passed to the agent CLI verbatim.
    passthrough: Vec<String>,
    rest: Vec<String>,
}

fn parse(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        state: None,
        env: EnvKind::Bounce,
        hidden: vec![32, 32],
        act: Act::Tanh,
        budget: 50,
        episodes: 512,
        cli: AgentCli::Claude,
        out: None,
        model: None,
        effort: None,
        max_turns: 80,
        max_budget_usd: None,
        dry_run: false,
        agents: Vec::new(),
        repeats: 1,
        graph: None,
        weights: None,
        goal: None,
        score_cmd: None,
        fpga: false,
        epochs: 6,
        stochastic: false,
        oneshot: false,
        budgets: Vec::new(),
        task_seed: 0,
        iters: 1000,
        seeds: 3,
        run: None,
        lr: None,
        gamma: None,
        internals_only: false,
        open: false,
        from_state: None,
        units: Vec::new(),
        into: Vec::new(),
        method: "ppo".into(),
        package: None,
        qa_practice: None,
        qa_heldout: None,
        heldout_n: 100,
        max_tokens: 256,
        device: "cpu".into(),
        dev_n: 0,
        dev_checks: 3,
        qa_general: None,
        prompt_style: "plain".into(),
        stop_at_answer: false,
        repetition_penalty: None,
        token_budget: None,
        keep_verified: false,
        max_edit: None,
        check_perturbed: false,
        check_truncation: false,
        dedupe: 0.6,
        targets: Vec::new(),
        rank: None,
        alpha: None,
        steps: None,
        eps: None,
        batch: None,
        train_package: None,
        mode: "solution".into(),
        prompts: Vec::new(),
        checkpoint: None,
        deny_tools: Vec::new(),
        opal: false,
        gate: None,
        average: false,
        snapshot_every: None,
        shots: 1,
        questions: 256,
        offset: 0,
        train_items: None,
        train_export: None,
        stages: Vec::new(),
        redo: Vec::new(),
        given: Default::default(),
        passthrough: Vec::new(),
        rest: Vec::new(),
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a.starts_with("--") {
            o.given.insert(a.clone());
        }
        let mut val = || it.next().cloned().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--state" => o.state = Some(val()?),
            "--env" => o.env = EnvKind::parse(&val()?).ok_or("unknown --env")?,
            "--hidden" => {
                o.hidden = val()?
                    .split(',')
                    .map(|s| s.parse().map_err(|_| format!("bad width `{s}`")))
                    .collect::<Result<_, _>>()?
            }
            "--act" => {
                o.act = match val()?.as_str() {
                    "tanh" => Act::Tanh,
                    "relu" => Act::Relu,
                    s => return Err(format!("unknown --act `{s}`")),
                }
            }
            "--budget" => o.budget = val()?.parse().map_err(|_| "bad --budget")?,
            "--episodes" => o.episodes = val()?.parse().map_err(|_| "bad --episodes")?,
            "--cli" => {
                let v = val()?;
                o.cli = AgentCli::parse(&v).ok_or(format!("unknown --cli `{v}`"))?
            }
            "--out" => o.out = Some(val()?),
            "--model" => o.model = Some(val()?),
            "--effort" => o.effort = Some(val()?),
            "--max-turns" => o.max_turns = val()?.parse().map_err(|_| "bad --max-turns")?,
            "--max-budget-usd" => {
                o.max_budget_usd = Some(val()?.parse().map_err(|_| "bad --max-budget-usd")?)
            }
            "--dry-run" => o.dry_run = true,
            "--agents" => {
                o.agents = val()?
                    .split(',')
                    .filter(|e| !e.is_empty())
                    .map(|e| {
                        let (cli, model) = match e.split_once(':') {
                            Some((c, m)) => (c, Some(m.to_string())),
                            None => (e, None),
                        };
                        AgentCli::parse(cli)
                            .map(|c| (c, model))
                            .ok_or(format!("unknown cli `{cli}` in --agents"))
                    })
                    .collect::<Result<_, _>>()?
            }
            "--repeats" => o.repeats = val()?.parse().map_err(|_| "bad --repeats")?,
            "--graph" => o.graph = Some(val()?),
            "--weights" => o.weights = Some(val()?),
            "--goal" => o.goal = Some(val()?),
            "--score-cmd" => o.score_cmd = Some(val()?),
            "--fpga" => o.fpga = true,
            "--stochastic" => o.stochastic = true,
            "--oneshot" => o.oneshot = true,
            "--budgets" => {
                o.budgets = val()?
                    .split(',')
                    .map(|b| b.parse().map_err(|_| format!("bad budget `{b}`")))
                    .collect::<Result<_, _>>()?
            }
            "--task-seed" => o.task_seed = val()?.parse().map_err(|_| "bad --task-seed")?,
            "--iters" => o.iters = val()?.parse().map_err(|_| "bad --iters")?,
            "--seeds" => o.seeds = val()?.parse().map_err(|_| "bad --seeds")?,
            "--run" => o.run = Some(val()?),
            "--lr" => o.lr = Some(val()?.parse().map_err(|_| "bad --lr")?),
            "--internals-only" => o.internals_only = true,
            "--open" => o.open = true,
            "--method" => o.method = val()?,
            "--package" => o.package = Some(val()?),
            "--qa-practice" => o.qa_practice = Some(val()?),
            "--qa-heldout" => o.qa_heldout = Some(val()?),
            "--heldout-n" => o.heldout_n = val()?.parse().map_err(|_| "bad --heldout-n")?,
            "--max-tokens" => o.max_tokens = val()?.parse().map_err(|_| "bad --max-tokens")?,
            "--device" => o.device = val()?,
            "--dev-n" => o.dev_n = val()?.parse().map_err(|_| "bad --dev-n")?,
            "--dev-checks" => o.dev_checks = val()?.parse().map_err(|_| "bad --dev-checks")?,
            "--qa-general" => o.qa_general = Some(val()?),
            "--prompt-style" => o.prompt_style = val()?,
            "--stop-at-answer" => o.stop_at_answer = true,
            "--repetition-penalty" => {
                o.repetition_penalty = Some(val()?.parse().map_err(|_| "bad --repetition-penalty")?)
            }
            "--token-budget" => {
                o.token_budget = Some(val()?.parse().map_err(|_| "bad --token-budget")?)
            }
            "--keep" => {
                o.keep_verified = match val()?.as_str() {
                    "verified" => true,
                    "final" => false,
                    k => return Err(format!("--keep `{k}`: final | verified")),
                }
            }
            "--max-edit" => o.max_edit = Some(val()?.parse().map_err(|_| "bad --max-edit")?),
            "--check-perturbed" => o.check_perturbed = true,
            "--check-truncation" => o.check_truncation = true,
            "--dedupe" => o.dedupe = val()?.parse().map_err(|_| "bad --dedupe")?,
            "--targets" => o.targets = val()?.split(',').map(str::to_string).collect(),
            "--rank" => o.rank = Some(val()?.parse().map_err(|_| "bad --rank")?),
            "--alpha" => o.alpha = Some(val()?.parse().map_err(|_| "bad --alpha")?),
            "--steps" => o.steps = Some(val()?.parse().map_err(|_| "bad --steps")?),
            "--eps" => o.eps = Some(val()?.parse().map_err(|_| "bad --eps")?),
            "--batch" => o.batch = Some(val()?.parse().map_err(|_| "bad --batch")?),
            "--train-package" => o.train_package = Some(val()?),
            "--mode" => o.mode = val()?,
            "--prompt" => o.prompts.push(val()?),
            "--checkpoint" => o.checkpoint = Some(val()?),
            "--opal" => o.opal = true,
            "--gate" => o.gate = Some(val()?.parse().map_err(|_| "bad --gate")?),
            "--average" => o.average = true,
            "--shots" => o.shots = val()?.parse().map_err(|_| "bad --shots")?,
            "--questions" => o.questions = val()?.parse().map_err(|_| "bad --questions")?,
            "--offset" => o.offset = val()?.parse().map_err(|_| "bad --offset")?,
            "--train-items" => o.train_items = Some(val()?),
            "--train-export" => o.train_export = Some(val()?),
            "--stages" => o.stages = val()?.split(',').map(String::from).collect(),
            "--redo" => o.redo = val()?.split(',').map(String::from).collect(),
            "--snapshot-every" => {
                o.snapshot_every = Some(val()?.parse().map_err(|_| "bad --snapshot-every")?)
            }
            "--deny-tools" => {
                o.deny_tools = val()?
                    .split(',')
                    .filter(|t| !t.is_empty())
                    .map(str::to_string)
                    .collect()
            }
            "--from-state" => o.from_state = Some(val()?),
            "--units" => o.units = val()?.split(',').map(str::to_string).collect(),
            "--into" => o.into = val()?.split(',').map(str::to_string).collect(),
            "--gamma" => o.gamma = Some(val()?.parse().map_err(|_| "bad --gamma")?),
            "--epochs" => o.epochs = val()?.parse().map_err(|_| "bad --epochs")?,
            "--" => {
                o.passthrough = it.by_ref().cloned().collect();
            }
            _ => o.rest.push(a.clone()),
        }
    }
    Ok(o)
}

fn load(path: &str) -> Result<WeightSpace, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let st: State = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    Ok(WeightSpace::from_state(st))
}

fn save(path: &str, ws: &WeightSpace) -> Result<(), String> {
    let text = ws.to_json()?;
    std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))
}

fn need_state(o: &Opts) -> Result<&str, String> {
    o.state
        .as_deref()
        .ok_or_else(|| "--state <file> is required".into())
}

/// mnist-fix references: the tampered model as handed over, a model trained
/// on clean data, and the tampered model with the trigger pixels' input
/// weights zeroed (a fix that knows where the trigger is).
fn bench_mnist(o: &Opts) -> Result<(), String> {
    use rlx_wsp::mnist::trained;
    let row = |label: &str, ws: &mut WeightSpace| -> Result<(), String> {
        let h = ws.held_out(o.episodes)?;
        println!(
            "{label:<44} {:>8.2}   {}",
            h["score"].as_f64().unwrap_or(0.0),
            h["detail"].as_str().unwrap_or("")
        );
        Ok(())
    };
    let trigger = o.env.trigger().ok_or("not a repair task")?;
    println!(
        "{:?} — trigger {trigger:?}; score = clean accuracy − attack success (percentage points)\n",
        o.env
    );
    let mut ws = WeightSpace::try_new(o.env, &o.hidden, o.act, o.budget)?;
    row("tampered model, as handed to the agent", &mut ws)?;
    let tampered = ws.state.params.clone();
    let mut st = ws.state.clone();
    st.params = trained(None)?;
    row(
        "reference: trained on clean data",
        &mut WeightSpace::from_state(st),
    )?;
    ws.edit("l0.w", "zero trigger rows", |t| {
        for (i, _) in trigger.cells() {
            t.data[i * t.cols..(i + 1) * t.cols].fill(0.0);
        }
        Ok(())
    })?;
    row("reference fix: zero the trigger pixels' rows", &mut ws)?;
    // Knowing the trigger: switch off the layer-0 units it excites most
    // (mean activation, stamped minus clean, over validation digits).
    let data = rlx_wsp::mnist::data()?;
    let net =
        rlx_wsp::train::Trainer::from_params(&ws.state.mlp.widths, ws.state.mlp.act, &tampered);
    let n = 1000.min(data.val.len());
    let width = ws.state.mlp.widths[1];
    let mut lift = vec![0f32; width];
    for i in 0..n {
        let img = data.val.image(i);
        let mut stamped = img.to_vec();
        trigger.stamp(&mut stamped);
        let (a, b) = (net.forward(img), net.forward(&stamped));
        for u in 0..width {
            lift[u] += (b[1][u] - a[1][u]) / n as f32;
        }
    }
    let mut order: Vec<usize> = (0..width).collect();
    order.sort_by(|&a, &b| lift[b].total_cmp(&lift[a]));
    let mut best = (f64::NEG_INFINITY, 0, String::new());
    ws.set_params(tampered.clone());
    for k in 1..=12 {
        let units: Vec<rlx_wsp::graphops::Unit> = order[..k]
            .iter()
            .map(|&u| rlx_wsp::graphops::Unit {
                node: "l0.h".into(),
                index: u,
            })
            .collect();
        ws.set_masks(units);
        let h = ws.held_out(o.episodes);
        ws.set_masks(Vec::new());
        let h = h?;
        let sc = h["score"].as_f64().unwrap_or(f64::NEG_INFINITY);
        if sc > best.0 {
            best = (sc, k, h["detail"].as_str().unwrap_or("").to_string());
        }
    }
    println!(
        "{:<44} {:>8.2}   {} ({} units off)",
        "reference fix: ablate top trigger units", best.0, best.2, best.1
    );
    Ok(())
}

/// References for reading / audit / steering tasks.
fn bench_tasks(o: &Opts) -> Result<(), String> {
    match o.env {
        EnvKind::ReadRules | EnvKind::ReadDigit | EnvKind::AuditPair => {
            let n = match o.env {
                EnvKind::ReadRules => 3,
                EnvKind::ReadDigit => 10,
                _ => 2,
            };
            println!(
                "{:?}: answer-scored; chance = {:.1}%",
                o.env,
                100.0 / n as f32
            );
            // How well each copy-trained policy plays under its own rules.
            if o.env == EnvKind::ReadRules {
                use rlx_wsp::Env;
                for (i, (l, rules, _)) in rlx_wsp::tasks::RULE_CANDIDATES.iter().enumerate() {
                    let net = rlx_wsp::train::Trainer::from_params(
                        &[370, 32, 32, 3],
                        Act::Tanh,
                        &rlx_wsp::tasks::read_rules_policy(i)?,
                    );
                    let mut buf = vec![0f32; 370];
                    let policy = play_policy(
                        || Orchard::with_rules(*rules),
                        o.episodes,
                        SCORE_SEED,
                        |e| {
                            e.observe(&mut buf);
                            net.predict(&buf)
                        },
                    );
                    let planner = play_policy(
                        || Orchard::with_rules(*rules),
                        o.episodes,
                        SCORE_SEED,
                        Orchard::plan_action,
                    );
                    println!(
                        "  candidate {l}: copy-trained policy {:.2} (its planner {:.2})",
                        policy.mean_return, planner.mean_return
                    );
                }
            }
        }
        EnvKind::FixPolicy => {
            let mut ws = WeightSpace::try_new(o.env, &o.hidden, o.act, o.budget)?;
            let r = ws.evaluate(o.episodes, SCORE_SEED);
            println!(
                "fix-policy — held-out return over {} episodes\n",
                r.episodes
            );
            println!(
                "{:<44} {:>8.2}",
                "trained policy, as handed to the agent", r.mean_return
            );
            let mut best = WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, 0);
            bounce_landing_program(&mut best).map_err(|e| e.to_string())?;
            let r = best.evaluate(o.episodes, SCORE_SEED);
            println!(
                "{:<44} {:>8.2}",
                "reference: written landing program (max)", r.mean_return
            );
        }
        EnvKind::MnistSteer => {
            use rlx_wsp::tasks::{STEER_FROM, STEER_TO, steer_score};
            let mut ws = WeightSpace::try_new(o.env, &o.hidden, o.act, o.budget)?;
            let h = steer_score(&mut ws)?;
            println!("mnist-steer — score = mean(3s read as 8, other digits correct), %\n");
            println!(
                "{:<44} {:>8.2}   {}",
                "clean model, as handed over",
                h["score"].as_f64().unwrap_or(0.0),
                h["detail"].as_str().unwrap_or("")
            );
            // Logit-layer rewrite: logit8 += a·logit3, digit3 never predicted.
            let base = ws.state.params.clone();
            let mut best = (f64::NEG_INFINITY, 0.0f32, String::new());
            for a in [0.25f32, 0.5, 0.75, 1.0, 1.5, 2.0] {
                let mut p = base.clone();
                let w = p.get_mut("l2.w").unwrap();
                for r in 0..w.rows {
                    let v = w.at(r, STEER_FROM);
                    w.data[r * w.cols + STEER_TO] += a * v;
                    w.data[r * w.cols + STEER_FROM] = 0.0;
                }
                let b = p.get_mut("l2.b").unwrap();
                b.data[STEER_TO] += a * b.data[STEER_FROM];
                b.data[STEER_FROM] = -100.0;
                ws.set_params(p);
                let h = steer_score(&mut ws)?;
                let sc = h["score"].as_f64().unwrap_or(0.0);
                if sc > best.0 {
                    best = (sc, a, h["detail"].as_str().unwrap_or("").to_string());
                }
            }
            println!(
                "{:<44} {:>8.2}   {} (a = {})",
                "reference: logit8 += a·logit3, drop digit3", best.0, best.2, best.1
            );
        }
        _ => {}
    }
    Ok(())
}

fn bench(o: &Opts) {
    if o.env.is_reading() || matches!(o.env, EnvKind::MnistSteer | EnvKind::FixPolicy) {
        if let Err(e) = bench_tasks(o) {
            eprintln!("rlx-wsp: {e}");
        }
        return;
    }
    if o.env.is_classifier() {
        if let Err(e) = bench_mnist(o) {
            eprintln!("rlx-wsp: {e}");
        }
        return;
    }
    let protocol = if o.stochastic {
        rlx_wsp::space::Protocol::STOCHASTIC
    } else {
        rlx_wsp::space::Protocol::GREEDY
    };
    let new = || {
        let mut ws = WeightSpace::new(o.env, &o.hidden, o.act, o.budget);
        ws.state.protocol = protocol;
        ws
    };
    let print = |label: &str, r: &RolloutReport, fwd: usize| {
        let n = r.episodes as f32;
        println!(
            "{label:<40} {:>8.2} {:>8.2} {:>8.2} {:>6}",
            r.mean_return,
            r.positive_events as f32 / n,
            r.negative_events as f32 / n,
            fwd
        );
    };
    let row = |label: &str, ws: &mut WeightSpace| {
        let r = ws.evaluate(o.episodes, SCORE_SEED);
        print(label, &r, ws.state.used);
    };
    let mut ws = new();
    println!(
        "experiment 1 — {:?}, MLP {:?} {:?} ({} weights), held-out {} episodes, {}\n",
        o.env,
        ws.state.mlp.widths,
        ws.state.mlp.act,
        ws.state.mlp.n_weights(),
        o.episodes,
        protocol.describe()
    );
    println!(
        "{:<40} {:>8} {:>8} {:>8} {:>6}",
        "program", "return", "+ev/ep", "-ev/ep", "fwd"
    );
    row("zero weights", &mut ws);

    let mut ws = new();
    random_init(&mut ws, 1);
    row("random init (He)", &mut ws);

    for (budget, sigma) in [(o.budget, 0.05f32), (o.budget * 40, 0.05)] {
        let mut ws = WeightSpace::new(o.env, &o.hidden, o.act, budget);
        ws.state.protocol = protocol;
        one_plus_one_es(&mut ws, sigma, 64, 0xA11CE);
        row(&format!("(1+1)-ES from zero, {budget} fwd"), &mut ws);
    }

    match o.env {
        EnvKind::Bounce => {
            let mut ws = new();
            bounce_tracker_program(&mut ws).expect("tracker program");
            row("written: follow current ball x", &mut ws);
            let mut ws = new();
            bounce_landing_program(&mut ws).expect("landing program");
            row("written: steer to landing column", &mut ws);
        }
        EnvKind::Orchard => {
            let r = play_policy_with(
                Orchard::new,
                o.episodes,
                SCORE_SEED,
                protocol,
                Orchard::plan_action,
            );
            print("reference: lookahead planner (no network)", &r, 0);
        }
        EnvKind::MnistFix
        | EnvKind::MnistFixHard
        | EnvKind::ReadRules
        | EnvKind::ReadDigit
        | EnvKind::AuditPair
        | EnvKind::MnistSteer
        | EnvKind::FixPolicy
        | EnvKind::Qa
        | EnvKind::Graph => {}
        EnvKind::Gusts => {
            let r = play_policy_with(
                Orchard::gusts,
                o.episodes,
                SCORE_SEED,
                protocol,
                Orchard::plan_action,
            );
            print("reference: planner, true hidden rules", &r, 0);
            let r = play_policy_with(Orchard::gusts, o.episodes, SCORE_SEED, protocol, |e| {
                e.plan_action_assuming(Rules::ORCHARD)
            });
            print("planner assuming Orchard's rules", &r, 0);
        }
        EnvKind::GustsHard => {
            let r = play_policy_with(
                Orchard::gusts_hard,
                o.episodes,
                SCORE_SEED,
                protocol,
                Orchard::plan_action,
            );
            print("reference: planner, true hidden rules", &r, 0);
            let r = play_policy_with(Orchard::gusts_hard, o.episodes, SCORE_SEED, protocol, |e| {
                e.plan_action_assuming(Rules::ORCHARD)
            });
            print("planner assuming Orchard's rules", &r, 0);
        }
    }
}

/// A fresh space from the options: a loaded graph (`--graph`) or the task's
/// network (`--env`).
/// A branch of the run whose state is `--from-state` (a state file or a run
/// directory), at `--checkpoint` (default: its current weights), with a
/// fresh `--budget` and the guard / check flags given overriding its own.
fn branch_from_opts(o: &Opts) -> Result<WeightSpace, String> {
    let from = o
        .from_state
        .as_deref()
        .ok_or("--from-state <state.json | run dir> is required")?;
    let mut path = std::path::PathBuf::from(from);
    if path.is_dir() {
        path = path.join("state.json");
    }
    let path = abs_path(&path.display().to_string())?;
    let parent = load(&path)?;
    let version = o.checkpoint.as_deref().unwrap_or("current");
    let mut ws = parent.branch(&path, version, o.budget)?;
    let given = |f: &str| o.given.contains(f);
    if o.opal {
        ws.state.opal = opal_config(o)?;
    }
    if given("--deny-tools") {
        ws.state.denied_tools = o.deny_tools.clone();
    }
    if given("--goal") {
        ws.state.goal = o.goal.clone();
    }
    if given("--score-cmd") {
        ws.state.score_cmd = score_cmd(o);
    }
    if given("--token-budget") {
        ws.state.token_budget = o.token_budget;
    }
    if given("--keep") {
        ws.state.keep_verified = o.keep_verified;
    }
    if given("--max-edit") {
        ws.state.max_edit = o.max_edit;
    }
    // A dev split asked for here (or more checks): the branch gets checks
    // of its own.
    if given("--dev-checks") || (given("--dev-n") && o.dev_n > 0) {
        ws.state.dev_checks_left = o.dev_checks;
    }
    if let Some(q) = ws.state.qa.as_mut() {
        if given("--max-tokens") {
            q.max_tokens = o.max_tokens;
        }
        if given("--heldout-n") {
            q.heldout_n = o.heldout_n;
        }
        if given("--dev-n") {
            q.dev_n = o.dev_n;
        }
        if given("--qa-general") {
            q.general = o.qa_general.as_deref().map(abs_path).transpose()?;
        }
        if given("--prompt-style") {
            q.prompt_style = o.prompt_style.clone();
        }
        q.stop_at_answer |= o.stop_at_answer;
        if let Some(r) = o.repetition_penalty {
            q.repetition_penalty = r;
        }
        q.check_perturbed |= o.check_perturbed;
        q.check_truncation |= o.check_truncation;
        if given("--device") {
            q.device = o.device.clone();
        }
    }
    Ok(ws)
}

/// The steered-training config the flags ask for.
fn opal_config(o: &Opts) -> Result<Option<rlx_wsp::opal::OpalConfig>, String> {
    if !o.opal {
        return Ok(None);
    }
    let tp = o
        .train_package
        .as_deref()
        .ok_or("--opal needs --train-package <trainable .rlxp> (rlx-export … --variant train)")?;
    let d = rlx_wsp::lora::LoraOptions::default();
    Ok(Some(rlx_wsp::opal::OpalConfig {
        train_package: abs_path(tp)?,
        targets: if o.targets.is_empty() {
            d.targets
        } else {
            o.targets.clone()
        },
        rank: o.rank.unwrap_or(d.rank),
        alpha: o.alpha.unwrap_or(d.alpha),
        lr: o.lr.unwrap_or(1e-4),
        max_steps: o.steps.unwrap_or(500),
        steps_done: 0,
        mode: o.mode.clone(),
        updater: None,
    }))
}

/// `--score-cmd`, split into program and arguments.
fn score_cmd(o: &Opts) -> Option<Vec<String>> {
    o.score_cmd
        .as_ref()
        .map(|c| c.split_whitespace().map(str::to_string).collect())
}

fn space_from_opts(o: &Opts) -> Result<WeightSpace, String> {
    let mut ws = space_base(o)?;
    ws.state.task_seed = o.task_seed;
    if o.stochastic {
        ws.state.protocol = rlx_wsp::space::Protocol::STOCHASTIC;
    }
    ws.state.internals_only = o.internals_only;
    ws.state.open = o.open;
    ws.state.denied_tools = o.deny_tools.clone();
    ws.state.score_cmd = score_cmd(o);
    ws.state.opal = opal_config(o)?;
    if o.oneshot {
        ws.state.budget = 0;
        ws.state.tools = Some(ONESHOT_TOOLS.iter().map(|t| t.to_string()).collect());
    }
    Ok(ws)
}

/// One-reply mode, as in the post's from-scratch experiments: no rollouts,
/// no probes, no sandbox — the agent reads the rules and writes weights.
const ONESHOT_TOOLS: [&str; 11] = [
    "describe",
    "status",
    "list_params",
    "set",
    "set_col",
    "set_row",
    "write",
    "fill",
    "scale",
    "add_rank1",
    "answer",
];

/// Practice indices that nearly repeat one of the first `n` held-out
/// questions (Jaccard ≥ `threshold`; 1 or more turns it off).
fn near_duplicate_practice(
    practice: &str,
    heldout: &str,
    n: usize,
    threshold: f64,
) -> Result<Vec<usize>, String> {
    if threshold >= 1.0 {
        return Ok(Vec::new());
    }
    let p = rlx_wsp::lm::load_items(practice)?;
    let h = rlx_wsp::lm::load_items(heldout)?;
    let h = &h[..n.min(h.len())];
    let mut idx: Vec<usize> = rlx_wsp::checks::near_duplicates(&p, h, threshold)
        .into_iter()
        .map(|d| d.0)
        .collect();
    idx.sort_unstable();
    idx.dedup();
    Ok(idx)
}

fn space_base(o: &Opts) -> Result<WeightSpace, String> {
    if let Some(pkg) = &o.package {
        let qa = match (&o.qa_practice, &o.qa_heldout) {
            (Some(p), Some(h)) => Some(rlx_wsp::lm::QaSpec {
                practice: abs_path(p)?,
                heldout: abs_path(h)?,
                heldout_n: o.heldout_n,
                max_tokens: o.max_tokens,
                device: o.device.clone(),
                dev_n: o.dev_n,
                general: o.qa_general.as_deref().map(abs_path).transpose()?,
                prompt_style: o.prompt_style.clone(),
                stop_at_answer: o.stop_at_answer,
                excluded_practice: near_duplicate_practice(p, h, o.heldout_n, o.dedupe)?,
                check_perturbed: o.check_perturbed,
                check_truncation: o.check_truncation,
                repetition_penalty: o.repetition_penalty.unwrap_or(1.0),
            }),
            (None, None) => None,
            _ => return Err("give both --qa-practice and --qa-heldout".into()),
        };
        let mut ws = WeightSpace::from_package(&abs_path(pkg)?, o.budget, qa, o.goal.clone())?;
        ws.state.token_budget = o.token_budget;
        ws.state.dev_checks_left = if o.dev_n > 0 { o.dev_checks } else { 0 };
        ws.state.keep_verified = o.keep_verified;
        ws.state.max_edit = o.max_edit;
        return Ok(ws);
    }
    let Some(path) = &o.graph else {
        return WeightSpace::for_task(o.env, &o.hidden, o.act, o.budget, o.task_seed);
    };
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let weights = match &o.weights {
        Some(w) => {
            let t = std::fs::read_to_string(w).map_err(|e| format!("{w}: {e}"))?;
            serde_json::from_str(&t)
                .map_err(|e| format!("{w}: expected {{name: [values]}}: {e}"))?
        }
        None => Default::default(),
    };
    WeightSpace::from_graph(text, weights, o.budget, o.goal.clone())
}

fn abs_path(p: &str) -> Result<String, String> {
    std::path::absolute(p)
        .map(|p| p.display().to_string())
        .map_err(|e| format!("{p}: {e}"))
}

fn agent(o: &Opts) -> Result<(), String> {
    let out = o.out.as_deref().ok_or("--out <run dir> is required")?;
    let abs = |p: &std::path::Path| std::path::absolute(p).map_err(|e| e.to_string());
    let spec = RunSpec {
        cli: o.cli,
        out: abs(std::path::Path::new(out))?,
        wsp_exe: std::env::current_exe().map_err(|e| e.to_string())?,
        model: o.model.clone(),
        effort: o.effort.clone(),
        max_turns: o.max_turns,
        max_budget_usd: o.max_budget_usd,
        extra_args: o.passthrough.clone(),
    };
    if o.dry_run {
        let (prog, args) = command_line(&spec, "<prompt>");
        println!(
            "{prog} {}",
            args.iter()
                .map(|a| format!("{a:?}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        return Ok(());
    }
    let ws = if o.from_state.is_some() {
        branch_from_opts(o)?
    } else {
        space_from_opts(o)?
    };
    let prompt = setup(&spec, &ws)?;
    let task = match &ws.state.package {
        Some(p) => format!("package {p}"),
        None => format!("{:?} {:?}", ws.state.env, ws.state.mlp.widths),
    };
    eprintln!(
        "rlx-wsp agent: {} on {task}, budget {} forward passes -> {}",
        spec.cli.program(),
        ws.state.budget,
        spec.out.display()
    );
    let report = rlx_wsp::agent::run(&spec, &prompt, o.episodes)?;
    println!("{report:#}");
    Ok(())
}

/// The stages from a model and its questions to a comparison of methods
/// (see `rlx_wsp::pipeline`).
fn pipeline(o: &Opts) -> Result<(), String> {
    let out = abs_path(o.out.as_deref().ok_or("--out <dir> is required")?)?;
    let pkg = o.package.as_deref().ok_or("--package is required")?;
    let (Some(pr), Some(h)) = (&o.qa_practice, &o.qa_heldout) else {
        return Err("--qa-practice and --qa-heldout are required".into());
    };
    for st in o.stages.iter().chain(&o.redo) {
        if !rlx_wsp::pipeline::STAGES.contains(&st.as_str()) {
            return Err(format!(
                "no stage `{st}` (have: {})",
                rlx_wsp::pipeline::STAGES.join(", ")
            ));
        }
    }
    let given = |f: &str| o.given.contains(f);
    // What a comparison needs, unless told otherwise.
    let mut g = o.clone();
    g.check_perturbed = true;
    g.check_truncation = true;
    if !given("--dev-n") {
        g.dev_n = 50;
    }
    if !given("--keep") {
        g.keep_verified = true;
    }
    // The token budget is the agent's and the trainers', not the task's.
    g.token_budget = None;
    let mut task: Vec<String> = ["--package", pkg, "--qa-practice", pr, "--qa-heldout", h]
        .iter()
        .map(|s| s.to_string())
        .collect();
    task.extend([
        "--heldout-n".into(),
        o.heldout_n.to_string(),
        "--max-tokens".into(),
        o.max_tokens.to_string(),
        "--device".into(),
        o.device.clone(),
        "--dedupe".into(),
        o.dedupe.to_string(),
        "--budget".into(),
        "0".into(),
    ]);
    task.extend(qa_guard_args(&g));
    let mut agent = vec![
        "--cli".to_string(),
        o.cli.program().into(),
        "--budget".into(),
        o.budget.to_string(),
        "--max-turns".into(),
        o.max_turns.to_string(),
    ];
    let mut opt = |flag: &str, v: Option<String>| {
        if let Some(v) = v {
            agent.extend([flag.to_string(), v]);
        }
    };
    opt("--model", o.model.clone());
    opt("--effort", o.effort.clone());
    opt("--max-budget-usd", o.max_budget_usd.map(|b| b.to_string()));
    opt("--goal", o.goal.clone());
    opt("--token-budget", o.token_budget.map(|t| t.to_string()));
    opt(
        "--deny-tools",
        (!o.deny_tools.is_empty()).then(|| o.deny_tools.join(",")),
    );
    let distill = vec![
        "--shots".to_string(),
        o.shots.to_string(),
        "--questions".into(),
        (if given("--questions") {
            o.questions
        } else {
            512
        })
        .to_string(),
    ];
    let mut lora = vec![
        "--steps".to_string(),
        o.steps.unwrap_or(300).to_string(),
        "--gate".into(),
        o.gate.unwrap_or(3.0).to_string(),
        "--average".into(),
    ];
    let mut opt = |flag: &str, v: Option<String>| {
        if let Some(v) = v {
            lora.extend([flag.to_string(), v]);
        }
    };
    opt("--rank", o.rank.map(|r| r.to_string()));
    opt("--alpha", o.alpha.map(|a| a.to_string()));
    opt("--lr", o.lr.map(|l| l.to_string()));
    opt("--token-budget", o.token_budget.map(|t| t.to_string()));
    opt(
        "--targets",
        (!o.targets.is_empty()).then(|| o.targets.join(",")),
    );
    let plan = rlx_wsp::pipeline::Plan {
        out: out.into(),
        exe: std::env::current_exe().map_err(|e| e.to_string())?,
        task,
        agent,
        distill,
        lora,
        train_package: o.train_package.as_deref().map(abs_path).transpose()?,
        train_export: o.train_export.clone(),
        only: o.stages.clone(),
        redo: o.redo.clone(),
        heldout_n: o.heldout_n,
    };
    let r = rlx_wsp::pipeline::run(&plan)?;
    println!("{r:#}");
    let failed = r
        .as_object()
        .is_some_and(|m| m.values().any(|s| s["status"] == "failed"));
    if failed {
        return Err("a stage failed; run the same command again to retry it".into());
    }
    Ok(())
}

/// Run every `--agents` entry `--repeats` times in parallel, each as its own
/// `rlx-wsp agent` process, then tabulate the reports.
/// The QA guard and check flags, as given, for child runs.
fn qa_guard_args(o: &Opts) -> Vec<String> {
    let mut a = vec![
        "--dev-n".to_string(),
        o.dev_n.to_string(),
        "--dev-checks".into(),
        o.dev_checks.to_string(),
        "--prompt-style".into(),
        o.prompt_style.clone(),
        "--keep".into(),
        if o.keep_verified { "verified" } else { "final" }.into(),
    ];
    if let Some(g) = &o.qa_general {
        a.extend(["--qa-general".into(), g.clone()]);
    }
    if let Some(t) = o.token_budget {
        a.extend(["--token-budget".into(), t.to_string()]);
    }
    if let Some(m) = o.max_edit {
        a.extend(["--max-edit".into(), m.to_string()]);
    }
    if let Some(r) = o.repetition_penalty {
        a.extend(["--repetition-penalty".into(), r.to_string()]);
    }
    for (on, flag) in [
        (o.stop_at_answer, "--stop-at-answer"),
        (o.check_perturbed, "--check-perturbed"),
        (o.check_truncation, "--check-truncation"),
    ] {
        if on {
            a.push(flag.into());
        }
    }
    a
}

fn sweep(o: &Opts) -> Result<(), String> {
    let out = std::path::absolute(o.out.as_deref().ok_or("--out <sweep dir> is required")?)
        .map_err(|e| e.to_string())?;
    if o.agents.is_empty() {
        return Err("--agents cli[:model],... is required".into());
    }
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let word = |v: Value| v.as_str().unwrap_or_default().to_string();
    let env_name = word(serde_json::to_value(o.env).map_err(|e| e.to_string())?);
    let act_name = word(serde_json::to_value(o.act).map_err(|e| e.to_string())?);
    let hidden = o
        .hidden
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let mut children = Vec::new();
    let budgets = if o.budgets.is_empty() {
        vec![o.budget]
    } else {
        o.budgets.clone()
    };
    for &budget in &budgets {
        for (cli, model) in &o.agents {
            for i in 1..=o.repeats {
                let tag: String = model
                    .as_deref()
                    .map(|m| {
                        m.chars()
                            .map(|c| {
                                if c.is_ascii_alphanumeric() || c == '.' {
                                    c
                                } else {
                                    '-'
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_else(|| "default".into());
                let name = if o.budgets.is_empty() {
                    format!("{}-{tag}-{i}", cli.program())
                } else {
                    format!("{}-{tag}-b{budget}-{i}", cli.program())
                };
                let mut cmd = std::process::Command::new(&exe);
                cmd.args(["agent", "--cli", cli.program(), "--env", &env_name])
                    .args(["--hidden", &hidden, "--act", &act_name])
                    .args(["--budget", &budget.to_string()])
                    .args(["--task-seed", &(o.task_seed + i as u64 - 1).to_string()])
                    .args(["--episodes", &o.episodes.to_string()])
                    .args(["--max-turns", &o.max_turns.to_string()])
                    .arg("--out")
                    .arg(out.join(&name));
                if let Some(m) = model {
                    cmd.args(["--model", m]);
                }
                if let Some(e) = &o.effort {
                    cmd.args(["--effort", e]);
                }
                if let Some(b) = o.max_budget_usd {
                    cmd.args(["--max-budget-usd", &b.to_string()]);
                }
                if o.stochastic {
                    cmd.arg("--stochastic");
                }
                if o.oneshot {
                    cmd.arg("--oneshot");
                }
                if o.internals_only {
                    cmd.arg("--internals-only");
                }
                if o.open {
                    cmd.arg("--open");
                }
                if !o.deny_tools.is_empty() {
                    cmd.args(["--deny-tools", &o.deny_tools.join(",")]);
                }
                if let Some(g) = &o.goal {
                    cmd.args(["--goal", g]);
                }
                if let Some(c) = &o.score_cmd {
                    cmd.args(["--score-cmd", c]);
                }
                if let Some(p) = &o.package {
                    cmd.args(["--package", p]);
                    cmd.args(["--heldout-n", &o.heldout_n.to_string()]);
                    cmd.args(["--max-tokens", &o.max_tokens.to_string()]);
                    cmd.args(["--device", &o.device]);
                    if let (Some(pr), Some(h)) = (&o.qa_practice, &o.qa_heldout) {
                        cmd.args(["--qa-practice", pr, "--qa-heldout", h]);
                    }
                    cmd.args(qa_guard_args(o));
                }
                if !o.passthrough.is_empty() {
                    cmd.arg("--").args(&o.passthrough);
                }
                let log = std::fs::File::create(out.join(format!("{name}.progress")))
                    .map_err(|e| e.to_string())?;
                let child = cmd
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(log)
                    .spawn()
                    .map_err(|e| format!("{name}: {e}"))?;
                eprintln!("started {name} (pid {})", child.id());
                children.push((name, child));
            }
        }
    }
    let mut rows = Vec::new();
    for (name, mut child) in children {
        let status = child.wait().map_err(|e| e.to_string())?;
        let report: Option<Value> = std::fs::read_to_string(out.join(&name).join("report.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok());
        eprintln!("finished {name}: {status}");
        rows.push((name, report));
    }
    let mut md = format!(
        "# rlx-wsp sweep — {env_name}, budget {}, {}\n\n\
         | run | model | score | detail | fwd | edits | tool calls | minutes | cost |\n\
         |---|---|---:|---|---:|---:|---:|---:|---:|\n",
        o.budget,
        if o.env.is_classifier() {
            "held-out test set".to_string()
        } else {
            format!("{} held-out episodes", o.episodes)
        }
    );
    for (name, r) in &rows {
        let Some(r) = r else {
            md.push_str(&format!(
                "| {name} | — | failed (see {name}.progress) ||||||||\n"
            ));
            continue;
        };
        let a = &r["agent"];
        let models: Vec<String> = a["models"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|m| m.as_str().map(str::to_string))
            .collect();
        let calls: u64 = a["tool_calls"]
            .as_object()
            .map(|m| m.values().filter_map(Value::as_u64).sum())
            .unwrap_or(0);
        let num = |v: &Value| v.as_f64().map_or("—".to_string(), |x| format!("{x:.2}"));
        md.push_str(&format!(
            "| {name} | {} | {} | {} | {}/{} | {} | {calls} | {} | {} |\n",
            models.join(", "),
            num(&r["held_out"]["score"]),
            r["held_out"]["detail"].as_str().unwrap_or(""),
            r["forward_passes"]["used"],
            r["forward_passes"]["budget"],
            r["edits"],
            a["duration_ms"]
                .as_f64()
                .map_or("—".to_string(), |ms| format!("{:.1}", ms / 60_000.0)),
            a["cost_usd"]
                .as_f64()
                .map_or("—".to_string(), |c| format!("${c:.2}")),
        ));
    }
    md.push_str(&group_stats(&rows));
    std::fs::write(out.join("summary.md"), &md).map_err(|e| e.to_string())?;
    println!("{md}");
    Ok(())
}

/// Mean, spread and a bootstrap 95% interval of the score per agent and
/// budget, over repeats.
fn group_stats(rows: &[(String, Option<Value>)]) -> String {
    let mut groups: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    // QA runs: each run's score minus the unedited model's.
    let mut deltas: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    for (name, r) in rows {
        let Some(score) = r.as_ref().and_then(|r| r["held_out"]["score"].as_f64()) else {
            continue;
        };
        if let Some(u) = r
            .as_ref()
            .and_then(|r| r["held_out_unedited"]["score"].as_f64())
        {
            let key = name
                .rsplit_once('-')
                .map_or(name.as_str(), |(k, _)| k)
                .to_string();
            deltas.entry(key).or_default().push(score - u);
        }
        // Name is cli-model[-bN]-i: the group drops the repeat index.
        let key = name
            .rsplit_once('-')
            .map_or(name.as_str(), |(k, _)| k)
            .to_string();
        groups.entry(key).or_default().push(score);
    }
    if groups.values().all(|v| v.len() < 2) {
        return String::new();
    }
    let mut md = String::from(
        "\n## Over repeats\n\n| group | n | mean | sd | 95% CI (bootstrap) |\n|---|---:|---:|---:|---|\n",
    );
    let mut rng = rlx_ir::Philox4x32::new(0xB007);
    for (k, v) in groups {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let sd = if v.len() > 1 {
            (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
        } else {
            0.0
        };
        let mut means: Vec<f64> = (0..2000)
            .map(|_| {
                (0..v.len())
                    .map(|_| v[rng.next_u32() as usize % v.len()])
                    .sum::<f64>()
                    / n
            })
            .collect();
        means.sort_by(f64::total_cmp);
        md.push_str(&format!(
            "| {k} | {} | {mean:.2} | {sd:.2} | {:.2} – {:.2} |\n",
            v.len(),
            means[50],
            means[1949]
        ));
    }
    if !deltas.is_empty() {
        md.push_str(
            "\n### Against the unedited model\n\n| group | n | mean change | 95% CI (bootstrap) | runs above |\n|---|---:|---:|---|---:|\n",
        );
        for (k, v) in deltas {
            let n = v.len() as f64;
            let mean = v.iter().sum::<f64>() / n;
            let mut means: Vec<f64> = (0..2000)
                .map(|_| {
                    (0..v.len())
                        .map(|_| v[rng.next_u32() as usize % v.len()])
                        .sum::<f64>()
                        / n
                })
                .collect();
            means.sort_by(f64::total_cmp);
            md.push_str(&format!(
                "| {k} | {} | {mean:+.2} | {:+.2} – {:+.2} | {}/{} |\n",
                v.len(),
                means[50],
                means[1949],
                v.iter().filter(|d| **d > 0.0).count(),
                v.len()
            ));
        }
    }
    md
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = args.split_first().ok_or(USAGE)?;
    let o = parse(rest)?;
    match cmd.as_str() {
        "tools" => println!("{:#}", tool_schemas()),
        "new" => {
            let path = need_state(&o)?;
            let ws = space_from_opts(&o)?;
            save(path, &ws)?;
            match ws.store() {
                Some(st) => println!(
                    "{path}: package {} ({} tensors), budget {}",
                    st.path,
                    st.names().count(),
                    o.budget
                ),
                None => println!(
                    "{path}: {:?} widths {:?}, {} weights, budget {}",
                    o.env,
                    ws.state.mlp.widths,
                    ws.state.mlp.n_weights(),
                    o.budget
                ),
            }
        }
        "weights" => {
            let path = need_state(&o)?;
            let out = o.out.as_deref().ok_or("--out <file> is required")?;
            let ws = load(path)?;
            let weights: std::collections::BTreeMap<&str, &[f32]> = ws
                .state
                .params
                .iter()
                .map(|(n, t)| (n.as_str(), t.data.as_slice()))
                .collect();
            let text = serde_json::to_string(&weights).map_err(|e| e.to_string())?;
            std::fs::write(out, text).map_err(|e| format!("{out}: {e}"))?;
            println!("{out}: {} tensors", weights.len());
        }
        "call" => {
            let path = need_state(&o)?;
            let mut ws = load(path)?;
            let body = match o.rest.first().map(String::as_str) {
                Some("-") | None => {
                    let mut s = String::new();
                    std::io::stdin()
                        .read_to_string(&mut s)
                        .map_err(|e| e.to_string())?;
                    s
                }
                Some(s) => s.to_string(),
            };
            let reply = handle_json(&mut ws, &body);
            save(path, &ws)?;
            println!("{reply:#}");
        }
        "serve" => {
            let mut ws = match &o.state {
                Some(p) if std::path::Path::new(p).exists() => load(p)?,
                _ => WeightSpace::new(o.env, &o.hidden, o.act, o.budget),
            };
            let stdout = std::io::stdout();
            for line in std::io::stdin().lock().lines() {
                let line = line.map_err(|e| e.to_string())?;
                if line.trim().is_empty() {
                    continue;
                }
                let reply = handle_json(&mut ws, &line);
                let mut out = stdout.lock();
                writeln!(out, "{reply}").map_err(|e| e.to_string())?;
                out.flush().map_err(|e| e.to_string())?;
                if let Some(p) = &o.state {
                    save(p, &ws)?;
                }
            }
        }
        "score" => {
            let mut ws = load(need_state(&o)?)?;
            let mut checks = serde_json::Value::Null;
            if o.check_perturbed || o.check_truncation {
                if let Some(q) = ws.state.qa.as_mut() {
                    q.check_perturbed |= o.check_perturbed;
                    q.check_truncation |= o.check_truncation;
                }
                checks = rlx_wsp::checks::report_checks(&mut ws)?;
                println!("{checks:#}");
            }
            let h = ws.held_out(o.episodes)?;
            println!(
                "held-out score {:.2} ({}), forward passes used {}/{}",
                h["score"].as_f64().unwrap_or(0.0),
                h["detail"].as_str().unwrap_or(""),
                ws.state.used,
                ws.state.budget
            );
            // `--out report.json`: the scores a run's report carries, for a
            // run whose scoring did not finish.
            if let Some(out) = &o.out {
                let untouched = ws.state.params.is_empty()
                    && ws.state.rewrites.is_empty()
                    && ws.state.lowrank.is_empty();
                let u = if untouched {
                    h.clone()
                } else {
                    rlx_wsp::checks::unedited_held_out(&mut ws)?
                };
                println!("unedited: {}", u["detail"].as_str().unwrap_or(""));
                let report = serde_json::json!({
                    "held_out_final": h, "held_out_unedited": u, "checks": checks,
                    "forward_passes": { "budget": ws.state.budget, "used": ws.state.used },
                });
                std::fs::write(out, format!("{report:#}\n")).map_err(|e| format!("{out}: {e}"))?;
            }
        }
        "bench" => bench(&o),
        "mcp" => {
            let mut ws = match &o.state {
                Some(p) if std::path::Path::new(p).exists() => load(p)?,
                _ => WeightSpace::new(o.env, &o.hidden, o.act, o.budget),
            };
            let state = o.state.clone();
            rlx_wsp::mcp::serve(&mut ws, |ws| match &state {
                Some(p) => save(p, ws),
                None => Ok(()),
            })?;
        }
        "agent" => agent(&o)?,
        "pipeline" => pipeline(&o)?,
        "finish" => {
            // A run stopped after its agent ended: score what it left.
            let dir = o.run.as_deref().ok_or("--run <run dir> is required")?;
            let dir = std::path::absolute(dir).map_err(|e| e.to_string())?;
            let report = rlx_wsp::agent::finish(&dir, o.cli, o.episodes)?;
            println!("{:#}", report["held_out"]);
        }
        "sweep" => sweep(&o)?,
        "graft" => {
            // Graft units from another state file (any architecture) into --state.
            let path = need_state(&o)?;
            let mut ws = load(path)?;
            let src = load(
                o.from_state
                    .as_deref()
                    .ok_or("--from-state <file> is required")?,
            )?;
            ws.state.networks.insert("donor".into(), src.state.params);
            let r = ws.graft("donor", &o.units, &o.into);
            ws.state.networks.remove("donor");
            let r = r?;
            save(path, &ws)?;
            println!("{r:#}");
        }
        "bundle" => {
            let run = o.run.as_deref().ok_or("--run <run dir> is required")?;
            let out = o.out.as_deref().ok_or("--out <dir> is required")?;
            let r = rlx_wsp::bundle::bundle(
                std::path::Path::new(run),
                std::path::Path::new(out),
                o.episodes,
            )?;
            println!("{r:#}");
        }
        "rl-baseline" => {
            let protocol = if o.stochastic {
                rlx_wsp::space::Protocol::STOCHASTIC
            } else {
                rlx_wsp::space::Protocol::GREEDY
            };
            let d = rlx_wsp::pg::PgOptions::default();
            let opts = rlx_wsp::pg::PgOptions {
                iters: o.iters,
                seeds: o.seeds,
                eval_episodes: o.episodes.min(256),
                lr: o.lr.unwrap_or(d.lr),
                gamma: o.gamma.unwrap_or(d.gamma),
                ..d
            };
            let r = match o.method.as_str() {
                "ppo" => rlx_wsp::pg::train_ppo_baseline(o.env, &o.hidden, o.act, protocol, opts)?,
                "reinforce" => {
                    rlx_wsp::pg::train_baseline(o.env, &o.hidden, o.act, protocol, opts)?
                }
                m => return Err(format!("unknown --method `{m}` (ppo | reinforce)")),
            };
            println!("{r:#}");
        }
        "warmstart" => {
            let ws = load(need_state(&o)?)?;
            let opts = rlx_wsp::warmstart::WarmstartOptions {
                epochs: o.epochs,
                ..Default::default()
            };
            let r = rlx_wsp::warmstart::warmstart(&ws, opts)?;
            println!("{r:#}");
        }
        "verify" => {
            // Re-check a finished run: unit importance vs the agent's account.
            let state = need_state(&o)?;
            let mut ws = load(state)?;
            let report = std::path::Path::new(state).with_file_name("report.json");
            let text = std::fs::read_to_string(&report)
                .ok()
                .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                .and_then(|r| r["agent"]["final_text"].as_str().map(str::to_string))
                .unwrap_or_default();
            let r = rlx_wsp::verify::check_run(&mut ws, &text, o.episodes.min(128))?;
            println!("{r:#}");
        }
        "job" => {
            // A background job's child (see rlx_wsp::jobs).
            let state = need_state(&o)?;
            let out = o.out.as_deref().ok_or("--out <job dir> is required")?;
            let call = o.rest.first().ok_or("give the call JSON")?;
            rlx_wsp::jobs::run_child(std::path::Path::new(state), std::path::Path::new(out), call)?;
        }
        "branch" => {
            let ws = branch_from_opts(&o)?;
            let out = o
                .out
                .as_deref()
                .ok_or("--out <run dir | state.json> is required")?;
            let target = if out.ends_with(".json") {
                std::path::PathBuf::from(out)
            } else {
                std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
                std::path::Path::new(out).join("state.json")
            };
            if target.exists() {
                return Err(format!("{} exists; pick a new --out", target.display()));
            }
            save(&target.display().to_string(), &ws)?;
            let last = ws.state.lineage.last().cloned().unwrap_or_default();
            println!(
                "{}: branched from {} at `{}` ({} forward passes used there); budget {}, checkpoints {:?}",
                target.display(),
                last["from"].as_str().unwrap_or(""),
                last["version"].as_str().unwrap_or(""),
                last["forward_passes_used"],
                ws.state.budget,
                ws.state.checkpoints.keys().collect::<Vec<_>>()
            );
        }
        "qa-hygiene" => {
            let (p, h) = (
                o.qa_practice
                    .as_deref()
                    .ok_or("--qa-practice is required")?,
                o.qa_heldout.as_deref().ok_or("--qa-heldout is required")?,
            );
            let pi = rlx_wsp::lm::load_items(p)?;
            let hi = rlx_wsp::lm::load_items(h)?;
            let hi = &hi[..o.heldout_n.min(hi.len())];
            let d = rlx_wsp::checks::near_duplicates(&pi, hi, o.dedupe);
            let twins = rlx_wsp::checks::perturbed_set(hi, 7);
            println!(
                "{:#}",
                serde_json::json!({
                    "practice": pi.len(), "heldout": hi.len(), "threshold": o.dedupe,
                    "near_duplicates": d.iter().map(|(a, b, j)| serde_json::json!({
                        "practice": a, "heldout": b, "jaccard": j,
                        "practice_question": pi[*a].question, "heldout_question": hi[*b].question,
                    })).collect::<Vec<_>>(),
                    "heldout_perturbable": twins.len(),
                    "example_twins": twins.iter().take(3).map(|(i, t)| serde_json::json!({
                        "original": hi[*i].question, "gold": hi[*i].gold,
                        "perturbed": t.question, "perturbed_gold": t.gold,
                    })).collect::<Vec<_>>(),
                })
            );
        }
        "explore-chart" => {
            // How a finished (or running) run explored, as SVG.
            let dir = o
                .run
                .as_deref()
                .or(o.state.as_deref())
                .ok_or("--run <run dir> is required")?;
            let dir = std::path::Path::new(dir);
            let title = format!(
                "How {} explored the weight space",
                dir.file_name()
                    .map_or_else(|| "the run".into(), |n| n.to_string_lossy().to_string())
            );
            let svg = rlx_wsp::explore::write_chart(dir, &title)?;
            match &o.out {
                Some(out) => {
                    std::fs::copy(&svg, out).map_err(|e| format!("{out}: {e}"))?;
                    println!("{out}");
                }
                None => println!("{}", svg.display()),
            }
        }
        "qa-distill" => {
            // Correct answers given with worked examples in front, as rows
            // to train on with the plain prompt (`lora-train --train-items`).
            let mut ws = load(need_state(&o)?)?;
            let out = o.out.as_deref().ok_or("--out <file.jsonl> is required")?;
            let mut f = std::io::BufWriter::new(
                std::fs::File::create(out).map_err(|e| format!("{out}: {e}"))?,
            );
            let summary =
                rlx_wsp::lora::distill(&mut ws, o.shots, o.questions, o.offset, &mut |row| {
                    use std::io::Write;
                    writeln!(f, "{row}")
                        .and_then(|_| f.flush())
                        .map_err(|e| e.to_string())
                })?;
            println!("{summary:#}");
        }
        "lora-train" => {
            let mut ws = load(need_state(&o)?)?;
            let out = o.out.as_deref().ok_or("--out <dir> is required")?;
            let d = rlx_wsp::lora::LoraOptions::default();
            let method = if o.method == "ppo" {
                "mezo"
            } else {
                o.method.as_str()
            };
            let opts = rlx_wsp::lora::LoraOptions {
                targets: if o.targets.is_empty() {
                    d.targets.clone()
                } else {
                    o.targets.clone()
                },
                rank: o.rank.unwrap_or(d.rank),
                alpha: o.alpha.unwrap_or(d.alpha),
                steps: o.steps.unwrap_or(d.steps),
                lr: o
                    .lr
                    .unwrap_or(if method == "backprop" { 1e-4 } else { d.lr }),
                batch: o.batch.unwrap_or(d.batch),
                eps: o.eps.unwrap_or(d.eps),
                mode: o.mode.clone(),
                eval_every: d.eval_every,
                seed: o.task_seed,
                max_tokens: o.token_budget,
                gate: o.gate,
                average: o.average,
                snapshot_every: o.snapshot_every.unwrap_or(d.snapshot_every),
                train_items: o.train_items.as_deref().map(abs_path).transpose()?,
            };
            let record = match method {
                "mezo" => rlx_wsp::lora::train_mezo(&mut ws, &opts)?,
                "backprop" => {
                    let tp = o
                        .train_package
                        .as_deref()
                        .ok_or("--train-package <trainable .rlxp> is required (rlx-export … --variant train)")?;
                    #[cfg(feature = "training")]
                    {
                        rlx_wsp::lora::train_backprop(&mut ws, &abs_path(tp)?, &opts)?
                    }
                    #[cfg(not(feature = "training"))]
                    {
                        let _ = tp;
                        return Err(
                            "backprop LoRA needs rlx-wsp built with --features training".into()
                        );
                    }
                }
                m => return Err(format!("--method `{m}`: mezo | backprop")),
            };
            let dir = std::path::Path::new(out);
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            let state = dir.join("state.json");
            save(&state.display().to_string(), &ws)?;
            let mut files = vec!["state.json".to_string()];
            if ws.store().is_some() {
                let a = ws.model_path()?;
                std::fs::copy(&a, dir.join("adapter.rlxp")).map_err(|e| e.to_string())?;
                files.push("adapter.rlxp".into());
            }
            let record = serde_json::json!({ "training": record, "files": files,
                "score_with": format!("rlx-wsp score --state {}", state.display()) });
            std::fs::write(dir.join("lora.json"), format!("{record:#}"))
                .map_err(|e| e.to_string())?;
            println!("{record:#}");
        }
        "run" => {
            // Generate with a space's model (overlay and rewrites applied), or
            // with any package or adapter package.
            if o.prompts.is_empty() {
                return Err("give --prompt \"…\" (repeatable)".into());
            }
            if let Some(path) = o.state.as_deref() {
                let mut ws = load(path)?;
                let outs = ws.lm_generate(&o.prompts, o.max_tokens)?;
                for (p, c) in o.prompts.iter().zip(outs) {
                    println!("=== {p}\n{c}");
                }
            } else {
                let pkg = o.package.as_deref().ok_or("give --state or --package")?;
                let mut m = rlx_runtime::model::open_source(
                    &abs_path(pkg)?,
                    serde_json::json!({ "device": o.device }),
                )
                .map_err(|e| format!("{pkg}: {e:#}"))?;
                for p in &o.prompts {
                    let c = m
                        .generate_text(
                            p,
                            &serde_json::json!({ "max_tokens": o.max_tokens, "temperature": 0.0 }),
                        )
                        .map_err(|e| format!("{e:#}"))?;
                    println!("=== {p}\n{c}");
                }
            }
        }
        "export" => {
            let mut ws = load(need_state(&o)?)?;
            let out = o.out.as_deref().ok_or("--out <dir> is required")?;
            let r =
                rlx_wsp::export::export(&mut ws, std::path::Path::new(out), o.fpga, o.episodes)?;
            println!("{r:#}");
        }
        _ => return Err(USAGE.into()),
    }
    Ok(())
}

const USAGE: &str = "usage: rlx-wsp <tools|new|call|weights|serve|score|bench|mcp|agent|finish|sweep|warmstart|export|verify|rl-baseline|explore-chart|qa-distill|lora-train|run|branch|qa-hygiene|bundle|graft> [options] (see the crate README)";

/// `--detach`: run this command in a session of its own, its output in
/// `--log <file>` (else `<--out>.log`), so ending the shell or the agent
/// session that started it does not end it.
fn detach() -> Result<bool, String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let end = args.iter().position(|a| a == "--").unwrap_or(args.len());
    let Some(i) = args[..end].iter().position(|a| a == "--detach") else {
        return Ok(false);
    };
    args.remove(i);
    let end = args.iter().position(|a| a == "--").unwrap_or(args.len());
    let value_of = |flag: &str| {
        let i = args[..end].iter().position(|a| a == flag)?;
        args.get(i + 1).cloned()
    };
    let log = value_of("--log")
        .or_else(|| value_of("--out").map(|o| format!("{}.log", o.trim_end_matches('/'))))
        .unwrap_or_else(|| format!("rlx-wsp-{}.log", std::process::id()));
    if let Some(i) = args[..end].iter().position(|a| a == "--log") {
        args.drain(i..i + 2);
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log)
        .map_err(|e| format!("{log}: {e}"))?;
    let mut child = std::process::Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
    child
        .args(&args)
        .stdin(std::process::Stdio::null())
        .stdout(file.try_clone().map_err(|e| e.to_string())?)
        .stderr(file);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe extern "C" {
            fn setsid() -> i32;
        }
        // SAFETY: `setsid` is async-signal-safe and touches no Rust state.
        unsafe {
            child.pre_exec(|| {
                setsid();
                Ok(())
            });
        }
    }
    let c = child.spawn().map_err(|e| format!("detaching: {e}"))?;
    println!("detached: pid {}, log {log}", c.id());
    Ok(true)
}

fn main() {
    match detach() {
        Ok(true) => return,
        Ok(false) => {}
        Err(e) => {
            eprintln!("rlx-wsp: {e}");
            std::process::exit(2);
        }
    }
    if let Err(e) = run() {
        eprintln!("rlx-wsp: {e}");
        std::process::exit(2);
    }
}
