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
        passthrough: Vec::new(),
        rest: Vec::new(),
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
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
    let text = serde_json::to_string(&ws.state).map_err(|e| e.to_string())?;
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
        let units: Vec<String> = order[..k].iter().map(|u| format!("l0.h{u}")).collect();
        let off = ws.ablated(&units)?;
        ws.set_params(off);
        let h = ws.held_out(o.episodes)?;
        let sc = h["score"].as_f64().unwrap_or(f64::NEG_INFINITY);
        if sc > best.0 {
            best = (sc, k, h["detail"].as_str().unwrap_or("").to_string());
        }
        ws.set_params(tampered.clone());
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
fn space_from_opts(o: &Opts) -> Result<WeightSpace, String> {
    let mut ws = space_base(o)?;
    if o.stochastic {
        ws.state.protocol = rlx_wsp::space::Protocol::STOCHASTIC;
    }
    ws.state.internals_only = o.internals_only;
    ws.state.open = o.open;
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

fn space_base(o: &Opts) -> Result<WeightSpace, String> {
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
    let ws = space_from_opts(o)?;
    let prompt = setup(&spec, &ws)?;
    eprintln!(
        "rlx-wsp agent: {} on {:?} {:?}, budget {} forward passes -> {}",
        spec.cli.program(),
        o.env,
        ws.state.mlp.widths,
        o.budget,
        spec.out.display()
    );
    let report = rlx_wsp::agent::run(&spec, &prompt, o.episodes)?;
    println!("{report:#}");
    Ok(())
}

/// Run every `--agents` entry `--repeats` times in parallel, each as its own
/// `rlx-wsp agent` process, then tabulate the reports.
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
    std::fs::write(out.join("summary.md"), &md).map_err(|e| e.to_string())?;
    println!("{md}");
    Ok(())
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
            println!(
                "{path}: {:?} widths {:?}, {} weights, budget {}",
                o.env,
                ws.state.mlp.widths,
                ws.state.mlp.n_weights(),
                o.budget
            );
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
            let h = ws.held_out(o.episodes)?;
            println!(
                "held-out score {:.2} ({}), forward passes used {}/{}",
                h["score"].as_f64().unwrap_or(0.0),
                h["detail"].as_str().unwrap_or(""),
                ws.state.used,
                ws.state.budget
            );
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

const USAGE: &str = "usage: rlx-wsp <tools|new|call|serve|score|bench|mcp|agent|sweep|warmstart|export|verify|rl-baseline|bundle|graft> [options] (see the crate README)";

fn main() {
    if let Err(e) = run() {
        eprintln!("rlx-wsp: {e}");
        std::process::exit(2);
    }
}
