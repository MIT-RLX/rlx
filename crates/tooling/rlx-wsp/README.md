# rlx-wsp

**Weight-space programming harness.** An LLM programs a network by reading and
writing its weights directly, reasoning about what each unit computes, instead
of training it. `rlx-wsp` is the sandbox: the weights of an rlx graph as a
labelled, checkpointed store, a JSON tool surface an agent loop or MCP server
can register as-is, and a **forward-pass budget** that is the agent's only way
to run the network.

```
 describe ──▶ read (stats / slice / topk / norms / svd) ──▶ edit (set / set_col / add_rank1 …)
                 ▲                                              │
                 └──── probe / rollout  (1 forward pass each) ◀─┘
```

- Reading and editing are free; `probe` (≤32 hand-written observations, returns
  logits *and* hidden activations) and `rollout` (a batch of episodes) each
  cost one forward pass.
- Every row/column is addressable by **label**: `ball>(x=3,y=5)`,
  `paddle(x=4)`, `l0.h7`, `right`. Views are sized for a context window (raw
  slices capped at 4096 values; otherwise summaries).
- The network is an rlx graph compiled once; edits upload through
  `CompiledGraph::set_param` with no recompile.
- The held-out score (`rlx-wsp score`) uses a seed the agent's rollouts never
  see and is not a tool.

## Experiment 1 — Bounce, weights written from zero

A ball falls one row per step and moves diagonally, reflecting off the side
walls; a width-1 paddle must be under it when it lands (+1 catch, −1 miss).
Policy: MLP `210 → 32 → 32 → 3`, tanh, **all weights start at zero**.

```sh
cargo run -p rlx-wsp --release -- bench --budget 50
```

| program | return | catch | forward passes |
|---|---:|---:|---:|
| zero weights | −17.55 | 10.1% | 0 |
| random init (He) | −18.27 | 8.5% | 0 |
| (1+1)-ES from zero | −15.55 | 14.7% | 50 |
| (1+1)-ES from zero | −4.36 | 40.1% | 2000 |
| written: follow current ball x | −22.00 | 0.0% | 0 |
| written: steer to landing column | **22.00** | **100%** | 0 |

The written program (`baselines::bounce_landing_program`, built only from tool
calls) uses 2 units per layer: layer 0 computes `landing_x − paddle_x` (the
landing column of every ball cell is a constant, so this is linear in the
one-hot observation), layer 1 relays it, layer 2 maps its sign to
left / stay / right. Following the ball's *current* column scores below
standing still: the paddle moves first, so it ends each step one column behind
a diagonal ball.

This table is the harness's reference, not an LLM result: the program was
written by the same author as the environment. The experiment proper is an
agent that gets `describe` and a budget, nothing else.

## Driving it

```sh
rlx-wsp tools                                   # tool schemas: [{name, description, input_schema}]
rlx-wsp new   --state ws.json --budget 50       # zero weights
rlx-wsp call  --state ws.json '{"tool":"describe"}'
rlx-wsp call  --state ws.json '{"tool":"probe","observations":[{"active":["ball>(x=8,y=0)","paddle(x=5)"]}]}'
rlx-wsp call  --state ws.json '{"tool":"rollout","episodes":32,"trace_steps":10}'
rlx-wsp score --state ws.json                   # held-out, harness only
rlx-wsp serve --state ws.json                   # JSON lines on stdin/stdout
```

Replies are `{"ok": …}` or `{"error": "…"}`. The `--state` file holds weights,
checkpoints, budget and the edit log, so one-shot `call`s compose across
processes.

## Running an LLM against it

The LLM is an **agent CLI**; `rlx-wsp` makes no API requests itself. `rlx-wsp
mcp` serves the tools over stdio MCP, and `rlx-wsp agent` launches a CLI
against it, then scores the result on held-out episodes:

```sh
cargo build -p rlx-wsp --release
target/release/rlx-wsp agent --cli claude --out runs/claude-1 --budget 50 --effort high
target/release/rlx-wsp agent --cli agent   --out runs/cursor-1 --model sonnet-4
target/release/rlx-wsp agent --cli copilot --out runs/copilot-1
target/release/rlx-wsp agent --cli claude --out x --dry-run      # print the command line only
```

| `--cli` | launches | MCP config | tool restriction |
|---|---|---|---|
| `claude` | `claude -p` (stream-json) | `--mcp-config` + `--strict-mcp-config` | `--tools ""` (no built-ins) + `--allowedTools mcp__wsp`, `dontAsk` |
| `agent` | Cursor `agent -p` (stream-json) | `workspace/.cursor/mcp.json`, `--approve-mcps` | shell stays available in the sandboxed, empty workspace |
| `copilot` | `copilot -p` (JSONL) | `--additional-mcp-config @mcp.json` | `--allow-tool wsp`, shell and write denied |

`--model`, `--max-turns` (claude, default 80), `--effort` and
`--max-budget-usd` (claude only) are forwarded; anything after `--` is
appended verbatim. A new CLI is one `AgentCli` variant in `src/agent.rs`.

Each run writes `<out>/{state.json, mcp.json, prompt.txt, transcript.jsonl,
report.json}` and runs the CLI in the empty `<out>/workspace/`. The prompt
gives the goal and the budget only; rules and layout come from `describe`.
The held-out seed is not reachable through any tool. With `--cli claude` the
agent has no file or shell access, so it cannot read the state file or this
repository (including `baselines.rs`). The Cursor and Copilot launch lines are
checked by tests and `--dry-run`, but have not been run live.

## Tasks

| `--env` | what the agent does | reference points (held-out) |
|---|---|---|
| `bounce` | write a policy from zero weights; one ball | written landing program +22.00 (max) |
| `orchard` | write; fruit + bombs, up to 4 objects at once | lookahead planner +26.52, (1+1)-ES@2000 +5.58 |
| `gusts` | write; Orchard with **hidden** motion (wrapping walls, 2 cols/step from row 5, bombs fall 2 rows) | planner with true rules +27.02, planner assuming Orchard's rules −7.05 |
| `mnist-fix` | repair a backdoored MNIST MLP (trigger, place and target not told) | tampered −3.38, clean-trained +96.89, zero-trigger-rows +96.37 (score = clean acc − attack success, pp) |
| `mnist-fix-hard` | repair; the trigger is a 4×4 checkerboard in the image centre, where digits have ink | tampered −3.82, zero-trigger-rows 85.15, ablate 6 top trigger units 92.44, clean-trained 96.91 |
| `gusts-hard` | write; Gusts' hidden motion on a 16×12 grid, up to 6 objects, 45% spawn — not every fruit is reachable and one unit per (column, action) no longer fits in 32 | planner with true rules +29.43, planner assuming Orchard's rules −6.20, (1+1)-ES@2000 +1.06, zero −1.26 |
| `read-rules` | READ: which of 3 motion rule sets (reflect / wrap / Gusts) a copy-trained policy learned; no rollouts; `answer` A/B/C | chance 33%; the policies play their own variant at 21–22 (planners 26.5–27.0) |
| `read-digit` | READ: which digit an MNIST model never saw; `answer` 0–9 | chance 10% |
| `audit-pair` | AUDIT: two MNIST models, one backdoored (centre checkerboard); `switch` between them, `answer` A/B | chance 50% |
| `mnist-steer` | STEER: every 3 must read as 8, other digits unchanged; score = mean of the two rates | clean model 48.93; logit rewrite `logit8 += 1.5·logit3`, drop 3: 97.64 (easy) |
| `--graph m.rlxmodel` | any rlx graph + `--weights w.json` + `--goal "..."`; no harness score | — |

`rlx-wsp bench --env <task>` prints the reference rows. `mnist-fix` reads the
IDX files from `$RLX_MNIST_DIR` (default `$HOME/.mnist`) and trains the
tampered model once (plain Rust, cached under `$HOME/.cache/rlx-wsp/`).

## Tools

Free: `describe`, `status`, `list_params`, `stats`, `slice`, `topk`, `norms`,
`svd`, `compose`, `list_nodes`, `sample_images`, edits (`set`, `set_col`,
`set_row`, `write`, `fill`, `scale`, `add_rank1`), `checkpoint` / `revert` /
`diff`, `log`, `script`.
Budgeted: `probe` (1), `rollout` (1), `eval` (1), `trace` (1), `compare` (2),
`ablate` (2).

- **`trace`** — run once and return any named node (MLP: `l0.z`, `l0.h`, …,
  `logits`; loaded graphs: node names or `%id`), via a graph rewrite that
  makes them extra outputs.
- **`compose`** — summarise `A·B` of two params (svd / topk / norms / stats),
  e.g. `l0.w·l1.w` maps inputs to layer-1 units.
- **`compare`** — paired A/B of two checkpoints on the same episodes / digits:
  mean difference, standard error, win counts.
- **`ablate`** — switch units off and compare on the same episodes.
- **`script`** — JavaScript in the embedded `quickrs` engine with no file,
  network, timer or host access and a 5·10⁸-instruction budget; reads params
  via `read`, returns JSON, and a returned `{calls: [...]}` is applied.

## After a run

```sh
rlx-wsp verify    --state run/state.json          # per-unit ablation vs the agent's own account
rlx-wsp warmstart --state run/state.json          # SGD from the agent's weights vs random
rlx-wsp export    --state run/state.json --out x [--fpga]   # .rlxmodel, int8 check, SystemVerilog
```

- `agent` reports include `claim_check` automatically: important units the
  agent never named, and named units that do nothing (heuristic text parse).
- `warmstart` behaviour-clones the task's reference controller. On Claude's
  Orchard program (6 epochs, 12.8k samples): teacher agreement 0.45 → 0.96
  from the agent's weights vs 0.37 → 0.73 from random; held-out return
  26.3 → 1.0 after the first epoch, back to 20.5 by epoch 6 (random: 2.9).
- `export` bakes the params into `Op::Constant`s of a batch-1 graph. Claude's
  Bounce program: 400 / 7907 nonzero, int8 per-tensor keeps +22.00. With
  `--features fpga`, ReLU networks go through `rlx_runtime::export`
  (the repaired MNIST model: DSP 6, LUT ≈ 656, 56k cycles); `rlx-fpga` has no
  tanh, so tanh programs stop at the `.rlxmodel`.

## Protocol, modes, baselines

- `--stochastic` — score as the post does: sampled actions and sticky actions
  (p = 0.1). Default is greedy. Programs need logit margins to survive it.
- `--oneshot` — budget 0 and only `describe` / `list_params` / write tools
  (no reads of runs, no probes, no `script`), the closest a tool-using CLI
  gets to "weights in one reply".
- `--task-seed N` — picks the hidden answer of reading / audit tasks; `sweep`
  gives repeat `i` seed `N + i − 1`.
- `sweep --budgets 0,10,25,50` — the post's budget curves.
- `rl-baseline --env <task> [--iters 1000] [--seeds 3] [--lr 0.05] [--gamma 0.9]`
  — REINFORCE with a mean baseline from random weights, best of seeds (a
  simpler stand-in for the post's PPO). Orchard: 10.16 after 3.2M steps;
  Bounce stalls at −0.6.
- `bundle --run <run dir> --out <dir>` — prompt, transcript, report, final
  state, baked `.rlxmodel` and a README; run paths and `$HOME` scrubbed.
- `verify` adds measured unit roles (the game feature each unit tracks over
  the network's own play) and checks the agent's text names them.

## Building blocks

- `inject {units, spec: {block, ...}}` writes a predefined block into units of
  one layer, activation-aware: `relay`, `threshold`, `and` / `or` (`levels`
  `01` or `pm1`), `gate` (signal passes only while a 0/1 input is on),
  `compare` (units `[a>b, b>a]`). Output labels address the logit layer.
- `insert_layer {after, width?}` adds a hidden layer that relays its input
  (exact for ReLU, ~1e-4 for tanh) so it can be programmed.
- `project_out` removes a direction exactly (rows or columns, optionally only
  some); `graft` copies units from a checkpoint or another network; `switch`
  changes the active network; `answer` submits a reading / audit answer.
