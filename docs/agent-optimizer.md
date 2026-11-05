# The agent optimizer

A coding agent rewrites the optimizer while the model trains: the protocol of
OPAL (*Reactive Optimizers*, Bai et al., [arXiv:2610.33826](https://arxiv.org/abs/2610.33826))
over RLX's `Trainer`. Code: `rlx_bind::runtime::agent` (feature `agent`),
`pyrlx.AgentOptimizer`, the `rlx-agent-opt` command, and
[`rlx-agent`](../crates/tooling/rlx-agent/README.md) for launching the agent CLI.

## How a run goes

Every update is made by the current **program**. At scheduled **decisions**
(and when the loss turns non-finite, or the agent's own analysis asks), the run
writes what it has seen into a workspace folder and asks the agent, which
answers by editing the program files there. A changed updater takes over at an
update boundary and runs **on trial** for `trial_steps` updates:

| failure | what happens |
|---|---|
| fails to build | rejected; the running program stays |
| fails at its first update | that update is undone exactly (from a copy of the params, up to `snapshot_limit_gb`) and made by the previous program |
| fails later in the trial | the previous program takes over; the failing update is not undone |
| fails after its trial | the previous program is rebuilt from its files (its optimizer state from zero) |
| no earlier program | the step returns an error |

The agent sees every error at its next decision. With no agent (`backend
"keep"`) a run is the plain `Trainer` run bit for bit, learning-rate schedule
included — check that before spending money.

## Programs

The agent edits two files:

- **The updater.** `program="rules"`: `updater.json` assigns params to groups
  (globs, optional `ndim`), each group to one of rlx-optim's algorithms
  (`adamw`, `muon`, `muon_adamw`, `soap`, …) with its hyperparameters and an
  `lr_mult`, puts a piecewise-linear multiplier on the learning rate over the
  run, and may skip updates (`skip_if`: loss or gradient norm above a bound).
  Runs natively; nothing the agent writes executes. `program="python"`
  (pyrlx): `updater.py`, a module whose `build(params, previous, spec)`
  returns an object with `step(params, grads, ctx)` over float32 numpy arrays —
  any update rule, run inside the training process (a rule, not a sandbox).
  `reference/example_updater.json` (or `.py`) uses every feature at values that
  make it the starting program: the form, not a recommendation (with
  `prompt="aggressive"`, the Python one adds a learning-rate controller, off).
  Before a Python program is built, its source is read for what it reaches
  past the program (files, `eval`, `sys.modules`, `__main__`, `gc`, frames, the
  model): each finding is a `boundary` event, a count in `results.json` and a
  banner in the report. Findings never stop a program, and this is no sandbox.
- **The analysis.** `analysis.json`: which statistics the reports hold
  (`grad_norm`, `weight_norm`, `update_ratio`, `grad_cosine`, activations of
  the graph's further outputs), how often, and `wake` rules that ask for a
  decision as soon as a statistic crosses a bound. A change to it alone does
  not rebuild the updater or open a trial.
- **The script's schedule.** When the trainer has its own schedule
  (`schedule=` on the `Trainer`), the agent is shown it, sampled across
  training, and told that its groups' rates already scale with it, so a decay
  it writes multiplies the script's.

## What the agent sees

The goal sentence, the run's facts (`epochs`, `train_size`, `batch_size`,
`same_order_each_epoch`, `task_facts`), the parameter table, the training
loss, its reports, errors, timing, and:

- **validation results** — `report_validation(loss, accuracy)` with
  `validation=N` (the agent is told a held-out set is evaluated about every N
  updates, at times not announced);
- **values you choose to show** — `log(values, show_agent=True)`; without
  `show_agent` they are only recorded;
- **read-only copies of your code** — `share=[paths]` (pyrlx shares the
  training script by default, `share_code=False` keeps it out);
- **your notes** — `rlx-agent-opt say <run> "…"` (or a file in `<run>/notes/`)
  reaches the next decision.

With validation on, an **integrity check** flags a program that moves the
params somewhere else for an evaluation and back (`eval_integrity`); the agent
is not told, the flags go to `results.json`.

The system prompt is `prompt="neutral"` (facts and protocol, no recommended
settings), `"aggressive"` (adds a review of every group's hyperparameters at
each decision), or the path of your own template with the same `{fields}`
(`PROMPT_FIELDS`).

## Decisions, waiting and cost

`decisions` unset: the count is estimated from the time of the first 10
updates so the agent adds at most `max_wait_ratio` (4) × the training time at
`seconds_per_decision` (40) each, between 1 and 50. A fifth fall in the first
5% of the run (`spread="front"`) or they are even (`"uniform"`).

`blocking`: `true` (wait at every decision), `false` (edits land when they
arrive), or `"auto"` (wait when an answer would land more than
`max_lag_updates` late). A non-finite loss always waits (`pause_on_nonfinite`).
`max_cost_usd` stops the asking. With the built-in prompts the agent is told
when its next scheduled decision is (`show_next_decision`, off with a template)
and not what it has cost (a template's message says).

## Agents

`backend`: `"claude"` (default model `claude-opus-5-5`), `"cursor"`,
`"copilot"`, `"codex"` (with its own `codex_home`), through
[`rlx-agent`](../crates/tooling/rlx-agent/README.md): the CLI runs with no
built-in tools; its tools are one MCP server over the workspace that reads
every file (`list_files`, `read_file` by lines, `search`) and writes only the
program files and `NOTES.md`. Its `check_program` tool builds the program as
the files are now and runs a few updates of it on made-up params in a process
of its own (`rlx-agent-opt check-program <workspace>`, Python's through
`python -m pyrlx.agent`): whether it builds, stays finite and how far it moves
each param, before the agent finishes. Claude keeps one
session for the run. A failed call is made again with growing pauses
(`retries`, `retry_pause_seconds`); an agent that is unavailable (no login, a
usage limit, an overload) is waited for (`unavailable_timeout_seconds` to give
up); `timeout_seconds` bounds one decision.

**No internet on the training machine:** `backend="mailbox"` with
`{"dir": <shared folder>}`, and on a machine with the agent:

```text
rlx-agent-opt broker <shared folder> --backend claude
```

## Records and the command line

`rlx_runs/<id>/`: `workspace/`, `programs/vNNN/`, `decisions/NNN/` (message,
response, transcript), `events.jsonl`, `losses.csv`, `validation.jsonl`,
`metrics.jsonl`, `reports.jsonl` (each report, with every group's learning
rate), `notes/`, `FINAL`, `summary.json`, `results.json`. Credentials in
answers, errors and transcripts are redacted. `show` and `report` list what
each applied program changed: every JSON setting that differs
(`updater.json groups[0].optimizer.lr: 0.001 → 0.003`), else the lines added
and removed; the report also tables each group's learning rate over the run.

```text
rlx-agent-opt doctor [--live]                    is the agent CLI ready? (--live: one tiny paid call)
rlx-agent-opt estimate --total-steps N --sec-per-step S
rlx-agent-opt show <run>                         one line per decision
rlx-agent-opt watch <run>                        events as they are written
rlx-agent-opt report <run>                       one HTML page: loss, decisions, rationales, diffs
rlx-agent-opt compare <a> <b>
rlx-agent-opt check-program <workspace>          build its program, run it on made-up params (the agent's tool)
```

From Python: `python -m pyrlx.agent <command>`.

`save(dir, trainer)` / `resume(dir, …, trainer)` (Rust: `save_with` /
`resume_with`) checkpoint the whole run in one folder, the trainer's own
`trainer.gguf` beside the agent's files; without the trainer they checkpoint
the agent's side alone. A resumed run continues bit for bit. `replay(run, …)` runs
a finished run's final program again without an agent.

## Example (Python)

```python
import pyrlx as rlx

opt = {"kind": "adamw", "lr": 3e-4}
trainer = rlx.Trainer(loss_graph, wrt=names, init=weights, optimizer=opt)
agent = rlx.AgentOptimizer("MLP on MNIST; judged by test accuracy after the last update",
                           10_000, optimizer=opt, program="python", backend="claude",
                           max_cost_usd=20, validation=500)
for step, batch in enumerate(data):
    agent.train_step(trainer, batch)
    if step % 500 == 0:
        agent.report_validation(loss=val_loss(trainer.params()))
print(agent.close())
```

## Data-parallel runs

With `Trainer::set_data_parallel(group)` (Python: `trainer.set_data_parallel(pyrlx.Group…)`)
every rank trains a replica on its own batches with the mean gradient, and
`train_step` runs the agent over the same ranks (OPAL's protocol over DDP):
rank 0 alone asks the agent; at each check it sends the others the schedule
and any answer, so every rank installs the same program at the same update; a
build must succeed on every rank; during a trial the ranks agree after each
update on whether it failed and revert together (exactly at its first update),
and a trial that makes the ranks' params differ is reverted with rank 0's
params sent to all. After its trial, a failure on any rank stops the run with
`crashed-v<N>-r<rank>.json`; resuming from the checkpoint before it starts from
the program before the crashed one. Every `replica_check_every` updates (100;
0 for never) and after the last, the ranks check that their params are equal;
when they are not, rank 0's are sent to all and the agent is told which
tensors differed (a program keeping state per rank). Each rank keeps its own
run folder; rank 0's has the decisions.

## Training options

The `Trainer` options the agent works alongside (every binding takes them in
its options; Python as keyword arguments):

| option | what it does | PyTorch |
|---|---|---|
| `schedule` | `{kind: constant \| linear \| cosine \| step \| multistep \| exponential \| polynomial \| onecycle, total_steps, warmup_steps, …}`: the factor on the base learning rate per update | `torch.optim.lr_scheduler` |
| `accumulate` | micro-batches per update; the update uses their mean gradient | gradient accumulation |
| `loss_scale` | `"dynamic"` (×2 after 2000 good updates, ×0.5 and skip on overflow), a fixed scale, or `{init, growth, backoff, interval}` | `torch.amp.GradScaler` |
| further graph outputs | outputs after the loss reach `step_with` (`StepInfo::outputs`) and the analysis's `activations` | forward hooks |
| `set_data_parallel` | replicas on several ranks, mean gradient | `DistributedDataParallel` |

`run(…, eval_batch, eval_every, patience, min_delta, checkpoint,
checkpoint_every)` evaluates during training, keeps `best.gguf`, stops early and
writes periodic checkpoints. Under a schedule, the agent's programs follow the
scheduled rate (`StepInfo::lr`); with no agent the run is the plain one bit for
bit.

## Fine-tuning a package (LoRA)

Any `.rlxp` with a static, differentiable function trains through
`Trainer::from_package` (`rlx_runtime::model::trainer`: `function`, `dims`,
`loss`, `train`, over the defaults in the package's model block), and the
agent runs over it as over any `Trainer`. Qwen3.5 0.8B, from the BF16 GGUF:

```text
rlx-export Qwen3.5-0.8B-BF16.gguf -o qwen-train.rlxp --variant train --seqs 64
```

The `train` variant (rlx-models' `docs/exporting.md`, the `qwen35` row)
is one function `forward` (`input_ids` `[1, 64]` → `logits`), its weights dense
f32 params, and these defaults: next-token cross-entropy against `labels`,
LoRA rank 8 on `*attn_q*` / `*attn_v*` (the full-attention layers' q and v,
the delta-net layers' fused qkv: 30 sites). In Python:

```python
trainer = rlx.Trainer.from_package("qwen-train.rlxp", device="metal",
                                   optimizer={"kind": "adamw", "lr": 1e-3}, clip_norm=1.0)
agent = rlx.AgentOptimizer(goal, steps, optimizer={"kind": "adamw", "lr": 1e-3},
                           backend="claude", validation=25, max_cost_usd=2)
for step, batch in enumerate(windows, 1):        # {"input_ids": [1, 64], "labels": [64]}
    agent.train_step(trainer, batch)
    if step % 25 == 0:
        agent.report_validation(loss=mean(trainer.evaluate(b) for b in validation))
```

Judge the run on a test set the agent never sees: shown the loss it is judged
on, an agent can tune its programs to that set.

`schedule`, `accumulate` and `loss_scale` go in the same options.
[`examples/agent_qwen35.py`](../crates/bindings/pyrlx/examples/agent_qwen35.py)
is the whole run: `train` and `heldout` are text files or folders of `.md`
files; the held-out text is split into a validation half (shown to the agent)
and a test half (the result, never shown). `--plain` first trains a plain
`Trainer` on the same windows, measured at the same updates, prints both
curves side by side and reports whether the losses are equal; the run folder
keeps them in `comparison.json`. Before spending money on an agent:

1. `--backend keep --plain`: the run must be identical to the plain one.
2. With `resident=True`, the fused path must make the host path's losses
   (gradient clipping is host-only, so the script drops it there).

On an M-series Mac (Metal, 100 updates, 64-token windows of RLX's docs):

| check | result |
|---|---|
| `keep` against plain | identical losses; held-out loss 4.553 → 4.316 in both |
| time per update | about 0.85 s (host path) |
| resident `muon_adamw` against host, 20 updates | within 1e-5 per update; 0.59 s against 0.84 s |
| `keep` on the device (`--resident --plain`) against the plain resident run | identical losses; held-out loss 4.553 → 4.356 in both |
| Claude on the device, 2 decisions, $0.36 | both programs applied and passed their trials; held-out loss 4.339 |
| 1,000 updates (1.7 epochs of 580 windows): plain against Claude, 6 decisions, $2.54 | plain 4.553 → 4.155; Claude's lowest 3.927 (update 600), then the second epoch overfit and it ended at 4.286; it called `check_program` at every decision |
| 1,000 updates on 3.4 MB of Markdown (no window twice), test set never shown: plain (constant rate) against Claude, 6 decisions, $1.79 | test loss 4.625 → plain 4.112, Claude 3.712; Claude added a decay of the rate (cut earlier when validation rose); against a constant rate |
| the same, both runs under a cosine schedule (warm-up 5%), Claude 6 decisions, $1.96 | test loss: plain 3.705, Claude 3.816; Claude doubled the LoRA `up` rate, then cut it and decayed further on top of the cosine. With a standard schedule the agent did not help here |
| the same, after the device reports gained gradient and weight statistics and the agent was shown the script's schedule, Claude 6 decisions, $2.62 | test loss: plain 3.705, Claude 3.740 (validation: plain 3.369, Claude 3.364); Claude kept the cosine, read the weight norms and update ratios, and halved the rate once at update 367 after one rise of the validation loss. A tie within the noise of 32 windows |

## On the device

On a resident trainer (`resident=True`, the update fused into the graph)
`train_step` runs the program on the device: a rules program becomes a plan of
each param's fused rule (`adam`, `adamw`, `muon`, `muon_adamw`) and learning
rate, and the trainer's fused step makes the update. A new rate costs nothing;
new rules build the fused step again, keeping the params and, within a family
of rules and the group's `carry_state`, the moments. The protocol runs on the
loss: a program the device cannot run (another kind, `skip_if`, a Python
program) is rejected when it is built; at a swap the params and moments are
read back once, and a non-finite loss during the trial puts them back and the
previous program with them. At a report the fused step also reads back each
param's gradient norm (a scalar each, no gradient download), and the params
when the analysis asks for `weight_norm` or `update_ratio`, so the reports
match the host's; `grad_cosine` needs the gradients and is left out, and
`update_ratio` starts at the report after the decision that asks for it. With
no agent the run is the plain resident run.

## License

MIT OR Apache-2.0.
