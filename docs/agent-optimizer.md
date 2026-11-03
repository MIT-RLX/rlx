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
- **The analysis.** `analysis.json`: which statistics the reports hold
  (`grad_norm`, `weight_norm`, `update_ratio`, `grad_cosine`, activations of
  the graph's further outputs), how often, and `wake` rules that ask for a
  decision as soon as a statistic crosses a bound. A change to it alone does
  not rebuild the updater or open a trial.

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
`max_cost_usd` stops the asking.

## Agents

`backend`: `"claude"` (default model `claude-opus-5-5`), `"cursor"`,
`"copilot"`, `"codex"` (with its own `codex_home`), through
[`rlx-agent`](../crates/tooling/rlx-agent/README.md): the CLI runs with no
built-in tools; its tools are one MCP server over the workspace that reads
every file and writes only the program files and `NOTES.md`. Claude keeps one
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
`metrics.jsonl`, `notes/`, `FINAL`, `summary.json`, `results.json`. Credentials
in answers, errors and transcripts are redacted.

```text
rlx-agent-opt doctor [--live]                    is the agent CLI ready? (--live: one tiny paid call)
rlx-agent-opt estimate --total-steps N --sec-per-step S
rlx-agent-opt show <run>                         one line per decision
rlx-agent-opt watch <run>                        events as they are written
rlx-agent-opt report <run>                       one HTML page: loss, decisions, rationales, diffs
rlx-agent-opt compare <a> <b>
```

From Python: `python -m pyrlx.agent <command>`.

`save(dir)` / `resume(dir, …)` checkpoint the agent's side (save the trainer's
params alongside); a resumed run continues bit for bit. `replay(run, …)` runs
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
the program before the crashed one. Each rank keeps its own run folder; rank
0's has the decisions.

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

## Not covered

The agent works on the host path (`Trainer` without `resident`).

## License

MIT OR Apache-2.0.
