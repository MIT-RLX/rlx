# rlx-wsp

**Weight-space programming on rlx graphs.** An agent reads and edits a
network's weights directly, reasoning about what each unit computes, instead
of training it — following
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
`rlx-wsp` is the harness: any rlx graph's weights as a labelled, checkpointed
store; a tool surface an agent CLI drives over MCP; a **forward-pass budget**
that is the agent's only way to run the network; and held-out scoring it
never sees.

```
 describe ─▶ read (slice / topk / norms / svd / compose / list_nodes) ─▶ edit (set / inject / project_out / insert_linear …)
                ▲                                                                   │
                └──────── probe / rollout / eval / trace / compare / ablate (budgeted) ◀─┘
```

## Approach: model-independent, graph-first

The tools never assume an architecture. Everything they do is read off the
rlx [`Graph`](../../core/rlx-ir/src/graph.rs) and its params:

- **The network is a graph.** A space holds an rlx graph (inline `.rlxmodel`
  text, or a task's recipe that builds one) and its params. Tasks are the
  only code that knows what the network is for; tools see nodes, params and
  labels.
- **A unit is a channel of a node:** `node[i]` — the last-axis index of any
  named node. Tasks may give channels names (an MLP's `l0.h3`, an output
  node's action or digit names); those are aliases, not assumptions.
- **Labels are data.** Rows and columns of a param, and channels of a node,
  carry the labels a task attached (observation cells `ball>(x=3,y=5)`,
  pixels `px(r=12,c=14)`, actions `right`), else generic `rI` / `cJ` /
  `node[i]`. Every tool accepts labels wherever it takes an index.
- **Provenance comes from the graph** (`graphops::channel_params`): the
  weight column and bias that produce a channel (walking back through
  elementwise ops to a `MatMul` with a param right-hand side) and the weight
  rows that read it. `inject`, `graft` and the edit checks use it, so they
  work on any node fed by a linear map.
- **Structural operations are graph rewrites** (`graphops::insert_after`):
  the graph is rebuilt in order with new nodes spliced in, and outputs,
  output names, variables and symbolic attributes are carried across with
  ids remapped — so they apply to any graph, decode graphs included.
  - `ablate` = a channel mask (`Mul` by a constant) after the node;
  - `insert_linear` = an identity `x @ W` after any node, a new param the
    agent can program without changing the function;
  - `repeat` = a shape-preserving span (`from` → `to`, any number of layers
    or blocks) run `times` times in a row, with shared weights or cloned
    params (`{param}.rep{k}`) that can be edited apart; spans that touch
    variables or caches are refused;
  - `trace` = extra graph outputs for the requested nodes.
- **Running is budgeted and the rules are enforced.** One forward pass = one
  `probe` / `rollout` / `eval` / `trace` (two for `compare` / `ablate`). The
  `script` sandbox cannot see weight values, and a script that carries copied
  weights (≥16 pasted non-round weight values, or >200 literals) is refused —
  otherwise an agent can run the network in JavaScript for free (one did).
  Reading weights, reasoning about them and `compose` are free, as in the
  post.
- **Scoring is held out.** Each task's score uses episodes or test data no
  tool exposes; reading / audit answers are checked against a key stored in
  the state but never shown.

- **Any packaged model, edited as an overlay.** A package space opens an
  `.rlxp` (`--package m.rlxp`): tensors are read from the package on demand
  (f32 / f16 / bf16; quantized ones are listed, read-only), and only the
  tensors the agent changes are stored, compactly, in the state. To run the
  edited model, the overlay is written as an **adapter package** over the
  base — each changed tensor as `W + B·A` with one factor the identity, so
  the merge reproduces the edit exactly — and opened through rlx-runtime's
  `Model` API, which generates with whatever decoder the package carries.
  Nothing in `rlx-wsp` knows the architecture; the model-specific step is the
  export (`rlx-export` in rlx-models).
- **Graph tools on packages through taps.** A package runs its own graphs
  (decode steps with caches); an `rlx_runtime::taps::GraphTap` sees each one
  as it compiles, replays the recorded rewrites (`repeat`), applies masks and
  adds the requested nodes as outputs. So `list_nodes` / `use_function`,
  `trace`, `node_stats`, `contrast`, `lens`, `ablate` and `compare` read and
  change any package fed by texts (language models) or named tensors, with
  one value per item pooled over decode steps (`last` or `mean`).
- **Measure before keeping.** Accuracy on 64 questions moves ±6 points by
  chance, and two question sets differed by 17 points for the same model in
  one run. `qa_loglik` scores the reference answer by teacher forcing
  (`LmRunner::score_continuation` in rlx-runtime: one cached decode step a
  token, for any language model) — continuous and deterministic. `compare`,
  `ablate`, `sensitivity` and `search` score both versions on the same
  items and report paired statistics; `search` confirms its winner on items
  no round used, since the best of many on the same items is overstated.
- **Targeted edits from the graph.** `channel_bias` / `token_bias` / `steer`
  add constants to a node; on a node that is exactly `x + bias` they edit
  that bias (an ordinary weight edit), elsewhere they are recorded rewrites
  replayed on every run and open, and kept by checkpoints. `write_kv` finds
  the `MatMul` reading a weight (through dtype casts) and its input node.
  Listed nodes show the params they read (`params`); unnamed nodes are
  addressed as `%id`.
- **Low-rank edits stay low-rank.** On package spaces `add_rank1` and
  `project_out` keep their factors; the state stores them, not the dense
  tensor, and the adapter carries them as a true-rank `W + B·A`. Any other
  edit of the tensor makes it dense.

What is still task-level rather than general: the training baselines
(`rl-baseline`, `warmstart`) train MLPs on the host, and unit roles use
features of the built-in games.

## Tools

| group | tools | cost |
|---|---|---|
| orient | `describe`, `status`, `list_params` / `list_nodes` (filter, `offset`/`limit`, sizes), `use_function` (packages: which graph), `log` | free |
| read weights | `stats`, `slice` (≤4096 values, filter rows by label), `topk`, `norms`, `svd`, `compose` (A·B of two params) | free |
| read data | `sample_images` (classifier tasks, text art) | free |
| edit | `set`, `set_row`, `set_col`, `write`, `fill`, `scale`, `add_rank1`, `project_out` (remove a direction exactly), `inject` (blocks), `graft` (copy units between networks of any shape) | free |
| structure | `insert_linear` (identity map after any node), `repeat` (any span of layers / blocks, any number of times; shared or cloned weights) | free |
| history | `checkpoint`, `revert`, `diff` | free |
| sandbox | `script` (JavaScript, no host access, no weight values, 5·10⁸ instructions; returned `{calls:[…]}` are applied) | free |
| run | `probe` (≤32 inputs: logits + hidden), `rollout`, `eval`, `trace` (any nodes) | 1 pass |
| read activations | `node_stats` (per-channel mean / sd, most variable), `contrast` (mean difference of two input sets, stored as a named direction for `project_out`), `lens` (a node read through a param, decoded to labels or tokens) | 1 pass |
| test | `compare` (paired A/B of checkpoints, `current`, or `unedited`), `ablate` (mask units, paired); QA: on the same practice questions, metric `loglik` (default) or `accuracy`, with the paired mean difference, a bootstrap 95% interval, p (sign-flip, or exact McNemar for accuracy) and a verdict | 2 passes |
| targeted edits | `channel_bias` (constants on channels of any node), `token_bias` (a token's logit, by text or id), `steer` (add a direction, e.g. one `contrast` stored, to a node), `rewrites` (list / remove) | free |
| closed-form write | `write_kv`: make a weight map a key (its MatMul input on chosen items) to a target output (another item set's, or current + `delta`): `W += k (v − kᵀW)ᵀ / kᵀk`, kept low-rank | 1–2 passes |
| where / which | `sensitivity` (scale each of ≤16 tensors by 1 + eps; metric change, paired, largest first), `search` (successive halving over ≤16 candidate edits on growing item sets, then the winner vs the live network on fresh items; `apply_best` only if clearly better) | 1 + tensors / reported before running |
| answer | `answer` (+ confidence), `switch` (active network in pair tasks) | free |
| language models | `qa_practice` (practice questions) | free |
| language models | `generate` (≤32 prompts), `qa_eval` (≤64 practice questions: accuracy, outcomes — correct / wrong answer / loop / unfinished `<think>` / no final marker — token counts, hit-limit rate, time), `qa_loglik` (teacher-forced mean log-probability of the reference solution, or only its `#### n` line) | 1 pass |

**`inject` blocks** write a circuit into channels of one node, replacing the
producing column and bias found in the graph, activation-aware (tanh off =
−1, ReLU off = 0): `relay`, `threshold`, `and` / `or` (`01` or `pm1`
inputs), `gate` (passes a signal only while a 0/1 input is on), `compare`
(`[a>b, b>a]`). The Bounce program, rebuilt from two thresholds and relays,
scores the maximum.

## Tasks

| `--env` | what the agent does | references (held-out) |
|---|---|---|
| `bounce` | write a policy from zero; one ball | written landing program +22.00 (max) |
| `orchard` | write; fruit + bombs, ≤4 objects | planner +26.52, ES@2000 +5.58 |
| `gusts` | write; Orchard with **hidden** motion rules | planner (true rules) +27.02, planner assuming Orchard −7.05 |
| `gusts-hard` | write; hidden rules, 16×12, ≤6 objects | planner +29.43 greedy / +26.38 stochastic; ES −1.14 stochastic |
| `fix-policy` | fix a stuck policy-gradient Bounce policy, given one recorded episode | handed over −0.40, max 22.00 |
| `mnist-fix` | repair a backdoored MNIST MLP (corner trigger) | tampered −3.38, zero trigger rows 96.37, clean-trained 96.89 |
| `mnist-fix-hard` | repair; 4×4 checkerboard in the image centre | zero trigger rows 85.15, mask 6 trigger units 92.44, clean-trained 96.91 |
| `mnist-steer` | make every 3 read as 8, keep other digits | handed over 48.93, logit rewrite 97.64 (easy) |
| `read-rules` | READ: which motion rules a copy-trained policy learned (`--open`: free text) | chance 33% |
| `read-digit` | READ: which digit an MNIST model never saw | chance 10% |
| `audit-pair` | AUDIT: which of two MNIST models is backdoored (`--internals-only`: no outputs) | chance 50% |
| `--graph m.rlxmodel` | any graph + `--weights w.json` + `--goal "…"`; no harness score. `--score-cmd "prog args"` adds the `external_score` tool (1 pass): the command gets a `{name: [values]}` JSON file of the live weights as its last argument and prints a JSON score on its last line. Keep the final test outside it; `rlx-wsp weights --state s --out w.json` writes the edited weights | — |
| `--package m.rlxp` | any packaged model, edited as an overlay; no score | — |
| `--package lm.rlxp --qa-practice p.jsonl --qa-heldout h.jsonl` | QA: improve a language model's answers (GSM8K-style `#### n`, else last number) | the unedited model's held-out accuracy |

Scores: control tasks = mean episode return; repair = clean accuracy − attack
success (pp); steer = mean of the two rates; reading / audit = 100 if right
(plus a Brier score when a confidence is given). `rlx-wsp bench --env <task>`
prints the references. MNIST reads IDX files from `$RLX_MNIST_DIR` (default
`$HOME/.mnist`); trained task models are cached in `$HOME/.cache/rlx-wsp/`.

## Running agents

The LLM is an **agent CLI** (`rlx-agent`): Claude Code, Cursor `agent`, or
Copilot, restricted to the `wsp` MCP tools. `rlx-wsp` makes no API requests.

```sh
cargo build -p rlx-wsp --release
rlx-wsp agent --cli claude --env gusts-hard --out runs/g1 --budget 50 --stochastic
rlx-wsp sweep --agents claude,agent:gpt-5.2,copilot --env orchard --repeats 3 --out runs/s1
rlx-wsp sweep --agents claude --env bounce --budgets 0,10,25,50 --out runs/curve
```

Flags: `--stochastic` (sampled + sticky actions, the post's protocol),
`--oneshot` (budget 0, write tools only), `--internals-only`, `--open`,
`--task-seed N`, `--model`, `--effort`, `--max-turns`, `--max-budget-usd`.
Each run writes `state.json`, `prompt.txt`, `transcript.jsonl` and
`report.json` (held-out score, budget audit, and the check of the agent's
account), and runs the CLI in an empty `workspace/`.

`sweep` reports mean ± sd and a bootstrap 95% interval per configuration
over repeats, and for QA the change against the unedited model (its
held-out score, cached, is in every report) with an interval and how many
runs beat it; every report records the `rlx-wsp` version, task seed, action
protocol, rewrites and package. Long passes send MCP
`notifications/progress` when the client gives a progress token.

Local driving without an agent: `rlx-wsp new / call / serve / score / mcp`
(`call` takes one `{"tool": …}` JSON; replies are `{"ok"}` / `{"error"}`).

### A language model (Qwen3.5-0.8B)

```sh
# once, in rlx-models: the model as a self-contained package
rlx-export Qwen3.5-0.8B-BF16.gguf -o qwen35-0.8b-bf16.rlxp --context 1024 --weights bf16 --embed-weights
# a QA space: practice / held-out JSONL ({"question", "answer"}), e.g. GSM8K
rlx-wsp new --state q.json --package qwen35-0.8b-bf16.rlxp \
    --qa-practice gsm8k-practice.jsonl --qa-heldout gsm8k-heldout.jsonl \
    --heldout-n 100 --max-tokens 256 --device metal --budget 50
rlx-wsp agent --cli claude --package … (same flags) --out runs/qwen-1
```

Build with `--features metal` for `--device metal`: a decode-only package
feeds prompts token by token, and 4 questions × 64 tokens took 392 s on CPU,
23 s on Metal. The unedited Qwen3.5-0.8B scored 22% on 100 held-out GSM8K
questions at 256 tokens (64% gave a final answer; 1018 s on Metal) with the
first answer parser, which read `####` followed by a `<think>` block as the
answer; the parser now takes the first number after a marker, so that
baseline is measured again. The unedited result is cached per package,
question set and parser version, so `score` does not repeat it. Zeroing `blk.15.ffn_down.weight` through the tools changes
"12 + 30" from 42 to 32 — the overlay reaches the model.

### Guards against fooling ourselves (QA)

| flag | what it does |
|---|---|
| `--dev-n N` (+ `--dev-checks K`, default 3) | the last N practice questions are hidden from every tool; `dev_check {a, b}` compares two versions on them, at most K times, and keeps a clear winner as checkpoint `verified` |
| `--keep verified` | the report's headline score is checkpoint `verified` (or the unedited model when nothing was verified), not whatever the run ended with; both are reported |
| `--qa-general file` | general text (JSONL `{"text"}` or paragraphs); `compare` and `search` also compare its log-likelihood and warn when an edit makes it worse |
| `--token-budget N` | tokens processed (prompts and outputs) as a second budget, so a 64-question, 256-token evaluation is not priced like a 3-prompt `generate` |
| `--check-perturbed` | the report also scores the held-out questions' twins — numbers changed, answers recomputed from the reference calculator steps (`<<48/2=24>>`) — for the edited and the unedited model; a gain that holds on the originals but not the twins is learned answers. `qa_eval {perturbed: true}` does the same on practice questions for the agent. Numbers the steps use transformed (150% as 1.5) or also as constants (half = /2) are kept; 46 of the first 100 GSM8K test questions get a twin |
| `--check-truncation` | the report also scores at twice the token limit, edited and unedited |
| `--dedupe J` (default 0.6) | practice questions whose word 3-grams overlap a held-out question's at Jaccard ≥ J are dropped from the tools; `rlx-wsp qa-hygiene` lists them and shows example twins |
| `--max-edit R` | package spaces: refuse any edit that leaves a tensor more than R of its norm from the package's; every report has `edit_size` (per-tensor and total relative change) |
| `--prompt-style chat`, `--stop-at-answer` | score with the package's chat template (thinking off), and cut each answer after its first `#### n` line — to measure how much of a score is the prompt, not the weights |
| `--repetition-penalty r` | damp loops in every generation (edited and unedited alike: it is part of the task, and keys the held-out cache) — about 40% of Qwen3.5-0.8B's GSM8K answers loop at 1.0 |

Scored answers are generated only until their first `#### n` line is complete: the score reads that answer, so what would follow cannot change it (the `generate` tool still shows whole completions).

### LoRA baselines

```sh
# forward-only (MeZO-style SPSA on the LoRA factors; any package, slow)
rlx-wsp lora-train --state q.json --method mezo --targets '*attn_q*,*attn_v*' \
    --rank 8 --alpha 16 --steps 200 --batch 4 --lr 1e-4 --eps 1e-3 --out runs/lora-mezo
# backprop (rlx_runtime::model::trainer; needs --features training and a
# package with a trainable full-sequence function)
rlx-export Qwen3.5-0.8B-BF16.gguf -o qwen35-train.rlxp --variant train --seqs 256 --embed-weights  # rlx-models
rlx-wsp lora-train --state q.json --method backprop --train-package qwen35-train.rlxp \
    --targets '*attn_q*,*attn_v*' --steps 300 --lr 1e-4 --out runs/lora-bp
rlx-wsp score --state runs/lora-bp/state.json          # held-out, like any run
rlx-wsp run --state runs/lora-bp/state.json --prompt "Question: …"
rlx-wsp run --package runs/lora-bp/adapter.rlxp --prompt "…"   # any package or adapter
```

`--token-budget N` stops either at N tokens processed (backprop counts a
token three times, forward and backward) — the unit the agent's
`budget_audit.tokens_used` reports, so a LoRA and an agent can be compared
at equal compute; `lora.json` records `tokens_processed`.

Both leave the adapter in the space as low-rank factors (`W + Σ u vᵀ`) and
write `state.json`, `adapter.rlxp` (over the space's package) and
`lora.json` (the training curve). Backprop trains on the trainable package
(loss: next-token cross-entropy masked to the reference answer) and applies
the factors to the space's package by tensor name — both exports read the
same checkpoint tensors. These are references for what training reaches,
not agent tools.

Context distillation: when worked examples in front of the prompt fix
what the model gets wrong (on GSM8K, Qwen3.5-0.8B stops writing plans
without arithmetic), train on its own correct answers given that way, with
the plain prompt:

```sh
rlx-wsp qa-distill --state q.json --shots 1 --questions 512 --out runs/distill.jsonl
rlx-wsp lora-train --state q.json --method backprop --train-package qwen35-train.rlxp \
    --train-items runs/distill.jsonl --average --gate 3 --out runs/lora-distill
```

`qa-distill` answers practice questions after the first `--shots`
practice items' reference solutions, keeps the correct answers as GSM8K
rows (`{question, answer}`), and never draws on the dev split or the
near-duplicates of held-out questions. `--train-items` trains on those rows
while the dev split (for `--average`) and the held-out score stay the
task's.

### The whole comparison, for any model

```sh
rlx-wsp pipeline --package model.rlxp --qa-practice train.jsonl --qa-heldout test.jsonl \
    --cli claude --effort high --budget 40 --max-budget-usd 10 \
    --train-export 'rlx-export model.gguf -o {out} --variant train --seqs 256 --embed-weights' \
    --out runs/model --detach
```

From a package and a question set to a comparison of what improves it:

| stage | what it does |
|---|---|
| `base` | the task: near-duplicates of held-out questions left out, a dev split (50 unless `--dev-n`) |
| `baseline` | the unedited model on the held-out questions, perturbed and at twice the tokens |
| `agent` | an agent edits the weights (`--cli`, `--budget`, `--goal`, `--deny-tools`), scored, `exploration.svg` |
| `distill` | the model's correct answers given with worked examples (`--shots`, `--questions`) |
| `export` | `--train-export`, the one model-specific step (`{out}` is the path to write); or give `--train-package` |
| `lora-reference` | LoRA on the reference solutions, scored |
| `lora-distill` | LoRA on the distilled answers, scored |
| `summary` | `summary.md`, `summary.json`, `summary.svg`: each method side by side, and what each stage came to |

Stages run one at a time, as child processes, logging to
`<out>/logs/<stage>.log`. Their outcomes are in `<out>/pipeline.json`: the
same command again skips what is done, retries what failed, and finishes an
agent run that ended before its scoring did. `--stages a,b` runs only those,
`--redo a,b` runs them again. A stage whose input failed is skipped and the
rest go on; without a trainable package the LoRA stages say so. LoRA stages
need rlx-wsp built with `--features training`.

### Long runs

```sh
rlx-wsp agent … --out runs/qwen-1 --detach     # any command; log in runs/qwen-1.log
rlx-wsp finish --run runs/qwen-1               # a run stopped after its agent ended
```

`--detach` runs the command in a session of its own (`--log <file>`, else
`<--out>.log`), so ending the shell or agent session that started it does
not end it. Scoring keeps each finished stage in `scoring.json`, tied to
that `state.json`: `finish` scores what is left and writes the whole report
and chart, the agent's account read back from the transcript. A verified
version that is also the live one is scored once.

### How a run explored

Every tool call leaves a step in the state's `trail` (tool, arguments,
reply, passes used, the edits live after it), recorded by the server, so
it is the same whichever agent drives the run. When an agent run ends,
`exploration.svg` is written beside `report.json`; for any run directory:

```sh
rlx-wsp explore-chart --run runs/qwen-1 --out qwen-1.svg
```

Three views on one call axis: the accuracy each measurement found
(practice and dev questions, held-out scores at the end), the paired
changes comparisons found (one panel per metric, mean and 95% interval),
and a lane per edit for the calls it was live, with checkpoints, reverts
and failed calls. Colour is the edit family (activation, token, weight);
hovering a mark names the call. `exploration.json` holds the same points
as a table. A run recorded before the trail existed is read from its
transcript, its live edits followed from the edit calls.

### Branching off a run

```sh
rlx-wsp branch --from-state runs/qwen-1 --checkpoint q125 --budget 30 --out runs/qwen-1-b
rlx-wsp agent  --from-state runs/qwen-1 --checkpoint q125 --budget 30 --cli claude \
    --dev-n 50 --keep verified --check-perturbed --out runs/qwen-1-b
```

`--deny-tools scale,set_row` takes tools away from a run (hidden from the
tool list, refused when called directly, from a script or in a `search`
candidate) — e.g. to make an agent look past a lever it keeps reaching
for; `--goal "…"` replaces the task's goal text. Both work on new runs,
branches and sweeps. A denied tool's effect can still be reached through
other edit tools; the option shapes what the agent tries, it does not make
an edit impossible.

A branch is a new run from another run's current weights or one of its
checkpoints — a finished run or one still going (its state file is read
once): the same task, that version's overlay, low-rank factors and
rewrites, its checkpoints plus `branch_start`, and a fresh budget. Its
history moves into `lineage`, which `describe` shows the agent as
`branched_from`; branches of branches keep every link. Guard and check
flags given override the parent's (a dev split added on a branch gets its
own checks — its questions may have been practice questions for the
parent).

### An agent in the training loop (after OPAL, arXiv:2610.33826)

```sh
rlx-wsp agent --cli claude --package qwen35-0.8b-bf16.rlxp --qa-practice … --qa-heldout … \
    --opal --train-package qwen35-train.rlxp --targets '*attn_q*,*attn_v*' --steps 600 \
    --dev-n 50 --keep verified --out runs/opal-1          # build with --features training
```

A LoRA trains on the practice set (the package trainer's function, forward
+ backward + optimizer in one graph, a rate and a freeze gate per factor)
while the agent watches and changes it:

| tool | what it does |
|---|---|
| `train {steps, background}` | advance training (a step budget); `background` trains in a thread while the agent keeps working |
| `set_updater {code}` | a JavaScript policy run before every step with `STATS` (recent losses; per site the update norm, update-to-weight ratio, cosine with the previous update; rates; optimizer); returns `lr`, `site_lr`, `freeze` / `unfreeze`, `gate` (skip a batch whose loss spikes, scored before the step), `skip`, `optimizer` (`adamw` / `muon`: rebuilt, factors kept) |
| `analyze {code}` | the agent's own probes over the whole `HISTORY` |
| `lora_snapshot {tag}` | the current adapter on the starting version, as a checkpoint and the live weights — measured with `qa_loglik` / `compare` / `dev_check`, mixed with `average` |
| `train_status` | steps, losses, gated / skipped steps, rates, frozen sites, errors |

The same loop runs over any trainer: a program that embeds this crate
implements `steer::Trainee` (describe, stats, apply an updater's answer, step,
validate, params), `attach`es it to a space and serves it. `train`,
`set_updater`, `analyze`, `train_status`, `train_validate` and `train_snapshot`
then steer that run; its data, model and optimizer stay in the program, and
what an updater may return is the trainee's to define (`describe` says).
`SteeredRun::finish` runs what is left under the last updater, so an agent that
stops early does not leave a schedule unfinished.

### Other tools from the same idea

- **Background jobs:** `"async": true` on `qa_eval`, `qa_loglik`,
  `generate`, `trace`, `node_stats`, `lens`, `contrast`, `compare`, `ablate`
  or `sensitivity` charges its passes and runs it in a child process on a
  snapshot of the weights of that moment; the agent keeps working and
  collects it with `jobs {collect, wait}` (one at a time by default: the
  child opens its own model).
- **Spike gate and dev-selected averaging** for both LoRA trainers:
  `lora-train --gate 3 --average --snapshot-every 20` skips batches whose
  loss exceeds 3× the recent median, keeps adapters along the way and picks
  the final one or the mean of the last 2 / 4 / 8 by dev log-likelihood.
- **`average {tags, weights}`** mixes versions (checkpoints, `current`,
  `unedited`) — e.g. halfway between the model and an edit.
- **`precision {dtype}`**: how much of each edit survives rounding to bf16
  / f16 (a small edit can vanish below the dtype's resolution).
- **Time in reports:** seconds per tool and outside tools (the agent
  reasoning); **saved scripts:** `script {save_as}` / `script {saved}`.

## After a run

```sh
rlx-wsp verify    --state run/state.json   # writing tasks: unit ablations + measured roles vs the agent's text
                                           # repair / fix: what changed and what each change is worth
rlx-wsp bundle    --run run --out pub/     # prompt, transcript, report, weights, .rlxmodel, README
rlx-wsp export    --state run/state.json --out x [--fpga]   # baked .rlxmodel, int8 check, SystemVerilog
rlx-wsp warmstart --state run/state.json   # SGD from the agent's weights vs random (MLP tasks)
rlx-wsp rl-baseline --env orchard [--method ppo|reinforce]  # trained baseline
rlx-wsp graft --state dst.json --from-state src.json --units l0.h3 --into l1.h0
```

## Results so far (Claude Code, default model)

| task | result | references |
|---|---|---|
| bounce | 22.00 ×3, 1 pass | max 22.00 |
| orchard | 26.74 | planner 26.52 |
| gusts | 26.88–27.12 | planner 27.02 |
| gusts-hard (stochastic) | 26.62 / 27.60 / 27.98 | planner 26.38 |
| mnist-fix-hard | 95.94 ×3 (`project_out` on the 16 trigger pixels) | best simple fix 92.44, clean 96.91 |
| audit-pair | 3/3 correct | chance 50% |
| read-rules (weights only) | 3/3 correct | chance 33% |

The `mnist-fix-hard` and `gusts-hard` runs predate the script restrictions
(their scripts could read weights); the reading and audit runs do not. One
run per configuration elsewhere; differences of a few tenths are within
noise.

## Limits

- On package spaces `repeat` shares weights and `insert_linear` is refused:
  both would need tensors the base package does not have, which the adapter
  cannot add. `inject` edits existing producing
  columns, so it works where a channel comes from a `MatMul` with a package
  weight.
- A dense edit stores the whole tensor in the overlay (a 3584×1024 matrix is
  ~20 MB of state); only `add_rank1` / `project_out` stay low-rank.
- Tools have no per-call timeout; a long `qa_eval` reports progress but runs
  to the end.
- `qa_loglik` scores one question at a time (no batched step), and a
  decode-only package feeds the prompt token by token.
- Backprop LoRA runs on the CPU (the package trainer's device) and needs a
  trainable export; forward-only LoRA reopens the model for every
  evaluation (two per step).
- Recorded rewrites (`steer`, `token_bias` on nodes without a bias param,
  `repeat`) are not in the adapter: `export` lists them in `export.json`;
  running them needs the tap (rlx-wsp) or a re-export of the graph.
- Tensors over 64M elements (e.g. a large embedding table) are not loaded
  whole.
- `rl-baseline`, `warmstart` and unit roles are task-level (host MLP training,
  built-in game features).
- Claim and role checks parse free text; treat them as signals, not proofs.
- Agents can still copy weights by hand in their own reasoning; only
  mechanised copying into `script` is blocked, as the post's harness
  relied on reviewing logs for the same gap.
