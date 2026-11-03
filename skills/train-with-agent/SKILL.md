---
name: train-with-agent
description: Set up an RLX training run with the agent optimizer (pyrlx.AgentOptimizer or rlx_bind::runtime::agent), so a coding agent rewrites the optimizer while the model trains. Use when someone asks to train an RLX model "with an agent", "with OPAL", or to let Claude tune the optimizer during training.
---

# Train an RLX model with the agent optimizer

Read [docs/agent-optimizer.md](../../docs/agent-optimizer.md) first. Then, in
the person's project:

1. **Check the machine.** `python -m pyrlx.agent doctor` (free, offline). Run
   `doctor --live` (one tiny paid call) only once they approve an agent run.
   If the agent CLI is missing or not logged in, say what to run; do not log
   in for them.
2. **Find the training loop.** Report the `Trainer` (or the loop that calls
   `step`), its optimizer config, the number of steps, any learning-rate
   schedule (`set_lr`), clipping, and whether it is `resident` (the agent needs
   the host path).
3. **Ask for the goal sentence**: what is trained and what it is judged by at
   the end. Do not invent the metric. Ask whether there is a held-out
   validation set to report (`validation=N`, `report_validation`).
4. **Wire it in** and show the diff before running anything: replace
   `trainer.step(batch)` with `agent.train_step(trainer, batch)`, pass the same
   optimizer config as `optimizer=`, forward the learning rate with
   `agent.set_lr(lr)` where the script sets it, and `agent.close()` at the end.
5. **Check it changes nothing.** About 100 steps unwrapped and 100 with
   `backend="keep"`: the losses must match step for step. Show both.
6. **Before the real run**, time a step and run `python -m pyrlx.agent
   estimate --total-steps N --sec-per-step S`; show the waiting and ask for a
   cost cap (`max_cost_usd`). Wait for the go-ahead, then give the command and
   how to follow it (`show`, `watch`, `report`).

Do not edit RLX's own files; configure everything through the
`AgentOptimizer` arguments.

## License

MIT OR Apache-2.0.
