# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""``pyrlx.AgentOptimizer``: OPAL's protocol from Python.

With no agent a run is the plain ``Trainer`` run bit for bit, for both program
languages. A stand-in agent CLI rewrites ``updater.py`` through the workspace
MCP server (``python -m pyrlx.workspace_mcp``), as Claude Code would: the new
program takes over at the decision; one that raises at its first update is
undone exactly. A checkpointed run resumes bit for bit.
"""

import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

import pyrlx as rlx

pytestmark = pytest.mark.skipif(not hasattr(rlx, "AgentOptimizer"), reason="built without `training`")

GOAL = "a small regression; judged by the final loss"
OPT = {"kind": "adamw", "lr": 0.05, "weight_decay": 0.01}
BATCH = {"x": np.array([1, 2], np.float32), "t": np.array([1, -1], np.float32)}


def graph():
    g = rlx.Graph("bias")
    x = g.input("x", [1, 2], "f32")
    t = g.input("t", [1, 2], "f32")
    w = g.param("w", [2, 2], "f32")
    b = g.param("b", [1, 2], "f32")
    d = g.sub(g.add(g.matmul(x, w), b), t)
    g.set_outputs([g.mean(g.mul(d, d), [0, 1], False)])
    return g


def trainer(optimizer=OPT):
    init = {"w": np.array([0.3, -0.2, 0.1, 0.4], np.float32), "b": np.zeros(2, np.float32)}
    return rlx.Trainer(graph(), wrt=["w", "b"], init=init, optimizer=optimizer, clip_norm=0.5)


def options(tmp_path, decisions=1, **kw):
    return dict(decisions=decisions, spread="uniform", every=5, run_root=str(tmp_path / "runs"), **kw)


def events(agent_dir):
    return [json.loads(line) for line in Path(agent_dir, "events.jsonl").read_text().splitlines()]


def stand_in_cli(tmp_path, updater_py):
    """An executable that acts as `claude`: it writes `updater_py` through the
    workspace MCP server named in its --mcp-config and prints stream-json."""
    script = f"""#!{sys.executable}
import json, subprocess, sys
args = sys.argv[1:]
cfg = json.load(open(args[args.index("--mcp-config") + 1]))
srv = cfg["mcpServers"]["rlx_files"]
p = subprocess.Popen([srv["command"], *srv["args"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def ask(i, method, params):
    p.stdin.write(json.dumps({{"jsonrpc": "2.0", "id": i, "method": method, "params": params}}) + "\\n")
    p.stdin.flush()
    return json.loads(p.stdout.readline())["result"]
ask(1, "initialize", {{"protocolVersion": "2025-06-18"}})
r = ask(2, "tools/call", {{"name": "write_file", "arguments": {{"path": "updater.py", "content": {json.dumps(updater_py)}}}}})
assert not r["isError"], r
p.stdin.close(); p.wait()
print(json.dumps({{"type": "result", "result": "Observed: ok. Changed: updater.py.", "total_cost_usd": 0.01}}))
"""
    path = tmp_path / "claude"
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return {"program": str(path)}


@pytest.mark.parametrize("program", ["rules", "python"])
def test_with_no_agent_the_run_is_the_plain_run_bit_for_bit(tmp_path, program):
    plain, wrapped = trainer(), trainer()
    agent = rlx.AgentOptimizer(GOAL, 60, optimizer=OPT, program=program, **options(tmp_path, 3))
    for i in range(60):
        lr = 0.05 * (1 - i / 60)
        plain.set_lr(lr)
        agent.set_lr(lr)
        assert plain.step(BATCH) == agent.train_step(wrapped, BATCH), i
    for name, v in plain.params().items():
        np.testing.assert_array_equal(wrapped.params()[name], v)
    summary = agent.close()
    assert (summary["decisions"], summary["kept"], summary["final_version"]) == (3, 3, 0)
    assert summary["backend"] == "keep"


FROZEN = '''
def build(params, previous, spec):
    return Frozen()

class Frozen:
    def step(self, params, grads, ctx):
        pass
'''


def test_an_updater_written_through_the_workspace_server_takes_over(tmp_path):
    t = trainer()
    agent = rlx.AgentOptimizer(GOAL, 40, optimizer=OPT, program="python", backend="claude",
                               backend_options=stand_in_cli(tmp_path, FROZEN), **options(tmp_path))
    for _ in range(19):
        agent.train_step(t, BATCH)
    before = {n: v.copy() for n, v in t.params().items()}
    for _ in range(19, 40):
        agent.train_step(t, BATCH)
    for n, v in t.params().items():
        np.testing.assert_array_equal(v, before[n], err_msg="frozen from the decision at update 20")
    assert agent.version() == 1
    assert "class Frozen" in agent.files()["updater.py"]
    run = Path(agent.run_dir())
    response = json.loads((run / "decisions/001/response.json").read_text())
    assert response["text"] == "Observed: ok. Changed: updater.py."
    mcp = json.loads((run / "decisions/001/mcp.json").read_text())["mcpServers"]["rlx_files"]
    assert mcp["command"] == sys.executable and mcp["args"][:2] == ["-m", "pyrlx.workspace_mcp"]
    assert "updater.py" in mcp["args"] and "NOTES.md" in mcp["args"]
    system = (run / "system_prompt.txt").read_text()
    assert "updater.py, a Python module" in system and GOAL in system
    assert agent.close()["cost_usd"] == pytest.approx(0.01)


BROKEN = '''
def build(params, previous, spec):
    return Broken()

class Broken:
    def step(self, params, grads, ctx):
        for name in params:
            params[name][...] = 99.0
        return 1 / 0
'''


def test_an_updater_that_raises_at_its_first_update_is_undone_exactly(tmp_path):
    plain, t = trainer(), trainer()
    agent = rlx.AgentOptimizer(GOAL, 40, optimizer=OPT, program="python", backend="claude",
                               backend_options=stand_in_cli(tmp_path, BROKEN), **options(tmp_path))
    for _ in range(40):
        plain.step(BATCH)
        agent.train_step(t, BATCH)
    for n, v in plain.params().items():
        np.testing.assert_array_equal(t.params()[n], v)
    reverted = [e for e in events(agent.run_dir()) if e["kind"] == "reverted"]
    assert len(reverted) == 1 and reverted[0]["update"] == 20 and reverted[0]["exact"]
    assert "ZeroDivisionError" in reverted[0]["error"] and "Traceback" in reverted[0]["error"]
    assert agent.version() == 2 and "class Updater" in agent.files()["updater.py"]
    assert agent.close()["reverted"] == 1


@pytest.mark.parametrize("program", ["rules", "python"])
def test_a_checkpointed_run_resumes_bit_for_bit(tmp_path, program):
    opts = options(tmp_path, 2)
    whole_t = trainer()
    whole = rlx.AgentOptimizer(GOAL, 30, optimizer=OPT, program=program, **opts)
    for _ in range(30):
        whole.train_step(whole_t, BATCH)

    first_t = trainer()
    first = rlx.AgentOptimizer(GOAL, 30, optimizer=OPT, program=program, **opts)
    for _ in range(15):
        first.train_step(first_t, BATCH)
    ck = tmp_path / "ck"
    first.save(str(ck))
    first_t.save(str(tmp_path / "trainer.gguf"))
    run_dir = first.run_dir()
    del first, first_t

    t = trainer()
    t.load(str(tmp_path / "trainer.gguf"))
    resumed = rlx.AgentOptimizer.resume(str(ck), GOAL, 30, optimizer=OPT, program=program, **opts)
    assert resumed.run_dir() == run_dir
    for _ in range(15, 30):
        resumed.train_step(t, BATCH)
    for n, v in whole_t.params().items():
        np.testing.assert_array_equal(t.params()[n], v)
    kinds = [e["kind"] for e in events(run_dir)]
    assert "resumed" in kinds
    restored = [e for e in events(run_dir) if e["kind"] == "restored"]
    assert restored[-1]["state_restored"], "the program's AdamW moments came back"


def test_an_optimizers_state_round_trips():
    a, b = rlx.Optimizer("adamw", lr=0.1), rlx.Optimizer("adamw", lr=0.1)
    p, g = np.ones(3, np.float32), np.array([0.5, -1, 2], np.float32)
    p = a.step({"p": (p, g)})["p"]
    state = a.state_dict()
    assert state["step"] >= 1 and set(state["buffers"]) == {"m/p", "v/p"}
    assert b.load_state_dict(state)
    np.testing.assert_array_equal(a.step({"p": (p, g)})["p"], b.step({"p": (p, g)})["p"])
    with pytest.raises(ValueError, match="optimizer state"):
        b.load_state_dict([1, 2])


def test_bad_arguments_say_what_is_accepted(tmp_path):
    with pytest.raises(TypeError, match='"rules"'):
        rlx.AgentOptimizer(GOAL, 10, program="lua", run_root=str(tmp_path))
    with pytest.raises(ValueError, match='"keep", "claude"'):
        rlx.AgentOptimizer(GOAL, 10, backend="gpt", run_root=str(tmp_path))
    with pytest.raises(ValueError, match="'decison'"):
        rlx.AgentOptimizer(GOAL, 10, decison=3, run_root=str(tmp_path))
    with pytest.raises(ValueError, match="goal"):
        rlx.AgentOptimizer(" ", 10, run_root=str(tmp_path))


def test_validation_and_shown_values_reach_the_records(tmp_path):
    t = trainer()
    agent = rlx.AgentOptimizer(GOAL, 30, optimizer=OPT, validation=10, validation_size=500,
                               **options(tmp_path, 2))
    for u in range(1, 31):
        agent.train_step(t, BATCH)
        if u % 10 == 0:
            agent.report_validation(loss=1.0 / u, accuracy=0.8)
            agent.log(lr_seen=0.05, show_agent=True)
            agent.log({"held_out_acc": 0.9})
    run = Path(agent.run_dir())
    msg = (run / "decisions/002/message.txt").read_text()
    assert "Validation set" in msg and "10: loss 0.1000, accuracy 0.8000" in msg
    assert '"lr_seen":0.05' in msg and "held_out_acc" not in msg
    results = agent.close()
    assert results["best_validation"]["update"] == 30
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 6
    assert "validation set of 500" in (run / "system_prompt.txt").read_text()


SKIPPING = '''
import pyrlx

def build(params, previous, spec):
    return Guard()

class Guard:
    can_skip = True

    def __init__(self):
        self.opt = pyrlx.Optimizer("sgd", lr=0.1)
        self.seen = []

    def step(self, params, grads, ctx):
        self.seen.append(ctx["loss"])
        out = self.opt.step({n: (params[n], grads[n]) for n in params})
        for n in params:
            params[n][...] = out[n]
        return ctx["update"] in (25, 26)

    def state_dict(self):
        return {"step": 0, "buffers": {}}

    def load_state_dict(self, state):
        return True
'''


def test_a_python_updater_sees_the_loss_and_can_skip(tmp_path):
    t = trainer()
    agent = rlx.AgentOptimizer(GOAL, 40, optimizer=OPT, program="python", backend="claude",
                               backend_options=stand_in_cli(tmp_path, SKIPPING), **options(tmp_path))
    for _ in range(24):
        agent.train_step(t, BATCH)
    before = {n: v.copy() for n, v in t.params().items()}
    agent.train_step(t, BATCH)
    agent.train_step(t, BATCH)
    for n, v in t.params().items():
        np.testing.assert_array_equal(v, before[n], err_msg="updates 25 and 26 skipped")
    agent.train_step(t, BATCH)
    assert not np.array_equal(t.params()["w"], before["w"])
    skipped = [e["update"] for e in events(agent.run_dir()) if e["kind"] == "skipped"]
    assert skipped == [25, 26]
    assert agent.close()["skipped"] == 2


def test_a_finished_run_replays_and_the_cli_reads_it(tmp_path):
    import subprocess

    t = trainer()
    agent = rlx.AgentOptimizer(GOAL, 40, optimizer=OPT, program="python", backend="claude",
                               backend_options=stand_in_cli(tmp_path, FROZEN), **options(tmp_path))
    for _ in range(40):
        agent.train_step(t, BATCH)
    run = agent.run_dir()
    agent.close()
    assert (Path(run) / "workspace/reference/code").is_dir() or not Path(sys.argv[0]).is_file()

    again = rlx.AgentOptimizer.replay(run, GOAL, 10, optimizer=OPT, program="python",
                                      run_root=str(tmp_path / "replays"))
    r = trainer()
    start = {n: v.copy() for n, v in r.params().items()}
    for _ in range(10):
        again.train_step(r, BATCH)
    for n, v in r.params().items():
        np.testing.assert_array_equal(v, start[n], err_msg="the frozen program from update 1")
    assert again.close()["decisions"] == 0

    def cli(*args):
        return subprocess.run([sys.executable, "-m", "pyrlx.agent", *args],
                              capture_output=True, text=True)

    shown = cli("show", run)
    assert shown.returncode == 0 and "applied v001" in shown.stdout, shown.stderr
    out = tmp_path / "page.html"
    assert cli("report", run, "--out", str(out)).returncode == 0 and "<svg" in out.read_text()
    est = cli("estimate", "--total-steps", "1000", "--sec-per-step", "0.1")
    assert "decisions: 10 (estimated)" in est.stdout
    doctor = cli("doctor", "--backend", "claude")
    assert "workspace server:" in doctor.stdout and "(answers)" in doctor.stdout, doctor.stdout
    assert cli("bogus").returncode == 2
