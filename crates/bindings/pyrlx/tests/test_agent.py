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


@pytest.mark.parametrize("program", ["rules", "python"])
def test_save_with_the_trainer_checkpoints_the_whole_run(tmp_path, program):
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
    first.save(str(ck), trainer=first_t)
    assert (ck / "trainer.gguf").is_file()
    del first, first_t

    t = trainer()
    resumed = rlx.AgentOptimizer.resume(str(ck), GOAL, 30, optimizer=OPT, program=program, trainer=t, **opts)
    for _ in range(15, 30):
        resumed.train_step(t, BATCH)
    for n, v in whole_t.params().items():
        np.testing.assert_array_equal(t.params()[n], v)


SNOOPS = """\
import sys
import numpy as np

def build(params, previous, spec):
    return U()

class U:
    def step(self, params, grads, ctx):
        main = sys.modules["__main__"]
        for n in params:
            params[n][...] = params[n] - 0.05 * grads[n]
"""


def test_a_python_updater_reaching_past_its_program_is_flagged_not_stopped(tmp_path):
    agent = rlx.AgentOptimizer(GOAL, 40, optimizer=OPT, program="python", backend="claude",
                               backend_options=stand_in_cli(tmp_path, SNOOPS), **options(tmp_path))
    t = trainer()
    for _ in range(40):
        agent.train_step(t, BATCH)
    run_dir = agent.run_dir()
    results = agent.close()
    assert results["applied"] == 1, "flags never stop a program"
    flags = [e for e in events(run_dir) if e["kind"] == "boundary"]
    assert {f["category"] for f in flags} == {"introspection"}, flags
    assert flags[0]["version"] == 1 and flags[0]["file"] == "updater.py"
    assert results["boundary_flags"] == len(flags)
    import subprocess
    out = subprocess.run([sys.executable, "-m", "pyrlx.agent", "report", run_dir], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Boundary flags:</b> 1 static findings" in Path(run_dir, "report.html").read_text()


def test_the_boundary_check_names_each_kind_of_reach():
    from pyrlx import _boundary

    src = "\n".join([
        "import os, gc, subprocess",
        "import pickle as pk",
        "x = gc.get_objects()",
        "y = os.listdir('.')",
        "z = eval('1')",
        "f = lambda fr: fr.f_back",
        "pk.load(open('a'))",
        "def g(model): model.weight = 1",
    ])
    cats = {(f["line"], f["category"]) for f in _boundary.check({"updater.py": src, "notes.md": "import os"})}
    assert (1, "introspection") in cats and (1, "io") in cats  # gc, subprocess
    assert (3, "introspection") in cats
    assert (4, "io") in cats
    assert (5, "dynamic") in cats
    assert (6, "introspection") in cats
    assert (7, "io") in cats
    assert (8, "model_mutation") in cats
    assert _boundary.check({"updater.py": "def (:"}) == [], "a syntax error is the build's to report"


def test_the_python_example_is_the_starting_program_and_check_program_runs_it(tmp_path):
    opts = options(tmp_path, 1)
    agent = rlx.AgentOptimizer(GOAL, 30, optimizer=OPT, program="python", **opts)
    plain_t = trainer()
    t = trainer()
    for _ in range(30):
        agent.train_step(t, BATCH)
        plain_t.step(BATCH)
    ws = Path(agent.run_dir(), "workspace")
    example = (ws / "reference/example_updater.py").read_text()
    assert "Nothing here is a recommendation" in example and "LR_CONTROLLER" not in example

    # Installed by an agent, the example trains as the starting program did.
    agent2 = rlx.AgentOptimizer(GOAL, 30, optimizer=OPT, program="python", backend="claude",
                                backend_options=stand_in_cli(tmp_path, example), **opts)
    t2 = trainer()
    for _ in range(30):
        agent2.train_step(t2, BATCH)
    assert agent2.version() == 1
    for n, v in t.params().items():
        np.testing.assert_array_equal(t2.params()[n], v)

    # check-program builds it in a process of its own.
    ws2 = Path(agent2.run_dir(), "workspace")
    import subprocess
    out = subprocess.run([sys.executable, "-m", "pyrlx.agent", "check-program", str(ws2), "--updates", "2"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr + out.stdout
    assert "builds for 2 params" in out.stdout and "finite" in out.stdout

    aggressive = rlx.AgentOptimizer(GOAL, 30, optimizer=OPT, program="python", prompt="aggressive",
                                    **options(tmp_path / "a", 1))
    aggressive.train_step(trainer(), BATCH)
    text = Path(aggressive.run_dir(), "workspace/reference/example_updater.py").read_text()
    assert "LR_CONTROLLER = False" in text
    compile(text, "example_updater.py", "exec")


def device_trainer():
    init = {"w": np.array([0.3, -0.2, 0.1, 0.4], np.float32), "b": np.zeros(2, np.float32)}
    try:
        return rlx.Trainer(graph(), wrt=["w", "b"], init=init, optimizer=OPT, device="metal", resident=True)
    except Exception:
        return None


GROUPS_JSON = json.dumps({"groups": [{"names": ["w"], "optimizer": {"kind": "muon", "lr": 0.02}},
                                     {"names": ["*"], "optimizer": {"kind": "adamw", "lr": 0.05}}]})


def stand_in_rules_cli(tmp_path, updater_json):
    """`stand_in_cli`, writing updater.json."""
    backend = stand_in_cli(tmp_path, "")
    path = Path(backend["program"])
    path.write_text(path.read_text().replace('"updater.py", "content": ""',
                                             f'"updater.json", "content": {json.dumps(updater_json)}'))
    return backend


def test_on_a_resident_trainer_the_program_runs_on_the_device(tmp_path):
    plain = device_trainer()
    if plain is None:
        pytest.skip("no metal backend in this build")
    plain_losses = [plain.step(BATCH) for _ in range(30)]
    t = device_trainer()
    agent = rlx.AgentOptimizer(GOAL, 30, optimizer=OPT, **options(tmp_path, 2))
    losses = [agent.train_step(t, BATCH) for _ in range(30)]
    np.testing.assert_allclose(losses, plain_losses, rtol=1e-6)

    t = device_trainer()
    agent = rlx.AgentOptimizer(GOAL, 30, optimizer=OPT, backend="claude",
                               backend_options=stand_in_rules_cli(tmp_path, GROUPS_JSON),
                               **options(tmp_path, 1, blocking=True, trial_steps=5))
    losses = [agent.train_step(t, BATCH) for _ in range(30)]
    assert agent.version() == 1 and all(np.isfinite(losses))
    kinds = [e["kind"] for e in events(agent.run_dir())]
    assert "applied" in kinds and "trial_passed" in kinds

    # A Python program cannot run there: the run says so.
    with pytest.raises(Exception, match="device"):
        py = rlx.AgentOptimizer(GOAL, 30, optimizer=OPT, program="python", **options(tmp_path / "py", 1))
        py.train_step(device_trainer(), BATCH)


def test_a_rules_run_from_python_offers_check_program(tmp_path):
    agent = rlx.AgentOptimizer(GOAL, 30, optimizer=OPT, backend="claude",
                               backend_options=stand_in_rules_cli(tmp_path, GROUPS_JSON), **options(tmp_path))
    t = trainer()
    for _ in range(30):
        agent.train_step(t, BATCH)
    ws = Path(agent.run_dir(), "workspace")
    spec = json.loads((ws / "reference/check.json").read_text())
    assert spec["host"]["program"] == "rules"
    import subprocess
    out = subprocess.run([sys.executable, "-m", "pyrlx.agent", "check-program", str(ws)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr + out.stdout
    assert "builds for 2 params" in out.stdout
