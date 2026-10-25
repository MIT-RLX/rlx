# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""The runtime examples (`examples/{inference,training,loading,distributed,
backends,models}.py`) run and print what they claim. The Swift and JS examples of the
same names check the same lines, so the bindings agree on the answers too."""

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"
WORKSPACE = EXAMPLES.parents[3]

CASES = {
    "inference": ["compiled on: cpu", "probs sum: 1.0000"],
    "training": [
        "trainer w: 1.999 -0.999 0.498",
        "checkpoint: 3 tensors",
        "unchanged: True",
        "manual loss: 0.000000 w: 2.000 -1.000 0.500",
    ],
    "loading": ["gguf architecture: demo", "gguf f32 exact: True", "exact: True", "argmax: 1"],
    "distributed": [
        "all_reduce sum: [6.0]",
        "all_gather: [1.0, 2.0, 3.0]",
        "rank 0 w: 1.500 -2.000 0.500",
        "rank 1 w agrees: True",
        "rank 2 w agrees: True",
    ],
    "backends": ["session cpu: max|Δ| vs first = 0.00e+00", "router ran on:"],
    "models": [
        "model: affine (graph) on cpu",
        "model run: 2.000 3.000",
        "plugin model: count (text-generation)",
        "model text: defg",
        "model ids: 3 4 5",
        "chat: defg",
        "voice: 5 samples at 16000 Hz",
        "heard: hello",
        "embed: 5 2 3 0",
    ],
}


def model_plugin():
    """The reference model plugin (`rlx-model-plugin`), built if need be."""
    if os.environ.get("RLX_MODEL_PLUGINS"):
        return os.environ["RLX_MODEL_PLUGINS"]
    if not hasattr(__import__("pyrlx"), "Model"):
        pytest.skip("pyrlx built without `models`")
    if shutil.which("cargo") is None:
        pytest.skip("no cargo to build rlx-model-plugin")
    subprocess.run(["cargo", "build", "-q", "-p", "rlx-model-plugin"], cwd=WORKSPACE, check=True)
    name = {"darwin": "librlx_model_plugin.dylib", "win32": "rlx_model_plugin.dll"}.get(
        sys.platform, "librlx_model_plugin.so")
    target = os.environ.get("CARGO_TARGET_DIR", WORKSPACE / "target")
    return str(pathlib.Path(target) / "debug" / name)


@pytest.mark.parametrize("name", sorted(CASES))
def test_example_runs(name):
    env = dict(os.environ)
    if name == "models":
        env["RLX_MODEL_PLUGINS"] = model_plugin()
    out = subprocess.run(
        [sys.executable, str(EXAMPLES / f"{name}.py")],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    for want in CASES[name]:
        assert want in out.stdout, f"{name}: missing {want!r} in\n{out.stdout}{out.stderr}"
