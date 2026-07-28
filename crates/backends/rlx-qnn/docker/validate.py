#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
# rlx-qnn Docker validation driver — cross-platform (macOS / Linux / Windows).
#
#   python3 validate.py harness-test [--dims M K N] [--artifacts DIR]
#       SDK-free. Builds nothing proprietary: emits the artifacts, then runs the
#       host harness (verify.py) end-to-end in a stock python image with a numpy
#       stand-in for qnn-net-run. Needs only Docker. Good for CI.
#
#   python3 validate.py run [--dims M K N] [--artifacts DIR]
#       Real validation. Builds the model lib with qnn-model-lib-generator and
#       runs it on the QNN x86 reference backend (libQnnCpu.so) inside Docker.
#       Needs Docker AND the proprietary QNN SDK: set QNN_SDK_ROOT to its path.
#
# The QNN SDK is gated behind a Qualcomm account (Qualcomm Package Manager) and
# is NOT publicly pullable, so `run` can only work once you provide it; the SDK
# is mounted read-only into the container, never baked into an image.

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]  # crates/rlx-qnn/docker -> repo root
PYTHON_IMAGE = "python:3.12-slim"
VALIDATE_IMAGE = "rlx-qnn-validate"


def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)


def require(tool):
    if shutil.which(tool) is None:
        sys.exit(f"error: `{tool}` not found on PATH")


def emit_artifacts(dims, artifacts):
    """Return a dir holding qnn_model.cpp / verify.py / run_qnn.sh.

    With --artifacts, use (and trust) a pre-emitted dir; otherwise emit a fresh
    set with the rlx-qnn-emit binary via cargo.
    """
    if artifacts:
        d = Path(artifacts).resolve()
        if not (d / "run_qnn.sh").exists():
            sys.exit(f"error: {d} has no run_qnn.sh — not an rlx-qnn artifact dir")
        return d
    require("cargo")
    d = Path(tempfile.mkdtemp(prefix="rlx-qnn-"))
    run(
        ["cargo", "run", "-q", "-p", "rlx-qnn", "--bin", "rlx-qnn-emit",
         "--", *map(str, dims), str(d)],
        cwd=REPO_ROOT,
    )
    return d


def harness_test(dims, artifacts):
    require("docker")
    work = emit_artifacts(dims, artifacts)
    shutil.copy(HERE / "mock_net_run.py", work / "mock_net_run.py")
    m, k, n = dims
    script = (
        "set -e\n"
        "bash -n run_qnn.sh && echo 'run_qnn.sh: shell-syntax OK'\n"
        "pip install -q numpy\n"
        "python3 verify.py --gen\n"
        f"python3 mock_net_run.py {m} {k} {n}\n"
        "python3 verify.py --check\n"
    )
    run([
        "docker", "run", "--rm",
        "-v", f"{work}:/work", "-w", "/work",
        PYTHON_IMAGE, "bash", "-c", script,
    ])
    print("\nharness-test PASSED — host harness validated (QNN lowering still needs `run`).")


def real_run(dims, artifacts):
    require("docker")
    sdk = os.environ.get("QNN_SDK_ROOT")
    if not sdk:
        sys.exit("error: set QNN_SDK_ROOT to your Qualcomm AI Engine Direct SDK")
    sdk = Path(sdk).resolve()
    if not (sdk / "bin" / "envsetup.sh").exists():
        sys.exit(f"error: {sdk} doesn't look like a QNN SDK (no bin/envsetup.sh)")
    work = emit_artifacts(dims, artifacts)
    run(["docker", "build", "-t", VALIDATE_IMAGE, str(HERE)])
    run([
        "docker", "run", "--rm",
        "-v", f"{sdk}:/opt/qnn:ro",
        "-v", f"{work}:/work",
        VALIDATE_IMAGE,
    ])
    print("\nrun PASSED — QNN model built + executed on the x86 reference backend.")


def main():
    ap = argparse.ArgumentParser(description="rlx-qnn Docker validation driver")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("harness-test", "run"):
        p = sub.add_parser(name)
        p.add_argument("--dims", nargs=3, type=int, metavar=("M", "K", "N"),
                       default=[8, 16, 4])
        p.add_argument("--artifacts", help="use a pre-emitted artifact dir "
                                           "instead of running rlx-qnn-emit")
    args = ap.parse_args()
    if args.cmd == "harness-test":
        harness_test(args.dims, args.artifacts)
    else:
        real_run(args.dims, args.artifacts)


if __name__ == "__main__":
    main()
