#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""One-command setup for the rlx-cortexm crate.

Installs everything a developer needs to train, build, flash, and test
the nRF52840 MNIST demo. Cross-platform: macOS, Linux, Windows.

Run:
    python3 tools/setup.py            # required dev tools only
    python3 tools/setup.py --probe-rs # also install probe-rs-tools (slow)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from typing import Iterable


def run(cmd: Iterable[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = list(cmd)
    pretty = " ".join(cmd)
    print(f"  $ {pretty}", file=sys.stderr)
    return subprocess.run(cmd, check=check,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None,
                          text=True)


def have(prog: str) -> bool:
    return shutil.which(prog) is not None


def step(title: str):
    print(f"\n── {title} ─────────────────", file=sys.stderr)


def install_just():
    step("just (recipe runner)")
    if have("just"):
        print("  already installed", file=sys.stderr)
        return
    if not have("cargo"):
        print("  ERROR: cargo not in PATH; install Rust first", file=sys.stderr)
        sys.exit(1)
    run(["cargo", "install", "just", "--locked"])


def install_rust_target():
    step("rustup target: thumbv7em-none-eabihf")
    out = run(["rustup", "target", "list", "--installed"], capture=True)
    if "thumbv7em-none-eabihf" in out.stdout:
        print("  already installed", file=sys.stderr)
    else:
        run(["rustup", "target", "add", "thumbv7em-none-eabihf"])


def install_llvm_tools():
    step("rustup component: llvm-tools-preview (for size, objcopy)")
    out = run(["rustup", "component", "list", "--installed"], capture=True)
    if "llvm-tools" in out.stdout:
        print("  already installed", file=sys.stderr)
    else:
        run(["rustup", "component", "add", "llvm-tools-preview"])


def install_python_deps():
    step("python: pyserial (for the host client)")
    try:
        import serial  # noqa: F401
        print("  already installed", file=sys.stderr)
        return
    except ImportError:
        pass
    # Prefer pipx if available (clean isolation); fall back to pip --user.
    if have("pipx"):
        run(["pipx", "install", "pyserial"])
    else:
        run([sys.executable, "-m", "pip", "install", "--user", "pyserial"])


def install_nrfutil():
    step("nrfutil (for DFU flashing)")
    if have("nrfutil"):
        print("  already installed", file=sys.stderr)
        return
    if have("pipx"):
        run(["pipx", "install", "nrfutil"])
    else:
        # Direct pip install of nrfutil tends to break system Python on
        # macOS/Homebrew; require pipx if available, else fall back.
        run([sys.executable, "-m", "pip", "install", "--user", "nrfutil"])


def install_probe_rs():
    step("probe-rs-tools (for SWD flashing — slow, ~5–10 min)")
    if have("probe-rs"):
        print("  already installed", file=sys.stderr)
        return
    if not have("cargo"):
        print("  ERROR: cargo not in PATH; install Rust first", file=sys.stderr)
        sys.exit(1)
    run(["cargo", "install", "probe-rs-tools", "--locked"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-rs", action="store_true",
                    help="Also install probe-rs-tools for SWD flashing.")
    ap.add_argument("--no-nrfutil", action="store_true",
                    help="Skip nrfutil — only needed for DFU flashing.")
    args = ap.parse_args()

    install_just()
    install_rust_target()
    install_llvm_tools()
    install_python_deps()
    if not args.no_nrfutil:
        install_nrfutil()
    if args.probe_rs:
        install_probe_rs()

    print("\n✓ setup complete. Try `just demo` (or `just --list`).",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
