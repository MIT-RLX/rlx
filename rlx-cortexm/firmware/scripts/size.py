#!/usr/bin/env python3
"""Print firmware section sizes — cross-platform.

Looks for llvm-size in PATH first, then in the rustup sysroot.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ELF = HERE.parent / "target" / "thumbv7em-none-eabihf" / "release" / "rlx-cortexm-firmware"


def find_size_tool() -> str:
    for c in ("rust-size", "llvm-size"):
        if shutil.which(c):
            return c
    sysroot = subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    host = next(line.split()[1] for line in
                subprocess.check_output(["rustc", "-vV"], text=True).splitlines()
                if line.startswith("host:"))
    candidate = Path(sysroot) / "lib" / "rustlib" / host / "bin" / "llvm-size"
    if sys.platform == "win32":
        candidate = candidate.with_suffix(".exe")
    if candidate.is_file():
        return str(candidate)
    print("error: no size tool found. Run `just setup`.", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    if not ELF.is_file():
        print(f"error: {ELF} not found. Run `just build-fw` first.",
              file=sys.stderr)
        return 1
    subprocess.run([find_size_tool(), str(ELF)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
