#!/usr/bin/env python3
"""Cross-platform flasher for the nRF52840 USB Dongle.

Tries probe-rs (SWD) first if a debug probe is detected, otherwise
falls back to nrfutil DFU through the Nordic open USB bootloader.

Run from anywhere — the script locates the firmware crate relative to
its own path. macOS, Linux, and Windows all supported.

Usage:
    python3 firmware/scripts/flash.py            # auto-pick path
    python3 firmware/scripts/flash.py --dfu      # force DFU
    python3 firmware/scripts/flash.py --swd      # force SWD
    python3 firmware/scripts/flash.py --port COM4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
FIRMWARE_DIR = HERE.parent
CRATE_DIR = FIRMWARE_DIR.parent
ELF = FIRMWARE_DIR / "target" / "thumbv7em-none-eabihf" / "release" / "rlx-cortexm-firmware"
HEX = ELF.with_suffix(".hex")
PKG = FIRMWARE_DIR / "target" / "rlx-cortexm-firmware-dfu.zip"
CHIP = "nRF52840_xxAA"


def run(cmd, *, cwd=None, check=True, capture=False):
    pretty = " ".join(str(c) for c in cmd)
    print(f"  $ {pretty}", file=sys.stderr)
    return subprocess.run(cmd, cwd=cwd, check=check,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None,
                          text=True)


def have(prog: str) -> bool:
    return shutil.which(prog) is not None


def find_objcopy() -> str:
    """Find an objcopy that can produce ihex from an ARM ELF.
    Order: rust-objcopy (from rustup llvm-tools), llvm-objcopy on PATH,
    arm-none-eabi-objcopy.
    """
    for c in ("rust-objcopy", "llvm-objcopy", "arm-none-eabi-objcopy"):
        if have(c):
            return c
    # Last resort: dig into rustup sysroot.
    try:
        sysroot = subprocess.check_output(
            ["rustc", "--print", "sysroot"], text=True
        ).strip()
        host = subprocess.check_output(
            ["rustc", "-vV"], text=True)
        host = next(line.split()[1] for line in host.splitlines()
                    if line.startswith("host:"))
        candidate = Path(sysroot) / "lib" / "rustlib" / host / "bin" / "llvm-objcopy"
        if sys.platform == "win32":
            candidate = candidate.with_suffix(".exe")
        if candidate.is_file():
            return str(candidate)
    except Exception:
        pass
    print("error: no objcopy found. Run `python3 tools/setup.py` first.",
          file=sys.stderr)
    sys.exit(1)


def find_bootloader_port() -> Optional[str]:
    """Locate the Nordic open USB bootloader's serial port.

    The bootloader advertises VID 0x1915 PID 0x521f. We deliberately do
    NOT fall back to "any CDC port" here — flashing the wrong device
    would be confusing at best. Returns None if no bootloader is found.
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        print("error: pyserial missing. Run `python3 tools/setup.py` first.",
              file=sys.stderr)
        sys.exit(1)
    for p in list_ports.comports():
        if (p.vid, p.pid) == (0x1915, 0x521f):
            return p.device
    return None


def cargo_build():
    print("building firmware (release)…", file=sys.stderr)
    run(["cargo", "build", "--release"], cwd=FIRMWARE_DIR)


def has_probe() -> bool:
    if not have("probe-rs"):
        return False
    try:
        out = run(["probe-rs", "list"], capture=True, check=False)
    except Exception:
        return False
    if out.returncode != 0:
        return False
    text = out.stdout + (out.stderr or "")
    # `probe-rs list` prints "No debug probes were found." when empty.
    return "No debug probes" not in text and bool(text.strip())


def flash_swd():
    if not have("probe-rs"):
        print("error: probe-rs not installed. "
              "Run `python3 tools/setup.py --probe-rs`.", file=sys.stderr)
        sys.exit(1)
    cargo_build()
    run(["probe-rs", "run", "--chip", CHIP, str(ELF)])


def flash_dfu(port: Optional[str]):
    if not have("nrfutil"):
        print("error: nrfutil not installed. Run `python3 tools/setup.py`.",
              file=sys.stderr)
        sys.exit(1)
    cargo_build()
    objcopy = find_objcopy()
    run([objcopy, "-O", "ihex", str(ELF), str(HEX)])

    if PKG.exists():
        PKG.unlink()
    run([
        "nrfutil", "pkg", "generate",
        "--hw-version", "52",
        "--sd-req", "0x00",
        "--application-version", "1",
        "--application", str(HEX),
        str(PKG),
    ])

    p = port or find_bootloader_port()
    if not p:
        print("error: no serial port found. Put the dongle in DFU mode "
              "(press the small RESET button on the side; red LED should "
              "pulse) and re-run.", file=sys.stderr)
        sys.exit(1)
    print(f"flashing via DFU on {p}…", file=sys.stderr)
    run(["nrfutil", "dfu", "usb-serial", "-pkg", str(PKG), "-p", p])


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--swd", action="store_true",
                   help="Force SWD via probe-rs.")
    g.add_argument("--dfu", action="store_true",
                   help="Force DFU via nrfutil.")
    ap.add_argument("--port", help="Serial port override for DFU.")
    args = ap.parse_args()

    if args.swd:
        flash_swd()
    elif args.dfu:
        flash_dfu(args.port)
    else:
        # Auto: if a debug probe is connected, prefer SWD; else DFU.
        if has_probe():
            print("debug probe detected — using SWD via probe-rs.",
                  file=sys.stderr)
            flash_swd()
        else:
            print("no debug probe — using DFU via nrfutil.",
                  file=sys.stderr)
            flash_dfu(args.port)
    print("✓ flash complete.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
