#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# GPL-3.0-only. See LICENSE.
"""Build the rlx-web WebAssembly bundle and (optionally) serve the demo.

One command, all platforms (macOS / Linux / Windows):

    python3 crates/rlx-web/build.py            # CPU build -> web/pkg
    python3 crates/rlx-web/build.py --webgpu   # also bring up a WebGPU device
    python3 crates/rlx-web/build.py --serve     # build + serve the demo

The `wasm-bindgen` CLI must match the `wasm-bindgen` crate version. If it is
missing or mismatched, this script prints the exact `cargo install` command.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

CRATE = "rlx-web"
WASM_ARTIFACT = "rlx_web.wasm"  # cargo replaces '-' with '_'
OUT_NAME = "rlx_web"
TARGET = "wasm32-unknown-unknown"

HERE = Path(__file__).resolve().parent          # crates/rlx-web
WORKSPACE = HERE.parent.parent                  # repo root
WEB_DIR = HERE / "web"
PKG_DIR = WEB_DIR / "pkg"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("·", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def wasm_bindgen_crate_version() -> str:
    """Read the locked wasm-bindgen version from `cargo metadata`."""
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1"],
        cwd=WORKSPACE, check=True, capture_output=True, text=True,
    )
    meta = json.loads(out.stdout)
    for pkg in meta["packages"]:
        if pkg["name"] == "wasm-bindgen":
            return pkg["version"]
    raise SystemExit("wasm-bindgen not found in cargo metadata")


def ensure_wasm_bindgen(expected: str) -> str:
    exe = shutil.which("wasm-bindgen")
    install = f"cargo install wasm-bindgen-cli --version {expected}"
    if exe is None:
        raise SystemExit(
            f"wasm-bindgen CLI not found.\n  Install the matching version:\n    {install}"
        )
    got = subprocess.run([exe, "--version"], check=True, capture_output=True, text=True)
    ver = got.stdout.strip().split()[-1]
    if ver != expected:
        raise SystemExit(
            f"wasm-bindgen CLI {ver} != crate {expected} (they must match).\n"
            f"  Fix:\n    {install}"
        )
    return exe


def target_dir() -> Path:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=WORKSPACE, check=True, capture_output=True, text=True,
    )
    return Path(json.loads(out.stdout)["target_directory"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the rlx-web wasm bundle.")
    ap.add_argument("--webgpu", action="store_true", help="enable the WebGPU compute path")
    ap.add_argument("--webgl", action="store_true", help="enable the WebGL2 GPGPU path")
    ap.add_argument("--all", action="store_true", help="enable every GPU backend")
    ap.add_argument("--serve", action="store_true", help="serve the demo after building")
    ap.add_argument("--debug", action="store_true", help="debug build (default: release)")
    ap.add_argument("--port", type=int, default=8000, help="port for --serve")
    args = ap.parse_args()

    expected = wasm_bindgen_crate_version()
    bindgen = ensure_wasm_bindgen(expected)

    run(["rustup", "target", "add", TARGET])

    features = []
    if args.webgpu or args.all:
        features.append("webgpu")
    if args.webgl or args.all:
        features.append("webgl")

    cargo = ["cargo", "build", "-p", CRATE, "--target", TARGET]
    if not args.debug:
        cargo.append("--release")
    if features:
        cargo += ["--features", ",".join(features)]
    run(cargo, cwd=WORKSPACE)

    profile = "debug" if args.debug else "release"
    wasm = target_dir() / TARGET / profile / WASM_ARTIFACT
    if not wasm.exists():
        raise SystemExit(f"expected wasm artifact not found: {wasm}")

    PKG_DIR.mkdir(parents=True, exist_ok=True)
    run([bindgen, "--target", "web", "--out-dir", str(PKG_DIR),
         "--out-name", OUT_NAME, str(wasm)])

    print(f"\n✓ bundle ready: {PKG_DIR}")
    print(f"  backends: {', '.join(['cpu', *features]) if features else 'cpu'}")

    if args.serve:
        import functools
        import http.server
        import socketserver

        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB_DIR))
        url = f"http://localhost:{args.port}/"
        print(f"\nServing {WEB_DIR} at {url}  (Ctrl-C to stop)")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        with socketserver.TCPServer(("", args.port), handler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nstopped.")


if __name__ == "__main__":
    main()
