#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Serve the rlx-web static demo (no wasm rebuild).

ES modules + wasm require http:// — do not open HTML as file://.

Examples:
    python3 crates/bindings/rlx-web/serve.py
    python3 crates/bindings/rlx-web/serve.py --backend npx
    python3 crates/bindings/rlx-web/serve.py --backend miniserve --port 8080
    python3 crates/bindings/rlx-web/serve.py --open vision-bench.html
"""

from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import socketserver
import subprocess
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"
PKG = WEB_DIR / "pkg" / "rlx_web.js"

BACKENDS = ("python", "npx", "miniserve", "basic-http-server")


def ensure_bundle() -> None:
    if not PKG.exists():
        raise SystemExit(
            f"wasm bundle missing: {PKG}\n"
            "  Build first:\n"
            "    just build-web --all\n"
            "    python3 crates/bindings/rlx-web/build.py --all"
        )


def serve_python(port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(WEB_DIR))
    with socketserver.TCPServer(("", port), handler) as httpd:
        httpd.serve_forever()


def serve_npx(port: int) -> None:
    if shutil.which("npx") is None:
        raise SystemExit("npx not found — install Node.js or use --backend python")
    subprocess.run(
        ["npx", "--yes", "serve", str(WEB_DIR), "-l", str(port)],
        check=True,
    )


def serve_miniserve(port: int) -> None:
    exe = shutil.which("miniserve")
    if exe is None:
        raise SystemExit(
            "miniserve not found.\n"
            "  Install: cargo install miniserve\n"
            "  Or use:  --backend python | npx"
        )
    subprocess.run([exe, "-p", str(port), str(WEB_DIR)], check=True)


def serve_basic_http(port: int) -> None:
    exe = shutil.which("basic-http-server")
    if exe is None:
        raise SystemExit(
            "basic-http-server not found.\n"
            "  Install: cargo install basic-http-server\n"
            "  Or use:  --backend python | npx"
        )
    subprocess.run([exe, str(WEB_DIR), "--addr", f"127.0.0.1:{port}"], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve rlx-web static files.")
    ap.add_argument(
        "--backend",
        choices=BACKENDS,
        default="python",
        help="static file server (default: python stdlib)",
    )
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--open",
        default="vision-bench.html",
        help="page to open in browser (default: vision-bench.html)",
    )
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--skip-check", action="store_true", help="do not require pkg/ bundle")
    args = ap.parse_args()

    if not args.skip_check:
        ensure_bundle()

    url = f"http://127.0.0.1:{args.port}/{args.open}"
    print(f"Serving {WEB_DIR}")
    print(f"  backend: {args.backend}")
    print(f"  URL:     {url}")
    print("  Pages:   index.html (MLP), vision-bench.html (vision)")
    print("  Ctrl-C to stop\n")

    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    runners = {
        "python": serve_python,
        "npx": serve_npx,
        "miniserve": serve_miniserve,
        "basic-http-server": serve_basic_http,
    }
    try:
        runners[args.backend](args.port)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
