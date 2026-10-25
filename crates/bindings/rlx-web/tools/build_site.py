#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Assemble the tab-cluster demo into a standalone static site.

    python3 crates/bindings/rlx-web/tools/build_site.py           # → ./examples/tab-cluster
    python3 crates/bindings/rlx-web/tools/build_site.py --out /tmp/site --open
    python3 crates/bindings/rlx-web/tools/build_site.py --all-pages --out _site  # GitHub Pages
    python3 crates/bindings/rlx-web/tools/build_site.py --all-pages --out _site \
        --qwen ../rlx-models/crates/rlx-models-web/web     # + the Qwen3.5 chat, at qwen/

The output is deployable to any static host — GitHub Pages, Netlify, S3 — with
no server component: the page builds, trains and distributes entirely in the
browser, and the only runtime network traffic is WebRTC signalling to public
relays when someone opens a room.

Files are **discovered, not listed**. The script walks the page's real
dependency graph — `<script src>`, `<img src>`, and `import … from "./x.js"`
chains — so adding a module to the demo does not require editing a manifest
here. A hardcoded list is the usual way a deploy silently ships a stale or
incomplete bundle.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRATE = HERE.parent
WEB = CRATE / "web"
WORKSPACE = CRATE.parent.parent.parent

ENTRY = "cluster.html"
# A static host serves `index.html` at the root; the page itself uses only
# relative paths, so renaming it is safe.
ENTRY_AS = "index.html"

# `--all-pages`: every page under its own name, the demo at the root. The
# pages link each other by these names (the nav in index.html and
# vision-bench.html), so none is renamed.
PAGES = ["index.html", "vision-bench.html", "cluster.html"]

# `<script src>`, `<img src>`, `<link href>` — local only.
HTML_REFS = re.compile(r'(?:src|href)="(?!https?:|//|#|mailto:)([^"]+)"')
# `import x from "./y.js"`, `export … from "./y.js"`, `import("./y.js")`, and
# **worker constructors** — `new Worker("./y.js")` is not an import, and missing
# it shipped a site that loaded fine and then died creating its worker. The
# worker's own imports are then followed from there, which is how
# `cluster-protocol.js` arrives.
JS_REFS = re.compile(
    r'(?:from|import)\s*\(?\s*["\'](\.[^"\']+)["\']'
    r'|new\s+(?:Shared)?Worker\s*\(\s*["\'](\.[^"\']+)["\']'
    r'|importScripts\s*\(\s*["\'](\.[^"\']+)["\']'
    # A worker URL made first, `new URL("./y.js", import.meta.url)`, then
    # handed to `new Worker(url)` (rlx-models-web's pipeline.js).
    r'|new\s+URL\s*\(\s*["\'](\.[^"\']+\.m?js)["\']\s*,\s*import\.meta\.url'
)

# `--qwen DIR`: rlx-models-web's chat page (Qwen3.5 on WebGPU, weights streamed
# from the Hugging Face Hub) under `qwen/`, with its own wasm package.
QWEN_ENTRY = "chat.html"
QWEN_AT = "qwen"
QWEN_NAV = '      <a href="./qwen/chat.html">Qwen chat</a>\n    </nav>'


def discover(entry: Path, web: Path = WEB) -> set[Path]:
    """Every local file the page needs, following imports transitively."""
    seen: set[Path] = set()
    queue = [entry]
    while queue:
        path = queue.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        text = path.read_text(errors="replace")
        pattern = HTML_REFS if path.suffix in {".html", ".htm"} else JS_REFS
        for match in pattern.findall(text):
            # `findall` yields a tuple when the pattern has several groups.
            ref = next((g for g in (match if isinstance(match, tuple) else (match,)) if g), "")
            if not ref:
                continue
            ref = ref.split("?")[0].split("#")[0]
            target = (path.parent / ref).resolve()
            if web.resolve() in target.parents or target.parent == web.resolve():
                queue.append(target)
            elif target.is_file():
                queue.append(target)
    return seen


def ensure_bundle() -> None:
    """The wasm bundle must exist and carry the GPU backends."""
    pkg = WEB / "pkg" / "rlx_web.js"
    if pkg.exists():
        return
    print("· wasm bundle missing — building with --all")
    subprocess.run(
        [sys.executable, str(CRATE / "build.py"), "--all"], cwd=WORKSPACE, check=True
    )


def check_ascii_prologue(page: Path) -> str | None:
    """Everything before `<head>` must be ASCII.

    Hugging Face Spaces injects a `window.huggingface = {...}` script into the
    served HTML, and its insertion offset does not survive multi-byte UTF-8
    before `<head>`: the offset is computed in one unit and applied in the
    other, so each multi-byte character shifts the insertion earlier. One em
    dash in the licence comment (3 bytes, 1 char) was enough to drift it by two
    and land the script *inside the opening tag*:

        <html lang="en">
          <hea<script>window.huggingface={...};</script>d>

    The browser then parsed `<hea<script>` as a bogus tag and rendered the
    leftover `d>` as text at the top of the page. The page's own markup was
    valid; the drift was entirely in the prologue's byte count. Keeping that
    region ASCII makes the offset exact, whichever unit is used.
    """
    text = page.read_text()
    at = text.find("<head")
    if at < 0:
        return "no <head> in the page"
    prologue = text[:at]
    offenders = sorted({c for c in prologue if ord(c) > 127})
    if offenders:
        drift = len(prologue.encode()) - len(prologue)
        return (
            f"non-ASCII before <head> ({', '.join(repr(c) for c in offenders)}) "
            f"shifts a byte/char offset by {drift}; Hugging Face's injected "
            "script would land inside the opening tag"
        )
    return None


def verify(out: Path) -> list[tuple[str, str]]:
    """Every local reference in the output must resolve inside the output.

    The tool checking its own result is the point: a scanner that misses a
    reference pattern produces a site that loads and then fails at runtime —
    which is exactly what `new Worker("./cluster-worker.js")` did before it was
    added to the patterns. A missing file is referenced but absent, so this
    catches any future gap without knowing what the gap is.
    """
    missing: list[tuple[str, str]] = []
    for path in sorted(out.rglob("*")):
        if path.suffix not in {".html", ".js", ".mjs"}:
            continue
        text = path.read_text(errors="replace")
        pattern = HTML_REFS if path.suffix in {".html", ".htm"} else JS_REFS
        for match in pattern.findall(text):
            ref = next(
                (g for g in (match if isinstance(match, tuple) else (match,)) if g), ""
            )
            if not ref or ref.startswith("data:"):
                continue
            ref = ref.split("?")[0].split("#")[0]
            if not (path.parent / ref).exists():
                missing.append((ref, str(path.relative_to(out))))
    return missing


# Hugging Face Space card. `sdk: static` serves `app_file` directly — there is
# no Python app, so `sdk_version`/`python_version`/`app_port` do not apply
# (they are Gradio and Docker fields). Colours must come from HF's fixed
# palette: red, yellow, green, blue, indigo, purple, pink, gray.
#
# Deliberately **no** `custom_headers`: COOP/COEP would create a cross-origin
# isolated context, which this page does not need (the wasm is single-threaded,
# no SharedArrayBuffer) and which would break the two cross-origin resources it
# does use — the GitHub avatar and the WebRTC mesh module from esm.sh.
README = """---
title: RLX Tab Cluster
emoji: 🦀
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
header: mini
pinned: false
license: apache-2.0
short_description: Data-parallel training across browser tabs, in Rust via wasm
tags:
  - rust
  - webassembly
  - webgpu
  - distributed-training
  - compiler
  - webrtc
---

# RLX tab cluster

Data-parallel training across browser tabs and devices, with no backend. Every
tab is a **rank**: it compiles an [RLX](https://github.com/MIT-RLX/rlx) backward
graph in a Web Worker, computes gradients over its own shard of the data, and
the cohort averages them each round.

- **Open another tab** — it joins over a `BroadcastChannel` and samples/s rises.
- **Open a room** — other devices join a **WebRTC mesh**; scan the QR code on a
  phone. Signalling goes to public relays, every gradient travels peer-to-peer.
- **Pick a backend** per rank: `webgpu`, `webgl` or `cpu`. A mixed cohort shows
  up in the roster.
- **Pick a wire format**: `f32`, `f16` or `f8` payloads, trading bandwidth for
  precision — f8 moves 3.1× less data.

The page ends with the same model written three ways — Rust, JavaScript and
Python — each building a graph, letting autodiff rewrite it into a backward
graph, and training it.

## Deploy

The directory is self-contained: copy it to any static host.

    # Hugging Face Space — `sdk: static` serves index.html
    git push  # with this folder at the Space root

    # GitHub Pages, Netlify, Vercel, S3 — upload as-is
    python3 -m http.server 8000     # or serve locally

`.nojekyll` is for GitHub Pages, which otherwise skips directories it treats as
private and would drop the wasm bundle. It is harmless elsewhere.

## Two things worth knowing

**Cross-device rooms need HTTPS.** `RTCPeerConnection` is only available in a
secure context, with `localhost` the sole exception — so a LAN IP over plain
http pairs tabs within one browser but never reaches a second device. Hugging
Face Spaces, Pages and Netlify all serve HTTPS already.

**Do not add COOP/COEP headers.** They would cross-origin isolate the page,
which it does not need (the wasm is single-threaded), and would break the two
cross-origin resources it does use: the GitHub avatar and the mesh module.

Built from `crates/bindings/rlx-web` in <https://github.com/MIT-RLX/rlx>, which
also carries the measurements and the failure modes behind these choices.
Licensed MIT OR Apache-2.0.
"""


def copy_qwen(web: Path, out: Path) -> int:
    """The chat page, its modules and workers, and its wasm package."""
    pkg = web / "pkg"
    if not (pkg / "rlx_models_web_bg.wasm").exists():
        raise SystemExit(f"\n✗ no wasm package in {pkg}: run `just web-models-build` in rlx-models")
    files = discover(web / QWEN_ENTRY, web)
    files |= {f.resolve() for f in pkg.glob("rlx_models_web*") if f.suffix in {".js", ".wasm"}}
    total = 0
    for src in sorted(files):
        dest = out / src.relative_to(web.resolve())
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        total += dest.stat().st_size
    print(f"· qwen/: {len(files)} files, {total / 1e6:.2f} MB")
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--out",
        default=str(WORKSPACE / "examples" / "tab-cluster"),
        help="output directory (default: ./examples/tab-cluster)",
    )
    ap.add_argument(
        "--all-pages",
        action="store_true",
        help="every page (demo, vision bench, tab cluster) under its own name — "
        "the GitHub Pages site",
    )
    ap.add_argument(
        "--qwen",
        metavar="DIR",
        help="rlx-models-web's web/ directory (built with `just web-models-build`): "
        "adds its Qwen3.5 chat at qwen/ and links it from every page",
    )
    ap.add_argument("--open", action="store_true", help="serve and open it")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()

    ensure_bundle()
    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    entries = PAGES if args.all_pages else [ENTRY]
    rename = {} if args.all_pages else {ENTRY: ENTRY_AS}
    files: set[Path] = set()
    for entry in entries:
        files |= discover(WEB / entry)
    # The wasm glue loads its `.wasm` through `new URL(…, import.meta.url)`,
    # which no static scan can see — copy the package wholesale.
    for extra in (WEB / "pkg").glob("*"):
        # `.d.ts` is a development artifact; a deployed page never reads it.
        if extra.suffix in {".js", ".wasm"} and not extra.name.endswith(".d.ts"):
            files.add(extra.resolve())

    total = 0
    for src in sorted(files):
        rel = src.relative_to(WEB.resolve())
        dest = out / rename.get(str(rel), rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        total += dest.stat().st_size

    if args.qwen:
        total += copy_qwen(Path(args.qwen).resolve(), out / QWEN_AT)
        for entry in entries:
            page = out / rename.get(entry, entry)
            text = page.read_text()
            # cluster.html has no nav (it is also the standalone Space page).
            if "</nav>" not in text:
                continue
            if text.count("    </nav>") != 1:
                raise SystemExit(f"\n✗ {page.name}: expected one <nav> to link the Qwen chat from")
            page.write_text(text.replace("    </nav>", QWEN_NAV, 1))

    (out / ".nojekyll").touch()
    if not args.all_pages:
        # The Space card describes the tab cluster alone.
        (out / "README.md").write_text(README)

    for entry in entries:
        page = out / rename.get(entry, entry)
        prologue_problem = check_ascii_prologue(page)
        if prologue_problem:
            raise SystemExit(f"\n✗ {page.name}: {prologue_problem}")

    missing = verify(out)
    if missing:
        print("\n✗ incomplete site — these references do not resolve:", file=sys.stderr)
        for ref, src in missing:
            print(f"    {src}: {ref}", file=sys.stderr)
        raise SystemExit(
            "The dependency scan missed something. Teach `JS_REFS`/`HTML_REFS` the\n"
            "pattern rather than copying the file by hand — a `new Worker(…)` that\n"
            "the scan did not know about is how this check came to exist."
        )

    print(f"\n✓ standalone site: {out}")
    print(f"  {len(files)} files, {total / 1e6:.2f} MB")
    for src in sorted(files):
        rel = src.relative_to(WEB.resolve())
        name = rename.get(str(rel), str(rel))
        print(f"    {name:34} {src.stat().st_size / 1024:8.1f} KB")

    if args.open:
        url = f"http://localhost:{args.port}/"
        print(f"\nServing {out} at {url}  (Ctrl-C to stop)")
        webbrowser.open(url)
        subprocess.run(
            [sys.executable, "-m", "http.server", str(args.port)], cwd=out, check=False
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
