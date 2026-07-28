#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regroup the rlx workspace crates into a layered directory taxonomy.

Behavior-preserving: crate *names* are unchanged; only directory locations and
the relative `path = "../.."` deps in Cargo.tomls (plus the root `members`,
`.gitmodules`, `Justfile`, and `scripts/`) are rewritten. Fully reversible.

Layout (grouped):

    crates/
      rlx/                     # umbrella prelude — stays at top level
      core/     rlx-ir rlx-tensor rlx-flow rlx-fusion rlx-autodiff rlx-compile
                rlx-opt rlx-optim rlx-driver rlx-macros rlx-runtime rlx-collectives
      backends/ rlx-cpu rlx-metal rlx-mlx rlx-mlx-sys rlx-coreml rlx-wgpu
                rlx-cuda rlx-rocm rlx-gpu-kernels rlx-tpu rlx-vulkan rlx-oneapi
                rlx-qnn rlx-cortexm rlx-fpga rlx-cerebras rlx-webgl
      io/       rlx-gguf rlx-gguf-convert rlx-nemo rlx-onnx rlx-onnx-import
                rlx-onnx-conformance rlx-text
      numerics/ rlx-linalg rlx-sparse rlx-vq rlx-fdm rlx-rl rlx-bbo rlx-umap
      tooling/  rlx-bench
      bindings/ pyrlx rlx-web

Design rule: mutually-referencing crates share a group so their `../sibling`
build.rs refs don't change (cuda/rocm/gpu-kernels co-locate in backends/).

Usage:
    python3 scripts/regroup_crates.py            # dry-run forward (prints plan)
    python3 scripts/regroup_crates.py --apply    # execute forward
    python3 scripts/regroup_crates.py --revert           # dry-run revert
    python3 scripts/regroup_crates.py --revert --apply   # execute revert
"""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRATES = os.path.join(ROOT, "crates")

# crate dir name -> group (None => stays directly under crates/)
GROUPS = {
    # core spine (backend-agnostic: IR -> compile -> runtime)
    "rlx-ir": "core", "rlx-tensor": "core", "rlx-flow": "core", "rlx-fusion": "core",
    "rlx-autodiff": "core", "rlx-compile": "core", "rlx-opt": "core", "rlx-optim": "core",
    "rlx-driver": "core", "rlx-macros": "core", "rlx-runtime": "core", "rlx-collectives": "core",
    # backends
    "rlx-cpu": "backends", "rlx-metal": "backends", "rlx-mlx": "backends", "rlx-mlx-sys": "backends",
    "rlx-coreml": "backends", "rlx-wgpu": "backends", "rlx-cuda": "backends", "rlx-rocm": "backends",
    "rlx-gpu-kernels": "backends", "rlx-tpu": "backends", "rlx-vulkan": "backends", "rlx-oneapi": "backends",
    "rlx-qnn": "backends", "rlx-cortexm": "backends", "rlx-fpga": "backends", "rlx-cerebras": "backends",
    "rlx-webgl": "backends",
    # io: model formats, loaders, conversion, tokenizer
    "rlx-gguf": "io", "rlx-gguf-convert": "io", "rlx-nemo": "io", "rlx-onnx": "io",
    "rlx-onnx-import": "io", "rlx-onnx-conformance": "io", "rlx-text": "io",
    # numerics: downstream custom-op / domain crates
    "rlx-linalg": "numerics", "rlx-sparse": "numerics", "rlx-vq": "numerics", "rlx-fdm": "numerics",
    "rlx-rl": "numerics", "rlx-bbo": "numerics", "rlx-umap": "numerics",
    # tooling
    "rlx-bench": "tooling",
    # bindings
    "pyrlx": "bindings", "rlx-web": "bindings",
    # umbrella prelude stays at crates/rlx
    "rlx": None,
}

GROUP_DIRS = sorted({g for g in GROUPS.values() if g})

# rlx-mlx-sys carries a git submodule (vendor/mlx); move it with `git mv` so the
# gitlink + .gitmodules path are relocated correctly.
SUBMODULE_CRATE = "rlx-mlx-sys"

# Nested sub-crate members (dir relative to its parent crate).
NESTED = {"rlx-cortexm": ["trainer"]}

# Tooling files that hard-code `crates/<name>/...` paths.
TOOLING_FILES = ["Justfile", "rig.sh", ".gitmodules"]


def reldir(name, grouped):
    """Directory of a crate relative to repo root, in the given layout."""
    if not grouped:
        return f"crates/{name}"
    g = GROUPS[name]
    return f"crates/{name}" if g is None else f"crates/{g}/{name}"


def sh(args, dry):
    print("  $ " + " ".join(args))
    if not dry:
        subprocess.run(args, cwd=ROOT, check=True)


def _merge_newer(src, dst):
    """Move files from src into dst; on conflict keep whichever mtime is newer."""
    for dirpath, _, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        target_dir = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(target_dir, exist_ok=True)
        for fn in filenames:
            s = os.path.join(dirpath, fn)
            d = os.path.join(target_dir, fn)
            if not os.path.exists(d) or os.path.getmtime(s) > os.path.getmtime(d):
                shutil.move(s, d)


def move_dirs(grouped, dry):
    """Move each crate to its target layout. grouped=True forward, False revert."""
    if grouped:
        for g in GROUP_DIRS:
            d = os.path.join(CRATES, g)
            if not os.path.isdir(d) and not dry:
                os.makedirs(d, exist_ok=True)
        print(f"  (ensured group dirs: {', '.join(GROUP_DIRS)})")
    for name in GROUPS:
        src = os.path.join(ROOT, reldir(name, not grouped))
        dst = os.path.join(ROOT, reldir(name, grouped))
        if os.path.abspath(src) == os.path.abspath(dst):
            continue  # umbrella / no-op
        if not os.path.isdir(src):
            print(f"  ! skip {name}: source {src} missing")
            continue
        if name == SUBMODULE_CRATE:
            sh(["git", "mv", os.path.relpath(src, ROOT), os.path.relpath(dst, ROOT)], dry)
        elif os.path.isdir(dst):
            # dst already exists (e.g. an editor autosaved to the old path after a
            # prior run moved it): merge src into dst, newer file wins, then drop src.
            print(f"  merge {os.path.relpath(src, ROOT)} -> {os.path.relpath(dst, ROOT)} (dst exists)")
            if not dry:
                _merge_newer(src, dst)
                shutil.rmtree(src)
        else:
            print(f"  mv {os.path.relpath(src, ROOT)} -> {os.path.relpath(dst, ROOT)}")
            if not dry:
                os.rename(src, dst)
    if not grouped and not dry:
        for g in GROUP_DIRS:  # remove now-empty group dirs on revert
            d = os.path.join(CRATES, g)
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)


PATH_RE = re.compile(r'(?P<pre>path\s*=\s*")(?P<val>[^"]*)(?P<post>")')


def rewrite_manifests(grouped, dry):
    """Rewrite every `path = "..."` dep whose basename is a known crate."""
    changed = 0
    for dirpath, dirnames, filenames in os.walk(CRATES):
        dirnames[:] = [d for d in dirnames if d not in ("vendor", "target", ".git", "node_modules")]
        if "Cargo.toml" not in filenames:
            continue
        manifest = os.path.join(dirpath, "Cargo.toml")
        manifest_dir = os.path.relpath(dirpath, ROOT)
        with open(manifest) as f:
            text = f.read()

        def repl(m):
            target = os.path.basename(m.group("val").rstrip("/"))
            if target not in GROUPS:
                return m.group(0)
            newrel = os.path.relpath(reldir(target, grouped), manifest_dir)
            return m.group("pre") + newrel + m.group("post")

        new = PATH_RE.sub(repl, text)
        if new != text:
            changed += 1
            print(f"  rewrite deps: {manifest_dir}/Cargo.toml")
            if not dry:
                with open(manifest, "w") as f:
                    f.write(new)
    print(f"  ({changed} manifests updated)")


def rewrite_members(grouped, dry):
    """Rewrite the `members` paths in the root Cargo.toml."""
    root_manifest = os.path.join(ROOT, "Cargo.toml")
    with open(root_manifest) as f:
        text = f.read()
    orig = text
    # longest names first so `rlx` doesn't shadow `rlx-ir` (we match full quoted token anyway)
    for name in sorted(GROUPS, key=len, reverse=True):
        text = text.replace(f'"crates/{name}"', f'"{reldir(name, grouped)}"')
        for sub in NESTED.get(name, []):
            old = f'"crates/{name}/{sub}"'
            new = f'"{reldir(name, grouped)}/{sub}"'
            text = text.replace(old, new)
    if text != orig:
        print("  rewrite members: Cargo.toml")
        if not dry:
            with open(root_manifest, "w") as f:
                f.write(text)


def rewrite_tooling(grouped, dry):
    """Rewrite hard-coded crates/<name> refs in Justfile / scripts / .gitmodules."""
    names_by_len = sorted(GROUPS, key=len, reverse=True)
    targets = [os.path.join(ROOT, f) for f in TOOLING_FILES]
    scripts_dir = os.path.join(ROOT, "scripts")
    if os.path.isdir(scripts_dir):
        for fn in os.listdir(scripts_dir):
            if fn == os.path.basename(__file__):
                continue
            p = os.path.join(scripts_dir, fn)
            if os.path.isfile(p):
                targets.append(p)
    for p in targets:
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            continue  # skip binaries (e.g. scripts/rig)
        orig = text
        for name in names_by_len:
            # match `crates/<name>` only when followed by a path boundary
            pat = re.compile(r'crates/' + re.escape(name) + r'(?=[/"\'\s)]|$)')
            text = pat.sub(reldir(name, grouped), text)
        if text != orig:
            print(f"  rewrite tooling: {os.path.relpath(p, ROOT)}")
            if not dry:
                with open(p, "w") as f:
                    f.write(text)


def main():
    revert = "--revert" in sys.argv
    dry = "--apply" not in sys.argv
    grouped = not revert  # forward => grouped layout; revert => flat
    print(f"== regroup_crates: {'REVERT (flat)' if revert else 'FORWARD (grouped)'}"
          f" | {'DRY-RUN' if dry else 'APPLY'} ==")
    move_dirs(grouped, dry)
    rewrite_manifests(grouped, dry)
    rewrite_members(grouped, dry)
    rewrite_tooling(grouped, dry)
    print("== done ==" + ("  (dry-run — re-run with --apply)" if dry else ""))


if __name__ == "__main__":
    main()
