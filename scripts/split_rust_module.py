#!/usr/bin/env python3
"""Behavior-preserving splitter: carve a big single-file Rust module into a
`mod.rs` + per-topic submodules, grouping items by a name->module map.

Reusable across Phase-2 targets. Items are matched by KEY:
  * `fn NAME`                    -> key = NAME
  * `struct|enum|union|type NAME`-> key = NAME
  * `impl ... for TYPE` / `impl TYPE` -> key = TYPE   (travels with its type)
Items whose key is in MAP move to MAP[key].rs; everything else stays in mod.rs.
Moved private `fn`/`struct` are bumped so the parent can still reach them
(pub(super) for fn, pub(crate) for types); already-`pub` items are untouched.
mod.rs re-exports each submodule (`pub use m::*` when REEXPORT_PUB else `use`),
so external `pub use <thismod>::{...}` keep resolving. Submodules get the file's
import block (with `super::` paths absolutized) + `use super::*` under
`#![allow(unused_imports)]`. Child modules can see the parent's private items,
so shared helpers may stay in mod.rs.

Configure via the CONFIG block, then run once (deletes the flat file).
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── CONFIG ──────────────────────────────────────────────────────────────
SRC = os.path.join(ROOT, "crates/numerics/rlx-linalg/src/lib.rs")
OUTDIR = os.path.join(ROOT, "crates/numerics/rlx-linalg/src")
REEXPORT_PUB = False                      # internal registration types: `use m::*` into lib.rs
SUPER_REWRITE = {}
# auto-group X{Ext,Cpu} structs + their trait impls by stem X -> op_<x>
AUTO_STEM = ("Ext", "Cpu")
MAP = {}
# ────────────────────────────────────────────────────────────────────────

ITEM_RE = re.compile(
    r'^(?P<vis>pub(\([^)]*\))? )?(async )?(unsafe )?(extern "[^"]*" )?'
    r'(?P<kw>fn|struct|enum|union|impl|trait|const|static|type|mod|macro_rules!?)\b')


def item_key(sig):
    """(kind, key-name) for a top-level item signature line."""
    m = ITEM_RE.match(sig)
    if not m:
        return None, None
    kw = m.group("kw")
    if kw == "impl":
        after = sig[sig.index("impl") + 4:]
        after = re.sub(r'^\s*<[^>]*>', '', after).strip()   # drop generics after impl
        after = after.split(" for ", 1)[1] if " for " in after else after
        nm = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', after)
        return "impl", (nm.group(1) if nm else None)
    if kw in ("fn", "struct", "enum", "union", "type", "trait", "const", "static", "mod"):
        nm = re.search(r'\b' + kw + r'\s+([A-Za-z_][A-Za-z0-9_]*)', sig)
        return kw, (nm.group(1) if nm else None)
    return kw, None


def main():
    lines = open(SRC).read().split("\n")
    n = len(lines)
    starts = [i for i, l in enumerate(lines) if ITEM_RE.match(l)]

    def attach_back(idx):
        j = idx
        while j > 0:
            p = lines[j - 1].strip()
            if p.startswith(("#[", "///", "//!", "//", "#!")):
                j -= 1
            else:
                break
        return j

    bounds = [(attach_back(s), s) for s in starts]
    preamble = lines[:bounds[0][0]]

    if AUTO_STEM:  # group X{Ext,Cpu} (+ their impls) by stem X -> op_<x>
        stems = {}
        for _, sl in bounds:
            kk = item_key(lines[sl])[1]
            if not kk:
                continue
            stem = kk
            for suf in AUTO_STEM:
                if kk.endswith(suf) and len(kk) > len(suf):
                    stem = kk[:-len(suf)]
                    break
            stems.setdefault(stem, set()).add(kk)
        for stem, kks in stems.items():
            if len(kks) >= 2:
                mod = "op_" + re.sub(r'(?<!^)(?=[A-Z])', '_', stem).lower()
                for kk in kks:
                    MAP[kk] = mod

    keep, fam_chunks = [], {}
    moved = []
    for k, (astart, sline) in enumerate(bounds):
        end = bounds[k + 1][0] if k + 1 < len(bounds) else n
        blk = lines[astart:end]
        kind, key = item_key(lines[sline])
        mod = MAP.get(key)
        if mod:
            rel = sline - astart
            vis = ITEM_RE.match(lines[sline]).group("vis")
            if not vis:  # bump private items so parent/siblings can reach them
                if kind == "fn":
                    blk[rel] = "pub(super) " + blk[rel]
                elif kind in ("struct", "enum", "union", "type", "const", "static"):
                    blk[rel] = "pub(crate) " + blk[rel]
            fam_chunks.setdefault(mod, []).append("\n".join(blk))
            moved.append((mod, kind, key))
        else:
            keep.append("\n".join(blk))

    # headers
    use_start = next((i for i, l in enumerate(preamble) if l.startswith("use ")), len(preamble))
    license_text = "\n".join(preamble[:use_start]).rstrip()
    use_block = "\n".join(preamble[use_start:]).rstrip()
    for a, b in SUPER_REWRITE.items():
        use_block = use_block.replace(a, b)
    fam_header = (license_text + "\n\n#![allow(unused_imports)]\n\n"
                  + use_block + "\n\nuse super::*;\n")

    os.makedirs(OUTDIR, exist_ok=True)
    fams = sorted(fam_chunks)
    kw = "pub use" if REEXPORT_PUB else "use"
    decls = "\n".join(f"mod {f};" for f in fams) + "\n\n" \
        + "\n".join(f"{kw} {f}::*;" for f in fams) + "\n"
    mod_rs = "\n".join(preamble).rstrip() + "\n\n" + decls + "\n" + "\n\n".join(keep) + "\n"
    # If OUTDIR is the source's own dir, keep SRC as the module root (lib.rs / an
    # existing mod.rs) and add siblings. Otherwise convert the file into a dir module.
    same_dir = os.path.abspath(OUTDIR) == os.path.abspath(os.path.dirname(SRC))
    root_path = SRC if same_dir else os.path.join(OUTDIR, "mod.rs")
    open(root_path, "w").write(mod_rs)
    for f in fams:
        open(os.path.join(OUTDIR, f + ".rs"), "w").write(
            fam_header + "\n" + "\n\n".join(fam_chunks[f]) + "\n")
    if not same_dir:
        os.remove(SRC)

    print(f"moved {len(moved)} items into {len(fams)} modules; kept {len(keep)} in mod.rs")
    for f in fams:
        items = [f"{k}:{key}" for m, k, key in moved if m == f]
        print(f"  {f:22} {', '.join(items)}")


if __name__ == "__main__":
    main()
