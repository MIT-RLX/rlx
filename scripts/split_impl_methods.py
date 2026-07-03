#!/usr/bin/env python3
"""Behavior-preserving splitter for a single giant inherent `impl` block.

Carves the methods of one big `impl<..> TYPE<..>` into per-topic sibling modules
(each its own `impl<..> TYPE<..> { <methods> }`). Inherent methods resolve via
the type across modules, so `self.foo()` keeps working — every moved method is
bumped to `pub(crate)`. Submodules get the file's import block (`use super::` ->
`use crate::`) + `use super::*` under `#![allow(unused_imports)]`.

Auto-groups methods by name prefix (first `_`-token); a prefix with >= MIN_GROUP
methods becomes a module, singletons stay in mod.rs. Converts a plain file module
(foo.rs) into a directory module (foo/): writes foo/mod.rs + foo/<grp>.rs and
removes foo.rs. For an existing mod.rs/lib.rs it edits in place.

Usage: python3 split_impl_methods.py <src.rs> <ImplType> [min_group]
"""
import os
import re
import sys

MIN_GROUP = 2
METHOD_RE = re.compile(
    r'^    (?P<vis>pub(\([^)]*\))? )?(async )?(unsafe )?fn (?P<name>[a-z_][a-z0-9_]*)')


def main():
    SRC = os.path.abspath(sys.argv[1])
    IMPL_TYPE = sys.argv[2]
    min_group = int(sys.argv[3]) if len(sys.argv) > 3 else MIN_GROUP
    impl_re = re.compile(r'^impl(<[^>]*>)?\s+' + IMPL_TYPE + r'\b')

    srcdir = os.path.dirname(SRC)
    base = os.path.basename(SRC)
    if base in ("mod.rs", "lib.rs"):
        outdir, root_path, remove_src = srcdir, SRC, False
    else:
        outdir = os.path.join(srcdir, base[:-3])
        root_path, remove_src = os.path.join(outdir, "mod.rs"), True
    os.makedirs(outdir, exist_ok=True)

    lines = open(SRC).read().split("\n")
    n = len(lines)
    # pick the BIGGEST inherent impl block of the type (files may have several)
    cands = [i for i, l in enumerate(lines) if impl_re.match(l) and " for " not in l]
    def endof(st):
        return next(i for i in range(st + 1, n) if lines[i] == "}")
    impl_start = max(cands, key=lambda st: endof(st) - st)
    impl_header = lines[impl_start]
    impl_end = endof(impl_start)
    prefix, suffix, body = lines[:impl_start], lines[impl_end + 1:], lines[impl_start + 1:impl_end]

    starts = [i for i, l in enumerate(body) if METHOD_RE.match(l)]

    def attach(idx):
        j = idx
        while j > 0 and body[j - 1].strip().startswith(("#[", "///", "//!", "//")):
            j -= 1
        return j

    bounds = [(attach(s), s) for s in starts]
    body_prefix = "\n".join(body[:bounds[0][0]]) if bounds else "\n".join(body)
    names = [METHOD_RE.match(body[s]).group("name") for _, s in bounds]

    # auto-group by first `_`-token; prefixes with >= min_group methods -> module.
    # Sanitize module names that collide with Rust keywords.
    KW = {"const", "type", "ref", "match", "impl", "fn", "let", "mut", "move", "self",
          "super", "crate", "use", "mod", "pub", "struct", "enum", "trait", "where",
          "for", "in", "if", "else", "while", "loop", "break", "continue", "return",
          "as", "dyn", "async", "await", "unsafe", "extern", "static", "box", "yield"}
    pref = {}
    for nm in names:
        pref.setdefault(nm.split("_")[0], []).append(nm)
    amap = {nm: (p + "_ops" if p in KW else p)
            for p, nms in pref.items() if len(nms) >= min_group for nm in nms}

    def bump(block, rel):
        bl = block.split("\n")
        if not METHOD_RE.match(bl[rel]).group("vis"):
            bl[rel] = bl[rel].replace("    fn ", "    pub(crate) fn ", 1)
        return "\n".join(bl)

    kept, fam = [], {}
    for k, (a, s) in enumerate(bounds):
        end = bounds[k + 1][0] if k + 1 < len(bounds) else len(body)
        blk = bump("\n".join(body[a:end]), s - a)
        mod = amap.get(METHOD_RE.match(body[s]).group("name"))
        (fam.setdefault(mod, []).append(blk) if mod else kept.append(blk))

    # collect full (possibly multi-line) `use` statements up to their `;`
    uses, i = [], 0
    while i < len(prefix):
        if prefix[i].startswith("use "):
            stmt = [prefix[i]]
            while ";" not in stmt[-1] and i + 1 < len(prefix):
                i += 1
                stmt.append(prefix[i])
            uses.append("\n".join(stmt).replace("use super::", "use crate::"))
        i += 1
    lic_end = next((i for i, l in enumerate(prefix)
                    if l.startswith(("use ", "mod ", "const ", "pub ", "struct ", "#!["))), 0)
    fam_header = ("\n".join(prefix[:lic_end]).rstrip() + "\n\n#![allow(unused_imports)]\n\n"
                  + "\n".join(uses) + "\n\nuse super::*;\n")

    fams = sorted(fam)
    decls = "\n".join(f"mod {f};" for f in fams)
    mod_impl = (impl_header + "\n" + (body_prefix + "\n\n" if body_prefix.strip() else "")
                + "\n\n".join(kept) + "\n}")
    mod_rs = ("\n".join(prefix).rstrip() + "\n\n" + decls + "\n\n" + mod_impl + "\n"
              + ("\n" + "\n".join(suffix) if any(s.strip() for s in suffix) else "") + "\n")
    open(root_path, "w").write(mod_rs)
    for f in fams:
        open(os.path.join(outdir, f + ".rs"), "w").write(
            fam_header + "\n" + impl_header + "\n" + "\n\n".join(fam[f]) + "\n}\n")
    if remove_src and os.path.abspath(root_path) != SRC:
        os.remove(SRC)

    print(f"impl {IMPL_TYPE}: {impl_start+1}-{impl_end+1}, {len(bounds)} methods")
    print(f"moved {sum(len(v) for v in fam.values())} into {len(fams)} modules "
          f"({', '.join(f'{f}:{len(fam[f])}' for f in fams)}); kept {len(kept)} in mod.rs")


if __name__ == "__main__":
    main()
