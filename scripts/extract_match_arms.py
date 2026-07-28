#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Extract each named arm of the giant `match &node.op` in
`unfuse_fused_for_autodiff` into a free fn `unfuse_<op>(node, new_inputs, out)`,
replacing the arm with a call. Behavior-preserving: arm bodies moved verbatim;
the variant's fields are re-bound in the fn via `let-else`; the `_` default and
any guarded/multi-pattern arms stay inline. `new_inputs` is passed by value so
arms that index it (copy) and arms that move it both work unchanged.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "crates/core/rlx-fusion/src/unfuse.rs")
FN = "pub fn unfuse_fused_for_autodiff"
NODE_TY = "&rlx_ir::Node"


def snake(name):
    return "unfuse_" + re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def main():
    lines = open(SRC).read().split("\n")
    n = len(lines)
    fn_start = next(i for i, l in enumerate(lines) if l.startswith(FN))
    match_open = next(i for i in range(fn_start, n) if "= match &node.op {" in lines[i])
    match_close = next(i for i in range(match_open + 1, n) if lines[i] == "        };")
    fn_close = next(i for i in range(match_close + 1, n) if lines[i] == "}")

    starts = [i for i in range(match_open + 1, match_close)
              if re.match(r'^            (Op::|_)', lines[i])]

    def attach(idx):  # pull preceding 12-space comments/attrs with the arm
        j = idx
        while j - 1 > match_open and lines[j - 1].strip().startswith(("//", "#[")):
            j -= 1
        return j

    bounds = [(attach(s), s) for s in starts]

    fns, rebuilt_arms, kept = [], [], 0
    for k, (a, s) in enumerate(bounds):
        end = bounds[k + 1][0] if k + 1 < len(bounds) else match_close
        arm_body = "\n".join(lines[s:end])
        pos = arm_body.find("=>")
        pat = arm_body[:pos].rstrip()
        body = arm_body[pos + 2:].strip().rstrip()
        if body.endswith(","):
            body = body[:-1].rstrip()
        m = re.search(r'Op::(\w+)', pat)
        # keep inline: default, guarded, multi-pattern, or unparseable
        if lines[s].strip().startswith("_") or " if " in pat or "|" in pat or not m:
            rebuilt_arms += lines[a:end]
            kept += 1
            continue
        variant = m.group(1)
        fname = snake(variant)
        # `out` is now `&mut Graph` (was owned): reborrow explicit `&mut out` args.
        body = re.sub(r'&mut out(?![\w.])', '&mut *out', body)
        comments = [c.lstrip() for c in lines[a:s]]  # move arm comments onto the fn
        fn_text = "\n".join(comments) + ("\n" if comments else "") + (
            f"fn {fname}(node: {NODE_TY}, new_inputs: Vec<NodeId>, out: &mut Graph) -> NodeId {{\n"
            f"    let {pat.lstrip()} = &node.op else {{ unreachable!() }};\n"
            f"    {body}\n"
            f"}}")
        fns.append(fn_text)
        rebuilt_arms.append(
            f"            Op::{variant} {{ .. }} => {fname}(node, new_inputs, &mut out),")

    out_lines = lines[:match_open + 1] + rebuilt_arms + lines[match_close:fn_close + 1]
    for ft in fns:
        out_lines += ["", ft]
    out_lines += lines[fn_close + 1:]
    open(SRC, "w").write("\n".join(out_lines))

    print(f"fn@{fn_start+1} match {match_open+1}-{match_close+1} fnclose@{fn_close+1}")
    print(f"extracted {len(fns)} arms into free fns; kept {kept} inline")
    for ft in fns:
        print("  " + [l for l in ft.split("\n") if l.startswith("fn ")][0])


if __name__ == "__main__":
    main()
