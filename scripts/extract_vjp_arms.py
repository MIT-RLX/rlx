#!/usr/bin/env python3
"""Extract each simple arm of vjp's giant `match &node.op` into a free fn
`vjp_<op>(node, upstream, upstream_shape, fwd_map, bwd)`, replacing the arm with
a call. Behavior-preserving: patterns kept VERBATIM (handles tuple+discriminant
`Op::Binary(BinaryOp::Add)` and multi-arm-per-variant `Op::Reduce{..}` x4), the
arm's fields re-bound in the fn via `let-else`. Multi-pattern (`|`) and guarded
(`if`) arms and `_` stay inline. `upstream_shape` (pre-match local) is passed by
value. `#[allow(unused_variables)]` covers the now-unused outer bindings and any
context param an arm doesn't use.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "crates/core/rlx-autodiff/src/autodiff.rs")
SIG = ("node: &Node, upstream: NodeId, upstream_shape: Shape, "
       "fwd_map: &HashMap<NodeId, NodeId>, bwd: &mut Graph")
CALL = "node, upstream, upstream_shape, fwd_map, bwd"
RET = "Vec<(usize, NodeId)>"


def snake(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def main():
    lines = open(SRC).read().split("\n")
    n = len(lines)
    vs = next(i for i, l in enumerate(lines) if l.startswith("fn vjp("))
    mo = next(i for i in range(vs, n) if lines[i] == "    match &node.op {")
    mc = next(i for i in range(mo + 1, n) if lines[i] == "    }")
    fc = next(i for i in range(mc + 1, n) if lines[i] == "}")

    # arm starts: exactly 8-space then an identifier/underscore (Op::/_/named
    # catch-all like `other`); deeper lines and `        }` are continuations.
    starts = [i for i in range(mo + 1, mc) if re.match(r'^        [A-Za-z_]', lines[i])]

    def attach(idx):
        j = idx
        while j - 1 > mo and lines[j - 1].strip().startswith(("//", "#[")):
            j -= 1
        return j

    bounds = [(attach(s), s) for s in starts]
    used_names, fns, rebuilt, kept = {}, [], [], 0
    for k, (a, s) in enumerate(bounds):
        end = bounds[k + 1][0] if k + 1 < len(bounds) else mc
        arm = "\n".join(lines[s:end])
        pos = arm.find("=>")
        pat = arm[:pos].rstrip()
        body = arm[pos + 2:].strip().rstrip()
        if body.endswith(","):
            body = body[:-1].rstrip()
        m = re.search(r'Op::(\w+)', pat)
        if lines[s].strip().startswith("_") or " if " in pat or "|" in pat or not m:
            rebuilt += lines[a:end]
            kept += 1
            continue
        base = "vjp_" + snake(m.group(1))
        disc = re.search(r'Op::\w+\(\s*\w+::(\w+)', pat)   # tuple discriminant e.g. BinaryOp::Add
        if disc:
            base += "_" + snake(disc.group(1))
        used_names[base] = used_names.get(base, 0) + 1
        fname = base if used_names[base] == 1 else f"{base}_{used_names[base]}"
        body = re.sub(r'&mut bwd(?![\w.])', '&mut *bwd', body)
        comments = [c.lstrip() for c in lines[a:s]]
        fn_text = "\n".join(comments) + ("\n" if comments else "") + (
            f"#[allow(unused_variables)]\n"
            f"fn {fname}({SIG}) -> {RET} {{\n"
            f"    let {pat.lstrip()} = &node.op else {{ unreachable!() }};\n"
            f"    {body}\n"
            f"}}")
        fns.append(fn_text)
        rebuilt.append(arm[:pos] + f"=> {fname}({CALL}),")

    out = lines[:mo + 1] + rebuilt + lines[mc:fc + 1]
    for ft in fns:
        out += ["", ft]
    out += lines[fc + 1:]
    # dispatcher keeps unused outer-arm bindings -> allow on vjp
    for i, l in enumerate(out):
        if l == "fn vjp(":
            out.insert(i, "#[allow(unused_variables)]")
            break
    open(SRC, "w").write("\n".join(out))
    print(f"vjp@{vs+1} match {mo+1}-{mc+1} fnclose@{fc+1}")
    print(f"extracted {len(fns)} arms; kept {kept} inline (multi-pattern/default)")


if __name__ == "__main__":
    main()
