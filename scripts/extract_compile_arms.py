#!/usr/bin/env python3
"""Extract each simple arm of compile_thunks_with_rng's `let t = match &node.op`
(compile-time, ~3800 lines, 133 arms) into a free fn
`compile_<op>(node, graph, arena, matmul_fold, rng_shared) -> Thunk`. Patterns
kept verbatim + re-bound via let-else; multi-pattern/guard/catch-all stay inline.
Same technique as extract_vjp_arms; only the fn/indent/context differ.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "crates/backends/rlx-cpu/src/thunk.rs")
SIG = ("node: &rlx_ir::Node, graph: &Graph, arena: &crate::arena::Arena, "
       "matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>, "
       "rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>, "
       "rng: rlx_ir::RngOptions")
CALL = "node, graph, arena, &matmul_fold, &rng_shared, rng"
RET = "Thunk"
MATCH_LINE = "        let t = match &node.op {"
MATCH_CLOSE = "        };"
ARM_RE = re.compile(r'^            [A-Za-z_]')      # 12-space arm starts


def snake(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def main():
    lines = open(SRC).read().split("\n")
    n = len(lines)
    vs = next(i for i, l in enumerate(lines) if l.startswith("pub fn compile_thunks_with_rng"))
    mo = next(i for i in range(vs, n) if lines[i] == MATCH_LINE)
    mc = next(i for i in range(mo + 1, n) if lines[i] == MATCH_CLOSE)
    fc = next(i for i in range(mc + 1, n) if lines[i] == "}")

    starts = [i for i in range(mo + 1, mc) if ARM_RE.match(lines[i])]

    def attach(idx):
        j = idx
        while j - 1 > mo and lines[j - 1].strip().startswith(("//", "#[")):
            j -= 1
        return j

    bounds = [(attach(s), s) for s in starts]
    used, fns, rebuilt, kept = {}, [], [], 0
    for k, (a, s) in enumerate(bounds):
        end = bounds[k + 1][0] if k + 1 < len(bounds) else mc
        arm = "\n".join(lines[s:end])
        pos = arm.find("=>")
        pat = arm[:pos].rstrip()
        body = arm[pos + 2:].strip().rstrip()
        if body.endswith(","):
            body = body[:-1].rstrip()
        m = re.search(r'Op::(\w+)', pat)
        # keep inline: loop-control / accumulator / un-passed maps
        if re.search(r'\b(continue|break|thunks|sgd_fold)\b', body):
            rebuilt += lines[a:end]
            kept += 1
            continue
        if lines[s].strip().startswith("_") or " if " in pat or "|" in pat or not m:
            rebuilt += lines[a:end]
            kept += 1
            continue
        base = "compile_" + snake(m.group(1))
        disc = re.search(r'Op::\w+\(\s*\w+::(\w+)', pat)
        if disc:
            base += "_" + snake(disc.group(1))
        used[base] = used.get(base, 0) + 1
        fname = base if used[base] == 1 else f"{base}_{used[base]}"
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
    # dispatcher keeps unused outer-arm bindings -> allow on the fn
    for i, l in enumerate(out):
        if l.startswith("pub fn compile_thunks_with_rng"):
            out.insert(i, "#[allow(unused_variables)]")
            break
    open(SRC, "w").write("\n".join(out))
    print(f"compile@{vs+1} match {mo+1}-{mc+1} fnclose@{fc+1}")
    print(f"extracted {len(fns)} arms; kept {kept} inline")


if __name__ == "__main__":
    main()
