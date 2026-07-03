#!/usr/bin/env python3
"""Extract self-contained arms of execute_thunks's `match thunk` (the per-forward
HOT LOOP) into `#[inline(always)] fn exec_<t>(t: &Thunk[, base])`, replacing each
arm with a call. `#[inline(always)]` => the compiler folds them back into the
loop, so codegen (and perf) is unchanged — VERIFIED by the benchmark gate.

Clean, zero-warning:
- outer patterns simplified to `{ .. }` / `(..)` (no bindings => no unused-var
  warnings, no #[allow]); the fn re-binds the fields via `let-else` from `t`.
- `base` param omitted when an arm doesn't use it (no unused-param warnings).
Arms that touch the pre-alloc scratch buffers / thresholds / loop-control / other
per-iter locals STAY INLINE (they can't be moved without dragging that context).
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "crates/backends/rlx-cpu/src/thunk.rs")
FN = "pub fn execute_thunks("
MATCH_LINE = "        match thunk {"
MATCH_CLOSE = "        }"
ARM_RE = re.compile(r'^            Thunk::')
# keep inline if the body needs anything beyond (t, base):
TRIG = re.compile(r'\b(zero_bias|sdpa_scores|fl_qkv|fl_attn|fl_res|fl_normed|fl_ffn'
                  r'|fl_sc|mask_thr|mask_neg|score_thr|trace_done|continue|break'
                  r'|thunks|prof_prev|profile|schedule|arena_buf)\b')


def snake(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def main():
    lines = open(SRC).read().split("\n")
    n = len(lines)
    vs = next(i for i, l in enumerate(lines) if l.startswith(FN))
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
        m = re.search(r'Thunk::(\w+)', pat)
        # keep inline: multi-pattern (`A | B`) and guarded arms can't be
        # simplified to `{ .. }` without dropping alternatives.
        if not m or TRIG.search(body) or "|" in pat or " if " in pat:
            rebuilt += lines[a:end]
            kept += 1
            continue
        variant = m.group(1)
        base_n = "exec_" + snake(variant)
        used[base_n] = used.get(base_n, 0) + 1
        fname = base_n if used[base_n] == 1 else f"{base_n}_{used[base_n]}"
        uses_base = bool(re.search(r'\bbase\b', body))
        sig = "t: &Thunk, base: *mut u8" if uses_base else "t: &Thunk"
        call = "thunk, base" if uses_base else "thunk"
        if "{" in pat:
            outer = f"            Thunk::{variant} {{ .. }}"
        elif "(" in pat:
            outer = f"            Thunk::{variant}(..)"
        else:
            outer = f"            Thunk::{variant}"
        comments = [c.lstrip() for c in lines[a:s]]
        fn_text = "\n".join(comments) + ("\n" if comments else "") + (
            f"#[inline(always)]\nfn {fname}({sig}) {{\n"
            f"    let {pat.lstrip()} = t else {{ unreachable!() }};\n"
            f"    {body}\n}}")
        fns.append(fn_text)
        rebuilt.append(f"{outer} => {fname}({call}),")

    out = lines[:mo + 1] + rebuilt + lines[mc:fc + 1]
    for ft in fns:
        out += ["", ft]
    out += lines[fc + 1:]
    open(SRC, "w").write("\n".join(out))
    print(f"execute@{vs+1} match {mo+1}-{mc+1} fnclose@{fc+1}")
    print(f"extracted {len(fns)}; kept {kept} inline")


if __name__ == "__main__":
    main()
