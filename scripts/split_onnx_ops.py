#!/usr/bin/env python3
"""One-shot module split for crates/io/rlx-onnx-import/src/lower/ops.rs.

Behavior-preserving: moves each `lower_<op>` handler fn into an op-family
submodule under `lower/ops/`, keeping all shared helpers, `LowerCtx`, and the
`lower_node` dispatcher in `lower/ops/mod.rs`. Handlers reach shared items via
`use super::*` (child modules can see the parent's private items). Moved fns are
bumped to `pub(super)` so the dispatcher can call them via `use <family>::*`.

Idempotent-ish: run once; it deletes the flat ops.rs and writes ops/.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "crates/io/rlx-onnx-import/src/lower/ops.rs")
OUTDIR = os.path.join(ROOT, "crates/io/rlx-onnx-import/src/lower/ops")

# lower_<op> handler fn -> family module. lower_node is the dispatcher (stays).
FAMILY = {
    "lower_binary": "binary", "lower_mod": "binary", "lower_is_nan": "binary",
    "lower_pow": "binary", "lower_clip": "binary", "lower_where": "binary",
    "lower_compare": "binary",
    "lower_matmul": "matmul", "lower_qmatmul": "matmul", "lower_gemm": "matmul",
    "lower_activation": "activation", "lower_activation_map": "activation",
    "lower_leaky_relu": "activation", "lower_act_copy": "activation",
    "lower_dropout": "activation", "lower_identity": "activation",
    "lower_cast": "cast_quant", "lower_dynamic_quant": "cast_quant",
    "lower_dynamic_quantize_lstm": "cast_quant",
    "lower_transpose": "shape_ops", "lower_reshape": "shape_ops",
    "lower_gather": "shape_ops", "lower_concat": "shape_ops",
    "lower_slice": "shape_ops", "lower_slice_stub": "shape_ops",
    "lower_shape_op": "shape_ops", "lower_expand": "shape_ops",
    "lower_softmax": "norm", "lower_layer_norm": "norm",
    "lower_instance_norm": "norm", "lower_batch_norm": "norm",
    "lower_conv": "conv_pool", "lower_conv_transpose_decomposed": "conv_pool",
    "lower_pool": "conv_pool",
    "lower_reduce": "reduce", "lower_cumsum": "reduce", "lower_cumprod": "reduce",
    "lower_topk": "reduce",
    "lower_scatter_nd": "gather_scatter", "lower_scatter_elements": "gather_scatter",
    "lower_gather_nd": "gather_scatter", "lower_one_hot": "gather_scatter",
    "lower_non_zero": "gather_scatter",
    "lower_if": "control", "lower_if_stub": "control",
    "lower_control_flow": "control", "lower_scan": "control",
    "lower_loop": "control", "lower_split_to_sequence": "control",
    "lower_sequence_empty": "control", "lower_concat_from_sequence": "control",
    "lower_range": "generators", "lower_constant_of_shape": "generators",
    "lower_random": "generators", "lower_random_like": "generators",
    "lower_resize": "generators", "lower_pad_as_concat": "generators",
    "lower_einsum": "generators",
}

ITEM_RE = re.compile(
    r'^(pub(\([^)]*\))? )?(async )?(unsafe )?(extern "[^"]*" )?'
    r'(fn|struct|enum|union|impl|trait|const|static|type|mod|macro_rules!?)\b')
FN_NAME_RE = re.compile(r'\bfn\s+([A-Za-z_][A-Za-z0-9_]*)')


def main():
    lines = open(SRC).read().split("\n")
    n = len(lines)

    # 1. find item boundary lines (column-0 items), attach preceding attrs/doc/comments
    starts = [i for i, l in enumerate(lines) if ITEM_RE.match(l)]
    first = starts[0]
    preamble = lines[:first]

    def attach_back(idx):
        j = idx
        while j > 0:
            p = lines[j - 1].strip()
            if p.startswith("#[") or p.startswith("///") or p.startswith("//!") \
               or p.startswith("//") or p.startswith("#!"):
                j -= 1
            else:
                break
        return j

    # spans: [attached_start, next_attached_start)
    bounds = [(attach_back(s), s) for s in starts]
    spans = []
    for k, (astart, sline) in enumerate(bounds):
        end = bounds[k + 1][0] if k + 1 < len(bounds) else n
        spans.append((astart, sline, end))

    # 2. classify each item -> ('keep') or family
    keep_chunks = []          # (astart, text) staying in mod.rs
    fam_chunks = {}           # family -> list of text blocks
    moved = {}
    for astart, sline, end in spans:
        block = "\n".join(lines[astart:end])
        sig = lines[sline]
        # only a top-level `fn <name>` (optionally pub) declaration is movable
        fam = None
        name = None
        if re.match(r'^(pub(\([^)]*\))? )?fn ', sig):
            m = FN_NAME_RE.search(sig)
            if m:
                name = m.group(1)
                fam = FAMILY.get(name)
        if fam:
            # bump visibility: `fn ` -> `pub(super) fn ` on the signature line
            rel = sline - astart
            blines = block.split("\n")
            if blines[rel].startswith("fn "):
                blines[rel] = "pub(super) " + blines[rel]
            fam_chunks.setdefault(fam, []).append("\n".join(blines))
            moved[name] = fam
        else:
            keep_chunks.append((astart, block))

    # 3. build headers
    preamble_text = "\n".join(preamble)
    license_lines, use_start = [], 0
    for i, l in enumerate(preamble):
        if l.startswith("use "):
            use_start = i
            break
        license_lines.append(l)
    license_text = "\n".join(preamble[:use_start]).rstrip()
    use_block = "\n".join(preamble[use_start:]).rstrip()
    fam_use_block = use_block.replace("use super::options::", "use crate::lower::options::")
    fam_header = (license_text + "\n\n#![allow(unused_imports)]\n\n"
                  + fam_use_block + "\n\nuse super::*;\n")

    # 4. write ops/ dir
    os.makedirs(OUTDIR, exist_ok=True)
    fams = sorted(fam_chunks)
    mod_decls = "\n".join(f"mod {f};" for f in fams) + "\n\n" \
        + "\n".join(f"use {f}::*;" for f in fams) + "\n"
    keep_body = "\n\n".join(t for _, t in keep_chunks)
    mod_rs = preamble_text.rstrip() + "\n\n" + mod_decls + "\n" + keep_body + "\n"
    open(os.path.join(OUTDIR, "mod.rs"), "w").write(mod_rs)
    for f in fams:
        body = "\n\n".join(fam_chunks[f])
        open(os.path.join(OUTDIR, f + ".rs"), "w").write(fam_header + "\n" + body + "\n")

    os.remove(SRC)  # remove flat ops.rs (ops/mod.rs supersedes it)

    print(f"moved {len(moved)} handlers into {len(fams)} families:")
    for f in fams:
        names = sorted(k for k, v in moved.items() if v == f)
        print(f"  {f:14} ({len(names)}): {', '.join(names)}")
    print(f"kept {len(keep_chunks)} items in mod.rs")


if __name__ == "__main__":
    main()
