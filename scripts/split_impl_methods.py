#!/usr/bin/env python3
"""Behavior-preserving splitter for a single giant inherent `impl` block.

Carves the methods of one big `impl<..> TYPE<..> { ... }` into per-topic sibling
modules, each holding its own `impl<..> TYPE<..> { <methods> }`. Inherent methods
resolve via the type across modules, so `self.foo()` keeps working — provided
every method is `pub(crate)` (we bump them). No `use`/re-export needed for the
calls; submodules only need the type + helpers in scope (`use super::*` + the
file's import block, with helper-module paths made `super::`-relative).

mod.rs keeps: everything before/after the impl, plus the impl block reduced to
its unmapped (core/dispatch/helper) methods. Configure CONFIG, run once.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── CONFIG ──────────────────────────────────────────────────────────────
SRC = os.path.join(ROOT, "crates/backends/rlx-coreml/src/mil/mod.rs")
OUTDIR = os.path.join(ROOT, "crates/backends/rlx-coreml/src/mil")
IMPL_TYPE = "LowerCtx"                    # inherent impl to split
FAM_USE_REWRITE = {"use helpers::": "use super::helpers::"}   # sibling-mod paths
MAP = {
    # norm
    "lower_layer_norm": "norm", "lower_rms_norm": "norm",
    "lower_rms_norm_backward_input": "norm", "lower_rms_norm_backward_gamma": "norm",
    "lower_rms_norm_backward_beta": "norm", "lower_layer_norm_backward_input": "norm",
    "lower_layer_norm_backward_gamma": "norm", "lower_group_norm_backward_input": "norm",
    "lower_group_norm_backward_gamma": "norm", "lower_group_norm_backward_beta": "norm",
    "lower_batch_norm": "norm", "lower_group_norm": "norm", "lower_layer_norm2d": "norm",
    "normalize_chain": "norm", "affine_nchw": "norm",
    # attention
    "lower_attention": "attention", "bshd_to_bhsd": "attention", "fused_to_bhsd": "attention",
    "fused_to_bhsd_kv": "attention", "apply_score_mask": "attention", "attention_core": "attention",
    "lower_attention_backward": "attention", "attention_backward_core": "attention",
    # rope
    "lower_rope": "rope", "rope_insert_head_axis": "rope", "lower_axial_rope2d": "rope",
    # conv / pool
    "lower_conv": "conv_pool", "lower_conv2d_backward_weight": "conv_pool",
    "lower_pool": "conv_pool", "lower_max_pool2d_backward": "conv_pool",
    # loss / softmax
    "lower_softmax_cross_entropy_with_logits": "loss", "lower_softmax_cross_entropy_backward": "loss",
    # activation / reduce / matmul
    "lower_activation": "activation",
    "lower_topk": "reduce_index", "lower_argreduce": "reduce_index", "lower_reverse": "reduce_index",
    "lower_lora_matmul": "matmul", "lower_grouped_matmul": "matmul",
    # quant / dequant
    "quant_bytes": "quant", "bake_ondevice_weight": "quant", "lower_dequant_matmul_ondevice": "quant",
    "lower_dequant_matmul": "quant", "lower_dequant_moe_weights": "quant",
    "lower_dequant_grouped_matmul_ondevice": "quant", "lower_dequant_grouped_matmul": "quant",
    "bake_affine": "quant", "lower_dequantize": "quant", "lower_quantize": "quant",
    # state-space models
    "lower_selective_scan": "ssm", "lower_gated_delta_net": "ssm",
    "gdn_vec": "ssm", "gdn_scalar": "ssm",
}
# ────────────────────────────────────────────────────────────────────────

METHOD_RE = re.compile(r'^    (?P<vis>pub(\([^)]*\))? )?(async )?(unsafe )?fn (?P<name>[a-z_][a-z0-9_]*)')
IMPL_RE = re.compile(r'^impl(<[^>]*>)?\s+' + IMPL_TYPE + r'\b')


def main():
    lines = open(SRC).read().split("\n")
    n = len(lines)
    impl_start = next(i for i, l in enumerate(lines) if IMPL_RE.match(l) and " for " not in l)
    impl_header = lines[impl_start]
    impl_end = next(i for i in range(impl_start + 1, n) if lines[i] == "}")
    prefix = lines[:impl_start]
    suffix = lines[impl_end + 1:]
    body = lines[impl_start + 1:impl_end]

    # method starts within the impl body (4-space fn), attaching docs/attrs
    starts = [i for i, l in enumerate(body) if METHOD_RE.match(l)]

    def attach(idx):
        j = idx
        while j > 0 and body[j - 1].strip().startswith(("#[", "///", "//!", "//")):
            j -= 1
        return j

    bounds = [(attach(s), s) for s in starts]
    body_prefix = "\n".join(body[:bounds[0][0]]) if bounds else "\n".join(body)

    def bump(block, sline_rel):
        bl = block.split("\n")
        m = METHOD_RE.match(bl[sline_rel])
        if not m.group("vis"):
            bl[sline_rel] = bl[sline_rel].replace("    fn ", "    pub(crate) fn ", 1)
        return "\n".join(bl)

    kept, fam = [], {}
    for k, (a, s) in enumerate(bounds):
        end = bounds[k + 1][0] if k + 1 < len(bounds) else len(body)
        blk = bump("\n".join(body[a:end]), s - a)
        name = METHOD_RE.match(body[s]).group("name")
        mod = MAP.get(name)
        (fam.setdefault(mod, []).append(blk) if mod else kept.append(blk))

    # family import header from the file's `use` lines (helpers -> super::helpers)
    uses = []
    for l in prefix:
        if l.startswith("use "):
            for a, b in FAM_USE_REWRITE.items():
                l = l.replace(a, b)
            uses.append(l)
    license_txt = "\n".join(prefix[:next((i for i, l in enumerate(prefix)
                            if l.startswith(("use ", "mod ", "const ", "pub ", "struct "))), 0)]).rstrip()
    fam_header = (license_txt + "\n\n#![allow(unused_imports)]\n\n"
                  + "\n".join(uses) + "\n\nuse super::*;\n")

    fams = sorted(fam)
    decls = "\n".join(f"mod {f};" for f in fams)
    mod_impl = impl_header + "\n" + (body_prefix + "\n\n" if body_prefix.strip() else "") \
        + "\n\n".join(kept) + "\n}"
    mod_rs = "\n".join(prefix).rstrip() + "\n\n" + decls + "\n\n" + mod_impl + "\n" \
        + ("\n" + "\n".join(suffix) if any(s.strip() for s in suffix) else "") + "\n"
    open(os.path.join(OUTDIR, "mod.rs"), "w").write(mod_rs)
    for f in fams:
        blocks = "\n\n".join(fam[f])
        open(os.path.join(OUTDIR, f + ".rs"), "w").write(
            fam_header + "\n" + impl_header + "\n" + blocks + "\n}\n")

    print(f"impl {IMPL_TYPE}: lines {impl_start+1}-{impl_end+1}, {len(bounds)} methods")
    print(f"moved {sum(len(v) for v in fam.values())} into {len(fams)} modules; kept {len(kept)} in mod.rs")
    for f in fams:
        print(f"  {f:14} ({len(fam[f])})")


if __name__ == "__main__":
    main()
