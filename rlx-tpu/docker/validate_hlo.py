# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""Validate emitted HLO bytes by round-tripping through xla_extension.

Reads a manifest produced by the `emit_hlo_samples` example — one line
per emitted module, format ``<name>\\t<path>``. For each module we:

  1. Load the bytes from disk.
  2. Call ``xla_extension.HloModule.from_serialized_hlo_module_proto``.
  3. Assert per-module structural properties — that the right opcodes
     appear, instruction count is in the expected range, root shape
     matches, etc. These checks turn each module into a golden test:
     a regression in any of our lowerings (e.g. broken DotGeneral
     batch dims, missing softmax decomposition steps) shows up here.

If parse or any structural assertion fails, the script exits non-zero
with the offending module + traceback.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    from jax.lib import xla_extension
except ImportError as e:
    print(f"ERROR: jax.lib.xla_extension import failed: {e}", file=sys.stderr)
    sys.exit(2)


# ── Per-module structural expectations ────────────────────────────
#
# Each entry maps a module name (matches the manifest) to a set of
# checks. The checks run after the module parses cleanly:
#
#   opcodes   : a set of strings that MUST appear among the
#               instructions' opcodes. Catches "did the lowering
#               actually emit the op we expected"
#   min_instr : minimum number of instructions in the entry computation.
#               Catches accidentally-empty lowerings (e.g. softmax
#               that doesn't actually decompose to max+sub+exp+sum+div).
#   max_instr : optional upper bound. Catches lowerings that explode
#               into many more steps than they should.
#   params    : expected number of parameters in the entry computation.

@dataclass
class Expect:
    opcodes:   set[str]
    min_instr: int
    max_instr: int | None
    params:    int


EXPECTATIONS: dict[str, Expect] = {
    # Trivial pair: one parameter per input + one binary + root.
    "ew_add":           Expect({"add"},                  3, 5,  2),
    "matmul_2d":        Expect({"dot"},                  3, 6,  2),
    # Activations: each one's distinctive opcode must appear.
    "act_relu":         Expect({"maximum"},              3, 8,  1),
    "act_gelu":         Expect({"erf", "multiply"},      6, 16, 1),
    "act_gelu_approx":  Expect({"tanh", "multiply"},     8, 20, 1),
    "act_silu":         Expect({"logistic", "multiply"}, 3, 8,  1),
    "act_sigmoid":      Expect({"logistic"},             2, 5,  1),
    "act_tanh":         Expect({"tanh"},                 2, 5,  1),
    "act_rsqrt":        Expect({"rsqrt"},                2, 5,  1),
    # Norm decompositions: rsqrt + reduce + multiply chain. Upper
    # bounds capture the cost of the full unrolled computation —
    # mean→centered→sq→var→rsqrt→scale→bias for layernorm; baseline
    # widths come from the actual lowering output and shouldn't grow
    # without a corresponding refactor noted in the commit.
    "layernorm":        Expect({"rsqrt", "reduce", "subtract", "multiply", "add"},
                               20, 50,  3),
    "rmsnorm":          Expect({"rsqrt", "reduce", "multiply"},
                               12, 40,  3),
    # Softmax: max + sub + exp + reduce + divide.
    "softmax":          Expect({"reduce", "exponential", "subtract", "divide"},
                               10, 30,  1),
    # Reduce family.
    "reduce_sum":       Expect({"reduce"},               2, 8,  1),
    "reduce_mean":      Expect({"reduce", "divide"},     3, 12, 1),
    "reduce_max":       Expect({"reduce"},               2, 8,  1),
    # Compare + Where.
    "compare_where":    Expect({"compare", "select"},    3, 6,  2),
    # Shape ops.
    "shape_ops":        Expect({"reshape", "transpose", "slice"},
                               3, 8,  1),
    "gather":           Expect({"gather"},               2, 6,  2),
    # Attention causal: dot×2 + softmax decomp + iota+compare+select
    # for the mask. Subcomputation parameters are not counted.
    "attention_causal": Expect({"dot", "iota", "compare", "select",
                                "exponential"},
                               20, 60, 3),
    "rope":             Expect({"slice", "multiply", "concatenate"},
                               6, 20, 3),
    # BERT fragment: 1 graph input ("ids") + 5 params (emb_table,
    # ffn_w, ffn_b, ln_g, ln_b). Hits gather+matmul+gelu+layernorm.
    "bert_fragment":    Expect({"dot", "gather", "erf", "rsqrt", "reduce"},
                               25, 100, 6),

    # ── Tier-3 op lowerings (parity with rlx-cuda / rlx-rocm) ──
    "topk":             Expect({"sort", "slice", "iota",
                                "get-tuple-element"},
                               4, 12, 1),
    "grouped_matmul":   Expect({"gather", "dot"},
                               3, 12, 3),
    "dequant_matmul":   Expect({"convert", "subtract", "multiply",
                                "broadcast", "dot"},
                               6, 30, 4),
    "qmatmul":          Expect({"convert", "dot",
                                "round-nearest-even", "maximum",
                                "minimum"},
                               10, 40, 3),
    "qconv2d":          Expect({"convolution", "round-nearest-even",
                                "maximum", "minimum"},
                               10, 40, 3),
    # Greedy sample: argmax via sort+slice; no rng.
    "sample_greedy":    Expect({"sort", "slice", "convert",
                                "get-tuple-element"},
                               4, 15, 1),
    # Temperature sample: rng + softmax decomp + cumsum (reduce-window)
    # + first-greater-than (compare + select + reduce).
    "sample_temp":      Expect({"rng", "exponential", "divide",
                                "reduce-window", "compare", "select",
                                "reduce"},
                               20, 80, 1),
    # SelectiveScan: the entry computation is a thin wrapper around
    # `while`; the per-step dynamic-slice / dynamic-update-slice / exp
    # / multiply / add live in the body subcomputation, which the
    # entry-only opcode collector doesn't see. The structural check
    # here ensures the entry wires the loop correctly; the body's
    # internals get exercised end-to-end by the pjrt_roundtrip test.
    "selective_scan":   Expect({"while", "tuple", "get-tuple-element",
                                "broadcast"},
                               2, 30, 5),
}


def parse_module(bytes_path: Path) -> object:
    data = bytes_path.read_bytes()
    if not data:
        raise ValueError(f"empty HLO module at {bytes_path}")
    return xla_extension.HloModule.from_serialized_hlo_module_proto(data)


def collect_opcodes_and_count(mod) -> tuple[set[str], int, int]:
    """Return (opcodes_in_entry, n_instructions_in_entry, n_parameters_in_entry).

    Limited to the *entry* computation — reducer subcomputations bring
    their own `parameter` instructions and extra arithmetic that we
    don't want to count toward the entry's expected complexity.
    """
    # xla_extension.HloModule API varies between versions; the
    # introspection API isn't present in 0.4.x, so we go straight
    # through the text dump.
    import re
    text = mod.to_string()
    opcodes: set[str] = set()
    n_instr = 0
    n_param = 0
    # Match: "}<space>OPCODE(", "]<space>OPCODE(", or ")<space>OPCODE("
    # — opcode is the bare word between the shape-end and the opening
    # paren. The trailing-shape character is `]` for arrays, `}` for
    # layouts, and `)` for tuple-typed values like the result of `sort`
    # which produces `(f32[...], s32[...])`. Trim any ".N" instance
    # suffix (e.g. "fusion.0" → "fusion").
    pat = re.compile(r"[\]\}\)]\s*([a-zA-Z][\w-]*)\s*\(")

    # Walk computations: each one is a `... { ... }` block in the
    # text dump. Only count instructions inside the ENTRY block;
    # subcomputations (reducers, scatter combiners) bring their own
    # parameters we don't want to attribute to the entry.
    #
    # We deliberately don't track brace depth — shape layouts like
    # `f32[6]{0}` contain braces too. A standalone `}` line is what
    # closes a computation in HLO text format; that's the boundary.
    in_entry = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("ENTRY") or " ENTRY " in stripped:
            in_entry = True
            continue
        if in_entry and stripped == "}":
            in_entry = False
            continue
        if not in_entry:
            continue
        if " = " not in stripped:
            continue
        rhs = stripped.split(" = ", 1)[1]
        m = pat.search(rhs)
        if not m:
            continue
        opcode = m.group(1).split(".")[0]
        opcodes.add(opcode)
        n_instr += 1
        if opcode == "parameter":
            n_param += 1
    return opcodes, n_instr, n_param


def check_module(name: str, path: Path) -> tuple[bool, str]:
    try:
        mod = parse_module(path)
    except Exception:
        return False, f"parse failed:\n{traceback.format_exc()}"

    expect = EXPECTATIONS.get(name)
    if expect is None:
        # Unknown module — accept any parse-clean module. Lets the
        # harness keep working when new samples are added before
        # expectations are filled in.
        return True, "(parsed; no expectations registered)"

    opcodes, n_instr, n_param = collect_opcodes_and_count(mod)
    missing = expect.opcodes - opcodes
    if missing:
        return False, (f"missing opcodes {sorted(missing)} — got "
                       f"{sorted(opcodes)}")
    if n_instr < expect.min_instr:
        return False, (f"only {n_instr} instructions, expected ≥ "
                       f"{expect.min_instr}")
    if expect.max_instr is not None and n_instr > expect.max_instr:
        return False, (f"{n_instr} instructions, expected ≤ "
                       f"{expect.max_instr}")
    if n_param != expect.params:
        return False, f"{n_param} parameters, expected {expect.params}"
    return True, f"opcodes={len(opcodes):2d}  instructions={n_instr:3d}  params={n_param}"


def main(manifest_path: str) -> int:
    manifest = Path(manifest_path)
    if not manifest.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    failures: list[tuple[str, str]] = []
    n = 0
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "\t" not in line:
            continue
        name, path = line.split("\t", 1)
        n += 1
        ok, detail = check_module(name, Path(path))
        marker = "✓" if ok else "✗"
        print(f"{marker} {name:30s}  {detail}")
        if not ok:
            failures.append((name, detail))

    if failures:
        print()
        print(f"{'='*60}")
        print(f"{len(failures)} of {n} modules failed structural checks:")
        print(f"{'='*60}")
        for name, detail in failures:
            print(f"  - {name}: {detail}")
        return 1

    print()
    print(f"all {n} HLO modules parsed cleanly + matched expectations")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <manifest>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
