#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Scan a compiled CoreML `model.mil` for reshapes whose element counts disagree.

rlx-ir does not verify element counts on a fully-concrete reshape target, so a
mis-built graph reaches the backend intact: the CPU backend reinterprets the
buffer and the model silently returns garbage, while Metal/CoreML hand the same
reshape to a verifying compiler and `abort()` the process.

Every CoreML compile leaves its compiled model in
`$TMPDIR/rlx-coreml-cache/<hash>.mlmodelc`, so one glob sweeps a whole run:

    python3 scripts/mil-reshape-check.py "$TMPDIR"/rlx-coreml-cache/*.mlmodelc/model.mil

Exits non-zero if any mismatch is found.
"""
import re
import sys
from math import prod

DECL = re.compile(r"tensor<\w+, \[([0-9, ]*)\]> (\w+) = (\w+)\(")
XARG = re.compile(r"\bx = (\w+)\b")


def dims(s):
    return [int(d) for d in s.replace(" ", "").split(",") if d]


def scan(path):
    shapes, bad = {}, []
    for ln, line in enumerate(open(path), 1):
        m = DECL.search(line)
        if not m:
            continue
        d, name, op = dims(m.group(1)), m.group(2), m.group(3)
        shapes[name] = d
        if op != "reshape":
            continue
        x = XARG.search(line)
        if not x or x.group(1) not in shapes:
            continue
        src = shapes[x.group(1)]
        if prod(src) != prod(d):
            bad.append((ln, name, x.group(1), src, d))
    return bad


def main(argv):
    rc = 0
    for path in argv:
        bad = scan(path)
        if bad:
            rc = 1
            print(f"{path}")
            for ln, name, src_name, src, dst in bad:
                print(
                    f"  :{ln} {name} = reshape({src_name}) "
                    f"{src} ({prod(src)} elems) -> {dst} ({prod(dst)} elems)"
                )
    if rc == 0:
        print(f"no mismatched reshapes in {len(argv)} file(s)")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
