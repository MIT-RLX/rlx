#!/usr/bin/env python3
# RLX — scaffold helper for adding a new Op.
# Prints the checklist from AGENTS.md and optional stub paths.
# Does not invent IR definitions — use --write only for empty reminder files.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKLIST = """\
# New op: {name}

## Checklist (from AGENTS.md)

1. `rlx-ir` — define `Op`, inference, verifier
2. Backends — thunk/executor, cost, and that crate's `SUPPORTED_OPS`
   (`crates/backends/rlx-*/src/supported_ops.rs`; Vulkan/OneAPI in `backend.rs`).
   Runtime wiring: `rlx-runtime/src/backend/<device>_backend.rs`.
3. `rlx-fusion` / `rlx-compile` if fusion/legalization applies
4. Parity test (`rlx-runtime/tests/` or downstream)
5. `just gen-op-coverage` to refresh `docs/op-coverage.md`
6. Host-staged GPU ops: implement in `rlx-gpu-host`, thin forwarder via `DeviceArena`

## Suggested touch points

- crates/core/rlx-ir/src/op.rs (or op module)
- crates/core/rlx-ir/src/infer.rs / verify.rs
- crates/backends/rlx-cpu/src/supported_ops.rs + executor/thunk
- crates/backends/rlx-{{metal,cuda,rocm,wgpu,mlx}}/src/supported_ops.rs (+ kernels)
- crates/backends/rlx-gpu-host/src/  (if host-fallback)
- crates/core/rlx-runtime/tests/
- docs/op-coverage.md (via `just gen-op-coverage`)

Probe: `RLX_DISPATCH_REPORT=1`
"""


def main() -> int:
    p = argparse.ArgumentParser(description="RLX new-op checklist / stub reminder")
    p.add_argument("name", help="Op variant name, e.g. FooBar")
    p.add_argument(
        "--write",
        action="store_true",
        help="Write docs/new-op-{name}.md checklist (does not modify Rust sources)",
    )
    args = p.parse_args()
    name = args.name.strip()
    if not name or not name.isidentifier():
        print(f"error: need a Rust identifier Op name, got {name!r}", file=sys.stderr)
        return 1
    text = CHECKLIST.format(name=name)
    print(text)
    if args.write:
        out = ROOT / "docs" / f"new-op-{name}.md"
        if out.exists():
            print(f"error: {out} already exists", file=sys.stderr)
            return 1
        out.write_text(text)
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
