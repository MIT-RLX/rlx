#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail when two workspace packages ship an example under the same name.

Cargo writes every package's examples into ONE `target/<profile>/examples/`
directory, keyed by the bare target name. Two packages naming an example the
same thing therefore overwrite each other's unsuffixed binary: whichever built
last wins, and a harness that runs `./target/release/examples/<name>` measures
that one — under both labels.

That is not hypothetical. `schedule_matmul_ab` shipped in both rlx-cuda and
rlx-wgpu (`example-name-collision-across-packages` in the evolution ledger);
the wgpu copy is now `wgpu_schedule_matmul_ab`, and the remedy this file
enforces is the same one: prefix the example with its backend.

`cargo metadata` is the source of truth rather than a walk of
`crates/*/*/examples/*.rs`, because several packages declare `[[example]]`
explicitly with a custom path and `crates/rlx/examples` is not three levels
deep. A filesystem walk would quietly miss both.

Usage:
  python3 scripts/check-example-names.py    # exit 1 on an unlisted collision
"""

from __future__ import annotations

import collections
import json
import subprocess
import sys

# Collisions that are allowed, each with the reason it cannot bite.
#
# An entry is a promise that the packages listed can never be built into the
# same `target/` directory. "Nobody runs it by the bare path today" is NOT a
# reason — that is a property of the harness, which changes; the platform split
# below is a property of the toolchains, which does not.
ALLOWED: dict[str, tuple[set[str], str]] = {
    "reference_perf": (
        {"rlx-cuda", "rlx-metal"},
        "rlx-corpus's reference register derives the anchor path per backend as "
        "`crates/backends/rlx-{backend}/examples/reference_perf.rs`, so the shared "
        "name is load-bearing (see reference.rs::a_backend_with_an_anchor_has_at_"
        "least_one_measured_row). Safe because Metal builds only on Apple and CUDA "
        "only off it — the two can never land in one target/ directory.",
    ),
}


def examples_by_name() -> dict[str, set[str]]:
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        capture_output=True,
        text=True,
        check=True,
    )
    names: dict[str, set[str]] = collections.defaultdict(set)
    for pkg in json.loads(out.stdout)["packages"]:
        for target in pkg["targets"]:
            if "example" in target["kind"]:
                names[target["name"]].add(pkg["name"])
    return names


def main() -> int:
    names = examples_by_name()
    collisions = {n: p for n, p in names.items() if len(p) > 1}

    failures: list[str] = []

    for name, packages in sorted(collisions.items()):
        allowed = ALLOWED.get(name)
        if allowed is None:
            failures.append(
                f"  {name}: {sorted(packages)}\n"
                f"      Both write target/<profile>/examples/{name}; the last build wins.\n"
                f"      Prefix one with its backend (see wgpu_schedule_matmul_ab), or add an\n"
                f"      ALLOWED entry in this script stating why they cannot collide."
            )
        elif packages != allowed[0]:
            failures.append(
                f"  {name}: {sorted(packages)}\n"
                f"      ALLOWED here covers {sorted(allowed[0])}. The set changed, so the\n"
                f"      stated reason no longer describes what is in the tree — recheck it."
            )

    # A stale exemption is its own failure: it reads as "reviewed and safe"
    # while covering nothing, and the next real collision under that name would
    # be waved through by it.
    for name, (packages, _) in sorted(ALLOWED.items()):
        if name not in collisions:
            failures.append(
                f"  {name}: listed in ALLOWED for {sorted(packages)} but no longer collides.\n"
                f"      Delete the entry — an exemption that covers nothing still excuses the\n"
                f"      next collision that takes this name."
            )

    total = sum(len(p) for p in names.values())
    print(f"{total} example(s) across the workspace, {len(names)} distinct name(s)")
    for name, (packages, reason) in sorted(ALLOWED.items()):
        if name in collisions:
            print(f"  ALLOWED {name}: {sorted(packages)} — {reason.splitlines()[0]}…")

    if failures:
        print("\nexample-name collisions:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1

    print("no unlisted example-name collisions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
