#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
# Cargo target runner for Apple *simulator* targets — iOS, tvOS, watchOS and
# visionOS.
#
# Cargo invokes this as `apple-sim-runner.sh <test-binary> [args...]`. The
# binary is a simulator Mach-O (libtest harness); `simctl spawn` runs it on a
# booted simulator and forwards stdio + the exit code, so `cargo test` works
# end-to-end.
#
# The simulator family is read out of the binary itself (`vtool -show-build`),
# not guessed: a tvOS test binary spawned on a booted iPhone fails with a
# mismatched-platform error that looks nothing like its cause. RLX_SIM_DEVICE
# (a name fragment or a UDID) narrows the choice *within* that family.
#
# Wire it up via:
#   CARGO_TARGET_AARCH64_APPLE_TVOS_SIM_RUNNER=scripts/apple-sim-runner.sh
set -euo pipefail

BIN="$1"; shift || true

# `$1 == "platform"` and not /platform/: vtool echoes the file name first, and
# a test binary called `apple_platform_sim-<hash>` matches a loose pattern —
# yielding an empty field 2 and a runner that claims the Mach-O is unreadable.
PLATFORM="$(xcrun vtool -show-build "$BIN" 2>/dev/null | awk '$1 == "platform" {print $2; exit}')"
if [ -z "${PLATFORM}" ]; then
  echo "apple-sim-runner: cannot read the Mach-O platform of ${BIN}" >&2
  exit 1
fi

# bash 3.2 — what /bin/bash still is on macOS — cannot parse a here-doc inside
# $(...), so the picker is read into a variable first and passed to `python3 -c`.
PICK_SIM=""
read -r -d '' PICK_SIM <<'PY' || true
import json, os, subprocess, sys

# Mach-O platform (what the test binary was built for) -> the simctl runtime
# identifier fragment for that family, plus the device to reach for when the
# caller named none. visionOS answers to both spellings: Apple's SDK and
# simctl runtime say "xrOS", the Mach-O load command says "VISIONOS".
FAMILY = {
    "IOSSIMULATOR":      (".SimRuntime.iOS-",     "iPhone"),
    "TVOSSIMULATOR":     (".SimRuntime.tvOS-",    "Apple TV"),
    "WATCHOSSIMULATOR":  (".SimRuntime.watchOS-", "Apple Watch"),
    "VISIONOSSIMULATOR": (".SimRuntime.xrOS-",    "Apple Vision"),
    "XROSSIMULATOR":     (".SimRuntime.xrOS-",    "Apple Vision"),
}

platform = os.environ["RLX_SIM_PLATFORM"]
if platform not in FAMILY:
    sys.exit(f"apple-sim-runner: {platform} is not a simulator platform — "
             "this runner only drives simulator targets")
runtime_key, default_name = FAMILY[platform]
want = os.environ.get("RLX_SIM_DEVICE") or default_name

devices = json.loads(subprocess.run(
    ["xcrun", "simctl", "list", "devices", "available", "-j"],
    capture_output=True, text=True, check=True).stdout)["devices"]

# Only devices of this family are candidates — a booted iPhone cannot run a
# tvOS binary, so "prefer whatever is booted" has to be family-scoped.
cands = [d for rt, ds in devices.items() if runtime_key in rt for d in ds]
if not cands:
    sys.exit(f"apple-sim-runner: no {platform} simulator runtime installed "
             f"(looked for {runtime_key})")

matched = [d for d in cands if want in d["name"] or d["udid"] == want]
if not matched:
    names = ", ".join(sorted({d["name"] for d in cands}))
    sys.exit(f"apple-sim-runner: no {platform} simulator matching '{want}'. "
             f"Available: {names}")

# A booted match needs no boot; otherwise take the first and let the caller boot it.
booted = next((d for d in matched if d.get("state") == "Booted"), None)
print((booted or matched[0])["udid"])
PY

udid="$(RLX_SIM_PLATFORM="$PLATFORM" RLX_SIM_DEVICE="${RLX_SIM_DEVICE:-}" \
        /usr/bin/python3 -c "$PICK_SIM")"

if ! xcrun simctl list devices booted | grep -q "${udid}"; then
  echo "apple-sim-runner: booting ${PLATFORM} simulator ${udid}" >&2
  xcrun simctl boot "${udid}" 2>/dev/null || true
  # `simctl spawn` on a device still coming up fails; wait for the data layer.
  xcrun simctl bootstatus "${udid}" -b >/dev/null 2>&1 || true
fi

# Run the test binary inside the simulator; -s forwards stdout/stderr.
exec xcrun simctl spawn -s "${udid}" "${BIN}" "$@"
