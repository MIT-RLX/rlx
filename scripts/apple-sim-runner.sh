#!/usr/bin/env bash
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
# Cargo target runner for Apple *simulator* targets (iOS / tvOS / visionOS sim).
#
# Cargo invokes this as `apple-sim-runner.sh <test-binary> [args...]`. The
# binary is a simulator Mach-O (libtest harness); `simctl spawn` runs it on a
# booted simulator and forwards stdio + the exit code, so `cargo test` works
# end-to-end.
#
# Pick the simulator with RLX_SIM_DEVICE (a name or UDID); defaults to a booted
# one, else boots the first available iPhone. Wire it up via:
#   CARGO_TARGET_AARCH64_APPLE_IOS_SIM_RUNNER=scripts/apple-sim-runner.sh
set -euo pipefail

BIN="$1"; shift || true

# Find (or boot) a simulator. Prefer an already-booted device.
udid="$(xcrun simctl list devices booted -j 2>/dev/null \
  | /usr/bin/python3 -c 'import json,sys; d=json.load(sys.stdin)["devices"]; print(next((x["udid"] for v in d.values() for x in v if x.get("state")=="Booted"), ""))' 2>/dev/null || true)"

if [ -z "${udid}" ]; then
  want="${RLX_SIM_DEVICE:-iPhone}"
  udid="$(xcrun simctl list devices available -j \
    | /usr/bin/python3 -c "import json,sys;
d=json.load(sys.stdin)['devices'];
cands=[x for v in d.values() for x in v if '${want}' in x['name'] or x['udid']=='${want}'];
print(cands[0]['udid'] if cands else '')")"
  if [ -z "${udid}" ]; then
    echo "apple-sim-runner: no simulator matching '${want}' found" >&2
    exit 1
  fi
  echo "apple-sim-runner: booting simulator ${udid}" >&2
  xcrun simctl boot "${udid}" 2>/dev/null || true
fi

# Run the test binary inside the simulator; -s forwards stdout/stderr.
exec xcrun simctl spawn -s "${udid}" "${BIN}" "$@"
