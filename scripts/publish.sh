#!/usr/bin/env bash
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
# scripts/publish.sh — workspace-wide publish driver for rlx.
#
# Walks the dep graph in tier order (leaves first), publishing one
# crate at a time with rate-limit-respecting delays AND active
# polling of the sparse index until each just-published version is
# resolvable. This combination is what keeps the script safe to
# leave running unattended:
#
#   * Hard rate-limit floor: a `MIN_INTERVAL` sleep (default 65 s,
#     above the documented "1 new crate per minute" + "1 version per
#     minute" crates.io throttle).
#   * Active readiness check: after every successful `cargo publish`,
#     poll `https://index.crates.io/<partition>/<crate>` until the
#     just-uploaded version appears. Until that happens, the next
#     crate's `cargo publish` would either 404 on dep resolution or
#     get stale metadata.
#   * Exponential backoff on HTTP 429 ("Too Many Requests") from
#     `cargo publish` itself — we never burst past the limit, but if
#     we ever did the script self-throttles and retries.
#
# Crates.io documented limits (as of 2026-05):
#   * New crates:        1 per minute   (10 per 10 min)
#   * New versions:      1 per minute   (30 per 10 min)
#   * Sparse-index serve: typically <30 s, occasionally up to ~5 min
#
# Default flow:
#
#   1. Pre-flight: cargo fmt --check, cargo clippy -- -D warnings,
#      cargo test --workspace --release. Aborts if any fails.
#   2. Confirm prompt (skip with --yes).
#   3. Per tier, publish each crate sequentially. After each crate:
#        a. wait for the sparse index to report the new version
#           (poll interval `POLL_INTERVAL`, hard cap `POLL_TIMEOUT`),
#        b. sleep `MIN_INTERVAL` (rate-limit safety floor),
#      before issuing the next publish. After the last crate of a
#      tier the loop additionally sleeps `BETWEEN_DELAY` to let
#      downstream crates' dep resolution catch up.
#   4. Crates marked `publish = false` (rlx-mlx, rlx-rocm, pyrlx,
#      rlx-cortexm-trainer) are skipped automatically by cargo —
#      this script doesn't list them.
#
# Usage:
#
#   scripts/publish.sh --dry-run                    # safe — no upload
#   scripts/publish.sh --list                       # print tier order, exit
#   scripts/publish.sh --no-gate                    # skip fmt/clippy/test
#   scripts/publish.sh --yes                        # skip confirm prompt
#   scripts/publish.sh --start-tier 3               # resume from tier 3
#   scripts/publish.sh --start-crate rlx-runtime    # resume from a crate
#   scripts/publish.sh --min-interval 90            # rate-limit floor (sec)
#   scripts/publish.sh --between-delay 120          # between-tier extra (sec)
#   scripts/publish.sh --poll-interval 10           # index-poll interval
#   scripts/publish.sh --poll-timeout 600           # index-poll cap (sec)
#   scripts/publish.sh --max-retries 3              # 429 retries per crate
#   scripts/publish.sh --no-verify                  # skip cargo's local rebuild
#   scripts/publish.sh --no-poll                    # disable index polling
#
# Resuming:
#
#   If a publish fails (network blip, transient crates.io error),
#   re-run with `--start-crate <name>` (or `--start-tier N`) to pick
#   up where you stopped. Already-published crates aren't revisited.
#
# Dry-run limitations:
#
#   `cargo publish --dry-run` does a full local rebuild + dep
#   resolution against crates.io. For NON-LEAF crates (tier ≥ 1)
#   that fails until their leaf deps actually exist on crates.io at
#   the new version. The script's --dry-run mode handles this by
#   recording the resolution-failure crates separately at the end —
#   metadata + packaging are still validated for them. The only
#   pre-publish step that can fully dry-run is the leaf tier (tier 0
#   in this workspace: rlx-ir, rlx-gguf, rlx-macros, rlx-cortexm).
#
# Environment:
#
#   CARGO_REGISTRY_TOKEN — must be set, or run `cargo login` first.
#                          Not validated here; cargo errors clearly.

set -euo pipefail

DRY_RUN=0
LIST_ONLY=0
NO_GATE=0
NO_VERIFY=0
NO_POLL=0
ASSUME_YES=0
MIN_INTERVAL=65          # rate-limit safety floor (sec)
BETWEEN_DELAY=90         # extra sleep at tier boundaries (sec)
POLL_INTERVAL=10         # index poll cadence (sec)
POLL_TIMEOUT=600         # max time we'll wait for the index (sec)
MAX_RETRIES=3            # `cargo publish` retries on 429
START_TIER=0
START_CRATE=""

# Tier definitions. Each array entry is a single tier; space-separated
# crate names within. Order within a tier doesn't matter for
# correctness (tier members don't depend on each other), but we
# publish in this order for determinism.
TIERS=(
    "rlx-ir rlx-gguf rlx-macros rlx-cortexm"
    "rlx-driver"
    "rlx-opt"
    "rlx-cpu rlx-metal rlx-wgpu rlx-cuda rlx-tpu rlx-fpga"
    "rlx-runtime"
    "rlx rlx-bench rlx-sparse rlx-linalg"
)

usage() {
    sed -n '2,80p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while (( $# > 0 )); do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --list)           LIST_ONLY=1; shift ;;
        --no-gate)        NO_GATE=1; shift ;;
        --no-verify)      NO_VERIFY=1; shift ;;
        --no-poll)        NO_POLL=1; shift ;;
        --yes|-y)         ASSUME_YES=1; shift ;;
        --min-interval)   MIN_INTERVAL="$2"; shift 2 ;;
        --between-delay)  BETWEEN_DELAY="$2"; shift 2 ;;
        --poll-interval)  POLL_INTERVAL="$2"; shift 2 ;;
        --poll-timeout)   POLL_TIMEOUT="$2"; shift 2 ;;
        --max-retries)    MAX_RETRIES="$2"; shift 2 ;;
        --start-tier)     START_TIER="$2"; shift 2 ;;
        --start-crate)    START_CRATE="$2"; shift 2 ;;
        --help|-h)        usage ;;
        # Legacy aliases for the old --within-delay flag — map onto
        # --min-interval so older invocations don't silently no-op.
        --within-delay)   MIN_INTERVAL="$2"; shift 2 ;;
        *)
            echo "unknown arg: $1" >&2
            echo "run with --help for usage" >&2
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")/.."

# Extract the workspace version once so the index-readiness check
# knows what to look for. Stops at the next `[…]` header so we don't
# accidentally read a version line from another table.
WORKSPACE_VERSION="$(awk '
    BEGIN              { in_block = 0 }
    /^\[workspace\.package\]/ { in_block = 1; next }
    /^\[/              { in_block = 0; next }
    in_block && $1 == "version" {
        # Line looks like:  version       = "0.2.0"
        match($0, /"[^"]+"/)
        if (RSTART > 0) {
            v = substr($0, RSTART + 1, RLENGTH - 2)
            print v
            exit
        }
    }
' Cargo.toml)"
if [[ -z "$WORKSPACE_VERSION" ]]; then
    echo "could not parse [workspace.package].version from Cargo.toml" >&2
    exit 1
fi

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
bold()   { printf "\033[1m%s\033[0m\n" "$*"; }

list_tiers() {
    bold "Publish order (workspace version $WORKSPACE_VERSION):"
    local i=0
    for tier in "${TIERS[@]}"; do
        echo "  tier $i:"
        for c in $tier; do
            echo "    - $c"
        done
        ((i++))
    done
    echo
    bold "Skipped (publish = false):"
    echo "  - rlx-mlx                  (build.rs reads ../vendor/mlx)"
    echo "  - rlx-rocm                 (include_str! to ../../rlx-cuda)"
    echo "  - pyrlx                    (PyPI via maturin)"
    echo "  - rlx-cortexm-trainer      (binary tool; nested workspace member"
    echo "                              under rlx-cortexm/trainer)"
    echo "  - rlx-cortexm-firmware     (no_std firmware binary; ships from git)"
}

if (( LIST_ONLY )); then
    list_tiers
    exit 0
fi

# ── Pre-flight gates ────────────────────────────────────────────
if (( ! NO_GATE )); then
    bold "[1/3] cargo fmt --check"
    cargo fmt --all -- --check

    bold "[2/3] cargo clippy --workspace --all-targets -- -D warnings"
    cargo clippy --workspace --all-targets -- -D warnings

    bold "[3/3] cargo test --workspace --release"
    cargo test --workspace --release
    green "Pre-flight gates passed."
fi

# ── Confirmation ────────────────────────────────────────────────
list_tiers
echo
if (( DRY_RUN )); then
    yellow "Mode: DRY RUN — no actual uploads, no rate-limit sleeps, no index polling."
else
    yellow "Mode: REAL PUBLISH — uploads to crates.io."
    yellow "Rate-limit floor:      ${MIN_INTERVAL}s between publishes."
    yellow "Between-tier extra:    ${BETWEEN_DELAY}s after each tier."
    if (( NO_POLL )); then
        yellow "Index polling:         DISABLED (--no-poll) — relying on fixed sleeps only."
    else
        yellow "Index polling:         every ${POLL_INTERVAL}s, hard cap ${POLL_TIMEOUT}s per crate."
    fi
    yellow "429 retries per crate: ${MAX_RETRIES} (exponential backoff)."
    if [[ -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
        # cargo will use ~/.cargo/credentials if no env var.
        yellow "CARGO_REGISTRY_TOKEN not set — relying on \`cargo login\` credentials."
    fi
fi
echo

if (( ! ASSUME_YES )); then
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        red "Aborted."
        exit 1
    fi
fi

# ── Sparse-index URL builder ────────────────────────────────────
#
# crates.io's sparse-index partitioning mirrors crates.io-index's
# git-side layout — encoded in the URL prefix so each shard stays
# small:
#
#   1 char:   /1/<crate>
#   2 chars:  /2/<crate>
#   3 chars:  /3/<first_char>/<crate>
#   4+ chars: /<first_2>/<next_2>/<crate>
#
# Names are always lower-case and hyphens are kept verbatim. See
# https://doc.rust-lang.org/cargo/reference/registry-index.html
sparse_index_path() {
    local name="$1"
    local n="${#name}"
    if   (( n == 1 )); then printf '1/%s\n'     "$name"
    elif (( n == 2 )); then printf '2/%s\n'     "$name"
    elif (( n == 3 )); then printf '3/%s/%s\n'  "${name:0:1}" "$name"
    else                    printf '%s/%s/%s\n' "${name:0:2}" "${name:2:2}" "$name"
    fi
}

# Returns 0 if the sparse index serves <version> for <crate>, 1 if
# the version isn't there yet, or 2 if the crate itself isn't in the
# index at all (pre-first-publish — perfectly fine, just means we
# should keep polling).
check_index() {
    local crate="$1"
    local version="$2"
    local url="https://index.crates.io/$(sparse_index_path "$crate")"
    local body
    # `curl -fsS` exits non-zero on 4xx/5xx; 404 means "crate doesn't
    # exist on the index yet" which is fine — keep polling.
    body="$(curl -fsS --max-time 10 "$url" 2>/dev/null || true)"
    if [[ -z "$body" ]]; then
        return 2
    fi
    # Index lines are NDJSON. Match `"vers":"<version>"` anywhere in
    # the file. `grep -F` keeps the dots from being regex-interpreted.
    if printf '%s' "$body" | grep -Fq "\"vers\":\"$version\""; then
        return 0
    fi
    return 1
}

# Block until check_index says we're good or the timeout trips.
wait_for_index() {
    local crate="$1"
    local version="$2"
    if (( NO_POLL )); then
        yellow "  (--no-poll set; skipping index readiness check)"
        return 0
    fi
    local elapsed=0
    yellow "  polling https://index.crates.io/.../$crate for $version (every ${POLL_INTERVAL}s, cap ${POLL_TIMEOUT}s)..."
    while (( elapsed < POLL_TIMEOUT )); do
        if check_index "$crate" "$version"; then
            green "  index ready after ${elapsed}s — $crate@$version resolvable."
            return 0
        fi
        sleep "$POLL_INTERVAL"
        elapsed=$(( elapsed + POLL_INTERVAL ))
    done
    yellow "  index didn't report $crate@$version within ${POLL_TIMEOUT}s — continuing anyway,"
    yellow "    next crate's publish may fail dep resolution if it depends on this one."
    return 0
}

# Sleep with a single live countdown line — easier to watch than
# silent waits during long publishes.
sleep_with_countdown() {
    local seconds="$1"
    local label="$2"
    local remaining=$seconds
    while (( remaining > 0 )); do
        printf "\r  %s — %3ds remaining " "$label" "$remaining"
        sleep 1
        remaining=$(( remaining - 1 ))
    done
    printf "\r  %s — done.                  \n" "$label"
}

# ── Walk tiers ──────────────────────────────────────────────────
DRY_RUN_PASS=()
DRY_RUN_FAIL=()

publish_one_attempt() {
    local crate="$1"
    local args=()
    args+=("--package" "$crate")
    if (( DRY_RUN )); then
        args+=("--dry-run")
    fi
    if (( NO_VERIFY )); then
        args+=("--no-verify")
    fi
    bold ">> cargo publish ${args[*]}"
    # Capture stderr so we can detect HTTP 429 ("Too Many Requests")
    # without losing the user-visible output.
    local tmp_err
    tmp_err="$(mktemp)"
    if cargo publish "${args[@]}" 2> >(tee "$tmp_err" >&2); then
        rm -f "$tmp_err"
        return 0
    fi
    local rc=$?
    if grep -qE "429|Too Many Requests|rate.?limit" "$tmp_err"; then
        rm -f "$tmp_err"
        return 42  # special: signals rate-limit, caller will back off
    fi
    rm -f "$tmp_err"
    return $rc
}

publish_one() {
    local crate="$1"
    local attempt=1
    local backoff=$MIN_INTERVAL
    while true; do
        if publish_one_attempt "$crate"; then
            if (( DRY_RUN )); then
                DRY_RUN_PASS+=("$crate")
            fi
            return 0
        fi
        local rc=$?
        if (( DRY_RUN )); then
            # Non-leaf crates can't fully dry-run; record + continue.
            DRY_RUN_FAIL+=("$crate")
            yellow "  (dry-run: dep resolution failed — expected for non-leaf crates pre-publish)"
            return 0
        fi
        if (( rc == 42 )); then
            if (( attempt > MAX_RETRIES )); then
                red "Publish failed for $crate after $MAX_RETRIES retries (still rate-limited)."
                red "Re-run with --start-crate $crate to resume later."
                exit 1
            fi
            yellow "  crates.io returned 429 / rate-limit (attempt $attempt/$MAX_RETRIES)."
            sleep_with_countdown "$backoff" "    backoff"
            backoff=$(( backoff * 2 ))
            attempt=$(( attempt + 1 ))
            continue
        fi
        red "Publish failed for $crate (cargo exit $rc)."
        red "Re-run with --start-crate $crate (or a later one) to resume."
        exit 1
    done
}

# Resolve start position.
skip_until_tier=$START_TIER
skip_until_crate="$START_CRATE"

for tier_idx in "${!TIERS[@]}"; do
    if (( tier_idx < skip_until_tier )); then
        continue
    fi
    tier="${TIERS[$tier_idx]}"
    bold "── Tier $tier_idx ────────────────────────────────────────"

    in_tier=0
    for crate in $tier; do
        # Skip past --start-crate if specified.
        if [[ -n "$skip_until_crate" ]]; then
            if [[ "$crate" != "$skip_until_crate" ]]; then
                yellow "  skip $crate (resume target: $skip_until_crate)"
                continue
            fi
            skip_until_crate=""
        fi

        # Pre-publish floor: between any two publishes (within or
        # across tiers) we wait at least MIN_INTERVAL. The first
        # crate of a tier already had its delay paid during the
        # previous tier's BETWEEN_DELAY (or is the very first
        # publish, which needs no upstream wait).
        if (( in_tier > 0 )) && (( ! DRY_RUN )); then
            sleep_with_countdown "$MIN_INTERVAL" "rate-limit floor"
        fi

        publish_one "$crate"

        # Post-publish: poll the index until the new version is
        # actually queryable, so the *next* crate's dep resolution
        # doesn't 404. Dry runs skip this entirely.
        if (( ! DRY_RUN )); then
            wait_for_index "$crate" "$WORKSPACE_VERSION"
        fi
        ((in_tier++))
    done

    # Between-tier extra delay (downstream tiers tend to resolve
    # multiple just-published crates at once, and the sparse index
    # can serve a name while not yet serving all its sibling
    # crates' latest versions).
    if (( ! DRY_RUN )) && (( tier_idx + 1 < ${#TIERS[@]} )) && (( in_tier > 0 )); then
        sleep_with_countdown "$BETWEEN_DELAY" "between-tier (let crates.io index settle)"
    fi
done

if (( DRY_RUN )); then
    echo
    bold "Dry-run summary:"
    green "  passed (full dry-run incl. dep resolution): ${#DRY_RUN_PASS[@]}"
    for c in "${DRY_RUN_PASS[@]}"; do echo "    ✓ $c"; done
    if (( ${#DRY_RUN_FAIL[@]} > 0 )); then
        yellow "  metadata + packaging ok, dep resolution fails (expected pre-publish): ${#DRY_RUN_FAIL[@]}"
        for c in "${DRY_RUN_FAIL[@]}"; do echo "    ⚠ $c"; done
        echo
        yellow "Non-leaf crates can only fully dry-run after their leaves are"
        yellow "actually published — that's why they show ⚠ above. Real publish"
        yellow "in tier order will succeed."
    fi
else
    green "All tiers published successfully."
fi
