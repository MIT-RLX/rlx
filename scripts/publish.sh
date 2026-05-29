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
#   * Skip published: before each upload, resolve that crate's effective
#     version from its Cargo.toml (`version = "…"` or
#     `version.workspace = true` → workspace.package). Skip only when
#     *that* version is already on the sparse index — so a crate bumped
#     to 0.2.2 is still published even when 0.2.1 is on crates.io.
#     Use `--no-skip-published` to force an upload attempt anyway.
#   * Registry HTTP errors (429, 408, 5xx, timeouts): parse cargo output as
#     it finishes, sleep until crates.io's "try again after <GMT>" when
#     present (+ pad), else status-specific backoff — then retry the same
#     crate automatically until it uploads. --max-retries 0 (default) never
#     gives up on retryable codes.
#   * Missing registry deps: poll the sparse index for the dependency,
#     then retry (no manual --start-crate for index lag).
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
#   4. Crates marked `publish = false` (pyrlx, rlx-cortexm-trainer) are
#      skipped automatically by cargo — this script lists the rest.
#
# Usage:
#
#   scripts/publish.sh --dry-run                    # safe — no upload
#   scripts/publish.sh --list                       # tier order + per-crate versions
#   scripts/publish.sh --plan                       # crates needing publish only
#   scripts/publish.sh --no-gate                    # skip fmt/clippy/test
#   scripts/publish.sh --yes                        # skip confirm prompt
#   scripts/publish.sh --start-tier 3               # resume from tier 3
#   scripts/publish.sh --start-crate rlx-runtime    # resume from a crate
#   scripts/publish.sh --min-interval 90            # rate-limit floor (sec)
#   scripts/publish.sh --between-delay 120          # between-tier extra (sec)
#   scripts/publish.sh --poll-interval 10           # index-poll interval
#   scripts/publish.sh --poll-timeout 600           # index-poll cap (sec)
#   scripts/publish.sh --max-retries 5              # cap 429 backoff retries (0 = unlimited)
#   scripts/publish.sh --rate-limit-pad 15          # pad after server retry-after
#   scripts/publish.sh --no-skip-published          # upload even if version exists
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
#   pre-publish step that can fully dry-run is tier 0 (no RLX path
#   deps). Run `scripts/publish.sh --list` to print the full tier
#   order; `validate_publish_order` checks every `[dependencies]` and
#   `[dev-dependencies]` path dep against that order before publishing.
#
# Environment:
#
#   CARGO_REGISTRY_TOKEN — must be set, or run `cargo login` first.
#                          Not validated here; cargo errors clearly.

set -euo pipefail

DRY_RUN=0
LIST_ONLY=0
PLAN_ONLY=0
NO_GATE=0
NO_VERIFY=0
NO_POLL=0
SKIP_PUBLISHED=1
ASSUME_YES=0
MIN_INTERVAL=65          # rate-limit safety floor (sec)
BETWEEN_DELAY=90         # extra sleep at tier boundaries (sec)
POLL_INTERVAL=10         # index poll cadence (sec)
POLL_TIMEOUT=600         # max time we'll wait for the index (sec)
MAX_RETRIES=0            # 429 backoff cap when retry-after unparsed (0 = unlimited)
RATE_LIMIT_PAD=15        # extra seconds after server "try again after" time
START_TIER=0
START_CRATE=""
LAST_PUBLISH_ERR=""      # temp log from the last failed publish attempt

# Crates with `publish = false` in their Cargo.toml (cargo skips them;
# listed here for tier-coverage validation only).
SKIPPED=(
    pyrlx
    rlx-cortexm-trainer
)

# Tier definitions. Each array entry is a single tier; space-separated
# crate names within. Order within a tier matters when one member
# depends on another in the same tier (e.g. rlx-ir before rlx-flow).
# Publish order: `cargo publish` resolves every path dep in `[dependencies]`
# and `[dev-dependencies]` (including optional) against crates.io. Within
# a tier, list deps before dependents (e.g. rlx-cpu before rlx-splat).
TIERS=(
    "rlx-ir rlx-gguf rlx-gpu-kernels rlx-mlx-sys rlx-macros rlx-cortexm rlx-bbo"
    "rlx-flow rlx-fusion rlx-driver"
    "rlx-autodiff"
    "rlx-compile"
    "rlx-opt"
    "rlx-cpu rlx-wgpu rlx-cuda rlx-rocm rlx-mlx rlx-tpu rlx-fpga"
    "rlx-splat"
    "rlx-metal"
    "rlx-runtime"
    "rlx-sparse rlx-linalg rlx-umap"
    "rlx-fdm rlx-bench"
    "rlx-rl"
    "rlx"
)

usage() {
    sed -n '2,80p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while (( $# > 0 )); do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --list)           LIST_ONLY=1; shift ;;
        --plan)           PLAN_ONLY=1; shift ;;
        --no-gate)        NO_GATE=1; shift ;;
        --no-verify)      NO_VERIFY=1; shift ;;
        --no-poll)        NO_POLL=1; shift ;;
        --no-skip-published) SKIP_PUBLISHED=0; shift ;;
        --yes|-y)         ASSUME_YES=1; shift ;;
        --min-interval)   MIN_INTERVAL="$2"; shift 2 ;;
        --between-delay)  BETWEEN_DELAY="$2"; shift 2 ;;
        --poll-interval)  POLL_INTERVAL="$2"; shift 2 ;;
        --poll-timeout)   POLL_TIMEOUT="$2"; shift 2 ;;
        --max-retries)    MAX_RETRIES="$2"; shift 2 ;;
        --rate-limit-pad) RATE_LIMIT_PAD="$2"; shift 2 ;;
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

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
bold()   { printf "\033[1m%s\033[0m\n" "$*"; }

validate_tier_coverage() {
    local -a listed=() missing=() extra=()
    local name tier c s

    for tier in "${TIERS[@]}"; do
        for c in $tier; do
            listed+=("$c")
        done
    done

    if ! command -v jq >/dev/null 2>&1; then
        yellow "jq not found — skipping tier coverage check (install jq to enable)."
        return 0
    fi

    while IFS= read -r name; do
        [[ -z "$name" ]] && continue
        for s in "${SKIPPED[@]}"; do
            if [[ "$name" == "$s" ]]; then
                continue 2
            fi
        done
        local found=0
        for c in "${listed[@]}"; do
            if [[ "$name" == "$c" ]]; then
                found=1
                break
            fi
        done
        if (( ! found )); then
            missing+=("$name")
        fi
    done < <(
        cargo metadata --no-deps --format-version 1 2>/dev/null \
            | jq -r '.workspace_members[] as $m | .packages[] | select(.id == $m) | .name'
    )

    for c in "${listed[@]}"; do
        local found=0
        while IFS= read -r name; do
            [[ "$name" == "$c" ]] && found=1 && break
        done < <(
            cargo metadata --no-deps --format-version 1 2>/dev/null \
                | jq -r '.workspace_members[] as $m | .packages[] | select(.id == $m) | .name'
        )
        if (( ! found )); then
            extra+=("$c")
        fi
    done

    if (( ${#missing[@]} > 0 )); then
        red "publish.sh TIERS missing workspace crates: ${missing[*]}"
        exit 1
    fi
    if (( ${#extra[@]} > 0 )); then
        red "publish.sh TIERS list unknown workspace crates: ${extra[*]}"
        exit 1
    fi
}

validate_tier_coverage

# Every rlx-* path dep in [dependencies] / [dev-dependencies] must appear
# in an earlier tier (or the same tier, listed before this crate).
validate_publish_order() {
    if ! command -v python3 >/dev/null 2>&1; then
        yellow "python3 not found — skipping publish-order check (install python3 to enable)."
        return 0
    fi
    local err
    err="$(python3 - "$PWD" <<'PY'
import re, sys
from pathlib import Path

root = Path(sys.argv[1])
script = (root / "scripts/publish.sh").read_text()
m = re.search(r'TIERS=\(\n((?:\s+"[^"]+"\n)+)\)', script)
if not m:
    print("could not parse TIERS from publish.sh", file=sys.stderr)
    sys.exit(2)
tier_lines = re.findall(r'"([^"]+)"', m.group(1))
crate_tier = {}
for i, line in enumerate(tier_lines):
    for j, c in enumerate(line.split()):
        crate_tier[c] = (i, j)

def parse_rlx_deps(toml_path: Path) -> set[str]:
    text = toml_path.read_text()
    deps: set[str] = set()
    for section in ("dependencies", "dev-dependencies"):
        sm = re.search(rf"\[{section}\](.*?)(?=\n\[|\Z)", text, re.S)
        if not sm:
            continue
        for line in sm.group(1).splitlines():
            m2 = re.match(r"^(rlx-[a-z0-9-]+)\s*=", line.strip())
            if m2:
                deps.add(m2.group(1))
    return deps

violations: list[str] = []
for toml in sorted(root.glob("rlx-*/Cargo.toml")):
    name = toml.parent.name
    if name.endswith("-trainer"):
        continue
    if re.search(r"^publish\s*=\s*false", toml.read_text(), re.M):
        continue
    if name not in crate_tier:
        violations.append(f"{name} is publishable but missing from TIERS")
        continue
    my_tier, my_pos = crate_tier[name]
    for dep in sorted(parse_rlx_deps(toml)):
        if dep not in crate_tier:
            violations.append(f"{name}: path dep {dep} is not listed in TIERS")
            continue
        dep_tier, dep_pos = crate_tier[dep]
        if dep_tier > my_tier or (dep_tier == my_tier and dep_pos >= my_pos):
            violations.append(
                f"{name} (tier {dep_tier} pos {dep_pos} needs {dep} before it): "
                f"publish {dep} before {name}"
            )

if violations:
    for v in violations:
        print(v, file=sys.stderr)
    sys.exit(1)
PY
)" || true
    if [[ -n "$err" ]]; then
        red "Publish tier order does not match Cargo.toml path dependencies:"
        while IFS= read -r line; do
            [[ -n "$line" ]] && red "  $line"
        done <<< "$err"
        red "Fix scripts/publish.sh TIERS (or remove path deps from dev-dependencies)."
        exit 1
    fi
}

validate_publish_order

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

# ── Sparse-index helpers (needed for per-crate skip/plan checks) ─
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
    body="$(curl -fsS --max-time 10 "$url" 2>/dev/null || true)"
    if [[ -z "$body" ]]; then
        return 2
    fi
    if printf '%s' "$body" | grep -Fq "\"vers\":\"$version\""; then
        return 0
    fi
    return 1
}

# Returns 0 if <crate>@<version> is already on the sparse index.
version_on_crates_io() {
    check_index "$1" "$2"
}

# Per-crate effective publish versions (explicit `version =` or workspace default).
CRATE_VERSIONS_FILE=""

load_crate_versions() {
    local all_crates="" tier c
    for tier in "${TIERS[@]}"; do
        for c in $tier; do
            all_crates+="$c "
        done
    done

    CRATE_VERSIONS_FILE="$(mktemp)"
    python3 - "$PWD" "$WORKSPACE_VERSION" "$all_crates" <<'PY' > "$CRATE_VERSIONS_FILE"
import re, sys
from pathlib import Path

root = Path(sys.argv[1])
workspace_version = sys.argv[2]
crates = sys.argv[3].split()

def effective_version(crate: str) -> str:
    toml = root / crate / "Cargo.toml"
    if not toml.is_file():
        raise SystemExit(f"could not find Cargo.toml for {crate}")
    text = toml.read_text()
    if re.search(r"^version\.workspace\s*=\s*true", text, re.M):
        return workspace_version
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    if m:
        return m.group(1)
    raise SystemExit(f"could not parse version in {toml}")

for crate in crates:
    print(f"{crate}\t{effective_version(crate)}")
PY
}

crate_version() {
    local crate="$1"
    local ver
    ver="$(awk -F'\t' -v c="$crate" '$1 == c { print $2; exit }' "$CRATE_VERSIONS_FILE")"
    if [[ -z "$ver" ]]; then
        red "unknown crate in version map: $crate" >&2
        exit 1
    fi
    echo "$ver"
}

load_crate_versions
trap 'rm -f "${CRATE_VERSIONS_FILE:-}"' EXIT

# Crates whose effective local version is not yet on the sparse index.
crates_needing_publish() {
    local tier c ver
    for tier in "${TIERS[@]}"; do
        for c in $tier; do
            ver="$(crate_version "$c")"
            if ! version_on_crates_io "$c" "$ver"; then
                echo "$c@$ver"
            fi
        done
    done
}

list_tiers() {
    local check_index="${1:-0}"
    bold "Publish order ([workspace.package] version $WORKSPACE_VERSION):"
    local i=0 tier c ver suffix
    for tier in "${TIERS[@]}"; do
        echo "  tier $i:"
        for c in $tier; do
            ver="$(crate_version "$c")"
            suffix=""
            if (( check_index )); then
                if version_on_crates_io "$c" "$ver"; then
                    suffix="  (on crates.io)"
                else
                    suffix="  (needs publish)"
                fi
            elif [[ "$ver" != "$WORKSPACE_VERSION" ]]; then
                suffix="  (explicit version)"
            fi
            echo "    - $c@$ver$suffix"
        done
        ((i++))
    done
    echo
    bold "Skipped (publish = false):"
    for s in "${SKIPPED[@]}"; do
        case "$s" in
            pyrlx)
                echo "  - pyrlx                    (PyPI via maturin)"
                ;;
            rlx-cortexm-trainer)
                echo "  - rlx-cortexm-trainer      (binary tool; nested under rlx-cortexm/trainer)"
                ;;
            *)
                echo "  - $s"
                ;;
        esac
    done
    echo "  - rlx-cortexm-firmware     (no_std firmware binary; not a workspace member)"
}

print_publish_plan() {
    local -a need=()
    local entry
    while IFS= read -r entry; do
        [[ -z "$entry" ]] && continue
        need+=("$entry")
    done < <(crates_needing_publish)

    if (( ${#need[@]} == 0 )); then
        green "Nothing to publish — every listed crate is already at its local version on crates.io."
        return 0
    fi

    bold "Crates needing publish (${#need[@]}):"
    for entry in "${need[@]}"; do
        echo "  - $entry"
    done
}

if (( LIST_ONLY )); then
    list_tiers 1
    echo
    print_publish_plan
    exit 0
fi

if (( PLAN_ONLY )); then
    print_publish_plan
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
list_tiers 1
echo
print_publish_plan
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
    if (( MAX_RETRIES == 0 )); then
        yellow "Registry retries:        auto-wait + retry until success (unlimited)."
    else
        yellow "Registry retries:        up to ${MAX_RETRIES} backoff attempts per crate."
    fi
    yellow "Retry-after parsing:     HTTP-date + status-specific backoff (+${RATE_LIMIT_PAD}s pad)."
    if (( SKIP_PUBLISHED )); then
        yellow "Already on crates.io:    skip when local version matches index."
    else
        yellow "Already on crates.io:    still attempt upload (--no-skip-published)."
    fi
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
# (sparse_index_path, check_index, version_on_crates_io are defined
# above for per-crate skip/plan checks.)

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

# True when a path dep was rewritten for publish but is not on crates.io yet.
is_missing_registry_dep_error() {
    local err_file="$1"
    [[ -f "$err_file" ]] || return 1
    grep -qE 'no matching package named|location searched: crates\.io index' "$err_file"
}

missing_registry_dep_name() {
    local err_file="$1"
    grep -oE 'no matching package named `[^`]+`' "$err_file" 2>/dev/null \
        | head -1 \
        | sed -E 's/no matching package named `([^`]+)`/\1/'
}

# True when cargo says this version was uploaded already.
is_already_exists_error() {
    local err_file="$1"
    [[ -f "$err_file" ]] || return 1
    grep -qiE 'already exists on crates\.io|already exists on the registry' "$err_file"
}

# Last HTTP status from a `status NNN` line in cargo/crates.io output (e.g. 429).
registry_http_status_from_log() {
    local err_file="$1"
    [[ -f "$err_file" ]] || return 0
    grep -oiE 'status [0-9]{3}' "$err_file" 2>/dev/null \
        | tail -1 \
        | grep -oE '[0-9]{3}' \
        || true
}

# Transient registry / network failures — retry after a computed wait.
is_retryable_registry_error() {
    local err_file="$1"
    [[ -f "$err_file" ]] || return 1
    local status
    status="$(registry_http_status_from_log "$err_file")"
    case "$status" in
        408|429|500|502|503|504) return 0 ;;
    esac
    grep -qiE \
        'Too Many Requests|rate.?limit|published too many new crates|try again after|temporarily unavailable|service unavailable|bad gateway|gateway timeout|timed out|connection reset|connection refused|unexpected eof|broken pipe|error sending request|operation timed out|dns error' \
        "$err_file"
}

# Convert crates.io HTTP-date ("Wed, 27 May 2026 11:09:09 GMT") → epoch.
http_date_to_epoch() {
    local when="$1"
    if [[ "$(uname -s)" == Darwin ]]; then
        date -j -u -f '%a, %d %b %Y %H:%M:%S GMT' "$when" '+%s' 2>/dev/null
    else
        date -u -d "$when" '+%s' 2>/dev/null
    fi
}

# Default backoff when no explicit retry window is in the log.
registry_status_default_wait() {
    local status="$1"
    case "$status" in
        429) echo $(( MIN_INTERVAL * 2 )) ;;
        408) echo 60 ;;
        500|502|503|504) echo 90 ;;
        *) echo "$MIN_INTERVAL" ;;
    esac
}

# Seconds to wait before retrying a registry error.
# Prints: wait_seconds parsed_flag http_status
#   parsed_flag=1 → wait derived from server retry window or status default
registry_retry_wait_seconds() {
    local err_file="$1"
    local when epoch now wait parsed=0 status
    status="$(registry_http_status_from_log "$err_file")"
    wait="$(registry_status_default_wait "$status")"

    when="$(
        grep -oiE '(please )?try again after [A-Za-z]{3}, [0-9]+ [A-Za-z]+ [0-9]+ [0-9:]+ GMT' \
            "$err_file" 2>/dev/null \
            | tail -1 \
            | sed -E 's/^[Pp]lease [Tt]ry again after //; s/^[Tt]ry again after //'
    )"
    if [[ -n "$when" ]]; then
        epoch="$(http_date_to_epoch "$when" || true)"
        if [[ -n "${epoch:-}" ]]; then
            now="$(date -u '+%s')"
            wait=$(( epoch - now + RATE_LIMIT_PAD ))
            parsed=1
            if (( wait < MIN_INTERVAL )); then
                wait=$MIN_INTERVAL
            fi
            if (( wait > 86400 )); then
                wait=86400
            fi
            echo "$wait $parsed ${status:-0}"
            return 0
        fi
        if grep -qiE 'try again after|published too many new crates' "$err_file"; then
            wait=$(( MIN_INTERVAL * 2 ))
            parsed=1
            echo "$wait $parsed ${status:-0}"
            return 0
        fi
    fi

    local retry_after
    retry_after="$(
        grep -oiE 'retry-after: *[0-9]+' "$err_file" 2>/dev/null \
            | tail -1 \
            | grep -oE '[0-9]+' \
            || true
    )"
    if [[ -n "$retry_after" ]]; then
        wait=$(( retry_after + RATE_LIMIT_PAD ))
        parsed=1
        echo "$wait $parsed ${status:-0}"
        return 0
    fi

    if is_retryable_registry_error "$err_file"; then
        parsed=1
    fi
    echo "$wait $parsed ${status:-0}"
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
ALREADY_ON_CRATES_IO=()
PUBLISHED_THIS_RUN=()

publish_one_attempt() {
    local crate="$1"
    local args=()
    args+=("--package" "$crate")
    if (( DRY_RUN )); then
        args+=("--dry-run" "--allow-dirty")
    fi
    if (( NO_VERIFY )); then
        args+=("--no-verify")
    fi
    bold ">> cargo publish ${args[*]}"
    local tmp_err
    tmp_err="$(mktemp)"
    LAST_PUBLISH_ERR=""
    # Do not trust cargo's exit code alone — crates.io 429 often returns 0.
    set +e
    cargo publish "${args[@]}" 2>&1 | tee "$tmp_err"
    local rc=${PIPESTATUS[0]}
    set -e

    if is_already_exists_error "$tmp_err"; then
        rm -f "$tmp_err"
        return 43
    fi

    if is_retryable_registry_error "$tmp_err"; then
        LAST_PUBLISH_ERR="$tmp_err"
        return 42
    fi

    if is_missing_registry_dep_error "$tmp_err"; then
        LAST_PUBLISH_ERR="$tmp_err"
        return 44
    fi

    if (( rc == 0 )) && ! grep -qE 'error(\[[0-9]+\])?: failed|error(\[[0-9]+\])?: failed to publish' "$tmp_err"; then
        rm -f "$tmp_err"
        return 0
    fi

    LAST_PUBLISH_ERR="$tmp_err"
    return 1
}

# 0 = published (or dry-run ok), 1 = hard failure, 2 = already on crates.io
publish_one() {
    local crate="$1"
    local version
    version="$(crate_version "$crate")"
    local attempt=1
    local backoff=$MIN_INTERVAL
    local missing_dep_attempts=0
    while true; do
        # `|| rc=$?` keeps `set -e` from aborting on non-zero returns (42/44).
        local rc=0
        publish_one_attempt "$crate" || rc=$?
        if (( rc == 0 )); then
            rm -f "${LAST_PUBLISH_ERR:-}"
            LAST_PUBLISH_ERR=""
            if (( DRY_RUN )); then
                DRY_RUN_PASS+=("$crate")
            fi
            return 0
        fi
        if (( rc == 43 )); then
            green "  $crate@$version already on crates.io (cargo confirmed) — skip."
            return 2
        fi
        if (( DRY_RUN )); then
            DRY_RUN_FAIL+=("$crate")
            yellow "  (dry-run: dep resolution failed — expected for non-leaf crates pre-publish)"
            rm -f "${LAST_PUBLISH_ERR:-}"
            LAST_PUBLISH_ERR=""
            return 0
        fi
        if (( rc == 44 )); then
            local missing=""
            if [[ -n "${LAST_PUBLISH_ERR:-}" && -f "$LAST_PUBLISH_ERR" ]]; then
                missing="$(missing_registry_dep_name "$LAST_PUBLISH_ERR")"
            fi
            if [[ -n "$missing" ]]; then
                local missing_ver
                missing_ver="$(crate_version "$missing")"
                missing_dep_attempts=$(( missing_dep_attempts + 1 ))
                yellow "  $crate: waiting for $missing@$missing_ver on sparse index (attempt $missing_dep_attempts), then retry."
                wait_for_index "$missing" "$missing_ver"
                rm -f "${LAST_PUBLISH_ERR:-}"
                LAST_PUBLISH_ERR=""
                sleep_with_countdown "$POLL_INTERVAL" "index settle after $missing"
                continue
            fi
            red "Publish blocked for $crate: dependency not on crates.io yet."
            red "  Check scripts/publish.sh tier order."
            exit 1
        fi
        if (( rc == 42 )); then
            local wait_sec=$MIN_INTERVAL parsed=0 http_status=0 when_human=""
            if [[ -n "${LAST_PUBLISH_ERR:-}" && -f "$LAST_PUBLISH_ERR" ]]; then
                read -r wait_sec parsed http_status < <(registry_retry_wait_seconds "$LAST_PUBLISH_ERR")
                if grep -qiE 'try again after' "$LAST_PUBLISH_ERR"; then
                    when_human="$(
                        grep -oiE '(please )?try again after [A-Za-z]{3}, [0-9]+ [A-Za-z]+ [0-9]+ [0-9:]+ GMT' \
                            "$LAST_PUBLISH_ERR" 2>/dev/null \
                            | tail -1 \
                            | sed -E 's/^[Pp]lease [Tt]ry again after //; s/^[Tt]ry again after //'
                    )"
                fi
                rm -f "$LAST_PUBLISH_ERR"
                LAST_PUBLISH_ERR=""
            fi
            if (( parsed )); then
                if (( http_status > 0 )); then
                    yellow "  crates.io HTTP $http_status for $crate — waiting ${wait_sec}s then retry (automatic)."
                else
                    yellow "  registry transient error for $crate — waiting ${wait_sec}s then retry (automatic)."
                fi
                if [[ -n "$when_human" ]]; then
                    yellow "    server window ends: $when_human (+${RATE_LIMIT_PAD}s pad)"
                fi
                attempt=1
                backoff=$MIN_INTERVAL
            else
                if (( MAX_RETRIES > 0 && attempt > MAX_RETRIES )); then
                    red "Publish failed for $crate after $MAX_RETRIES registry retries."
                    red "Re-run with --start-crate $crate to resume later."
                    exit 1
                fi
                wait_sec=$backoff
                if (( MAX_RETRIES > 0 )); then
                    yellow "  registry retry for $crate (attempt $attempt/$MAX_RETRIES) — backoff ${wait_sec}s."
                else
                    yellow "  registry retry for $crate (attempt $attempt) — backoff ${wait_sec}s."
                fi
                backoff=$(( backoff * 2 ))
                if (( backoff > 600 )); then
                    backoff=600
                fi
                attempt=$(( attempt + 1 ))
            fi
            sleep_with_countdown "$wait_sec" "crates.io registry cooldown"
            continue
        fi
        if [[ -n "${LAST_PUBLISH_ERR:-}" && -f "$LAST_PUBLISH_ERR" ]]; then
            red "  last cargo log: $LAST_PUBLISH_ERR"
        fi
        red "Publish failed for $crate (non-retryable, exit $rc)."
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

    published_in_tier=0
    for crate in $tier; do
        local version
        version="$(crate_version "$crate")"

        # Skip past --start-crate if specified.
        if [[ -n "$skip_until_crate" ]]; then
            if [[ "$crate" != "$skip_until_crate" ]]; then
                yellow "  skip $crate (resume target: $skip_until_crate)"
                continue
            fi
            skip_until_crate=""
        fi

        # Already on crates.io at this crate's local version — no upload.
        if (( SKIP_PUBLISHED )) && (( ! DRY_RUN )); then
            if version_on_crates_io "$crate" "$version"; then
                green "  skip $crate@$version (already on crates.io)"
                ALREADY_ON_CRATES_IO+=("$crate@$version")
                continue
            fi
        fi

        # Rate-limit floor only between actual uploads (not after skips).
        if (( published_in_tier > 0 )) && (( ! DRY_RUN )); then
            sleep_with_countdown "$MIN_INTERVAL" "rate-limit floor"
        fi

        publish_one "$crate"
        pub_rc=$?
        if (( pub_rc == 2 )); then
            ALREADY_ON_CRATES_IO+=("$crate@$version")
            continue
        fi
        if (( pub_rc != 0 )); then
            exit 1
        fi

        PUBLISHED_THIS_RUN+=("$crate@$version")
        published_in_tier=$(( published_in_tier + 1 ))

        # Post-publish: poll the index until the new version is
        # actually queryable, so the *next* crate's dep resolution
        # doesn't 404. Dry runs skip this entirely.
        if (( ! DRY_RUN )); then
            wait_for_index "$crate" "$version"
        fi
    done

    # Between-tier extra delay only when this tier uploaded something.
    if (( ! DRY_RUN )) && (( tier_idx + 1 < ${#TIERS[@]} )) && (( published_in_tier > 0 )); then
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
    echo
    if (( ${#PUBLISHED_THIS_RUN[@]} > 0 )); then
        green "Published this run (${#PUBLISHED_THIS_RUN[@]}):"
        for c in "${PUBLISHED_THIS_RUN[@]}"; do echo "    ✓ $c"; done
    fi
    if (( ${#ALREADY_ON_CRATES_IO[@]} > 0 )); then
        yellow "Already on crates.io — skipped (${#ALREADY_ON_CRATES_IO[@]}):"
        for c in "${ALREADY_ON_CRATES_IO[@]}"; do echo "    ○ $c"; done
    fi
    if (( ${#PUBLISHED_THIS_RUN[@]} == 0 )) && (( ${#ALREADY_ON_CRATES_IO[@]} > 0 )); then
        green "Nothing left to publish — every listed crate is already at its local version on crates.io."
    elif (( ${#PUBLISHED_THIS_RUN[@]} > 0 )); then
        green "Publish run finished."
    else
        green "All tiers processed."
    fi
fi
