#!/usr/bin/env bash
# Build RlxNode.xcframework — the RLX distributed node as a linkable Apple
# framework, one slice per Apple OS × (device, simulator).
#
#   ios/build-xcframework.sh                        # every Apple OS, CPU node
#   ios/build-xcframework.sh --apple                # + Metal / ANE
#   ios/build-xcframework.sh --apple --model        # + `mode = "model"` (dist_node MODE=model);
#                                                   #   unwinding `release` profile, for lane fallback
#   ios/build-xcframework.sh --platforms ios,visionos
#   ios/build-xcframework.sh --all-archs            # + arm64_32 watch, x86_64 sims
#   ios/build-xcframework.sh --device-only          # no simulator slices: one Rust build
#                                                   #   per platform instead of two or three
#   ios/build-xcframework.sh --capi                 # RLX.xcframework: the whole rlx C ABI
#   ios/build-xcframework.sh --capi-crate <Cargo.toml>   # … with model plugins linked in
#
# `--capi` packages `rlx-capi` (graphs, sessions, training, text, weights — the
# library under `crates/bindings/rlx-swift`) instead of the node, as
# RLX.xcframework with a `CRLX` module over `rlx.h`, into
# crates/bindings/rlx-swift/build/. Its default platforms are ios and macos, so
# the same artifact serves an iOS app and `swift test` on a Mac; `--apple`
# there means the `metal` feature.
#
# An iOS app cannot load a model plugin, so it links them: `--capi-crate`
# packages a static library crate that re-exports rlx-capi and names its
# plugins with `rlx_runtime::link_model_plugins!` (as
# crates/bindings/rlx-capi/tests/linked does), in the same RLX.xcframework.
# The app calls `rlx_link_model_plugins()` once before its first model call.
#
# Built with the size-tuned `apple` cargo profile (see [profile.apple] in the
# workspace Cargo.toml): LTO, one codegen unit, and `panic = "abort"`, which
# alone removes ~1.1 MB of unwinding tables from a linked app. `--profile`
# overrides it; `--debug` is shorthand for the dev profile.
#
# Platforms: ios, tvos, watchos, visionos (Apple and xcodebuild call the last
# one `xros`; Rust calls it `visionos`, and this script follows Rust). `macos`
# is understood but off by default — a Mac links the Rust staticlib directly.
# Add it (`--platforms ios,tvos,watchos,visionos,macos`) if you consume this
# through Package.swift on a Mac, since that manifest declares `.macOS`.
#
# Missing Rust targets are installed on demand; a platform whose targets cannot
# be installed is skipped with a note rather than failing the whole build.
set -euo pipefail

CALLER="$PWD"
cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="$ROOT/ios/build"
FEATURES=()
APPLE=0
MODEL=0
CAPI=0
PROFILE=apple
PROFILE_SET=0
PLATFORMS=""
# Tier-3 Rust targets (no prebuilt std) are off by default: they need a nightly
# toolchain and a from-source std, which turns a 10-minute build into a much
# longer one. What they buy is real coverage, not completeness for its own
# sake — Apple Watch Series 4-8 are 32-bit (`arm64_32`) and simply cannot run
# the default slice, and an Intel Mac cannot run the arm64 simulator ones.
ALL_ARCHS=0
BUILD_STD=0
# --device-only: physical devices only. iOS has two simulator architectures
# (arm64, x86_64), each a full build of the Rust stack, so this cuts an iOS
# build to a third.
DEVICE_ONLY=0

# Deployment targets. Rust stamps every object with one, and a link that mixes
# floors emits 'built for newer <OS>' for each mismatched object, so set them
# here rather than inheriting whatever SDK is current.
#
# iOS and tvOS sit at 17.0 because that is where Metal gained native `bfloat`,
# which MLX's kernels use. Measured against the Metal compiler: `bfloat` is an
# unknown type at 16.0 on both and available from 17.0; visionOS 1.0 already
# shipped past that line. Below the floor MLX does not fail on `bfloat` alone —
# it falls back to its emulated bf16 header and the build dies in a wall of
# `redefinition of 'abs'`. A CPU-only node would be happy at 15.0; keeping one
# floor for both avoids an xcframework whose slices disagree.
#
# watchOS is the odd one: rustup's prebuilt `std` for aarch64-apple-watchos is
# itself compiled at 26.0, so a lower floor cannot be honoured without
# rebuilding std from source (`-Zbuild-std`). Overriding this below that is
# accepted by rustc and then warned about by the linker — raise the app's
# deployment target instead, or build std yourself.
export IPHONEOS_DEPLOYMENT_TARGET=${IPHONEOS_DEPLOYMENT_TARGET:-17.0}
export TVOS_DEPLOYMENT_TARGET=${TVOS_DEPLOYMENT_TARGET:-17.0}
export WATCHOS_DEPLOYMENT_TARGET=${WATCHOS_DEPLOYMENT_TARGET:-26.0}
export XROS_DEPLOYMENT_TARGET=${XROS_DEPLOYMENT_TARGET:-1.0}
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-12.0}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apple) APPLE=1; shift ;;
    --model) MODEL=1; shift ;;
    --capi) CAPI=1; shift ;;
    --capi-crate)
      [[ $# -ge 2 ]] || { echo "--capi-crate needs a Cargo.toml" >&2; exit 2; }
      CAPI=1; CAPI_MANIFEST="$(cd "$CALLER" && cd "$(dirname "$2")" && pwd)/$(basename "$2")"
      shift 2 ;;
    --debug) PROFILE=dev; PROFILE_SET=1; shift ;;
    --profile)
      [[ $# -ge 2 ]] || { echo "--profile needs a value" >&2; exit 2; }
      PROFILE="$2"; PROFILE_SET=1; shift 2 ;;
    --profile=*) PROFILE="${1#--profile=}"; PROFILE_SET=1; shift ;;
    --all-archs) ALL_ARCHS=1; BUILD_STD=1; shift ;;
    --device-only) DEVICE_ONLY=1; shift ;;
    --build-std) BUILD_STD=1; shift ;;
    --platforms)
      [[ $# -ge 2 ]] || { echo "--platforms needs a value" >&2; exit 2; }
      PLATFORMS="${2//,/ }"; shift 2 ;;
    --platforms=*) PLATFORMS="${1#--platforms=}"; PLATFORMS="${PLATFORMS//,/ }"; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

# Rust targets per platform: the first is the device, the rest are simulator
# architectures to lipo into one slice.
#
# `--all-archs` adds the tier-3 ones. There is deliberately no x86_64 visionOS
# simulator: rustc has no such target, and the Vision Pro simulator requires an
# Apple Silicon Mac anyway.
device_target() {
  case "$1" in
    ios)      echo aarch64-apple-ios ;;
    tvos)     echo aarch64-apple-tvos ;;
    watchos)
      # arm64_32 is Apple Watch Series 4-8 — 32-bit pointers, a different CPU
      # subtype, and the only slice those watches can load.
      if [[ $ALL_ARCHS == 1 ]]; then echo "aarch64-apple-watchos arm64_32-apple-watchos"
      else echo aarch64-apple-watchos; fi ;;
    visionos) echo aarch64-apple-visionos ;;
    macos)    echo "aarch64-apple-darwin x86_64-apple-darwin" ;;
    *) echo "unknown platform: $1" >&2; exit 2 ;;
  esac
}

sim_targets() {
  case "$1" in
    ios)      echo "aarch64-apple-ios-sim x86_64-apple-ios" ;;
    tvos)
      if [[ $ALL_ARCHS == 1 ]]; then echo "aarch64-apple-tvos-sim x86_64-apple-tvos"
      else echo aarch64-apple-tvos-sim; fi ;;
    watchos)
      if [[ $ALL_ARCHS == 1 ]]; then echo "aarch64-apple-watchos-sim x86_64-apple-watchos-sim"
      else echo aarch64-apple-watchos-sim; fi ;;
    visionos) echo aarch64-apple-visionos-sim ;;
    macos)    echo "" ;;   # a Mac has no simulator variant
    *) echo "unknown platform: $1" >&2; exit 2 ;;
  esac
}

if [[ $CAPI == 1 ]]; then
  CRATE=rlx-capi; lib=librlx_capi.a; NAME=RLX; MODULE=CRLX
  PACKAGE=(-p "$CRATE")
  if [[ -n ${CAPI_MANIFEST:-} ]]; then
    # Its own workspace, built into this one's target/ so the paths below hold.
    CRATE=$(cargo metadata --no-deps --format-version 1 --manifest-path "$CAPI_MANIFEST" \
      | python3 -c 'import json, sys; print(json.load(sys.stdin)["packages"][0]["name"])')
    lib="lib${CRATE//-/_}.a"
    PACKAGE=(--manifest-path "$CAPI_MANIFEST" --target-dir "$ROOT/target")
  fi
  HEADER="$ROOT/crates/bindings/rlx-capi/include/rlx.h"
  OUT="$ROOT/crates/bindings/rlx-swift/build"
  PLATFORMS=${PLATFORMS:-"ios macos"}
  # A `--capi-crate` crate picks its own features, Metal included.
  [[ $APPLE == 1 && -z ${CAPI_MANIFEST:-} ]] && FEATURES=(--features metal,blas-accelerate)
  # Static archive only: rlx-capi also declares `cdylib`, and a shared library
  # must resolve every symbol at link time, which zstd-sys does not on iOS.
  CARGO_FLAGS=("${PACKAGE[@]}" --lib --crate-type staticlib --profile "$PROFILE")
  SUBCOMMAND=rustc
else
  CRATE=rlx-ffi; lib=librlx_ffi.a; NAME=RlxNode; MODULE=RlxNodeFFI
  HEADER="$ROOT/crates/bindings/rlx-ffi/include/rlx_node.h"
  PLATFORMS=${PLATFORMS:-"ios tvos watchos visionos"}
  feats=()
  [[ $APPLE == 1 ]] && feats+=(apple)
  [[ $MODEL == 1 ]] && feats+=(model)
  # BLE=1: the node can take mesh links over Bluetooth LE (`via = "ble"`).
  [[ ${BLE:-0} == 1 ]] && feats+=(ble)
  # A `model` node runs training lanes, and a lane that fails (a backend that
  # cannot run the graph — Core ML missing an op) is dropped by catching its
  # panic. `panic = "abort"` (the `apple` profile) turns that into the whole
  # app dying, so unwind unless a profile was asked for.
  if [[ $MODEL == 1 && $PROFILE_SET == 0 ]]; then
    PROFILE=release
  fi
  if [[ ${#feats[@]} -gt 0 ]]; then
    FEATURES=(--features "$(IFS=,; echo "${feats[*]}")")
  fi
  CARGO_FLAGS=(-p "$CRATE" --lib --profile "$PROFILE")
  SUBCOMMAND=build
fi
# `--profile dev` writes to target/<triple>/debug, every other profile to a
# directory named after itself.
OUTDIR=$PROFILE
[[ "$PROFILE" == dev ]] && OUTDIR=debug

if [[ $BUILD_STD == 1 ]]; then
  rustup toolchain install nightly >/dev/null
  rustup component add rust-src --toolchain nightly >/dev/null
  # `panic_abort` has to be in the list or a panic=abort profile has no
  # panic runtime to link against.
  CARGO=(cargo +nightly)
  STD_FLAGS=(-Zbuild-std=std,panic_abort)
else
  CARGO=(cargo)
  STD_FLAGS=()
fi

# True if the Rust target is usable — already installed, installable now, or
# (when building std from source) merely known to rustc.
have_target() {
  if [[ $BUILD_STD == 1 ]]; then
    rustc --print target-list | grep -qx "$1" && return 0
  fi
  rustup target list --installed | grep -qx "$1" && return 0
  rustup target add "$1" >/dev/null 2>&1
}

build() {
  echo "==> $1 (profile $PROFILE)"
  # Only this target's own deployment variable: clang reads every
  # *_DEPLOYMENT_TARGET in the environment and refuses to compile a C/ObjC
  # shim (rlx-coreml's, under --apple) when several disagree.
  local keep
  case "$1" in
    *-ios|*-ios-sim|*-ios-macabi) keep=IPHONEOS_DEPLOYMENT_TARGET ;;
    *-tvos|*-tvos-sim) keep=TVOS_DEPLOYMENT_TARGET ;;
    *-watchos|*-watchos-sim) keep=WATCHOS_DEPLOYMENT_TARGET ;;
    *-visionos|*-visionos-sim) keep=XROS_DEPLOYMENT_TARGET ;;
    *-darwin) keep=MACOSX_DEPLOYMENT_TARGET ;;
    *) keep="" ;;
  esac
  local unset=()
  for v in IPHONEOS_DEPLOYMENT_TARGET TVOS_DEPLOYMENT_TARGET WATCHOS_DEPLOYMENT_TARGET \
           XROS_DEPLOYMENT_TARGET MACOSX_DEPLOYMENT_TARGET; do
    [[ $v == "$keep" ]] || unset+=(-u "$v")
  done
  env "${unset[@]}" "${CARGO[@]}" "$SUBCOMMAND" "${STD_FLAGS[@]+"${STD_FLAGS[@]}"}" \
    "${CARGO_FLAGS[@]}" "${FEATURES[@]+"${FEATURES[@]}"}" --target "$1"
}

rm -rf "$OUT"; mkdir -p "$OUT/headers"
cp "$HEADER" "$OUT/headers/"
cat > "$OUT/headers/module.modulemap" <<MAP
module $MODULE {
    header "$(basename "$HEADER")"
    export *
}
MAP

# One `-library`/`-headers` pair per slice, accumulated for a single
# `-create-xcframework` call: xcodebuild reads each archive's own
# LC_BUILD_VERSION to name the slice, so device and simulator never collide.
SLICES=()
BUILT=()

# $1 = platform, $2 = slice kind (device|simulator), rest = Rust targets.
add_slice() {
  local platform="$1" kind="$2"; shift 2
  [[ $# -gt 0 ]] || return 1        # e.g. macOS, which has no simulator
  local usable=()
  for t in "$@"; do
    if have_target "$t"; then usable+=("$t"); else
      echo "note: $platform/$kind — Rust target $t unavailable, skipping that arch" >&2
    fi
  done
  [[ ${#usable[@]} -gt 0 ]] || { echo "note: $platform/$kind has no usable target — slice omitted" >&2; return 1; }

  # `set -e` does not fire inside a function invoked from a `&&` list, which is
  # exactly how the platform loop calls this. Without the explicit exit a failed
  # cargo build just omits a slice, and `-create-xcframework` reports it as an
  # invalid *library path* — a message that points at the packaging step rather
  # than at the compile error 40 lines up.
  for t in "${usable[@]}"; do
    build "$t" || { echo "ERROR: cargo build failed for $t — aborting" >&2; exit 1; }
  done

  local dir="$OUT/$platform-$kind"
  mkdir -p "$dir"
  if [[ ${#usable[@]} -gt 1 ]]; then
    local inputs=()
    for t in "${usable[@]}"; do inputs+=("target/$t/$OUTDIR/$lib"); done
    lipo -create "${inputs[@]}" -output "$dir/$lib"
  else
    cp "target/${usable[0]}/$OUTDIR/$lib" "$dir/$lib"
  fi
  SLICES+=(-library "$dir/$lib" -headers "$OUT/headers")
}

for platform in $PLATFORMS; do
  ok=0
  # shellcheck disable=SC2046  # word splitting is how the target lists are passed
  add_slice "$platform" device $(device_target "$platform") && ok=1
  # shellcheck disable=SC2046
  [[ $DEVICE_ONLY == 1 ]] || { add_slice "$platform" simulator $(sim_targets "$platform") && ok=1; }
  [[ $ok == 1 ]] && BUILT+=("$platform") || true
done

[[ ${#SLICES[@]} -gt 0 ]] || { echo "no slices built" >&2; exit 1; }

xcodebuild -create-xcframework "${SLICES[@]}" -output "$OUT/$NAME.xcframework"

# MLX ships its GPU kernels in a separate `mlx.metallib`, loaded lazily at the
# first kernel and looked up NEXT TO THE EXECUTABLE — a static archive cannot
# carry it. Without it MLX links, reports itself available, and then fails at
# first use with:
#
#   Failed to load the default metallib. library not found
#
# Observed on an iPad exactly that way. Emit it beside the xcframework so an app
# target can copy it into its bundle; there is nothing to emit unless `--apple`
# built MLX in.
if [[ ${#FEATURES[@]} -gt 0 ]]; then
  for platform in ${BUILT[*]:-}; do
    for t in $(device_target "$platform") $(sim_targets "$platform"); do
      lib_path="target/$t/$OUTDIR/build"
      # A target this run did not build (a simulator under --device-only) has
      # no build directory, and a failing find would end the script here.
      found=$(find "$lib_path" -name mlx.metallib -path "*/out/lib/*" 2>/dev/null | head -1 || true)
      if [[ -n "$found" ]]; then
        mkdir -p "$OUT/metallib/$t"
        cp "$found" "$OUT/metallib/$t/mlx.metallib"
      fi
    done
  done
  if [[ -d "$OUT/metallib" ]]; then
    echo
    echo "MLX kernel libraries -> $OUT/metallib/<target>/mlx.metallib"
    echo "  Copy the one matching your slice into the .app bundle NEXT TO the"
    echo "  executable (or as default.metallib). MLX loads it at the first GPU"
    echo "  kernel and fails at that point if it is absent."
  fi
fi

echo "built $OUT/$NAME.xcframework for: ${BUILT[*]:-}"
