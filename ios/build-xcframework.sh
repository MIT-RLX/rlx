#!/usr/bin/env bash
# Build RlxNode.xcframework — the RLX distributed node as a linkable Apple
# framework (device + simulator).
#
#   ios/build-xcframework.sh              # CPU-only node
#   ios/build-xcframework.sh --apple      # + Metal / ANE
#
# Requires the Rust Apple targets:
#   rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="$ROOT/ios/build"
FEATURES=()
PROFILE=release
# Match the app's minimum. Without this the Rust objects are stamped with
# the SDK's current version and every link emits 'built for newer iOS'.
export IPHONEOS_DEPLOYMENT_TARGET=${IPHONEOS_DEPLOYMENT_TARGET:-15.0}

for arg in "$@"; do
  case "$arg" in
    --apple) FEATURES=(--features apple) ;;
    --debug) PROFILE=debug ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

CARGO_FLAGS=(-p rlx-ffi --lib)
[[ "$PROFILE" == release ]] && CARGO_FLAGS+=(--release)

DEVICE=aarch64-apple-ios
SIM_ARM=aarch64-apple-ios-sim
SIM_X86=x86_64-apple-ios

build() {
  echo "==> $1"
  cargo build "${CARGO_FLAGS[@]}" "${FEATURES[@]+"${FEATURES[@]}"}" --target "$1"
}

build "$DEVICE"
build "$SIM_ARM"
# The Intel simulator slice is optional — skip it if the target isn't installed
# rather than failing a build that an Apple Silicon Mac doesn't need.
HAVE_X86=0
if rustup target list --installed | grep -qx "$SIM_X86"; then
  build "$SIM_X86"; HAVE_X86=1
else
  echo "note: $SIM_X86 not installed — simulator slice will be arm64-only"
fi

lib=librlx_ffi.a
rm -rf "$OUT"; mkdir -p "$OUT/sim" "$OUT/headers"
cp "$ROOT/crates/bindings/rlx-ffi/include/rlx_node.h" "$OUT/headers/"
cat > "$OUT/headers/module.modulemap" <<MAP
module RlxNodeFFI {
    header "rlx_node.h"
    export *
}
MAP

if [[ $HAVE_X86 == 1 ]]; then
  lipo -create \
    "target/$SIM_ARM/$PROFILE/$lib" \
    "target/$SIM_X86/$PROFILE/$lib" \
    -output "$OUT/sim/$lib"
else
  cp "target/$SIM_ARM/$PROFILE/$lib" "$OUT/sim/$lib"
fi

xcodebuild -create-xcframework \
  -library "target/$DEVICE/$PROFILE/$lib" -headers "$OUT/headers" \
  -library "$OUT/sim/$lib"                -headers "$OUT/headers" \
  -output "$OUT/RlxNode.xcframework"

echo "built $OUT/RlxNode.xcframework"
