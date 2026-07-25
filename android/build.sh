#!/usr/bin/env bash
# Cross-build librlx_jni.so for arm64 Android and stage it under
# app/src/main/jniLibs/arm64-v8a/ for the Gradle project.
#
# Usage:
#   ./build.sh              # scalar CPU (default, no OpenBLAS)
#   ./build.sh --blas       # static OpenBLAS (run build-openblas.sh first)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET=aarch64-linux-android
ABI=arm64-v8a
JNI_DIR="$ROOT/app/src/main/jniLibs/$ABI"
OUT_SO="$ROOT/target/$TARGET/release/librlx_jni.so"
RLX_JNI_FEATURES="scalar"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --blas)
      RLX_JNI_FEATURES="blas"
      export OPENBLAS_LIB_DIR="$ROOT/third_party/openblas-android/lib"
      if [[ ! -f "$OPENBLAS_LIB_DIR/libopenblas.a" && ! -f "$OPENBLAS_LIB_DIR/libopenblas.so" ]]; then
        echo "OpenBLAS not found at $OPENBLAS_LIB_DIR — run ./build-openblas.sh first." >&2
        exit 1
      fi
      shift
      ;;
    *)
      echo "Unknown arg: $1 (try --blas)" >&2
      exit 1
      ;;
  esac
done

# shellcheck disable=SC1091
eval "$("$ROOT/ndk-env.sh")"

rustup target add "$TARGET" >/dev/null 2>&1 || true

echo "Building rlx-jni for $TARGET (features=$RLX_JNI_FEATURES, NDK: $ANDROID_NDK_HOME)"
cargo build --manifest-path "$ROOT/rlx-jni/Cargo.toml" \
  --release --target "$TARGET" \
  --no-default-features --features "$RLX_JNI_FEATURES"

mkdir -p "$JNI_DIR"
cp "$OUT_SO" "$JNI_DIR/"
echo "Staged $JNI_DIR/librlx_jni.so"

# Dynamic OpenBLAS only — copy alongside rlx_jni if present.
if [[ -f "${OPENBLAS_LIB_DIR:-}/libopenblas.so" ]]; then
  cp "$OPENBLAS_LIB_DIR/libopenblas.so" "$JNI_DIR/"
  echo "Staged $JNI_DIR/libopenblas.so"
fi
