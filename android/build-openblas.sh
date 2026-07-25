#!/usr/bin/env bash
# Cross-build static OpenBLAS for aarch64-linux-android.
# Install prefix: android/third_party/openblas-android/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PREFIX="$ROOT/third_party/openblas-android"
SRC="$ROOT/third_party/OpenBLAS-src"
TARGET=aarch64-linux-android
TARGET_ENV="$(printf '%s' "$TARGET" | tr '[:lower:]-' '[:upper:]_')"

default_android_home() {
  if [[ -d /opt/homebrew/share/android-commandlinetools ]]; then
    echo /opt/homebrew/share/android-commandlinetools
  fi
}

find_ndk() {
  if [[ -n "${ANDROID_NDK_HOME:-}" && -d "$ANDROID_NDK_HOME" ]]; then
    echo "$ANDROID_NDK_HOME"
    return
  fi
  local home="${ANDROID_HOME:-$(default_android_home)}"
  if [[ -n "$home" && -d "$home/ndk" ]]; then
    find "$home/ndk" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -V | tail -1
  fi
}

NDK="$(find_ndk)"
if [[ -z "$NDK" || ! -d "$NDK" ]]; then
  echo "Set ANDROID_NDK_HOME or ANDROID_HOME (with NDK installed)." >&2
  exit 1
fi

case "$(uname -s)" in
  Darwin) host=darwin-x86_64 ;;
  Linux) host=linux-x86_64 ;;
  *) echo "Unsupported host OS." >&2; exit 1 ;;
esac

BIN="$NDK/toolchains/llvm/prebuilt/$host/bin"
export CC="$BIN/${TARGET}34-clang"
export AR="$BIN/llvm-ar"
export RANLIB="$BIN/llvm-ranlib"

if [[ ! -d "$SRC/.git" ]]; then
  echo "Cloning OpenBLAS into $SRC"
  git clone --depth 1 --branch v0.3.28 https://github.com/xianyi/OpenBLAS.git "$SRC"
fi

echo "Building OpenBLAS for $TARGET (prefix: $PREFIX)"
make -C "$SRC" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" clean
make -C "$SRC" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
  TARGET=ARMV8 \
  BINARY=64 \
  CC="$CC" \
  AR="$AR" \
  RANLIB="$RANLIB" \
  HOSTCC=cc \
  NOFORTRAN=1 \
  NO_SHARED=1 \
  USE_THREAD=0 \
  CFLAGS="-O2 -fno-sanitize=undefined"

rm -rf "$PREFIX"
mkdir -p "$PREFIX/lib"
cp "$SRC"/libopenblas.a "$PREFIX/lib/"
echo "Installed $PREFIX/lib/libopenblas.a"
