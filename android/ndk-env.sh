#!/usr/bin/env bash
# Print export lines for the aarch64-linux-android NDK toolchain.
# Usage: eval "$(android/ndk-env.sh)"
set -euo pipefail

TARGET=aarch64-linux-android
# cc-rs: CC_aarch64_linux_android (lowercase, hyphens → underscores)
TARGET_CC_ENV="${TARGET//-/_}"
# cargo: CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER (uppercase)
TARGET_CARGO_ENV="$(printf '%s' "$TARGET_CC_ENV" | tr '[:lower:]' '[:upper:]')"

default_android_home() {
  if [[ -d /opt/homebrew/share/android-commandlinetools ]]; then
    echo /opt/homebrew/share/android-commandlinetools
  elif [[ -d "${HOME}/Library/Android/sdk" ]]; then
    echo "${HOME}/Library/Android/sdk"
  fi
}

find_ndk() {
  if [[ -n "${ANDROID_NDK_HOME:-}" && -d "$ANDROID_NDK_HOME" ]]; then
    echo "$ANDROID_NDK_HOME"
    return
  fi
  local home="${ANDROID_HOME:-$(default_android_home)}"
  if [[ -n "$home" && -d "$home/ndk" ]]; then
    local candidate
    candidate="$(find "$home/ndk" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -V | tail -1)"
    if [[ -n "$candidate" && -d "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  fi
  echo "Set ANDROID_NDK_HOME or ANDROID_HOME (with an installed NDK)." >&2
  return 1
}

NDK="$(find_ndk)" || exit 1
case "$(uname -s)" in
  Darwin) host=darwin-x86_64 ;;
  Linux) host=linux-x86_64 ;;
  *) echo "Unsupported host OS for NDK toolchain detection." >&2; exit 1 ;;
esac
BIN="$NDK/toolchains/llvm/prebuilt/$host/bin"
CLANG="$BIN/${TARGET}34-clang"
if [[ ! -x "$CLANG" ]]; then
  CLANG="$(ls "$BIN"/${TARGET}*-clang 2>/dev/null | grep -v '\+\+' | sort -V | tail -1 || true)"
fi
if [[ -z "$CLANG" || ! -x "$CLANG" ]]; then
  echo "No ${TARGET}*-clang under $BIN" >&2
  exit 1
fi

printf 'export ANDROID_NDK_HOME=%q\n' "$NDK"
printf 'export CC_%s=%q\n' "$TARGET_CC_ENV" "$CLANG"
printf 'export AR_%s=%q\n' "$TARGET_CC_ENV" "$BIN/llvm-ar"
printf 'export CARGO_TARGET_%s_LINKER=%q\n' "$TARGET_CARGO_ENV" "$CLANG"
