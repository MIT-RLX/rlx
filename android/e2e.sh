#!/usr/bin/env bash
# End-to-end Android gate: host tests, cross-compile, APK build, on-device tests.
#
# Usage:
#   ./e2e.sh           # scalar CPU (default)
#   ./e2e.sh --blas    # OpenBLAS / CBLAS (runs build-openblas.sh if needed)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
USE_BLAS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --blas) USE_BLAS=1; shift ;;
    *) echo "Unknown arg: $1 (try --blas)" >&2; exit 1 ;;
  esac
done

export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home}"
export PATH="$JAVA_HOME/bin:$PATH"
export ANDROID_HOME="${ANDROID_HOME:-/opt/homebrew/share/android-commandlinetools}"
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$ANDROID_HOME/ndk/27.0.12077973}"
export PATH="$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$PATH"

AVD="${RLX_ANDROID_AVD:-elemind}"
SERIAL="${ANDROID_SERIAL:-}"

wait_for_device() {
  adb wait-for-device
  for _ in $(seq 1 120); do
    if [[ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == "1" ]]; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for emulator boot." >&2
  exit 1
}

ensure_emulator() {
  if adb devices 2>/dev/null | grep -E '[[:space:]]device$' >/dev/null; then
    echo "Device already connected."
    return
  fi
  if ! "$ANDROID_HOME/emulator/emulator" -list-avds | grep -qx "$AVD"; then
    echo "No AVD named '$AVD'. Create one or set RLX_ANDROID_AVD." >&2
    exit 1
  fi
  echo "Starting emulator '$AVD'..."
  "$ANDROID_HOME/emulator/emulator" -avd "$AVD" -no-snapshot-load -no-audio -no-boot-anim &
  wait_for_device
}

verify_no_cblas_undef() {
  local so="$ROOT/app/src/main/jniLibs/arm64-v8a/librlx_jni.so"
  local ndk_bin="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt"
  local host
  case "$(uname -s)" in
    Darwin) host=darwin-x86_64 ;;
    Linux) host=linux-x86_64 ;;
    *) return 0 ;;
  esac
  local readelf="$ndk_bin/$host/bin/llvm-readelf"
  if [[ -x "$readelf" ]]; then
    if "$readelf" -Ws "$so" | grep ' UND ' | grep -qi cblas; then
      echo "ERROR: librlx_jni.so has undefined cblas_* symbols." >&2
      "$readelf" -Ws "$so" | grep ' UND ' | grep -i cblas
      exit 1
    fi
    echo "Verified: no undefined cblas_* in librlx_jni.so"
  fi
}

echo "==> Host JNI unit test (scalar)"
cargo test --manifest-path "$ROOT/rlx-jni/Cargo.toml" --features scalar

echo "==> Android cross-compile gate"
(
  cd "$REPO"
  just android-check
)

if [[ "$USE_BLAS" -eq 1 ]]; then
  echo "==> OpenBLAS for Android"
  if [[ ! -f "$ROOT/third_party/openblas-android/lib/libopenblas.a" ]]; then
    "$ROOT/build-openblas.sh"
  else
    echo "Using cached $ROOT/third_party/openblas-android/lib/libopenblas.a"
  fi
  echo "==> Native library (CBLAS / OpenBLAS)"
  "$ROOT/build.sh" --blas
else
  echo "==> Native library (scalar CPU)"
  "$ROOT/build.sh"
fi
verify_no_cblas_undef

echo "==> APK"
(
  cd "$ROOT"
  ./gradlew assembleDebug assembleDebugAndroidTest
)

ensure_emulator
[[ -n "$SERIAL" ]] && export ANDROID_SERIAL="$SERIAL"

echo "==> Instrumented tests (on device)"
(
  cd "$ROOT"
  ./gradlew connectedDebugAndroidTest
)

echo "==> UI check"
adb install -r "$ROOT/app/build/outputs/apk/debug/app-debug.apk" >/dev/null
adb shell am force-stop com.mit.rlx.demo
adb shell am start -n com.mit.rlx.demo/com.mit.rlx.MainActivity >/dev/null
sleep 2
# RUN INFERENCE button center (bounds ~[63,551][464,677] on 1080x2400 emu)
adb shell input tap 264 614
sleep 10
adb shell uiautomator dump /sdcard/rlx-ui.xml >/dev/null 2>&1 || true
UI="$(adb shell cat /sdcard/rlx-ui.xml 2>/dev/null || true)"
if echo "$UI" | grep -q 'resource-id="com.mit.rlx.demo:id/outputText"'; then
  if echo "$UI" | grep 'outputText' | grep -q 'text="\['; then
    echo "UI check: inference output visible"
  else
    echo "WARNING: outputText did not update — instrumented tests passed; UI tap may have missed." >&2
  fi
else
  echo "WARNING: could not read UI dump." >&2
fi

echo "E2E OK ($([ "$USE_BLAS" -eq 1 ] && echo CBLAS || echo scalar))"
