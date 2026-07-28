# RLX Android demo

Minimal NDK + JNI sample: a Kotlin app loads `librlx_jni.so`, compiles a
tiny RLX graph (`matmul → bias → GELU`), and runs it on **CPU (NEON)** or
**GPU (Vulkan / wgpu)** depending on what the device exposes.

Layout:

```text
android/
  build.sh              # cross-build .so → app/src/main/jniLibs/arm64-v8a/
  rlx-jni/              # Rust cdylib (standalone workspace)
  app/                  # Gradle application (Kotlin)
```

## Prerequisites

- Rust stable + `aarch64-linux-android` target
- Android NDK (r26+; r27 recommended) via Android Studio or the SDK Manager
- Android Studio Ladybug (2024.2+) or Gradle 8.7+ for the APK build
- Physical **arm64** device or emulator with API 26+ (Vulkan optional)

Set one of:

```bash
export ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/27.0.12077973   # macOS
export ANDROID_HOME=$HOME/Library/Android/sdk
```

## Build the native library

From the repo root:

```bash
./android/build.sh
# or
just android-build
```

This writes `android/app/src/main/jniLibs/arm64-v8a/librlx_jni.so`.

## Build and install the APK

1. Open the `android/` folder in Android Studio.
2. Let Gradle sync (Android Studio creates the wrapper if missing).
3. **Run** on a connected device or emulator.

CLI (after `./gradlew wrapper` once inside `android/`):

```bash
cd android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Tap **Run inference** in the app. You should see the active backend and a
two-element float vector (GELU outputs).

## What the JNI layer does

| JNI method | Rust | Purpose |
|------------|------|---------|
| `runInference()` | tiny `matmul → bias → GELU` | Returns `[f32; 2]` output |
| `backendName()` | `pick_device()` → `Device::Cpu` or `Device::Gpu` | Label for the UI |
| `runMnist()` / `mnistPredict()` | Embedded MLP `784→32→10` | Logits / argmax for a bundled MNIST digit |
| `mnistExpectedLabel()` | Sample ground truth | For instrumented tests |

Regenerate the embedded weights (needs local MNIST IDX files):

```bash
cargo run --manifest-path android/rlx-jni/Cargo.toml --example gen_mnist_assets --release
```

The graph is a small 1×4 × 4×2 matmul with identity-ish weights — fast to
compile on-device.

## BLAS on Android

Unlike iOS (Accelerate) or desktop Linux (OpenBLAS), **the Android NDK does
not ship a system CBLAS/LAPACK**. Default builds use:

| Layer | What runs |
|-------|-----------|
| **CPU** | NEON kernels + portable GEMM (`rlx-cpu` skips OpenBLAS on aarch64 unless `OPENBLAS_LIB_DIR` is set) |
| **GPU** | wgpu → Vulkan when an adapter is available |

The demo enables `rlx-runtime/android` (`cpu` + `gpu`).

**Optional OpenBLAS:** cross-compile CBLAS for arm64 and link it statically into
`librlx_jni.so`:

```bash
./android/build-openblas.sh    # once — produces third_party/openblas-android/lib/libopenblas.a
./android/build.sh --blas      # rlx-jni with rlx-runtime/android-blas
cd android && ./gradlew assembleDebug
```

Without `--blas`, `./build.sh` uses portable GEMM (no CBLAS).

## Host-side check

The JNI crate includes a CPU unit test (no Android required):

```bash
cargo test --manifest-path android/rlx-jni/Cargo.toml
```

Cross-compile gate (needs NDK linker for a full link; `cargo check` is enough
for IR/runtime):

```bash
just android-check
```

End-to-end on an emulator (build `.so`, APK, instrumented tests):

```bash
just android-e2e
# or
./android/e2e.sh
```

## Notes

- **Vulkan:** `Device::Gpu` is selected when wgpu finds a Vulkan adapter.
  Emulators may fall back to CPU only.
- **Next steps:** load a `.gguf`, wire `ExpertPool`, or expose `Session` from
  Kotlin via additional JNI methods.

## License

MIT OR Apache-2.0 — same as the RLX workspace.
