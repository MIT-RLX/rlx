# rlx for Kotlin (JVM, Android)

Kotlin bindings for [rlx](../../../README.md) over its C ABI
([`rlx-capi`](../rlx-capi)) through JNI, so they run on Android as well as any
JVM: graphs, sessions, device selection, autodiff, training, weights, text and
distributed nodes. `rlx-bindgen` generates both halves from rlx-bind's tables —
the JNI glue (`src/main/c/rlx_jni.c`) and the Kotlin classes
(`src/main/kotlin/org/rlx/kt/RLX.kt`).

```kotlin
import org.rlx.kt.*

val g = Graph("mlp")
val x = g.input("x", longArrayOf(1, 4))
val w = g.param("w", longArrayOf(4, 4))
g.setOutputs(intArrayOf(g.matmul(x, w)))
Session(device = "cpu").compile(g).use { m ->         // compile consumes g
    m.setParam("w", floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f))
    println(m.run(mapOf("x" to floatArrayOf(1f, 2f, 3f, 4f)))[0].toList())
}
```

## Build

`librlx_jni` (the glue) links `librlx_capi`. With CMake — the same file serves
the desktop and Android's `externalNativeBuild`:

```bash
cargo build -p rlx-capi
cmake -S crates/bindings/rlx-kotlin/src/main/c -B build/rlx-jni -DRLX_CAPI_LIB_DIR=$PWD/target/debug
cmake --build build/rlx-jni
cd crates/bindings/rlx-kotlin
gradle rlxTest -Prlx.jni.lib=<path to librlx_jni>
gradle example -Pname=inference -Prlx.jni.lib=<path to librlx_jni>
```

On Android, build `librlx_capi.so` per ABI (`cargo ndk -t arm64-v8a build -p
rlx-capi`), point `RLX_CAPI_LIB_DIR` at it and package both libraries in
`jniLibs`; the classes load `rlx_jni` with `System.loadLibrary`. Elsewhere,
`-Drlx.jni.lib=<path>` names the library.

## Conventions

| | |
|---|---|
| Handles | `Graph`, `Session`, `Compiled`, … — `AutoCloseable` (`use { }`), with a finalizer as a backstop |
| Free functions / statics | `grad(g, wrt)`, `devices()`; `Gguf.open(path)`, `Group.connect(rank, world, …)` |
| Optional parameters | named arguments with the tables' defaults (`g.softmax(x, axis = 1)`) |
| Tensors | `FloatArray`; shapes `LongArray`; nodes `Int` |
| Errors | `RlxException` with `kind` (`TYPE`, `RANGE`, `RUNTIME`, `PANIC`, `NOT_FOUND`) |
| Options in, reports out | JSON strings |
| Hand-written | `Optimizer.step` (updates your arrays in place), `Trainer.run` (lambdas), in `Support.kt` |

Names: camelCase, as in the JS column of [`docs/bindings-api.md`](../../../docs/bindings-api.md).

## Examples

| | |
|---|---|
| [`Inference.kt`](examples/Inference.kt) | compile a classifier for a device, set weights, run |
| [`Training.kt`](examples/Training.kt) | `Trainer.run` with an LR schedule and early stop; checkpoint + resume; freezing; `grad` + `Optimizer` by hand |
| [`Loading.kt`](examples/Loading.kt) | GGUF write / read, safetensors into a graph, quantize |
| [`Distributed.kt`](examples/Distributed.kt) | three ranks on threads: collectives, pipelined inference, data-parallel training |
| [`Backends.kt`](examples/Backends.kt) | devices, cost-model pick, `FlexibleSession`, `Runner`, `Router` |
| [`Models.kt`](examples/Models.kt) | save a graph as an `.rlxp` model and run it by name; generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

`cargo test -p rlx-capi --test ffi_languages` builds the glue and runs the tests and examples.

License: MIT OR Apache-2.0.
