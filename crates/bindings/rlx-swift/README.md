# RLX for Swift

Swift bindings for rlx over its C ABI ([`rlx-capi`](../rlx-capi)).

```swift
import RLX

let g = try Graph("mse")
let x = try g.input("x", shape: [1, 2])
let w = try g.param("w", shape: [2, 2])
let t = try g.input("t", shape: [1, 2])
let d = try g.sub(g.matmul(x, rhs: w), b: t)
try g.setOutputs([try g.mean(g.mul(d, b: d), axes: [0, 1])])

let trainer = try Trainer(g, wrt: ["w"], initial: ["w": [0, 0, 0, 0]],
                          options: ["optimizer": ["kind": "adamw", "lr": 0.1]])
print(try trainer.params()["w"]!)
try trainer.run(steps: 200, batch: { _ in ["x": [1, 1], "t": [1, -1]] })
```

* `Sources/RLX/GraphOps.swift` — every graph builder, **generated** by
  `rlx-bindgen` from rlx-bind's op table (`just gen-bindings`). The first
  argument is unlabeled, the rest labeled with the table's defaults; a row's
  options are trailing labeled parameters (`conv2d(x, weight: k, kernelSize:
  [3, 3], stride: [2, 2])`).
* `Sources/RLX/API.swift` — every runtime class and function (`Session`,
  `FlexibleSession`, `Compiled`, `Runner`, `Router`, `grad` / `jvp` / …,
  `check`, devices, `Gguf` / `GgufWriter`, weights and RLXP, `Optimizer`,
  `Trainer`, `Tokenizer`, `Sampler`, `Group` for distributed nodes),
  **generated** from rlx-bind's runtime API table (`api.idl`) after checking
  each row against the C function it calls. One Swift member per row, named
  as in [`docs/bindings-api.md`](../../../docs/bindings-api.md).
* `Sources/RLX/Manual.swift` — the two rows that take Swift closures or update
  arrays in place: `Trainer.run(steps:batch:lr:every:onStep:)` and
  `Optimizer.step(_:grads:)`.
* `Sources/RLX/Core.swift` — `Graph`, `Node`, `RLXError` (`.kind` is the
  `RLX_ERR_*` status: `.type`, `.range`, `.runtime`, `.panic`, `.notFound`),
  `TypedTensor`, and the bridging helpers.

Every C function in `rlx.h` has a Swift wrapper; `cargo test -p rlx-capi`
fails when one is added without one.

## Examples

Six runnable examples — inference, training, loading, distributed, backends,
models — in [`Examples/`](Examples/README.md), the same six as Python and JS:

```bash
swift run --package-path crates/bindings/rlx-swift rlx-example-distributed
```

## Building

```bash
cargo build -p rlx-capi                  # → target/debug/librlx_capi.{dylib,a}
swift run --package-path crates/bindings/rlx-swift rlx-example-inference
```

`RLX_CAPI_LIB_DIR` points the package at another build of the library
(`target/release`, say). Rows behind an rlx-capi feature off by default
(`export_fpga`, `convert_to_gguf`) compile in when `RLX_CAPI_FEATURES` names it
(`RLX_CAPI_FEATURES=fpga,gguf-convert`, for a library built with those). `cargo test -p rlx-capi` runs the examples and the
XCTest suite (`Tests/RLXTests`, also `swift test --package-path
crates/bindings/rlx-swift`).

## In an iOS / macOS app

```bash
ios/build-xcframework.sh --capi --apple   # → crates/bindings/rlx-swift/build/RLX.xcframework
RLX_XCFRAMEWORK=1 swift test --package-path crates/bindings/rlx-swift
```

The XCFramework holds the static library for iOS devices, the iOS simulator
and macOS (`--platforms "ios macos"` is the default; `--apple` adds the Metal
backend). With `RLX_XCFRAMEWORK=1` the package's `CRLX` module is that binary
target and links the system frameworks rlx needs (Accelerate, Metal, MPS,
MPSGraph), so an Xcode project adds the package as a local dependency and
links `RLX`; no dylib to embed.

License: MIT OR Apache-2.0.
