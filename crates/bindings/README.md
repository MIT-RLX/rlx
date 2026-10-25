# RLX in your language

One API, every language. Each binding below exposes the same graph builders
and the same runtime — sessions, devices, autodiff, training, weights, text and
distributed nodes — generated from one source in [`rlx-bind`](rlx-bind), and
tested one-to-one against it (see [How the bindings stay one-to-one](#how-the-bindings-stay-one-to-one)).

| Language | Package | Examples | Build / run |
|----------|---------|----------|-------------|
| **Python** | [`pyrlx`](pyrlx) | [`pyrlx/examples/`](pyrlx/examples/README.md) | `maturin develop --features cpu` · `python examples/inference.py` |
| **JavaScript** (embedded QuickJS) | [`rlx-js`](rlx-js) | [`rlx-js/examples/`](rlx-js/examples/README.md) | `cargo run -p rlx-js --bin rlx-js -- examples/inference.js` |
| **Swift** (iOS, macOS) | [`rlx-swift`](rlx-swift) | [`rlx-swift/Examples/`](rlx-swift/Examples/README.md) | `cargo build -p rlx-capi` · `swift run --package-path crates/bindings/rlx-swift rlx-example-inference` |
| **Go** | [`rlx-go`](rlx-go/README.md) | [`rlx-go/examples/`](rlx-go/examples) | `go test ./...` · `go run ./examples/inference` |
| **Java** (FFM, 22+) | [`rlx-java`](rlx-java/README.md) | [`rlx-java/examples/`](rlx-java/examples/org/rlx/examples) | `javac` + `java --enable-native-access=ALL-UNNAMED` · `mvn test` |
| **C#** / .NET | [`rlx-csharp`](rlx-csharp/README.md) | [`rlx-csharp/examples/`](rlx-csharp/examples/Rlx.Examples) | `dotnet run --project examples/Rlx.Examples -- inference` |
| **Ruby** | [`rlx-ruby`](rlx-ruby/README.md) | [`rlx-ruby/examples/`](rlx-ruby/examples) | `ruby -Ilib examples/inference.rb` |
| **Julia** | [`rlx-julia`](rlx-julia/README.md) | [`rlx-julia/examples/`](rlx-julia/examples) | `julia --project=. examples/inference.jl` |
| **Zig** | [`rlx-zig`](rlx-zig/README.md) | [`rlx-zig/examples/`](rlx-zig/examples) | `zig build test` · `zig build inference` |
| **C++** | [`rlx-cpp`](rlx-cpp/README.md) | [`rlx-cpp/examples/`](rlx-cpp/examples) | `cmake -B build -DRLX_CAPI_LIB_DIR=…` · `build/inference` |
| **Node.js / TypeScript** | [`rlx-node`](rlx-node/README.md) | [`rlx-node/examples/`](rlx-node/examples) | `npm install` · `npm test` · `node examples/inference.js` |
| **Kotlin (JVM, Android)** | [`rlx-kotlin`](rlx-kotlin/README.md) | [`rlx-kotlin/examples/`](rlx-kotlin/examples) | CMake the JNI glue · `gradle example -Pname=inference` |
| **Dart / Flutter** | [`rlx-dart`](rlx-dart/README.md) | [`rlx-dart/example/`](rlx-dart/example) | `dart test` · `dart run example/inference.dart` |
| **Lua** | [`rlx-lua`](rlx-lua/README.md) | [`rlx-lua/examples/`](rlx-lua/examples) | build `rlx_core` · `lua examples/inference.lua` |
| **MATLAB / Octave** | [`rlx-matlab`](rlx-matlab/README.md) | [`rlx-matlab/examples/`](rlx-matlab/examples) | `mkoctfile --mex …` · `example_inference()` |
| **R** | [`rlx-r`](rlx-r/README.md) | [`rlx-r/inst/examples/`](rlx-r/inst/examples) | `R CMD INSTALL` · `Rscript inst/examples/inference.R` |
| **PHP** | [`rlx-php`](rlx-php/README.md) | [`rlx-php/examples/`](rlx-php/examples) | `php -d ffi.enable=1 examples/inference.php` |
| **Nim** | [`rlx-nim`](rlx-nim/README.md) | [`rlx-nim/examples/`](rlx-nim/examples) | `nim r --path:src examples/inference.nim` |
| **D** | [`rlx-d`](rlx-d/README.md) | [`rlx-d/examples/`](rlx-d/examples) | `ldc2 -i -Isource examples/inference.d -L-lrlx_capi` |
| **V** | [`rlx-v`](rlx-v/README.md) | [`rlx-v/examples/`](rlx-v/examples) | `v run examples/inference.v` |
| **Haskell** | [`rlx-haskell`](rlx-haskell/README.md) | [`rlx-haskell/examples/`](rlx-haskell/examples) | `cabal run rlx-example -- inference` |
| **OCaml** | [`rlx-ocaml`](rlx-ocaml/README.md) | [`rlx-ocaml/examples/`](rlx-ocaml/examples) | `dune exec examples/inference.exe` |
| **Elixir** | [`rlx-elixir`](rlx-elixir/README.md) | [`rlx-elixir/examples/`](rlx-elixir/examples) | `make` · `mix run examples/inference.exs` |
| **Odin** | [`rlx-odin`](rlx-odin/README.md) | [`rlx-odin/examples/`](rlx-odin/examples) | `odin run examples/inference -collection:rlx=.` |
| **C** / any FFI language | [`rlx-capi`](rlx-capi) (`include/rlx.h`) | [`rlx-capi/examples/`](rlx-capi/examples/README.md) | `cargo build -p rlx-capi` · link `librlx_capi` |
| **C / C++ on microcontrollers** (bare-metal Cortex-M) | [`rlx-micro`](rlx-micro/README.md) (`rlx_micro.h`, `rlx/micro.hpp`) | [`rlx-micro/examples/`](rlx-micro/examples) | `cargo build -p rlx-micro --target thumbv7em-none-eabihf` · link `librlx_micro.a` |
| **Browser** (wasm) | [`rlx-web`](rlx-web) | [`rlx-web/web/`](rlx-web/README.md) | `just build-web` · `just serve-web` |
| **Rust** | [`rlx`](../rlx) | [`docs/`](../../docs/README.md) | `cargo add rlx --features cpu` |

`rlx-micro` is the microcontroller tier: INT8 kernels and fixed models, with
no OS or heap (~25 KB flash). It sits outside the generated API below.
Embedded Linux boards use `rlx-capi` itself, cross-built for their target.

Mobile and embedded nodes that only join a distributed mesh as workers use the
smaller [`rlx-ffi`](rlx-ffi) (C ABI of the node alone; `ios/`, `android/`).

## The same six examples everywhere

Every language above ships the same six runnable examples, printing the same
numbers — each binding's tests run them and check those lines:

| Example | What it shows |
|---------|---------------|
| `inference` | build a graph, compile it for a device, set weights, run (and a typed run) |
| `training` | `Trainer` with a callback loop, LR schedule and early stop; GGUF checkpoint and resume; freezing; the same fit by hand with `grad` + `Optimizer` |
| `loading` | write and read GGUF (f32 + Q8_0), load safetensors, quantize in memory, tokenizer |
| `distributed` | a three-rank mesh: collectives, pipelined inference (ship a stage per worker), data-parallel training where every replica ends identical |
| `backends` | which devices exist and suit a graph; pick one by name, by cost model, by policy (`FlexibleSession`), by benchmark (`Runner`), or per run with fallback (`Router`) |
| `models` | save a graph as an `.rlxp` model package and run it by name on any device (`Model.open`); then, from the reference [`rlx-model-plugin`](rlx-model-plugin/README.md), each task's method: `chat` with a language model, `synthesize` then `transcribe` (speech out and back in), and `embed` |

```python
import pyrlx, numpy as np                       # Python
g = pyrlx.Graph("mlp")
x = g.input("x", [1, 4])
g.set_outputs([g.relu(g.matmul(x, g.param("w", [4, 4])))])
m = pyrlx.Session("cpu").compile(g)
m.set_param("w", np.eye(4, dtype=np.float32))
print(m.run({"x": np.ones((1, 4), np.float32)}))
```

```js
const g = new rlx.Graph("mlp");                 // JavaScript
const x = g.input("x", [1, 4]);
g.setOutputs([g.relu(g.matmul(x, g.param("w", [4, 4])))]);
const m = new rlx.Session("cpu").compile(g);
m.setParam("w", new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]));
console.log(m.run({ x: new Float32Array(4).fill(1) }));
```

```swift
let g = try Graph("mlp")                         // Swift
let x = try g.input("x", shape: [1, 4])
try g.setOutputs([g.relu(g.matmul(x, rhs: g.param("w", shape: [4, 4])))])
let m = try Session(device: "cpu").compile(g)
try m.setParam("w", data: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
print(try m.run(["x": [1, 1, 1, 1]]))
```

## Packages

Each directory is the root of the package it will be published as: Python
wheels from `pyrlx` (maturin), the Swift package from `rlx-swift` (SwiftPM, or
`RLX.xcframework` from `ios/build-xcframework.sh --capi` for apps), the Go
module `rlx-go`, the Maven artifact `rlx-java` (`pom.xml`), the NuGet package
`rlx-csharp/src/Rlx`, the gem `rlx-ruby` (`rlx.gemspec`), the Julia package
`rlx-julia` (`Project.toml`), the Zig package `rlx-zig` (`build.zig.zon`), the
CMake target `rlx::rlx` (`rlx-cpp`), the npm package `@rlx/rlx` (`rlx-node`),
the Gradle / Android library `rlx-kotlin`, the pub package `rlx-dart`, the Lua
module `rlx-lua`, the MATLAB / Octave toolbox `rlx-matlab`, the R package
`rlx-r`, the composer package `rlx-php`, the nimble package `rlx-nim`, the dub
package `rlx-d`, the V module `rlx-v`, the cabal package `rlx-haskell`, the
opam / dune package `rlx-ocaml`, the hex package `rlx-elixir`, the Odin package
`rlx-odin`, the C library and `rlx.h` from `rlx-capi`, the browser bundle from
`rlx-web`. The
FFI packages load `librlx_capi`, shipped alongside (or `RLX_CAPI_LIB`). Examples
and tests live inside their package, so they ship with it.

Each package is released on its own, with prebuilt `librlx_capi` for macOS
(arm64, x64), Linux (x64, arm64) and Windows (x64) inside, under
`native/<os>-<arch>/`, where its loader looks before `RLX_CAPI_LIB` and the
workspace build. All of them carry the workspace version (`just
sync-binding-versions`); none is published to crates.io. A `v*` tag builds them
all into a draft GitHub Release; [`docs/releasing-bindings.md`](../../docs/releasing-bindings.md)
covers publishing to each registry.

## How the bindings stay one-to-one

* **Graph builders** — one row per op in [`rlx-bind/src/table.rs`](rlx-bind/src/table.rs),
  expanded at compile time into the JS and Python methods and generated into C
  (`rlx_graph_*`) and Swift (`GraphOps.swift`).
* **Runtime** — one row per class member or function in
  [`rlx-bind/src/api.idl`](rlx-bind/src/api.idl), with its kinds and its name in
  every language. `rlx-bindgen` checks every row against the exported C
  function's parameter types (and every C function against a row) and
  generates the Swift classes from the rows; the rlx-js and pyrlx tests check
  every row resolves to an export and every export is a row.
* **Every FFI language** — Swift, Go, Java, C#, Ruby, Julia, Zig, C++, Node,
  Kotlin (JNI), Dart, Lua, MATLAB / Octave (MEX), R, PHP, Nim, D, V, Haskell,
  OCaml, Elixir (NIF), Odin — is generated
  by `rlx-bindgen` from one lowered view of both tables (`rlx-bindgen/src/ir.rs`):
  each graph op and runtime row with typed parameters, checked against the C
  signature rustc will export. An emitter maps about 25 kinds to its language;
  it never lists functions, so a new row reaches every language at once.
* `just gen-bindings` regenerates; `just check-bindings` (in `just ci`) fails on
  drift. The full name map is [`docs/bindings-api.md`](../../docs/bindings-api.md);
  how to add an op, a runtime function or a language is
  [`docs/bindings.md`](../../docs/bindings.md).

## License

MIT OR Apache-2.0.
