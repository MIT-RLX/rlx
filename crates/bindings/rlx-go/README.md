# rlx for Go

Go bindings for [rlx](../../../README.md) over its C ABI ([`rlx-capi`](../rlx-capi)):
graphs, sessions, device selection, autodiff, training, weights, text and
distributed nodes. Every C function is a Go function or method, generated from
rlx-bind's tables (`generated.go`, by `rlx-bindgen`).

```go
import rlx "github.com/MIT-RLX/rlx/crates/bindings/rlx-go"

g, _ := rlx.NewGraph(&rlx.GraphOpts{Name: rlx.Ptr("mlp")})
x, _ := g.Input("x", []int{1, 4}, nil)
w, _ := g.Param("w", []int{4, 4}, nil)
y, _ := g.Matmul(x, w)
g.SetOutputs([]rlx.Node{y})

s, _ := rlx.NewSession(nil)                 // nil options: every default (cpu, f32)
m, _ := s.Compile(g, nil)                   // compile consumes g
defer m.Close()
m.SetParam("w", []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1})
out, _ := m.Run(map[string][]float32{"x": {1, 2, 3, 4}})
```

## Build

```bash
cargo build -p rlx-capi                    # → target/debug/librlx_capi.{dylib,so,a}
cd crates/bindings/rlx-go && go test ./...
go run ./examples/inference
```

cgo links the workspace's `target/debug`; for another build set `CGO_LDFLAGS`
(`-L<dir> -lrlx_capi`) and the runtime library path.

## Conventions

| | |
|---|---|
| Handles | `*Graph`, `*Session`, `*Compiled`, `*Trainer`, … — `Close()` frees now, a finalizer frees otherwise |
| Constructors / statics | `NewSession(opts)`, `NewTrainer(graph, wrt, initial, opts)`; `RunnerFromEnv(g)`, `GroupConnect(rank, world, opts)`, `GgufOpen(path)` |
| Optional parameters | the function's `…Opts` struct; `nil` takes every default, a nil field keeps its default (`rlx.Ptr(v)` for scalars) |
| Errors | every call returns `error`; `*rlx.Error` has `Kind` (`ErrType`, `ErrRange`, `ErrRuntime`, `ErrPanic`, `ErrNotFound`) and `Message` |
| Options in, reports out | `map[string]any` → JSON; reports decode to `map[string]any` / `[]map[string]any` |
| Hand-written | `Optimizer.Step` (updates your slices in place), `Trainer.Run` (Go callbacks) |
| Optional C features | `ExportFpga` / `ConvertToGguf` need `-tags rlx_fpga` / `-tags rlx_gguf_convert` and a library built with that feature |

The full name map is [`docs/bindings-api.md`](../../../docs/bindings-api.md) —
Go spells each row in PascalCase.

## Examples

The same six as every binding, printing the same numbers
([all languages](../README.md)):

| | |
|---|---|
| [`examples/inference`](examples/inference/main.go) | compile a classifier for a device, set weights, run |
| [`examples/training`](examples/training/main.go) | `Trainer.Run` with an LR schedule and early stop; checkpoint + resume; freezing; `Grad` + `Optimizer` by hand |
| [`examples/loading`](examples/loading/main.go) | GGUF write / read (f32 + Q8_0), safetensors into a graph, quantize |
| [`examples/distributed`](examples/distributed/main.go) | three ranks as goroutines: collectives, pipelined inference, data-parallel training |
| [`examples/backends`](examples/backends/main.go) | devices, cost-model pick, `FlexibleSession`, `Runner` benchmark / fallback, `Router` |
| [`examples/models`](examples/models/main.go) | save a graph as an `.rlxp` model and run it by name; generate with a plugin's language model |

`cargo test -p rlx-capi --test ffi_languages` runs the tests and examples of
every FFI binding.

License: MIT OR Apache-2.0.
