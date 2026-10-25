# rlx for Haskell

Haskell bindings for [rlx](../../../README.md) over its C ABI
([`rlx-capi`](../rlx-capi)): graphs, sessions, device selection, autodiff,
training, weights, text and distributed nodes. `rlx-bindgen` generates
`src/RLX/Generated.hs` (every `foreign import ccall` and its wrapper) from
rlx-bind's tables; `RLX.Support` and `RLX.Manual` are hand-written.

```haskell
import qualified RLX as R

main :: IO ()
main = do
  g <- R.newGraphWith R.defNewGraph {R.newGraphName = "mlp"}
  x <- R.input g "x" [1, 4]
  w <- R.param g "w" [4, 4]
  R.matmul g x w >>= \y -> R.setOutputs g [y]
  m <- R.newSession >>= \s -> R.sessionCompile s g   -- compile consumes g
  R.compiledSetParam m "w" [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  R.compiledRun m [("x", [1, 2, 3, 4])] >>= print
```

## Build

```bash
cargo build -p rlx-capi
cd crates/bindings/rlx-haskell
cabal build --extra-lib-dirs=$PWD/../../../target/debug
LD_LIBRARY_PATH=$PWD/../../../target/debug cabal run test-rlx --extra-lib-dirs=$PWD/../../../target/debug
LD_LIBRARY_PATH=$PWD/../../../target/debug cabal run rlx-example --extra-lib-dirs=$PWD/../../../target/debug -- inference
```

`DYLD_LIBRARY_PATH` on macOS. Functions behind a feature the default library
lacks need `--ghc-options=-DRLX_FEATURE_<NAME>` (e.g. `RLX_FEATURE_FPGA`).
`RLX.Generated` is large; `--ghc-options=-O0` builds it much faster.

## Conventions

| | |
|---|---|
| Import | qualified (`import qualified RLX as R`): many op names match Prelude's (`sum`, `exp`, `reverse`) |
| Handles | `Graph`, `Session`, `Compiled`, … — `ForeignPtr`s finalized by the C free function |
| Names | the graph's ops take the graph first and carry no prefix (`R.matmul g a b`); other members carry the class (`R.compiledRun`, `R.ggufOpen`, `R.groupConnect`); constructors `R.newGraph` |
| Optional parameters | `f` uses the defaults; `fWith` takes a record whose `def…` value holds them (`R.softmaxWith R.defSoftmax {R.softmaxAxis = 1} g x`) |
| Tensors | `[Float]`; named tensors `[(String, [Float])]` |
| Errors | `RlxError` exceptions (`RlxError ErrRange message`) in `IO` |
| Options in, reports out | aeson `Value`s; typed reports as lists |
| Several results | tuples (`(device, outputs) <- R.runnerRun r x`) |
| Blocking | calls are `safe`, so other Haskell threads run meanwhile |
| Hand-written | `optimizerStep` (returns the updated params), `trainerRun` (IO callbacks), in `RLX.Manual` |

Names: camelCase, as in the JS column of [`docs/bindings-api.md`](../../../docs/bindings-api.md).
A mesh rank is a process (`rlx-example distributed rank world peers`).

## Examples

One executable, `rlx-example`, one argument per example:

| | |
|---|---|
| [`Inference.hs`](examples/Inference.hs) | compile a classifier for a device, set weights, run |
| [`Training.hs`](examples/Training.hs) | `trainerRun` with an LR schedule and early stop; checkpoint + resume; freezing; `grad` + `optimizerStep` by hand |
| [`Loading.hs`](examples/Loading.hs) | GGUF write / read, safetensors into a graph, quantize |
| [`Distributed.hs`](examples/Distributed.hs) | one process per rank: collectives, pipelined inference, data-parallel training |
| [`Backends.hs`](examples/Backends.hs) | devices, cost-model pick, `FlexibleSession`, `Runner`, `Router` |
| [`Models.hs`](examples/Models.hs) | save a graph as an `.rlxp` model and run it by name; generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

`cargo test -p rlx-capi --test ffi_languages` runs the tests and examples (with
`RLX_TEST_DOCKER=1`, in the `haskell:9` image when `cabal` is not installed).

License: MIT OR Apache-2.0.
