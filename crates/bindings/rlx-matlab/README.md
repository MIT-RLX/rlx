# rlx for MATLAB and GNU Octave

MATLAB / Octave bindings for [rlx](../../../README.md) over its C ABI
([`rlx-capi`](../rlx-capi)): graphs, sessions, device selection, autodiff,
training, weights, text and distributed nodes. One MEX gateway (`rlx_mex`,
from `src/`) reaches every C function by name; the `+rlx` package wraps them in
handle classes and functions. `rlx-bindgen` generates both from rlx-bind's
tables (`src/rlx_mex.c` and every `+rlx/*.m` except `+rlx/+internal`,
`flat.m`, `unflat.m`, `NONE.m`).

```matlab
g = rlx.Graph('name', 'mlp');
x = g.input('x', [1 4]);
w = g.param('w', [4 4]);
g.set_outputs(g.matmul(x, w));
m = rlx.Session('device', 'cpu').compile(g);   % compile consumes g
m.set_param('w', rlx.flat(eye(4)));
y = m.run(struct('x', [1 2 3 4]));              % y{1}: a single row vector
```

## Build

Build the library, then the gateway (from the repository root):

```bash
cargo build -p rlx-capi
cd crates/bindings/rlx-matlab
mkoctfile --mex -I../rlx-capi/include src/rlx_mex.c src/rlx_mex_support.c \
    -L../../../target/debug -lrlx_capi -Wl,-rpath,$PWD/../../../target/debug -o rlx_mex.mex
octave --eval "addpath('test'); test_rlx()"
octave --eval "addpath('examples'); example_inference()"
```

In MATLAB the same sources build with
`mex -R2018a -I../rlx-capi/include src/rlx_mex.c src/rlx_mex_support.c -L../../../target/debug -lrlx_capi`.
Put this directory (it holds `+rlx` and the built `rlx_mex`) on the path with `addpath`.

## Conventions

| | |
|---|---|
| Handles | `rlx.Graph`, `rlx.Session`, `rlx.Compiled`, … — handle classes, freed when the last reference goes (or `delete(obj)`) |
| Constructors / statics | `rlx.Graph('name', 'g')`, `rlx.Gguf.open(path)`, `rlx.Group.connect(rank, world, 'options', …)` |
| Free functions | `rlx.grad(g, wrt)`, `rlx.devices()`, `rlx.quantize(values, scheme)` |
| Optional parameters | name-value pairs or one struct, with the tables' defaults (`g.softmax(x, 'axis', 1)`) |
| Tensors | flat **row-major** vectors (`rlx.flat(A)` / `rlx.unflat(v, shape)` convert MATLAB's column-major arrays); results are `single` |
| Named tensors | a struct, or a `containers.Map` when names are not identifiers (`'proj.weight'`); tensor maps come back as `containers.Map` |
| Errors | `error` with identifier `rlx:<kind>` — `rlx:type`, `rlx:range`, `rlx:runtime`, `rlx:panic`, `rlx:not_found` |
| Options in, reports out | structs / cell arrays (lists) → JSON; reports come back through `jsondecode` |
| Renamed | `eq`, `ne`, `lt`, `gt`, `le`, `ge`, `transpose` → `eq_`, … (as methods they would overload `==`, `.'`) |
| Hand-written | `Optimizer.step` (returns the updated params; a `containers.Map` is updated in place), `Trainer.run` (function handles), in `+rlx/+internal` |

Names: snake_case, as in the Python column of [`docs/bindings-api.md`](../../../docs/bindings-api.md).
A mesh rank is a process (`octave --eval` / `matlab -batch`).

## Examples

Functions, so each takes its arguments from the command line's `--eval` / `-batch`:

| | |
|---|---|
| [`example_inference.m`](examples/example_inference.m) | compile a classifier for a device, set weights, run |
| [`example_training.m`](examples/example_training.m) | `Trainer.run` with an LR schedule and early stop; checkpoint + resume; freezing; `grad` + `Optimizer` by hand |
| [`example_loading.m`](examples/example_loading.m) | GGUF write / read, safetensors into a graph, quantize |
| [`example_distributed.m`](examples/example_distributed.m) | one process per rank: collectives, pipelined inference, data-parallel training |
| [`example_backends.m`](examples/example_backends.m) | devices, cost-model pick, `FlexibleSession`, `Runner`, `Router` |
| [`example_models.m`](examples/example_models.m) | save a graph as an `.rlxp` model and run it by name; generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

`cargo test -p rlx-capi --test ffi_languages` builds the gateway and runs the tests and examples under Octave.

License: MIT OR Apache-2.0.
