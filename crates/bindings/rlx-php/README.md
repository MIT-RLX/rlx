# rlx for PHP

PHP 8.1+ bindings for [rlx](../../../README.md) over its C ABI
([`rlx-capi`](../rlx-capi)) through the FFI extension: graphs, sessions, device
selection, autodiff, training, weights, text and distributed nodes.
`rlx-bindgen` generates `src/Generated.php` and the header PHP's FFI parses
(`src/rlx_php.h`) from rlx-bind's tables; `src/Support.php` and
`src/Manual.php` are hand-written.

```php
require 'src/rlx.php';

$g = new Rlx\Graph(name: 'mlp');
$x = $g->input('x', [1, 4]);
$w = $g->param('w', [4, 4]);
$g->setOutputs([$g->matmul($x, $w)]);
$m = (new Rlx\Session(device: 'cpu'))->compile($g);   // compile consumes $g
$m->setParam('w', [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
print_r($m->run(['x' => [1, 2, 3, 4]]));
```

## Build

No build step: the FFI extension loads `librlx_capi` at the first call.

```bash
cargo build -p rlx-capi
cd crates/bindings/rlx-php
RLX_CAPI_LIB=../../../target/debug/librlx_capi.dylib php -d ffi.enable=1 tests/test_rlx.php
RLX_CAPI_LIB=../../../target/debug/librlx_capi.dylib php -d ffi.enable=1 examples/inference.php
```

The library is `RLX_CAPI_LIB` when set, else `librlx_capi` on the search path.
The CLI needs `ffi.enable=1` (or `preload`); composer's autoload includes
`src/rlx.php` (`composer.json`).

## Conventions

| | |
|---|---|
| Handles | `Rlx\Graph`, `Rlx\Session`, `Rlx\Compiled`, … — freed in `__destruct` (deterministic), or now with `close()` |
| Free functions / statics | `Rlx\grad($g, $wrt)`, `Rlx\devices()`; `Rlx\Gguf::open($path)`, `Rlx\Group::connect($rank, $world, options: …)` |
| Optional parameters | PHP defaults, so named arguments work (`$g->softmax($x, axis: 1)`) |
| Tensors | arrays of numbers; bytes are strings |
| Errors | `Rlx\RlxException` with `kind` (`type`, `range`, `runtime`, `panic`, `not_found`) |
| Options in, reports out | arrays → JSON; reports come back as arrays |
| Features | a function behind a feature the library lacks is declared on first use, so the library still loads |
| Hand-written | `Optimizer::step` (`&$params` updated in place), `Trainer::run` (callables), in `src/Manual.php` |

Names: camelCase, as in the JS column of [`docs/bindings-api.md`](../../../docs/bindings-api.md).
A mesh rank is a process (`php examples/distributed.php rank world peers`).

## Examples

| | |
|---|---|
| [`inference.php`](examples/inference.php) | compile a classifier for a device, set weights, run |
| [`training.php`](examples/training.php) | the Trainer with an LR schedule and early stop; checkpoint + resume; freezing; `grad` + `Optimizer` by hand |
| [`loading.php`](examples/loading.php) | GGUF write / read, safetensors into a graph, quantize |
| [`distributed.php`](examples/distributed.php) | one process per rank: collectives, pipelined inference, data-parallel training |
| [`backends.php`](examples/backends.php) | devices, cost-model pick, `FlexibleSession`, `Runner`, `Router` |
| [`models.php`](examples/models.php) | save a graph as an `.rlxp` model and run it by name; generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

`cargo test -p rlx-capi --test ffi_languages` runs the tests and examples (with
`RLX_TEST_DOCKER=1`, in a container when the toolchain is not installed).

License: MIT OR Apache-2.0.
