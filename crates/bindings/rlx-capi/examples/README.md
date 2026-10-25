# C examples

```bash
cargo build -p rlx-capi
cc crates/bindings/rlx-capi/examples/train_linear.c \
   -I crates/bindings/rlx-capi/include -L target/debug -lrlx_capi \
   -Wl,-rpath,target/debug -o train_linear && ./train_linear
```

| File | What it shows |
|------|---------------|
| [`train_linear.c`](train_linear.c) | build a graph with the generated `rlx_graph_*` builders, compile and run it, differentiate it, and train it with `RlxTrainer` |
| [`compose.c`](compose.c) | splice two layers into one graph, split it in two, run the parts in order (the boundary tensor passed by name), merge them back, and check the error path |
| [`models.c`](models.c) | save a graph as an `.rlxp` model, open and run it with `rlx_model_*`, then generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

Every call returns `RLX_OK` or an `RLX_ERR_*` status, with the message in
`rlx_last_error()`; see [`include/rlx.h`](../include/rlx.h). The higher-level
examples (inference, training, loading, distributed, backends) are in the
bindings built on this ABI — [Swift](../../rlx-swift/Examples/README.md) is the
closest to C. `cargo test -p rlx-capi` compiles and runs all three.

For microcontrollers (bare-metal Cortex-M, no OS or heap), see
[`rlx-micro`](../../rlx-micro/README.md). Its C and C++ examples run on the
host and as firmware under QEMU.

License: MIT OR Apache-2.0.
