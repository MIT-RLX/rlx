# rlx-cortexm-trainer

Native fp32 trainer for the [`rlx-cortexm`](https://crates.io/crates/rlx-cortexm)
TinyConv-MNIST demo. Replaces the legacy PyTorch `tools/train_mnist.py`:
builds the same TinyConv architecture as an `rlx_ir::Graph`, derives a
gradient graph via `rlx_opt::autodiff::grad_with_loss`, runs SGD
through the `rlx_cpu` executor, then quantizes the trained weights to
INT8 and emits `src/model_weights.rs` for the firmware to consume.

## Run

```sh
cargo run -p rlx-cortexm-trainer --release -- \
    --epochs 2 --batch 128 \
    --data ~/.cache/torchvision-mnist/MNIST/raw \
    --out  rlx-cortexm/src/model_weights.rs
```

Reaches ~97 % test accuracy in ~50 s on a 2-epoch run.

## Flags

```
--epochs N            Number of training epochs (default: 2)
--batch N             Mini-batch size (default: 128)
--lr F                Learning rate (default: 0.05)
--momentum F          SGD momentum (default: 0.9)
--data PATH           Directory containing MNIST IDX files
--out PATH            Output path for model_weights.rs
--seed N              RNG seed for weight init + shuffling (default: 0)
--train-limit N       Use only the first N training images per epoch (default: 0 = all)
--eval-limit N        Evaluate on N test images (default: 0 = all 10,000)
--val-set N           Write N test images to tests/data/test_set.bin for the
                      firmware bulk-validation test (default: 500)
--weight-bits N       8 (default), 4 (nibble-packed), or 2 (ternary)
--qat MODE            Quantization-aware training: on/off/auto
                      (default: auto = on when --weight-bits < 8)
```

## Execution modes

| env | effect |
|---|---|
| `RLX_FAST_CONV=0` | reference scalar conv instead of im2col+BLAS (default on) |
| `RLX_GRAPH_FUSED=1` | fold the SGD+momentum update into the graph; run fwd+bwd+update as one fused schedule |
| `RLX_REGIONS=1` | with `RLX_GRAPH_FUSED`, keep scalar elementwise chains fused for the region interpreter |

## Status

Binary-only crate — not for crates.io. Distributed via the workspace
git tree.

## License

MIT OR Apache-2.0.