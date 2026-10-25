# rlx-js examples

Run any of them with the `rlx-js` CLI (it allows file access; `scriptArgs`
holds the arguments after the script):

```bash
cargo run -p rlx-js --bin rlx-js -- crates/bindings/rlx-js/examples/inference.js
just js crates/bindings/rlx-js/examples/training.js        # same, through just
FEATURES=metal just js crates/bindings/rlx-js/examples/backends.js
```

## The runtime, end to end

The same six examples exist for Python and Swift, printing the same numbers
([`crates/bindings/README.md`](../../README.md)); `tests/examples.rs` runs these.

| Script | What it shows |
|--------|---------------|
| [`inference.js`](inference.js) | build a classifier, compile for a device (`inference.js metal`), set weights, run |
| [`training.js`](training.js) | `Trainer.run` with an LR schedule and early stop; GGUF checkpoint + resume; freezing; `grad` + `Optimizer` by hand |
| [`loading.js`](loading.js) | write / read GGUF (f32 + Q8_0), load safetensors into a graph, quantize; `RLX_TOKENIZER=…` adds the tokenizer |
| [`distributed.js`](distributed.js) | one rank per process: collectives, pipelined inference, data-parallel training (usage in the file) |
| [`backends.js`](backends.js) | devices, cost-model pick, `FlexibleSession`, `Runner` benchmark / fallback chain, `Router` |
| [`models.js`](models.js) | save a graph as an `.rlxp` model, run it by name with `rlx.Model`; generate with a plugin's language model (`RLX_MODEL_PLUGINS`) |

## More

| Script | What it shows |
|--------|---------------|
| [`hello.js`](hello.js) | the smallest graph |
| [`train.js`](train.js) | gradient descent with the update written in JS |
| [`train_classifier.js`](train_classifier.js) | a classifier trained with the `Trainer` |
| [`mnist.js`](mnist.js) | MNIST MLP / CNN with checkpoints and early stopping |
| [`mnist_lora.js`](mnist_lora.js) | LoRA fine-tuning on MNIST |
| [`transformer_block.js`](transformer_block.js) / [`transformer_block_dsl.js`](transformer_block_dsl.js) | a transformer block, with the graph API and the `Tensor` DSL |
| [`gguf.js`](gguf.js) | inspect a GGUF checkpoint |

License: MIT OR Apache-2.0.
