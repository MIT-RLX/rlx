# rlx-optim

RLX training-step optimizers. Host-side `f32` step functions for the
optimizer families surveyed in *"A Systematic Review of Optimization
Algorithms for Modern Deep Learning"* (arXiv:[2509.02046v1](https://arxiv.org/abs/2509.02046)).

| Struct       | Family                              | Reference                                           |
|--------------|-------------------------------------|-----------------------------------------------------|
| `Sgd`        | SGD ± momentum / Nesterov           | Polyak '64 / Nesterov '83                           |
| `Adam`       | Adam                                | Kingma & Ba 2014                                    |
| `AdamW`      | AdamW (decoupled decay)             | Loshchilov & Hutter 2017                            |
| `NAdamW`     | Nesterov AdamW                      | Dozat 2016 + AdamW                                  |
| `RAdam`      | Rectified Adam                      | Liu et al. 2019                                     |
| `QHAdamW`    | Quasi-hyperbolic AdamW              | Ma & Yarats 2019                                    |
| `Lamb`       | LAMB (layer-wise adaptive)          | You et al. 2019                                     |
| `Adafactor`  | Adafactor (factored 2nd moment)     | Shazeer & Stern 2018                                |
| `Lion`       | Lion (sign of EMA)                  | Chen et al. 2023                                    |
| `Soap`       | SOAP (Shampoo-in-Adam-basis)        | Vyas et al. 2024                                    |
| `KronPsgd`   | Kron / PSGD                         | Li 2018                                             |
| `Muon`       | Muon (Newton–Schulz orthogonal)     | Jordan et al. 2024                                  |
| `Sophia`     | Sophia-H (diagonal-Hessian)         | Liu et al. 2023                                     |
| `Mars`       | MARS (variance-reduced)             | Yuan et al. 2024                                    |

## Usage

```rust
use rlx_optim::{AdamW, Optimizer};

let mut opt = AdamW::new(3e-4).with_weight_decay(0.1);
let shape = [768, 768];
let mut w = vec![0.0f32; 768 * 768];
let g = vec![0.01f32; 768 * 768];

for _ in 0..100 {
    opt.step("transformer.layers.0.attn.q_proj", &shape, &mut w, &g);
    opt.end_iteration(); // advances the global step counter
}
```

Per-parameter moments are keyed by `name`, so one optimizer instance
holds the state for *every* tensor in a model. Matrix-aware
optimizers (Adafactor, SOAP, Muon, Kron-PSGD) look at `shape` and fall
back to a plain elementwise rule for 1-D / higher-rank tensors.

## Design notes

* No external dependencies. Reference Rust; backends that ship a
  fused step kernel (see `rlx-metal::splat_adam`) bypass this crate
  for their hot path.
* Pure `&mut [f32]` / `&[f32]` slices — call from anywhere holding a
  flat parameter buffer, including `rlx-umap::WeightStore` or a
  hand-rolled training loop.
* `forbid(unsafe_code)`.

## Implementing for a backend

The `Optimizer` trait is intentionally minimal — `(name, shape, &mut [f32], &[f32])`
— so backends can write a fused step kernel and impl the trait
without owning host buffers:

```rust,ignore
use rlx_optim::Optimizer;

pub struct MetalFusedAdamW {
    pipeline: ComputePipelineState,
    lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32,
    step: u32,
    state: HashMap<String, (Buffer, Buffer)>, // m, v on the device
}

impl Optimizer for MetalFusedAdamW {
    fn step(&mut self, name: &str, _shape: &[usize], param: &mut [f32], grad: &[f32]) {
        // Upload param + grad to device, dispatch the fused kernel,
        // download the updated param. The lr/eps/beta args go to a
        // small uniform buffer; m and v stay resident on the device.
        // (See rlx-metal::splat_adam for a worked example.)
    }
    fn end_iteration(&mut self) { self.step += 1; }
}
```

The existing `rlx-metal::splat_adam` kernel is the canonical
fused-step example. It currently exposes a free function rather than
an `Optimizer` impl because it carries per-attribute scaling specific
to Gaussian splat training; a thin adapter struct in `rlx-metal`
could wrap it into the trait if you want a uniform interface from a
generic trainer.

## Cross-crate integration

| Caller       | Path                                                              |
|--------------|-------------------------------------------------------------------|
| `rlx` prelude | `rlx::optim::*` behind feature `optim`                            |
| `rlx-umap`   | `rlx_umap::optim_adapter::step_weight_store` behind feature `optim` (bridges `WeightStore` ↔ any `Optimizer`) |

## Performance

Enable the `parallel` feature to dispatch the elementwise inner loops
of Adam, AdamW and Lion to rayon when a tensor crosses 64k elements.
LAMB and MARS cache their scratch buffers across iterations, so a
trainer running for thousands of steps allocates exactly once per
parameter (not per step).

## Status

| Property                | Notes                                     |
|-------------------------|-------------------------------------------|
| Numerical reference     | Yes; matches PyTorch / Optax conventions  |
| CPU parallelism         | Optional via `parallel` feature (rayon)   |
| Backend-fused kernels   | Trait is impl'able from any backend crate; see "Implementing for a backend" above |
| Distributed reductions  | No (single-host)                          |
| Mixed precision         | Caller-side (cast to f32 before stepping) |

## License

GPL-3.0-only.
