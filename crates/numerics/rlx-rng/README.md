# rlx-rng

Bit-exact reproductions of the numpy and PyTorch random streams. Pure Rust, no
dependencies — not even on `rlx-ir`.

Porting a Python reference usually stops at the arithmetic, and then fails on
anything seeded: a random projection, a subsample, a shuffled split. Matching
those needs the *same stream*, not merely the same distribution — a
different-but-valid Gaussian gives a different projection matrix, and every
number downstream moves.

## What's here

| entry point | matches |
|---|---|
| `numpy::RandomState` | `np.random.RandomState(seed)` — MT19937 + legacy polar Gaussian |
| `numpy::SeedSequence` + `numpy::Generator` | `np.random.Generator(PCG64(seed))` |
| `numpy::integers` / `choice_pm1` | `Generator.integers` (Lemire), `±1` draws |
| `numpy::gaussian_random_matrix` | sklearn-style Gaussian random projection |
| `numpy::deterministic_hash` / `hashing_projection` | FNV-1a keyed bucket + sign assignment |
| `torch::RandomState` / `torch::randperm` | `torch.manual_seed(s); torch.randperm(n)` |
| `torch::subsample_indices` | seeded subsample without replacement |

## The details that are invisible until you diff the streams

Each generator was verified against its reference implementation, and each
needed one of these. Every one produces a perfectly reasonable random sequence
when wrong, which is why they are worth stating:

- numpy's legacy Gaussian returns `f·x2` and **caches `f·x1`** — dropping the
  cached variate desynchronises everything after the first draw.
- PCG64's `set_seed` reads word 0 as the **high** half.
- `Generator.integers` uses **Lemire** rejection, not the masked rejection
  `RandomState` uses, and dispatches to a **32-bit** path when the range fits —
  where PCG64 splits one `u64` and buffers the high half. The two paths consume
  the stream at different rates, so the wrong one matches for a draw or two and
  then diverges.
- torch's `randperm` is Fisher–Yates walking **forward**.
- `deterministic_hash` is deliberately *not* Python's `hash()`, which is salted
  by `PYTHONHASHSEED` and differs between runs.

## Install

```toml
[dependencies]
rlx-rng = "0.2"
```

## Quickstart

```rust
// np.random.RandomState(0).normal(0.0, 1.0, 4)
let mut rs = rlx_rng::numpy::RandomState::new(0);
let xs = rs.normal(0.0, 1.0, 4);

// np.random.Generator(PCG64(42)).integers(0, 10, 4)
let mut g = rlx_rng::numpy::Generator::new(42);
let idx = rlx_rng::numpy::integers(&mut g, 10, 4);

// torch.manual_seed(7); torch.randperm(5)
let perm = rlx_rng::torch::randperm(5, 7);
```

## License

MIT OR Apache-2.0.
