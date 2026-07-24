# rlx-bake

Offline bake for RLX: **merge graph + weights**, optimize where weight
values allow, and write a single deployable `*.rlx` file.

Same idea as CoreML / ONNX with initializers: the artifact carries the
model and the weights together. Weight-aware passes can cut work
(skip zeros, pack ternary / quants) before the file is written — **fewer
bytes to load and less compute per forward**, not just a zip of an
unoptimized graph plus a weight dump.

**Full walkthrough (concepts + MNIST train → bake → encrypt → run):**
[docs/rlx-bake.md](../../../docs/rlx-bake.md).

Also: [weight-compute caching](../../../docs/weight-compute-caching.md).

## What `*.rlx` contains

| Field | Role |
|-------|------|
| `graph` | Optimized MIR graph (params specialized to constants / packed ops) |
| `weights` | Explicit weight table: name, shape, encoding (`f32` / `gguf_tq2_0` / `gguf_q8_0`), bytes |
| `meta` | I/O shapes, bake stats (skipped / ternary / quant counts) |

Format: magic `RLXBAKE1` + `u32` schema (v2) + bincode. Schema v1
(graph-only) still loads with an empty weight table.

Ship packages (`.rlxp`): see [docs/rlxp.md](../../../docs/rlxp.md). Default
container is **flat mmap** (`RLXPFLAT`); use `--container zip|dir` when you need
inspectability. Convert or bake with `--format rlxp`:

```bash
cargo run -p rlx-bake -- graph.json -o model.rlxp --format rlxp --weights w.safetensors
cargo run -p rlx-bake -- graph.json -o model.rlxp --weights mlx-dir --weights-policy auto --opt merge
cargo run -p rlx-bake -- convert model.rlx -o model.rlxp
```

### `--weights-policy` (f32-first go / no-go)

| Policy | f32-first | Use when |
|--------|-----------|----------|
| `f32` (default) | **GO** | Dense MatMul graphs; `--opt exact`/`size` ternary+Q8 |
| `packed` | **NO-GO** if MLX packs or DDUF half exist | Space/speed — keep `mlx_*` / `f16` encodings |
| `auto` | **NO-GO** if packs/half and ternary/quant off; else **GO** | Prefer packs without breaking size profile |

`exact`/`size` always force **GO** (rewrites need floats).

Optional ONNX → RLXP (executable MIR graph by default):

```bash
cargo run -p rlx-bake --features onnx -- import-onnx model.onnx -o model.rlxp
cargo run -p rlx-bake --features onnx -- import-onnx model.onnx -o weights.rlxp --no-graph
```

## Pipeline

1. Specialize `Op::Param` → named `Op::Constant` from bindings
2. **skip** — `MatMul(x, 0)` → zero constant
3. **ternary** — exact `{−1,0,+1}` MatMul weights → GGUF TQ2_0 + `DequantMatMul`
4. **quant** (opt-in) — remaining F32 MatMul weights → GGUF Q8_0 + `DequantMatMul`
5. **unfold** — catalog surviving weight Constants into the weight table
6. Algebraic simplify → DCE → constant fold

## Optimization profiles

| `--opt` | Intent |
|---------|--------|
| `merge` | One file; same dense MatMul compute |
| `fold` | Fold weight-only math; keep dense GEMM |
| `exact` | **Default** — skip zeros + ternary TQ2 (lossless) |
| `size` | `exact` + Q8_0 for remaining dense matmuls |

Fine flags override the profile: `--no-skip`, `--no-ternary`, `--quant` /
`--no-quant`, `--no-unfold`, `--no-fold`, `--no-dce`, `--no-simplify`,
`--no-cleanup`, `--memory duplex|runtime|compact`, `--dedupe-constants`,
`--no-folded-bindings`. Details: [docs/rlx-bake.md](../../../docs/rlx-bake.md).

## CLI

```bash
# Bundle dir (model.hir.json + weights.safetensors)
cargo run -p rlx-bake -- path/to/bundle -o model.rlx

# MIR / HIR JSON + safetensors
cargo run -p rlx-bake -- graph.json --weights weights.safetensors -o model.rlx

# Profiles + overrides
cargo run -p rlx-bake -- bundle -o model.rlx --opt size
cargo run -p rlx-bake -- bundle -o model.rlx --opt merge
cargo run -p rlx-bake -- bundle -o model.rlx --opt exact --no-ternary
cargo run -p rlx-bake -- bundle -o model.rlx --quant          # same idea as --opt size for Q8
```

## Encryption (`--features encrypt`)

Opt-in cargo feature (pulls ChaCha20-Poly1305 + Argon2id). Seals the
**entire** file; magic `RLXENC01`. Plaintext `read_rlx` refuses encrypted
files so a password cannot be skipped by accident.

```bash
cargo run -p rlx-bake --features encrypt -- bundle -o model.rlx --password 'secret'
cargo run -p rlx-bake --features encrypt -- bundle -o model.rlx --password-env RLX_BAKE_PASSWORD
cargo run -p rlx-bake --features encrypt -- decrypt model.rlx -o model.plain.rlx --password 'secret'
```

```rust
// Cargo.toml: rlx-bake = { version = "…", features = ["encrypt"] }
write_rlx_encrypted("model.rlx", &file, "secret")?;
let file = read_rlx_with_password("model.rlx", "secret")?;
```

## MNIST demos

See **[docs/rlx-bake.md](../../../docs/rlx-bake.md)** for the full train → bake
→ encrypt → run walkthrough.

```bash
export RLX_BAKE_PASSWORD='demo-secret'
cargo run -p rlx-bake --example mnist_train_bake --features encrypt,runtime --release
cargo run -p rlx-bake --example mnist_run_encrypted --features encrypt,runtime --release
# artifact: examples/out/mnist.rlx  (or $RLX_BAKE_OUT)

# Bench bake → .rlx / .rlxp load+compile+run (no encrypt):
just throttle
RLX_ALLOW_THROTTLE=1 cargo run -p rlx-bake --example mnist_bench_rlxp --features runtime --release
```

Uses real MNIST from `RLX_MNIST_DIR` / `~/.cache/torchvision-mnist/…` when
present; otherwise synthetic digit-ish data.

## Library

```rust
use rlx_bake::{bake, BakeProfile, write_rlx};

let (file, report) = bake(&graph, &bindings, &BakeProfile::Exact.options());
assert!(!file.weights.is_empty());
write_rlx("model.rlx", &file)?;
// Compact memory (default for exact/size): materialize before Session::compile
// let graph = file.into_runtime_graph()?;
// Ship mmap pack: write_rlxp("model.rlxp", &file, None)?;
// Smaller compute+payload: BakeProfile::Size.options()
// With features = ["encrypt"]: write_rlx_encrypted / read_rlx_with_password
// With features = ["mmap"]: read_rlx_mmap for plaintext loads
// With features = ["onnx"]: onnx_to_rlxp / CLI import-onnx
```

## Features

| Feature | Role |
|---------|------|
| `encrypt` | Full-file `RLXENC01` seal |
| `runtime` | MNIST examples that `Session::compile` / `run` |
| `mmap` | mmap-backed plaintext `read_rlx` |
| `onnx` | `onnx_to_rlxp` + `import-onnx` CLI |

## License

GPL-3.0-only.
