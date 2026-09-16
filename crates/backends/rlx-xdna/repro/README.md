# AIE2 codegen bugs: nested reductions in `aie.core`

Five minimal `aie.device(npu1_1col)` kernels, identical except for loop
structure. Three produce exact results on hardware; two produce garbage. All
were run on a Ryzen AI NPU (`RyzenAI-npu1`, AIE 1.1, fw 1.5.5.391) with
XRT 2.21.75, `mlir_aie` 0.0.1.2026020504 and `llvm-aie`
22.0.0.2026082501 (peano).

Every kernel takes two `64xf32` inputs `A`, `B` and writes `64xf32` `O`, with
`A[i] = ((i%13)-6)*0.1` and `B[i] = ((i%11)-5)*0.1`. Nothing in any kernel
depends on the outer index `i`, so **every element of `O` must be identical** —
which makes a wrong answer obvious without needing a reference.

Reproduce with the harness in `../examples/xdna_repro.rs`:

```
AIECC=<...>/aiecc.py PEANO=<...>/llvm-aie RLX_XDNA_SHIM=<...>/librlx_xdna_shim.so \
REPRO_MLIR=$PWD/02_two_sibling_reductions_FAIL.mlir \
cargo run --release -p rlx-xdna --features xrt,direct --example xdna_repro
```

## Bug 1 — two sibling nested reductions corrupt the FIRST outer iteration

`01` and `02` differ only in that `02` runs the same reduction **twice** in one
loop body and adds the results. `03` runs the identical two reductions in two
**separate** outer loops.

| file | shape | `O[0..6]` |
|---|---|---|
| `01_one_reduction_OK` | `for i { R(j){dot(k)} }` | `0.44999987` x6 — correct (0.45) |
| `02_two_sibling_reductions_FAIL` | `for i { R1; R2 }` | **`[-0.4, 0.9, 0.9, 0.9, 0.9, 0.9]`** |
| `03_split_outer_loops_OK` | `for i { R1 }; for i { R2 }` | `0.89999974` x6 — correct (0.9) |

Only `O[0]` is wrong in `02`; all 63 other elements are exact. Splitting the
two reductions into separate outer loops is a complete fix for this case.

## Bug 2 — a nested reduction plus `exp` in the same body produces garbage

`04` and `05` both compute a sum of exponentials inside a reduction. The only
difference is where the exponent comes from: `04` loads it, `05` computes it
with a **nested** reduction first.

| file | shape | `O[0..6]` |
|---|---|---|
| `04_exp_in_reduction_OK` | `for i { S = for j { load; exp; add } }` | `7.9011407` x6 — correct (7.901141) |
| `05_nested_reduction_plus_exp_FAIL` | `for i { S = for j { dot(k); exp; add } }` | **`NaN` x6** (also seen: `-1.6e33`, `2.8e21`) |

So `exp` inside a reduction is fine, and a nested reduction is fine; the two
together are not. The garbage is not deterministic across builds, which points
at uninitialised state rather than a wrong constant.

`exp` is a red herring in the sense that matters: delete it and keep the same
loop shape, and the result is still garbage (`[-0.2655, -0.2725, 1e-44, 0.0,
...]` for values that must all be equal). The `exp` expansion here is a pure
`arith` sequence — no `math.exp`, which AIE2 does not lower.

## What is NOT the cause

Checked directly on hardware, each ruled out:

* **Optimisation level** — `-O0`, `-O1`, `-O2`, `-O3` behave identically.
* **`math.exp`** — not used; the expansion is `arith` only.
* **Stack size** — the failing core's frame is 704 B against a 1024 B stack;
  raising it to 4096 via `stack_size` on `aie.core` changes nothing.
* **`aie.buffer`** — the failing kernels use none.
* **The `-inf` reduction initialiser** — a finite `-1e30` behaves identically.
* **The accumulator initialiser** — seeding `iter_args` with a runtime zero the
  compiler cannot constant-fold changes nothing.
* **Constant pool / f32 constants** — reusing an already-present constant in
  place of new ones changes nothing.
* **The data path** — a trivial copy kernel through the same three FIFOs is
  exact.

## Why it matters

This blocks a softmax-attention kernel, which inherently wants a max reduction
and a sum reduction over the same axis in one body (bug 1) with `exp` over a
dot product (bug 2). Every restructuring that dodges one manifestation exposes
the other: splitting bug 2 into a dot pass and an exp pass makes elements 2..N
correct but corrupts the first two, and flattening the dot pass into a single
loop with `arith.remui` is worse again.
## License

MIT OR Apache-2.0.
