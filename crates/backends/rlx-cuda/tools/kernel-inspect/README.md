# kernel-inspect — rlx GPU kernel inspection harness

A harness for studying the GPU kernels rlx actually ships. The `.cu` kernel
sources are **shared** across the CUDA and ROCm backends (`rlx-gpu-kernels`),
so one tool inspects both — it captures the exact translation unit each backend
JIT-compiles, then lowers it to **disassembly**, **register/occupancy**, and
**opcode-profile** reports on the target GPU:

| | CUDA (NVIDIA) | ROCm (AMD) |
|---|---|---|
| compile | NVRTC → PTX | hipRTC → code object (`.hsaco`) |
| disasm | `ptxas` + `cuobjdump` → SASS | `llvm-objdump` → GCN ISA |
| resources | registers / smem / spills | VGPR / SGPR / LDS / scratch-spills |
| occupancy | warps/SM (block=256) | waves/SIMD |

It answers, per backend: which kernels **spill** (scratch/local traffic →
latency)? What's the **register/VGPR pressure** and resulting **occupancy
ceiling**? Which kernels use **tensor/matrix cores** (HMMA / MFMA), fp64,
transcendentals? What's the **opcode mix**? Did a source change help or hurt?
(`diff` two runs.) The same kernel often behaves differently per arch — e.g.
`attention` is smem-limited on sm_86 but VGPR-spilling on MI100.

Everything is derived from rlx's *own* JIT output (not a re-implementation), so
the reports match what runs in production.

## How it works

1. `rlx-cuda`'s and `rlx-rocm`'s `kernels::compile()` each have an env-gated
   hook: with `RLX_DUMP_KERNELS=<dir>` set, every kernel they JIT-compile is
   snapshotted (both the static `kernel_cache!` kernels and the dynamic
   `{Cuda,Rocm}GpuKernel` registry route through this one function):
   - `cu/<entry>.cu`  — the exact source the JIT saw (post `gelu.cuh` / codegen assembly)
   - CUDA: `ptx/<entry>.ptx`   ·   ROCm: `codeobj/<entry>.hsaco`
   - `manifest.jsonl` — `{entry, src_hash, arch, …}` per kernel
2. `kinspect.py` auto-detects the dump kind (`ptx/` → CUDA, `codeobj/` → ROCm),
   deduplicates translation units, disassembles each, and writes the reports.

Coverage = the kernels the test run actually exercises. The manifest records
exactly what was captured.

## Usage

```bash
# one-shot: build the backend, run its tests to trigger every kernel, analyze.
# auto-detects CUDA (nvidia-smi) vs ROCm (rocminfo).
python3 crates/backends/rlx-cuda/tools/kernel-inspect/kinspect.py run

python3 …/kinspect.py run --target rocm       # force ROCm (on the AMD rig)
python3 …/kinspect.py run --release conv3d    # release + a cargo test filter

# analyze an existing dump without rebuilding
RLX_DUMP_KERNELS=/tmp/kd cargo test -p rlx-rocm -- --test-threads=1
python3 …/kinspect.py analyze /tmp/kd

# before/after a kernel edit (works for both backends)
python3 …/kinspect.py diff before/report.json after/report.json
```

From the mac, drive it on the rigs via `rig.sh` (see the repo's rig setup):

```bash
./rig.sh --msi exec 'python3 crates/backends/rlx-cuda/tools/kernel-inspect/kinspect.py run'  # CUDA on msi
./rig.sh --amd exec 'python3 crates/backends/rlx-cuda/tools/kernel-inspect/kinspect.py run'  # ROCm on amd
```

## Output

Under `<dump>/report/` (default `target/kernel-inspect/dump-<target>/report/`):
- `report.md` — human summary: per-kernel table sorted by register/VGPR
  pressure, a **spills** section (perf risk), a **tensor/matrix-core** section,
  and per-kernel opcode profiles.
- `report.json` — machine-readable, the input to `diff`.
- `sass/<kernel>.sass` (CUDA) or `isa/<kernel>.isa` (ROCm) — full disassembly.

## Notes

- CUDA `--arch` auto-detects the live GPU's compute capability (RTX 3080 Ti →
  `sm_86`). ROCm reads the code object's arch from the manifest (MI100 →
  `gfx908`); the `.hsaco` is arch-specific, so it reports the arch hipRTC
  targeted at runtime.
- Occupancy is a **theoretical upper bound** from register/VGPR (and smem)
  pressure; real launch geometry lives in the Rust launch config. gfx908 limits
  are exact (the rig's GPU); RDNA3 rows are best-effort.
- `analyze` / `diff` need only the disasm tools (`ptxas`+`cuobjdump`, or
  `llvm-objdump`+`llvm-readobj`), no GPU.
- The `RLX_DUMP_KERNELS` hook is off unless the env var is set — zero cost to
  normal builds.
