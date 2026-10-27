# `cargo rlx doctor`

One command that tells you what RLX can run on this machine, what is missing,
and how to fix it.

```sh
cargo rlx doctor
```

For every device RLX knows, `doctor` reports whether its backend is built into
the binary, whether the hardware and driver are present, and — for each
available device — runs a small graph on it and checks the answer. It then
matches the machine against a table of known problems and prints the fix for
each one that applies, and lists the `RLX_*` settings in effect.

- [Install and run](#install-and-run)
- [Reading the output](#reading-the-output)
- [Flags and exit codes](#flags-and-exit-codes)
- [Known issues](#known-issues)
- [Settings: RLX_* variables, profiles, config files](#settings-rlx_-variables-profiles-config-files)
- [Toolchains: discovery and tested versions](#toolchains-discovery-and-tested-versions)
- [XDNA (Ryzen AI NPU) in detail](#xdna-ryzen-ai-npu-in-detail)
- [Reporting a bug](#reporting-a-bug)
- [For maintainers](#for-maintainers)

---

## Install and run

`doctor` lives in the `cargo-rlx` binary (`crates/tooling/rlx-check`). It can
only check backends compiled into it, so build it with the features for your
hardware.

**From a checkout, without installing:**

```sh
cargo run -p rlx-check --bin cargo-rlx --features <features> -- doctor
```

**Installed, so `cargo rlx doctor` works anywhere:**

```sh
cargo install --path crates/tooling/rlx-check --features <features>
cargo rlx doctor
```

| Machine | `<features>` |
|---|---|
| macOS, Apple silicon | `metal,gpu` (add `mlx`, `coreml` if you use them) |
| Linux + NVIDIA | `cuda,vulkan,gpu` |
| Linux + AMD GPU | `rocm,vulkan,gpu` |
| Ryzen AI laptop (NPU + Radeon iGPU) | `xdna,rocm,vulkan,gpu` |
| Intel GPU | `oneapi,vulkan,gpu` |
| CPU only | *(none)* |

Backends not built in are listed on one line at the end of the device table
("not built into this binary"), so a missing feature is never mistaken for
missing hardware.

---

## Reading the output

```text
rlx 0.2.19 on linux-x86_64

Devices
  ok  CPU         ready       check rel err 0.0e0 in 5 ms
  ok  ROCm        ready       check rel err 0.0e0 in 622 ms
  ok  XDNA NPU    ready       check rel err 2.9e-3 in 2842 ms
        mlir-aie 1.4.4 found at ~/aie144/lib/python3.14/site-packages/mlir_aie/bin/aiecc
  ok  GPU (wgpu)  ready       check rel err 0.0e0 in 12 ms
        AMD Radeon 780M Graphics (RADV PHOENIX)
  ok  Vulkan      ready       check rel err 0.0e0 in 1 ms
        AMD Radeon 780M Graphics (RADV PHOENIX)
  not built into this binary (enable the Cargo feature): metal, mlx, ane, cuda, …

Environment (1 RLX_* set)
  ok  RLX_XDNA_XRT_LIB=~/xrt-root/usr/lib/x86_64-linux-gnu  [xdna]

Known issues on this host (docs/known-issues.md)
  info    RLX-ROCM-001  MIOpen has no kernel database
        fix: For MIOpen convolutions, install AMD's packaged MIOpen (repo.radeon.com) or set
             MIOPEN_SYSTEM_DB_PATH to a directory holding its *.kdb files.

healthy — ready: CPU, ROCm, XDNA NPU, GPU (wgpu), Vulkan
```

### Device status

| Status | Meaning |
|---|---|
| `ready` | Available, and the check graph matched the reference |
| `BROKEN` | Available, but the check graph errored or gave wrong numbers |
| `unavailable` | Backend built in, but hardware, driver or runtime is missing — the line below says which, in the backend's own words |
| *not built* | Backend not compiled into this binary (listed on one line) |

### The check run

Each available device runs `relu(a·b + c)` on small deterministic inputs
(`a` 4×8, `b` 8×4) and is compared against an f64 reference computed by
`doctor` itself — not against RLX's CPU backend, so a bug shared by CPU and the
device cannot hide. The relative error must be within the device's tolerance:

| Device | Tolerance | Why |
|---|---|---|
| CPU, Metal, CUDA, ROCm, wgpu, Vulkan, … | `1e-4` | f32 compute |
| TPU | `1e-2` | bf16 matmul |
| XDNA NPU, Hexagon, ANE | `5e-2` | INT8 / f16 engines |

A compile error or a panic inside the run is caught and shown as the device's
error instead of crashing `doctor`. `--no-run` skips the check runs.

### Detail lines

Under a device, `doctor` shows what it detected: the adapter name (wgpu,
Vulkan), the CUDA driver and NVRTC versions, the mlir-aie version and where it
was found. A `warning:` line flags a toolchain outside what RLX supports (for
example a CUDA driver older than 12.3, or a Peano build that does not match the
installed mlir-aie).

### Environment

Every `RLX_*` variable that is set, classified:

| Mark | Meaning |
|---|---|
| `ok` | A registered variable; `[group]` is its area |
| `old` | Deprecated spelling; use the name shown |
| `??` | Not a variable RLX reads — almost always a typo; the nearest registered name is suggested |

Variables switched on indirectly (by `RLX_PROFILE`, `RLX_CONFIG` or a list
variable) are listed under *set by profile / config / list* with their source.

### Known issues

Each matching entry shows its severity, id, title and the fix. See
[Known issues](#known-issues).

### Last line

`healthy — ready: …` or `problems found — ready: …`, with the devices that are
ready.

---

## Flags and exit codes

| Command | What it does |
|---|---|
| `cargo rlx doctor` | Full report (devices, check runs, known issues, settings) |
| `cargo rlx doctor --device cuda,xdna` | Only these devices |
| `cargo rlx doctor --no-run` | Skip the check runs (availability only; faster) |
| `cargo rlx doctor --report` | Paste-ready Markdown for a GitHub issue; home directory and user name redacted |
| `cargo rlx doctor --json` | Machine-readable report (devices, checks, toolchain versions, issues, settings) |
| `cargo rlx doctor --known` | Every known issue with its symptom and fix |
| `cargo rlx doctor --known --markdown` | The same, as `docs/known-issues.md` |
| `cargo rlx setup xdna [--version X.Y.Z] [--dry-run]` | Install the XDNA toolchain (mlir-aie + pinned Peano) without root |

**Exit code** is non-zero when any of these hold, so scripts and CI can use
`doctor` as a gate:

- an available device failed its check run (`BROKEN`);
- a known issue of severity *error* matches this machine;
- an `RLX_*` variable is set that RLX does not read.

`--known` always exits 0.

When a program fails to create a session, the panic message ends with
*Run `cargo rlx doctor` for a diagnosis and the fix*, so users are pointed here
from the first error they see.

---

## Known issues

[`docs/known-issues.md`](docs/known-issues.md) lists every problem users have hit,
each with a permanent id (`RLX-<AREA>-<NNN>`), the symptom text to search for,
the cause and the fix. It is generated from
[`known_issues.rs`](crates/core/rlx-runtime/src/doctor/known_issues.rs).

- **Detected** entries are checked automatically by `doctor` against facts about
  the machine (files, sysfs, env, detected versions, each device's status and
  reason) and printed when they match.
- **Build-time** entries cannot be detected — they stop the build before
  `doctor` exists — so they are listed with the exact compiler/linker error to
  search for (e.g. `unable to find library -lopenblas`, `` `default-features =
  false` cannot override workspace's `default-features` ``).

Current areas: XDNA (XRT runtime and headers, device permissions, memlock,
XDNA 2 parts, mlir-aie/Peano toolchain and version compatibility), ROCm (MIOpen
database, 780M/gfx1103 override), CUDA (driver floor, NVRTC, missing driver),
wgpu/Vulkan (host-routed elementwise ops, missing ICD), environment (unknown
`RLX_*` variables), check-run failures, and build-time failures (cargo version,
libpython, OpenBLAS, MLX submodule, glibc floor for ONNX Runtime).

Severity: **error** blocks a device or gives wrong results; **warning** works
but is slower or one step from breaking; **info** is expected behaviour worth
knowing about.

---

## Settings: RLX_* variables, profiles, config files

RLX reads its switches from `RLX_*` environment variables. All of them are
registered in one table (`crates/core/rlx-ir/src/env_registry_data.inc.rs`,
listed in [`docs/rlx-env-vars.md`](docs/rlx-env-vars.md)).

**Typos are caught.** On the first `RLX_*` read, RLX prints a one-time warning
for any set `RLX_*` variable it does not read, with the nearest registered name:

```text
rlx: RLX_DISPATCH_REPOT is not a variable rlx reads; did you mean RLX_DISPATCH_REPORT?
```

Silence it with `RLX_ENV_WARN_UNKNOWN=0`. `doctor` lists the same.

**Several switches at once.** Anything set explicitly always wins over these:

- **Lists** — `RLX_DISABLE=cse,sccp` sets `RLX_DISABLE_CSE` and
  `RLX_DISABLE_SCCP`; `RLX_TRACE=metal,gguf` sets `RLX_METAL_TRACE` and
  `RLX_GGUF_TRACE`; `RLX_DEBUG` likewise for `RLX_*_DEBUG`. Case-insensitive.
- **Profiles** — `RLX_PROFILE=<name>[,<name>…]`, later ones win:

  | Profile | Sets |
  |---|---|
  | `debug` | runtime logging, dispatch report, static NaN lint, deprecation notices |
  | `nans` | runtime NaN/Inf localizer (slow) + static lint |
  | `parity` | tightest numerics on CUDA, ROCm and Metal, for cross-backend comparison |
  | `deterministic` | reproducible collective reductions (set on every rank) |
  | `ci` | a missing GPU fails tests instead of skipping; memory-plan self-check |

- **A settings file** — `RLX_CONFIG=rlx.toml`:

  ```toml
  # applied whenever RLX_CONFIG points here
  RLX_DISPATCH_REPORT = 1
  LINT_NUMERICS = true          # the RLX_ prefix is optional

  [profile.bench]               # selected with RLX_PROFILE=bench
  RLX_CUDA_TMA = true
  ```

  `true`/`false` become `1`/`0`. Unknown keys and malformed lines are reported
  once and skipped.

Precedence, highest first: explicit variable → list variable → `RLX_CONFIG` top
level → `RLX_PROFILE`. `doctor` shows which variables each of these set and
from where.

---

## Toolchains: discovery and tested versions

**Discovery.** Where a backend needs an external toolchain, RLX looks for it
instead of requiring environment variables; variables still override. For XDNA
the mlir-aie compiler (`aiecc`), the Peano core compiler (`llvm-aie`) and the
`aie_kernels` include tree are searched in this order:

1. the `cargo rlx setup` directory (`~/.local/share/rlx/toolchains/*`);
2. `RLX_XDNA_TOOLCHAIN_DIR` (any venv or install prefix);
3. the active environment (`$VIRTUAL_ENV`, `$CONDA_PREFIX`);
4. `aiecc` on `PATH` (its venv), including source builds (`<prefix>/bin/aiecc`
   next to `<prefix>/include/aie_kernels`);
5. `~/.local`, venvs up to two levels under `$HOME` (`~/aie144`,
   `~/mlir-aie/ironenv`), conda/miniforge/mamba envs, pyenv versions;
6. the system Python's packages and `/opt/*`.

A symlinked `aiecc` is followed to its real install. When several mlir-aie
releases are installed, RLX picks a **working combination**: releases that pass
the NPU suite before known-bad ones, newest first, paired with the Peano build
that release pins if it is installed anywhere, else the one next to it, else
the newest. The XRT runtime is found in `RLX_XDNA_XRT_LIB`, `LD_LIBRARY_PATH`,
`$XILINX_XRT/lib`, `/opt/xilinx/xrt/lib` and the distro library directories.

**When nothing is found**, the error says how many places were searched, lists
anything half-installed it saw (an mlir-aie that ships only `aiecc.py`, a Peano
without mlir-aie), and gives the one-command fix.

**Tested versions.** `doctor` shows, next to each device, the external toolchain
versions RLX has actually been run against, taken from
[`tested_toolchains.tsv`](crates/core/rlx-runtime/src/doctor/tested_toolchains.tsv).
That file is written by the version matrix (see
[For maintainers](#for-maintainers)); it only ever contains versions that ran.

---

## XDNA (Ryzen AI NPU) in detail

What `doctor` needs to report the NPU **ready**:

1. **The `amdxdna` kernel driver** and `/dev/accel/accelN` read/write for your
   user (`sudo usermod -aG render $USER`, then log in again).
2. **The XRT runtime.** Ubuntu 26.04: `sudo apt install libxrt2 libxrt-npu2
   libxrt-utils-npu`. Without root, `apt-get download` those packages,
   `dpkg-deb -x` them into a directory and set `RLX_XDNA_XRT_LIB` to its
   `usr/lib/x86_64-linux-gnu`. Do **not** put that directory on
   `LD_LIBRARY_PATH` if you also use ROCm (its `libxrt_hip.so` hides ROCm's HIP —
   known issue RLX-XDNA-008).
3. **RLX's XRT shim.** Built automatically from `csrc/xrt_gemm_shim.cpp` the
   first time it is needed, against the XRT headers it finds, and cached in
   `~/.cache/rlx-xdna`. Needs a C++ compiler and the XRT headers (`libxrt-dev`).
   Set `RLX_XDNA_SHIM` to use your own build, or `RLX_XDNA_NO_SHIM_BUILD=1` to
   disable the automatic build.
4. **mlir-aie and its pinned Peano.** One command, no root:

   ```sh
   cargo rlx setup xdna                 # mlir-aie 1.4.4 + Peano 22.0.0.2026092101
   cargo rlx setup xdna --version 1.4.2 # another tested release
   cargo rlx setup xdna --dry-run       # show the commands only
   ```

   It installs into a venv under `~/.local/share/rlx/toolchains/` (searched
   first, so nothing needs configuring), checks what RLX now resolves, and, if
   the XRT runtime is missing, prints the commands for your distro (those need
   root). Only releases that pass the NPU suite are offered; mlir-aie publishes
   toolchains for Linux x86-64 only. By hand, any venv works and is found
   automatically:

   ```sh
   python3 -m venv ~/mlir-aie && ~/mlir-aie/bin/pip install mlir_aie==1.4.4 \
     -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.4
   ~/mlir-aie/bin/pip install llvm-aie==22.0.0.2026092101 \
     -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
   ```

**mlir-aie compatibility**, measured on a Ryzen 9 8945HS (npu1, firmware
1.5.5.391) with the full NPU op suite:

| mlir-aie | Result | Notes |
|---|---|---|
| 1.4.0 – 1.4.4 | **pass** | 1.4.4 recommended |
| 1.3.4 | **pass** | |
| 1.3.2 | **pass** | |
| 1.3.3 | refused | emits LLVM IR no Peano parses (RLX-XDNA-011) |
| 1.3.0, 1.3.1 | refused | softmax/attention compile but return wrong results (1.3.0) or run out of registers (1.3.1); their own Peano builds are no longer published (RLX-XDNA-011) |
| 1.2.x and older | cannot run | no native `aiecc`, only `aiecc.py` (RLX-XDNA-009) |

Each mlir-aie release pins one Peano build (`utils/peano-requirements.txt` in
that release); `doctor` warns when the installed Peano differs (RLX-XDNA-010).
Mismatches often still work — 1.4.4 runs on the 1.4.0 Peano — but when kernels
fail with `opt: … expected value token`, install the pinned build.

*Refused* releases are not used even when installed or named by `AIECC`: XDNA
reports unavailable with the reason and the working releases found on the
machine, so a release that can return wrong numbers never runs silently.
`RLX_XDNA_ALLOW_KNOWN_BAD=1` uses one anyway (the version matrix does, to keep
measuring them).

**How RLX adapts to each release.**

- **Probing.** The first time RLX sees a toolchain it predicts the AIE-MLIR
  syntax (1.4.0 changed `aie.dma_bd`, 1.4.3 removed `!aie.objectfifosubview`)
  and the way outputs are requested (`--aie-generate-*`, `--get-xclbin
  --get-npu-insts`, `--get=…`) from the wheel version and `aiecc --help`, then
  confirms it by compiling a small kernel that uses every construct that has
  changed. If the prediction fails it tries the other combinations. The result
  is cached per toolchain (path, size, mtime) in
  `~/.cache/rlx-xdna/recipes.tsv`, so a release that changes one of these known
  points — or an install whose version cannot be read — works without a code
  change. Measured: an unversioned copy of 1.4.1 was predicted wrong, adapted,
  and passed the full suite.
- **Retries.** An oversized kernel is retried at lower optimisation levels
  (`-O 1`, then `-O 0`); release 1.4.0's Chess default is switched off when the
  driver offers `--no-xchesscc`.
- **Peano recovery.** When every attempt fails with a Peano-mismatch signature
  (`expected value token`, `ran out of registers`), RLX retries with the other
  Peano builds installed, pinned build first, and remembers one that works.
- **Readable failures.** aiecc's output is captured; a failure shows its last
  lines instead of pages of compiler output. `RLX_VERBOSE=1` shows everything.

`RLX_XDNA_AIE_DIALECT=<version>` forces a syntax instead of probing.

**XRT versions:** 2.21.75 (Ubuntu 26.04 archive) passes with mlir-aie 1.3.4 and
1.4.4. 2.25.0 (`ppa:amd-team/xrt`) is in the matrix but not yet recorded.

---

## Reporting a bug

1. Build `cargo-rlx` with the backends you use (see
   [Install and run](#install-and-run)).
2. Run `cargo rlx doctor`. If a known issue matches, its fix is printed — try it
   first.
3. Otherwise open an issue and paste `cargo rlx doctor --report` into the
   template's *cargo rlx doctor --report* field. The report contains the RLX
   version and features, every device's status and check result, detected
   toolchain versions, matched known issues and your `RLX_*` settings, with your
   home directory and user name redacted.
4. If the build itself fails, search [`docs/known-issues.md`](docs/known-issues.md)
   for the error text, and include the error and `rustc --version` in the issue.

---

## For maintainers

### Where things live

| Piece | Location |
|---|---|
| Report, check run, rendering | `crates/core/rlx-runtime/src/doctor/mod.rs` |
| Host facts (fakeable for tests) | `crates/core/rlx-runtime/src/doctor/facts.rs` |
| Known-issue table | `crates/core/rlx-runtime/src/doctor/known_issues.rs` |
| `--report` Markdown | `crates/core/rlx-runtime/src/doctor/report.rs` |
| Tested versions | `crates/core/rlx-runtime/src/doctor/tested_toolchains.tsv` |
| CLI | `crates/tooling/rlx-check/src/bin/cargo-rlx.rs` |
| XDNA toolchain discovery and pairing | `crates/backends/rlx-xdna/src/toolchain.rs` |
| XDNA probing, recipe cache, recovery | `crates/backends/rlx-xdna/src/compile.rs` |
| `cargo rlx setup xdna` plan | `crates/backends/rlx-xdna/src/setup.rs` |
| Version matrix | `scripts/toolchain_matrix.py`, `scripts/toolchain_matrix.toml` |
| Issue template | `.github/ISSUE_TEMPLATE/bug_report.yml` |

### Adding a known issue

When a GitHub issue is resolved, turn it into an entry so the next user gets the
fix from `doctor`:

1. Append a `KnownIssue` to `ISSUES` in `known_issues.rs` with a new id (ids are
   permanent; never reuse one), the symptom text users see, the cause and the
   exact fix.
2. If it can be recognised on a running machine, add a `detect` function over
   `HostFacts` — device status and reason text, detected versions, env, or files
   under the (fakeable) root.
3. Add a fake-host case in that file's tests. `every_detected_issue_has_a_fake_host_case`
   fails until you do, and `healthy_hosts_stay_quiet` makes sure the check does
   not fire on a working machine.
4. Run `just gen-known-issues` to regenerate `docs/known-issues.md`;
   `known_issues_doc_is_up_to_date` fails while it is stale.

Fake hosts are temp directories laid out like the real thing (`/sys/class/kfd/…`,
`/opt/rocm/share/miopen/db`, a venv's `site-packages`, …) plus a fake env, so
every diagnosis is tested on any machine without the hardware.

### The version matrix

`scripts/toolchain_matrix.toml` lists one run per (device, toolchain version,
rig). Each version is installed side by side (its own venv or directory); `env`
selects it; `pass` is text the command must print in addition to exiting 0.

```sh
just toolchain-matrix                          # run everything, print a table
just toolchain-matrix --only xdna              # one device or component
just toolchain-matrix --only mlir-aie --version 1.4.4
just toolchain-matrix --update                 # record results in tested_toolchains.tsv
```

Every run gets a fresh `TMPDIR`, so kernels compiled by one version can never be
reused by another and make a run pass that compiled nothing. `--update` writes
one row per combination, passing or failing, with the pairing (e.g. the Peano
build) in the notes column; `doctor` shows the passing rows.

To cover a new release: install it next to the others, add a `[[run]]`, and run
with `--update`.

### Watching upstream

```sh
just toolchain-upstream
```

compares each project in the spec's `[[upstream]]` list (mlir-aie, XRT) with the
newest version in `tested_toolchains.tsv` and exits 1 when a newer release
exists. It uses the highest version across all releases, not GitHub's "latest"
marker, and ignores date prefixes, non-version tags and old year-based schemes.
Run it nightly or before a release so a breaking upstream change is found here
before a user reports it.

### Tests

```sh
cargo test -p rlx-runtime --lib doctor     # report, known issues (fake hosts), doc freshness
cargo test -p rlx-xdna --lib               # toolchain discovery (fake installs), aiecc CLI handling
cargo test -p rlx-ir --lib env             # typo suggestions, lists, profiles, config file
```
