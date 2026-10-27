# Known issues

Generated from [`known_issues.rs`](../crates/core/rlx-runtime/src/doctor/known_issues.rs) — do not edit by hand; run `just gen-known-issues`.

`cargo rlx doctor` detects the entries marked *detected* on your machine and prints the fix. Build-time entries stop the build before `doctor` can run; search this page for the error text.

| Id | Severity | Title | |
|----|----------|-------|---|
| [RLX-XDNA-001](#rlx-xdna-001) | warning | mlir-aie is newer than any release rlx has been run against | detected |
| [RLX-XDNA-002](#rlx-xdna-002) | error | rlx's XRT shim cannot be built: XRT headers missing | detected |
| [RLX-XDNA-003](#rlx-xdna-003) | error | XDNA NPU present but no XRT runtime | detected |
| [RLX-XDNA-004](#rlx-xdna-004) | error | NPU device node is not writable by this user | detected |
| [RLX-XDNA-005](#rlx-xdna-005) | warning | Locked-memory limit too low for XRT buffers | detected |
| [RLX-XDNA-006](#rlx-xdna-006) | error | XDNA 2 (Strix-class) NPU is not supported yet | detected |
| [RLX-XDNA-007](#rlx-xdna-007) | error | mlir-aie toolchain (aiecc + Peano) not found | detected |
| [RLX-XDNA-008](#rlx-xdna-008) | warning | XRT on LD_LIBRARY_PATH hides ROCm's HIP | detected |
| [RLX-XDNA-009](#rlx-xdna-009) | error | mlir-aie older than 1.3 has no native aiecc | detected |
| [RLX-XDNA-010](#rlx-xdna-010) | warning | Peano (llvm-aie) is not the build this mlir-aie release pins | detected |
| [RLX-XDNA-011](#rlx-xdna-011) | error | mlir-aie 1.3.0, 1.3.1 and 1.3.3 cannot compile every rlx kernel correctly | detected |
| [RLX-ROCM-001](#rlx-rocm-001) | info | MIOpen has no kernel database | detected |
| [RLX-ROCM-002](#rlx-rocm-002) | error | Radeon 780M (gfx1103) is not enumerated by ROCm | detected |
| [RLX-ROCM-003](#rlx-rocm-003) | info | APU GPU resets under NUMA balancing (gfx11 MES queue-eviction hang) | detected |
| [RLX-CUDA-001](#rlx-cuda-001) | error | NVIDIA driver older than CUDA 12.3 | detected |
| [RLX-CUDA-002](#rlx-cuda-002) | error | NVRTC (CUDA runtime compiler) not found | detected |
| [RLX-CUDA-003](#rlx-cuda-003) | error | CUDA backend built, but no NVIDIA driver loaded | detected |
| [RLX-WGPU-001](#rlx-wgpu-001) | info | wgpu on Vulkan/DX12 runs elementwise ops on the host | detected |
| [RLX-VULKAN-001](#rlx-vulkan-001) | error | Vulkan backend built, but no Vulkan driver (ICD) installed | detected |
| [RLX-ENV-001](#rlx-env-001) | warning | An RLX_* variable is set that nothing reads | detected |
| [RLX-CHECK-001](#rlx-check-001) | error | A device failed its check run | detected |
| [RLX-BUILD-001](#rlx-build-001) | error | cargo older than 1.99 rejects the workspace manifests | build time |
| [RLX-BUILD-002](#rlx-build-002) | error | Linking libpython fails in a cargo build | build time |
| [RLX-BUILD-003](#rlx-build-003) | error | OpenBLAS not installed | build time |
| [RLX-BUILD-004](#rlx-build-004) | error | MLX submodule not checked out | build time |
| [RLX-BUILD-005](#rlx-build-005) | error | ONNX Runtime prebuilt needs glibc 2.38+ | build time |

## RLX-XDNA-001

**mlir-aie is newer than any release rlx has been run against** (warning)

- **Symptom:** `aiecc: ... error: expected ')' / unknown type in dialect `aie``
- **Cause:** mlir-aie changes its AIE-MLIR syntax between releases (1.4.0 dma_bd, 1.4.3 objectfifosubview). rlx rewrites its kernels for every release up to 1.4.x; a newer one may have changed something else.
- **Fix:** Run `cargo rlx doctor --report` and open an issue with it. Meanwhile pin a known release: `pip install mlir_aie==1.4.4`, or set RLX_XDNA_AIE_DIALECT=1.4.4 to force the newest known syntax.

## RLX-XDNA-002

**rlx's XRT shim cannot be built: XRT headers missing** (error)

- **Symptom:** `building it needs the XRT headers (xrt/xrt_device.h)`
- **Cause:** RLX_XDNA_SHIM is unset, so rlx builds librlx_xdna_shim.so itself, and that needs XRT's development headers, which the runtime packages do not ship.
- **Fix:** Ubuntu/Debian: `sudo apt install libxrt-dev`. Or set XILINX_XRT to an XRT tree that has include/xrt/, or build the shim yourself and set RLX_XDNA_SHIM.

## RLX-XDNA-003

**XDNA NPU present but no XRT runtime** (error)

- **Symptom:** `AMD XDNA NPU detected ... but no userspace runtime is available`
- **Cause:** The amdxdna kernel driver is loaded, but the XRT userspace libraries (libxrt_coreutil, libxrt_driver_xdna) are not found.
- **Fix:** Ubuntu 26.04: `sudo apt install libxrt2 libxrt-npu2 libxrt-utils-npu`. Without root: `apt-get download` those packages, `dpkg-deb -x` them into a directory, and set RLX_XDNA_XRT_LIB to its usr/lib/x86_64-linux-gnu. Check with `xrt-smi examine`.

## RLX-XDNA-004

**NPU device node is not writable by this user** (error)

- **Symptom:** `/dev/accel/accel0 is not read/write for this user`
- **Cause:** XRT opens /dev/accel/accelN read/write; the user is not in its group.
- **Fix:** `sudo usermod -aG render $USER`, then log out and back in.

## RLX-XDNA-005

**Locked-memory limit too low for XRT buffers** (warning)

- **Symptom:** `XRT buffer mmap fails with EAGAIN`
- **Cause:** The distro default RLIMIT_MEMLOCK (8 MiB) is smaller than larger NPU buffers.
- **Fix:** Add `@render - memlock unlimited` to /etc/security/limits.d/rlx.conf, then log in again.

## RLX-XDNA-006

**XDNA 2 (Strix-class) NPU is not supported yet** (error)

- **Symptom:** `this is an XDNA 2 (Strix-class, AIE2P) NPU`
- **Cause:** rlx-xdna's kernels target XDNA 1 (Phoenix / Hawk Point npu1); an npu1 xclbin cannot load on AIE2P.
- **Fix:** No workaround yet. Use Device::Rocm or Device::Vulkan on the same machine; follow the XDNA 2 tracking issue.

## RLX-XDNA-007

**mlir-aie toolchain (aiecc + Peano) not found** (error)

- **Symptom:** `no mlir-aie toolchain found`
- **Cause:** On-demand NPU kernels are compiled with mlir-aie's aiecc and the Peano (llvm-aie) core compiler; neither was found in PATH, ~/.local, venvs under $HOME, /usr or /opt.
- **Fix:** `cargo rlx setup xdna` installs mlir-aie 1.4.4 with its pinned Peano into ~/.local/share/rlx/toolchains (no root). Any venv also works and is found automatically; or set AIECC and PEANO, or RLX_XDNA_TOOLCHAIN_DIR.

## RLX-XDNA-008

**XRT on LD_LIBRARY_PATH hides ROCm's HIP** (warning)

- **Symptom:** `Device::Rocm unavailable after adding XRT to LD_LIBRARY_PATH`
- **Cause:** The extracted XRT tree ships its own libxrt_hip.so; first on LD_LIBRARY_PATH, it shadows ROCm's HIP runtime.
- **Fix:** Remove the XRT directory from LD_LIBRARY_PATH and set RLX_XDNA_XRT_LIB to it instead — rlx loads XRT from there without affecting other libraries.

## RLX-XDNA-009

**mlir-aie older than 1.3 has no native aiecc** (error)

- **Symptom:** `ships only aiecc.py`
- **Cause:** Releases before 1.3.0 ship the Python driver aiecc.py only; rlx drives the native aiecc binary (no Python in the loop).
- **Fix:** Install mlir-aie 1.3.0 or newer (1.4.4 recommended) together with its pinned Peano: `pip install mlir_aie==1.4.4 -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.4` and `pip install -r utils/peano-requirements.txt` from that release.

## RLX-XDNA-010

**Peano (llvm-aie) is not the build this mlir-aie release pins** (warning)

- **Symptom:** `llvm-aie/bin/opt: …peano-compat.ll: error: expected value token`
- **Cause:** Each mlir-aie release pins one llvm-aie build (utils/peano-requirements.txt); LLVM IR from one can fail to parse in another's opt. Some mismatches work (1.4.4 runs on the 1.4.0 Peano).
- **Fix:** Install the pinned build next to mlir-aie: `pip install -r utils/peano-requirements.txt` from the matching mlir-aie tag (e.g. 1.4.4 → `llvm-aie==22.0.0.2026092101 -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly`).

## RLX-XDNA-011

**mlir-aie 1.3.0, 1.3.1 and 1.3.3 cannot compile every rlx kernel correctly** (error)

- **Symptom:** `cannot compile every rlx kernel correctly / ran out of registers / expected value token (f0x… literal)`
- **Cause:** Measured on the NPU rig against every published Peano build: on 1.3.0 softmax and attention compile but return wrong results, 1.3.1 runs out of registers on them, and 1.3.3 emits LLVM IR float literals (f0x…) that no Peano parses. rlx refuses these releases (XDNA reports unavailable) unless RLX_XDNA_ALLOW_KNOWN_BAD=1. 1.3.2, 1.3.4 and 1.4.0-1.4.4 pass the full suite.
- **Fix:** `cargo rlx setup xdna` (installs 1.4.4 with its pinned Peano), or install 1.4.4 by hand: `pip install mlir_aie==1.4.4 -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.4` and `pip install -r utils/peano-requirements.txt` from that release.

## RLX-ROCM-001

**MIOpen has no kernel database** (info)

- **Symptom:** `MIOpen has no kernel database`
- **Cause:** Distro MIOpen builds (+dfsg) ship without the precompiled kernel database, so rlx uses its own convolution kernels instead.
- **Fix:** For MIOpen convolutions, install AMD's packaged MIOpen (repo.radeon.com) or set MIOPEN_SYSTEM_DB_PATH to a directory holding its *.kdb files.

## RLX-ROCM-002

**Radeon 780M (gfx1103) is not enumerated by ROCm** (error)

- **Symptom:** `no ROCm-capable device is detected`
- **Cause:** ROCm does not list gfx1103 as supported; HSA hides it unless told to treat it as gfx1100.
- **Fix:** `export HSA_OVERRIDE_GFX_VERSION=11.0.0` before running (hipRTC then compiles for gfx1100, which runs on the 780M).

## RLX-ROCM-003

**APU GPU resets under NUMA balancing (gfx11 MES queue-eviction hang)** (info)

- **Symptom:** `HipError(719); kernel log: MES failed to respond to msg=REMOVE_QUEUE … Failed to evict process queues … GPU reset`
- **Cause:** On a multi-node host with kernel.numa_balancing on, the kernel's page sampling makes amdgpu evict a process's GPU queues, and gfx11 APU firmware (MES) can hang doing it.
- **Fix:** rlx-rocm opts its process out (MPOL_LOCAL, before hipInit). Threads started before ROCm comes up keep the default: run under `numactl --localalloc`, or `sudo sysctl kernel.numa_balancing=0`. RLX_ROCM_NUMA_GUARD=0 turns the opt-out off.

## RLX-CUDA-001

**NVIDIA driver older than CUDA 12.3** (error)

- **Symptom:** `GetProcAddress / symbol not found loading libcuda`
- **Cause:** rlx-cuda resolves the CUDA 12.3 driver API; older drivers lack symbols.
- **Fix:** Update the NVIDIA driver (`nvidia-smi` shows the CUDA version it supports).

## RLX-CUDA-002

**NVRTC (CUDA runtime compiler) not found** (error)

- **Symptom:** `libnvrtc.so: cannot open shared object file`
- **Cause:** rlx-cuda JIT-compiles its kernels with NVRTC; the driver is present but libnvrtc is not on the library path.
- **Fix:** Install it (Ubuntu: `sudo apt install cuda-nvrtc-13-1`, or the CUDA toolkit) and make it findable: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64`.

## RLX-CUDA-003

**CUDA backend built, but no NVIDIA driver loaded** (error)

- **Symptom:** `device CUDA is not available`
- **Cause:** No /dev/nvidia0: the NVIDIA kernel driver is not loaded (or no NVIDIA GPU).
- **Fix:** Install the NVIDIA driver and check with `nvidia-smi`.

## RLX-WGPU-001

**wgpu on Vulkan/DX12 runs elementwise ops on the host** (info)

- **Symptom:** `most wgpu schedule steps are host_op (RLX_DISPATCH_REPORT=1)`
- **Cause:** Deliberate: elementwise SPIR-V produced wrong model output on discrete NVIDIA, so wgpu host-routes those ops on every Vulkan/DX12 adapter (integrated ones too). Results are correct but slower.
- **Fix:** Prefer Device::Cuda or Device::Vulkan on this GPU. RLX_WGPU_SHARD_GPU=1 forces GPU elementwise if you want to measure it.

## RLX-VULKAN-001

**Vulkan backend built, but no Vulkan driver (ICD) installed** (error)

- **Symptom:** `device Vulkan is not available`
- **Cause:** No Vulkan ICD manifest in /usr/share/vulkan/icd.d or /etc/vulkan/icd.d.
- **Fix:** AMD/Intel: `sudo apt install mesa-vulkan-drivers`; NVIDIA: the proprietary driver ships its ICD. Check with `vulkaninfo --summary`.

## RLX-ENV-001

**An RLX_* variable is set that nothing reads** (warning)

- **Symptom:** `RLX_... is not a variable rlx reads`
- **Cause:** Most often a typo; the setting silently does nothing.
- **Fix:** Fix the name (`cargo rlx doctor` suggests the nearest one) or unset it.

## RLX-CHECK-001

**A device failed its check run** (error)

- **Symptom:** `BROKEN … check graph failed / disagrees with the f64 reference`
- **Cause:** The device is available but returned an error or wrong numbers for a small relu(a·b + c) graph.
- **Fix:** Rerun with RLX_DISPATCH_REPORT=1 to see what ran where, then open an issue with `cargo rlx doctor --report`.

## RLX-BUILD-001

**cargo older than 1.99 rejects the workspace manifests** (error)

- **Symptom:** ``default-features = false` cannot override workspace's `default-features``
- **Cause:** Workspace dependency overrides that cargo 1.99 accepts.
- **Fix:** `rustup update stable` (cargo >= 1.99).

## RLX-BUILD-002

**Linking libpython fails in a cargo build** (error)

- **Symptom:** `unable to find library -lpython3`
- **Cause:** A PyO3 build without `extension-module` links libpython, whose unversioned .so comes only with the -dev package.
- **Fix:** Fixed in pyrlx (extension-module is on by default). Elsewhere: `sudo apt install python3-dev`, or build Python packages with maturin.

## RLX-BUILD-003

**OpenBLAS not installed** (error)

- **Symptom:** `unable to find library -lopenblas`
- **Cause:** rlx-cpu links OpenBLAS by default on Linux x86-64.
- **Fix:** `sudo apt install libopenblas-dev`, or build rlx-cpu with `--no-default-features --features parity-gemm` (pure Rust GEMM).

## RLX-BUILD-004

**MLX submodule not checked out** (error)

- **Symptom:** `rlx-mlx-sys: vendor/mlx is empty`
- **Cause:** crates/backends/rlx-mlx-sys/vendor/mlx is a git submodule.
- **Fix:** `git submodule update --init crates/backends/rlx-mlx-sys/vendor/mlx`.

## RLX-BUILD-005

**ONNX Runtime prebuilt needs glibc 2.38+** (error)

- **Symptom:** `undefined reference to `__isoc23_strtol` (ort-sys)`
- **Cause:** The prebuilt libonnxruntime.a that ort-sys downloads targets glibc 2.38.
- **Fix:** Build on a distro with glibc >= 2.38 (Ubuntu 24.04+, Debian trixie), or disable the onnx feature.
