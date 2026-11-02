# Local patches to the MLX submodule

`vendor/mlx` is a **git submodule** pinned to a clean upstream commit, so a
local fix cannot live in the tree: `git submodule update` discards it and the
parent repo never records it. These patches are tracked here instead and applied
to the submodule's working tree by `build.rs` before CMake configures.

`build.rs` applies them in filename order and is idempotent — it tries
`git apply --check --reverse` first, and a patch that applies backwards is
already in the tree. A patch that applies **neither** way is a hard error naming
the submodule commit, not a silent skip: the failure it would otherwise cause is
a metallib stamped for the wrong platform, which does not surface until Metal
refuses to load it on a device.

To bump the submodule, run `just update-mlx [REF]` (`scripts/update-mlx.sh`;
default ref is the newest release tag, `main` for the tip). It takes the patches
out, moves the pin, re-applies them and names any that no longer fit. Re-base
those by hand:

```sh
cd vendor/mlx
git apply --3way ../../patches/0001-metal-target-sdk.patch
# resolve, then re-export just that patch's files
git diff -- CMakeLists.txt mlx/backend/metal/kernels/CMakeLists.txt \
    > ../../patches/0001-metal-target-sdk.patch
```

## 0001 — build the metallib for the target platform

Upstream hardcodes `xcrun -sdk macosx metal` and `-mmacosx-version-min=` for
both the `.air` compile and the metallib link, so an iOS / tvOS / visionOS
cross-build still emits a **macOS** metallib. A metallib carries its platform in
its header, and the Metal runtime refuses one built for a different platform —
at first use, not at build time. `MLX_METAL_JIT` does not rescue it either: it
shrinks the metallib rather than removing it, and `device.cpp` loads that file
or nothing.

The patch derives the SDK from `CMAKE_SYSTEM_NAME` plus simulator-ness of the
sysroot (CMake may have expanded `CMAKE_OSX_SYSROOT` from a name to a full path
by then), and picks the matching version-min spelling. visionOS is the odd one:
there is no `-mxros-version-min`, it takes `-mtargetos=xros<ver>[-simulator]`.

It also adds `tvOS` and `visionOS` to the branch that keeps `MLX_BUILD_METAL` on.
**watchOS is deliberately absent** — no public Metal API, so it falls through to
the `OFF` branch.

Verified platform byte at offset `0x0B` of the produced metallib:

| target | byte | | target | byte |
|---|---|---|---|---|
| macOS | `81` | | iOS sim | `87` |
| iOS | `82` | | tvOS sim | `88` |
| tvOS | `83` | | visionOS sim | `8c` |
| visionOS | `8b` | | | |

## 0002 — no process spawning on tvOS / visionOS

MLX's CPU backend compiles kernels at runtime by shelling out to `g++`. tvOS,
watchOS and visionOS mark both `system()` and `popen()` unavailable, and the
build fails on the *capability probe* itself:

```
jit_compiler.cpp:212:28: error: 'system' is unavailable: not available on tvOS
```

The patch makes `JitCompiler::available()` answer `false` on those platforms
without asking, and keeps `exec()` compiling. MLX already treats an unavailable
JIT compiler as a supported state and falls back to its non-compiled CPU
kernels, so this costs a fast path that could never have existed there.
## 0003 — CUDA JIT header roots

MLX's NVRTC JIT looks for CUDA headers under `CUDA_HOME` / `CUDA_PATH` or a
hard-coded `/usr/local/cuda`, and for the CCCL headers next to its own
binary (`<bin>/../include/cccl`). A static `libmlx` linked into a Rust
binary is never in that install tree, and `/usr/local/cuda` may be a
different toolkit than the one MLX was built with — on CUDA 13 its CCCL
headers also moved under `include/cccl`, so kernels fail with `cannot open
source file "cuda/std/tuple"`.

The patch points `MLX_CCCL_DIR` at the CCCL copy MLX installs (outside test
builds too), and adds `MLX_CUDA_TOOLKIT_DIR`, which `build.rs` sets to the
toolkit it chose, as the JIT's default header root. `CUDA_HOME` /
`CUDA_PATH` still win.

## License

MIT OR Apache-2.0.
