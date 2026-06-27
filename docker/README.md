# Linux validation for `rlx-vulkan`

A Docker image that validates the native Vulkan backend on Linux using Mesa
**lavapipe** — a software (CPU) Vulkan implementation — so it runs with **no
physical GPU** (CI-friendly). The same image validates against a real GPU when
you pass one through.

What it runs (`run-validation.sh`):
1. `vulkaninfo --summary` — show the selected Vulkan device.
2. `cargo test -p rlx-vulkan` — backend unit smoke tests.
3. `cargo test -p rlx-runtime --features vulkan --test vulkan_parity` — the
   Vulkan ↔ CPU cross-backend parity suite (matmul, softmax, norms, RoPE,
   attention, gather, transpose, reduce, elementwise).

The source tree is **mounted**, not copied, so the image is tiny and always
tests your current checkout.

## Quick start

```bash
# from the repo root
docker build -t rlx-vulkan-linux ./docker
docker run --rm -v "$PWD":/work rlx-vulkan-linux
```

Or with caching volumes (fast repeat runs) via compose:

```bash
docker compose -f docker/docker-compose.yml run --rm validate
```

## Notes

- **lavapipe** is a conformant CPU Vulkan device, so a green parity run is real
  evidence the SPIR-V kernels and push-constant (std430) layout are correct —
  independent of the GLSL→SPIR-V toolchain on the host.
- Build artifacts land in the `/target` volume (or a container-local dir),
  never in the host's macOS `target/`.
- Apple Silicon hosts run the `arm64` image; `run-validation.sh` globs the
  arch-correct `lvp_icd.*.json`, so x86_64 and aarch64 both work.

## Validate against a real Linux GPU

Pass the DRI device and a vendor ICD instead of lavapipe:

```bash
docker run --rm \
  --device /dev/dri \
  -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  -e VK_LOADER_DRIVERS_SELECT= \
  -v "$PWD":/work \
  rlx-vulkan-linux
```

(Install the matching vendor driver / `nvidia-container-toolkit` on the host.)
