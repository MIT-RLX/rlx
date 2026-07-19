# rlx-hwprofile

Host hardware profiler for RLX. Probes the local machine for compute backends,
GPUs, and VRAM so RLX can do VRAM-aware device selection and feed a topology
planner.

All probes shell out to standard vendor tools via `std::process::Command` and
degrade gracefully when a tool is missing — a host with no GPU tooling still
returns a valid profile that at minimum reports the CPU backend.

## Probes

| Platform        | RAM                     | GPU / VRAM                                                        |
| --------------- | ----------------------- | ---------------------------------------------------------------- |
| macOS / Apple   | `sysctl -n hw.memsize`  | Metal, unified memory = total RAM; name via `system_profiler`/`sysctl` |
| Linux / Windows | `/proc/meminfo` (Linux) | `nvidia-smi` (CUDA), `rocminfo` (ROCm), `vulkaninfo` (best-effort) |

CPU is always reported as available.

## Usage

```rust
let profile = rlx_hwprofile::detect();
println!("primary backend: {:?}", profile.primary_backend());
if let Some(vram) = profile.largest_gpu_vram_bytes() {
    println!("largest GPU VRAM: {} bytes", vram);
}
for gpu in &profile.gpus {
    // Build a topology `NodeSpec { vram_bytes: gpu.vram_bytes }` here.
    println!("{} — {:?} — {} bytes", gpu.name, gpu.backend, gpu.vram_bytes);
}
```

The public types (`BackendKind`, `GpuInfo`, `HostProfile`) are `serde`-serializable
so a profile can be persisted or shipped to a topology planner. This crate depends
on no RLX backend or planner crate.

## Caveats

- `nvidia-smi` reports `memory.total` in MiB; converted to bytes.
- `rocminfo` VRAM is approximated from the largest memory pool `Size:` (KiB).
- `vulkaninfo --summary` does not expose VRAM, so Vulkan-only devices report
  `vram_bytes == 0`; software rasterizers (llvmpipe/lavapipe/swiftshader) and
  devices already found by a more specific probe are skipped.
- On Apple Silicon, unified memory means the Metal device's usable "VRAM" equals
  total system RAM.
