# rlx-driver

Driver layer for RLX — `Device` enum + cross-cutting types (arenas,
buffers, command streams).

This crate sits between `rlx-ir` (vocabulary) and the per-backend crates
(execution). Backends consume the types here so the runtime can dispatch
to any of them through a uniform interface.

## What's here

- **`Device`** — `Cpu`, `Metal`, `Mlx`, `Ane`, `Cuda`, `Rocm`, `Tpu`,
  `Gpu` (wgpu), `Vulkan`, `OpenGl`, `DirectX`, `WebGpu`, `Fpga`. Used
  by the runtime to pick a backend and by tests to pin graphs to a
  specific path.
- **`registry`** — process-global table of which `Device` variants have
  a registered backend (set at runtime, queried by
  `rlx_runtime::Device::is_available`).
- **Collective primitives** — the `Transport` trait + `ProcessGroup`
  (`all_reduce` / `all_gather` / `broadcast` / `barrier`) with `TcpTransport`
  (mesh) and `NetTransport` (star) for distributed / multi-stream scenarios
  that the runtime composes on top.
- **`IrohTransport`** (feature `iroh`) — a NAT-traversing `Transport` over
  iroh QUIC (relays + pkarr/DNS discovery): ranks addressed by `EndpointId`,
  no `ip:port`. See [`../../../docs/iroh-transport.md`](../../../docs/iroh-transport.md).

## Install

```toml
[dependencies]
rlx-driver = "0.2"
```

Usually pulled in transitively via [`rlx-runtime`](https://crates.io/crates/rlx-runtime)
or [`rlx`](https://crates.io/crates/rlx).

## License

MIT OR Apache-2.0.