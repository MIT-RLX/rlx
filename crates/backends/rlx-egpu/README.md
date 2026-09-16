# rlx-egpu

External GPU over USB4/Thunderbolt on Apple Silicon — PCI discovery and a
driver-extension transport.

## The gap this addresses

PCIe tunnelling works on Apple Silicon. A device behind a Thunderbolt/USB4
tunnel enumerates as a normal `IOPCIDevice` with `IOPCITunnelled = true` — an
NVMe enclosure on a Mac mini's USB4 port shows up as a bridge plus a storage
function and runs at link speed. What macOS does not ship on arm64 is a driver
for PCI base class 0x03, so a discrete GPU enumerates and is left unclaimed.

The missing piece is a driver, not a bus.

**[`docs/egpu.md`](../../../docs/egpu.md)** is the full guide — building the
crate, the firmware manifest, the ahead-of-time kernel path, and **how to build
and sign the DriverKit extension** this talks to (entitlement, IOKit personality,
user-client contract, helper wire protocol, notarizing, installing).

```sh
just test-egpu                  # crate tests + the Device::Egpu runtime seam
just egpu-probe                 # what is on the PCIe tunnel right now
just egpu-firmware gc_12_0      # fetch + verify signed firmware for an IP family
just egpu-inspect k.rlxisa      # read a kernel pack — no GPU toolchain needed
```

## Stages

| Stage | Feature | State |
|-------|---------|-------|
| PCI discovery over the tunnel | *(default)* | complete |
| Config space / BAR / DMA transport | `dext` | complete |
| Ahead-of-time kernel packs | *(default)* | complete |
| Signed-firmware manifest + verification | `firmware` | complete |
| AMD RDNA3/4 device bring-up | `am` | **written, never executed** |
| NVIDIA GSP bring-up | `nv` | reserved, not written |
| Ring submission | — | not started |

`am` splits by what can be decided without a device: `am::pm4` (packet encoding)
and `am::pt` (page-table math) are pure computation and unit tested; `am::psp`
and the IP sequencing are written but have never run. `am::bring_up()` refuses
rather than starting a sequence it cannot finish. A reference DriverKit
extension lives in [`dext/`](dext/) — it builds, and has never been loaded.

`is_available()` is gated on the bring-up stage, so it reports `false` even with
a supported card attached and the transport working. Device selection will not
dispatch a graph here.

```sh
cargo run -p rlx-egpu --example egpu_probe
cargo run -p rlx-egpu --features dext --example egpu_pci
cargo run -p rlx-egpu --example egpu_inspect -- kernels.rlxisa
cargo run -p rlx-egpu --features firmware --example egpu_firmware
```

## Two signatures, doing different jobs

They get conflated whenever someone asks whether the driver can be skipped:

- A **driver extension** is signed by Apple. It governs whether a process may
  claim the PCI device. The card never sees it.
- **Firmware** is signed by AMD or NVIDIA. The PSP and the GSP bootloader verify
  it against keys fused into the die and refuse anything else.

rlx cannot produce the second kind and does not try. Those blobs are
redistributable, carried by `linux-firmware`, and bring-up consists of handing
the card its own firmware and letting it boot itself:

```sh
scripts/pull_gpu_firmware.sh --list
scripts/pull_gpu_firmware.sh gc_12_0 psp_14_0 smu_14_0 sdma_7_0   # RDNA4
scripts/pull_gpu_firmware.sh --check                              # verify, fetch nothing
```

Each file is checked against the SHA-256 in `firmware/manifest.tsv`, pinned to
one linux-firmware commit. A mismatch is a hard failure — an unverified blob is
one the PSP will reject, and a silent bad file turns a clear checksum error into
an opaque firmware-load hang.

## Kernels are not signed — so compile them elsewhere

A compiled code object is ordinary bytes: DMA it into VRAM and point the command
processor at it. Nothing checks its provenance. That removes the worst
dependency in this stack, because the host driving the card has the weakest
toolchain story — macOS has no ROCm, and the NVIDIA route otherwise runs `nvcc`
in a Linux container.

So build on a machine that has a compiler and carry the bytes:

```sh
# on a ROCm host
cargo run -p rlx-egpu --example egpu_bake -- -o k.rlxisa --hip k.hip:gfx1100,gfx1201
# on a CUDA host — ptxas alone, no nvcc frontend, no host headers
cargo run -p rlx-egpu --example egpu_bake -- -o k.rlxisa --ptx k.ptx:sm_86,sm_89
# anywhere, including a host with no GPU toolchain at all
cargo run -p rlx-egpu --example egpu_inspect -- k.rlxisa
```

`codeobj` reads clang offload bundles, AMD code objects and NVIDIA cubins,
naming the architecture from `e_flags` rather than the filename. The `aot` pack
is a length-prefixed record list with a checksum, because truncation is the
realistic failure for an artifact copied between machines and a pack that
silently loses a kernel is worse than one that refuses to open.

## Two external dependencies

**The entitlement.** Claiming a PCIe device on macOS requires a DriverKit
extension holding `com.apple.developer.driverkit.transport.pci`, which Apple
grants per development team. rlx does not have it, ships no extension, and names
none in code. The `dext` feature speaks to whichever signed extension is
installed on the host, provided it is the thin kind: matching
`IOPCIClassMatch = 0x03000000` with `IOPCITunnelCompatible`, and exposing config
reads/writes, BAR mappings, function-level reset, and `IODMACommand` buffers
whose physical scatter-gather addresses are handed back to userspace.

Three identifiers name that external component, and all three are its published
names rather than ours. `DextConfig` layers them, most specific first: a value
set through the builder, then the environment variable, then the built-in
default. Swapping extensions is configuration, not a code change.

| | Builder | Environment |
|---|---|---|
| helper executable path | `.helper(..)` | `RLX_EGPU_HELPER` |
| IOService name it publishes | `.service(..)` | `RLX_EGPU_SERVICE` |
| UNIX socket it listens on | `.socket(..)` | `RLX_EGPU_SOCK` |

```rust
let config = DextConfig::builder()
    .helper("/opt/acme/bin/pcie-helper")
    .service("acmepci")
    .build();
let gpu = PciTransport::connect_with(&config)?;
```

Leave the socket unset and it follows the service name, so pointing at a second
extension cannot silently reuse the first one's socket. `DextConfig::default()`
is the built-in set with the environment ignored, for embedders and tests that
need a predictable configuration; `DextConfig::from_env()` is the layered form
`PciTransport::connect()` uses.

Owning this end to end means applying for the entitlement and shipping a dext of
our own; the transport above it would not change.

**The bring-up.** Config space, BARs, and DMA-able memory with known physical
addresses are the Linux equivalent of `/sys/bus/pci/devices/*` plus `/dev/vfio`.
Turning that into a device that runs kernels means loading signed firmware
through the PSP (or booting GSP), starting the memory controller, SMU, and
GFX/SDMA rings, and building GPUVM page tables — several thousand lines of
register sequencing per architecture. tinygrad's `AM` driver is the reference
for scale.

## Cost model

The link is a PCIe x4-class tunnel. DMA buffers are shared by file descriptor
and mapped directly, so host memory access is zero-copy; BAR access is a socket
round-trip into the process holding the driver connection, on the order of tens
of microseconds per MMIO operation. Anything built on this must keep command
buffers in DMA memory and touch BARs only for doorbells. Resident weights and
few large kernels survive the transport; per-kernel host traffic does not.

## Supported devices

`ids::is_supported` mirrors the device sets a working userspace bring-up covers:
AMD Navi 31/33 (RDNA3), Navi 44/48 (RDNA4), MI300X — matched exactly; NVIDIA
GA10x through GB20x (Ampere through Blackwell) — matched by device-ID family.
Membership means the register interface and firmware sequence are known, not
that rlx can execute on it.
## License

MIT OR Apache-2.0.
