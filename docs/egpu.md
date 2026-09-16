# External GPU over USB4 / Thunderbolt (`Device::Egpu`)

How to build and use `rlx-egpu`, and how to build the DriverKit extension it
talks to.

## What is actually missing on the host

PCIe tunnelling over Thunderbolt/USB4 works on Apple Silicon. A device behind the
tunnel enumerates as a normal `IOPCIDevice` and carries `IOPCITunnelled = true` —
an NVMe enclosure on a Mac mini's USB4 port appears as a bridge plus a storage
function and runs at link speed. What macOS does not ship on arm64 is a driver
for PCI base class 0x03, so a discrete GPU enumerates and is then left unclaimed.

The gap is a driver, not a bus. `rlx-egpu` fills the parts of it that do not
need a signed extension, and talks to one for the part that does.

| Stage | Feature | Needs | State |
|-------|---------|-------|-------|
| PCI discovery over the tunnel | *(default)* | nothing | complete |
| Config space / BAR / DMA transport | `dext` | a signed driver extension | complete |
| Ahead-of-time kernel packs | *(default)* | a GPU compiler, on some other host | complete |
| Signed-firmware manifest + verification | `firmware` | network, once | complete |
| AMD RDNA3/4 device bring-up | `am` | a card, to run it against | **written, never executed** |
| NVIDIA GSP bring-up | `nv` | — | reserved, not written |
| Ring submission | — | — | not started |

`is_available()` is gated on `am::VALIDATED_ON_HARDWARE`, not on
`cfg!(feature = "am")` — compiling an unexecuted driver must not make the device
a dispatch target. It returns `false` even with a supported card attached and the
transport working. Device selection will not dispatch a graph to `Device::Egpu`, and `EgpuBackend::compile` reports what is
missing instead of falling back to the CPU.

## Building

```sh
# discovery only — no dependencies, builds and runs anywhere
cargo build -p rlx-egpu
cargo run -p rlx-egpu --example egpu_probe

# with the transport (macOS) and firmware verification
cargo build -p rlx-egpu --features dext,firmware
cargo test  -p rlx-egpu --all-features

# through the runtime, registering Device::Egpu
cargo build -p rlx-runtime --features egpu
cargo build -p rlx --features egpu
just test-egpu
```

The crate has no build script and needs no SDK. `dext` pulls `libc` for
`recvmsg`/`mmap`; `firmware` pulls `sha2`. Discovery compiles on macOS (IOKit
registry) and Linux (`/sys/bus/pci`), and returns an empty bus elsewhere — which
is why the tunnel logic can be exercised on a Linux box with a real GPU on it
rather than only on the machine that needs the workaround.

### Examples

| Example | Features | What it does |
|---------|----------|--------------|
| `egpu_probe` | — | Enumerate PCI, report tunnel/root-complex and driver binding |
| `egpu_inspect` | — | Read a kernel pack or code object. **No toolchain, no driver, no device** |
| `egpu_bake` | — / `nvrtc` | Build a kernel pack (needs `hipcc`, `ptxas`, or NVRTC on the host) |
| `egpu_firmware` | `firmware` | Audit the firmware cache against the pinned manifest |
| `egpu_pci` | `dext` | Open a claimed device, read config space, allocate DMA |

## Signed firmware

Two different signatures get conflated when people ask whether the driver can be
skipped:

- A **driver extension** is signed by Apple. It governs whether a process may
  claim the PCI device. The card never sees it.
- **Firmware** is signed by AMD or NVIDIA. The PSP and the GSP bootloader verify
  it against keys fused into the die and refuse anything else.

rlx cannot produce the second kind and does not try. Those blobs are
redistributable, carried by `linux-firmware`, and bring-up consists of handing
the card its own firmware and letting it boot itself.

```sh
scripts/pull_gpu_firmware.sh --list                             # what is pinned
scripts/pull_gpu_firmware.sh gc_12_0 psp_14_0 smu_14_0 sdma_7_0 # RDNA4
scripts/pull_gpu_firmware.sh arcturus                           # MI100 / gfx908
scripts/pull_gpu_firmware.sh amd                                # every AMD family
scripts/pull_gpu_firmware.sh --check                            # verify, download nothing
```

`ids::firmware_families(vendor, device)` maps a detected card to the families it
needs, and `egpu_probe` prints them, so the fetch command follows from what is
on the bus. Firmware coverage is deliberately wider than bring-up coverage: the
MI100 (`arcturus`) has no bring-up sequence but its blobs are pinned, because a
real card is worth more as a development target than as a gap in a table.

Destination is `$RLX_FW_DIR`, else `$XDG_CACHE_HOME/rlx/firmware`, else
`~/.cache/rlx/firmware`. Each file is checked against the SHA-256 in
`crates/backends/rlx-egpu/firmware/manifest.tsv`, pinned to one linux-firmware
commit. A mismatch is a hard failure and the partial download is deleted rather
than left where a loader would find it: an unverified blob is one the PSP will
reject, and a silent bad file turns a clear checksum error into an opaque
firmware-load hang. `--check` exits non-zero if anything is missing or corrupt.

`rlx_egpu::firmware::load(name)` re-verifies before handing bytes to a caller.

## Ahead-of-time kernels

Kernels are the one part of this stack that is **not** signed. A compiled code
object is ordinary bytes: DMA it into VRAM and point the command processor at
it. Nothing checks its provenance.

So the host driving the card — which has the weakest toolchain story of any host
in the workspace — never needs a GPU compiler. Build elsewhere, carry the bytes:

```sh
# on a ROCm host
cargo run -p rlx-egpu --example egpu_bake -- -o k.rlxisa --hip k.hip:gfx1100,gfx1201

# on a CUDA host, from CUDA C++ via NVRTC (feature `nvrtc`). NVRTC compiles
# in-process and reads no system headers, so it works where the nvcc frontend
# collides with glibc's math declarations
cargo run -p rlx-egpu --features nvrtc --example egpu_bake -- -o k.rlxisa --cuda k.cu:sm_86,sm_89

# or straight from PTX. ptxas alone: no nvcc frontend, no host headers, no container
cargo run -p rlx-egpu --example egpu_bake -- -o k.rlxisa --ptx k.ptx:sm_86,sm_89

# from artifacts someone else compiled
cargo run -p rlx-egpu --example egpu_bake -- -o k.rlxisa a.hsaco b.cubin

# anywhere, including a host with no GPU toolchain at all
cargo run -p rlx-egpu --example egpu_inspect -- k.rlxisa
```

`codeobj` reads clang offload bundles (`hipcc --genco`), bare AMD code objects,
and NVIDIA cubins, naming the architecture from `e_flags` rather than the
filename. The `aot` pack is a length-prefixed record list with a checksum,
because truncation is the realistic failure for an artifact copied between
machines and a pack that silently loses a kernel is worse than one that refuses
to open.

Two things worth knowing when wiring this into a build:

- `hipcc` is **not reproducible** — two identical invocations produce different
  bytes. Content-addressing a kernel cache on the compiler output will miss every
  time; key on the source and flags instead.
- PTX for the NVIDIA path comes from NVRTC, which `rlx-cuda` already drives.
  `ptxas` then turns it into a cubin without touching a host compiler, which is
  what makes the CUDA path free of the `nvcc`-in-a-container arrangement.

## The driver extension

### Why one is needed

Claiming a PCIe device on macOS requires a DriverKit extension holding
`com.apple.developer.driverkit.transport.pci`. rlx does not have that
entitlement, ships no extension, and names none in code — the `dext` feature
talks to whichever signed extension is installed, provided it implements the
contract below.

**The entitlement is the gate, not the code.** It is granted by Apple per
development team through the developer portal, and there is no self-serve path.
The extension itself is small — a few hundred lines — and everything above it in
this crate is already written.

### A reference implementation

[`crates/backends/rlx-egpu/dext/`](../crates/backends/rlx-egpu/dext/) is a
complete, buildable extension implementing everything below, plus the socket
helper. It **builds** — `./build.sh` produces `RlxPciDriver.dext` and
`rlxpci-helper` — and has **never been loaded**, because loading it needs the
entitlement. Use it as a starting point or as an executable specification.

### What the extension must do

Match the device, hand it to userspace, and get out of the way.

**IOKit personality** (`Info.plist`):

```xml
<key>IOKitPersonalities</key>
<dict><key>YourDriver</key><dict>
  <key>IOProviderClass</key>          <string>IOPCIDevice</string>
  <key>IOClass</key>                  <string>IOUserService</string>
  <key>IOUserClass</key>              <string>YourDriver</string>
  <key>IOPCIClassMatch</key>          <string>0x03000000</string>
  <key>IOPCITunnelCompatible</key>    <true/>
  <key>IOResourceMatch</key>          <string>IOKit</string>
  <key>IOUserServerName</key>         <string>com.example.egpu.Driver</string>
  <key>YourDriverUserClientProperties</key>
  <dict>
    <key>IOClass</key>     <string>IOUserUserClient</string>
    <key>IOUserClass</key> <string>YourDriverUserClient</string>
  </dict>
</dict></dict>
```

`IOPCIClassMatch = 0x03000000` matches base class 0x03 (display).
`IOPCITunnelCompatible` is what permits attaching to a Thunderbolt-tunnelled
device; without it the personality never matches an eGPU.

**Entitlements**:

```xml
<key>com.apple.developer.driverkit</key><true/>
<key>com.apple.developer.driverkit.transport.pci</key>
<array><dict>
  <key>IOPCIPrimaryMatch</key><string>0xFFFFFFFF&amp;0x00000000</string>
</dict></array>
```

Narrow `IOPCIPrimaryMatch` to the vendor/device range you intend to drive rather
than requesting everything; Apple reviews the request, and a narrower ask is
easier to justify.

**On `Start`**: `Open()` the `IOPCIDevice`, OR `BusMaster | Memory | IOSpace`
into the PCI command register, `SetName()` to the service name rlx will look up,
then `RegisterService()`.

**User client external methods** — the selectors rlx expects:

| Selector | Name | Inputs | Outputs |
|----------|------|--------|---------|
| 0 | `ReadCfg` | `offset`, `size` (1/2/4) | value |
| 1 | `WriteCfg` | `offset`, `size`, `value` | — |
| 2 | `Reset` | — | — (function-level reset, fall back to hot reset) |
| 3 | `PrepareDMA` | struct in/out | physical segments written to the output |

**`CopyClientMemoryForType(type)`** carries two meanings, which is how BARs and
DMA memory both arrive in userspace:

- `type < 6` — map BAR `type` and return it.
- `type >= 6` — allocate `type` bytes of DMA-capable memory, run it through
  `IODMACommand` (`maxAddressBits = 40`), and write the physical scatter-gather
  table into the head of the buffer as `[addr0, len0, addr1, len1, …, 0, 0]`.

That last part is the whole point: the physical addresses are what a GPU's page
tables and ring descriptors need, and nothing else on macOS hands them out.

### The helper and its protocol

The extension's `io_connect_t` is held by a helper process, which republishes it
over a UNIX socket. rlx is a client of that socket, so it links no IOKit
userclient code and needs no entitlement of its own.

Request — 33 bytes, packed, little-endian:

| Offset | Type | Field |
|--------|------|-------|
| 0 | `u8` | command |
| 1 | `u32` | device index |
| 5 | `u32` | BAR index |
| 9 | `u64` | arg0 |
| 17 | `u64` | arg1 |
| 25 | `u64` | arg2 |

Response — 17 bytes, packed, little-endian: `u8` status (0 = ok), then two
`u64`s. On failure, status is non-zero, `resp0` is a message length, and that
many bytes of message follow.

| Cmd | Name | Request | Response |
|-----|------|---------|----------|
| 1 | `MAP_BAR` | BAR index | `resp1` = BAR size |
| 2 | `MAP_SYSMEM_FD` | arg0 = size, arg1 = contiguous | `resp0` = mapped length; descriptor over `SCM_RIGHTS` |
| 3 | `CFG_READ` | arg0 = offset, arg1 = size | `resp0` = value |
| 4 | `CFG_WRITE` | arg0 = offset, arg1 = size, arg2 = value | — |
| 5 | `RESET` | — | — |
| 6 | `MMIO_READ` | BAR, arg0 = offset, arg1 = length | response, then `length` bytes |
| 7 | `MMIO_WRITE` | BAR, arg0 = offset, arg1 = length, then bytes | *no response* |

For `MAP_SYSMEM_FD` the helper passes a shared-memory descriptor over
`SCM_RIGHTS`; rlx `mmap`s it directly, so host access to DMA memory is zero-copy.
The physical segment table sits at the head of that mapping and is parsed once.

The helper must accept `server <socket-path>` as arguments — that is how rlx
starts one when the socket is not already live. It may serve a single client at a
time; rlx reuses a running helper rather than starting a second.

### Signing, notarizing, installing

```sh
# build (Xcode project or xcodebuild)
xcodebuild -project YourDriver.xcodeproj -scheme YourDriver -configuration Release

# sign the dext with the entitlements, then the containing app
codesign --force --sign "Developer ID Application: …" \
         --entitlements YourDriver.entitlements \
         --options runtime YourDriver.dext
codesign --force --sign "Developer ID Application: …" \
         --entitlements app.entitlements --options runtime YourApp.app

# notarize the containing app, then staple
xcrun notarytool submit YourApp.zip --keychain-profile "AC" --wait
xcrun stapler staple YourApp.app

cp -R YourApp.app /Applications && /Applications/YourApp.app/Contents/MacOS/YourApp install
```

Enable it under **System Settings → General → Login Items & Extensions → Driver
Extensions**. Installing prompts for this; if the prompt is missed, the toggle is
there.

**Iterating before the entitlement is granted:** `systemextensionsctl developer on`
plus a machine with SIP partially disabled lets an unnotarized, locally-signed
extension load. That is a development arrangement, not a distribution one — a
machine in that state should not be one you care about.

### Verifying it works

```sh
systemextensionsctl list                     # is the extension staged and enabled
ioreg -c IOPCIDevice -r -l | grep -i tunnel  # is a display-class device on the tunnel
log stream --predicate 'sender CONTAINS "YourDriver"'   # driver-side logging

cargo run -p rlx-egpu --example egpu_probe   # does rlx see the device
cargo run -p rlx-egpu --features dext --example egpu_pci   # can rlx open it
```

`egpu_pci` reads vendor and device ID out of config space and allocates a DMA
buffer. Getting the right IDs back is the end-to-end check that the whole path —
extension, helper, socket, transport — works.

### Pointing rlx at your extension

Three identifiers name the external component, and all three are its published
names rather than rlx's. `DextConfig` layers them, most specific first: a builder
value, then the environment variable, then the built-in default.

| | Builder | Environment | Config file |
|---|---|---|---|
| helper executable path | `.helper(..)` | `RLX_EGPU_HELPER` | `helper` |
| IOService name it publishes | `.service(..)` | `RLX_EGPU_SERVICE` | `service` |
| UNIX socket it listens on | `.socket(..)` | `RLX_EGPU_SOCK` | `socket` |

The fourth layer is a config file — `RLX_EGPU_CONFIG`, else
`$XDG_CONFIG_HOME/rlx/egpu.conf`, else `~/.config/rlx/egpu.conf` — read between
the environment and the defaults:

```text
# ~/.config/rlx/egpu.conf
helper  = /Applications/YourApp.app/Contents/MacOS/YourApp
service = yourdriver
```

```rust
use rlx_egpu::dext::{DextConfig, PciTransport};

let config = DextConfig::builder()
    .helper("/Applications/YourApp.app/Contents/MacOS/YourApp")
    .service("yourdriver")
    .build();
let mut gpu = PciTransport::connect_with(&config)?;
let (vendor, device) = gpu.identity()?;
```

Leave the socket unset and it follows the service name, so pointing at a second
extension cannot silently reuse the first one's socket.
`DextConfig::default()` is the built-in set with the environment ignored, for
embedders and tests that need a predictable configuration;
`DextConfig::from_env()` is the layered form `PciTransport::connect()` uses.

The example takes the same three as flags:

```sh
cargo run -p rlx-egpu --features dext --example egpu_pci -- \
    --helper /Applications/YourApp.app/Contents/MacOS/YourApp --service yourdriver
```

## The bring-up module

`rlx_egpu::am` (feature `am`) is the AMD bring-up. It is split by what can be
decided without a device:

| Module | Kind | Verified |
|--------|------|----------|
| `am::pm4` | PM4 packet encoding | yes — unit tested |
| `am::pt` | GPUVM page-table math | yes — unit tested |
| `am::psp` | signed-firmware load sequence | no — needs a card |
| `am` | IP bring-up order | no — needs a card |

`am::plan()` reports each stage and what it is still missing;
`am::first_blocked_stage()` is where work restarts once hardware is available
(the IP discovery table, which every later stage needs for firmware filenames
and register offsets). `am::bring_up()` refuses rather than running a partial
sequence: stopping midway leaves the card needing a bus reset, and on a
Thunderbolt-attached device that can take the tunnel down with it.

Packet encoding and page-table arithmetic are written properly and tested. The
sequencing is written as structure with the register access it needs named, and
it does not poke registers whose offsets were never confirmed — a wrong offset
writes a live register and surfaces somewhere unrelated.

## What is left before a graph runs

The transport gives config space, BARs, and DMA-able memory with known physical
addresses — the Linux equivalent of `/sys/bus/pci/devices/*` plus `/dev/vfio`.
Turning that into a device that executes kernels means loading the signed
firmware through the PSP (or booting GSP), starting the memory controller, SMU,
and GFX/SDMA rings, and building GPUVM page tables. In the reference
implementation that is ~1,000 lines of register sequencing for AMD and ~800 for
NVIDIA, on top of ~900 lines of shared memory/queue code and tens of thousands of
lines of *generated* register tables. It is the `am` feature, which is not
written.

It also cannot be written blind. Bring-up fails by hanging on a firmware
handshake, and the only way to find that is to run it on an attached card.

## Cost model

The link is a PCIe x4-class tunnel. DMA buffers are shared by descriptor and
mapped directly, so host memory access is zero-copy; BAR access is a socket
round-trip into the helper, on the order of tens of microseconds per MMIO
operation. Anything built on this must keep command buffers in DMA memory and
touch BARs only for doorbells. Resident weights and few large kernels survive the
transport; per-kernel host traffic does not.

## Supported devices

`ids::is_supported` covers the parts with a known userspace bring-up sequence:
AMD Navi 31/33 (RDNA3), Navi 44/48 (RDNA4) and MI300X, matched exactly; NVIDIA
GA10x through GB20x (Ampere through Blackwell), matched by device-ID family.
Membership means the register interface and firmware sequence are known — not
that rlx can execute on it.
## License

MIT OR Apache-2.0.
