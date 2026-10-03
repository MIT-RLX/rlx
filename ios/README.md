# RLX distributed node on Apple platforms

Runs an RLX mesh worker inside an Apple app: the device joins a cluster as a
rank, receives a stage from the coordinator, and streams activations. iPhone and
iPad, Apple TV, Apple Watch and Apple Vision Pro are all ranks like any other.

The directory is still called `ios/` because that is where it started; nothing
in it is iPhone-specific any more.

## Platform matrix

| Platform | Backends | Slice | Notes |
|---|---|---|---|
| iOS | CPU (Accelerate/AMX), Metal, CoreML/ANE, MLX | `ios-arm64`, `ios-arm64_x86_64-simulator` | the widest surface |
| tvOS | CPU, Metal, CoreML/ANE, MLX | `tvos-arm64`, `tvos-arm64-simulator` | the tvOS **simulator** SDK ships MPS without MPSGraph, so the MPSGraph fast path is off there (device is unaffected) |
| visionOS | CPU, Metal, CoreML/ANE, MLX | `xros-arm64`, `xros-arm64-simulator` | Apple's SDK calls it `xros`; Rust calls it `visionos` |
| watchOS | **CPU/Accelerate only** | `watchos-arm64`, `watchos-arm64-simulator` | no public Metal API, and no CoreML runtime model-compile |

MLX builds for every Metal-capable Apple platform, device and simulator. It
needs two local patches to the MLX submodule (`crates/backends/rlx-mlx-sys/
patches/`): upstream hardcodes `xcrun -sdk macosx` when building its metallib,
which silently produces a macOS metallib that Metal refuses to load anywhere
else, and its CPU backend calls `system()`/`popen()`, which tvOS and visionOS do
not have. MLX on watchOS is not a gap to close — there is no Metal there at all.

**MLX needs its kernel library in the app bundle.** `mlx.metallib` is looked up
next to the executable at the first GPU kernel; a static archive cannot carry it,
so `ios/build-xcframework.sh --apple` emits it to `build/metallib/<target>/` for
the app target to copy in. Without it MLX links, reports itself available, and
fails at first use with `Failed to load the default metallib`.

**Validated on hardware** (iPad Pro 11-inch, iOS 26.5.2): Metal, CoreML/ANE, MLX
and wgpu all match the CPU reference at `max|Δ| = 1.49e-8`. `just test-apple-sim`
cannot check those three — a headless `simctl spawn` exposes no Metal device, so
they skip there. tvOS, watchOS and visionOS hardware is still untested.

Everything above is verified by two gates: `just check-apple` cross-compiles
each one, and `just test-apple-sim` **runs** the backend smoke + parity tests on
all four simulators.

## Build

```sh
ios/build-xcframework.sh            # CPU node, every Apple OS
ios/build-xcframework.sh --apple    # + Metal / ANE
ios/build-xcframework.sh --platforms ios,visionos
ios/build-xcframework.sh --all-archs        # + arm64_32 watch, x86_64 sims
ios/build-xcframework.sh --profile release  # stock release instead of `apple`
```

That writes `ios/build/RlxNode.xcframework` with a device and a simulator slice
per platform. Add `ios/` as a local Swift package, or drag the `.xcframework`
into an Xcode target.

Rust targets are installed on demand; a platform whose targets cannot be
installed is skipped with a note rather than failing the build.

### The `apple` cargo profile

The default build uses `[profile.apple]` (workspace `Cargo.toml`), not the
stock release profile. It is release plus LTO, one codegen unit and
`panic = "abort"`.

`panic = "abort"` is the big one and it is not a behaviour change: every entry
point in `rlx-ffi` is `extern "C"`, and a panic across that boundary already
aborts. Dropping the unwinder takes `__eh_frame`, `__gcc_except_tab` and
`__unwind_info` with it.

`opt-level` deliberately stays at 3. This is numeric kernel code on a
thermally-limited device; trading throughput for bytes is the wrong trade, and
the size comes from the linker and the dependency graph instead.

### Architecture and deployment-target limits

- **arm64 only by default** on tvOS, watchOS and visionOS. Their x86_64
  simulator targets and watchOS's 32-bit `arm64_32` are Rust tier 3 — no
  prebuilt `std` — so they need a nightly toolchain and a from-source std.
  `--all-archs` turns that on: it adds `arm64_32-apple-watchos` (Apple Watch
  Series 4–8, which cannot load the arm64 slice at all), `x86_64-apple-tvos`
  and `x86_64-apple-watchos-sim` (Intel Macs). iOS keeps its fat
  arm64 + x86_64 simulator slice either way.
- There is **no x86_64 visionOS simulator**: rustc has no such target, and the
  Vision Pro simulator requires an Apple Silicon Mac regardless.
- **watchOS floor is 26.0.** rustup's prebuilt `std` for
  `aarch64-apple-watchos` is itself compiled at 26.0; a lower deployment target
  is accepted by rustc and then warned about by the linker, object by object.
  `--build-std` rebuilds std at whatever `WATCHOS_DEPLOYMENT_TARGET` you set,
  which is the only way below it.
- **iOS and tvOS floors are 17.0**, because that is where Metal gained native
  `bfloat` and MLX's kernels use it. Measured against the Metal compiler:
  `bfloat` is an unknown type at iOS 16.0 and tvOS 16.0 and available from 17.0
  on both; visionOS 1.0 already shipped past that line. Below the floor MLX
  does not fail on `bfloat` alone — it falls back to its emulated bf16 header
  and the build dies in a wall of `redefinition of 'abs'`. A CPU-only node
  would be happy at 15.0; one floor for both keeps the xcframework's slices
  from disagreeing.
- visionOS is 1.0. All floors are set by the build script and mirrored in
  `Package.swift` and `Demo/project.yml`. Change one, change all three.

## Demo apps

`ios/Demo` is one XcodeGen project with four app targets sharing one source
tree — `RlxDemo` (iOS), `RlxDemoTV`, `RlxDemoWatch`, `RlxDemoVision`.

```sh
ios/build-xcframework.sh            # once — produces ios/build/RlxNode.xcframework
cd ios/Demo && xcodegen generate    # brew install xcodegen
open RlxDemo.xcodeproj
```

Start the desktop coordinator, then tap **Join mesh**:

```sh
cargo run -p rlx-ffi --example node_coordinator -- --world 2 --peers <mac-ip>:29500
```

Enter that same `<mac-ip>:29500` as the coordinator address on the device. A
worker only needs the coordinator's address — it dials out, and nothing dials
it back.

Verify the desktop half on its own first (no device needed):

```sh
cargo run -p rlx-ffi --example node_coordinator -- --world 2 --self-test
```

The shared `NodeView` carries three `#if os(...)` seams, all of them real
SwiftUI differences rather than styling: `.pickerStyle(.segmented)` exists on
neither tvOS nor watchOS, `.navigationViewStyle(.stack)` is wrong on visionOS,
and watchOS has no `keyboardType` (its rows stack instead of sitting
side-by-side).

### Headless / scripted runs

The apps read `-key value` launch arguments via `UserDefaults`, so a simulator
run needs no UI driving:

```sh
xcrun simctl launch <sim> com.mit.rlx.nodedemo \
    -rank 1 -world 2 -peers 127.0.0.1:29500 -autojoin YES
```

Bundle IDs are `com.mit.rlx.nodedemo` plus `.tv`, `.watch` and `.vision`. The
simulator shares the host network stack, so `127.0.0.1` reaches a coordinator
running on the Mac.

### Linking notes

A static archive carries no link directives, so an app target must name what
the Rust code needs itself — `-framework Accelerate` (rlx-cpu's BLAS / LAPACK /
vForce), `-lresolv`, `-lc++`. `ios/Demo/project.yml` shows the full set, shared
across all four targets.

The non-iOS targets also pin `ARCHS: arm64`. Without it Xcode builds a fat
simulator binary, finds no x86_64 in the arm64-only slice, and buries the real
error under hundreds of `ignoring file ... found architecture 'arm64'` lines.

## Required Info.plist / entitlements

iOS, tvOS and visionOS gate local-network access, and the failure mode is
silent — sockets appear to work while no peer is ever reached. Both keys are
needed for a node to see the rest of a LAN mesh:

| Key | Where | Why |
|---|---|---|
| `NSLocalNetworkUsageDescription` | Info.plist | **Required for any LAN peer traffic**, including plain TCP to a `host:port`. Without it the first connection triggers a permission prompt and, if denied or absent, connects to nothing. |
| `NSBonjourServices` | Info.plist | Only if you use mDNS discovery (`mdns` feature). List the service type, e.g. `_rlx-coord._udp`. |
| `com.apple.developer.networking.multicast` | Entitlements | Only for the **UDP broadcast discovery** path (empty `peers`). Apple grants this by request; until then, use a static peer list or unicast rendezvous. |
| `WKApplication` | Info.plist (watchOS) | Marks a single-target watch app — one with no companion iOS app. |

```xml
<key>NSLocalNetworkUsageDescription</key>
<string>Joins a nearby RLX compute cluster to share model execution.</string>
```

If you cannot get the multicast entitlement, pass an explicit peer list — that
path needs neither the entitlement nor broadcast, and is what a coordinator on
a known address gives you anyway.

## Lifecycle

A serving node holds a socket and burns battery. Every one of these OSes
suspends a backgrounded process within seconds, and **a suspended rank stalls
every peer waiting on it** — the mesh has no timeout that will rescue you.

```swift
.onChange(of: scenePhase) { _, phase in
    if phase == .background { RlxNode.stop() }
}
```

`stop()` is cooperative: a node parked in `recv` exits when its peer sends or
the link drops, so `status` can report `running` briefly afterwards.

## Thermals

None of these is a server. Sustained participation will thermally throttle the
SoC, and no Apple OS offers a way to reserve the GPU. Prefer `device: "auto"`
and let placement give each device proportionate work — or run it as a CPU
rank. A watch is a CPU rank by construction.

## Known limits

- **Backgrounding is the hard constraint.** There is no background mode that
  keeps a compute node alive indefinitely; treat any of these as foreground-only.
- Peer discovery over cellular does not work — nodes must share a LAN.
- **iOS is hardware-validated; tvOS / watchOS / visionOS are not.** The
  simulators execute CPU and CoreML/ANE only — a headless `simctl spawn` exposes
  no Metal device, so Metal, MLX and wgpu skip there and their parity assertions
  do not actually run. Treat a green simulator run as covering CPU and ANE.
- A watch reaches the LAN through its paired phone unless it is on Wi-Fi
  directly; an unpaired, Wi-Fi-less watch has no path to a coordinator.
