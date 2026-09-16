# RLX distributed node on iOS

Runs an RLX mesh worker inside an iOS app: the handset joins a cluster as a
rank, receives a stage from the coordinator, and streams activations.

## Build

```sh
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
ios/build-xcframework.sh            # CPU node
ios/build-xcframework.sh --apple    # + Metal / ANE
```

That writes `ios/build/RlxNode.xcframework`. Add `ios/` as a local Swift
package, or drag the `.xcframework` into an Xcode target.

## Demo app

A SwiftUI app that joins a mesh as a worker rank lives in `ios/Demo`.

```sh
ios/build-xcframework.sh            # once — produces ios/build/RlxNode.xcframework
cd ios/Demo && xcodegen generate    # brew install xcodegen
open RlxDemo.xcodeproj
```

Start the desktop coordinator, then tap **Join mesh**:

```sh
cargo run -p rlx-ffi --example node_coordinator -- --world 2 --peers <mac-ip>:29500
```

Enter that same `<mac-ip>:29500` as the coordinator address on the phone. A
worker only needs the coordinator's address — it dials out, and nothing dials
it back.

Verify the desktop half on its own first (no phone needed):

```sh
cargo run -p rlx-ffi --example node_coordinator -- --world 2 --self-test
```

### Headless / scripted runs

The app reads `-key value` launch arguments via `UserDefaults`, so a simulator
run needs no UI driving:

```sh
xcrun simctl launch <sim> com.mit.rlx.nodedemo \
    -rank 1 -world 2 -peers 127.0.0.1:29500 -autojoin YES
```

The simulator shares the host network stack, so `127.0.0.1` reaches a
coordinator running on the Mac.

### Linking notes

A static archive carries no link directives, so an app target must name what
the Rust code needs itself — `-framework Accelerate` (rlx-cpu's BLAS / LAPACK /
vForce), `-lresolv`, `-lc++`. `ios/Demo/project.yml` shows the full set.

## Required Info.plist / entitlements

iOS gates local-network access, and the failure mode is silent — sockets appear
to work while no peer is ever reached. Both keys are needed for a node to see
the rest of a LAN mesh:

| Key | Where | Why |
|---|---|---|
| `NSLocalNetworkUsageDescription` | Info.plist | **Required for any LAN peer traffic**, including plain TCP to a `host:port`. Without it the first connection triggers a permission prompt and, if denied or absent, connects to nothing. |
| `NSBonjourServices` | Info.plist | Only if you use mDNS discovery (`mdns` feature). List the service type, e.g. `_rlx-coord._udp`. |
| `com.apple.developer.networking.multicast` | Entitlements | Only for the **UDP broadcast discovery** path (empty `peers`). Apple grants this by request; until then, use a static peer list or unicast rendezvous. |

```xml
<key>NSLocalNetworkUsageDescription</key>
<string>Joins a nearby RLX compute cluster to share model execution.</string>
```

If you cannot get the multicast entitlement, pass an explicit peer list — that
path needs neither the entitlement nor broadcast, and is what a coordinator on
a known address gives you anyway.

## Lifecycle

A serving node holds a socket and burns battery. iOS suspends a backgrounded
process within seconds, and **a suspended rank stalls every peer waiting on
it** — the mesh has no timeout that will rescue you.

```swift
.onChange(of: scenePhase) { _, phase in
    if phase == .background { RlxNode.stop() }
}
```

`stop()` is cooperative: a node parked in `recv` exits when its peer sends or
the link drops, so `status` can report `running` briefly afterwards.

## Thermals

A phone is not a server. Sustained participation will thermally throttle the
SoC, and iOS offers no way to reserve the GPU. Prefer `device: "auto"` and let
placement give the handset proportionate work — or run it as a CPU rank.

## Known limits

- **Backgrounding is the hard constraint.** There is no background mode that
  keeps a compute node alive indefinitely; treat an iOS rank as foreground-only.
- Peer discovery over cellular does not work — nodes must share a LAN.
