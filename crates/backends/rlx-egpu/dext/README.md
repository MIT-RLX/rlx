# Reference PCIe driver extension

A minimal DriverKit extension that claims a display-class device on a
Thunderbolt/USB4 PCIe tunnel and hands it to userspace, plus the helper that
republishes the connection over a UNIX socket.

This is the component `rlx-egpu`'s `dext` feature talks to. rlx ships no signed
extension — this is source you can build, sign with your own team's entitlement,
and point rlx at. See [`docs/egpu.md`](../../../../docs/egpu.md) for the whole
picture.

## Status

The extension **has never been loaded**. Building it needs the DriverKit SDK;
loading it needs `com.apple.developer.driverkit.transport.pci`, which Apple
grants per development team and which this tree does not have. The code is
reviewed and complete against the documented contract, and it has not been run.

The helper builds and runs anywhere — it just fails to find a service.

## Files

| File | What |
|------|------|
| `Info.plist` | IOKit personality: `IOPCIClassMatch = 0x03000000` + `IOPCITunnelCompatible` |
| `RlxPciDriver.entitlements` | The restricted PCI entitlement, scoped to AMD and NVIDIA vendors |
| `RlxPciDriver.iig` / `.cpp` | Claims the device, enables bus mastering, exposes config/BAR/DMA/reset |
| `RlxPciDriverUserClient.iig` / `.cpp` | The four selectors and `CopyClientMemoryForType` |
| `helper/helper.c` | Holds the connection, serves the socket protocol |
| `build.sh` | Builds both |

## Building

```sh
./build.sh helper     # no entitlement, no Xcode needed
./build.sh dext       # needs full Xcode (DriverKit SDK)
RLX_CODESIGN_IDENTITY="Developer ID Application: …" ./build.sh dext
```

**Your editor will show errors in the `.cpp` files.** `RlxPciDriver.h` and
`RlxPciDriverUserClient.h` do not exist in the tree — `iig` generates them from
the `.iig` files during the build, and the DriverKit SDK headers are not on
clang's default include path. That is expected, not a broken checkout.

## Installing

The extension has to be delivered inside an application bundle and activated
through `OSSystemExtensionManager`; there is no way to install a `.dext`
directly. The containing app needs the
`com.apple.developer.system-extension.install` entitlement and roughly this:

```swift
import SystemExtensions

let request = OSSystemExtensionRequest.activationRequest(
    forExtensionWithIdentifier: "org.rlx.egpu.RlxPciDriver",
    queue: .main)
request.delegate = delegate          // must implement OSSystemExtensionRequestDelegate
OSSystemExtensionManager.shared.submitRequest(request)
```

Then notarize the app, staple it, put it in `/Applications`, and enable the
extension under **System Settings → General → Login Items & Extensions → Driver
Extensions**.

To iterate before the entitlement is granted: `systemextensionsctl developer on`
on a machine with SIP partially disabled will load a locally-signed,
unnotarized extension. That is a development arrangement — not a machine you
should care about.

## Pointing rlx at it

The driver registers as `rlxpci` (see `SetName` in `RlxPciDriver.cpp`), so:

```sh
cargo run -p rlx-egpu --features dext --example egpu_pci -- \
    --service rlxpci --helper /path/to/rlxpci-helper
```

or `RLX_EGPU_SERVICE=rlxpci`, or `service = rlxpci` in
`~/.config/rlx/egpu.conf`. Renaming the service in `SetName` means changing it
in all three places — and in `SERVICE_NAME` in `helper/helper.c`, which looks the
service up independently.

## Contract

The selectors, the dual meaning of `CopyClientMemoryForType`, and the socket
wire protocol are specified in [`docs/egpu.md`](../../../../docs/egpu.md). Any
extension implementing them works with `rlx-egpu`; this one is a reference, not
a requirement.
## License

MIT OR Apache-2.0.
