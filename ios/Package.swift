// swift-tools-version:5.9
import PackageDescription

// Run `ios/build-xcframework.sh` first — it produces ios/build/RlxNode.xcframework
// with a device and a simulator slice for iOS, tvOS, watchOS and visionOS.
//
// The minimums below match the deployment targets that script stamps into the
// Rust objects; raising one here without raising it there gets you a linker
// warning per object. iOS and tvOS sit at 17.0 because that is where Metal
// gained native `bfloat`, which MLX's kernels require; watchOS sits at 26.0
// because rustup's prebuilt `std` for aarch64-apple-watchos is compiled at
// 26.0 — see the script's header for both.
//
// `.macOS` is declared because the Swift wrapper is useful on a Mac, but the
// default xcframework carries no macOS slice: pass
// `--platforms ios,tvos,watchos,visionos,macos` to add one.
let package = Package(
    name: "RlxNode",
    platforms: [
        .iOS(.v17),
        .macOS(.v12),
        .tvOS(.v17),
        .watchOS("26.0"),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "RlxNode", targets: ["RlxNode"])
    ],
    targets: [
        .binaryTarget(name: "RlxNodeFFI", path: "build/RlxNode.xcframework"),
        .target(name: "RlxNode", dependencies: ["RlxNodeFFI"]),
    ]
)
