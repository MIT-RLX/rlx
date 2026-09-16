// swift-tools-version:5.9
import PackageDescription

// Run `ios/build-xcframework.sh` first — it produces ios/build/RlxNode.xcframework.
let package = Package(
    name: "RlxNode",
    platforms: [.iOS(.v15), .macOS(.v12), .tvOS(.v15), .visionOS(.v1)],
    products: [
        .library(name: "RlxNode", targets: ["RlxNode"])
    ],
    targets: [
        .binaryTarget(name: "RlxNodeFFI", path: "build/RlxNode.xcframework"),
        .target(name: "RlxNode", dependencies: ["RlxNodeFFI"]),
    ]
)
