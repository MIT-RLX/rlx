// swift-tools-version:5.9
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Swift bindings for rlx over its C ABI (crates/bindings/rlx-capi).
//
// Build the library first, then this package:
//
//     cargo build -p rlx-capi            # → target/debug/librlx_capi.{dylib,a}
//     swift run --package-path crates/bindings/rlx-swift rlx-example-inference
//     swift test --package-path crates/bindings/rlx-swift
//
// Examples (Examples/<Name>/main.swift): rlx-example-inference, -training,
// -loading, -distributed, -backends, -models.
//
// `RLX_CAPI_FEATURES` lists the optional rlx-capi features the library was
// built with (`fpga,gguf-convert`), so the Swift rows that need them compile;
// `linked-plugins` for a library with model plugins linked in (an iOS app's,
// `ios/build-xcframework.sh --capi-crate`), which adds `linkModelPlugins()`.
//
// `RLX_CAPI_LIB_DIR` points at a different build (`target/release`, an
// iOS archive's directory); it defaults to the workspace's `target/debug`.
//
// For an app (iOS or macOS), package the static library instead and build
// against it with `RLX_XCFRAMEWORK=1`:
//
//     ios/build-xcframework.sh --capi [--apple]   # → build/RLX.xcframework
//     RLX_XCFRAMEWORK=1 swift test --package-path crates/bindings/rlx-swift

import Foundation
import PackageDescription

let libDir = ProcessInfo.processInfo.environment["RLX_CAPI_LIB_DIR"]
    ?? URL(fileURLWithPath: Context.packageDirectory)
        .appendingPathComponent("../../../target/debug").standardized.path

let env = ProcessInfo.processInfo.environment
let useXCFramework = env["RLX_XCFRAMEWORK"].map { $0 != "0" && !$0.isEmpty } ?? false

// The static archive carries no link list of its own: these are what rustc
// reports as its native libraries (`--print native-static-libs`), with the
// Metal set for a `--apple` build (unused frameworks cost nothing).
let crlx: Target = useXCFramework
    ? .binaryTarget(name: "CRLX", path: "build/RLX.xcframework")
    : .systemLibrary(name: "CRLX", path: "Sources/CRLX")
let rlxLinker: [LinkerSetting] = useXCFramework
    ? [
        .linkedFramework("Accelerate"),
        .linkedFramework("Metal"),
        .linkedFramework("MetalPerformanceShaders"),
        .linkedFramework("MetalPerformanceShadersGraph"),
        .linkedLibrary("bz2"),
        .linkedLibrary("iconv"),
        .linkedLibrary("objc"),
    ]
    : [.unsafeFlags(["-L", libDir, "-Xlinker", "-rpath", "-Xlinker", libDir])]

let features = (env["RLX_CAPI_FEATURES"] ?? "")
    .split(separator: ",")
    .map { $0.trimmingCharacters(in: .whitespaces) }
    .filter { !$0.isEmpty }
let defines: [SwiftSetting] = features.map {
    .define("RLX_" + $0.uppercased().replacingOccurrences(of: "-", with: "_"))
}

let examples = ["Inference", "Training", "Loading", "Distributed", "Backends", "Models"]

let package = Package(
    name: "RLX",
    products: [
        .library(name: "RLX", targets: ["RLX"]),
    ] + examples.map { .executable(name: "rlx-example-\($0.lowercased())", targets: ["Example\($0)"]) },
    targets: [
        crlx,
        .target(name: "RLX", dependencies: ["CRLX"], swiftSettings: defines, linkerSettings: rlxLinker),
        .testTarget(name: "RLXTests", dependencies: ["RLX"], swiftSettings: defines),
    ] + examples.map {
        .executableTarget(name: "Example\($0)", dependencies: ["RLX"], path: "Examples/\($0)",
                          swiftSettings: defines)
    }
)
