// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

import Foundation

// SwiftPM consumes the C ABI as a module (via the xcframework's modulemap); an
// Xcode app target usually reaches it through a bridging header instead, where
// the symbols are already in scope and no import exists to make.
#if canImport(RlxNodeFFI)
import RlxNodeFFI
#endif

/// An RLX mesh worker running inside this app.
///
/// The node serves on its own thread, so every call here returns immediately.
/// Only one node may run per process.
///
/// ```swift
/// try RlxNode.start(rank: 1, world: 2,
///                   peers: ["192.168.1.10:29500", "192.168.1.11:29501"])
/// print(RlxNode.status)   // "running"
/// RlxNode.stop()
/// ```
///
/// - Important: iOS needs `NSLocalNetworkUsageDescription` in Info.plist to
///   talk to LAN peers at all, and the `com.apple.developer.networking.multicast`
///   entitlement for the UDP-discovery path. See `ios/README.md`.
public enum RlxNode {
    /// What this node joins the mesh to do.
    public enum Mode: String {
        /// Serve a shipped inference stage; stoppable between activations.
        case infer
        /// Join a data-parallel training run; **not** stoppable partway.
        case train
    }

    public struct Error: Swift.Error, CustomStringConvertible {
        public let code: Int32
        public let message: String
        public var description: String { "RlxNode(\(code)): \(message)" }
    }

    /// Join a mesh as worker `rank` of `world`.
    ///
    /// - Parameters:
    ///   - peers: `host:port` per rank. Pass an empty array to find the
    ///     coordinator by UDP broadcast (needs the multicast entitlement).
    ///   - device: `"auto"`, or a backend name (`"cpu"`, `"metal"`, …).
    ///   - mode: ``Mode/infer`` to serve a shipped stage, ``Mode/train`` to
    ///     join a data-parallel training run.
    ///
    /// - Important: A training rank cannot drop out partway — the gradient
    ///   reduce is a barrier, so stopping one stalls every other rank.
    ///   ``stop()`` is honoured between inference activations but **not**
    ///   mid-training-run, so only start a training node when the app will
    ///   stay foregrounded for it.
    public static func start(
        rank: Int,
        world: Int,
        peers: [String],
        device: String = "auto",
        mode: Mode = .infer
    ) throws {
        let rc = peers.joined(separator: ",").withCString { p in
            device.withCString { d in
                mode.rawValue.withCString { m in
                    rlx_node_start(Int32(rank), Int32(world), p, d, m)
                }
            }
        }
        guard rc == RLX_NODE_OK else {
            throw Error(code: rc, message: lastError)
        }
    }

    /// `idle` | `running` | `stopping` | `ok: …` | `error: …`.
    public static var status: String { read(rlx_node_status) }

    /// Detail for the most recent failure.
    public static var lastError: String { read(rlx_node_last_error) }

    /// Platform tag the library was built for (`"ios"`).
    public static var platform: String {
        guard let p = rlx_node_platform() else { return "unknown" }
        return String(cString: p)
    }

    /// Ask the node to leave the mesh after its current activation.
    ///
    /// Cooperative: a node parked in `recv` exits when its peer sends or the
    /// link drops, so ``status`` may report `running` briefly afterwards. Call
    /// this from `scenePhase == .background` — iOS suspends the process
    /// shortly after, and a suspended rank stalls every peer that waits on it.
    @discardableResult
    public static func stop() -> Bool {
        rlx_node_stop() == RLX_NODE_OK
    }

    private static func read(_ f: (UnsafeMutablePointer<CChar>?, Int) -> Int32) -> String {
        var buf = [CChar](repeating: 0, count: 512)
        let n = buf.withUnsafeMutableBufferPointer { f($0.baseAddress, $0.count) }
        guard n >= 0 else { return "" }
        return String(cString: buf)
    }
}
