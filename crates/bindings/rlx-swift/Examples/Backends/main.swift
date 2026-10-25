// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Picking a backend: what this build has, what suits a graph, and the four
// ways to choose — name it, ask for the fastest, let a policy decide per
// graph, or benchmark and route per run.
//
//     swift run --package-path crates/bindings/rlx-swift rlx-example-backends

import Foundation
import RLX

setvbuf(stdout, nil, _IOLBF, 0)  // one line at a time, also when piped

func model() throws -> Graph {
    let g = try Graph("mlp")
    let x = try g.input("x", shape: [4, 16])
    let w = try g.param("w", shape: [16, 16])
    try g.setOutputs([g.tanh(g.matmul(x, rhs: w))])
    return g
}

let x = (0..<64).map { Float($0 % 7) / 7 }
let w = (0..<256).map { Float($0 % 5) / 10 - 0.2 }

do {
    // 1. What is compiled in and has hardware here.
    let available = try devices()
    print("devices: \(available.joined(separator: " "))")
    print("metal available: \(isAvailable("metal"))")
    print("parse gpu: \(try parseDevice("gpu"))")
    let manifest = try backendsManifest()
    print("manifest keys: \(manifest.keys.sorted().prefix(4).joined(separator: " "))")

    // 2. What suits this graph: the cost model's pick and every device's verdict.
    let g = try model()
    print("fastest for graph: \(try fastestDeviceFor(g))")
    for row in try deviceReport(g) where (row["available"] as? Bool) == true {
        print("  \(row["device"] ?? "?"): supports=\(row["supports_graph"] ?? "?")")
    }

    // 3. Name it: one session per device, same answer from each.
    var reference: [Float]?
    for device in available {
        let c = try Session(device: device).compile(g.clone())
        try c.setParam("w", data: w)
        let y = try c.run(["x": x])[0]
        let err = reference.map { r in zip(r, y).map { abs($0 - $1) }.max() ?? 0 } ?? 0
        reference = reference ?? y
        print("session \(device): max|Δ| vs first = \(String(format: "%.2e", err))")
    }

    // 4. A policy decides per graph: prefer Metal, never the CPU fallback
    //    unless nothing else runs it.
    let flex = try FlexibleSession(policy: ["prefer": ["metal", "cpu"]])
    let picked = try flex.compile(g.clone())
    print("flexible session picked: \(try picked.device())")

    // 5. Benchmark every allowed device, then run on the winner (or a named one).
    let runner = try Runner(g.clone())
    try runner.setParam("w", data: w)
    print("runner warmed: \(try runner.warmAll().joined(separator: " "))")
    for row in try runner.benchmark(["x": x], runs: 3) {
        print("  bench \(row["device"] ?? "?"): exec_ms=\(row["exec_ms"] ?? "?")")
    }
    let (device, outputs) = try runner.run(["x": x])
    print("runner ran on: \(device) (\(outputs[0].count) values)")
    let chain = try runner.runTry(["metal", "cpu"], inputs: ["x": x])
    print("run_try chain: \(chain.device)")
    _ = try runner.runOn("cpu", inputs: ["x": x])

    // 6. Route per run, falling back down the policy when a device fails.
    let router = try Router(g)
    try router.setParam("w", data: w)
    print("router ran on: \(try router.runChain(["x": x]).device)")
} catch {
    print("error: \(error)")
    exit(1)
}
