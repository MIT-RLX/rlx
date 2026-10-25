// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Inference: build a graph, compile it for a device, set its weights, run it.
//
//     swift run --package-path crates/bindings/rlx-swift rlx-example-inference [device]
//
// A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].
// The same compiled graph also runs with typed (non-f32) inputs.

import Foundation
import RLX

setvbuf(stdout, nil, _IOLBF, 0)  // one line at a time, also when piped

func classifier() throws -> Graph {
    let g = try Graph("classifier")
    let x = try g.input("x", shape: [1, 4])
    let w1 = try g.param("w1", shape: [4, 8])
    let w2 = try g.param("w2", shape: [8, 3])
    let hidden = try g.relu(g.matmul(x, rhs: w1))
    try g.setOutputs([g.softmax(g.matmul(hidden, rhs: w2))])
    return g
}

/// Deterministic weights in [-0.5, 0.5).
func weights(_ n: Int, seed: Int) -> [Float] {
    (0..<n).map { Float((($0 * 7919 + seed * 104_729) % 1000)) / 1000 - 0.5 }
}

do {
    let device = CommandLine.arguments.dropFirst().first ?? "cpu"
    let g = try classifier()
    print("graph: \(try g.shapeOf(Node(id: 0)).dims) → \(try g.shapeOf(g.outputs()[0]).dims)")

    // Compile: the session picks the device and precision; compile consumes
    // the graph, so keep a clone for the typed run below.
    let session = try Session(device: device)
    let model = try session.compile(g.clone())
    print("compiled on: \(try model.device())")
    try model.setParams(["w1": weights(32, seed: 1), "w2": weights(24, seed: 2)])

    let probs = try model.run(["x": [0.5, -1, 2, 0.25]])[0]
    print("probs: \(probs.map { String(format: "%.4f", $0) }.joined(separator: " "))")
    print("probs sum: \(String(format: "%.4f", probs.reduce(0, +)))")
    print("output shapes: \(try model.outputShapes()) dtypes: \(try model.outputDtypes())")

    // Typed I/O: f32 bytes in, typed tensors out — the path for f16 / i8 / i64
    // inputs and outputs.
    let typed = try session.compile(g)
    try typed.setParams(["w1": weights(32, seed: 1), "w2": weights(24, seed: 2)])
    let bytes = [Float](arrayLiteral: 0.5, -1, 2, 0.25).withUnsafeBytes { Array($0) }
    let out = try typed.runTyped(["x": TypedTensor(bytes: bytes, dtype: "f32")])[0]
    let back = out.bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    print("typed run: \(out.dtype) \(back == probs ? "matches" : "DIFFERS")")
} catch {
    print("error: \(error)")
    exit(1)
}
