// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Training: fit y = x·w_true with the Trainer (forward + backward + optimizer
// compiled once, on one device), then the same fit by hand with `grad` and a
// standalone Optimizer.
//
//     swift run --package-path crates/bindings/rlx-swift rlx-example-training [device]

import Foundation
import RLX

setvbuf(stdout, nil, _IOLBF, 0)  // one line at a time, also when piped

let wTrue: [Float] = [2, -1, 0.5]
let batch = 8

/// loss = mean((x·w − y)²); x [8, 3], w [3, 1], y [8, 1].
func regression() throws -> (Graph, Node) {
    let g = try Graph("regression")
    let x = try g.input("x", shape: [batch, 3])
    let y = try g.input("y", shape: [batch, 1])
    let w = try g.param("w", shape: [3, 1])
    let d = try g.sub(g.matmul(x, rhs: w), b: y)
    try g.setOutputs([g.mean(g.mul(d, b: d), axes: [0, 1])])
    return (g, w)
}

/// Batch `step` of a fixed synthetic data set.
func data(_ step: UInt64) -> [String: [Float]] {
    var x: [Float] = []
    var y: [Float] = []
    for i in 0..<batch {
        let row = (0..<3).map { k in Float(((Int(step) * 31 + i * 7 + k * 3) % 17)) / 8.5 - 1 }
        x += row
        y.append(zip(row, wTrue).map(*).reduce(0, +))
    }
    return ["x": x, "y": y]
}

func fmt(_ v: [Float]) -> String { v.map { String(format: "%.3f", $0) }.joined(separator: " ") }

do {
    let device = CommandLine.arguments.dropFirst().first ?? "cpu"

    // ── Trainer: one call per step, or `run` with callbacks ────────────────
    let (g, w) = try regression()
    let trainer = try Trainer(
        g.clone(), wrt: ["w"], initial: ["w": [0, 0, 0]],
        options: ["device": device, "optimizer": ["kind": "adamw", "lr": 0.1]]
    )
    print("trainer device: \(try trainer.device()) resident: \(trainer.isResident())")
    let first = try trainer.evaluate(data(0))
    // Cosine-decay the learning rate; stop once the loss is tiny.
    let summary = try trainer.run(
        steps: 400,
        batch: { data($0) },
        lr: { step in 0.1 * Float(0.5 * (1 + cos(Double.pi * Double(step) / 400))) },
        onStep: { _, loss in loss > 1e-6 }
    )
    let last = try trainer.evaluate(data(0))
    print("trainer loss: \(String(format: "%.4f", first)) -> \(String(format: "%.6f", last)) in \(try trainer.steps()) steps")
    print("trainer summary keys: \(summary.keys.sorted().joined(separator: " "))")
    print("trainer w: \(fmt(try trainer.params()["w"]!))")

    // Checkpoint (GGUF: params + optimizer state), restore into a fresh trainer.
    let path = FileManager.default.temporaryDirectory.appendingPathComponent("rlx-example-train.gguf").path
    let written = try trainer.save(path)
    let restored = try Trainer(g.clone(), wrt: ["w"], initial: ["w": [0, 0, 0]], options: ["device": device])
    let report = try restored.load(path)
    print("checkpoint: \(written) tensors, restored step \(report["step"] ?? "?")")
    print("restored loss: \(String(format: "%.6f", try restored.evaluate(data(0))))")
    try? FileManager.default.removeItem(atPath: path)

    // Freezing: a frozen param keeps its value through steps.
    try restored.freeze(["w"])
    let before = try restored.params()["w"]!
    try restored.step(data(1))
    print("frozen: \(try restored.frozen()) unchanged: \(try restored.params()["w"]! == before)")
    try restored.unfreeze()

    // ── By hand: grad → compile → Optimizer ────────────────────────────────
    let backward = try Session(device: device).compile(grad(g, wrt: [w]))
    let opt = try Optimizer(kind: "sgd", options: ["lr": 0.2])
    var params: [String: [Float]] = ["w": [0, 0, 0]]
    var loss: Float = 0
    for step in 0..<300 {
        try backward.setParam("w", data: params["w"]!)
        var feed = data(UInt64(step))
        feed["d_output"] = [1]  // the upstream gradient of the loss
        let outs = try backward.run(feed)
        loss = outs[0][0]
        try opt.step(&params, grads: ["w": outs[1]])
    }
    print("manual loss: \(String(format: "%.6f", loss)) w: \(fmt(params["w"]!))")
} catch {
    print("error: \(error)")
    exit(1)
}
