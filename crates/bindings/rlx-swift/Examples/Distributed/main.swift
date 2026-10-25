// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Distributed: a mesh of rlx nodes. Here all three ranks are threads of one
// process on loopback; across machines, give each rank the same `peers` list
// (or `discover`) and run one rank per device — an iPhone, a Mac, a Linux box.
//
//     swift run --package-path crates/bindings/rlx-swift rlx-example-distributed
//
// 1. Collectives: all_reduce, all_gather, broadcast.
// 2. Pipelined inference: rank 0 ships each worker a stage of a model; the
//    workers compile it on their own hardware and load its weights locally
//    (only the spec and the activations cross the wire).
// 3. Data-parallel training: rank 0 ships a training job; every rank trains
//    its own shard and the gradients are averaged each step.

import Foundation
import RLX

setvbuf(stdout, nil, _IOLBF, 0)  // one line at a time, also when piped

let world: UInt32 = 3
let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("rlx-dist-\(ProcessInfo.processInfo.processIdentifier)")

/// Raw little-endian f32 at `name`, as the `file://` URI a node reads.
func blob(_ name: String, _ values: [Float]) throws -> String {
    try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    let url = tmp.appendingPathComponent(name)
    try values.withUnsafeBytes { Data($0) }.write(to: url)
    return "file://\(url.path)"
}

/// One pipeline stage: relu(x · w), x [1, 4], w [4, 4].
func stage() throws -> Graph {
    let g = try Graph("stage")
    let x = try g.input("x", shape: [1, 4])
    try g.setOutputs([g.relu(g.matmul(x, rhs: g.param("w", shape: [4, 4])))])
    return g
}

/// The backward graph of mean((X·w − y)²) for a batch of 4: outputs [loss, dw].
func trainingGraph() throws -> Graph {
    let g = try Graph("lr")
    let x = try g.input("X", shape: [4, 3])
    let y = try g.input("y", shape: [4, 1])
    let w = try g.param("w", shape: [3, 1])
    let d = try g.sub(g.matmul(x, rhs: w), b: y)
    try g.setOutputs([g.mean(g.mul(d, b: d), axes: [0, 1])])
    return try grad(g, wrt: [w])
}

func matvec(_ x: [Float], _ w: [Float]) -> [Float] {
    (0..<4).map { j in max(0, (0..<4).map { x[$0] * w[$0 * 4 + j] }.reduce(0, +)) }
}

let stageWeights: [UInt32: [Float]] = [
    1: (0..<16).map { Float(($0 * 5) % 7) / 7 - 0.3 },
    2: (0..<16).map { Float(($0 * 3) % 5) / 5 - 0.2 },
]
let perRank = 16
let wTrue: [Float] = [1.5, -2, 0.5]
let samples = perRank * Int(world)
let xAll: [Float] = (0..<(samples * 3)).map { Float(($0 * 37) % 23) / 11.5 - 1 }
let yAll: [Float] = (0..<samples).map { i in (0..<3).map { xAll[i * 3 + $0] * wTrue[$0] }.reduce(0, +) }

/// Rank `rank`'s training job: its shard of X / y, the shared init and schedule.
func trainSpec(rank: UInt32, xURI: String, yURI: String, wURI: String) -> [String: Any] {
    let start = Int(rank) * perRank
    return [
        "params": [["name": "w", "uri": wURI]],
        "data": [
            ["input": "X", "uri": xURI, "elem": 3, "shard_start": start, "shard_len": perRank],
            ["input": "y", "uri": yURI, "elem": 1, "shard_start": start, "shard_len": perRank],
        ],
        "batch": 4, "lr_per_epoch": [Float](repeating: 0.2, count: 40), "device": "cpu",
    ]
}

/// A worker: everything it runs, the coordinator ships it.
func worker(_ rank: UInt32, peers: [String]) throws -> [Float] {
    let g = try Group.connect(rank: rank, world: world, options: ["peers": peers, "device": "cpu"])
    // 1. Collectives — every rank calls each one.
    let mine: [Float] = [Float(rank + 1)]
    _ = try g.allReduce(mine, op: "sum")
    _ = try g.allGather(mine)
    _ = try g.broadcast(root: 0, data: [0])  // the root's length; it overwrites
    // 2. Serve two activations of whatever stage arrives.
    _ = try g.serveWorker(["steps": 2])
    // 3. Train whatever job arrives; return the final weights.
    let report = try g.serveTrainer()
    return ((report["params"] as? [String: Any])?["w"] as? [NSNumber])?.map(\.floatValue) ?? []
}

do {
    let peers = try loopbackPeers(world: world)
    var workerResults: [UInt32: Result<[Float], Error>] = [:]
    let lock = NSLock()
    let done = DispatchGroup()
    for rank in 1..<world {
        done.enter()
        Thread {
            let r = Result { try worker(rank, peers: peers) }
            lock.lock()
            workerResults[rank] = r
            lock.unlock()
            done.leave()
        }.start()
    }

    // ── rank 0: the coordinator ────────────────────────────────────────────
    let g = try Group.connect(rank: 0, world: world, options: ["peers": peers, "device": "cpu"])
    print("mesh: rank \(try g.rank()) of \(try g.world()), leader: \(try g.isLeader())")

    // 1. Collectives.
    print("all_reduce sum: \(try g.allReduce([1], op: "sum"))")
    print("all_gather: \(try g.allGather([1]))")
    print("broadcast: \(try g.broadcast(root: 0, data: [42]))")

    // 2. Pipelined inference: rank 1 runs stage A, rank 2 stage B, back to 0.
    let caps = try g.collectCaps()
    print("workers: \(caps.map { "rank \($0["rank"] ?? "?") on \($0["platform"] ?? "?")" }.joined(separator: ", "))")
    let stageGraph = try stage()
    for rank in 1..<world {
        let uri = try blob("w\(rank).bin", stageWeights[rank]!)
        try g.shipStage(rank: rank, graph: stageGraph, options: ["device": "cpu", "weights": ["w": uri]])
    }
    for x in [[Float](arrayLiteral: 1, -1, 0.5, 2), [0.25, 0.5, -0.75, 1]] {
        try g.sendActivation(to: 1, data: x)
        let y = try g.recvActivation(from: world - 1)
        let expected = matvec(matvec(x, stageWeights[1]!), stageWeights[2]!)
        let err = zip(y, expected).map { abs($0 - $1) }.max() ?? 0
        print("pipeline: \(y.map { String(format: "%.3f", $0) }.joined(separator: " ")) max|Δ| vs local = \(String(format: "%.1e", err))")
    }

    // 3. Data-parallel training.
    _ = try g.collectCaps()
    let bwd = try trainingGraph()
    let xURI = try blob("X.bin", xAll), yURI = try blob("y.bin", yAll), wURI = try blob("w0.bin", [0, 0, 0])
    for rank in 1..<world {
        try g.shipTrain(rank: rank, graph: bwd, spec: trainSpec(rank: rank, xURI: xURI, yURI: yURI, wURI: wURI))
    }
    let mine = try g.runTrain(bwd, spec: trainSpec(rank: 0, xURI: xURI, yURI: yURI, wURI: wURI))
    let metrics = mine["metrics"] as? [String: Any] ?? [:]
    let w0 = ((mine["params"] as? [String: Any])?["w"] as? [NSNumber])?.map(\.floatValue) ?? []
    print("training loss: \(metrics["first_loss"] ?? "?") -> \(metrics["last_loss"] ?? "?")")
    print("rank 0 w: \(w0.map { String(format: "%.3f", $0) }.joined(separator: " ")) (true: 1.5 -2 0.5)")

    done.wait()
    for rank in 1..<world {
        let theirs = try workerResults[rank]!.get()
        let err = zip(w0, theirs).map { abs($0 - $1) }.max() ?? .infinity
        print("rank \(rank) w agrees: \(err < 1e-5 && theirs.count == w0.count)")
    }
    try? FileManager.default.removeItem(at: tmp)
} catch {
    print("error: \(error)")
    exit(1)
}
