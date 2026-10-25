// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The Swift binding against the real library: generated builders (labels,
// defaults, options, optionals, body graphs, row arguments), errors, running,
// autodiff, training and checkpoints, text, GGUF, and devices.

import Foundation
import XCTest
@testable import RLX

final class RLXTests: XCTestCase {
    /// loss = mean((x·w - t)²), x/t [1, 2], w [2, 2].
    private func lossGraph() throws -> (Graph, Node) {
        let g = try Graph("mse")
        let x = try g.input("x", shape: [1, 2])
        let t = try g.input("t", shape: [1, 2])
        let w = try g.param("w", shape: [2, 2])
        let d = try g.sub(g.matmul(x, rhs: w), b: t)
        try g.setOutputs([g.mean(g.mul(d, b: d), axes: [0, 1])])
        return (g, w)
    }

    private func tmp(_ name: String) -> String {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("rlx-swift-\(ProcessInfo.processInfo.processIdentifier)-\(name)").path
    }

    // ── builders ────────────────────────────────────────────

    func testBuildersInferShapesAndTakeTableDefaults() throws {
        let g = try Graph("shapes")
        let x = try g.input("x", shape: [1, 3, 8, 8])
        let k = try g.param("k", shape: [4, 3, 3, 3])
        XCTAssertEqual(try g.shapeOf(g.conv2d(x, weight: k, kernelSize: [3, 3])).dims, [1, 4, 6, 6])
        // A pair option takes one value for both axes.
        let strided = try g.conv2d(x, weight: k, kernelSize: [3], stride: [2], padding: [1])
        XCTAssertEqual(try g.shapeOf(strided).dims, [1, 4, 4, 4])
        // `maxPool2d`'s kernel defaults to 2 and its stride to the kernel.
        XCTAssertEqual(try g.shapeOf(g.maxPool2d(x)).dims, [1, 3, 4, 4])
        let v = try g.input("v", shape: [2, 6])
        XCTAssertEqual(try g.shapeOf(g.softmax(v)).dims, [2, 6])
        XCTAssertEqual(try g.shapeOf(g.cast(v, to: "i32")).dtype, "i32")
        XCTAssertEqual(try g.shapeOf(g.argmax(v, axis: 1, keepDim: true)).dims, [2, 1])
        XCTAssertEqual(try g.paramNames(), ["k"])
    }

    func testOptionalAndRowArgumentsReachTheCore() throws {
        let g = try Graph("opts")
        let q = try g.input("q", shape: [1, 4, 8])
        let cos = try g.input("cos", shape: [4, 2])
        let sin = try g.input("sin", shape: [4, 2])
        // `nRot` is optional: nil = full head, a value = partial rotary.
        XCTAssertEqual(try g.shapeOf(g.rope(q, cos: cos, sin: sin, headDim: 4)).dims, [1, 4, 8])
        XCTAssertEqual(try g.shapeOf(g.rope(q, cos: cos, sin: sin, headDim: 4, nRot: 4, style: "gptj")).dims, [1, 4, 8])
        // Optional node + optional scalars.
        let att = try g.attentionOpts(q, k: q, v: q, numHeads: 2, headDim: 4, scale: 0.5, softcap: 30)
        XCTAssertEqual(try g.shapeOf(att).dims, [1, 4, 8])
        // Rows: a symmetric pad row is one value.
        let s = try g.input("s", shape: [2, 16])
        XCTAssertEqual(try g.shapeOf(g.pad(s, pads: [[0], [1, 2]])).dims, [2, 19])
        XCTAssertEqual(try g.shapeOf(g.sosfilt(s, sos: [[1, 0, 0, 1, 0, 0]])).dims, [2, 16])
        // A wrong-width row is refused before it reaches C.
        XCTAssertThrowsError(try g.sosfilt(s, sos: [[1, 0]]))
    }

    func testBodyGraphsAreConsumed() throws {
        let body = try Graph("step")
        let carry = try body.input("carry", shape: [2])
        let xt = try body.input("xt", shape: [2])
        try body.setOutputs([body.add(carry, b: xt)])
        let g = try Graph("scan")
        let xs = try g.input("xs", shape: [3, 2])
        let out = try g.scan(g.input("init", shape: [2]), body: body, length: 3, xs: [xs])
        XCTAssertEqual(try g.shapeOf(out).dims, [2])
        XCTAssertTrue(body.isConsumed)
        // Running the scan sums the slices onto the carry.
        try g.setOutputs([out])
        let c = try Session().compile(g)
        XCTAssertEqual(try c.run(["init": [0, 100], "xs": [1, 2, 3, 4, 5, 6]])[0], [9, 112])
    }

    func testErrorsCarryTheStatusAndMessage() throws {
        let g = try Graph("err")
        XCTAssertThrowsError(try g.relu(Node(id: 42))) { e in
            let err = e as! RLXError
            XCTAssertEqual(err.kind, .range)
            XCTAssertTrue(err.message.contains("does not exist"), err.message)
        }
        let x = try g.input("x", shape: [2])
        XCTAssertThrowsError(try g.activation("nope", input: x)) { e in
            XCTAssertTrue((e as! RLXError).message.contains("LogSigmoid"))
        }
        XCTAssertThrowsError(try Session(device: "no-such-device"))
    }

    // ── running ─────────────────────────────────────────────

    func testRunAndTypedRun() throws {
        let g = try Graph("typed")
        let x = try g.input("x", shape: [2, 3])
        try g.setOutputs([g.relu(x), g.cast(g.argmax(x, axis: 1), to: "i32")])
        let s = try Session(device: "cpu")
        XCTAssertEqual(try s.precision(), "f32")
        XCTAssertNil(try s.policy())
        let c = try s.compile(g)
        XCTAssertEqual(try c.device(), "cpu")
        XCTAssertEqual(try c.outputShapes(), [[2, 3], [2]])
        XCTAssertEqual(try c.outputDtypes(), ["f32", "i32"])
        // `run` refuses the i32 output rather than reinterpreting it.
        XCTAssertThrowsError(try c.run(["x": [1, -2, 3, -4, 5, -6]]))
        let values: [Float] = [1, -2, 3, -4, 5, -6]
        let bytes = values.withUnsafeBytes { Array($0) }
        let outs = try c.runTyped(["x": TypedTensor(bytes: bytes, dtype: "f32")])
        XCTAssertEqual(outs[1].dtype, "i32")
        let idx = outs[1].bytes.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        XCTAssertEqual(idx, [2, 1])
    }

    func testPoliciesAndCheck() throws {
        XCTAssertEqual(try Session(device: "cpu", policy: "mixed").policy(), "mixed")
        XCTAssertEqual(try Session(device: "cpu", policy: ["compute": "f16"]).policy(), "custom")
        let (g, _) = try lossGraph()
        let report = try check(g, options: ["backends": ["cpu"]])
        XCTAssertEqual(report["errors"] as? Int, 0)
        XCTAssertTrue(try checkBackends().contains("cuda"))
    }

    // ── autodiff ────────────────────────────────────────────

    func testGradJvpAndHigherOrder() throws {
        let (g, w) = try lossGraph()
        let bwd = try Session().compile(grad(g, wrt: [w]))
        try bwd.setParam("w", data: [1, 0, 0, 1])
        XCTAssertEqual(try bwd.run(["x": [1, 2], "t": [0, 0], "d_output": [1]])[1], [1, 2, 2, 4])

        // d³(x³)/dx³ = 6.
        let cube = try Graph("cube")
        let x = try cube.input("x", shape: [])
        try cube.setOutputs([cube.mul(cube.mul(x, b: x), b: x)])
        let third = try Session().compile(nthOrderGrad(cube, wrt: "x", order: 3))
        XCTAssertEqual(try third.run(["x": [2]])[0][0], 6, accuracy: 1e-4)
        XCTAssertFalse(try directionalNthGrad(cube, wrt: "x", order: 2).isConsumed)
        XCTAssertFalse(try jvp(cube, tangentFor: [x]).isConsumed)
    }

    // ── training ────────────────────────────────────────────

    func testTrainerFreezesSavesAndResumes() throws {
        let batch: [String: [Float]] = ["x": [1, 1], "t": [1, -1]]
        let opts: [String: Any] = ["optimizer": ["kind": "adamw", "lr": 0.05]]
        let a = try Trainer(try lossGraph().0, wrt: ["w"], initial: ["w": [0, 0, 0, 0]], options: opts)
        try a.run(steps: 20, batch: { _ in batch })
        XCTAssertNotNil(try a.gradNorm())
        XCTAssertFalse(a.isResident())
        XCTAssertEqual(try a.trainable(), ["w"])
        let path = tmp("ck.gguf")
        defer { try? FileManager.default.removeItem(atPath: path) }
        XCTAssertGreaterThanOrEqual(try a.save(path), 1)
        let continued = try a.run(steps: 20, batch: { _ in batch })["last_loss"] as! Double

        let b = try Trainer(try lossGraph().0, wrt: ["w"], initial: ["w": [0, 0, 0, 0]], options: opts)
        let report = try b.load(path)
        XCTAssertEqual(report["step"] as? Int, 20)
        XCTAssertEqual(report["optimizer_restored"] as? Bool, true)
        let resumed = try b.run(steps: 20, batch: { _ in batch })["last_loss"] as! Double
        XCTAssertEqual(resumed, continued, accuracy: 1e-6)

        XCTAssertEqual(try b.freeze(), 1)
        XCTAssertEqual(try b.frozen(), ["w"])
        let before = try b.params()["w"]!
        try b.step(batch)
        XCTAssertEqual(try b.params()["w"]!, before)
        XCTAssertEqual(try b.unfreeze(), 0)
        XCTAssertEqual(try b.setParams(["w": [1, 0, 0, 1]]), 1)
    }

    func testRunStopsEarlyAndOptimizerSteps() throws {
        let t = try Trainer(try lossGraph().0, wrt: ["w"], initial: ["w": [0, 0, 0, 0]],
                            options: ["optimizer": "sgd"])
        var seen: [UInt64] = []
        let summary = try t.run(steps: 10, batch: { _ in ["x": [1, 1], "t": [1, -1]] }) { i, _ in
            seen.append(i)
            return i < 3
        }
        XCTAssertEqual(seen, [0, 1, 2, 3])
        XCTAssertEqual(summary["stopped"] as? Bool, true)

        let opt = try Optimizer(kind: "adam", options: ["lr": 0.1])
        try opt.setLr(0.1)
        var p: [String: [Float]] = ["p": [1, -1]]
        for _ in 0..<50 { try opt.step(&p, grads: ["p": p["p"]!.map { 2 * $0 }]) }
        XCTAssertTrue(p["p"]!.allSatisfy { abs($0) < 0.2 }, "\(p)")
        XCTAssertThrowsError(try Optimizer(kind: "adamx"))
        XCTAssertThrowsError(try Optimizer(kind: "adam", options: ["learning_rate": 0.1]))
    }

    // ── text ────────────────────────────────────────────────

    func testSamplingIsDeterministicPerSeed() throws {
        let logits: [Float] = [0.1, 3.0, 0.2, 2.9]
        XCTAssertEqual(try argmax(logits), 1)
        XCTAssertEqual(try sampleNext(logits), 1)
        let a = try sampleNext(logits, options: ["temperature": 1.0, "seed": 7])
        XCTAssertEqual(a, try sampleNext(logits, options: ["temperature": 1.0, "seed": 7]))
        let s = try Sampler(options: ["temperature": 1.0, "seed": 3])
        try s.prime([5, 6])
        for _ in 0..<8 { _ = try s.next([1, 1, 1, 1]) }
        XCTAssertEqual(try s.history().prefix(2), [5, 6])
        try s.reset(3)
        XCTAssertEqual(try s.history(), [])
        XCTAssertThrowsError(try Sampler(options: ["temprature": 1.0]))
    }

    func testChatTemplatesAndTokenizer() throws {
        let tpl = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}"
        XCTAssertEqual(try renderChat(tpl, messages: [["role": "user", "content": "hi"]],
                                      addGenerationPrompt: false), "<user>hi")
        XCTAssertThrowsError(try renderChat(tpl, messages: [["role": "bot", "content": "x"]]))
        let models = ProcessInfo.processInfo.environment["RLX_MODELS_DIR"] ?? "../../../../rlx-models"
        let path = URL(fileURLWithPath: models).appendingPathComponent("weights/qwen3-0.6b/tokenizer.json").path
        guard FileManager.default.fileExists(atPath: path) else { throw XCTSkip("no qwen3 tokenizer fixture") }
        let tok = try Tokenizer.load(path)
        let text = "Hello, 世界 — 👨‍👩‍👧 done."
        let ids = try tok.encode(text, addSpecial: false)
        let streamed = try ids.map { try tok.decodeIncremental(id: $0) }.joined() + tok.finishStream()
        XCTAssertEqual(streamed, try tok.decode(ids))
        try tok.resetStream()
    }

    // ── GGUF / weights ──────────────────────────────────────

    func testGgufWriteReadRoundTrip() throws {
        let path = tmp("w.gguf")
        defer { try? FileManager.default.removeItem(atPath: path) }
        let w = (0..<256).map { Float($0) / 128 - 1 }
        let packed = try quantize(w, dtype: "Q8_0")
        let writer = try GgufWriter()
        try writer.addF32("dense", shape: [256], dtype: "F32", data: w)
        try writer.addBytes("q8", shape: [256], dtype: "Q8_0", data: packed)
        XCTAssertEqual(try writer.write(path, architecture: "test", metadata: ["test.answer": 42]), 2)

        let f = try Gguf.open(path)
        XCTAssertEqual(try f.tensorNames(), ["dense", "q8"])
        XCTAssertEqual(try f.info("q8")["dtype"] as? String, "Q8_0")
        XCTAssertEqual(try f.architecture(), "test")
        XCTAssertEqual(try f.metadataValue("test.answer") as? Int, 42)
        XCTAssertThrowsError(try f.info("missing"))
        XCTAssertEqual(try f.tensor("dense"), w)
        XCTAssertEqual(try f.tensorBytes("q8"), packed)
        XCTAssertEqual(try f.metadata()["test.answer"] as? Int, 42)
        let back = try dequantize(packed, dtype: "Q8_0")
        XCTAssertLessThan(zip(w, back).map { abs($0 - $1) }.max()!, 0.01)
        XCTAssertThrowsError(try Gguf.open(tmp("absent.gguf"))) { e in
            XCTAssertEqual((e as! RLXError).kind, .notFound)
        }
        XCTAssertThrowsError(try loadWeights("nope", path: path))
        XCTAssertThrowsError(try openRlxp(tmp("absent.rlxp")))
    }

    // ── devices ─────────────────────────────────────────────

    func testDevicesRunnerAndRouter() throws {
        XCTAssertTrue(try devices().contains("cpu"))
        XCTAssertTrue(isAvailable("cpu"))
        XCTAssertFalse(isAvailable("no-such-device"))
        XCTAssertFalse(try backendsManifest().isEmpty)
        let (g, _) = try lossGraph()
        XCTAssertFalse(try deviceReport(g).isEmpty)
        XCTAssertFalse(try fastestDeviceFor(g).isEmpty)

        let inputs: [String: [Float]] = ["x": [1, 2], "t": [0, 0]]
        let r = try Runner(try lossGraph().0, policy: ["only": ["cpu"]])
        XCTAssertEqual(try r.devices(), ["cpu"])
        try r.setParam("w", data: [1, 0, 0, 1])
        let (dev, outs) = try r.run(inputs)
        XCTAssertEqual(dev, "cpu")
        XCTAssertEqual(outs[0][0], 2.5, accuracy: 1e-6)
        XCTAssertEqual(try r.runTry(["cpu"], inputs: inputs).device, "cpu")
        XCTAssertEqual(try r.warmAll(), ["cpu"])
        XCTAssertEqual(try r.benchmark(inputs, runs: 1).count, 1)
        XCTAssertFalse(try r.report().isEmpty)

        XCTAssertEqual(try r.fastest(), "cpu")
        XCTAssertEqual(try r.resolve(), "cpu")
        XCTAssertEqual(try r.setParams(["w": [1, 0, 0, 1]]), 1)
        XCTAssertEqual(try r.runOn("cpu", inputs: inputs)[0][0], 2.5, accuracy: 1e-6)
        XCTAssertEqual(try r.runChain(inputs).device, "cpu")
        XCTAssertEqual(try r.runWithInputs(inputs)[0][0], 2.5, accuracy: 1e-6)

        let router = try Router(try lossGraph().0, policy: ["only": ["cpu"]])
        try router.setParam("w", data: [1, 0, 0, 1])
        XCTAssertEqual(try router.devices(), ["cpu"])
        XCTAssertEqual(try router.run(inputs).outputs[0][0], 2.5, accuracy: 1e-6)
        XCTAssertEqual(try router.runOn("cpu", inputs: inputs)[0][0], 2.5, accuracy: 1e-6)
        XCTAssertEqual(try router.runChain(inputs, hint: "cpu").device, "cpu")
        try router.setRebenchOnThrottle(false)

        let flex = try FlexibleSession(policy: ["only": ["cpu"]])
        XCTAssertEqual(try flex.compile(try lossGraph().0).device(), "cpu")
        XCTAssertEqual(try parseDevice("cpu"), "cpu")
        XCTAssertThrowsError(try parseDevice("no-such-device"))
    }

    // ── distributed ─────────────────────────────────────────

    func testATwoRankMeshReducesAndShipsAStage() throws {
        let peers = try loopbackPeers(world: 2)
        XCTAssertEqual(peers.count, 2)
        let worker = Thread {
            guard let g = try? Group.connect(rank: 1, world: 2, options: ["peers": peers]) else { return }
            _ = try? g.allReduce([3], op: "max")
            _ = try? g.serveWorker(["steps": 1])
        }
        worker.start()
        let g = try Group.connect(rank: 0, world: 2, options: ["peers": peers])
        XCTAssertEqual(try g.world(), 2)
        XCTAssertEqual(try g.allReduce([5], op: "max"), [5])
        XCTAssertThrowsError(try g.allReduce([1], op: "median"))
        XCTAssertEqual(try g.collectCaps().count, 1)
        let s = try Graph("double")
        let x = try s.input("x", shape: [2])
        try s.setOutputs([s.add(x, b: x)])
        try g.shipStage(rank: 1, graph: s, options: ["device": "cpu"])
        try g.sendActivation(to: 1, data: [1, 2])
        XCTAssertEqual(try g.recvActivation(from: 1), [2, 4])
        XCTAssertThrowsError(try Group.connect(rank: 3, world: 2)) { e in
            XCTAssertEqual((e as! RLXError).kind, .range)
        }
    }
}
