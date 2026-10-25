// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Models: any provider's model, opened by path or id and run by name.
//
//     RLX_MODEL_PLUGINS=…/librlx_model_plugin.dylib \
//         swift run --package-path crates/bindings/rlx-swift rlx-example-models
//
// A graph saved as an .rlxp package runs anywhere; a plugin's models (a
// downstream collection's LLMs, or the reference plugin's demo:count) load
// through RLX_MODEL_PLUGINS or loadModelPlugin, or, in an iOS app, are
// linked into its RLX.xcframework. The Python, JS and C examples of the same
// name do the same steps.

import Foundation
import RLX

setvbuf(stdout, nil, _IOLBF, 0)  // one line at a time, also when piped
#if RLX_LINKED_PLUGINS
linkModelPlugins()  // where plugins are linked in, not loaded (iOS)
#endif

do {
    let g = try Graph("affine")
    let x = try g.input("x", shape: [1, 2])
    try g.setOutputs([g.matmul(x, rhs: g.param("w", shape: [2, 2]))])
    let path = FileManager.default.temporaryDirectory.appendingPathComponent("rlx-example-swift.rlxp").path
    try saveModel(g, path: path, weights: ["w": [2, 0, 0, 3]],
                  model: ["task": "graph", "family": "affine", "outputs": ["y"]])

    let m = try Model.open(path, options: ["device": "cpu"])
    let info = try m.info()
    print("model: \(info["family"] ?? "?") (\(info["task"] ?? "?")) on \(info["device"] ?? "?")")
    let y = try m.run(["x": [1, 1]])[0]
    print(String(format: "model run: %.3f %.3f", y[0], y[1]))
    try FileManager.default.removeItem(atPath: path)

    let lm = try Model.open("demo:count")
    let lmInfo = try lm.info()
    print("plugin model: \(lmInfo["family"] ?? "?") (\(lmInfo["task"] ?? "?"))")
    print("model text: \(try lm.generateText("c", options: ["max_tokens": 4]))")
    print("model ids: \(try lm.generate([2], maxTokens: 3).map(String.init).joined(separator: " "))")

    // By task: a chat, speech out and back in, embeddings.
    print("chat: \(try lm.chat([["role": "user", "content": "c"]], options: ["max_tokens": 4]))")
    let (audio, rate) = try Model.open("demo:voice").synthesize("hello")
    print("voice: \(audio.count) samples at \(rate) Hz")
    print("heard: \(try Model.open("demo:speech").transcribe(audio, sampleRate: rate))")
    let rows = try Model.open("demo:embed").embed(["hello", "sky"]) as? [[Double]] ?? []
    print("embed: \(rows.flatMap { $0 }.map { String(Int($0)) }.joined(separator: " "))")
} catch {
    print("error: \(error)")
    exit(1)
}
