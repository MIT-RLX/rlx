// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Loading: weights from disk into a compiled graph — a GGUF file (written
// here, f32 and quantized), a safetensors file, and quantizing in memory. With
// RLX_TOKENIZER=<tokenizer.json> it also tokenizes and samples.
//
//     swift run --package-path crates/bindings/rlx-swift rlx-example-loading

import Foundation
import RLX

setvbuf(stdout, nil, _IOLBF, 0)  // one line at a time, also when piped

let tmp = FileManager.default.temporaryDirectory
let w: [Float] = (0..<64).map { Float($0 % 9) / 4 - 1 }

/// A safetensors file: an 8-byte header length, a JSON header, raw data.
func writeSafetensors(_ path: String, name: String, shape: [Int], data: [Float]) throws {
    let bytes = data.withUnsafeBytes { Data($0) }
    var header = "{\"\(name)\":{\"dtype\":\"F32\",\"shape\":\(shape),\"data_offsets\":[0,\(bytes.count)]}}"
    while header.utf8.count % 8 != 0 { header += " " }
    var file = Data()
    withUnsafeBytes(of: UInt64(header.utf8.count).littleEndian) { file.append(contentsOf: $0) }
    file.append(Data(header.utf8))
    file.append(bytes)
    try file.write(to: URL(fileURLWithPath: path))
}

do {
    // ── GGUF: write f32 + Q8_0 tensors with metadata, read them back ───────
    let ggufPath = tmp.appendingPathComponent("rlx-example.gguf").path
    let writer = try GgufWriter()
    try writer.addF32("proj.weight", shape: [8, 8], dtype: "f32", data: w)
    try writer.addF32("proj.weight_q8", shape: [8, 8], dtype: "q8_0", data: w)
    let count = try writer.write(ggufPath, architecture: "demo", metadata: ["demo.layers": 1])
    print("gguf written: \(count) tensors")

    let file = try Gguf.open(ggufPath)
    print("gguf architecture: \(try file.architecture() ?? "none")")
    print("gguf tensors: \(try file.tensorNames().joined(separator: " "))")
    let info = try file.info("proj.weight_q8")
    print("gguf q8 info: dtype=\(info["dtype"] ?? "?") shape=\(info["shape"] ?? "?")")
    let exact = try file.tensor("proj.weight")
    let q8 = try file.tensor("proj.weight_q8")
    let q8err = zip(exact, q8).map { abs($0 - $1) }.max() ?? 0
    print("gguf f32 exact: \(exact == w) q8 max|Δ|: \(String(format: "%.4f", q8err))")
    print("gguf metadata demo.layers: \(try file.metadataValue("demo.layers"))")

    // Into a graph: the loaded tensor is the param of a compiled matmul.
    let g = try Graph("proj")
    let x = try g.input("x", shape: [1, 8])
    try g.setOutputs([g.matmul(x, rhs: g.param("proj.weight", shape: [8, 8]))])
    let c = try Session(device: "cpu").compile(g)
    try c.setParam("proj.weight", data: exact)
    print("gguf → graph: \(try c.run(["x": [Float](repeating: 1, count: 8)])[0].prefix(3).map { String(format: "%.2f", $0) })")

    // ── safetensors → every tensor as f32 ───────────────────────────────────
    let stPath = tmp.appendingPathComponent("rlx-example.safetensors").path
    try writeSafetensors(stPath, name: "proj.weight", shape: [8, 8], data: w)
    let loaded = try loadWeights("safetensors", path: stPath)
    print("safetensors: \(loaded.keys.sorted()) exact: \(loaded["proj.weight"] == w)")

    // ── quantize in memory ──────────────────────────────────────────────────
    let packed = try quantize(w, dtype: "q8_0")
    let unpacked = try dequantize(packed, dtype: "q8_0", count: w.count)
    print("quantize q8_0: \(w.count * 4) → \(packed.count) bytes, round trip \(unpacked.count) values")

    try? FileManager.default.removeItem(atPath: ggufPath)
    try? FileManager.default.removeItem(atPath: stPath)

    // ── text: tokenizer + sampling (needs a tokenizer.json) ────────────────
    if let tok = ProcessInfo.processInfo.environment["RLX_TOKENIZER"] {
        let tokenizer = try Tokenizer.load(tok)
        let ids = try tokenizer.encode("Hello from Swift")
        print("tokenizer: \(ids.count) ids → \"\(try tokenizer.decode(ids))\"")
        let sampler = try Sampler(options: ["temperature": 0.7, "top_k": 5, "seed": 1])
        print("sampled: \(try sampler.next([0.1, 2, 0.3, 1.5]))")
    } else {
        print("tokenizer: skipped (set RLX_TOKENIZER=<tokenizer.json>)")
    }
    print("argmax: \(try argmax([0.1, 2, 0.3]))")
} catch {
    print("error: \(error)")
    exit(1)
}
