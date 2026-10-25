// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The pieces every generated and hand-written wrapper shares: errors, node
// ids, the `Graph` handle, and the bridging helpers for optional pointers.

import CRLX
import Foundation

/// An rlx call that failed: the `RLX_ERR_*` status and the message.
public struct RLXError: Error, CustomStringConvertible {
    /// What went wrong, by `RLX_ERR_*` status.
    public enum Kind: Equatable {
        /// A wrong argument: unknown name, bad shape, wrong arity, `NULL`.
        case type
        /// An index or value out of range (a node id the graph does not have).
        case range
        /// Well-formed but not carried out: I/O, a backend not in this build.
        case runtime
        /// A Rust assertion inside rlx, caught at the boundary.
        case panic
        /// A file that does not exist.
        case notFound
        case other(Int32)
    }

    public let code: Int32
    public let message: String

    public var kind: Kind {
        switch code {
        case RLX_ERR_TYPE: return .type
        case RLX_ERR_RANGE: return .range
        case RLX_ERR_RUNTIME: return .runtime
        case RLX_ERR_PANIC: return .panic
        case RLX_ERR_NOT_FOUND: return .notFound
        default: return .other(code)
        }
    }

    public var description: String { "RLXError(\(kind)): \(message)" }
}

/// Throw the last error unless `rc` is `RLX_OK`.
@inline(__always)
func check(_ rc: Int32) throws {
    if rc != RLX_OK {
        throw RLXError(code: rc, message: String(cString: rlx_last_error()))
    }
}

/// A node of a graph — the id every builder returns.
public struct Node: Hashable, CustomStringConvertible {
    public let id: UInt32
    public init(id: UInt32) { self.id = id }
    public var description: String { "%\(id)" }
}

/// A node's static dims (`-1` for a dynamic axis) and dtype.
public struct Shape: Decodable, Equatable {
    public let dims: [Int]
    public let dtype: String
}

/// A symbolic rlx graph. The builder methods (`matmul`, `conv2d`, …) are in
/// `GraphOps.swift`, generated from rlx-bind's op table.
public final class Graph {
    let handle: OpaquePointer

    public init(_ name: String = "graph") throws {
        var h: OpaquePointer?
        try check(rlx_graph_new(name, &h))
        handle = h!
    }

    init(handle: OpaquePointer) { self.handle = handle }

    deinit { rlx_graph_free(handle) }

    /// A deep copy — compile one, keep building on the other.
    public func clone() throws -> Graph {
        var h: OpaquePointer?
        try check(rlx_graph_clone(handle, &h))
        return Graph(handle: h!)
    }

    /// Whether a compile (or a body-graph argument) consumed this graph.
    public var isConsumed: Bool { rlx_graph_is_consumed(handle) }
}

// ── bridging helpers ────────────────────────────────────────

/// Take ownership of a string rlx returned.
func takeString(_ p: UnsafeMutablePointer<CChar>?) -> String {
    guard let p else { return "" }
    defer { rlx_string_free(p) }
    return String(cString: p)
}

/// Take ownership of an id array rlx returned.
func takeU32s(_ v: RlxU32s) -> [UInt32] {
    defer { rlx_u32s_free(v) }
    guard let p = v.ptr else { return [] }
    return Array(UnsafeBufferPointer(start: p, count: v.len))
}

/// Take ownership of a float array rlx returned.
func takeF32s(_ v: RlxF32s) -> [Float] {
    defer { rlx_f32s_free(v) }
    guard let p = v.ptr else { return [] }
    return Array(UnsafeBufferPointer(start: p, count: v.len))
}

/// Take ownership of a byte array rlx returned.
func takeBytes(_ v: RlxBytes) -> [UInt8] {
    defer { rlx_bytes_free(v) }
    guard let p = v.ptr else { return [] }
    return Array(UnsafeBufferPointer(start: p, count: v.len))
}

func decodeJSON<T: Decodable>(_ type: T.Type, _ text: String) throws -> T {
    do {
        return try JSONDecoder().decode(type, from: Data(text.utf8))
    } catch {
        throw RLXError(code: RLX_ERR_RUNTIME, message: "decoding rlx JSON: \(error)")
    }
}

/// JSON text → a Foundation value (`[String: Any]`, `[Any]`, …), for reports.
func parseJSON(_ text: String) throws -> Any {
    do {
        return try JSONSerialization.jsonObject(with: Data(text.utf8), options: [.fragmentsAllowed])
    } catch {
        throw RLXError(code: RLX_ERR_RUNTIME, message: "decoding rlx JSON: \(error)")
    }
}

/// A Foundation value → JSON text, for options (`nil` → `NULL`).
func jsonText(_ value: Any?) throws -> String? {
    guard let value else { return nil }
    do {
        let data = try JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed])
        return String(decoding: data, as: UTF8.self)
    } catch {
        throw RLXError(code: RLX_ERR_TYPE, message: "options are not JSON-serializable: \(error)")
    }
}

func withOptionalArray<T, R>(_ a: [T]?, _ body: (UnsafePointer<T>?, Int) throws -> R) rethrows -> R {
    guard let a else { return try body(nil, 0) }
    return try a.withUnsafeBufferPointer { try body($0.baseAddress, a.count) }
}

func withOptionalValue<T, R>(_ v: T?, _ body: (UnsafePointer<T>?) throws -> R) rethrows -> R {
    guard var v else { return try body(nil) }
    return try withUnsafePointer(to: &v) { try body($0) }
}

func withOptionalCString<R>(_ s: String?, _ body: (UnsafePointer<CChar>?) throws -> R) rethrows -> R {
    guard let s else { return try body(nil) }
    return try s.withCString { try body($0) }
}

/// `rows` of exactly `width` values, flattened; `splat` lets a 1-value row
/// stand for `width` copies (a symmetric pad).
func flatRows<T>(_ rows: [[T]], width: Int, what: String, splat: Bool = false) throws -> [T] {
    var out: [T] = []
    out.reserveCapacity(rows.count * width)
    for (i, r) in rows.enumerated() {
        if r.count == width {
            out.append(contentsOf: r)
        } else if splat, r.count == 1 {
            out.append(contentsOf: Array(repeating: r[0], count: width))
        } else {
            throw RLXError(code: RLX_ERR_TYPE, message: "\(what): row \(i) needs \(width) values, got \(r.count)")
        }
    }
    return out
}

/// Named f32 tensors as the parallel C arrays every run / step call takes.
func withInputs<R>(
    _ inputs: [String: [Float]],
    _ body: (UnsafePointer<UnsafePointer<CChar>?>, UnsafePointer<UnsafePointer<Float>?>, UnsafePointer<Int>, Int) throws -> R
) throws -> R {
    let names = inputs.keys.sorted()
    let cNames = names.map { strdup($0) }
    defer { cNames.forEach { free($0) } }
    let buffers: [UnsafeMutableBufferPointer<Float>] = names.map { name in
        let values = inputs[name]!
        let buf = UnsafeMutableBufferPointer<Float>.allocate(capacity: max(values.count, 1))
        _ = buf.initialize(from: values)
        return buf
    }
    defer { buffers.forEach { $0.deallocate() } }
    let namePtrs: [UnsafePointer<CChar>?] = cNames.map { UnsafePointer($0) }
    let dataPtrs: [UnsafePointer<Float>?] = buffers.map { UnsafePointer($0.baseAddress) }
    let lens: [Int] = names.map { inputs[$0]!.count }
    return try namePtrs.withUnsafeBufferPointer { n in
        try dataPtrs.withUnsafeBufferPointer { d in
            try lens.withUnsafeBufferPointer { l in
                try body(n.baseAddress!, d.baseAddress!, l.baseAddress!, names.count)
            }
        }
    }
}

/// A list of strings as a C `const char* const*`.
func withCStrings<R>(_ strings: [String], _ body: (UnsafePointer<UnsafePointer<CChar>?>?, Int) throws -> R) rethrows -> R {
    let c = strings.map { strdup($0) }
    defer { c.forEach { free($0) } }
    let ptrs: [UnsafePointer<CChar>?] = c.map { UnsafePointer($0) }
    return try ptrs.withUnsafeBufferPointer { try body($0.baseAddress, strings.count) }
}

/// `nil` → `(NULL, 0)`, else a list of strings as a C `const char* const*`.
func withOptionalCStrings<R>(_ strings: [String]?, _ body: (UnsafePointer<UnsafePointer<CChar>?>?, Int) throws -> R) rethrows -> R {
    guard let strings else { return try body(nil, 0) }
    return try withCStrings(strings, body)
}

/// Raw bytes of one tensor and its dtype name (`"f32"`, `"i64"`, …).
public struct TypedTensor: Equatable {
    public let bytes: [UInt8]
    public let dtype: String
    public init(bytes: [UInt8], dtype: String) {
        self.bytes = bytes
        self.dtype = dtype
    }
}

/// Named typed tensors as the parallel C arrays a typed run takes: names,
/// byte pointers, byte lengths, dtype names, count.
func withTypedInputs<R>(
    _ inputs: [String: TypedTensor],
    _ body: (UnsafePointer<UnsafePointer<CChar>?>?, UnsafePointer<UnsafePointer<UInt8>?>, UnsafePointer<Int>, UnsafePointer<UnsafePointer<CChar>?>?, Int) throws -> R
) throws -> R {
    let names = inputs.keys.sorted()
    let tensors = names.map { inputs[$0]! }
    let store = tensors.map { t -> UnsafeMutableBufferPointer<UInt8> in
        let buf = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: max(t.bytes.count, 1))
        _ = buf.initialize(from: t.bytes)
        return buf
    }
    defer { store.forEach { $0.deallocate() } }
    let data: [UnsafePointer<UInt8>?] = store.map { UnsafePointer($0.baseAddress) }
    let lens = tensors.map(\.bytes.count)
    return try withCStrings(names) { n, count in
        try withCStrings(tensors.map(\.dtype)) { d, _ in
            try data.withUnsafeBufferPointer { dp in
                try lens.withUnsafeBufferPointer { lp in
                    try body(n, dp.baseAddress!, lp.baseAddress!, d, count)
                }
            }
        }
    }
}

/// A JSON string rlx returned, parsed.
func takeJSON(_ p: UnsafeMutablePointer<CChar>?) throws -> Any {
    try parseJSON(takeString(p))
}

/// A JSON object rlx returned.
func takeObject(_ p: UnsafeMutablePointer<CChar>?) throws -> [String: Any] {
    guard let o = try takeJSON(p) as? [String: Any] else {
        throw RLXError(code: RLX_ERR_RUNTIME, message: "rlx returned JSON that is not an object")
    }
    return o
}

/// A JSON array of objects rlx returned.
func takeRows(_ p: UnsafeMutablePointer<CChar>?) throws -> [[String: Any]] {
    guard let rows = try takeJSON(p) as? [[String: Any]] else {
        throw RLXError(code: RLX_ERR_RUNTIME, message: "rlx returned JSON that is not a list of objects")
    }
    return rows
}

/// Take ownership of run outputs: one f32 array per graph output.
func collectOutputs(_ outs: OpaquePointer) throws -> [[Float]] {
    defer { rlx_outputs_free(outs) }
    var result: [[Float]] = []
    for i in 0..<rlx_outputs_len(outs) {
        var p: UnsafePointer<Float>?
        var n = 0
        try check(rlx_outputs_get(outs, i, &p, &n))
        result.append(Array(UnsafeBufferPointer(start: p, count: n)))
    }
    return result
}

/// Take ownership of typed run outputs.
func takeTypedOutputs(_ outs: OpaquePointer) throws -> [TypedTensor] {
    defer { rlx_typed_outputs_free(outs) }
    var result: [TypedTensor] = []
    for i in 0..<rlx_typed_outputs_len(outs) {
        var p: UnsafePointer<UInt8>?
        var len = 0
        var dtype: UnsafePointer<CChar>?
        try check(rlx_typed_outputs_get(outs, i, &p, &len, &dtype))
        result.append(TypedTensor(bytes: Array(UnsafeBufferPointer(start: p, count: len)), dtype: String(cString: dtype!)))
    }
    return result
}

/// Take ownership of named tensors (weights, trainer params).
func takeTensorMap(_ m: OpaquePointer) throws -> [String: [Float]] {
    defer { rlx_tensor_map_free(m) }
    var out: [String: [Float]] = [:]
    for i in 0..<rlx_tensor_map_len(m) {
        var name: UnsafePointer<CChar>?
        var data: UnsafePointer<Float>?
        var n = 0
        try check(rlx_tensor_map_get(m, i, &name, &data, &n))
        out[String(cString: name!)] = Array(UnsafeBufferPointer(start: data, count: n))
    }
    return out
}
