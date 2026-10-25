// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The rows of api.idl marked `manual`: the ones whose Swift shape is not a
// call with arguments and results — a callback-driven training loop, and an
// optimizer step that updates the caller's arrays in place. Everything else
// is generated into API.swift.

import CRLX
import Foundation

extension Optimizer {
    /// Update `params[name]` in place from `grads[name]`.
    public func step(_ params: inout [String: [Float]], grads: [String: [Float]]) throws {
        let names = params.keys.sorted()
        // Stable storage the C call can write through, copied back after.
        let pStore = names.map { name -> UnsafeMutableBufferPointer<Float> in
            let values = params[name]!
            let buf = UnsafeMutableBufferPointer<Float>.allocate(capacity: max(values.count, 1))
            _ = buf.initialize(from: values)
            return buf
        }
        defer { pStore.forEach { $0.deallocate() } }
        let gStore = try names.map { name -> UnsafeMutableBufferPointer<Float> in
            guard let g = grads[name] else {
                throw RLXError(code: RLX_ERR_TYPE, message: "Optimizer.step: no gradient for '\(name)'")
            }
            let buf = UnsafeMutableBufferPointer<Float>.allocate(capacity: max(g.count, 1))
            _ = buf.initialize(from: g)
            return buf
        }
        defer { gStore.forEach { $0.deallocate() } }
        let lens = names.map { params[$0]!.count }
        let pp: [UnsafeMutablePointer<Float>?] = pStore.map { $0.baseAddress }
        let gp: [UnsafePointer<Float>?] = gStore.map { UnsafePointer($0.baseAddress) }
        try withCStrings(names) { namePtrs, n in
            try check(rlx_optimizer_step(handle, namePtrs, pp, gp, lens, nil, nil, n))
        }
        for (i, name) in names.enumerated() {
            params[name] = Array(pStore[i].prefix(lens[i]))
        }
    }
}

extension Trainer {
    /// Run `steps` steps; `batch(step)` returns the inputs or `nil` when the
    /// data is exhausted, `lr(step)` (optional) sets each step's learning
    /// rate, `onStep(step, loss)` returns `false` to stop early. Returns the
    /// run summary.
    @discardableResult
    public func run(
        steps: Int,
        batch: @escaping (UInt64) -> [String: [Float]]?,
        lr: ((UInt64) -> Float)? = nil,
        every: Int = 1,
        onStep: ((UInt64, Float) -> Bool)? = nil
    ) throws -> [String: Any] {
        final class Box {
            let batch: (UInt64) -> [String: [Float]]?
            let lr: ((UInt64) -> Float)?
            let onStep: ((UInt64, Float) -> Bool)?
            init(_ b: @escaping (UInt64) -> [String: [Float]]?, _ l: ((UInt64) -> Float)?, _ s: ((UInt64, Float) -> Bool)?) {
                batch = b
                lr = l
                onStep = s
            }
        }
        let box = Box(batch, lr, onStep)
        let user = Unmanaged.passUnretained(box).toOpaque()
        let batchFn: RlxBatchFn = { user, step, b in
            let box = Unmanaged<Box>.fromOpaque(user!).takeUnretainedValue()
            guard let inputs = box.batch(step) else { return 0 }
            for (name, data) in inputs where rlx_batch_set(b, name, data, data.count) != RLX_OK {
                return -1
            }
            return 1
        }
        let lrFn: RlxLrFn = { user, step, lr in
            guard let f = Unmanaged<Box>.fromOpaque(user!).takeUnretainedValue().lr else { return 0 }
            lr?.pointee = f(step)
            return 1
        }
        let stepFn: RlxStepFn = { user, step, loss in
            let box = Unmanaged<Box>.fromOpaque(user!).takeUnretainedValue()
            return (box.onStep?(step, loss) ?? true) ? 1 : 0
        }
        var s: UnsafeMutablePointer<CChar>?
        try withExtendedLifetime(box) {
            try check(rlx_trainer_run(handle, steps, batchFn, lr == nil ? nil : lrFn, onStep == nil ? nil : stepFn, every, user, &s))
        }
        return try takeObject(s)
    }
}

#if RLX_LINKED_PLUGINS
/// Register the model plugins linked into the library (an iOS app's
/// RLX.xcframework built with `ios/build-xcframework.sh --capi-crate`) and
/// return how many providers they add. Call it once, before the first
/// `Model.open`. Built with `RLX_CAPI_FEATURES=linked-plugins`.
@discardableResult
public func linkModelPlugins() -> Int {
    Int(rlx_link_model_plugins())
}
#endif
