// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// What the generated `rlx.zig` stands on: the C import, errors, the types that
// carry tensors across, copying rlx's buffers into the caller's allocator, and
// the two rows whose shape is not a plain call (`Optimizer.step` updates the
// caller's arrays in place; `Trainer.run` takes C callbacks).

const std = @import("std");
const rlx = @import("rlx.zig");
pub const c = @cImport(@cInclude("rlx.h"));

/// A failed call, by `RLX_ERR_*` status; `lastError()` has the message.
pub const Error = error{ Type, Range, Runtime, Panic, NotFound, Unknown, OutOfMemory };

/// A node of a graph — the id every builder returns.
pub const Node = u32;
/// "No node", for an optional node argument.
pub const NONE: Node = std.math.maxInt(u32);

/// One named f32 tensor: a run's input, a parameter.
pub const Input = struct { name: [*:0]const u8, data: []const f32 };
/// One named tensor of any dtype, as raw bytes.
pub const TypedInput = struct { name: [*:0]const u8, bytes: []const u8, dtype: [*:0]const u8 };
/// A typed result: raw bytes and a dtype name, owned by the caller's allocator.
pub const TypedTensor = struct { bytes: []u8, dtype: []u8 };
/// A named f32 tensor rlx returned (weights, trainer params).
pub const Tensor = struct { name: []u8, data: []f32 };

/// The message of the last failed call on this thread.
pub fn lastError() []const u8 {
    return std.mem.span(c.rlx_last_error());
}

pub fn check(rc: c_int) Error!void {
    return switch (rc) {
        0 => {},
        -1 => Error.Type,
        -2 => Error.Range,
        -3 => Error.Runtime,
        -4 => Error.Panic,
        -5 => Error.NotFound,
        else => Error.Unknown,
    };
}

pub fn takeString(alloc: std.mem.Allocator, p: [*c]u8) Error![]u8 {
    if (p == null) return alloc.alloc(u8, 0);
    defer c.rlx_string_free(p);
    return alloc.dupe(u8, std.mem.span(p));
}

pub fn takeSlice(comptime T: type, alloc: std.mem.Allocator, v: anytype, free: anytype) Error![]T {
    defer free(v);
    if (v.ptr == null) return alloc.alloc(T, 0);
    return alloc.dupe(T, v.ptr[0..v.len]);
}

pub fn takeOutputs(alloc: std.mem.Allocator, o: *c.RlxOutputs) Error![][]f32 {
    defer c.rlx_outputs_free(o);
    const n = c.rlx_outputs_len(o);
    const out = try alloc.alloc([]f32, n);
    for (out, 0..) |*slot, i| {
        var p: [*c]const f32 = null;
        var len: usize = 0;
        try check(c.rlx_outputs_get(o, i, &p, &len));
        slot.* = try alloc.dupe(f32, p[0..len]);
    }
    return out;
}

pub fn takeTypedOutputs(alloc: std.mem.Allocator, o: *c.RlxTypedOutputs) Error![]TypedTensor {
    defer c.rlx_typed_outputs_free(o);
    const n = c.rlx_typed_outputs_len(o);
    const out = try alloc.alloc(TypedTensor, n);
    for (out, 0..) |*slot, i| {
        var p: [*c]const u8 = null;
        var len: usize = 0;
        var dtype: [*c]const u8 = null;
        try check(c.rlx_typed_outputs_get(o, i, &p, &len, &dtype));
        slot.* = .{ .bytes = try alloc.dupe(u8, p[0..len]), .dtype = try alloc.dupe(u8, std.mem.span(dtype)) };
    }
    return out;
}

pub fn takeTensors(alloc: std.mem.Allocator, m: *c.RlxTensorMap) Error![]Tensor {
    defer c.rlx_tensor_map_free(m);
    const n = c.rlx_tensor_map_len(m);
    const out = try alloc.alloc(Tensor, n);
    for (out, 0..) |*slot, i| {
        var name: [*c]const u8 = null;
        var p: [*c]const f32 = null;
        var len: usize = 0;
        try check(c.rlx_tensor_map_get(m, i, &name, &p, &len));
        slot.* = .{ .name = try alloc.dupe(u8, std.mem.span(name)), .data = try alloc.dupe(f32, p[0..len]) };
    }
    return out;
}

/// The parallel C arrays a run's named inputs travel as.
pub const CInputs = struct {
    names: [][*c]const u8,
    data: [][*c]const f32,
    lens: []usize,

    pub fn deinit(self: CInputs, alloc: std.mem.Allocator) void {
        alloc.free(self.names);
        alloc.free(self.data);
        alloc.free(self.lens);
    }
};

pub fn cInputs(alloc: std.mem.Allocator, inputs: []const Input) Error!CInputs {
    const r = CInputs{
        .names = try alloc.alloc([*c]const u8, inputs.len),
        .data = try alloc.alloc([*c]const f32, inputs.len),
        .lens = try alloc.alloc(usize, inputs.len),
    };
    for (inputs, 0..) |in, i| {
        r.names[i] = in.name;
        r.data[i] = in.data.ptr;
        r.lens[i] = in.data.len;
    }
    return r;
}

pub const CTypedInputs = struct {
    names: [][*c]const u8,
    data: [][*c]const u8,
    lens: []usize,
    dtypes: [][*c]const u8,

    pub fn deinit(self: CTypedInputs, alloc: std.mem.Allocator) void {
        alloc.free(self.names);
        alloc.free(self.data);
        alloc.free(self.lens);
        alloc.free(self.dtypes);
    }
};

pub fn cTypedInputs(alloc: std.mem.Allocator, inputs: []const TypedInput) Error!CTypedInputs {
    const r = CTypedInputs{
        .names = try alloc.alloc([*c]const u8, inputs.len),
        .data = try alloc.alloc([*c]const u8, inputs.len),
        .lens = try alloc.alloc(usize, inputs.len),
        .dtypes = try alloc.alloc([*c]const u8, inputs.len),
    };
    for (inputs, 0..) |in, i| {
        r.names[i] = in.name;
        r.data[i] = in.bytes.ptr;
        r.lens[i] = in.bytes.len;
        r.dtypes[i] = in.dtype;
    }
    return r;
}

// ── the manual rows ─────────────────────────────────────────

/// One parameter for `Optimizer.step`: updated in place from `grad`.
pub const StepParam = struct { name: [*:0]const u8, param: []f32, grad: []const f32 };

/// `Optimizer.step`: update every `param` in place from its `grad`.
pub fn optimizer_step(self: rlx.Optimizer, alloc: std.mem.Allocator, params: []const StepParam) Error!void {
    const names = try alloc.alloc([*c]const u8, params.len);
    defer alloc.free(names);
    const ps = try alloc.alloc([*c]f32, params.len);
    defer alloc.free(ps);
    const gs = try alloc.alloc([*c]const f32, params.len);
    defer alloc.free(gs);
    const lens = try alloc.alloc(usize, params.len);
    defer alloc.free(lens);
    for (params, 0..) |p, i| {
        names[i] = p.name;
        ps[i] = p.param.ptr;
        gs[i] = p.grad.ptr;
        lens[i] = p.param.len;
    }
    try check(c.rlx_optimizer_step(self.h, names.ptr, ps.ptr, gs.ptr, lens.ptr, null, null, params.len));
}

/// Fill one step's batch from inside a `Trainer.run` batch callback.
pub fn batchSet(batch: ?*c.RlxBatch, name: [*:0]const u8, data: []const f32) bool {
    return c.rlx_batch_set(batch, name, data.ptr, data.len) == 0;
}

/// The callbacks `Trainer.run` drives; `user` is handed back to each.
pub const RunCallbacks = struct {
    user: ?*anyopaque = null,
    /// Fill the batch (with `batchSet`); return 1 to go on, 0 when the data is
    /// exhausted, negative to abort.
    batch: c.RlxBatchFn,
    /// Write `*lr` and return 1, or return 0 to keep the current rate.
    lr: c.RlxLrFn = null,
    /// Called every `every` steps with the loss: return 0 to stop.
    on_step: c.RlxStepFn = null,
    every: usize = 1,
};

/// `Trainer.run`: `steps` steps pulling batches from `cb.batch`; the summary JSON.
pub fn trainer_run(self: rlx.Trainer, alloc: std.mem.Allocator, steps: usize, cb: RunCallbacks) Error![]u8 {
    var summary: [*c]u8 = null;
    try check(c.rlx_trainer_run(self.h, steps, cb.batch, cb.lr, cb.on_step, cb.every, cb.user, &summary));
    return takeString(alloc, summary);
}
