// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Training: fit y = x·w_true with the Trainer (its loop driven by C callbacks),
// checkpoint and resume it, then the same fit by hand with `grad` and an
// Optimizer.
//
//     zig build training

const std = @import("std");
const rlx = @import("rlx");
const print = std.debug.print;

/// A scratch file under `TMPDIR` (or the working directory).
fn scratch(a: std.mem.Allocator, name: []const u8) ![:0]u8 {
    const dir = if (std.c.getenv("TMPDIR")) |d| std.mem.span(d) else ".";
    return std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ std.mem.trimEnd(u8, dir, "/"), name }, 0);
}

const W_TRUE = [3]f32{ 2, -1, 0.5 };
const BATCH = 8;

const Batch = struct { x: [BATCH * 3]f32, y: [BATCH]f32 };

fn data(step: u64) Batch {
    var b: Batch = .{ .x = undefined, .y = .{0} ** BATCH };
    for (0..BATCH) |i| {
        for (0..3) |k| {
            const v = @as(f32, @floatFromInt((step * 31 + i * 7 + k * 3) % 17)) / 8.5 - 1;
            b.x[i * 3 + k] = v;
            b.y[i] += v * W_TRUE[k];
        }
    }
    return b;
}

fn inputs(b: *const Batch) [2]rlx.Input {
    return .{ .{ .name = "x", .data = &b.x }, .{ .name = "y", .data = &b.y } };
}

// The C callbacks `Trainer.run` drives.
fn batchFn(_: ?*anyopaque, step: u64, batch: ?*rlx.c.RlxBatch) callconv(.c) i32 {
    const b = data(step);
    if (!rlx.batchSet(batch, "x", &b.x) or !rlx.batchSet(batch, "y", &b.y)) return -1;
    return 1;
}
fn lrFn(_: ?*anyopaque, step: u64, lr: [*c]f32) callconv(.c) i32 {
    lr.* = @floatCast(0.1 * 0.5 * (1 + @cos(std.math.pi * @as(f64, @floatFromInt(step)) / 400)));
    return 1;
}
fn onStepFn(_: ?*anyopaque, _: u64, loss: f32) callconv(.c) i32 {
    return if (loss > 1e-6) 1 else 0;
}

fn regression() !struct { g: rlx.Graph, w: rlx.Node } {
    const g = try rlx.Graph.init(.{ .name = "regression" });
    const x = try g.input("x", &.{ BATCH, 3 }, .{});
    const y = try g.input("y", &.{ BATCH, 1 }, .{});
    const w = try g.param("w", &.{ 3, 1 }, .{});
    const d = try g.sub(try g.matmul(x, w), y);
    try g.set_outputs(&.{try g.mean(try g.mul(d, d), &.{ 0, 1 }, .{})});
    return .{ .g = g, .w = w };
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try regression();
    const zero = [3]f32{ 0, 0, 0 };

    const trainer = try rlx.Trainer.init(a, try r.g.clone(), &.{"w"}, &.{.{ .name = "w", .data = &zero }}, .{
        .options = "{\"device\": \"cpu\", \"optimizer\": {\"kind\": \"adamw\", \"lr\": 0.1}}",
    });
    defer trainer.deinit();
    var b0 = data(0);
    const first = try trainer.evaluate(a, &inputs(&b0));
    _ = try trainer.run(a, 400, .{ .batch = batchFn, .lr = lrFn, .on_step = onStepFn });
    print("trainer loss: {d:.4} -> {d:.6} in {d} steps\n", .{ first, try trainer.evaluate(a, &inputs(&b0)), try trainer.steps() });
    const w = (try trainer.params(a))[0].data;
    print("trainer w: {d:.3} {d:.3} {d:.3}\n", .{ w[0], w[1], w[2] });

    const path = try scratch(a, "rlx-example-train-zig.gguf");
    const written = try trainer.save(path);
    const restored = try rlx.Trainer.init(a, try r.g.clone(), &.{"w"}, &.{.{ .name = "w", .data = &zero }}, .{ .options = "{\"device\": \"cpu\"}" });
    defer restored.deinit();
    print("checkpoint: {d} tensors, restored {s}\n", .{ written, try restored.load(a, path) });
    _ = std.c.unlink(path);
    _ = try restored.freeze(.{ .names = &.{"w"} });
    const before = (try restored.params(a))[0].data;
    var b1 = data(1);
    _ = try restored.step(a, &inputs(&b1));
    print("frozen: {s} unchanged: {}\n", .{ try restored.frozen(a), std.mem.eql(f32, before, (try restored.params(a))[0].data) });
    _ = try restored.unfreeze(.{});

    const session = try rlx.Session.init(.{});
    defer session.deinit();
    const backward = try session.compile(try rlx.grad(r.g, &.{r.w}), .{}); // outputs [loss, dw]
    defer backward.deinit();
    const opt = try rlx.Optimizer.init(.{ .kind = "sgd", .options = "{\"lr\": 0.2}" });
    defer opt.deinit();
    var params = [3]f32{ 0, 0, 0 };
    var loss: f32 = 0;
    for (0..300) |step| {
        try backward.set_param("w", &params);
        var b = data(step);
        const outs = try backward.run(a, &.{ .{ .name = "x", .data = &b.x }, .{ .name = "y", .data = &b.y }, .{ .name = "d_output", .data = &.{1} } });
        loss = outs[0][0];
        try opt.step(a, &.{.{ .name = "w", .param = &params, .grad = outs[1] }});
    }
    print("manual loss: {d:.6} w: {d:.3} {d:.3} {d:.3}\n", .{ loss, params[0], params[1], params[2] });
    r.g.deinit();
}
