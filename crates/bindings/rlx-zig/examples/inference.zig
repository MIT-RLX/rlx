// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Inference: build a graph, compile it for a device, set its weights, run it.
//
//     zig build inference
//
// A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].

const std = @import("std");
const rlx = @import("rlx");
const print = std.debug.print;

fn weights(comptime n: usize, seed: usize) [n]f32 {
    var w: [n]f32 = undefined;
    for (&w, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7919 + seed * 104729) % 1000)) / 1000 - 0.5;
    return w;
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const g = try rlx.Graph.init(.{ .name = "classifier" });
    defer g.deinit();
    const x = try g.input("x", &.{ 1, 4 }, .{});
    const hidden = try g.relu(try g.matmul(x, try g.param("w1", &.{ 4, 8 }, .{})));
    try g.set_outputs(&.{try g.softmax(try g.matmul(hidden, try g.param("w2", &.{ 8, 3 }, .{})), .{})});

    const session = try rlx.Session.init(.{ .device = "cpu" });
    defer session.deinit();
    const model = try session.compile(g, .{}); // consumes g
    defer model.deinit();
    print("compiled on: {s}\n", .{try model.device(a)});
    const w1 = weights(32, 1);
    const w2 = weights(24, 2);
    _ = try model.set_params(a, &.{ .{ .name = "w1", .data = &w1 }, .{ .name = "w2", .data = &w2 } });
    const probs = (try model.run(a, &.{.{ .name = "x", .data = &.{ 0.5, -1, 2, 0.25 } }}))[0];
    print("probs:", .{});
    var sum: f32 = 0;
    for (probs) |p| {
        print(" {d:.4}", .{p});
        sum += p;
    }
    print("\nprobs sum: {d:.4}\n", .{sum});
    print("output shapes: {s} dtypes: {s}\n", .{ try model.output_shapes(a), try model.output_dtypes(a) });
}
