// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Picking a backend: what this build has, what suits a graph, and the ways to
// choose — by name, by cost model, by policy, by benchmark, per run.
//
//     zig build backends

const std = @import("std");
const rlx = @import("rlx");
const print = std.debug.print;

fn model() !rlx.Graph {
    const g = try rlx.Graph.init(.{ .name = "mlp" });
    const x = try g.input("x", &.{ 4, 16 }, .{});
    try g.set_outputs(&.{try g.tanh(try g.matmul(x, try g.param("w", &.{ 16, 16 }, .{})))});
    return g;
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var xs: [64]f32 = undefined;
    for (&xs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) / 7;
    var w: [256]f32 = undefined;
    for (&w, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) / 10 - 0.2;
    const x = [_]rlx.Input{.{ .name = "x", .data = &xs }};

    print("devices: {s}\n", .{try rlx.devices(a)});
    print("metal available: {}\n", .{rlx.is_available("metal")});
    const probe = try model();
    defer probe.deinit();
    print("fastest for graph: {s}\n", .{try rlx.fastest_device_for(a, probe, .{})});

    // One session per device; every device's answer against the first's.
    const names = (try std.json.parseFromSlice([][]const u8, a, try rlx.devices(a), .{})).value;
    var reference: ?[]f32 = null;
    for (names) |name| {
        const s = try rlx.Session.init(.{ .device = try a.dupeZ(u8, name) });
        defer s.deinit();
        const c = try s.compile(try model(), .{});
        defer c.deinit();
        try c.set_param("w", &w);
        const y = (try c.run(a, &x))[0];
        const ref = reference orelse y;
        reference = ref;
        var maxd: f32 = 0;
        for (y, ref) |p, q| maxd = @max(maxd, @abs(p - q));
        print("session {s}: max|Δ| vs first = {e:.2}\n", .{ name, maxd });
    }

    const flex = try rlx.FlexibleSession.init(.{ .policy = "{\"prefer\": [\"metal\", \"cpu\"]}" });
    defer flex.deinit();
    const picked = try flex.compile(try model(), .{});
    defer picked.deinit();
    print("flexible session picked: {s}\n", .{try picked.device(a)});

    const runner = try rlx.Runner.init(try model(), .{});
    defer runner.deinit();
    try runner.set_param("w", &w);
    print("runner warmed: {s}\n", .{try runner.warm_all(a)});
    print("bench: {s}\n", .{try runner.benchmark(a, &x, .{ .runs = 3 })});
    print("runner ran on: {s}\n", .{(try runner.run(a, &x, .{})).device});
    print("run_try chain: {s}\n", .{(try runner.run_try(a, &.{ "metal", "cpu" }, &x)).device});
    const router = try rlx.Router.init(try model(), .{});
    defer router.deinit();
    try router.set_param("w", &w);
    print("router ran on: {s}\n", .{(try router.run_chain(a, &x, .{})).device});
}
