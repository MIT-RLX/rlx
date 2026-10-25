// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Distributed: three ranks as threads on loopback; across machines, one rank
// per host with the same peers. Collectives, pipelined inference (a stage
// shipped per worker), data-parallel training.
//
//     zig build distributed

const std = @import("std");
const rlx = @import("rlx");
const print = std.debug.print;

/// A scratch file under `TMPDIR` (or the working directory).
fn scratch(a: std.mem.Allocator, name: []const u8) ![:0]u8 {
    const dir = if (std.c.getenv("TMPDIR")) |d| std.mem.span(d) else ".";
    return std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ std.mem.trimEnd(u8, dir, "/"), name }, 0);
}

const WORLD = 3;
const PER_RANK = 16;

fn writeBlob(a: std.mem.Allocator, name: []const u8, values: []const f32) ![:0]u8 {
    const path = try scratch(a, try std.fmt.allocPrint(a, "rlx-dist-zig-{s}", .{name}));
    const fd = std.c.fopen(path, "wb").?;
    const bytes = std.mem.sliceAsBytes(values);
    _ = std.c.fwrite(bytes.ptr, 1, bytes.len, fd);
    _ = std.c.fclose(fd);
    return std.fmt.allocPrintSentinel(a, "file://{s}", .{path}, 0);
}

fn matvec(x: [4]f32, w: [16]f32) [4]f32 {
    var out: [4]f32 = undefined;
    for (0..4) |j| {
        var s: f32 = 0;
        for (0..4) |i| s += x[i] * w[i * 4 + j];
        out[j] = @max(0, s);
    }
    return out;
}

fn num(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => 0,
    };
}

/// A rank's training job: its shard of X / y, the shared init and schedule.
fn trainSpec(a: std.mem.Allocator, rank: usize, ux: []const u8, uy: []const u8, uw: []const u8) ![:0]u8 {
    const start = rank * PER_RANK;
    return std.fmt.allocPrintSentinel(a,
        \\{{"params": [{{"name": "w", "uri": "{s}"}}], "data": [{{"input": "X", "uri": "{s}", "elem": 3, "shard_start": {d}, "shard_len": {d}}}, {{"input": "y", "uri": "{s}", "elem": 1, "shard_start": {d}, "shard_len": {d}}}], "batch": 4, "lr": 0.2, "epochs": 40, "device": "cpu"}}
    , .{ uw, ux, start, PER_RANK, uy, start, PER_RANK }, 0);
}

/// `params.w` of a training report, as f32.
fn weightsOf(a: std.mem.Allocator, report: []const u8) ![3]f32 {
    const parsed = try std.json.parseFromSlice(std.json.Value, a, report, .{});
    const w = parsed.value.object.get("params").?.object.get("w").?.array.items;
    var out: [3]f32 = undefined;
    for (&out, w) |*o, v| o.* = @floatCast(num(v));
    return out;
}

fn worker(rank: u32, peers: [:0]const u8, result: *[3]f32) void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const opts = std.fmt.allocPrintSentinel(a, "{{\"peers\": {s}, \"device\": \"cpu\"}}", .{peers}, 0) catch return;
    const g = rlx.Group.connect(rank, WORLD, .{ .options = opts }) catch return;
    defer g.deinit();
    const me = [_]f32{@floatFromInt(rank + 1)};
    _ = g.all_reduce(a, &me, .{ .op = "sum" }) catch return;
    _ = g.all_gather(a, &me) catch return;
    _ = g.broadcast(a, 0, &.{0}) catch return; // the root's length; it overwrites
    _ = g.serve_worker(a, .{ .options = "{\"steps\": 2}" }) catch return;
    const report = g.serve_trainer(a, .{}) catch return;
    result.* = weightsOf(a, report) catch return;
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var stage_w: [3][16]f32 = undefined;
    for (0..16) |i| {
        stage_w[1][i] = @as(f32, @floatFromInt(i * 5 % 7)) / 7 - 0.3;
        stage_w[2][i] = @as(f32, @floatFromInt(i * 3 % 5)) / 5 - 0.2;
    }
    const w_true = [3]f32{ 1.5, -2, 0.5 };
    var x_all: [PER_RANK * WORLD * 3]f32 = undefined;
    for (&x_all, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i * 37 % 23)) / 11.5 - 1;
    var y_all: [PER_RANK * WORLD]f32 = .{0} ** (PER_RANK * WORLD);
    for (&y_all, 0..) |*v, i| {
        for (0..3) |k| v.* += x_all[i * 3 + k] * w_true[k];
    }

    const peers = try rlx.loopback_peers(a, WORLD);
    const peers_z = try a.dupeZ(u8, peers);
    var results: [WORLD][3]f32 = undefined;
    var threads: [WORLD - 1]std.Thread = undefined;
    for (&threads, 1..) |*t, rank| t.* = try std.Thread.spawn(.{}, worker, .{ @as(u32, @intCast(rank)), peers_z, &results[rank] });

    const opts = try std.fmt.allocPrintSentinel(a, "{{\"peers\": {s}, \"device\": \"cpu\"}}", .{peers}, 0);
    const g = try rlx.Group.connect(0, WORLD, .{ .options = opts });
    defer g.deinit();
    print("mesh: rank {d} of {d}, leader: {}\n", .{ try g.rank(), try g.world(), try g.is_leader() });
    print("all_reduce sum: {any}\n", .{try g.all_reduce(a, &.{1}, .{ .op = "sum" })});
    print("all_gather: {any}\n", .{try g.all_gather(a, &.{1})});
    print("broadcast: {any}\n", .{try g.broadcast(a, 0, &.{42})});
    print("workers: {s}\n", .{try g.collect_caps(a)});

    const stage = try rlx.Graph.init(.{ .name = "stage" });
    defer stage.deinit();
    const sx = try stage.input("x", &.{ 1, 4 }, .{});
    try stage.set_outputs(&.{try stage.relu(try stage.matmul(sx, try stage.param("w", &.{ 4, 4 }, .{})))});
    for (1..WORLD) |r| {
        const uri = try writeBlob(a, try std.fmt.allocPrint(a, "w{d}.bin", .{r}), &stage_w[r]);
        const spec = try std.fmt.allocPrintSentinel(a, "{{\"device\": \"cpu\", \"weights\": {{\"w\": \"{s}\"}}}}", .{uri}, 0);
        try g.ship_stage(@intCast(r), stage, spec);
    }
    for ([_][4]f32{ .{ 1, -1, 0.5, 2 }, .{ 0.25, 0.5, -0.75, 1 } }) |xv| {
        try g.send_activation(1, &xv);
        const yv = try g.recv_activation(a, WORLD - 1);
        const want = matvec(matvec(xv, stage_w[1]), stage_w[2]);
        var maxd: f32 = 0;
        for (yv, want) |got, exp| maxd = @max(maxd, @abs(got - exp));
        print("pipeline max|Δ| vs local = {e:.1}\n", .{maxd});
    }

    _ = try g.collect_caps(a);
    const lr = try rlx.Graph.init(.{ .name = "lr" });
    defer lr.deinit();
    const X = try lr.input("X", &.{ 4, 3 }, .{});
    const Y = try lr.input("y", &.{ 4, 1 }, .{});
    const wv = try lr.param("w", &.{ 3, 1 }, .{});
    const d = try lr.sub(try lr.matmul(X, wv), Y);
    try lr.set_outputs(&.{try lr.mean(try lr.mul(d, d), &.{ 0, 1 }, .{})});
    const bwd = try rlx.grad(lr, &.{wv});
    defer bwd.deinit();
    const ux = try writeBlob(a, "X.bin", &x_all);
    const uy = try writeBlob(a, "y.bin", &y_all);
    const uw = try writeBlob(a, "w0.bin", &.{ 0, 0, 0 });
    for (1..WORLD) |r| try g.ship_train(@intCast(r), bwd, try trainSpec(a, r, ux, uy, uw));
    const mine = try g.run_train(a, bwd, try trainSpec(a, 0, ux, uy, uw));
    const metrics = (try std.json.parseFromSlice(std.json.Value, a, mine, .{})).value.object.get("metrics").?.object;
    print("training loss: {d:.4} -> {e:.2}\n", .{ num(metrics.get("first_loss").?), num(metrics.get("last_loss").?) });
    const w0 = try weightsOf(a, mine);
    print("rank 0 w: {d:.3} {d:.3} {d:.3} (true: 1.5 -2 0.5)\n", .{ w0[0], w0[1], w0[2] });
    for (&threads) |t| t.join();
    for (1..WORLD) |r| {
        var agree = true;
        for (0..3) |i| agree = agree and @abs(results[r][i] - w0[i]) < 1e-5;
        print("rank {d} w agrees: {}\n", .{ r, agree });
    }
}
