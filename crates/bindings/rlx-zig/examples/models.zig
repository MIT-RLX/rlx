// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Models: any provider's model, opened by path or id and run by name. A graph
// saved as an .rlxp package runs anywhere; a plugin's models (a downstream
// collection's LLMs, or the reference plugin's demo:count) load through
// RLX_MODEL_PLUGINS.
//
//     RLX_MODEL_PLUGINS=…/librlx_model_plugin.dylib zig build models

const std = @import("std");
const rlx = @import("rlx");
const print = std.debug.print;

const Info = struct { family: []const u8, task: []const u8, device: []const u8 };

fn infoOf(a: std.mem.Allocator, m: rlx.Model) !Info {
    const opts: std.json.ParseOptions = .{ .ignore_unknown_fields = true };
    return (try std.json.parseFromSlice(Info, a, try m.info(a), opts)).value;
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const g = try rlx.Graph.init(.{ .name = "affine" });
    defer g.deinit();
    const x = try g.input("x", &.{ 1, 2 }, .{});
    try g.set_outputs(&.{try g.matmul(x, try g.param("w", &.{ 2, 2 }, .{}))});
    const dir = if (std.c.getenv("TMPDIR")) |d| std.mem.span(d) else ".";
    const path = try std.fmt.allocPrintSentinel(a, "{s}/rlx-example-zig.rlxp", .{dir}, 0);
    try rlx.save_model(a, g, path, &.{.{ .name = "w", .data = &.{ 2, 0, 0, 3 } }}, .{
        .model = "{\"task\": \"graph\", \"family\": \"affine\", \"outputs\": [\"y\"]}",
    });
    defer _ = std.c.unlink(path);

    const m = try rlx.Model.open(path, .{ .options = "{\"device\": \"cpu\"}" });
    defer m.deinit();
    const info = try infoOf(a, m);
    print("model: {s} ({s}) on {s}\n", .{ info.family, info.task, info.device });
    const y = (try m.run(a, &.{.{ .name = "x", .data = &.{ 1, 1 } }}, .{}))[0];
    print("model run: {d:.3} {d:.3}\n", .{ y[0], y[1] });

    const lm = try rlx.Model.open("demo:count", .{});
    defer lm.deinit();
    const lm_info = try infoOf(a, lm);
    print("plugin model: {s} ({s})\n", .{ lm_info.family, lm_info.task });
    print("model text: {s}\n", .{try lm.generate_text(a, "c", .{ .options = "{\"max_tokens\": 4}" })});
    print("model ids:", .{});
    for (try lm.generate(a, &.{2}, .{ .max_tokens = 3 })) |id| print(" {d}", .{id});
    print("\n", .{});

    // By task: a chat, speech out and back in, embeddings.
    const msgs = "[{\"role\": \"user\", \"content\": \"c\"}]";
    print("chat: {s}\n", .{try lm.chat(a, msgs, .{ .options = "{\"max_tokens\": 4}" })});
    const voice = try rlx.Model.open("demo:voice", .{});
    defer voice.deinit();
    const speech = try voice.synthesize(a, "hello", .{});
    print("voice: {d} samples at {d} Hz\n", .{ speech.audio.len, speech.sample_rate });
    const ear = try rlx.Model.open("demo:speech", .{});
    defer ear.deinit();
    print("heard: {s}\n", .{try ear.transcribe(a, speech.audio, .{ .sample_rate = speech.sample_rate })});
    const embedder = try rlx.Model.open("demo:embed", .{});
    defer embedder.deinit();
    const rows_json = try embedder.embed(a, &.{ "hello", "sky" }, .{});
    const rows = try std.json.parseFromSliceLeaky([]const []const f64, a, rows_json, .{});
    print("embed:", .{});
    for (rows) |row| for (row) |v| print(" {d}", .{@as(i64, @intFromFloat(v))});
    print("\n", .{});
}
