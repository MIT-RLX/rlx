// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
// into a compiled graph, and quantizing in memory.
//
//     zig build loading

const std = @import("std");
const rlx = @import("rlx");
const print = std.debug.print;

/// A scratch file under `TMPDIR` (or the working directory).
fn scratch(a: std.mem.Allocator, name: []const u8) ![:0]u8 {
    const dir = if (std.c.getenv("TMPDIR")) |d| std.mem.span(d) else ".";
    return std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ std.mem.trimEnd(u8, dir, "/"), name }, 0);
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var w: [64]f32 = undefined;
    for (&w, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 9)) / 4 - 1;

    const gguf = try scratch(a, "rlx-example-zig.gguf");
    const writer = try rlx.GgufWriter.init();
    defer writer.deinit();
    try writer.add_f32("proj.weight", &.{ 8, 8 }, "F32", &w);
    try writer.add_f32("proj.weight_q8", &.{ 8, 8 }, "Q8_0", &w);
    _ = try writer.write(gguf, .{ .architecture = "demo", .metadata = "{\"demo.layers\": 1}" });
    const f = try rlx.Gguf.open(gguf);
    defer f.deinit();
    print("gguf architecture: {s}\n", .{(try f.architecture(a)).?});
    print("gguf tensors: {s}\n", .{try f.tensor_names(a)});
    const exact = try f.tensor(a, "proj.weight");
    const q8 = try f.tensor(a, "proj.weight_q8");
    var maxd: f32 = 0;
    for (exact, q8) |e, q| maxd = @max(maxd, @abs(e - q));
    print("gguf f32 exact: {} q8 max|Δ|: {d:.4}\n", .{ std.mem.eql(f32, exact, &w), maxd });

    const g = try rlx.Graph.init(.{ .name = "proj" });
    defer g.deinit();
    const x = try g.input("x", &.{ 1, 8 }, .{});
    try g.set_outputs(&.{try g.matmul(x, try g.param("proj.weight", &.{ 8, 8 }, .{}))});
    const s = try rlx.Session.init(.{});
    defer s.deinit();
    const c = try s.compile(g, .{});
    defer c.deinit();
    try c.set_param("proj.weight", exact);
    const y = (try c.run(a, &.{.{ .name = "x", .data = &(.{1} ** 8) }}))[0];
    print("gguf -> graph: {d:.2} {d:.2} {d:.2}\n", .{ y[0], y[1], y[2] });

    // A safetensors file: an 8-byte header length, a JSON header, raw data.
    const st = try scratch(a, "rlx-example-zig.safetensors");
    const header = "{\"proj.weight\":{\"dtype\":\"F32\",\"shape\":[8,8],\"data_offsets\":[0,256]}}    ";
    var file = try a.alloc(u8, 8 + header.len + 256);
    std.mem.writeInt(u64, file[0..8], header.len, .little);
    @memcpy(file[8 .. 8 + header.len], header);
    @memcpy(file[8 + header.len ..], std.mem.sliceAsBytes(&w));
    const fd = std.c.fopen(st, "wb").?;
    _ = std.c.fwrite(file.ptr, 1, file.len, fd);
    _ = std.c.fclose(fd);
    const loaded = try rlx.load_weights(a, "safetensors", st);
    print("safetensors: [{s}] exact: {}\n", .{ loaded[0].name, std.mem.eql(f32, loaded[0].data, &w) });

    const packed_ = try rlx.quantize(a, &w, "Q8_0");
    const back = try rlx.dequantize(a, packed_, "Q8_0", .{ .count = w.len });
    print("quantize q8_0: {d} -> {d} bytes, round trip {d} values\n", .{ w.len * 4, packed_.len, back.len });
    _ = std.c.unlink(gguf);
    _ = std.c.unlink(st);
    print("argmax: {d}\n", .{try rlx.argmax(&.{ 0.1, 2, 0.3 })});
}
