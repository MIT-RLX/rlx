// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The Zig binding against the real library: every generated declaration
// analyses, and inference, training, loading, backends and a mesh run.
// `testing.allocator` fails a test that leaks.

const std = @import("std");
const rlx = @import("rlx.zig");
const t = std.testing;

test "every generated declaration compiles" {
    inline for (@typeInfo(rlx).@"struct".decls) |d| {
        // `c` is the C import (its untranslatable system macros are not ours).
        if (comptime std.mem.eql(u8, d.name, "c")) continue;
        // Behind rlx-capi features off by default; they link when built with them.
        if (comptime std.mem.eql(u8, d.name, "export_fpga") or std.mem.eql(u8, d.name, "convert_to_gguf")) continue;
        const v = @field(rlx, d.name);
        _ = &v;
        if (@TypeOf(v) == type and @typeInfo(v) == .@"struct") t.refAllDecls(v);
    }
}

/// loss = mean((x·w − y)²), x [1, 2], w [2, 2].
fn lossGraph() !rlx.Graph {
    const g = try rlx.Graph.init(.{ .name = "mse" });
    const x = try g.input("x", &.{ 1, 2 }, .{});
    const y = try g.input("y", &.{ 1, 2 }, .{});
    const w = try g.param("w", &.{ 2, 2 }, .{});
    const d = try g.sub(try g.matmul(x, w), y);
    try g.set_outputs(&.{try g.mean(try g.mul(d, d), &.{ 0, 1 }, .{})});
    return g;
}

test "inference: compile, set a param, run" {
    const a = t.allocator;
    const g = try lossGraph();
    defer g.deinit();
    const s = try rlx.Session.init(.{});
    defer s.deinit();
    const m = try s.compile(g, .{});
    defer m.deinit();
    try t.expect(g.is_consumed());
    try m.set_param("w", &.{ 1, 0, 0, 1 });
    const out = try m.run(a, &.{ .{ .name = "x", .data = &.{ 1, 2 } }, .{ .name = "y", .data = &.{ 0, 0 } } });
    defer {
        for (out) |o| a.free(o);
        a.free(out);
    }
    try t.expectApproxEqAbs(@as(f32, 2.5), out[0][0], 1e-6);
    const dev = try m.device(a);
    defer a.free(dev);
    try t.expectEqualStrings("cpu", dev);
}

test "errors carry the status and the message" {
    const g = try rlx.Graph.init(.{});
    defer g.deinit();
    try t.expectError(rlx.Error.Range, g.relu(42));
    try t.expect(std.mem.indexOf(u8, rlx.lastError(), "does not exist") != null);
    try t.expectError(rlx.Error.Type, rlx.Session.init(.{ .device = "no-such-device" }));
}

test "training: grad, the Trainer, an Optimizer" {
    const a = t.allocator;
    const fwd = try lossGraph();
    defer fwd.deinit();
    const w = (try fwd.param_names(a));
    a.free(w);
    const tr = try rlx.Trainer.init(a, try fwd.clone(), &.{"w"}, &.{.{ .name = "w", .data = &.{ 0, 0, 0, 0 } }}, .{ .options = "{\"optimizer\": {\"kind\": \"adamw\", \"lr\": 0.1}}" });
    defer tr.deinit();
    const batch = [_]rlx.Input{ .{ .name = "x", .data = &.{ 1, 1 } }, .{ .name = "y", .data = &.{ 1, -1 } } };
    const first = try tr.step(a, &batch);
    var last: f32 = first;
    for (0..200) |_| last = try tr.step(a, &batch);
    try t.expect(first > 0.5 and last < 1e-3);
    try t.expectEqual(@as(u64, 201), try tr.steps());

    const opt = try rlx.Optimizer.init(.{ .kind = "sgd", .options = "{\"lr\": 0.5}" });
    defer opt.deinit();
    var p = [_]f32{ 1, -1 };
    try opt.step(a, &.{.{ .name = "p", .param = &p, .grad = &.{ 2, -2 } }});
    try t.expectEqualSlices(f32, &.{ 0, 0 }, &p);
}

test "loading: write and read a GGUF file, quantize" {
    const a = t.allocator;
    const dir = if (std.c.getenv("TMPDIR")) |d| std.mem.span(d) else ".";
    const path = try std.fmt.allocPrintSentinel(a, "{s}/rlx-zig-test.gguf", .{std.mem.trimEnd(u8, dir, "/")}, 0);
    defer a.free(path);
    const w = [_]f32{ 0.5, -1, 2, 0.25 };
    const writer = try rlx.GgufWriter.init();
    defer writer.deinit();
    try writer.add_f32("w", &.{4}, "F32", &w);
    try t.expectEqual(@as(usize, 1), try writer.write(path, .{ .architecture = "demo" }));
    const f = try rlx.Gguf.open(path);
    defer f.deinit();
    const arch = (try f.architecture(a)).?;
    defer a.free(arch);
    try t.expectEqualStrings("demo", arch);
    const back = try f.tensor(a, "w");
    defer a.free(back);
    try t.expectEqualSlices(f32, &w, back);
    const packed_ = try rlx.quantize(a, &w ** 8, "Q8_0");
    defer a.free(packed_);
    try t.expect(packed_.len < 32 * 4);
    _ = std.c.unlink(path);
}

test "backends and a one-rank mesh" {
    const a = t.allocator;
    const devs = try rlx.devices(a);
    defer a.free(devs);
    try t.expect(std.mem.indexOf(u8, devs, "cpu") != null);
    try t.expect(rlx.is_available("cpu"));
    const g = try lossGraph();
    defer g.deinit();
    const r = try rlx.Runner.init(g, .{ .policy = "{\"only\": [\"cpu\"]}" });
    defer r.deinit();
    try r.set_param("w", &.{ 1, 0, 0, 1 });
    const res = try r.run(a, &.{ .{ .name = "x", .data = &.{ 1, 2 } }, .{ .name = "y", .data = &.{ 0, 0 } } }, .{});
    defer {
        a.free(res.device);
        for (res.outputs) |o| a.free(o);
        a.free(res.outputs);
    }
    try t.expectEqualStrings("cpu", res.device);

    const mesh = try rlx.Group.connect(0, 1, .{});
    defer mesh.deinit();
    const sum = try mesh.all_reduce(a, &.{ 2, 3 }, .{ .op = "max" });
    defer a.free(sum);
    try t.expectEqualSlices(f32, &.{ 2, 3 }, sum);
}
