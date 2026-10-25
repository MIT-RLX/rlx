// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The `rlx` Zig module over rlx-capi. `-Dlib-dir` is where librlx_capi lives
// (default: the package's bundled native/<os>-<arch>/ when present, else the
// workspace's target/debug); `zig build test` runs src/tests.zig.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const lib_dir = b.option([]const u8, "lib-dir", "directory holding librlx_capi") orelse
        bundledLibDir(b, target) orelse b.pathFromRoot("../../../target/debug");

    const rlx = b.addModule("rlx", .{
        .root_source_file = b.path("src/rlx.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    rlx.addIncludePath(b.path("../rlx-capi/include"));
    rlx.addLibraryPath(.{ .cwd_relative = lib_dir });
    rlx.addRPath(.{ .cwd_relative = lib_dir });
    rlx.linkSystemLibrary("rlx_capi", .{});

    const tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }) });
    tests.root_module.addIncludePath(b.path("../rlx-capi/include"));
    tests.root_module.addLibraryPath(.{ .cwd_relative = lib_dir });
    tests.root_module.addRPath(.{ .cwd_relative = lib_dir });
    tests.root_module.linkSystemLibrary("rlx_capi", .{});
    // `zig build <example>` runs examples/<example>.zig against the module.
    inline for (.{ "inference", "training", "loading", "distributed", "backends", "models" }) |name| {
        const exe = b.addExecutable(.{ .name = name, .root_module = b.createModule(.{
            .root_source_file = b.path("examples/" ++ name ++ ".zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "rlx", .module = rlx }},
        }) });
        b.step(name, "run examples/" ++ name ++ ".zig").dependOn(&b.addRunArtifact(exe).step);
    }

    const run = b.addRunArtifact(tests);
    b.step("test", "run the rlx Zig tests against librlx_capi").dependOn(&run.step);
}

/// `native/<os>-<arch>/` in a packaged release, if it exists for `target`.
fn bundledLibDir(b: *std.Build, target: std.Build.ResolvedTarget) ?[]const u8 {
    const os = switch (target.result.os.tag) {
        .macos => "macos",
        .linux => "linux",
        .windows => "windows",
        else => return null,
    };
    const arch = switch (target.result.cpu.arch) {
        .aarch64 => "arm64",
        .x86_64 => "x64",
        else => return null,
    };
    const dir = b.pathFromRoot(b.fmt("native/{s}-{s}", .{ os, arch }));
    std.Io.Dir.accessAbsolute(b.graph.io, dir, .{}) catch return null;
    return dir;
}
