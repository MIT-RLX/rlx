#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Standalone releases of the language bindings in crates/bindings/.

    python3 scripts/bindings_release.py versions          # stamp the workspace version
    python3 scripts/bindings_release.py versions --check  # fail on drift (CI)
    python3 scripts/bindings_release.py stage --natives natives/ --out dist/
    python3 scripts/bindings_release.py stage --natives natives/ --out dist/ --build
    python3 scripts/bindings_release.py pyrlx-name pyrlx-cu12      # rename the wheel's distribution

One version: `[workspace.package].version` in Cargo.toml. `versions` writes it
into every binding manifest that carries its own (package.json, gemspec,
pubspec, csproj, …); Swift, Go, Lua, MATLAB and Odin version by git tag.

`stage` turns prebuilt C ABI libraries into self-contained packages. `--natives`
holds one directory per platform (`natives/<platform>/`, as the release
workflow uploads them) with `librlx_capi.{so,dylib}` / `rlx_capi.dll` and the
static library. A platform may be a backend variant of another
(`linux-x64-cuda`): same target, more backends compiled in.

The registries that can select a package per platform get one package per
platform, so an install downloads one library rather than all of them:

  npm     `@rlx/rlx` plus `@rlx/rlx-<os>-<cpu>` as optionalDependencies, which
          npm filters by `os` / `cpu` / `libc`; a variant (`…-cuda`) is opt-in
          and preferred by the loader when installed.
  gem     a platform gem per platform (`rlx-<v>-arm64-darwin.gem`, …) and a
          plain `rlx-<v>.gem` without a library for every other platform.
  NuGet   `Rlx` (managed only) and `Rlx.Native.<rid>` (`Rlx.Native.<rid>.Cuda`)
          carrying `runtimes/<rid>/native/`, the layout .NET probes by itself.
  Maven   the `rlx` jar without libraries and one classifier jar per platform
          (`natives-<platform>`) with `native/<os>-<arch>/`, extracted on first
          use.

SwiftPM resolves a package from a git repository with `Package.swift` at its
root, so `rlx-swift` is staged as that repository: a `Package.swift` whose
`CRLX` is `binaryTarget(url:checksum:)` over this release's
`RLX.xcframework.zip` (its checksum read from `RLX.xcframework.zip.sha256` in
`--out`), for a mirror repository tagged with the version.

Every other binding is copied (its git-tracked files only) with every base
platform's shared library under `native/<platform>/` — where its loader looks
before `RLX_CAPI_LIB` and the workspace build — and archived as
`dist/<binding>-<version>.tar.gz`. One `rlx-capi-<version>-<platform>.tar.gz`
per platform (variants included) carries the header and both libraries, for
the bindings that link at build time (C++, Zig, Go, Swift, …) and for C itself,
with `lib/pkgconfig/rlx.pc` and `lib/cmake/rlx/rlxConfig.cmake` (`rlx::rlx`,
`rlx::rlx_static`). A static link also needs the system libraries rustc names
(`--print native-static-libs`); the release workflow saves them as
`natives/<platform>/native-static-libs.txt`.

`--build` also runs each ecosystem's own packer where it is installed (`npm
pack`, `gem build`, `dotnet pack`, `R CMD build`, `cabal sdist`, `mix
hex.build`); a missing tool is reported and skipped. Publishing stays a manual
step: see docs/releasing-bindings.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINDINGS = ROOT / "crates" / "bindings"

# platform → (Rust target triple, .NET runtime id, shared lib, static lib)
PLATFORMS = {
    "macos-arm64": ("aarch64-apple-darwin", "osx-arm64", "librlx_capi.dylib", "librlx_capi.a"),
    "macos-x64": ("x86_64-apple-darwin", "osx-x64", "librlx_capi.dylib", "librlx_capi.a"),
    "linux-x64": ("x86_64-unknown-linux-gnu", "linux-x64", "librlx_capi.so", "librlx_capi.a"),
    "linux-arm64": ("aarch64-unknown-linux-gnu", "linux-arm64", "librlx_capi.so", "librlx_capi.a"),
    "windows-x64": ("x86_64-pc-windows-msvc", "win-x64", "rlx_capi.dll", "rlx_capi.lib"),
}

# Backend variants: `<platform>-<variant>` is the same target built with more
# backends. The bundled-library loaders key on the base platform, so a variant
# never goes into a package that carries every platform.
VARIANTS = ("cuda",)

# base platform → (npm os, npm cpu, npm libc, RubyGems platform)
REGISTRY_PLATFORMS = {
    "macos-arm64": ("darwin", "arm64", None, "arm64-darwin"),
    "macos-x64": ("darwin", "x64", None, "x86_64-darwin"),
    "linux-x64": ("linux", "x64", "glibc", "x86_64-linux-gnu"),
    "linux-arm64": ("linux", "arm64", "glibc", "aarch64-linux-gnu"),
    "windows-x64": ("win32", "x64", None, "x64-mingw-ucrt"),
}

# Each binding manifest that states its own version: (file, pattern). The
# pattern's single group is the version; only its first match is rewritten.
VERSION_FIELDS = [
    ("rlx-node/package.json", r'^\s*"version":\s*"([^"]+)"'),
    ("rlx-ruby/rlx.gemspec", r'^\s*s\.version\s*=\s*"([^"]+)"'),
    ("rlx-dart/pubspec.yaml", r"^version:\s*(\S+)"),
    ("rlx-elixir/mix.exs", r'^\s*version:\s*"([^"]+)"'),
    ("rlx-julia/Project.toml", r'^version\s*=\s*"([^"]+)"'),
    ("rlx-kotlin/build.gradle.kts", r'^version\s*=\s*"([^"]+)"'),
    ("rlx-nim/rlx.nimble", r'^version\s*=\s*"([^"]+)"'),
    ("rlx-php/composer.json", r'^\s*"version":\s*"([^"]+)"'),
    ("rlx-d/dub.json", r'^\s*"version":\s*"([^"]+)"'),
    ("rlx-haskell/rlx.cabal", r"^version:\s*(\S+)"),
    ("rlx-zig/build.zig.zon", r'^\s*\.version\s*=\s*"([^"]+)"'),
    ("rlx-r/DESCRIPTION", r"^Version:\s*(\S+)"),
    ("rlx-v/v.mod", r"^\s*version:\s*'([^']+)'"),
    ("rlx-ocaml/dune-project", r"^\(version\s+([^)\s]+)\)"),
    ("rlx-cpp/CMakeLists.txt", r"^project\(rlx_cpp VERSION ([0-9.]+)"),
    ("rlx-java/pom.xml", r"^  <version>([^<]+)</version>"),
    ("rlx-csharp/src/Rlx/Rlx.csproj", r"<Version>([^<]+)</Version>"),
    ("pyrlx/pyproject.toml", r'^version\s*=\s*"([^"]+)"'),
]

# Bindings packaged here, and the native layout each gets. Rust crates
# (rlx-bind, rlx-capi, rlx-js, rlx-micro, rlx-model-plugin) are not language
# packages; pyrlx ships as maturin wheels; rlx-web deploys to GitHub Pages;
# rlx-ffi is the iOS node XCFramework (ios/build-xcframework.sh).
PACKAGES = {
    "rlx-node": "npm", "rlx-ruby": "gem", "rlx-csharp": "nuget",
    "rlx-java": "jar", "rlx-dart": "native", "rlx-php": "native",
    "rlx-julia": "native", "rlx-kotlin": "native", "rlx-lua": "native",
    "rlx-matlab": "native", "rlx-go": "native", "rlx-cpp": "native",
    "rlx-zig": "native", "rlx-d": "native", "rlx-nim": "native",
    "rlx-haskell": "native", "rlx-ocaml": "native", "rlx-elixir": "native",
    "rlx-r": "native", "rlx-v": "native", "rlx-odin": "native",
    "rlx-swift": "swiftpm",
}

# `--build`: (tool to probe, command run in the staged directory). The
# per-platform npm, gem and NuGet packages are packed in `stage_split`.
BUILDERS = {
    "rlx-node": ("npm", ["npm", "pack", "--pack-destination", "{dist}"]),
    "rlx-ruby": ("gem", ["gem", "build", "rlx.gemspec", "--output", "{dist}/rlx-{version}.gem"]),
    "rlx-csharp": ("dotnet", ["dotnet", "pack", "src/Rlx/Rlx.csproj", "-c", "Release", "-o", "{dist}"]),
    "rlx-r": ("R", ["R", "CMD", "build", "."]),
    "rlx-haskell": ("cabal", ["cabal", "sdist", "-o", "{dist}"]),
    "rlx-elixir": ("mix", ["mix", "hex.build", "-o", "{dist}/rlx-{version}.tar"]),
}


def workspace_version() -> str:
    data = tomllib.loads((ROOT / "Cargo.toml").read_text())
    return data["workspace"]["package"]["version"]


def cmd_versions(check: bool) -> int:
    version = workspace_version()
    drift = []
    for rel, pattern in VERSION_FIELDS:
        path = BINDINGS / rel
        text = path.read_text()
        m = re.search(pattern, text, re.M)
        if not m:
            raise SystemExit(f"{rel}: no version matches {pattern!r}")
        if m.group(1) == version:
            continue
        drift.append((rel, m.group(1)))
        if not check:
            path.write_text(text[: m.start(1)] + version + text[m.end(1) :])
    for rel, old in drift:
        print(f"  {rel}: {old} → {version}" + ("" if check else " (written)"))
    if check and drift:
        print(f"binding versions drift from the workspace ({version}); run: just sync-binding-versions")
        return 1
    print(f"binding versions: {version} ({len(VERSION_FIELDS)} manifests)")
    return 0


def tracked_files(directory: Path) -> list[Path]:
    """The binding's git-tracked files — never build output, caches or node_modules."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", str(directory.relative_to(ROOT))],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    return [ROOT / p for p in out.decode().split("\0") if p]


def split_variant(plat: str) -> tuple[str, str | None]:
    """`linux-x64-cuda` → (`linux-x64`, `cuda`); `linux-x64` → (`linux-x64`, None)."""
    for variant in VARIANTS:
        if plat.endswith("-" + variant):
            return plat[: -len(variant) - 1], variant
    return plat, None


def natives_found(natives: Path) -> dict[str, Path]:
    found = {}
    for d in sorted(natives.iterdir()) if natives.is_dir() else []:
        base, _ = split_variant(d.name)
        if base in PLATFORMS and (d / PLATFORMS[base][2]).is_file():
            found[d.name] = d
    if not found:
        raise SystemExit(
            f"no prebuilt libraries under {natives}: expected {natives}/<platform>[-<variant>]/"
            f"librlx_capi.* for one of {', '.join(PLATFORMS)}"
        )
    return found


def run_packer(tool: str, command: list[str], cwd: Path, env: dict | None = None) -> bool | None:
    """Run an ecosystem packer: None when the tool is missing, else whether it succeeded."""
    if shutil.which(tool) is None:
        return None
    return subprocess.run(command, cwd=cwd, env={**os.environ, **(env or {})}).returncode == 0


def stage_npm(dest: Path, found: dict[str, Path], version: str, out: Path, build: bool) -> list[str]:
    """`@rlx/rlx` and one `@rlx/rlx-<os>-<cpu>[-<variant>]` package per platform."""
    manifest = json.loads((dest / "package.json").read_text())
    name, packages, optional = manifest["name"], [], {}
    for plat, d in found.items():
        base, variant = split_variant(plat)
        npm_os, npm_cpu, libc, _ = REGISTRY_PLATFORMS[base]
        pkg = f"{name}-{npm_os}-{npm_cpu}" + (f"-{variant}" if variant else "")
        pdir = dest.parent / pkg.replace("/", "__")
        pdir.mkdir()
        shared = PLATFORMS[base][2]
        shutil.copy2(d / shared, pdir)
        meta = {
            "name": pkg,
            "version": version,
            "description": f"librlx_capi for {plat}" + (f" ({variant} backend)" if variant else "")
            + f": the native library of {name}.",
            "license": manifest["license"],
            "os": [npm_os],
            "cpu": [npm_cpu],
            **({"libc": [libc]} if libc else {}),
            "files": [shared],
        }
        (pdir / "package.json").write_text(json.dumps(meta, indent=2) + "\n")
        packages.append(pdir)
        if not variant:  # a variant is opt-in: `npm install <name>-linux-x64-cuda`
            optional[pkg] = version
    manifest["optionalDependencies"] = optional
    manifest["files"] = [f for f in manifest["files"] if f != "native"]
    (dest / "package.json").write_text(json.dumps(manifest, indent=2) + "\n")
    failed = []
    for pdir in packages if build else []:
        ok = run_packer("npm", ["npm", "pack", "--pack-destination", str(out)], pdir)
        if ok is False:
            failed.append(pdir.name)
    return failed


def stage_gem(dest: Path, found: dict[str, Path], version: str, out: Path, build: bool) -> list[str]:
    """One platform gem per base platform; the plain gem (built by BUILDERS) has no library."""
    failed = []
    for plat, d in found.items():
        base, variant = split_variant(plat)
        if variant:  # a platform gem cannot name a backend; use the C SDK + RLX_CAPI_LIB
            continue
        gem_plat = REGISTRY_PLATFORMS[base][3]
        pdir = dest.parent / f"rlx-ruby-{gem_plat}"
        shutil.copytree(dest, pdir)
        (pdir / "native" / base).mkdir(parents=True)
        shutil.copy2(d / PLATFORMS[base][2], pdir / "native" / base)
        if build:
            command = ["gem", "build", "rlx.gemspec", "--output", str(out / f"rlx-{version}-{gem_plat}.gem")]
            if run_packer("gem", command, pdir, {"RLX_GEM_PLATFORM": gem_plat}) is False:
                failed.append(pdir.name)
    return failed


NATIVE_CSPROJ = """<Project Sdk="Microsoft.NET.Sdk">
  <!-- Generated by scripts/bindings_release.py: librlx_capi for one runtime id. -->
  <PropertyGroup>
    <TargetFramework>netstandard2.0</TargetFramework>
    <IncludeBuildOutput>false</IncludeBuildOutput>
    <NoWarn>NU5128</NoWarn>
    <PackageId>{package}</PackageId>
    <Version>{version}</Version>
    <Authors>Eugene Hauptmann; Nataliya Kosmyna</Authors>
    <Description>librlx_capi for {rid}{variant}, the native library of the Rlx package.</Description>
    <PackageLicenseExpression>MIT OR Apache-2.0</PackageLicenseExpression>
  </PropertyGroup>
  <ItemGroup>
    <None Include="runtimes/**/*" Pack="true" PackagePath="runtimes" />
  </ItemGroup>
</Project>
"""


def stage_nuget(dest: Path, found: dict[str, Path], version: str, out: Path, build: bool) -> list[str]:
    """`Rlx.Native.<rid>[.<Variant>]`, each with `runtimes/<rid>/native/`."""
    failed = []
    for plat, d in found.items():
        base, variant = split_variant(plat)
        rid = PLATFORMS[base][1]
        package = f"Rlx.Native.{rid}" + (f".{variant.capitalize()}" if variant else "")
        pdir = dest.parent / package
        lib_dir = pdir / "runtimes" / rid / "native"
        lib_dir.mkdir(parents=True)
        shutil.copy2(d / PLATFORMS[base][2], lib_dir)
        (pdir / f"{package}.csproj").write_text(NATIVE_CSPROJ.format(
            package=package, version=version, rid=rid, variant=f" ({variant} backend)" if variant else ""))
        if build:
            command = ["dotnet", "pack", f"{package}.csproj", "-c", "Release", "-o", str(out)]
            if run_packer("dotnet", command, pdir) is False:
                failed.append(package)
    return failed


SWIFT_PACKAGE = """// swift-tools-version:5.9
// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Generated by scripts/bindings_release.py for rlx {version}: the Swift
// bindings over the prebuilt RLX.xcframework (iOS + macOS, CPU + Metal) from
// the rlx release. The development manifest is crates/bindings/rlx-swift in
// the rlx repository.
//
//     .package(url: "<this repository>", from: "{version}")

import PackageDescription

let examples = ["Inference", "Training", "Loading", "Distributed", "Backends", "Models"]

let package = Package(
    name: "RLX",
    platforms: [.macOS(.v12), .iOS(.v17)],
    products: [
        .library(name: "RLX", targets: ["RLX"]),
    ] + examples.map {{ .executable(name: "rlx-example-\\($0.lowercased())", targets: ["Example\\($0)"]) }},
    targets: [
        .binaryTarget(
            name: "CRLX",
            url: "{url}",
            checksum: "{checksum}"
        ),
        // What rustc reports as the static archive's native libraries
        // (`--print native-static-libs`) for a `--apple` build.
        .target(name: "RLX", dependencies: ["CRLX"], linkerSettings: [
            .linkedFramework("Accelerate"),
            .linkedFramework("Metal"),
            .linkedFramework("MetalPerformanceShaders"),
            .linkedFramework("MetalPerformanceShadersGraph"),
            .linkedLibrary("bz2"),
            .linkedLibrary("iconv"),
            .linkedLibrary("objc"),
        ]),
        .testTarget(name: "RLXTests", dependencies: ["RLX"]),
    ] + examples.map {{
        .executableTarget(name: "Example\\($0)", dependencies: ["RLX"], path: "Examples/\\($0)")
    }}
)
"""


def stage_swiftpm(dest: Path, version: str, out: Path) -> None:
    """Rewrite the staged rlx-swift as the SwiftPM mirror repository's contents."""
    sha = out / "RLX.xcframework.zip.sha256"
    if not sha.is_file():
        print(f"  rlx-swift: no {sha.name} in {out}; staged with the development Package.swift")
        return
    repo = os.environ.get("GITHUB_REPOSITORY", "MIT-RLX/rlx")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    url = f"{server}/{repo}/releases/download/v{version}/RLX.xcframework.zip"
    checksum = sha.read_text().split()[0]
    (dest / "Package.swift").write_text(SWIFT_PACKAGE.format(version=version, url=url, checksum=checksum))
    # The modulemap that points at a local librlx_capi is the development path's.
    shutil.rmtree(dest / "Sources" / "CRLX", ignore_errors=True)


def stage_jars(found: dict[str, Path], version: str, out: Path) -> None:
    """`rlx-<v>-natives-<platform>.jar`: `native/<os>-<arch>/` for Support.java to extract."""
    for plat, d in found.items():
        base, _ = split_variant(plat)
        shared = PLATFORMS[base][2]
        with zipfile.ZipFile(out / f"rlx-{version}-natives-{plat}.jar", "w", zipfile.ZIP_DEFLATED) as jar:
            jar.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\nCreated-By: bindings_release.py\n")
            jar.write(d / shared, f"native/{base}/{shared}")


CMAKE_CONFIG = """# Generated by scripts/bindings_release.py: the rlx C ABI {version} ({platform}).
#   find_package(rlx CONFIG REQUIRED)   # CMAKE_PREFIX_PATH=<this SDK>
#   target_link_libraries(app PRIVATE rlx::rlx)          # or rlx::rlx_static
get_filename_component(_rlx_prefix "${{CMAKE_CURRENT_LIST_DIR}}/../../.." ABSOLUTE)
set(rlx_VERSION {version})
if(NOT TARGET rlx::rlx)
  add_library(rlx::rlx SHARED IMPORTED)
  set_target_properties(rlx::rlx PROPERTIES
    IMPORTED_LOCATION "${{_rlx_prefix}}/lib/{shared}"{implib}
    INTERFACE_INCLUDE_DIRECTORIES "${{_rlx_prefix}}/include")
  add_library(rlx::rlx_static STATIC IMPORTED)
  set_target_properties(rlx::rlx_static PROPERTIES
    IMPORTED_LOCATION "${{_rlx_prefix}}/lib/{static}"
    INTERFACE_INCLUDE_DIRECTORIES "${{_rlx_prefix}}/include"
    INTERFACE_LINK_LIBRARIES "{static_libs}")
endif()
unset(_rlx_prefix)
"""


def native_static_libs(d: Path) -> list[str]:
    """rustc's `native-static-libs` for this build, `-framework X` kept as one
    item, each once (rustc repeats system libraries)."""
    f = d / "native-static-libs.txt"
    if not f.is_file():
        return []
    words = f.read_text().split("native-static-libs:")[-1].split()
    libs, i = [], 0
    while i < len(words):
        if words[i] == "-framework" and i + 1 < len(words):
            libs.append(f"-framework {words[i + 1]}")
            i += 2
        else:
            libs.append(words[i])
            i += 1
    return list(dict.fromkeys(libs))


def write_sdk_configs(sdk: Path, plat: str, d: Path, version: str) -> None:
    """pkg-config and CMake files, relative to the SDK wherever it is unpacked."""
    base, _ = split_variant(plat)
    shared, static = PLATFORMS[base][2:]
    libs = native_static_libs(d)
    pc = sdk / "lib" / "pkgconfig"
    pc.mkdir(parents=True)
    rpath = "" if shared.endswith(".dll") else " -Wl,-rpath,${libdir}"
    (pc / "rlx.pc").write_text(
        "prefix=${pcfiledir}/../..\nlibdir=${prefix}/lib\nincludedir=${prefix}/include\n\n"
        f"Name: rlx\nDescription: rlx C ABI ({plat})\nVersion: {version}\n"
        f"Libs: -L${{libdir}}{rpath} -lrlx_capi\n"
        + (f"Libs.private: {' '.join(libs)}\n" if libs else "")
        + "Cflags: -I${includedir}\n"
    )
    implib = '\n    IMPORTED_IMPLIB "${_rlx_prefix}/lib/rlx_capi.dll.lib"' if shared.endswith(".dll") else ""
    cmake = sdk / "lib" / "cmake" / "rlx"
    cmake.mkdir(parents=True)
    (cmake / "rlxConfig.cmake").write_text(CMAKE_CONFIG.format(
        version=version, platform=plat, shared=shared, static=static, implib=implib,
        static_libs=";".join(libs)))
    (cmake / "rlxConfigVersion.cmake").write_text(
        f'set(PACKAGE_VERSION "{version}")\n'
        "if(PACKAGE_FIND_VERSION VERSION_GREATER PACKAGE_VERSION)\n"
        "  set(PACKAGE_VERSION_COMPATIBLE FALSE)\nelse()\n  set(PACKAGE_VERSION_COMPATIBLE TRUE)\n"
        "  if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)\n    set(PACKAGE_VERSION_EXACT TRUE)\n  endif()\nendif()\n"
    )


def archive(src: Path, dest: Path) -> None:
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(src, arcname=src.name)


def cmd_stage(natives: Path, out: Path, build: bool) -> int:
    version = workspace_version()
    found = natives_found(natives.resolve())
    out = out.resolve()
    staging = out / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # The C ABI SDK, one per platform.
    for plat, d in found.items():
        name = f"rlx-capi-{version}-{plat}"
        sdk = staging / name
        (sdk / "include").mkdir(parents=True)
        (sdk / "lib").mkdir()
        shutil.copy2(BINDINGS / "rlx-capi" / "include" / "rlx.h", sdk / "include")
        for lib in PLATFORMS[split_variant(plat)[0]][2:] + ("rlx_capi.dll.lib",):
            if (d / lib).is_file():
                shutil.copy2(d / lib, sdk / "lib")
        write_sdk_configs(sdk, plat, d, version)
        shutil.copy2(BINDINGS / "rlx-capi" / "README.md", sdk)
        archive(sdk, out / f"{name}.tar.gz")

    failed = []
    for binding, layout in PACKAGES.items():
        src = BINDINGS / binding
        dest = staging / binding
        for f in tracked_files(src):
            target = dest / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
        if layout == "npm":
            failed += stage_npm(dest, found, version, out, build)
        elif layout == "gem":
            failed += stage_gem(dest, found, version, out, build)
        elif layout == "nuget":
            failed += stage_nuget(dest, found, version, out, build)
        elif layout == "jar":
            stage_jars(found, version, out)
        elif layout == "swiftpm":
            stage_swiftpm(dest, version, out)
        else:
            for plat, d in found.items():
                if split_variant(plat)[1]:
                    continue
                lib_dir = dest / "native" / plat
                lib_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(d / PLATFORMS[plat][2], lib_dir)
        archive(dest, out / f"{binding}-{version}.tar.gz")

        if build and binding in BUILDERS:
            tool, command = BUILDERS[binding]
            ok = run_packer(tool, [c.format(dist=out, version=version) for c in command], dest)
            if ok is None:
                print(f"  skip {binding}: no {tool}")
                continue
            if not ok:
                failed.append(binding)
            elif binding == "rlx-r":
                for t in dest.glob("*.tar.gz"):
                    shutil.move(t, out / t.name)

    print(f"\n✓ {version}: {len(found)} platform(s) ({', '.join(found)}) → {out}")
    for p in sorted(out.iterdir()):
        if p.is_file():
            print(f"    {p.name:46} {p.stat().st_size / 1e6:8.2f} MB")
    if failed:
        print(f"✗ packer failed for: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def cmd_pyrlx_name(name: str) -> int:
    """Set pyrlx's distribution name (`pyrlx-cu12`): the module stays `pyrlx`."""
    path = BINDINGS / "pyrlx" / "pyproject.toml"
    text = path.read_text()
    m = re.search(r'^name\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        raise SystemExit(f"{path}: no [project] name")
    path.write_text(text[: m.start(1)] + name + text[m.end(1) :])
    print(f"pyrlx distribution: {m.group(1)} → {name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("versions", help="stamp the workspace version into every binding manifest")
    v.add_argument("--check", action="store_true", help="report drift and fail; write nothing")
    s = sub.add_parser("stage", help="package every binding with prebuilt libraries")
    s.add_argument("--natives", type=Path, required=True, help="natives/<platform>/ directories")
    s.add_argument("--out", type=Path, default=ROOT / "dist", help="output directory (default ./dist)")
    s.add_argument("--build", action="store_true", help="also run each ecosystem's packer")
    n = sub.add_parser("pyrlx-name", help="rename pyrlx's distribution for a backend variant wheel")
    n.add_argument("name", help="e.g. pyrlx-cu12")
    args = ap.parse_args()
    if args.cmd == "versions":
        return cmd_versions(args.check)
    if args.cmd == "pyrlx-name":
        return cmd_pyrlx_name(args.name)
    return cmd_stage(args.natives, args.out, args.build)


if __name__ == "__main__":
    raise SystemExit(main())
