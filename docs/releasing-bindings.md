# Releasing the language bindings

Every binding in [`crates/bindings/`](../crates/bindings/README.md) ships as a
standalone package for its own ecosystem, with a prebuilt `librlx_capi`, so
users need no Rust toolchain. Where the registry can pick a package per
platform (npm, RubyGems, NuGet, Maven), each platform's library is its own
package, so an install downloads one ~10 MB library instead of all of them. Building is automated; **publishing to each
registry is a manual, reviewed step**.

## 1. One version

`[workspace.package].version` in the root `Cargo.toml` is the only version.
After bumping it:

    just sync-binding-versions     # stamps it into every binding manifest
    just check-binding-versions    # what CI runs; fails on drift

`scripts/bindings_release.py` lists the 18 manifests that carry a version
(package.json, gemspec, pubspec, csproj, pom.xml, …). Swift, Go, Lua, MATLAB and
Odin version by git tag.

## 2. Build: tag, then wait for the draft release

Pushing a `v<version>` tag runs `.github/workflows/bindings-release.yml`
(also runnable by hand from the Actions tab). It:

1. checks the binding versions;
2. builds `librlx_capi` (shared and static) for `macos-arm64`, `macos-x64`,
   `linux-x64`, `linux-arm64` and `windows-x64`, and checks the Linux builds
   link no system BLAS (rlx-cpu falls back to its portable GEMM and pure-Rust
   linear algebra, so the library runs on a bare machine);
3. builds the pyrlx wheels (maturin, manylinux) and sdist, and the Swift
   `RLX.xcframework` (iOS + macOS) with its SwiftPM checksum;
4. stages the packages (below) and runs the packers it can (`npm pack`,
   `gem build`, `dotnet pack`);
5. uploads everything to a **draft** GitHub Release.

### Backends in the prebuilt libraries

| build | backends |
|---|---|
| macOS (`macos-arm64`, `macos-x64`), the XCFramework | CPU + Metal + Accelerate BLAS: both ship with every macOS |
| Linux, Windows | CPU |
| `linux-x64-cuda`, `windows-x64-cuda` | CPU + CUDA |

The CUDA builds need no toolkit: cudarc loads libcuda, NVRTC and cuBLAS on
first device use (the workflow checks the library links none of them), so a
CUDA package still starts, CPU-only, on a machine without the driver. They need
a CUDA 12.3+ driver to use the GPU. CUDA ships as its own package, the way
PyTorch and ONNX Runtime do:

| ecosystem | CPU | CUDA |
|---|---|---|
| Python | `pip install pyrlx` | `pip install pyrlx-cu12` (same `import pyrlx`; install one) |
| Node | `npm install @rlx/rlx` | also `npm install @rlx/rlx-linux-x64-cuda`; preferred when installed |
| .NET | `Rlx` + `Rlx.Native.<rid>` | `Rlx` + `Rlx.Native.<rid>.Cuda` |
| Java | `rlx` + classifier `natives-<platform>` | classifier `natives-<platform>-cuda` |
| C / other FFI | `rlx-capi-<v>-<platform>.tar.gz` | `rlx-capi-<v>-<platform>-cuda.tar.gz` |

Ruby has no CUDA gem (a platform gem cannot name a backend): point
`RLX_CAPI_LIB` at the CUDA C SDK's library.

### One package per platform

| registry | packages |
|---|---|
| npm | `@rlx/rlx` lists `@rlx/rlx-<os>-<cpu>` (`darwin-arm64`, `darwin-x64`, `linux-x64`, `linux-arm64`, `win32-x64`) as `optionalDependencies`; npm installs the one whose `os` / `cpu` / `libc` match. `@rlx/rlx-<os>-<cpu>-cuda` is opt-in. |
| RubyGems | `rlx-<v>-<platform>.gem` for `arm64-darwin`, `x86_64-darwin`, `x86_64-linux-gnu`, `aarch64-linux-gnu`, `x64-mingw-ucrt`; `gem install rlx` picks the matching one. The plain `rlx-<v>.gem` carries no library (`RLX_CAPI_LIB`) for any other platform. |
| NuGet | `Rlx` (managed) and `Rlx.Native.<rid>` (`osx-arm64`, `osx-x64`, `linux-x64`, `linux-arm64`, `win-x64`), each `runtimes/<rid>/native/`. An app references `Rlx` plus the runtime ids it deploys to; without one, loading fails naming the package to add. |
| Maven | `org.rlx:rlx` (no library) plus `org.rlx:rlx:<v>:natives-<platform>` jars, each `native/<os>-<arch>/`, extracted on first use. |

Every other binding still ships one archive with all five platforms' libraries.

Locally, for the host platform only: `just package-bindings` (→ `./dist`).

### Where a package looks for the library

Runtime loaders (Node, Ruby, C#, Java, Dart, PHP, Julia) search, in order:

1. `RLX_CAPI_LIB` — a full path, always wins;
2. the bundled library: `native/<os>-<arch>/` in the package (`<os>` is
   `macos` / `linux` / `windows`, `<arch>` is `arm64` / `x64`). Node first
   resolves the installed `@rlx/rlx-<os>-<cpu>-cuda`, then `@rlx/rlx-<os>-<cpu>`;
   a platform gem carries `native/<os>-<arch>/`; C# uses NuGet's
   `runtimes/<rid>/native/` from `Rlx.Native.<rid>`; Java reads it from the
   classpath (a `natives-<platform>` jar) and extracts it to a temporary file;
3. the workspace's `target/debug` — a checkout.

Build-time bindings take a library directory (`RLX_CAPI_LIB_DIR`, Zig's
`-Dlib-dir`). C++, Kotlin's JNI glue, Zig and Elixir default to the bundled
`native/<os>-<arch>/` when the package has one. The others link against the
per-platform C SDK: `rlx-capi-<version>-<platform>.tar.gz` holds `include/rlx.h`
and `lib/`.

## 3. Publish

### Automated: PyPI, npm, RubyGems, NuGet

Publishing the draft release runs `.github/workflows/bindings-publish.yml`
(or run it by hand with the tag). Each registry is one job behind a GitHub
Environment of the same name (`pypi`, `npm`, `rubygems`, `nuget`), and
uploads nothing until a reviewer approves that environment. The jobs sign in
with OIDC Trusted Publishing, skip versions the registry already has (a
re-run publishes only what is missing), and publish npm and NuGet platform
packages before the packages that depend on them.

One-time setup:

1. **Environments** — Settings → Environments: create the four, each with
   required reviewers and deployment limited to `v*` tags.
2. **PyPI** — for each of `pyrlx` and `pyrlx-cu12`, add a trusted publisher:
   this repository, workflow `bindings-publish.yml`, environment `pypi`. (A
   project that does not exist yet: add a *pending* publisher.)
3. **npm** — trusted publishing is configured on an existing package, so
   publish each package (`@rlx/rlx` and every `@rlx/rlx-<platform>`) once by
   hand, then in each package's settings add this repository, workflow
   `bindings-publish.yml`, environment `npm`. Packages are published with
   `--provenance`.
4. **RubyGems** — on the `rlx` gem (pending publisher before its first
   release): trusted publisher, this repository, `bindings-publish.yml`,
   environment `rubygems`.
5. **NuGet** — nuget.org → Trusted Publishing: a policy for this repository
   and `bindings-publish.yml`, owned by the account that owns `Rlx` and
   `Rlx.Native.*`; set the repository variable `NUGET_USER` to that account's
   user name.

### By hand, the rest

Download the draft release's assets, check them, then:

| binding | artifact | publish |
|---|---|---|
| Python, Node, Ruby, C# | wheels + sdist, `.tgz`, `.gem`, `.nupkg` | automated, above (by hand: `twine upload`, `npm publish --access public` platform packages first, `gem push`, `dotnet nuget push` native packages first) |
| Swift (`rlx-swift`) | `rlx-swift-<v>.tar.gz`, `RLX.xcframework.zip` | publish the release first (the package downloads the zip from it), then push the extracted `rlx-swift-<v>/` to the SwiftPM repository (`rlx-swift`, a mirror with `Package.swift` at its root) and tag it `<v>`. Its `Package.swift` is generated: `CRLX` is `binaryTarget(url:checksum:)` over this release's zip. |
| Java (`rlx-java`) | `rlx-java-<v>.tar.gz`, `rlx-<v>-natives-<platform>.jar` | extract, `mvn deploy`; then each jar with `mvn deploy:deploy-file -DgroupId=org.rlx -DartifactId=rlx -Dversion=<v> -Dclassifier=natives-<platform> -Dfile=<jar>` |
| Kotlin (`rlx-kotlin`) | `rlx-kotlin-<v>.tar.gz` | extract, then `gradle publish` |
| Dart (`rlx-dart`) | `rlx-dart-<v>.tar.gz` | extract, then `dart pub publish` |
| Zig (`rlx-zig`) | `rlx-zig-<v>.tar.gz` | users `zig fetch --save <release asset URL>`; nothing to upload |
| C / C++ (`rlx-capi`, `rlx-cpp`) | `rlx-capi-<v>-<platform>.tar.gz`, `rlx-cpp-<v>.tar.gz` | release assets are the distribution. Each SDK has `lib/pkgconfig/rlx.pc` (`pkg-config --cflags --libs rlx`; `--static` adds the system libraries) and `lib/cmake/rlx/` (`find_package(rlx CONFIG)` → `rlx::rlx` / `rlx::rlx_static`), both relative to wherever it is unpacked. The macOS dylib's install name is `@rpath/librlx_capi.dylib`. |
| Go (`rlx-go`) | git tag | tag `crates/bindings/rlx-go/v<v>`; users point cgo at the C SDK (`CGO_LDFLAGS=-L<sdk>/lib`) |
| PHP (`rlx-php`) | git tag | Packagist follows the repository; users set `RLX_CAPI_LIB` or unpack the C SDK |
| Julia (`rlx-julia`) | git tag | register via Registrator; users set `RLX_CAPI_LIB` or unpack the C SDK |
| Elixir, Haskell, R, Nim, D, OCaml, Lua, MATLAB, V, Odin | `rlx-<lang>-<v>.tar.gz` | each registry's own upload (`mix hex.publish`, `cabal upload`, r-universe, `nimble publish`, `dub` registry, opam, luarocks), building against the C SDK |

The git-tag registries (Go, Packagist, Julia's General, opam) distribute from
the repository, where the prebuilt libraries are not committed. Those packages
find the library through `RLX_CAPI_LIB` / `RLX_CAPI_LIB_DIR` and the C SDK
asset.

Then publish the draft release.

## Not covered here

- **crates.io:** `./scripts/publish.sh` publishes the Rust crates. Every crate
  under `crates/bindings/` is `publish = false` and listed in its `SKIPPED`.
- **rlx-web:** deployed to GitHub Pages by `.github/workflows/pages.yml`.
- **rlx-js** (the embedded-JS CLI), **rlx-micro** (microcontroller C ABI) and
  **rlx-ffi** (the iOS node, `ios/build-xcframework.sh`) have their own build
  instructions in their READMEs.
## License

MIT OR Apache-2.0.
