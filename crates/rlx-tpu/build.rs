// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Compile the vendored xla protos into Rust types via prost-build.
//!
//! Runs at `cargo build` time. Output lands under
//! `$OUT_DIR/xla.rs` (and `xla.service.rs`) — included by `src/lib.rs`
//! through `tonic-prost`-style `include!` macros.
//!
//! protoc is supplied by the `protoc-bin-vendored` crate; no system
//! install required. The vendored binary supports macOS aarch64,
//! Linux x86_64, and Linux aarch64 — same set the Docker image runs
//! on.

fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path()
        .expect("protoc-bin-vendored: no binary for this target");
    // SAFETY: `protoc-bin-vendored` returns an absolute path; we
    // export it so prost-build picks it up via PROTOC env var.
    unsafe {
        std::env::set_var("PROTOC", protoc);
    }

    let mut config = prost_build::Config::new();
    // The xla.* messages are referenced by `xla.service.HloModuleProto`
    // — both must be in scope. prost-build infers package paths from
    // the `package xla;` / `package xla.service;` declarations and
    // emits one Rust file per package.
    config
        .compile_protos(
            &[
                "xla/xla_data.proto",
                "xla/service/hlo.proto",
                "xla/service/metrics.proto",
            ],
            &["proto"],
        )
        .expect("prost-build: compile_protos failed");

    println!("cargo:rerun-if-changed=proto/xla/xla_data.proto");
    println!("cargo:rerun-if-changed=proto/xla/service/hlo.proto");
    println!("cargo:rerun-if-changed=proto/xla/service/metrics.proto");
}
