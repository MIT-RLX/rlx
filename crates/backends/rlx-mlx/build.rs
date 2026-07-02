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

fn main() {
    println!("cargo::rustc-check-cfg=cfg(rlx_mlx_host)");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // MLX has a real backend on macOS / Linux / Windows and on iOS (device +
    // simulator — MLX's CMake builds a Metal backend there). tvOS / watchOS /
    // visionOS are not MLX-supported, so they keep the non-host stub.
    if matches!(os.as_str(), "macos" | "linux" | "windows" | "ios") {
        println!("cargo:rustc-cfg=rlx_mlx_host");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
