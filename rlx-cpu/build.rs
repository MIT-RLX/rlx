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
    // The `blas` feature gates both the FFI extern in src/blas.rs and
    // the link directives below. `--no-default-features` skips both
    // and the kernels fall back to a portable scalar gemm.
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_BLAS");
    if std::env::var_os("CARGO_FEATURE_BLAS").is_none() {
        return;
    }

    // macOS: vendored Accelerate framework provides cblas_sgemm.
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // Windows / Linux: link OpenBLAS. Honour OPENBLAS_DIR (matches
    // openblas-src / burn-ndarray's blas-openblas-system convention)
    // and fall back to system search paths.
    #[cfg(not(target_os = "macos"))]
    {
        println!("cargo:rerun-if-env-changed=OPENBLAS_DIR");
        println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");
        if let Ok(dir) = std::env::var("OPENBLAS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
        } else if let Ok(root) = std::env::var("OPENBLAS_DIR") {
            println!("cargo:rustc-link-search=native={root}/lib");
        }
        // OpenBLAS provides cblas_sgemm under either name on different
        // distributions — try the explicit `libopenblas` first (Win
        // MSVC + most Linux distros), letting the linker fall back to
        // `cblas` if needed via the user's RUSTFLAGS.
        println!("cargo:rustc-link-lib=openblas");
    }
}
