// Link Apple's Accelerate framework on macOS so Muon's Newton–Schulz
// orthogonalization can call `cblas_sgemm` (AMX-backed BLAS) instead of the
// portable hand-rolled matmul — the dominant cost of Muon on large models.
// No-op on other platforms (the naive parallel path is used there).
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
