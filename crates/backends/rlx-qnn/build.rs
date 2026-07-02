// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// GPL-3.0-only.
//
// Compiles the QNN FFI shim (`runtime/rlx_qnn_shim.c`) against the SDK headers
// when the `runtime` feature is on. Without that feature this is a no-op, so
// the codegen tool stays dependency- and SDK-free (the empty-prelude / optional
// -backend posture).

fn main() {
    println!("cargo:rerun-if-changed=runtime/rlx_qnn_shim.c");
    println!("cargo:rerun-if-changed=runtime/rlx_qnn_shim.h");
    println!("cargo:rerun-if-env-changed=QNN_SDK_ROOT");

    if std::env::var_os("CARGO_FEATURE_RUNTIME").is_none() {
        return;
    }

    let sdk = std::env::var("QNN_SDK_ROOT").unwrap_or_else(|_| {
        panic!(
            "rlx-qnn `runtime` feature requires QNN_SDK_ROOT to point at a \
             Qualcomm AI Engine Direct SDK (for include/QNN headers)"
        )
    });

    cc::Build::new()
        .file("runtime/rlx_qnn_shim.c")
        .include(format!("{sdk}/include/QNN"))
        .std("c11")
        .warnings(false)
        .compile("rlx_qnn_shim");

    // The shim dlopen's the backend library at run time.
    println!("cargo:rustc-link-lib=dl");
}
