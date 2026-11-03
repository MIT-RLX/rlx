// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regenerate `src/onnx.rs` from `onnx.proto` with the pure-Rust parser.

use std::path::Path;

const HEADER: &str = "// RLX — versatile ML compiler + runtime.\n\
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n\
// SPDX-License-Identifier: MIT OR Apache-2.0\n";

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let out = std::env::temp_dir().join("rlx-onnx-proto-regen");
    std::fs::create_dir_all(&out).unwrap();
    protobuf_codegen::Codegen::new()
        .pure()
        .include(root)
        .input(root.join("onnx.proto"))
        .out_dir(&out)
        .run_from_script();
    let generated = std::fs::read_to_string(out.join("onnx.rs")).unwrap();
    let dest = root.join("src/onnx.rs");
    std::fs::write(&dest, format!("{HEADER}{generated}")).unwrap();
    println!("wrote {}", dest.display());
}
