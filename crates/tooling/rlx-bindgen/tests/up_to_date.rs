// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The checked-in generated bindings match what the op table generates now.
//!
//! Fails after an op-table (or rlx-capi) change until `just gen-bindings` has
//! been run, so the C ABI, header, Swift and `.pyi` cannot fall behind rlx-js
//! and pyrlx, which expand the table at compile time.

#[test]
fn generated_bindings_are_up_to_date() {
    let root = rlx_bindgen::workspace_root();
    let stale = rlx_bindgen::stale(&root);
    assert!(
        stale.is_empty(),
        "stale generated bindings — run `just gen-bindings`: {stale:?}"
    );
}

#[test]
fn every_op_has_a_c_function_in_the_header() {
    let root = rlx_bindgen::workspace_root();
    let header =
        std::fs::read_to_string(root.join("crates/bindings/rlx-capi/include/rlx.h")).unwrap();
    for op in rlx_bind::manifest() {
        let decl = format!("int32_t rlx_graph_{}(", op.c_name());
        assert!(header.contains(&decl), "rlx.h has no `{decl}`");
    }
}
