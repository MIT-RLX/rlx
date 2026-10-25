// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `rlx` global against rlx-bind's runtime API table (`api.idl`), both
//! ways: every row this build's features cover resolves to a JS name, and
//! every name `rlx` exports is a row or a listed host helper. The C ABI and
//! Swift are held to the same table by rlx-bindgen, so a row missing here is
//! a binding that is no longer one-to-one with the others.

use rlx_js::Runtime;

/// The rlx-capi feature names the table uses, as this crate's features.
fn enabled(feature: &str) -> bool {
    match feature {
        "gguf" => cfg!(feature = "gguf"),
        "weights" => cfg!(feature = "weights"),
        "training" => cfg!(feature = "training"),
        "text" => cfg!(feature = "text"),
        "gguf-convert" => cfg!(feature = "gguf-convert"),
        "fpga" => cfg!(feature = "fpga"),
        "models" => cfg!(feature = "models"),
        other => panic!("api.idl names a feature rlx-js does not map: {other}"),
    }
}

fn eval(source: &str) -> String {
    Runtime::new()
        .eval(source)
        .unwrap_or_else(|e| panic!("script failed:\n{e}"))
}

fn js_list(names: &[String]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("'{n}'")).collect();
    format!("[{}]", quoted.join(", "))
}

#[test]
fn every_api_row_has_a_js_name() {
    let want = rlx_bind::api::contract("js", enabled);
    let missing = eval(&format!(
        "{}.filter(path => {{
            const [head, member] = path.split('.');
            const owner = rlx[head];
            if (owner === undefined) return true;
            if (member === undefined) return false;
            const proto = owner.prototype;
            return (proto === undefined || proto[member] === undefined) && owner[member] === undefined;
        }}).join(' ')",
        js_list(&want)
    ));
    assert!(
        missing.is_empty(),
        "api.idl rows with no JS name (add them to rlx-js, or mark the row \
         `js=-` with a why): {missing}"
    );
}

#[test]
fn every_js_export_is_a_row_or_a_host_helper() {
    let a = rlx_bind::api::api();
    let mut known = rlx_bind::api::contract("js", |_| true);
    known.extend(a.host("js").into_iter().map(str::to_string));
    // Members of the graph classes are the op table's, checked by rlx-bind.
    let exported = eval(
        "const out = [];
         for (const k of Object.getOwnPropertyNames(rlx)) {
             const v = rlx[k];
             out.push(k);
             if (k === 'Graph' || k === 'Tensor' || typeof v !== 'function' || !v.prototype) continue;
             for (const m of Object.getOwnPropertyNames(v.prototype))
                 if (m !== 'constructor' && m !== 'toString' && m !== '__class') out.push(k + '.' + m);
             for (const m of Object.getOwnPropertyNames(v))
                 if (!['length', 'name', 'prototype', '__className'].includes(m)) out.push(k + '.' + m);
         }
         out.join(' ')",
    );
    let extra: Vec<&str> = exported
        .split_whitespace()
        .filter(|n| !known.iter().any(|k| k == n))
        .collect();
    assert!(
        extra.is_empty(),
        "rlx exports names that are no api.idl row (add a row, so every \
         binding gets it, or list them under `host js`): {extra:?}"
    );
}
