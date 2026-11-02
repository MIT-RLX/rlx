// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fused matmul / conv epilogues (`apply_act`) divide through
//! `epi_div_rn` and exponentiate through `epi_exp`, copies of rlxsl's
//! `rlx_div_rn` / `rlx_exp`, and spell the erf polynomial with explicit `fma`. A raw WGSL `/` or `exp`, or a
//! plain `a * b + c` chain is compiled differently by native wgpu and by a
//! browser's WebGPU on the same Metal GPU: BCG-FM's squeeze-excite sigmoid
//! (an `m = 1` gemv epilogue) and every fused GELU moved by ULPs between them.
//! These shaders are not built through build.rs, so the copies are pinned here;
//! the normalization, softmax and attention kernels carry the same copies.

use std::path::Path;

/// The body of `fn <name>(…) -> f32 { … }` in `src`, braces included.
fn body<'a>(src: &'a str, name: &str) -> Option<&'a str> {
    let start = src.find(&format!("fn {name}("))?;
    let open = start + src[start..].find('{')?;
    let mut depth = 0usize;
    for (i, ch) in src[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&src[open..=open + i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// rlxsl's helper `rlx_<name>` as the kernels copy it (`rlx_*` → `epi_*`).
fn helper(name: &str) -> String {
    let want = body(rlxsl::exact::wgsl_numeric_helpers(), &format!("rlx_{name}"))
        .unwrap_or_else(|| panic!("rlxsl's rlx_{name}"));
    ["div_rn", "sqrt_rn", "rsqrt_rn", "exp"]
        .iter()
        .fold(want.to_string(), |b, n| {
            b.replace(&format!("rlx_{n}"), &format!("epi_{n}"))
        })
}

/// Whether `src` calls the builtin `name(` (not a helper ending in it).
fn calls_builtin(src: &str, name: &str) -> bool {
    src.match_indices(&format!("{name}("))
        .any(|(i, _)| !src[..i].ends_with(|c: char| c.is_ascii_alphanumeric() || c == '_'))
}

#[test]
fn every_epilogue_divides_exactly_and_fuses_its_polynomials() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernels");
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(&path).unwrap_or_default();
        if !name.ends_with(".wgsl") {
            continue;
        }
        // every copy of a helper, in any kernel, is rlxsl's
        for h in ["div_rn", "sqrt_rn", "rsqrt_rn", "exp"] {
            if let Some(got) = body(&src, &format!("epi_{h}")) {
                assert_eq!(got, helper(h), "{name}: epi_{h} is not rlxsl's rlx_{h}");
            }
        }
        // elementwise_region's `apply_activation` is the generated rlxsl math
        if !src.contains("fn apply_act(") {
            continue;
        }
        seen += 1;
        for h in ["div_rn", "exp"] {
            assert!(
                body(&src, &format!("epi_{h}")).is_some(),
                "{name}: no epi_{h}"
            );
        }
        for raw in [
            "1.0 / (1.0 + exp",
            "v / (1.0 + exp",
            "1.0 / (1.0 + 0.3275911",
            "1.0 - poly *",
            "0.044715 * x3)",
        ] {
            assert!(
                !src.contains(raw),
                "{name}: `{raw}…` is back (inexact `/` or a contractible chain)"
            );
        }
        assert!(
            !calls_builtin(&src, "exp"),
            "{name}: calls the builtin `exp` (its rounding depends on the math mode)"
        );
    }
    assert_eq!(seen, 18, "the epilogue shaders moved; update this test");
}

/// The normalization, softmax and attention kernels (forward and backward) take
/// `1/n`, `1/sqrt` and `exp` from the helper copies (group norm from the helpers
/// build.rs prepends): a raw `/`, `inverseSqrt` or `exp` rounds by the
/// math mode, and LayerNorm alone moved BIOT's first token embedding.
#[test]
fn norm_softmax_attention_kernels_use_the_exact_helpers() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernels");
    for name in [
        "layernorm.wgsl",
        "fused_residual_ln.wgsl",
        "fused_residual_ln_tee.wgsl",
        "fused_residual_rms_norm.wgsl",
        "ada_layer_norm.wgsl",
        "softmax.wgsl",
        "attention.wgsl",
        "layer_norm_backward.wgsl",
        "rms_norm_backward.wgsl",
        "ada_layer_norm_backward.wgsl",
        "attention_bwd.wgsl",
        "softmax_cross_entropy.wgsl",
        "softmax_cross_entropy_bwd.wgsl",
        "group_norm.wgsl",
        "group_norm_backward.wgsl",
    ] {
        let src = std::fs::read_to_string(dir.join(name)).unwrap();
        assert!(
            !calls_builtin(&src, "inverseSqrt"),
            "{name}: raw inverseSqrt"
        );
        assert!(!calls_builtin(&src, "exp"), "{name}: raw exp");
        assert!(!src.contains("1.0 / "), "{name}: raw `1.0 / …`");
    }
}

/// rlxsl's WGSL `erf` (gelu in the elementwise kernels) uses explicit `fma`,
/// and its `exp` is the helper's.
#[test]
fn generated_wgsl_erf_is_fused() {
    let gelu = rlxsl::emit_activation(rlx_ir::op::Activation::Gelu, rlxsl::Lang::Wgsl);
    let text = format!("{gelu:?}");
    assert!(
        text.contains("fma("),
        "rlxsl's WGSL gelu lost its explicit fma: {text}"
    );
    assert!(
        text.contains("rlx_exp("),
        "rlxsl's WGSL gelu calls the builtin exp: {text}"
    );
}
