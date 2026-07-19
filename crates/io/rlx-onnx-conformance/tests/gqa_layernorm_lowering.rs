//! Lowering coverage for the Microsoft contrib fused ops emitted by the
//! transformers.js / ONNX Runtime LM exporters (ChatterBox's `language_model`,
//! Phi/Qwen/Llama ONNX): `GroupQueryAttention`, `SkipSimplifiedLayerNormalization`,
//! `SimplifiedLayerNormalization`, `ArgMax`. Proves each decomposes into the
//! expected native rlx primitives (no stubs / unsupported).

use std::collections::HashMap;

use rlx_onnx_import::{ImportOptions, build_hir_from_onnx_file};

#[test]
fn gqa_layernorm_fixture_lowers_to_native_ops() {
    let path = rlx_onnx_conformance::synthetic::gqa_layernorm_fixture();

    let mut named = HashMap::new();
    named.insert("seq".to_string(), 3usize);
    named.insert("batch".to_string(), 1usize);
    named.insert("past".to_string(), 0usize);
    let opts = ImportOptions {
        sequence_length: 3,
        named_lengths: named,
        strict: true, // fail on ANY stub / unsupported op
        ..ImportOptions::default()
    };

    let (hir, _params, report, _manifest) =
        build_hir_from_onnx_file(&path, opts).expect("import GQA + layernorm fixture (strict)");
    assert_eq!(report.stubbed, 0, "no node should be stubbed");
    assert!(
        report.unsupported.is_empty(),
        "unexpected unsupported ops: {:?}",
        report.unsupported
    );

    let graph = rlx_ir::hir_to_graph(hir).expect("hir_to_graph");
    let mut hist: HashMap<String, usize> = HashMap::new();
    for n in graph.nodes() {
        *hist.entry(format!("{:?}", n.op.kind())).or_insert(0) += 1;
    }
    let count = |k: &str| hist.get(k).copied().unwrap_or(0);

    // GroupQueryAttention → 1 Attention + 2 Rope (Q, K) + 3 Narrow (packed QKV split).
    assert_eq!(count("Attention"), 1, "GQA → one Attention: {hist:?}");
    assert_eq!(count("Rope"), 2, "GQA → RoPE on Q and K: {hist:?}");
    assert!(count("Narrow") >= 3, "GQA → packed-QKV split: {hist:?}");
    // Skip + Simplified LayerNorm → 2 RmsNorm.
    assert_eq!(count("RmsNorm"), 2, "two RMSNorms: {hist:?}");
    // ArgMax preserved.
    assert_eq!(count("ArgMax"), 1, "ArgMax lowered: {hist:?}");
}
