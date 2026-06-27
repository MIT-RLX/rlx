// Host-portable lowering tests: validate that an IR graph lowers to a
// well-formed CoreML ML Program proto. These do NOT touch CoreML.framework
// and so run on every platform (including the Linux CI), guarding the
// lowering logic independently of on-device execution.

use std::collections::HashMap;

use rlx_coreml::mil::lower_graph;
use rlx_coreml::proto;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};

fn main_block(model: &proto::Model) -> &proto::Block {
    let proto::model::Type::MlProgram(program) = model.r#type.as_ref().unwrap();
    let func = program.functions.get("main").expect("main function");
    func.block_specializations
        .values()
        .next()
        .expect("a block specialization")
}

fn op_types(block: &proto::Block) -> Vec<String> {
    block.operations.iter().map(|o| o.r#type.clone()).collect()
}

#[test]
fn lowers_matmul_relu_chain() {
    let mut g = Graph::new("m");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("W", Shape::new(&[3, 4], DType::F32));
    let mm = g.matmul(x, w, Shape::new(&[2, 4], DType::F32));
    let y = g.activation(Activation::Relu, mm, Shape::new(&[2, 4], DType::F32));
    g.set_outputs(vec![y]);

    let mut params = HashMap::new();
    params.insert("W".to_string(), vec![0.0f32; 12]);

    let lowered = lower_graph(&g, &params, &Default::default()).expect("lower");
    let block = main_block(&lowered.model);

    // const(W) + matmul + relu
    let types = op_types(block);
    assert!(types.contains(&"const".to_string()), "{types:?}");
    assert!(types.contains(&"matmul".to_string()), "{types:?}");
    assert!(types.contains(&"relu".to_string()), "{types:?}");

    // One input feature (x), one output feature.
    assert_eq!(lowered.inputs.len(), 1);
    assert_eq!(lowered.inputs[0].ir_name, "x");
    assert_eq!(lowered.outputs.len(), 1);
    assert_eq!(lowered.outputs[0].dims, vec![2, 4]);

    // Spec version / opset sanity.
    assert!(lowered.model.specification_version >= 6);
}

/// The native RMSNorm input-gradient kernel (training) lowers to a tight,
/// broadcast-driven MIL graph — `rsqrt` + two `reduce_mean`s, no decomposition
/// `expand`s — directly, without going through the autodiff decomposer.
#[cfg(feature = "training")]
#[test]
fn rms_norm_backward_input_lowers_to_native_kernel() {
    let (rows, h) = (2usize, 4usize);
    let mut g = Graph::new("rms_bwd_in");
    let x = g.input("x", Shape::new(&[rows, h], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[h], DType::F32));
    let beta = g.param("beta", Shape::new(&[h], DType::F32));
    let dy = g.input("dy", Shape::new(&[rows, h], DType::F32));
    let dx = g.rms_norm_backward_input(x, gamma, beta, dy, -1, 1e-6);
    g.set_outputs(vec![dx]);

    let mut params = HashMap::new();
    params.insert("gamma".to_string(), vec![1.0f32; h]);
    params.insert("beta".to_string(), vec![0.0f32; h]);

    let lowered = lower_graph(&g, &params, &Default::default()).expect("lower native rms bwd");
    let block = main_block(&lowered.model);
    let types = op_types(block);

    // Native composition markers; the autodiff decomposition instead emits a
    // forest of `expand`/`reshape`/`const` ones — assert we did NOT take it.
    assert_eq!(
        types.iter().filter(|t| *t == "rsqrt").count(),
        1,
        "expected one rsqrt: {types:?}"
    );
    assert_eq!(
        types.iter().filter(|t| *t == "reduce_mean").count(),
        2,
        "expected mean(x^2) + mean(x*dy*gamma): {types:?}"
    );
    assert!(
        !types.contains(&"expand".to_string()),
        "native kernel should not expand: {types:?}"
    );
    assert_eq!(lowered.outputs[0].dims, vec![rows as i64, h as i64]);
}

/// The native MaxPool2d-backward kernel lowers at real CNN scale — where the
/// autodiff decomposition would build a ~200k×200k dense scatter and blow RLX's
/// size cap. Host-portable: proves the fix without a device.
#[cfg(feature = "training")]
#[test]
fn max_pool2d_backward_lowers_at_cnn_scale() {
    // MNIST conv feature map: [8,32,28,28] → 2×2/2 → [8,32,14,14].
    let mut g = Graph::new("mp_big");
    let x = g.input("x", Shape::new(&[8, 32, 28, 28], DType::F32));
    let dy = g.input("dy", Shape::new(&[8, 32, 14, 14], DType::F32));
    let dx = g.maxpool2d_backward(x, dy, vec![2, 2], vec![2, 2], vec![0, 0]);
    g.set_outputs(vec![dx]);

    let lowered = lower_graph(&g, &HashMap::new(), &Default::default())
        .expect("native maxpool backward lowers at CNN scale");
    let types = op_types(main_block(&lowered.model));
    assert!(types.iter().any(|t| t == "reduce_max"), "{types:?}");
    assert!(
        !types.iter().any(|t| t == "scatter" || t == "scatter_nd"),
        "native kernel must avoid scatter: {types:?}"
    );
    assert_eq!(lowered.outputs[0].dims, vec![8, 32, 28, 28]);
}

/// The native LayerNorm input-gradient kernel lowers to a composed MIL graph —
/// `rsqrt` + three `reduce_mean`s (mean, var, and the two backward means share the
/// mean op count), no decomposition `expand`s — directly, not via the decomposer.
#[cfg(feature = "training")]
#[test]
fn layer_norm_backward_input_lowers_to_native_kernel() {
    let (rows, h) = (2usize, 4usize);
    let mut g = Graph::new("ln_bwd_in");
    let x = g.input("x", Shape::new(&[rows, h], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[h], DType::F32));
    let dy = g.input("dy", Shape::new(&[rows, h], DType::F32));
    let dx = g.layer_norm_backward_input(x, gamma, dy, -1, 1e-5);
    g.set_outputs(vec![dx]);

    let mut params = HashMap::new();
    params.insert("gamma".to_string(), vec![1.0f32; h]);

    let lowered = lower_graph(&g, &params, &Default::default()).expect("lower native ln bwd");
    let types = op_types(main_block(&lowered.model));
    assert_eq!(
        types.iter().filter(|t| *t == "rsqrt").count(),
        1,
        "expected one rsqrt: {types:?}"
    );
    // mean(x), mean(xc²), mean(sy), mean(sy·x_hat)
    assert_eq!(
        types.iter().filter(|t| *t == "reduce_mean").count(),
        4,
        "expected 4 reduce_means: {types:?}"
    );
    assert!(
        !types.contains(&"expand".to_string()),
        "native kernel should not expand: {types:?}"
    );
    assert_eq!(lowered.outputs[0].dims, vec![rows as i64, h as i64]);
}

/// Native fused attention backward lowers to a tight MIL graph — one `softmax`
/// (recomputed P), ≥3 `matmul`s (QKᵀ, dP=dO·Vᵀ, dQ=ds·K) and the softmax-JVP
/// `reduce_sum` — NOT the autodiff decomposition's large reconstructed-forward graph.
#[cfg(feature = "training")]
#[test]
fn attention_backward_lowers_to_fused_kernel() {
    use rlx_ir::op::{AttentionBwdWrt, MaskKind};
    let (b, h, s, d) = (1usize, 1usize, 3usize, 4usize);
    let mut g = Graph::new("attn_bwd");
    let q = g.input("q", Shape::new(&[b, h, s, d], DType::F32));
    let k = g.input("k", Shape::new(&[b, h, s, d], DType::F32));
    let v = g.input("v", Shape::new(&[b, h, s, d], DType::F32));
    let dy = g.input("dy", Shape::new(&[b, h, s, d], DType::F32));
    let dq = g.attention_backward(
        AttentionBwdWrt::Query,
        q,
        k,
        v,
        dy,
        h,
        d,
        MaskKind::Causal,
        None,
    );
    g.set_outputs(vec![dq]);

    let lowered =
        lower_graph(&g, &HashMap::new(), &Default::default()).expect("lower native attn bwd");
    let types = op_types(main_block(&lowered.model));
    assert_eq!(
        types.iter().filter(|t| *t == "softmax").count(),
        1,
        "one recomputed-P softmax: {types:?}"
    );
    assert!(
        types.iter().filter(|t| *t == "matmul").count() >= 3,
        "QKᵀ + dP=dO·Vᵀ + dQ=ds·K: {types:?}"
    );
    assert!(
        types.iter().any(|t| t == "reduce_sum"),
        "softmax-JVP rowsum: {types:?}"
    );
    assert_eq!(
        lowered.outputs[0].dims,
        vec![b as i64, h as i64, s as i64, d as i64]
    );
}

/// Native attention backward in the fused `[B,S,H·D]` layout + a SlidingWindow mask:
/// it canonicalizes each operand (reshape+transpose in), runs the core (shared
/// `apply_score_mask` handles the window), then transposes+reshapes the gradient back.
#[cfg(feature = "training")]
#[test]
fn attention_backward_fused_layout_and_mask_lower() {
    use rlx_ir::op::{AttentionBwdWrt, MaskKind};
    let (b, s, h, d) = (1usize, 3usize, 2usize, 4usize);
    let hd = h * d;
    let mut g = Graph::new("attn_bwd_fused");
    let q = g.input("q", Shape::new(&[b, s, hd], DType::F32));
    let k = g.input("k", Shape::new(&[b, s, hd], DType::F32));
    let v = g.input("v", Shape::new(&[b, s, hd], DType::F32));
    let dy = g.input("dy", Shape::new(&[b, s, hd], DType::F32));
    let dq = g.attention_backward(
        AttentionBwdWrt::Query,
        q,
        k,
        v,
        dy,
        h,
        d,
        MaskKind::SlidingWindow(2),
        None,
    );
    g.set_outputs(vec![dq]);

    let lowered = lower_graph(&g, &HashMap::new(), &Default::default())
        .expect("fused-layout + sliding-window attention backward lowers");
    let types = op_types(main_block(&lowered.model));
    // 4 operands × (reshape+transpose) in + transpose+reshape out.
    assert!(
        types.iter().filter(|t| *t == "transpose").count() >= 5,
        "operand canonicalization + output un-canonicalization: {types:?}"
    );
    assert!(types.contains(&"softmax".to_string()), "{types:?}");
    assert_eq!(lowered.outputs[0].dims, vec![b as i64, s as i64, hd as i64]);
}

/// Native GroupNorm input-gradient lowers via a single `[N,C,H,W]→[N,G,M]` reshape
/// + group-axis reduces — NOT the decomposition's per-group `slice`/`concat` loop.
#[cfg(feature = "training")]
#[test]
fn group_norm_backward_input_lowers_to_native_kernel() {
    let (n, c, hh, w, ng) = (2usize, 4usize, 2usize, 2usize, 2usize);
    let mut g = Graph::new("gn_bwd_in");
    let x = g.input("x", Shape::new(&[n, c, hh, w], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[c], DType::F32));
    let beta = g.param("beta", Shape::new(&[c], DType::F32));
    let dy = g.input("dy", Shape::new(&[n, c, hh, w], DType::F32));
    let dx = g.group_norm_backward_input(x, gamma, beta, dy, ng, 1e-5);
    g.set_outputs(vec![dx]);

    let mut params = HashMap::new();
    params.insert("gamma".to_string(), vec![1.0f32; c]);
    params.insert("beta".to_string(), vec![0.0f32; c]);

    let lowered = lower_graph(&g, &params, &Default::default()).expect("lower native gn bwd");
    let types = op_types(main_block(&lowered.model));
    assert_eq!(
        types.iter().filter(|t| *t == "rsqrt").count(),
        1,
        "one group-norm rsqrt: {types:?}"
    );
    assert!(
        !types.iter().any(|t| t == "concat" || t == "slice_by_size"),
        "native kernel must avoid the per-group slice/concat loop: {types:?}"
    );
    assert_eq!(
        lowered.outputs[0].dims,
        vec![n as i64, c as i64, hh as i64, w as i64]
    );
}

/// Native softmax-cross-entropy (forward `WithLogits` + backward) lower to MIL's
/// single `one_hot` op — NOT the O(C) column-concat decomposition. At LLM vocab
/// sizes the decompose emits thousands of `concat`/`reshape` nodes; the native
/// kernel is one node regardless of C. Host-portable (no device).
#[cfg(feature = "training")]
#[test]
fn softmax_cross_entropy_lowers_to_native_one_hot() {
    let (n, c) = (3usize, 5usize);

    // backward: dlogits = (softmax − onehot)·d_loss
    let mut g = Graph::new("sce_bwd");
    let logits = g.input("logits", Shape::new(&[n, c], DType::F32));
    let labels = g.input("labels", Shape::new(&[n], DType::F32));
    let d_loss = g.input("d_loss", Shape::new(&[n], DType::F32));
    let dlogits = g.softmax_cross_entropy_backward(logits, labels, d_loss);
    g.set_outputs(vec![dlogits]);
    let lowered =
        lower_graph(&g, &HashMap::new(), &Default::default()).expect("lower native sce bwd");
    let types = op_types(main_block(&lowered.model));
    assert!(
        types.contains(&"one_hot".to_string()),
        "bwd must use native one_hot: {types:?}"
    );
    assert!(types.contains(&"softmax".to_string()), "bwd: {types:?}");
    assert!(
        !types.contains(&"concat".to_string()),
        "bwd must avoid the O(C) concat decompose: {types:?}"
    );
    assert_eq!(lowered.outputs[0].dims, vec![n as i64, c as i64]);

    // forward: loss[n] = logsumexp(logits) − logits[label]
    let mut g = Graph::new("sce_fwd");
    let logits = g.input("logits", Shape::new(&[n, c], DType::F32));
    let labels = g.input("labels", Shape::new(&[n], DType::F32));
    let loss = g.softmax_cross_entropy_with_logits(logits, labels);
    g.set_outputs(vec![loss]);
    let lowered =
        lower_graph(&g, &HashMap::new(), &Default::default()).expect("lower native sce fwd");
    let types = op_types(main_block(&lowered.model));
    assert!(
        types.contains(&"one_hot".to_string()),
        "fwd must use native one_hot: {types:?}"
    );
    assert!(
        !types.contains(&"concat".to_string()),
        "fwd must avoid the O(C) concat decompose: {types:?}"
    );
}

/// MIL `log` is emitted with its required `epsilon` (else CoreML rejects the
/// model — the failure that broke softmax-cross-entropy training).
#[test]
fn log_activation_binds_epsilon() {
    use rlx_ir::op::Activation;
    let mut g = Graph::new("log");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.activation(Activation::Log, x, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![y]);
    let lowered = lower_graph(&g, &HashMap::new(), &Default::default()).expect("lower log");
    let block = main_block(&lowered.model);
    let log_op = block
        .operations
        .iter()
        .find(|o| o.r#type == "log")
        .expect("a log op");
    assert!(
        log_op.inputs.contains_key("epsilon"),
        "log must bind epsilon: {:?}",
        log_op.inputs.keys().collect::<Vec<_>>()
    );
}

#[test]
fn silu_composes_to_sigmoid_and_mul() {
    let mut g = Graph::new("silu");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.activation(Activation::Silu, x, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![y]);

    let lowered = lower_graph(&g, &HashMap::new(), &Default::default()).expect("lower");
    let types = op_types(main_block(&lowered.model));
    assert!(types.contains(&"sigmoid".to_string()), "{types:?}");
    assert!(types.contains(&"mul".to_string()), "{types:?}");
}

#[test]
fn rms_norm_expands_to_primitive_chain() {
    let mut g = Graph::new("rms");
    let x = g.input("x", Shape::new(&[2, 8], DType::F32));
    let gamma = g.param("g", Shape::new(&[8], DType::F32));
    let y = g.append_node(
        rlx_ir::Op::RmsNorm {
            axis: -1,
            eps: 1e-6,
        },
        vec![x, gamma],
        Shape::new(&[2, 8], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);

    let mut params = HashMap::new();
    params.insert("g".to_string(), vec![1.0f32; 8]);

    let lowered = lower_graph(&g, &params, &Default::default()).expect("lower");
    let types = op_types(main_block(&lowered.model));
    for needed in ["mul", "reduce_mean", "add", "rsqrt"] {
        assert!(
            types.contains(&needed.to_string()),
            "missing {needed} in {types:?}"
        );
    }
}

#[test]
fn binary_div_maps_to_real_div() {
    let mut g = Graph::new("div");
    let a = g.input("a", Shape::new(&[3], DType::F32));
    let b = g.input("b", Shape::new(&[3], DType::F32));
    let y = g.binary(BinaryOp::Div, a, b, Shape::new(&[3], DType::F32));
    g.set_outputs(vec![y]);

    let lowered = lower_graph(&g, &HashMap::new(), &Default::default()).expect("lower");
    let types = op_types(main_block(&lowered.model));
    assert!(types.contains(&"real_div".to_string()), "{types:?}");
}

#[test]
fn large_consts_go_to_weight_blob() {
    // A 12-element weight (≥ 10) must move to the blob; the const's value
    // becomes a blobFileValue, and the blob bytes are non-empty.
    let mut g = Graph::new("blob");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("W", Shape::new(&[3, 4], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[2, 4], DType::F32));
    g.set_outputs(vec![y]);
    let mut params = HashMap::new();
    params.insert("W".to_string(), (0..12).map(|i| i as f32).collect());

    let lowered = lower_graph(&g, &params, &Default::default()).expect("lower");
    assert!(!lowered.blob.is_empty(), "weight blob should be populated");
    // Header count (first u32) should be 1 (one weight).
    let count = u32::from_le_bytes([
        lowered.blob[0],
        lowered.blob[1],
        lowered.blob[2],
        lowered.blob[3],
    ]);
    assert_eq!(count, 1);
    // The const op for W references the blob, not an inline tensor.
    let block = main_block(&lowered.model);
    let w_const = block
        .operations
        .iter()
        .find(|o| o.r#type == "const")
        .expect("a const op");
    let val = w_const.attributes.get("val").expect("val attr");
    assert!(
        matches!(val.value, Some(proto::value::Value::BlobFileValue(_))),
        "large const must use blobFileValue"
    );
}

#[test]
fn small_consts_stay_inline() {
    // A 3-element const stays inline (no blob).
    let mut g = Graph::new("inline");
    let x = g.input("x", Shape::new(&[3], DType::F32));
    let w = g.param("w", Shape::new(&[3], DType::F32));
    let y = g.binary(BinaryOp::Add, x, w, Shape::new(&[3], DType::F32));
    g.set_outputs(vec![y]);
    let mut params = HashMap::new();
    params.insert("w".to_string(), vec![1.0f32, 2.0, 3.0]);

    let lowered = lower_graph(&g, &params, &Default::default()).expect("lower");
    assert!(lowered.blob.is_empty(), "small const should stay inline");
}

#[test]
fn oversize_model_is_an_explicit_error() {
    use rlx_coreml::mlpackage::{PROTO_MAX_BYTES, check_model_size};

    let mut g = Graph::new("sz");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    let y = g.activation(Activation::Relu, x, Shape::new(&[4], DType::F32));
    g.set_outputs(vec![y]);
    let lowered = lower_graph(&g, &HashMap::new(), &Default::default()).expect("lower");

    // Real model comfortably under the 2 GiB cap.
    assert!(check_model_size(&lowered.model, PROTO_MAX_BYTES).is_ok());

    // A tiny synthetic limit must trip the explicit TooLarge error.
    match check_model_size(&lowered.model, 8) {
        Err(rlx_coreml::CoremlError::TooLarge { bytes, limit, .. }) => {
            assert_eq!(limit, 8);
            assert!(bytes > 8);
        }
        other => panic!("expected TooLarge, got {other:?}"),
    }
}

#[test]
fn dangling_reference_is_caught() {
    use std::collections::HashMap as Map;
    // A quantized Param fed to a plain matmul (not a Dequant* op): the
    // param emits no const, so the matmul would reference an undefined
    // value. The verify pass must reject this with a clear error rather
    // than producing a model CoreML can't parse.
    let mut g = Graph::new("dangle");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let w = g.param("W", Shape::new(&[4, 3], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
    g.set_outputs(vec![y]);

    let mut typed: Map<String, (Vec<u8>, DType)> = Map::new();
    typed.insert("W".to_string(), (vec![0u8; 12], DType::I8)); // quantized, no f32

    match lower_graph(&g, &HashMap::new(), &typed) {
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("dangling reference"), "got: {msg}");
        }
        Ok(_) => panic!("expected a dangling-reference error"),
    }
}

#[test]
fn missing_param_is_an_error() {
    let mut g = Graph::new("p");
    let x = g.input("x", Shape::new(&[2, 2], DType::F32));
    let w = g.param("W", Shape::new(&[2, 2], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[2, 2], DType::F32));
    g.set_outputs(vec![y]);

    // No params provided → must error rather than silently produce a bad model.
    assert!(lower_graph(&g, &HashMap::new(), &Default::default()).is_err());
}
