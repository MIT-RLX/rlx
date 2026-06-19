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
