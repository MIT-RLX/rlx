// Portable MIL tests for P1–P5 infrastructure (FP16, flex shapes, constexpr dequant).

use std::collections::HashMap;

use rlx_coreml::mil::{LowerOptions, lower_graph_with_options};
use rlx_coreml::proto;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, Graph, Op, Shape};

fn main_block(model: &proto::Model) -> &proto::Block {
    let proto::model::Type::MlProgram(program) = model.r#type.as_ref().unwrap();
    let func = program.functions.get("main").expect("main function");
    func.block_specializations
        .values()
        .next()
        .expect("a block specialization")
}

#[test]
fn lower_fp16_matmul_emits_float16_io() {
    let mut g = Graph::new("fp16");
    let x = g.input("x", Shape::new(&[2, 4], DType::F16));
    let w = g.param("w", Shape::new(&[4, 3], DType::F16));
    let y = g.matmul(x, w, Shape::new(&[2, 3], DType::F16));
    g.set_outputs(vec![y]);
    let params = HashMap::from([("w".to_string(), vec![0.1f32; 12])]);
    let opts = LowerOptions {
        float_dtype: DType::F16,
        ..Default::default()
    };
    let lowered = lower_graph_with_options(&g, &params, &Default::default(), &opts).expect("lower");
    assert_eq!(lowered.inputs[0].dtype, DType::F16);
}

#[test]
fn lower_flexible_input_emits_shape_range() {
    let mut g = Graph::new("flex");
    let x = g.input(
        "x",
        Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(8)], DType::F32),
    );
    let w = g.param("w", Shape::new(&[8, 4], DType::F32));
    let y = g.matmul(
        x,
        w,
        Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(4)], DType::F32),
    );
    g.set_outputs(vec![y]);
    let params = HashMap::from([("w".to_string(), vec![0.01f32; 32])]);
    let opts = LowerOptions {
        flexible_inputs: true,
        ..Default::default()
    };
    let lowered = lower_graph_with_options(&g, &params, &Default::default(), &opts).expect("lower");
    assert!(lowered.inputs[0].flex_dims[0]);
    assert_eq!(lowered.inputs[0].dims[0], -1);
}

#[test]
fn lower_q8_0_ondevice_uses_block_mul() {
    let (m, k, n) = (1usize, 32usize, 2usize);
    let w_nk = vec![0.1f32; n * k];
    let packed = rlx_gguf::quantize::quantize_q8_0(&w_nk).expect("quantize");
    let mut g = Graph::new("dq");
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("W", Shape::new(&[n, k], DType::F32));
    let y = g.append_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ8_0,
        },
        vec![xi, w],
        Shape::new(&[m, n], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);
    let mut typed = rlx_coreml::mil::TypedParams::new();
    typed.insert("W".to_string(), (packed, DType::U8));
    let opts = LowerOptions {
        ondevice_dequant: true,
        ..Default::default()
    };
    let lowered = lower_graph_with_options(&g, &HashMap::new(), &typed, &opts).expect("lower");
    let ops: Vec<_> = main_block(&lowered.model)
        .operations
        .iter()
        .map(|o| o.r#type.clone())
        .collect();
    assert!(
        ops.iter().any(|t| t == "mul"),
        "expected on-device block mul, got {ops:?}"
    );
}

fn assert_ondevice_mul_with_packed(scheme: QuantScheme, packed: Vec<u8>, k: usize, n: usize) {
    let m = 1usize;
    let mut g = Graph::new("dq");
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("W", Shape::new(&[n, k], DType::F32));
    let y = g.append_node(
        Op::DequantMatMul { scheme },
        vec![xi, w],
        Shape::new(&[m, n], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);
    let mut typed = rlx_coreml::mil::TypedParams::new();
    typed.insert("W".to_string(), (packed, DType::U8));
    let opts = LowerOptions {
        ondevice_dequant: true,
        ..Default::default()
    };
    let lowered = lower_graph_with_options(&g, &HashMap::new(), &typed, &opts).expect("lower");
    let ops: Vec<_> = main_block(&lowered.model)
        .operations
        .iter()
        .map(|o| o.r#type.clone())
        .collect();
    assert!(
        ops.iter().any(|t| t == "mul"),
        "{scheme} expected mul, got {ops:?}"
    );
}

fn assert_ondevice_mul_zeros(scheme: QuantScheme, k: usize, n: usize) {
    let block_sz = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    assert_eq!(
        (k * n) % block_sz,
        0,
        "{scheme}: {n}x{k} not divisible by block size {block_sz}"
    );
    let nb = (k * n) / block_sz;
    let packed = vec![0u8; nb * block_bytes];
    assert_ondevice_mul_with_packed(scheme, packed, k, n);
}

fn assert_ondevice_kquant_mul(scheme: QuantScheme, ggml: rlx_gguf::GgmlType) {
    let (m, k, n) = (1usize, 256usize, 2usize);
    let w_nk = vec![0.1f32; n * k];
    let packed = rlx_gguf::quantize(&w_nk, ggml).expect("quantize");
    let mut g = Graph::new("dq");
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("W", Shape::new(&[n, k], DType::F32));
    let y = g.append_node(
        Op::DequantMatMul { scheme },
        vec![xi, w],
        Shape::new(&[m, n], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);
    let mut typed = rlx_coreml::mil::TypedParams::new();
    typed.insert("W".to_string(), (packed, DType::U8));
    let opts = LowerOptions {
        ondevice_dequant: true,
        ..Default::default()
    };
    let lowered = lower_graph_with_options(&g, &HashMap::new(), &typed, &opts).expect("lower");
    let ops: Vec<_> = main_block(&lowered.model)
        .operations
        .iter()
        .map(|o| o.r#type.clone())
        .collect();
    assert!(
        ops.iter().any(|t| t == "mul"),
        "{scheme} expected mul, got {ops:?}"
    );
}

#[test]
fn lower_q2_k_ondevice_uses_block_mul() {
    assert_ondevice_kquant_mul(QuantScheme::GgufQ2K, rlx_gguf::GgmlType::Q2K);
}

#[test]
fn lower_q3_k_ondevice_uses_block_mul() {
    assert_ondevice_kquant_mul(QuantScheme::GgufQ3K, rlx_gguf::GgmlType::Q3K);
}

#[test]
fn lower_q6_k_ondevice_uses_block_mul() {
    assert_ondevice_kquant_mul(QuantScheme::GgufQ6K, rlx_gguf::GgmlType::Q6K);
}

#[test]
fn lower_q5_0_ondevice_uses_block_mul() {
    assert_ondevice_kquant_mul(QuantScheme::GgufQ5_0, rlx_gguf::GgmlType::Q5_0);
}

#[test]
fn lower_q5_1_ondevice_uses_block_mul() {
    assert_ondevice_kquant_mul(QuantScheme::GgufQ5_1, rlx_gguf::GgmlType::Q5_1);
}

#[test]
fn lower_iq4_nl_ondevice_uses_block_mul() {
    assert_ondevice_mul_zeros(QuantScheme::GgufIQ4NL, 32, 2);
}

#[test]
fn lower_iq4_xs_ondevice_uses_block_mul() {
    assert_ondevice_mul_zeros(QuantScheme::GgufIQ4XS, 256, 2);
}

#[test]
fn lower_mxfp4_ondevice_uses_block_mul() {
    assert_ondevice_mul_zeros(QuantScheme::GgufMXFP4, 32, 2);
}

#[test]
fn lower_tq2_0_ondevice_uses_block_mul() {
    assert_ondevice_mul_zeros(QuantScheme::GgufTQ2_0, 256, 2);
}

#[test]
fn lower_tq1_0_ondevice_uses_block_mul() {
    assert_ondevice_mul_zeros(QuantScheme::GgufTQ1_0, 256, 2);
}

#[test]
fn lower_nvfp4_ondevice_uses_block_mul() {
    assert_ondevice_kquant_mul(QuantScheme::GgufNVFP4, rlx_gguf::GgmlType::NVFP4);
}

#[test]
fn lower_iq2_xxs_ondevice_uses_block_mul() {
    assert_ondevice_kquant_mul(QuantScheme::GgufIQ2XXS, rlx_gguf::GgmlType::IQ2XXS);
}

#[test]
fn lower_iq2_xs_ondevice_uses_block_mul() {
    assert_ondevice_kquant_mul(QuantScheme::GgufIQ2XS, rlx_gguf::GgmlType::IQ2XS);
}

#[test]
fn lower_iq2_s_ondevice_uses_block_mul() {
    assert_ondevice_kquant_mul(QuantScheme::GgufIQ2S, rlx_gguf::GgmlType::IQ2S);
}

#[test]
fn lower_iq3_xxs_ondevice_uses_block_mul() {
    assert_ondevice_kquant_mul(QuantScheme::GgufIQ3XXS, rlx_gguf::GgmlType::IQ3XXS);
}

#[test]
fn lower_iq3_s_ondevice_uses_block_mul() {
    assert_ondevice_kquant_mul(QuantScheme::GgufIQ3S, rlx_gguf::GgmlType::IQ3S);
}

#[test]
fn lower_iq1_s_ondevice_uses_block_mul() {
    assert_ondevice_kquant_mul(QuantScheme::GgufIQ1S, rlx_gguf::GgmlType::IQ1S);
}

#[test]
fn lower_iq1_m_ondevice_uses_block_mul() {
    assert_ondevice_kquant_mul(QuantScheme::GgufIQ1M, rlx_gguf::GgmlType::IQ1M);
}
