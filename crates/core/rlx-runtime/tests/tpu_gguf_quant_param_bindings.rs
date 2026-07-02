//! TPU GGUF param bake via [`CompileOptions::quant_param_bindings`] (HLO only).

#![cfg(feature = "tpu")]

use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Shape};
use rlx_tpu::lower::{LowerParamBytes, lower_graph_with_rng_and_params};

fn contains_opcode(bytes: &[u8], needle: &str) -> bool {
    let pat: Vec<u8> = std::iter::once(needle.len() as u8)
        .chain(needle.bytes())
        .collect();
    bytes.windows(pat.len()).any(|w| w == pat)
}

#[test]
fn quant_param_bindings_lowers_to_baked_dot() {
    let k = 32usize;
    let n = 2usize;
    let w_f32: Vec<f32> = (0..k * n).map(|i| 0.04 * (i as f32).cos()).collect();
    let packed = rlx_gguf::quantize::quantize_q8_0(&w_f32).unwrap();
    let mut g = Graph::new("tpu_qpb");
    let x = g.input("x", Shape::new(&[1, k], DType::F32));
    let w = g.param("weights", Shape::new(&[n, k], DType::U8));
    let y = g.add_node(
        rlx_ir::Op::DequantMatMul {
            scheme: QuantScheme::GgufQ8_0,
        },
        vec![x, w],
        Shape::new(&[1, n], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut bindings: LowerParamBytes = LowerParamBytes::new();
    bindings.insert("weights".to_string(), packed);
    let b = lower_graph_with_rng_and_params(&g, rlx_ir::RngOptions::zero(), Some(&bindings)).bytes;
    assert!(contains_opcode(&b, "dot"));
    assert!(contains_opcode(&b, "constant"));
}
