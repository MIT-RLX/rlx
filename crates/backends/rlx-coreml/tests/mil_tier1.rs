// Host-portable MIL lowering tests for Tier-1 ANE ops.

use std::collections::HashMap;

use rlx_coreml::mil::lower_graph;
use rlx_coreml::proto;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};

fn main_block(model: &proto::Model) -> &proto::Block {
    let proto::model::Type::MlProgram(program) = model.r#type.as_ref().unwrap();
    program
        .functions
        .get("main")
        .expect("main")
        .block_specializations
        .values()
        .next()
        .expect("block")
}

fn types(model: &proto::Model) -> Vec<String> {
    main_block(model)
        .operations
        .iter()
        .map(|o| o.r#type.clone())
        .collect()
}

#[test]
fn lowers_argmax_via_reduce_argmax() {
    let mut g = Graph::new("argmax");
    let x = g.input("x", Shape::new(&[2, 5], DType::F32));
    let y = g.argmax(x, 1, false, Shape::new(&[2], DType::F32));
    g.set_outputs(vec![y]);
    let lowered = lower_graph(&g, &HashMap::new(), &Default::default()).expect("lower");
    let t = types(&lowered.model);
    assert!(t.contains(&"reduce_argmax".to_string()), "{t:?}");
}

#[test]
fn lowers_reverse_via_gather() {
    let mut g = Graph::new("rev");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let y = g.reverse(x, vec![1]);
    g.set_outputs(vec![y]);
    let lowered = lower_graph(&g, &HashMap::new(), &Default::default()).expect("lower");
    let t = types(&lowered.model);
    assert!(t.iter().filter(|op| *op == "gather").count() >= 1, "{t:?}");
}

#[test]
fn lowers_sliding_window_attention() {
    let mut g = Graph::new("sw");
    let q = g.input("q", Shape::new(&[1, 2, 4, 8], DType::F32));
    let k = g.input("k", Shape::new(&[1, 2, 4, 8], DType::F32));
    let v = g.input("v", Shape::new(&[1, 2, 4, 8], DType::F32));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: 2,
            head_dim: 8,
            mask_kind: MaskKind::SlidingWindow(2),
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[1, 2, 4, 8], DType::F32),
    );
    g.set_outputs(vec![y]);
    let lowered = lower_graph(&g, &HashMap::new(), &Default::default()).expect("lower");
    let t = types(&lowered.model);
    assert!(t.iter().any(|op| op == "const"), "{t:?}");
}
