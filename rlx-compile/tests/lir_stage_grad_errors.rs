// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

use rlx_autodiff::{AutodiffError, grad_with_loss_module};
use rlx_compile::CompilePipeline;
use rlx_ir::{DType, Graph, GraphModule, GraphStage, MirModule, NodeId, Shape};

#[test]
fn lir_stage_grad_errors() {
    let mut g = Graph::new("t");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    g.set_outputs(vec![x]);
    let lir = CompilePipeline::default().plan_lir(MirModule::from_graph(g));
    let err = grad_with_loss_module(GraphModule::from_lir(lir), &[NodeId(0)]).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::WrongStage {
            got: GraphStage::Lir,
            ..
        }
    ));
}
