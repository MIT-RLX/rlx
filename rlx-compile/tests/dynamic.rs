// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Dynamic shape compile-once / specialize-at-runtime integration tests.

use rlx_ir::DType;
use rlx_ir::DimBinding;
use rlx_ir::Shape;
use rlx_ir::hir::HirModule;
use rlx_ir::sym;
use rlx_compile::{CompilePipeline, FusionTarget};

#[test]
fn compile_dynamic_hir_then_specialize_seq_lengths() {
    let mut hir = HirModule::new("dyn_linear");
    let x = hir.input_batch_seq("x", sym::BATCH, sym::SEQ, 64, DType::F32);
    let w = hir.param("w", Shape::new(&[64, 64], DType::F32));
    let y = hir.linear(x, w, None, None, Shape::batch_seq(sym::BATCH, sym::SEQ, 64, DType::F32));
    hir.set_outputs(vec![y]);

    let pipe = CompilePipeline::new(FusionTarget::Cpu);
    let compiled = pipe.compile_hir(hir).expect("compile dynamic HIR");
    assert!(compiled.has_dynamic_dims());
    assert!(compiled.dynamic_symbols().contains(&sym::SEQ));

    let short = compiled.specialize(&pipe, &DimBinding::batch_seq(1, 8));
    assert!(short.lir.is_fully_static());
    assert_eq!(
        short.lir.as_graph().node(short.lir.as_graph().outputs[0]).shape,
        Shape::new(&[1, 8, 64], DType::F32)
    );
    let short_arena = short.lir.arena_size();
    assert!(short_arena > 0);

    let long = compiled.specialize(&pipe, &DimBinding::batch_seq(1, 128));
    assert!(long.lir.is_fully_static());
    assert_eq!(
        long.lir.as_graph().node(long.lir.as_graph().outputs[0]).shape,
        Shape::new(&[1, 128, 64], DType::F32)
    );
    assert!(long.lir.arena_size() > short_arena);
}
