// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Compiled UMAP model graphs.

use rlx_runtime::{CompiledGraph, Session};

use crate::encoder::loss::UmapTrainGraph;
use crate::encoder::mlp::{ModelSpec, build_forward_graph};
use crate::weights::WeightStore;

pub struct CompiledUmap {
    pub forward: CompiledGraph,
    pub train: CompiledGraph,
    pub train_meta: UmapTrainGraph,
    pub spec: ModelSpec,
}

impl CompiledUmap {
    pub fn compile(session: &Session, spec: &ModelSpec, n_pos: usize, n_neg: usize) -> Self {
        let (fwd, _, _) = build_forward_graph(spec);
        let train_meta = crate::encoder::build_train_graph(spec, n_pos, n_neg);
        Self {
            forward: session.compile(fwd),
            train: session.compile(train_meta.backward.clone()),
            train_meta,
            spec: spec.clone(),
        }
    }

    pub fn set_weights(&mut self, w: &WeightStore) {
        w.apply(&mut self.forward);
        w.apply(&mut self.train);
    }

    pub fn forward_embedding(&mut self, x: &[f32]) -> Vec<f32> {
        let outs = self.forward.run(&[("x", x)]);
        outs.into_iter().next().unwrap_or_default()
    }
}
