// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

    /// The embedding of `x` (row-major, `spec.input_dim` values per row), for
    /// any number of rows. The forward graph is compiled for `spec.n` rows, so
    /// other counts go through it `spec.n` at a time, the last chunk
    /// zero-padded and its padding dropped. The encoder is a per-row MLP, so
    /// the padding does not change the real rows.
    pub fn forward_embedding(&mut self, x: &[f32]) -> Vec<f32> {
        let (rows, d) = (self.spec.n, self.spec.input_dim);
        if d == 0 || rows == 0 || x.len() == rows * d {
            return self.run_forward(x);
        }
        assert_eq!(
            x.len() % d,
            0,
            "forward_embedding: {} values is not whole rows of {d}",
            x.len()
        );
        let mut out = Vec::new();
        let mut chunk = vec![0.0f32; rows * d];
        for part in x.chunks(rows * d) {
            chunk[..part.len()].copy_from_slice(part);
            chunk[part.len()..].fill(0.0);
            let emb = self.run_forward(&chunk);
            let out_d = emb.len() / rows;
            out.extend_from_slice(&emb[..(part.len() / d) * out_d]);
        }
        out
    }

    fn run_forward(&mut self, x: &[f32]) -> Vec<f32> {
        let outs = self.forward.run(&[("x", x)]);
        outs.into_iter().next().unwrap_or_default()
    }
}
