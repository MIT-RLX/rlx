// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Best-effort MIR graph from NeMo YAML dims (when present).

use crate::{NemoConfig, NemoModel};
use anyhow::Result;
use rlx_ir::{DType, Graph, Op, Shape};

/// Build a tiny encoder probe graph when `encoder.d_model` (and friends) exist.
///
/// Not a full ASR/LLM model — binds a real Linear weight from the checkpoint
/// when a common EncDec name is present so RLXP can compile/run a slice.
pub fn build_nemo_probe_graph(model: &NemoModel, name: &str) -> Result<Option<Graph>> {
    let cfg = model.config();
    let Some(d_model) = cfg
        .get_usize("encoder.d_model")
        .or_else(|| cfg.get_usize("model.encoder.d_model"))
        .or_else(|| cfg.get_usize("cfg.encoder.d_model"))
    else {
        return Ok(None);
    };
    let features = cfg
        .get_usize("preprocessor.features")
        .or_else(|| cfg.get_usize("cfg.preprocessor.features"))
        .unwrap_or(d_model);

    let w_names = [
        "encoder.layers.0.feed_forward.linear1.weight",
        "encoder.layers.0.fc1.weight",
        "encoder.layers.0.linear1.weight",
        "encoder.layers.0.self_attn.linear_q.weight",
    ];
    let mut chosen: Option<(String, Vec<usize>)> = None;
    for n in w_names {
        if let Some(sh) = model.shape_of(n) {
            if sh.len() == 2 {
                chosen = Some((n.to_string(), sh.to_vec()));
                break;
            }
        }
    }

    let mut g = Graph::new(name);
    if let Some((wname, shape)) = chosen {
        // NeMo Linear `[out, in]`; MatMul needs `[in, out]` → Transpose.
        let in_dim = shape[1];
        let out_dim = shape[0];
        let x = g.input("x", Shape::new(&[1, in_dim], DType::F32));
        let w = g.add_node(
            Op::Param { name: wname },
            vec![],
            Shape::new(&shape, DType::F32),
        );
        let wt = g.add_node(
            Op::Transpose { perm: vec![1, 0] },
            vec![w],
            Shape::new(&[in_dim, out_dim], DType::F32),
        );
        let y = g.add_node(
            Op::MatMul,
            vec![x, wt],
            Shape::new(&[1, out_dim], DType::F32),
        );
        g.set_outputs(vec![y]);
    } else {
        // No matching Linear — identity Input sized to features / d_model.
        let dim = features.min(d_model).max(1);
        let x = g.input("x", Shape::new(&[1, dim], DType::F32));
        g.set_outputs(vec![x]);
    }
    Ok(Some(g))
}

/// Summarize config fields useful for graph builders.
pub fn nemo_arch_summary(cfg: &NemoConfig) -> String {
    format!(
        "d_model={:?} n_layers={:?} n_heads={:?} features={:?}",
        cfg.get_usize("encoder.d_model"),
        cfg.get_usize("encoder.n_layers"),
        cfg.get_usize("encoder.n_heads"),
        cfg.get_usize("preprocessor.features"),
    )
}
