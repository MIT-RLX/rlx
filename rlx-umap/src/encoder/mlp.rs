// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Parametric UMAP encoder (MLP with ReLU hidden layers).

use rlx_compile::legalize_broadcast;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};

use crate::config::UmapConfig;
use crate::weights::{WeightStore, init_mat, init_vec};

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub n: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub hidden: Vec<usize>,
}

impl ModelSpec {
    pub fn from_config(config: &UmapConfig, n: usize, input_dim: usize) -> Self {
        Self {
            n,
            input_dim,
            output_dim: config.n_components,
            hidden: config.hidden_sizes.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParamSlot {
    pub name: String,
    pub param: NodeId,
    pub grad: Option<NodeId>,
}

pub fn init_model_weights(spec: &ModelSpec, seed: u64) -> WeightStore {
    let mut w = WeightStore::default();
    let mut s = seed;
    let mut in_d = spec.input_dim;
    for (li, &hd) in spec.hidden.iter().enumerate() {
        init_mat(&mut w, &format!("umap_w{li}"), in_d, hd, &mut s);
        init_vec(&mut w, &format!("umap_b{li}"), hd, &mut s);
        in_d = hd;
    }
    init_mat(&mut w, "umap_w_out", in_d, spec.output_dim, &mut s);
    init_vec(&mut w, "umap_b_out", spec.output_dim, &mut s);
    w
}

/// Forward graph: `x [n, d_in]` → `embed [n, d_out]`.
pub fn build_forward_graph(spec: &ModelSpec) -> (Graph, NodeId, Vec<ParamSlot>) {
    let f = DType::F32;
    let mut g = Graph::new("umap_forward");
    let mut params = Vec::new();
    let x = g.input("x", Shape::new(&[spec.n, spec.input_dim], f));
    let mut h = x;
    let mut in_d = spec.input_dim;

    for (li, &hd) in spec.hidden.iter().enumerate() {
        let w_name = format!("umap_w{li}");
        let b_name = format!("umap_b{li}");
        let w = g.param(&w_name, Shape::new(&[in_d, hd], f));
        let b = g.param(&b_name, Shape::new(&[hd], f));
        params.push(ParamSlot {
            name: w_name,
            param: w,
            grad: None,
        });
        params.push(ParamSlot {
            name: b_name,
            param: b,
            grad: None,
        });
        let mm = g.mm(h, w);
        let lin = g.add(mm, b);
        h = g.relu(lin);
        in_d = hd;
    }

    let w = g.param("umap_w_out", Shape::new(&[in_d, spec.output_dim], f));
    let b = g.param("umap_b_out", Shape::new(&[spec.output_dim], f));
    params.push(ParamSlot {
        name: "umap_w_out".into(),
        param: w,
        grad: None,
    });
    params.push(ParamSlot {
        name: "umap_b_out".into(),
        param: b,
        grad: None,
    });
    let mm_out = g.mm(h, w);
    let embed = g.add(mm_out, b);
    g.set_outputs(vec![embed]);
    let embed_node = embed;

    let (g, remap) = legalize_broadcast::run_with_remap(g);
    let embed = remap[&embed_node];
    let params = params
        .into_iter()
        .map(|mut p| {
            p.param = remap[&p.param];
            p
        })
        .collect();
    (g, embed, params)
}
