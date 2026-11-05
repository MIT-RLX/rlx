// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reading and ablating inside any packaged model, through rlx-runtime taps.
//!
//! A package runs through rlx-runtime's `Model` API, which compiles and runs
//! its own graphs (a language model's decode step, with caches). A
//! [`rlx_runtime::taps::GraphTap`] sees every graph compiled in its scope: the
//! tap here applies ablation masks and appends the requested nodes as extra
//! outputs to each graph that has them, and collects their values after every
//! run. So `trace`, `node_stats`, `contrast`, `lens` and `ablate` work on any
//! package without knowing its architecture — the model is simply opened and
//! run inside the scope.
//!
//! A language model is read on texts (one run per decode step); another
//! package on named input tensors. Per item, a node's values are pooled over
//! the runs: `last` (the final step: the last token of a text) or `mean`.

use crate::graphops::{Unit, find_node, mask_channels};
use crate::space::WeightSpace;
use rlx_ir::{DType, Graph};
use rlx_runtime::taps::GraphTap;
use serde_json::Value;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

/// What a read is fed: texts (language models) or named f32 tensors.
#[derive(Debug, Clone)]
pub enum Item {
    Text(String),
    Tensors(Vec<(String, Vec<usize>, Vec<f32>)>),
}

struct ReadTap {
    rewrites: Vec<crate::space::Rewrite>,
    /// Read only graphs of this name (set when a node is addressed by
    /// `%id`, which is only meaningful in the graph it came from).
    only_graph: Option<String>,
    nodes: Vec<String>,
    masks: Vec<Unit>,
    /// One entry per run of a graph that had every requested node.
    captured: RefCell<Vec<Vec<Vec<f32>>>>,
}

/// The packaged function a compiled graph is: its name without the sizes
/// it was specialized at (`decode@cap=65,upper=64`) or the device a build
/// is for (`decode:metal`). Node ids survive both.
fn function_of(name: &str) -> &str {
    let name = name.split('@').next().unwrap_or(name);
    name.split(':').next().unwrap_or(name)
}

impl GraphTap for ReadTap {
    fn on_compile(&self, graph: Graph) -> (Graph, usize) {
        let read_here = self
            .only_graph
            .as_ref()
            .is_none_or(|n| n == function_of(&graph.name));
        let mut g = crate::space::apply_rewrites(graph, &self.rewrites);
        let mut by_node: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for u in &self.masks {
            by_node.entry(u.node.as_str()).or_default().push(u.index);
        }
        for (node, idx) in by_node {
            if let Some(id) = find_node(&g, node)
                && let Ok(m) = mask_channels(&g, id, &idx)
            {
                g = m;
            }
        }
        let ids: Vec<_> = self.nodes.iter().filter_map(|n| find_node(&g, n)).collect();
        if !read_here || self.nodes.is_empty() || ids.len() != self.nodes.len() {
            return (g, 0);
        }
        let mut outs = g.outputs.clone();
        outs.extend(&ids);
        g.set_outputs(outs);
        (g, ids.len())
    }

    fn on_outputs(&self, extra: Vec<(Vec<u8>, DType)>) {
        self.captured.borrow_mut().push(
            extra
                .iter()
                .map(|(b, d)| rlx_runtime::taps::to_f32(b, *d))
                .collect(),
        );
    }
}

/// Per item, per node: the pooled values (`last` step or `mean` over steps).
pub type Read = Vec<Vec<Vec<f32>>>;

impl WeightSpace {
    /// Run `items` through the package (with the overlay applied) and read
    /// `nodes`, with `masks` zeroed. Not budgeted (callers charge).
    pub fn read_package(
        &mut self,
        items: &[Item],
        nodes: &[String],
        masks: &[Unit],
        pool_mean: bool,
    ) -> Result<Read, String> {
        let path = self.model_path()?;
        let device = self.lm_device();
        let only_graph = if nodes.iter().any(|n| n.starts_with('%')) {
            Some(self.base_graph()?.name)
        } else {
            None
        };
        let tap = Rc::new(ReadTap {
            rewrites: self.state.rewrites.clone(),
            only_graph,
            nodes: nodes.to_vec(),
            masks: masks.to_vec(),
            captured: RefCell::new(Vec::new()),
        });
        let mut bounds = Vec::with_capacity(items.len());
        let t2 = tap.clone();
        let result: Result<(), String> = rlx_runtime::taps::with_tap(t2, || {
            let mut model = rlx_runtime::model::open_source(
                &path.display().to_string(),
                serde_json::json!({ "device": device }),
            )
            .map_err(|e| format!("opening the package: {e:#}"))?;
            for item in items {
                let start = tap.captured.borrow().len();
                match item {
                    Item::Text(t) => {
                        model
                            .generate_text(
                                t,
                                &serde_json::json!({ "max_tokens": 1, "temperature": 0.0 }),
                            )
                            .map_err(|e| format!("reading a text: {e:#}"))?;
                    }
                    Item::Tensors(ts) => {
                        let inputs: Vec<rlx_runtime::model::Tensor> = ts
                            .iter()
                            .map(|(n, shape, v)| {
                                rlx_runtime::model::Tensor::f32(n.clone(), shape.clone(), v)
                            })
                            .collect();
                        model
                            .run(&inputs, &serde_json::json!({}))
                            .map_err(|e| format!("running the package: {e:#}"))?;
                    }
                }
                bounds.push((start, tap.captured.borrow().len()));
            }
            Ok(())
        });
        result?;
        let captured = tap.captured.borrow();
        let mut out = Vec::with_capacity(items.len());
        for (a, b) in bounds {
            if a == b && !nodes.is_empty() {
                return Err(format!(
                    "{} {nodes:?} (see list_nodes / function)",
                    crate::tools::NO_NODES
                ));
            }
            let runs = &captured[a..b];
            let per_node: Vec<Vec<f32>> = (0..nodes.len())
                .map(|k| {
                    if pool_mean {
                        let w = runs[0][k].len();
                        let mut m = vec![0f32; w];
                        for r in runs {
                            for (x, v) in m.iter_mut().zip(&r[k]) {
                                *x += v / runs.len() as f32;
                            }
                        }
                        m
                    } else {
                        runs[runs.len() - 1][k].clone()
                    }
                })
                .collect();
            out.push(per_node);
        }
        Ok(out)
    }

    /// Read `nodes` on a task network for observation vectors (one run).
    pub fn read_task(
        &mut self,
        observations: &[Vec<f32>],
        nodes: &[String],
    ) -> Result<Read, String> {
        let d = self.spec().obs_dim;
        if observations.len() > crate::space::BATCH {
            return Err(format!("at most {} observations", crate::space::BATCH));
        }
        let mut obs = vec![0f32; crate::space::BATCH * d];
        for (i, o) in observations.iter().enumerate() {
            if o.len() != d {
                return Err(format!("observation has {} values, expected {d}", o.len()));
            }
            obs[i * d..(i + 1) * d].copy_from_slice(o);
        }
        let outs = self.activations(&obs, nodes)?;
        Ok((0..observations.len())
            .map(|i| {
                outs[1..]
                    .iter()
                    .map(|v| {
                        let w = v.len() / crate::space::BATCH;
                        v[i * w..(i + 1) * w].to_vec()
                    })
                    .collect()
            })
            .collect())
    }
}

/// Inputs as the tools accept them: `texts` for language-model packages,
/// `observations` for task networks, `tensors` for other packages.
pub fn parse_items(ws: &WeightSpace, args: &Value) -> Result<(Vec<Item>, Vec<Vec<f32>>), String> {
    if let Some(texts) = args.get("texts").and_then(Value::as_array) {
        let items = texts
            .iter()
            .map(|t| {
                t.as_str()
                    .map(|s| Item::Text(s.to_string()))
                    .ok_or("texts must be strings")
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((items, Vec::new()));
    }
    if let Some(sets) = args.get("tensors").and_then(Value::as_array) {
        let mut items = Vec::new();
        for set in sets {
            let obj = set
                .as_object()
                .ok_or("each tensors item is {name: {shape, values}}")?;
            let mut ts = Vec::new();
            for (name, t) in obj {
                let shape: Vec<usize> =
                    serde_json::from_value(t["shape"].clone()).map_err(|e| e.to_string())?;
                let values: Vec<f32> =
                    serde_json::from_value(t["values"].clone()).map_err(|e| e.to_string())?;
                ts.push((name.clone(), shape, values));
            }
            items.push(Item::Tensors(ts));
        }
        return Ok((items, Vec::new()));
    }
    if let Some(obs) = args.get("observations").and_then(Value::as_array) {
        let dense = obs
            .iter()
            .map(|o| {
                let o: crate::tools::Observation =
                    serde_json::from_value(o.clone()).map_err(|e| e.to_string())?;
                crate::tools::dense_observation(ws, &o)
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok((Vec::new(), dense));
    }
    Err("give `texts` (language models), `observations` (task networks) or `tensors`".into())
}

/// Rewrites (and masks) without reading anything: for running a package
/// with recorded structure changes.
pub(crate) fn rewrite_tap(
    rewrites: Vec<crate::space::Rewrite>,
    masks: Vec<Unit>,
) -> Rc<dyn GraphTap> {
    Rc::new(ReadTap {
        rewrites,
        only_graph: None,
        nodes: Vec::new(),
        masks,
        captured: RefCell::new(Vec::new()),
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_specialized_graph_is_its_function() {
        assert_eq!(super::function_of("decode@cap=65,upper=64"), "decode");
        assert_eq!(super::function_of("decode:metal"), "decode");
        assert_eq!(super::function_of("decode"), "decode");
    }
}
