// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Combining and splitting graphs — [`rlx_ir::compose`] for every binding.
//!
//! `splice` grows a graph in place; `parallel`, `sequential` and `extract`
//! return a new graph; a [`Partition`] holds the parts of a split graph.
//! Graphs are borrowed, never consumed, so the same graph can be spliced or
//! split as often as needed.
//!
//! ```js
//! const g = new rlx.Graph("stack");
//! let [h] = rlx.splice(g, layer, { prefix: "l0." });
//! [h] = rlx.splice(g, layer, { prefix: "l1.", bind: { x: h } });
//! g.setOutputs([h]);
//!
//! const p = rlx.Partition.splitAt(g, [h0]);   // run p.graph(0), then p.graph(1)
//! ```

use rlx_ir::compose::{self, ComposeError, Splice};
use rlx_ir::{Graph, NodeId};
use serde_json::Value;

use crate::kinds::{Arg, node, nodes};
use crate::runtime::opts::Opts;
use crate::{ApiError, ApiResult};

fn api(e: ComposeError) -> ApiError {
    match e {
        ComposeError::UnknownNode(_)
        | ComposeError::NoParts
        | ComposeError::PartOutOfRange { .. } => ApiError::range(e.to_string()),
        ComposeError::UnknownLeaf { .. }
        | ComposeError::DuplicateBinding { .. }
        | ComposeError::ShapeMismatch { .. }
        | ComposeError::LeafKindClash { .. }
        | ComposeError::ArityMismatch { .. } => ApiError::type_(e.to_string()),
        _ => ApiError::runtime(e.to_string()),
    }
}

/// Copy `source` into `target` and return `source`'s outputs as `target`
/// node ids. `options`:
///
/// * `bind`: `{leaf name: node}` — feed an input (or param) of `source` from
///   a node of `target` instead of a leaf;
/// * `prefix`: prepended to `source`'s unbound leaf names (`"enc."`), so
///   they don't share with same-named leaves of `target`;
/// * `adapt_shapes`: accept bound nodes of another shape and re-infer.
///
/// On error `target` is unchanged.
pub fn splice(target: &mut Graph, source: &Graph, options: &Value) -> ApiResult<Vec<u32>> {
    let opts = Opts::new("splice", options)?;
    let mut s = Splice::new();
    if let Some(bind) = opts.get("bind") {
        let Value::Object(map) = bind else {
            return Err(ApiError::type_(
                "splice option 'bind': expected an object {leaf name: node}",
            ));
        };
        for (name, id) in map {
            let id = id
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| {
                    ApiError::type_(format!("splice bind '{name}': expected a node id"))
                })?;
            s = s.bind(name.clone(), node(target, id, Arg("splice", "bind"))?);
        }
    }
    if let Some(prefix) = opts.str("prefix")? {
        s = s.prefix(prefix);
    }
    if opts.bool("adapt_shapes", false)? {
        s = s.adapt_shapes();
    }
    opts.finish()?;
    let spliced = target.splice(source, &s).map_err(api)?;
    Ok(spliced.outputs.iter().map(|id| id.0).collect())
}

/// `a` and `b` side by side: `a`'s outputs, then `b`'s. Same-named leaves
/// are shared. Chain calls for more graphs.
pub fn parallel(a: &Graph, b: &Graph, name: Option<&str>) -> ApiResult<Graph> {
    compose::parallel(name.unwrap_or("parallel"), &[a, b]).map_err(api)
}

/// `b` after `a`: `b`'s inputs, in creation order, are fed by `a`'s outputs.
/// Same-named params are shared. Chain calls for more graphs.
pub fn sequential(a: &Graph, b: &Graph, name: Option<&str>) -> ApiResult<Graph> {
    compose::sequential(name.unwrap_or("sequential"), &[a, b]).map_err(api)
}

/// The subgraph computing `outputs`, with them as its outputs.
pub fn extract(graph: &Graph, outputs: Vec<u32>) -> ApiResult<Graph> {
    let ids = nodes(graph, outputs, Arg("extract", "outputs"))?;
    graph.extract(&ids).map_err(api)
}

/// A graph split into parts that run in order. Part `i` reads the values
/// named [`imports`](Self::imports) — exported, under the same name, by an
/// earlier part — and its outputs are named [`exports`](Self::exports).
#[derive(Debug, Clone)]
pub struct Partition(pub compose::Partition);

impl Partition {
    /// Split `graph`, node `i` going to part `assign[i]` (one entry per
    /// node, `node_count()` long; entries of inputs, params and constants
    /// are ignored — those are copied into every part that reads them).
    /// Every edge must run to the same or a later part.
    pub fn new(graph: &Graph, assign: &[u32]) -> ApiResult<Self> {
        if assign.len() != graph.len() {
            return Err(ApiError::range(format!(
                "Partition assign: {} entries for a graph of {} nodes (one part per node id)",
                assign.len(),
                graph.len()
            )));
        }
        let parts = assign
            .iter()
            .zip(graph.nodes())
            .filter(|(_, n)| !n.op.is_leaf())
            .map(|(&p, _)| p as usize + 1)
            .max()
            .unwrap_or(1);
        graph
            .partition(parts, |id: NodeId| assign[id.0 as usize] as usize)
            .map(Self)
            .map_err(api)
    }

    /// Two parts: `cut` and everything it reads, then the rest.
    pub fn split_at(graph: &Graph, cut: Vec<u32>) -> ApiResult<Self> {
        let ids = nodes(graph, cut, Arg("Partition.split_at", "cut"))?;
        graph.split_at(&ids).map(Self).map_err(api)
    }

    /// One part per independent subgraph; the parts import nothing.
    pub fn components(graph: &Graph) -> ApiResult<Self> {
        graph.components().map(Self).map_err(api)
    }

    /// The number of parts.
    pub fn len(&self) -> usize {
        self.0.parts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.parts.is_empty()
    }

    fn part(&self, i: usize, what: &str) -> ApiResult<&compose::Part> {
        self.0.parts.get(i).ok_or_else(|| {
            ApiError::range(format!(
                "Partition.{what}: part {i} of {} parts",
                self.0.parts.len()
            ))
        })
    }

    /// A copy of part `i`'s graph.
    pub fn graph(&self, i: usize) -> ApiResult<Graph> {
        Ok(self.part(i, "graph")?.graph.clone())
    }

    /// The boundary input names part `i` reads.
    pub fn imports(&self, i: usize) -> ApiResult<Vec<String>> {
        Ok(self.part(i, "imports")?.imports.clone())
    }

    /// The name of each of part `i`'s outputs.
    pub fn exports(&self, i: usize) -> ApiResult<Vec<String>> {
        Ok(self.part(i, "exports")?.exports.clone())
    }

    /// Where each source output went: `[[part, output index], …]`.
    pub fn outputs(&self) -> Value {
        Value::from(
            self.0
                .outputs
                .iter()
                .map(|&(p, j)| Value::from(vec![p, j]))
                .collect::<Vec<_>>(),
        )
    }

    /// The parts as one graph again.
    pub fn merge(&self) -> ApiResult<Graph> {
        self.0.merge().map_err(api)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{DType, GraphExt, Shape};
    use serde_json::json;

    fn layer() -> Graph {
        let s = Shape::new(&[2, 4], DType::F32);
        let mut g = Graph::new("layer");
        let x = g.input("x", s.clone());
        let w = g.param("w", Shape::new(&[4, 4], DType::F32));
        let y = g.matmul(x, w, s);
        let y = g.tanh(y);
        g.set_outputs(vec![y]);
        g
    }

    #[test]
    fn splice_reads_options_and_rejects_unknown_ones() {
        let mut g = Graph::new("t");
        let h = splice(&mut g, &layer(), &json!({"prefix": "l0."})).unwrap();
        let h2 = splice(
            &mut g,
            &layer(),
            &json!({"prefix": "l1.", "bind": {"x": h[0]}}),
        )
        .unwrap();
        assert!(g.param_id("l0.w").is_some() && g.param_id("l1.w").is_some());
        assert!(g.input_id("l1.x").is_none());
        assert_ne!(h, h2);

        let err = splice(&mut g, &layer(), &json!({"prefx": "a."})).unwrap_err();
        assert!(err.message.contains("prefx"), "{}", err.message);
        let err = splice(&mut g, &layer(), &json!({"bind": {"y": 0}})).unwrap_err();
        assert!(err.message.contains("\"y\""), "{}", err.message);
        let err = splice(&mut g, &layer(), &json!({"bind": {"x": 999}})).unwrap_err();
        assert!(err.message.contains("999"), "{}", err.message);
    }

    #[test]
    fn partition_by_node_list_round_trips() {
        let g = sequential(&layer(), &layer(), None).unwrap();
        let n = g.len();
        let assign: Vec<u32> = (0..n as u32)
            .map(|i| u32::from(i >= n as u32 / 2))
            .collect();
        let p = Partition::new(&g, &assign).unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p.imports(1).unwrap(), p.exports(0).unwrap());
        assert_eq!(p.merge().unwrap().len(), g.len());
        assert!(Partition::new(&g, &assign[1..]).is_err());
        assert!(p.graph(2).is_err());
        assert_eq!(p.outputs(), json!([[1, 0]]));
    }
}
