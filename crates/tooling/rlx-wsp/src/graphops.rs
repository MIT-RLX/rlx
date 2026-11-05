// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model-independent operations on an rlx [`Graph`].
//!
//! The weight-space tools never assume an architecture. A **unit** is one
//! channel (last-axis index) of a named node: `node[i]`. Where a channel comes
//! from and goes to is read off the graph ([`channel_params`]): the weight
//! column and bias that produce it (through elementwise ops back to a
//! `MatMul` with a param right-hand side) and the weight rows that consume
//! it. Rewrites ([`insert_after`]) rebuild the graph in order with new nodes
//! spliced in, carrying every piece of graph metadata (outputs and their
//! names, variables, symbolic attributes) across with ids remapped — so they
//! apply to any graph, decode graphs with caches included.

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Node, NodeId, Op, Shape};
use std::collections::BTreeMap;

/// A channel of a named node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Unit {
    pub node: String,
    pub index: usize,
}

impl std::fmt::Display for Unit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}[{}]", self.node, self.index)
    }
}

/// Parse `node[i]`, or the MLP alias `l{k}.h{i}` (= `l{k}.h[i]`).
pub fn parse_unit(s: &str) -> Option<Unit> {
    if let Some((node, rest)) = s.split_once('[') {
        let index = rest.strip_suffix(']')?.parse().ok()?;
        return Some(Unit {
            node: node.to_string(),
            index,
        });
    }
    let (layer, index) = s.strip_prefix('l')?.split_once(".h")?;
    layer.parse::<usize>().ok()?;
    Some(Unit {
        node: format!("l{layer}.h"),
        index: index.parse().ok()?,
    })
}

/// The node named `name` (or `%id`).
pub fn find_node(g: &Graph, name: &str) -> Option<NodeId> {
    if let Some(id) = name.strip_prefix('%').and_then(|s| s.parse::<u32>().ok())
        && (id as usize) < g.nodes().len()
    {
        return Some(NodeId(id));
    }
    g.nodes()
        .iter()
        .find(|n| n.name.as_deref() == Some(name))
        .map(|n| n.id)
}

/// Channels (last-axis size) of a node.
pub fn width(g: &Graph, id: NodeId) -> Option<usize> {
    g.node(id).shape.dims().last().and_then(|d| match d {
        rlx_ir::Dim::Static(v) => Some(*v),
        _ => None,
    })
}

/// The param a node is (directly, or through dtype casts: a package's
/// bf16 weight widened to f32 is still that weight).
pub fn param_name(g: &Graph, id: NodeId) -> Option<String> {
    let n = g.node(id);
    match &n.op {
        Op::Param { name } => Some(name.clone()),
        Op::Cast { .. } if n.inputs.len() == 1 => param_name(g, n.inputs[0]),
        _ => None,
    }
}

/// Where a node's channels come from and go to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChannelParams {
    /// The `[fan_in, width]` weight whose column `i` produces channel `i`.
    pub weight: Option<String>,
    /// The `[width]` bias added to it.
    pub bias: Option<String>,
    /// `[width, fan_out]` weights whose row `i` reads channel `i`.
    pub readers: Vec<String>,
}

/// Elementwise ops a channel passes through unchanged in index.
fn channelwise(op: &Op) -> bool {
    matches!(op, Op::Activation(_) | Op::Binary(_))
}

/// Trace channel provenance through elementwise ops: back to the
/// `MatMul(x, W)` that produces the node (and an added param bias), forward
/// to `MatMul(node, W2)` consumers.
pub fn channel_params(g: &Graph, id: NodeId) -> ChannelParams {
    let mut out = ChannelParams::default();
    // Producer.
    let mut cur = id;
    for _ in 0..8 {
        let n = g.node(cur);
        match &n.op {
            Op::MatMul if n.inputs.len() == 2 => {
                out.weight = param_name(g, n.inputs[1]);
                break;
            }
            Op::Binary(BinaryOp::Add) if n.inputs.len() == 2 => {
                let (a, b) = (n.inputs[0], n.inputs[1]);
                if let Some(p) = param_name(g, b) {
                    out.bias.get_or_insert(p);
                    cur = a;
                } else if let Some(p) = param_name(g, a) {
                    out.bias.get_or_insert(p);
                    cur = b;
                } else {
                    break;
                }
            }
            op if channelwise(op) && !n.inputs.is_empty() => cur = n.inputs[0],
            _ => break,
        }
    }
    // Readers (directly, or through channelwise ops on the node).
    let mut frontier = vec![id];
    let mut seen = vec![id];
    while let Some(x) = frontier.pop() {
        for n in g.nodes() {
            if !n.inputs.contains(&x) {
                continue;
            }
            match &n.op {
                Op::MatMul if n.inputs.len() == 2 && n.inputs[0] == x => {
                    if let Some(p) = param_name(g, n.inputs[1])
                        && !out.readers.contains(&p)
                    {
                        out.readers.push(p);
                    }
                }
                op if channelwise(op) && !seen.contains(&n.id) && n.inputs[0] == x => {
                    seen.push(n.id);
                    frontier.push(n.id);
                }
                _ => {}
            }
        }
    }
    out
}

/// Rebuild `g` in order; right after `target`, `splice` appends nodes and
/// returns the node later consumers (and outputs) should use instead.
/// Metadata (outputs, output names, variables, symbolic attributes and
/// definitions) is carried over with ids remapped.
pub fn insert_after(
    g: &Graph,
    target: NodeId,
    splice: impl FnOnce(&mut Graph, NodeId, &[NodeId]) -> NodeId,
) -> Graph {
    let mut out = Graph::new(g.name.clone());
    let mut map: Vec<NodeId> = Vec::with_capacity(g.nodes().len());
    let mut real: Vec<NodeId> = Vec::with_capacity(g.nodes().len());
    let mut splice = Some(splice);
    for n in g.nodes() {
        let inputs: Vec<NodeId> = n.inputs.iter().map(|i| map[i.0 as usize]).collect();
        let id = out.add_node(n.op.clone(), inputs.clone(), n.shape.clone());
        *out.node_mut(id) = Node {
            id,
            inputs,
            ..n.clone()
        };
        real.push(id);
        if n.id == target {
            let f = splice.take().expect("one target");
            let r = f(&mut out, id, &map);
            map.push(r);
        } else {
            map.push(id);
        }
    }
    out.outputs = g.outputs.iter().map(|o| map[o.0 as usize]).collect();
    out.set_output_names(g.output_names().to_vec());
    // A variable is an input node: keep pointing at the node itself.
    out.set_vars(
        g.vars()
            .iter()
            .map(|(v, k)| (real[v.0 as usize], *k))
            .collect(),
    );
    out.set_sym_attrs(
        g.sym_attrs()
            .iter()
            .map(|a| {
                let mut a = a.clone();
                a.node = real[a.node.0 as usize];
                a
            })
            .collect(),
    );
    out.set_sym_defs(g.sym_defs().to_vec());
    out
}

/// `g` with channels `indices` of node `target` multiplied by zero (an
/// ablation): everything downstream sees them as 0.
pub fn mask_channels(g: &Graph, target: NodeId, indices: &[usize]) -> Result<Graph, String> {
    let w = width(g, target).ok_or("the node has no static last axis")?;
    if let Some(i) = indices.iter().find(|&&i| i >= w) {
        return Err(format!("channel {i} out of range ({w})"));
    }
    let mut mask = vec![1f32; w];
    for &i in indices {
        mask[i] = 0.0;
    }
    let shape = g.node(target).shape.clone();
    let data: Vec<u8> = mask.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(insert_after(g, target, |g, x, _| {
        let m = g.add_node(Op::Constant { data }, vec![], Shape::new(&[w], DType::F32));
        g.add_node(Op::Binary(BinaryOp::Mul), vec![x, m], shape)
    }))
}

/// `g` with the constant `add` (one value per channel) added to node
/// `target` at every position.
pub fn add_channels(g: &Graph, target: NodeId, add: &[f32]) -> Result<Graph, String> {
    let w = width(g, target).ok_or("the node has no static last axis")?;
    if add.len() != w {
        return Err(format!("{} values for a node of {w} channels", add.len()));
    }
    let shape = g.node(target).shape.clone();
    let data: Vec<u8> = add.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(insert_after(g, target, |g, x, _| {
        let c = g.add_node(Op::Constant { data }, vec![], Shape::new(&[w], DType::F32));
        g.add_node(Op::Binary(BinaryOp::Add), vec![x, c], shape)
    }))
}

/// The bias param `node` is exactly `x + bias` of (a `[width]` param), if
/// so: adding a constant to the node is then an edit of that param.
pub fn bias_of(g: &Graph, node: NodeId) -> Option<String> {
    let n = g.node(node);
    if !matches!(n.op, Op::Binary(BinaryOp::Add)) || n.inputs.len() != 2 {
        return None;
    }
    let w = width(g, node)?;
    n.inputs.iter().find_map(|&i| {
        let p = param_name(g, i)?;
        (g.node(i).shape.dims().len() == 1 && width(g, i) == Some(w)).then_some(p)
    })
}

/// `MatMul` nodes whose right-hand side is param `param`, with their
/// left-hand input (the keys a row of the param reads).
pub fn matmuls_reading(g: &Graph, param: &str) -> Vec<(NodeId, NodeId)> {
    g.nodes()
        .iter()
        .filter(|n| {
            matches!(n.op, Op::MatMul)
                && n.inputs.len() == 2
                && param_name(g, n.inputs[1]).as_deref() == Some(param)
        })
        .map(|n| (n.id, n.inputs[0]))
        .collect()
}

/// A node's name for reads: its own, else `%id`.
pub fn node_ref(g: &Graph, id: NodeId) -> String {
    g.node(id)
        .name
        .clone()
        .unwrap_or_else(|| format!("%{}", id.0))
}

/// `g` with a square linear map `x @ W` (param `param`, `[w, w]`) after node
/// `target`; returns the graph and the new node's name. Initialised to the
/// identity by the caller, it leaves the function unchanged.
pub fn insert_linear(g: &Graph, target: NodeId, param: &str) -> Result<(Graph, String), String> {
    let w = width(g, target).ok_or("the node has no static last axis")?;
    let shape = g.node(target).shape.clone();
    let name = format!("{param}.out");
    let label = name.clone();
    let p = param.to_string();
    let out = insert_after(g, target, |g, x, _| {
        let wp = g.param(p, Shape::new(&[w, w], DType::F32));
        let y = g.matmul(x, wp, shape);
        g.node_mut(y).name = Some(label);
        y
    });
    Ok((out, name))
}

/// The nodes of the span from `from` (its input, exclusive) to `to` (its
/// output, inclusive): those on a path between them.
pub fn span(g: &Graph, from: NodeId, to: NodeId) -> Result<Vec<NodeId>, String> {
    if from.0 >= to.0 {
        return Err("`from` must come before `to`".into());
    }
    let n = g.nodes().len();
    let mut fwd = vec![false; n];
    fwd[from.0 as usize] = true;
    for node in g.nodes() {
        if node.inputs.iter().any(|i| fwd[i.0 as usize]) {
            fwd[node.id.0 as usize] = true;
        }
    }
    if !fwd[to.0 as usize] {
        return Err("`to` does not depend on `from`".into());
    }
    let mut back = vec![false; n];
    back[to.0 as usize] = true;
    for node in g.nodes().iter().rev() {
        if back[node.id.0 as usize] {
            for i in &node.inputs {
                back[i.0 as usize] = true;
            }
        }
    }
    Ok((from.0 + 1..=to.0)
        .map(NodeId)
        .filter(|id| fwd[id.0 as usize] && back[id.0 as usize])
        .collect())
}

/// `g` with the span `from → to` run `times` times in a row: copy `k`
/// reads copy `k-1`'s output where the original read `from`, and the
/// original's consumers of `to` read the last copy. Nodes the span reads
/// from outside (weights, positions, masks) are shared, except params when
/// `clone_params` is set: copy `k` then reads `{param}.rep{k}`. Returns the
/// graph and `(new param, original)` pairs to initialise. Refuses spans that
/// read or write a variable (a cache): copies would share it.
pub fn repeat_span(
    g: &Graph,
    from: NodeId,
    to: NodeId,
    times: usize,
    clone_params: bool,
) -> Result<(Graph, Vec<(String, String)>), String> {
    if times < 2 {
        return Err("`times` must be at least 2 (1 is the graph as it is)".into());
    }
    if g.node(from).shape != g.node(to).shape {
        return Err(format!(
            "the span's input and output shapes differ ({:?} vs {:?}); only shape-preserving spans repeat",
            g.node(from).shape.dims(),
            g.node(to).shape.dims()
        ));
    }
    let body = span(g, from, to)?;
    let var_inputs: Vec<NodeId> = g.vars().iter().map(|(v, _)| *v).collect();
    let var_outputs: Vec<NodeId> = g.vars().iter().map(|(_, k)| g.outputs[*k]).collect();
    for id in &body {
        let n = g.node(*id);
        if n.inputs.iter().any(|i| var_inputs.contains(i)) || var_outputs.contains(id) {
            return Err(format!(
                "the span touches a cache / state variable (at {}); copies would share it",
                n.name.clone().unwrap_or_else(|| format!("%{}", id.0))
            ));
        }
    }
    let in_body = |id: NodeId| body.binary_search(&id).is_ok();
    let mut new_params: Vec<(String, String)> = Vec::new();
    let mut added_params: BTreeMap<String, NodeId> = BTreeMap::new();
    let out = insert_after(g, to, |out, last, map| {
        let mut prev = last;
        for k in 1..times {
            let mut copy: BTreeMap<NodeId, NodeId> = BTreeMap::new();
            for &id in &body {
                let n = g.node(id);
                let inputs: Vec<NodeId> = n
                    .inputs
                    .iter()
                    .map(|i| {
                        if *i == from {
                            prev
                        } else if in_body(*i) {
                            copy[i]
                        } else if clone_params && let Op::Param { name } = &g.node(*i).op {
                            let nm = format!("{name}.rep{k}");
                            *added_params.entry(nm.clone()).or_insert_with(|| {
                                new_params.push((nm.clone(), name.clone()));
                                out.param(nm, g.node(*i).shape.clone())
                            })
                        } else {
                            map[i.0 as usize]
                        }
                    })
                    .collect();
                let nid = out.add_node(n.op.clone(), inputs.clone(), n.shape.clone());
                *out.node_mut(nid) = Node {
                    id: nid,
                    inputs,
                    name: n.name.as_ref().map(|s| format!("{s}.rep{k}")),
                    ..n.clone()
                };
                copy.insert(id, nid);
            }
            prev = copy[&to];
        }
        prev
    });
    Ok((out, new_params))
}

/// Named nodes whose channels count as units by default: activations.
pub fn unit_nodes(g: &Graph) -> Vec<String> {
    g.nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Activation(_)))
        .filter_map(|n| n.name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlp::{Act, MlpSpec};

    fn mlp() -> Graph {
        MlpSpec {
            widths: vec![5, 4, 3, 2],
            act: Act::Tanh,
        }
        .build_graph(2)
    }

    #[test]
    fn units_parse_and_resolve() {
        assert_eq!(
            parse_unit("l1.h2"),
            Some(Unit {
                node: "l1.h".into(),
                index: 2
            })
        );
        assert_eq!(
            parse_unit("blk3.attn[7]"),
            Some(Unit {
                node: "blk3.attn".into(),
                index: 7
            })
        );
        assert_eq!(parse_unit("logits"), None);
        let g = mlp();
        assert_eq!(width(&g, find_node(&g, "l0.h").unwrap()), Some(4));
    }

    #[test]
    fn channel_params_follow_the_graph() {
        let g = mlp();
        let c = channel_params(&g, find_node(&g, "l0.h").unwrap());
        assert_eq!(c.weight.as_deref(), Some("l0.w"));
        assert_eq!(c.bias.as_deref(), Some("l0.b"));
        assert_eq!(c.readers, vec!["l1.w".to_string()]);
        let c = channel_params(&g, find_node(&g, "logits").unwrap());
        assert_eq!(
            (c.weight.as_deref(), c.bias.as_deref()),
            (Some("l2.w"), Some("l2.b"))
        );
        assert!(c.readers.is_empty());
    }

    #[test]
    fn spans_repeat_with_shared_or_cloned_weights() {
        let g = MlpSpec {
            widths: vec![5, 4, 4, 4, 2],
            act: Act::Tanh,
        }
        .build_graph(2);
        let (from, to) = (
            find_node(&g, "l0.h").unwrap(),
            find_node(&g, "l2.h").unwrap(),
        );
        // Layers 1 and 2 (two layers) as one span, three times.
        let body = span(&g, from, to).unwrap();
        let (r, added) = repeat_span(&g, from, to, 3, false).unwrap();
        assert!(added.is_empty());
        assert_eq!(r.nodes().len(), g.nodes().len() + 2 * body.len());
        for n in r.nodes() {
            assert!(n.inputs.iter().all(|i| i.0 < n.id.0));
        }
        assert!(find_node(&r, "l2.h.rep2").is_some());
        // The output layer now reads the last copy.
        let c = channel_params(&r, find_node(&r, "l2.h.rep2").unwrap());
        assert_eq!(c.readers, vec!["l3.w".to_string()]);
        let (_, added) = repeat_span(&g, from, to, 2, true).unwrap();
        let names: Vec<&str> = added.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["l1.w.rep1", "l1.b.rep1", "l2.w.rep1", "l2.b.rep1"]);
        // Shape-changing spans are refused.
        let x = find_node(&g, "obs").unwrap();
        assert!(repeat_span(&g, x, from, 2, false).is_err());
    }

    #[test]
    fn rewrites_keep_order_and_outputs() {
        let g = mlp();
        let t = find_node(&g, "l0.h").unwrap();
        let m = mask_channels(&g, t, &[1]).unwrap();
        assert_eq!(m.nodes().len(), g.nodes().len() + 2);
        assert_eq!(m.outputs.len(), g.outputs.len());
        // Every input precedes its consumer.
        for n in m.nodes() {
            assert!(n.inputs.iter().all(|i| i.0 < n.id.0));
        }
        let (l, name) = insert_linear(&g, t, "l0.h.ins.w").unwrap();
        assert!(find_node(&l, &name).is_some());
        assert!(l.param_id("l0.h.ins.w").is_some());
        // The next layer now reads the inserted map.
        let c = channel_params(&l, find_node(&l, &name).unwrap());
        assert_eq!(c.weight.as_deref(), Some("l0.h.ins.w"));
        assert_eq!(c.readers, vec!["l1.w".to_string()]);
    }
}
