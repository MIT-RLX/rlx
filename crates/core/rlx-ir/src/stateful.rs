// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode state inside the graph: a step that takes the token and its
//! `position` and returns caches the caller feeds straight back.
//!
//! A bucketed decode graph leaves a protocol to its host: RoPE rows from a
//! table at the position, a mask that admits the cached rows and a spare one
//! at `upper`, a new K/V row written at `upper` that the host then moves to
//! `past`. [`stateful_decode`] moves all of it into the graph:
//!
//! * each table input becomes `Gather(table, position)`, the table held as a
//!   param;
//! * the mask becomes `iota[0..cap] <= position`;
//! * each `KvAppend` becomes a `DynamicUpdateSlice` writing the row in place
//!   at `position`,
//!   so a cache output is the next step's cache input, as is; each such
//!   cache is declared a variable ([`Graph::vars`]).
//!
//! A step of several tokens (`tokens`, a speculative decoder's check of its
//! draft) appends its rows by `concat(cache, rows)`: the cache input takes
//! the concat's length and the concat becomes a `DynamicUpdateSlice` of all
//! the rows at `position`. Its tables give the rows `position..position +
//! tokens`, and its mask is a `[batch, heads, tokens, keys]` bias that lets
//! token `i` see keys `0..=position + i`.

use std::collections::HashMap;

use crate::dynamic::SymAttr;
use crate::op::CmpOp;
use crate::shape::{Dim, Shape};
use crate::{DType, Graph, Node, NodeId, Op};

/// Rebuild `graph` node by node. For each node, `f` gets the graph being
/// built, its symbolic attributes, the node, and its inputs' new ids; it
/// returns `None` to keep the node as is, or the id of what stands for it
/// after adding whatever nodes that takes. Kept nodes keep their symbolic
/// attributes; a replaced node's are dropped (its replacement declares its
/// own). Outputs follow.
pub fn rebuild(
    graph: &Graph,
    mut f: impl FnMut(&mut Graph, &mut Vec<SymAttr>, &Node, &[NodeId]) -> Result<Option<NodeId>, String>,
) -> Result<Graph, String> {
    let mut out = Graph::new(&graph.name);
    let mut attrs = Vec::new();
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|i| {
                map.get(i)
                    .copied()
                    .ok_or_else(|| format!("{} reads {i}, which comes after it", node.id))
            })
            .collect::<Result<_, _>>()?;
        let new = match f(&mut out, &mut attrs, node, &inputs)? {
            Some(id) => id,
            None => {
                let id = out.push_ext(
                    node.op.clone(),
                    inputs,
                    node.shape.clone(),
                    node.name.clone(),
                    node.origin.clone(),
                );
                for a in graph.sym_attrs().iter().filter(|a| a.node == node.id) {
                    attrs.push(SymAttr {
                        node: id,
                        ..a.clone()
                    });
                }
                id
            }
        };
        map.insert(node.id, new);
    }
    let outputs = graph.outputs.iter().map(|o| map[o]).collect();
    out.set_outputs(outputs);
    out.set_sym_attrs(attrs);
    out.set_vars(graph.vars().iter().map(|(i, o)| (map[i], *o)).collect());
    out.set_sym_defs(graph.sym_defs().to_vec());
    Ok(out)
}

/// What [`stateful_decode`] rewrites.
#[derive(Debug, Clone)]
pub struct StatefulDecode {
    /// The input to add: the token's position, `f32[1]` (the number of
    /// tokens before it).
    pub position: String,
    /// Inputs that are one row of a table at the position (RoPE `cos` /
    /// `sin`): `(input, table param, table rows)`. The table becomes a param
    /// of `[rows, width]`.
    pub tables: Vec<(String, String, usize)>,
    /// The attention mask input, `[batch, cap]`, and the iota param it is
    /// computed from: `(input, iota param, iota rows)`.
    pub mask: Option<(String, String, usize)>,
    /// The symbol of the cache length (`cap`).
    pub cache_symbol: u32,
    /// Tokens a step takes: 1, or a check's several (see the module docs).
    pub tokens: usize,
    /// An input to add, `f32[batch]`: the row each sequence starts at. Its
    /// mask then admits only `starts[b] ..= position`, so a sequence can
    /// join a batch mid-way (continuous batching): RoPE rotates by the
    /// difference of positions, so starting at row `s` computes what
    /// starting at 0 does.
    pub starts: Option<String>,
    /// An input to add, `f32[1]`: added to `position` where the tables are
    /// read, not where the caches are written or masked. A prompt whose
    /// positions are not its rows (multimodal RoPE: an image's tokens share
    /// positions) goes on from where its positions left off.
    pub rope_offset: Option<String>,
    /// Further mask inputs, each the step's mask (from `mask`'s iota) that
    /// also admits only the last `window` positions, `position - window <
    /// key`: a sliding-window layer's (Gemma, Mistral).
    pub windows: Vec<(String, usize)>,
}

/// `x` (`[1, n]`) as `shape` (`[batch, n]`): itself at batch 1, else
/// broadcast over the batch (a symbolic batch's target is re-derived when
/// the graph is specialized).
fn repeat_rows(g: &mut Graph, x: NodeId, batch: Dim, shape: &Shape) -> NodeId {
    if batch == Dim::Static(1) {
        return x;
    }
    let target = shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n as i64,
            Dim::Dynamic(_) => -1,
        })
        .collect();
    g.push_ext(
        Op::Expand {
            target_shape: target,
        },
        vec![x],
        shape.clone(),
        None,
        None,
    )
}

/// The decode step `graph` with its state inside: see the module docs.
/// `graph` is symbolic in the cache length (`spec.cache_symbol`), and may be
/// in its batch: every sequence is at the same position. An error names
/// what does not fit: a table input that is not `[batch, width]`, a
/// `KvAppend` on an axis other than 1, …
pub fn stateful_decode(graph: &Graph, spec: &StatefulDecode) -> Result<Graph, String> {
    let f32s = |d: &[Dim]| Shape::from_dims(d, DType::F32);
    let mut position: Option<NodeId> = None;
    let mut position_of = |g: &mut Graph| -> NodeId {
        *position.get_or_insert_with(|| {
            g.push_ext(
                Op::Input {
                    name: spec.position.clone(),
                },
                vec![],
                Shape::new(&[1], DType::F32),
                Some(spec.position.clone()),
                None,
            )
        })
    };
    // One node per param name: two nodes of one name would each need
    // binding, and a backend binds a name once.
    let mut params_made: HashMap<String, NodeId> = HashMap::new();
    let mut param_of = |g: &mut Graph, name: &String, shape: Shape| -> NodeId {
        *params_made.entry(name.clone()).or_insert_with(|| {
            g.push_ext(
                Op::Param { name: name.clone() },
                vec![],
                shape,
                Some(name.clone()),
                None,
            )
        })
    };
    let mut offset: Option<NodeId> = None;
    // The position the tables are read at: `position + rope_offset`.
    let mut rope_position = |g: &mut Graph, pos: NodeId| -> NodeId {
        let Some(name) = &spec.rope_offset else {
            return pos;
        };
        let off = *offset.get_or_insert_with(|| {
            g.push_ext(
                Op::Input { name: name.clone() },
                vec![],
                Shape::new(&[1], DType::F32),
                Some(name.clone()),
                None,
            )
        });
        g.push_ext(
            Op::Binary(crate::op::BinaryOp::Add),
            vec![pos, off],
            Shape::new(&[1], DType::F32),
            None,
            None,
        )
    };
    let mut kv_appends = 0usize;
    // Caches grown by `concat(cache, rows)` along axis 1 and returned: the
    // cache input takes the concat's shape.
    let concat_caches: HashMap<NodeId, Shape> = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Concat { axis: 1 }) && graph.outputs.contains(&n.id))
        .filter_map(|n| {
            let cache = graph.node(*n.inputs.first()?);
            let grows = matches!(cache.op, Op::Input { .. })
                && matches!(cache.shape.dims().get(1), Some(Dim::Dynamic(_)));
            grows.then(|| (cache.id, n.shape.clone()))
        })
        .collect();
    // `position + [0, 1, …, tokens - 1]`, `f32[tokens]`.
    let tokens = spec.tokens.max(1);
    let positions_of = |g: &mut Graph, pos: NodeId| -> NodeId {
        let offsets: Vec<u8> = (0..tokens).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let offsets = g.push_ext(
            Op::Constant { data: offsets },
            vec![],
            Shape::new(&[tokens], DType::F32),
            None,
            None,
        );
        g.push_ext(
            Op::Binary(crate::op::BinaryOp::Add),
            vec![offsets, pos],
            Shape::new(&[tokens], DType::F32),
            None,
            None,
        )
    };
    let g = rebuild(graph, |g, attrs, node, inputs| {
        let dims = node.shape.dims().to_vec();
        match &node.op {
            Op::Input { name } if concat_caches.contains_key(&node.id) => Ok(Some(g.push_ext(
                node.op.clone(),
                vec![],
                concat_caches[&node.id].clone(),
                Some(name.clone()),
                None,
            ))),
            Op::Concat { axis: 1 }
                if inputs.len() == 2 && concat_caches.contains_key(&node.inputs[0]) =>
            {
                kv_appends += 1;
                let pos = position_of(g);
                let next_name = match &graph.node(node.inputs[0]).op {
                    Op::Input { name } => Some(format!("{name}.next")),
                    _ => node.name.clone(),
                };
                Ok(Some(g.push_ext(
                    Op::DynamicUpdateSlice { axis: 1 },
                    vec![inputs[0], inputs[1], pos],
                    node.shape.clone(),
                    next_name,
                    node.origin.clone(),
                )))
            }
            Op::Input { name } => {
                if let Some((_, table, rows)) = spec.tables.iter().find(|(i, _, _)| i == name)
                    && tokens > 1
                {
                    // `[tokens, width]`: the rows at `position..position+tokens`.
                    let [Dim::Static(t), Dim::Static(width)] = dims[..] else {
                        return Err(format!(
                            "table input '{name}' is {}, not [tokens, width]",
                            node.shape
                        ));
                    };
                    if t != tokens {
                        return Err(format!(
                            "table input '{name}' has {t} rows for {tokens} tokens"
                        ));
                    }
                    let pos = position_of(g);
                    let pos = rope_position(g, pos);
                    let at = positions_of(g, pos);
                    let t = param_of(g, table, Shape::new(&[*rows, width], DType::F32));
                    return Ok(Some(g.push_ext(
                        Op::Gather { axis: 0 },
                        vec![t, at],
                        node.shape.clone(),
                        Some(name.clone()),
                        None,
                    )));
                }
                if let Some((_, table, rows)) = spec.tables.iter().find(|(i, _, _)| i == name) {
                    // `[batch, width]`: every sequence of a batched step is
                    // at the same position, so one row, repeated.
                    let [batch, Dim::Static(width)] = dims[..] else {
                        return Err(format!(
                            "table input '{name}' is {}, not [batch, width]",
                            node.shape
                        ));
                    };
                    let pos = position_of(g);
                    let pos = rope_position(g, pos);
                    let t = param_of(g, table, Shape::new(&[*rows, width], DType::F32));
                    let row = g.push_ext(
                        Op::Gather { axis: 0 },
                        vec![t, pos],
                        Shape::new(&[1, width], DType::F32),
                        Some(name.clone()),
                        None,
                    );
                    return Ok(Some(repeat_rows(g, row, batch, &node.shape)));
                }
                if let Some((_, iota, rows)) = spec.mask.as_ref().filter(|(i, _, _)| i == name)
                    && let [_, _, Dim::Static(q), keys @ Dim::Dynamic(keys_sym)] = dims[..]
                {
                    // A check's bias: token `i` sees keys `0..=position + i`.
                    let pos = position_of(g);
                    let at = positions_of(g, pos);
                    let iota = param_of(g, iota, Shape::new(&[*rows], DType::F32));
                    let head = g.push_ext(
                        Op::Narrow {
                            axis: 0,
                            start: 0,
                            len: 0,
                        },
                        vec![iota],
                        f32s(&[keys]),
                        None,
                        None,
                    );
                    attrs.push(SymAttr {
                        node: head,
                        path: vec!["len".into()],
                        sym: keys_sym,
                        scale: 1,
                        offset: 0,
                    });
                    let grid = f32s(&[Dim::Static(q), keys]);
                    let expand = |g: &mut Graph, x: NodeId, shape: Vec<Dim>| {
                        let s = f32s(&shape);
                        let r = g.push_ext(
                            Op::Reshape {
                                new_shape: shape
                                    .iter()
                                    .map(|d| match d {
                                        Dim::Static(n) => *n as i64,
                                        Dim::Dynamic(_) => -1,
                                    })
                                    .collect(),
                            },
                            vec![x],
                            s,
                            None,
                            None,
                        );
                        g.push_ext(
                            Op::Expand {
                                target_shape: vec![q as i64, -1],
                            },
                            vec![r],
                            grid.clone(),
                            None,
                            None,
                        )
                    };
                    let keys_grid = expand(g, head, vec![Dim::Static(1), keys]);
                    let tok_grid = expand(g, at, vec![Dim::Static(q), Dim::Static(1)]);
                    let seen = g.push_ext(
                        Op::Compare(CmpOp::Le),
                        vec![keys_grid, tok_grid],
                        Shape::from_dims(&[Dim::Static(q), keys], DType::Bool),
                        None,
                        None,
                    );
                    let seen = g.push_ext(
                        Op::Cast { to: DType::F32 },
                        vec![seen],
                        grid.clone(),
                        None,
                        None,
                    );
                    // (seen - 1) · 1e9: 0 where seen, -1e9 where not.
                    let one = g.push_ext(
                        Op::Constant {
                            data: 1f32.to_le_bytes().to_vec(),
                        },
                        vec![],
                        Shape::new(&[1], DType::F32),
                        None,
                        None,
                    );
                    let big = g.push_ext(
                        Op::Constant {
                            data: 1e9f32.to_le_bytes().to_vec(),
                        },
                        vec![],
                        Shape::new(&[1], DType::F32),
                        None,
                        None,
                    );
                    let off = g.push_ext(
                        Op::Binary(crate::op::BinaryOp::Sub),
                        vec![seen, one],
                        grid.clone(),
                        None,
                        None,
                    );
                    let bias = g.push_ext(
                        Op::Binary(crate::op::BinaryOp::Mul),
                        vec![off, big],
                        grid.clone(),
                        None,
                        None,
                    );
                    let four = f32s(&[Dim::Static(1), Dim::Static(1), Dim::Static(q), keys]);
                    let bias = g.push_ext(
                        Op::Reshape {
                            new_shape: vec![1, 1, q as i64, -1],
                        },
                        vec![bias],
                        four,
                        None,
                        None,
                    );
                    let target = node
                        .shape
                        .dims()
                        .iter()
                        .map(|d| match d {
                            Dim::Static(n) => *n as i64,
                            Dim::Dynamic(_) => -1,
                        })
                        .collect();
                    return Ok(Some(g.push_ext(
                        Op::Expand {
                            target_shape: target,
                        },
                        vec![bias],
                        node.shape.clone(),
                        Some(name.clone()),
                        None,
                    )));
                }
                let window = spec
                    .windows
                    .iter()
                    .find(|(i, _)| i == name)
                    .map(|(_, w)| *w);
                if let Some((_, iota, rows)) = spec
                    .mask
                    .as_ref()
                    .filter(|(i, _, _)| i == name || window.is_some())
                {
                    let [batch, cap @ Dim::Dynamic(_)] = dims[..] else {
                        return Err(format!(
                            "mask input '{name}' is {}, not [batch, cap]",
                            node.shape
                        ));
                    };
                    let pos = position_of(g);
                    let iota = param_of(g, iota, Shape::new(&[*rows], DType::F32));
                    // iota[0..cap]: `len` is the cache length.
                    let head = g.push_ext(
                        Op::Narrow {
                            axis: 0,
                            start: 0,
                            len: 0,
                        },
                        vec![iota],
                        f32s(&[cap]),
                        None,
                        None,
                    );
                    attrs.push(SymAttr {
                        node: head,
                        path: vec!["len".into()],
                        sym: spec.cache_symbol,
                        scale: 1,
                        offset: 0,
                    });
                    let seen = g.push_ext(
                        Op::Compare(CmpOp::Le),
                        vec![head, pos],
                        Shape::from_dims(&[cap], DType::Bool),
                        None,
                        None,
                    );
                    let mut as_f32 = g.push_ext(
                        Op::Cast { to: DType::F32 },
                        vec![seen],
                        f32s(&[cap]),
                        None,
                        None,
                    );
                    if let Some(w) = window {
                        // `key + window > position`: within the window.
                        let w = g.push_ext(
                            Op::Constant {
                                data: (w as f32).to_le_bytes().to_vec(),
                            },
                            vec![],
                            Shape::new(&[1], DType::F32),
                            None,
                            None,
                        );
                        let ahead = g.push_ext(
                            Op::Binary(crate::op::BinaryOp::Add),
                            vec![head, w],
                            f32s(&[cap]),
                            None,
                            None,
                        );
                        let recent = g.push_ext(
                            Op::Compare(CmpOp::Gt),
                            vec![ahead, pos],
                            Shape::from_dims(&[cap], DType::Bool),
                            None,
                            None,
                        );
                        let recent = g.push_ext(
                            Op::Cast { to: DType::F32 },
                            vec![recent],
                            f32s(&[cap]),
                            None,
                            None,
                        );
                        as_f32 = g.push_ext(
                            Op::Binary(crate::op::BinaryOp::Mul),
                            vec![as_f32, recent],
                            f32s(&[cap]),
                            None,
                            None,
                        );
                    }
                    let mask = g.push_ext(
                        Op::Reshape {
                            new_shape: vec![1, -1],
                        },
                        vec![as_f32],
                        f32s(&[Dim::Static(1), cap]),
                        Some(name.clone()),
                        None,
                    );
                    let mask = repeat_rows(g, mask, batch, &node.shape);
                    let Some(starts) = &spec.starts else {
                        return Ok(Some(mask));
                    };
                    // Each sequence's keys from its own start.
                    let grid = node.shape.clone();
                    let starts = g.push_ext(
                        Op::Input {
                            name: starts.clone(),
                        },
                        vec![],
                        f32s(&[batch]),
                        Some(starts.clone()),
                        None,
                    );
                    let col = g.push_ext(
                        Op::Reshape {
                            new_shape: vec![-1, 1],
                        },
                        vec![starts],
                        f32s(&[batch, Dim::Static(1)]),
                        None,
                        None,
                    );
                    let col = g.push_ext(
                        Op::Expand {
                            target_shape: vec![-1, -1],
                        },
                        vec![col],
                        grid.clone(),
                        None,
                        None,
                    );
                    let row = g.push_ext(
                        Op::Reshape {
                            new_shape: vec![1, -1],
                        },
                        vec![head],
                        f32s(&[Dim::Static(1), cap]),
                        None,
                        None,
                    );
                    let row = g.push_ext(
                        Op::Expand {
                            target_shape: vec![-1, -1],
                        },
                        vec![row],
                        grid.clone(),
                        None,
                        None,
                    );
                    let after = g.push_ext(
                        Op::Compare(CmpOp::Ge),
                        vec![row, col],
                        Shape::from_dims(&[batch, cap], DType::Bool),
                        None,
                        None,
                    );
                    let after = g.push_ext(
                        Op::Cast { to: DType::F32 },
                        vec![after],
                        grid.clone(),
                        None,
                        None,
                    );
                    return Ok(Some(g.push_ext(
                        Op::Binary(crate::op::BinaryOp::Mul),
                        vec![mask, after],
                        grid,
                        Some(name.clone()),
                        None,
                    )));
                }
                Ok(None)
            }
            Op::KvAppend { axis, .. } => {
                let (cache, row) = (inputs[0], inputs[1]);
                let cache_dims = g.node(cache).shape.dims().to_vec();
                let [_, Dim::Dynamic(_), Dim::Static(_)] = cache_dims[..] else {
                    return Err(format!(
                        "{}: KvAppend on a cache of {}; a stateful step writes [batch, cap, width]",
                        node.id,
                        g.node(cache).shape
                    ));
                };
                if *axis != 1 {
                    return Err(format!("{}: KvAppend on axis {axis}, not 1", node.id));
                }
                kv_appends += 1;
                let pos = position_of(g);
                // `%past_k_0.next`: the cache's next value, by its name.
                let next_name = match &g.node(cache).op {
                    Op::Input { name } => Some(format!("{name}.next")),
                    _ => node.name.clone(),
                };
                // The row in place at `position`: one row a step, whatever
                // the cache's length.
                let written = g.push_ext(
                    Op::DynamicUpdateSlice { axis: 1 },
                    vec![cache, row, pos],
                    g.node(cache).shape.clone(),
                    next_name,
                    node.origin.clone(),
                );
                Ok(Some(written))
            }
            _ => Ok(None),
        }
    })?;
    if kv_appends == 0 {
        return Err("the graph has no KvAppend: no cache to keep".into());
    }
    // Each written cache that the step returns is a variable: its input
    // now, that output next.
    let mut g = g;
    let mut vars = g.vars().to_vec();
    for (index, out) in g.outputs.iter().enumerate() {
        let n = g.node(*out);
        if matches!(n.op, Op::DynamicUpdateSlice { .. })
            && let Some(&cache) = n.inputs.first()
            && matches!(g.node(cache).op, Op::Input { .. })
            && !vars.iter().any(|(i, _)| *i == cache)
        {
            vars.push((cache, index));
        }
    }
    g.set_vars(vars);
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuild_keeps_what_it_is_not_told_to_replace() {
        let s = Shape::new(&[2], DType::F32);
        let mut g = Graph::new("g");
        let x = g.input("x", s.clone());
        let y = g.add_node(
            Op::Activation(crate::op::Activation::Gelu),
            vec![x],
            s.clone(),
        );
        g.set_outputs(vec![y]);
        let same = rebuild(&g, |_, _, _, _| Ok(None)).unwrap();
        assert_eq!(format!("{same}"), format!("{g}"));
        // Replace the activation by two nodes; the output follows.
        let two = rebuild(&g, |out, _, node, inputs| {
            if !matches!(node.op, Op::Activation(_)) {
                return Ok(None);
            }
            let a = out.push_ext(
                Op::Activation(crate::op::Activation::Relu),
                inputs.to_vec(),
                s.clone(),
                None,
                None,
            );
            Ok(Some(out.push_ext(
                Op::Activation(crate::op::Activation::Gelu),
                vec![a],
                s.clone(),
                None,
                None,
            )))
        })
        .unwrap();
        assert_eq!(two.len(), 3);
        assert_eq!(two.outputs, [NodeId(2)]);
    }

    /// A decode step's shape: a table input, a `[1, cap]` mask, one cache
    /// with its `KvAppend` at `upper` (`cap - 1`).
    #[cfg(feature = "serialize")]
    fn toy_step() -> Graph {
        let f = |d: &[Dim]| Shape::from_dims(d, DType::F32);
        let (cap, s) = (Dim::Dynamic(0), |n| Dim::Static(n));
        let mut g = Graph::new("decode");
        let cos = g.input("rope_cos", f(&[s(1), s(2)]));
        let mask = g.input("mask", f(&[s(1), cap]));
        let past = g.input("past_k_0", f(&[s(1), cap, s(2)]));
        let row = g.add_node(
            Op::Reshape {
                new_shape: vec![1, 1, 2],
            },
            vec![cos],
            f(&[s(1), s(1), s(2)]),
        );
        let k = g.add_node(
            Op::KvAppend { axis: 1, pos: 0 },
            vec![past, row],
            f(&[s(1), cap, s(2)]),
        );
        g.set_outputs(vec![mask, k]);
        g.set_sym_attrs(vec![SymAttr {
            node: k,
            path: vec!["pos".into()],
            sym: 0,
            scale: 1,
            offset: -1,
        }]);
        g
    }

    // Writing the symbolic `len` back (in `specialize`) reads fields by name.
    #[cfg(feature = "serialize")]
    #[test]
    fn the_state_moves_into_the_graph() {
        let spec = StatefulDecode {
            position: "position".into(),
            tables: vec![("rope_cos".into(), "rope_cos.table".into(), 8)],
            mask: Some(("mask".into(), "decode.iota".into(), 8)),
            cache_symbol: 0,
            tokens: 1,
            starts: None,
            rope_offset: None,
            windows: Vec::new(),
        };
        let g = stateful_decode(&toy_step(), &spec).unwrap();
        let ops: Vec<String> = g
            .nodes()
            .iter()
            .map(|n| format!("{:?}", n.op.kind()))
            .collect();
        let has = |k: &str| ops.iter().any(|o| o == k);
        assert!(
            has("Gather") && has("DynamicUpdateSlice") && has("Compare"),
            "{ops:?}"
        );
        assert!(!has("KvAppend"), "{ops:?}");
        let inputs: Vec<&str> = g
            .nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Input { name } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(inputs, ["position", "past_k_0"]);
        // The cache is a variable: its input now, the written cache next.
        let past = g
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name } if name == "past_k_0"))
            .unwrap()
            .id;
        assert_eq!(g.vars(), [(past, 1)]);
        // The iota's `len` is the cache length; KvAppend's `pos` is gone.
        assert_eq!(g.sym_attrs().len(), 1);
        assert_eq!(g.sym_attrs()[0].path, ["len"]);
        // It specializes to a static graph at any cache length.
        let mut b = crate::DimBinding::new();
        b.set(0, 5);
        let at5 = crate::dynamic::specialize(&g, &b);
        assert!(!at5.is_symbolic());
        assert_eq!(
            at5.node(at5.outputs[1]).shape.dims(),
            [Dim::Static(1), Dim::Static(5), Dim::Static(2)]
        );
        // A cache it cannot write in place is named.
        let mut bad = toy_step();
        if let Op::KvAppend { axis, .. } = &mut bad.node_mut(NodeId(4)).op {
            *axis = 2;
        }
        let e = stateful_decode(&bad, &spec).unwrap_err();
        assert!(e.contains("axis 2"), "{e}");
    }
}
