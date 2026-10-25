// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Combine graphs into one, and split one graph into several.
//!
//! **Combining.** [`Graph::splice`] copies a graph into another, feeding
//! named leaves from nodes already there. It is the one copy-in primitive:
//! the others here, `rlx_compile::inline_into` and autodiff's
//! `merge_subgraph` are all calls to it.
//!
//! * [`parallel`] — side by side: one graph whose outputs are every graph's
//!   outputs in order.
//! * [`sequential`] — a chain: each graph's inputs are fed, in declaration
//!   order, by the previous graph's outputs (`f ∘ g`).
//!
//! Leaves are bound by name, so a combined graph keeps one leaf per name
//! (see [`crate::verify_unique_leaf_names`]): an unbound input or param whose
//! name the target already has is **shared** — the same input fed once, a
//! tied weight loaded once — after checking that both agree on kind and
//! shape. [`Splice::prefix`] gives a graph its own namespace instead;
//! [`Splice::bind`] ties a leaf to any node, a weight of another name
//! included.
//!
//! **Splitting.** Each of these returns a [`Partition`]: self-contained part
//! graphs that run in order, and [`Partition::merge`] to put them back.
//!
//! * [`Graph::partition`] — any node→part assignment whose edges run from a
//!   part to the same or a later one. A value crossing parts becomes a named
//!   boundary ([`boundary_name`]): an output of its producer's part, an
//!   `Op::Input` of each consumer's part.
//! * [`Graph::split_at`] — the two-part cut at given nodes.
//! * [`Graph::components`] — the independent subgraphs (no compute edge
//!   between them; reading the same input does not connect them), as parts
//!   with no imports.
//!
//! [`Graph::extract`] is the single subgraph computing some nodes.
//!
//! Variables ([`Graph::vars`]) and symbolic op attributes
//! ([`Graph::sym_attrs`]) follow their nodes through every operation.
//! Symbols themselves are not renumbered: graphs combined must agree on
//! what each symbol id means.

use crate::collections::{HashMap, HashSet};
use crate::dynamic::{SymAttr, SymDef};
use crate::infer_shape::infer_output_shape;
use crate::prelude::*;
use crate::{Graph, NodeId, Op, Shape};

/// Prefix of the leaf names a [`Partition`] gives values crossing parts.
pub const BOUNDARY_PREFIX: &str = "__boundary.";

/// The name a [`Partition`] gives the value of source node `id` where it
/// crosses parts: `%17` is `__boundary.17`.
pub fn boundary_name(id: NodeId) -> String {
    format!("{BOUNDARY_PREFIX}{}", id.0)
}

/// Why graphs could not be combined or split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComposeError {
    /// A binding names a leaf (input or param) the source graph lacks.
    UnknownLeaf { graph: String, name: String },
    /// The same leaf is bound twice.
    DuplicateBinding { name: String },
    /// A node id outside the graph.
    UnknownNode(NodeId),
    /// A leaf and what it is bound or shared with disagree on shape.
    ShapeMismatch {
        name: String,
        expected: Box<Shape>,
        found: Box<Shape>,
    },
    /// The same name is an `Op::Input` on one side and an `Op::Param` on
    /// the other.
    LeafKindClash { name: String },
    /// [`sequential`]: graph `stage` declares `inputs` free inputs but the
    /// graph before it has `outputs` outputs.
    ArityMismatch {
        stage: usize,
        outputs: usize,
        inputs: usize,
    },
    /// Two graphs share a variable input; both would write its next value.
    SharedVar { name: String },
    /// Two graphs define the same derived symbol differently.
    SymDefConflict { sym: u32 },
    /// [`Graph::partition`] into zero parts.
    NoParts,
    /// [`Graph::partition`]: a node was assigned past the last part.
    PartOutOfRange {
        node: NodeId,
        part: usize,
        parts: usize,
    },
    /// [`Graph::partition`]: an edge runs from a later part to an earlier
    /// one. Parts run in order, so the consumer would read a value not yet
    /// computed.
    BackwardEdge {
        from: NodeId,
        from_part: usize,
        to: NodeId,
        to_part: usize,
    },
    /// [`Partition::merge`]: a part imports a boundary no earlier part
    /// exports.
    MissingImport { part: usize, name: String },
}

impl core::fmt::Display for ComposeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownLeaf { graph, name } => {
                write!(f, "graph {graph:?} has no input or param named {name:?}")
            }
            Self::DuplicateBinding { name } => write!(f, "{name:?} is bound twice"),
            Self::UnknownNode(id) => write!(f, "node {id} is not in the graph"),
            Self::ShapeMismatch {
                name,
                expected,
                found,
            } => write!(f, "{name:?}: expected shape {expected}, found {found}"),
            Self::LeafKindClash { name } => write!(
                f,
                "{name:?} is an input on one side and a param on the other"
            ),
            Self::ArityMismatch {
                stage,
                outputs,
                inputs,
            } => write!(
                f,
                "graph {stage} declares {inputs} free inputs, the graph before it has {outputs} outputs"
            ),
            Self::SharedVar { name } => {
                write!(f, "variable {name:?} would be written by two graphs")
            }
            Self::SymDefConflict { sym } => {
                write!(f, "symbol {sym} is defined differently by two graphs")
            }
            Self::NoParts => write!(f, "a partition needs at least one part"),
            Self::PartOutOfRange { node, part, parts } => {
                write!(f, "node {node} assigned part {part} of {parts}")
            }
            Self::BackwardEdge {
                from,
                from_part,
                to,
                to_part,
            } => write!(
                f,
                "edge {from} (part {from_part}) → {to} (part {to_part}) runs to an earlier part"
            ),
            Self::MissingImport { part, name } => write!(
                f,
                "part {part} imports {name:?}, which no earlier part exports"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ComposeError {}

/// How [`Graph::splice`] wires the source graph in.
#[derive(Debug, Clone, Default)]
pub struct Splice {
    bind: Vec<(String, NodeId)>,
    prefix: String,
    adapt_shapes: bool,
}

impl Splice {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed the source's leaf `name` from `node`, a node of the target.
    /// The leaf is an input, or a param: bind one to tie it to a weight of
    /// another name, or to compute it. `name` is the source's own, before
    /// any [`prefix`](Self::prefix).
    pub fn bind(mut self, name: impl Into<String>, node: NodeId) -> Self {
        self.bind.push((name.into(), node));
        self
    }

    /// [`bind`](Self::bind) each `(name, node)` pair.
    pub fn bind_all<S: Into<String>>(
        mut self,
        pairs: impl IntoIterator<Item = (S, NodeId)>,
    ) -> Self {
        self.bind
            .extend(pairs.into_iter().map(|(name, node)| (name.into(), node)));
        self
    }

    /// Prepend `prefix` to the names of the source's unbound leaves and its
    /// node names (`"enc."` → `enc.x`), so they don't share with same-named
    /// leaves already in the target.
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Accept bound nodes whose shape differs from the leaf they feed, and
    /// infer every copied node's shape from its new operands (keeping the
    /// source's where inference gives up). Autodiff splices per-step bodies
    /// this way. Without it a differing shape is
    /// [`ComposeError::ShapeMismatch`].
    pub fn adapt_shapes(mut self) -> Self {
        self.adapt_shapes = true;
        self
    }
}

/// Where [`Graph::splice`] put the source graph.
#[derive(Debug, Clone)]
pub struct Spliced {
    /// The source's outputs, as target nodes.
    pub outputs: Vec<NodeId>,
    /// `map[i]` is the target node that source node `%i` became.
    pub map: Vec<NodeId>,
    /// The source's variables whose input stayed unbound, as target
    /// `(input, next value)` nodes. The caller owns the target's outputs, so
    /// it decides where each next value goes ([`Graph::set_vars`]).
    pub vars: Vec<(NodeId, NodeId)>,
}

/// One part of a [`Partition`].
#[derive(Debug, Clone)]
pub struct Part {
    /// A self-contained graph: the part's nodes, the leaves they read, and
    /// an `Op::Input` for every value an earlier part computes. It carries
    /// a variable when it holds both the variable's input and next value.
    pub graph: Graph,
    /// Names of the boundary inputs this part reads.
    pub imports: Vec<String>,
    /// The name of each of `graph.outputs`.
    pub exports: Vec<String>,
}

/// A graph split into parts that run in order. Run each part's graph,
/// feeding its imports from earlier parts' exports of the same name, or
/// [`merge`](Self::merge) them back into one graph.
#[derive(Debug, Clone)]
pub struct Partition {
    /// The source graph's name.
    pub name: String,
    pub parts: Vec<Part>,
    /// Source output `k` is `parts[p].graph.outputs[j]` where
    /// `outputs[k] == (p, j)`.
    pub outputs: Vec<(usize, usize)>,
    /// The source's variables as `(input name, source output index)`,
    /// whichever parts their ends landed in.
    pub vars: Vec<(String, usize)>,
}

fn leaf_name(op: &Op) -> Option<&str> {
    match op {
        Op::Input { name } | Op::Param { name } => Some(name),
        _ => None,
    }
}

fn prefixed(prefix: &str, name: &str) -> String {
    format!("{prefix}{name}")
}

fn merge_sym_defs(into: &mut Vec<SymDef>, from: &[SymDef]) -> Result<(), ComposeError> {
    for d in from {
        match into.iter().find(|e| e.sym == d.sym) {
            Some(e) if e != d => return Err(ComposeError::SymDefConflict { sym: d.sym }),
            Some(_) => {}
            None => into.push(*d),
        }
    }
    Ok(())
}

fn check_nodes(g: &Graph, ids: &[NodeId]) -> Result<(), ComposeError> {
    match ids.iter().find(|id| id.0 as usize >= g.len()) {
        Some(&bad) => Err(ComposeError::UnknownNode(bad)),
        None => Ok(()),
    }
}

/// `seen[i]` for every node `ids` read, transitively, and `ids` themselves.
fn cone(g: &Graph, ids: &[NodeId]) -> Vec<bool> {
    let mut seen = vec![false; g.len()];
    for id in ids {
        seen[id.0 as usize] = true;
    }
    for i in (0..g.len()).rev() {
        if seen[i] {
            for input in &g.nodes()[i].inputs {
                seen[input.0 as usize] = true;
            }
        }
    }
    seen
}

/// The source's variables by input name, for a [`Partition`].
fn vars_by_name(g: &Graph) -> Vec<(String, usize)> {
    g.vars()
        .iter()
        .filter_map(|&(input, out)| Some((leaf_name(&g.node(input).op)?.to_string(), out)))
        .collect()
}

impl Graph {
    /// Copy `src` into this graph and return where it went.
    ///
    /// Leaves named in [`Splice::bind`] read the given nodes; the source's
    /// other inputs and params become leaves of this graph, shared with any
    /// same-named leaf already here (see the [module docs](crate::compose)).
    /// Outputs are left alone: wire [`Spliced::outputs`] in as needed.
    ///
    /// Everything is checked before anything is added, so on error the
    /// graph is unchanged.
    pub fn splice(&mut self, src: &Graph, splice: &Splice) -> Result<Spliced, ComposeError> {
        // ── Check ──────────────────────────────────────────────────────
        let mut bound: HashMap<&str, NodeId> = HashMap::new();
        for (name, node) in &splice.bind {
            let Some(leaf) = src.node_id_by_name(name) else {
                return Err(ComposeError::UnknownLeaf {
                    graph: src.name.clone(),
                    name: name.clone(),
                });
            };
            check_nodes(self, &[*node])?;
            if bound.insert(name, *node).is_some() {
                return Err(ComposeError::DuplicateBinding { name: name.clone() });
            }
            let (expected, found) = (src.shape(leaf), self.shape(*node));
            if !splice.adapt_shapes && expected != found {
                return Err(ComposeError::ShapeMismatch {
                    name: name.clone(),
                    expected: Box::new(expected.clone()),
                    found: Box::new(found.clone()),
                });
            }
        }
        let mut leaves: HashMap<String, NodeId> = HashMap::new();
        for node in self.nodes() {
            if let Some(name) = leaf_name(&node.op) {
                leaves.entry(name.to_string()).or_insert(node.id);
            }
        }
        for node in src.nodes() {
            let Some(name) = leaf_name(&node.op).filter(|n| !bound.contains_key(n)) else {
                continue;
            };
            let name = prefixed(&splice.prefix, name);
            if let Some(&existing) = leaves.get(&name) {
                let here = self.node(existing);
                if core::mem::discriminant(&here.op) != core::mem::discriminant(&node.op) {
                    return Err(ComposeError::LeafKindClash { name });
                }
                if here.shape != node.shape {
                    return Err(ComposeError::ShapeMismatch {
                        name,
                        expected: Box::new(here.shape.clone()),
                        found: Box::new(node.shape.clone()),
                    });
                }
            }
        }
        let mut sym_defs = self.sym_defs().to_vec();
        merge_sym_defs(&mut sym_defs, src.sym_defs())?;

        // ── Copy ───────────────────────────────────────────────────────
        let mut map: Vec<NodeId> = Vec::with_capacity(src.len());
        for node in src.nodes() {
            let name = node.name.as_deref().map(|n| prefixed(&splice.prefix, n));
            let id = match leaf_name(&node.op) {
                Some(leaf) => match bound.get(leaf) {
                    Some(&node) => node,
                    None => {
                        let leaf = prefixed(&splice.prefix, leaf);
                        match leaves.get(&leaf) {
                            Some(&existing) => existing,
                            None => {
                                let op = match node.op {
                                    Op::Input { .. } => Op::Input { name: leaf.clone() },
                                    _ => Op::Param { name: leaf.clone() },
                                };
                                let shape = node.shape.clone();
                                let id =
                                    self.push_ext(op, vec![], shape, name, node.origin.clone());
                                leaves.insert(leaf, id);
                                id
                            }
                        }
                    }
                },
                None => {
                    let inputs = node.inputs.iter().map(|i| map[i.0 as usize]).collect();
                    let shape = node.shape.clone();
                    let id =
                        self.push_ext(node.op.clone(), inputs, shape, name, node.origin.clone());
                    if splice.adapt_shapes
                        && !node.op.is_leaf()
                        && let Some(shape) = infer_output_shape(self, self.node(id))
                    {
                        self.node_mut(id).shape = shape;
                    }
                    id
                }
            };
            map.push(id);
        }

        let mut sym_attrs = self.sym_attrs().to_vec();
        sym_attrs.extend(src.sym_attrs().iter().map(|attr| SymAttr {
            node: map[attr.node.0 as usize],
            ..attr.clone()
        }));
        self.set_sym_attrs(sym_attrs);
        self.set_sym_defs(sym_defs);

        let vars = src
            .vars()
            .iter()
            .filter(|(input, _)| {
                leaf_name(&src.node(*input).op).is_none_or(|n| !bound.contains_key(n))
            })
            .map(|&(input, out)| (map[input.0 as usize], map[src.outputs[out].0 as usize]))
            .collect();

        Ok(Spliced {
            outputs: src.outputs.iter().map(|o| map[o.0 as usize]).collect(),
            map,
            vars,
        })
    }

    /// The subgraph computing `outputs`: those nodes and everything they
    /// read, in a new graph whose outputs are `outputs` in order. Variables
    /// survive when both their input and their next value do.
    pub fn extract(&self, outputs: &[NodeId]) -> Result<Graph, ComposeError> {
        check_nodes(self, outputs)?;
        let keep = cone(self, outputs);
        let mut g = Graph::new(self.name.clone());
        let mut map: Vec<Option<NodeId>> = vec![None; self.len()];
        for node in self.nodes().iter().filter(|n| keep[n.id.0 as usize]) {
            let inputs = node
                .inputs
                .iter()
                .map(|i| map[i.0 as usize].unwrap())
                .collect();
            map[node.id.0 as usize] = Some(g.push_ext(
                node.op.clone(),
                inputs,
                node.shape.clone(),
                node.name.clone(),
                node.origin.clone(),
            ));
        }
        g.outputs = outputs.iter().map(|o| map[o.0 as usize].unwrap()).collect();
        g.set_sym_attrs(
            self.sym_attrs()
                .iter()
                .filter_map(|a| map[a.node.0 as usize].map(|node| SymAttr { node, ..a.clone() }))
                .collect(),
        );
        g.set_sym_defs(self.sym_defs().to_vec());
        let vars = self
            .vars()
            .iter()
            .filter_map(|&(input, out)| {
                let input = map[input.0 as usize]?;
                let next = self.outputs[out];
                let j = outputs.iter().position(|&o| o == next)?;
                Some((input, j))
            })
            .collect();
        g.set_vars(vars);
        // An extracted output keeps the name it had as an output here.
        let names = outputs
            .iter()
            .enumerate()
            .filter_map(|(j, o)| {
                let i = self.outputs.iter().position(|x| x == o)?;
                Some((j, self.output_name(i)?.to_string()))
            })
            .collect();
        g.set_output_names(names);
        Ok(g)
    }

    /// Split into independent subgraphs, one part per group of outputs that
    /// share compute. The parts import nothing, so they run in any order —
    /// on different devices at once. A variable's input and next value stay
    /// in one part.
    pub fn components(&self) -> Result<Partition, ComposeError> {
        check_nodes(self, &self.outputs)?;
        let mut parent: Vec<u32> = (0..self.len() as u32).collect();
        fn find(parent: &mut [u32], mut x: u32) -> u32 {
            while parent[x as usize] != x {
                parent[x as usize] = parent[parent[x as usize] as usize];
                x = parent[x as usize];
            }
            x
        }
        fn union(parent: &mut [u32], a: NodeId, b: NodeId) {
            let (ra, rb) = (find(parent, a.0), find(parent, b.0));
            if ra != rb {
                parent[ra.max(rb) as usize] = ra.min(rb);
            }
        }
        for node in self.nodes().iter().filter(|n| !n.op.is_leaf()) {
            for &input in &node.inputs {
                if !self.node(input).op.is_leaf() {
                    union(&mut parent, input, node.id);
                }
            }
        }
        for &(input, out) in self.vars() {
            union(&mut parent, input, self.outputs[out]);
        }

        // (root, the group's distinct output nodes)
        let mut groups: Vec<(u32, Vec<NodeId>)> = Vec::new();
        let mut outputs = Vec::with_capacity(self.outputs.len());
        for &o in &self.outputs {
            let root = find(&mut parent, o.0);
            let p = match groups.iter().position(|(r, _)| *r == root) {
                Some(p) => p,
                None => {
                    groups.push((root, Vec::new()));
                    groups.len() - 1
                }
            };
            let nodes = &mut groups[p].1;
            let j = nodes.iter().position(|&n| n == o).unwrap_or_else(|| {
                nodes.push(o);
                nodes.len() - 1
            });
            outputs.push((p, j));
        }
        let parts = groups
            .into_iter()
            .enumerate()
            .map(|(p, (_, nodes))| {
                let mut graph = self.extract(&nodes)?;
                graph.name = format!("{}__component{p}", self.name);
                Ok(Part {
                    graph,
                    imports: Vec::new(),
                    exports: nodes.iter().map(|&n| boundary_name(n)).collect(),
                })
            })
            .collect::<Result<_, ComposeError>>()?;
        Ok(Partition {
            name: self.name.clone(),
            parts,
            outputs,
            vars: vars_by_name(self),
        })
    }

    /// Split into `parts` graphs that run in order, compute node `id` going
    /// to part `assign(id)`. Leaves (inputs, params, constants) are copied
    /// into every part that reads them. Every edge must run to the same or
    /// a later part.
    pub fn partition(
        &self,
        parts: usize,
        assign: impl Fn(NodeId) -> usize,
    ) -> Result<Partition, ComposeError> {
        if parts == 0 {
            return Err(ComposeError::NoParts);
        }
        let mut part_of: Vec<usize> = vec![0; self.len()];
        for node in self.nodes().iter().filter(|n| !n.op.is_leaf()) {
            let part = assign(node.id);
            if part >= parts {
                return Err(ComposeError::PartOutOfRange {
                    node: node.id,
                    part,
                    parts,
                });
            }
            part_of[node.id.0 as usize] = part;
        }
        self.partition_by(parts, &part_of, &HashSet::new())
    }

    /// Cut into two parts: `cut` and everything they read, then the rest.
    /// Each compute node in `cut` is an output of the first part (an input
    /// of the second when it reads it), even if nothing else needs it there.
    pub fn split_at(&self, cut: &[NodeId]) -> Result<Partition, ComposeError> {
        check_nodes(self, cut)?;
        let part_of: Vec<usize> = cone(self, cut)
            .iter()
            .map(|&head| if head { 0 } else { 1 })
            .collect();
        let force: HashSet<NodeId> = cut
            .iter()
            .copied()
            .filter(|c| !self.node(*c).op.is_leaf())
            .collect();
        self.partition_by(2, &part_of, &force)
    }

    fn partition_by(
        &self,
        parts: usize,
        part_of: &[usize],
        force: &HashSet<NodeId>,
    ) -> Result<Partition, ComposeError> {
        let is_leaf = |id: NodeId| self.node(id).op.is_leaf();
        let mut used_by_later = vec![false; self.len()];
        for node in self.nodes().iter().filter(|n| !is_leaf(n.id)) {
            let to_part = part_of[node.id.0 as usize];
            for &input in node.inputs.iter().filter(|&&i| !is_leaf(i)) {
                let from_part = part_of[input.0 as usize];
                if from_part > to_part {
                    return Err(ComposeError::BackwardEdge {
                        from: input,
                        from_part,
                        to: node.id,
                        to_part,
                    });
                }
                if from_part < to_part {
                    used_by_later[input.0 as usize] = true;
                }
            }
        }
        let is_output: HashSet<NodeId> = self.outputs.iter().copied().collect();

        let mut out_parts = Vec::with_capacity(parts);
        // Source node → (part, output index) for each exported value.
        let mut exported: HashMap<NodeId, (usize, usize)> = HashMap::new();
        for p in 0..parts {
            let mut g = Graph::new(format!("{}__part{p}", self.name));
            let mut local: HashMap<NodeId, NodeId> = HashMap::new();
            let mut imports = Vec::new();
            let materialize = |g: &mut Graph,
                               local: &mut HashMap<NodeId, NodeId>,
                               id: NodeId,
                               imports: &mut Vec<String>| {
                if let Some(&l) = local.get(&id) {
                    return l;
                }
                let src = self.node(id);
                let l = if is_leaf(id) {
                    g.push_ext(
                        src.op.clone(),
                        vec![],
                        src.shape.clone(),
                        src.name.clone(),
                        src.origin.clone(),
                    )
                } else {
                    let name = boundary_name(id);
                    imports.push(name.clone());
                    g.push_ext(
                        Op::Input { name: name.clone() },
                        vec![],
                        src.shape.clone(),
                        Some(name),
                        None,
                    )
                };
                local.insert(id, l);
                l
            };

            let mut exports = Vec::new();
            let mut outs = Vec::new();
            // Outputs that are leaves are passed through by the first part.
            if p == 0 {
                for &o in self.outputs.iter().filter(|&&o| is_leaf(o)) {
                    exported.entry(o).or_insert_with(|| {
                        let l = materialize(&mut g, &mut local, o, &mut imports);
                        outs.push(l);
                        exports.push(boundary_name(o));
                        (p, outs.len() - 1)
                    });
                }
            }
            for node in self
                .nodes()
                .iter()
                .filter(|n| !is_leaf(n.id) && part_of[n.id.0 as usize] == p)
            {
                let inputs = node
                    .inputs
                    .iter()
                    .map(|&i| materialize(&mut g, &mut local, i, &mut imports))
                    .collect();
                let l = g.push_ext(
                    node.op.clone(),
                    inputs,
                    node.shape.clone(),
                    node.name.clone(),
                    node.origin.clone(),
                );
                local.insert(node.id, l);
                let id = node.id;
                if is_output.contains(&id) || used_by_later[id.0 as usize] || force.contains(&id) {
                    exported.insert(id, (p, outs.len()));
                    outs.push(l);
                    exports.push(boundary_name(id));
                }
            }
            g.outputs = outs;
            g.set_sym_attrs(
                self.sym_attrs()
                    .iter()
                    .filter_map(|a| {
                        let node = *local.get(&a.node)?;
                        (!is_leaf(a.node)).then(|| SymAttr { node, ..a.clone() })
                    })
                    .collect(),
            );
            g.set_sym_defs(self.sym_defs().to_vec());
            g.set_vars(
                self.vars()
                    .iter()
                    .filter_map(|&(input, out)| {
                        let input = *local.get(&input)?;
                        match exported.get(&self.outputs[out]) {
                            Some(&(q, j)) if q == p => Some((input, j)),
                            _ => None,
                        }
                    })
                    .collect(),
            );
            out_parts.push(Part {
                graph: g,
                imports,
                exports,
            });
        }

        Ok(Partition {
            name: self.name.clone(),
            parts: out_parts,
            outputs: self.outputs.iter().map(|o| exported[o]).collect(),
            vars: vars_by_name(self),
        })
    }
}

impl Partition {
    /// One graph again: the parts spliced in order, each import fed by the
    /// part that exports it, the source's outputs and variables restored.
    pub fn merge(&self) -> Result<Graph, ComposeError> {
        let mut g = Graph::new(self.name.clone());
        let mut produced: HashMap<&str, NodeId> = HashMap::new();
        let mut part_outputs: Vec<Vec<NodeId>> = Vec::with_capacity(self.parts.len());
        for (p, part) in self.parts.iter().enumerate() {
            let mut splice = Splice::new();
            for name in &part.imports {
                let Some(&node) = produced.get(name.as_str()) else {
                    return Err(ComposeError::MissingImport {
                        part: p,
                        name: name.clone(),
                    });
                };
                splice = splice.bind(name.clone(), node);
            }
            let spliced = g.splice(&part.graph, &splice)?;
            for (name, &id) in part.exports.iter().zip(&spliced.outputs) {
                produced.insert(name, id);
            }
            part_outputs.push(spliced.outputs);
        }
        g.outputs = self
            .outputs
            .iter()
            .map(|&(p, j)| part_outputs[p][j])
            .collect();
        let vars = self
            .vars
            .iter()
            .filter_map(|(name, out)| Some((g.input_id(name)?, *out)))
            .collect();
        g.set_vars(vars);
        Ok(g)
    }
}

/// The graphs side by side: one graph with every graph's outputs, in order
/// (a variable's next value stays where its graph put it). Same-named
/// leaves are shared (see the [module docs](crate::compose)).
pub fn parallel(name: impl Into<String>, graphs: &[&Graph]) -> Result<Graph, ComposeError> {
    let mut g = Graph::new(name);
    let mut state = State::default();
    let mut names = Vec::new();
    for src in graphs {
        let base = g.outputs.len();
        let spliced = g.splice(src, &Splice::new())?;
        for &(input, out) in src.vars() {
            state.add(&g, spliced.map[input.0 as usize], base + out)?;
        }
        names.extend(
            src.output_names()
                .iter()
                .map(|(i, n)| (base + i, n.clone())),
        );
        g.outputs.extend(spliced.outputs);
    }
    g.set_vars(state.vars);
    g.set_output_names(names);
    Ok(g)
}

/// The graphs chained: graph `k`'s free inputs (inputs that are not
/// variables), in declaration order, are fed by graph `k - 1`'s outputs
/// that are not variables' next values, in order. The result's outputs are
/// the last graph's such outputs, then every variable's next value (graph
/// order). Same-named params and unfed inputs are shared.
pub fn sequential(name: impl Into<String>, graphs: &[&Graph]) -> Result<Graph, ComposeError> {
    let mut g = Graph::new(name);
    let mut carry: Vec<NodeId> = Vec::new();
    let mut carry_names: Vec<(usize, String)> = Vec::new();
    let mut next_values: Vec<(NodeId, NodeId)> = Vec::new();
    for (stage, src) in graphs.iter().enumerate() {
        let is_var_input = |id: NodeId| src.vars().iter().any(|v| v.0 == id);
        let mut splice = Splice::new();
        if stage > 0 {
            let free: Vec<&str> = src
                .nodes()
                .iter()
                .filter(|n| !is_var_input(n.id))
                .filter_map(|n| match &n.op {
                    Op::Input { name } => Some(name.as_str()),
                    _ => None,
                })
                .collect();
            if free.len() != carry.len() {
                return Err(ComposeError::ArityMismatch {
                    stage,
                    outputs: carry.len(),
                    inputs: free.len(),
                });
            }
            splice = splice.bind_all(free.into_iter().zip(carry.iter().copied()));
        }
        let spliced = g.splice(src, &splice)?;
        next_values.extend(&spliced.vars);
        let free: Vec<usize> = (0..spliced.outputs.len())
            .filter(|k| !src.vars().iter().any(|v| v.1 == *k))
            .collect();
        carry = free.iter().map(|&k| spliced.outputs[k]).collect();
        carry_names = free
            .iter()
            .enumerate()
            .filter_map(|(j, &k)| Some((j, src.output_name(k)?.to_string())))
            .collect();
    }
    let mut state = State::default();
    for (j, &(input, _)) in next_values.iter().enumerate() {
        state.add(&g, input, carry.len() + j)?;
    }
    g.outputs = carry;
    g.outputs.extend(next_values.iter().map(|&(_, next)| next));
    g.set_vars(state.vars);
    // The last graph's outputs keep their names.
    g.set_output_names(carry_names);
    Ok(g)
}

/// A combined graph's variables, refusing one input written twice.
#[derive(Default)]
struct State {
    vars: Vec<(NodeId, usize)>,
}

impl State {
    fn add(&mut self, g: &Graph, input: NodeId, out: usize) -> Result<(), ComposeError> {
        if self.vars.iter().any(|(i, _)| *i == input) {
            return Err(ComposeError::SharedVar {
                name: leaf_name(&g.node(input).op).unwrap_or_default().to_string(),
            });
        }
        self.vars.push((input, out));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::GraphExt;
    use crate::{DType, verify_all};

    fn f32s(dims: &[usize]) -> Shape {
        Shape::new(dims, DType::F32)
    }

    /// `y = relu(x @ w + b)`, `x: [2, 4]`, `w: [4, 4]`.
    fn dense(name: &str) -> Graph {
        let mut g = Graph::new(name);
        let x = g.input("x", f32s(&[2, 4]));
        let w = g.param("w", f32s(&[4, 4]));
        let b = g.param("b", f32s(&[4]));
        let mm = g.matmul(x, w, f32s(&[2, 4]));
        let y = g.add(mm, b);
        let y = g.relu(y);
        g.set_outputs(vec![y]);
        g
    }

    fn leaf_names(g: &Graph) -> Vec<String> {
        let mut v: Vec<String> = g
            .nodes()
            .iter()
            .filter_map(|n| leaf_name(&n.op).map(String::from))
            .collect();
        v.sort();
        v
    }

    fn assert_valid(g: &Graph) {
        let errors = verify_all(g);
        assert!(errors.is_empty(), "{g}\n{errors:?}");
        assert!(crate::verify_unique_leaf_names(g).is_empty(), "{g}");
    }

    #[test]
    fn sequential_feeds_outputs_into_inputs_and_shares_params() {
        let g = sequential("two", &[&dense("a"), &dense("b")]).unwrap();
        assert_valid(&g);
        // One x, one tied w and b: both layers read the same weights.
        assert_eq!(leaf_names(&g), ["b", "w", "x"]);
        assert_eq!(g.outputs.len(), 1);
        let matmuls = g
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::MatMul))
            .count();
        assert_eq!(matmuls, 2);
    }

    #[test]
    fn prefix_keeps_weights_apart() {
        let mut g = Graph::new("two");
        let a = g.splice(&dense("a"), &Splice::new().prefix("l0.")).unwrap();
        let b = g
            .splice(
                &dense("b"),
                &Splice::new().prefix("l1.").bind("x", a.outputs[0]),
            )
            .unwrap();
        g.set_outputs(b.outputs);
        assert_valid(&g);
        assert_eq!(leaf_names(&g), ["l0.b", "l0.w", "l0.x", "l1.b", "l1.w"]);
    }

    #[test]
    fn parallel_concatenates_outputs() {
        let mut other = Graph::new("other");
        let x = other.input("x", f32s(&[2, 4]));
        let y = other.exp(x);
        other.set_outputs(vec![y]);
        let g = parallel("both", &[&dense("a"), &other]).unwrap();
        assert_valid(&g);
        assert_eq!(g.outputs.len(), 2);
        // `x` is shared.
        assert_eq!(leaf_names(&g), ["b", "w", "x"]);
    }

    #[test]
    fn errors_leave_target_unchanged() {
        let mut g = Graph::new("t");
        let z = g.input("z", f32s(&[3]));
        let before = g.len();
        let err = g
            .splice(&dense("a"), &Splice::new().bind("x", z))
            .unwrap_err();
        assert!(matches!(err, ComposeError::ShapeMismatch { .. }), "{err}");
        let err = g
            .splice(&dense("a"), &Splice::new().bind("nope", z))
            .unwrap_err();
        assert!(matches!(err, ComposeError::UnknownLeaf { .. }), "{err}");
        // `w` as an input clashes with `w` the param.
        let _ = g.input("w", f32s(&[4, 4]));
        let before_clash = g.len();
        let err = g.splice(&dense("a"), &Splice::new()).unwrap_err();
        assert_eq!(err, ComposeError::LeafKindClash { name: "w".into() });
        assert_eq!(g.len(), before_clash);
        assert!(before < before_clash);
    }

    #[test]
    fn sequential_checks_arity() {
        let mut two = Graph::new("two");
        let p = two.input("p", f32s(&[2, 4]));
        let q = two.input("q", f32s(&[2, 4]));
        let s = two.add(p, q);
        two.set_outputs(vec![s]);
        let err = sequential("bad", &[&dense("a"), &two]).unwrap_err();
        assert_eq!(
            err,
            ComposeError::ArityMismatch {
                stage: 1,
                outputs: 1,
                inputs: 2
            }
        );
    }

    #[test]
    fn extract_keeps_only_the_cone() {
        let mut g = dense("a");
        let x = g.input_id("x").unwrap();
        let side = g.exp(x);
        let y = g.outputs[0];
        g.set_outputs(vec![y, side]);
        let e = g.extract(&[side]).unwrap();
        assert_valid(&e);
        assert_eq!(leaf_names(&e), ["x"]);
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn components_split_independent_outputs() {
        let mut other = Graph::new("other");
        let x = other.input("x", f32s(&[2, 4]));
        let y = other.exp(x);
        other.set_outputs(vec![y]);
        // Outputs: [dense, exp, dense] — two components sharing input `x`.
        let mut g = parallel("both", &[&dense("a"), &other]).unwrap();
        let first = g.outputs[0];
        g.outputs.push(first);
        let p = g.components().unwrap();
        assert_eq!(p.parts.len(), 2);
        // The repeated output is one export, read twice.
        assert_eq!(p.outputs, [(0, 0), (1, 0), (0, 0)]);
        for part in &p.parts {
            assert_valid(&part.graph);
            assert!(part.imports.is_empty());
        }
        assert_eq!(leaf_names(&p.parts[1].graph), ["x"]);
        let merged = p.merge().unwrap();
        assert_valid(&merged);
        assert_eq!(merged.outputs.len(), 3);
        assert_eq!(merged.outputs[0], merged.outputs[2]);
        assert_eq!(leaf_names(&merged), ["b", "w", "x"]);
    }

    #[test]
    fn bind_ties_a_param_to_another_weight() {
        // Decoder `w_out` tied to the encoder's `w`.
        let mut dec = Graph::new("dec");
        let h = dec.input("h", f32s(&[2, 4]));
        let w_out = dec.param("w_out", f32s(&[4, 4]));
        let y = dec.matmul(h, w_out, f32s(&[2, 4]));
        dec.set_outputs(vec![y]);

        let mut g = Graph::new("tied");
        let enc = g.splice(&dense("enc"), &Splice::new()).unwrap();
        let w = g.param_id("w").unwrap();
        let out = g
            .splice(
                &dec,
                &Splice::new().bind("h", enc.outputs[0]).bind("w_out", w),
            )
            .unwrap();
        g.set_outputs(out.outputs);
        assert_valid(&g);
        assert_eq!(leaf_names(&g), ["b", "w", "x"]);

        let x = g.input_id("x").unwrap();
        let err = g
            .splice(&dec, &Splice::new().bind("h", x).bind("h", x))
            .unwrap_err();
        assert_eq!(err, ComposeError::DuplicateBinding { name: "h".into() });
    }

    #[test]
    fn adapt_shapes_reinfers_from_bound_operands() {
        // Body written for [2, 4]; fed a [3, 4] value.
        let mut body = Graph::new("body");
        let x = body.input("x", f32s(&[2, 4]));
        let y = body.exp(x);
        let z = body.add(y, x);
        body.set_outputs(vec![z]);

        let mut g = Graph::new("t");
        let v = g.input("v", f32s(&[3, 4]));
        let err = g.splice(&body, &Splice::new().bind("x", v)).unwrap_err();
        assert!(matches!(err, ComposeError::ShapeMismatch { .. }), "{err}");
        let s = g
            .splice(&body, &Splice::new().bind("x", v).adapt_shapes())
            .unwrap();
        g.set_outputs(s.outputs.clone());
        assert_eq!(g.shape(s.outputs[0]), &f32s(&[3, 4]));
        assert_valid(&g);
    }

    #[test]
    fn partition_round_trips() {
        let g = sequential("deep", &[&dense("a"), &dense("b"), &dense("c")]).unwrap();
        let n = g.len();
        let p = g.partition(3, |id| (id.0 as usize * 3 / n).min(2)).unwrap();
        assert_eq!(p.parts.len(), 3);
        for part in &p.parts {
            assert_valid(&part.graph);
            assert_eq!(part.graph.outputs.len(), part.exports.len());
        }
        let merged = p.merge().unwrap();
        assert_valid(&merged);
        assert_eq!(merged.len(), g.len());
        assert_eq!(leaf_names(&merged), leaf_names(&g));
    }

    #[test]
    fn partition_rejects_backward_edges() {
        let g = dense("a");
        // relu (last node) in part 0, its input in part 1.
        let last = NodeId(g.len() as u32 - 1);
        let err = g
            .partition(2, |id| if id == last { 0 } else { 1 })
            .unwrap_err();
        assert!(matches!(err, ComposeError::BackwardEdge { .. }), "{err}");
        let err = g.partition(2, |_| 5).unwrap_err();
        assert!(matches!(err, ComposeError::PartOutOfRange { .. }), "{err}");
        assert_eq!(g.partition(0, |_| 0).unwrap_err(), ComposeError::NoParts);
    }

    #[test]
    fn split_at_exports_the_cut() {
        let g = dense("a");
        let mm = g
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::MatMul))
            .unwrap()
            .id;
        let p = g.split_at(&[mm]).unwrap();
        let (head, tail) = (&p.parts[0], &p.parts[1]);
        assert_eq!(head.exports, [format!("{BOUNDARY_PREFIX}{}", mm.0)]);
        assert_eq!(tail.imports, head.exports);
        assert_eq!(leaf_names(&head.graph), ["w", "x"]);
        // Model output lives in the tail.
        assert_eq!(p.outputs, [(1, 0)]);
        let merged = p.merge().unwrap();
        assert_valid(&merged);
    }

    #[test]
    fn output_names_survive_extract_parallel_and_sequential() {
        let mut a = Graph::new("a");
        let x = a.input("x", f32s(&[4]));
        let e = a.exp(x);
        let r = a.relu(x);
        a.set_named_outputs([("exp", e), ("relu", r)]);

        let only = a.extract(&[r]).unwrap();
        assert_eq!(only.output_names(), &[(0, "relu".to_string())]);

        let mut b = Graph::new("b");
        let q = b.input("q", f32s(&[1]));
        let t = b.exp(q);
        b.set_outputs(vec![t]);
        b.name_output(0, "t");
        let p = parallel("p", &[&a, &b]).unwrap();
        assert_eq!(p.output_index("t"), Some(2));
        assert_eq!(p.output_index("relu"), Some(1));

        // `b` after `a`: two outputs of `a` feed `c`'s two inputs.
        let mut c = Graph::new("c");
        let u = c.input("u", f32s(&[4]));
        let v = c.input("v", f32s(&[4]));
        let w = c.add(u, v);
        c.set_named_outputs([("sum", w)]);
        let s = sequential("s", &[&a, &c]).unwrap();
        assert_eq!(s.output_names(), &[(0, "sum".to_string())]);
    }

    #[test]
    fn vars_survive_parallel_and_round_trip() {
        // state' = state + x: a running sum carried across calls.
        let mut acc = Graph::new("acc");
        let s = acc.input("state", f32s(&[4]));
        let x = acc.input("x", f32s(&[4]));
        let y = acc.add(s, x);
        let z = acc.exp(x);
        acc.set_outputs(vec![z, y]);
        acc.set_vars(vec![(s, 1)]);

        let g = parallel("p", &[&dense("d"), &acc]).unwrap_err();
        // `x` is [2, 4] in dense and [4] here.
        assert!(matches!(g, ComposeError::ShapeMismatch { .. }));

        let mut other = Graph::new("o");
        let q = other.input("q", f32s(&[1]));
        let r = other.exp(q);
        other.set_outputs(vec![r]);
        let g = parallel("p", &[&other, &acc]).unwrap();
        let state = g.input_id("state").unwrap();
        assert_eq!(g.vars(), [(state, 2)]);

        let err = parallel("pp", &[&acc, &acc]).unwrap_err();
        assert_eq!(
            err,
            ComposeError::SharedVar {
                name: "state".into()
            }
        );

        let p = g.split_at(&[g.outputs[0]]).unwrap();
        // The tail holds both ends of `state`, so it carries the variable.
        let tail = &p.parts[1].graph;
        assert_eq!(tail.vars().len(), 1);
        let (input, j) = tail.vars()[0];
        assert_eq!(
            tail.node(input).op,
            Op::Input {
                name: "state".into()
            }
        );
        assert_eq!(p.outputs[2], (1, j));
        assert!(p.parts[0].graph.vars().is_empty());
        let merged = p.merge().unwrap();
        assert_eq!(merged.vars(), [(merged.input_id("state").unwrap(), 2)]);
    }

    #[test]
    fn sequential_threads_vars() {
        let mut acc = Graph::new("acc");
        let s = acc.input("state", f32s(&[2, 4]));
        let x = acc.input("x", f32s(&[2, 4]));
        let y = acc.add(s, x);
        acc.set_outputs(vec![y, y]);
        acc.set_vars(vec![(s, 1)]);
        let g = sequential("seq", &[&dense("d"), &acc]).unwrap();
        assert_valid(&g);
        // The free input `x` of `acc` is fed by dense; `state` stays a var.
        assert_eq!(leaf_names(&g), ["b", "state", "w", "x"]);
        assert_eq!(g.outputs.len(), 2);
        assert_eq!(g.vars(), [(g.input_id("state").unwrap(), 1)]);
    }
}
