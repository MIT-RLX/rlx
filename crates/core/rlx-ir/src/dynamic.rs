// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dynamic / symbolic dimensions — compile once, specialize at runtime.
//!
//! Plan #54: graphs built with [`Dim::Dynamic`] symbols specialize via
//! [`DimBinding`] before buffer planning and backend lowering.

use crate::collections::HashMap;
use crate::prelude::*;
use alloc::collections::BTreeSet;

use crate::shape::{Dim, DimBinding, Shape};
use crate::{DType, Graph, Op};

/// Well-known dynamic dimension symbols. Reuse the same id across shapes
/// so `[?0, ?1, H]` and `[?0, ?1, 4]` share batch/seq bindings.
pub mod sym {
    pub const BATCH: u32 = 0;
    pub const SEQ: u32 = 1;
    /// Cached prefix length for decode KV (past_k / past_v axis 1).
    pub const PAST_SEQ: u32 = 3;
    /// Product of leading axes (e.g. `batch * seq` flatten).
    pub const ROWS: u32 = 2;
}

/// Allocate named dynamic symbols for model builders.
#[derive(Debug, Clone, Default)]
pub struct DimEnv {
    next: u32,
    names: HashMap<String, u32>,
}

impl DimEnv {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the symbol id for `name`, allocating on first use.
    pub fn sym(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.names.get(name) {
            return id;
        }
        let id = self.next;
        self.next += 1;
        self.names.insert(name.into(), id);
        id
    }

    pub fn name(&self, symbol: u32) -> Option<&str> {
        self.names
            .iter()
            .find_map(|(n, &s)| (s == symbol).then_some(n.as_str()))
    }
}

impl Shape {
    /// `[batch, seq, hidden]` with symbolic leading axes.
    pub fn batch_seq(batch: u32, seq: u32, hidden: usize, dtype: DType) -> Self {
        Self::from_dims(
            &[Dim::Dynamic(batch), Dim::Dynamic(seq), Dim::Static(hidden)],
            dtype,
        )
    }

    /// `[batch, seq]` matrix.
    pub fn batch_seq_2d(batch: u32, seq: u32, dtype: DType) -> Self {
        Self::from_dims(&[Dim::Dynamic(batch), Dim::Dynamic(seq)], dtype)
    }

    /// `[batch, seq, heads, head_dim]` attention layout.
    pub fn batch_seq_heads(
        batch: u32,
        seq: u32,
        heads: usize,
        head_dim: usize,
        dtype: DType,
    ) -> Self {
        Self::from_dims(
            &[
                Dim::Dynamic(batch),
                Dim::Dynamic(seq),
                Dim::Static(heads),
                Dim::Static(head_dim),
            ],
            dtype,
        )
    }
}

impl DimBinding {
    pub fn from_pairs(pairs: &[(u32, usize)]) -> Self {
        let mut b = Self::new();
        for &(sym, size) in pairs {
            b.set(sym, size);
        }
        b
    }

    pub fn batch_seq(batch: usize, seq: usize) -> Self {
        let mut b = Self::from_pairs(&[(sym::BATCH, batch), (sym::SEQ, seq)]);
        if batch > 1 {
            b.set(sym::ROWS, batch * seq);
        }
        b
    }

    pub fn batch_past_seq(batch: usize, past_seq: usize) -> Self {
        Self::from_pairs(&[(sym::BATCH, batch), (sym::PAST_SEQ, past_seq)])
    }
}

/// True if any node shape references a dynamic dimension.
pub fn has_dynamic_dims(graph: &Graph) -> bool {
    graph
        .nodes()
        .iter()
        .any(|n| n.shape.dims().iter().any(|d| matches!(d, Dim::Dynamic(_))))
}

/// Collect all dynamic symbols referenced anywhere in the graph.
pub fn collect_dynamic_symbols(graph: &Graph) -> Vec<u32> {
    let mut syms = BTreeSet::new();
    for node in graph.nodes() {
        for s in node.shape.dynamic_symbols() {
            syms.insert(s);
        }
    }
    syms.into_iter().collect()
}

/// An op attribute that is a symbol's size plus `offset`: a decode step's
/// cache write position `pos = upper`, a slice's `start = seq - 1`. `path`
/// names the field in the op's serde form (`["pos"]`; an index for a tuple
/// field). [`generalize`] finds them, [`specialize`] writes them, and
/// `.rlxmodel` prints them in place (`KvAppend{axis: 1, pos: upper}`).
/// Reading and writing a field by name needs the `serialize` feature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymAttr {
    pub node: crate::NodeId,
    pub path: Vec<String>,
    pub sym: u32,
    /// The attribute is the size times this, plus `offset` (−1: a start
    /// that falls as the size grows, `1000 − seq`).
    pub scale: i64,
    pub offset: i64,
}

/// A symbol that is another's size (or the product of two symbols' sizes)
/// times `scale` plus `offset`: the `seq*16` rows of a `[seq, 16, 128]`
/// flattened to two dims, the `seq+1` of a cache with one more row, the
/// `h*w*4` patches of an `h × w` grid of 2×2 blocks — plus `linear` times
/// `of` for a square (`times == of`): the `seq*seq*2+seq-1` an attention's
/// relative positions flatten to — and that over `div`, rounded down
/// (`inner` added first): an STFT's `(audio)/256+1` frames, a stride-2
/// conv's `(T+1)/2` — plus another symbol times `plus_scale`: the frames a
/// decoder generates past a prompt's, `(audio*-1+255)/256+duration*1-1`.
/// Bound wherever its
/// bases are ([`bind_graph`], [`specialize`]), and binds a missing base
/// where it is bound itself (an image's `h*32` rows give `h`);
/// [`generalize`] finds them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymDef {
    pub sym: u32,
    pub of: u32,
    /// A second base the size is the product with (`h*w`), if any.
    pub times: Option<u32>,
    pub scale: i64,
    /// `of`'s coefficient besides the product (a square's linear term).
    pub linear: i64,
    /// Added before the division by `div` (1: no division).
    pub inner: i64,
    pub div: i64,
    /// Another symbol, added after the division times `plus_scale`.
    pub plus: Option<u32>,
    pub plus_scale: i64,
    pub offset: i64,
}

impl SymDef {
    /// Its size when `of` is `size` (and `times`, if any, `other`); `None`
    /// below zero.
    pub fn size(&self, size: usize) -> Option<usize> {
        self.size_with(size, 1, 0)
    }

    /// Its size when `of` is `size`, `times` (if any) `other` and `plus`
    /// (if any) `plus`.
    fn size_with(&self, size: usize, other: usize, plus: usize) -> Option<usize> {
        let n = i128::from(self.scale) * size as i128 * other as i128
            + i128::from(self.linear) * size as i128
            + i128::from(self.inner);
        usize::try_from(
            n.div_euclid(i128::from(self.div.max(1)))
                + i128::from(self.plus_scale) * plus as i128
                + i128::from(self.offset),
        )
        .ok()
    }

    /// Its size where `size` gives each base's, if it gives them all.
    fn size_of(&self, size: impl Fn(u32) -> Option<usize>) -> Option<usize> {
        let other = self.times.map_or(Some(1), &size)?;
        let plus = self.plus.map_or(Some(0), &size)?;
        self.size_with(size(self.of)?, other, plus)
    }

    /// The same definition, whatever its own symbol's number.
    fn same(&self, other: &SymDef) -> bool {
        SymDef { sym: 0, ..*self } == SymDef { sym: 0, ..*other }
    }

    /// `of`'s size (and `times`'s) as `scale`, `linear` and `offset` give it.
    fn line(of: u32, times: Option<u32>, scale: i64, linear: i64, offset: i64) -> Self {
        SymDef {
            sym: 0,
            of,
            times,
            scale,
            linear,
            inner: 0,
            div: 1,
            plus: None,
            plus_scale: 0,
            offset,
        }
    }

    /// The base's size when this is `size` and the other base's (if any)
    /// `other`: `None` unless it divides out whole.
    fn base(&self, size: usize, other: usize) -> Option<usize> {
        if self.div != 1 || self.plus.is_some() {
            return None;
        }
        let per = i128::from(self.scale) * other as i128 + i128::from(self.linear);
        let rest = size as i128 - i128::from(self.offset);
        (per > 0 && rest >= 0 && rest % per == 0)
            .then(|| usize::try_from(rest / per).ok())
            .flatten()
    }
}

/// `bindings` with each derived symbol of `graph` whose bases it binds, and
/// each base a bound derived symbol determines, until nothing changes.
fn with_derived(graph: &Graph, bindings: &DimBinding) -> DimBinding {
    let mut b = bindings.clone();
    loop {
        let mut changed = false;
        for d in graph.sym_defs() {
            if b.get(d.sym).is_none()
                && let Some(n) = d.size_of(|s| b.get(s))
            {
                b.set(d.sym, n);
                changed = true;
                continue;
            }
            if d.plus.is_some() {
                continue;
            }
            let other = d.times.map(|t| b.get(t));
            match (b.get(d.sym), b.get(d.of), other) {
                (Some(n), None, None) => {
                    if let Some(v) = d.base(n, 1) {
                        b.set(d.of, v);
                        changed = true;
                    }
                }
                (Some(n), None, Some(Some(w))) => {
                    if let Some(v) = d.base(n, w) {
                        b.set(d.of, v);
                        changed = true;
                    }
                }
                (Some(n), Some(v), Some(None)) => {
                    if let (Some(t), Some(w)) = (d.times, d.base(n, v)) {
                        b.set(t, w);
                        changed = true;
                    }
                }
                _ => {}
            }
        }
        if !changed {
            return b;
        }
    }
}

/// Specialize every node's shape against `bindings` (and the symbols
/// derived from them).
///
/// Node ids are preserved (nodes are cloned in insertion order), so
/// edges and outputs remain valid without remapping.
pub fn bind_graph(graph: &Graph, bindings: &DimBinding) -> Graph {
    let bindings = &with_derived(graph, bindings);
    let mut out = Graph::new(&graph.name);
    for node in graph.nodes() {
        let bound = node.shape.bind(bindings);
        out.push_ext(
            node.op.clone(),
            node.inputs.clone(),
            bound,
            node.name.clone(),
            node.origin.clone(),
        );
    }
    out.set_outputs(graph.outputs.clone());
    out.set_sym_attrs(graph.sym_attrs().to_vec());
    out.set_vars(graph.vars().to_vec());
    out.set_sym_defs(
        graph
            .sym_defs()
            .iter()
            .filter(|d| bindings.get(d.sym).is_none())
            .copied()
            .collect(),
    );
    out
}

/// A static specialization of a symbolic `graph`: every symbol bound, the
/// shape attributes (`Reshape` / `Expand` targets) taken from the bound node
/// shapes, then every shape re-inferred.
///
/// The node shapes are the symbolic truth: an attribute is one size's copy
/// of them. Re-inferring from the attributes first would turn a reshape to
/// `[1, seq, 12, 64]` back into the size it was built at.
pub fn specialize(graph: &Graph, bindings: &DimBinding) -> Graph {
    let bindings = &with_derived(graph, bindings);
    let mut g = bind_graph(graph, bindings);
    apply_sym_attrs(&mut g, bindings);
    sync_reshape_ops(&mut g);
    sync_expand_ops(&mut g);
    sync_graph_shapes(&mut g);
    g
}

/// Write the symbolic attributes `bindings` gives a size; the rest stay.
fn apply_sym_attrs(g: &mut Graph, bindings: &DimBinding) {
    let mut left = Vec::new();
    for a in g.sym_attrs().to_vec() {
        let Some(size) = bindings.get(a.sym) else {
            left.push(a);
            continue;
        };
        #[cfg(feature = "serialize")]
        {
            let op = &g.node(a.node).op;
            match crate::rlxmodel::set_op_int(
                op,
                &a.path,
                size as i128 * i128::from(a.scale) + i128::from(a.offset),
            ) {
                Ok(op) => g.node_mut(a.node).op = op,
                Err(e) => panic!("rlx-ir: symbolic attribute of {}: {e}", a.node),
            }
        }
        #[cfg(not(feature = "serialize"))]
        panic!(
            "rlx-ir: {} has a symbolic attribute ({}); writing it needs the `serialize` feature \
             (size {size})",
            a.node,
            a.path.join(".")
        );
    }
    g.set_sym_attrs(left);
}

/// One graph from the same model built at two sizes: `a` and `b`, where
/// `sizes` lists each symbol with its size in `a` and in `b`. A dim that is
/// a symbol's size in both becomes the symbol; one that is the same in both
/// stays. [`specialize`] then makes the model at any size.
///
/// For a builder that only knows static sizes (most model builders): build
/// it twice, at sizes no other dim has (37 and 53, say), and generalize.
/// A `Reshape` / `Expand` target is re-derived from its node's shape. Any
/// other integer attribute that is a symbol's size plus a constant in both
/// (a cache write position, a slice's start) becomes a [`SymAttr`]. Anything
/// else that differs (a constant sized to the sequence, another structure)
/// is an error naming the node: the graph is not symbolic in these sizes.
pub fn generalize(a: &Graph, b: &Graph, sizes: &[(u32, usize, usize)]) -> Result<Graph, String> {
    generalize_with(a, b, sizes, None)
}

/// [`generalize`], a derived dim's offset allowed up to `max_offset` rather
/// than its scale: a convolution's padding (`4·frames + 6`) is past the
/// default. Two sizes fit any line, so a caller that widens this must check
/// the result against the model built at a third size ([`specialize`] it
/// there and compare).
pub fn generalize_with(
    a: &Graph,
    b: &Graph,
    sizes: &[(u32, usize, usize)],
    max_offset: Option<i64>,
) -> Result<Graph, String> {
    generalize_from(a, b, &[], sizes, max_offset)
}

/// [`generalize_with`] with the model built at a third size too: `sizes`
/// lists each symbol with its size in `a`, `b` and `c`. Two sizes fit a
/// line through any two points, so a dim that is a square (`2·seq² +
/// seq − 1`) can pass for a line, and a line with a large offset for a
/// square; of the fits through `a` and `b`, the one through `c` too is
/// taken. The result is still to be checked against `c` ([`specialize`]
/// it there and compare): this fixes the dims, not the rest.
pub fn generalize_checked(
    a: &Graph,
    b: &Graph,
    c: &Graph,
    sizes: &[(u32, usize, usize, usize)],
    max_offset: Option<i64>,
) -> Result<Graph, String> {
    let sizes: Vec<(u32, Vec<usize>)> = sizes
        .iter()
        .map(|&(s, x, y, z)| (s, vec![x, y, z]))
        .collect();
    generalize_many(&[a, b, c], &sizes, max_offset)
}

/// [`generalize_checked`] with the model built at any number of sizes past
/// two: each symbol's size in each of `graphs`. Of the fits through the
/// first two, the one through every other is taken — three points fit
/// `⌊n / 7⌋` where the dim is `⌈n / 8⌉` (37, 46, 61); a fourth tells them
/// apart. The result is still to be checked against them.
pub fn generalize_many(
    graphs: &[&Graph],
    sizes: &[(u32, Vec<usize>)],
    max_offset: Option<i64>,
) -> Result<Graph, String> {
    if graphs.len() < 3 || sizes.iter().any(|(_, v)| v.len() != graphs.len()) {
        return Err(
            "generalize_many takes three graphs or more, and each symbol's size in each".into(),
        );
    }
    // Two symbols whose sizes step in proportion (`37, 45, 53` and `301,
    // 333, 365`: one is `4·the other + 153` at all three) cannot be told
    // apart: a dim of one fits the other.
    for (i, (sa, va)) in sizes.iter().enumerate() {
        for (sb, vb) in &sizes[i + 1..] {
            let d = |v: &[usize]| {
                v.iter()
                    .map(|x| *x as i128 - v[0] as i128)
                    .collect::<Vec<_>>()
            };
            let (da, db) = (d(va), d(vb));
            let proportional =
                (1..da.len()).all(|j| (1..db.len()).all(|k| da[j] * db[k] == da[k] * db[j]));
            if proportional {
                return Err(format!(
                    "symbols {sa} and {sb} step in proportion across the sizes ({va:?} and \
                     {vb:?}): choose sizes that do not"
                ));
            }
        }
    }
    for g in &graphs[2..] {
        if g.len() != graphs[0].len() {
            return Err(format!(
                "the graphs differ in size ({} vs {} nodes): the builder's structure depends on \
                 the size",
                graphs[0].len(),
                g.len()
            ));
        }
    }
    let two: Vec<(u32, usize, usize)> = sizes.iter().map(|(s, v)| (*s, v[0], v[1])).collect();
    let extras: Vec<(&Graph, Vec<(u32, usize)>)> = (2..graphs.len())
        .map(|k| (graphs[k], sizes.iter().map(|(s, v)| (*s, v[k])).collect()))
        .collect();
    generalize_from(graphs[0], graphs[1], &extras, &two, max_offset)
}

fn generalize_from(
    a: &Graph,
    b: &Graph,
    extras: &[(&Graph, Vec<(u32, usize)>)],
    sizes: &[(u32, usize, usize)],
    max_offset: Option<i64>,
) -> Result<Graph, String> {
    if a.len() != b.len() {
        return Err(format!(
            "the two graphs differ in size ({} vs {} nodes): the builder's structure depends on \
             the size",
            a.len(),
            b.len()
        ));
    }
    if a.outputs != b.outputs {
        return Err("the two graphs have different outputs".into());
    }
    let mut out = Graph::new(&a.name);
    let mut sym_attrs = Vec::new();
    // Derived symbols take numbers past every symbol either graph uses.
    let mut defs: Vec<SymDef> = Vec::new();
    let mut next_sym = a
        .nodes()
        .iter()
        .chain(b.nodes())
        .flat_map(|n| n.shape.dims().iter())
        .filter_map(|d| match d {
            Dim::Dynamic(s) => Some(*s + 1),
            Dim::Static(_) => None,
        })
        .chain(sizes.iter().map(|(s, _, _)| s + 1))
        .max()
        .unwrap_or(0);
    for (na, nb) in a.nodes().iter().zip(b.nodes()) {
        let what = || format!("node {} ({})", na.id, na.op);
        if na.inputs != nb.inputs {
            return Err(format!("{}: different inputs in the two graphs", what()));
        }
        let (sa, sb) = (&na.shape, &nb.shape);
        if sa.dtype() != sb.dtype() || sa.rank() != sb.rank() {
            return Err(format!("{}: {sa} in one graph, {sb} in the other", what()));
        }
        let mut dims = Vec::with_capacity(sa.rank());
        // This dim at each further size (all of them, or none).
        let at_c = |i: usize| -> Option<Vec<usize>> {
            if extras.is_empty() {
                return None;
            }
            extras
                .iter()
                .map(
                    |(c, _)| match c.nodes()[na.id.0 as usize].shape.dims().get(i) {
                        Some(Dim::Static(z)) => Some(*z),
                        _ => None,
                    },
                )
                .collect()
        };
        let size_in = |k: usize, sym: u32| extras[k].1.iter().find(|t| t.0 == sym).map(|t| t.1);
        let fits_c = |i: usize, d: &SymDef| match at_c(i) {
            None => true,
            Some(zs) => zs
                .iter()
                .enumerate()
                .all(|(k, z)| d.size_of(|s| size_in(k, s)) == Some(*z)),
        };
        for (i, (da, db)) in sa.dims().iter().zip(sb.dims()).enumerate() {
            dims.push(match (da, db) {
                // The same at two sizes and at the third: fixed. (A
                // quotient can be the same at two: `⌈37/16⌉ = ⌈46/16⌉`.)
                (x, y)
                    if x == y
                        && at_c(i).is_none_or(|zs| zs.iter().all(|z| *x == Dim::Static(*z))) =>
                {
                    *x
                }
                (Dim::Static(x), Dim::Static(y)) => {
                    let direct = sizes.iter().find(|(sym, s_a, s_b)| {
                        s_a == x && s_b == y && fits_c(i, &SymDef::line(*sym, None, 1, 0, 0))
                    });
                    let fit = derived(*x, *y, sizes, max_offset)
                        .into_iter()
                        .find(|f| fits_c(i, f))
                        .or_else(|| {
                            // Through a third size, a quotient: an STFT's
                            // frames, a strided conv's.
                            let zs = at_c(i)?;
                            divided(*x, *y, &zs, sizes, &size_in)
                        });
                    match (direct, fit) {
                        (Some((sym, _, _)), _) => Dim::Dynamic(*sym),
                        (None, Some(f)) => {
                            let def = match defs.iter().find(|d| d.same(&f)) {
                                Some(d) => *d,
                                None => {
                                    let d = SymDef { sym: next_sym, ..f };
                                    next_sym += 1;
                                    defs.push(d);
                                    d
                                }
                            };
                            Dim::Dynamic(def.sym)
                        }
                        (None, None) => {
                            return Err(format!(
                                "{}: dim {i} is {x} at one size and {y} at the other, which is no \
                                 symbol's size ({sa} vs {sb})",
                                what()
                            ));
                        }
                    }
                }
                _ => return Err(format!("{}: dim {i} differs ({sa} vs {sb})", what())),
            });
        }
        let shape = Shape::from_dims(&dims, sa.dtype());
        let attrs_follow_shape = matches!(
            (&na.op, &nb.op),
            (Op::Reshape { .. }, Op::Reshape { .. }) | (Op::Expand { .. }, Op::Expand { .. })
        );
        // The op at every size: the two, then the further ones.
        let mut ops = vec![&na.op, &nb.op];
        ops.extend(extras.iter().map(|(c, _)| &c.nodes()[na.id.0 as usize].op));
        if ops[1..].iter().any(|op| *op != &na.op) {
            // Symbolic attributes where they are a symbol's size plus a
            // constant, so the text says `[1, cap, 8, 128]`; a Reshape /
            // Expand target that is not is still re-derived from its shape.
            // A derived symbol's size too: a slice as long as a padded
            // dim (`4·frames + 6`), or a GRU's last step (`⌈n / 64⌉ − 1`),
            // is that dim's symbol.
            let base = |sym: u32, k: usize| match k {
                0 | 1 => sizes.iter().find(|t| t.0 == sym).map(|t| [t.1, t.2][k]),
                k => size_in(k - 2, sym),
            };
            let bases: Vec<(u32, Vec<usize>)> = sizes
                .iter()
                .filter_map(|&(sym, _, _)| {
                    Some((
                        sym,
                        (0..ops.len())
                            .map(|k| base(sym, k))
                            .collect::<Option<_>>()?,
                    ))
                })
                .collect();
            let found = attr_symbols(na.id, &ops, &bases).or_else(|e| {
                let more: Vec<(u32, Vec<usize>)> = bases
                    .iter()
                    .cloned()
                    .chain(defs.iter().filter_map(|d| {
                        let at: Option<Vec<usize>> =
                            (0..ops.len()).map(|k| d.size_of(|s| base(s, k))).collect();
                        Some((d.sym, at?))
                    }))
                    .collect();
                if more.len() == bases.len() {
                    return Err(e);
                }
                attr_symbols(na.id, &ops, &more)
            });
            match found {
                Ok(attrs) => sym_attrs.extend(attrs),
                Err(_) if attrs_follow_shape => {}
                Err(e) => {
                    let other = ops[1..]
                        .iter()
                        .find(|op| **op != &na.op)
                        .expect("one differs");
                    return Err(format!(
                        "{}: the op differs between the sizes ({} vs {other}): {e}",
                        what(),
                        na.op,
                    ));
                }
            }
        }
        out.push_ext(
            na.op.clone(),
            na.inputs.clone(),
            shape,
            na.name.clone(),
            na.origin.clone(),
        );
    }
    out.set_outputs(a.outputs.clone());
    out.set_sym_attrs(sym_attrs);
    out.set_sym_defs(defs);
    Ok(out)
}

/// A dim that is `x` at one size and `y` at the other as `scale * symbol +
/// offset`, or failing that `scale * symbol * other + offset` for two
/// different symbols (`h*w*4`): a whole positive `scale`, and an `offset` no
/// larger than it (`seq*16`, `(seq-1)*16`, `seq+1`). Two sizes fit any line,
/// so a larger offset — `seq²` flattened, read as `90*seq - 1961` — is
/// refused rather than taken for one. Products need sizes chosen so no two
/// candidates fit one dim (the builder's two sizes per symbol).
/// The derived dims through `x` and `y` (a symbol's, a product's or a
/// square's), the likeliest first.
fn derived(
    x: usize,
    y: usize,
    sizes: &[(u32, usize, usize)],
    max_offset: Option<i64>,
) -> Vec<SymDef> {
    let limit = |scale: i128| max_offset.map_or(scale, |m| scale.max(i128::from(m)));
    let fit = |sa: usize, sb: usize| -> Option<(i64, i64)> {
        let (dx, ds) = (y as i128 - x as i128, sb as i128 - sa as i128);
        if ds == 0 || dx % ds != 0 {
            return None;
        }
        let scale = dx / ds;
        let offset = x as i128 - scale * sa as i128;
        (scale >= 1 && offset.abs() <= limit(scale))
            .then(|| Some((i64::try_from(scale).ok()?, i64::try_from(offset).ok()?)))
            .flatten()
    };
    // `scale·s² + linear·s + offset` through the two sizes of `s`, for
    // each scale up to what the change allows: (scale, linear, offset).
    let squares = |sa: usize, sb: usize| -> Vec<(i64, i64, i64)> {
        let (sa, sb) = (sa as i128, sb as i128);
        let (dx, ds, d2) = (y as i128 - x as i128, sb - sa, sb * sb - sa * sa);
        if ds == 0 {
            return Vec::new();
        }
        (1..=(dx / d2 + 2).clamp(1, 1 << 12))
            .filter_map(|scale| {
                let rest = dx - scale * d2;
                if rest % ds != 0 {
                    return None;
                }
                let linear = rest / ds;
                let offset = x as i128 - scale * sa * sa - linear * sa;
                (offset.abs() <= limit(scale) && linear.abs() <= limit(scale))
                    .then(|| {
                        Some((
                            i64::try_from(scale).ok()?,
                            i64::try_from(linear).ok()?,
                            i64::try_from(offset).ok()?,
                        ))
                    })
                    .flatten()
            })
            .collect()
    };
    // Two sizes fit a line through any two points, so a product (`h*w`)
    // or a square (`seq*seq`, an attention's relative positions) can pass
    // for a line with a large offset, and the reverse: the fits with the
    // smallest constants first (a line first on a tie), for a third size
    // to choose from ([`generalize_checked`]).
    let lines = sizes
        .iter()
        .filter_map(|&(sym, sa, sb)| fit(sa, sb).map(|(s, o)| (sym, None, s, 0, o)));
    let products = sizes.iter().enumerate().flat_map(|(i, &(p, pa, pb))| {
        sizes[i + 1..].iter().filter_map(move |&(q, qa, qb)| {
            (p != q)
                .then(|| fit(pa * qa, pb * qb).map(|(s, o)| (p, Some(q), s, 0, o)))
                .flatten()
        })
    });
    let squares = sizes.iter().flat_map(|&(p, pa, pb)| {
        squares(pa, pb)
            .into_iter()
            .map(move |(s, l, o)| (p, Some(p), s, l, o))
    });
    let mut fits: Vec<_> = lines.chain(products).chain(squares).collect();
    fits.sort_by_key(|(_, _, _, l, o)| l.unsigned_abs() + o.unsigned_abs());
    fits.into_iter()
        .map(|(of, times, scale, linear, offset)| SymDef::line(of, times, scale, linear, offset))
        .collect()
}

/// `floor((size + inner) / div) + offset` through `x`, `y` and `z` (a
/// dim at the three sizes): of every symbol's, the smallest `div` that
/// fits (three points fit a large one by chance), the smallest `inner`
/// for it. Three sizes of different remainders pin it.
fn divided(
    x: usize,
    y: usize,
    zs: &[usize],
    sizes: &[(u32, usize, usize)],
    size_in: &dyn Fn(usize, u32) -> Option<usize>,
) -> Option<SymDef> {
    // Each symbol's size at every point: the two, then the further ones.
    let syms: Vec<(u32, Vec<i128>)> = sizes
        .iter()
        .filter_map(|&(sym, sa, sb)| {
            let mut n = vec![sa as i128, sb as i128];
            for k in 0..zs.len() {
                n.push(size_in(k, sym)? as i128);
            }
            Some((sym, n))
        })
        .collect();
    let mut v = vec![x as i128, y as i128];
    v.extend(zs.iter().map(|z| *z as i128));
    let pts = v.len();
    // Another symbol `m` times plus, or minus, or none.
    let pluses = |m: i128| {
        let mut p: Vec<Option<(u32, &[i128], i128)>> = vec![None];
        for (q, n) in &syms {
            p.push(Some((*q, n.as_slice(), m)));
            p.push(Some((*q, n.as_slice(), -m)));
        }
        p
    };
    // A quotient alone before one with another symbol; of each, the
    // smallest offset, then divisor: a few points fit a quotient by chance,
    // with a large offset (`⌊(n + 67) / 111⌋ − t + 181`).
    let fits = |plus: &Option<(u32, &[i128], i128)>, m: i128| {
        let mut all = Vec::new();
        for div in 1..=4096i128 {
            for (sym, n) in &syms {
                for sign in [m, -m] {
                    if (div == 1 && plus.is_none()) || plus.is_some_and(|(q, ..)| q == *sym) {
                        continue; // a line is `derived`'s
                    }
                    // What is left once the other symbol is taken off.
                    let rest: Vec<i128> = (0..pts)
                        .map(|i| v[i] - plus.map_or(0, |(_, m, b)| b * m[i]))
                        .collect();
                    // Each step's growth is its size's over `div`, give or take one.
                    let close = (1..pts).all(|i| {
                        let (dn, dv) = (sign * (n[i] - n[0]), rest[i] - rest[0]);
                        (dv * div - dn).abs() <= div
                    });
                    if !close || (div > 1 && n.iter().all(|k| k % div == n[0] % div)) {
                        continue;
                    }
                    all.extend((0..div).filter_map(|inner| {
                        let off = |i: usize| rest[i] - (sign * n[i] + inner).div_euclid(div);
                        let o = off(0);
                        (1..pts).all(|i| off(i) == o).then(|| SymDef {
                            inner: inner as i64,
                            div: div as i64,
                            plus: plus.map(|(q, ..)| q),
                            plus_scale: plus.map_or(0, |(_, _, b)| b as i64),
                            ..SymDef::line(*sym, None, sign as i64, 0, o as i64)
                        })
                    }));
                }
            }
        }
        // Of fits as small, a power-of-two divisor (a stride, a hop) before
        // another: five points fit `⌊(n + 8) / 15⌋` where four stride-2
        // convs make `⌈n / 16⌉`.
        all.into_iter().min_by_key(|d| {
            (
                d.offset.unsigned_abs(),
                !(d.div as u64).is_power_of_two(),
                d.div,
                d.inner.unsigned_abs(),
            )
        })
    };
    // Each symbol once; failing that, one or both twice — the mel frames
    // of a prompt's and a generation's tokens together (`2·t + 2·p − 1`),
    // a prompt's frames padded to them (`2·t − l`).
    let at = |m: i128, b: i128| {
        fits(&None, m).or_else(|| {
            pluses(b)[1..]
                .iter()
                .filter_map(|p| fits(p, m))
                .min_by_key(|d| (d.offset.unsigned_abs(), d.div))
        })
    };
    at(1, 1)
        .or_else(|| at(2, 2))
        .or_else(|| at(2, 1))
        .or_else(|| at(1, 2))
}

/// The attributes where `a` and `b` (one op at two sizes) differ, each as a
/// symbol's size plus an offset: `x - size_a == y - size_b`. A symbol with no
/// offset is preferred.
#[cfg(feature = "serialize")]
fn attr_symbols(
    node: crate::NodeId,
    ops: &[&Op],
    sizes: &[(u32, Vec<usize>)],
) -> Result<Vec<SymAttr>, String> {
    // Every attribute that differs anywhere, with its value at each size.
    let mut paths: Vec<(Vec<String>, Vec<i128>)> = Vec::new();
    for (k, op) in ops.iter().enumerate().skip(1) {
        let diffs = crate::rlxmodel::op_int_diffs(ops[0], op)
            .ok_or("it differs in more than integer attributes, so it is not symbolic")?;
        for (path, x, y) in diffs {
            let i = match paths.iter().position(|(p, _)| *p == path) {
                Some(i) => i,
                None => {
                    paths.push((path, vec![x; ops.len()]));
                    paths.len() - 1
                }
            };
            paths[i].1[k] = y;
        }
    }
    paths
        .into_iter()
        .map(|(path, v)| {
            let fits = |sym_v: &[usize], scale: i128| {
                let off = v[0] - scale * sym_v[0] as i128;
                (1..v.len())
                    .all(|k| v[k] - scale * sym_v[k] as i128 == off)
                    .then_some(off)
            };
            let offsets = sizes
                .iter()
                .filter_map(|(sym, sv)| fits(sv, 1).map(|off| (*sym, 1, off)));
            // Else a whole multiple of a size (`2·seq`, `1000 − seq`).
            let lines = sizes.iter().filter_map(|(sym, sv)| {
                let k = (1..v.len()).find(|&k| sv[k] != sv[0])?;
                let (dv, ds) = (v[k] - v[0], sv[k] as i128 - sv[0] as i128);
                (dv != 0 && dv % ds == 0).then(|| {
                    let scale = dv / ds;
                    fits(sv, scale).map(|off| (*sym, scale, off))
                })?
            });
            let (sym, scale, off) = offsets
                .min_by_key(|(_, _, off)| off.unsigned_abs())
                .or_else(|| lines.min_by_key(|(_, s, off)| (s.unsigned_abs(), off.unsigned_abs())))
                .ok_or_else(|| {
                    format!(
                        "attribute {} is {v:?} at the sizes, which is no symbol's size times a \
                         whole number plus a constant",
                        path.join(".")
                    )
                })?;
            Ok(SymAttr {
                node,
                path,
                sym,
                scale: scale as i64,
                offset: off as i64,
            })
        })
        .collect()
}

#[cfg(not(feature = "serialize"))]
fn attr_symbols(
    _node: crate::NodeId,
    _ops: &[&Op],
    _sizes: &[(u32, Vec<usize>)],
) -> Result<Vec<SymAttr>, String> {
    Err("its attributes depend on the size; finding which needs the `serialize` feature".into())
}

/// After [`bind_graph`], sync `Op::Reshape { new_shape }` with bound node shapes.
pub fn sync_reshape_ops(graph: &mut Graph) {
    use crate::Op;
    for node in graph.nodes_mut() {
        if let Op::Reshape { new_shape } = &mut node.op {
            if node.shape.is_static() {
                *new_shape = node
                    .shape
                    .dims()
                    .iter()
                    .map(|d| d.unwrap_static() as i64)
                    .collect();
            }
        }
    }
}

/// Recompute all inferrable output shapes after binding (propagates concat fixes).
///
/// Refining a shape is the point of this pass, but *replacing* a fully-static
/// declaration with a different fully-static one is not a refinement — it is the
/// graph telling us two incompatible things, and the planner has already sized
/// arena slots from the declaration. Silently taking the inferred side is how a
/// mis-shaped `GroupedMatMul` bank (`[E, N, K]` for `[E, K, N]`) used to sail
/// through: the check in each backend's lowering compares the declared shape
/// against the operands, so rewriting the declaration first makes that check
/// compare a value with itself. In debug the IR verifier caught it; release
/// builds — the ones that ship — had no guard at all.
///
/// So: refine freely where the declaration was dynamic or absent, and reject a
/// static-vs-static contradiction — comparing only the non-unit extents, because
/// `[]` and `[1]` are the same scalar and the graph uses both spellings
/// interchangeably (a rank-0 `Reduce` result feeding a rank-1 consumer). Across
/// the workspace suite that distinction is the whole story: comparing raw dims
/// flags 20 of these harmless scalar-rank pairs for every real conflict.
pub fn sync_graph_shapes(graph: &mut Graph) {
    // Extents that actually address memory — unit axes carry no elements, so
    // they cannot be the difference between a correct and an overrunning kernel.
    fn extents(s: &Shape) -> Vec<usize> {
        s.dims()
            .iter()
            .map(|d| d.unwrap_static())
            .filter(|&v| v != 1)
            .collect()
    }
    let nodes = graph.nodes().to_vec();
    for node in &nodes {
        if let Some(shape) = crate::infer_shape::infer_output_shape(graph, node) {
            if node.shape.is_static()
                && shape.is_static()
                && extents(&node.shape) != extents(&shape)
            {
                panic!(
                    "rlx-ir: node {:?} ({:?}) output shape mismatch: declares {} but its \
                     operands give {}. These disagree and both are static, so this is a \
                     malformed graph, not a shape that needs refining — the arena is sized \
                     from the declaration while kernels read their extents from the operands, \
                     so letting it through under- or over-writes the output slot",
                    node.id, node.op, node.shape, shape
                );
            }
            graph.node_mut(node.id).shape = shape;
        }
    }
}

/// Recompute `Op::Concat` output shapes from bound inputs (fixes mixed static+dynamic axes).
pub fn sync_concat_shapes(graph: &mut Graph) {
    use crate::Op;
    let nodes = graph.nodes().to_vec();
    // Concats whose shape this pass actually corrected. Their *consumers* were
    // built from the pre-bind shape and are now stale too, so the correction has
    // to be carried forward — see the propagation loop below.
    let mut corrected: BTreeSet<crate::NodeId> = BTreeSet::new();
    for node in &nodes {
        let Op::Concat { axis } = &node.op else {
            continue;
        };
        let shapes: Vec<Shape> = node
            .inputs
            .iter()
            .map(|&id| graph.node(id).shape.clone())
            .collect();
        let refs: Vec<&Shape> = shapes.iter().collect();
        if let Ok(out) = crate::shape::concat_shape(&refs, *axis) {
            if graph.node(node.id).shape != out {
                corrected.insert(node.id);
            }
            graph.node_mut(node.id).shape = out;
        }
    }
    if corrected.is_empty() {
        return;
    }
    // Carry the correction downstream.
    //
    // `concat_shape` deliberately under-reports a mixed static+dynamic axis as
    // just `Dim::Dynamic(sym)` (it cannot add `1` to a symbol), so every node
    // derived from that concat inherits the symbol *without* the static part.
    // Binding then turns both into statics that disagree by exactly the dropped
    // amount: a decode KV concat `[1, PAST_SEQ, d] ++ [1, 1, d]` bound at
    // PAST_SEQ=4 gives a concat of 5 rows feeding a `Narrow` that still declares
    // 4. `sync_graph_shapes` sees two static shapes that disagree and panics as
    // if the graph were malformed, which is how `rlx-gemma`/`rlx-llama32`
    // dynamic decode failed.
    //
    // Only nodes actually downstream of a corrected concat are re-inferred, so
    // a genuinely malformed graph elsewhere still trips the check.
    for node in &nodes {
        if !node.inputs.iter().any(|id| corrected.contains(id)) {
            continue;
        }
        if let Some(shape) = crate::infer_shape::infer_output_shape(graph, node) {
            if graph.node(node.id).shape != shape {
                corrected.insert(node.id);
            }
            graph.node_mut(node.id).shape = shape;
        } else {
            // Shape not inferable here; treat it as still-suspect so consumers
            // downstream of it are re-inferred too rather than silently kept.
            corrected.insert(node.id);
        }
    }
}

/// Clamp `Op::Narrow` start indices after bind (template may bake in max_seq placeholders).
pub fn sync_narrow_ops(graph: &mut Graph) {
    use crate::Op;
    let nodes = graph.nodes().to_vec();
    for node in &nodes {
        let Op::Narrow { axis, start, len } = &node.op else {
            continue;
        };
        let in_shape = graph.node(node.inputs[0]).shape.clone();
        if *axis >= in_shape.rank() || !in_shape.is_static() {
            continue;
        }
        let ax_len = in_shape.dims()[*axis].unwrap_static();
        if *start + *len > ax_len {
            graph.node_mut(node.id).op = Op::Narrow {
                axis: *axis,
                start: ax_len.saturating_sub(*len),
                len: *len,
            };
        }
    }
}

/// After [`bind_graph`], sync `Op::Expand { target_shape }` with bound output shapes.
pub fn sync_expand_ops(graph: &mut Graph) {
    use crate::Op;
    let nodes = graph.nodes().to_vec();
    for node in &nodes {
        let Op::Expand { .. } = &node.op else {
            continue;
        };
        if !node.shape.is_static() {
            continue;
        }
        let target: Vec<i64> = node
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        graph.node_mut(node.id).op = Op::Expand {
            target_shape: target,
        };
    }
}

/// Infer symbol sizes from runtime input element counts.
///
/// Each `Op::Input` may have at most one dynamic dimension; its size is
/// `data_len / product(static_dims)`.
pub fn infer_bindings_from_inputs(
    graph: &Graph,
    inputs: &[(&str, usize)],
) -> Result<DimBinding, String> {
    let by_name: HashMap<&str, usize> = inputs.iter().copied().collect();
    let mut binding = DimBinding::new();
    for node in graph.nodes() {
        let Op::Input { name } = &node.op else {
            continue;
        };
        let Some(&n_elems) = by_name.get(name.as_str()) else {
            continue;
        };
        let mut static_prod: usize = 1;
        let mut dynamic_sym: Option<u32> = None;
        for d in node.shape.dims() {
            match d {
                Dim::Static(n) => static_prod *= *n,
                Dim::Dynamic(sym) => {
                    if dynamic_sym.is_some() {
                        return Err(format!(
                            "Input '{name}' has multiple dynamic dims; \
                             pass an explicit DimBinding"
                        ));
                    }
                    dynamic_sym = Some(*sym);
                }
            }
        }
        let Some(sym) = dynamic_sym else {
            continue;
        };
        if static_prod == 0 {
            return Err(format!("Input '{name}': static dim product is zero"));
        }
        if n_elems % static_prod != 0 {
            return Err(format!(
                "Input '{name}': len {n_elems} not divisible by static product {static_prod}"
            ));
        }
        let size = n_elems / static_prod;
        if let Some(prev) = binding.get(sym) {
            if prev != size {
                return Err(format!(
                    "symbol {sym} bound to {prev} and {size} from different inputs"
                ));
            }
        } else {
            binding.set(sym, size);
        }
    }
    complete_im2col_row_bindings(graph, &mut binding);
    Ok(binding)
}

/// When `sym::BATCH` is bound on NCHW inputs feeding `Op::Im2Col`, derive
/// `sym::ROWS = batch · H_out · W_out` for im2col/matmul reshape consumers.
pub fn complete_im2col_row_bindings(graph: &Graph, binding: &mut DimBinding) {
    let Some(batch) = binding.get(sym::BATCH) else {
        return;
    };
    if binding.get(sym::ROWS).is_some() {
        return;
    }
    for node in graph.nodes() {
        let Op::Im2Col {
            kernel_size,
            stride,
            padding,
            dilation,
        } = &node.op
        else {
            continue;
        };
        let x_shape = &graph.node(node.inputs[0]).shape;
        if x_shape.rank() != 4 {
            continue;
        }
        if !x_shape.dim(2).is_static() || !x_shape.dim(3).is_static() {
            continue;
        }
        let h = x_shape.dim(2).unwrap_static();
        let w = x_shape.dim(3).unwrap_static();
        let kh = kernel_size.first().copied().unwrap_or(1);
        let kw = kernel_size.get(1).copied().unwrap_or(1);
        let sh = stride.first().copied().unwrap_or(1);
        let sw = stride.get(1).copied().unwrap_or(1);
        let ph = padding.first().copied().unwrap_or(0);
        let pw = padding.get(1).copied().unwrap_or(0);
        let dh = dilation.first().copied().unwrap_or(1);
        let dw = dilation.get(1).copied().unwrap_or(1);
        let h_out = crate::shape::conv2d_spatial_output(h, kh, sh, ph, dh);
        let w_out = crate::shape::conv2d_spatial_output(w, kw, sw, pw, dw);
        binding.set(sym::ROWS, batch * h_out * w_out);
        return;
    }
}

/// Infer bindings from f32 slice lengths (convenience for tests/runtime).
pub fn infer_bindings_from_f32_inputs(
    graph: &Graph,
    inputs: &[(&str, &[f32])],
) -> Result<DimBinding, String> {
    infer_bindings_from_inputs(
        graph,
        &inputs
            .iter()
            .map(|(n, d)| (*n, d.len()))
            .collect::<Vec<_>>(),
    )
}

pub fn same_binding(a: &DimBinding, b: &DimBinding) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().all(|(sym, size)| b.get(sym) == Some(size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::GraphExt;

    /// `[1, seq, 4]` → `[seq, 4]` · `w` → `[1, seq, 3]`, then a `[2, seq, 3]`
    /// broadcast: shape attributes that hold `seq`, as a builder writes them.
    fn affine_at(seq: usize) -> Graph {
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let mut g = Graph::new("affine");
        let x = g.input("x", f(&[1, seq, 4]));
        let w = g.param("w", f(&[4, 3]));
        let flat = g.reshape(x, vec![seq as i64, 4], f(&[seq, 4]));
        let y = g.matmul(flat, w, f(&[seq, 3]));
        let y = g.reshape(y, vec![1, seq as i64, 3], f(&[1, seq, 3]));
        let wide = g.add_node(
            Op::Expand {
                target_shape: vec![2, seq as i64, 3],
            },
            vec![y],
            f(&[2, seq, 3]),
        );
        g.set_outputs(vec![wide]);
        g
    }

    #[test]
    fn two_sizes_generalize_to_one_graph_that_specializes_to_any() {
        let g = generalize(&affine_at(37), &affine_at(53), &[(0, 37, 53)]).unwrap();
        assert!(g.has_dynamic_dims());
        assert_eq!(g.node(g.outputs[0]).shape.dims()[1], Dim::Dynamic(0));
        let mut b = DimBinding::new();
        b.set(0, 10);
        // Exactly the graph the builder makes at 10.
        assert_eq!(
            format!("{}", specialize(&g, &b)),
            format!("{}", affine_at(10))
        );
    }

    /// A pad block `[l − 5, 96]` flattened (`96·l − 480`) and an attention's
    /// relative positions (`2·l² + l − 1`): through 8 and 11 each fits a
    /// line and a square; the size 14 tells which.
    #[test]
    fn a_third_size_tells_a_line_from_a_square() {
        let at = |l: usize| {
            let mut g = Graph::new("rel");
            let pad = g.input("pad", Shape::new(&[96 * (l - 5)], DType::F32));
            let rel = g.input("rel", Shape::new(&[2 * l * l + l - 1], DType::F32));
            g.set_outputs(vec![pad, rel]);
            g
        };
        let g =
            generalize_checked(&at(8), &at(11), &at(14), &[(0, 8, 11, 14)], Some(1 << 16)).unwrap();
        let defs: Vec<_> = g
            .sym_defs()
            .iter()
            .map(|d| (d.of, d.times, d.scale, d.linear, d.offset))
            .collect();
        assert_eq!(defs, [(0, None, 96, 0, -480), (0, Some(0), 2, 1, -1)]);
        let mut b = DimBinding::new();
        b.set(0, 40);
        assert_eq!(format!("{}", specialize(&g, &b)), format!("{}", at(40)));
    }

    /// A flow decoder's dims over all its tokens `s`, its prompt's `p` and
    /// its prompt's mel frames `l`: the generated tokens (`s − p`), the
    /// relative positions of all of them upsampled (`2·s − 1`) and the
    /// prompt's frames padded to the whole (`2·s − l`) — a symbol twice
    /// beside another once.
    #[test]
    fn a_dim_of_a_symbol_twice_and_another_is_derived() {
        let at = |s: usize, p: usize, l: usize| {
            let mut g = Graph::new("flow");
            let tok = g.input("token", Shape::new(&[s - p], DType::F32));
            let prompt = g.input("prompt", Shape::new(&[p], DType::F32));
            let feat = g.input("feat", Shape::new(&[l], DType::F32));
            let rel = g.input("rel", Shape::new(&[2 * s - 1], DType::F32));
            let pad = g.input("pad", Shape::new(&[2 * s - l], DType::F32));
            g.set_outputs(vec![tok, prompt, feat, rel, pad]);
            g
        };
        let builds = [(9, 4, 8), (17, 9, 19), (11, 5, 9), (18, 7, 15)];
        let graphs: Vec<Graph> = builds.iter().map(|&(s, p, l)| at(s, p, l)).collect();
        let refs: Vec<&Graph> = graphs.iter().collect();
        let sizes = [
            (0, builds.iter().map(|b| b.0).collect()),
            (1, builds.iter().map(|b| b.1).collect()),
            (2, builds.iter().map(|b| b.2).collect()),
        ];
        let g = generalize_many(&refs, &sizes, Some(1 << 16)).unwrap();
        let mut b = DimBinding::new();
        b.set(0, 63);
        b.set(1, 23);
        b.set(2, 47);
        assert_eq!(
            format!("{}", specialize(&g, &b)),
            format!("{}", at(63, 23, 47))
        );
    }

    #[test]
    fn a_size_dependent_attribute_or_dim_is_named_not_guessed() {
        // A constant holding one value per position is not symbolic in it.
        let table_at = |seq: usize| {
            let mut g = Graph::new("table");
            let c = g.add_node(
                Op::Constant {
                    data: vec![0u8; seq * 4],
                },
                vec![],
                Shape::new(&[seq], DType::F32),
            );
            g.set_outputs(vec![c]);
            g
        };
        let e = generalize(&table_at(37), &table_at(53), &[(0, 37, 53)]).unwrap_err();
        assert!(e.contains("node %0") && e.contains("op differs"), "{e}");
        // A dim one more than a symbol's size is a symbol derived from it
        // (`seq+1`), bound with it.
        let g = generalize(&affine_at(37), &affine_at(53), &[(0, 36, 52)]).unwrap();
        assert_eq!(
            g.sym_defs(),
            [SymDef {
                sym: 1,
                of: 0,
                times: None,
                scale: 1,
                linear: 0,
                inner: 0,
                div: 1,
                plus: None,
                plus_scale: 0,
                offset: 1
            }]
        );
        let mut b = DimBinding::new();
        b.set(0, 9);
        assert_eq!(
            format!("{}", specialize(&g, &b)),
            format!("{}", affine_at(10))
        );
    }

    #[test]
    fn a_padded_length_is_derived_only_when_the_caller_allows_its_offset() {
        // A convolution padded by 3 on each side of `4·frames` positions.
        let padded_at = |frames: usize| affine_at(4 * frames + 6);
        let e = generalize(&padded_at(5), &padded_at(7), &[(0, 5, 7)]).unwrap_err();
        assert!(e.contains("no symbol's size"), "{e}");
        let g = generalize_with(&padded_at(5), &padded_at(7), &[(0, 5, 7)], Some(16)).unwrap();
        assert_eq!(
            g.sym_defs(),
            [SymDef {
                sym: 1,
                of: 0,
                times: None,
                scale: 4,
                linear: 0,
                inner: 0,
                div: 1,
                plus: None,
                plus_scale: 0,
                offset: 6
            }]
        );
        let mut b = DimBinding::new();
        b.set(0, 3);
        assert_eq!(
            format!("{}", specialize(&g, &b)),
            format!("{}", padded_at(3))
        );
    }

    #[test]
    fn bind_graph_specializes_matmul() {
        let batch = sym::BATCH;
        let seq = sym::SEQ;
        let mut g = Graph::new("dyn");
        let x = g.input("x", Shape::batch_seq(batch, seq, 4, DType::F32));
        let w = g.param("w", Shape::new(&[4, 8], DType::F32));
        let y = g.mm(x, w);
        g.set_outputs(vec![y]);

        assert!(has_dynamic_dims(&g));
        let binding = DimBinding::batch_seq(2, 16);
        let bound = bind_graph(&g, &binding);
        assert!(!has_dynamic_dims(&bound));
        assert_eq!(
            bound.node(bound.outputs[0]).shape,
            Shape::new(&[2, 16, 8], DType::F32)
        );
    }

    #[test]
    fn infer_bindings_from_input_data() {
        let mut g = Graph::new("dyn");
        let x = g.input(
            "x",
            Shape::from_dims(
                &[Dim::Static(3), Dim::Dynamic(sym::SEQ), Dim::Static(64)],
                DType::F32,
            ),
        );
        g.set_outputs(vec![x]);

        let b = infer_bindings_from_f32_inputs(&g, &[("x", &vec![0.0f32; 3 * 128 * 64])])
            .expect("infer");
        assert_eq!(b.get(sym::SEQ), Some(128));
    }

    #[test]
    fn infer_bindings_sets_im2col_rows_from_batch() {
        let mut g = Graph::new("im2col_rows");
        let x = g.input(
            "x",
            Shape::from_dims(
                &[
                    Dim::Dynamic(sym::BATCH),
                    Dim::Static(1),
                    Dim::Static(4),
                    Dim::Static(4),
                ],
                DType::F32,
            ),
        );
        let _col = g.im2col(x, [3, 3], [1, 1], [1, 1], [1, 1]);
        g.set_outputs(vec![x]);
        let b = infer_bindings_from_f32_inputs(&g, &[("x", &[0.0f32; 2 * 16])]).expect("infer");
        assert_eq!(b.get(sym::BATCH), Some(2));
        assert_eq!(b.get(sym::ROWS), Some(2 * 4 * 4));
    }
}
