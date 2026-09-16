// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! rlx-ir → sequential-engine descriptors.
//!
//! Three passes:
//!
//! 1. **Absorb.** Bias adds and output activations fold into the descriptor of
//!    the op they follow, so `Conv → Binary(Add) → Activation(Relu)` is one
//!    stage, not three.
//! 2. **Place.** Every value gets an address range in the activation RAM.
//!    `Reshape` and `Transpose` never move data — they are back-propagated into
//!    the producing stage's destination strides, which is why the flatten in
//!    front of an LSTM costs nothing. `Concat` forces its operands to be
//!    adjacent, which is what lets a concatenated tap run be one dot product.
//! 3. **Emit.** Walk the ops and write descriptors.
//!
//! Padding shows up as `Concat` with zero-valued params; those simply reserve
//! space in a zero-initialised RAM, so the address generator needs no bounds
//! check.

use std::collections::HashMap;

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{Graph, NodeId, Op};

use super::{Descriptor, SeqAct, SeqConfig, SeqModel, SeqOp, quantise_pow2};

/// Where a value lives: `c` rows of `l` elements at `base`, strided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Place {
    base: usize,
    sc: usize,
    sl: usize,
    c: usize,
    l: usize,
}

impl Place {
    fn contiguous(base: usize, c: usize, l: usize) -> Self {
        Self {
            base,
            sc: l,
            sl: 1,
            c,
            l,
        }
    }
    fn len(&self) -> usize {
        self.c * self.l
    }
    fn end(&self) -> usize {
        self.base + self.len()
    }
}

fn idx(id: NodeId) -> usize {
    id.0 as usize
}

/// Collapse a shape to `(rows, cols)`. Conv activations here are `[N,C,H,W]`
/// with one spatial axis degenerate, which is the length-in-H convention the
/// rest of rlx uses for 1-D convolution.
fn dims_of(shape: &rlx_ir::Shape) -> Result<(usize, usize), String> {
    let d: Vec<usize> = shape
        .dims()
        .iter()
        .map(|x| {
            if x.is_static() {
                Ok(x.unwrap_static())
            } else {
                Err("rlx-fpga/seq: dynamic dimensions are not supported".to_string())
            }
        })
        .collect::<Result<_, _>>()?;
    Ok(match d.len() {
        1 => (1, d[0]),
        2 if d[0] == 1 => (1, d[1]),
        2 => (d[0], d[1]),
        3 => (d[1], d[2]),
        4 => (d[1], d[2] * d[3]),
        _ => return Err(format!("rlx-fpga/seq: unsupported rank {}", d.len())),
    })
}

struct Ctx<'a> {
    graph: &'a Graph,
    params: HashMap<&'a str, &'a [f32]>,
    /// Value placement, keyed by node index.
    place: HashMap<usize, Place>,
    /// Nodes folded into another stage's descriptor.
    absorbed: Vec<bool>,
    /// Carried state: output node index → input node index it updates.
    alias: HashMap<usize, usize>,
    /// Activation RAM bump pointer.
    arena: usize,
    weights: Vec<i16>,
    /// Param name → (offset into `weights`, fractional bits).
    wmap: HashMap<String, (usize, u32)>,
    offsets: Vec<u16>,
    descriptors: Vec<Descriptor>,
    labels: Vec<String>,
}

impl<'a> Ctx<'a> {
    fn param(&self, id: NodeId) -> Option<&'a [f32]> {
        match &self.graph.nodes()[idx(id)].op {
            Op::Param { name } => self.params.get(name.as_str()).copied(),
            _ => None,
        }
    }

    fn param_name(&self, id: NodeId) -> Option<&'a str> {
        match &self.graph.nodes()[idx(id)].op {
            Op::Param { name } => self.params.get_key_value(name.as_str()).map(|(k, _)| *k),
            _ => None,
        }
    }

    /// Quantise a param once and return `(offset, frac_bits)`.
    fn intern(&mut self, name: &str, values: &[f32]) -> (usize, u32) {
        if let Some(&hit) = self.wmap.get(name) {
            return hit;
        }
        let (q, frac) = quantise_pow2(values);
        let at = self.weights.len();
        self.weights.extend_from_slice(&q);
        self.wmap.insert(name.to_string(), (at, frac));
        (at, frac)
    }

    /// Intern under a synthetic name, for params the lowering reorders.
    fn intern_owned(&mut self, name: String, values: &[f32]) -> (usize, u32) {
        if let Some(&hit) = self.wmap.get(&name) {
            return hit;
        }
        let (q, frac) = quantise_pow2(values);
        let at = self.weights.len();
        self.weights.extend_from_slice(&q);
        self.wmap.insert(name, (at, frac));
        (at, frac)
    }

    /// Follow aliases to the node that actually owns the storage. Chains occur
    /// when a stage absorbs both a bias add and an activation.
    fn resolve(&self, mut n: usize) -> usize {
        for _ in 0..8 {
            match self.alias.get(&n) {
                Some(&next) if next != n => n = next,
                _ => break,
            }
        }
        n
    }

    fn get_place(&self, n: usize) -> Option<Place> {
        self.place.get(&self.resolve(n)).copied()
    }

    fn alloc(&mut self, n: usize, c: usize, l: usize) -> Place {
        let p = Place::contiguous(self.arena, c, l);
        self.arena += p.len();
        self.set_place(n, p);
        p
    }

    /// Record a placement, pushing it back through views so the stage that
    /// actually produces the data writes straight into final storage.
    fn set_place(&mut self, n: usize, p: Place) {
        let n = self.resolve(n);
        self.place.insert(n, p);
        let node = &self.graph.nodes()[n];
        match &node.op {
            // A reshape is the same bytes. If it does not change the (rows,
            // cols) split the strides carry over untouched; otherwise the
            // region has to be densely packed for a re-split to be expressible.
            // A single-row view is dense whatever its row stride says, which is
            // the case for one operand of a wider concat.
            Op::Reshape { .. } => {
                let src = idx(node.inputs[0]);
                if let Ok((c, l)) = dims_of(&self.graph.nodes()[src].shape) {
                    if (c, l) == (p.c, p.l) {
                        self.set_place(src, p);
                    } else if p.sl == 1 && (p.c == 1 || p.sc == p.l) && c * l == p.len() {
                        self.set_place(src, Place::contiguous(p.base, c, l));
                    }
                }
            }
            // A transpose is an index swap: give the source the mirrored strides
            // so its producer scatters into the transposed layout for free.
            Op::Transpose { .. } => {
                let src = idx(node.inputs[0]);
                if let Ok((c, l)) = dims_of(&self.graph.nodes()[src].shape) {
                    let q = if (c, l) == (p.l, p.c) {
                        Place {
                            base: p.base,
                            sc: p.sl,
                            sl: p.sc,
                            c,
                            l,
                        }
                    } else {
                        Place {
                            base: p.base,
                            sc: p.sc,
                            sl: p.sl,
                            c,
                            l,
                        }
                    };
                    self.set_place(src, q);
                }
            }
            _ => {}
        }
    }
}

/// Chase a bias operand back through the reshape/expand that broadcasts it.
fn bias_source(graph: &Graph, mut id: NodeId) -> NodeId {
    loop {
        match &graph.nodes()[idx(id)].op {
            Op::Reshape { .. } | Op::Expand { .. } => {
                id = graph.nodes()[idx(id)].inputs[0];
            }
            _ => return id,
        }
    }
}

pub fn lower_graph(
    graph: &Graph,
    params: &[(String, Vec<f32>)],
    cfg: &SeqConfig,
) -> Result<SeqModel, String> {
    let nodes = graph.nodes();
    let n = nodes.len();

    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node) in nodes.iter().enumerate() {
        for &inp in &node.inputs {
            consumers[idx(inp)].push(i);
        }
    }

    let mut ctx = Ctx {
        graph,
        params: params
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect(),
        place: HashMap::new(),
        absorbed: vec![false; n],
        alias: HashMap::new(),
        arena: 0,
        weights: Vec::new(),
        wmap: HashMap::new(),
        offsets: Vec::new(),
        descriptors: Vec::new(),
        labels: Vec::new(),
    };

    // ---- carried state: an output and the input it updates share storage ----
    let outs = &graph.outputs;
    for (name, out_ix) in &cfg.carry {
        let out = *outs
            .get(*out_ix)
            .ok_or_else(|| format!("rlx-fpga/seq: carry output index {out_ix} is out of range"))?;
        let inp = nodes
            .iter()
            .position(|nd| matches!(&nd.op, Op::Input { name: nm } if nm == name))
            .ok_or_else(|| format!("rlx-fpga/seq: no input named {name} to carry"))?;
        ctx.alias.insert(idx(out), inp);
    }

    // ---- pass 1: fold bias adds and activations into their producer --------
    // `stage_bias[i]` / `stage_act[i]` describe the descriptor for node i, and
    // the folded nodes are marked absorbed so the emit pass skips them.
    let mut stage_bias: Vec<Option<NodeId>> = vec![None; n];
    let mut stage_act: Vec<SeqAct> = vec![SeqAct::None; n];
    let mut stage_out: Vec<usize> = (0..n).collect();
    for i in 0..n {
        if !matches!(nodes[i].op, Op::Conv { .. } | Op::MatMul) {
            continue;
        }
        let mut tail = i;
        if consumers[tail].len() == 1 {
            let c = consumers[tail][0];
            if let Op::Binary(BinaryOp::Add) = &nodes[c].op {
                let other = nodes[c].inputs.iter().copied().find(|&x| idx(x) != tail);
                if let Some(other) = other {
                    let src = bias_source(graph, other);
                    if ctx.param(src).is_some() {
                        stage_bias[i] = Some(src);
                        ctx.absorbed[c] = true;
                        ctx.alias.insert(c, i);
                        // Mark the broadcast chain absorbed too.
                        let mut w = other;
                        while idx(w) != idx(src) {
                            ctx.absorbed[idx(w)] = true;
                            w = nodes[idx(w)].inputs[0];
                        }
                        tail = c;
                    }
                }
            }
        }
        if consumers[tail].len() == 1 {
            let c = consumers[tail][0];
            if let Op::Activation(kind) = &nodes[c].op {
                let act = match kind {
                    Activation::Relu => Some(SeqAct::Relu),
                    Activation::Sigmoid => Some(SeqAct::Sigmoid),
                    _ => None,
                };
                // A gate's activations belong to the Gate descriptor, not here.
                let feeds_gate =
                    consumers[c].len() == 1 && matches!(nodes[consumers[c][0]].op, Op::Binary(_));
                if let Some(act) = act {
                    if !feeds_gate {
                        stage_act[i] = act;
                        ctx.absorbed[c] = true;
                        ctx.alias.insert(c, i);
                        tail = c;
                    }
                }
            }
        }
        stage_out[i] = tail;
    }

    // ---- pass 2: placement ------------------------------------------------
    // Feature input first, so the host writes a known address range.
    let feat_node = nodes
        .iter()
        .position(
            |nd| matches!(&nd.op, Op::Input { name } if !cfg.carry.iter().any(|(c, _)| c == name)),
        )
        .ok_or("rlx-fpga/seq: no feature input")?;
    let (fc, fl) = dims_of(&nodes[feat_node].shape)?;
    let feat_place = ctx.alloc(feat_node, fc, fl);
    let (feat_base, feat_len) = (feat_place.base, feat_place.len());

    // Concats fix adjacency, so resolve them before anything else is placed.
    let mut concat_perm: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let Op::Concat { .. } = &nodes[i].op else {
            continue;
        };
        let (cc, _) = dims_of(&nodes[i].shape)?;
        let operands: Vec<usize> = nodes[i].inputs.iter().map(|&x| idx(x)).collect();
        let lens: Vec<usize> = operands
            .iter()
            .map(|&o| dims_of(&nodes[o].shape).map(|(_, l)| l))
            .collect::<Result<_, _>>()?;
        let total: usize = lens.iter().sum();
        let placed: Vec<Option<Place>> = operands.iter().map(|&o| ctx.get_place(o)).collect();

        if placed.iter().all(Option::is_none) {
            // Fresh block; each operand takes a column range.
            let base = ctx.arena;
            ctx.arena += cc * total;
            let mut col = 0;
            for (k, &o) in operands.iter().enumerate() {
                let (oc, _) = dims_of(&nodes[o].shape)?;
                let p = Place {
                    base: base + col,
                    sc: total,
                    sl: 1,
                    c: oc,
                    l: lens[k],
                };
                if ctx.param(NodeId(o as u32)).is_none() {
                    ctx.set_place(o, p);
                }
                col += lens[k];
            }
            ctx.place.insert(
                i,
                Place {
                    base,
                    sc: total,
                    sl: 1,
                    c: cc,
                    l: total,
                },
            );
        } else if placed.iter().all(Option::is_some) {
            // Everything already lives somewhere: accept any order that happens
            // to be contiguous and permute the consumer's weights instead.
            let mut order: Vec<usize> = (0..operands.len()).collect();
            order.sort_by_key(|&k| placed[k].unwrap().base);
            let mut cursor = placed[order[0]].unwrap().base;
            for &k in &order {
                let p = placed[k].unwrap();
                if p.base != cursor {
                    return Err(format!(
                        "rlx-fpga/seq: concat %{i} operands are not contiguous in memory; \
                         the sequential engine reads a concatenated tap run as one span"
                    ));
                }
                cursor += p.len();
            }
            let base = placed[order[0]].unwrap().base;
            ctx.place.insert(
                i,
                Place {
                    base,
                    sc: total,
                    sl: 1,
                    c: cc,
                    l: total,
                },
            );
            if order != (0..operands.len()).collect::<Vec<_>>() {
                concat_perm.insert(i, order);
            }
        } else {
            // Extend: the placed operands must already sit at the arena tail.
            let tail: usize = placed.iter().flatten().map(Place::end).max().unwrap();
            if tail != ctx.arena {
                return Err(format!(
                    "rlx-fpga/seq: concat %{i} cannot extend storage that is not at the arena tail"
                ));
            }
            let base = placed.iter().flatten().map(|p| p.base).min().unwrap();
            let mut col = base;
            for (k, &o) in operands.iter().enumerate() {
                let (oc, _) = dims_of(&nodes[o].shape)?;
                match placed[k] {
                    Some(p) => col = p.end(),
                    None => {
                        let p = Place {
                            base: col,
                            sc: lens[k],
                            sl: 1,
                            c: oc,
                            l: lens[k],
                        };
                        if ctx.param(NodeId(o as u32)).is_none() {
                            ctx.set_place(o, p);
                        }
                        ctx.arena = ctx.arena.max(p.end());
                        col = p.end();
                    }
                }
            }
            ctx.place.insert(
                i,
                Place {
                    base,
                    sc: total,
                    sl: 1,
                    c: cc,
                    l: total,
                },
            );
        }
    }

    // Pin everything whose address has to outlive a single inference before
    // any of it can be mistaken for scratch. Carried state is the trap: `h1`
    // and `h2` happen to sit in concats and so are placed already, but `c1`
    // and `c2` are only read by the gate that updates them, so a live-range
    // analysis scoped to one frame sees them die and hands their space away.
    // The result survives frame 0 and diverges from frame 1 on.
    for (name, _) in &cfg.carry {
        let inp = nodes
            .iter()
            .position(|nd| matches!(&nd.op, Op::Input { name: nm } if nm == name))
            .ok_or_else(|| format!("rlx-fpga/seq: no input named {name} to carry"))?;
        if ctx.get_place(inp).is_none() {
            let (c, l) = dims_of(&nodes[inp].shape)?;
            ctx.alloc(inp, c, l);
        }
    }
    for &out in &graph.outputs {
        if ctx.get_place(idx(out)).is_none() {
            let (c, l) = dims_of(&nodes[idx(out)].shape)?;
            ctx.alloc(idx(out), c, l);
        }
    }

    // Everything still unplaced is scratch: written by one stage, read by the
    // next, dead thereafter. Giving each its own block is what a bump
    // allocator does and it is wasteful — the conv intermediates alone are
    // nearly a third of the activation RAM and never coexist. Allocate them by
    // live range instead, so a dead buffer's space is reused.
    //
    // Concat operands, carried state, the feature input and the output are
    // already placed above and are not touched here: their addresses are
    // pinned by adjacency or by persisting across inferences.
    let mut scratch: Vec<(usize, usize, usize)> = Vec::new(); // (def, node, words)
    for i in 0..n {
        if ctx.absorbed[i] || ctx.place.contains_key(&ctx.resolve(i)) {
            continue;
        }
        match &nodes[i].op {
            Op::Param { .. } | Op::Constant { .. } => continue,
            Op::Reshape { .. } | Op::Transpose { .. } => {
                continue;
            }
            _ => {}
        }
        let (c, l) = dims_of(&nodes[i].shape)?;
        scratch.push((i, i, c * l));
    }

    // Last stage that can still read node `i`, following views and absorbed
    // nodes to whatever finally consumes them.
    let mut last_use: Vec<usize> = (0..n).collect();
    for i in (0..n).rev() {
        let mut end = i;
        for &c in &consumers[i] {
            let through = matches!(
                nodes[c].op,
                Op::Reshape { .. } | Op::Transpose { .. } | Op::Concat { .. } | Op::Narrow { .. }
            ) || ctx.absorbed[c];
            end = end.max(if through { last_use[c] } else { c });
        }
        last_use[i] = end;
    }

    // Linear scan over a free list. Buffers are freed when their last reader
    // has run, and reused first-fit.
    let scratch_base = ctx.arena;
    let mut free: Vec<(usize, usize)> = Vec::new(); // (offset, words)
    let mut active: Vec<(usize, usize, usize)> = Vec::new(); // (end, offset, words)
    let mut top = 0usize;
    scratch.sort_by_key(|&(def, ..)| def);
    for (def, node, words) in scratch {
        // Retire anything whose last reader ran before this stage.
        active.retain(|&(end, off, w)| {
            if end < def {
                free.push((off, w));
                false
            } else {
                true
            }
        });
        free.sort_unstable();
        // Coalesce, so a run of small dead buffers can host a large one.
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for (off, w) in free.drain(..) {
            match merged.last_mut() {
                Some((poff, pw)) if *poff + *pw == off => *pw += w,
                _ => merged.push((off, w)),
            }
        }
        free = merged;

        let slot = free.iter().position(|&(_, w)| w >= words);
        let off = match slot {
            Some(k) => {
                let (off, w) = free[k];
                if w == words {
                    free.remove(k);
                } else {
                    free[k] = (off + words, w - words);
                }
                off
            }
            None => {
                let off = top;
                top += words;
                off
            }
        };
        let (c, l) = dims_of(&nodes[node].shape)?;
        ctx.set_place(node, Place::contiguous(scratch_base + off, c, l));
        active.push((last_use[node], off, words));
    }
    ctx.arena = scratch_base + top;
    // Views inherit from their source once everything concrete is placed.
    for i in 0..n {
        if ctx.place.contains_key(&ctx.resolve(i)) {
            continue;
        }
        if let Op::Reshape { .. } | Op::Transpose { .. } = &nodes[i].op {
            if let Some(p) = ctx.get_place(idx(nodes[i].inputs[0])) {
                let (c, l) = dims_of(&nodes[i].shape)?;
                let q = if matches!(nodes[i].op, Op::Transpose { .. }) && (c, l) == (p.l, p.c) {
                    Place {
                        base: p.base,
                        sc: p.sl,
                        sl: p.sc,
                        c,
                        l,
                    }
                } else {
                    Place {
                        base: p.base,
                        sc: p.sc,
                        sl: p.sl,
                        c,
                        l,
                    }
                };
                ctx.place.insert(ctx.resolve(i), q);
            }
        }
    }

    lower_ops(
        &mut ctx,
        &consumers,
        &stage_bias,
        &stage_act,
        &concat_perm,
        cfg,
    )?;

    let prob_node = *outs.first().ok_or("rlx-fpga/seq: graph has no outputs")?;
    let prob = ctx
        .get_place(idx(prob_node))
        .ok_or("rlx-fpga/seq: output was never placed")?;

    ctx.descriptors.push(Descriptor {
        op: SeqOp::Done as u32,
        ..Default::default()
    });
    ctx.labels.push("done".into());

    if ctx.arena > cfg.aram_words {
        return Err(format!(
            "rlx-fpga/seq: activation RAM needs {} words, configured for {}",
            ctx.arena, cfg.aram_words
        ));
    }

    Ok(SeqModel {
        descriptors: ctx.descriptors,
        weights: ctx.weights,
        offsets: ctx.offsets,
        aram_words: cfg.aram_words,
        aram_used: ctx.arena,
        feat_base,
        feat_len,
        prob_addr: prob.base,
        cfg: cfg.clone(),
        labels: ctx.labels,
    })
}

/// Raw static dims, for ops that care about the spatial layout `dims_of`
/// flattens away.
fn raw_dims(shape: &rlx_ir::Shape) -> Vec<usize> {
    shape.dims().iter().map(|d| d.unwrap_static()).collect()
}

#[allow(clippy::too_many_lines)]
fn lower_ops(
    ctx: &mut Ctx<'_>,
    consumers: &[Vec<usize>],
    stage_bias: &[Option<NodeId>],
    stage_act: &[SeqAct],
    concat_perm: &HashMap<usize, Vec<usize>>,
    cfg: &SeqConfig,
) -> Result<(), String> {
    let nodes = ctx.graph.nodes();
    let af = cfg.act_frac;

    for i in 0..nodes.len() {
        if ctx.absorbed[i] {
            continue;
        }
        match &nodes[i].op {
            Op::Conv {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                if padding.iter().any(|&p| p != 0) {
                    return Err(format!(
                        "rlx-fpga/seq: conv %{i} has implicit padding; express it as a Concat of \
                         zeros so the address generator can fold it into a base offset"
                    ));
                }
                if dilation.iter().any(|&d| d != 1) {
                    return Err(format!("rlx-fpga/seq: conv %{i} dilation is unsupported"));
                }
                let src = idx(nodes[i].inputs[0]);
                let wid = nodes[i].inputs[1];
                let name = ctx
                    .param_name(wid)
                    .ok_or_else(|| format!("rlx-fpga/seq: conv %{i} weight is not a Param"))?;
                let wv = ctx.param(wid).unwrap().to_vec();
                let (base_w, w_frac) = ctx.intern(name, &wv);

                let ip = ctx
                    .get_place(src)
                    .ok_or_else(|| format!("rlx-fpga/seq: conv %{i} input is unplaced"))?;
                let op_place = ctx
                    .get_place(i)
                    .ok_or_else(|| format!("rlx-fpga/seq: conv %{i} output is unplaced"))?;

                let win = raw_dims(&nodes[src].shape);
                let wout = raw_dims(&nodes[i].shape);
                let (c_in, c_out) = (win[1], wout[1]);
                let (kh, kw) = (kernel_size[0], *kernel_size.get(1).unwrap_or(&1));
                let sh = stride[0];
                let n_i = wout[2] * wout[3];

                let mut d = Descriptor {
                    op: SeqOp::MatVec as u32,
                    act: stage_act[i] as u32,
                    base_a: ip.base as u32,
                    base_w: base_w as u32,
                    w_frac,
                    base_d: op_place.base as u32,
                    n_i: n_i as u32,
                    ..Default::default()
                };

                // Order matters: a 3x3 window over a single plane also satisfies
                // `groups == c_in == c_out == 1`, but its taps are not
                // contiguous along the length axis, so it must be recognised
                // as a 2-D window first.
                if kh > 1 && kw > 1 {
                    if c_in != 1 || c_out != 1 {
                        return Err(format!(
                            "rlx-fpga/seq: conv %{i} is a multi-channel 2-D window \
                             (c_in={c_in}, c_out={c_out}); only single-plane is supported"
                        ));
                    }
                    // Single-plane 2-D window: the tap table carries the row jump.
                    let off_ptr = ctx.offsets.len();
                    for ki in 0..kh {
                        for kj in 0..kw {
                            ctx.offsets.push((ki * win[3] + kj) as u16);
                        }
                    }
                    d.n_o = 1;
                    d.n_tap = (kh * kw) as u32;
                    d.use_off = 1;
                    d.off_ptr = off_ptr as u32;
                    d.sa_i = (stride.last().copied().unwrap_or(1) * ip.sl) as u32;
                    d.sw_t = 1;
                    d.sd_i = op_place.sl as u32;
                } else if *groups == c_in && c_in == c_out {
                    // Depthwise: one channel in, one out, taps along the length.
                    d.n_o = c_out as u32;
                    d.n_tap = (kh * kw) as u32;
                    d.sa_o = ip.sc as u32;
                    d.sa_i = (sh * ip.sl) as u32;
                    d.sa_t = ip.sl as u32;
                    d.sw_o = (kh * kw) as u32;
                    d.sw_t = 1;
                    d.sd_o = op_place.sc as u32;
                    d.sd_i = op_place.sl as u32;
                } else if kh == 1 && kw == 1 && *groups == 1 {
                    // Pointwise: taps run across input channels.
                    d.n_o = c_out as u32;
                    d.n_tap = c_in as u32;
                    d.sa_i = ip.sl as u32;
                    d.sa_t = ip.sc as u32;
                    d.sw_o = c_in as u32;
                    d.sw_t = 1;
                    d.sd_o = op_place.sc as u32;
                    d.sd_i = op_place.sl as u32;
                } else {
                    return Err(format!(
                        "rlx-fpga/seq: conv %{i} is neither depthwise, pointwise, nor \
                         single-plane (c_in={c_in}, c_out={c_out}, groups={groups})"
                    ));
                }

                apply_bias(ctx, &mut d, stage_bias[i], af, w_frac, true)?;
                ctx.descriptors.push(d);
                ctx.labels.push(format!("conv %{i}"));
            }

            Op::Pool {
                kernel_size,
                stride,
                padding,
                ..
            } => {
                if padding.iter().any(|&p| p != 0) {
                    return Err(format!("rlx-fpga/seq: pool %{i} padding is unsupported"));
                }
                let src = idx(nodes[i].inputs[0]);
                let ip = ctx
                    .get_place(src)
                    .ok_or("rlx-fpga/seq: pool input unplaced")?;
                let op_place = ctx
                    .get_place(i)
                    .ok_or("rlx-fpga/seq: pool output unplaced")?;
                let wout = raw_dims(&nodes[i].shape);
                let k: usize = kernel_size.iter().product();
                ctx.descriptors.push(Descriptor {
                    op: SeqOp::Pool as u32,
                    n_o: wout[1] as u32,
                    n_i: (wout[2] * wout[3]) as u32,
                    n_tap: k as u32,
                    base_a: ip.base as u32,
                    sa_o: ip.sc as u32,
                    sa_i: (stride[0] * ip.sl) as u32,
                    sa_t: ip.sl as u32,
                    base_d: op_place.base as u32,
                    sd_o: op_place.sc as u32,
                    sd_i: op_place.sl as u32,
                    ..Default::default()
                });
                ctx.labels.push(format!("pool %{i}"));
            }

            Op::MatMul => {
                let src = idx(nodes[i].inputs[0]);
                let wid = nodes[i].inputs[1];
                let name = ctx
                    .param_name(wid)
                    .ok_or_else(|| format!("rlx-fpga/seq: matmul %{i} weight is not a Param"))?
                    .to_string();
                let wv = ctx.param(wid).unwrap().to_vec();
                let ip = ctx
                    .get_place(src)
                    .ok_or("rlx-fpga/seq: matmul input unplaced")?;
                let wdims = raw_dims(&nodes[wid.0 as usize].shape);
                let (k, out_n) = (wdims[0], wdims[1]);

                // If the input concat is stored in a different order than the
                // graph wrote it, permute the contraction rows to match memory
                // rather than copying the activations.
                let (base_w, w_frac) = match concat_perm.get(&src) {
                    Some(order) => {
                        let seg: Vec<usize> = nodes[src]
                            .inputs
                            .iter()
                            .map(|&x| dims_of(&nodes[idx(x)].shape).map(|(_, l)| l))
                            .collect::<Result<_, _>>()?;
                        let starts: Vec<usize> = seg
                            .iter()
                            .scan(0, |a, &l| {
                                let s = *a;
                                *a += l;
                                Some(s)
                            })
                            .collect();
                        let mut rows = Vec::with_capacity(k);
                        for &o in order {
                            for r in 0..seg[o] {
                                rows.push(starts[o] + r);
                            }
                        }
                        let mut perm = Vec::with_capacity(wv.len());
                        for &r in &rows {
                            perm.extend_from_slice(&wv[r * out_n..(r + 1) * out_n]);
                        }
                        ctx.intern_owned(format!("{name}#memorder"), &perm)
                    }
                    None => ctx.intern(&name, &wv),
                };

                let op_place = ctx
                    .get_place(i)
                    .ok_or("rlx-fpga/seq: matmul output unplaced")?;
                let mut d = Descriptor {
                    op: SeqOp::MatVec as u32,
                    act: stage_act[i] as u32,
                    n_o: 1,
                    n_i: out_n as u32,
                    n_tap: k as u32,
                    base_a: ip.base as u32,
                    sa_t: ip.sl as u32,
                    base_w: base_w as u32,
                    sw_i: 1,
                    sw_t: out_n as u32,
                    w_frac,
                    base_d: op_place.base as u32,
                    sd_i: op_place.sl as u32,
                    ..Default::default()
                };
                apply_bias(ctx, &mut d, stage_bias[i], af, w_frac, false)?;
                ctx.descriptors.push(d);
                ctx.labels.push(format!("matmul %{i}"));

                if let Some(g) = detect_gate(ctx, consumers, i)? {
                    ctx.descriptors.push(g);
                    ctx.labels.push(format!("gate %{i}"));
                }
            }

            Op::Input { .. }
            | Op::Param { .. }
            | Op::Constant { .. }
            | Op::Reshape { .. }
            | Op::Transpose { .. }
            | Op::Expand { .. }
            | Op::Concat { .. }
            | Op::Narrow { .. } => {}

            other => {
                return Err(format!(
                    "rlx-fpga/seq: %{i} {other:?} has no sequential-engine lowering"
                ));
            }
        }
    }
    Ok(())
}

/// Fold a bias into the descriptor, resolving the shift that lifts it from its
/// own scale into the accumulator's.
///
/// `per_channel` selects which loop indexes the bias: a conv's bias is one
/// value per output channel (the `oo` loop), a matvec's is one per output (the
/// `ii` loop).
fn apply_bias(
    ctx: &mut Ctx<'_>,
    d: &mut Descriptor,
    bias: Option<NodeId>,
    act_frac: u32,
    w_frac: u32,
    per_channel: bool,
) -> Result<(), String> {
    let Some(bid) = bias else { return Ok(()) };
    let name = ctx
        .param_name(bid)
        .ok_or("rlx-fpga/seq: bias is not a Param")?
        .to_string();
    let bv = ctx.param(bid).unwrap().to_vec();
    let (base_b, b_frac) = ctx.intern(&name, &bv);
    let shift = i64::from(act_frac) + i64::from(w_frac) - i64::from(b_frac);
    if shift < 0 {
        return Err(format!(
            "rlx-fpga/seq: bias {name} needs a negative shift ({shift}); it carries more \
             fractional bits than the accumulator holds"
        ));
    }
    d.has_bias = 1;
    d.base_b = base_b as u32;
    d.b_shift = shift as u32;
    if per_channel {
        d.sb_o = 1;
    } else {
        d.sb_i = 1;
    }
    Ok(())
}

/// Recognise the LSTM cell that follows a gate projection and emit one `Gate`
/// descriptor for it.
///
/// The graph writes the cell out in full — four `Narrow`s, four activations,
/// two multiplies, an add, a `tanh` and a final multiply. All of it collapses
/// into a single hardware stage, so every node in the pattern is marked
/// absorbed. Returns `None` when the projection is an ordinary matmul.
fn detect_gate(
    ctx: &mut Ctx<'_>,
    consumers: &[Vec<usize>],
    proj: usize,
) -> Result<Option<Descriptor>, String> {
    let nodes = ctx.graph.nodes();
    // Consumers of the projection *after* its bias add was absorbed.
    let tail = consumers[proj]
        .iter()
        .copied()
        .find(|&c| ctx.absorbed[c])
        .map_or(proj, |c| consumers[c].first().copied().map_or(c, |_| c));
    let outs = &consumers[tail];
    if outs.len() != 4 {
        return Ok(None);
    }
    let mut narrows = [usize::MAX; 4];
    let hidden = match &nodes[outs[0]].op {
        Op::Narrow { len, .. } => *len,
        _ => return Ok(None),
    };
    for &c in outs {
        let Op::Narrow {
            axis: _,
            start,
            len,
        } = &nodes[c].op
        else {
            return Ok(None);
        };
        if *len != hidden || start % hidden != 0 {
            return Ok(None);
        }
        let k = start / hidden;
        if k > 3 {
            return Ok(None);
        }
        narrows[k] = c;
    }
    if narrows.contains(&usize::MAX) {
        return Ok(None);
    }

    // Each narrow feeds exactly one activation: i, f, o are sigmoid, g is tanh.
    let mut acts = [usize::MAX; 4];
    for (k, &nw) in narrows.iter().enumerate() {
        if consumers[nw].len() != 1 {
            return Ok(None);
        }
        let a = consumers[nw][0];
        let want = if k == 2 {
            Activation::Tanh
        } else {
            Activation::Sigmoid
        };
        match &nodes[a].op {
            Op::Activation(kind) if *kind == want => acts[k] = a,
            _ => return Ok(None),
        }
    }

    // f·c_prev — the operand that is not the forget gate is the cell state.
    let fc = consumers[acts[1]]
        .iter()
        .copied()
        .find(|&c| matches!(&nodes[c].op, Op::Binary(BinaryOp::Mul)))
        .ok_or("rlx-fpga/seq: LSTM forget gate does not feed a multiply")?;
    let c_prev = nodes[fc]
        .inputs
        .iter()
        .copied()
        .map(idx)
        .find(|&x| x != acts[1])
        .ok_or("rlx-fpga/seq: cannot identify the cell state")?;
    let ig = consumers[acts[0]]
        .iter()
        .copied()
        .find(|&c| matches!(&nodes[c].op, Op::Binary(BinaryOp::Mul)))
        .ok_or("rlx-fpga/seq: LSTM input gate does not feed a multiply")?;
    let add = consumers[fc]
        .iter()
        .copied()
        .find(|&c| matches!(&nodes[c].op, Op::Binary(BinaryOp::Add)))
        .ok_or("rlx-fpga/seq: LSTM cell update has no add")?;
    let tanh_c = consumers[add]
        .iter()
        .copied()
        .find(|&c| matches!(&nodes[c].op, Op::Activation(Activation::Tanh)))
        .ok_or("rlx-fpga/seq: LSTM cell state is not passed through tanh")?;
    let h_new = consumers[tanh_c]
        .iter()
        .copied()
        .find(|&c| matches!(&nodes[c].op, Op::Binary(BinaryOp::Mul)))
        .ok_or("rlx-fpga/seq: LSTM output gate does not feed a multiply")?;

    let z = ctx
        .get_place(tail)
        .ok_or("rlx-fpga/seq: gate projection is unplaced")?;
    let cp = ctx
        .get_place(c_prev)
        .ok_or("rlx-fpga/seq: cell state is unplaced")?;
    let hp = ctx
        .get_place(h_new)
        .ok_or("rlx-fpga/seq: hidden state is unplaced")?;

    for nd in narrows
        .iter()
        .chain(acts.iter())
        .chain([&fc, &ig, &add, &tanh_c, &h_new])
    {
        ctx.absorbed[*nd] = true;
    }
    // The new cell state overwrites the old in place.
    ctx.alias.insert(add, c_prev);

    Ok(Some(Descriptor {
        op: SeqOp::Gate as u32,
        n_o: 1,
        n_i: hidden as u32,
        base_a: z.base as u32,
        base_c: cp.base as u32,
        base_d: hp.base as u32,
        ..Default::default()
    }))
}
