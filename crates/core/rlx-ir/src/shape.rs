// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tensor shapes with static and dynamic dimensions.
//!
//! Shapes are first-class in RLX IR — every node's output shape is known
//! (or symbolically bounded) at graph construction time. This enables
//! buffer size computation for memory planning.

use crate::DType;
use smallvec::SmallVec;

/// A single dimension — either a concrete size or a symbolic dynamic dim.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dim {
    /// Known at graph construction time.
    Static(usize),
    /// Unknown until runtime. Identified by a symbol index so that
    /// `Dim::Dynamic(0)` in two shapes means "same unknown size".
    Dynamic(u32),
}

impl Dim {
    pub fn unwrap_static(self) -> usize {
        match self {
            Self::Static(n) => n,
            Self::Dynamic(s) => panic!("expected static dim, got dynamic symbol {s}"),
        }
    }

    pub fn is_static(self) -> bool {
        matches!(self, Self::Static(_))
    }
}

impl From<usize> for Dim {
    fn from(n: usize) -> Self {
        Self::Static(n)
    }
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static(n) => write!(f, "{n}"),
            Self::Dynamic(s) => write!(f, "?{s}"),
        }
    }
}

/// Tensor shape: ordered list of dimensions + element type.
///
/// SmallVec<[Dim; 4]> avoids heap allocation for up to 4D tensors (the common case).
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: SmallVec<[Dim; 4]>,
    dtype: DType,
}

impl Shape {
    /// Create a shape from static dimensions.
    pub fn new(dims: &[usize], dtype: DType) -> Self {
        Self {
            dims: dims.iter().map(|&d| Dim::Static(d)).collect(),
            dtype,
        }
    }

    /// Create a shape with mixed static/dynamic dimensions.
    pub fn from_dims(dims: &[Dim], dtype: DType) -> Self {
        Self {
            dims: dims.into(),
            dtype,
        }
    }

    /// Scalar (0-dimensional).
    pub fn scalar(dtype: DType) -> Self {
        Self {
            dims: SmallVec::new(),
            dtype,
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn dims(&self) -> &[Dim] {
        &self.dims
    }
    pub fn dim(&self, i: usize) -> Dim {
        self.dims.get(i).copied().unwrap_or_else(|| {
            let dims: Vec<_> = self.dims.iter().map(|d| d.unwrap_static()).collect();
            panic!(
                "Shape::dim({i}) out of bounds for rank {} dims={dims:?}",
                self.rank()
            );
        })
    }

    /// How many positions a gather/scatter along `axis` takes, when `self` is
    /// the shape of the INDEX operand.
    ///
    /// Gather's output rank is `data_rank - 1 + index_rank`, so an index may be
    /// either:
    ///
    ///   * rank > `axis` — the usual case, where the index carries a dimension
    ///     at the gather axis and that dimension is the count; or
    ///   * a flat list (typically rank 1) that has no dimension at `axis` and
    ///     whose whole extent IS the count.
    ///
    /// Every backend previously wrote `idx_shape.dim(axis)` here, which panics
    /// with "Shape::dim(1) out of bounds for rank 1" on the second form — so a
    /// rank-1 gather built a correct forward graph and then blew up the moment
    /// it needed a backward pass. Sharing the rule keeps the five backends that
    /// need it from disagreeing about it.
    pub fn gather_index_count(&self, axis: usize) -> usize {
        if self.rank() > axis {
            self.dim(axis).unwrap_static()
        } else {
            self.dims
                .iter()
                .map(|d| d.unwrap_static())
                .product::<usize>()
                .max(1)
        }
    }

    /// Set of dynamic dim symbols this shape references. Useful for
    /// "what bindings does this graph need?" queries on inputs.
    pub fn dynamic_symbols(&self) -> Vec<u32> {
        let mut syms: Vec<u32> = self
            .dims
            .iter()
            .filter_map(|d| match d {
                Dim::Dynamic(s) => Some(*s),
                _ => None,
            })
            .collect();
        syms.sort();
        syms.dedup();
        syms
    }

    /// Specialize the shape against a binding (`symbol → static
    /// size`). Unknown symbols stay [`Dim::Dynamic`]. Plan #54: the
    /// step that takes a "compile once, run at any seq length" graph
    /// and produces the runtime-specific concrete shape.
    pub fn bind(&self, bindings: &DimBinding) -> Self {
        let dims = self
            .dims
            .iter()
            .map(|d| match d {
                Dim::Dynamic(s) => match bindings.get(*s) {
                    Some(n) => Dim::Static(n),
                    None => *d,
                },
                _ => *d,
            })
            .collect();
        Self {
            dims,
            dtype: self.dtype,
        }
    }

    /// Total number of elements (only if all dims are static).
    pub fn num_elements(&self) -> Option<usize> {
        let mut total = 1usize;
        for d in &self.dims {
            match d {
                Dim::Static(n) => total = total.checked_mul(*n)?,
                Dim::Dynamic(_) => return None,
            }
        }
        Some(total)
    }

    /// Total size in bytes (only if all dims are static).
    pub fn size_bytes(&self) -> Option<usize> {
        self.num_elements().map(|n| n * self.dtype.size_bytes())
    }

    /// True if all dimensions are statically known.
    pub fn is_static(&self) -> bool {
        self.dims.iter().all(|d| d.is_static())
    }

    /// Replace a dimension.
    pub fn with_dim(mut self, axis: usize, dim: Dim) -> Self {
        self.dims[axis] = dim;
        self
    }

    /// Change dtype (for cast operations).
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Numpy-style broadcast with another shape (fusion / lowering).
    pub fn broadcast_with(&self, other: &Shape) -> Result<Shape, String> {
        broadcast(self, other)
    }
}

/// Pack leading-dim broadcast metadata for DiT modulation kernels
/// ([`Op::AdaLayerNorm`](crate::op::Op::AdaLayerNorm) /
/// [`Op::GatedResidual`](crate::op::Op::GatedResidual)).
///
/// Layout: `[lead_rank, x_lead[0..8], mod_lead[0..8]]` — right-aligns
/// `mod_dims` to `x_dims` with size-1 padding (NumPy broadcast). Last axis
/// of both shapes is the feature dim and is omitted from the pack.
pub fn ada_modulation_lead_pack(x_dims: &[usize], mod_dims_in: &[usize]) -> [u32; 17] {
    let rank = x_dims.len();
    assert!(rank >= 1, "AdaLayerNorm/GatedResidual: rank ≥ 1");
    let lead = rank - 1;
    assert!(
        lead <= 8,
        "AdaLayerNorm/GatedResidual: at most 8 leading dims"
    );
    let pad = rank
        .checked_sub(mod_dims_in.len())
        .expect("modulation rank exceeds x");
    let mut mod_dims = vec![1usize; rank];
    mod_dims[pad..].copy_from_slice(mod_dims_in);
    let mut pack = [0u32; 17];
    pack[0] = lead as u32;
    for j in 0..lead {
        pack[1 + j] = x_dims[j] as u32;
        pack[9 + j] = mod_dims[j] as u32;
    }
    pack
}

/// Launch geometry for DiT modulation backward kernels: one block/threadgroup
/// per unique modulation row; that unit loops `seq_per_mod` feature-rows that
/// share it (typical DiT `[B,S,D]` / `[B,1,D]` → `(B, S)`).
pub fn ada_modulation_launch(x_dims: &[usize], mod_dims: &[usize]) -> (u32, u32) {
    assert!(!x_dims.is_empty() && !mod_dims.is_empty());
    let xr = x_dims.len() - 1;
    let mr = mod_dims.len() - 1;
    let mut seq = 1u32;
    let mut mods = 1u32;
    for i in 0..xr {
        let xd = x_dims[i] as u32;
        let md = if i + mr >= xr {
            mod_dims[i - (xr - mr)] as u32
        } else {
            1
        };
        if md == 1 && xd > 1 {
            seq = seq.saturating_mul(xd);
        } else {
            mods = mods.saturating_mul(xd.max(1));
        }
    }
    (mods.max(1), seq.max(1))
}

// ── Shape inference functions ────────────────────────────────────────────

/// Numpy-style broadcast of two shapes. Returns the broadcast result.
pub fn broadcast(a: &Shape, b: &Shape) -> Result<Shape, String> {
    let max_rank = a.rank().max(b.rank());
    let mut dims = SmallVec::new();
    for i in 0..max_rank {
        let ad = if i < max_rank - a.rank() {
            Dim::Static(1)
        } else {
            a.dims[i - (max_rank - a.rank())]
        };
        let bd = if i < max_rank - b.rank() {
            Dim::Static(1)
        } else {
            b.dims[i - (max_rank - b.rank())]
        };
        let d = broadcast_dim(ad, bd)?;
        dims.push(d);
    }
    Ok(Shape {
        dims,
        dtype: a.dtype,
    })
}

fn broadcast_dim(a: Dim, b: Dim) -> Result<Dim, String> {
    match (a, b) {
        (Dim::Static(1), d) | (d, Dim::Static(1)) => Ok(d),
        (Dim::Static(x), Dim::Static(y)) if x == y => Ok(Dim::Static(x)),
        (Dim::Static(x), Dim::Static(y)) => Err(format!("cannot broadcast {x} with {y}")),
        (Dim::Dynamic(s), Dim::Dynamic(t)) if s == t => Ok(Dim::Dynamic(s)),
        (Dim::Dynamic(_), _) | (_, Dim::Dynamic(_)) => Ok(a), // keep first dynamic
    }
}

/// Operand dims of a [`crate::op::Op::GroupedMatMul`], as every backend needs
/// them: `input [.., M, K] · weight[idx[r]] [K, N] → out [M, N]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedMatMulDims {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub num_experts: usize,
}

/// Validate a [`crate::op::Op::GroupedMatMul`]'s operands and derive `(M, K, N, E)`.
///
/// The expert bank is `[E, K, N]` — K **second**, N last — which is the
/// transpose of the `[out, in]` layout checkpoints ship. Getting that backwards
/// is the classic MoE porting bug and, until this check existed, an entirely
/// silent one: a bank left in `[E, N, K]` has the right rank and the right
/// element count, so nothing rejects it. The kernels then read `N` off
/// `weight.dim(2)` — which is really K — and write `M·K` floats into the `M·N`
/// slot the planner sized from the declared output shape. When `K < N` that
/// leaves the tail of every output row holding whatever the arena had (looks
/// like a non-causal model); when `K > N` it runs off the end of the slot and
/// corrupts the neighbouring tensor.
///
/// So: K must agree between the input and the bank, and — when `out` is given —
/// the declared output must be `M × N`.
pub fn grouped_matmul_dims(
    input: &Shape,
    weight: &Shape,
    out: Option<&Shape>,
) -> Result<GroupedMatMulDims, String> {
    if input.rank() < 2 {
        return Err(format!(
            "GroupedMatMul input must be rank >= 2 ([.., M, K]), got {input}"
        ));
    }
    if weight.rank() != 3 {
        return Err(format!(
            "GroupedMatMul expert bank must be rank 3 ([E, K, N]), got {weight}"
        ));
    }
    let statics = |d: Dim, what: &str| match d {
        Dim::Static(v) => Ok(v),
        Dim::Dynamic(s) => Err(format!(
            "GroupedMatMul {what} must be static, got dynamic symbol {s}"
        )),
    };
    let m = statics(input.dim(input.rank() - 2), "M")?;
    let k = statics(input.dim(input.rank() - 1), "K")?;
    let num_experts = statics(weight.dim(0), "E")?;
    let k_w = statics(weight.dim(1), "the bank's K axis")?;
    let n = statics(weight.dim(2), "N")?;

    if k != k_w {
        return Err(format!(
            "GroupedMatMul K mismatch: input is {input} (K={k}) but the expert bank \
             is {weight} (K={k_w}, N={n}). The bank must be [E, K, N]; a checkpoint \
             tensor stored [E, N, K] ([out, in], the usual `x @ W.T` layout) has to \
             be transposed before it reaches this op"
        ));
    }
    if let Some(out) = out {
        let out_n = statics(out.dim(out.rank().saturating_sub(1)), "the output's N")?;
        let out_elems = out.num_elements().unwrap_or(0);
        if out_n != n || out_elems != m * n {
            return Err(format!(
                "GroupedMatMul output shape mismatch: declared {out} but the operands \
                 give [{m}, {n}] (input {input}, bank {weight})"
            ));
        }
    }
    Ok(GroupedMatMulDims {
        m,
        k,
        n,
        num_experts,
    })
}

/// `[.., M, K]` × `[E, K, N]` → `[M, N]`. See [`grouped_matmul_dims`].
pub fn grouped_matmul_shape(input: &Shape, weight: &Shape) -> Result<Shape, String> {
    let d = grouped_matmul_dims(input, weight, None)?;
    Ok(Shape::new(&[d.m, d.n], input.dtype()))
}

/// MatMul output shape: `[..,M,K] × [..,K,N] → [..,M,N]`.
pub fn matmul_shape(lhs: &Shape, rhs: &Shape) -> Result<Shape, String> {
    if lhs.rank() < 2 || rhs.rank() < 2 {
        return Err(format!(
            "matmul requires rank >= 2, got {} and {}",
            lhs.rank(),
            rhs.rank()
        ));
    }
    let m = lhs.dims[lhs.rank() - 2];
    let k1 = lhs.dims[lhs.rank() - 1];
    let k2 = rhs.dims[rhs.rank() - 2];
    let n = rhs.dims[rhs.rank() - 1];

    // Verify K dimensions match
    match (k1, k2) {
        (Dim::Static(a), Dim::Static(b)) if a != b => {
            return Err(format!("matmul K mismatch: {a} vs {b}"));
        }
        (Dim::Dynamic(s), Dim::Dynamic(t)) if s != t => {
            return Err(format!("matmul K mismatch: ?{s} vs ?{t}"));
        }
        _ => {}
    }

    // Broadcast batch dimensions
    let lhs_batch = &lhs.dims[..lhs.rank() - 2];
    let rhs_batch = &rhs.dims[..rhs.rank() - 2];
    let batch_a = Shape::from_dims(lhs_batch, lhs.dtype);
    let batch_b = Shape::from_dims(rhs_batch, rhs.dtype);
    let batch = if lhs_batch.is_empty() && rhs_batch.is_empty() {
        SmallVec::new()
    } else if lhs_batch.is_empty() {
        rhs_batch.into()
    } else if rhs_batch.is_empty() {
        lhs_batch.into()
    } else {
        broadcast(&batch_a, &batch_b)?.dims.clone()
    };

    let mut dims = batch;
    dims.push(m);
    dims.push(n);
    Ok(Shape {
        dims,
        dtype: lhs.dtype,
    })
}

/// GGUF [`crate::Op::DequantMatMul`] output shape: `[.., M, K] × [N, K] → [.., M, N]`.
///
/// The weight is `[n, k]`, **not** the `[k, n]` [`matmul_shape`] assumes: GGUF
/// stores a linear's weight in `[out_dim, in_dim]` order (GGML `ne = [in, out]`),
/// so the op contracts the *last* axis of both operands. The same order is why
/// `Graph::dequant_grouped_matmul_packed` can take an expert bank straight from
/// a `ffn_*_exps.weight` blob with no transpose.
///
/// Usually a GGUF weight reaches the graph as a rank-1 packed byte blob whose
/// shape says nothing about `n` or `k` — the block layout carries that, and the
/// caller declares the output shape directly. This rule is for the case where a
/// caller declares the *logical* rank-2 shape instead.
pub fn dequant_matmul_shape(lhs: &Shape, rhs: &Shape) -> Result<Shape, String> {
    if lhs.rank() < 2 {
        return Err(format!(
            "dequant matmul activations require rank >= 2, got {}",
            lhs.rank()
        ));
    }
    if rhs.rank() != 2 {
        return Err(format!(
            "dequant matmul weight must be rank 2 `[n, k]`, got rank {}",
            rhs.rank()
        ));
    }
    let m = lhs.dims[lhs.rank() - 2];
    let k_lhs = lhs.dims[lhs.rank() - 1];
    let n = rhs.dims[0];
    let k_rhs = rhs.dims[1];

    match (k_lhs, k_rhs) {
        (Dim::Static(a), Dim::Static(b)) if a != b => {
            return Err(format!(
                "dequant matmul K mismatch: {a} vs {b} — the GGUF weight is \
                 `[n, k]`, so K is its last axis, not its first"
            ));
        }
        (Dim::Dynamic(s), Dim::Dynamic(t)) if s != t => {
            return Err(format!("dequant matmul K mismatch: ?{s} vs ?{t}"));
        }
        _ => {}
    }

    let mut dims: SmallVec<[Dim; 4]> = lhs.dims[..lhs.rank() - 2].into();
    dims.push(m);
    dims.push(n);
    Ok(Shape {
        dims,
        dtype: lhs.dtype,
    })
}

/// ONNX Expand: broadcast `input` to `target` (numpy-style).
pub fn expand_shape(input: &Shape, target: &[i64]) -> Result<Shape, String> {
    if target.iter().any(|&d| d < 0) {
        return Err("expand target has negative dim".into());
    }
    let target_s = Shape::new(
        &target.iter().map(|&d| d as usize).collect::<Vec<_>>(),
        input.dtype(),
    );
    broadcast(input, &target_s)
}

/// Binary element-wise shape (broadcast).
pub fn binary_shape(lhs: &Shape, rhs: &Shape) -> Result<Shape, String> {
    broadcast(lhs, rhs)
}

/// Unary op: output = input shape.
pub fn unary_shape(input: &Shape) -> Shape {
    input.clone()
}

/// Cast: change dtype, keep shape.
pub fn cast_shape(input: &Shape, to: DType) -> Shape {
    input.clone().with_dtype(to)
}

/// Compare: broadcast + Bool dtype.
pub fn compare_shape(lhs: &Shape, rhs: &Shape) -> Result<Shape, String> {
    Ok(broadcast(lhs, rhs)?.with_dtype(DType::Bool))
}

/// Reduce along axes.
pub fn reduce_shape(input: &Shape, axes: &[usize], keep_dim: bool) -> Result<Shape, String> {
    let mut dims = SmallVec::new();
    for (i, &d) in input.dims.iter().enumerate() {
        if axes.contains(&i) {
            if keep_dim {
                dims.push(Dim::Static(1));
            }
        } else {
            dims.push(d);
        }
    }
    Ok(Shape {
        dims,
        dtype: input.dtype,
    })
}

/// Softmax: preserves shape.
pub fn softmax_shape(input: &Shape) -> Shape {
    input.clone()
}

/// Transpose: permute dims.
pub fn transpose_shape(input: &Shape, perm: &[usize]) -> Result<Shape, String> {
    if perm.len() != input.rank() {
        return Err(format!("perm len {} != rank {}", perm.len(), input.rank()));
    }
    let dims: SmallVec<[Dim; 4]> = perm.iter().map(|&i| input.dims[i]).collect();
    Ok(Shape {
        dims,
        dtype: input.dtype,
    })
}

/// Narrow: slice along one axis.
pub fn narrow_shape(input: &Shape, axis: usize, len: usize) -> Result<Shape, String> {
    if axis >= input.rank() {
        return Err(format!("axis {axis} >= rank {}", input.rank()));
    }
    Ok(input.clone().with_dim(axis, Dim::Static(len)))
}

/// Tile: axis `i` grows by factor `reps[i]` (aligned to the input rank).
pub fn tile_shape(input: &Shape, reps: &[usize]) -> Result<Shape, String> {
    if reps.len() != input.rank() {
        return Err(format!(
            "tile: reps length {} != rank {}",
            reps.len(),
            input.rank()
        ));
    }
    let mut out = input.clone();
    for (axis, &r) in reps.iter().enumerate() {
        if r == 1 {
            continue;
        }
        match input.dims[axis] {
            Dim::Static(n) => out = out.with_dim(axis, Dim::Static(n * r)),
            Dim::Dynamic(_) => return Err(format!("tile: cannot tile dynamic axis {axis}")),
        }
    }
    Ok(out)
}

/// Strided slice: output axis `axis` has length `len` (the number of strided
/// reads); all other axes unchanged.
pub fn slice_shape(input: &Shape, axis: usize, len: usize) -> Result<Shape, String> {
    if axis >= input.rank() {
        return Err(format!("slice: axis {axis} >= rank {}", input.rank()));
    }
    Ok(input.clone().with_dim(axis, Dim::Static(len)))
}

/// Pad: each axis `i` grows by `pads[i][0] + pads[i][1]`. `pads` is aligned to
/// the input rank. Padding a dynamic axis is rejected (the padded extent isn't
/// statically known).
pub fn pad_shape(input: &Shape, pads: &[[usize; 2]]) -> Result<Shape, String> {
    if pads.len() != input.rank() {
        return Err(format!(
            "pad: pads length {} != rank {}",
            pads.len(),
            input.rank()
        ));
    }
    let mut out = input.clone();
    for (axis, &[before, after]) in pads.iter().enumerate() {
        if before == 0 && after == 0 {
            continue;
        }
        match input.dims[axis] {
            Dim::Static(n) => out = out.with_dim(axis, Dim::Static(n + before + after)),
            Dim::Dynamic(_) => {
                return Err(format!(
                    "pad: cannot pad dynamic axis {axis} by ({before},{after})"
                ));
            }
        }
    }
    Ok(out)
}

/// Concat along axis.
pub fn concat_shape(inputs: &[&Shape], axis: usize) -> Result<Shape, String> {
    if inputs.is_empty() {
        return Err("concat: no inputs".into());
    }
    let base = inputs[0];
    let mut static_sum = 0usize;
    let mut dyn_sym: Option<u32> = None;
    for s in inputs {
        if s.rank() == 0 {
            return Err("concat: input has rank 0".into());
        }
        if s.rank() != base.rank() {
            return Err(format!(
                "concat: rank mismatch {} vs {}",
                s.rank(),
                base.rank()
            ));
        }
        let ax = axis.min(s.rank().saturating_sub(1));
        // Every axis OTHER than the concat axis must agree, as it does in
        // numpy and torch. This used to go unchecked, and the output simply
        // inherited `inputs[0]`'s dims — so a graph that concatenated a
        // [B,1,1,31] pad onto a [B,1,C,T] activation was accepted and silently
        // declared [B,1,1,·], collapsing the channel axis. CPU/Metal/wgpu/
        // CoreML all ran it; only MLX refused, so the defect looked like a
        // backend gap instead of a malformed graph (`rlx-eeginceptionerp`).
        // Dynamic dims are skipped: their extent is not known here.
        for d in 0..s.rank() {
            if d == ax {
                continue;
            }
            if let (Dim::Static(a), Dim::Static(b)) = (s.dims[d], base.dims[d])
                && a != b
            {
                return Err(format!(
                    "concat: axis {d} mismatch {a} vs {b} (concat axis is {axis});                      all axes but the concat axis must match"
                ));
            }
        }
        match s.dims[ax] {
            Dim::Static(n) => static_sum += n,
            Dim::Dynamic(sym) => {
                if let Some(prev) = dyn_sym {
                    if prev != sym {
                        return Err(format!(
                            "concat: mismatched dynamic symbols {prev} vs {sym} on axis {axis}"
                        ));
                    }
                }
                dyn_sym = Some(sym);
            }
        }
    }
    let out_dim = match dyn_sym {
        None => Dim::Static(static_sum),
        Some(sym) if static_sum == 0 => Dim::Dynamic(sym),
        Some(sym) => {
            // Mixed static + dynamic (e.g. conv_state || qkv). After `bind_graph`,
            // `sync_concat_shapes` recomputes from concrete input shapes.
            let _ = static_sum;
            Dim::Dynamic(sym)
        }
    };
    let out_axis = axis.min(base.rank().saturating_sub(1));
    Ok(base.clone().with_dim(out_axis, out_dim))
}

/// Gather (embedding lookup): table\[V,D\] + indices\[B,S\] → \[B,S,D\].
pub fn gather_shape(table: &Shape, indices: &Shape, axis: usize) -> Result<Shape, String> {
    if axis >= table.rank() {
        return Err(format!("gather: axis {axis} >= rank {}", table.rank()));
    }
    // ONNX Gather output = table[:axis] ++ indices.shape ++ table[axis+1:].
    // The leading `table[:axis]` dims were previously DROPPED — correct only for
    // axis 0 (empty prefix); an axis!=0 gather like `[4,3]` on axis 1 wrongly
    // collapsed to `[6]` instead of `[4,6]` (StyleTTS2 nearest-resize / any
    // batched/channel gather).
    let mut dims: SmallVec<[Dim; 4]> = table.dims[..axis].iter().copied().collect();
    dims.extend(indices.dims.iter().copied());
    for i in (axis + 1)..table.rank() {
        dims.push(table.dims[i]);
    }
    Ok(Shape {
        dims,
        dtype: table.dtype,
    })
}

/// Reshape with -1 wildcard support.
pub fn reshape_shape(input: &Shape, new_shape: &[i64]) -> Result<Shape, String> {
    let neg_count = new_shape.iter().filter(|&&d| d == -1).count();
    if neg_count > 1 {
        return Err("reshape: at most one -1".into());
    }

    if input.is_static() {
        // Only the `-1` case needs the input's element count. Computing `total`
        // eagerly made a reshape to a FULLY-CONCRETE target fail whenever the
        // input's product couldn't be formed (a stray over-large/garbage static
        // dim from an upstream mis-resolved dynamic axis → `num_elements()` = None
        // via checked_mul), aborting the whole import ("input has dynamic dims").
        // A fully-specified target shape is well-defined regardless of the input.
        let known_product: i64 = new_shape.iter().filter(|&&d| d != -1).product();
        let mut dims = SmallVec::new();
        for &d in new_shape {
            if d == -1 {
                let total = input
                    .num_elements()
                    .ok_or_else(|| "reshape: input has dynamic dims".to_string())?;
                let inferred = total as i64 / known_product.max(1);
                dims.push(Dim::Static(inferred as usize));
            } else if d < 0 {
                return Err(format!("reshape: invalid dim {d}"));
            } else {
                dims.push(Dim::Static(d as usize));
            }
        }
        // A reshape must preserve the element count. Without this check a
        // fully-concrete target silently reinterprets the buffer: an MoE router
        // reshaping `[rows, 1, k]` to `[rows, 1]` kept the first `rows` floats
        // and dropped the rest, so every token was weighted by another token's
        // routing probability — in range, plausible, and wrong.
        //
        // Only enforced when the input's product is actually computable.
        // `num_elements()` returns None for an over-large/garbage static dim
        // left by an upstream mis-resolved dynamic axis, and the comment above
        // documents that a fully-specified target should still be honoured
        // there rather than aborting the whole import.
        if let Some(total) = input.num_elements() {
            let out_total: usize = dims
                .iter()
                .map(|d: &Dim| d.unwrap_static())
                .try_fold(1usize, |a, d| a.checked_mul(d))
                .unwrap_or(usize::MAX);
            if out_total != total {
                return Err(format!(
                    "reshape: {:?} ({total} elements) -> {new_shape:?} ({out_total} elements); \
                     a reshape must preserve the element count",
                    input.dims()
                ));
            }
        }

        return Ok(Shape {
            dims,
            dtype: input.dtype,
        });
    }

    // Symbolic input: map `-1` to the sole dynamic symbol when unambiguous
    // (qwen35 prefill with batch=1 and `sym::SEQ`), otherwise keep dynamic.
    let dyn_syms = input.dynamic_symbols();
    let neg_idx = new_shape.iter().position(|&d| d == -1);
    let mut out_dims: SmallVec<[Dim; 4]> = SmallVec::new();
    for (i, &d) in new_shape.iter().enumerate() {
        if Some(i) == neg_idx {
            continue;
        }
        if d < 0 {
            return Err(format!("reshape: invalid dim {d}"));
        }
        out_dims.push(Dim::Static(d as usize));
    }
    if let Some(ni) = neg_idx {
        let inferred = if dyn_syms.len() == 1 {
            Dim::Dynamic(dyn_syms[0])
        } else if dyn_syms.is_empty() {
            return Err("reshape: cannot infer -1 on static input".into());
        } else {
            Dim::Dynamic(crate::dynamic::sym::ROWS)
        };
        out_dims.insert(ni, inferred);
    }
    Ok(Shape {
        dims: out_dims,
        dtype: input.dtype,
    })
}

/// Flatten leading axes to `[∏leading, H]` — used by `FuseRmsNormReshape` and shape verify.
pub fn leading_flatten_fused_shape(input: &Shape) -> Option<Shape> {
    if input.rank() < 2 {
        return None;
    }
    let Dim::Static(h) = input.dim(input.rank() - 1) else {
        return None;
    };
    let leading = &input.dims()[..input.rank() - 1];
    let lead_dim = if leading.iter().all(|d| d.is_static()) {
        Dim::Static(leading.iter().map(|d| d.unwrap_static()).product::<usize>())
    } else {
        let mut syms: Vec<u32> = leading
            .iter()
            .filter_map(|d| match d {
                Dim::Dynamic(s) => Some(*s),
                _ => None,
            })
            .collect();
        syms.sort();
        syms.dedup();
        match syms.len() {
            0 => Dim::Static(leading.iter().map(|d| d.unwrap_static()).product::<usize>()),
            1 => Dim::Dynamic(syms[0]),
            _ => Dim::Dynamic(crate::dynamic::sym::ROWS),
        }
    };
    Some(Shape::from_dims(&[lead_dim, Dim::Static(h)], input.dtype()))
}

/// Match `Reshape { new_shape }` after RmsNorm when fusing to a single op.
pub fn leading_flatten_shape(input: &Shape, new_shape: &[i64]) -> Option<Shape> {
    if new_shape.len() != 2 {
        return None;
    }
    let flat = leading_flatten_fused_shape(input)?;
    let Dim::Static(h) = input.dim(input.rank() - 1) else {
        return None;
    };
    if new_shape[1] as usize != h {
        return None;
    }
    match flat.dim(0) {
        Dim::Static(lead) if new_shape[0] as usize == lead => Some(flat),
        Dim::Dynamic(_) if new_shape[0] == -1 => Some(flat),
        _ => None,
    }
}

/// Attention: output shape = Q shape.
pub fn attention_shape(q: &Shape) -> Shape {
    q.clone()
}

/// Attention output shape accounting for an asymmetric V/output width. The
/// output has the same layout as `q` except the per-head width is `v_head_dim`:
/// rank-4 `[.., D]` → `[.., v_head_dim]`; rank ≤ 3 `[.., H·head_dim]` →
/// `[.., num_heads·v_head_dim]`. Equals `q` when `head_dim == v_head_dim`.
pub fn attention_shape_vdim(
    q: &Shape,
    num_heads: usize,
    head_dim: usize,
    v_head_dim: usize,
) -> Shape {
    if head_dim == v_head_dim || q.rank() == 0 {
        return q.clone();
    }
    let rank = q.rank();
    let new_last = if rank >= 4 {
        v_head_dim
    } else {
        num_heads * v_head_dim
    };
    q.clone().with_dim(rank - 1, Dim::Static(new_last))
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "] {}", self.dtype)
    }
}

/// Spatial output size for NCHW `Op::Conv` / `conv2d`.
pub fn conv2d_spatial_output(
    in_size: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let dil_k = dilation.saturating_mul(kernel.saturating_sub(1));
    (in_size + 2 * padding)
        .saturating_sub(dil_k)
        .saturating_sub(1)
        / stride
        + 1
}

/// Spatial output size for NCHW `Op::ConvTranspose2d`.
pub fn conv_transpose2d_spatial_output(
    in_size: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output_padding: usize,
) -> usize {
    let dil_k = dilation.saturating_mul(kernel.saturating_sub(1));
    (in_size - 1) * stride + output_padding + dil_k - 2 * padding + 1
}

/// Output shape for `conv2d` given NCHW `input` and weight `[C_out, C_in/g, kH, kW]`.
pub fn conv2d_output_shape(
    input: &Shape,
    weight: &Shape,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Result<Shape, String> {
    if input.rank() != 4 || weight.rank() != 4 {
        return Err("conv2d requires NCHW input and 4-D weight".into());
    }
    let n = input.dim(0);
    let c_in = input.dim(1).unwrap_static();
    let h = input.dim(2).unwrap_static();
    let w = input.dim(3).unwrap_static();
    let c_out = weight.dim(0).unwrap_static();
    let w_cin = weight.dim(1).unwrap_static();
    if w_cin * groups != c_in {
        return Err(format!(
            "conv2d weight C_in/g={w_cin} * groups={groups} != input C={c_in}"
        ));
    }
    let h_out = conv2d_spatial_output(h, kernel_size[0], stride[0], padding[0], dilation[0]);
    let w_out = conv2d_spatial_output(w, kernel_size[1], stride[1], padding[1], dilation[1]);
    Ok(Shape::from_dims(
        &[
            n,
            Dim::Static(c_out),
            Dim::Static(h_out),
            Dim::Static(w_out),
        ],
        input.dtype(),
    ))
}

/// Row width, in elements, of a RoPE cos/sin table.
///
/// Every backend needs this and each used to derive it locally, which is how
/// the CPU and GPU rules drifted apart in the first place.
///
/// * A table shaped `[positions, width]` carries its row width in the last
///   dimension, and that is what must be used. A table allocated at
///   `head_dim/2` but only partly filled under partial rotation is *wider* than
///   `n_rot/2`; assuming otherwise reads the wrong row for every position past
///   the first.
/// * A **rank-1** table is different: a flat `positions * n_rot/2` run has no
///   row structure in its shape, so its last dimension is the whole table.
///   Taking that as the stride sends position 1 past the end, which reads as
///   zeros — silently zeroing every position after the first. Rows there are
///   `n_rot/2` wide by construction.
pub fn rope_table_stride(cos: &Shape, n_rot: usize) -> usize {
    if cos.rank() >= 2 {
        cos.dims()
            .last()
            .map_or_else(|| n_rot / 2, |d| d.unwrap_static())
            .max(1)
    } else {
        (n_rot / 2).max(1)
    }
}

/// Output shape for NCHW [`crate::Op::Pool`].
///
/// Pooling has no weight, so the channel count passes through; only the two
/// spatial extents shrink, by the same rule convolution uses with dilation 1.
pub fn pool2d_output_shape(
    input: &Shape,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<Shape, String> {
    if input.rank() != 4 {
        return Err("pool2d requires an NCHW input".into());
    }
    if stride[0] == 0 || stride[1] == 0 {
        return Err("pool2d stride must be non-zero".into());
    }
    let n = input.dim(0);
    let c = input.dim(1);
    let h = input.dim(2).unwrap_static();
    let w = input.dim(3).unwrap_static();
    let h_out = conv2d_spatial_output(h, kernel_size[0], stride[0], padding[0], 1);
    let w_out = conv2d_spatial_output(w, kernel_size[1], stride[1], padding[1], 1);
    if h_out == 0 || w_out == 0 {
        return Err(format!(
            "pool2d window {kernel_size:?} with stride {stride:?} and padding {padding:?} \
             leaves no output for a {h}x{w} input"
        ));
    }
    Ok(Shape::from_dims(
        &[n, c, Dim::Static(h_out), Dim::Static(w_out)],
        input.dtype(),
    ))
}

/// Output shape for NCHW `Op::Im2Col`: `[M, C·kH·kW]` with
/// `M = N · H_out · W_out`. Dynamic batch maps to `dynamic::sym::ROWS`.
pub fn im2col_output_shape(
    input: &Shape,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Result<Shape, String> {
    if input.rank() != 4 {
        return Err("im2col requires NCHW input".into());
    }
    let c_in = input.dim(1).unwrap_static();
    let h = input.dim(2).unwrap_static();
    let w = input.dim(3).unwrap_static();
    let kh = kernel_size[0];
    let kw = kernel_size[1];
    let h_out = conv2d_spatial_output(h, kh, stride[0], padding[0], dilation[0]);
    let w_out = conv2d_spatial_output(w, kw, stride[1], padding[1], dilation[1]);
    let k = c_in * kh * kw;
    let spatial = h_out * w_out;
    let m = match input.dim(0) {
        Dim::Static(n) => Dim::Static(n * spatial),
        Dim::Dynamic(crate::dynamic::sym::BATCH) | Dim::Dynamic(crate::dynamic::sym::ROWS) => {
            Dim::Dynamic(crate::dynamic::sym::ROWS)
        }
        Dim::Dynamic(_) => Dim::Dynamic(crate::dynamic::sym::ROWS),
    };
    Ok(Shape::from_dims(&[m, Dim::Static(k)], input.dtype()))
}

/// Output shape for `conv_transpose2d` (weight `[C_in, C_out/g, kH, kW]`).
pub fn conv_transpose2d_output_shape(
    input: &Shape,
    weight: &Shape,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    output_padding: [usize; 2],
    groups: usize,
) -> Result<Shape, String> {
    if input.rank() != 4 || weight.rank() != 4 {
        return Err("conv_transpose2d requires NCHW input and 4-D weight".into());
    }
    // Preserve the batch dim (it may be dynamic) — mirror `conv2d_output_shape`.
    let n = input.dim(0);
    let c_in = input.dim(1).unwrap_static();
    let h = input.dim(2).unwrap_static();
    let w = input.dim(3).unwrap_static();
    let w_cin = weight.dim(0).unwrap_static();
    let c_out_per_g = weight.dim(1).unwrap_static();
    if w_cin != c_in {
        return Err(format!(
            "conv_transpose2d weight C_in={w_cin} != input C={c_in}"
        ));
    }
    let h_out = conv_transpose2d_spatial_output(
        h,
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        output_padding[0],
    );
    let w_out = conv_transpose2d_spatial_output(
        w,
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        output_padding[1],
    );
    Ok(Shape::from_dims(
        &[
            n,
            Dim::Static(c_out_per_g * groups),
            Dim::Static(h_out),
            Dim::Static(w_out),
        ],
        input.dtype(),
    ))
}

/// Output shape for NCDHW `Op::Conv3d` given weight `[C_out, C_in/g, kD, kH, kW]`.
/// Reuses the 1-D `conv2d_spatial_output` formula per spatial axis.
#[allow(clippy::too_many_arguments)]
pub fn conv3d_output_shape(
    input: &Shape,
    weight: &Shape,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
) -> Result<Shape, String> {
    if input.rank() != 5 || weight.rank() != 5 {
        return Err("conv3d requires NCDHW input and 5-D weight".into());
    }
    let n = input.dim(0);
    let c_in = input.dim(1).unwrap_static();
    let d = input.dim(2).unwrap_static();
    let h = input.dim(3).unwrap_static();
    let w = input.dim(4).unwrap_static();
    let c_out = weight.dim(0).unwrap_static();
    let w_cin = weight.dim(1).unwrap_static();
    if w_cin * groups != c_in {
        return Err(format!(
            "conv3d weight C_in/g={w_cin} * groups={groups} != input C={c_in}"
        ));
    }
    let d_out = conv2d_spatial_output(d, kernel_size[0], stride[0], padding[0], dilation[0]);
    let h_out = conv2d_spatial_output(h, kernel_size[1], stride[1], padding[1], dilation[1]);
    let w_out = conv2d_spatial_output(w, kernel_size[2], stride[2], padding[2], dilation[2]);
    Ok(Shape::from_dims(
        &[
            n,
            Dim::Static(c_out),
            Dim::Static(d_out),
            Dim::Static(h_out),
            Dim::Static(w_out),
        ],
        input.dtype(),
    ))
}

/// Output shape for NCDHW `Op::ConvTranspose3d` (weight `[C_in, C_out/g, kD, kH, kW]`).
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose3d_output_shape(
    input: &Shape,
    weight: &Shape,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    output_padding: [usize; 3],
    groups: usize,
) -> Result<Shape, String> {
    if input.rank() != 5 || weight.rank() != 5 {
        return Err("conv_transpose3d requires NCDHW input and 5-D weight".into());
    }
    let n = input.dim(0);
    let c_in = input.dim(1).unwrap_static();
    let d = input.dim(2).unwrap_static();
    let h = input.dim(3).unwrap_static();
    let w = input.dim(4).unwrap_static();
    let w_cin = weight.dim(0).unwrap_static();
    let c_out_per_g = weight.dim(1).unwrap_static();
    if w_cin != c_in {
        return Err(format!(
            "conv_transpose3d weight C_in={w_cin} != input C={c_in}"
        ));
    }
    let d_out = conv_transpose2d_spatial_output(
        d,
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        output_padding[0],
    );
    let h_out = conv_transpose2d_spatial_output(
        h,
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        output_padding[1],
    );
    let w_out = conv_transpose2d_spatial_output(
        w,
        kernel_size[2],
        stride[2],
        padding[2],
        dilation[2],
        output_padding[2],
    );
    Ok(Shape::from_dims(
        &[
            n,
            Dim::Static(c_out_per_g * groups),
            Dim::Static(d_out),
            Dim::Static(h_out),
            Dim::Static(w_out),
        ],
        input.dtype(),
    ))
}

/// `KvAppend`: write `row` into `cache` at index `pos` along `axis`, and return
/// the `[..pos+1]` prefix (which ALIASES the cache buffer).
///
/// Every operand relationship is checked here rather than left to the backends,
/// because none of them can check it: by the time a row write reaches a kernel
/// it is a byte offset and a length, and a `pos` past the cache or a row of the
/// wrong width is an out-of-bounds store into whatever the memory planner put
/// next in the arena. That is a corrupted neighbouring tensor, not a crash —
/// wrong logits with no error anywhere. The op is also the one place where a
/// caller must size a buffer to a CAPACITY rather than to its current contents
/// (`pos` indexes into the spare rows), so an off-by-one here is easy to write
/// and invisible afterwards.
///
/// Dynamic dims are skipped rather than guessed at: a symbolic sequence
/// capacity is legitimate, and the check that matters (`pos < cap`) can only be
/// made against a static one.
pub fn kv_append_shape(
    cache: &Shape,
    row: &Shape,
    axis: usize,
    pos: usize,
) -> Result<Shape, String> {
    if axis >= cache.rank() {
        return Err(format!(
            "kv_append: axis {axis} is out of range for a rank-{} cache",
            cache.rank()
        ));
    }
    if row.rank() != cache.rank() {
        return Err(format!(
            "kv_append: row rank {} does not match cache rank {} — the row is a \
             one-step slice of the cache, not a squeezed vector",
            row.rank(),
            cache.rank()
        ));
    }
    if row.dtype() != cache.dtype() {
        return Err(format!(
            "kv_append: row dtype {:?} does not match cache dtype {:?} — the write \
             is a raw copy, so a mismatch reinterprets the row's bits",
            row.dtype(),
            cache.dtype()
        ));
    }
    if let Dim::Static(cap) = cache.dim(axis)
        && pos >= cap
    {
        return Err(format!(
            "kv_append: pos {pos} is past the cache's axis-{axis} capacity {cap} — \
             the cache must be sized to the CAPACITY it will grow to, not to the \
             number of rows it currently holds"
        ));
    }
    if let Dim::Static(n) = row.dim(axis)
        && n != 1
    {
        return Err(format!(
            "kv_append: row axis-{axis} extent is {n}, must be 1 (one step)"
        ));
    }
    for i in 0..cache.rank() {
        if i == axis {
            continue;
        }
        if let (Dim::Static(c), Dim::Static(r)) = (cache.dim(i), row.dim(i))
            && c != r
        {
            return Err(format!(
                "kv_append: row dim {i} is {r}, cache dim {i} is {c} — every axis but \
                 {axis} must match or the copy walks off the row"
            ));
        }
    }
    Ok(cache.clone().with_dim(axis, Dim::Static(pos + 1)))
}

#[cfg(test)]
mod tests {

    /// A `Concat` whose inputs disagree on a NON-concat axis must be rejected,
    /// not silently shaped from `inputs[0]`.
    ///
    /// `rlx-eeginceptionerp` declared its zero-padding as [B,cin,1,l] while the
    /// activation was [B,cin,C,T]. The old inference returned [B,cin,1,·],
    /// collapsing the channel axis of the first block. CPU/Metal/wgpu/CoreML
    /// executed it anyway; only MLX refused, so a malformed graph looked like a
    /// backend gap.
    #[test]
    fn concat_rejects_non_concat_axis_mismatch() {
        let pad = Shape::new(&[3, 1, 1, 31], DType::F32);
        let act = Shape::new(&[3, 1, 8, 1000], DType::F32);
        let err = concat_shape(&[&pad, &act], 3).unwrap_err();
        assert!(
            err.contains("axis 2 mismatch"),
            "expected the channel-axis mismatch to be named, got: {err}"
        );

        // The well-formed version still infers normally.
        let pad_ok = Shape::new(&[3, 1, 8, 31], DType::F32);
        let out = concat_shape(&[&pad_ok, &act], 3).expect("matching dims must concat");
        assert_eq!(out.dims()[3].unwrap_static(), 1031);
        assert_eq!(out.dims()[2].unwrap_static(), 8);
    }

    /// The concat axis itself is exempt — that is the whole point of it.
    #[test]
    fn concat_axis_itself_may_differ() {
        let a = Shape::new(&[2, 3], DType::F32);
        let b = Shape::new(&[5, 3], DType::F32);
        let out = concat_shape(&[&a, &b], 0).expect("concat axis may differ");
        assert_eq!(out.dims()[0].unwrap_static(), 7);
    }

    use super::*;

    /// A GGUF weight is `[n, k]`. Reading it as `[k, n]` rejects correct graphs
    /// — a square weight hides that, so both cases here are non-square.
    #[test]
    fn dequant_matmul_contracts_the_weights_last_axis() {
        let (m, k, n) = (2usize, 64usize, 3usize);
        let x = Shape::new(&[m, k], DType::F32);
        let w_nk = Shape::new(&[n, k], DType::F32);

        assert_eq!(
            dequant_matmul_shape(&x, &w_nk).expect("[n,k] weight is the GGUF layout"),
            Shape::new(&[m, n], DType::F32),
        );

        // The plain matmul rule reads the same operands backwards. This is the
        // rejection `coreml_quant::dequant_matmul_through_session` used to hit.
        assert!(matmul_shape(&x, &w_nk).unwrap_err().contains("K mismatch"));

        // Leading batch axes ride along; only the last two participate.
        assert_eq!(
            dequant_matmul_shape(&Shape::new(&[5, m, k], DType::F32), &w_nk).unwrap(),
            Shape::new(&[5, m, n], DType::F32),
        );

        // A genuinely wrong K is still caught, and says which axis it means.
        let err = dequant_matmul_shape(&x, &Shape::new(&[n, k + 1], DType::F32)).unwrap_err();
        assert!(err.contains("K mismatch"), "{err}");
        assert!(err.contains("last axis"), "{err}");

        // A packed byte blob has no `[n, k]` to read; the caller declares the
        // output shape instead, and `infer_shape` never reaches this rule.
        assert!(dequant_matmul_shape(&x, &Shape::new(&[1024], DType::U8)).is_err());
    }

    /// The two index forms a gather backward has to accept.
    #[test]
    fn gather_index_count_handles_both_index_forms() {
        // Indexed AT the axis: the count is that dimension.
        let batched = Shape::new(&[2, 11], DType::F32);
        assert_eq!(batched.gather_index_count(1), 11);
        assert_eq!(batched.gather_index_count(0), 2);

        // A flat list with no dimension at the axis: its extent IS the count.
        // `dim(1)` would panic here, which is exactly the bug this replaced.
        let flat = Shape::new(&[11], DType::F32);
        assert_eq!(flat.gather_index_count(1), 11);
        assert_eq!(flat.gather_index_count(0), 11);
        assert_eq!(flat.gather_index_count(3), 11, "any axis beyond rank");

        // A scalar index selects exactly one position, never zero.
        assert_eq!(Shape::new(&[], DType::F32).gather_index_count(0), 1);
    }

    #[test]
    fn static_shape() {
        let s = Shape::new(&[4, 15, 384], DType::F32);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.num_elements(), Some(4 * 15 * 384));
        assert_eq!(s.size_bytes(), Some(4 * 15 * 384 * 4));
        assert!(s.is_static());
        assert_eq!(format!("{s}"), "[4, 15, 384] f32");
    }

    // ── GroupedMatMul operand checks ─────────────────────────

    #[test]
    fn grouped_matmul_derives_m_and_n() {
        let input = Shape::new(&[5, 6], DType::F32);
        let bank = Shape::new(&[8, 6, 12], DType::F32); // [E, K, N]
        let d = grouped_matmul_dims(&input, &bank, Some(&Shape::new(&[5, 12], DType::F32)))
            .expect("valid");
        assert_eq!(
            d,
            GroupedMatMulDims {
                m: 5,
                k: 6,
                n: 12,
                num_experts: 8
            }
        );
        assert_eq!(
            grouped_matmul_shape(&input, &bank).unwrap(),
            Shape::new(&[5, 12], DType::F32)
        );
    }

    /// The whole point: a bank still in the checkpoint's `[E, N, K]` order has
    /// the right rank and element count, so only the K axis gives it away.
    #[test]
    fn grouped_matmul_rejects_an_untransposed_bank() {
        let input = Shape::new(&[5, 6], DType::F32);
        let bank = Shape::new(&[8, 12, 6], DType::F32); // [E, N, K] — wrong way round
        let err = grouped_matmul_dims(&input, &bank, None).unwrap_err();
        assert!(err.contains("K mismatch"), "unhelpful: {err}");
        assert!(err.contains("[E, N, K]"), "should name the cause: {err}");
    }

    /// `2·inter == hidden` makes both bank layouts the *same shape*, so the K
    /// check passes and only the declared output catches it. This is the case
    /// that silently under-writes half of every output row.
    #[test]
    fn grouped_matmul_rejects_a_square_bank_by_output_shape() {
        let input = Shape::new(&[4, 12, 12], DType::F32);
        let bank = Shape::new(&[8, 12, 12], DType::F32);
        // K and N agree, so operand-only validation cannot object…
        assert!(grouped_matmul_dims(&input, &bank, None).is_ok());
        // …but a declared output of a different width is still caught.
        let err = grouped_matmul_dims(&input, &bank, Some(&Shape::new(&[12, 6], DType::F32)))
            .unwrap_err();
        assert!(err.contains("output shape mismatch"), "unhelpful: {err}");
    }

    #[test]
    fn grouped_matmul_rejects_a_non_bank_weight() {
        let input = Shape::new(&[5, 6], DType::F32);
        let err = grouped_matmul_dims(&input, &Shape::new(&[6, 12], DType::F32), None).unwrap_err();
        assert!(err.contains("rank 3"), "unhelpful: {err}");
    }

    // ── Shape inference tests ────────────────────────────────

    #[test]
    fn broadcast_same() {
        let a = Shape::new(&[4, 15, 384], DType::F32);
        let r = broadcast(&a, &a).unwrap();
        assert_eq!(r.dims(), a.dims());
    }

    #[test]
    fn broadcast_bias() {
        let a = Shape::new(&[4, 15, 384], DType::F32);
        let b = Shape::new(&[384], DType::F32);
        let r = broadcast(&a, &b).unwrap();
        assert_eq!(r, Shape::new(&[4, 15, 384], DType::F32));
    }

    #[test]
    fn broadcast_scalar() {
        let a = Shape::new(&[4, 15, 384], DType::F32);
        let b = Shape::scalar(DType::F32);
        let r = broadcast(&a, &b).unwrap();
        assert_eq!(r, a);
    }

    #[test]
    fn broadcast_mismatch() {
        let a = Shape::new(&[4, 15, 384], DType::F32);
        let b = Shape::new(&[4, 15, 256], DType::F32);
        assert!(broadcast(&a, &b).is_err());
    }

    #[test]
    fn matmul_basic() {
        let a = Shape::new(&[4, 15, 384], DType::F32);
        let b = Shape::new(&[384, 1536], DType::F32);
        let r = matmul_shape(&a, &b).unwrap();
        assert_eq!(r, Shape::new(&[4, 15, 1536], DType::F32));
    }

    #[test]
    fn matmul_batched() {
        let a = Shape::new(&[4, 15, 384], DType::F32);
        let b = Shape::new(&[4, 384, 1536], DType::F32);
        let r = matmul_shape(&a, &b).unwrap();
        assert_eq!(r, Shape::new(&[4, 15, 1536], DType::F32));
    }

    #[test]
    fn matmul_k_mismatch() {
        let a = Shape::new(&[4, 15, 384], DType::F32);
        let b = Shape::new(&[512, 1536], DType::F32);
        assert!(matmul_shape(&a, &b).is_err());
    }

    #[test]
    fn reduce_keepdim() {
        let a = Shape::new(&[4, 15, 384], DType::F32);
        let r = reduce_shape(&a, &[2], true).unwrap();
        assert_eq!(r, Shape::new(&[4, 15, 1], DType::F32));
    }

    #[test]
    fn reduce_no_keepdim() {
        let a = Shape::new(&[4, 15, 384], DType::F32);
        let r = reduce_shape(&a, &[2], false).unwrap();
        assert_eq!(r, Shape::new(&[4, 15], DType::F32));
    }

    #[test]
    fn concat_basic() {
        let a = Shape::new(&[4, 15, 384], DType::F32);
        let b = Shape::new(&[4, 15, 384], DType::F32);
        let r = concat_shape(&[&a, &b], 2).unwrap();
        assert_eq!(r, Shape::new(&[4, 15, 768], DType::F32));
    }

    #[test]
    fn gather_embedding() {
        let table = Shape::new(&[30522, 384], DType::F32);
        let indices = Shape::new(&[4, 15], DType::I64);
        let r = gather_shape(&table, &indices, 0).unwrap();
        assert_eq!(
            r,
            Shape::from_dims(
                &[Dim::Static(4), Dim::Static(15), Dim::Static(384)],
                DType::F32
            )
        );
    }

    #[test]
    fn gather_nonzero_axis_keeps_prefix() {
        // ONNX Gather output = table[:axis] ++ indices.shape ++ table[axis+1:].
        // Regression: axis!=0 must NOT drop the leading dims (a nearest-upsample
        // gather `[4,3]` on axis 1 with a `[6]` index → `[4,6]`, not `[6]`).
        let table = Shape::new(&[4, 3], DType::F32);
        let indices = Shape::new(&[6], DType::I64);
        let r = gather_shape(&table, &indices, 1).unwrap();
        assert_eq!(r, Shape::new(&[4, 6], DType::F32));
        // axis 1 of a rank-3 table, 2-D index.
        let table = Shape::new(&[2, 5, 7], DType::F32);
        let indices = Shape::new(&[3, 4], DType::I64);
        let r = gather_shape(&table, &indices, 1).unwrap();
        assert_eq!(r, Shape::new(&[2, 3, 4, 7], DType::F32));
    }

    #[test]
    fn reshape_with_neg1() {
        let a = Shape::new(&[4, 15, 384], DType::F32);
        let r = reshape_shape(&a, &[60, -1]).unwrap();
        assert_eq!(r, Shape::new(&[60, 384], DType::F32));
    }

    #[test]
    fn transpose_basic() {
        let a = Shape::new(&[4, 15, 384], DType::F32);
        let r = transpose_shape(&a, &[0, 2, 1]).unwrap();
        assert_eq!(r, Shape::new(&[4, 384, 15], DType::F32));
    }

    #[test]
    fn narrow_basic() {
        let a = Shape::new(&[4, 15, 1152], DType::F32);
        let r = narrow_shape(&a, 2, 384).unwrap();
        assert_eq!(r, Shape::new(&[4, 15, 384], DType::F32));
    }

    #[test]
    fn compare_bool_output() {
        let a = Shape::new(&[4, 15], DType::F32);
        let b = Shape::new(&[4, 15], DType::F32);
        let r = compare_shape(&a, &b).unwrap();
        assert_eq!(r.dtype(), DType::Bool);
        assert_eq!(r.rank(), 2);
    }

    // ── Original tests ──────────────────────────────────────

    #[test]
    fn dynamic_shape() {
        let s = Shape::from_dims(
            &[Dim::Dynamic(0), Dim::Dynamic(1), Dim::Static(384)],
            DType::F32,
        );
        assert_eq!(s.rank(), 3);
        assert_eq!(s.num_elements(), None);
        assert!(!s.is_static());
        assert_eq!(format!("{s}"), "[?0, ?1, 384] f32");
    }

    #[test]
    fn dynamic_symbols_lists_distinct_dims() {
        let s = Shape::from_dims(
            &[
                Dim::Dynamic(1),
                Dim::Static(384),
                Dim::Dynamic(0),
                Dim::Dynamic(1),
            ],
            DType::F32,
        );
        assert_eq!(s.dynamic_symbols(), vec![0, 1]);
    }

    #[test]
    fn bind_specializes_known_symbols() {
        let s = Shape::from_dims(
            &[Dim::Dynamic(0), Dim::Dynamic(1), Dim::Static(384)],
            DType::F32,
        );
        let mut b = DimBinding::new();
        b.set(0, 8);
        b.set(1, 64);
        let s2 = s.bind(&b);
        assert!(s2.is_static());
        assert_eq!(s2.num_elements(), Some(8 * 64 * 384));
    }

    #[test]
    fn bind_leaves_unknown_symbols_alone() {
        let s = Shape::from_dims(&[Dim::Dynamic(0), Dim::Dynamic(99)], DType::F32);
        let mut b = DimBinding::new();
        b.set(0, 4);
        let s2 = s.bind(&b);
        assert!(!s2.is_static()); // ?99 still dynamic
        assert_eq!(s2.dynamic_symbols(), vec![99]);
    }
}

/// Mapping from a dynamic-dim symbol to its concrete size at
/// runtime. Plan #54.
#[derive(Debug, Clone, Default)]
pub struct DimBinding {
    map: std::collections::HashMap<u32, usize>,
}

impl DimBinding {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set(&mut self, symbol: u32, size: usize) -> Option<usize> {
        self.map.insert(symbol, size)
    }
    pub fn get(&self, symbol: u32) -> Option<usize> {
        self.map.get(&symbol).copied()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn iter(&self) -> impl Iterator<Item = (u32, usize)> + '_ {
        self.map.iter().map(|(&s, &n)| (s, n))
    }
}
