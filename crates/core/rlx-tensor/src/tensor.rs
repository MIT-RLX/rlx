// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Symbolic tensor handle — zero payload bytes until compile + run.

use std::ops::{Add, Div, Mul, Neg, Sub};

use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{DType, Dim, GraphExt, NodeId, Op, ScaleLayout, ScaledFormat, Shape};

use crate::handle::GraphHandle;
use crate::scalar::{Scalar, promote_scalar};
use crate::slice::SliceSpec;

/// Generate same-shape unary [`Activation`] methods (`name => Variant`).
/// Each delegates to [`Tensor::map_act`], so adding a JAX-style elementwise
/// op is a one-line table entry, not a copied method body.
macro_rules! unary_ops {
    ($( $(#[$doc:meta])* $name:ident => $variant:ident ),+ $(,)?) => {
        impl Tensor {
            $(
                $(#[$doc])*
                pub fn $name(&self) -> Self {
                    self.map_act(Activation::$variant)
                }
            )+
        }
    };
}

/// Generate elementwise comparison methods (`name => Variant`) → `Bool` tensor.
macro_rules! cmp_ops {
    ($( $(#[$doc:meta])* $name:ident => $variant:ident ),+ $(,)?) => {
        impl Tensor {
            $(
                $(#[$doc])*
                pub fn $name(&self, rhs: &Tensor) -> Self {
                    self.map_cmp(BinaryRhs::Tensor(rhs), CmpOp::$variant)
                }
            )+
        }
    };
}

/// Generate axis-reduction methods (`name => Variant`).
macro_rules! reduce_ops {
    ($( $(#[$doc:meta])* $name:ident => $variant:ident ),+ $(,)?) => {
        impl Tensor {
            $(
                $(#[$doc])*
                pub fn $name(&self, axes: impl Into<Vec<usize>>, keep_dim: bool) -> Self {
                    self.map_reduce(ReduceOp::$variant, axes.into(), keep_dim)
                }
            )+
        }
    };
}

/// Convert a [`Dim`] to a reshape-style `i64` (dynamic → `-1`).
fn dim_to_i64(d: &Dim) -> i64 {
    match d {
        Dim::Static(n) => *n as i64,
        Dim::Dynamic(_) => -1,
    }
}

/// Render row-major `data` as nested brackets matching `dims`.
#[cfg(feature = "eval")]
fn fmt_nested(data: &[f32], dims: &[usize]) -> String {
    match dims {
        [] => format!("{}", data.first().copied().unwrap_or(0.0)),
        [_] => {
            let body = data
                .iter()
                .map(|x| format!("{x}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{body}]")
        }
        [outer, rest @ ..] => {
            let inner: usize = rest.iter().product();
            let parts = (0..*outer)
                .map(|i| fmt_nested(&data[i * inner..(i + 1) * inner], rest))
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{parts}]")
        }
    }
}

#[cfg(feature = "eval")]
impl std::fmt::Display for Tensor {
    /// Realizes the tensor (compile + run) and prints values — the lazy graph
    /// is forced here, like ndarray's eager `Display`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.show())
    }
}

/// A symbolic tensor in a traced graph.
///
/// **Lazy by construction.** Despite the eager-looking operator syntax, a
/// `Tensor` is a deferred handle: `&a + &b`, `.relu()`, `.matmul(..)` etc.
/// only append IR nodes — no host or device buffers are allocated and nothing
/// executes. The graph is compiled (fused + memory-planned) and run **only**
/// when you force it with [`to_vec`](Self::to_vec) / [`realize`](Self::realize)
/// (both require the `eval` feature). This is what lets the whole expression
/// fuse instead of materializing every intermediate.
///
/// **Zero-copy clone & slice.** `clone()` just bumps a refcount on the shared
/// graph handle — no data is duplicated. `slice`/`narrow`/`reshape`/`select`
/// add a *view* node into the same graph (referencing the parent by id), so
/// they move zero payload bytes either. Use [`shares_graph`](Self::shares_graph)
/// to observe storage sharing and [`storage_bytes`](Self::storage_bytes) to
/// confirm no data was copied. (Combining two *independently-constructed*
/// tensors is the one case that copies — the right operand's nodes are merged
/// in once; see the `handle` module.)
#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) handle: GraphHandle,
    pub(crate) id: NodeId,
}

impl Tensor {
    pub(crate) fn new(handle: GraphHandle, id: NodeId) -> Self {
        Self { handle, id }
    }

    /// IR node id (mix with raw [`rlx_ir::Graph`] builders via [`NodeId`]).
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn shape(&self) -> Shape {
        self.handle.with_graph(|g| g.shape(self.id).clone())
    }

    pub fn dtype(&self) -> DType {
        self.shape().dtype()
    }

    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    /// Static dimensions as a `Vec<usize>` (panics on a dynamic axis).
    pub fn dims(&self) -> Vec<usize> {
        self.shape()
            .dims()
            .iter()
            .map(|d| match d {
                Dim::Static(n) => *n,
                Dim::Dynamic(_) => panic!("dims(): tensor has a dynamic axis"),
            })
            .collect()
    }

    /// Total number of elements (panics on a dynamic axis).
    pub fn numel(&self) -> usize {
        self.dims().iter().product()
    }

    /// True when every dimension is statically known.
    pub fn is_static_shape(&self) -> bool {
        self.shape().is_static()
    }

    /// True if `self` and `other` are backed by the **same** graph — i.e. one
    /// is a [`clone`](Clone) or a view (`slice`/`reshape`/`narrow`/…) of the
    /// other, sharing storage with zero copy. Tensors from independent
    /// constructors return `false` until they're combined in an op.
    pub fn shares_graph(&self, other: &Tensor) -> bool {
        self.handle.same(&other.handle)
    }

    /// Total bytes of constant (payload) data physically held in the backing
    /// graph. **Cloning and slicing do not increase this** — they share
    /// storage and only add zero-payload view nodes. Combining tensors from
    /// different graphs copies the right-hand operand's data in (once).
    pub fn storage_bytes(&self) -> usize {
        self.handle.with_graph(|g| {
            g.nodes()
                .iter()
                .filter_map(|n| match &n.op {
                    Op::Constant { data } => Some(data.len()),
                    _ => None,
                })
                .sum()
        })
    }

    /// View slice — lowers to [`narrow`](Self::narrow) (no copy at IR level).
    pub fn slice(&self, spec: SliceSpec) -> Self {
        spec.apply(self)
    }

    fn map_unary(&self, f: impl FnOnce(&mut rlx_ir::Graph, NodeId) -> NodeId) -> Self {
        let id = self.handle.with_graph(|g| f(g, self.id));
        Self::new(self.handle.clone(), id)
    }

    /// Pull `other`'s graph into this tensor's graph (no-op when they already
    /// share one), returning the operand's node id in *this* graph.
    pub(crate) fn adopt(&self, other: &Tensor) -> NodeId {
        self.handle.adopt(&other.handle, other.id)
    }

    fn map_binary(
        &self,
        rhs: BinaryRhs<'_>,
        f: impl FnOnce(&mut rlx_ir::Graph, NodeId, NodeId) -> NodeId,
    ) -> Self {
        let rhs_id = match rhs {
            BinaryRhs::Tensor(v) => self.adopt(v),
            BinaryRhs::Scalar(s) => self.handle.with_graph(|g| {
                let dtype = g.shape(self.id).dtype();
                promote_scalar(g, s, dtype)
            }),
        };
        let id = self.handle.with_graph(|g| f(g, self.id, rhs_id));
        Self::new(self.handle.clone(), id)
    }

    /// Apply a unary [`Activation`] with same-shape output.
    fn map_act(&self, act: Activation) -> Self {
        self.map_unary(|g, x| {
            let s = rlx_ir::shape::unary_shape(g.shape(x));
            g.activation(act, x, s)
        })
    }

    /// Elementwise [`BinaryOp`] with broadcast shape inference.
    fn map_binary_op(&self, rhs: BinaryRhs<'_>, op: BinaryOp) -> Self {
        self.map_binary(rhs, |g, a, b| {
            let s = rlx_ir::shape::binary_shape(g.shape(a), g.shape(b))
                .expect("binary shape inference");
            g.binary(op, a, b, s)
        })
    }

    /// Elementwise comparison ([`CmpOp`]) → `Bool` tensor.
    fn map_cmp(&self, rhs: BinaryRhs<'_>, op: CmpOp) -> Self {
        self.map_binary(rhs, |g, a, b| {
            let s = rlx_ir::shape::compare_shape(g.shape(a), g.shape(b))
                .expect("compare shape inference");
            g.add_node(Op::Compare(op), vec![a, b], s)
        })
    }

    /// Reduction ([`ReduceOp`]) over `axes`.
    fn map_reduce(&self, op: ReduceOp, axes: Vec<usize>, keep_dim: bool) -> Self {
        self.map_unary(|g, x| {
            let s = rlx_ir::shape::reduce_shape(g.shape(x), &axes, keep_dim)
                .expect("reduce shape inference");
            g.reduce(x, op, axes, keep_dim, s)
        })
    }

    /// Matrix multiply (`@` in Python / JAX).
    #[doc(alias = "@")]
    pub fn mm(&self, rhs: &Tensor) -> Self {
        self.matmul(rhs)
    }

    pub fn matmul(&self, rhs: &Tensor) -> Self {
        self.map_binary(BinaryRhs::Tensor(rhs), |g, a, b| g.mm(a, b))
    }

    /// Native low-precision GEMM (TN: `self [m,k] · rhs [n,k]ᵀ → [m,n]`).
    /// Both operands are dynamically quantized to the minifloat `fmt` +
    /// `layout`; `rhs` must be K-last (`[n, k]`). `fmt` may be any
    /// [`ScaledFormat`], including a parameterized `Custom` — e.g.
    /// `t.scaled_matmul(&w, ScaledFormat::custom(3, 0), ScaleLayout::mx())` runs
    /// the matmul in `f4e3m0`.
    pub fn scaled_matmul(&self, rhs: &Tensor, fmt: ScaledFormat, layout: ScaleLayout) -> Self {
        self.map_binary(BinaryRhs::Tensor(rhs), move |g, a, b| {
            g.scaled_matmul(a, b, fmt, layout)
        })
    }

    pub fn softmax(&self, axis: i32) -> Self {
        self.map_unary(|g, x| g.sm(x, axis))
    }

    pub fn reshape(&self, new_shape: impl Into<Vec<i64>>) -> Self {
        self.map_unary(|g, x| g.reshape_(x, new_shape.into()))
    }

    pub fn transpose(&self, perm: impl Into<Vec<usize>>) -> Self {
        self.map_unary(|g, x| g.transpose_(x, perm.into()))
    }

    pub fn t(&self) -> Self {
        let rank = self.rank();
        assert!(rank >= 2, "t() requires rank >= 2");
        let perm: Vec<usize> = (0..rank - 2).chain([rank - 1, rank - 2]).collect();
        self.transpose(perm)
    }

    pub fn cast(&self, to: DType) -> Self {
        self.map_unary(|g, x| g.cast(x, to))
    }

    pub fn narrow(&self, axis: usize, start: usize, len: usize) -> Self {
        self.map_unary(|g, x| g.narrow_(x, axis, start, len))
    }

    /// Broadcast (expand) to an explicit target shape — NumPy `broadcast_to`.
    /// A `1` axis stretches; sizes must otherwise match. Use `-1`-free static
    /// targets.
    pub fn broadcast_to(&self, shape: impl Into<Vec<i64>>) -> Self {
        let target = shape.into();
        self.map_unary(|g, x| {
            let s = rlx_ir::shape::expand_shape(g.shape(x), &target)
                .expect("broadcast_to shape inference");
            g.add_node(
                Op::Expand {
                    target_shape: target,
                },
                vec![x],
                s,
            )
        })
    }

    /// Flatten to a 1-D tensor (row-major).
    pub fn flatten(&self) -> Self {
        self.reshape(vec![-1_i64])
    }

    /// Insert a size-1 axis at `axis` (NumPy `expand_dims` / PyTorch
    /// `unsqueeze`).
    pub fn unsqueeze(&self, axis: usize) -> Self {
        let mut dims = self.dims_i64();
        assert!(axis <= dims.len(), "unsqueeze: axis {axis} out of range");
        dims.insert(axis, 1);
        self.reshape(dims)
    }

    /// Remove the size-1 axis at `axis` (panics if it is not size 1).
    pub fn squeeze(&self, axis: usize) -> Self {
        let dims = self.shape();
        let dims = dims.dims();
        assert!(
            matches!(dims.get(axis), Some(Dim::Static(1))),
            "squeeze: axis {axis} is not size 1"
        );
        let new: Vec<i64> = dims
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, d)| dim_to_i64(d))
            .collect();
        self.reshape(new)
    }

    /// Remove every size-1 axis.
    pub fn squeeze_all(&self) -> Self {
        let new: Vec<i64> = self
            .shape()
            .dims()
            .iter()
            .filter(|d| !matches!(d, Dim::Static(1)))
            .map(dim_to_i64)
            .collect();
        self.reshape(new)
    }

    /// Current dims as `i64` (dynamic → `-1`), for reshape-style builders.
    fn dims_i64(&self) -> Vec<i64> {
        self.shape().dims().iter().map(dim_to_i64).collect()
    }

    /// Split `axis` into contiguous pieces of the given `sizes` (must sum to
    /// the axis length). NumPy `split` / PyTorch `split_with_sizes`.
    pub fn split(&self, axis: usize, sizes: &[usize]) -> Vec<Tensor> {
        let mut out = Vec::with_capacity(sizes.len());
        let mut start = 0;
        for &len in sizes {
            out.push(self.narrow(axis, start, len));
            start += len;
        }
        out
    }

    /// Split `axis` into `n` near-equal chunks (first `len % n` get one extra).
    pub fn chunk(&self, axis: usize, n: usize) -> Vec<Tensor> {
        assert!(n >= 1, "chunk: n must be >= 1");
        let total = self.dims()[axis];
        let base = total / n;
        let rem = total % n;
        let sizes: Vec<usize> = (0..n).map(|i| base + usize::from(i < rem)).collect();
        self.split(axis, &sizes)
    }

    /// Repeat the whole tensor `reps` times along `axis` (NumPy `tile` on one
    /// axis): output axis length becomes `len * reps`.
    pub fn tile(&self, axis: usize, reps: usize) -> Tensor {
        assert!(reps >= 1, "tile: reps must be >= 1");
        let copies = vec![self; reps];
        crate::array::cat(&copies, axis)
    }

    /// Reverse the order of elements along `axis` (NumPy `flip`).
    pub fn flip(&self, axis: usize) -> Tensor {
        let n = self.dims()[axis];
        let rev: Vec<i64> = (0..n as i64).rev().collect();
        self.index_select(axis, &Tensor::index_vec(rev))
    }

    /// Cyclically shift elements along `axis` by `shift` (NumPy `roll`).
    pub fn roll(&self, axis: usize, shift: usize) -> Tensor {
        let n = self.dims()[axis];
        let s = if n == 0 { 0 } else { shift % n };
        if s == 0 {
            return self.clone();
        }
        let tail = self.narrow(axis, n - s, s);
        let head = self.narrow(axis, 0, n - s);
        crate::array::cat(&[&tail, &head], axis)
    }

    /// Pad `axis` with `before`/`after` entries of `value` (constant pad).
    pub fn pad(&self, axis: usize, before: usize, after: usize, value: f32) -> Tensor {
        let mut parts: Vec<Tensor> = Vec::new();
        if before > 0 {
            let mut d = self.dims();
            d[axis] = before;
            parts.push(Tensor::full(&d, value));
        }
        parts.push(self.clone());
        if after > 0 {
            let mut d = self.dims();
            d[axis] = after;
            parts.push(Tensor::full(&d, value));
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        crate::array::cat(&refs, axis)
    }

    pub fn gather(&self, indices: &Tensor, axis: usize) -> Self {
        self.map_binary(BinaryRhs::Tensor(indices), |g, table, idx| {
            g.gather_(table, idx, axis)
        })
    }

    /// Gather along `axis` by an index tensor — NumPy `take` / PyTorch
    /// `index_select`. Alias for [`gather`](Self::gather).
    pub fn index_select(&self, axis: usize, indices: &Tensor) -> Self {
        self.gather(indices, axis)
    }

    /// Pick a single position along `axis`, **dropping** that axis (NumPy
    /// `a[…, i, …]`). Negative `index` counts from the end.
    pub fn select(&self, axis: usize, index: i64) -> Self {
        let dim = match self.shape().dims().get(axis) {
            Some(Dim::Static(n)) => Some(*n),
            _ => None,
        };
        let pos = crate::slice::resolve_index(index, dim, axis);
        self.narrow(axis, pos, 1).squeeze(axis)
    }

    pub fn stop_gradient(&self) -> Self {
        self.map_unary(|g, x| g.stop_gradient(x))
    }

    /// Index of the maximum along `axis` (f32-encoded indices). Drops the axis
    /// unless `keep_dim`.
    pub fn argmax(&self, axis: usize, keep_dim: bool) -> Self {
        self.map_unary(|g, x| {
            let s = rlx_ir::shape::reduce_shape(g.shape(x), &[axis], keep_dim)
                .expect("argmax shape inference");
            g.argmax(x, axis, keep_dim, s)
        })
    }

    /// Index of the minimum along `axis` (f32-encoded indices).
    pub fn argmin(&self, axis: usize, keep_dim: bool) -> Self {
        self.map_unary(|g, x| {
            let s = rlx_ir::shape::reduce_shape(g.shape(x), &[axis], keep_dim)
                .expect("argmin shape inference");
            g.argmin(x, axis, keep_dim, s)
        })
    }

    /// Matrix inverse of a square 2-D tensor. Composite over `DenseSolve`
    /// (`inv(A) = solve(A, I)`), so it needs a BLAS-backed eval build (LAPACK)
    /// to run, not the default splat backend.
    pub fn inv(&self) -> Self {
        let dims = self.dims();
        assert!(
            dims.len() == 2 && dims[0] == dims[1],
            "inv() requires a square 2-D matrix, got {dims:?}"
        );
        let n = dims[0];
        let eye_id = self.adopt(&Tensor::eye(n));
        let id = self.handle.with_graph(|g| {
            let s = Shape::new(&[n, n], DType::F32);
            g.dense_solve(self.id, eye_id, s)
        });
        Self::new(self.handle.clone(), id)
    }

    /// Solve the linear system `self @ x = b` for `x` (LAPACK via `DenseSolve`;
    /// needs a BLAS eval build). `x` takes `b`'s shape.
    pub fn solve(&self, b: &Tensor) -> Self {
        let s = b.shape();
        let b_id = self.adopt(b);
        let id = self.handle.with_graph(|g| g.dense_solve(self.id, b_id, s));
        Self::new(self.handle.clone(), id)
    }

    /// Cumulative sum along `axis`. `exclusive` shifts so `output[0]` = 0.
    pub fn cumsum(&self, axis: i32, exclusive: bool) -> Self {
        self.map_unary(|g, x| {
            let s = rlx_ir::shape::unary_shape(g.shape(x));
            g.cumsum(x, axis, exclusive, s)
        })
    }

    /// Population variance over `axes`: `mean((x - mean(x))²)`.
    pub fn var(&self, axes: impl Into<Vec<usize>>, keep_dim: bool) -> Self {
        let axes = axes.into();
        let mu = self.mean(axes.clone(), true);
        let centered = self - &mu;
        (&centered * &centered).mean(axes, keep_dim)
    }

    /// Population standard deviation over `axes`.
    pub fn std(&self, axes: impl Into<Vec<usize>>, keep_dim: bool) -> Self {
        self.var(axes, keep_dim).sqrt()
    }

    /// L2 (Frobenius) norm over `axes`: `sqrt(sum(x²))`.
    pub fn norm(&self, axes: impl Into<Vec<usize>>, keep_dim: bool) -> Self {
        (self * self).sum(axes, keep_dim).sqrt()
    }

    /// Numerically-stable `log(sum(exp(x)))` along `axis`
    /// (`max + log(sum(exp(x - max)))`).
    pub fn logsumexp(&self, axis: usize, keep_dim: bool) -> Self {
        let m = self.max([axis], true);
        let shifted = self - &m;
        let summed = shifted.exp().sum([axis], true);
        let lse = &summed.log() + &m;
        if keep_dim { lse } else { lse.squeeze(axis) }
    }

    /// 1-D FFT along the last axis (unnormalized; `ifft(fft(x)) = N·x`).
    /// Complex data uses the `[..., 2N]` block layout (real plane then imag
    /// plane) for F32/F64, or a `C64` last axis. Output matches input shape.
    pub fn fft(&self) -> Self {
        self.map_unary(|g, x| g.fft(x, false))
    }

    /// Inverse 1-D FFT along the last axis (unnormalized — pairs with [`Self::fft`]
    /// up to the `N` factor).
    pub fn ifft(&self) -> Self {
        self.map_unary(|g, x| g.fft(x, true))
    }

    /// 1-D FFT along an arbitrary `axis` (transpose → fft → transpose).
    pub fn fft_axis(&self, axis: usize, inverse: bool) -> Self {
        self.map_unary(|g, x| g.fft_axis(x, axis, inverse))
    }

    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> Self {
        let g_id = self.adopt(gamma);
        let b_id = self.adopt(beta);
        let id = self.handle.with_graph(|g| {
            let s = rlx_ir::shape::unary_shape(g.shape(self.id));
            g.layer_norm(self.id, g_id, b_id, -1, eps, s)
        });
        Self::new(self.handle.clone(), id)
    }

    pub fn rms_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> Self {
        let g_id = self.adopt(gamma);
        let b_id = self.adopt(beta);
        let id = self
            .handle
            .with_graph(|g| g.rms_norm(self.id, g_id, b_id, eps));
        Self::new(self.handle.clone(), id)
    }

    pub fn conv2d(
        &self,
        weight: &Tensor,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Self {
        let w_id = self.adopt(weight);
        let id = self.handle.with_graph(|g| {
            g.conv2d(
                self.id,
                w_id,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            )
        });
        Self::new(self.handle.clone(), id)
    }

    pub fn attention(
        &self,
        k: &Tensor,
        v: &Tensor,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
    ) -> Self {
        let k_id = self.adopt(k);
        let v_id = self.adopt(v);
        let id = self.handle.with_graph(|g| {
            let s = rlx_ir::shape::attention_shape(g.shape(self.id));
            g.attention_kind(self.id, k_id, v_id, num_heads, head_dim, mask_kind, s)
        });
        Self::new(self.handle.clone(), id)
    }

    pub fn rope(&self, cos: &Tensor, sin: &Tensor, head_dim: usize) -> Self {
        let cos_id = self.adopt(cos);
        let sin_id = self.adopt(sin);
        let id = self
            .handle
            .with_graph(|g| g.rope(self.id, cos_id, sin_id, head_dim));
        Self::new(self.handle.clone(), id)
    }

    pub fn where_(&self, on_true: &Tensor, on_false: &Tensor) -> Self {
        let t_id = self.adopt(on_true);
        let f_id = self.adopt(on_false);
        let id = self.handle.with_graph(|g| {
            let s = rlx_ir::shape::binary_shape(g.shape(t_id), g.shape(f_id))
                .expect("where shape inference");
            g.add_node(Op::Where, vec![self.id, t_id, f_id], s)
        });
        Self::new(self.handle.clone(), id)
    }

    /// Replace entries of `self` where `mask` is true with `value` (NumPy
    /// `masked_fill` / the attention-mask primitive: `scores.masked_fill(causal,
    /// f32::NEG_INFINITY)` then `softmax`). Requires a static shape.
    pub fn masked_fill(&self, mask: &Tensor, value: f64) -> Self {
        let dims: Vec<usize> = self
            .shape()
            .dims()
            .iter()
            .map(|d| match d {
                Dim::Static(n) => *n,
                Dim::Dynamic(_) => panic!("masked_fill requires a static shape"),
            })
            .collect();
        let fill = Tensor::full(dims, value as f32);
        mask.where_(&fill, self)
    }

    /// Elementwise maximum.
    pub fn maximum(&self, rhs: &Tensor) -> Self {
        self.map_binary_op(BinaryRhs::Tensor(rhs), BinaryOp::Max)
    }

    /// Elementwise minimum.
    pub fn minimum(&self, rhs: &Tensor) -> Self {
        self.map_binary_op(BinaryRhs::Tensor(rhs), BinaryOp::Min)
    }

    /// Clamp values to `[min, max]`.
    pub fn clamp(&self, min: f64, max: f64) -> Self {
        self.clamp_min(min).clamp_max(max)
    }

    /// Lower-bound values at `min`.
    pub fn clamp_min(&self, min: f64) -> Self {
        self.map_binary_op(BinaryRhs::Scalar(min.into()), BinaryOp::Max)
    }

    /// Upper-bound values at `max`.
    pub fn clamp_max(&self, max: f64) -> Self {
        self.map_binary_op(BinaryRhs::Scalar(max.into()), BinaryOp::Min)
    }

    pub fn pow(&self, rhs: &Tensor) -> Self {
        self.map_binary_op(BinaryRhs::Tensor(rhs), BinaryOp::Pow)
    }

    pub fn pow_scalar(&self, exp: f64) -> Self {
        self.map_binary_op(BinaryRhs::Scalar(exp.into()), BinaryOp::Pow)
    }
}

unary_ops! {
    /// Rectified linear unit `max(x, 0)`.
    relu => Relu,
    /// Gaussian error linear unit (exact).
    gelu => Gelu,
    /// GELU, tanh approximation.
    gelu_approx => GeluApprox,
    /// Sigmoid-weighted linear unit `x · σ(x)` (a.k.a. SwiGLU gate).
    silu => Silu,
    /// Hyperbolic tangent.
    tanh => Tanh,
    /// Natural exponential `e^x`.
    exp => Exp,
    /// Square root.
    sqrt => Sqrt,
    /// Logistic sigmoid `1 / (1 + e^-x)`.
    sigmoid => Sigmoid,
    /// Natural logarithm.
    log => Log,
    /// Reciprocal square root `1 / sqrt(x)`.
    rsqrt => Rsqrt,
    /// Absolute value.
    abs => Abs,
    /// Sine.
    sin => Sin,
    /// Cosine.
    cos => Cos,
    /// Tangent.
    tan => Tan,
    /// Arctangent.
    atan => Atan,
    /// Round to nearest (half-to-even); straight-through gradient.
    round => Round,
}

cmp_ops! {
    /// Elementwise `==`.
    eq => Eq,
    /// Elementwise `!=`.
    ne => Ne,
    /// Elementwise `<`.
    lt => Lt,
    /// Elementwise `<=`.
    le => Le,
    /// Elementwise `>`.
    gt => Gt,
    /// Elementwise `>=`.
    ge => Ge,
}

reduce_ops! {
    /// Sum over `axes`.
    sum => Sum,
    /// Mean over `axes`.
    mean => Mean,
    /// Maximum over `axes`.
    max => Max,
    /// Minimum over `axes`.
    min => Min,
    /// Product over `axes`.
    prod => Prod,
}

/// Generate typed little-endian readback methods (`name => (T, DType, width)`):
/// realize, assert the tensor dtype, then decode fixed-width POD elements.
/// `f32` (the default) and `bool` (multi-encoding) are hand-rolled separately.
#[cfg(feature = "eval")]
macro_rules! pod_readers {
    ($($(#[$m:meta])* $name:ident => ($ty:ty, $dt:ident, $width:literal)),+ $(,)?) => {
        $(
            $(#[$m])*
            pub fn $name(&self) -> Vec<$ty> {
                let (bytes, dt) = self.eval_typed();
                assert_eq!(
                    dt, DType::$dt,
                    concat!(stringify!($name), ": tensor dtype is {:?}"), dt,
                );
                bytes
                    .chunks_exact($width)
                    .map(|c| <$ty>::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            }
        )+
    };
}

/// Materialization — compile this tensor's graph through `rlx_runtime` and
/// read the result back to host memory. The graph stays lazy until one of
/// these is called, so the compiler fuses and memory-plans the whole
/// expression first. Available with the `eval` feature.
#[cfg(feature = "eval")]
impl Tensor {
    /// Compile + run and copy the output to a `Vec<f32>`. The device is chosen
    /// automatically — the fastest backend compiled into this build that can
    /// run the graph (CPU when nothing faster is available).
    pub fn to_vec(&self) -> Vec<f32> {
        self.to_vec_on(self.auto_device())
    }

    /// Compile + run on an explicit device.
    pub fn to_vec_on(&self, device: rlx_runtime::Device) -> Vec<f32> {
        // A raw constant's realized value is its own host data — device-independent,
        // so return it directly and skip backend compilation. This is both an
        // optimization (no GPU round-trip for a constant) and a robustness fix: the
        // MLX backend crashes when asked to compile a constant-only trace.
        if let Some((bytes, DType::F32)) = self.constant_payload() {
            return bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
        }
        let compiled = self.compiled_on(device);
        let mut outputs = compiled.borrow_mut().run(&[]);
        outputs.pop().expect("eval produced no output")
    }

    /// If this tensor's output node is a raw `Op::Constant` that physically holds its
    /// full contiguous payload, returns `(bytes, dtype)` — its realized value with no
    /// computation to run. (Views/slices of a constant are *not* raw constants, so
    /// they fall through to normal evaluation.)
    fn constant_payload(&self) -> Option<(Vec<u8>, DType)> {
        self.handle.with_graph(|g| {
            let node = g.node(self.id);
            if let Op::Constant { data } = &node.op {
                if node.shape.size_bytes() == Some(data.len()) {
                    return Some((data.clone(), node.shape.dtype()));
                }
            }
            None
        })
    }

    /// Fastest available backend that can run this tensor's graph (no clone).
    fn auto_device(&self) -> rlx_runtime::Device {
        self.handle
            .with_graph(|g| rlx_runtime::fastest_device_for(g))
    }

    /// Compiled graph for this tensor's output — borrows the shared graph for
    /// the cache fingerprint, cloning only on a miss (zero-copy on hits).
    fn compiled_on(
        &self,
        device: rlx_runtime::Device,
    ) -> std::rc::Rc<std::cell::RefCell<rlx_runtime::CompiledGraph>> {
        self.handle
            .with_graph(|g| crate::cache::compiled_output(g, self.id, device))
    }

    /// Force evaluation — alias for [`to_vec`](Self::to_vec). Building a
    /// `Tensor` allocates **nothing** and runs **nothing**; it only records IR.
    /// `realize` is the single point where the accumulated graph is compiled
    /// (fused + memory-planned) and executed. Use it to make the lazy model
    /// explicit at call sites.
    pub fn realize(&self) -> Vec<f32> {
        self.to_vec()
    }

    /// [`realize`](Self::realize) on an explicit device.
    pub fn realize_on(&self, device: rlx_runtime::Device) -> Vec<f32> {
        self.to_vec_on(device)
    }

    /// Realize a scalar (rank-0 or single-element) tensor to one `f32`.
    pub fn item(&self) -> f32 {
        let v = self.to_vec();
        assert_eq!(v.len(), 1, "item(): tensor has {} elements, not 1", v.len());
        v[0]
    }

    pod_readers! {
        /// Realize and read back as `f64` (tensor dtype must be `F64`).
        to_vec_f64 => (f64, F64, 8),
        /// Realize and read back as `i64` (tensor dtype must be `I64`).
        to_vec_i64 => (i64, I64, 8),
        /// Realize and read back as `i32` (tensor dtype must be `I32`).
        to_vec_i32 => (i32, I32, 4),
    }

    /// Realize and read back a `Bool` tensor. Robust to the backend storing
    /// bools 1-byte or f32-encoded (an element is `true` iff any of its bytes
    /// is non-zero).
    pub fn to_vec_bool(&self) -> Vec<bool> {
        let (bytes, dt) = self.eval_typed();
        assert_eq!(dt, DType::Bool, "to_vec_bool: tensor dtype is {dt:?}");
        let n = self.numel();
        assert!(n > 0 && bytes.len() % n == 0, "to_vec_bool: ragged output");
        let w = bytes.len() / n;
        bytes
            .chunks_exact(w)
            .map(|c| c.iter().any(|&b| b != 0))
            .collect()
    }

    /// Compile + run, returning the output's raw little-endian bytes + dtype.
    /// Device is auto-selected; backs the typed `to_vec_*` readers.
    fn eval_typed(&self) -> (Vec<u8>, DType) {
        // Raw constant → its bytes are the result (see `to_vec_on`).
        if let Some(ct) = self.constant_payload() {
            return ct;
        }
        let compiled = self.compiled_on(self.auto_device());
        let mut outputs = compiled.borrow_mut().run_typed(&[]);
        outputs.pop().expect("eval produced no output")
    }

    /// Realize and pretty-print with shape + dtype, NumPy/ndarray-style nesting.
    /// `println!("{}", t)` does the same (with the `eval` feature).
    pub fn show(&self) -> String {
        let dims = self.dims();
        let data = self.to_vec();
        format!(
            "Tensor{dims:?} {:?}\n{}",
            self.dtype(),
            fmt_nested(&data, &dims)
        )
    }

    /// Select a device fluently: `t.on(Device::Metal).to_vec()`.
    pub fn on(&self, device: rlx_runtime::Device) -> Materialize<'_> {
        Materialize {
            tensor: self,
            device,
        }
    }
}

/// Fluent device selector returned by [`Tensor::on`].
#[cfg(feature = "eval")]
pub struct Materialize<'a> {
    tensor: &'a Tensor,
    device: rlx_runtime::Device,
}

#[cfg(feature = "eval")]
impl Materialize<'_> {
    /// Compile + run on the selected device and copy the output to host.
    pub fn to_vec(self) -> Vec<f32> {
        self.tensor.to_vec_on(self.device)
    }
}

/// Reverse-mode autodiff. Available with the `grad` feature; pair with `eval`
/// to read the resulting gradient values.
#[cfg(feature = "autodiff")]
impl Tensor {
    /// Gradients of this tensor (treated as the loss) w.r.t. each `wrt`,
    /// returned as symbolic [`Tensor`]s in the same order as `wrt`.
    ///
    /// The seed `∂self/∂self = 1` is baked in as a constant (ones, matching
    /// `self`'s shape — so a non-scalar `self` differentiates `sum(self)`),
    /// making each returned gradient a self-contained graph you can
    /// `to_vec()` or compose further.
    ///
    /// ```ignore
    /// let loss = (&a * &b).sum([0], false);
    /// let g = loss.grad(&[&a, &b]);   // [∂/∂a, ∂/∂b]
    /// assert_eq!(g[0].to_vec(), b.to_vec());
    /// ```
    pub fn grad(&self, wrt: &[&Tensor]) -> Vec<Tensor> {
        // Resolve each `wrt` to its node id *within this loss's graph*.
        // Memoized adoption guarantees these are the exact nodes the loss was
        // built from, so gradient actually flows (not to a stale duplicate).
        let wrt_ids: Vec<NodeId> = wrt.iter().map(|t| self.adopt(t)).collect();
        let loss_id = self.id;
        let bwd = self.handle.with_graph(|g| {
            let mut forward = g.clone();
            forward.set_outputs(vec![loss_id]);
            let mut bwd = rlx_autodiff::grad(&forward, &wrt_ids);
            bake_unit_seed(&mut bwd);
            bwd
        });
        let handle = GraphHandle::new(bwd);
        let outputs = handle.with_graph(|g| g.outputs.clone());
        outputs
            .into_iter()
            .map(|id| Tensor::new(handle.clone(), id))
            .collect()
    }
}

/// Replace `rlx_autodiff`'s `d_output` seed input with a constant of ones, so
/// the backward graph runs with no external inputs.
#[cfg(feature = "autodiff")]
pub(crate) fn bake_unit_seed(bwd: &mut rlx_ir::Graph) {
    let seed = bwd.nodes().iter().find_map(|n| match &n.op {
        Op::Input { name } if name == "d_output" => Some(n.id),
        _ => None,
    });
    if let Some(id) = seed {
        let n = bwd.node(id).shape.num_elements().unwrap_or(1);
        let data: Vec<u8> = (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let node = bwd.node_mut(id);
        node.op = Op::Constant { data };
        node.inputs = Vec::new();
    }
}

impl From<Tensor> for NodeId {
    fn from(v: Tensor) -> Self {
        v.id
    }
}

impl From<&Tensor> for NodeId {
    fn from(v: &Tensor) -> Self {
        v.id
    }
}

pub(crate) enum BinaryRhs<'a> {
    Tensor(&'a Tensor),
    Scalar(Scalar),
}

impl<'a> BinaryRhs<'a> {
    fn scalar(s: Scalar) -> Self {
        Self::Scalar(s)
    }
}

macro_rules! impl_scalar_rhs {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl From<$ty> for BinaryRhs<'_> {
                fn from(v: $ty) -> Self {
                    Self::Scalar(v.into())
                }
            }
        )+
    };
}

impl_scalar_rhs!(bool, i32, i64, f32, f64);

macro_rules! impl_tensor_binop {
    ($trait:ident, $method:ident, $graph:ident) => {
        impl $trait<&Tensor> for &Tensor {
            type Output = Tensor;

            fn $method(self, rhs: &Tensor) -> Tensor {
                self.map_binary(BinaryRhs::Tensor(rhs), |g, a, b| g.$graph(a, b))
            }
        }

        impl $trait<Tensor> for &Tensor {
            type Output = Tensor;

            fn $method(self, rhs: Tensor) -> Tensor {
                self.$method(&rhs)
            }
        }

        impl $trait<f64> for &Tensor {
            type Output = Tensor;

            fn $method(self, rhs: f64) -> Tensor {
                self.map_binary(BinaryRhs::scalar(rhs.into()), |g, a, b| g.$graph(a, b))
            }
        }

        impl $trait<f32> for &Tensor {
            type Output = Tensor;

            fn $method(self, rhs: f32) -> Tensor {
                self.$method(rhs as f64)
            }
        }

        impl $trait<i32> for &Tensor {
            type Output = Tensor;

            fn $method(self, rhs: i32) -> Tensor {
                self.map_binary(BinaryRhs::scalar(rhs.into()), |g, a, b| g.$graph(a, b))
            }
        }

        impl $trait<&Tensor> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: &Tensor) -> Tensor {
                (&self).$method(rhs)
            }
        }

        impl $trait<Tensor> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: Tensor) -> Tensor {
                (&self).$method(&rhs)
            }
        }

        impl $trait<f64> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: f64) -> Tensor {
                (&self).$method(rhs)
            }
        }

        impl $trait<f32> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: f32) -> Tensor {
                (&self).$method(rhs)
            }
        }

        impl $trait<i32> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: i32) -> Tensor {
                (&self).$method(rhs)
            }
        }
    };
}

impl_tensor_binop!(Add, add, add);
impl_tensor_binop!(Sub, sub, sub);
impl_tensor_binop!(Mul, mul, mul);
impl_tensor_binop!(Div, div, div);

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self.map_unary(|g, x| g.neg(x))
    }
}

macro_rules! impl_scalar_left {
    ($trait:ident, $method:ident, $graph:ident) => {
        impl $trait<&Tensor> for f64 {
            type Output = Tensor;

            fn $method(self, rhs: &Tensor) -> Tensor {
                let id = rhs.handle.with_graph(|g| {
                    let lhs_id = promote_scalar(g, self.into(), g.shape(rhs.id).dtype());
                    g.$graph(lhs_id, rhs.id)
                });
                Tensor::new(rhs.handle.clone(), id)
            }
        }

        impl $trait<&Tensor> for f32 {
            type Output = Tensor;

            fn $method(self, rhs: &Tensor) -> Tensor {
                (self as f64).$method(rhs)
            }
        }

        impl $trait<&Tensor> for i32 {
            type Output = Tensor;

            fn $method(self, rhs: &Tensor) -> Tensor {
                (self as i64 as f64).$method(rhs)
            }
        }
    };
}

impl_scalar_left!(Add, add, add);
impl_scalar_left!(Sub, sub, sub);
impl_scalar_left!(Mul, mul, mul);
impl_scalar_left!(Div, div, div);

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        (&self).neg()
    }
}
