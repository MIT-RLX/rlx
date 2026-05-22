// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

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
        self.dims[i]
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

/// Concat along axis.
pub fn concat_shape(inputs: &[&Shape], axis: usize) -> Result<Shape, String> {
    if inputs.is_empty() {
        return Err("concat: no inputs".into());
    }
    let base = inputs[0];
    let mut static_sum = 0usize;
    let mut dyn_sym: Option<u32> = None;
    for s in inputs {
        if s.rank() != base.rank() {
            return Err(format!(
                "concat: rank mismatch {} vs {}",
                s.rank(),
                base.rank()
            ));
        }
        match s.dims[axis] {
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
    Ok(base.clone().with_dim(axis, out_dim))
}

/// Gather (embedding lookup): table\[V,D\] + indices\[B,S\] → \[B,S,D\].
pub fn gather_shape(table: &Shape, indices: &Shape, axis: usize) -> Result<Shape, String> {
    if axis >= table.rank() {
        return Err(format!("gather: axis {axis} >= rank {}", table.rank()));
    }
    let mut dims: SmallVec<[Dim; 4]> = indices.dims.clone();
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
        let total = input
            .num_elements()
            .ok_or_else(|| "reshape: input has dynamic dims".to_string())?;
        let known_product: i64 = new_shape.iter().filter(|&&d| d != -1).product();
        let mut dims = SmallVec::new();
        for &d in new_shape {
            if d == -1 {
                let inferred = total as i64 / known_product;
                dims.push(Dim::Static(inferred as usize));
            } else if d < 0 {
                return Err(format!("reshape: invalid dim {d}"));
            } else {
                dims.push(Dim::Static(d as usize));
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
        Dim::Static(
            leading
                .iter()
                .map(|d| d.unwrap_static())
                .product::<usize>(),
        )
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
            0 => Dim::Static(
                leading
                    .iter()
                    .map(|d| d.unwrap_static())
                    .product::<usize>(),
            ),
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
    (in_size + 2 * padding).saturating_sub(dil_k).saturating_sub(1) / stride + 1
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
    let n = input.dim(0).unwrap_static();
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
    Ok(Shape::new(&[n, c_out, h_out, w_out], input.dtype()))
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
    let n = input.dim(0).unwrap_static();
    let c_in = input.dim(1).unwrap_static();
    let h = input.dim(2).unwrap_static();
    let w = input.dim(3).unwrap_static();
    let w_cin = weight.dim(0).unwrap_static();
    let c_out_per_g = weight.dim(1).unwrap_static();
    if w_cin != c_in {
        return Err(format!("conv_transpose2d weight C_in={w_cin} != input C={c_in}"));
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
    Ok(Shape::new(
        &[n, c_out_per_g * groups, h_out, w_out],
        input.dtype(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_shape() {
        let s = Shape::new(&[4, 15, 384], DType::F32);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.num_elements(), Some(4 * 15 * 384));
        assert_eq!(s.size_bytes(), Some(4 * 15 * 384 * 4));
        assert!(s.is_static());
        assert_eq!(format!("{s}"), "[4, 15, 384] f32");
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
