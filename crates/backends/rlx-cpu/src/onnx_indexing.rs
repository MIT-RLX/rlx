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

//! Reference CPU kernels for ONNX indexing / contraction ops that lower to
//! `Op::Custom("onnx.*")`: GatherND, ScatterND, OneHot, NonZero, CumProd, and
//! Einsum. These are correctness-first reference implementations exercised by
//! the ONNX import path on the CPU backend; the per-element loops are not the
//! perf-critical path (those ops are rare in the LLM/vision hot loops).

use std::sync::Arc;

use rlx_ir::DType;

use crate::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

const GATHER_ND: &str = "onnx.GatherND";
const SCATTER_ND: &str = "onnx.ScatterND";
const ONE_HOT: &str = "onnx.OneHot";
const NON_ZERO: &str = "onnx.NonZero";
const CUM_PROD: &str = "onnx.CumProd";
const EINSUM: &str = "onnx.Einsum";

fn dims_of(shape: &rlx_ir::Shape) -> Vec<usize> {
    shape
        .dims()
        .iter()
        .map(|d| match d {
            rlx_ir::Dim::Static(n) => *n,
            rlx_ir::Dim::Dynamic(_) => 0,
        })
        .collect()
}

fn dims_usize(t: &CpuTensorRef<'_>) -> Vec<usize> {
    dims_of(t.shape())
}

/// Row-major strides for `shape` (element strides, not bytes).
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// GatherND
// ---------------------------------------------------------------------------

/// Source element offsets, in output order, for an ONNX GatherND.
///
/// `data_shape`/`indices_shape` are the full operand shapes, `indices` the flat
/// I64 index buffer, `batch_dims` the leading shared-batch rank. Each returned
/// offset is the start of a contiguous `slice_size`-element run in `data`.
fn gather_nd_src_offsets(
    data_shape: &[usize],
    indices: &[i64],
    indices_shape: &[usize],
    batch_dims: usize,
) -> (Vec<usize>, usize) {
    let k = indices_shape.last().copied().unwrap_or(0);
    let b = batch_dims.min(data_shape.len()).min(indices_shape.len());
    let data_strides = row_major_strides(data_shape);
    let slice_size: usize = data_shape
        .get(b + k..)
        .map(|s| s.iter().product())
        .unwrap_or(1);
    let batch_count: usize = data_shape[..b].iter().product();
    let tuples_per_batch: usize = indices_shape
        .get(b..indices_shape.len().saturating_sub(1))
        .map(|s| s.iter().product())
        .unwrap_or(1);
    // Element stride between consecutive batch entries in `data`.
    let batch_stride: usize = data_shape.get(b..).map(|s| s.iter().product()).unwrap_or(1);

    let mut offsets = Vec::with_capacity(batch_count * tuples_per_batch);
    for bi in 0..batch_count.max(1) {
        let batch_base = bi * batch_stride;
        for ti in 0..tuples_per_batch.max(1) {
            let tuple = (bi * tuples_per_batch + ti) * k;
            let mut off = batch_base;
            for m in 0..k {
                let mut idx = indices.get(tuple + m).copied().unwrap_or(0);
                let dim = data_shape.get(b + m).copied().unwrap_or(1) as i64;
                if idx < 0 {
                    idx += dim;
                }
                let idx = idx.clamp(0, dim.saturating_sub(1).max(0)) as usize;
                off += idx * data_strides[b + m];
            }
            offsets.push(off);
        }
    }
    (offsets, slice_size)
}

struct GatherNdKernel;

impl CpuKernel for GatherNdKernel {
    fn name(&self) -> &str {
        GATHER_ND
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.len() < 2 {
            return Err(format!(
                "onnx.GatherND: expected 2 inputs, got {}",
                inputs.len()
            ));
        }
        let batch_dims = if attrs.len() >= 4 {
            i32::from_le_bytes(attrs[0..4].try_into().unwrap()).max(0) as usize
        } else {
            0
        };
        let data_shape = dims_usize(&inputs[0]);
        let indices_shape = dims_usize(&inputs[1]);
        let indices = inputs[1].expect_i64("indices")?;
        let (offsets, slice) =
            gather_nd_src_offsets(&data_shape, indices, &indices_shape, batch_dims);
        match inputs[0].dtype() {
            DType::I64 => {
                let data = inputs[0].expect_i64("data")?;
                let out = output.expect_i64_mut("out")?;
                copy_slices(data, out, &offsets, slice);
            }
            _ => {
                let data = inputs[0].expect_f32("data")?;
                let out = output.expect_f32_mut("out")?;
                copy_slices(data, out, &offsets, slice);
            }
        }
        Ok(())
    }
}

fn copy_slices<T: Copy>(src: &[T], dst: &mut [T], src_offsets: &[usize], slice: usize) {
    for (t, &off) in src_offsets.iter().enumerate() {
        let d0 = t * slice;
        for j in 0..slice {
            if let (Some(s), Some(d)) = (src.get(off + j), dst.get_mut(d0 + j)) {
                *d = *s;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ScatterND
// ---------------------------------------------------------------------------

/// Destination element offsets, in updates order, for an ONNX ScatterND.
fn scatter_nd_dst_offsets(
    data_shape: &[usize],
    indices: &[i64],
    indices_shape: &[usize],
) -> (Vec<usize>, usize) {
    let k = indices_shape.last().copied().unwrap_or(0);
    let data_strides = row_major_strides(data_shape);
    let slice_size: usize = data_shape.get(k..).map(|s| s.iter().product()).unwrap_or(1);
    let num_updates: usize = indices_shape
        .get(..indices_shape.len().saturating_sub(1))
        .map(|s| s.iter().product())
        .unwrap_or(1);
    let mut offsets = Vec::with_capacity(num_updates);
    for u in 0..num_updates.max(1) {
        let mut off = 0usize;
        for m in 0..k {
            let mut idx = indices.get(u * k + m).copied().unwrap_or(0);
            let dim = data_shape.get(m).copied().unwrap_or(1) as i64;
            if idx < 0 {
                idx += dim;
            }
            let idx = idx.clamp(0, dim.saturating_sub(1).max(0)) as usize;
            off += idx * data_strides[m];
        }
        offsets.push(off);
    }
    (offsets, slice_size)
}

struct ScatterNdKernel;

impl CpuKernel for ScatterNdKernel {
    fn name(&self) -> &str {
        SCATTER_ND
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.len() < 3 {
            return Err(format!(
                "onnx.ScatterND: expected 3 inputs, got {}",
                inputs.len()
            ));
        }
        let data_shape = dims_usize(&inputs[0]);
        let indices_shape = dims_usize(&inputs[1]);
        let indices = inputs[1].expect_i64("indices")?;
        let (offsets, slice) = scatter_nd_dst_offsets(&data_shape, indices, &indices_shape);
        match inputs[0].dtype() {
            DType::I64 => {
                let data = inputs[0].expect_i64("data")?;
                let updates = inputs[2].expect_i64("updates")?;
                let out = output.expect_i64_mut("out")?;
                scatter_into(data, updates, out, &offsets, slice);
            }
            _ => {
                let data = inputs[0].expect_f32("data")?;
                let updates = inputs[2].expect_f32("updates")?;
                let out = output.expect_f32_mut("out")?;
                scatter_into(data, updates, out, &offsets, slice);
            }
        }
        Ok(())
    }
}

fn scatter_into<T: Copy>(
    data: &[T],
    updates: &[T],
    out: &mut [T],
    dst_offsets: &[usize],
    slice: usize,
) {
    // Start from a copy of `data` (skip when aliased in place).
    if !std::ptr::eq(data.as_ptr(), out.as_ptr()) {
        let n = data.len().min(out.len());
        out[..n].copy_from_slice(&data[..n]);
    }
    for (u, &off) in dst_offsets.iter().enumerate() {
        let u0 = u * slice;
        for j in 0..slice {
            if let (Some(s), Some(d)) = (updates.get(u0 + j), out.get_mut(off + j)) {
                *d = *s;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OneHot
// ---------------------------------------------------------------------------

struct OneHotKernel;

impl CpuKernel for OneHotKernel {
    fn name(&self) -> &str {
        ONE_HOT
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.len() < 3 {
            return Err(format!(
                "onnx.OneHot: expected 3 inputs, got {}",
                inputs.len()
            ));
        }
        let axis_attr = if attrs.len() >= 4 {
            i32::from_le_bytes(attrs[0..4].try_into().unwrap())
        } else {
            -1
        };
        let out_shape = dims_of(output.shape());
        let rank = out_shape.len().max(1);
        let axis = if axis_attr < 0 {
            (rank as i32 + axis_attr).max(0) as usize
        } else {
            (axis_attr as usize).min(rank - 1)
        };
        let depth = out_shape.get(axis).copied().unwrap_or(0);
        let inner: usize = out_shape
            .get(axis + 1..)
            .map(|s| s.iter().product())
            .unwrap_or(1);
        let indices = inputs[0].expect_i64("indices")?;

        match output.dtype() {
            DType::I64 => {
                let (off, on) = onehot_values_i64(&inputs[2]);
                let out = output.expect_i64_mut("out")?;
                one_hot_fill(out, indices, depth, inner, off, on);
            }
            _ => {
                let (off, on) = onehot_values_f32(&inputs[2]);
                let out = output.expect_f32_mut("out")?;
                one_hot_fill(out, indices, depth, inner, off, on);
            }
        }
        Ok(())
    }
}

fn onehot_values_f32(v: &CpuTensorRef<'_>) -> (f32, f32) {
    let s = v.as_f32().unwrap_or(&[]);
    (
        s.first().copied().unwrap_or(0.0),
        s.get(1).copied().unwrap_or(1.0),
    )
}

fn onehot_values_i64(v: &CpuTensorRef<'_>) -> (i64, i64) {
    let s = v.as_i64().unwrap_or(&[]);
    (
        s.first().copied().unwrap_or(0),
        s.get(1).copied().unwrap_or(1),
    )
}

fn one_hot_fill<T: Copy>(
    out: &mut [T],
    indices: &[i64],
    depth: usize,
    inner: usize,
    off: T,
    on: T,
) {
    out.fill(off);
    if depth == 0 {
        return;
    }
    for (p, &v) in indices.iter().enumerate() {
        let mut v = v;
        if v < 0 {
            v += depth as i64;
        }
        if v < 0 || v as usize >= depth {
            continue;
        }
        let outer_idx = p / inner;
        let inner_idx = p % inner;
        let dst = outer_idx * depth * inner + (v as usize) * inner + inner_idx;
        if let Some(d) = out.get_mut(dst) {
            *d = on;
        }
    }
}

// ---------------------------------------------------------------------------
// NonZero
// ---------------------------------------------------------------------------

struct NonZeroKernel;

impl CpuKernel for NonZeroKernel {
    fn name(&self) -> &str {
        NON_ZERO
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.is_empty() {
            return Err("onnx.NonZero: missing input".into());
        }
        let in_shape = dims_usize(&inputs[0]);
        let rank = in_shape.len().max(1);
        // Collect flat positions of non-zero elements (row-major order).
        let nz: Vec<usize> = match inputs[0].dtype() {
            DType::I64 => nonzero_positions(inputs[0].expect_i64("x")?, |&v| v != 0),
            DType::Bool => nonzero_positions(inputs[0].expect_bool("x")?, |&v| v != 0),
            _ => nonzero_positions(inputs[0].expect_f32("x")?, |&v| v != 0.0),
        };
        let out = output.expect_i64_mut("indices")?;
        out.fill(0);
        // Output is laid out [rank, cols]; column count comes from the buffer.
        let cols = out.len().checked_div(rank).unwrap_or(0);
        let strides = row_major_strides(&in_shape);
        for (c, &flat) in nz.iter().enumerate() {
            if c >= cols {
                break;
            }
            let mut rem = flat;
            for d in 0..rank {
                let coord = rem.checked_div(strides[d]).unwrap_or(0);
                rem %= strides[d].max(1);
                out[d * cols + c] = coord as i64;
            }
        }
        Ok(())
    }
}

fn nonzero_positions<T>(data: &[T], is_nz: impl Fn(&T) -> bool) -> Vec<usize> {
    data.iter()
        .enumerate()
        .filter_map(|(i, v)| if is_nz(v) { Some(i) } else { None })
        .collect()
}

// ---------------------------------------------------------------------------
// CumProd
// ---------------------------------------------------------------------------

struct CumProdKernel;

impl CpuKernel for CumProdKernel {
    fn name(&self) -> &str {
        CUM_PROD
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        if inputs.is_empty() {
            return Err("onnx.CumProd: missing input".into());
        }
        let shape = dims_usize(&inputs[0]);
        let rank = shape.len().max(1) as i64;
        // `axis` arrives either as a second operand (codegen path) or baked
        // into the first attr word (import path); `exclusive`/`reverse` always
        // live in the attr blob at bytes 4/5.
        let raw_axis = inputs
            .get(1)
            .and_then(|t| t.as_i64())
            .and_then(|v| v.first().copied())
            .unwrap_or_else(|| {
                if attrs.len() >= 4 {
                    i32::from_le_bytes(attrs[0..4].try_into().unwrap()) as i64
                } else {
                    0
                }
            });
        let axis = if raw_axis < 0 {
            raw_axis + rank
        } else {
            raw_axis
        }
        .clamp(0, rank - 1) as usize;
        let exclusive = attrs.get(4).copied().unwrap_or(0) != 0;
        let reverse = attrs.get(5).copied().unwrap_or(0) != 0;
        match inputs[0].dtype() {
            DType::I64 => {
                let x = inputs[0].expect_i64("x")?;
                let out = output.expect_i64_mut("out")?;
                cumprod(x, out, &shape, axis, exclusive, reverse, 1i64);
            }
            _ => {
                let x = inputs[0].expect_f32("x")?;
                let out = output.expect_f32_mut("out")?;
                cumprod(x, out, &shape, axis, exclusive, reverse, 1f32);
            }
        }
        Ok(())
    }
}

fn cumprod<T: Copy + std::ops::Mul<Output = T>>(
    x: &[T],
    out: &mut [T],
    shape: &[usize],
    axis: usize,
    exclusive: bool,
    reverse: bool,
    one: T,
) {
    if shape.is_empty() {
        let n = x.len().min(out.len());
        out[..n].copy_from_slice(&x[..n]);
        return;
    }
    let axis = axis.min(shape.len() - 1);
    let l = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let outer: usize = shape[..axis].iter().product();
    for o in 0..outer {
        for i in 0..inner {
            let base = o * l * inner + i;
            let mut running = one;
            for step in 0..l {
                let k = if reverse { l - 1 - step } else { step };
                let pos = base + k * inner;
                if exclusive {
                    out[pos] = running;
                    running = running * x[pos];
                } else {
                    running = running * x[pos];
                    out[pos] = running;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Einsum
// ---------------------------------------------------------------------------

struct EinsumKernel;

impl CpuKernel for EinsumKernel {
    fn name(&self) -> &str {
        EINSUM
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let equation = std::str::from_utf8(attrs)
            .map_err(|_| "onnx.Einsum: equation is not valid UTF-8".to_string())?;
        let shapes: Vec<Vec<usize>> = inputs.iter().map(dims_usize).collect();
        let plan = EinsumPlan::parse(equation, &shapes)?;
        match output.dtype() {
            DType::I64 => {
                let operands: Vec<&[i64]> =
                    inputs.iter().map(|t| t.as_i64().unwrap_or(&[])).collect();
                let out = output.expect_i64_mut("out")?;
                plan.contract(&operands, out, 0i64, |a, b| a + b, |a, b| a * b);
            }
            _ => {
                let operands: Vec<&[f32]> =
                    inputs.iter().map(|t| t.as_f32().unwrap_or(&[])).collect();
                let out = output.expect_f32_mut("out")?;
                plan.contract(&operands, out, 0f32, |a, b| a + b, |a, b| a * b);
            }
        }
        Ok(())
    }
}

/// A parsed Einsum equation bound to concrete operand shapes. Does not support
/// ellipsis (`...`) — those equations fall back to an error at parse time.
struct EinsumPlan {
    /// Per-operand label sequence (indices into `label_sizes`).
    operand_labels: Vec<Vec<usize>>,
    /// Output label sequence (indices into `label_sizes`).
    output_labels: Vec<usize>,
    /// Size of each distinct label.
    label_sizes: Vec<usize>,
}

impl EinsumPlan {
    fn parse(equation: &str, shapes: &[Vec<usize>]) -> Result<EinsumPlan, String> {
        let eq: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
        if eq.contains("...") {
            return Err("onnx.Einsum: ellipsis equations are not supported".into());
        }
        let (lhs, rhs) = match eq.split_once("->") {
            Some((l, r)) => (l.to_string(), Some(r.to_string())),
            None => (eq.clone(), None),
        };
        let terms: Vec<&str> = lhs.split(',').collect();
        if terms.len() != shapes.len() {
            return Err(format!(
                "onnx.Einsum: equation has {} terms but {} operands",
                terms.len(),
                shapes.len()
            ));
        }
        // Assign a dense id + size to each distinct label.
        let mut label_id: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        let mut label_sizes: Vec<usize> = Vec::new();
        let mut label_counts: std::collections::HashMap<char, usize> =
            std::collections::HashMap::new();
        let mut operand_labels: Vec<Vec<usize>> = Vec::with_capacity(terms.len());
        for (ti, term) in terms.iter().enumerate() {
            let chars: Vec<char> = term.chars().collect();
            if chars.len() != shapes[ti].len() {
                return Err(format!(
                    "onnx.Einsum: term '{term}' rank {} != operand rank {}",
                    chars.len(),
                    shapes[ti].len()
                ));
            }
            let mut ids = Vec::with_capacity(chars.len());
            for (ci, &c) in chars.iter().enumerate() {
                let size = shapes[ti][ci];
                let id = *label_id.entry(c).or_insert_with(|| {
                    label_sizes.push(size);
                    label_sizes.len() - 1
                });
                // Broadcast/consistency: keep the largest non-unit size seen.
                if label_sizes[id] <= 1 {
                    label_sizes[id] = size;
                }
                *label_counts.entry(c).or_insert(0) += 1;
                ids.push(id);
            }
            operand_labels.push(ids);
        }
        let output_chars: Vec<char> = match rhs {
            Some(r) => r.chars().collect(),
            None => {
                // Implicit output: labels appearing exactly once, sorted.
                let mut once: Vec<char> = label_counts
                    .iter()
                    .filter(|&(_, &n)| n == 1)
                    .map(|(&c, _)| c)
                    .collect();
                once.sort_unstable();
                once
            }
        };
        let output_labels: Vec<usize> = output_chars
            .iter()
            .map(|c| {
                label_id
                    .get(c)
                    .copied()
                    .ok_or_else(|| format!("onnx.Einsum: output label '{c}' absent from inputs"))
            })
            .collect::<Result<_, _>>()?;
        Ok(EinsumPlan {
            operand_labels,
            output_labels,
            label_sizes,
        })
    }

    /// Contract operands into `out`. `add`/`mul` are the accumulate/product ops;
    /// `zero` seeds the output. Output buffer is overwritten.
    fn contract<T: Copy>(
        &self,
        operands: &[&[T]],
        out: &mut [T],
        zero: T,
        add: impl Fn(T, T) -> T,
        mul: impl Fn(T, T) -> T,
    ) {
        out.fill(zero);
        let n_labels = self.label_sizes.len();
        // Per-operand strides over the global label space (0 if absent).
        let operand_strides: Vec<Vec<usize>> = self
            .operand_labels
            .iter()
            .map(|labels| {
                let local = row_major_strides(
                    &labels
                        .iter()
                        .map(|&l| self.label_sizes[l])
                        .collect::<Vec<_>>(),
                );
                let mut g = vec![0usize; n_labels];
                for (pos, &l) in labels.iter().enumerate() {
                    g[l] += local[pos];
                }
                g
            })
            .collect();
        let out_dims: Vec<usize> = self
            .output_labels
            .iter()
            .map(|&l| self.label_sizes[l])
            .collect();
        let out_local = row_major_strides(&out_dims);
        let mut out_stride = vec![0usize; n_labels];
        for (pos, &l) in self.output_labels.iter().enumerate() {
            out_stride[l] += out_local[pos];
        }
        // Odometer over all label index combinations.
        let mut idx = vec![0usize; n_labels];
        loop {
            let mut prod: Option<T> = None;
            for (oi, ops) in operands.iter().enumerate() {
                let off: usize = (0..n_labels).map(|l| idx[l] * operand_strides[oi][l]).sum();
                let v = ops.get(off).copied().unwrap_or(zero);
                prod = Some(match prod {
                    None => v,
                    Some(p) => mul(p, v),
                });
            }
            let prod = prod.unwrap_or(zero);
            let out_off: usize = (0..n_labels).map(|l| idx[l] * out_stride[l]).sum();
            if let Some(d) = out.get_mut(out_off) {
                *d = add(*d, prod);
            }
            // Increment odometer.
            let mut carry = n_labels;
            for l in (0..n_labels).rev() {
                idx[l] += 1;
                if idx[l] < self.label_sizes[l].max(1) {
                    carry = 0;
                    break;
                }
                idx[l] = 0;
            }
            if carry != 0 || n_labels == 0 {
                break;
            }
        }
    }
}

/// Register the ONNX indexing / contraction reference kernels.
pub fn register_onnx_indexing_kernels() {
    register_cpu_kernel(Arc::new(GatherNdKernel));
    register_cpu_kernel(Arc::new(ScatterNdKernel));
    register_cpu_kernel(Arc::new(OneHotKernel));
    register_cpu_kernel(Arc::new(NonZeroKernel));
    register_cpu_kernel(Arc::new(CumProdKernel));
    register_cpu_kernel(Arc::new(EinsumKernel));
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::Shape;

    fn f32_ref<'a>(d: &'a [f32], s: &'a Shape) -> CpuTensorRef<'a> {
        CpuTensorRef::F32 { data: d, shape: s }
    }
    fn i64_ref<'a>(d: &'a [i64], s: &'a Shape) -> CpuTensorRef<'a> {
        CpuTensorRef::I64 { data: d, shape: s }
    }

    #[test]
    fn gather_nd_basic() {
        // data [2,2] = [[0,1],[2,3]], indices [2,2] = [[0,0],[1,1]] -> [0,3]
        let ds = Shape::new(&[2, 2], DType::F32);
        let data = [0.0f32, 1.0, 2.0, 3.0];
        let is = Shape::new(&[2, 2], DType::I64);
        let idx = [0i64, 0, 1, 1];
        let os = Shape::new(&[2], DType::F32);
        let mut out = [0.0f32; 2];
        GatherNdKernel
            .execute(
                &[f32_ref(&data, &ds), i64_ref(&idx, &is)],
                CpuTensorMut::F32 {
                    data: &mut out,
                    shape: &os,
                },
                &0i32.to_le_bytes(),
            )
            .unwrap();
        assert_eq!(out, [0.0, 3.0]);
    }

    #[test]
    fn gather_nd_slice() {
        // data [2,2] indices [2,1] -> rows: [[0,1],[2,3]] reversed -> [[2,3],[0,1]]
        let ds = Shape::new(&[2, 2], DType::F32);
        let data = [0.0f32, 1.0, 2.0, 3.0];
        let is = Shape::new(&[2, 1], DType::I64);
        let idx = [1i64, 0];
        let os = Shape::new(&[2, 2], DType::F32);
        let mut out = [0.0f32; 4];
        GatherNdKernel
            .execute(
                &[f32_ref(&data, &ds), i64_ref(&idx, &is)],
                CpuTensorMut::F32 {
                    data: &mut out,
                    shape: &os,
                },
                &0i32.to_le_bytes(),
            )
            .unwrap();
        assert_eq!(out, [2.0, 3.0, 0.0, 1.0]);
    }

    #[test]
    fn scatter_nd_basic() {
        // data [4,4] of ones, scatter rows 0 and 2 with zeros/twos.
        let ds = Shape::new(&[4, 4], DType::F32);
        let data = [1.0f32; 16];
        let is = Shape::new(&[2, 1], DType::I64);
        let idx = [0i64, 2];
        let us = Shape::new(&[2, 4], DType::F32);
        let updates = [0.0f32, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0];
        let mut out = [0.0f32; 16];
        ScatterNdKernel
            .execute(
                &[
                    f32_ref(&data, &ds),
                    i64_ref(&idx, &is),
                    f32_ref(&updates, &us),
                ],
                CpuTensorMut::F32 {
                    data: &mut out,
                    shape: &ds,
                },
                &[],
            )
            .unwrap();
        assert_eq!(&out[0..4], &[0.0, 0.0, 0.0, 0.0]); // row 0
        assert_eq!(&out[4..8], &[1.0, 1.0, 1.0, 1.0]); // row 1 untouched
        assert_eq!(&out[8..12], &[2.0, 2.0, 2.0, 2.0]); // row 2
    }

    #[test]
    fn one_hot_basic() {
        // indices [2] = [1,2], depth 3, axis -1 -> [2,3]
        let is = Shape::new(&[2], DType::I64);
        let idx = [1i64, 2];
        let depth = [3i64];
        let dsh = Shape::new(&[1], DType::I64);
        let values = [0.0f32, 1.0];
        let vsh = Shape::new(&[2], DType::F32);
        let os = Shape::new(&[2, 3], DType::F32);
        let mut out = [0.0f32; 6];
        OneHotKernel
            .execute(
                &[
                    i64_ref(&idx, &is),
                    i64_ref(&depth, &dsh),
                    f32_ref(&values, &vsh),
                ],
                CpuTensorMut::F32 {
                    data: &mut out,
                    shape: &os,
                },
                &(-1i32).to_le_bytes(),
            )
            .unwrap();
        assert_eq!(out, [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn non_zero_basic() {
        // x [2,3] = [[0,1,0],[2,0,3]] -> coords (row,col) for 1,2,3
        // nonzero (row-major): (0,1),(1,0),(1,2) -> [[0,1,1],[1,0,2]]
        let xs = Shape::new(&[2, 3], DType::F32);
        let x = [0.0f32, 1.0, 0.0, 2.0, 0.0, 3.0];
        let os = Shape::new(&[2, 3], DType::I64); // cols upper bound = numel/rank = 3
        let mut out = [0i64; 6];
        NonZeroKernel
            .execute(
                &[f32_ref(&x, &xs)],
                CpuTensorMut::I64 {
                    data: &mut out,
                    shape: &os,
                },
                &[],
            )
            .unwrap();
        // rows: dim0 = [0,1,1], dim1 = [1,0,2]
        assert_eq!(&out[0..3], &[0, 1, 1]);
        assert_eq!(&out[3..6], &[1, 0, 2]);
    }

    #[test]
    fn cumprod_basic() {
        // x [2,3], axis 1, inclusive forward
        let xs = Shape::new(&[2, 3], DType::F32);
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let os = xs.clone();
        let mut out = [0.0f32; 6];
        let attrs = {
            let mut a = 1i32.to_le_bytes().to_vec();
            a.push(0); // exclusive=false
            a.push(0); // reverse=false
            a
        };
        CumProdKernel
            .execute(
                &[f32_ref(&x, &xs)],
                CpuTensorMut::F32 {
                    data: &mut out,
                    shape: &os,
                },
                &attrs,
            )
            .unwrap();
        assert_eq!(out, [1.0, 2.0, 6.0, 4.0, 20.0, 120.0]);
    }

    #[test]
    fn cumprod_exclusive_reverse() {
        let xs = Shape::new(&[3], DType::F32);
        let x = [2.0f32, 3.0, 4.0];
        let os = xs.clone();
        let mut out = [0.0f32; 3];
        let attrs = {
            let mut a = 0i32.to_le_bytes().to_vec();
            a.push(1); // exclusive
            a.push(1); // reverse
            a
        };
        CumProdKernel
            .execute(
                &[f32_ref(&x, &xs)],
                CpuTensorMut::F32 {
                    data: &mut out,
                    shape: &os,
                },
                &attrs,
            )
            .unwrap();
        // reverse exclusive: out[2]=1, out[1]=4, out[0]=12
        assert_eq!(out, [12.0, 4.0, 1.0]);
    }

    #[test]
    fn cumprod_axis_as_input() {
        // codegen path: axis supplied as a second operand (here -1 -> last axis)
        let xs = Shape::new(&[2, 3], DType::F32);
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let axis_s = Shape::new(&[1], DType::I64);
        let axis = [-1i64];
        let os = xs.clone();
        let mut out = [0.0f32; 6];
        // attrs carry only exclusive/reverse (placeholder axis word).
        let attrs = vec![0u8, 0, 0, 0, 0, 0];
        CumProdKernel
            .execute(
                &[f32_ref(&x, &xs), i64_ref(&axis, &axis_s)],
                CpuTensorMut::F32 {
                    data: &mut out,
                    shape: &os,
                },
                &attrs,
            )
            .unwrap();
        assert_eq!(out, [1.0, 2.0, 6.0, 4.0, 20.0, 120.0]);
    }

    #[test]
    fn einsum_matmul() {
        // ij,jk->ik with A[2,3], B[3,2]
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let asx = Shape::new(&[2, 3], DType::F32);
        let b = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let bsx = Shape::new(&[3, 2], DType::F32);
        let os = Shape::new(&[2, 2], DType::F32);
        let mut out = [0.0f32; 4];
        EinsumKernel
            .execute(
                &[f32_ref(&a, &asx), f32_ref(&b, &bsx)],
                CpuTensorMut::F32 {
                    data: &mut out,
                    shape: &os,
                },
                b"ij,jk->ik",
            )
            .unwrap();
        // [[58,64],[139,154]]
        assert_eq!(out, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn einsum_trace_and_transpose() {
        // transpose ij->ji
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let asx = Shape::new(&[2, 3], DType::F32);
        let os = Shape::new(&[3, 2], DType::F32);
        let mut out = [0.0f32; 6];
        EinsumKernel
            .execute(
                &[f32_ref(&a, &asx)],
                CpuTensorMut::F32 {
                    data: &mut out,
                    shape: &os,
                },
                b"ij->ji",
            )
            .unwrap();
        assert_eq!(out, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // diagonal sum (trace) ii->
        let m = [1.0f32, 2.0, 3.0, 4.0];
        let ms = Shape::new(&[2, 2], DType::F32);
        let scalar = Shape::new(&[1], DType::F32);
        let mut tr = [0.0f32; 1];
        EinsumKernel
            .execute(
                &[f32_ref(&m, &ms)],
                CpuTensorMut::F32 {
                    data: &mut tr,
                    shape: &scalar,
                },
                b"ii->",
            )
            .unwrap();
        assert_eq!(tr, [5.0]);
    }

    #[test]
    fn einsum_implicit_output() {
        // implicit: "ij,jk" -> sorted free labels "ik"
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let asx = Shape::new(&[2, 2], DType::F32);
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let bsx = Shape::new(&[2, 2], DType::F32);
        let os = Shape::new(&[2, 2], DType::F32);
        let mut out = [0.0f32; 4];
        EinsumKernel
            .execute(
                &[f32_ref(&a, &asx), f32_ref(&b, &bsx)],
                CpuTensorMut::F32 {
                    data: &mut out,
                    shape: &os,
                },
                b"ij,jk",
            )
            .unwrap();
        assert_eq!(out, [5.0, 6.0, 7.0, 8.0]); // identity * B = B
    }
}
