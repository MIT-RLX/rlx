// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Data-bearing constructors — the NumPy-style entry points.
//!
//! Each returns a [`Tensor`] backed by its own freshly-rooted graph with the
//! host data embedded as an [`rlx_ir::Op::Constant`]. Combining tensors from
//! different constructors merges their graphs transparently (see
//! [`crate::handle::GraphHandle::adopt`]), so callers never touch a graph
//! scope:
//!
//! ```rust
//! use rlx_tensor::Tensor;
//!
//! let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
//! let b = Tensor::ones([3]);
//! let c = (&a + &b).relu();
//! assert_eq!(c.rank(), 1);
//! ```

use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};

use crate::Tensor;
use crate::handle::GraphHandle;

/// Concatenate tensors along an existing `axis` (NumPy `concatenate`). Inputs
/// from different graphs are merged transparently.
pub fn cat(tensors: &[&Tensor], axis: usize) -> Tensor {
    assert!(!tensors.is_empty(), "cat: need at least one tensor");
    let root = tensors[0];
    let ids: Vec<NodeId> = tensors.iter().map(|t| root.adopt(t)).collect();
    let id = root.handle.with_graph(|g| g.concat_(ids, axis));
    Tensor::new(root.handle.clone(), id)
}

/// Stack tensors along a new `axis` (NumPy `stack`): each gains a size-1 axis
/// at `axis`, then they are concatenated there.
pub fn stack(tensors: &[&Tensor], axis: usize) -> Tensor {
    assert!(!tensors.is_empty(), "stack: need at least one tensor");
    let expanded: Vec<Tensor> = tensors.iter().map(|t| t.unsqueeze(axis)).collect();
    let refs: Vec<&Tensor> = expanded.iter().collect();
    cat(&refs, axis)
}

impl Tensor {
    /// Constant tensor from row-major host data (F32). The product of `shape`
    /// must equal `data.len()`.
    pub fn from_vec(data: impl Into<Vec<f32>>, shape: impl AsRef<[usize]>) -> Self {
        let data = data.into();
        let dims = shape.as_ref();
        let shape = Shape::new(dims, DType::F32);
        let n = shape
            .num_elements()
            .expect("from_vec: shape must be static");
        assert_eq!(
            n,
            data.len(),
            "from_vec: shape {dims:?} expects {n} elements, got {}",
            data.len()
        );
        Self::constant_f32(&data, shape)
    }

    /// All-zeros constant of the given shape (F32).
    pub fn zeros(shape: impl AsRef<[usize]>) -> Self {
        Self::full(shape, 0.0)
    }

    /// All-ones constant of the given shape (F32).
    pub fn ones(shape: impl AsRef<[usize]>) -> Self {
        Self::full(shape, 1.0)
    }

    /// Constant of the given shape filled with `value` (F32).
    pub fn full(shape: impl AsRef<[usize]>, value: f32) -> Self {
        let dims = shape.as_ref();
        let shape = Shape::new(dims, DType::F32);
        let n = shape.num_elements().expect("full: shape must be static");
        Self::constant_f32(&vec![value; n], shape)
    }

    /// `n × n` identity matrix (F32).
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::constant_f32(&data, Shape::new(&[n, n], DType::F32))
    }

    /// 1-D constant `[0, 1, …, end)` (F32).
    pub fn arange(end: i64) -> Self {
        Self::arange_step(0, end, 1)
    }

    /// 1-D constant `[start, start+step, …)` up to (excluding) `end` (F32).
    pub fn arange_step(start: i64, end: i64, step: i64) -> Self {
        assert!(step != 0, "arange: step must be non-zero");
        let mut v = Vec::new();
        let mut x = start;
        if step > 0 {
            while x < end {
                v.push(x as f32);
                x += step;
            }
        } else {
            while x > end {
                v.push(x as f32);
                x += step;
            }
        }
        let len = v.len();
        Self::constant_f32(&v, Shape::new(&[len], DType::F32))
    }

    /// `F64` (double-precision) constant from host data.
    pub fn from_f64(data: impl AsRef<[f64]>, shape: impl AsRef<[usize]>) -> Self {
        let data = data.as_ref();
        let shape = Shape::new(shape.as_ref(), DType::F64);
        let n = shape
            .num_elements()
            .expect("from_f64: shape must be static");
        assert_eq!(
            n,
            data.len(),
            "from_f64: shape expects {n} elements, got {}",
            data.len()
        );
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::constant_raw(bytes, shape)
    }

    /// `I64` constant from host data (general N-D).
    pub fn from_i64(data: impl AsRef<[i64]>, shape: impl AsRef<[usize]>) -> Self {
        let data = data.as_ref();
        let shape = Shape::new(shape.as_ref(), DType::I64);
        let n = shape
            .num_elements()
            .expect("from_i64: shape must be static");
        assert_eq!(
            n,
            data.len(),
            "from_i64: shape expects {n} elements, got {}",
            data.len()
        );
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::constant_raw(bytes, shape)
    }

    /// 1-D `I64` index vector for [`gather`](Tensor::gather) /
    /// [`index_select`](Tensor::index_select).
    pub fn index_vec(data: impl AsRef<[i64]>) -> Self {
        let data = data.as_ref();
        Self::from_i64(data, [data.len()])
    }

    /// Build a fresh-graph F32 constant node from host data.
    fn constant_f32(data: &[f32], shape: Shape) -> Self {
        let bytes = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        Self::constant_raw(bytes, shape)
    }

    /// Uniform random in `[0, 1)` (F32), reproducible from `seed`.
    pub fn rand(shape: impl AsRef<[usize]>, seed: u64) -> Self {
        Self::rand_range(shape, seed, 0.0, 1.0)
    }

    /// Uniform random in `[lo, hi)` (F32), reproducible from `seed`.
    pub fn rand_range(shape: impl AsRef<[usize]>, seed: u64, lo: f32, hi: f32) -> Self {
        let shape = Shape::new(shape.as_ref(), DType::F32);
        let n = shape
            .num_elements()
            .expect("rand_range: shape must be static");
        let mut s = seed_state(seed);
        let data: Vec<f32> = (0..n).map(|_| lo + (hi - lo) * next_unit(&mut s)).collect();
        Self::constant_f32(&data, shape)
    }

    /// Standard-normal random (mean 0, std 1), reproducible from `seed`.
    pub fn randn(shape: impl AsRef<[usize]>, seed: u64) -> Self {
        Self::randn_std(shape, seed, 0.0, 1.0)
    }

    /// Normal random with the given `mean`/`std`, reproducible from `seed`.
    /// Useful for weight init, e.g. `randn_std(shape, seed, 0.0, (2.0/fan_in).sqrt())`.
    pub fn randn_std(shape: impl AsRef<[usize]>, seed: u64, mean: f32, std: f32) -> Self {
        let shape = Shape::new(shape.as_ref(), DType::F32);
        let n = shape
            .num_elements()
            .expect("randn_std: shape must be static");
        let mut s = seed_state(seed);
        let data: Vec<f32> = (0..n).map(|_| mean + std * next_normal(&mut s)).collect();
        Self::constant_f32(&data, shape)
    }

    /// Build a fresh-graph constant node from raw little-endian bytes.
    fn constant_raw(bytes: Vec<u8>, shape: Shape) -> Self {
        let mut graph = Graph::new("rlx");
        let id = graph.add_node(Op::Constant { data: bytes }, vec![], shape);
        Tensor::new(GraphHandle::new(graph), id)
    }
}

/// SplitMix64 — a tiny, dependency-free PRNG so random constructors are
/// reproducible (host-generated, baked as constants) without a `rand` dep.
fn seed_state(seed: u64) -> u64 {
    // Avoid the all-zero fixed point.
    seed ^ 0x9E37_79B9_7F4A_7C15
}

fn next_u64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform `f32` in `[0, 1)` from 24 random mantissa bits.
fn next_unit(s: &mut u64) -> f32 {
    (next_u64(s) >> 40) as f32 / (1u64 << 24) as f32
}

/// Standard normal via Box–Muller.
fn next_normal(s: &mut u64) -> f32 {
    let u1 = next_unit(s).max(f32::MIN_POSITIVE);
    let u2 = next_unit(s);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}
