// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Numerical validation against an authority that is not the thing under
//! test.**
//!
//! CAKE Table 1 lists numerical validation as an *execution gate* whose job is
//! to "compare compiled outputs with an authoritative external reference", and
//! §3.1 adds that "final acceptance requires end-to-end evaluation in the
//! corresponding target framework".
//!
//! rlx's numerical tests almost all compare a backend against **rlx's own CPU
//! backend**. That catches backend-specific defects and misses everything the
//! CPU path gets wrong too — which is not hypothetical here: `rms_norm_backward`
//! carried an extra `1/r` on the input gradient in *all seven* implementations
//! at once, and no backend-vs-backend comparison could have seen it.
//!
//! # The three authorities, and why the distinction is recorded
//!
//! [`Authority::SelfBackend`] is not a validation authority; it is a
//! consistency check. Recording that difference is the whole point of this
//! module: a suite that reports "1200 numerical tests passing" without saying
//! how many were self-comparisons is reporting a number that cannot be acted
//! on.
//!
//! [`Authority::Independent`] is the workhorse — a closed-form evaluation in
//! `f64`, written here, sharing no code with any rlx kernel. It is the forward
//! analogue of what `fd_backward_gate` does for gradients: the oracle must not
//! be another implementation of the same thing.
//!
//! [`Authority::External`] is a third-party runtime. It is the strongest and
//! the least available: ONNX Runtime can only answer for graphs that came from
//! a `.onnx` file, because rlx has no `Graph` → ONNX exporter. Cases that
//! cannot reach an external authority say so rather than quietly downgrading.

use std::collections::HashMap;

use rlx_ir::{Graph, NodeId, Op};

/// Where a case's expected values came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Authority {
    /// A third-party runtime (ONNX Runtime). Strongest.
    External(&'static str),
    /// Closed-form `f64` evaluation written independently of every rlx kernel.
    Independent,
    /// rlx's own CPU backend. A consistency check, **not** validation.
    SelfBackend,
}

impl Authority {
    /// Whether this authority is independent of the implementation under test.
    /// `SelfBackend` is the only one that is not.
    pub const fn is_independent(self) -> bool {
        !matches!(self, Self::SelfBackend)
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::External(n) => n,
            Self::Independent => "independent-f64",
            Self::SelfBackend => "self-cpu (NOT validation)",
        }
    }
}

/// Backends whose *own runtime* is a legitimate external authority.
///
/// The module docs above say `External` is "the least available" because ONNX
/// Runtime can only answer for graphs that came from a `.onnx` file, and rlx
/// has no `Graph` -> ONNX exporter. That is true of ONNX Runtime and false as a
/// general claim, and the difference is worth a function.
///
/// A **delegating** backend does not run rlx's kernels — it hands rlx's lowered
/// IR to a vendor compiler and runtime that share no code with this project.
/// Comparing rlx-on-CoreML against rlx-on-CPU therefore compares two
/// independent implementations of the same mathematics, which is exactly what
/// [`Authority::External`] means. It is not [`Authority::SelfBackend`]: the
/// only rlx code in the loop is the lowering, and a lowering bug is precisely
/// what the comparison is good at finding.
///
/// The catch, stated because it bounds what a green result proves: rlx authors
/// the *lowering*, so a comparison of this kind validates the lowering and the
/// vendor's execution of it. It cannot catch a shared misunderstanding of the
/// op's semantics — for that the closed-form [`Authority::Independent`] oracle
/// is still the stronger instrument.
///
/// Returns `None` for backends that run rlx's own kernels; there the vendor
/// runtime is not in the picture and calling the result "external" would be a
/// category error.
pub const fn external_authority_for(backend: &str) -> Option<Authority> {
    // `match` on &str is not const-friendly; compare bytes.
    let b = backend.as_bytes();
    match b {
        b"mlx" => Some(Authority::External("MLX (Apple)")),
        b"coreml" => Some(Authority::External("CoreML / ANE (Apple)")),
        b"tpu" => Some(Authority::External("XLA")),
        b"qnn" => Some(Authority::External("Qualcomm QNN")),
        b"xdna" => Some(Authority::External("AMD XDNA / MLIR-AIE")),
        // Everything else runs rlx's kernels: cpu, cuda, rocm, metal, wgpu,
        // vulkan, oneapi, webgl, cerebras, cortexm, fpga.
        _ => None,
    }
}

/// Outcome of validating one case.
#[derive(Debug, Clone)]
pub struct OracleResult {
    pub authority: Authority,
    /// `None` when no authority could evaluate the graph.
    pub max_rel_err: Option<f64>,
    pub detail: String,
}

/// Tolerance for a single f32 reduction checked against an f64 reference.
///
/// Not bit-exactness: the reference accumulates in f64 and in a different
/// order, so agreement to the last bit would be the surprise.
pub const REL_TOL: f64 = 2e-4;

/// Upper bound on the derived tolerance. Past this a "pass" stops meaning
/// anything, so a graph needing more is reported rather than accommodated.
pub const MAX_DERIVED_TOL: f64 = 5e-3;

/// Tolerance for `graph`, scaled by how many reductions its result passes
/// through.
///
/// A single matmul agrees with the f64 reference to ~1e-5. A two-layer
/// transformer block does not, and it is not wrong: fourteen chained
/// reductions with residual adds compound f32 rounding, and the adds amplify
/// relative error wherever the sum nearly cancels. Measured here: 3.0e-4 at
/// one layer, 1.0e-3 at two.
///
/// So the bound is derived from the graph — one `REL_TOL` per reduction on
/// the path — rather than being a single global constant loosened until the
/// deepest case passed. That constant would have silently weakened the check
/// for every shallow case too, which is where a real defect is easiest to see.
pub fn tolerance_for(graph: &Graph) -> f64 {
    let reductions = graph
        .nodes()
        .iter()
        .filter(|n| {
            matches!(
                n.op,
                Op::MatMul
                    | Op::Reduce { .. }
                    | Op::RmsNorm { .. }
                    | Op::LayerNorm { .. }
                    | Op::Attention { .. }
            )
        })
        .count();
    (REL_TOL * reductions.max(1) as f64).min(MAX_DERIVED_TOL)
}

/// A dense f64 tensor for the reference evaluation.
#[derive(Clone)]
struct Ref {
    dims: Vec<usize>,
    data: Vec<f64>,
}

impl Ref {
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// Decode a GGUF **Q8_0** blob to f64, written from the format description
/// rather than from `rlx-gguf`.
///
/// The independence is the entire point. rlx-cpu's dequant path calls into
/// `rlx-gguf`, so scoring it against `rlx-gguf` would be a self-comparison
/// wearing an oracle's clothes — the `SelfBackend` case this module exists to
/// keep separate. This reads the layout straight from the spec:
///
/// > `block_q8_0` = 34 bytes = one little-endian **f16** scale `d`, then
/// > **32 int8** quants. Element `i` of the block is `d * q[i]`.
///
/// Q8_0 is the scheme to start with because *any* byte string is a valid block
/// (an f16 and 32 signed bytes — no sub-scale can be degenerate), so a fixture
/// needs no encoder. The K-quants pack scales into shared nibble planes, where
/// arbitrary bytes are not necessarily meaningful; they stay unvalidated and
/// say so rather than being scored against a fixture nobody can defend.
fn gguf_q8_0_decode(bytes: &[u8], n_elems: usize) -> Option<Vec<f64>> {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 34;
    if !n_elems.is_multiple_of(QK) {
        return None;
    }
    let blocks = n_elems / QK;
    if bytes.len() < blocks * BLOCK_BYTES {
        return None;
    }
    let mut out = Vec::with_capacity(n_elems);
    for b in 0..blocks {
        let base = b * BLOCK_BYTES;
        let d = f16_to_f64(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
        for i in 0..QK {
            out.push(d * (bytes[base + 2 + i] as i8) as f64);
        }
    }
    Some(out)
}

/// IEEE-754 binary16 → f64, by hand.
///
/// `half::f16` would do, but this file's contract is that it shares no code
/// with the path under test, and the dequant kernels use `half`.
fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits >> 15 == 1 { -1.0f64 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as f64;
    match exp {
        // Subnormal (and zero): no implicit leading 1, fixed exponent -14.
        0 => sign * frac * 2f64.powi(-24),
        // Inf / NaN.
        0x1f => {
            if frac == 0.0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => sign * (1.0 + frac / 1024.0) * 2f64.powi(exp - 15),
    }
}

/// Row-major strides for `dims`.
fn strides_of(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1];
    }
    s
}

/// Write the multi-index of row-major element `flat` of `dims` into `out`.
///
/// Takes a scratch buffer rather than allocating: the reindexing arms call this
/// once per output element, and a `Vec` per element made the transpose case
/// visibly slower than the kernel it is checking.
fn unravel_into(flat: usize, dims: &[usize], out: &mut [usize]) {
    let mut rem = flat;
    for i in (0..dims.len()).rev() {
        let d = dims[i].max(1);
        out[i] = rem % d;
        rem /= d;
    }
}

/// Broadcast `a` and `b` to a common shape, numpy-style, and apply `f`.
///
/// Written out rather than reusing any rlx broadcasting helper: sharing that
/// code would make this evaluator agree with rlx by construction, which is
/// exactly the failure this module exists to avoid. (rlx has shipped a
/// broadcast bug — trailing size-1 axes in `Thunk::Compare` — that a shared
/// helper would have reproduced identically on both sides.)
fn broadcast_zip(a: &Ref, b: &Ref, f: impl Fn(f64, f64) -> f64) -> Option<Ref> {
    let rank = a.dims.len().max(b.dims.len());
    let pad = |d: &[usize]| -> Vec<usize> {
        let mut v = vec![1usize; rank - d.len()];
        v.extend_from_slice(d);
        v
    };
    let (da, db) = (pad(&a.dims), pad(&b.dims));
    let mut out_dims = Vec::with_capacity(rank);
    for i in 0..rank {
        out_dims.push(match (da[i], db[i]) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        });
    }
    let total: usize = out_dims.iter().product();
    let strides = |d: &[usize]| -> Vec<usize> {
        let mut s = vec![0usize; rank];
        let mut acc = 1usize;
        for i in (0..rank).rev() {
            s[i] = if d[i] == 1 { 0 } else { acc };
            acc *= d[i];
        }
        s
    };
    let (sa, sb) = (strides(&da), strides(&db));
    let mut data = Vec::with_capacity(total);
    let mut idx = vec![0usize; rank];
    for _ in 0..total {
        let (mut ia, mut ib) = (0usize, 0usize);
        for i in 0..rank {
            ia += idx[i] * sa[i];
            ib += idx[i] * sb[i];
        }
        data.push(f(a.data[ia], b.data[ib]));
        for i in (0..rank).rev() {
            idx[i] += 1;
            if idx[i] < out_dims[i] {
                break;
            }
            idx[i] = 0;
        }
    }
    Some(Ref {
        dims: out_dims,
        data,
    })
}

/// Naive f64 matmul over the trailing two axes. Triple loop on purpose: this
/// must not share a blocking or accumulation strategy with anything rlx does.
fn matmul_ref(a: &Ref, b: &Ref) -> Option<Ref> {
    if a.dims.len() < 2 || b.dims.len() != 2 {
        return None;
    }
    let k = *a.dims.last()?;
    if b.dims[0] != k {
        return None;
    }
    let n = b.dims[1];
    let m: usize = a.dims[..a.dims.len() - 1].iter().product();
    let mut data = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += a.data[i * k + p] * b.data[p * n + j];
            }
            data[i * n + j] = acc;
        }
    }
    let mut dims = a.dims[..a.dims.len() - 1].to_vec();
    dims.push(n);
    Some(Ref { dims, data })
}

/// Evaluate `graph` in f64 from the given inputs and parameters.
///
/// Returns `None` for any op this evaluator does not implement — silently
/// producing a wrong reference would be far worse than declining, so coverage
/// is partial and explicit.
fn evaluate(
    graph: &Graph,
    inputs: &HashMap<String, Ref>,
    params: &HashMap<String, Ref>,
    packed: &HashMap<String, Vec<u8>>,
) -> Option<Ref> {
    let mut vals: HashMap<NodeId, Ref> = HashMap::new();
    for node in graph.nodes() {
        let dims: Vec<usize> = node
            .shape
            .dims()
            .iter()
            .map(|d| match d {
                rlx_ir::Dim::Static(n) => Some(*n),
                rlx_ir::Dim::Dynamic(_) => None,
            })
            .collect::<Option<Vec<_>>>()?;
        let got: Ref = match &node.op {
            Op::Input { name } => inputs.get(name)?.clone(),
            // A packed quantized weight has no f64 dense form until its
            // consumer decodes it, so it carries a placeholder here. Without
            // this the walk bailed at the `Param` node and every quantized case
            // reported "no independent oracle covers this graph's ops" — the
            // oracle looking like it lacked a rule it in fact had.
            Op::Param { name } if packed.contains_key(name) => Ref {
                dims: dims.clone(),
                data: Vec::new(),
            },
            Op::Param { name } => params.get(name)?.clone(),
            Op::Binary(op) => {
                let a = vals.get(&node.inputs[0])?;
                let b = vals.get(&node.inputs[1])?;
                use rlx_ir::op::BinaryOp as B;
                let f: fn(f64, f64) -> f64 = match op {
                    B::Add => |x, y| x + y,
                    B::Sub => |x, y| x - y,
                    B::Mul => |x, y| x * y,
                    B::Div => |x, y| x / y,
                    _ => return None,
                };
                broadcast_zip(a, b, f)?
            }
            Op::MatMul => {
                let a = vals.get(&node.inputs[0])?;
                let b = vals.get(&node.inputs[1])?;
                matmul_ref(a, b)?
            }
            Op::RmsNorm { axis, eps } => {
                if *axis != -1 {
                    return None;
                }
                let x = vals.get(&node.inputs[0])?;
                let gamma = vals.get(&node.inputs[1])?;
                let beta = node.inputs.get(2).and_then(|i| vals.get(i));
                let r = *x.dims.last()?;
                let rows = x.len() / r;
                let mut data = vec![0.0f64; x.len()];
                for row in 0..rows {
                    let s = &x.data[row * r..(row + 1) * r];
                    let ms = s.iter().map(|v| v * v).sum::<f64>() / r as f64;
                    let inv = 1.0 / (ms + *eps as f64).sqrt();
                    for c in 0..r {
                        let mut v = s[c] * inv * gamma.data[c];
                        if let Some(b) = beta {
                            v += b.data[c];
                        }
                        data[row * r + c] = v;
                    }
                }
                Ref {
                    dims: x.dims.clone(),
                    data,
                }
            }
            Op::LayerNorm { axis, eps } => {
                if *axis != -1 {
                    return None;
                }
                let x = vals.get(&node.inputs[0])?;
                let gamma = vals.get(&node.inputs[1])?;
                let beta = node.inputs.get(2).and_then(|i| vals.get(i));
                let r = *x.dims.last()?;
                let rows = x.len() / r;
                let mut data = vec![0.0f64; x.len()];
                for row in 0..rows {
                    let s = &x.data[row * r..(row + 1) * r];
                    let mean = s.iter().sum::<f64>() / r as f64;
                    let var = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / r as f64;
                    let inv = 1.0 / (var + *eps as f64).sqrt();
                    for c in 0..r {
                        let mut v = (s[c] - mean) * inv * gamma.data[c];
                        if let Some(b) = beta {
                            v += b.data[c];
                        }
                        data[row * r + c] = v;
                    }
                }
                Ref {
                    dims: x.dims.clone(),
                    data,
                }
            }
            Op::Reduce { op, axes, keep_dim } => {
                if axes.len() != 1 || *keep_dim {
                    return None;
                }
                let x = vals.get(&node.inputs[0])?;
                if axes[0] + 1 != x.dims.len() {
                    return None; // trailing-axis reductions only
                }
                let r = *x.dims.last()?;
                let rows = x.len() / r;
                use rlx_ir::op::ReduceOp as R;
                let mut data = Vec::with_capacity(rows);
                for row in 0..rows {
                    let s = &x.data[row * r..(row + 1) * r];
                    data.push(match op {
                        R::Sum => s.iter().sum::<f64>(),
                        R::Mean => s.iter().sum::<f64>() / r as f64,
                        R::Max => s.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                        R::Min => s.iter().copied().fold(f64::INFINITY, f64::min),
                        _ => return None,
                    });
                }
                Ref {
                    dims: x.dims[..x.dims.len() - 1].to_vec(),
                    data,
                }
            }
            Op::Attention {
                num_heads,
                head_dim,
                v_head_dim,
                mask_kind,
                score_scale,
                attn_logit_softcap,
            } => {
                // Scaled dot-product attention in f64, written out rather than
                // shared with any rlx kernel. Only the plain configuration is
                // covered; anything with a softcap, a non-default v-head-dim or
                // a mask this evaluator does not model returns None so the case
                // is reported unvalidated instead of scored against a reference
                // that quietly computes something else.
                use rlx_ir::op::MaskKind as M;
                if v_head_dim.is_some() || attn_logit_softcap.is_some() {
                    return None;
                }
                let causal = match mask_kind {
                    M::None => false,
                    M::Causal => true,
                    _ => return None,
                };
                let q = vals.get(&node.inputs[0])?;
                let k = vals.get(&node.inputs[1])?;
                let v = vals.get(&node.inputs[2])?;
                if q.dims.len() != 4 {
                    return None;
                }
                let (b, h, sq, d) = (q.dims[0], q.dims[1], q.dims[2], q.dims[3]);
                if h != *num_heads || d != *head_dim || k.dims != q.dims || v.dims != q.dims {
                    return None;
                }
                let scale = score_scale.map_or(1.0 / (d as f64).sqrt(), |s| s as f64);
                let mut out = vec![0.0f64; q.len()];
                for bi in 0..b {
                    for hi in 0..h {
                        let base = ((bi * h) + hi) * sq * d;
                        for i in 0..sq {
                            let last = if causal { i } else { sq - 1 };
                            let mut scores = vec![f64::NEG_INFINITY; sq];
                            for (j, sc) in scores.iter_mut().enumerate().take(last + 1) {
                                let mut acc = 0.0f64;
                                for t in 0..d {
                                    acc += q.data[base + i * d + t] * k.data[base + j * d + t];
                                }
                                *sc = acc * scale;
                            }
                            let m = scores[..=last]
                                .iter()
                                .copied()
                                .fold(f64::NEG_INFINITY, f64::max);
                            let mut denom = 0.0f64;
                            for sc in scores[..=last].iter_mut() {
                                *sc = (*sc - m).exp();
                                denom += *sc;
                            }
                            for t in 0..d {
                                let mut acc = 0.0f64;
                                for (j, sc) in scores[..=last].iter().enumerate() {
                                    acc += sc * v.data[base + j * d + t];
                                }
                                out[base + i * d + t] = acc / denom;
                            }
                        }
                    }
                }
                Ref {
                    dims: q.dims.clone(),
                    data: out,
                }
            }
            Op::Activation(act) => {
                let x = vals.get(&node.inputs[0])?;
                use rlx_ir::op::Activation as A;
                let f: fn(f64) -> f64 = match act {
                    A::Sigmoid => |v| 1.0 / (1.0 + (-v).exp()),
                    A::Relu => |v| v.max(0.0),
                    A::Tanh => |v| v.tanh(),
                    A::Exp => |v| v.exp(),
                    A::Sqrt => |v| v.sqrt(),
                    A::Neg => |v| -v,
                    _ => return None,
                };
                Ref {
                    dims: x.dims.clone(),
                    data: x.data.iter().copied().map(f).collect(),
                }
            }
            Op::Reshape { .. } => {
                let x = vals.get(&node.inputs[0])?;
                Ref {
                    dims: dims.clone(),
                    data: x.data.clone(),
                }
            }

            // `y[m,n] = x[m,k] · Wᵀ`, where the packed blob decodes to `[n, k]`.
            //
            // The `[n, k]` orientation is not incidental: rlx-rocm shipped this
            // op as `sgemm(N,N)` against a `[n,k]` weight and produced a
            // transposed answer on both paths, hidden because the only test
            // used `n = 1`. An oracle that took the layout on faith would have
            // agreed with the bug.
            Op::DequantMatMul { scheme } => {
                let x = vals.get(&node.inputs[0])?;
                let name = match &graph.node(node.inputs[1]).op {
                    Op::Param { name } | Op::Input { name } => name,
                    _ => return None,
                };
                let raw = packed.get(name)?;
                let (m, k) = (*x.dims.first()?, *x.dims.get(1)?);
                let n = *dims.get(1)?;
                // Only Q8_0 has a decoder here; everything else reports as
                // uncovered rather than guessing.
                if *scheme != rlx_ir::quant::QuantScheme::GgufQ8_0 {
                    return None;
                }
                let w = gguf_q8_0_decode(raw, n * k)?;
                let mut data = vec![0.0f64; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0f64;
                        for kk in 0..k {
                            acc += x.data[i * k + kk] * w[j * k + kk];
                        }
                        data[i * n + j] = acc;
                    }
                }
                Ref {
                    dims: dims.clone(),
                    data,
                }
            }

            // ── Pure reindexing ─────────────────────────────────────────────
            //
            // These move elements without computing on them, so an f64
            // reference is exact rather than merely more precise. They are here
            // because without them the `structural` family scored 0/3 on EVERY
            // device — the ceiling was this evaluator, not the backends, and an
            // unvalidated case is indistinguishable from an absent one in the
            // coverage number.
            //
            // Reindexing is also where the real defects in this tree live:
            // broadcast axes given a non-zero stride, a narrow reading past an
            // extent, a transpose whose permutation is inverted. Every one of
            // those produces a *plausible* wrong answer, which is exactly what
            // an independent authority is for.
            Op::Transpose { perm } => {
                let x = vals.get(&node.inputs[0])?;
                if perm.len() != x.dims.len() || perm.len() != dims.len() {
                    return None;
                }
                let in_str = strides_of(&x.dims);
                let mut data = vec![0.0f64; x.len()];
                let mut idx = vec![0usize; dims.len()];
                for (flat, slot) in data.iter_mut().enumerate() {
                    unravel_into(flat, &dims, &mut idx);
                    // out_dims[i] == in_dims[perm[i]], so out index i selects
                    // input axis perm[i].
                    let src: usize = idx
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| v * in_str[perm[i]])
                        .sum();
                    *slot = *x.data.get(src)?;
                }
                Ref {
                    dims: dims.clone(),
                    data,
                }
            }
            Op::Expand { .. } => {
                let x = vals.get(&node.inputs[0])?;
                // Right-aligned against the output, as broadcasting is
                // everywhere else in the IR.
                let lead = dims.len().checked_sub(x.dims.len())?;
                let in_str = strides_of(&x.dims);
                let mut data = vec![0.0f64; dims.iter().product()];
                let mut idx = vec![0usize; dims.len()];
                for (flat, slot) in data.iter_mut().enumerate() {
                    unravel_into(flat, &dims, &mut idx);
                    let mut src = 0usize;
                    for (ax, &d) in x.dims.iter().enumerate() {
                        // An axis of extent 1 is broadcast: stride 0, not
                        // `product(trailing dims)`. Getting this wrong is the
                        // attention-mask defect, so the reference states it.
                        if d != 1 {
                            src += idx[lead + ax] * in_str[ax];
                        }
                    }
                    *slot = *x.data.get(src)?;
                }
                Ref {
                    dims: dims.clone(),
                    data,
                }
            }
            Op::Narrow { axis, start, len } => {
                let x = vals.get(&node.inputs[0])?;
                let ax = *axis;
                if ax >= x.dims.len() || start + len > x.dims[ax] {
                    return None;
                }
                let in_str = strides_of(&x.dims);
                let mut data = vec![0.0f64; dims.iter().product()];
                let mut idx = vec![0usize; dims.len()];
                for (flat, slot) in data.iter_mut().enumerate() {
                    unravel_into(flat, &dims, &mut idx);
                    let mut src = 0usize;
                    for (i, &v) in idx.iter().enumerate() {
                        src += (if i == ax { v + start } else { v }) * in_str[i];
                    }
                    *slot = *x.data.get(src)?;
                }
                Ref {
                    dims: dims.clone(),
                    data,
                }
            }
            Op::Concat { axis } => {
                let ax = *axis;
                let outer: usize = dims.iter().take(ax).product();
                let inner: usize = dims.iter().skip(ax + 1).product();
                let mut data = vec![0.0f64; dims.iter().product()];
                let mut written = 0usize; // running offset along `ax`
                for &src_id in &node.inputs {
                    let s = vals.get(&src_id)?;
                    let s_ax = *s.dims.get(ax)?;
                    for o in 0..outer {
                        for a in 0..s_ax {
                            let dst = (o * dims[ax] + written + a) * inner;
                            let src = (o * s_ax + a) * inner;
                            data.get_mut(dst..dst + inner)?
                                .copy_from_slice(s.data.get(src..src + inner)?);
                        }
                    }
                    written += s_ax;
                }
                if written != dims[ax] {
                    return None;
                }
                Ref {
                    dims: dims.clone(),
                    data,
                }
            }
            _ => return None,
        };
        vals.insert(node.id, got);
    }
    vals.get(graph.outputs.first()?).cloned()
}

/// Validate `actual` (produced by a backend) for `graph` against the strongest
/// authority available.
pub fn validate(
    graph: &Graph,
    inputs: &[(&str, Vec<f32>)],
    params: &[(&str, Vec<f32>)],
    actual: &[f32],
) -> OracleResult {
    validate_with_packed(graph, inputs, params, &[], actual)
}

/// [`validate`], plus params fed as raw bytes.
///
/// A quantized weight is not expressible as `&[f32]` — `CompiledGraph::set_param`
/// would write four bytes per element into a `U8` slot sized for one, which is a
/// silent arena overflow, not a type error. Backends take those through
/// `set_param_typed`, so the oracle needs the same channel or the quantized
/// family can never be scored against anything.
pub fn validate_with_packed(
    graph: &Graph,
    inputs: &[(&str, Vec<f32>)],
    params: &[(&str, Vec<f32>)],
    packed_params: &[(&str, Vec<u8>)],
    actual: &[f32],
) -> OracleResult {
    let packed: HashMap<String, Vec<u8>> = packed_params
        .iter()
        .map(|(n, v)| ((*n).to_string(), v.clone()))
        .collect();
    let packed = &packed;
    let to_ref = |name: &str, v: &Vec<f32>| -> Option<(String, Ref)> {
        let node = graph.nodes().iter().find(|n| match &n.op {
            Op::Input { name: nm } | Op::Param { name: nm } => nm == name,
            _ => false,
        })?;
        let dims: Vec<usize> = node
            .shape
            .dims()
            .iter()
            .map(|d| match d {
                rlx_ir::Dim::Static(n) => Some(*n),
                rlx_ir::Dim::Dynamic(_) => None,
            })
            .collect::<Option<Vec<usize>>>()?;
        Some((
            name.to_string(),
            Ref {
                dims,
                data: v.iter().map(|x| *x as f64).collect(),
            },
        ))
    };
    let Some(in_map) = inputs
        .iter()
        .map(|(n, v)| to_ref(n, v))
        .collect::<Option<HashMap<_, _>>>()
    else {
        return OracleResult {
            authority: Authority::SelfBackend,
            max_rel_err: None,
            detail: "inputs could not be resolved to static shapes".into(),
        };
    };
    let Some(param_map) = params
        .iter()
        .map(|(n, v)| to_ref(n, v))
        .collect::<Option<HashMap<_, _>>>()
    else {
        return OracleResult {
            authority: Authority::SelfBackend,
            max_rel_err: None,
            detail: "params could not be resolved to static shapes".into(),
        };
    };

    match evaluate(graph, &in_map, &param_map, packed) {
        Some(expect) if expect.len() == actual.len() => {
            // Non-finite values are checked BEFORE the fold, not through it.
            // `f64::max(0.0, NaN)` returns `0.0`, so folding a NaN difference
            // reports *perfect agreement* — which is what this oracle did on
            // its first run: a graph whose activations overflowed to `inf` on
            // both sides scored max_rel = 0.0 and passed. A comparison that
            // cannot fail on garbage is not a comparison.
            let bad_ref = expect.data.iter().position(|v| !v.is_finite());
            let bad_act = actual.iter().position(|v| !v.is_finite());
            if bad_ref.is_some() || bad_act.is_some() {
                return OracleResult {
                    authority: Authority::Independent,
                    max_rel_err: None,
                    detail: format!(
                        "non-finite value(s): reference at {bad_ref:?}, actual at {bad_act:?} \
                         — comparison withheld"
                    ),
                };
            }
            let max_rel = expect
                .data
                .iter()
                .zip(actual)
                .map(|(e, a)| (e - *a as f64).abs() / (1.0 + e.abs()))
                .fold(0.0f64, f64::max);
            OracleResult {
                authority: Authority::Independent,
                max_rel_err: Some(max_rel),
                detail: format!("{} element(s) compared", expect.len()),
            }
        }
        Some(expect) => OracleResult {
            authority: Authority::Independent,
            max_rel_err: None,
            detail: format!(
                "length mismatch: reference {} vs actual {}",
                expect.len(),
                actual.len()
            ),
        },
        None => OracleResult {
            authority: Authority::SelfBackend,
            max_rel_err: None,
            // The honest outcome: no independent authority covers this graph,
            // so nothing here is validated. Saying so is the contribution.
            detail: "no independent oracle covers this graph's ops".into(),
        },
    }
}
