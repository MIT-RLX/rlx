#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

/// Fused sampling step: logits → top-k filter → top-p truncation
/// → softmax → multinomial sample. Operates on one row of length
/// `vocab` and returns the sampled index. Plan #42.
///
/// Internal scratch is on the stack via SmallVec-style fallback —
/// for `vocab > 8192` we heap-allocate a working buffer; below
/// that we keep things in a fixed array. (TODO: thread the
/// scratch through ThunkSchedule like sdpa_scores does.)
pub(crate) fn sample_row(
    logits: &[f32],
    top_k: usize,
    top_p: f32,
    temperature: f32,
    rng: &mut rlx_ir::Philox4x32,
) -> usize {
    let v = logits.len();
    if v == 0 {
        return 0;
    }
    let temp = temperature.max(1e-6);
    // Copy + temperature-scale into a working buffer.
    let mut scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();

    // Top-k: zero out everything but the k largest by setting to -inf.
    if top_k > 0 && top_k < v {
        // Partial selection: find k-th largest then mask below.
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        // Sort descending; partial would be O(n log k), full sort is fine
        // for typical vocab sizes (32k-128k) — single-row work.
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let cutoff = indexed[top_k - 1].1;
        for x in scaled.iter_mut() {
            if *x < cutoff {
                *x = f32::NEG_INFINITY;
            }
        }
    }

    // Stable softmax.
    let mut max_l = f32::NEG_INFINITY;
    for &x in &scaled {
        if x > max_l {
            max_l = x;
        }
    }
    let mut sum = 0.0f32;
    for x in scaled.iter_mut() {
        *x = (*x - max_l).exp();
        sum += *x;
    }
    let inv = 1.0 / sum.max(f32::MIN_POSITIVE);
    for x in scaled.iter_mut() {
        *x *= inv;
    }

    // Top-p: keep the smallest set of tokens whose cumulative
    // probability exceeds top_p (after sorting descending).
    if top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cum = 0.0f32;
        let mut keep = vec![false; v];
        for (idx, p) in indexed.iter() {
            keep[*idx] = true;
            cum += *p;
            if cum >= top_p {
                break;
            }
        }
        let mut new_sum = 0.0f32;
        for (i, x) in scaled.iter_mut().enumerate() {
            if !keep[i] {
                *x = 0.0;
            }
            new_sum += *x;
        }
        let inv = 1.0 / new_sum.max(f32::MIN_POSITIVE);
        for x in scaled.iter_mut() {
            *x *= inv;
        }
    }

    // Multinomial sample via inverse-CDF.
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (i, &p) in scaled.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i;
        }
    }
    v - 1 // floating-point edge case fallback
}

#[allow(unused_variables)]
pub(crate) fn compile_sample(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Sample {
        top_k,
        top_p,
        temperature,
        seed,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        // Logits are [batch, vocab] (or [vocab] → batch=1).
        let (batch, vocab) = if in_shape.rank() >= 2 {
            (
                in_shape.dim(0).unwrap_static(),
                in_shape.dim(in_shape.rank() - 1).unwrap_static(),
            )
        } else {
            (1, in_shape.num_elements().unwrap_or(0))
        };
        Thunk::Sample {
            logits: node_offset(arena, node.inputs[0]),
            dst: node_offset(arena, node.id),
            batch: batch as u32,
            vocab: vocab as u32,
            top_k: *top_k as u32,
            top_p: *top_p,
            temperature: *temperature,
            seed: *seed,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_rng_normal(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::RngNormal {
        mean,
        scale,
        key,
        op_seed,
    } = &node.op
    else {
        unreachable!()
    };
    Thunk::RngNormal {
        dst: node_offset(arena, node.id),
        len: node.shape.num_elements().unwrap_or(0) as u32,
        mean: *mean,
        scale: *scale,
        key: *key,
        op_seed: *op_seed,
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_rng_uniform(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::RngUniform {
        low,
        high,
        key,
        op_seed,
    } = &node.op
    else {
        unreachable!()
    };
    Thunk::RngUniform {
        dst: node_offset(arena, node.id),
        len: node.shape.num_elements().unwrap_or(0) as u32,
        low: *low,
        high: *high,
        key: *key,
        op_seed: *op_seed,
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_cumsum(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Cumsum { axis, exclusive } = &node.op else {
        unreachable!()
    };
    {
        // For now CPU only supports last-axis cumsum (the
        // common case for sampling / ragged offsets).
        // Other axes can lower via Transpose → Cumsum →
        // Transpose; not on the hot path today.
        let rank = node.shape.rank();
        let ax = if *axis < 0 {
            (rank as i32 + axis) as usize
        } else {
            *axis as usize
        };
        assert_eq!(
            ax,
            rank - 1,
            "Cumsum only supports the last axis on CPU today"
        );
        let cols = node.shape.dim(ax).unwrap_static();
        let total = node.shape.num_elements().unwrap();
        Thunk::Cumsum {
            src: node_offset(arena, node.inputs[0]),
            dst: node_offset(arena, node.id),
            rows: (total / cols) as u32,
            cols: cols as u32,
            exclusive: *exclusive,
            dtype: node.shape.dtype(),
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_top_k(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::TopK { k } = &node.op else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let rank = in_shape.rank();
        let axis_dim = in_shape.dim(rank - 1).unwrap_static();
        let outer = in_shape.num_elements().unwrap() / axis_dim;
        let indices_i64 = u8::from(graph.node(node.id).shape.dtype() == rlx_ir::DType::I64);
        Thunk::TopK {
            src: node_offset(arena, node.inputs[0]),
            dst: node_offset(arena, node.id),
            outer: outer as u32,
            axis_dim: axis_dim as u32,
            k: *k as u32,
            indices_i64,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_reduce(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Reduce {
        op,
        axes,
        keep_dim: _,
    } = &node.op
    else {
        unreachable!()
    };
    {
        // Decompose the input shape into [outer, reduced, inner]
        // around the reduced axis range. Non-contiguous reduced
        // axes aren't supported here — caller must transpose them
        // contiguous first (the coverage tool would surface the
        // gap if a model needs it).
        let in_shape = &graph.node(node.inputs[0]).shape;
        let rank = in_shape.rank();
        let mut sorted = axes.clone();
        sorted.sort();
        sorted.dedup();
        let contiguous = sorted.windows(2).all(|w| w[1] == w[0] + 1)
            && !sorted.is_empty()
            && *sorted.last().unwrap() < rank;
        if !contiguous {
            Thunk::Nop
        } else {
            let first = sorted[0];
            let last = *sorted.last().unwrap();
            let outer: usize = (0..first)
                .map(|i| in_shape.dim(i).unwrap_static())
                .product::<usize>()
                .max(1);
            let reduced: usize = (first..=last)
                .map(|i| in_shape.dim(i).unwrap_static())
                .product();
            let inner: usize = (last + 1..rank)
                .map(|i| in_shape.dim(i).unwrap_static())
                .product::<usize>()
                .max(1);
            let src = node_offset(arena, node.inputs[0]);
            let dst = node_offset(arena, node.id);
            if node.shape.dtype() == rlx_ir::DType::F64 && matches!(op, ReduceOp::Sum) {
                Thunk::ReduceSumF64 {
                    src,
                    dst,
                    outer: outer as u32,
                    reduced: reduced as u32,
                    inner: inner as u32,
                }
            } else {
                Thunk::Reduce {
                    src,
                    dst,
                    outer: outer as u32,
                    reduced: reduced as u32,
                    inner: inner as u32,
                    op: *op,
                }
            }
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_cumsum_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::CumsumBackward { exclusive, .. } = &node.op else {
        unreachable!()
    };
    {
        let dy_shape = &graph.node(node.inputs[0]).shape;
        let rank = dy_shape.rank();
        let cols = dy_shape.dim(rank - 1).unwrap_static();
        let rows = dy_shape.num_elements().unwrap() / cols;
        Thunk::CumsumBackward {
            dy: node_offset(arena, node.inputs[0]),
            dx: node_offset(arena, node.id),
            rows: rows as u32,
            cols: cols as u32,
            exclusive: *exclusive,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_softmax_cross_entropy(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::SoftmaxCrossEntropy = &node.op else {
        unreachable!()
    };
    {
        let logits_shape = &graph.node(node.inputs[0]).shape;
        if logits_shape.rank() == 2 {
            Thunk::SoftmaxCrossEntropyDense {
                logits: node_offset(arena, node.inputs[0]),
                targets: node_offset(arena, node.inputs[1]),
                dst: node_offset(arena, node.id),
                n: logits_shape.dim(0).unwrap_static() as u32,
                c: logits_shape.dim(1).unwrap_static() as u32,
            }
        } else {
            Thunk::Nop
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_softmax_cross_entropy_with_logits(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::SoftmaxCrossEntropyWithLogits = &node.op else {
        unreachable!()
    };
    {
        let logits_shape = &graph.node(node.inputs[0]).shape;
        if logits_shape.rank() == 2 {
            Thunk::SoftmaxCrossEntropy {
                logits: node_offset(arena, node.inputs[0]),
                labels: node_offset(arena, node.inputs[1]),
                dst: node_offset(arena, node.id),
                n: logits_shape.dim(0).unwrap_static() as u32,
                c: logits_shape.dim(1).unwrap_static() as u32,
            }
        } else {
            Thunk::Nop
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_softmax_cross_entropy_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::SoftmaxCrossEntropyBackward = &node.op else {
        unreachable!()
    };
    {
        let logits_shape = &graph.node(node.inputs[0]).shape;
        if logits_shape.rank() == 2 {
            Thunk::SoftmaxCrossEntropyBackward {
                logits: node_offset(arena, node.inputs[0]),
                labels: node_offset(arena, node.inputs[1]),
                d_loss: node_offset(arena, node.inputs[2]),
                dlogits: node_offset(arena, node.id),
                n: logits_shape.dim(0).unwrap_static() as u32,
                c: logits_shape.dim(1).unwrap_static() as u32,
            }
        } else {
            Thunk::Nop
        }
    }
}

#[inline(always)]
pub(crate) fn exec_reduce_sum_f64(t: &Thunk, base: *mut u8) {
    let Thunk::ReduceSumF64 {
        src,
        dst,
        outer,
        reduced,
        inner,
    } = t
    else {
        unreachable!()
    };
    {
        let (o, r, n) = (*outer as usize, *reduced as usize, *inner as usize);
        unsafe {
            let inp = sl_f64(*src, base, o * r * n);
            let out = sl_mut_f64(*dst, base, o * n);
            reduce_sum_f64(inp, out, o, r, n);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_softmax(t: &Thunk, base: *mut u8) {
    let Thunk::Softmax { data, rows, cols } = t else {
        unreachable!()
    };
    {
        let (rows, cols) = (*rows as usize, *cols as usize);
        unsafe {
            crate::kernels::neon_softmax(sl_mut(*data, base, rows * cols), rows, cols);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_cumsum(t: &Thunk, base: *mut u8) {
    let Thunk::Cumsum {
        src,
        dst,
        rows,
        cols,
        exclusive,
        dtype,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        cumsum_typed(
            base,
            *src,
            *dst,
            *rows as usize,
            *cols as usize,
            *exclusive,
            *dtype,
        );
    }
}

/// Row-wise cumulative sum along the last axis, honoring the element dtype. f32 is
/// the hot path (sampling / ragged offsets); I64 covers integer position ids /
/// mask cumsums (the f32 path would reinterpret the 8-byte i64 buffer as f32 and
/// scramble it, e.g. MOSS-TTS's RoPE `cumsum(attention_mask)`).
pub(crate) unsafe fn cumsum_typed(
    base: *mut u8,
    src: usize,
    dst: usize,
    rows: usize,
    cols: usize,
    exclusive: bool,
    dtype: rlx_ir::DType,
) {
    macro_rules! scan {
        ($sl:expr, $slm:expr, $zero:expr) => {{
            let s = $sl(src, base, rows * cols);
            let d = $slm(dst, base, rows * cols);
            for r in 0..rows {
                let mut acc = $zero;
                for c in 0..cols {
                    if exclusive {
                        d[r * cols + c] = acc;
                        acc += s[r * cols + c];
                    } else {
                        acc += s[r * cols + c];
                        d[r * cols + c] = acc;
                    }
                }
            }
        }};
    }
    match dtype {
        rlx_ir::DType::I64 => scan!(sl_i64, sl_mut_i64, 0i64),
        rlx_ir::DType::I32 => scan!(sl_i32, sl_mut_i32, 0i32),
        _ => scan!(sl, sl_mut, 0.0f32),
    }
}

#[inline(always)]
pub(crate) fn exec_sample(t: &Thunk, base: *mut u8) {
    let Thunk::Sample {
        logits,
        dst,
        batch,
        vocab,
        top_k,
        top_p,
        temperature,
        seed,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_sample_f32(
            *logits,
            *dst,
            *batch as usize,
            *vocab as usize,
            *top_k as usize,
            *top_p,
            *temperature,
            *seed,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_reduce(t: &Thunk, base: *mut u8) {
    let Thunk::Reduce {
        src,
        dst,
        outer,
        reduced,
        inner,
        op,
    } = t
    else {
        unreachable!()
    };
    {
        let outer = *outer as usize;
        let reduced = *reduced as usize;
        let inner = *inner as usize;
        let in_total = outer * reduced * inner;
        let out_total = outer * inner;
        unsafe {
            let inp = sl(*src, base, in_total);
            let out = sl_mut(*dst, base, out_total);
            // Each output element reduces a disjoint strided strip, so
            // the output range parallelizes (the bias-gradient reductions
            // read the big [N,C,H,W] tensors down to [C]; that's C-way).
            let reduce_one = |oi: usize| -> f32 {
                let o = oi / inner;
                let i = oi % inner;
                let mut acc = match op {
                    ReduceOp::Max => f32::NEG_INFINITY,
                    ReduceOp::Min => f32::INFINITY,
                    ReduceOp::Prod => 1.0f32,
                    _ => 0.0f32, // Sum / Mean
                };
                for r in 0..reduced {
                    let v = inp[o * reduced * inner + r * inner + i];
                    acc = match op {
                        ReduceOp::Sum | ReduceOp::Mean => acc + v,
                        ReduceOp::Max => acc.max(v),
                        ReduceOp::Min => acc.min(v),
                        ReduceOp::Prod => acc * v,
                    };
                }
                if matches!(op, ReduceOp::Mean) {
                    acc /= reduced as f32;
                }
                acc
            };
            if fast_conv_enabled() && crate::pool::should_parallelize(in_total) && out_total > 1 {
                let out_addr = out.as_mut_ptr() as usize;
                crate::pool::par_for(
                    out_total,
                    crate::pool::outer_chunk(out_total),
                    &|off, cnt| {
                        for oi in off..off + cnt {
                            *((out_addr as *mut f32).add(oi)) = reduce_one(oi);
                        }
                    },
                );
            } else {
                for oi in 0..out_total {
                    out[oi] = reduce_one(oi);
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_arg_reduce(t: &Thunk, base: *mut u8) {
    let Thunk::ArgReduce {
        src,
        dst,
        outer,
        reduced,
        inner,
        is_max,
    } = t
    else {
        unreachable!()
    };
    {
        let outer = *outer as usize;
        let reduced = *reduced as usize;
        let inner = *inner as usize;
        let in_total = outer * reduced * inner;
        let out_total = outer * inner;
        unsafe {
            let inp = sl(*src, base, in_total);
            let out = sl_mut(*dst, base, out_total);
            for o in 0..outer {
                for i in 0..inner {
                    let mut best = inp[o * reduced * inner + i];
                    let mut best_idx = 0usize;
                    for r in 1..reduced {
                        let v = inp[o * reduced * inner + r * inner + i];
                        let better = if *is_max { v > best } else { v < best };
                        if better {
                            best = v;
                            best_idx = r;
                        }
                    }
                    out[o * inner + i] = best_idx as f32;
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_cumsum_backward(t: &Thunk, base: *mut u8) {
    let Thunk::CumsumBackward {
        dy,
        dx,
        rows,
        cols,
        exclusive,
    } = t
    else {
        unreachable!()
    };
    {
        let (rows, cols) = (*rows as usize, *cols as usize);
        unsafe {
            let dys = sl(*dy, base, rows * cols);
            let out = sl_mut(*dx, base, rows * cols);
            for r in 0..rows {
                crate::training_bwd::cumsum_backward_row(
                    &dys[r * cols..(r + 1) * cols],
                    &mut out[r * cols..(r + 1) * cols],
                    *exclusive,
                );
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_softmax_cross_entropy_dense(t: &Thunk, base: *mut u8) {
    let Thunk::SoftmaxCrossEntropyDense {
        logits,
        targets,
        dst,
        n,
        c,
    } = t
    else {
        unreachable!()
    };
    {
        let n = *n as usize;
        let c = *c as usize;
        unsafe {
            let lg = sl(*logits, base, n * c);
            let tg = sl(*targets, base, n * c);
            let out = sl_mut(*dst, base, n);
            for ni in 0..n {
                let row = &lg[ni * c..(ni + 1) * c];
                let trow = &tg[ni * c..(ni + 1) * c];
                // log-sum-exp: max-subtract for stability.
                let mut m = f32::NEG_INFINITY;
                for &v in row {
                    if v > m {
                        m = v;
                    }
                }
                let mut sum = 0f32;
                for &v in row {
                    sum += (v - m).exp();
                }
                let lse = m + sum.ln();
                // loss = lse - Σ_c targets[c]·logits[c].
                let mut dot = 0f32;
                for k in 0..c {
                    dot += trow[k] * row[k];
                }
                out[ni] = lse - dot;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_softmax_cross_entropy(t: &Thunk, base: *mut u8) {
    let Thunk::SoftmaxCrossEntropy {
        logits,
        labels,
        dst,
        n,
        c,
    } = t
    else {
        unreachable!()
    };
    {
        let n = *n as usize;
        let c = *c as usize;
        unsafe {
            let lg = sl(*logits, base, n * c);
            let lb = sl(*labels, base, n);
            let out = sl_mut(*dst, base, n);
            for ni in 0..n {
                let row = &lg[ni * c..(ni + 1) * c];
                // log-sum-exp: max-subtract for stability.
                let mut m = f32::NEG_INFINITY;
                for &v in row {
                    if v > m {
                        m = v;
                    }
                }
                let mut sum = 0f32;
                for &v in row {
                    sum += (v - m).exp();
                }
                let lse = m + sum.ln();
                let label_idx = lb[ni] as usize;
                // loss = -(logits[label] - lse) = lse - logits[label].
                out[ni] = lse - row[label_idx];
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_softmax_cross_entropy_backward(t: &Thunk, base: *mut u8) {
    let Thunk::SoftmaxCrossEntropyBackward {
        logits,
        labels,
        d_loss,
        dlogits,
        n,
        c,
    } = t
    else {
        unreachable!()
    };
    {
        let n = *n as usize;
        let c = *c as usize;
        unsafe {
            let lg = sl(*logits, base, n * c);
            let lb = sl(*labels, base, n);
            let dl = sl(*d_loss, base, n);
            let out = sl_mut(*dlogits, base, n * c);
            for ni in 0..n {
                let row = &lg[ni * c..(ni + 1) * c];
                let label_idx = lb[ni] as usize;
                let scale = dl[ni];
                let mut m = f32::NEG_INFINITY;
                for &v in row {
                    if v > m {
                        m = v;
                    }
                }
                let mut sum = 0f32;
                for &v in row {
                    sum += (v - m).exp();
                }
                let inv_sum = 1.0 / sum;
                let dst_row = &mut out[ni * c..(ni + 1) * c];
                for k in 0..c {
                    let p = (row[k] - m).exp() * inv_sum;
                    let one_hot = if k == label_idx { 1.0 } else { 0.0 };
                    dst_row[k] = (p - one_hot) * scale;
                }
            }
        }
    }
}

/// Reference for `Op::ArgMax`/`Op::ArgMin` (f32-encoded indices, first-hit
/// tie-break). Reduces the middle `reduced` axis: input is logically
/// `[outer, reduced, inner]`, output `[outer, inner]`. Shared by the CPU thunk
/// and the Metal/WGPU host paths.
pub unsafe fn execute_argreduce_f32(
    src: usize,
    dst: usize,
    outer: usize,
    reduced: usize,
    inner: usize,
    is_max: bool,
    base: *mut u8,
) {
    let bptr = base as usize;
    unsafe {
        let inp = std::slice::from_raw_parts((bptr + src) as *const f32, outer * reduced * inner);
        let out = std::slice::from_raw_parts_mut((bptr + dst) as *mut f32, outer * inner);
        for o in 0..outer {
            for i in 0..inner {
                let mut best = inp[o * reduced * inner + i];
                let mut best_idx = 0usize;
                for r in 1..reduced {
                    let v = inp[o * reduced * inner + r * inner + i];
                    let better = if is_max { v > best } else { v < best };
                    if better {
                        best = v;
                        best_idx = r;
                    }
                }
                out[o * inner + i] = best_idx as f32;
            }
        }
    }
}

/// Token-sampling reference. Shared by the CPU `Thunk::Sample` arm and the
/// Metal unified-memory host fallback. `logits` is `[batch, vocab]` f32;
/// writes one f32-encoded index per batch row to `dst`. With a fixed `seed`
/// (and same top_k/top_p/temperature) the result is deterministic.
pub unsafe fn execute_sample_f32(
    logits: usize,
    dst: usize,
    batch: usize,
    vocab: usize,
    top_k: usize,
    top_p: f32,
    temperature: f32,
    seed: u64,
    base: *mut u8,
) {
    let (b, v) = (batch, vocab);
    let k = top_k.min(v);
    unsafe {
        let lg = sl(logits, base, b * v);
        let out = sl_mut(dst, base, b);
        let mut rng = rlx_ir::Philox4x32::new(if seed == 0 { 0xDEADBEEF } else { seed });
        for bi in 0..b {
            let row = &lg[bi * v..(bi + 1) * v];
            out[bi] = sample_row(row, k, top_p, temperature, &mut rng) as f32;
        }
    }
}

pub unsafe fn execute_cumsum_backward_f32(
    dy: usize,
    dx: usize,
    rows: u32,
    cols: u32,
    exclusive: bool,
    base: *mut u8,
) {
    let (rows, cols) = (rows as usize, cols as usize);
    let dys = sl(dy, base, rows * cols);
    let out = sl_mut(dx, base, rows * cols);
    for r in 0..rows {
        crate::training_bwd::cumsum_backward_row(
            &dys[r * cols..(r + 1) * cols],
            &mut out[r * cols..(r + 1) * cols],
            exclusive,
        );
    }
}

/// f64 sum reduction over a contiguous middle range.
/// Layout: input is `[outer, reduced, inner]`, output is `[outer, inner]`.
pub(crate) fn reduce_sum_f64(
    inp: &[f64],
    out: &mut [f64],
    outer: usize,
    reduced: usize,
    inner: usize,
) {
    for o in 0..outer {
        for n in 0..inner {
            let mut acc = 0.0_f64;
            for r in 0..reduced {
                acc += inp[o * reduced * inner + r * inner + n];
            }
            out[o * inner + n] = acc;
        }
    }
}

/// Host-side RNG fill against a byte arena (Metal/CUDA unified-memory fallback).
///
/// # Safety
///
/// `arena` must point to a valid allocation with at least `dst_off + len * 4` bytes.
pub unsafe fn fill_rng_normal_arena(
    dst_off: usize,
    len: usize,
    mean: f32,
    scale: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: rlx_ir::RngOptions,
    arena: *mut u8,
) {
    if len == 0 {
        return;
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut((arena.add(dst_off)) as *mut f32, len);
        rlx_ir::fill_normal_like(out, mean, scale, opts, key, op_seed);
    }
}

pub unsafe fn fill_rng_uniform_arena(
    dst_off: usize,
    len: usize,
    low: f32,
    high: f32,
    key: u64,
    op_seed: Option<f32>,
    opts: rlx_ir::RngOptions,
    arena: *mut u8,
) {
    if len == 0 {
        return;
    }
    unsafe {
        let out = std::slice::from_raw_parts_mut((arena.add(dst_off)) as *mut f32, len);
        rlx_ir::fill_uniform_like(out, low, high, opts, key, op_seed);
    }
}
