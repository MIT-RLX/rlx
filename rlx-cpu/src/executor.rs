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

//! Graph executor — runs a fused IR graph on CPU using the arena + kernels.
//!
//! The executor is the runtime hot path. For a 6-layer BERT, it makes
//! ~24 kernel calls total (one per fused node). Everything else is
//! inside the kernels — SIMD, BLAS, pre-allocated arena buffers.

use crate::arena::Arena;
use crate::kernels;
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{Graph, NodeId, Op};
use std::collections::HashMap;

/// External data provided at runtime (model weights + inputs).
pub struct ExternalBuffers<'a> {
    /// Map from node ID (Input/Param nodes) to external f32 data.
    pub buffers: HashMap<NodeId, &'a [f32]>,
}

/// Execute a compiled graph on CPU.
///
/// The graph should already be fused and memory-planned.
/// `arena` holds all intermediate buffers.
/// `external` provides input data and model weights.
///
/// Returns the output node IDs (data is in the arena).
pub fn execute(graph: &Graph, arena: &mut Arena, external: &ExternalBuffers) {
    let schedule: Vec<NodeId> = arena.schedule().to_vec();
    for &node_id in &schedule {
        let node = graph.node(node_id);

        match &node.op {
            // External data — skip (data provided via get_data which reads
            // from external buffers directly without copying to arena)
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => {}

            // ── Fused matmul + bias + optional activation ───────────
            Op::FusedMatMulBiasAct { activation } => {
                let input_id = node.inputs[0];
                let weight_id = node.inputs[1];
                let bias_id = node.inputs[2];

                let input = get_data(arena, external, input_id);
                let weight = get_data(arena, external, weight_id);
                let bias = get_data(arena, external, bias_id);
                let output = get_output(arena, node_id);

                // Compute output shape for sgemm
                let shape = &node.shape;
                let n = shape.dim(shape.rank() - 1).unwrap_static();
                let m = shape.num_elements().unwrap() / n;
                let k = input.len() / m;

                // sgemm: output = input @ weight
                // TODO: call cblas_sgemm via FFI
                // For now, naive matmul as placeholder
                matmul(input, weight, output, m, k, n);

                // Fused bias + activation (parallel NEON kernels)
                match activation {
                    Some(Activation::Gelu) => kernels::par_bias_gelu(output, bias, m, n),
                    Some(Activation::Silu) => {
                        crate::blas::bias_add(output, bias, m, n);
                        kernels::silu_inplace(output);
                    }
                    _ => crate::blas::bias_add(output, bias, m, n),
                }
            }

            // ── Fused residual + LayerNorm (parallel NEON) ──────────
            Op::FusedResidualLN { has_bias, eps } => {
                let x_id = node.inputs[0];
                let residual_id = node.inputs[1];
                let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                let zero_bias = vec![0f32; h];
                let (gamma_id, beta_id, bias_slice) = if *has_bias {
                    let b = get_data(arena, external, node.inputs[2]);
                    (node.inputs[3], node.inputs[4], b)
                } else {
                    (node.inputs[2], node.inputs[3], zero_bias.as_slice())
                };

                let x = get_data(arena, external, x_id);
                let residual = get_data(arena, external, residual_id);
                let gamma = get_data(arena, external, gamma_id);
                let beta = get_data(arena, external, beta_id);
                let output = get_output(arena, node_id);

                let n = x.len() / h;

                // Parallel: each thread processes a chunk of rows
                let x_ptr = x.as_ptr() as usize;
                let r_ptr = residual.as_ptr() as usize;
                let o_ptr = output.as_mut_ptr() as usize;
                let bi_ptr = bias_slice.as_ptr() as usize;
                let g_ptr = gamma.as_ptr() as usize;
                let b_ptr = beta.as_ptr() as usize;
                let e = *eps;
                crate::pool::par_for(n, 4, &|off, cnt| unsafe {
                    let x_s =
                        std::slice::from_raw_parts((x_ptr as *const f32).add(off * h), cnt * h);
                    let r_s =
                        std::slice::from_raw_parts((r_ptr as *const f32).add(off * h), cnt * h);
                    let o_s =
                        std::slice::from_raw_parts_mut((o_ptr as *mut f32).add(off * h), cnt * h);
                    let bi = std::slice::from_raw_parts(bi_ptr as *const f32, h);
                    let g = std::slice::from_raw_parts(g_ptr as *const f32, h);
                    let b = std::slice::from_raw_parts(b_ptr as *const f32, h);
                    kernels::residual_bias_layer_norm(x_s, r_s, bi, g, b, o_s, cnt, h, e);
                });
            }

            // ── Fused residual + RMSNorm (parallel) ─────────────────
            Op::FusedResidualRmsNorm { has_bias, eps } => {
                let x_id = node.inputs[0];
                let residual_id = node.inputs[1];
                let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                let zero_bias = vec![0f32; h];
                let (gamma_id, beta_id, bias_slice) = if *has_bias {
                    let b = get_data(arena, external, node.inputs[2]);
                    (node.inputs[3], node.inputs[4], b)
                } else {
                    (node.inputs[2], node.inputs[3], zero_bias.as_slice())
                };

                let x = get_data(arena, external, x_id);
                let residual = get_data(arena, external, residual_id);
                let gamma = get_data(arena, external, gamma_id);
                let beta = get_data(arena, external, beta_id);
                let output = get_output(arena, node_id);

                let n = x.len() / h;

                let x_ptr = x.as_ptr() as usize;
                let r_ptr = residual.as_ptr() as usize;
                let o_ptr = output.as_mut_ptr() as usize;
                let bi_ptr = bias_slice.as_ptr() as usize;
                let g_ptr = gamma.as_ptr() as usize;
                let b_ptr = beta.as_ptr() as usize;
                let e = *eps;
                crate::pool::par_for(n, 4, &|off, cnt| unsafe {
                    let x_s =
                        std::slice::from_raw_parts((x_ptr as *const f32).add(off * h), cnt * h);
                    let r_s =
                        std::slice::from_raw_parts((r_ptr as *const f32).add(off * h), cnt * h);
                    let o_s =
                        std::slice::from_raw_parts_mut((o_ptr as *mut f32).add(off * h), cnt * h);
                    let bi = std::slice::from_raw_parts(bi_ptr as *const f32, h);
                    let g = std::slice::from_raw_parts(g_ptr as *const f32, h);
                    let b = std::slice::from_raw_parts(b_ptr as *const f32, h);
                    kernels::residual_bias_rms_norm(x_s, r_s, bi, g, b, o_s, cnt, h, e);
                });
            }

            // ── Plain matmul ────────────────────────────────────────
            Op::MatMul => {
                let lhs = get_data(arena, external, node.inputs[0]);
                let rhs = get_data(arena, external, node.inputs[1]);
                let output = get_output(arena, node_id);

                let shape = &node.shape;
                let lhs_shape = &graph.node(node.inputs[0]).shape;
                let rhs_shape = &graph.node(node.inputs[1]).shape;
                let n = shape.dim(shape.rank() - 1).unwrap_static();
                let out_m_inner = shape.dim(shape.rank() - 2).unwrap_static();
                let k = lhs_shape.dim(lhs_shape.rank() - 1).unwrap_static();

                // Outer batch dims — present when either input has rank > 2.
                // Compute total batch as output.num_elements / (M * N).
                let total = shape.num_elements().unwrap();
                let per_batch_out = out_m_inner * n;
                let batches = total / per_batch_out;

                if batches == 1 {
                    matmul(lhs, rhs, output, out_m_inner, k, n);
                } else {
                    let lhs_batched = lhs_shape.num_elements().unwrap_or(0)
                        == batches * out_m_inner * k;
                    let rhs_batched =
                        rhs_shape.num_elements().unwrap_or(0) == batches * k * n;
                    for b in 0..batches {
                        let l_off = if lhs_batched { b * out_m_inner * k } else { 0 };
                        let r_off = if rhs_batched { b * k * n } else { 0 };
                        let o_off = b * out_m_inner * n;
                        let l_slice = &lhs[l_off..l_off + out_m_inner * k];
                        let r_slice = &rhs[r_off..r_off + k * n];
                        let o_slice = &mut output[o_off..o_off + out_m_inner * n];
                        matmul(l_slice, r_slice, o_slice, out_m_inner, k, n);
                    }
                }
            }

            // ── Element-wise binary ─────────────────────────────────
            Op::Binary(op) => {
                let lhs = get_data(arena, external, node.inputs[0]);
                let rhs = get_data(arena, external, node.inputs[1]);
                let output = get_output(arena, node_id);
                let len = output.len();
                let rhs_len = rhs.len();

                // Fast path: Add with broadcast bias → NEON bias_add
                if matches!(op, BinaryOp::Add) && rhs_len < len && len.is_multiple_of(rhs_len) {
                    output.copy_from_slice(lhs);
                    crate::blas::bias_add(output, rhs, len / rhs_len, rhs_len);
                } else if rhs_len == len {
                    for i in 0..len {
                        output[i] = binary_op(*op, lhs[i], rhs[i]);
                    }
                } else {
                    for i in 0..len {
                        output[i] = binary_op(*op, lhs[i], rhs[i % rhs_len]);
                    }
                }
            }

            // ── Unary activation ────────────────────────────────────
            Op::Activation(act) => {
                let input = get_data(arena, external, node.inputs[0]);
                let output = get_output(arena, node_id);
                output.copy_from_slice(input);
                let zeros = vec![0f32; node.shape.dim(node.shape.rank() - 1).unwrap_static()];
                let m = output.len() / zeros.len();
                let n = zeros.len();
                match act {
                    Activation::Gelu => kernels::par_bias_gelu(output, &zeros, m, n),
                    Activation::Silu => kernels::silu_inplace(output),
                    Activation::Relu => {
                        for v in output.iter_mut() {
                            *v = v.max(0.0);
                        }
                    }
                    Activation::Exp => {
                        for v in output.iter_mut() {
                            *v = v.exp();
                        }
                    }
                    Activation::Sqrt => {
                        for v in output.iter_mut() {
                            *v = v.sqrt();
                        }
                    }
                    Activation::Neg => {
                        for v in output.iter_mut() {
                            *v = -*v;
                        }
                    }
                    Activation::Tanh => {
                        for v in output.iter_mut() {
                            *v = v.tanh();
                        }
                    }
                    Activation::Sigmoid => {
                        for v in output.iter_mut() {
                            *v = 1.0 / (1.0 + (-*v).exp());
                        }
                    }
                    _ => {}
                }
            }

            // ── Gather (embedding lookup) ───────────────────────────
            Op::Gather { axis } => {
                let table = get_data(arena, external, node.inputs[0]);
                let indices = get_data(arena, external, node.inputs[1]);
                let output = get_output(arena, node_id);

                let table_shape = &graph.node(node.inputs[0]).shape;
                let _out_shape = &node.shape;

                // For axis=0 (embedding): table[V, D...], indices[B, S] → [B, S, D...]
                if *axis == 0 {
                    let trailing: usize = (1..table_shape.rank())
                        .map(|i| table_shape.dim(i).unwrap_static())
                        .product();
                    for (i, &idx_f32) in indices.iter().enumerate() {
                        let idx = idx_f32 as usize;
                        let src = idx * trailing;
                        let dst = i * trailing;
                        output[dst..dst + trailing].copy_from_slice(&table[src..src + trailing]);
                    }
                } else {
                    // General gather — fallback
                    output.fill(0.0);
                }
            }

            // ── Narrow (slice along axis) ───────────────────────────
            Op::Narrow { axis, start, len } => {
                let input = get_data(arena, external, node.inputs[0]);
                let output = get_output(arena, node_id);
                let in_shape = &graph.node(node.inputs[0]).shape;

                let rank = in_shape.rank();
                let outer: usize = (0..*axis)
                    .map(|i| in_shape.dim(i).unwrap_static())
                    .product::<usize>()
                    .max(1);
                let inner: usize = (*axis + 1..rank)
                    .map(|i| in_shape.dim(i).unwrap_static())
                    .product::<usize>()
                    .max(1);
                let in_axis_size = in_shape.dim(*axis).unwrap_static();

                for o in 0..outer {
                    for s in 0..*len {
                        let src_off = o * in_axis_size * inner + (*start + s) * inner;
                        let dst_off = o * len * inner + s * inner;
                        output[dst_off..dst_off + inner]
                            .copy_from_slice(&input[src_off..src_off + inner]);
                    }
                }
            }

            // ── Transpose ───────────────────────────────────────────
            Op::Transpose { perm } => {
                let input = get_data(arena, external, node.inputs[0]);
                let output = get_output(arena, node_id);
                let in_shape = &graph.node(node.inputs[0]).shape;
                let rank = in_shape.rank();

                let in_dims: Vec<usize> =
                    (0..rank).map(|i| in_shape.dim(i).unwrap_static()).collect();
                let out_dims: Vec<usize> = perm.iter().map(|&i| in_dims[i]).collect();

                // Row-major strides for input and output spaces.
                // For a shape [d0, d1, ..., d_{r-1}], stride[i] = product(d_{i+1..r}).
                let mut in_strides = vec![1usize; rank];
                for i in (0..rank - 1).rev() {
                    in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
                }
                let mut out_strides = vec![1usize; rank];
                for i in (0..rank - 1).rev() {
                    out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
                }

                let total = output.len();
                for flat_out in 0..total {
                    let mut in_flat = 0;
                    for d in 0..rank {
                        // out_coord[d] decoded from flat_out via output strides.
                        let coord = (flat_out / out_strides[d]) % out_dims[d];
                        // Output dim d came from input dim perm[d].
                        in_flat += coord * in_strides[perm[d]];
                    }
                    output[flat_out] = input[in_flat];
                }
            }

            // ── Concat ──────────────────────────────────────────────
            Op::Concat { axis } => {
                let output = get_output(arena, node_id);
                let out_shape = &node.shape;
                let rank = out_shape.rank();

                let outer: usize = (0..*axis)
                    .map(|i| out_shape.dim(i).unwrap_static())
                    .product::<usize>()
                    .max(1);
                let inner: usize = (*axis + 1..rank)
                    .map(|i| out_shape.dim(i).unwrap_static())
                    .product::<usize>()
                    .max(1);

                let mut dst_off = 0;
                for o in 0..outer {
                    for &inp_id in &node.inputs {
                        let inp = get_data(arena, external, inp_id);
                        let inp_shape = &graph.node(inp_id).shape;
                        let inp_axis = inp_shape.dim(*axis).unwrap_static();
                        let chunk = inp_axis * inner;
                        let src_off = o * chunk;
                        output[dst_off..dst_off + chunk]
                            .copy_from_slice(&inp[src_off..src_off + chunk]);
                        dst_off += chunk;
                    }
                }
            }

            // ── Reshape (zero-copy: same data, different shape) ─────
            Op::Reshape { .. } | Op::Expand { .. } => {
                let input = get_data(arena, external, node.inputs[0]);
                let output = get_output(arena, node_id);
                output[..input.len()].copy_from_slice(input);
            }

            // ── LayerNorm (parallel NEON) ────────────────────────────
            Op::LayerNorm { eps, .. } => {
                let input = get_data(arena, external, node.inputs[0]);
                let gamma = get_data(arena, external, node.inputs[1]);
                let beta = get_data(arena, external, node.inputs[2]);
                let output = get_output(arena, node_id);
                let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                let n = input.len() / h;
                for row in 0..n {
                    let base = row * h;
                    kernels::layer_norm_row(
                        &input[base..base + h],
                        gamma,
                        beta,
                        &mut output[base..base + h],
                        h,
                        *eps,
                    );
                }
            }

            Op::GroupNorm {
                num_groups,
                eps,
            } => {
                let input = get_data(arena, external, node.inputs[0]);
                let gamma = get_data(arena, external, node.inputs[1]);
                let beta = get_data(arena, external, node.inputs[2]);
                let output = get_output(arena, node_id);
                let n = node.shape.dim(0).unwrap_static();
                let c = node.shape.dim(1).unwrap_static();
                let h = node.shape.dim(2).unwrap_static();
                let w = node.shape.dim(3).unwrap_static();
                kernels::group_norm_nchw(
                    input,
                    gamma,
                    beta,
                    output,
                    n,
                    c,
                    h,
                    w,
                    *num_groups,
                    *eps,
                );
            }

            Op::ResizeNearest2x => {
                let input = get_data(arena, external, node.inputs[0]);
                let output = get_output(arena, node_id);
                let n = node.shape.dim(0).unwrap_static();
                let c = node.shape.dim(1).unwrap_static();
                let h = node.shape.dim(2).unwrap_static() / 2;
                let w = node.shape.dim(3).unwrap_static() / 2;
                let in_plane = c * h * w;
                let out_plane = c * h * 2 * w * 2;
                for ni in 0..n {
                    kernels::resize_nearest_2x_nchw(
                        &input[ni * in_plane..(ni + 1) * in_plane],
                        &mut output[ni * out_plane..(ni + 1) * out_plane],
                        c,
                        h,
                        w,
                    );
                }
            }

            Op::AxialRope2d {
                end_x,
                end_y,
                head_dim,
                num_heads,
                theta,
                repeat_factor,
            } => {
                let input = get_data(arena, external, node.inputs[0]);
                let output = get_output(arena, node_id);
                let batch = node.shape.dim(0).unwrap_static();
                let seq = node.shape.dim(1).unwrap_static();
                let plane = seq * node.shape.dim(2).unwrap_static();
                for bi in 0..batch {
                    let rotated = rlx_ir::ops::axial_rope2d::apply_axial_rope2d(
                        &input[bi * plane..(bi + 1) * plane],
                        *num_heads,
                        seq,
                        *head_dim,
                        *end_x,
                        *end_y,
                        *theta,
                        *repeat_factor,
                    );
                    output[bi * plane..(bi + 1) * plane].copy_from_slice(&rotated);
                }
            }

            // ── Softmax ─────────────────────────────────────────────
            Op::Softmax { axis } => {
                let input = get_data(arena, external, node.inputs[0]);
                let output = get_output(arena, node_id);
                output.copy_from_slice(input);
                let rank = node.shape.rank();
                let ax = if *axis < 0 {
                    (rank as i32 + axis) as usize
                } else {
                    *axis as usize
                };
                let cols = node.shape.dim(ax).unwrap_static();
                let rows = output.len() / cols;
                crate::naive::softmax(output, rows, cols);
            }

            // ── Attention (SDPA) — BLAS-accelerated ─────────────────
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
            } => {
                let q = get_data(arena, external, node.inputs[0]);
                let k = get_data(arena, external, node.inputs[1]);
                let v = get_data(arena, external, node.inputs[2]);
                // For non-Custom mask kinds the IR emits no mask input —
                // synthesize an empty slice so the masking branch below
                // sees `mask.len() < ...` and skips.
                let mask: &[f32] = if matches!(
                    mask_kind,
                    rlx_ir::op::MaskKind::Custom | rlx_ir::op::MaskKind::Bias
                ) {
                    get_data(arena, external, node.inputs[3])
                } else {
                    &[]
                };
                let output = get_output(arena, node_id);

                let q_shape = &graph.node(node.inputs[0]).shape;
                let k_shape = &graph.node(node.inputs[1]).shape;
                let hs = num_heads * head_dim;
                let scale = (*head_dim as f32).powf(-0.5);
                let (batch_size, s_q) = if q_shape.rank() >= 3 {
                    (
                        q_shape.dim(0).unwrap_static(),
                        q_shape.dim(1).unwrap_static(),
                    )
                } else {
                    (1, q_shape.dim(0).unwrap_static())
                };
                // K and V share Lk. In decode mode Lk = past+1 and Lq = 1;
                // in prefill Lq = Lk. Causal/SlidingWindow masking is
                // expressed in absolute positions: Q-row qi is at absolute
                // position (Lk - Lq) + qi, so masking shifts accordingly.
                let s_k = if k_shape.rank() >= 3 {
                    k_shape.dim(1).unwrap_static()
                } else {
                    k_shape.dim(0).unwrap_static()
                };
                let q_offset = s_k.saturating_sub(s_q);

                // Pre-allocate buffers ONCE (reused across heads)
                let q_buf_len = s_q * head_dim;
                let k_buf_len = s_k * head_dim;
                let mut q_head = vec![0f32; q_buf_len];
                let mut k_head = vec![0f32; k_buf_len];
                let mut v_head = vec![0f32; k_buf_len];
                let mut scores = vec![0f32; s_q * s_k];
                let mut out_head = vec![0f32; q_buf_len];

                for bi in 0..batch_size {
                    for hi in 0..*num_heads {
                        // Gather per-head Q (Lq rows).
                        for si in 0..s_q {
                            let off = bi * s_q * hs + si * hs + hi * head_dim;
                            q_head[si * head_dim..(si + 1) * head_dim]
                                .copy_from_slice(&q[off..off + head_dim]);
                        }
                        // Gather per-head K, V (Lk rows).
                        for si in 0..s_k {
                            let off = bi * s_k * hs + si * hs + hi * head_dim;
                            k_head[si * head_dim..(si + 1) * head_dim]
                                .copy_from_slice(&k[off..off + head_dim]);
                            v_head[si * head_dim..(si + 1) * head_dim]
                                .copy_from_slice(&v[off..off + head_dim]);
                        }
                        // Q@K^T: scores[Lq, Lk]. Use NEON dots when the
                        // larger of Lq/Lk is small; BLAS otherwise.
                        if s_q.max(s_k) <= 32 {
                            for qi in 0..s_q {
                                for ki in 0..s_k {
                                    let q_off = qi * head_dim;
                                    let k_off = ki * head_dim;
                                    #[cfg(target_arch = "aarch64")]
                                    let mut dot;
                                    #[cfg(not(target_arch = "aarch64"))]
                                    let mut dot = 0f32;
                                    #[cfg(target_arch = "aarch64")]
                                    unsafe {
                                        use std::arch::aarch64::*;
                                        let chunks = head_dim / 4;
                                        let mut acc = vdupq_n_f32(0.0);
                                        for c in 0..chunks {
                                            let vq = vld1q_f32(q_head.as_ptr().add(q_off + c * 4));
                                            let vk = vld1q_f32(k_head.as_ptr().add(k_off + c * 4));
                                            acc = vfmaq_f32(acc, vq, vk);
                                        }
                                        dot = vaddvq_f32(acc);
                                        for d in (chunks * 4)..*head_dim {
                                            dot += q_head[q_off + d] * k_head[k_off + d];
                                        }
                                    }
                                    #[cfg(not(target_arch = "aarch64"))]
                                    {
                                        for d in 0..*head_dim {
                                            dot += q_head[q_off + d] * k_head[k_off + d];
                                        }
                                    }
                                    scores[qi * s_k + ki] = dot * scale;
                                }
                            }
                        } else {
                            crate::blas::sgemm_bt(
                                &q_head,
                                &k_head,
                                &mut scores,
                                s_q,
                                *head_dim,
                                s_k,
                                scale,
                            );
                        }
                        // Mask: branch on kind so None / Causal skip the
                        // mask load entirely. Causal/SlidingWindow use
                        // absolute positions so they handle Lq != Lk
                        // (decode-mode with cached K/V).
                        match mask_kind {
                            rlx_ir::op::MaskKind::None => {}
                            rlx_ir::op::MaskKind::Causal => {
                                for qi in 0..s_q {
                                    let abs_q = q_offset + qi;
                                    for ki in (abs_q + 1)..s_k {
                                        scores[qi * s_k + ki] = -1e9;
                                    }
                                }
                            }
                            rlx_ir::op::MaskKind::SlidingWindow(w) => {
                                for qi in 0..s_q {
                                    let abs_q = q_offset + qi;
                                    let lo = abs_q.saturating_sub(*w);
                                    for ki in 0..s_k {
                                        if ki < lo || ki > abs_q {
                                            scores[qi * s_k + ki] = -1e9;
                                        }
                                    }
                                }
                            }
                            rlx_ir::op::MaskKind::Custom => {
                                if mask.len() >= (bi + 1) * s_k {
                                    let m = &mask[bi * s_k..(bi + 1) * s_k];
                                    for qi in 0..s_q {
                                        for ki in 0..s_k {
                                            if m[ki] < 0.5 {
                                                scores[qi * s_k + ki] = -1e9;
                                            }
                                        }
                                    }
                                }
                            }
                            rlx_ir::op::MaskKind::Bias => {
                                // Bias is [batch, num_heads, s_q, s_k]
                                // (additive, pre-softmax). Skip if the
                                // buffer wasn't supplied.
                                let per_bh = s_q * s_k;
                                let need = (bi * *num_heads + hi + 1) * per_bh;
                                if mask.len() >= need {
                                    let bias_off = (bi * *num_heads + hi) * per_bh;
                                    let b = &mask[bias_off..bias_off + per_bh];
                                    for i in 0..per_bh {
                                        scores[i] += b[i];
                                    }
                                }
                            }
                        }
                        crate::naive::softmax(&mut scores, s_q, s_k);
                        // scores[Lq, Lk] @ V[Lk, head_dim] → out_head[Lq, head_dim]
                        if s_q.max(s_k) <= 32 {
                            out_head.fill(0.0);
                            for qi in 0..s_q {
                                for ki in 0..s_k {
                                    let sc = scores[qi * s_k + ki];
                                    if sc > 1e-8 {
                                        let v_off = ki * head_dim;
                                        let o_off = qi * head_dim;
                                        #[cfg(target_arch = "aarch64")]
                                        unsafe {
                                            use std::arch::aarch64::*;
                                            let vsc = vdupq_n_f32(sc);
                                            let chunks = head_dim / 4;
                                            for c in 0..chunks {
                                                let off = c * 4;
                                                let vo =
                                                    vld1q_f32(out_head.as_ptr().add(o_off + off));
                                                let vv =
                                                    vld1q_f32(v_head.as_ptr().add(v_off + off));
                                                vst1q_f32(
                                                    out_head.as_mut_ptr().add(o_off + off),
                                                    vfmaq_f32(vo, vsc, vv),
                                                );
                                            }
                                        }
                                        #[cfg(not(target_arch = "aarch64"))]
                                        for d in 0..*head_dim {
                                            out_head[o_off + d] += sc * v_head[v_off + d];
                                        }
                                    }
                                }
                            }
                        } else {
                            crate::blas::sgemm(
                                &scores,
                                &v_head,
                                &mut out_head,
                                s_q,
                                s_k,
                                *head_dim,
                            );
                        }
                        // Scatter back into [B, Lq, hs].
                        for si in 0..s_q {
                            let off = bi * s_q * hs + si * hs + hi * head_dim;
                            output[off..off + head_dim]
                                .copy_from_slice(&out_head[si * head_dim..(si + 1) * head_dim]);
                        }
                    }
                }
            }

            // ── Rotary position embedding ────────────────────────────
            Op::Rope { head_dim, n_rot } => {
                let head_dim = *head_dim;
                let n_rot = *n_rot;
                let x = get_data(arena, external, node.inputs[0]);
                let cos_cache = get_data(arena, external, node.inputs[1]);
                let sin_cache = get_data(arena, external, node.inputs[2]);
                let output = get_output(arena, node_id);
                output.copy_from_slice(x);

                let rot_half = n_rot / 2;
                let tab_half = head_dim / 2;
                let total = output.len();
                let num_chunks = total / head_dim;
                for chunk in 0..num_chunks {
                    let off = chunk * head_dim;
                    let cos_len = cos_cache.len();
                    let max_seq = cos_len / tab_half;
                    let pos = chunk % max_seq;
                    let cos_off = pos * tab_half;

                    for i in 0..rot_half {
                        let cos_v = cos_cache[cos_off + i];
                        let sin_v = sin_cache[cos_off + i];
                        let x1 = output[off + i];
                        let x2 = output[off + rot_half + i];
                        output[off + i] = x1 * cos_v - x2 * sin_v;
                        output[off + rot_half + i] = x2 * cos_v + x1 * sin_v;
                    }
                    for j in n_rot..head_dim {
                        output[off + j] = x[off + j];
                    }
                }
            }

            // ── Compare ─────────────────────────────────────────────
            Op::Compare(cmp) => {
                let lhs = get_data(arena, external, node.inputs[0]);
                let rhs = get_data(arena, external, node.inputs[1]);
                let output = get_output(arena, node_id);
                let rhs_len = rhs.len();
                for i in 0..output.len() {
                    let a = lhs[i];
                    let b = rhs[i % rhs_len];
                    output[i] = if compare_op(*cmp, a, b) { 1.0 } else { 0.0 };
                }
            }

            // ── Where (conditional select) ──────────────────────────
            Op::Where => {
                let cond = get_data(arena, external, node.inputs[0]);
                let on_true = get_data(arena, external, node.inputs[1]);
                let on_false = get_data(arena, external, node.inputs[2]);
                let output = get_output(arena, node_id);
                for i in 0..output.len() {
                    output[i] = if cond[i] > 0.5 {
                        on_true[i]
                    } else {
                        on_false[i]
                    };
                }
            }

            // ── Reduce ──────────────────────────────────────────────
            Op::Reduce {
                op: reduce_op,
                axes,
                keep_dim: _,
            } => {
                let input = get_data(arena, external, node.inputs[0]);
                let output = get_output(arena, node_id);
                output.fill(0.0);
                // Simple: only handle single-axis reduction for now
                if axes.len() == 1 {
                    let in_shape = &graph.node(node.inputs[0]).shape;
                    let axis = axes[0];
                    let rank = in_shape.rank();
                    let outer: usize = (0..axis)
                        .map(|i| in_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);
                    let axis_size = in_shape.dim(axis).unwrap_static();
                    let inner: usize = (axis + 1..rank)
                        .map(|i| in_shape.dim(i).unwrap_static())
                        .product::<usize>()
                        .max(1);

                    match reduce_op {
                        ReduceOp::Sum | ReduceOp::Mean => {
                            for o in 0..outer {
                                for i in 0..inner {
                                    let mut acc = 0f32;
                                    for a in 0..axis_size {
                                        acc += input[o * axis_size * inner + a * inner + i];
                                    }
                                    if matches!(reduce_op, ReduceOp::Mean) {
                                        acc /= axis_size as f32;
                                    }
                                    output[o * inner + i] = acc;
                                }
                            }
                        }
                        ReduceOp::Max => {
                            output.fill(f32::NEG_INFINITY);
                            for o in 0..outer {
                                for i in 0..inner {
                                    for a in 0..axis_size {
                                        let v = input[o * axis_size * inner + a * inner + i];
                                        let idx = o * inner + i;
                                        if v > output[idx] {
                                            output[idx] = v;
                                        }
                                    }
                                }
                            }
                        }
                        _ => {} // Min, Prod — TODO
                    }
                }
            }

            // ── Cast ────────────────────────────────────────────────
            Op::Cast { .. } => {
                let input = get_data(arena, external, node.inputs[0]);
                let output = get_output(arena, node_id);
                output[..input.len()].copy_from_slice(input);
            }

            // ── Fused SwiGLU ────────────────────────────────────────
            // Input layout: concatenated [..., 2N] tensor where the first
            // N elements per row are the "up" projection and the next N
            // are the "gate" projection. Output: [..., N] = up * silu(gate).
            // `cast_to` is currently advisory: rlx-cpu always operates in
            // f32, so backends that distinguish dtypes apply the cast; the
            // CPU executor stores the f32 result regardless.
            Op::FusedSwiGLU { cast_to: _, .. } => {
                let input = get_data(arena, external, node.inputs[0]);
                let output = get_output(arena, node_id);
                // n = last-dim half (read from the node's own shape, NOT
                // derived from buffer lengths — those count total elements
                // including all leading dims).
                let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
                let outer = output.len() / n;
                debug_assert_eq!(
                    outer * 2 * n,
                    input.len(),
                    "FusedSwiGLU: input/output shape mismatch"
                );
                for o in 0..outer {
                    let in_row = &input[o * 2 * n..(o + 1) * 2 * n];
                    let out_row = &mut output[o * n..(o + 1) * n];
                    for i in 0..n {
                        let up = in_row[i];
                        let gate = in_row[n + i];
                        let silu_gate = gate / (1.0 + (-gate).exp());
                        out_row[i] = up * silu_gate;
                    }
                }
            }

            // ── DenseSolve: x = A⁻¹ b (F32 / F64 via LAPACK) ────────
            Op::DenseSolve => {
                let a_shape = &graph.node(node.inputs[0]).shape;
                let n = a_shape.dim(0).unwrap_static();
                let b_elems = node.shape.num_elements().unwrap();
                let nrhs = b_elems / n.max(1);
                match node.shape.dtype() {
                    rlx_ir::DType::F32 => {
                        let a = get_data(arena, external, node.inputs[0]);
                        let b = get_data(arena, external, node.inputs[1]);
                        let x = get_output(arena, node_id);
                        let mut a_scratch = a.to_vec();
                        let mut x_buf = b.to_vec();
                        let info = crate::blas::sgesv(&mut a_scratch, &mut x_buf, n, nrhs);
                        if info != 0 {
                            panic!("DenseSolve: singular matrix (info={info})");
                        }
                        x[..x_buf.len()].copy_from_slice(&x_buf);
                    }
                    rlx_ir::DType::F64 => {
                        let (a_ptr, a_len) = arena.raw_ptr(node.inputs[0]);
                        let (b_ptr, b_len) = arena.raw_ptr(node.inputs[1]);
                        let (x_ptr, x_len) = arena.raw_ptr(node_id);
                        unsafe {
                            let a_src =
                                std::slice::from_raw_parts(a_ptr as *const f64, a_len / 8);
                            let b_src =
                                std::slice::from_raw_parts(b_ptr as *const f64, b_len / 8);
                            let mut a_scratch = a_src.to_vec();
                            let mut x_buf = b_src.to_vec();
                            let info = crate::blas::dgesv(&mut a_scratch, &mut x_buf, n, nrhs);
                            if info != 0 {
                                panic!("DenseSolve: singular matrix (info={info})");
                            }
                            std::slice::from_raw_parts_mut(x_ptr as *mut f64, x_len / 8)
                                .copy_from_slice(&x_buf);
                        }
                    }
                    other => panic!("DenseSolve executor: unsupported dtype {other:?}"),
                }
            }

            // ── Passthrough for unimplemented ops ───────────────────
            _ => {
                if !node.inputs.is_empty() && arena.has_buffer(node_id) {
                    let input = get_data(arena, external, node.inputs[0]);
                    let output = get_output(arena, node_id);
                    let len = output.len().min(input.len());
                    output[..len].copy_from_slice(&input[..len]);
                }
            }
        }
    }
}

/// Get read-only data for a node — from external or arena via raw pointer.
/// SAFETY: the memory planner guarantees that input buffers don't overlap with
/// the output buffer being written, so concurrent read+write is safe.
fn get_data<'a>(arena: &'a Arena, external: &'a ExternalBuffers, id: NodeId) -> &'a [f32] {
    // Check external first (test mode, or runtime inputs copied via run())
    // Then arena (params pre-stored by set_param, computed intermediates)
    if let Some(&ext) = external.buffers.get(&id) {
        ext
    } else if arena.has_buffer(id) {
        let (ptr, len) = arena.raw_ptr(id);
        unsafe { std::slice::from_raw_parts(ptr, len) }
    } else {
        panic!("no data for node {id}")
    }
}

/// Get mutable output buffer via raw pointer (doesn't borrow arena).
/// Takes `&Arena` (not `&mut Arena`) on purpose — the executor walks
/// the schedule with multiple node-buffer references live at once;
/// the arena allocator already partitioned them into non-overlapping
/// regions at compile time.
#[allow(clippy::mut_from_ref)]
fn get_output(arena: &Arena, id: NodeId) -> &mut [f32] {
    let (ptr, len) = arena.raw_ptr(id);
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

/// Matrix multiply — uses BLAS when linked, naive fallback otherwise.
#[inline]
fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Use BLAS sgemm (Accelerate/MKL/OpenBLAS) — linked via build.rs
    crate::blas::sgemm(a, b, c, m, k, n);
}

fn binary_op(op: rlx_ir::op::BinaryOp, a: f32, b: f32) -> f32 {
    use rlx_ir::op::BinaryOp::*;
    match op {
        Add => a + b,
        Sub => a - b,
        Mul => a * b,
        Div => a / b,
        Max => a.max(b),
        Min => a.min(b),
        Pow => a.powf(b),
    }
}

fn compare_op(op: rlx_ir::op::CmpOp, a: f32, b: f32) -> bool {
    use rlx_ir::op::CmpOp::*;
    match op {
        Eq => a == b,
        Ne => a != b,
        Lt => a < b,
        Le => a <= b,
        Gt => a > b,
        Ge => a >= b,
    }
}

// Reference scalar GELU — kept as a parity oracle for SIMD paths.
#[allow(dead_code)]
fn scalar_gelu(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0f32 } else { -1.0 };
    let xa = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * xa);
    let y = t
        * (0.254_829_6
            + t * (-0.284_496_72 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    let erf = sign * (1.0 - y * (-xa * xa).exp());
    x * 0.5 * (1.0 + erf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::*;

    use rlx_opt::fusion::FuseMatMulBiasAct;
    use rlx_opt::memory;
    use rlx_opt::pass::Pass;

    /// End-to-end test: build graph → fuse → plan memory → execute.
    #[test]
    fn execute_fused_matmul_bias_gelu() {
        // Build graph: x @ w + b → gelu
        let mut g = Graph::new("test");
        let x_id = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w_id = g.param("w", Shape::new(&[4, 3], DType::F32));
        let b_id = g.param("b", Shape::new(&[3], DType::F32));
        let mm = g.matmul(x_id, w_id, Shape::new(&[2, 3], DType::F32));
        let add = g.binary(BinaryOp::Add, mm, b_id, Shape::new(&[2, 3], DType::F32));
        let out = g.activation(Activation::Gelu, add, Shape::new(&[2, 3], DType::F32));
        g.set_outputs(vec![out]);

        // Fuse
        let fused = FuseMatMulBiasAct.run(g);
        println!("{fused}");

        // Plan memory
        let plan = memory::plan_memory(&fused);
        println!("Arena: {} bytes", plan.arena_size);

        // Prepare data
        let x_data = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // [2, 4] identity-ish
        let w_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]; // [4, 3]
        let b_data = vec![0.5, -0.5, 0.0]; // [3]

        let mut ext = ExternalBuffers {
            buffers: HashMap::new(),
        };
        ext.buffers.insert(fused.outputs[0], &[]); // placeholder
        // Find input/param node IDs in fused graph
        for node in fused.nodes() {
            match &node.op {
                Op::Input { name } if name == "x" => {
                    ext.buffers.insert(node.id, &x_data);
                }
                Op::Param { name } if name == "w" => {
                    ext.buffers.insert(node.id, &w_data);
                }
                Op::Param { name } if name == "b" => {
                    ext.buffers.insert(node.id, &b_data);
                }
                _ => {}
            }
        }

        // Execute
        let mut arena = Arena::from_plan(plan);
        execute(&fused, &mut arena, &ext);

        // Check output
        let output_id = fused.outputs[0];
        let result = arena.slice(output_id);
        println!("Result: {result:?}");

        // x @ w = [[1,0,0], [0,1,0]]; + bias = [[1.5,-0.5,0], [0.5,0.5,0]]
        // gelu(1.5) ≈ 1.399, gelu(-0.5) ≈ -0.154, gelu(0) = 0
        // gelu(0.5) ≈ 0.346
        assert!((result[0] - 1.399).abs() < 0.01, "got {}", result[0]);
        assert!((result[1] - -0.154).abs() < 0.01, "got {}", result[1]);
        assert!((result[2] - 0.0).abs() < 0.01, "got {}", result[2]);
        assert!((result[3] - 0.346).abs() < 0.01, "got {}", result[3]);
    }

    /// Test Gather (embedding lookup).
    #[test]
    fn execute_gather() {
        use rlx_ir::infer::GraphExt;
        let mut g = Graph::new("gather_test");
        // Embedding table [4, 3] and indices [2] → output [2, 3]
        let table = g.param("table", Shape::new(&[4, 3], DType::F32));
        let indices = g.input("ids", Shape::new(&[2], DType::F32)); // f32 indices
        let out = g.gather_(table, indices, 0);
        g.set_outputs(vec![out]);

        let plan = memory::plan_memory(&g);
        let mut arena = Arena::from_plan(plan);

        let table_data = vec![
            10.0, 11.0, 12.0, // row 0
            20.0, 21.0, 22.0, // row 1
            30.0, 31.0, 32.0, // row 2
            40.0, 41.0, 42.0, // row 3
        ];
        let ids_data = vec![2.0, 0.0]; // gather rows 2 and 0

        let mut ext = ExternalBuffers {
            buffers: HashMap::new(),
        };
        for node in g.nodes() {
            match &node.op {
                Op::Param { name } if name == "table" => {
                    ext.buffers.insert(node.id, &table_data);
                }
                Op::Input { name } if name == "ids" => {
                    ext.buffers.insert(node.id, &ids_data);
                }
                _ => {}
            }
        }

        execute(&g, &mut arena, &ext);
        let result = arena.slice(g.outputs[0]);
        assert_eq!(&result[..3], &[30.0, 31.0, 32.0]); // row 2
        assert_eq!(&result[3..6], &[10.0, 11.0, 12.0]); // row 0
    }

    /// Test Narrow (slice).
    #[test]
    fn execute_narrow() {
        use rlx_ir::infer::GraphExt;
        let mut g = Graph::new("narrow_test");
        let x = g.input("x", Shape::new(&[2, 6], DType::F32));
        let sliced = g.narrow_(x, 1, 2, 3); // take cols 2..5
        g.set_outputs(vec![sliced]);

        let plan = memory::plan_memory(&g);
        let mut arena = Arena::from_plan(plan);

        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let mut ext = ExternalBuffers {
            buffers: HashMap::new(),
        };
        for node in g.nodes() {
            if let Op::Input { .. } = &node.op {
                ext.buffers.insert(node.id, &data);
            }
        }

        execute(&g, &mut arena, &ext);
        let result = arena.slice(g.outputs[0]);
        assert_eq!(result, &[2.0, 3.0, 4.0, 8.0, 9.0, 10.0]);
    }

    /// Test Softmax.
    #[test]
    fn execute_softmax() {
        use rlx_ir::infer::GraphExt;
        let mut g = Graph::new("softmax_test");
        let x = g.input("x", Shape::new(&[1, 4], DType::F32));
        let sm = g.sm(x, -1);
        g.set_outputs(vec![sm]);

        let plan = memory::plan_memory(&g);
        let mut arena = Arena::from_plan(plan);

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut ext = ExternalBuffers {
            buffers: HashMap::new(),
        };
        for node in g.nodes() {
            if let Op::Input { .. } = &node.op {
                ext.buffers.insert(node.id, &data);
            }
        }

        execute(&g, &mut arena, &ext);
        let result = arena.slice(g.outputs[0]);
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax should sum to 1, got {sum}"
        );
        // Values should be monotonically increasing
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
        assert!(result[2] < result[3]);
    }

    /// Test RoPE (rotary position embedding).
    #[test]
    fn execute_rope() {
        use rlx_ir::infer::GraphExt;
        let head_dim = 4;
        let half = head_dim / 2;
        let seq = 2;

        let mut g = Graph::new("rope_test");
        // x: [seq, head_dim], cos: [seq, half], sin: [seq, half]
        let x = g.input("x", Shape::new(&[seq, head_dim], DType::F32));
        let cos = g.param("cos", Shape::new(&[seq, half], DType::F32));
        let sin = g.param("sin", Shape::new(&[seq, half], DType::F32));
        let rotated = g.rope(x, cos, sin, head_dim);
        g.set_outputs(vec![rotated]);

        let plan = memory::plan_memory(&g);
        let mut arena = Arena::from_plan(plan);

        // x = [[1, 0, 0, 1], [1, 1, 0, 0]] (2 positions, head_dim=4)
        let x_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0f32];
        // cos = [[1, 0], [0, 1]], sin = [[0, 1], [1, 0]] (identity-ish rotation)
        let cos_data = vec![1.0, 0.0, 0.0, 1.0f32];
        let sin_data = vec![0.0, 1.0, 1.0, 0.0f32];

        let mut ext = ExternalBuffers {
            buffers: HashMap::new(),
        };
        for node in g.nodes() {
            match &node.op {
                Op::Input { name } if name == "x" => {
                    ext.buffers.insert(node.id, &x_data);
                }
                Op::Param { name } if name == "cos" => {
                    ext.buffers.insert(node.id, &cos_data);
                }
                Op::Param { name } if name == "sin" => {
                    ext.buffers.insert(node.id, &sin_data);
                }
                _ => {}
            }
        }

        execute(&g, &mut arena, &ext);
        let result = arena.slice(g.outputs[0]);

        // Position 0: cos=[1,0], sin=[0,1]
        //   x1=1, x2=0 → x1*cos[0]-x2*sin[0] = 1*1-0*0 = 1
        //   x1=0, x2=1 → same half: x2*cos[0]+x1*sin[0] = 0*1+1*0 → wait
        // Actually: for i=0: x[0]=1, x[half+0]=0 → out[0]=1*1-0*0=1, out[2]=0*1+1*0=0
        //           for i=1: x[1]=0, x[half+1]=1 → out[1]=0*0-1*1=-1, out[3]=1*0+0*1=0
        assert!((result[0] - 1.0).abs() < 1e-5, "pos0[0]={}", result[0]);
        assert!((result[1] - -1.0).abs() < 1e-5, "pos0[1]={}", result[1]);
        assert!((result[2] - 0.0).abs() < 1e-5, "pos0[2]={}", result[2]);
        assert!((result[3] - 0.0).abs() < 1e-5, "pos0[3]={}", result[3]);

        // Position 1: cos=[0,1], sin=[1,0]
        //   x=[1,1,0,0]: for i=0: 1*0-0*1=-0=0, out[half+0]=0*0+1*1=1
        //                 for i=1: 1*1-0*0=1, out[half+1]=0*1+1*0=0
        assert!((result[4] - 0.0).abs() < 1e-5, "pos1[0]={}", result[4]);
        assert!((result[5] - 1.0).abs() < 1e-5, "pos1[1]={}", result[5]);
        assert!((result[6] - 1.0).abs() < 1e-5, "pos1[2]={}", result[6]);
        assert!((result[7] - 0.0).abs() < 1e-5, "pos1[3]={}", result[7]);
    }

    /// Test LayerNorm standalone.
    #[test]
    fn execute_layer_norm() {
        use rlx_ir::infer::GraphExt;
        let mut g = Graph::new("ln_test");
        let x = g.input("x", Shape::new(&[1, 4], DType::F32));
        let gamma = g.param("g", Shape::new(&[4], DType::F32));
        let beta = g.param("b", Shape::new(&[4], DType::F32));
        let ln = g.ln(x, gamma, beta, 1e-5);
        g.set_outputs(vec![ln]);

        let plan = memory::plan_memory(&g);
        let mut arena = Arena::from_plan(plan);

        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let g_data = vec![1.0, 1.0, 1.0, 1.0];
        let b_data = vec![0.0, 0.0, 0.0, 0.0];

        let mut ext = ExternalBuffers {
            buffers: HashMap::new(),
        };
        for node in g.nodes() {
            match &node.op {
                Op::Input { name } if name == "x" => {
                    ext.buffers.insert(node.id, &x_data);
                }
                Op::Param { name } if name == "g" => {
                    ext.buffers.insert(node.id, &g_data);
                }
                Op::Param { name } if name == "b" => {
                    ext.buffers.insert(node.id, &b_data);
                }
                _ => {}
            }
        }

        execute(&g, &mut arena, &ext);
        let result = arena.slice(g.outputs[0]);
        let sum: f32 = result.iter().sum();
        assert!(
            sum.abs() < 1e-3,
            "LN output should be zero-centered, sum={sum}"
        );
    }
}
