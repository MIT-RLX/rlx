#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

#[allow(unused_variables)]
pub(crate) fn compile_fused_mat_mul_bias_act(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::FusedMatMulBiasAct { activation } = &node.op else {
        unreachable!()
    };
    {
        let shape = &node.shape;
        let n = shape.dim(shape.rank() - 1).unwrap_static();
        let total = shape.num_elements().unwrap();
        let m = total / n;
        let a_len = get_len(graph, node.inputs[0]);
        let k = a_len / m;
        Thunk::FusedMmBiasAct {
            a: node_offset(arena, node.inputs[0]),
            w: node_offset(arena, node.inputs[1]),
            bias: node_offset(arena, node.inputs[2]),
            c: node_offset(arena, node.id),
            m: m as u32,
            k: k as u32,
            n: n as u32,
            act: *activation,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_mat_mul(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::MatMul = &node.op else { unreachable!() };
    {
        let shape = &node.shape;
        let a_shape = &graph.node(node.inputs[0]).shape;
        let b_shape = &graph.node(node.inputs[1]).shape;
        // Prefer inferred matmul shape from operands — ONNX bundle
        // meta often over-ranks outputs (e.g. [seq, seq, H]).
        let eff = rlx_ir::shape::matmul_shape(a_shape, b_shape).unwrap_or_else(|_| shape.clone());
        let rank = eff.rank().max(2);
        let n = eff.dim(rank - 1).unwrap_static();
        let k_dim = a_shape.dim(a_shape.rank().max(2) - 1).unwrap_static();
        if shape.dtype() == rlx_ir::DType::C64 {
            // Complex GEMM (interleaved re/im). Handles 2D and
            // 3D×2D (flatten M); both-operand batched C64 is not
            // yet wired.
            let both = a_shape.rank() >= 3 && b_shape.rank() >= 3;
            assert!(!both, "batched (both-operand) C64 matmul not yet supported");
            let m: usize = if a_shape.rank() >= 3 {
                (0..a_shape.rank() - 1)
                    .map(|d| a_shape.dim(d).unwrap_static())
                    .product()
            } else {
                a_shape.dim(a_shape.rank() - 2).unwrap_static()
            };
            Thunk::CgemmC64 {
                a: node_offset(arena, node.inputs[0]),
                b: node_offset(arena, node.inputs[1]),
                c: node_offset(arena, node.id),
                m: m as u32,
                k: k_dim as u32,
                n: n as u32,
            }
        } else {
            // Batched GEMM only when both operands carry batch dimensions.
            // 3D×2D (activations × shared weight) must flatten to one Sgemm.
            let both_batched = a_shape.rank() >= 3 && b_shape.rank() >= 3;
            let batched_3d = rank >= 3 && both_batched && a_shape.rank() + b_shape.rank() > 4;
            if batched_3d && shape.dtype() == rlx_ir::DType::F64 {
                let mut batch_prod = 1usize;
                for d in 0..rank - 2 {
                    batch_prod *= eff.dim(d).unwrap_static();
                }
                let m_dim = eff.dim(rank - 2).unwrap_static();
                Thunk::BatchedDgemmF64 {
                    a: node_offset(arena, node.inputs[0]),
                    b: node_offset(arena, node.inputs[1]),
                    c: node_offset(arena, node.id),
                    batch: batch_prod as u32,
                    m: m_dim as u32,
                    k: k_dim as u32,
                    n: n as u32,
                }
            } else if batched_3d && shape.dtype() == rlx_ir::DType::F32 {
                let mut batch_prod = 1usize;
                for d in 0..rank - 2 {
                    batch_prod *= eff.dim(d).unwrap_static();
                }
                let m_dim = eff.dim(rank - 2).unwrap_static();
                // Per-operand batch counts: an operand whose batch product is 1
                // (while the output batch is >1) is BROADCAST across the batch.
                let a_batch: usize = (0..a_shape.rank().saturating_sub(2))
                    .map(|d| a_shape.dim(d).unwrap_static())
                    .product();
                let b_batch: usize = (0..b_shape.rank().saturating_sub(2))
                    .map(|d| b_shape.dim(d).unwrap_static())
                    .product();
                Thunk::BatchedSgemm {
                    a: node_offset(arena, node.inputs[0]),
                    b: node_offset(arena, node.inputs[1]),
                    c: node_offset(arena, node.id),
                    batch: batch_prod as u32,
                    m: m_dim as u32,
                    k: k_dim as u32,
                    n: n as u32,
                    a_bcast: a_batch == 1 && batch_prod > 1,
                    b_bcast: b_batch == 1 && batch_prod > 1,
                }
            } else {
                let m = if a_shape.rank() >= 3 && b_shape.rank() <= 2 {
                    let mut m_prod = 1usize;
                    for d in 0..a_shape.rank() - 1 {
                        m_prod *= a_shape.dim(d).unwrap_static();
                    }
                    m_prod
                } else if a_shape.rank() >= 2 {
                    a_shape.dim(a_shape.rank() - 2).unwrap_static()
                } else {
                    eff.num_elements().unwrap_or(1) / n.max(1)
                };
                match shape.dtype() {
                    rlx_ir::DType::F64 => Thunk::Dgemm {
                        a: node_offset(arena, node.inputs[0]),
                        b: node_offset(arena, node.inputs[1]),
                        c: node_offset(arena, node.id),
                        m: m as u32,
                        k: k_dim as u32,
                        n: n as u32,
                    },
                    _ => {
                        if let Some(&(asrc, ta, bsrc, tb)) = matmul_fold.get(&node.id) {
                            // Folded Transpose→MatMul: read pre-transpose
                            // operands, do the transpose via cblas flags.
                            Thunk::SgemmT {
                                a: node_offset(arena, asrc),
                                b: node_offset(arena, bsrc),
                                c: node_offset(arena, node.id),
                                m: m as u32,
                                k: k_dim as u32,
                                n: n as u32,
                                ta,
                                tb,
                            }
                        } else {
                            Thunk::Sgemm {
                                a: node_offset(arena, node.inputs[0]),
                                b: node_offset(arena, node.inputs[1]),
                                c: node_offset(arena, node.id),
                                m: m as u32,
                                k: k_dim as u32,
                                n: n as u32,
                            }
                        }
                    }
                }
            }
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_lora_mat_mul(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::LoraMatMul { scale } = &node.op else {
        unreachable!()
    };
    {
        // x [m, k], w [k, n], a [k, r], b [r, n].
        let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
        let total = node.shape.num_elements().unwrap();
        let m = total / n.max(1);
        let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
        let k = x_total / m.max(1);
        let a_total = graph.node(node.inputs[2]).shape.num_elements().unwrap();
        let r = a_total / k.max(1);
        Thunk::LoraMatMul {
            x: node_offset(arena, node.inputs[0]),
            w: node_offset(arena, node.inputs[1]),
            a: node_offset(arena, node.inputs[2]),
            b: node_offset(arena, node.inputs[3]),
            dst: node_offset(arena, node.id),
            m: m as u32,
            k: k as u32,
            n: n as u32,
            r: r as u32,
            scale: *scale,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_grouped_mat_mul(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::GroupedMatMul = &node.op else {
        unreachable!()
    };
    {
        // Inputs: [input(M, K), weight(E, K, N), expert_idx(M)]
        let in_shape = &graph.node(node.inputs[0]).shape;
        let w_shape = &graph.node(node.inputs[1]).shape;
        let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
        let k_dim = in_shape.dim(in_shape.rank() - 1).unwrap_static();
        let num_experts = w_shape.dim(0).unwrap_static();
        let n = w_shape.dim(2).unwrap_static();
        Thunk::GroupedMatMul {
            input: node_offset(arena, node.inputs[0]),
            weight: node_offset(arena, node.inputs[1]),
            expert_idx: node_offset(arena, node.inputs[2]),
            dst: node_offset(arena, node.id),
            m: m as u32,
            k_dim: k_dim as u32,
            n: n as u32,
            num_experts: num_experts as u32,
        }
    }
}

#[inline(always)]
pub(crate) fn exec_fused_mm_bias_act(t: &Thunk, base: *mut u8) {
    let Thunk::FusedMmBiasAct {
        a,
        w,
        bias,
        c,
        m,
        k,
        n,
        act,
    } = t
    else {
        unreachable!()
    };
    {
        let (m, k, n) = (*m as usize, *k as usize, *n as usize);
        unsafe {
            let out = sl_mut(*c, base, m * n);
            let bias_v = sl(*bias, base, n);
            // Match torch `F.linear` / `addmm(bias, A, W)` association order.
            for row in 0..m {
                out[row * n..(row + 1) * n].copy_from_slice(bias_v);
            }
            crate::blas::sgemm_accumulate_auto(
                sl(*a, base, m * k),
                sl(*w, base, k * n),
                out,
                m,
                k,
                n,
            );
            if let Some(act) = act {
                apply_activation_inplace(out, *act)
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_bias_add(t: &Thunk, base: *mut u8) {
    let Thunk::BiasAdd {
        src,
        bias,
        dst,
        m,
        n,
    } = t
    else {
        unreachable!()
    };
    {
        let (m, n) = (*m as usize, *n as usize);
        let len = m * n;
        unsafe {
            let out = sl_mut(*dst, base, len);
            if *src != *dst {
                let src_ptr = base.add(*src) as *const f32;
                let dst_ptr = base.add(*dst) as *mut f32;
                if src_ptr != dst_ptr {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len);
                }
            }
            crate::blas::bias_add(out, sl(*bias, base, n), m, n);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_lora_mat_mul(t: &Thunk, base: *mut u8) {
    let Thunk::LoraMatMul {
        x,
        w,
        a,
        b,
        dst,
        m,
        k,
        n,
        r,
        scale,
    } = t
    else {
        unreachable!()
    };
    {
        let (m, k, n, r) = (*m as usize, *k as usize, *n as usize, *r as usize);
        unsafe {
            let xs = sl(*x, base, m * k);
            let ws = sl(*w, base, k * n);
            let a_s = sl(*a, base, k * r);
            let bs = sl(*b, base, r * n);
            let out = sl_mut(*dst, base, m * n);
            crate::blas::sgemm(xs, ws, out, m, k, n);
            let mut tmp = vec![0f32; m * r];
            crate::blas::sgemm(xs, a_s, &mut tmp, m, k, r);
            if *scale != 1.0 {
                for v in tmp.iter_mut() {
                    *v *= *scale;
                }
            }
            crate::blas::sgemm_accumulate(&tmp, bs, out, m, r, n);
        }
    }
}
