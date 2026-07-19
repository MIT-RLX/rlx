#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

/// Per-output-row element offset into a modulation tensor (`scale`/`shift`/
/// `gate`) that broadcasts against `x` (NumPy right-alignment). Covers the DiT
/// `[B,1,D]` modulation and any rank-N broadcast whose axes are each `1` or
/// equal to `x`'s. The last axis is the feature axis `h` and is never
/// broadcast; the returned offsets index the start of each `h`-length slice.
pub(crate) fn broadcast_row_offsets(x_dims: &[usize], mod_dims_in: &[usize], h: usize) -> Vec<u32> {
    let rank = x_dims.len();
    // Right-align (NumPy broadcast): left-pad the modulation dims with 1s.
    let pad = rank
        .checked_sub(mod_dims_in.len())
        .expect("modulation tensor has higher rank than x");
    let mut mod_dims = vec![1usize; rank];
    mod_dims[pad..].copy_from_slice(mod_dims_in);

    let k = rank.saturating_sub(1); // leading (row) axes; last axis is `h`
    let rows: usize = x_dims[..k].iter().product();
    // Contiguous element strides of the modulation buffer's leading axes.
    let mut mod_strides = vec![0usize; k];
    let mut acc = h;
    for j in (0..k).rev() {
        mod_strides[j] = acc;
        acc *= mod_dims[j];
    }
    let mut offs = Vec::with_capacity(rows);
    let mut idx = vec![0usize; k];
    for _ in 0..rows {
        let mut off = 0usize;
        for j in 0..k {
            if mod_dims[j] != 1 {
                off += idx[j] * mod_strides[j];
            }
        }
        offs.push(off as u32);
        // Row-major increment of the leading multi-index (last axis fastest).
        for j in (0..k).rev() {
            idx[j] += 1;
            if idx[j] < x_dims[j] {
                break;
            }
            idx[j] = 0;
        }
    }
    offs
}

fn static_dims(shape: &rlx_ir::shape::Shape) -> Vec<usize> {
    shape.dims().iter().map(|d| d.unwrap_static()).collect()
}

#[allow(unused_variables)]
pub(crate) fn compile_ada_layer_norm(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::AdaLayerNorm { norm, eps } = &node.op else {
        unreachable!()
    };
    {
        let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
        let total = node.shape.num_elements().unwrap();
        let x_dims = static_dims(&graph.node(node.inputs[0]).shape);
        let scale_shape = &graph.node(node.inputs[1]).shape;
        debug_assert_eq!(
            scale_shape.dims(),
            graph.node(node.inputs[2]).shape.dims(),
            "AdaLayerNorm: scale and shift must share a shape"
        );
        let mod_dims = static_dims(scale_shape);
        let mod_len = mod_dims.iter().product::<usize>();
        let mod_off = broadcast_row_offsets(&x_dims, &mod_dims, h);
        Thunk::AdaLayerNorm {
            src: node_offset(arena, node.inputs[0]),
            scale: node_offset(arena, node.inputs[1]),
            shift: node_offset(arena, node.inputs[2]),
            dst: node_offset(arena, node.id),
            rows: (total / h) as u32,
            h: h as u32,
            eps: *eps,
            layer_norm: matches!(norm, rlx_ir::op::AdaNormKind::LayerNorm),
            mod_len: mod_len as u32,
            mod_off,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_gated_residual(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    debug_assert!(matches!(node.op, Op::GatedResidual));
    let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
    let total = node.shape.num_elements().unwrap();
    let x_dims = static_dims(&graph.node(node.inputs[0]).shape);
    let gate_dims = static_dims(&graph.node(node.inputs[2]).shape);
    let gate_len = gate_dims.iter().product::<usize>();
    let gate_off = broadcast_row_offsets(&x_dims, &gate_dims, h);
    Thunk::GatedResidual {
        x: node_offset(arena, node.inputs[0]),
        y: node_offset(arena, node.inputs[1]),
        gate: node_offset(arena, node.inputs[2]),
        dst: node_offset(arena, node.id),
        rows: (total / h) as u32,
        h: h as u32,
        gate_len: gate_len as u32,
        gate_off,
    }
}

pub(crate) fn compile_ada_layer_norm_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    _matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    _rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    _rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::AdaLayerNormBackward { norm, eps } = &node.op else {
        unreachable!()
    };
    let x_shape = &graph.node(node.inputs[0]).shape;
    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static();
    let nx = x_shape.num_elements().unwrap();
    let x_dims = static_dims(x_shape);
    let mod_dims = static_dims(&graph.node(node.inputs[1]).shape);
    let mod_len = mod_dims.iter().product::<usize>();
    let mod_off = broadcast_row_offsets(&x_dims, &mod_dims, h);
    Thunk::AdaLayerNormBackward {
        x: node_offset(arena, node.inputs[0]),
        scale: node_offset(arena, node.inputs[1]),
        shift: node_offset(arena, node.inputs[2]),
        dy: node_offset(arena, node.inputs[3]),
        dst: node_offset(arena, node.id),
        rows: (nx / h) as u32,
        h: h as u32,
        eps: *eps,
        layer_norm: matches!(norm, rlx_ir::op::AdaNormKind::LayerNorm),
        mod_len: mod_len as u32,
        mod_off,
    }
}

pub(crate) fn compile_gated_residual_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    _matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    _rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    _rng: rlx_ir::RngOptions,
) -> Thunk {
    debug_assert!(matches!(node.op, Op::GatedResidualBackward));
    let x_shape = &graph.node(node.inputs[0]).shape;
    let h = x_shape.dim(x_shape.rank() - 1).unwrap_static();
    let nx = x_shape.num_elements().unwrap();
    let x_dims = static_dims(x_shape);
    let gate_dims = static_dims(&graph.node(node.inputs[2]).shape);
    let gate_len = gate_dims.iter().product::<usize>();
    let gate_off = broadcast_row_offsets(&x_dims, &gate_dims, h);
    Thunk::GatedResidualBackward {
        x: node_offset(arena, node.inputs[0]),
        y: node_offset(arena, node.inputs[1]),
        gate: node_offset(arena, node.inputs[2]),
        dy: node_offset(arena, node.inputs[3]),
        dst: node_offset(arena, node.id),
        rows: (nx / h) as u32,
        h: h as u32,
        gate_len: gate_len as u32,
        gate_off,
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_fused_residual_l_n(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::FusedResidualLN { has_bias, eps } = &node.op else {
        unreachable!()
    };
    {
        let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
        let total = node.shape.num_elements().unwrap();
        let rows = total / h;
        let (g_idx, b_idx) = if *has_bias { (3, 4) } else { (2, 3) };
        Thunk::FusedResidualLN {
            x: node_offset(arena, node.inputs[0]),
            res: node_offset(arena, node.inputs[1]),
            bias: if *has_bias {
                node_offset(arena, node.inputs[2])
            } else {
                0
            },
            g: node_offset(arena, node.inputs[g_idx]),
            b: node_offset(arena, node.inputs[b_idx]),
            out: node_offset(arena, node.id),
            rows: rows as u32,
            h: h as u32,
            eps: *eps,
            has_bias: *has_bias,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_fused_residual_rms_norm(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::FusedResidualRmsNorm { has_bias, eps } = &node.op else {
        unreachable!()
    };
    {
        let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
        let total = node.shape.num_elements().unwrap();
        let rows = total / h;
        let (g_idx, b_idx) = if *has_bias { (3, 4) } else { (2, 3) };
        Thunk::FusedResidualRmsNorm {
            x: node_offset(arena, node.inputs[0]),
            res: node_offset(arena, node.inputs[1]),
            bias: if *has_bias {
                node_offset(arena, node.inputs[2])
            } else {
                0
            },
            g: node_offset(arena, node.inputs[g_idx]),
            b: node_offset(arena, node.inputs[b_idx]),
            out: node_offset(arena, node.id),
            rows: rows as u32,
            h: h as u32,
            eps: *eps,
            has_bias: *has_bias,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_rms_norm(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::RmsNorm { eps, .. } = &node.op else {
        unreachable!()
    };
    {
        let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
        let total = node.shape.num_elements().unwrap();
        Thunk::RmsNorm {
            src: node_offset(arena, node.inputs[0]),
            g: node_offset(arena, node.inputs[1]),
            b: node_offset(arena, node.inputs[2]),
            dst: node_offset(arena, node.id),
            rows: (total / h) as u32,
            h: h as u32,
            eps: *eps,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_layer_norm(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::LayerNorm { eps, .. } = &node.op else {
        unreachable!()
    };
    {
        let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
        let total = node.shape.num_elements().unwrap();
        Thunk::LayerNorm {
            src: node_offset(arena, node.inputs[0]),
            g: node_offset(arena, node.inputs[1]),
            b: node_offset(arena, node.inputs[2]),
            dst: node_offset(arena, node.id),
            rows: (total / h) as u32,
            h: h as u32,
            eps: *eps,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_group_norm(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::GroupNorm { num_groups, eps } = &node.op else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let (n, c, h, w) = conv_nchw_dims(in_shape);
        Thunk::GroupNorm {
            src: node_offset(arena, node.inputs[0]),
            g: node_offset(arena, node.inputs[1]),
            b: node_offset(arena, node.inputs[2]),
            dst: node_offset(arena, node.id),
            n,
            c,
            h,
            w,
            num_groups: *num_groups as u32,
            eps: *eps,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_batch_norm_inference(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::BatchNormInference { eps } = &node.op else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let rank = in_shape.rank();
        let channels = in_shape.dim(rank - 1).unwrap_static();
        let total = in_shape.num_elements().unwrap_or(0);
        let count = (total / channels.max(1)) as u32;
        Thunk::BatchNormInference {
            src: node_offset(arena, node.inputs[0]),
            g: node_offset(arena, node.inputs[1]),
            b: node_offset(arena, node.inputs[2]),
            mean: node_offset(arena, node.inputs[3]),
            var: node_offset(arena, node.inputs[4]),
            dst: node_offset(arena, node.id),
            count,
            channels: channels as u32,
            eps: *eps,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_batch_norm_inference_backward_input(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::BatchNormInferenceBackwardInput { eps } = &node.op else {
        unreachable!()
    };
    {
        let x_shape = &graph.node(node.inputs[0]).shape;
        let rank = x_shape.rank();
        let channels = x_shape.dim(rank - 1).unwrap_static();
        let total = x_shape.num_elements().unwrap_or(0);
        Thunk::BatchNormInferenceBackwardInput {
            x: node_offset(arena, node.inputs[0]),
            gamma: node_offset(arena, node.inputs[1]),
            mean: node_offset(arena, node.inputs[2]),
            var: node_offset(arena, node.inputs[3]),
            dy: node_offset(arena, node.inputs[4]),
            dx: node_offset(arena, node.id),
            count: (total / channels.max(1)) as u32,
            channels: channels as u32,
            eps: *eps,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_batch_norm_inference_backward_gamma(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::BatchNormInferenceBackwardGamma { eps } = &node.op else {
        unreachable!()
    };
    {
        let x_shape = &graph.node(node.inputs[0]).shape;
        let rank = x_shape.rank();
        let channels = x_shape.dim(rank - 1).unwrap_static();
        let total = x_shape.num_elements().unwrap_or(0);
        let _gamma_shape = &graph.node(node.id).shape;
        Thunk::BatchNormInferenceBackwardGamma {
            x: node_offset(arena, node.inputs[0]),
            mean: node_offset(arena, node.inputs[1]),
            var: node_offset(arena, node.inputs[2]),
            dy: node_offset(arena, node.inputs[3]),
            dgamma: node_offset(arena, node.id),
            count: (total / channels.max(1)) as u32,
            channels: channels as u32,
            eps: *eps,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_batch_norm_inference_backward_beta(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::BatchNormInferenceBackwardBeta = &node.op else {
        unreachable!()
    };
    {
        let dy_shape = &graph.node(node.inputs[0]).shape;
        let rank = dy_shape.rank();
        let channels = dy_shape.dim(rank - 1).unwrap_static();
        let total = dy_shape.num_elements().unwrap_or(0);
        Thunk::BatchNormInferenceBackwardBeta {
            dy: node_offset(arena, node.inputs[0]),
            dbeta: node_offset(arena, node.id),
            count: (total / channels.max(1)) as u32,
            channels: channels as u32,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_layer_norm2d(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::LayerNorm2d { eps } = &node.op else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let (n, c, h, w) = conv_nchw_dims(in_shape);
        Thunk::LayerNorm2d {
            src: node_offset(arena, node.inputs[0]),
            g: node_offset(arena, node.inputs[1]),
            b: node_offset(arena, node.inputs[2]),
            dst: node_offset(arena, node.id),
            n,
            c,
            h,
            w,
            eps: *eps,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_layer_norm_backward_input(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::LayerNormBackwardInput { eps, .. } = &node.op else {
        unreachable!()
    };
    {
        // axis = -1 only (matches forward LayerNorm thunk).
        let h = node.shape.dim(node.shape.rank() - 1).unwrap_static();
        let total = node.shape.num_elements().unwrap();
        Thunk::LayerNormBackwardInput {
            x: node_offset(arena, node.inputs[0]),
            gamma: node_offset(arena, node.inputs[1]),
            dy: node_offset(arena, node.inputs[2]),
            dx: node_offset(arena, node.id),
            rows: (total / h) as u32,
            h: h as u32,
            eps: *eps,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_layer_norm_backward_gamma(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::LayerNormBackwardGamma { eps, .. } = &node.op else {
        unreachable!()
    };
    {
        let x_shape = &graph.node(node.inputs[0]).shape;
        let h = x_shape.dim(x_shape.rank() - 1).unwrap_static();
        let x_total = x_shape.num_elements().unwrap();
        Thunk::LayerNormBackwardGamma {
            x: node_offset(arena, node.inputs[0]),
            dy: node_offset(arena, node.inputs[1]),
            dgamma: node_offset(arena, node.id),
            rows: (x_total / h) as u32,
            h: h as u32,
            eps: *eps,
        }
    }
}

#[inline(always)]
pub(crate) fn exec_layer_norm(t: &Thunk, base: *mut u8) {
    let Thunk::LayerNorm {
        src,
        g,
        b,
        dst,
        rows,
        h,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let (rows, h) = (*rows as usize, *h as usize);
        unsafe {
            let input = sl(*src, base, rows * h);
            let gamma = sl(*g, base, h);
            let beta = sl(*b, base, h);
            let output = sl_mut(*dst, base, rows * h);
            // Parallelize across rows (same pattern as FusedResidualLN)
            if rows >= 4 && rows * h >= 30_000 {
                let i_ptr = input.as_ptr() as usize;
                let o_ptr = output.as_mut_ptr() as usize;
                let g_ptr = gamma.as_ptr() as usize;
                let b_ptr = beta.as_ptr() as usize;
                let e = *eps;
                crate::pool::par_for(rows, 4, &|off, cnt| {
                    let inp =
                        std::slice::from_raw_parts((i_ptr as *const f32).add(off * h), cnt * h);
                    let out =
                        std::slice::from_raw_parts_mut((o_ptr as *mut f32).add(off * h), cnt * h);
                    let g = std::slice::from_raw_parts(g_ptr as *const f32, h);
                    let b = std::slice::from_raw_parts(b_ptr as *const f32, h);
                    for row in 0..cnt {
                        crate::kernels::layer_norm_row(
                            &inp[row * h..(row + 1) * h],
                            g,
                            b,
                            &mut out[row * h..(row + 1) * h],
                            h,
                            e,
                        );
                    }
                });
            } else {
                for row in 0..rows {
                    crate::kernels::layer_norm_row(
                        &input[row * h..(row + 1) * h],
                        gamma,
                        beta,
                        &mut output[row * h..(row + 1) * h],
                        h,
                        *eps,
                    );
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_group_norm(t: &Thunk, base: *mut u8) {
    let Thunk::GroupNorm {
        src,
        g,
        b,
        dst,
        n,
        c,
        h,
        w,
        num_groups,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let (n, c, h, w) = (*n as usize, *c as usize, *h as usize, *w as usize);
        let plane = c * h * w;
        unsafe {
            // Per-batch plane stride is `plane` *f32 elements*; `base`
            // is a byte pointer, so advance by `plane * size_of::<f32>()`.
            // (Was `base.add(ni * plane)` — a 4× under-advance that read
            // batch row 1+ from the wrong offset; only n=1 was tested.)
            let stride = plane * std::mem::size_of::<f32>();
            for ni in 0..n {
                let input = sl(*src, base.add(ni * stride), plane);
                let gamma = sl(*g, base, c);
                let beta = sl(*b, base, c);
                let output = sl_mut(*dst, base.add(ni * stride), plane);
                crate::kernels::group_norm_nchw(
                    input,
                    gamma,
                    beta,
                    output,
                    1,
                    c,
                    h,
                    w,
                    *num_groups as usize,
                    *eps,
                );
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_batch_norm_inference(t: &Thunk, base: *mut u8) {
    let Thunk::BatchNormInference {
        src,
        g,
        b,
        mean,
        var,
        dst,
        count,
        channels,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let count = *count as usize;
        let c = *channels as usize;
        let n = count * c;
        unsafe {
            crate::kernels::batch_norm_inference(
                sl(*src, base, n),
                sl(*g, base, c),
                sl(*b, base, c),
                sl(*mean, base, c),
                sl(*var, base, c),
                sl_mut(*dst, base, n),
                c,
                *eps,
            );
        }
    }
}

#[inline(always)]
pub(crate) fn exec_layer_norm2d(t: &Thunk, base: *mut u8) {
    let Thunk::LayerNorm2d {
        src,
        g,
        b,
        dst,
        n,
        c,
        h,
        w,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let (n, c, h, w) = (*n as usize, *c as usize, *h as usize, *w as usize);
        let plane = c * h * w;
        unsafe {
            let input = sl(*src, base, n * plane);
            let gamma = sl(*g, base, c);
            let beta = sl(*b, base, c);
            let output = sl_mut(*dst, base, n * plane);
            crate::kernels::layer_norm2d_nchw(input, gamma, beta, output, n, c, h, w, *eps);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_rms_norm(t: &Thunk, base: *mut u8) {
    let Thunk::RmsNorm {
        src,
        g,
        b,
        dst,
        rows,
        h,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let (rows, h) = (*rows as usize, *h as usize);
        unsafe {
            let input = sl(*src, base, rows * h);
            let gamma = sl(*g, base, h);
            let beta = sl(*b, base, h);
            let output = sl_mut(*dst, base, rows * h);
            let inv_h = 1.0 / h as f32;
            for row in 0..rows {
                let in_row = &input[row * h..(row + 1) * h];
                let out_row = &mut output[row * h..(row + 1) * h];
                // RMS = sqrt(mean(x^2) + eps); scale = 1/RMS.
                let mut sumsq = 0f32;
                for &v in in_row {
                    sumsq += v * v;
                }
                let inv_rms = (sumsq * inv_h + *eps).sqrt().recip();
                for i in 0..h {
                    out_row[i] = in_row[i] * inv_rms * gamma[i] + beta[i];
                }
            }
        }
    }
}

/// One row of DiT adaLN-Zero modulation: `out = norm(in)·(1+scale)+shift`.
/// `layer_norm` selects mean+variance normalization vs. RMS-only.
#[inline(always)]
fn ada_modulate_row(
    in_row: &[f32],
    scale: &[f32],
    shift: &[f32],
    out_row: &mut [f32],
    h: usize,
    eps: f32,
    layer_norm: bool,
) {
    let inv_h = 1.0 / h as f32;
    let (mean, inv) = if layer_norm {
        let mut sum = 0f32;
        let mut sumsq = 0f32;
        for &v in in_row {
            sum += v;
            sumsq += v * v;
        }
        let mean = sum * inv_h;
        let var = (sumsq * inv_h - mean * mean).max(0.0);
        (mean, (var + eps).sqrt().recip())
    } else {
        let mut sumsq = 0f32;
        for &v in in_row {
            sumsq += v * v;
        }
        (0.0, (sumsq * inv_h + eps).sqrt().recip())
    };
    for i in 0..h {
        let n = (in_row[i] - mean) * inv;
        out_row[i] = n * (1.0 + scale[i]) + shift[i];
    }
}

/// DiT adaLN-Zero modulation `out = norm(x)·(1+scale)+shift` (row-parallel).
#[inline(always)]
pub(crate) fn exec_ada_layer_norm(t: &Thunk, base: *mut u8) {
    let Thunk::AdaLayerNorm {
        src,
        scale,
        shift,
        dst,
        rows,
        h,
        eps,
        layer_norm,
        mod_len,
        mod_off,
    } = t
    else {
        unreachable!()
    };
    {
        let (nrows, h) = (*rows as usize, *h as usize);
        let eps = *eps;
        let ln = *layer_norm;
        let mlen = *mod_len as usize;
        unsafe {
            let input = sl(*src, base, nrows * h);
            let output = sl_mut(*dst, base, nrows * h);
            let scale_s = sl(*scale, base, mlen);
            let shift_s = sl(*shift, base, mlen);
            if nrows >= 4 && nrows * h >= 30_000 {
                let i_ptr = input.as_ptr() as usize;
                let o_ptr = output.as_mut_ptr() as usize;
                let sc_ptr = scale_s.as_ptr() as usize;
                let sh_ptr = shift_s.as_ptr() as usize;
                let mo_ptr = mod_off.as_ptr() as usize;
                crate::pool::par_for(nrows, 4, &|off, cnt| {
                    let inp =
                        std::slice::from_raw_parts((i_ptr as *const f32).add(off * h), cnt * h);
                    let out =
                        std::slice::from_raw_parts_mut((o_ptr as *mut f32).add(off * h), cnt * h);
                    let sc_all = std::slice::from_raw_parts(sc_ptr as *const f32, mlen);
                    let sh_all = std::slice::from_raw_parts(sh_ptr as *const f32, mlen);
                    let mo = std::slice::from_raw_parts(mo_ptr as *const u32, nrows);
                    for row in 0..cnt {
                        let mo_i = mo[off + row] as usize;
                        ada_modulate_row(
                            &inp[row * h..(row + 1) * h],
                            &sc_all[mo_i..mo_i + h],
                            &sh_all[mo_i..mo_i + h],
                            &mut out[row * h..(row + 1) * h],
                            h,
                            eps,
                            ln,
                        );
                    }
                });
            } else {
                for row in 0..nrows {
                    let mo_i = mod_off[row] as usize;
                    ada_modulate_row(
                        &input[row * h..(row + 1) * h],
                        &scale_s[mo_i..mo_i + h],
                        &shift_s[mo_i..mo_i + h],
                        &mut output[row * h..(row + 1) * h],
                        h,
                        eps,
                        ln,
                    );
                }
            }
        }
    }
}

/// DiT gated residual `out = x + gate·y` (row-parallel, broadcast gate).
#[inline(always)]
pub(crate) fn exec_gated_residual(t: &Thunk, base: *mut u8) {
    let Thunk::GatedResidual {
        x,
        y,
        gate,
        dst,
        rows,
        h,
        gate_len,
        gate_off,
    } = t
    else {
        unreachable!()
    };
    {
        let (nrows, h) = (*rows as usize, *h as usize);
        let glen = *gate_len as usize;
        unsafe {
            let xs = sl(*x, base, nrows * h);
            let ys = sl(*y, base, nrows * h);
            let gs = sl(*gate, base, glen);
            let output = sl_mut(*dst, base, nrows * h);
            let row = |xr: &[f32], yr: &[f32], gr: &[f32], out: &mut [f32]| {
                for i in 0..h {
                    out[i] = xr[i] + gr[i] * yr[i];
                }
            };
            if nrows >= 4 && nrows * h >= 30_000 {
                let x_ptr = xs.as_ptr() as usize;
                let y_ptr = ys.as_ptr() as usize;
                let g_ptr = gs.as_ptr() as usize;
                let o_ptr = output.as_mut_ptr() as usize;
                let go_ptr = gate_off.as_ptr() as usize;
                crate::pool::par_for(nrows, 4, &|off, cnt| {
                    let xr =
                        std::slice::from_raw_parts((x_ptr as *const f32).add(off * h), cnt * h);
                    let yr =
                        std::slice::from_raw_parts((y_ptr as *const f32).add(off * h), cnt * h);
                    let out =
                        std::slice::from_raw_parts_mut((o_ptr as *mut f32).add(off * h), cnt * h);
                    let g_all = std::slice::from_raw_parts(g_ptr as *const f32, glen);
                    let go = std::slice::from_raw_parts(go_ptr as *const u32, nrows);
                    for r in 0..cnt {
                        let gi = go[off + r] as usize;
                        row(
                            &xr[r * h..(r + 1) * h],
                            &yr[r * h..(r + 1) * h],
                            &g_all[gi..gi + h],
                            &mut out[r * h..(r + 1) * h],
                        );
                    }
                });
            } else {
                for r in 0..nrows {
                    let gi = gate_off[r] as usize;
                    row(
                        &xs[r * h..(r + 1) * h],
                        &ys[r * h..(r + 1) * h],
                        &gs[gi..gi + h],
                        &mut output[r * h..(r + 1) * h],
                    );
                }
            }
        }
    }
}

/// Packed reverse of AdaLayerNorm → `[dx ∥ dscale ∥ dshift]` (1-D).
#[inline(always)]
pub(crate) fn exec_ada_layer_norm_backward(t: &Thunk, base: *mut u8) {
    let Thunk::AdaLayerNormBackward {
        x,
        scale,
        shift: _,
        dy,
        dst,
        rows,
        h,
        eps,
        layer_norm,
        mod_len,
        mod_off,
    } = t
    else {
        unreachable!()
    };
    let (nrows, h) = (*rows as usize, *h as usize);
    let eps = *eps;
    let ln = *layer_norm;
    let mlen = *mod_len as usize;
    let nx = nrows * h;
    unsafe {
        let xs = sl(*x, base, nx);
        let sc = sl(*scale, base, mlen);
        let dys = sl(*dy, base, nx);
        let out = sl_mut(*dst, base, nx + 2 * mlen);
        let (dx, rest) = out.split_at_mut(nx);
        let (dscale, dshift) = rest.split_at_mut(mlen);
        dscale.fill(0.0);
        dshift.fill(0.0);
        let n_inv = 1.0 / h as f32;
        for r in 0..nrows {
            let mo = mod_off[r] as usize;
            let xr = &xs[r * h..(r + 1) * h];
            let dyr = &dys[r * h..(r + 1) * h];
            let sr = &sc[mo..mo + h];
            let dxr = &mut dx[r * h..(r + 1) * h];

            let (mean, inv) = if ln {
                let mut sum = 0.0f32;
                let mut sumsq = 0.0f32;
                for &v in xr {
                    sum += v;
                    sumsq += v * v;
                }
                let mean = sum * n_inv;
                let var = (sumsq * n_inv - mean * mean).max(0.0);
                (mean, (var + eps).sqrt().recip())
            } else {
                let mut sumsq = 0.0f32;
                for &v in xr {
                    sumsq += v * v;
                }
                (0.0, (sumsq * n_inv + eps).sqrt().recip())
            };

            // n, dn = dy·(1+scale); accumulate dscale/dshift.
            let mut dn = vec![0.0f32; h];
            let mut s_sy = 0.0f32;
            let mut s_sxh = 0.0f32;
            for d in 0..h {
                let n = (xr[d] - mean) * inv;
                let sy = dyr[d] * (1.0 + sr[d]);
                dn[d] = sy;
                dscale[mo + d] += dyr[d] * n;
                dshift[mo + d] += dyr[d];
                if ln {
                    s_sy += sy;
                    s_sxh += sy * n;
                } else {
                    // RMS: closed form with γ=1 uses mean(dy·x̂) only via
                    // training_bwd helper path — inline γ=1 RMS adjoint:
                    s_sxh += sy * n;
                }
            }
            if ln {
                let m_sy = s_sy * n_inv;
                let m_sxh = s_sxh * n_inv;
                for d in 0..h {
                    let n = (xr[d] - mean) * inv;
                    dxr[d] = inv * (dn[d] - m_sy - n * m_sxh);
                }
            } else {
                // RMSNorm γ=1, β=0: dx = inv · (dn - x̂ · mean(dn · x̂))
                let m_sxh = s_sxh * n_inv;
                for d in 0..h {
                    let n = xr[d] * inv;
                    dxr[d] = inv * (dn[d] - n * m_sxh);
                }
            }
        }
    }
}

/// Packed reverse of GatedResidual → `[dx ∥ dy ∥ dgate]` (1-D).
#[inline(always)]
pub(crate) fn exec_gated_residual_backward(t: &Thunk, base: *mut u8) {
    let Thunk::GatedResidualBackward {
        x: _,
        y,
        gate,
        dy,
        dst,
        rows,
        h,
        gate_len,
        gate_off,
    } = t
    else {
        unreachable!()
    };
    let (nrows, h) = (*rows as usize, *h as usize);
    let glen = *gate_len as usize;
    let nx = nrows * h;
    unsafe {
        let ys = sl(*y, base, nx);
        let gs = sl(*gate, base, glen);
        let dys = sl(*dy, base, nx);
        let out = sl_mut(*dst, base, nx + nx + glen);
        let (dx, rest) = out.split_at_mut(nx);
        let (dy_out, dgate) = rest.split_at_mut(nx);
        dgate.fill(0.0);
        for r in 0..nrows {
            let gi = gate_off[r] as usize;
            let yr = &ys[r * h..(r + 1) * h];
            let dyr = &dys[r * h..(r + 1) * h];
            let gr = &gs[gi..gi + h];
            for d in 0..h {
                dx[r * h + d] = dyr[d];
                dy_out[r * h + d] = dyr[d] * gr[d];
                dgate[gi + d] += dyr[d] * yr[d];
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_layer_norm_backward_input(t: &Thunk, base: *mut u8) {
    let Thunk::LayerNormBackwardInput {
        x,
        gamma,
        dy,
        dx,
        rows,
        h,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let rows = *rows as usize;
        let h = *h as usize;
        let eps = *eps;
        unsafe {
            let xs = sl(*x, base, rows * h);
            let g = sl(*gamma, base, h);
            let dys = sl(*dy, base, rows * h);
            let out = sl_mut(*dx, base, rows * h);
            let n_inv = 1.0 / h as f32;
            for r in 0..rows {
                let xr = &xs[r * h..(r + 1) * h];
                let dyr = &dys[r * h..(r + 1) * h];
                // Per-row mean and inv_std (recompute — no saved
                // tensor from the forward pass).
                let mut sum = 0f32;
                for &v in xr {
                    sum += v;
                }
                let mean = sum * n_inv;
                let mut var = 0f32;
                for &v in xr {
                    let d = v - mean;
                    var += d * d;
                }
                let inv_std = 1.0 / (var * n_inv + eps).sqrt();

                // sums needed for the closed-form:
                //   mean(dy·γ) and mean(dy·γ·x̂)
                let mut s_sy = 0f32;
                let mut s_sxh = 0f32;
                for d in 0..h {
                    let xh = (xr[d] - mean) * inv_std;
                    let sy = dyr[d] * g[d];
                    s_sy += sy;
                    s_sxh += sy * xh;
                }
                let m_sy = s_sy * n_inv;
                let m_sxh = s_sxh * n_inv;

                for d in 0..h {
                    let xh = (xr[d] - mean) * inv_std;
                    let sy = dyr[d] * g[d];
                    out[r * h + d] = inv_std * (sy - m_sy - xh * m_sxh);
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_batch_norm_inference_backward_input(t: &Thunk, base: *mut u8) {
    let Thunk::BatchNormInferenceBackwardInput {
        x,
        gamma,
        mean,
        var,
        dy,
        dx,
        count,
        channels,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let count = *count as usize;
        let c = *channels as usize;
        let n = count * c;
        let eps = *eps;
        unsafe {
            crate::kernels::batch_norm_inference_backward_input(
                sl(*x, base, n),
                sl(*gamma, base, c),
                sl(*mean, base, c),
                sl(*var, base, c),
                sl(*dy, base, n),
                sl_mut(*dx, base, n),
                c,
                eps,
            );
        }
    }
}

#[inline(always)]
pub(crate) fn exec_batch_norm_inference_backward_gamma(t: &Thunk, base: *mut u8) {
    let Thunk::BatchNormInferenceBackwardGamma {
        x,
        mean,
        var,
        dy,
        dgamma,
        count,
        channels,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let count = *count as usize;
        let c = *channels as usize;
        let n = count * c;
        let eps = *eps;
        unsafe {
            crate::kernels::batch_norm_inference_backward_gamma(
                sl(*x, base, n),
                sl(*mean, base, c),
                sl(*var, base, c),
                sl(*dy, base, n),
                sl_mut(*dgamma, base, c),
                c,
                eps,
            );
        }
    }
}

#[inline(always)]
pub(crate) fn exec_batch_norm_inference_backward_beta(t: &Thunk, base: *mut u8) {
    let Thunk::BatchNormInferenceBackwardBeta {
        dy,
        dbeta,
        count,
        channels,
    } = t
    else {
        unreachable!()
    };
    {
        let count = *count as usize;
        let c = *channels as usize;
        let n = count * c;
        unsafe {
            crate::kernels::batch_norm_inference_backward_beta(
                sl(*dy, base, n),
                sl_mut(*dbeta, base, c),
                c,
            );
        }
    }
}

#[inline(always)]
pub(crate) fn exec_layer_norm_backward_gamma(t: &Thunk, base: *mut u8) {
    let Thunk::LayerNormBackwardGamma {
        x,
        dy,
        dgamma,
        rows,
        h,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let rows = *rows as usize;
        let h = *h as usize;
        let eps = *eps;
        unsafe {
            let xs = sl(*x, base, rows * h);
            let dys = sl(*dy, base, rows * h);
            let out = sl_mut(*dgamma, base, h);
            for v in out.iter_mut() {
                *v = 0.0;
            }
            let n_inv = 1.0 / h as f32;
            for r in 0..rows {
                let xr = &xs[r * h..(r + 1) * h];
                let dyr = &dys[r * h..(r + 1) * h];
                let mut sum = 0f32;
                for &v in xr {
                    sum += v;
                }
                let mean = sum * n_inv;
                let mut var = 0f32;
                for &v in xr {
                    let d = v - mean;
                    var += d * d;
                }
                let inv_std = 1.0 / (var * n_inv + eps).sqrt();
                for d in 0..h {
                    let xh = (xr[d] - mean) * inv_std;
                    out[d] += dyr[d] * xh;
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_rms_norm_backward_input(t: &Thunk, base: *mut u8) {
    let Thunk::RmsNormBackwardInput {
        x,
        gamma,
        beta,
        dy,
        dx,
        rows,
        h,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let (rows, h) = (*rows as usize, *h as usize);
        unsafe {
            let xs = sl(*x, base, rows * h);
            let g = sl(*gamma, base, h);
            let b = sl(*beta, base, h);
            let dys = sl(*dy, base, rows * h);
            let out = sl_mut(*dx, base, rows * h);
            let mut dg = vec![0f32; h];
            let mut db = vec![0f32; h];
            for r in 0..rows {
                crate::training_bwd::rms_norm_backward_row(
                    &xs[r * h..(r + 1) * h],
                    g,
                    b,
                    &dys[r * h..(r + 1) * h],
                    &mut out[r * h..(r + 1) * h],
                    &mut dg,
                    &mut db,
                    *eps,
                );
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_rms_norm_backward_gamma(t: &Thunk, base: *mut u8) {
    let Thunk::RmsNormBackwardGamma {
        x,
        gamma,
        beta,
        dy,
        dgamma,
        rows,
        h,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let (rows, h) = (*rows as usize, *h as usize);
        unsafe {
            let xs = sl(*x, base, rows * h);
            let g = sl(*gamma, base, h);
            let b = sl(*beta, base, h);
            let dys = sl(*dy, base, rows * h);
            let out = sl_mut(*dgamma, base, h);
            for v in out.iter_mut() {
                *v = 0.0;
            }
            let mut dx = vec![0f32; h];
            let mut db = vec![0f32; h];
            for r in 0..rows {
                crate::training_bwd::rms_norm_backward_row(
                    &xs[r * h..(r + 1) * h],
                    g,
                    b,
                    &dys[r * h..(r + 1) * h],
                    &mut dx,
                    &mut *out,
                    &mut db,
                    *eps,
                );
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_rms_norm_backward_beta(t: &Thunk, base: *mut u8) {
    let Thunk::RmsNormBackwardBeta {
        x,
        gamma,
        beta,
        dy,
        dbeta,
        rows,
        h,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let (rows, h) = (*rows as usize, *h as usize);
        unsafe {
            let xs = sl(*x, base, rows * h);
            let g = sl(*gamma, base, h);
            let b = sl(*beta, base, h);
            let dys = sl(*dy, base, rows * h);
            let out = sl_mut(*dbeta, base, h);
            for v in out.iter_mut() {
                *v = 0.0;
            }
            let mut dx = vec![0f32; h];
            let mut dg = vec![0f32; h];
            for r in 0..rows {
                crate::training_bwd::rms_norm_backward_row(
                    &xs[r * h..(r + 1) * h],
                    g,
                    b,
                    &dys[r * h..(r + 1) * h],
                    &mut dx,
                    &mut dg,
                    &mut *out,
                    *eps,
                );
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_group_norm_backward_input(t: &Thunk, base: *mut u8) {
    let Thunk::GroupNormBackwardInput {
        x,
        gamma,
        beta: _beta,
        dy,
        dx,
        n,
        c,
        h,
        w,
        num_groups,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let (n, c, h, w) = (*n as usize, *c as usize, *h as usize, *w as usize);
        let plane = c * h * w;
        unsafe {
            let xs = sl(*x, base, n * plane);
            let g = sl(*gamma, base, c);
            let dys = sl(*dy, base, n * plane);
            let out = sl_mut(*dx, base, n * plane);
            crate::training_bwd::group_norm_backward_input_nchw(
                xs,
                g,
                dys,
                out,
                n,
                c,
                h,
                w,
                *num_groups as usize,
                *eps,
            );
        }
    }
}

#[inline(always)]
pub(crate) fn exec_group_norm_backward_gamma(t: &Thunk, base: *mut u8) {
    let Thunk::GroupNormBackwardGamma {
        x,
        dy,
        dgamma,
        n,
        c,
        h,
        w,
        num_groups,
        eps,
    } = t
    else {
        unreachable!()
    };
    {
        let (n, c, h, w) = (*n as usize, *c as usize, *h as usize, *w as usize);
        let plane = c * h * w;
        unsafe {
            let xs = sl(*x, base, n * plane);
            let dys = sl(*dy, base, n * plane);
            let out = sl_mut(*dgamma, base, c);
            crate::training_bwd::group_norm_backward_gamma_nchw(
                xs,
                dys,
                out,
                n,
                c,
                h,
                w,
                *num_groups as usize,
                *eps,
            );
        }
    }
}

#[inline(always)]
pub(crate) fn exec_group_norm_backward_beta(t: &Thunk, base: *mut u8) {
    let Thunk::GroupNormBackwardBeta {
        dy,
        dbeta,
        n,
        c,
        h,
        w,
    } = t
    else {
        unreachable!()
    };
    {
        let (n, c, h, w) = (*n as usize, *c as usize, *h as usize, *w as usize);
        let plane = c * h * w;
        unsafe {
            let dys = sl(*dy, base, n * plane);
            let out = sl_mut(*dbeta, base, c);
            crate::training_bwd::group_norm_backward_beta_nchw(dys, out, n, c, h, w);
        }
    }
}

/// Host-fallback: `Op::RmsNormBackwardInput` (GPU unified-memory / D2H arenas).
pub unsafe fn execute_rms_norm_backward_input_f32(
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dx: usize,
    rows: u32,
    h: u32,
    eps: f32,
    base: *mut u8,
) {
    let (rows, h) = (rows as usize, h as usize);
    let mut dg = vec![0f32; h];
    let mut db = vec![0f32; h];
    let xs = sl(x, base, rows * h);
    let dys = sl(dy, base, rows * h);
    let g = sl(gamma, base, h);
    let b = sl(beta, base, h);
    let out = sl_mut(dx, base, rows * h);
    for r in 0..rows {
        crate::training_bwd::rms_norm_backward_row(
            &xs[r * h..(r + 1) * h],
            g,
            b,
            &dys[r * h..(r + 1) * h],
            &mut out[r * h..(r + 1) * h],
            &mut dg,
            &mut db,
            eps,
        );
    }
}

pub unsafe fn execute_rms_norm_backward_gamma_f32(
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dgamma: usize,
    rows: u32,
    h: u32,
    eps: f32,
    base: *mut u8,
) {
    let (rows, h) = (rows as usize, h as usize);
    let out = sl_mut(dgamma, base, h);
    out.fill(0.0);
    let mut dx = vec![0f32; h];
    let mut db = vec![0f32; h];
    let xs = sl(x, base, rows * h);
    let dys = sl(dy, base, rows * h);
    let g = sl(gamma, base, h);
    let b = sl(beta, base, h);
    for r in 0..rows {
        crate::training_bwd::rms_norm_backward_row(
            &xs[r * h..(r + 1) * h],
            g,
            b,
            &dys[r * h..(r + 1) * h],
            &mut dx,
            out,
            &mut db,
            eps,
        );
    }
}

pub unsafe fn execute_rms_norm_backward_beta_f32(
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dbeta: usize,
    rows: u32,
    h: u32,
    eps: f32,
    base: *mut u8,
) {
    let (rows, h) = (rows as usize, h as usize);
    let out = sl_mut(dbeta, base, h);
    out.fill(0.0);
    let mut dx = vec![0f32; h];
    let mut dg = vec![0f32; h];
    let xs = sl(x, base, rows * h);
    let dys = sl(dy, base, rows * h);
    let g = sl(gamma, base, h);
    let b = sl(beta, base, h);
    for r in 0..rows {
        crate::training_bwd::rms_norm_backward_row(
            &xs[r * h..(r + 1) * h],
            g,
            b,
            &dys[r * h..(r + 1) * h],
            &mut dx,
            &mut dg,
            out,
            eps,
        );
    }
}

/// Host fallback for NCHW group norm (Metal unified-memory arena).
pub unsafe fn execute_group_norm_nchw_f32(
    src: usize,
    g: usize,
    b: usize,
    dst: usize,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    num_groups: usize,
    eps: f32,
    base: *mut u8,
) {
    let plane = c * h * w;
    for ni in 0..n {
        let input = unsafe { sl(src + ni * plane * std::mem::size_of::<f32>(), base, plane) };
        let gamma = unsafe { sl(g, base, c) };
        let beta = unsafe { sl(b, base, c) };
        let output = unsafe { sl_mut(dst + ni * plane * std::mem::size_of::<f32>(), base, plane) };
        crate::kernels::group_norm_nchw(input, gamma, beta, output, 1, c, h, w, num_groups, eps);
    }
}

/// Host fallback for NCHW LayerNorm2d (SAM / candle semantics).
pub unsafe fn execute_layer_norm2d_nchw_f32(
    src: usize,
    g: usize,
    b: usize,
    dst: usize,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    eps: f32,
    base: *mut u8,
) {
    let plane = c * h * w;
    unsafe {
        let input = sl(src, base, n * plane);
        let gamma = sl(g, base, c);
        let beta = sl(b, base, c);
        let output = sl_mut(dst, base, n * plane);
        crate::kernels::layer_norm2d_nchw(input, gamma, beta, output, n, c, h, w, eps);
    }
}
