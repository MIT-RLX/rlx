#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

/// NCL `[N,C,L]` or NCHW `[N,C,H,W]` → `(n, c, h, w)` for 2D conv/norm thunks.
pub(crate) fn conv_nchw_dims(shape: &Shape) -> (u32, u32, u32, u32) {
    match shape.rank() {
        3 => (
            shape.dim(0).unwrap_static() as u32,
            shape.dim(1).unwrap_static() as u32,
            1,
            shape.dim(2).unwrap_static() as u32,
        ),
        4 => (
            shape.dim(0).unwrap_static() as u32,
            shape.dim(1).unwrap_static() as u32,
            shape.dim(2).unwrap_static() as u32,
            shape.dim(3).unwrap_static() as u32,
        ),
        r => panic!(
            "conv_nchw_dims: expected rank 3 or 4, got {r} dims={:?}",
            shape.dims()
        ),
    }
}

/// (N, C, D, H, W) for a rank-5 NCDHW conv tensor.
pub(crate) fn conv_ncdhw_dims(shape: &Shape) -> (u32, u32, u32, u32, u32) {
    assert_eq!(
        shape.rank(),
        5,
        "conv_ncdhw_dims: expected rank 5, got {}",
        shape.rank()
    );
    (
        shape.dim(0).unwrap_static() as u32,
        shape.dim(1).unwrap_static() as u32,
        shape.dim(2).unwrap_static() as u32,
        shape.dim(3).unwrap_static() as u32,
        shape.dim(4).unwrap_static() as u32,
    )
}

#[allow(unused_variables)]
pub(crate) fn compile_conv_transpose2d(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ConvTranspose2d {
        kernel_size,
        stride,
        padding,
        dilation,
        output_padding: _,
        groups,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let out_shape = &node.shape;
        let (n, c_in, h, w_in) = conv_nchw_dims(in_shape);
        let (_, c_out, h_out, w_out) = conv_nchw_dims(out_shape);
        Thunk::ConvTranspose2d {
            src: node_offset(arena, node.inputs[0]),
            weight: node_offset(arena, node.inputs[1]),
            dst: node_offset(arena, node.id),
            n,
            c_in,
            h,
            w_in,
            c_out,
            h_out,
            w_out,
            kh: kernel_size[0] as u32,
            kw: kernel_size[1] as u32,
            sh: stride.first().copied().unwrap_or(1) as u32,
            sw: stride.get(1).copied().unwrap_or(1) as u32,
            ph: padding.first().copied().unwrap_or(0) as u32,
            pw: padding.get(1).copied().unwrap_or(0) as u32,
            dh: dilation.first().copied().unwrap_or(1) as u32,
            dw: dilation.get(1).copied().unwrap_or(1) as u32,
            groups: *groups as u32,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_resize_nearest2x(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ResizeNearest2x = &node.op else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let (n, c, h, w) = conv_nchw_dims(in_shape);
        Thunk::ResizeNearest2x {
            src: node_offset(arena, node.inputs[0]),
            dst: node_offset(arena, node.id),
            n,
            c,
            h,
            w,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_conv(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Conv {
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let w_shape = &graph.node(node.inputs[1]).shape;
        let out_shape = &node.shape;
        // 1×1 fast path (plan #26): kH=kW=1, stride=1,
        // padding=0, dilation=1, groups=1. Emits a single
        // Conv2D1x1 thunk that BLAS-dispatches per batch.
        let is_1x1_simple = kernel_size.len() == 2
            && kernel_size[0] == 1
            && kernel_size[1] == 1
            && stride.iter().all(|&s| s == 1)
            && padding.iter().all(|&p| p == 0)
            && dilation.iter().all(|&d| d == 1)
            && *groups == 1;
        if is_1x1_simple && in_shape.rank() >= 3 && out_shape.rank() >= 3 && w_shape.rank() >= 2 {
            let (n, c_in, h, w) = conv_nchw_dims(in_shape);
            let (_, c_out, _, _) = conv_nchw_dims(out_shape);
            Thunk::Conv2D1x1 {
                src: node_offset(arena, node.inputs[0]),
                weight: node_offset(arena, node.inputs[1]),
                dst: node_offset(arena, node.id),
                n,
                c_in,
                c_out,
                hw: h.saturating_mul(w),
            }
        } else if kernel_size.len() == 2
            && in_shape.rank() >= 3
            && w_shape.rank() >= 2
            && out_shape.rank() >= 3
        {
            let (n, c_in, h, w_in) = conv_nchw_dims(in_shape);
            let (_, c_out, h_out, w_out) = conv_nchw_dims(out_shape);
            // rlx lowers ONNX 1D convs as 2D NCHW with a unit H axis and the
            // length in W (`[N,C,1,L]`), but keeps the length kernel/stride/pad/
            // dilation at index 0 (`kernel=[k,1]`). A literal 2D conv would run
            // the k-tap kernel over the singleton H axis and ignore the length.
            // Since `[N,C,1,L]` and `[N,C,L,1]` share the same row-major layout,
            // relabel the length onto the H axis (no data copy) so the kernel
            // convolves it — matching the MLX 1D path and onnxruntime.
            let one_d_w = h == 1
                && w_in > 1
                && kernel_size[0] > 1
                && kernel_size.get(1).copied().unwrap_or(1) == 1;
            let (h, w_in, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw) = if one_d_w {
                (
                    w_in,
                    1,
                    w_out,
                    1,
                    kernel_size[0] as u32,
                    1,
                    stride.first().copied().unwrap_or(1) as u32,
                    1,
                    padding.first().copied().unwrap_or(0) as u32,
                    0,
                    dilation.first().copied().unwrap_or(1) as u32,
                    1,
                )
            } else {
                (
                    h,
                    w_in,
                    h_out,
                    w_out,
                    kernel_size[0] as u32,
                    kernel_size[1] as u32,
                    stride.first().copied().unwrap_or(1) as u32,
                    stride.get(1).copied().unwrap_or(1) as u32,
                    padding.first().copied().unwrap_or(0) as u32,
                    padding.get(1).copied().unwrap_or(0) as u32,
                    dilation.first().copied().unwrap_or(1) as u32,
                    dilation.get(1).copied().unwrap_or(1) as u32,
                )
            };
            Thunk::Conv2D {
                src: node_offset(arena, node.inputs[0]),
                weight: node_offset(arena, node.inputs[1]),
                dst: node_offset(arena, node.id),
                n,
                c_in,
                h,
                w: w_in,
                c_out,
                h_out,
                w_out,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
                groups: *groups as u32,
            }
        } else {
            Thunk::Nop
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_conv3d(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Conv3d {
        stride,
        padding,
        dilation,
        groups,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let w_shape = &graph.node(node.inputs[1]).shape;
        let out_shape = &node.shape;
        if in_shape.rank() == 5 && w_shape.rank() == 5 && out_shape.rank() == 5 {
            let (n, c_in, d, h, w) = conv_ncdhw_dims(in_shape);
            let (_, c_out, d_out, h_out, w_out) = conv_ncdhw_dims(out_shape);
            // Kernel extent from the weight [C_out, C_in/g, kD, kH, kW].
            let kd = w_shape.dim(2).unwrap_static() as u32;
            let kh = w_shape.dim(3).unwrap_static() as u32;
            let kw = w_shape.dim(4).unwrap_static() as u32;
            Thunk::Conv3d {
                src: node_offset(arena, node.inputs[0]),
                weight: node_offset(arena, node.inputs[1]),
                dst: node_offset(arena, node.id),
                n,
                c_in,
                d,
                h,
                w,
                c_out,
                d_out,
                h_out,
                w_out,
                kd,
                kh,
                kw,
                sd: stride[0] as u32,
                sh: stride[1] as u32,
                sw: stride[2] as u32,
                pd: padding[0] as u32,
                ph: padding[1] as u32,
                pw: padding[2] as u32,
                dd: dilation[0] as u32,
                dh: dilation[1] as u32,
                dw: dilation[2] as u32,
                groups: *groups as u32,
            }
        } else {
            Thunk::Nop
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_conv_transpose3d(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ConvTranspose3d {
        stride,
        padding,
        dilation,
        output_padding: _,
        groups,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let w_shape = &graph.node(node.inputs[1]).shape;
        let out_shape = &node.shape;
        if in_shape.rank() == 5 && w_shape.rank() == 5 && out_shape.rank() == 5 {
            let (n, c_in, d, h, w_in) = conv_ncdhw_dims(in_shape);
            let (_, c_out, d_out, h_out, w_out) = conv_ncdhw_dims(out_shape);
            // Kernel extent from the weight [C_in, C_out/g, kD, kH, kW].
            let kd = w_shape.dim(2).unwrap_static() as u32;
            let kh = w_shape.dim(3).unwrap_static() as u32;
            let kw = w_shape.dim(4).unwrap_static() as u32;
            Thunk::ConvTranspose3d {
                src: node_offset(arena, node.inputs[0]),
                weight: node_offset(arena, node.inputs[1]),
                dst: node_offset(arena, node.id),
                n,
                c_in,
                d,
                h,
                w_in,
                c_out,
                d_out,
                h_out,
                w_out,
                kd,
                kh,
                kw,
                sd: stride[0] as u32,
                sh: stride[1] as u32,
                sw: stride[2] as u32,
                pd: padding[0] as u32,
                ph: padding[1] as u32,
                pw: padding[2] as u32,
                dd: dilation[0] as u32,
                dh: dilation[1] as u32,
                dw: dilation[2] as u32,
                groups: *groups as u32,
            }
        } else {
            Thunk::Nop
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_pool(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Pool {
        kind,
        kernel_size,
        stride,
        padding,
    } = &node.op
    else {
        unreachable!()
    };
    {
        // Currently support 2D pooling on rank-4 NCHW tensors.
        let in_shape = &graph.node(node.inputs[0]).shape;
        let out_shape = &node.shape;
        if kernel_size.len() == 2 && in_shape.rank() == 4 && out_shape.rank() == 4 {
            Thunk::Pool2D {
                src: node_offset(arena, node.inputs[0]),
                dst: node_offset(arena, node.id),
                n: in_shape.dim(0).unwrap_static() as u32,
                c: in_shape.dim(1).unwrap_static() as u32,
                h: in_shape.dim(2).unwrap_static() as u32,
                w: in_shape.dim(3).unwrap_static() as u32,
                h_out: out_shape.dim(2).unwrap_static() as u32,
                w_out: out_shape.dim(3).unwrap_static() as u32,
                kh: kernel_size[0] as u32,
                kw: kernel_size[1] as u32,
                sh: stride.first().copied().unwrap_or(1) as u32,
                sw: stride.get(1).copied().unwrap_or(1) as u32,
                ph: padding.first().copied().unwrap_or(0) as u32,
                pw: padding.get(1).copied().unwrap_or(0) as u32,
                kind: *kind,
            }
        } else {
            Thunk::Nop
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_max_pool2d_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::MaxPool2dBackward {
        kernel_size,
        stride,
        padding,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let x_shape = &graph.node(node.inputs[0]).shape;
        let dy_shape = &graph.node(node.inputs[1]).shape;
        if kernel_size.len() == 2 && x_shape.rank() == 4 && dy_shape.rank() == 4 {
            Thunk::MaxPool2dBackward {
                x: node_offset(arena, node.inputs[0]),
                dy: node_offset(arena, node.inputs[1]),
                dx: node_offset(arena, node.id),
                n: x_shape.dim(0).unwrap_static() as u32,
                c: x_shape.dim(1).unwrap_static() as u32,
                h: x_shape.dim(2).unwrap_static() as u32,
                w: x_shape.dim(3).unwrap_static() as u32,
                h_out: dy_shape.dim(2).unwrap_static() as u32,
                w_out: dy_shape.dim(3).unwrap_static() as u32,
                kh: kernel_size[0] as u32,
                kw: kernel_size[1] as u32,
                sh: stride.first().copied().unwrap_or(1) as u32,
                sw: stride.get(1).copied().unwrap_or(1) as u32,
                ph: padding.first().copied().unwrap_or(0) as u32,
                pw: padding.get(1).copied().unwrap_or(0) as u32,
            }
        } else {
            Thunk::Nop
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_conv2d_backward_input(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Conv2dBackwardInput {
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let dy_shape = &graph.node(node.inputs[0]).shape;
        let w_shape = &graph.node(node.inputs[1]).shape;
        let out_shape = &node.shape;
        if kernel_size.len() == 2
            && dy_shape.rank() == 4
            && w_shape.rank() == 4
            && out_shape.rank() == 4
        {
            Thunk::Conv2dBackwardInput {
                dy: node_offset(arena, node.inputs[0]),
                w: node_offset(arena, node.inputs[1]),
                dx: node_offset(arena, node.id),
                n: out_shape.dim(0).unwrap_static() as u32,
                c_in: out_shape.dim(1).unwrap_static() as u32,
                h: out_shape.dim(2).unwrap_static() as u32,
                w_in: out_shape.dim(3).unwrap_static() as u32,
                c_out: dy_shape.dim(1).unwrap_static() as u32,
                h_out: dy_shape.dim(2).unwrap_static() as u32,
                w_out: dy_shape.dim(3).unwrap_static() as u32,
                kh: kernel_size[0] as u32,
                kw: kernel_size[1] as u32,
                sh: stride.first().copied().unwrap_or(1) as u32,
                sw: stride.get(1).copied().unwrap_or(1) as u32,
                ph: padding.first().copied().unwrap_or(0) as u32,
                pw: padding.get(1).copied().unwrap_or(0) as u32,
                dh: dilation.first().copied().unwrap_or(1) as u32,
                dw: dilation.get(1).copied().unwrap_or(1) as u32,
                groups: *groups as u32,
            }
        } else {
            Thunk::Nop
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_conv2d_backward_weight(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Conv2dBackwardWeight {
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let x_shape = &graph.node(node.inputs[0]).shape;
        let dy_shape = &graph.node(node.inputs[1]).shape;
        let dw_shape = &node.shape;
        if kernel_size.len() == 2
            && x_shape.rank() == 4
            && dy_shape.rank() == 4
            && dw_shape.rank() == 4
        {
            Thunk::Conv2dBackwardWeight {
                x: node_offset(arena, node.inputs[0]),
                dy: node_offset(arena, node.inputs[1]),
                dw: node_offset(arena, node.id),
                n: x_shape.dim(0).unwrap_static() as u32,
                c_in: x_shape.dim(1).unwrap_static() as u32,
                h: x_shape.dim(2).unwrap_static() as u32,
                w: x_shape.dim(3).unwrap_static() as u32,
                c_out: dy_shape.dim(1).unwrap_static() as u32,
                h_out: dy_shape.dim(2).unwrap_static() as u32,
                w_out: dy_shape.dim(3).unwrap_static() as u32,
                kh: kernel_size[0] as u32,
                kw: kernel_size[1] as u32,
                sh: stride.first().copied().unwrap_or(1) as u32,
                sw: stride.get(1).copied().unwrap_or(1) as u32,
                ph: padding.first().copied().unwrap_or(0) as u32,
                pw: padding.get(1).copied().unwrap_or(0) as u32,
                dh: dilation.first().copied().unwrap_or(1) as u32,
                dw_dil: dilation.get(1).copied().unwrap_or(1) as u32,
                groups: *groups as u32,
            }
        } else {
            Thunk::Nop
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_im2_col(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Im2Col {
        kernel_size,
        stride,
        padding,
        dilation,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let x_shape = &graph.node(node.inputs[0]).shape;
        let out_shape = &node.shape;
        if kernel_size.len() == 2 && x_shape.rank() == 4 && out_shape.rank() == 2 {
            let n = match x_shape.dim(0) {
                rlx_ir::shape::Dim::Static(v) => v as u32,
                _ => 0,
            };
            let c_in = x_shape.dim(1).unwrap_static() as u32;
            let h = x_shape.dim(2).unwrap_static() as u32;
            let w = x_shape.dim(3).unwrap_static() as u32;
            let kh = kernel_size[0] as u32;
            let kw = kernel_size[1] as u32;
            let sh = stride.first().copied().unwrap_or(1) as u32;
            let sw = stride.get(1).copied().unwrap_or(1) as u32;
            let ph = padding.first().copied().unwrap_or(0) as u32;
            let pw = padding.get(1).copied().unwrap_or(0) as u32;
            let dh = dilation.first().copied().unwrap_or(1) as u32;
            let dw_dil = dilation.get(1).copied().unwrap_or(1) as u32;
            let h_out = rlx_ir::shape::conv2d_spatial_output(
                h as usize,
                kh as usize,
                sh as usize,
                ph as usize,
                dh as usize,
            ) as u32;
            let w_out = rlx_ir::shape::conv2d_spatial_output(
                w as usize,
                kw as usize,
                sw as usize,
                pw as usize,
                dw_dil as usize,
            ) as u32;
            Thunk::Im2Col {
                x: node_offset(arena, node.inputs[0]),
                col: node_offset(arena, node.id),
                n,
                c_in,
                h,
                w,
                h_out,
                w_out,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw_dil,
            }
        } else {
            Thunk::Nop
        }
    }
}

#[inline(always)]
pub(crate) fn exec_conv_transpose2d(t: &Thunk, base: *mut u8) {
    let Thunk::ConvTranspose2d {
        src,
        weight,
        dst,
        n,
        c_in,
        h,
        w_in,
        c_out,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        groups,
    } = t
    else {
        unreachable!()
    };
    {
        let n = *n as usize;
        let c_in = *c_in as usize;
        let h = *h as usize;
        let w_in = *w_in as usize;
        let c_out = *c_out as usize;
        let h_out = *h_out as usize;
        let w_out = *w_out as usize;
        unsafe {
            let inp = sl(*src, base, n * c_in * h * w_in);
            let wt = sl(
                *weight,
                base,
                c_in * (c_out / *groups as usize) * (*kh as usize) * (*kw as usize),
            );
            let out = sl_mut(*dst, base, n * c_out * h_out * w_out);
            crate::kernels::conv_transpose2d_nchw(
                inp,
                wt,
                out,
                n,
                c_in,
                h,
                w_in,
                c_out,
                h_out,
                w_out,
                *kh as usize,
                *kw as usize,
                *sh as usize,
                *sw as usize,
                *ph as usize,
                *pw as usize,
                *dh as usize,
                *dw as usize,
                *groups as usize,
            );
        }
    }
}

pub(crate) fn exec_conv3d(t: &Thunk, base: *mut u8) {
    let Thunk::Conv3d {
        src,
        weight,
        dst,
        n,
        c_in,
        d,
        h,
        w,
        c_out,
        d_out,
        h_out,
        w_out,
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
        groups,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_conv3d_forward_f32(
            *src, *weight, *dst, *n, *c_in, *d, *h, *w, *c_out, *d_out, *h_out, *w_out, *kd, *kh,
            *kw, *sd, *sh, *sw, *pd, *ph, *pw, *dd, *dh, *dw, *groups, base,
        );
    }
}

pub(crate) fn exec_conv_transpose3d(t: &Thunk, base: *mut u8) {
    let Thunk::ConvTranspose3d {
        src,
        weight,
        dst,
        n,
        c_in,
        d,
        h,
        w_in,
        c_out,
        d_out,
        h_out,
        w_out,
        kd,
        kh,
        kw,
        sd,
        sh,
        sw,
        pd,
        ph,
        pw,
        dd,
        dh,
        dw,
        groups,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_conv_transpose3d_ncdhw_f32(
            *src, *weight, *dst, *n, *c_in, *d, *h, *w_in, *c_out, *d_out, *h_out, *w_out, *kd,
            *kh, *kw, *sd, *sh, *sw, *pd, *ph, *pw, *dd, *dh, *dw, *groups, base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_resize_nearest2x(t: &Thunk, base: *mut u8) {
    let Thunk::ResizeNearest2x {
        src,
        dst,
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
        let in_plane = c * h * w;
        let out_plane = c * h * 2 * w * 2;
        // `base` is a byte pointer; batch planes are f32 elements, so
        // advance by `plane * size_of::<f32>()`. (Was `base.add(ni *
        // plane)` — a 4× under-advance; only n=1 was ever tested.)
        let fsz = std::mem::size_of::<f32>();
        unsafe {
            for ni in 0..n {
                let input = sl(*src, base.add(ni * in_plane * fsz), in_plane);
                let output = sl_mut(*dst, base.add(ni * out_plane * fsz), out_plane);
                crate::kernels::resize_nearest_2x_nchw(input, output, c, h, w);
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_conv2_d1x1(t: &Thunk, base: *mut u8) {
    let Thunk::Conv2D1x1 {
        src,
        weight,
        dst,
        n,
        c_in,
        c_out,
        hw,
    } = t
    else {
        unreachable!()
    };
    {
        let n = *n as usize;
        let c_in = *c_in as usize;
        let c_out = *c_out as usize;
        let hw = *hw as usize;
        unsafe {
            let inp = sl(*src, base, n * c_in * hw);
            let wt = sl(*weight, base, c_out * c_in);
            let out = sl_mut(*dst, base, n * c_out * hw);
            // Per-batch sgemm: weight [c_out, c_in] @ input
            // [c_in, hw] = output [c_out, hw]. The weight is
            // shared across batches, so we get to dispatch
            // BLAS once per N (typically 1).
            for ni in 0..n {
                let in_off = ni * c_in * hw;
                let out_off = ni * c_out * hw;
                crate::blas::sgemm(
                    wt,
                    &inp[in_off..in_off + c_in * hw],
                    &mut out[out_off..out_off + c_out * hw],
                    c_out,
                    c_in,
                    hw,
                );
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_conv2_d(t: &Thunk, base: *mut u8) {
    let Thunk::Conv2D {
        src,
        weight,
        dst,
        n,
        c_in,
        h,
        w,
        c_out,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        groups,
    } = t
    else {
        unreachable!()
    };
    {
        let n = *n as usize;
        let c_in = *c_in as usize;
        let h = *h as usize;
        let w = *w as usize;
        let c_out = *c_out as usize;
        let h_out = *h_out as usize;
        let w_out = *w_out as usize;
        let kh = *kh as usize;
        let kw = *kw as usize;
        let sh = *sh as usize;
        let sw = *sw as usize;
        let ph = *ph as usize;
        let pw = *pw as usize;
        let dh = *dh as usize;
        let dw = *dw as usize;
        let groups = *groups as usize;
        let c_in_per_g = c_in / groups;
        unsafe {
            let inp = sl(*src, base, n * c_in * h * w);
            let wt = sl(*weight, base, c_out * c_in_per_g * kh * kw);
            let out = sl_mut(*dst, base, n * c_out * h_out * w_out);
            // Forward conv kernels (im2col+BLAS by default via `fast_conv_enabled`):
            //   * im2col + BLAS sgemm — measured CPU optimum for TinyConv /
            //     MNIST CNN shapes; mirrors Conv2D1x1 / Conv2dBackwardWeight.
            //   * Winograd / direct — opt-in (`RLX_WINOGRAD` / `RLX_DIRECT_CONV`);
            //     win only at higher channel counts.
            //   * reference scalar nested loop — `RLX_FAST_CONV=0` or when the
            //     im2col scratch would be huge (degenerate ISTFT-as-conv sizes).
            // Eligibility for Winograd/direct: stride-1, no pad, dilation-1.
            let s1_nopad = sh == 1 && sw == 1 && ph == 0 && pw == 0 && dh == 1 && dw == 1;
            let winograd_ok = s1_nopad && kh == 3 && kw == 3 && groups == 1;
            // im2col materializes a `[c_in_per_g·kh·kw, h_out·w_out]` patch matrix.
            // For a large-kernel / large-output conv — e.g. an ISTFT-as-conv vocoder
            // head (kernel 1024, decomposed stride-256 ConvTranspose → forward conv
            // over a 36k-sample length) that col buffer is c_in·1024·36k ≈ 38 billion
            // f32 (150+ GB) and OOMs the box, while the naive direct kernel needs
            // ZERO scratch. Guard the fast path on the col size so those degenerate
            // shapes take the (constant-memory) naive kernel instead.
            const IM2COL_MAX_COL_ELEMS: usize = 512 * 1024 * 1024; // 2 GB f32 scratch
            let col_elems = (c_in_per_g * kh * kw).saturating_mul(h_out * w_out);
            if fast_conv_enabled() && winograd_enabled() && winograd_ok {
                conv2d_forward_winograd(inp, wt, out, n, c_in, h, w, c_out, h_out, w_out);
            } else if fast_conv_enabled() && direct_conv_enabled() && s1_nopad {
                conv2d_forward_direct(
                    inp, wt, out, n, c_in, h, w, c_out, h_out, w_out, kh, kw, groups,
                );
            } else if fast_conv_enabled() && col_elems <= IM2COL_MAX_COL_ELEMS {
                conv2d_forward_im2col(
                    inp, wt, out, n, c_in, h, w, c_out, h_out, w_out, kh, kw, sh, sw, ph, pw, dh,
                    dw, groups,
                );
            } else {
                conv2d_forward_naive(
                    inp, wt, out, n, c_in, h, w, c_out, h_out, w_out, kh, kw, sh, sw, ph, pw, dh,
                    dw, groups,
                );
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_max_pool2d_backward(t: &Thunk, base: *mut u8) {
    let Thunk::MaxPool2dBackward {
        x,
        dy,
        dx,
        n,
        c,
        h,
        w,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_maxpool2d_backward_f32(
            *x, *dy, *dx, *n, *c, *h, *w, *h_out, *w_out, *kh, *kw, *sh, *sw, *ph, *pw, base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_conv2d_backward_input(t: &Thunk, base: *mut u8) {
    let Thunk::Conv2dBackwardInput {
        dy,
        w,
        dx,
        n,
        c_in,
        h,
        w_in,
        c_out,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        groups,
    } = t
    else {
        unreachable!()
    };
    {
        // Per-group GEMM + col2im. Two orders of magnitude faster
        // than the naive 6-deep nested loop on training shapes.
        //
        //   dcol_n_g = w_g^T  @  dy_n_g            (sgemm)
        //   dx_n_g  += col2im(dcol_n_g)            (scatter-add)
        //
        // Layouts (all row-major):
        //   w_g       [c_out_per_g, c_in_per_g · kh · kw]
        //   dy_n_g    [c_out_per_g, h_out · w_out]
        //   dcol_n_g  [c_in_per_g · kh · kw, h_out · w_out]
        //   dx_n_g    [c_in_per_g, h · w_in]
        let n = *n as usize;
        let c_in = *c_in as usize;
        let h = *h as usize;
        let w_in = *w_in as usize;
        let c_out = *c_out as usize;
        let h_out = *h_out as usize;
        let w_out = *w_out as usize;
        let kh = *kh as usize;
        let kw = *kw as usize;
        let sh = *sh as usize;
        let sw = *sw as usize;
        let ph = *ph as usize;
        let pw = *pw as usize;
        let dh = *dh as usize;
        let dw = *dw as usize;
        let groups = *groups as usize;
        let c_in_per_g = c_in / groups;
        let c_out_per_g = c_out / groups;

        let m_dim = c_in_per_g * kh * kw;
        let n_dim = h_out * w_out;
        let k_dim = c_out_per_g;

        let dy_stride_n = c_out * h_out * w_out;
        let dy_stride_g = c_out_per_g * h_out * w_out;
        let w_stride_g = c_out_per_g * c_in_per_g * kh * kw;
        let dx_stride_n = c_in * h * w_in;
        let dx_stride_g = c_in_per_g * h * w_in;

        unsafe {
            let dys = sl(*dy, base, n * c_out * h_out * w_out);
            let ws = sl(*w, base, c_out * c_in_per_g * kh * kw);
            let dxs = sl_mut(*dx, base, n * c_in * h * w_in);
            for v in dxs.iter_mut() {
                *v = 0.0;
            }

            // Each (ni, g) writes a disjoint dx_n_g region (col2im
            // scatter-adds into the freshly-zeroed slice), so the batch
            // loop fans out over the pool when RLX_FAST_CONV is set —
            // each worker owns a `dcol` scratch and a raw pointer it
            // offsets into its own dx window.
            if fast_conv_enabled() {
                let dx_addr = dxs.as_mut_ptr() as usize;
                crate::pool::par_for(n, 1, &|off, cnt| {
                    let mut dcol = vec![0f32; m_dim * n_dim];
                    for ni in off..off + cnt {
                        for g in 0..groups {
                            let w_g_off = g * w_stride_g;
                            let dy_n_g_off = ni * dy_stride_n + g * dy_stride_g;
                            let dx_n_g_off = ni * dx_stride_n + g * dx_stride_g;
                            crate::blas::sgemm_general(
                                ws.as_ptr().add(w_g_off),
                                dys.as_ptr().add(dy_n_g_off),
                                dcol.as_mut_ptr(),
                                m_dim,
                                n_dim,
                                k_dim,
                                1.0,
                                0.0,
                                m_dim,
                                n_dim,
                                n_dim,
                                true,
                                false,
                            );
                            let dx_g = std::slice::from_raw_parts_mut(
                                (dx_addr as *mut f32).add(dx_n_g_off),
                                dx_stride_g,
                            );
                            col2im(
                                &dcol, dx_g, c_in_per_g, h, w_in, h_out, w_out, kh, kw, sh, sw, ph,
                                pw, dh, dw,
                            );
                        }
                    }
                });
            } else {
                // Reused scratch buffer for the [m_dim, n_dim] dcol.
                let mut dcol = vec![0f32; m_dim * n_dim];
                for ni in 0..n {
                    for g in 0..groups {
                        let w_g_off = g * w_stride_g;
                        let dy_n_g_off = ni * dy_stride_n + g * dy_stride_g;
                        let dx_n_g_off = ni * dx_stride_n + g * dx_stride_g;

                        // dcol = w_g^T @ dy_n_g
                        // w_g  is stored as [k_dim rows, m_dim cols] row-major
                        // (i.e. K×M storage with lda = M = m_dim — exactly what
                        // sgemm_general wants for trans_a=true).
                        crate::blas::sgemm_general(
                            ws.as_ptr().add(w_g_off),
                            dys.as_ptr().add(dy_n_g_off),
                            dcol.as_mut_ptr(),
                            m_dim,
                            n_dim,
                            k_dim,
                            1.0,
                            0.0,
                            /*lda=*/ m_dim,
                            /*ldb=*/ n_dim,
                            /*ldc=*/ n_dim,
                            /*trans_a=*/ true,
                            /*trans_b=*/ false,
                        );

                        // dx_n_g += col2im(dcol)
                        col2im(
                            &dcol,
                            &mut dxs[dx_n_g_off..dx_n_g_off + dx_stride_g],
                            c_in_per_g,
                            h,
                            w_in,
                            h_out,
                            w_out,
                            kh,
                            kw,
                            sh,
                            sw,
                            ph,
                            pw,
                            dh,
                            dw,
                        );
                    }
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_conv2d_backward_weight(t: &Thunk, base: *mut u8) {
    let Thunk::Conv2dBackwardWeight {
        x,
        dy,
        dw,
        n,
        c_in,
        h,
        w,
        c_out,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw_dil,
        groups,
    } = t
    else {
        unreachable!()
    };
    {
        let n = *n as usize;
        let c_in = *c_in as usize;
        let h = *h as usize;
        let w = *w as usize;
        // Per-group im2col + GEMM, summed across batch.
        //
        //   col_n_g  = im2col(x_n_g)               (gather)
        //   dw_g    += dy_n_g  @  col_n_g^T        (sgemm, β=1)
        //
        // Layouts:
        //   x_n_g     [c_in_per_g, h · w]
        //   col_n_g   [c_in_per_g · kh · kw, h_out · w_out]
        //   dy_n_g    [c_out_per_g, h_out · w_out]
        //   dw_g      [c_out_per_g, c_in_per_g · kh · kw]
        let c_out = *c_out as usize;
        let h_out = *h_out as usize;
        let w_out = *w_out as usize;
        let kh = *kh as usize;
        let kw = *kw as usize;
        let sh = *sh as usize;
        let sw = *sw as usize;
        let ph = *ph as usize;
        let pw = *pw as usize;
        let dh = *dh as usize;
        let dw_dil = *dw_dil as usize;
        let groups = *groups as usize;
        let c_in_per_g = c_in / groups;
        let c_out_per_g = c_out / groups;

        let m_dim = c_out_per_g;
        let n_dim = c_in_per_g * kh * kw;
        let k_dim = h_out * w_out;

        let x_stride_n = c_in * h * w;
        let x_stride_g = c_in_per_g * h * w;
        let dy_stride_n = c_out * h_out * w_out;
        let dy_stride_g = c_out_per_g * h_out * w_out;
        let dw_stride_g = c_out_per_g * c_in_per_g * kh * kw;

        unsafe {
            let xs = sl(*x, base, n * c_in * h * w);
            let dys = sl(*dy, base, n * c_out * h_out * w_out);
            let dws = sl_mut(*dw, base, c_out * c_in_per_g * kh * kw);
            for v in dws.iter_mut() {
                *v = 0.0;
            }

            // dw is a cross-batch reduction (β=1 accumulate), so the
            // parallel path gives each worker a private `local` dw,
            // accumulates into it over its slice of the batch, then adds
            // it into the shared `dws` once under a lock — O(threads)
            // contention, not O(batch). RLX_FAST_CONV gates it.
            if fast_conv_enabled() {
                let dw_len = dws.len();
                let dws_addr = dws.as_mut_ptr() as usize;
                let lock = std::sync::Mutex::new(());
                crate::pool::par_for(n, 1, &|off, cnt| {
                    let mut col = vec![0f32; n_dim * k_dim];
                    let mut local = vec![0f32; dw_len];
                    for ni in off..off + cnt {
                        for g in 0..groups {
                            let x_n_g_off = ni * x_stride_n + g * x_stride_g;
                            let dy_n_g_off = ni * dy_stride_n + g * dy_stride_g;
                            let dw_g_off = g * dw_stride_g;
                            // Rows-layout im2col: col is [P, K] row-major, so
                            // the GEMM dy[c_out,P] @ col[P,K] reads col
                            // contiguously (trans_b=false) instead of the
                            // strided transposed read the [K,P] layout forced.
                            crate::im2col::im2col_rows_layout(
                                &xs[x_n_g_off..x_n_g_off + x_stride_g],
                                &mut col,
                                1,
                                c_in_per_g,
                                h,
                                w,
                                h_out,
                                w_out,
                                kh,
                                kw,
                                sh,
                                sw,
                                ph,
                                pw,
                                dh,
                                dw_dil,
                            );
                            crate::blas::sgemm_general(
                                dys.as_ptr().add(dy_n_g_off),
                                col.as_ptr(),
                                local.as_mut_ptr().add(dw_g_off),
                                m_dim,
                                n_dim,
                                k_dim,
                                1.0,
                                1.0,
                                k_dim,
                                n_dim,
                                n_dim,
                                false,
                                false,
                            );
                        }
                    }
                    let _guard = lock.lock().unwrap();
                    let dws = std::slice::from_raw_parts_mut(dws_addr as *mut f32, dw_len);
                    for (d, l) in dws.iter_mut().zip(local.iter()) {
                        *d += *l;
                    }
                });
            } else {
                let mut col = vec![0f32; n_dim * k_dim];
                for ni in 0..n {
                    for g in 0..groups {
                        let x_n_g_off = ni * x_stride_n + g * x_stride_g;
                        im2col(
                            &xs[x_n_g_off..x_n_g_off + x_stride_g],
                            &mut col,
                            c_in_per_g,
                            h,
                            w,
                            h_out,
                            w_out,
                            kh,
                            kw,
                            sh,
                            sw,
                            ph,
                            pw,
                            dh,
                            dw_dil,
                        );

                        let dy_n_g_off = ni * dy_stride_n + g * dy_stride_g;
                        let dw_g_off = g * dw_stride_g;

                        // dw_g += dy_n_g @ col^T
                        //
                        // Output shape m × n_out = c_out_per_g × (c_in_per_g·kh·kw).
                        // dy_n_g is stored M×K row-major (lda = K = k_dim).
                        // col is stored as N×K row-major; with trans_b=true,
                        // sgemm_general uses ldb = K = k_dim and treats it as
                        // transposed. β=1 accumulates across the batch loop.
                        crate::blas::sgemm_general(
                            dys.as_ptr().add(dy_n_g_off),
                            col.as_ptr(),
                            dws.as_mut_ptr().add(dw_g_off),
                            m_dim,
                            n_dim,
                            k_dim,
                            1.0,
                            1.0,
                            /*lda=*/ k_dim,
                            /*ldb=*/ k_dim,
                            /*ldc=*/ n_dim,
                            /*trans_a=*/ false,
                            /*trans_b=*/ true,
                        );
                    }
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_im2_col(t: &Thunk, base: *mut u8) {
    let Thunk::Im2Col {
        x,
        col,
        n,
        c_in,
        h,
        w,
        h_out,
        w_out,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw_dil,
    } = t
    else {
        unreachable!()
    };
    {
        let c_in = *c_in as usize;
        let h = *h as usize;
        let w = *w as usize;
        let h_out = *h_out as usize;
        let w_out = *w_out as usize;
        let kh = *kh as usize;
        let kw = *kw as usize;
        let sh = *sh as usize;
        let sw = *sw as usize;
        let ph = *ph as usize;
        let pw = *pw as usize;
        let dh = *dh as usize;
        let dw_dil = *dw_dil as usize;
        let per_batch = c_in * h * w;
        unsafe {
            let n_eff = if *n == 0 { 0usize } else { *n as usize };
            let x_floats = if n_eff == 0 {
                per_batch.max(1)
            } else {
                n_eff * per_batch
            };
            let xs = sl(*x, base, x_floats);
            let n = if *n == 0 {
                xs.len() / per_batch.max(1)
            } else {
                n_eff
            };
            let m = n * h_out * w_out;
            let k = c_in * kh * kw;
            let cols = sl_mut(*col, base, m * k);
            crate::im2col::im2col_rows_layout(
                xs, cols, n, c_in, h, w, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw_dil,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_conv2d_forward_f32(
    src: usize,
    weight: usize,
    dst: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w: u32,
    c_out: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    base: *mut u8,
) {
    let n = n as usize;
    let c_in = c_in as usize;
    let h = h as usize;
    let w = w as usize;
    let c_out = c_out as usize;
    let h_out = h_out as usize;
    let w_out = w_out as usize;
    let kh = kh as usize;
    let kw = kw as usize;
    let sh = sh as usize;
    let sw = sw as usize;
    let ph = ph as usize;
    let pw = pw as usize;
    let dh = dh as usize;
    let dw = dw as usize;
    let groups = groups as usize;
    let c_in_per_g = c_in / groups;
    let inp = sl(src, base, n * c_in * h * w);
    let wt = sl(weight, base, c_out * c_in_per_g * kh * kw);
    let out = sl_mut(dst, base, n * c_out * h_out * w_out);
    crate::conv_fwd::conv2d_forward_nchw_f32(
        inp, wt, out, n, c_in, h, w, c_out, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw, groups,
    );
}

/// Host / interpreter kernel for NCDHW `Thunk::Conv3d`. Slices `src`/`weight`
/// out of the arena and runs the naive 3-D forward conv. Mirrors
/// [`execute_conv2d_forward_f32`] with a depth axis.
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_conv3d_forward_f32(
    src: usize,
    weight: usize,
    dst: usize,
    n: u32,
    c_in: u32,
    d: u32,
    h: u32,
    w: u32,
    c_out: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dd: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    base: *mut u8,
) {
    let n = n as usize;
    let c_in = c_in as usize;
    let d = d as usize;
    let h = h as usize;
    let w = w as usize;
    let c_out = c_out as usize;
    let d_out = d_out as usize;
    let h_out = h_out as usize;
    let w_out = w_out as usize;
    let kd = kd as usize;
    let kh = kh as usize;
    let kw = kw as usize;
    let sd = sd as usize;
    let sh = sh as usize;
    let sw = sw as usize;
    let pd = pd as usize;
    let ph = ph as usize;
    let pw = pw as usize;
    let dd = dd as usize;
    let dh = dh as usize;
    let dw = dw as usize;
    let groups = groups as usize;
    let c_in_per_g = c_in / groups;
    let inp = sl(src, base, n * c_in * d * h * w);
    let wt = sl(weight, base, c_out * c_in_per_g * kd * kh * kw);
    let out = sl_mut(dst, base, n * c_out * d_out * h_out * w_out);
    crate::conv_fwd::conv3d_forward_ncdhw_f32(
        inp, wt, out, n, c_in, d, h, w, c_out, d_out, h_out, w_out, kd, kh, kw, sd, sh, sw, pd, ph,
        pw, dd, dh, dw, groups,
    );
}

/// Host / interpreter kernel for NCDHW `Thunk::ConvTranspose3d`. Mirrors
/// [`execute_conv_transpose2d_nchw_f32`] with a depth axis.
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_conv_transpose3d_ncdhw_f32(
    src: usize,
    weight: usize,
    dst: usize,
    n: u32,
    c_in: u32,
    d: u32,
    h: u32,
    w_in: u32,
    c_out: u32,
    d_out: u32,
    h_out: u32,
    w_out: u32,
    kd: u32,
    kh: u32,
    kw: u32,
    sd: u32,
    sh: u32,
    sw: u32,
    pd: u32,
    ph: u32,
    pw: u32,
    dd: u32,
    dh: u32,
    dw: u32,
    groups: u32,
    base: *mut u8,
) {
    let n = n as usize;
    let c_in = c_in as usize;
    let d = d as usize;
    let h = h as usize;
    let w_in = w_in as usize;
    let c_out = c_out as usize;
    let d_out = d_out as usize;
    let h_out = h_out as usize;
    let w_out = w_out as usize;
    let kd = kd as usize;
    let kh = kh as usize;
    let kw = kw as usize;
    let sd = sd as usize;
    let sh = sh as usize;
    let sw = sw as usize;
    let pd = pd as usize;
    let ph = ph as usize;
    let pw = pw as usize;
    let dd = dd as usize;
    let dh = dh as usize;
    let dw = dw as usize;
    let groups = groups as usize;
    let in_elems = n * c_in * d * h * w_in;
    let w_elems = c_in * (c_out / groups) * kd * kh * kw;
    let out_elems = n * c_out * d_out * h_out * w_out;
    let input = sl(src, base, in_elems);
    let wt = sl(weight, base, w_elems);
    let output = sl_mut(dst, base, out_elems);
    crate::kernels::conv_transpose3d_ncdhw(
        input, wt, output, n, c_in, d, h, w_in, c_out, d_out, h_out, w_out, kd, kh, kw, sd, sh, sw,
        pd, ph, pw, dd, dh, dw, groups,
    );
}

pub unsafe fn execute_maxpool2d_backward_f32(
    x: usize,
    dy: usize,
    dx: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    base: *mut u8,
) {
    let (n, c, h, w) = (n as usize, c as usize, h as usize, w as usize);
    let (h_out, w_out) = (h_out as usize, w_out as usize);
    let (kh, kw) = (kh as usize, kw as usize);
    let (sh, sw) = (sh as usize, sw as usize);
    let (ph, pw) = (ph as usize, pw as usize);
    let xs = sl(x, base, n * c * h * w);
    let dys = sl(dy, base, n * c * h_out * w_out);
    let dxs = sl_mut(dx, base, n * c * h * w);
    // Each (n, c) plane is independent — under RLX_FAST_CONV, fan the existing
    // per-plane kernel out over the channel-batch (disjoint dx windows).
    if fast_conv_enabled() && crate::pool::should_parallelize(n * c * h * w) {
        let (in_plane, out_plane) = (h * w, h_out * w_out);
        let x_addr = xs.as_ptr() as usize;
        let dy_addr = dys.as_ptr() as usize;
        let dx_addr = dxs.as_mut_ptr() as usize;
        crate::pool::par_for(n * c, crate::pool::outer_chunk(n * c), &|off, cnt| {
            for nc in off..off + cnt {
                let xp =
                    std::slice::from_raw_parts((x_addr as *const f32).add(nc * in_plane), in_plane);
                let dyp = std::slice::from_raw_parts(
                    (dy_addr as *const f32).add(nc * out_plane),
                    out_plane,
                );
                let dxp = std::slice::from_raw_parts_mut(
                    (dx_addr as *mut f32).add(nc * in_plane),
                    in_plane,
                );
                crate::training_bwd::maxpool2d_backward_nchw(
                    xp, dyp, dxp, 1, 1, h, w, h_out, w_out, kh, kw, sh, sw, ph, pw,
                );
            }
        });
    } else {
        crate::training_bwd::maxpool2d_backward_nchw(
            xs, dys, dxs, n, c, h, w, h_out, w_out, kh, kw, sh, sw, ph, pw,
        );
    }
}

/// Host fallback for NCHW ConvTranspose2d.
pub unsafe fn execute_conv_transpose2d_nchw_f32(
    src: usize,
    weight: usize,
    dst: usize,
    n: usize,
    c_in: usize,
    h: usize,
    w_in: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    groups: usize,
    base: *mut u8,
) {
    let in_elems = n * c_in * h * w_in;
    let w_elems = c_in * (c_out / groups) * kh * kw;
    let out_elems = n * c_out * h_out * w_out;
    unsafe {
        let input = sl(src, base, in_elems);
        let wt = sl(weight, base, w_elems);
        let output = sl_mut(dst, base, out_elems);
        crate::kernels::conv_transpose2d_nchw(
            input, wt, output, n, c_in, h, w_in, c_out, h_out, w_out, kh, kw, sh, sw, ph, pw, dh,
            dw, groups,
        );
    }
}

/// Host fallback for nearest 2× upsample on NCHW.
pub unsafe fn execute_resize_nearest_2x_f32(
    src: usize,
    dst: usize,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    base: *mut u8,
) {
    let in_plane = c * h * w;
    let out_plane = c * h * 2 * w * 2;
    for ni in 0..n {
        let input = unsafe {
            sl(
                src + ni * in_plane * std::mem::size_of::<f32>(),
                base,
                in_plane,
            )
        };
        let output = unsafe {
            sl_mut(
                dst + ni * out_plane * std::mem::size_of::<f32>(),
                base,
                out_plane,
            )
        };
        crate::kernels::resize_nearest_2x_nchw(input, output, c, h, w);
    }
}

/// im2col for one image (single batch + group slice).
///
/// Source `x` is `[c_in, H, W]` row-major. Destination `col` is
/// `[c_in · kH · kW, H_out · W_out]` row-major. Out-of-bounds positions
/// (in the padded region) are written as 0.
///
/// `col[(ci · kH · kW + ki · kW + kj) · n_dim + ho · W_out + wo] =
///    x[ci, ho·sh + ki·dh − ph, wo·sw + kj·dw_dil − pw]`
#[allow(clippy::too_many_arguments)]
/// Is the optimized CPU path enabled (im2col+BLAS forward conv, rayon fans
/// for pool / elementwise / bwd)? **On by default.** Set `RLX_FAST_CONV=0`
/// (or `off`/`false`/`no`) for the reference scalar kernels — used by the
/// cortexm trainer's unfused `rlx` bench bar. Process-global (`OnceLock`) so
/// a run is either fused or reference, never mixed mid-process.
pub(crate) fn fast_conv_enabled() -> bool {
    use std::sync::OnceLock;
    static FAST_CONV: OnceLock<bool> = OnceLock::new();
    *FAST_CONV.get_or_init(|| {
        let on = match rlx_ir::env::var("RLX_FAST_CONV").as_deref() {
            Some("0") | Some("off") | Some("false") | Some("no") => false,
            // unset, empty, 1/on/true/yes, or any other value → fast path
            _ => true,
        };
        if on {
            // Rayon fans + multi-threaded OpenBLAS would nest N×N workers.
            crate::blas::limit_inner_threads();
        }
        on
    })
}

/// Reference forward convolution: a direct scalar nested loop. Correct and
/// dependency-free, but scalar / single-threaded with a bounds-check branch
/// in the hot loop — orders of magnitude slower on training shapes. Kept as
/// the parity oracle for `conv2d_forward_im2col` and for `RLX_FAST_CONV=0`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_forward_naive(
    inp: &[f32],
    wt: &[f32],
    out: &mut [f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    groups: usize,
) {
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;
    for ni in 0..n {
        for co in 0..c_out {
            let g = co / c_out_per_g;
            let ci_start = g * c_in_per_g;
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc = 0f32;
                    for ci_off in 0..c_in_per_g {
                        let ci = ci_start + ci_off;
                        let in_chan = ((ni * c_in) + ci) * h * w;
                        let wt_chan = ((co * c_in_per_g) + ci_off) * kh * kw;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = ho * sh + ki * dh;
                                let wi = wo * sw + kj * dw;
                                if hi < ph || wi < pw {
                                    continue;
                                }
                                let hi = hi - ph;
                                let wi = wi - pw;
                                if hi >= h || wi >= w {
                                    continue;
                                }
                                acc += inp[in_chan + hi * w + wi] * wt[wt_chan + ki * kw + kj];
                            }
                        }
                    }
                    out[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo] = acc;
                }
            }
        }
    }
}

/// Winograd F(2×2,3×3) enabled? Read once from `RLX_WINOGRAD`. Applies only to
/// 3×3 stride-1 no-dilation groups-1 convs (the conv-net hot case).
pub(crate) fn winograd_enabled() -> bool {
    use std::sync::OnceLock;
    static W: OnceLock<bool> = OnceLock::new();
    *W.get_or_init(|| {
        matches!(
            rlx_ir::env::var("RLX_WINOGRAD").as_deref(),
            Some("1") | Some("on") | Some("true") | Some("yes")
        )
    })
}

/// Direct (no-im2col) forward conv enabled? `RLX_DIRECT_CONV`. Off by default —
/// im2col+BLAS is faster for low-channel convs; this wins only at high channels.
pub(crate) fn direct_conv_enabled() -> bool {
    use std::sync::OnceLock;
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| {
        matches!(
            rlx_ir::env::var("RLX_DIRECT_CONV").as_deref(),
            Some("1") | Some("on") | Some("true") | Some("yes")
        )
    })
}

/// Filter transform U = G·g·Gᵀ for one 3×3 filter → 4×4 (row-major).
#[inline]
pub(crate) fn winograd_filter_transform(g: &[f32]) -> [f32; 16] {
    let mut tmp = [0f32; 12]; // 4×3 = G·g
    for j in 0..3 {
        let (g0, g1, g2) = (g[j], g[3 + j], g[6 + j]);
        tmp[j] = g0;
        tmp[3 + j] = 0.5 * (g0 + g1 + g2);
        tmp[6 + j] = 0.5 * (g0 - g1 + g2);
        tmp[9 + j] = g2;
    }
    let mut u = [0f32; 16];
    for i in 0..4 {
        let (t0, t1, t2) = (tmp[i * 3], tmp[i * 3 + 1], tmp[i * 3 + 2]);
        u[i * 4] = t0;
        u[i * 4 + 1] = 0.5 * (t0 + t1 + t2);
        u[i * 4 + 2] = 0.5 * (t0 - t1 + t2);
        u[i * 4 + 3] = t2;
    }
    u
}

/// Input transform V = Bᵀ·d·B for one 4×4 tile → 4×4 (row-major).
#[inline]
pub(crate) fn winograd_input_transform(d: &[f32; 16]) -> [f32; 16] {
    let mut t = [0f32; 16];
    for j in 0..4 {
        let (d0, d1, d2, d3) = (d[j], d[4 + j], d[8 + j], d[12 + j]);
        t[j] = d0 - d2;
        t[4 + j] = d1 + d2;
        t[8 + j] = d2 - d1;
        t[12 + j] = d1 - d3;
    }
    let mut v = [0f32; 16];
    for i in 0..4 {
        let (a0, a1, a2, a3) = (t[i * 4], t[i * 4 + 1], t[i * 4 + 2], t[i * 4 + 3]);
        v[i * 4] = a0 - a2;
        v[i * 4 + 1] = a1 + a2;
        v[i * 4 + 2] = a2 - a1;
        v[i * 4 + 3] = a1 - a3;
    }
    v
}

/// Output transform Y = Aᵀ·m·A for one 4×4 accumulator → 2×2 (row-major).
#[inline]
pub(crate) fn winograd_output_transform(m: &[f32; 16]) -> [f32; 4] {
    let mut s = [0f32; 8];
    for j in 0..4 {
        let (m0, m1, m2, m3) = (m[j], m[4 + j], m[8 + j], m[12 + j]);
        s[j] = m0 + m1 + m2;
        s[4 + j] = m1 - m2 - m3;
    }
    let mut y = [0f32; 4];
    for i in 0..2 {
        let (a0, a1, a2, a3) = (s[i * 4], s[i * 4 + 1], s[i * 4 + 2], s[i * 4 + 3]);
        y[i * 2] = a0 + a1 + a2;
        y[i * 2 + 1] = a1 - a2 - a3;
    }
    y
}

/// Winograd F(2×2,3×3) forward convolution (3×3, stride 1, no dilation, groups
/// 1, valid). ~2.25× fewer multiplies than im2col by working on 4×4 tiles:
/// transform filter (G·g·Gᵀ) and input (Bᵀ·d·B), 16 channel-GEMMs (one per
/// tile position), then output transform (Aᵀ·m·A). Boundary tiles zero-pad the
/// input read and skip out-of-range output writes (handles odd H_out/W_out).
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_forward_winograd(
    inp: &[f32],
    wt: &[f32],
    out: &mut [f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
) {
    let th = h_out.div_ceil(2);
    let tw = w_out.div_ceil(2);
    let tiles_per = th * tw;
    let nt = n * tiles_per;
    if nt == 0 {
        return;
    }

    // Filter transform U[p][co][ci]  (p = 0..16).
    let mut u = vec![0f32; 16 * c_out * c_in];
    for co in 0..c_out {
        for ci in 0..c_in {
            let uu = winograd_filter_transform(&wt[(co * c_in + ci) * 9..(co * c_in + ci) * 9 + 9]);
            for (p, &val) in uu.iter().enumerate() {
                u[p * c_out * c_in + co * c_in + ci] = val;
            }
        }
    }

    // Input transform V[p][ci][tile].
    let mut v = vec![0f32; 16 * c_in * nt];
    let v_addr = v.as_mut_ptr() as usize;
    let in_xform = |tile: usize| {
        let ni = tile / tiles_per;
        let rem = tile % tiles_per;
        let (h0, w0) = (2 * (rem / tw), 2 * (rem % tw));
        for ci in 0..c_in {
            let mut d = [0f32; 16];
            for di in 0..4 {
                let hh = h0 + di;
                if hh >= h {
                    continue;
                }
                let base = ((ni * c_in + ci) * h + hh) * w;
                for dj in 0..4 {
                    let ww = w0 + dj;
                    if ww < w {
                        d[di * 4 + dj] = inp[base + ww];
                    }
                }
            }
            let vv = winograd_input_transform(&d);
            for (p, &val) in vv.iter().enumerate() {
                unsafe {
                    *((v_addr as *mut f32).add(p * c_in * nt + ci * nt + tile)) = val;
                }
            }
        }
    };
    if fast_conv_enabled() && crate::pool::should_parallelize(nt * c_in * 16) {
        crate::pool::par_for(nt, crate::pool::outer_chunk(nt), &|off, cnt| {
            for t in off..off + cnt {
                in_xform(t);
            }
        });
    } else {
        for t in 0..nt {
            in_xform(t);
        }
    }

    // 16 channel-GEMMs: M[p] = U[p]·V[p]  → [c_out, nt].
    let mut m = vec![0f32; 16 * c_out * nt];
    for p in 0..16 {
        crate::blas::sgemm(
            &u[p * c_out * c_in..(p + 1) * c_out * c_in],
            &v[p * c_in * nt..(p + 1) * c_in * nt],
            &mut m[p * c_out * nt..(p + 1) * c_out * nt],
            c_out,
            c_in,
            nt,
        );
    }

    // Output transform + scatter.
    let out_addr = out.as_mut_ptr() as usize;
    let out_xform = |tile: usize| {
        let ni = tile / tiles_per;
        let rem = tile % tiles_per;
        let (ho0, wo0) = (2 * (rem / tw), 2 * (rem % tw));
        for co in 0..c_out {
            let mut mm = [0f32; 16];
            for (p, slot) in mm.iter_mut().enumerate() {
                *slot = m[p * c_out * nt + co * nt + tile];
            }
            let y = winograd_output_transform(&mm);
            for yi in 0..2 {
                let oh = ho0 + yi;
                if oh >= h_out {
                    continue;
                }
                for yj in 0..2 {
                    let ow = wo0 + yj;
                    if ow < w_out {
                        unsafe {
                            *((out_addr as *mut f32)
                                .add(((ni * c_out + co) * h_out + oh) * w_out + ow)) =
                                y[yi * 2 + yj];
                        }
                    }
                }
            }
        }
    };
    if fast_conv_enabled() && crate::pool::should_parallelize(nt * c_out * 16) {
        crate::pool::par_for(nt, crate::pool::outer_chunk(nt), &|off, cnt| {
            for t in off..off + cnt {
                out_xform(t);
            }
        });
    } else {
        for t in 0..nt {
            out_xform(t);
        }
    }
}

/// Direct forward convolution for the stride-1, no-padding, dilation-1 case
/// (the conv-net hot path). Unlike im2col it never materialises the 9×-expanded
/// patch buffer — each output plane stays L1-resident while we accumulate over
/// (c_in, kh, kw), and the innermost loop is a contiguous SAXPY over the output
/// row (`out[wo] += w * in[wo+kj]`) that autovectorizes. Compute-bound, not
/// bandwidth-bound — the win over im2col for low-channel convs. Parallel over
/// (batch × c_out); caller guarantees stride/pad/dilation eligibility.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_forward_direct(
    inp: &[f32],
    wt: &[f32],
    out: &mut [f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    groups: usize,
) {
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;
    let out_plane = h_out * w_out;
    let in_plane = h * w;
    let out_addr = out.as_mut_ptr() as usize;
    let compute = |nco: usize| {
        let ni = nco / c_out;
        let co = nco % c_out;
        let ci_start = (co / c_out_per_g) * c_in_per_g;
        let out_base = (ni * c_out + co) * out_plane;
        let op = unsafe {
            std::slice::from_raw_parts_mut((out_addr as *mut f32).add(out_base), out_plane)
        };
        for v in op.iter_mut() {
            *v = 0.0;
        }
        for ci_off in 0..c_in_per_g {
            let in_base = (ni * c_in + ci_start + ci_off) * in_plane;
            let wt_base = (co * c_in_per_g + ci_off) * kh * kw;
            for ki in 0..kh {
                for kj in 0..kw {
                    let wv = wt[wt_base + ki * kw + kj];
                    for ho in 0..h_out {
                        let in_row = in_base + (ho + ki) * w + kj;
                        let dst = &mut op[ho * w_out..ho * w_out + w_out];
                        let src = &inp[in_row..in_row + w_out];
                        for wo in 0..w_out {
                            dst[wo] += wv * src[wo];
                        }
                    }
                }
            }
        }
    };
    if fast_conv_enabled() && crate::pool::should_parallelize(n * c_out * out_plane) {
        crate::pool::par_for(
            n * c_out,
            crate::pool::outer_chunk(n * c_out),
            &|off, cnt| {
                for nco in off..off + cnt {
                    compute(nco);
                }
            },
        );
    } else {
        for nco in 0..n * c_out {
            compute(nco);
        }
    }
}

/// im2col + BLAS forward convolution. For each (batch, group) we gather the
/// receptive-field patches into a `[c_in_per_g·kH·kW, H_out·W_out]` column
/// matrix and dispatch a single `sgemm`:
///
/// ```text
///   out_n_g [c_out_per_g, P]  =  weight_g [c_out_per_g, K]  @  col [K, P]
/// ```
///
/// The result lands in NCHW directly (matching `Conv2D1x1`). The batch loop
/// is embarrassingly parallel — each image writes a disjoint output region —
/// so it fans out over the thread pool, each worker owning a `col` scratch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_forward_im2col(
    inp: &[f32],
    wt: &[f32],
    out: &mut [f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    groups: usize,
) {
    let c_in_per_g = c_in / groups;
    let c_out_per_g = c_out / groups;
    let k_dim = c_in_per_g * kh * kw; // im2col rows (contraction dim)
    let p_dim = h_out * w_out; // spatial positions
    let x_stride_n = c_in * h * w;
    let x_stride_g = c_in_per_g * h * w;
    let out_stride_n = c_out * h_out * w_out;
    let out_stride_g = c_out_per_g * p_dim;
    let w_stride_g = c_out_per_g * k_dim;

    // When the batch (`n·groups`) is too small to fill the cores — the common
    // single-utterance generative case (n=1, groups=1), e.g. a HiFi-GAN vocoder
    // whose per-conv im2col over a long sequence dominates — batch-parallelism
    // leaves every core but one idle. Fall back to parallelizing the *im2col
    // fill* over `c_in` (each `ci` owns a disjoint block of col rows) and let
    // BLAS thread the GEMM. Bit-identical to the serial path (independent
    // writes / same reduction order).
    if n * groups < crate::pool::num_threads() {
        let mut col = vec![0f32; k_dim * p_dim];
        for ni in 0..n {
            for g in 0..groups {
                let x_off = ni * x_stride_n + g * x_stride_g;
                im2col_par(
                    &inp[x_off..x_off + x_stride_g],
                    &mut col,
                    c_in_per_g,
                    h,
                    w,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                );
                let w_off = g * w_stride_g;
                let o_off = ni * out_stride_n + g * out_stride_g;
                let out_g = &mut out[o_off..o_off + out_stride_g];
                // sgemm_auto: single-batch vocoder/DiT convs were stuck on
                // 1-thread BLAS while im2col alone used the pool.
                crate::blas::sgemm_auto(
                    &wt[w_off..w_off + w_stride_g],
                    &col,
                    out_g,
                    c_out_per_g,
                    k_dim,
                    p_dim,
                );
            }
        }
        return;
    }

    // Hand each worker the output base as a raw address: every (ni, g) writes
    // a disjoint `[out_stride_g]` window, so the aliasing is provably safe.
    let out_addr = out.as_mut_ptr() as usize;
    crate::pool::par_for(n, 1, &|off, cnt| {
        let mut col = vec![0f32; k_dim * p_dim];
        for ni in off..off + cnt {
            for g in 0..groups {
                let x_off = ni * x_stride_n + g * x_stride_g;
                im2col(
                    &inp[x_off..x_off + x_stride_g],
                    &mut col,
                    c_in_per_g,
                    h,
                    w,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                );
                let w_off = g * w_stride_g;
                let o_off = ni * out_stride_n + g * out_stride_g;
                let out_g = unsafe {
                    std::slice::from_raw_parts_mut((out_addr as *mut f32).add(o_off), out_stride_g)
                };
                crate::blas::sgemm(
                    &wt[w_off..w_off + w_stride_g],
                    &col,
                    out_g,
                    c_out_per_g,
                    k_dim,
                    p_dim,
                );
            }
        }
    });
}

/// `im2col` with the patch gather parallelized over input channels. Each `ci`
/// owns the disjoint col-row block `[ci·kH·kW, (ci+1)·kH·kW)`, so the concurrent
/// writes never overlap. Used for small-batch convs where the outer
/// batch-parallel loop can't keep the cores busy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn im2col_par(
    x: &[f32],
    col: &mut [f32],
    c_in: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
) {
    let n_dim = h_out * w_out;
    debug_assert_eq!(col.len(), c_in * kh * kw * n_dim);
    debug_assert_eq!(x.len(), c_in * h * w);
    if c_in < 2 || !crate::pool::should_parallelize(c_in * kh * kw * n_dim) {
        im2col(
            x, col, c_in, h, w, h_out, w_out, kh, kw, sh, sw, ph, pw, dh, dw_dil,
        );
        return;
    }
    let h_isz = h as isize;
    let w_isz = w as isize;
    let ph_isz = ph as isize;
    let pw_isz = pw as isize;
    let col_addr = col.as_mut_ptr() as usize;
    crate::pool::par_for(c_in, 1, &|off, cnt| {
        for ci in off..off + cnt {
            for ki in 0..kh {
                for kj in 0..kw {
                    let row = ((ci * kh) + ki) * kw + kj;
                    let row_off = row * n_dim;
                    // SAFETY: disjoint per (ci, ki, kj) — no two workers touch
                    // the same row window.
                    let col_row = unsafe {
                        std::slice::from_raw_parts_mut((col_addr as *mut f32).add(row_off), n_dim)
                    };
                    for ho in 0..h_out {
                        let hi = (ho * sh + ki * dh) as isize - ph_isz;
                        if hi < 0 || hi >= h_isz {
                            for wo in 0..w_out {
                                col_row[ho * w_out + wo] = 0.0;
                            }
                            continue;
                        }
                        let hi = hi as usize;
                        let in_row_off = (ci * h + hi) * w;
                        for wo in 0..w_out {
                            let wi = (wo * sw + kj * dw_dil) as isize - pw_isz;
                            col_row[ho * w_out + wo] = if wi < 0 || wi >= w_isz {
                                0.0
                            } else {
                                x[in_row_off + wi as usize]
                            };
                        }
                    }
                }
            }
        }
    });
}

pub(crate) fn im2col(
    x: &[f32],
    col: &mut [f32],
    c_in: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
) {
    let n_dim = h_out * w_out;
    debug_assert_eq!(col.len(), c_in * kh * kw * n_dim);
    debug_assert_eq!(x.len(), c_in * h * w);
    let h_isz = h as isize;
    let w_isz = w as isize;
    let ph_isz = ph as isize;
    let pw_isz = pw as isize;
    for ci in 0..c_in {
        for ki in 0..kh {
            for kj in 0..kw {
                let row = ((ci * kh) + ki) * kw + kj;
                let row_off = row * n_dim;
                for ho in 0..h_out {
                    let hi = (ho * sh + ki * dh) as isize - ph_isz;
                    if hi < 0 || hi >= h_isz {
                        for wo in 0..w_out {
                            col[row_off + ho * w_out + wo] = 0.0;
                        }
                        continue;
                    }
                    let hi = hi as usize;
                    let in_row_off = (ci * h + hi) * w;
                    for wo in 0..w_out {
                        let wi = (wo * sw + kj * dw_dil) as isize - pw_isz;
                        col[row_off + ho * w_out + wo] = if wi < 0 || wi >= w_isz {
                            0.0
                        } else {
                            x[in_row_off + wi as usize]
                        };
                    }
                }
            }
        }
    }
}

/// col2im — inverse of `im2col` with scatter-accumulation. The caller
/// is responsible for zeroing `x` if it doesn't already start zero
/// (the conv-input-grad path zeros once before the batch loop).
///
/// `x[ci, hi, wi] += col[(ci · kH · kW + ki · kW + kj) · n_dim + ho · W_out + wo]`
/// for all `(ki, kj, ho, wo)` whose `(hi, wi)` lands in `[0, H) × [0, W)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn col2im(
    col: &[f32],
    x: &mut [f32],
    c_in: usize,
    h: usize,
    w: usize,
    h_out: usize,
    w_out: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw_dil: usize,
) {
    let n_dim = h_out * w_out;
    debug_assert_eq!(col.len(), c_in * kh * kw * n_dim);
    debug_assert_eq!(x.len(), c_in * h * w);
    let h_isz = h as isize;
    let w_isz = w as isize;
    let ph_isz = ph as isize;
    let pw_isz = pw as isize;
    for ci in 0..c_in {
        for ki in 0..kh {
            for kj in 0..kw {
                let row = ((ci * kh) + ki) * kw + kj;
                let row_off = row * n_dim;
                for ho in 0..h_out {
                    let hi = (ho * sh + ki * dh) as isize - ph_isz;
                    if hi < 0 || hi >= h_isz {
                        continue;
                    }
                    let hi = hi as usize;
                    let in_row_off = (ci * h + hi) * w;
                    for wo in 0..w_out {
                        let wi = (wo * sw + kj * dw_dil) as isize - pw_isz;
                        if wi < 0 || wi >= w_isz {
                            continue;
                        }
                        x[in_row_off + wi as usize] += col[row_off + ho * w_out + wo];
                    }
                }
            }
        }
    }
}
