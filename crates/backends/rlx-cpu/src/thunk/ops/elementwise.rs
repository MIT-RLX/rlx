#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

#[allow(unused_variables)]
pub(crate) fn compile_gather(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Gather { axis } = &node.op else {
        unreachable!()
    };
    {
        // Non-zero axis: outer × num_idx × trailing layout.
        let table_shape = &graph.node(node.inputs[0]).shape;
        let rank = table_shape.rank();
        let outer: usize = (0..*axis)
            .map(|i| table_shape.dim(i).unwrap_static())
            .product::<usize>()
            .max(1);
        let trailing: usize = (*axis + 1..rank)
            .map(|i| table_shape.dim(i).unwrap_static())
            .product::<usize>()
            .max(1);
        let axis_dim = table_shape.dim(*axis).unwrap_static();
        let idx_len = get_len(graph, node.inputs[1]);
        let idx_i64 = u8::from(graph.node(node.inputs[1]).shape.dtype() == rlx_ir::DType::I64);
        let table_bytes = graph.node(node.inputs[0]).shape.dtype().size_bytes() as u8;
        Thunk::GatherAxis {
            table: node_offset(arena, node.inputs[0]),
            idx: node_offset(arena, node.inputs[1]),
            dst: node_offset(arena, node.id),
            outer: outer as u32,
            axis_dim: axis_dim as u32,
            num_idx: idx_len as u32,
            trailing: trailing as u32,
            idx_i64,
            table_bytes,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_narrow(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Narrow { axis, start, len } = &node.op else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let elem_bytes = in_shape.dtype().size_bytes() as u8;
        let rank = in_shape.rank();
        let outer: usize = (0..*axis)
            .map(|i| in_shape.dim(i).unwrap_static())
            .product::<usize>()
            .max(1);
        let inner: usize = (*axis + 1..rank)
            .map(|i| in_shape.dim(i).unwrap_static())
            .product::<usize>()
            .max(1);
        let in_axis = in_shape.dim(*axis).unwrap_static();
        let src_byte_offset =
            node_offset(arena, node.inputs[0]) + start * inner * elem_bytes as usize;
        Thunk::Narrow {
            src: src_byte_offset,
            dst: node_offset(arena, node.id),
            outer: outer as u32,
            src_stride: (in_axis * inner) as u32, // elements per outer step in source
            dst_stride: (*len * inner) as u32,    // elements per outer step in dest
            inner: (*len * inner) as u32,         // elements to copy per outer step
            elem_bytes,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_reverse(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Reverse { axes } = &node.op else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let rank = in_shape.rank();
        let dims: Vec<u32> = (0..rank)
            .map(|i| in_shape.dim(i).unwrap_static() as u32)
            .collect();
        let mut rev_mask = vec![false; rank];
        for &a in axes {
            if a < rank {
                rev_mask[a] = true;
            }
        }
        Thunk::Reverse {
            src: node_offset(arena, node.inputs[0]),
            dst: node_offset(arena, node.id),
            dims,
            rev_mask,
            elem_bytes: in_shape.dtype().size_bytes() as u8,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_cast(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Cast { to } = &node.op else {
        unreachable!()
    };
    {
        let in_node = graph.node(node.inputs[0]);
        let in_dtype = in_node.shape.dtype();
        let out_dtype = *to;
        let len = node.shape.num_elements().unwrap();
        let src = node_offset(arena, node.inputs[0]);
        let dst = node_offset(arena, node.id);
        if in_dtype == rlx_ir::DType::F32 && out_dtype == rlx_ir::DType::I64 {
            Thunk::CastF32ToI64 {
                src,
                dst,
                len: len as u32,
            }
        } else if in_dtype == rlx_ir::DType::F32 && out_dtype == rlx_ir::DType::F64 {
            Thunk::CastF32ToF64 {
                src,
                dst,
                len: len as u32,
            }
        } else if in_dtype == rlx_ir::DType::F32 && out_dtype == rlx_ir::DType::I32 {
            Thunk::CastF32ToI32 {
                src,
                dst,
                len: len as u32,
            }
        } else if in_dtype == rlx_ir::DType::I64 && out_dtype == rlx_ir::DType::F32 {
            Thunk::CastI64ToF32 {
                src,
                dst,
                len: len as u32,
            }
        } else if in_dtype == rlx_ir::DType::Bool && out_dtype == rlx_ir::DType::I32 {
            Thunk::CastBoolToI32 {
                src,
                dst,
                len: len as u32,
            }
        } else if in_dtype == rlx_ir::DType::Bool && out_dtype == rlx_ir::DType::F32 {
            // Bool is 1 byte; the generic f32 Copy below would misread it as
            // 4-byte f32. VITS sequence masks are `Cast(Less(...), f32)`.
            Thunk::CastBoolToF32 {
                src,
                dst,
                len: len as u32,
            }
        } else if in_dtype == rlx_ir::DType::F32 && out_dtype == rlx_ir::DType::Bool {
            // f32 → 1-byte Bool. Output is native-width Bool (1 byte); the
            // generic Copy would write 4 bytes. Used by logical And/Or.
            Thunk::CastF32ToBool {
                src,
                dst,
                len: len as u32,
            }
        } else if in_dtype == rlx_ir::DType::I32 && out_dtype == rlx_ir::DType::F32 {
            Thunk::CastI32ToF32 {
                src,
                dst,
                len: len as u32,
            }
        } else if in_dtype == rlx_ir::DType::I32 && out_dtype == rlx_ir::DType::I64 {
            Thunk::CastI32ToI64 {
                src,
                dst,
                len: len as u32,
            }
        } else if in_dtype == rlx_ir::DType::I32 && out_dtype == rlx_ir::DType::Bool {
            Thunk::CastI32ToBool {
                src,
                dst,
                len: len as u32,
            }
        } else if in_dtype == rlx_ir::DType::I64 && out_dtype == rlx_ir::DType::Bool {
            Thunk::CastI64ToBool {
                src,
                dst,
                len: len as u32,
            }
        } else if in_dtype == rlx_ir::DType::Bool && out_dtype == rlx_ir::DType::I64 {
            Thunk::CastBoolToI64 {
                src,
                dst,
                len: len as u32,
            }
        } else if in_dtype == out_dtype {
            match out_dtype {
                rlx_ir::DType::F64 => Thunk::CopyF64 {
                    src,
                    dst,
                    len: len as u32,
                },
                rlx_ir::DType::I64 => Thunk::CopyI64 {
                    src,
                    dst,
                    len: len as u32,
                },
                _ => Thunk::Copy {
                    src,
                    dst,
                    len: len as u32,
                },
            }
        } else {
            Thunk::Copy {
                src,
                dst,
                len: len as u32,
            }
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_expand(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Expand { .. } = &node.op else {
        unreachable!()
    };
    {
        // Broadcast: build per-output-dim strides where any input dim
        // of size 1 has stride 0 (read the same element repeatedly).
        // Reuses the Thunk::Transpose runtime — N-D walk with strides
        // is identical; only the strides differ.
        let in_shape = &graph.node(node.inputs[0]).shape;
        let out_shape = &node.shape;
        let in_rank = in_shape.rank();
        let out_rank = out_shape.rank();
        // Implicit leading 1s if input has lower rank.
        let pad = out_rank.saturating_sub(in_rank);
        let in_dims: Vec<usize> = (0..out_rank)
            .map(|i| {
                if i < pad {
                    1
                } else {
                    in_shape.dim(i - pad).unwrap_static()
                }
            })
            .collect();
        // Row-major input strides (over the padded shape).
        let mut in_strides_full = vec![1usize; out_rank];
        for d in (0..out_rank.saturating_sub(1)).rev() {
            in_strides_full[d] = in_strides_full[d + 1] * in_dims[d + 1];
        }
        let out_dims: Vec<u32> = (0..out_rank)
            .map(|i| out_shape.dim(i).unwrap_static() as u32)
            .collect();
        // Stride is 0 for broadcast dims (in_dim == 1 && out_dim > 1).
        let in_strides: Vec<u32> = (0..out_rank)
            .map(|i| {
                if in_dims[i] == 1 && (out_dims[i] as usize) > 1 {
                    0
                } else {
                    in_strides_full[i] as u32
                }
            })
            .collect();
        let in_total = in_dims.iter().product::<usize>() as u32;
        let src = node_offset(arena, node.inputs[0]);
        let dst = node_offset(arena, node.id);
        let elem_bytes = node.shape.dtype().size_bytes() as u8;
        match node.shape.dtype() {
            rlx_ir::DType::F64 => Thunk::TransposeF64 {
                src,
                dst,
                in_total,
                out_dims,
                in_strides,
            },
            _ => Thunk::Transpose {
                src,
                dst,
                in_total,
                out_dims,
                in_strides,
                elem_bytes,
            },
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_transpose(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Transpose { perm } = &node.op else {
        unreachable!()
    };
    {
        // Pre-compute (out_dims, in_strides_for_each_out_dim) so the
        // runtime loop is just an N-D index walk + scatter.
        let in_shape = &graph.node(node.inputs[0]).shape;
        let in_rank = in_shape.rank();
        if perm.iter().any(|&p| p >= in_rank) {
            Thunk::Nop
        } else {
            let in_dims: Vec<usize> = (0..in_rank)
                .map(|i| in_shape.dim(i).unwrap_static())
                .collect();
            // Row-major input strides: stride[d] = product of dims[d+1..].
            let mut in_strides_full = vec![1usize; in_rank];
            for d in (0..in_rank.saturating_sub(1)).rev() {
                in_strides_full[d] = in_strides_full[d + 1] * in_dims[d + 1];
            }
            let out_dims: Vec<u32> = perm.iter().map(|&p| in_dims[p] as u32).collect();
            let in_strides: Vec<u32> = perm.iter().map(|&p| in_strides_full[p] as u32).collect();
            let in_total = in_dims.iter().product::<usize>() as u32;
            let src = node_offset(arena, node.inputs[0]);
            let dst = node_offset(arena, node.id);
            let elem_bytes = node.shape.dtype().size_bytes() as u8;
            match node.shape.dtype() {
                rlx_ir::DType::F64 => Thunk::TransposeF64 {
                    src,
                    dst,
                    in_total,
                    out_dims,
                    in_strides,
                },
                _ => Thunk::Transpose {
                    src,
                    dst,
                    in_total,
                    out_dims,
                    in_strides,
                    elem_bytes,
                },
            }
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_scatter_add(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ScatterAdd = &node.op else {
        unreachable!()
    };
    {
        // updates: [num_updates, ...trailing], indices: [num_updates],
        // output: [out_dim, ...trailing]
        let upd_shape = &graph.node(node.inputs[0]).shape;
        let out_shape = &node.shape;
        let num_updates = upd_shape.dim(0).unwrap_static();
        let out_dim = out_shape.dim(0).unwrap_static();
        let trailing: usize = (1..out_shape.rank())
            .map(|i| out_shape.dim(i).unwrap_static())
            .product::<usize>()
            .max(1);
        Thunk::ScatterAdd {
            updates: node_offset(arena, node.inputs[0]),
            indices: node_offset(arena, node.inputs[1]),
            dst: node_offset(arena, node.id),
            num_updates: num_updates as u32,
            out_dim: out_dim as u32,
            trailing: trailing as u32,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_scatter_nd(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ScatterNd { reduction } = &node.op else {
        unreachable!()
    };
    let data_shape = &graph.node(node.inputs[0]).shape;
    let indices_shape = &graph.node(node.inputs[1]).shape;
    let updates_shape = &graph.node(node.inputs[2]).shape;
    let data_len = data_shape.num_elements().unwrap_or(0);
    let updates_len = updates_shape.num_elements().unwrap_or(0);
    let indices_len = indices_shape.num_elements().unwrap_or(0);
    let indices_i64 = u8::from(indices_shape.dtype() == rlx_ir::DType::I64);
    Thunk::ScatterNd {
        data: node_offset(arena, node.inputs[0]),
        indices: node_offset(arena, node.inputs[1]),
        updates: node_offset(arena, node.inputs[2]),
        dst: node_offset(arena, node.id),
        data_shape: (0..data_shape.rank())
            .map(|i| data_shape.dim(i).unwrap_static() as u32)
            .collect(),
        indices_shape: (0..indices_shape.rank())
            .map(|i| indices_shape.dim(i).unwrap_static() as u32)
            .collect(),
        data_len: data_len as u32,
        updates_len: updates_len as u32,
        indices_len: indices_len as u32,
        indices_i64,
        reduction: *reduction,
    }
}

fn shape_u32s(shape: &rlx_ir::Shape) -> Vec<u32> {
    (0..shape.rank())
        .map(|i| shape.dim(i).unwrap_static() as u32)
        .collect()
}

#[allow(unused_variables)]
pub(crate) fn compile_scatter_elements(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ScatterElements { axis, reduction } = &node.op else {
        unreachable!()
    };
    let data_shape = &graph.node(node.inputs[0]).shape;
    let indices_shape = &graph.node(node.inputs[1]).shape;
    let updates_shape = &graph.node(node.inputs[2]).shape;
    Thunk::ScatterElements {
        data: node_offset(arena, node.inputs[0]),
        indices: node_offset(arena, node.inputs[1]),
        updates: node_offset(arena, node.inputs[2]),
        dst: node_offset(arena, node.id),
        data_shape: shape_u32s(data_shape),
        data_len: data_shape.num_elements().unwrap_or(0) as u32,
        updates_len: updates_shape.num_elements().unwrap_or(0) as u32,
        indices_len: indices_shape.num_elements().unwrap_or(0) as u32,
        indices_i64: u8::from(indices_shape.dtype() == rlx_ir::DType::I64),
        axis: *axis,
        reduction: *reduction,
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_gather_nd(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::GatherNd { batch_dims } = &node.op else {
        unreachable!()
    };
    let data_shape = &graph.node(node.inputs[0]).shape;
    let indices_shape = &graph.node(node.inputs[1]).shape;
    Thunk::GatherNd {
        data: node_offset(arena, node.inputs[0]),
        indices: node_offset(arena, node.inputs[1]),
        dst: node_offset(arena, node.id),
        data_shape: shape_u32s(data_shape),
        indices_shape: shape_u32s(indices_shape),
        data_len: data_shape.num_elements().unwrap_or(0) as u32,
        indices_len: indices_shape.num_elements().unwrap_or(0) as u32,
        out_len: node.shape.num_elements().unwrap_or(0) as u32,
        indices_i64: u8::from(indices_shape.dtype() == rlx_ir::DType::I64),
        batch_dims: *batch_dims,
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_gather_elements(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::GatherElements { axis } = &node.op else {
        unreachable!()
    };
    let data_shape = &graph.node(node.inputs[0]).shape;
    let indices_shape = &graph.node(node.inputs[1]).shape;
    Thunk::GatherElements {
        data: node_offset(arena, node.inputs[0]),
        indices: node_offset(arena, node.inputs[1]),
        dst: node_offset(arena, node.id),
        data_shape: shape_u32s(data_shape),
        indices_shape: shape_u32s(indices_shape),
        data_len: data_shape.num_elements().unwrap_or(0) as u32,
        indices_len: indices_shape.num_elements().unwrap_or(0) as u32,
        out_len: node.shape.num_elements().unwrap_or(0) as u32,
        indices_i64: u8::from(indices_shape.dtype() == rlx_ir::DType::I64),
        axis: *axis,
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_compare(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Compare(cmp) = &node.op else {
        unreachable!()
    };
    {
        let len = node.shape.num_elements().unwrap();
        let lhs_n = graph.node(node.inputs[0]).shape.num_elements().unwrap();
        let rhs_n = graph.node(node.inputs[1]).shape.num_elements().unwrap();
        let lhs_scalar = lhs_n == 1 && len > 1;
        let rhs_scalar = rhs_n == 1 && len > 1;
        let in_dtype = graph.node(node.inputs[0]).shape.dtype();
        let inputs_i64 = u8::from(in_dtype == rlx_ir::DType::I64);
        Thunk::Compare {
            lhs: node_offset(arena, node.inputs[0]),
            rhs: node_offset(arena, node.inputs[1]),
            dst: node_offset(arena, node.id),
            len: len as u32,
            op: *cmp,
            inputs_i64,
            inputs_elem_bytes: in_dtype.size_bytes() as u8,
            dst_elem_bytes: node.shape.dtype().size_bytes() as u8,
            lhs_scalar,
            rhs_scalar,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_where(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Where = &node.op else { unreachable!() };
    {
        let len = node.shape.num_elements().unwrap();
        let cn = graph.node(node.inputs[0]).shape.num_elements().unwrap();
        let tn = graph.node(node.inputs[1]).shape.num_elements().unwrap();
        let fn_ = graph.node(node.inputs[2]).shape.num_elements().unwrap();
        let cond_scalar = cn == 1 && len > 1;
        let true_scalar = tn == 1 && len > 1;
        let false_scalar = fn_ == 1 && len > 1;
        let elem_bytes = node.shape.dtype().size_bytes() as u8;
        let cond_elem_bytes = graph.node(node.inputs[0]).shape.dtype().size_bytes() as u8;
        Thunk::Where {
            cond: node_offset(arena, node.inputs[0]),
            on_true: node_offset(arena, node.inputs[1]),
            on_false: node_offset(arena, node.inputs[2]),
            dst: node_offset(arena, node.id),
            len: len as u32,
            elem_bytes,
            cond_elem_bytes,
            cond_scalar,
            true_scalar,
            false_scalar,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_fma(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Fma = &node.op else { unreachable!() };
    {
        let len = node.shape.num_elements().unwrap();
        Thunk::Fma {
            a: node_offset(arena, node.inputs[0]),
            b: node_offset(arena, node.inputs[1]),
            c: node_offset(arena, node.inputs[2]),
            dst: node_offset(arena, node.id),
            len: len as u32,
            elem_bytes: node.shape.dtype().size_bytes() as u8,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_relu_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ReluBackward = &node.op else {
        unreachable!()
    };
    {
        let len: usize = (0..node.shape.rank())
            .map(|i| node.shape.dim(i).unwrap_static())
            .product();
        let x = node_offset(arena, node.inputs[0]);
        let dy = node_offset(arena, node.inputs[1]);
        let dx = node_offset(arena, node.id);
        match node.shape.dtype() {
            rlx_ir::DType::F64 => Thunk::ReluBackwardF64 {
                x,
                dy,
                dx,
                len: len as u32,
            },
            _ => Thunk::ReluBackward {
                x,
                dy,
                dx,
                len: len as u32,
            },
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_activation_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ActivationBackward { kind } = &node.op else {
        unreachable!()
    };
    {
        let len: usize = (0..node.shape.rank())
            .map(|i| node.shape.dim(i).unwrap_static())
            .product();
        let x = node_offset(arena, node.inputs[0]);
        let dy = node_offset(arena, node.inputs[1]);
        let dx = node_offset(arena, node.id);
        match node.shape.dtype() {
            rlx_ir::DType::F64 => Thunk::ActivationBackwardF64 {
                x,
                dy,
                dx,
                len: len as u32,
                kind: *kind,
            },
            _ => Thunk::ActivationBackward {
                x,
                dy,
                dx,
                len: len as u32,
                kind: *kind,
            },
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_gather_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::GatherBackward { .. } = &node.op else {
        unreachable!()
    };
    {
        let dy_shape = &graph.node(node.inputs[0]).shape;
        let idx_shape = &graph.node(node.inputs[1]).shape;
        let out_shape = &node.shape;
        let rank = out_shape.rank();
        let axis = match &node.op {
            Op::GatherBackward { axis } => *axis,
            _ => 0,
        };
        let axis_u = if axis < 0 {
            (rank as i32 + axis) as usize
        } else {
            axis as usize
        };
        let outer: usize = (0..axis_u)
            .map(|i| dy_shape.dim(i).unwrap_static())
            .product::<usize>()
            .max(1);
        let num_idx = idx_shape.dim(axis_u).unwrap_static();
        let trailing: usize = (axis_u + 1..dy_shape.rank())
            .map(|i| dy_shape.dim(i).unwrap_static())
            .product::<usize>()
            .max(1);
        let axis_dim = out_shape.dim(axis_u).unwrap_static();
        Thunk::GatherBackward {
            dy: node_offset(arena, node.inputs[0]),
            indices: node_offset(arena, node.inputs[1]),
            dst: node_offset(arena, node.id),
            outer: outer as u32,
            axis_dim: axis_dim as u32,
            num_idx: num_idx as u32,
            trailing: trailing as u32,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_concat(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Concat { axis } = &node.op else {
        unreachable!()
    };
    {
        // Compute outer/inner from the OUTPUT shape: all inputs share
        // the same shape except along `axis`. The output's leading
        // and trailing dims match.
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
        let total_axis = out_shape.dim(*axis).unwrap_static();
        let inputs: Vec<(usize, u32, u32)> = node
            .inputs
            .iter()
            .map(|&in_id| {
                let in_shape = &graph.node(in_id).shape;
                let in_axis = concat_axis_extent(in_shape, *axis, rank);
                let in_numel = in_shape.num_elements().unwrap_or(0) as u32;
                (node_offset(arena, in_id), in_axis as u32, in_numel)
            })
            .collect();
        let dst = node_offset(arena, node.id);
        // The `Concat` thunk copies 4-byte (f32-sized) elements; `ConcatF64`
        // copies 8-byte elements. Route by element SIZE, not just F64 — an I64
        // tensor (8-byte) sent through the 4-byte path reads/writes half the
        // bytes per element, so it copies only the first half of each input and
        // leaves the tail zero (e.g. `Tile([9..0], 3)` on i64 rel-shift indices
        // → `[9,8,7,6,5]×3 + zeros`). The 8-byte copy is a pure byte move, so
        // reinterpreting i64 as f64 for the copy is bit-preserving.
        match out_shape.dtype().size_bytes() {
            8 => Thunk::ConcatF64 {
                dst,
                outer: outer as u32,
                inner: inner as u32,
                total_axis: total_axis as u32,
                inputs,
            },
            _ => Thunk::Concat {
                dst,
                outer: outer as u32,
                inner: inner as u32,
                total_axis: total_axis as u32,
                inputs,
            },
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_elementwise_region(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ElementwiseRegion {
        chain,
        scalar_input_mask,
        input_modulus,
        prologue,
        ..
    } = &node.op
    else {
        unreachable!()
    };
    {
        // The scalar interpreter handles plain chains; prologue (resize)
        // regions are GPU-only and never reach here on the CPU path
        // (the graphfused fusion options disable prologue/FK fusion).
        if *prologue != rlx_ir::op::RegionPrologue::None {
            Thunk::Nop
        } else {
            let input_offs: Vec<usize> = node
                .inputs
                .iter()
                .map(|&id| node_offset(arena, id))
                .collect();
            Thunk::ElementwiseRegion {
                dst: node_offset(arena, node.id),
                len: node.shape.num_elements().unwrap_or(0) as u32,
                input_offs,
                chain: chain.clone(),
                scalar_input_mask: *scalar_input_mask,
                input_modulus: *input_modulus,
            }
        }
    }
}

pub(crate) fn get_len(graph: &Graph, id: NodeId) -> usize {
    graph.node(id).shape.num_elements().unwrap_or(0)
}

/// Static `usize` dims of a node's shape, or empty if any dim is dynamic.
pub(crate) fn get_static_dims(graph: &Graph, id: NodeId) -> Vec<usize> {
    let dims = graph.node(id).shape.dims();
    let mut out = Vec::with_capacity(dims.len());
    for d in dims {
        if let Some(s) = match d {
            rlx_ir::Dim::Static(s) => Some(*s),
            _ => None,
        } {
            out.push(s);
        } else {
            return Vec::new();
        }
    }
    out
}

/// Extent along `axis` for a concat input, treating leading implicit 1s when
/// `input.rank() < output.rank()` (ONNX / numpy concat broadcast rules).
pub(crate) fn concat_axis_extent(input: &rlx_ir::Shape, axis: usize, out_rank: usize) -> usize {
    let in_rank = input.rank();
    if axis >= out_rank {
        return 1;
    }
    if axis < in_rank {
        input.dim(axis).unwrap_static()
    } else {
        1
    }
}

pub(crate) fn broadcast_src_index(src_idx: usize, in_len: usize) -> usize {
    if in_len == 0 { 0 } else { src_idx % in_len }
}

/// Copy `inp` into row-strided `out` for a Concat input, with outer-dim
/// broadcasting. The body is dtype-agnostic (only `Copy` + slice ops), so the
/// f32/f64 variants are generated from one template rather than copy-pasted.
macro_rules! concat_copy_rows {
    ($name:ident, $t:ty) => {
        pub(crate) fn $name(
            out: &mut [$t],
            inp: &[$t],
            outer: usize,
            copy_per_row: usize,
            row_stride: usize,
            dst_col_off: usize,
            in_numel: usize,
        ) {
            let need = outer.saturating_mul(copy_per_row.max(1));
            let broadcast_outer = in_numel < need;
            let out_ptr = out.as_mut_ptr() as usize;
            let inp_ptr = inp.as_ptr() as usize;
            let inp_len = inp.len();
            let run = |o0: usize, o1: usize| {
                let out = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut $t, out.len()) };
                let inp = unsafe { std::slice::from_raw_parts(inp_ptr as *const $t, inp_len) };
                for o in o0..o1 {
                    let dst_row_start = o * row_stride + dst_col_off;
                    if broadcast_outer {
                        if in_numel == 1 {
                            if copy_per_row == 1 {
                                out[dst_row_start] = inp[0];
                            } else {
                                out[dst_row_start..dst_row_start + copy_per_row].fill(inp[0]);
                            }
                        } else if copy_per_row <= inp.len() {
                            out[dst_row_start..dst_row_start + copy_per_row]
                                .copy_from_slice(&inp[..copy_per_row]);
                        } else if !inp.is_empty() {
                            out[dst_row_start..dst_row_start + copy_per_row].fill(inp[0]);
                        }
                    } else {
                        let src_row_start = o * copy_per_row;
                        out[dst_row_start..dst_row_start + copy_per_row]
                            .copy_from_slice(&inp[src_row_start..src_row_start + copy_per_row]);
                    }
                }
            };
            // F5 DiT concatenates large mel/text streams — parallelize outer rows.
            if outer >= 8
                && copy_per_row >= 16
                && crate::pool::num_threads() > 1
                && crate::pool::should_parallelize(outer.saturating_mul(copy_per_row))
            {
                crate::pool::par_for(outer, 1, &|off, cnt| run(off, off + cnt));
            } else {
                run(0, outer);
            }
        }
    };
}

concat_copy_rows!(concat_copy_rows_f32, f32);
concat_copy_rows!(concat_copy_rows_f64, f64);

/// NumPy-style broadcast strides for one operand into the flat output
/// buffer. Returns a length-`out_dims.len()` `Vec<u32>` where entry
/// `d` is `0` if the input is size-1 (broadcast) at output dim `d`
/// (after left-padding with size-1 to match ranks), otherwise the
/// natural row-major stride into the *input* buffer.
///
/// Caller iterates output flat index `i` → output coords (row-major)
/// → input flat index = dot(coords, strides). The result is correct
/// for any broadcast pattern (scalar, last-axis, middle-axis,
/// bidirectional).
/// True when `rhs_dims` describes a *trailing* broadcast of `out_dims`
/// — i.e. every rhs dim either equals the corresponding output dim
/// (counting from the right) or rhs is shorter (left-padded with 1s).
/// Mid-shape singletons (e.g. rhs `[a, b, 1, d]` into out `[a, b, c, d]`
/// where `c > 1`) are NOT trailing broadcasts and require the
/// shape-aware `BinaryFull` slow path — `BiasAdd`'s linear bias-replicated
/// kernel silently miscomputes them.
pub(crate) fn is_trailing_bias_broadcast(
    rhs_dims: &[rlx_ir::Dim],
    out_dims: &[rlx_ir::Dim],
) -> bool {
    if rhs_dims.len() > out_dims.len() {
        return false;
    }
    let off = out_dims.len() - rhs_dims.len();
    for i in 0..rhs_dims.len() {
        let r = match rhs_dims[i] {
            rlx_ir::Dim::Static(n) => n,
            _ => return false,
        };
        let o = match out_dims[off + i] {
            rlx_ir::Dim::Static(n) => n,
            _ => return false,
        };
        if r != o {
            return false;
        }
    }
    true
}

pub(crate) fn broadcast_strides(in_dims: &[usize], out_dims: &[usize]) -> Vec<u32> {
    let r_out = out_dims.len();
    let r_in = in_dims.len();
    assert!(
        r_in <= r_out,
        "broadcast: input rank {r_in} > output rank {r_out}"
    );
    let pad = r_out - r_in;
    let mut strides = vec![0u32; r_out];
    let mut acc: usize = 1;
    for d in (0..r_out).rev() {
        let in_size = if d < pad { 1 } else { in_dims[d - pad] };
        if in_size == 1 {
            strides[d] = 0;
        } else {
            assert_eq!(
                in_size, out_dims[d],
                "broadcast: input dim {in_size} doesn't match output dim {} at axis {d}",
                out_dims[d]
            );
            strides[d] = acc as u32;
            acc *= in_size;
        }
    }
    strides
}

#[inline(always)]
pub(crate) fn exec_elementwise_region(t: &Thunk, base: *mut u8) {
    let Thunk::ElementwiseRegion {
        dst,
        len,
        input_offs,
        chain,
        scalar_input_mask,
        input_modulus,
    } = t
    else {
        unreachable!()
    };
    {
        let len = *len as usize;
        if !chain.is_empty() && len > 0 {
            let base_addr = base as usize;
            let dst = *dst;
            let scalar_mask = *scalar_input_mask;
            // Each output element is independent → fan over the range.
            let eval = |gid: usize| {
                let v = region_eval_elem(
                    gid,
                    base_addr as *const u8,
                    input_offs,
                    chain,
                    scalar_mask,
                    input_modulus,
                );
                unsafe {
                    *((base_addr as *mut u8).add(dst) as *mut f32).add(gid) = v;
                }
            };
            if fast_conv_enabled() && crate::pool::should_parallelize(len) {
                crate::pool::par_for(len, crate::pool::chunk_floor(len), &|off, cnt| {
                    for gid in off..off + cnt {
                        eval(gid);
                    }
                });
            } else {
                for gid in 0..len {
                    eval(gid);
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_transpose_f64(t: &Thunk, base: *mut u8) {
    let Thunk::TransposeF64 {
        src,
        dst,
        in_total,
        out_dims,
        in_strides,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        let inp = sl_f64(*src, base, *in_total as usize);
        let out_total: usize = out_dims.iter().map(|d| *d as usize).product();
        let out = sl_mut_f64(*dst, base, out_total);
        transpose_walk_f64(inp, out, out_dims, in_strides);
    }
}

#[inline(always)]
pub(crate) fn exec_activation_f64(t: &Thunk, base: *mut u8) {
    let Thunk::ActivationF64 {
        src,
        dst,
        len,
        kind,
    } = t
    else {
        unreachable!()
    };
    {
        let len = *len as usize;
        unsafe {
            let inp = sl_f64(*src, base, len);
            let out = sl_mut_f64(*dst, base, len);
            apply_activation_f64(inp, out, *kind);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_binary_full_f64(t: &Thunk, base: *mut u8) {
    let Thunk::BinaryFullF64 {
        lhs,
        rhs,
        dst,
        len,
        lhs_len,
        rhs_len,
        op,
        out_dims_bcast,
        bcast_lhs_strides,
        bcast_rhs_strides,
    } = t
    else {
        unreachable!()
    };
    {
        let len = *len as usize;
        let lhs_len = *lhs_len as usize;
        let rhs_len = *rhs_len as usize;
        unsafe {
            let l = sl_f64(*lhs, base, lhs_len);
            let r = sl_f64(*rhs, base, rhs_len);
            let d = sl_mut_f64(*dst, base, len);
            if lhs_len == len && rhs_len == len {
                for i in 0..len {
                    d[i] = binary_op_f64(*op, l[i], r[i]);
                }
            } else if !out_dims_bcast.is_empty() {
                // Shape-aware broadcast path: correct for
                // arbitrary NumPy-style broadcasts including
                // bidirectional `[N,1] op [1,S]`.
                let rank = out_dims_bcast.len();
                let mut coords = vec![0u32; rank];
                for i in 0..len {
                    let mut rem = i;
                    for ax in (0..rank).rev() {
                        let sz = out_dims_bcast[ax] as usize;
                        coords[ax] = (rem % sz) as u32;
                        rem /= sz;
                    }
                    let mut li: usize = 0;
                    let mut ri: usize = 0;
                    for ax in 0..rank {
                        li += coords[ax] as usize * bcast_lhs_strides[ax] as usize;
                        ri += coords[ax] as usize * bcast_rhs_strides[ax] as usize;
                    }
                    d[i] = binary_op_f64(*op, l[li], r[ri]);
                }
            } else {
                // Fallback: legacy modulo path (preserved for
                // dynamic-shape graphs where strides can't be
                // precomputed). Only correct for scalar /
                // last-axis broadcast.
                for i in 0..len {
                    d[i] = binary_op_f64(*op, l[i % lhs_len], r[i % rhs_len]);
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_binary_full_c64(t: &Thunk, base: *mut u8) {
    let Thunk::BinaryFullC64 {
        lhs,
        rhs,
        dst,
        len,
        lhs_len,
        rhs_len,
        op,
        out_dims_bcast,
        bcast_lhs_strides,
        bcast_rhs_strides,
    } = t
    else {
        unreachable!()
    };
    {
        // Complex element layout: [re_0, im_0, re_1, im_1, ...]
        // Underlying f32 buffer length is 2·N (N = complex
        // element count). All offsets are byte offsets; the
        // `sl` helper reads as f32 starting at the byte
        // offset, so f32-length = 2·complex-len.
        let n_out = *len as usize;
        let n_l = *lhs_len as usize;
        let n_r = *rhs_len as usize;
        unsafe {
            let l = sl(*lhs, base, 2 * n_l);
            let r = sl(*rhs, base, 2 * n_r);
            let d = sl_mut(*dst, base, 2 * n_out);
            let do_c64 = |a_re: f32, a_im: f32, b_re: f32, b_im: f32| -> (f32, f32) {
                match op {
                    BinaryOp::Add => (a_re + b_re, a_im + b_im),
                    BinaryOp::Sub => (a_re - b_re, a_im - b_im),
                    BinaryOp::Mul => (a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re),
                    BinaryOp::Div => {
                        let denom = b_re * b_re + b_im * b_im;
                        (
                            (a_re * b_re + a_im * b_im) / denom,
                            (a_im * b_re - a_re * b_im) / denom,
                        )
                    }
                    BinaryOp::Max | BinaryOp::Min | BinaryOp::Pow => {
                        unreachable!("C64 max/min/pow rejected at lowering")
                    }
                }
            };
            if n_l == n_out && n_r == n_out {
                for i in 0..n_out {
                    let (re, im) = do_c64(l[2 * i], l[2 * i + 1], r[2 * i], r[2 * i + 1]);
                    d[2 * i] = re;
                    d[2 * i + 1] = im;
                }
            } else if !out_dims_bcast.is_empty() {
                // Strided complex broadcast: strides are in
                // *complex element* units; multiply by 2 when
                // indexing into the f32 buffer.
                let rank = out_dims_bcast.len();
                let mut coords = vec![0u32; rank];
                for i in 0..n_out {
                    let mut rem = i;
                    for ax in (0..rank).rev() {
                        let sz = out_dims_bcast[ax] as usize;
                        coords[ax] = (rem % sz) as u32;
                        rem /= sz;
                    }
                    let mut li: usize = 0;
                    let mut ri: usize = 0;
                    for ax in 0..rank {
                        li += coords[ax] as usize * bcast_lhs_strides[ax] as usize;
                        ri += coords[ax] as usize * bcast_rhs_strides[ax] as usize;
                    }
                    let (re, im) = do_c64(l[2 * li], l[2 * li + 1], r[2 * ri], r[2 * ri + 1]);
                    d[2 * i] = re;
                    d[2 * i + 1] = im;
                }
            } else {
                // Modulo fallback (scalar / last-axis broadcast).
                for i in 0..n_out {
                    let li = if n_l == 1 { 0 } else { i % n_l };
                    let ri = if n_r == 1 { 0 } else { i % n_r };
                    let (re, im) = do_c64(l[2 * li], l[2 * li + 1], r[2 * ri], r[2 * ri + 1]);
                    d[2 * i] = re;
                    d[2 * i + 1] = im;
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_activation_c64(t: &Thunk, base: *mut u8) {
    let Thunk::ActivationC64 {
        src,
        dst,
        len,
        kind,
    } = t
    else {
        unreachable!()
    };
    {
        let n = *len as usize;
        unsafe {
            let s = sl(*src, base, 2 * n);
            let d = sl_mut(*dst, base, 2 * n);
            for i in 0..n {
                let a = s[2 * i];
                let b = s[2 * i + 1];
                let (re, im) = match kind {
                    Activation::Neg => (-a, -b),
                    Activation::Exp => {
                        // exp(a + bi) = e^a · (cos b + i·sin b)
                        let ea = a.exp();
                        (ea * b.cos(), ea * b.sin())
                    }
                    Activation::Log => {
                        // log(z) = log|z| + i·arg(z), principal branch
                        let r = (a * a + b * b).sqrt();
                        (r.ln(), b.atan2(a))
                    }
                    Activation::Sqrt => {
                        // sqrt(a+bi) = sqrt((|z|+a)/2) + sign(b)·i·sqrt((|z|-a)/2)
                        // Principal branch; for b == 0 and a < 0 returns +i·sqrt(|a|).
                        let r = (a * a + b * b).sqrt();
                        let re = ((r + a) * 0.5).max(0.0).sqrt();
                        let im_mag = ((r - a) * 0.5).max(0.0).sqrt();
                        let im = if b >= 0.0 { im_mag } else { -im_mag };
                        (re, im)
                    }
                    _ => unreachable!("non-C64 activation kind survived lowering"),
                };
                d[2 * i] = re;
                d[2 * i + 1] = im;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_gather(t: &Thunk, base: *mut u8) {
    let Thunk::Gather {
        table,
        table_len,
        idx,
        dst,
        num_idx,
        trailing,
        idx_i64,
        table_bytes,
    } = t
    else {
        unreachable!()
    };
    {
        let (ni, tr) = (*num_idx as usize, *trailing as usize);
        let rows = *table_len as usize / tr.max(1);
        unsafe {
            if *table_bytes == 8 {
                let tab = sl_i64(*table, base, *table_len as usize);
                let out = sl_mut_i64(*dst, base, ni * tr);
                if *idx_i64 != 0 {
                    let ids = sl_i64(*idx, base, ni);
                    for i in 0..ni {
                        let row = ids[i].max(0) as usize;
                        if row < rows {
                            out[i * tr..(i + 1) * tr]
                                .copy_from_slice(&tab[row * tr..(row + 1) * tr]);
                        }
                    }
                } else {
                    let ids = sl(*idx, base, ni);
                    for i in 0..ni {
                        let row = ids[i] as usize;
                        if row < rows {
                            out[i * tr..(i + 1) * tr]
                                .copy_from_slice(&tab[row * tr..(row + 1) * tr]);
                        }
                    }
                }
            } else {
                let tab = sl(*table, base, *table_len as usize);
                let out = sl_mut(*dst, base, ni * tr);
                if *idx_i64 != 0 {
                    let ids = sl_i64(*idx, base, ni);
                    for i in 0..ni {
                        let row = ids[i].max(0) as usize;
                        if row < rows {
                            out[i * tr..(i + 1) * tr]
                                .copy_from_slice(&tab[row * tr..(row + 1) * tr]);
                        }
                    }
                } else {
                    let ids = sl(*idx, base, ni);
                    for i in 0..ni {
                        let row = ids[i] as usize;
                        if row < rows {
                            out[i * tr..(i + 1) * tr]
                                .copy_from_slice(&tab[row * tr..(row + 1) * tr]);
                        }
                    }
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_activation_in_place(t: &Thunk, base: *mut u8) {
    let Thunk::ActivationInPlace { data, len, act } = t else {
        unreachable!()
    };
    {
        let len = *len as usize;
        unsafe {
            let d = sl_mut(*data, base, len);
            // Serial scalar transcendentals were a per-op cost for models heavy in
            // tanh/sigmoid/exp (LSTM/GRU gates, vocoders). Apply elementwise in
            // parallel for large tensors (bit-exact — elements independent), mirroring
            // the already-parallel Gelu/Silu kernels. `should_parallelize` guards the
            // small case so tiny activations stay serial (no dispatch overhead).
            let par = crate::pool::should_parallelize(len);
            macro_rules! apply {
                ($f:expr) => {{
                    let f = $f;
                    if par {
                        use rayon::prelude::*;
                        d.par_iter_mut().for_each(|v| *v = f(*v));
                    } else {
                        for v in d.iter_mut() {
                            *v = f(*v);
                        }
                    }
                }};
            }
            match act {
                Activation::Gelu => crate::kernels::par_gelu_inplace(d),
                Activation::GeluApprox => crate::kernels::par_gelu_approx_inplace(d),
                Activation::Silu => crate::kernels::par_silu_inplace(d),
                Activation::Relu => apply!(|x: f32| x.max(0.0)),
                Activation::Sigmoid => apply!(|x: f32| 1.0 / (1.0 + (-x).exp())),
                Activation::Tanh => apply!(|x: f32| x.tanh()),
                Activation::Exp => apply!(|x: f32| x.exp()),
                Activation::Log => apply!(|x: f32| x.ln()),
                Activation::Sqrt => apply!(|x: f32| x.sqrt()),
                Activation::Rsqrt => apply!(|x: f32| 1.0 / x.sqrt()),
                Activation::Neg => apply!(|x: f32| -x),
                Activation::Abs => apply!(|x: f32| x.abs()),
                Activation::Round => apply!(|x: f32| x.round()),
                Activation::Sin => apply!(|x: f32| x.sin()),
                Activation::Cos => apply!(|x: f32| x.cos()),
                Activation::Tan => apply!(|x: f32| x.tan()),
                Activation::Atan => apply!(|x: f32| x.atan()),
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_concat(t: &Thunk, base: *mut u8) {
    let Thunk::Concat {
        dst,
        outer,
        inner,
        total_axis,
        inputs,
    } = t
    else {
        unreachable!()
    };
    {
        let outer = *outer as usize;
        let inner = *inner as usize;
        let total_axis = *total_axis as usize;
        let row_stride = total_axis * inner;
        let out_total = outer * row_stride;
        unsafe {
            let out = sl_mut(*dst, base, out_total);
            let mut cum: usize = 0;
            for (src_off, in_axis, in_numel) in inputs {
                let in_axis = *in_axis as usize;
                let copy_per_row = in_axis * inner;
                let dst_col_off = cum * inner;
                let inp = sl(*src_off, base, (*in_numel as usize).max(1));
                concat_copy_rows_f32(
                    out,
                    inp,
                    outer,
                    copy_per_row,
                    row_stride,
                    dst_col_off,
                    *in_numel as usize,
                );
                cum += in_axis;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_concat_f64(t: &Thunk, base: *mut u8) {
    let Thunk::ConcatF64 {
        dst,
        outer,
        inner,
        total_axis,
        inputs,
    } = t
    else {
        unreachable!()
    };
    {
        let outer = *outer as usize;
        let inner = *inner as usize;
        let total_axis = *total_axis as usize;
        let row_stride = total_axis * inner;
        let out_total = outer * row_stride;
        unsafe {
            let out = sl_mut_f64(*dst, base, out_total);
            let mut cum: usize = 0;
            for (src_off, in_axis, in_numel) in inputs {
                let in_axis = *in_axis as usize;
                let copy_per_row = in_axis * inner;
                let dst_col_off = cum * inner;
                let inp = sl_f64(*src_off, base, (*in_numel as usize).max(1));
                concat_copy_rows_f64(
                    out,
                    inp,
                    outer,
                    copy_per_row,
                    row_stride,
                    dst_col_off,
                    *in_numel as usize,
                );
                cum += in_axis;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_scatter_add(t: &Thunk, base: *mut u8) {
    let Thunk::ScatterAdd {
        updates,
        indices,
        dst,
        num_updates,
        out_dim,
        trailing,
    } = t
    else {
        unreachable!()
    };
    {
        let num_updates = *num_updates as usize;
        let out_dim = *out_dim as usize;
        let trailing = *trailing as usize;
        unsafe {
            let upd = sl(*updates, base, num_updates * trailing);
            let ids = sl(*indices, base, num_updates);
            let out = sl_mut(*dst, base, out_dim * trailing);
            // Zero the output first — semantics are accumulate-into-zeros.
            for v in out.iter_mut() {
                *v = 0.0;
            }
            for i in 0..num_updates {
                let row = ids[i] as usize;
                debug_assert!(row < out_dim, "ScatterAdd index out of range");
                let src_off = i * trailing;
                let dst_off = row * trailing;
                for j in 0..trailing {
                    out[dst_off + j] += upd[src_off + j];
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_scatter_nd(t: &Thunk, base: *mut u8) {
    let Thunk::ScatterNd {
        data,
        indices,
        updates,
        dst,
        data_shape,
        indices_shape,
        data_len,
        updates_len,
        indices_len,
        indices_i64,
        reduction,
    } = t
    else {
        unreachable!()
    };
    let data_shape: Vec<usize> = data_shape.iter().map(|&d| d as usize).collect();
    let indices_shape: Vec<usize> = indices_shape.iter().map(|&d| d as usize).collect();
    let data_len = *data_len as usize;
    let updates_len = *updates_len as usize;
    let indices_len = *indices_len as usize;
    unsafe {
        let data_s = sl(*data, base, data_len);
        let updates_s = sl(*updates, base, updates_len);
        let out = sl_mut(*dst, base, data_len);
        let idx_i64: Vec<i64> = if *indices_i64 != 0 {
            let ptr = base.add(*indices) as *const i64;
            std::slice::from_raw_parts(ptr, indices_len).to_vec()
        } else {
            sl(*indices, base, indices_len)
                .iter()
                .map(|&x| x as i64)
                .collect()
        };
        let (offsets, slice) =
            crate::onnx_indexing::scatter_nd_dst_offsets(&data_shape, &idx_i64, &indices_shape);
        crate::onnx_indexing::scatter_nd_into_f32(
            data_s, updates_s, out, &offsets, slice, *reduction,
        );
    }
}

fn load_indices_i64(base: *mut u8, off: usize, len: usize, as_i64: u8) -> Vec<i64> {
    unsafe {
        if as_i64 != 0 {
            let ptr = base.add(off) as *const i64;
            std::slice::from_raw_parts(ptr, len).to_vec()
        } else {
            sl(off, base, len).iter().map(|&x| x as i64).collect()
        }
    }
}

#[inline(always)]
pub(crate) fn exec_scatter_elements(t: &Thunk, base: *mut u8) {
    let Thunk::ScatterElements {
        data,
        indices,
        updates,
        dst,
        data_shape,
        data_len,
        updates_len,
        indices_len,
        indices_i64,
        axis,
        reduction,
    } = t
    else {
        unreachable!()
    };
    let data_shape: Vec<usize> = data_shape.iter().map(|&d| d as usize).collect();
    unsafe {
        let data_s = sl(*data, base, *data_len as usize);
        let updates_s = sl(*updates, base, *updates_len as usize);
        let out = sl_mut(*dst, base, *data_len as usize);
        let idx = load_indices_i64(base, *indices, *indices_len as usize, *indices_i64);
        crate::onnx_indexing::scatter_elements_f32(
            data_s,
            updates_s,
            &idx,
            out,
            &data_shape,
            *axis,
            *reduction,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_gather_nd(t: &Thunk, base: *mut u8) {
    let Thunk::GatherNd {
        data,
        indices,
        dst,
        data_shape,
        indices_shape,
        data_len,
        indices_len,
        out_len,
        indices_i64,
        batch_dims,
    } = t
    else {
        unreachable!()
    };
    let data_shape: Vec<usize> = data_shape.iter().map(|&d| d as usize).collect();
    let indices_shape: Vec<usize> = indices_shape.iter().map(|&d| d as usize).collect();
    unsafe {
        let data_s = sl(*data, base, *data_len as usize);
        let out = sl_mut(*dst, base, *out_len as usize);
        let idx = load_indices_i64(base, *indices, *indices_len as usize, *indices_i64);
        let (offsets, slice) = crate::onnx_indexing::gather_nd_src_offsets(
            &data_shape,
            &idx,
            &indices_shape,
            (*batch_dims).max(0) as usize,
        );
        for (t, &off) in offsets.iter().enumerate() {
            let d0 = t * slice;
            for j in 0..slice {
                if let (Some(&s), Some(d)) = (data_s.get(off + j), out.get_mut(d0 + j)) {
                    *d = s;
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_gather_elements(t: &Thunk, base: *mut u8) {
    let Thunk::GatherElements {
        data,
        indices,
        dst,
        data_shape,
        indices_shape,
        data_len,
        indices_len,
        out_len,
        indices_i64,
        axis,
    } = t
    else {
        unreachable!()
    };
    let data_shape: Vec<usize> = data_shape.iter().map(|&d| d as usize).collect();
    let indices_shape: Vec<usize> = indices_shape.iter().map(|&d| d as usize).collect();
    unsafe {
        let data_s = sl(*data, base, *data_len as usize);
        let out = sl_mut(*dst, base, *out_len as usize);
        let idx = load_indices_i64(base, *indices, *indices_len as usize, *indices_i64);
        crate::onnx_indexing::gather_elements_f32(
            data_s,
            &idx,
            out,
            &data_shape,
            &indices_shape,
            *axis,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_relu_backward(t: &Thunk, base: *mut u8) {
    let Thunk::ReluBackward { x, dy, dx, len } = t else {
        unreachable!()
    };
    {
        let len = *len as usize;
        unsafe {
            let xs = sl(*x, base, len);
            let dys = sl(*dy, base, len);
            let out = sl_mut(*dx, base, len);
            if fast_conv_enabled() && crate::pool::should_parallelize(len) {
                let oa = out.as_mut_ptr() as usize;
                crate::pool::par_for(len, crate::pool::chunk_floor(len), &|off, cnt| {
                    for i in off..off + cnt {
                        *((oa as *mut f32).add(i)) = if xs[i] > 0.0 { dys[i] } else { 0.0 };
                    }
                });
            } else {
                for i in 0..len {
                    out[i] = if xs[i] > 0.0 { dys[i] } else { 0.0 };
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_relu_backward_f64(t: &Thunk, base: *mut u8) {
    let Thunk::ReluBackwardF64 { x, dy, dx, len } = t else {
        unreachable!()
    };
    {
        let len = *len as usize;
        unsafe {
            let xs = sl_f64(*x, base, len);
            let dys = sl_f64(*dy, base, len);
            let out = sl_mut_f64(*dx, base, len);
            for i in 0..len {
                out[i] = if xs[i] > 0.0 { dys[i] } else { 0.0 };
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_activation_backward(t: &Thunk, base: *mut u8) {
    let Thunk::ActivationBackward {
        x,
        dy,
        dx,
        len,
        kind,
    } = t
    else {
        unreachable!()
    };
    {
        let len = *len as usize;
        unsafe {
            let xs = sl(*x, base, len);
            let dys = sl(*dy, base, len);
            let out = sl_mut(*dx, base, len);
            activation_backward_kernel(*kind, xs, dys, out);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_activation_backward_f64(t: &Thunk, base: *mut u8) {
    let Thunk::ActivationBackwardF64 {
        x,
        dy,
        dx,
        len,
        kind,
    } = t
    else {
        unreachable!()
    };
    {
        let len = *len as usize;
        unsafe {
            let xs = sl_f64(*x, base, len);
            let dys = sl_f64(*dy, base, len);
            let out = sl_mut_f64(*dx, base, len);
            activation_backward_kernel_f64(*kind, xs, dys, out);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_gather_backward(t: &Thunk, base: *mut u8) {
    let Thunk::GatherBackward {
        dy,
        indices,
        dst,
        outer,
        axis_dim,
        num_idx,
        trailing,
    } = t
    else {
        unreachable!()
    };
    {
        let (outer, axis_dim, num_idx, trailing) = (
            *outer as usize,
            *axis_dim as usize,
            *num_idx as usize,
            *trailing as usize,
        );
        unsafe {
            let dys = sl(*dy, base, outer * num_idx * trailing);
            let ids = sl(*indices, base, num_idx);
            let out = sl_mut(*dst, base, outer * axis_dim * trailing);
            for v in out.iter_mut() {
                *v = 0.0;
            }
            crate::training_bwd::gather_axis_backward(
                dys, ids, out, outer, axis_dim, num_idx, trailing,
            );
        }
    }
}

#[inline(always)]
pub(crate) fn exec_gather_axis(t: &Thunk, base: *mut u8) {
    let Thunk::GatherAxis {
        table,
        idx,
        dst,
        outer,
        axis_dim,
        num_idx,
        trailing,
        idx_i64,
        table_bytes,
    } = t
    else {
        unreachable!()
    };
    {
        let outer = *outer as usize;
        let axis_dim = *axis_dim as usize;
        let num_idx = *num_idx as usize;
        let trailing = *trailing as usize;
        unsafe {
            if *table_bytes == 8 {
                let tab = sl_i64(*table, base, outer * axis_dim * trailing);
                let out = sl_mut_i64(*dst, base, outer * num_idx * trailing);
                for o in 0..outer {
                    let tab_outer = o * axis_dim * trailing;
                    let out_outer = o * num_idx * trailing;
                    if *idx_i64 != 0 {
                        let ids = sl_i64(*idx, base, num_idx);
                        for k in 0..num_idx {
                            let row = ids[k].max(0) as usize;
                            if row < axis_dim {
                                let tab_row = tab_outer + row * trailing;
                                let out_row = out_outer + k * trailing;
                                out[out_row..out_row + trailing]
                                    .copy_from_slice(&tab[tab_row..tab_row + trailing]);
                            }
                        }
                    } else {
                        let ids = sl(*idx, base, num_idx);
                        for k in 0..num_idx {
                            let row = ids[k] as usize;
                            if row < axis_dim {
                                let tab_row = tab_outer + row * trailing;
                                let out_row = out_outer + k * trailing;
                                out[out_row..out_row + trailing]
                                    .copy_from_slice(&tab[tab_row..tab_row + trailing]);
                            }
                        }
                    }
                }
            } else {
                let tab = sl(*table, base, outer * axis_dim * trailing);
                let out = sl_mut(*dst, base, outer * num_idx * trailing);
                for o in 0..outer {
                    let tab_outer = o * axis_dim * trailing;
                    let out_outer = o * num_idx * trailing;
                    if *idx_i64 != 0 {
                        let ids = sl_i64(*idx, base, num_idx);
                        for k in 0..num_idx {
                            let row = ids[k].max(0) as usize;
                            if row < axis_dim {
                                let tab_row = tab_outer + row * trailing;
                                let out_row = out_outer + k * trailing;
                                out[out_row..out_row + trailing]
                                    .copy_from_slice(&tab[tab_row..tab_row + trailing]);
                            }
                        }
                    } else {
                        let ids = sl(*idx, base, num_idx);
                        for k in 0..num_idx {
                            let row = ids[k] as usize;
                            if row < axis_dim {
                                let tab_row = tab_outer + row * trailing;
                                let out_row = out_outer + k * trailing;
                                out[out_row..out_row + trailing]
                                    .copy_from_slice(&tab[tab_row..tab_row + trailing]);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_reverse(t: &Thunk, base: *mut u8) {
    let Thunk::Reverse {
        src,
        dst,
        dims,
        rev_mask,
        elem_bytes,
    } = t
    else {
        unreachable!()
    };
    {
        let eb = *elem_bytes as usize;
        let rank = dims.len();
        let total: usize = dims.iter().map(|&d| d as usize).product::<usize>().max(1);
        let mut strides = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1] as usize;
        }
        unsafe {
            let src_base = base.add(*src);
            let dst_base = base.add(*dst);
            for o in 0..total {
                let mut rem = o;
                let mut in_flat = 0usize;
                for ax in 0..rank {
                    let idx = rem / strides[ax];
                    rem %= strides[ax];
                    let in_idx = if rev_mask[ax] {
                        dims[ax] as usize - 1 - idx
                    } else {
                        idx
                    };
                    in_flat += in_idx * strides[ax];
                }
                std::ptr::copy_nonoverlapping(src_base.add(in_flat * eb), dst_base.add(o * eb), eb);
            }
        }
    }
}

/// Host-fallback entry for `Op::GatedDeltaNet` (Metal / unified memory).
/// When `state == 0`, uses a zero-initialized scratch state per batch item.
/// Batch-general reverse/flip (dtype-agnostic, byte-copy). Shared by the CPU
/// `Thunk::Reverse` arm and the Metal/WGPU host paths. `dims` is the row-major
/// input shape; `rev_mask[a]` flips axis `a`. `src`/`dst` are byte offsets.
pub unsafe fn execute_reverse(
    src: usize,
    dst: usize,
    dims: &[u32],
    rev_mask: &[bool],
    elem_bytes: usize,
    base: *mut u8,
) {
    let rank = dims.len();
    let total: usize = dims.iter().map(|&d| d as usize).product::<usize>().max(1);
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1] as usize;
    }
    unsafe {
        let src_base = base.add(src);
        let dst_base = base.add(dst);
        for o in 0..total {
            let mut rem = o;
            let mut in_flat = 0usize;
            for ax in 0..rank {
                let idx = rem / strides[ax];
                rem %= strides[ax];
                let in_idx = if rev_mask[ax] {
                    dims[ax] as usize - 1 - idx
                } else {
                    idx
                };
                in_flat += in_idx * strides[ax];
            }
            std::ptr::copy_nonoverlapping(
                src_base.add(in_flat * elem_bytes),
                dst_base.add(o * elem_bytes),
                elem_bytes,
            );
        }
    }
}

pub unsafe fn execute_gather_backward_f32(
    dy: usize,
    indices: usize,
    dst: usize,
    outer: u32,
    axis_dim: u32,
    num_idx: u32,
    trailing: u32,
    base: *mut u8,
) {
    let (outer, axis_dim, num_idx, trailing) = (
        outer as usize,
        axis_dim as usize,
        num_idx as usize,
        trailing as usize,
    );
    let out = sl_mut(dst, base, outer * axis_dim * trailing);
    out.fill(0.0);
    crate::training_bwd::gather_axis_backward(
        sl(dy, base, outer * num_idx * trailing),
        sl(indices, base, num_idx),
        out,
        outer,
        axis_dim,
        num_idx,
        trailing,
    );
}

/// Scalar activation for the fused-region interpreter. Matches the GPU region
/// kernel's math (e.g. tanh-approx GELU) so CPU/GPU region results agree.
#[inline]
pub(crate) fn region_activation_scalar(act: rlx_ir::op::Activation, x: f32) -> f32 {
    use rlx_ir::op::Activation as A;
    const GC: f32 = 0.797_884_6; // sqrt(2/pi)
    match act {
        A::Relu => x.max(0.0),
        A::Gelu | A::GeluApprox => 0.5 * x * (1.0 + (GC * (x + 0.044715 * x * x * x)).tanh()),
        A::Silu => x / (1.0 + (-x).exp()),
        A::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        A::Tanh => x.tanh(),
        A::Exp => x.exp(),
        A::Log => x.ln(),
        A::Sqrt => x.sqrt(),
        A::Rsqrt => 1.0 / x.sqrt(),
        A::Neg => -x,
        A::Abs => x.abs(),
        A::Sin => x.sin(),
        A::Cos => x.cos(),
        A::Tan => x.tan(),
        A::Atan => x.atan(),
        A::Round => x.round(),
    }
}

#[inline]
pub(crate) fn region_binary_scalar(op: rlx_ir::op::BinaryOp, l: f32, r: f32) -> f32 {
    use rlx_ir::op::BinaryOp as B;
    match op {
        B::Add => l + r,
        B::Sub => l - r,
        B::Mul => l * r,
        B::Div => l / r,
        B::Max => l.max(r),
        B::Min => l.min(r),
        B::Pow => l.powf(r),
    }
}

#[inline]
pub(crate) fn region_compare_scalar(op: rlx_ir::op::CmpOp, l: f32, r: f32) -> bool {
    use rlx_ir::op::CmpOp as C;
    match op {
        C::Eq => l == r,
        C::Ne => l != r,
        C::Lt => l < r,
        C::Le => l <= r,
        C::Gt => l > r,
        C::Ge => l >= r,
    }
}

/// Resolve one chain operand for output element `gid`: a previous step result
/// (`scratch[s]`) or an external input read with broadcast (`input[gid % mod]`,
/// scalar inputs read element 0). `base` is the arena byte base.
#[inline]
pub(crate) fn region_resolve_operand(
    op: &rlx_ir::op::ChainOperand,
    gid: usize,
    base: *const u8,
    input_offs: &[usize],
    scalar_mask: u32,
    modulus: &[u32; 16],
    scratch: &[f32; 32],
) -> f32 {
    use rlx_ir::op::ChainOperand as O;
    match op {
        O::Step(s) => scratch[*s as usize],
        O::Input(i) => {
            let i = *i as usize;
            let row = if (scalar_mask >> i) & 1 == 1 {
                0
            } else if modulus[i] != 0 {
                gid % modulus[i] as usize
            } else {
                gid
            };
            unsafe { *(base.add(input_offs[i]) as *const f32).add(row) }
        }
    }
}

/// Evaluate a fused element-wise chain for output element `gid`, returning the
/// final step's value (what gets written to `dst[gid]`).
#[inline]
pub(crate) fn region_eval_elem(
    gid: usize,
    base: *const u8,
    input_offs: &[usize],
    chain: &[rlx_ir::op::ChainStep],
    scalar_mask: u32,
    modulus: &[u32; 16],
) -> f32 {
    use rlx_ir::op::ChainStep as S;
    let mut scratch = [0f32; 32];
    let r = |o: &rlx_ir::op::ChainOperand, sc: &[f32; 32]| {
        region_resolve_operand(o, gid, base, input_offs, scalar_mask, modulus, sc)
    };
    for (k, step) in chain.iter().enumerate() {
        scratch[k] = match step {
            S::Activation(a, x) => region_activation_scalar(*a, r(x, &scratch)),
            S::Cast(_, x) => r(x, &scratch), // f32→f32 identity (chains are same-dtype)
            S::Binary(op, l, rr) => region_binary_scalar(*op, r(l, &scratch), r(rr, &scratch)),
            S::Compare(op, l, rr) => {
                if region_compare_scalar(*op, r(l, &scratch), r(rr, &scratch)) {
                    1.0
                } else {
                    0.0
                }
            }
            S::Where(c, t, f) => {
                if r(c, &scratch) != 0.0 {
                    r(t, &scratch)
                } else {
                    r(f, &scratch)
                }
            }
        };
    }
    scratch[chain.len() - 1]
}

// Unsafe helpers to create slices from arena base + offset
/// In-place per-element activation. Mirrors the dispatch in
/// `Thunk::ActivationInPlace`. Used by `Thunk::FusedMmBiasAct` to
/// apply the activation after `bias_add` for all non-Gelu cases.
#[inline(always)]
pub(crate) fn apply_activation_inplace(d: &mut [f32], act: rlx_ir::op::Activation) {
    use rlx_ir::op::Activation;
    match act {
        Activation::Gelu => crate::kernels::par_gelu_inplace(d),
        Activation::GeluApprox => crate::kernels::par_gelu_approx_inplace(d),
        Activation::Silu => crate::kernels::par_silu_inplace(d),
        Activation::Relu => {
            for v in d.iter_mut() {
                *v = v.max(0.0);
            }
        }
        Activation::Sigmoid => {
            for v in d.iter_mut() {
                *v = 1.0 / (1.0 + (-*v).exp());
            }
        }
        Activation::Tanh => {
            for v in d.iter_mut() {
                *v = v.tanh();
            }
        }
        Activation::Exp => {
            for v in d.iter_mut() {
                *v = v.exp();
            }
        }
        Activation::Log => {
            for v in d.iter_mut() {
                *v = v.ln();
            }
        }
        Activation::Sqrt => {
            for v in d.iter_mut() {
                *v = v.sqrt();
            }
        }
        Activation::Rsqrt => {
            for v in d.iter_mut() {
                *v = 1.0 / v.sqrt();
            }
        }
        Activation::Neg => {
            for v in d.iter_mut() {
                *v = -*v;
            }
        }
        Activation::Abs => {
            for v in d.iter_mut() {
                *v = v.abs();
            }
        }
        Activation::Round => {
            for v in d.iter_mut() {
                *v = v.round();
            }
        }
        Activation::Sin => {
            for v in d.iter_mut() {
                *v = v.sin();
            }
        }
        Activation::Cos => {
            for v in d.iter_mut() {
                *v = v.cos();
            }
        }
        Activation::Tan => {
            for v in d.iter_mut() {
                *v = v.tan();
            }
        }
        Activation::Atan => {
            for v in d.iter_mut() {
                *v = v.atan();
            }
        }
    }
}

pub(crate) fn activation_backward_kernel(
    act: rlx_ir::op::Activation,
    xs: &[f32],
    dys: &[f32],
    out: &mut [f32],
) {
    use rlx_ir::op::Activation;
    let n = xs.len();
    debug_assert_eq!(dys.len(), n);
    debug_assert_eq!(out.len(), n);
    match act {
        Activation::Relu => {
            for i in 0..n {
                out[i] = if xs[i] > 0.0 { dys[i] } else { 0.0 };
            }
        }
        Activation::Sigmoid => {
            for i in 0..n {
                let s = 1.0 / (1.0 + (-xs[i]).exp());
                out[i] = s * (1.0 - s) * dys[i];
            }
        }
        Activation::Tanh => {
            for i in 0..n {
                let t = xs[i].tanh();
                out[i] = (1.0 - t * t) * dys[i];
            }
        }
        Activation::Silu => {
            // y = x * σ(x);  dy/dx = σ(x) * (1 + x * (1 - σ(x))).
            for i in 0..n {
                let s = 1.0 / (1.0 + (-xs[i]).exp());
                out[i] = s * (1.0 + xs[i] * (1.0 - s)) * dys[i];
            }
        }
        Activation::Gelu => {
            // Exact erf-based GELU:  y = 0.5 x (1 + erf(x / √2)).
            //   dy/dx = 0.5 (1 + erf(x/√2)) + (x / √(2π)) · exp(-x²/2)
            const INV_SQRT2: f32 = 0.707_106_77;
            const INV_SQRT_2PI: f32 = 0.398_942_3;
            for i in 0..n {
                let x = xs[i];
                let phi = 0.5 * (1.0 + erf_f32(x * INV_SQRT2));
                let pdf = INV_SQRT_2PI * (-(x * x) * 0.5).exp();
                out[i] = (phi + x * pdf) * dys[i];
            }
        }
        Activation::GeluApprox => {
            // Tanh-approximation:
            //   y = 0.5 x (1 + tanh(c · (x + 0.044715 x³))) where c = √(2/π).
            const C: f32 = 0.797_884_6; // √(2/π)
            const A: f32 = 0.044_715;
            for i in 0..n {
                let x = xs[i];
                let inner = C * (x + A * x * x * x);
                let t = inner.tanh();
                let dinner = C * (1.0 + 3.0 * A * x * x);
                let d = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner;
                out[i] = d * dys[i];
            }
        }
        Activation::Exp => {
            for i in 0..n {
                out[i] = xs[i].exp() * dys[i];
            }
        }
        Activation::Log => {
            for i in 0..n {
                out[i] = dys[i] / xs[i];
            }
        }
        Activation::Sqrt => {
            // d/dx √x = 0.5 / √x — undefined at x=0; clamp to 0.
            for i in 0..n {
                let s = xs[i].sqrt();
                out[i] = if s > 0.0 { 0.5 * dys[i] / s } else { 0.0 };
            }
        }
        Activation::Rsqrt => {
            // d/dx (1/√x) = -0.5 · x^(-3/2).
            for i in 0..n {
                let s = xs[i].sqrt();
                out[i] = if s > 0.0 {
                    -0.5 * dys[i] / (xs[i] * s)
                } else {
                    0.0
                };
            }
        }
        Activation::Neg => {
            for i in 0..n {
                out[i] = -dys[i];
            }
        }
        Activation::Abs => {
            // sign(x); 0 at x=0.
            for i in 0..n {
                let x = xs[i];
                let s = if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                out[i] = s * dys[i];
            }
        }
        Activation::Round => {
            // STE: pretend the round was identity in the backward
            // pass. The round step has zero gradient almost
            // everywhere, so without this trick the optimizer can't
            // learn through it.
            out.copy_from_slice(dys);
        }
        Activation::Sin => {
            // d/dx sin(x) = cos(x).
            for i in 0..n {
                out[i] = xs[i].cos() * dys[i];
            }
        }
        Activation::Cos => {
            for i in 0..n {
                out[i] = -xs[i].sin() * dys[i];
            }
        }
        Activation::Tan => {
            // d/dx tan(x) = sec²(x) = 1 + tan²(x)
            for i in 0..n {
                let t = xs[i].tan();
                out[i] = (1.0 + t * t) * dys[i];
            }
        }
        Activation::Atan => {
            // d/dx atan(x) = 1 / (1 + x²)
            for i in 0..n {
                let x = xs[i];
                out[i] = dys[i] / (1.0 + x * x);
            }
        }
    }
}

/// f64 sibling of `activation_backward_kernel`. Same math, twice the
/// precision — used by f64 graphs where the f32 kernel reading bytes
/// as `&[f32]` would silently discard half of every f64 value.
pub(crate) fn activation_backward_kernel_f64(
    act: rlx_ir::op::Activation,
    xs: &[f64],
    dys: &[f64],
    out: &mut [f64],
) {
    use rlx_ir::op::Activation;
    let n = xs.len();
    debug_assert_eq!(dys.len(), n);
    debug_assert_eq!(out.len(), n);
    match act {
        Activation::Relu => {
            for i in 0..n {
                out[i] = if xs[i] > 0.0 { dys[i] } else { 0.0 };
            }
        }
        Activation::Sigmoid => {
            for i in 0..n {
                let s = 1.0 / (1.0 + (-xs[i]).exp());
                out[i] = s * (1.0 - s) * dys[i];
            }
        }
        Activation::Tanh => {
            for i in 0..n {
                let t = xs[i].tanh();
                out[i] = (1.0 - t * t) * dys[i];
            }
        }
        Activation::Silu => {
            for i in 0..n {
                let s = 1.0 / (1.0 + (-xs[i]).exp());
                out[i] = s * (1.0 + xs[i] * (1.0 - s)) * dys[i];
            }
        }
        Activation::Gelu | Activation::GeluApprox => {
            // Both rare on f64 paths; use the high-quality libm erf.
            const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
            const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
            for i in 0..n {
                let x = xs[i];
                let phi = 0.5 * (1.0 + erf_f64(x * INV_SQRT2));
                let pdf = INV_SQRT_2PI * (-(x * x) * 0.5).exp();
                out[i] = (phi + x * pdf) * dys[i];
            }
        }
        Activation::Exp => {
            for i in 0..n {
                out[i] = xs[i].exp() * dys[i];
            }
        }
        Activation::Log => {
            for i in 0..n {
                out[i] = dys[i] / xs[i];
            }
        }
        Activation::Sqrt => {
            for i in 0..n {
                let s = xs[i].sqrt();
                out[i] = if s > 0.0 { 0.5 * dys[i] / s } else { 0.0 };
            }
        }
        Activation::Rsqrt => {
            for i in 0..n {
                let s = xs[i].sqrt();
                out[i] = if s > 0.0 {
                    -0.5 * dys[i] / (xs[i] * s)
                } else {
                    0.0
                };
            }
        }
        Activation::Neg => {
            for i in 0..n {
                out[i] = -dys[i];
            }
        }
        Activation::Abs => {
            for i in 0..n {
                let x = xs[i];
                let s = if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                out[i] = s * dys[i];
            }
        }
        Activation::Round => {
            out.copy_from_slice(dys);
        }
        Activation::Sin => {
            for i in 0..n {
                out[i] = xs[i].cos() * dys[i];
            }
        }
        Activation::Cos => {
            for i in 0..n {
                out[i] = -xs[i].sin() * dys[i];
            }
        }
        Activation::Tan => {
            for i in 0..n {
                let t = xs[i].tan();
                out[i] = (1.0 + t * t) * dys[i];
            }
        }
        Activation::Atan => {
            for i in 0..n {
                let x = xs[i];
                out[i] = dys[i] / (1.0 + x * x);
            }
        }
    }
}

/// f64 erf via A&S 7.1.26 — same coefficients as `erf_f32`, computed
/// at f64 width. Max error ~1.5e-7 (limited by the polynomial, not the
/// arithmetic). Adequate for gradient kernels; if higher precision is
/// needed, swap in a libm dependency.
#[inline(always)]
pub(crate) fn erf_f64(x: f64) -> f64 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_43 * t - 1.453_152_03) * t) + 1.421_413_75) * t - 0.284_496_74) * t
            + 0.254_829_59)
            * t
            * (-x * x).exp();
    s * y
}

/// Cheap erf approximation (Abramowitz & Stegun 7.1.26, max error ~1.5e-7
/// over all of ℝ — plenty for f32 gradient kernels).
#[inline(always)]
pub(crate) fn erf_f32(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    s * y
}

pub(crate) fn narrow_thunk_closure(
    src: usize,
    dst: usize,
    outer: u32,
    src_stride: u32,
    dst_stride: u32,
    inner: u32,
    elem_bytes: u8,
) -> Arc<dyn Fn(*mut u8) + Send + Sync> {
    let (outer, ss, ds, inner, eb) = (
        outer as usize,
        src_stride as usize,
        dst_stride as usize,
        inner as usize,
        elem_bytes as usize,
    );
    let row_bytes = inner.saturating_mul(eb);
    let src_row_stride = ss.saturating_mul(eb);
    let dst_row_stride = ds.saturating_mul(eb);
    Arc::new(move |base: *mut u8| unsafe {
        if row_bytes == 0 || src == dst {
            return;
        }
        // Compiled-fn path has no arena length; skip if offsets look bogus.
        let arena_len = usize::MAX;
        for o in 0..outer {
            let s_off = src + o * src_row_stride;
            let d_off = dst + o * dst_row_stride;
            if s_off == d_off {
                continue;
            }
            if s_off.saturating_add(row_bytes) > arena_len
                || d_off.saturating_add(row_bytes) > arena_len
            {
                break;
            }
            std::ptr::copy_nonoverlapping(base.add(s_off), base.add(d_off), row_bytes);
        }
    })
}

/// f64 N-D index walk used by Transpose and Expand. `out_dims` gives
/// the output shape; `in_strides` gives the source stride for each
/// output dim (broadcast axes have stride 0).
pub(crate) fn transpose_walk_f64(
    inp: &[f64],
    out: &mut [f64],
    out_dims: &[u32],
    in_strides: &[u32],
) {
    let rank = out_dims.len();
    let mut idx = vec![0u32; rank];
    for o in 0..out.len() {
        let mut src_off = 0usize;
        for d in 0..rank {
            src_off += idx[d] as usize * in_strides[d] as usize;
        }
        out[o] = inp[broadcast_src_index(src_off, inp.len())];
        // Increment index — last dim varies fastest.
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < out_dims[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

/// f64 elementwise activation. Reads `inp`, writes `out`. For now
/// covers what the autodiff-emitted gradient graph needs (Neg, Exp,
/// Log, Sqrt, Rsqrt, Abs, Tanh, Sigmoid, Relu — the
/// transcendental-free subset). Approximate Gelu/Silu deferred until a
/// workload demands them at f64.
pub(crate) fn apply_activation_f64(inp: &[f64], out: &mut [f64], kind: Activation) {
    match kind {
        Activation::Neg => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = -v;
            }
        }
        Activation::Exp => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.exp();
            }
        }
        Activation::Log => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.ln();
            }
        }
        Activation::Sqrt => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.sqrt();
            }
        }
        Activation::Rsqrt => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = 1.0 / v.sqrt();
            }
        }
        Activation::Abs => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.abs();
            }
        }
        Activation::Tanh => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.tanh();
            }
        }
        Activation::Sigmoid => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = 1.0 / (1.0 + (-v).exp());
            }
        }
        Activation::Relu => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.max(0.0);
            }
        }
        Activation::Round => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.round_ties_even();
            }
        }
        Activation::Sin => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.sin();
            }
        }
        Activation::Cos => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.cos();
            }
        }
        Activation::Tan => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.tan();
            }
        }
        Activation::Atan => {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.atan();
            }
        }
        Activation::Gelu | Activation::GeluApprox | Activation::Silu => {
            panic!(
                "apply_activation_f64: {kind:?} not yet implemented at f64. \
                    Add when a workload needs it."
            );
        }
    }
}

#[inline]
pub(crate) fn binary_op_f64(op: BinaryOp, a: f64, b: f64) -> f64 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Max => a.max(b),
        BinaryOp::Min => a.min(b),
        BinaryOp::Pow => a.powf(b),
    }
}
