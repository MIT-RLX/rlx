#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

/// Compile graph into thunk schedule.
pub fn compile_thunks(graph: &Graph, arena: &Arena) -> ThunkSchedule {
    compile_thunks_with_rng(graph, arena, rlx_ir::RngOptions::default())
}

/// Compile a scan body `Graph` — `1` carry + `num_bcast` broadcast +
/// `num_xs` per-step `Op::Input`s in NodeId order, a single output (the next
/// carry) — into a [`ScanBodyPlan`]. RNG ops inside the body are not supported
/// via this path (scan bodies are deterministic recurrences).
pub fn compile_scan_body(body: &Graph, num_bcast: usize, num_xs: usize) -> ScanBodyPlan {
    let body_plan = rlx_opt::memory::plan_memory(body);
    let body_offsets: HashMap<NodeId, usize> = body_plan
        .assignments
        .iter()
        .map(|(id, slot)| (*id, slot.offset))
        .collect();
    let body_inputs: Vec<NodeId> = body
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, Op::Input { .. }))
        .map(|n| n.id)
        .collect();
    assert_eq!(
        body_inputs.len(),
        1 + num_bcast + num_xs,
        "compile_scan_body: body has {} inputs, expected {}",
        body_inputs.len(),
        1 + num_bcast + num_xs
    );
    let body_input_off = body_offsets[&body_inputs[0]];
    let carry_bytes = body
        .node(body_inputs[0])
        .shape
        .size_bytes()
        .expect("scan carry must have static shape");
    let body_output_id = *body
        .outputs
        .first()
        .expect("scan body must declare one output");
    let body_output_off = body_offsets[&body_output_id];
    let bcast_body_offs: Vec<usize> = (0..num_bcast)
        .map(|i| body_offsets[&body_inputs[1 + i]])
        .collect();
    let xs_body_offs: Vec<usize> = (0..num_xs)
        .map(|i| body_offsets[&body_inputs[1 + num_bcast + i]])
        .collect();

    let mut body_arena = crate::arena::Arena::from_plan(body_plan);
    for n in body.nodes() {
        if let Op::Constant { data } = &n.op
            && body_arena.has_buffer(n.id)
            && !data.is_empty()
        {
            match n.shape.dtype() {
                rlx_ir::DType::F64
                | rlx_ir::DType::F16
                | rlx_ir::DType::BF16
                | rlx_ir::DType::I64
                | rlx_ir::DType::I32
                | rlx_ir::DType::U32
                | rlx_ir::DType::Bool
                | rlx_ir::DType::U8
                | rlx_ir::DType::I8 => {
                    let off = body_arena.byte_offset(n.id);
                    let buf = body_arena.raw_buf_mut();
                    let nbytes = (buf.len() - off).min(data.len());
                    buf[off..off + nbytes].copy_from_slice(&data[..nbytes]);
                }
                _ => {
                    let buf = body_arena.slice_mut(n.id);
                    let n_floats = data.len() / 4;
                    let n_lim = buf.len().min(n_floats);
                    for i in 0..n_lim {
                        buf[i] = f32::from_le_bytes([
                            data[i * 4],
                            data[i * 4 + 1],
                            data[i * 4 + 2],
                            data[i * 4 + 3],
                        ]);
                    }
                }
            }
        }
    }
    let body_init = body_arena.raw_buf().to_vec();
    let body_schedule = compile_thunks(body, &body_arena);
    ScanBodyPlan {
        body: body_schedule,
        body_init,
        body_input_off,
        body_output_off,
        carry_bytes,
        bcast_body_offs,
        xs_body_offs,
    }
}

/// Compile graph into thunk schedule with explicit RNG policy.
#[allow(unused_variables)]
pub fn compile_thunks_with_rng(
    graph: &Graph,
    arena: &Arena,
    rng: rlx_ir::RngOptions,
) -> ThunkSchedule {
    // Ensure the ONNX reference kernels (ScatterND / NonZero / GatherND / Einsum /
    // Mod / …) are registered before any `Op::Custom('onnx.*')` node is compiled.
    // `CpuBackend::compile` does this, but the direct thunk-compile paths (e.g.
    // rlx-tiny-tts `compile_named` for imported ONNX graphs) bypass it — so register
    // here too, at the chokepoint every path hits. Idempotent via `Once`.
    static ONNX_KERNELS: std::sync::Once = std::sync::Once::new();
    ONNX_KERNELS.call_once(crate::onnx_ref::register_onnx_reference_kernels);
    let rng_shared = Arc::new(std::sync::RwLock::new(rng));
    let mut thunks = Vec::with_capacity(graph.len());

    // ── Auto-fuse last-two-axis Transpose → MatMul into a trans-Sgemm ──
    // Matmul backprop emits `Transpose(operand) → MatMul` for `dA=g·Bᵀ` and
    // `dB=Aᵀ·g`; the transpose is a full copy (the dominant backward cost).
    // Fold it into cblas trans flags (no copy). Guarded to the safe 2-D F32
    // case where the transpose is used only by this matmul.
    let mut use_counts: std::collections::HashMap<NodeId, usize> = std::collections::HashMap::new();
    for n in graph.nodes() {
        for &i in &n.inputs {
            *use_counts.entry(i).or_insert(0) += 1;
        }
    }
    let is_t2 = |g: &Graph, id: NodeId| -> bool {
        matches!(&g.node(id).op, Op::Transpose { perm } if perm.as_slice() == [1, 0])
            && g.node(id).shape.rank() == 2
    };
    let mut folded_transpose: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    let mut matmul_fold: std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)> =
        std::collections::HashMap::new();
    for n in graph.nodes() {
        if !matches!(n.op, Op::MatMul) {
            continue;
        }
        let (a_id, b_id) = (n.inputs[0], n.inputs[1]);
        if graph.node(a_id).shape.rank() != 2
            || graph.node(b_id).shape.rank() != 2
            || n.shape.dtype() != rlx_ir::DType::F32
        {
            continue;
        }
        let fold_a = is_t2(graph, a_id) && use_counts.get(&a_id) == Some(&1);
        let fold_b = is_t2(graph, b_id) && use_counts.get(&b_id) == Some(&1);
        if !fold_a && !fold_b {
            continue;
        }
        let (asrc, ta) = if fold_a {
            (graph.node(a_id).inputs[0], true)
        } else {
            (a_id, false)
        };
        let (bsrc, tb) = if fold_b {
            (graph.node(b_id).inputs[0], true)
        } else {
            (b_id, false)
        };
        matmul_fold.insert(n.id, (asrc, ta, bsrc, tb));
        if fold_a {
            folded_transpose.insert(a_id);
        }
        if fold_b {
            folded_transpose.insert(b_id);
        }
    }

    // ── Auto-fuse the in-graph SGD-with-momentum update into one kernel ──
    // The fully-fused training step appends, per parameter, the chain
    //   v' = Add(Mul(vel, momᶜ), grad) ;  p' = Sub(param, Mul(v', lrᶜ))
    // where `momᶜ`/`lrᶜ` are full-size constant tensors. That lowers to 4
    // `BinaryFull` ops (+2 constants) per param and dominates the step
    // (~60% of CPU time on the MLP). Collapse the whole chain into a single
    // `SgdMomentum` thunk attached to the `Sub` (p'), which also writes v''s
    // slot. Guarded to exactly this shape: both p' and v' are graph outputs,
    // the inner nodes have a single in-graph use, and the scalars are
    // uniform full-size F32 constants.
    let out_set: std::collections::HashSet<NodeId> = graph.outputs.iter().copied().collect();
    let const_scalar = |g: &Graph, id: NodeId, n: usize| -> Option<f32> {
        if let Op::Constant { data } = &g.node(id).op {
            if data.len() == n * 4 {
                return Some(f32::from_le_bytes([data[0], data[1], data[2], data[3]]));
            }
        }
        None
    };
    // p' node → (param, vel, grad, v' node, lr, mom, len)
    let mut sgd_fold: std::collections::HashMap<
        NodeId,
        (NodeId, NodeId, NodeId, NodeId, f32, f32, usize),
    > = std::collections::HashMap::new();
    let mut sgd_elim: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    for n in graph.nodes() {
        // p' = Sub(param, lr_v)
        if !matches!(n.op, Op::Binary(BinaryOp::Sub)) || n.shape.dtype() != rlx_ir::DType::F32 {
            continue;
        }
        if !out_set.contains(&n.id) {
            continue;
        }
        let len = match n.shape.num_elements() {
            Some(l) => l,
            None => continue,
        };
        let (param, lr_v) = (n.inputs[0], n.inputs[1]);
        // lr_v = Mul(v', lrᶜ), single in-graph use
        let lrv = graph.node(lr_v);
        if !matches!(lrv.op, Op::Binary(BinaryOp::Mul)) || use_counts.get(&lr_v) != Some(&1) {
            continue;
        }
        let (v_new, lr_c) = (lrv.inputs[0], lrv.inputs[1]);
        let lr = match const_scalar(graph, lr_c, len) {
            Some(v) => v,
            None => continue,
        };
        // v' = Add(v_scaled, grad), graph output, single in-graph use
        let vnew = graph.node(v_new);
        if !matches!(vnew.op, Op::Binary(BinaryOp::Add))
            || use_counts.get(&v_new) != Some(&1)
            || !out_set.contains(&v_new)
        {
            continue;
        }
        let (v_scaled, grad) = (vnew.inputs[0], vnew.inputs[1]);
        // v_scaled = Mul(vel, momᶜ), single in-graph use
        let vs = graph.node(v_scaled);
        if !matches!(vs.op, Op::Binary(BinaryOp::Mul)) || use_counts.get(&v_scaled) != Some(&1) {
            continue;
        }
        let (vel, mom_c) = (vs.inputs[0], vs.inputs[1]);
        let mom = match const_scalar(graph, mom_c, len) {
            Some(v) => v,
            None => continue,
        };
        sgd_fold.insert(n.id, (param, vel, grad, v_new, lr, mom, len));
        sgd_elim.insert(v_scaled);
        sgd_elim.insert(v_new);
        sgd_elim.insert(lr_v);
    }

    for node in graph.nodes() {
        // View ops (Reshape / same-dtype Cast / axis-0 Narrow) are aliased
        // to their parent's slot by the memory planner — no copy needed.
        // Plan #46.
        if rlx_opt::is_pure_view(graph, node) {
            thunks.push(Thunk::Nop);
            continue;
        }
        // Transpose folded into a downstream matmul's trans flag — skip it.
        if folded_transpose.contains(&node.id) {
            thunks.push(Thunk::Nop);
            continue;
        }
        // Inner nodes of an SGD chain folded into one SgdMomentum thunk.
        if sgd_elim.contains(&node.id) {
            thunks.push(Thunk::Nop);
            continue;
        }
        let t = match &node.op {
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => Thunk::Nop,

            Op::FusedMatMulBiasAct { activation } => {
                compile_fused_mat_mul_bias_act(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::FusedResidualLN { has_bias, eps } => {
                compile_fused_residual_l_n(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::FusedResidualRmsNorm { has_bias, eps } => {
                compile_fused_residual_rms_norm(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::MatMul => compile_mat_mul(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Binary(op) => {
                if let Some(&(param, vel, grad, v_new, lr, mom, len)) = sgd_fold.get(&node.id) {
                    // Folded SGD-momentum: this `Sub` (p') drives a single
                    // SgdMomentum thunk that also writes v''s slot.
                    thunks.push(Thunk::SgdMomentum {
                        param: node_offset(arena, param),
                        vel: node_offset(arena, vel),
                        grad: node_offset(arena, grad),
                        p_out: node_offset(arena, node.id),
                        v_out: node_offset(arena, v_new),
                        lr,
                        mom,
                        len: len as u32,
                    });
                    continue;
                }
                let lhs_len = get_len(graph, node.inputs[0]);
                let rhs_len = get_len(graph, node.inputs[1]);
                let out_len = node.shape.num_elements().unwrap();
                if node.shape.dtype() == rlx_ir::DType::C64 {
                    // Native C64 element-wise. Add/Sub/Mul/Div lower
                    // to `BinaryFullC64`; the rest don't have a
                    // single natural complex definition.
                    match op {
                        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {}
                        BinaryOp::Max | BinaryOp::Min | BinaryOp::Pow => panic!(
                            "Op::Binary({op:?}) on DType::C64: complex \
                             max/min/pow have no single natural definition \
                             — caller should drop to 2N-real-block (see \
                             spike-ac) and pick a convention there"
                        ),
                    }
                }
                // Compute broadcast strides for the slow path. Empty
                // vectors when no broadcast is needed (the fast-path
                // kernel ignores them anyway).
                let (out_dims_bcast, bcast_lhs_strides, bcast_rhs_strides) =
                    if lhs_len == out_len && rhs_len == out_len {
                        (Vec::new(), Vec::new(), Vec::new())
                    } else {
                        let lhs_dims = get_static_dims(graph, node.inputs[0]);
                        let rhs_dims = get_static_dims(graph, node.inputs[1]);
                        let out_dims_v = get_static_dims(graph, node.id);
                        if lhs_dims.is_empty() || rhs_dims.is_empty() || out_dims_v.is_empty() {
                            // Dynamic shape — fall back to the legacy
                            // modulo path (correct for scalar / last-
                            // axis broadcast, which is the only
                            // dynamic case in practice).
                            (Vec::new(), Vec::new(), Vec::new())
                        } else {
                            let ls = broadcast_strides(&lhs_dims, &out_dims_v);
                            let rs = broadcast_strides(&rhs_dims, &out_dims_v);
                            let od: Vec<u32> = out_dims_v.iter().map(|x| *x as u32).collect();
                            (od, ls, rs)
                        }
                    };
                if node.shape.dtype() == rlx_ir::DType::C64 {
                    Thunk::BinaryFullC64 {
                        lhs: node_offset(arena, node.inputs[0]),
                        rhs: node_offset(arena, node.inputs[1]),
                        dst: node_offset(arena, node.id),
                        len: out_len as u32,
                        lhs_len: lhs_len as u32,
                        rhs_len: rhs_len as u32,
                        op: *op,
                        out_dims_bcast,
                        bcast_lhs_strides,
                        bcast_rhs_strides,
                    }
                } else if node.shape.dtype() == rlx_ir::DType::F64 {
                    // f64 path — no BiasAdd fast-path (yet); use the
                    // general binary-with-broadcast kernel.
                    Thunk::BinaryFullF64 {
                        lhs: node_offset(arena, node.inputs[0]),
                        rhs: node_offset(arena, node.inputs[1]),
                        dst: node_offset(arena, node.id),
                        len: out_len as u32,
                        lhs_len: lhs_len as u32,
                        rhs_len: rhs_len as u32,
                        op: *op,
                        out_dims_bcast,
                        bcast_lhs_strides,
                        bcast_rhs_strides,
                    }
                } else if matches!(op, BinaryOp::Add)
                    && node.shape.dtype() == rlx_ir::DType::F32
                    && rhs_len < out_len
                    && out_len % rhs_len == 0
                    && is_trailing_bias_broadcast(
                        graph.node(node.inputs[1]).shape.dims(),
                        graph.node(node.id).shape.dims(),
                    )
                {
                    // `BiasAdd` is an f32-only fast path. Rank-0 / scalar I64
                    // Add (`is_trailing_bias_broadcast([], [N])` is vacuously
                    // true) must NOT take it — BiasAdd reads/writes 4-byte
                    // lanes and packs two i32 results into each i64 slot
                    // (Soprano Vocos gather indices: `floor+1` became
                    // `value | (1<<32)`).
                    //
                    // Also only correct when the bias is a *trailing*
                    // broadcast — rhs dims match the right-hand side of the
                    // output dims (with size-1 only allowed in left-padded
                    // outer positions). SAM's rel-pos
                    // `[bh, h, w, 1, w] + [bh, h, w, h, w]` has rhs_len divide
                    // out_len cleanly but is a mid-shape singleton, NOT a
                    // trailing broadcast.
                    Thunk::BiasAdd {
                        src: node_offset(arena, node.inputs[0]),
                        bias: node_offset(arena, node.inputs[1]),
                        dst: node_offset(arena, node.id),
                        m: (out_len / rhs_len) as u32,
                        n: rhs_len as u32,
                    }
                } else {
                    let lhs_len = get_len(graph, node.inputs[0]);
                    Thunk::BinaryFull {
                        lhs: node_offset(arena, node.inputs[0]),
                        rhs: node_offset(arena, node.inputs[1]),
                        dst: node_offset(arena, node.id),
                        len: out_len as u32,
                        lhs_len: lhs_len as u32,
                        rhs_len: rhs_len as u32,
                        op: *op,
                        out_dims_bcast,
                        bcast_lhs_strides,
                        bcast_rhs_strides,
                        elem_bytes: node.shape.dtype().size_bytes() as u8,
                    }
                }
            }

            Op::Activation(act) => {
                let len = node.shape.num_elements().unwrap();
                let in_off = node_offset(arena, node.inputs[0]);
                let out_off = node_offset(arena, node.id);
                if node.shape.dtype() == rlx_ir::DType::C64 {
                    // Only Neg/Exp/Log/Sqrt have natural complex
                    // extensions used in signal-processing graphs.
                    // Everything else (Sigmoid, Tanh, Relu, Abs,
                    // Sin/Cos/Tan/Atan, Round, GeLU family) is rejected.
                    match act {
                        Activation::Neg | Activation::Exp | Activation::Log | Activation::Sqrt => {}
                        other => panic!(
                            "Op::Activation({other:?}) on DType::C64: no \
                             natural complex extension — supported on C64: \
                             Neg, Exp, Log, Sqrt"
                        ),
                    }
                    Thunk::ActivationC64 {
                        src: in_off,
                        dst: out_off,
                        len: len as u32,
                        kind: *act,
                    }
                } else if node.shape.dtype() == rlx_ir::DType::F64 {
                    Thunk::ActivationF64 {
                        src: in_off,
                        dst: out_off,
                        len: len as u32,
                        kind: *act,
                    }
                } else if in_off == out_off {
                    // ActivationInPlace operates on a single buffer. When the
                    // planner has assigned input and output the same slot
                    // (typical post-fusion case), we just run on that slot.
                    Thunk::ActivationInPlace {
                        data: out_off,
                        len: len as u32,
                        act: *act,
                    }
                } else {
                    // Two-step: copy input → output, then activate output in place.
                    // The schedule executes them in this order; downstream
                    // thunks see the activated output at out_off.
                    thunks.push(Thunk::Copy {
                        src: in_off,
                        dst: out_off,
                        len: len as u32,
                    });
                    Thunk::ActivationInPlace {
                        data: out_off,
                        len: len as u32,
                        act: *act,
                    }
                }
            }

            Op::Gather { axis } if *axis == 0 => {
                let table_shape = &graph.node(node.inputs[0]).shape;
                let table_total = table_shape.num_elements().unwrap();
                let trailing: usize = (1..table_shape.rank())
                    .map(|i| table_shape.dim(i).unwrap_static())
                    .product();
                let idx_len = get_len(graph, node.inputs[1]);
                let idx_i64 =
                    u8::from(graph.node(node.inputs[1]).shape.dtype() == rlx_ir::DType::I64);
                let table_bytes = graph.node(node.inputs[0]).shape.dtype().size_bytes() as u8;
                Thunk::Gather {
                    table: node_offset(arena, node.inputs[0]),
                    table_len: table_total as u32,
                    idx: node_offset(arena, node.inputs[1]),
                    dst: node_offset(arena, node.id),
                    num_idx: idx_len as u32,
                    trailing: trailing as u32,
                    idx_i64,
                    table_bytes,
                }
            }

            Op::Gather { axis } => {
                compile_gather(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::Narrow { axis, start, len } => {
                compile_narrow(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::Reverse { axes } => {
                compile_reverse(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::Reshape { .. } | Op::StopGradient => {
                // Pure layout change: same total element count, plain copy.
                let len = node.shape.num_elements().unwrap();
                let src = node_offset(arena, node.inputs[0]);
                let dst = node_offset(arena, node.id);
                match node.shape.dtype() {
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
            }

            Op::Cast { to } => compile_cast(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Quantize {
                axis,
                scales,
                zero_points,
            } => compile_quantize(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::FakeQuantize {
                bits,
                axis,
                ste,
                scale_mode,
            } => compile_fake_quantize(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::FakeQuantizeLSQ { bits, axis } => {
                compile_fake_quantize_l_s_q(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::FakeQuantizeLSQBackwardX { bits, axis } => compile_fake_quantize_l_s_q_backward_x(
                node,
                graph,
                arena,
                &matmul_fold,
                &rng_shared,
                rng,
            ),
            Op::FakeQuantizeLSQBackwardScale { bits, axis } => {
                compile_fake_quantize_l_s_q_backward_scale(
                    node,
                    graph,
                    arena,
                    &matmul_fold,
                    &rng_shared,
                    rng,
                )
            }
            Op::FakeQuantizeBackward { bits, axis, ste } => {
                compile_fake_quantize_backward(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::Dequantize {
                axis,
                scales,
                zero_points,
            } => compile_dequantize(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Expand { .. } => compile_expand(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::RmsNorm { eps, .. } => {
                compile_rms_norm(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::LayerNorm { eps, .. } => {
                compile_layer_norm(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::AdaLayerNorm { .. } => {
                compile_ada_layer_norm(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::GatedResidual => {
                compile_gated_residual(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::AdaLayerNormBackward { .. } => {
                compile_ada_layer_norm_backward(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::GatedResidualBackward => {
                compile_gated_residual_backward(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::GroupNorm { num_groups, eps } => {
                compile_group_norm(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::BatchNormInference { eps } => {
                compile_batch_norm_inference(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::BatchNormInferenceBackwardInput { eps } => {
                compile_batch_norm_inference_backward_input(
                    node,
                    graph,
                    arena,
                    &matmul_fold,
                    &rng_shared,
                    rng,
                )
            }
            Op::BatchNormInferenceBackwardGamma { eps } => {
                compile_batch_norm_inference_backward_gamma(
                    node,
                    graph,
                    arena,
                    &matmul_fold,
                    &rng_shared,
                    rng,
                )
            }
            Op::BatchNormInferenceBackwardBeta => compile_batch_norm_inference_backward_beta(
                node,
                graph,
                arena,
                &matmul_fold,
                &rng_shared,
                rng,
            ),
            Op::LayerNorm2d { eps } => {
                compile_layer_norm2d(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::ConvTranspose2d {
                kernel_size,
                stride,
                padding,
                dilation,
                output_padding: _,
                groups,
            } => compile_conv_transpose2d(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::ResizeNearest2x => {
                compile_resize_nearest2x(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::AxialRope2d {
                end_x,
                end_y,
                head_dim,
                num_heads,
                theta,
                repeat_factor,
            } => compile_axial_rope2d(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Softmax { axis } => {
                let rank = node.shape.rank();
                let ax = if *axis < 0 {
                    (rank as i32 + axis) as usize
                } else {
                    *axis as usize
                };
                let cols = node.shape.dim(ax).unwrap_static();
                let total = node.shape.num_elements().unwrap();
                let in_off = node_offset(arena, node.inputs[0]);
                let out_off = node_offset(arena, node.id);
                // Softmax kernel runs in-place on its data buffer. If the
                // planner gave input and output separate slots (their live
                // ranges overlap, so no aliasing), the output starts
                // uninitialized — emit a Copy first so the data is there.
                // Same pattern as Op::Activation.
                if in_off != out_off {
                    thunks.push(Thunk::Copy {
                        src: in_off,
                        dst: out_off,
                        len: total as u32,
                    });
                }
                Thunk::Softmax {
                    data: out_off,
                    rows: (total / cols) as u32,
                    cols: cols as u32,
                }
            }

            Op::SelectiveScan { state_size } => {
                compile_selective_scan(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::GatedDeltaNet {
                state_size,
                carry_state,
            } => compile_gated_delta_net(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Lstm {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
            } => compile_lstm(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Gru {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
            } => compile_gru(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Rnn {
                hidden_size,
                num_layers,
                bidirectional,
                carry,
                relu,
            } => compile_rnn(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Mamba2 {
                head_dim,
                state_size,
            } => compile_mamba2(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::QMatMul {
                x_zp,
                w_zp,
                out_zp,
                mult,
            } => compile_q_mat_mul(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::QConv2d {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                x_zp,
                w_zp,
                out_zp,
                mult,
            } => compile_q_conv2d(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::DequantMatMul { scheme } => {
                compile_dequant_mat_mul(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::ScaledMatMul {
                lhs_format,
                rhs_format,
                scale_layout,
                has_bias,
            } => compile_scaled_mat_mul(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::ScaledQuantize {
                format,
                scale_layout,
            } => compile_scaled_quantize(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::ScaledQuantScale {
                format,
                scale_layout,
            } => compile_scaled_quant_scale(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::ScaledDequantize {
                format,
                scale_layout,
            } => compile_scaled_dequantize(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::LoraMatMul { scale } => {
                compile_lora_mat_mul(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::Sample {
                top_k,
                top_p,
                temperature,
                seed,
            } => compile_sample(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::RngNormal {
                mean,
                scale,
                key,
                op_seed,
            } => compile_rng_normal(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::RngUniform {
                low,
                high,
                key,
                op_seed,
            } => compile_rng_uniform(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Cumsum { axis, exclusive } => {
                compile_cumsum(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
                score_scale,
                attn_logit_softcap,
            } => compile_attention(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::AttentionBackward {
                num_heads,
                head_dim,
                mask_kind,
                wrt,
            } => compile_attention_backward(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::FusedAttentionBlock {
                num_heads,
                head_dim,
                has_bias,
                has_rope,
            } => compile_fused_attention_block(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Rope {
                head_dim,
                n_rot,
                style,
            } => compile_rope(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::FusedSwiGLU {
                cast_to: _,
                gate_first,
            } => compile_fused_swi_g_l_u(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Conv {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => compile_conv(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Conv3d { .. } => compile_conv3d(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::ConvTranspose3d { .. } => {
                compile_conv_transpose3d(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::Pool {
                kind,
                kernel_size,
                stride,
                padding,
            } => compile_pool(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Transpose { perm } => {
                compile_transpose(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::ScatterAdd => {
                compile_scatter_add(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::ScatterNd { .. } => {
                compile_scatter_nd(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::ScatterElements { .. } => {
                compile_scatter_elements(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::GatherNd { .. } => {
                compile_gather_nd(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::GatherElements { .. } => {
                compile_gather_elements(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::GroupedMatMul => {
                compile_grouped_mat_mul(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::DequantGroupedMatMul { scheme } => {
                compile_dequant_grouped_mat_mul(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::DequantMoEWeights { scheme } => {
                compile_dequant_mo_e_weights(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::TopK { k } => compile_top_k(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Reduce {
                op,
                axes,
                keep_dim: _,
            } => compile_reduce(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::ArgMax { axis, keep_dim: _ } | Op::ArgMin { axis, keep_dim: _ } => {
                let in_shape = &graph.node(node.inputs[0]).shape;
                let rank = in_shape.rank();
                let outer: usize = (0..*axis)
                    .map(|i| in_shape.dim(i).unwrap_static())
                    .product::<usize>()
                    .max(1);
                let reduced = in_shape.dim(*axis).unwrap_static();
                let inner: usize = (*axis + 1..rank)
                    .map(|i| in_shape.dim(i).unwrap_static())
                    .product::<usize>()
                    .max(1);
                Thunk::ArgReduce {
                    src: node_offset(arena, node.inputs[0]),
                    dst: node_offset(arena, node.id),
                    outer: outer as u32,
                    reduced: reduced as u32,
                    inner: inner as u32,
                    is_max: matches!(node.op, Op::ArgMax { .. }),
                }
            }

            Op::Compare(cmp) => compile_compare(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Where => compile_where(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Fma => compile_fma(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::ReluBackward => {
                compile_relu_backward(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::ComplexNormSq => {
                compile_complex_norm_sq(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::ComplexNormSqBackward => {
                compile_complex_norm_sq_backward(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::Conjugate => compile_conjugate(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::ActivationBackward { kind } => {
                compile_activation_backward(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::LayerNormBackwardInput { eps, .. } => compile_layer_norm_backward_input(
                node,
                graph,
                arena,
                &matmul_fold,
                &rng_shared,
                rng,
            ),
            Op::LayerNormBackwardGamma { eps, .. } => compile_layer_norm_backward_gamma(
                node,
                graph,
                arena,
                &matmul_fold,
                &rng_shared,
                rng,
            ),
            Op::RmsNormBackwardInput { eps, .. }
            | Op::RmsNormBackwardGamma { eps, .. }
            | Op::RmsNormBackwardBeta { eps, .. } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                let h = x_shape.dim(x_shape.rank() - 1).unwrap_static();
                let rows = (x_shape.num_elements().unwrap() / h) as u32;
                let off = |i: usize| node_offset(arena, node.inputs[i]);
                let common = (off(0), off(1), off(2), off(3), rows, h as u32, *eps);
                match &node.op {
                    Op::RmsNormBackwardInput { .. } => Thunk::RmsNormBackwardInput {
                        x: common.0,
                        gamma: common.1,
                        beta: common.2,
                        dy: common.3,
                        dx: node_offset(arena, node.id),
                        rows: common.4,
                        h: common.5,
                        eps: common.6,
                    },
                    Op::RmsNormBackwardGamma { .. } => Thunk::RmsNormBackwardGamma {
                        x: common.0,
                        gamma: common.1,
                        beta: common.2,
                        dy: common.3,
                        dgamma: node_offset(arena, node.id),
                        rows: common.4,
                        h: common.5,
                        eps: common.6,
                    },
                    Op::RmsNormBackwardBeta { .. } => Thunk::RmsNormBackwardBeta {
                        x: common.0,
                        gamma: common.1,
                        beta: common.2,
                        dy: common.3,
                        dbeta: node_offset(arena, node.id),
                        rows: common.4,
                        h: common.5,
                        eps: common.6,
                    },
                    _ => unreachable!(),
                }
            }

            Op::RopeBackward { head_dim, n_rot } => {
                compile_rope_backward(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::CumsumBackward { exclusive, .. } => {
                compile_cumsum_backward(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::GatherBackward { .. } => {
                compile_gather_backward(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::GroupNormBackwardInput { num_groups, eps }
            | Op::GroupNormBackwardGamma { num_groups, eps }
            | Op::GroupNormBackwardBeta { num_groups, eps } => {
                let x_shape = &graph.node(node.inputs[0]).shape;
                let n = x_shape.dim(0).unwrap_static() as u32;
                let c = x_shape.dim(1).unwrap_static() as u32;
                let h = x_shape.dim(2).unwrap_static() as u32;
                let w = x_shape.dim(3).unwrap_static() as u32;
                match &node.op {
                    Op::GroupNormBackwardInput { .. } => Thunk::GroupNormBackwardInput {
                        x: node_offset(arena, node.inputs[0]),
                        gamma: node_offset(arena, node.inputs[1]),
                        beta: node_offset(arena, node.inputs[2]),
                        dy: node_offset(arena, node.inputs[3]),
                        dx: node_offset(arena, node.id),
                        n,
                        c,
                        h,
                        w,
                        num_groups: *num_groups as u32,
                        eps: *eps,
                    },
                    Op::GroupNormBackwardGamma { .. } => Thunk::GroupNormBackwardGamma {
                        x: node_offset(arena, node.inputs[0]),
                        dy: node_offset(arena, node.inputs[1]),
                        dgamma: node_offset(arena, node.id),
                        n,
                        c,
                        h,
                        w,
                        num_groups: *num_groups as u32,
                        eps: *eps,
                    },
                    Op::GroupNormBackwardBeta { .. } => Thunk::GroupNormBackwardBeta {
                        dy: node_offset(arena, node.inputs[1]),
                        dbeta: node_offset(arena, node.id),
                        n,
                        c,
                        h,
                        w,
                    },
                    _ => unreachable!(),
                }
            }

            Op::MaxPool2dBackward {
                kernel_size,
                stride,
                padding,
            } => compile_max_pool2d_backward(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Conv2dBackwardInput {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => compile_conv2d_backward_input(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Conv2dBackwardWeight {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => compile_conv2d_backward_weight(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Im2Col {
                kernel_size,
                stride,
                padding,
                dilation,
            } => compile_im2_col(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::SoftmaxCrossEntropy => {
                compile_softmax_cross_entropy(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::SoftmaxCrossEntropyWithLogits => compile_softmax_cross_entropy_with_logits(
                node,
                graph,
                arena,
                &matmul_fold,
                &rng_shared,
                rng,
            ),
            Op::SoftmaxCrossEntropyBackward => compile_softmax_cross_entropy_backward(
                node,
                graph,
                arena,
                &matmul_fold,
                &rng_shared,
                rng,
            ),
            Op::DenseSolve => {
                compile_dense_solve(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::BatchedDenseSolve => {
                compile_batched_dense_solve(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            // ── Riemannian / SPD-manifold layers ─────────────────
            // Routed through Thunk::CustomOp (same path as user custom
            // ops) with directly-instantiated F64 kernels — no bespoke
            // Thunk variant needed. See crate::spd_kernels.
            Op::BiMap => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::BiMapKernel),
            ),
            Op::ReEig { eps } => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::ReEigKernel { eps: *eps as f64 }),
            ),
            Op::LogEig { eps } => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::LogEigKernel { eps: *eps as f64 }),
            ),
            Op::SpdBatchNorm { eps } => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::SpdBatchNormKernel { eps: *eps as f64 }),
            ),
            Op::SpdKarcherMean { iters, tol } => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::SpdKarcherMeanKernel {
                    iters: *iters as usize,
                    tol: *tol as f64,
                }),
            ),
            Op::SpdKarcherMeanWeighted { iters, tol } => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::SpdKarcherMeanWeightedKernel {
                    iters: *iters as usize,
                    tol: *tol as f64,
                }),
            ),
            Op::SpdLogMap => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::SpdLogMapKernel),
            ),
            Op::SpdExpMap => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::SpdExpMapKernel),
            ),
            Op::SpdParallelTransport => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::SpdParallelTransportKernel),
            ),
            Op::SpdMatrixFnBatch { kind } => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::SpdMatrixFnBatchKernel { kind: *kind }),
            ),
            Op::SpdLogMapBackward => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::SpdLogMapBackwardKernel),
            ),
            Op::SpdExpMapBackward => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::SpdExpMapBackwardKernel),
            ),
            Op::SpdParallelTransportBackward => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::SpdParallelTransportBackwardKernel),
            ),
            Op::SpdMatrixFnBatchBackward { kind } => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::SpdMatrixFnBatchBackwardKernel {
                    kind: *kind,
                }),
            ),
            Op::Eigh => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::EighKernel),
            ),
            Op::EighBackward => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::EighBackwardKernel),
            ),
            Op::EighBatch => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::EighBatchKernel),
            ),
            Op::EighBatchBackward => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::EighBatchBackwardKernel),
            ),
            Op::ReEigBackward { eps } => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::ReEigBackwardKernel { eps: *eps as f64 }),
            ),
            Op::LogEigBackward { eps } => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::LogEigBackwardKernel { eps: *eps as f64 }),
            ),
            Op::SpdBatchNormBackwardX { eps } => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::SpdBnBackwardXKernel { eps: *eps as f64 }),
            ),
            Op::SpdBatchNormBackwardG { eps } => compile_spd_custom(
                node,
                graph,
                arena,
                std::sync::Arc::new(crate::spd_kernels::SpdBnBackwardGKernel { eps: *eps as f64 }),
            ),
            Op::Scan {
                body,
                length,
                save_trajectory,
                num_bcast,
                num_xs,
                num_checkpoints,
            } => compile_scan(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::ScanBackward {
                body_vjp,
                length,
                save_trajectory,
                num_xs,
                num_checkpoints,
                forward_body,
            } => compile_scan_backward(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::ScanBackwardXs {
                body_vjp,
                length,
                save_trajectory,
                num_xs,
                xs_idx,
                num_checkpoints,
                forward_body,
            } => compile_scan_backward_xs(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::Concat { axis } => {
                compile_concat(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::GaussianSplatRender {
                width,
                height,
                tile_size,
                radius_scale,
                alpha_cutoff,
                max_splat_steps,
                transmittance_threshold,
                max_list_entries,
            } => compile_gaussian_splat_render(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::GaussianSplatRenderBackward {
                width,
                height,
                tile_size,
                radius_scale,
                alpha_cutoff,
                max_splat_steps,
                transmittance_threshold,
                max_list_entries,
                loss_grad_clip,
                sh_band,
                max_anisotropy,
            } => compile_gaussian_splat_render_backward(
                node,
                graph,
                arena,
                &matmul_fold,
                &rng_shared,
                rng,
            ),
            Op::GaussianSplatPrepare {
                width,
                height,
                tile_size,
                radius_scale,
                alpha_cutoff,
                max_splat_steps,
                transmittance_threshold,
                max_list_entries,
            } => compile_gaussian_splat_prepare(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::GaussianSplatRasterize {
                width,
                height,
                tile_size,
                alpha_cutoff,
                max_splat_steps,
                transmittance_threshold,
                max_list_entries,
            } => {
                compile_gaussian_splat_rasterize(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::Custom { name, attrs, .. } => {
                compile_custom(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::Fft { inverse, norm } => {
                compile_fft(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::FftButterflyStage { stage, n_fft } => {
                compile_fft_butterfly_stage(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::LogMel => compile_log_mel(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::LogMelBackward => {
                compile_log_mel_backward(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::WelchPeaks { k, n_segments } => {
                compile_welch_peaks(node, graph, arena, &matmul_fold, &rng_shared, rng)
            }
            Op::CustomFn {
                fwd_body,
                num_inputs,
                ..
            } => compile_custom_fn(node, graph, arena, &matmul_fold, &rng_shared, rng),
            Op::ElementwiseRegion {
                chain,
                scalar_input_mask,
                input_modulus,
                prologue,
                ..
            } => compile_elementwise_region(node, graph, arena, &matmul_fold, &rng_shared, rng),
            _ => Thunk::Nop,
        };
        thunks.push(t);
    }

    let cfg = crate::config::RuntimeConfig::global();
    let mask_thr = cfg.mask_binary_threshold;
    let mask_neg = cfg.attn_mask_neg_inf;
    let score_skip = cfg.score_skip_threshold;

    // Pre-compile closures (skip Nops — they're filtered out)
    let compiled_fns: Vec<Arc<dyn Fn(*mut u8) + Send + Sync>> = thunks
        .iter()
        .filter(|t| !matches!(t, Thunk::Nop))
        .map(|thunk| {
            match thunk.clone() {
                Thunk::Nop => Arc::new(|_: *mut u8| {}) as Arc<dyn Fn(*mut u8) + Send + Sync>,

                Thunk::Sgemm { a, b, c, m, k, n } => {
                    let (m, k, n) = (m as usize, k as usize, n as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        crate::blas::sgemm(
                            sl(a, base, m * k),
                            sl(b, base, k * n),
                            sl_mut(c, base, m * n),
                            m,
                            k,
                            n,
                        );
                    })
                }

                Thunk::CgemmC64 { a, b, c, m, k, n } => {
                    let (m, k, n) = (m as usize, k as usize, n as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        cgemm_c64(a, b, c, m, k, n, base);
                    })
                }

                Thunk::DenseSolveF64 { a, b, x, n, nrhs } => {
                    let (n_, nrhs_) = (n as usize, nrhs as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let a_src = sl_f64(a, base, n_ * n_);
                        let b_src = sl_f64(b, base, n_ * nrhs_);
                        let mut a_scratch: Vec<f64> = a_src.to_vec();
                        let mut x_buf: Vec<f64> = b_src.to_vec();
                        let info = crate::blas::dgesv(&mut a_scratch, &mut x_buf, n_, nrhs_);
                        if info != 0 {
                            panic!("DenseSolveF64: singular (info={info})");
                        }
                        sl_mut_f64(x, base, n_ * nrhs_).copy_from_slice(&x_buf);
                    })
                }

                Thunk::DenseSolveF32 { a, b, x, n, nrhs } => {
                    let (n_, nrhs_) = (n as usize, nrhs as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let a_src = sl(a, base, n_ * n_);
                        let b_src = sl(b, base, n_ * nrhs_);
                        let mut a_scratch: Vec<f32> = a_src.to_vec();
                        let mut x_buf: Vec<f32> = b_src.to_vec();
                        let info = crate::blas::sgesv(&mut a_scratch, &mut x_buf, n_, nrhs_);
                        if info != 0 {
                            panic!("DenseSolveF32: singular (info={info})");
                        }
                        sl_mut(x, base, n_ * nrhs_).copy_from_slice(&x_buf);
                    })
                }

                Thunk::FusedMmBiasAct {
                    a,
                    w,
                    bias,
                    c,
                    m,
                    k,
                    n,
                    act,
                } => {
                    let (m, k, n) = (m as usize, k as usize, n as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let out = sl_mut(c, base, m * n);
                        let bias_v = sl(bias, base, n);
                        // Match torch `F.linear` / `addmm(bias, A, W)`: seed C
                        // with broadcast bias, then accumulate A @ W (β=1).
                        // Separate sgemm + bias_add differs by ~1 ULP and
                        // breaks hard abs parity against PyTorch.
                        for row in 0..m {
                            out[row * n..(row + 1) * n].copy_from_slice(bias_v);
                        }
                        crate::blas::sgemm_accumulate(
                            sl(a, base, m * k),
                            sl(w, base, k * n),
                            out,
                            m,
                            k,
                            n,
                        );
                        match act {
                            Some(Activation::Gelu) => {
                                apply_activation_inplace(out, Activation::Gelu);
                            }
                            Some(other) => apply_activation_inplace(out, other),
                            None => {}
                        }
                    })
                }

                Thunk::FusedResidualLN {
                    x,
                    res,
                    bias,
                    g,
                    b,
                    out,
                    rows,
                    h,
                    eps,
                    has_bias,
                } => {
                    let (rows, h) = (rows as usize, h as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let zero = vec![0f32; h]; // closure only — not hot path
                        let bi = if has_bias { sl(bias, base, h) } else { &zero };
                        let xp = sl(x, base, rows * h).as_ptr() as usize;
                        let rp = sl(res, base, rows * h).as_ptr() as usize;
                        let op = sl_mut(out, base, rows * h).as_mut_ptr() as usize;
                        let bp = bi.as_ptr() as usize;
                        let gp = sl(g, base, h).as_ptr() as usize;
                        let bbp = sl(b, base, h).as_ptr() as usize;
                        crate::pool::par_for(rows, 4, &|off, cnt| {
                            let xs = std::slice::from_raw_parts(
                                (xp as *const f32).add(off * h),
                                cnt * h,
                            );
                            let rs = std::slice::from_raw_parts(
                                (rp as *const f32).add(off * h),
                                cnt * h,
                            );
                            let os = std::slice::from_raw_parts_mut(
                                (op as *mut f32).add(off * h),
                                cnt * h,
                            );
                            let bi = std::slice::from_raw_parts(bp as *const f32, h);
                            let g = std::slice::from_raw_parts(gp as *const f32, h);
                            let b = std::slice::from_raw_parts(bbp as *const f32, h);
                            crate::kernels::residual_bias_layer_norm(
                                xs, rs, bi, g, b, os, cnt, h, eps,
                            );
                        });
                    })
                }

                Thunk::BiasAdd {
                    src,
                    bias,
                    dst,
                    m,
                    n,
                } => {
                    let (m, n) = (m as usize, n as usize);
                    let len = m * n;
                    Arc::new(move |base: *mut u8| unsafe {
                        let out = sl_mut(dst, base, len);
                        if src != dst {
                            let src_ptr = base.add(src) as *const f32;
                            let dst_ptr = base.add(dst) as *mut f32;
                            if src_ptr != dst_ptr {
                                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len);
                            }
                        }
                        crate::blas::bias_add(out, sl(bias, base, n), m, n);
                    })
                }

                Thunk::Gather {
                    table,
                    table_len,
                    idx,
                    dst,
                    num_idx,
                    trailing,
                    idx_i64,
                    table_bytes,
                } => {
                    let (ni, tr, tl) = (num_idx as usize, trailing as usize, table_len as usize);
                    let rows = tl / tr.max(1);
                    let (idx_i64, table_bytes) = (idx_i64, table_bytes);
                    Arc::new(move |base: *mut u8| unsafe {
                        if table_bytes == 8 {
                            let tab = sl_i64(table, base, tl);
                            let out = sl_mut_i64(dst, base, ni * tr);
                            if idx_i64 != 0 {
                                let ids = sl_i64(idx, base, ni);
                                for i in 0..ni {
                                    let row = ids[i].max(0) as usize;
                                    if row < rows {
                                        out[i * tr..(i + 1) * tr]
                                            .copy_from_slice(&tab[row * tr..(row + 1) * tr]);
                                    }
                                }
                            } else {
                                let ids = sl(idx, base, ni);
                                for i in 0..ni {
                                    let row = ids[i] as usize;
                                    if row < rows {
                                        out[i * tr..(i + 1) * tr]
                                            .copy_from_slice(&tab[row * tr..(row + 1) * tr]);
                                    }
                                }
                            }
                        } else {
                            let tab = sl(table, base, tl);
                            let out = sl_mut(dst, base, ni * tr);
                            if idx_i64 != 0 {
                                let ids = sl_i64(idx, base, ni);
                                for i in 0..ni {
                                    let row = ids[i].max(0) as usize;
                                    if row < rows {
                                        out[i * tr..(i + 1) * tr]
                                            .copy_from_slice(&tab[row * tr..(row + 1) * tr]);
                                    }
                                }
                            } else {
                                let ids = sl(idx, base, ni);
                                for i in 0..ni {
                                    let row = ids[i] as usize;
                                    if row < rows {
                                        out[i * tr..(i + 1) * tr]
                                            .copy_from_slice(&tab[row * tr..(row + 1) * tr]);
                                    }
                                }
                            }
                        }
                    })
                }

                Thunk::Narrow {
                    src,
                    dst,
                    outer,
                    src_stride,
                    dst_stride,
                    inner,
                    elem_bytes,
                } => {
                    narrow_thunk_closure(src, dst, outer, src_stride, dst_stride, inner, elem_bytes)
                }

                Thunk::Reverse {
                    src,
                    dst,
                    dims,
                    rev_mask,
                    elem_bytes,
                } => {
                    let eb = elem_bytes as usize;
                    let rank = dims.len();
                    let total: usize = dims.iter().map(|&d| d as usize).product::<usize>().max(1);
                    // Row-major element strides.
                    let mut strides = vec![1usize; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        strides[i] = strides[i + 1] * dims[i + 1] as usize;
                    }
                    let dims_u: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
                    Arc::new(move |base: *mut u8| unsafe {
                        let src_base = base.add(src);
                        let dst_base = base.add(dst);
                        for o in 0..total {
                            // Output flat index → multi-index → (axis-reversed)
                            // input flat index.
                            let mut rem = o;
                            let mut in_flat = 0usize;
                            for ax in 0..rank {
                                let idx = rem / strides[ax];
                                rem %= strides[ax];
                                let in_idx = if rev_mask[ax] {
                                    dims_u[ax] - 1 - idx
                                } else {
                                    idx
                                };
                                in_flat += in_idx * strides[ax];
                            }
                            std::ptr::copy_nonoverlapping(
                                src_base.add(in_flat * eb),
                                dst_base.add(o * eb),
                                eb,
                            );
                        }
                    })
                }

                Thunk::Copy { src, dst, len } => {
                    let len = len as usize;
                    Arc::new(move |base: *mut u8| unsafe {
                        if src == dst || len == 0 {
                            return;
                        }
                        let src_ptr = base.add(src) as *const f32;
                        let dst_ptr = base.add(dst) as *mut f32;
                        if src_ptr == dst_ptr {
                            return;
                        }
                        std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, len);
                    })
                }

                Thunk::Softmax { data, rows, cols } => {
                    let (rows, cols) = (rows as usize, cols as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        crate::naive::softmax(sl_mut(data, base, rows * cols), rows, cols);
                    })
                }

                Thunk::Cumsum {
                    src,
                    dst,
                    rows,
                    cols,
                    exclusive,
                    dtype,
                } => {
                    let (rows, cols) = (rows as usize, cols as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        cumsum_typed(base, src, dst, rows, cols, exclusive, dtype);
                    })
                }

                Thunk::Sample {
                    logits,
                    dst,
                    batch,
                    vocab,
                    top_k,
                    top_p,
                    temperature,
                    seed,
                } => {
                    let (b, v) = (batch as usize, vocab as usize);
                    let k = (top_k as usize).min(v);
                    Arc::new(move |base: *mut u8| unsafe {
                        let lg = sl(logits, base, b * v);
                        let out = sl_mut(dst, base, b);
                        let mut rng =
                            rlx_ir::Philox4x32::new(if seed == 0 { 0xDEADBEEF } else { seed });
                        for bi in 0..b {
                            let row = &lg[bi * v..(bi + 1) * v];
                            out[bi] = sample_row(row, k, top_p, temperature, &mut rng) as f32;
                        }
                    })
                }

                Thunk::RngNormal {
                    dst,
                    len,
                    mean,
                    scale,
                    key,
                    op_seed,
                } => {
                    let n = len as usize;
                    let rng = rng_shared.clone();
                    Arc::new(move |base: *mut u8| unsafe {
                        let out = sl_mut(dst, base, n);
                        let opts = *rng.read().unwrap();
                        rlx_ir::fill_normal_like(out, mean, scale, opts, key, op_seed);
                    })
                }

                Thunk::RngUniform {
                    dst,
                    len,
                    low,
                    high,
                    key,
                    op_seed,
                } => {
                    let n = len as usize;
                    let rng = rng_shared.clone();
                    Arc::new(move |base: *mut u8| unsafe {
                        let out = sl_mut(dst, base, n);
                        let opts = *rng.read().unwrap();
                        rlx_ir::fill_uniform_like(out, low, high, opts, key, op_seed);
                    })
                }

                Thunk::DequantMatMul {
                    x,
                    w_q,
                    scale,
                    zp,
                    dst,
                    m,
                    k,
                    n,
                    block_size,
                    is_asymmetric,
                } => {
                    let (m, k, n, bs) = (m as usize, k as usize, n as usize, block_size as usize);
                    let n_blocks_per_col = k.div_ceil(bs);
                    Arc::new(move |base: *mut u8| unsafe {
                        let xs = sl(x, base, m * k);
                        // w_q is packed i8 — use raw byte slice + reinterpret.
                        let raw = base.add(w_q);
                        let w_bytes = std::slice::from_raw_parts(raw as *const i8, k * n);
                        let scales = sl(scale, base, n_blocks_per_col * n);
                        let zps = if is_asymmetric {
                            sl(zp, base, n_blocks_per_col * n)
                        } else {
                            &[][..]
                        };
                        let out = sl_mut(dst, base, m * n);
                        dequant_matmul_int8(
                            xs,
                            w_bytes,
                            scales,
                            zps,
                            out,
                            m,
                            k,
                            n,
                            bs,
                            is_asymmetric,
                        );
                    })
                }

                Thunk::DequantMatMulGguf {
                    x,
                    w_q,
                    dst,
                    m,
                    k,
                    n,
                    scheme,
                } => {
                    let (m, k, n) = (m as usize, k as usize, n as usize);
                    let block_bytes = scheme.gguf_block_bytes() as usize;
                    let block_elems = scheme.gguf_block_size() as usize;
                    let total_bytes = (k * n) / block_elems * block_bytes;
                    Arc::new(move |base: *mut u8| unsafe {
                        let xs = sl(x, base, m * k);
                        let w_bytes =
                            std::slice::from_raw_parts(base.add(w_q) as *const u8, total_bytes);
                        let out = sl_mut(dst, base, m * n);
                        crate::gguf_matmul::gguf_matmul_bt_dispatch(xs, w_bytes, out, m, k, n, scheme);
                    })
                }

                Thunk::DequantMatMulInt4 {
                    x,
                    w_q,
                    scale,
                    zp,
                    dst,
                    m,
                    k,
                    n,
                    block_size,
                    is_asymmetric,
                } => {
                    let (m, k, n, bs) = (m as usize, k as usize, n as usize, block_size as usize);
                    let n_blocks = k.div_ceil(bs);
                    Arc::new(move |base: *mut u8| unsafe {
                        let xs = sl(x, base, m * k);
                        let w_bytes = std::slice::from_raw_parts(
                            base.add(w_q) as *const u8,
                            (k * n).div_ceil(2),
                        );
                        let scales = sl(scale, base, n_blocks * n);
                        let zps = if is_asymmetric {
                            sl(zp, base, n_blocks * n)
                        } else {
                            &[][..]
                        };
                        let out = sl_mut(dst, base, m * n);
                        dequant_matmul_int4(
                            xs,
                            w_bytes,
                            scales,
                            zps,
                            out,
                            m,
                            k,
                            n,
                            bs,
                            is_asymmetric,
                        );
                    })
                }

                Thunk::DequantMatMulFp8 {
                    x,
                    w_q,
                    scale,
                    dst,
                    m,
                    k,
                    n,
                    e5m2,
                } => {
                    let (m, k, n) = (m as usize, k as usize, n as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let xs = sl(x, base, m * k);
                        let w_bytes = std::slice::from_raw_parts(base.add(w_q) as *const u8, k * n);
                        let scales = sl(scale, base, n);
                        let out = sl_mut(dst, base, m * n);
                        dequant_matmul_fp8(xs, w_bytes, scales, out, m, k, n, e5m2);
                    })
                }

                Thunk::DequantMatMulNvfp4 {
                    x,
                    w_q,
                    scale,
                    global_scale,
                    dst,
                    m,
                    k,
                    n,
                } => {
                    let (m, k, n) = (m as usize, k as usize, n as usize);
                    let n_scale = k.div_ceil(rlx_ir::NVFP4_GROUP_SIZE) * n;
                    Arc::new(move |base: *mut u8| unsafe {
                        let xs = sl(x, base, m * k);
                        let w_bytes = std::slice::from_raw_parts(
                            base.add(w_q) as *const u8,
                            (k * n).div_ceil(2),
                        );
                        let scale_bytes =
                            std::slice::from_raw_parts(base.add(scale) as *const u8, n_scale);
                        let gs = sl(global_scale, base, 1)[0];
                        let out = sl_mut(dst, base, m * n);
                        dequant_matmul_nvfp4(xs, w_bytes, scale_bytes, gs, out, m, k, n);
                    })
                }

                Thunk::ScaledMatMul {
                    lhs,
                    rhs,
                    lhs_scale,
                    rhs_scale,
                    bias,
                    dst,
                    m,
                    k,
                    n,
                    lhs_fmt,
                    rhs_fmt,
                    layout,
                    has_bias,
                } => {
                    let (m, k, n) = (m as usize, k as usize, n as usize);
                    let nblk = lowp_nblk(k, layout);
                    let per_tensor = matches!(layout, rlx_ir::ScaleLayout::PerTensor);
                    let n_lscale = if per_tensor { 1 } else { m * nblk };
                    let n_rscale = if per_tensor { 1 } else { n * nblk };
                    Arc::new(move |base: *mut u8| unsafe {
                        let lhs_b = std::slice::from_raw_parts(base.add(lhs) as *const u8, m * k);
                        let rhs_b = std::slice::from_raw_parts(base.add(rhs) as *const u8, n * k);
                        let ls = lowp_read_scales(layout, base, lhs_scale, n_lscale);
                        let rs = lowp_read_scales(layout, base, rhs_scale, n_rscale);
                        let bias_s = if has_bias { Some(sl(bias, base, n)) } else { None };
                        let out = sl_mut(dst, base, m * n);
                        lowp_scaled_matmul(
                            lhs_b, rhs_b, &ls, &rs, bias_s, out, m, n, k, layout, lhs_fmt, rhs_fmt,
                        );
                    })
                }

                Thunk::ScaledQuantize {
                    x,
                    scale,
                    dst,
                    rows,
                    cols,
                    fmt,
                    layout,
                } => {
                    let (rows, cols) = (rows as usize, cols as usize);
                    let nblk = lowp_nblk(cols, layout);
                    let n_scale = if matches!(layout, rlx_ir::ScaleLayout::PerTensor) {
                        1
                    } else {
                        rows * nblk
                    };
                    Arc::new(move |base: *mut u8| unsafe {
                        let xs = sl(x, base, rows * cols);
                        let scales = lowp_read_scales(layout, base, scale, n_scale);
                        let out =
                            std::slice::from_raw_parts_mut(base.add(dst), rows * cols);
                        lowp_quantize(xs, &scales, fmt, layout, rows, cols, out);
                    })
                }

                Thunk::ScaledQuantScale {
                    x,
                    dst,
                    rows,
                    cols,
                    fmt,
                    layout,
                } => {
                    let (rows, cols) = (rows as usize, cols as usize);
                    let nblk = lowp_nblk(cols, layout);
                    Arc::new(move |base: *mut u8| unsafe {
                        let xs = sl(x, base, rows * cols);
                        let scales = lowp_compute_scales(xs, fmt, layout, rows, cols);
                        match layout {
                            rlx_ir::ScaleLayout::PerTensor => {
                                sl_mut(dst, base, 1)[0] = scales[0];
                            }
                            rlx_ir::ScaleLayout::BlockMxE8M0 { .. } => {
                                let out = std::slice::from_raw_parts_mut(
                                    base.add(dst),
                                    rows * nblk,
                                );
                                for (o, &s) in out.iter_mut().zip(&scales) {
                                    *o = rlx_ir::lowp_codec::f32_to_e8m0(s);
                                }
                            }
                            rlx_ir::ScaleLayout::Nvfp4 { .. } => {
                                let out = std::slice::from_raw_parts_mut(
                                    base.add(dst),
                                    rows * nblk,
                                );
                                for (o, &s) in out.iter_mut().zip(&scales) {
                                    *o = rlx_ir::lowp_codec::encode(rlx_ir::ScaledFormat::F8E4M3, s);
                                }
                            }
                        }
                    })
                }

                Thunk::ScaledDequantize {
                    codes,
                    scale,
                    dst,
                    rows,
                    cols,
                    fmt,
                    layout,
                } => {
                    let (rows, cols) = (rows as usize, cols as usize);
                    let nblk = lowp_nblk(cols, layout);
                    let n_scale = if matches!(layout, rlx_ir::ScaleLayout::PerTensor) {
                        1
                    } else {
                        rows * nblk
                    };
                    Arc::new(move |base: *mut u8| unsafe {
                        let cs = std::slice::from_raw_parts(base.add(codes) as *const u8, rows * cols);
                        let scales = lowp_read_scales(layout, base, scale, n_scale);
                        let out = sl_mut(dst, base, rows * cols);
                        lowp_dequantize(cs, &scales, fmt, layout, rows, cols, out);
                    })
                }

                Thunk::LoraMatMul {
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
                } => {
                    let (m, k, n, r) = (m as usize, k as usize, n as usize, r as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let xs = sl(x, base, m * k);
                        let ws = sl(w, base, k * n);
                        let a_s = sl(a, base, k * r);
                        let bs = sl(b, base, r * n);
                        let out = sl_mut(dst, base, m * n);
                        // Step 1: out = x · W.
                        crate::blas::sgemm(xs, ws, out, m, k, n);
                        // Step 2: tmp = x · A (rank-r intermediate; tiny).
                        let mut tmp = vec![0f32; m * r];
                        crate::blas::sgemm(xs, a_s, &mut tmp, m, k, r);
                        // Step 3: out += scale * (tmp · B).
                        // sgemm_accumulate uses alpha=1.0 internally, so
                        // scale tmp first.
                        if scale != 1.0 {
                            for v in tmp.iter_mut() {
                                *v *= scale;
                            }
                        }
                        crate::blas::sgemm_accumulate(&tmp, bs, out, m, r, n);
                    })
                }

                Thunk::LayerNorm {
                    src,
                    g,
                    b,
                    dst,
                    rows,
                    h,
                    eps,
                } => {
                    let (rows, h) = (rows as usize, h as usize);
                    Arc::new(move |base: *mut u8| unsafe {
                        let inp = sl(src, base, rows * h);
                        let gamma = sl(g, base, h);
                        let beta = sl(b, base, h);
                        let out = sl_mut(dst, base, rows * h);
                        for row in 0..rows {
                            crate::kernels::layer_norm_row(
                                &inp[row * h..(row + 1) * h],
                                gamma,
                                beta,
                                &mut out[row * h..(row + 1) * h],
                                h,
                                eps,
                            );
                        }
                    })
                }

                Thunk::BatchNormInference {
                    src,
                    g,
                    b,
                    mean,
                    var,
                    dst,
                    count,
                    channels,
                    eps,
                } => {
                    let count = count as usize;
                    let c = channels as usize;
                    let n = count * c;
                    let (src, g, b, mean, var, dst) = (src, g, b, mean, var, dst);
                    Arc::new(move |base: *mut u8| unsafe {
                        crate::kernels::batch_norm_inference(
                            sl(src, base, n),
                            sl(g, base, c),
                            sl(b, base, c),
                            sl(mean, base, c),
                            sl(var, base, c),
                            sl_mut(dst, base, n),
                            c,
                            eps,
                        );
                    })
                }

                Thunk::Attention {
                    q,
                    k,
                    v,
                    mask,
                    out,
                    batch,
                    seq,
                    kv_seq,
                    heads,
                    kv_heads,
                    head_dim,
                    mask_kind,
                    scale,
                    softcap,
                    q_row_stride,
                    k_row_stride,
                    v_row_stride,
                    bhsd,
                } => {
                    if std::env::var("RLX_ATTN_DEBUG").is_ok() {
                        eprintln!("[attn-compile] batch={batch} seq={seq} kv_seq={kv_seq} heads={heads} bhsd={bhsd}");
                    }
                    // Q seq length (`q_s`) and K/V seq length (`k_s`) differ
                    // during cached decode (`q_s=1`, `k_s=past_seq+1`). The
                    // earlier version of this kernel destructured
                    // `kv_seq: _` and used a single `s = seq` for both axes,
                    // so cached decode only scored 1×1 instead of 1×k_s —
                    // attention couldn't see the past K cache and decode
                    // collapsed into repetitive fragments
                    // (`Self-based on [1\nAnswer: Self-based on [1…`).
                    let (b, q_s, k_s, nh, dh) = (
                        batch as usize,
                        seq as usize,
                        kv_seq as usize,
                        heads as usize,
                        head_dim as usize,
                    );
                    let hs = nh * dh;
                    // GQA/MQA: `group` query heads share one KV head. group == 1
                    // for MHA (kv_heads == heads), so this is a no-op there.
                    let nkv = (kv_heads as usize).max(1);
                    let group = (nh / nkv).max(1);
                    let qrs = q_row_stride as usize;
                    let krs = k_row_stride as usize;
                    let vrs = v_row_stride as usize;
                    // honor Op::Attention::score_scale (e.g. Gemma 4 = 1.0)
                    Arc::new(move |base: *mut u8| unsafe {
                        if std::env::var("RLX_ATTN_DEBUG").is_ok() {
                            eprintln!("[attn] b={b} q_s={q_s} k_s={k_s} nh={nh} dh={dh} bhsd={bhsd} mask_kind={:?}", mask_kind);
                        }
                        // Slice lengths use the source's row stride so the
                        // compiler-emitted bounds checks cover the whole
                        // strided span (the kernel walks with q/k/v_rs).
                        // For [B, H, S, D] the buffer is dense B*H*S*D.
                        let (q_len, k_len, v_len, o_len) = if bhsd {
                            let qn = b * nh * q_s * dh;
                            let kn = b * nkv * k_s * dh;
                            (qn, kn, kn, qn)
                        } else {
                            (b * q_s * qrs, b * k_s * krs, b * k_s * vrs, b * q_s * hs)
                        };
                        let q_d = sl(q, base, q_len);
                        let k_d = sl(k, base, k_len);
                        let v_d = sl(v, base, v_len);
                        let m_d: &[f32] = match mask_kind {
                            rlx_ir::op::MaskKind::Custom => sl(mask, base, b * k_s),
                            rlx_ir::op::MaskKind::Bias => sl(mask, base, b * nh * q_s * k_s),
                            _ => &[],
                        };
                        let o_d = sl_mut(out, base, o_len);
                        let mut qh = vec![0f32; q_s * dh];
                        let mut kh = vec![0f32; k_s * dh];
                        let mut vh = vec![0f32; k_s * dh];
                        let mut sc = vec![0f32; q_s * k_s];
                        let mut oh = vec![0f32; q_s * dh];
                        for bi in 0..b {
                            for hi in 0..nh {
                                // Gather per-head Q.
                                for si in 0..q_s {
                                    let q_off = if bhsd {
                                        bi * nh * q_s * dh + hi * q_s * dh + si * dh
                                    } else {
                                        bi * q_s * qrs + si * qrs + hi * dh
                                    };
                                    qh[si * dh..(si + 1) * dh]
                                        .copy_from_slice(&q_d[q_off..q_off + dh]);
                                }
                                // Gather per-head K, V. GQA/MQA: several query
                                // heads share one KV head (kv_hi); group == 1 for
                                // MHA, so kv_hi == hi.
                                let kv_hi = hi / group;
                                for si in 0..k_s {
                                    let (k_off, v_off) = if bhsd {
                                        (
                                            bi * nkv * k_s * dh + kv_hi * k_s * dh + si * dh,
                                            bi * nkv * k_s * dh + kv_hi * k_s * dh + si * dh,
                                        )
                                    } else {
                                        (
                                            bi * k_s * krs + si * krs + kv_hi * dh,
                                            bi * k_s * vrs + si * vrs + kv_hi * dh,
                                        )
                                    };
                                    kh[si * dh..(si + 1) * dh]
                                        .copy_from_slice(&k_d[k_off..k_off + dh]);
                                    vh[si * dh..(si + 1) * dh]
                                        .copy_from_slice(&v_d[v_off..v_off + dh]);
                                }
                                for qi in 0..q_s {
                                    for ki in 0..k_s {
                                        let mut dot = 0f32;
                                        for d in 0..dh {
                                            dot += qh[qi * dh + d] * kh[ki * dh + d];
                                        }
                                        sc[qi * k_s + ki] = dot * scale;
                                    }
                                }
                                // Apply mask. Causal/SlidingWindow use absolute
                                // positions so they handle Lq != Lk (decode mode
                                // with cached K/V): q_offset = k_s - q_s.
                                let q_offset = k_s.saturating_sub(q_s);
                                match mask_kind {
                                    rlx_ir::op::MaskKind::None => {}
                                    rlx_ir::op::MaskKind::Causal => {
                                        for qi in 0..q_s {
                                            let abs_q = q_offset + qi;
                                            for ki in (abs_q + 1)..k_s {
                                                sc[qi * k_s + ki] = mask_neg;
                                            }
                                        }
                                    }
                                    rlx_ir::op::MaskKind::SlidingWindow(w) => {
                                        for qi in 0..q_s {
                                            let abs_q = q_offset + qi;
                                            let lo = abs_q.saturating_sub(w);
                                            for ki in 0..k_s {
                                                if ki < lo || ki > abs_q {
                                                    sc[qi * k_s + ki] = mask_neg;
                                                }
                                            }
                                        }
                                    }
                                    rlx_ir::op::MaskKind::Custom => {
                                        for qi in 0..q_s {
                                            for ki in 0..k_s {
                                                if m_d[bi * k_s + ki] < mask_thr {
                                                    sc[qi * k_s + ki] = mask_neg;
                                                }
                                            }
                                        }
                                    }
                                    rlx_ir::op::MaskKind::Bias => {
                                        let per_bh = q_s * k_s;
                                        let off = (bi * nh + hi) * per_bh;
                                        for i in 0..per_bh {
                                            sc[i] += m_d[off + i];
                                        }
                                    }
                                }
                                // Gemma 2 attention logit soft-cap (post-mask,
                                // pre-softmax) — matches rlx-cpu executor.rs and
                                // rlx-metal / rlx-cuda native attention.
                                if softcap > 0.0 {
                                    for s in sc.iter_mut() {
                                        *s = softcap * (*s / softcap).tanh();
                                    }
                                }
                                crate::naive::softmax(&mut sc, q_s, k_s);
                                oh.fill(0.0);
                                for qi in 0..q_s {
                                    for ki in 0..k_s {
                                        let w = sc[qi * k_s + ki];
                                        if w > score_skip {
                                            for d in 0..dh {
                                                oh[qi * dh + d] += w * vh[ki * dh + d];
                                            }
                                        }
                                    }
                                }
                                for si in 0..q_s {
                                    let off = if bhsd {
                                        bi * nh * q_s * dh + hi * q_s * dh + si * dh
                                    } else {
                                        bi * q_s * hs + si * hs + hi * dh
                                    };
                                    o_d[off..off + dh].copy_from_slice(&oh[si * dh..(si + 1) * dh]);
                                }
                            }
                        }
                    })
                }

                Thunk::FusedSwiGLU {
                    src,
                    dst,
                    n_half,
                    total,
                    gate_first,
                } => {
                    let n = n_half as usize;
                    let t = total as usize;
                    let outer = t / n;
                    let in_total = outer * 2 * n;
                    Arc::new(move |base: *mut u8| unsafe {
                        let inp = sl(src, base, in_total);
                        let out = sl_mut(dst, base, t);
                        for o in 0..outer {
                            let in_row = &inp[o * 2 * n..(o + 1) * 2 * n];
                            let out_row = &mut out[o * n..(o + 1) * n];
                            for i in 0..n {
                                let (up, gate) = if gate_first {
                                    (in_row[n + i], in_row[i])
                                } else {
                                    (in_row[i], in_row[n + i])
                                };
                                out_row[i] = up * (gate / (1.0 + (-gate).exp()));
                            }
                        }
                    })
                }

                Thunk::Concat {
                    dst,
                    outer,
                    inner,
                    total_axis,
                    inputs,
                } => {
                    let outer = outer as usize;
                    let inner = inner as usize;
                    let total_axis = total_axis as usize;
                    let out_total = outer * total_axis * inner;
                    let mut layout: Vec<(usize, usize, usize, usize)> =
                        Vec::with_capacity(inputs.len());
                    let mut cum: usize = 0;
                    for (src_off, in_axis, in_numel) in &inputs {
                        let in_axis = *in_axis as usize;
                        layout.push((*src_off, cum * inner, in_axis * inner, *in_numel as usize));
                        cum += in_axis;
                    }
                    Arc::new(move |base: *mut u8| unsafe {
                        let out = sl_mut(dst, base, out_total);
                        let row_stride = total_axis * inner;
                        for (src_off, dst_col_off, copy_per_row, in_numel) in &layout {
                            let inp = sl(*src_off, base, (*in_numel).max(1));
                            concat_copy_rows_f32(
                                out,
                                inp,
                                outer,
                                *copy_per_row,
                                row_stride,
                                *dst_col_off,
                                *in_numel,
                            );
                        }
                    })
                }

                Thunk::CustomOp {
                    kernel,
                    inputs,
                    output,
                    attrs,
                } => {
                    // Capture-by-move: clone the Arc and Vecs once into the
                    // closure. Dispatch by output dtype each call (the
                    // dtype is fixed at compile time but it's cheaper to
                    // branch once per execution than to monomorphize a
                    // dozen closure variants).
                    let kernel = kernel.clone();
                    let attrs = attrs.clone();
                    let inputs = inputs.clone();
                    let (out_off, out_len, out_shape) = output.clone();
                    Arc::new(move |base: *mut u8| unsafe {
                        dispatch_custom_op(
                            &*kernel, &inputs, out_off, out_len, &out_shape, &attrs, base,
                        );
                    })
                }

                Thunk::GaussianSplatRender {
                    positions_off,
                    positions_len,
                    scales_off,
                    scales_len,
                    rotations_off,
                    rotations_len,
                    opacities_off,
                    opacities_len,
                    colors_off,
                    colors_len,
                    sh_coeffs_off,
                    sh_coeffs_len,
                    meta_off,
                    dst_off,
                    dst_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => Arc::new(move |base: *mut u8| unsafe {
                    crate::splat::execute_gaussian_splat_render(
                        positions_off,
                        positions_len,
                        scales_off,
                        scales_len,
                        rotations_off,
                        rotations_len,
                        opacities_off,
                        opacities_len,
                        colors_off,
                        colors_len,
                        sh_coeffs_off,
                        sh_coeffs_len,
                        meta_off,
                        dst_off,
                        dst_len,
                        width,
                        height,
                        tile_size,
                        radius_scale,
                        alpha_cutoff,
                        max_splat_steps,
                        transmittance_threshold,
                        max_list_entries,
                        base,
                    );
                }),

                Thunk::GaussianSplatRenderBackward {
                    positions_off,
                    positions_len,
                    scales_off,
                    scales_len,
                    rotations_off,
                    rotations_len,
                    opacities_off,
                    opacities_len,
                    colors_off,
                    colors_len,
                    sh_coeffs_off,
                    sh_coeffs_len,
                    meta_off,
                    d_loss_off,
                    d_loss_len,
                    packed_off,
                    packed_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                    loss_grad_clip,
                    sh_band,
                    max_anisotropy,
                } => Arc::new(move |base: *mut u8| unsafe {
                    crate::splat::execute_gaussian_splat_render_backward(
                        positions_off,
                        positions_len,
                        scales_off,
                        scales_len,
                        rotations_off,
                        rotations_len,
                        opacities_off,
                        opacities_len,
                        colors_off,
                        colors_len,
                        sh_coeffs_off,
                        sh_coeffs_len,
                        meta_off,
                        d_loss_off,
                        d_loss_len,
                        packed_off,
                        packed_len,
                        width,
                        height,
                        tile_size,
                        radius_scale,
                        alpha_cutoff,
                        max_splat_steps,
                        transmittance_threshold,
                        max_list_entries,
                        loss_grad_clip,
                        sh_band,
                        max_anisotropy,
                        base,
                    );
                }),

                Thunk::GaussianSplatPrepare {
                    positions_off,
                    positions_len,
                    scales_off,
                    scales_len,
                    rotations_off,
                    rotations_len,
                    opacities_off,
                    opacities_len,
                    colors_off,
                    colors_len,
                    sh_coeffs_off,
                    sh_coeffs_len,
                    meta_off,
                    meta_len,
                    prep_off,
                    prep_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => Arc::new(move |base: *mut u8| unsafe {
                    crate::splat::execute_gaussian_splat_prepare(
                        positions_off,
                        positions_len,
                        scales_off,
                        scales_len,
                        rotations_off,
                        rotations_len,
                        opacities_off,
                        opacities_len,
                        colors_off,
                        colors_len,
                        sh_coeffs_off,
                        sh_coeffs_len,
                        meta_off,
                        meta_len,
                        prep_off,
                        prep_len,
                        width,
                        height,
                        tile_size,
                        radius_scale,
                        alpha_cutoff,
                        max_splat_steps,
                        transmittance_threshold,
                        max_list_entries,
                        base,
                    );
                }),

                Thunk::GaussianSplatRasterize {
                    prep_off,
                    prep_len,
                    meta_off,
                    meta_len,
                    dst_off,
                    dst_len,
                    count,
                    width,
                    height,
                    tile_size,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => Arc::new(move |base: *mut u8| unsafe {
                    crate::splat::execute_gaussian_splat_rasterize(
                        prep_off,
                        prep_len,
                        meta_off,
                        meta_len,
                        dst_off,
                        dst_len,
                        count,
                        width,
                        height,
                        tile_size,
                        alpha_cutoff,
                        max_splat_steps,
                        transmittance_threshold,
                        max_list_entries,
                        base,
                    );
                }),

                Thunk::Fft1d {
                    src,
                    dst,
                    outer,
                    n_complex,
                    inverse,
                    norm_tag,
                    dtype,
                } => {
                    let f: Arc<dyn Fn(*mut u8) + Send + Sync> = match dtype {
                        rlx_ir::DType::F64 => Arc::new(move |base: *mut u8| unsafe {
                            execute_fft1d_f64(
                                src,
                                dst,
                                outer as usize,
                                n_complex as usize,
                                inverse,
                                norm_tag,
                                base,
                            );
                        }),
                        rlx_ir::DType::F32 => Arc::new(move |base: *mut u8| unsafe {
                            execute_fft1d_f32(
                                src,
                                dst,
                                outer as usize,
                                n_complex as usize,
                                inverse,
                                norm_tag,
                                base,
                            );
                        }),
                        rlx_ir::DType::C64 => Arc::new(move |base: *mut u8| unsafe {
                            execute_fft1d_c64(
                                src,
                                dst,
                                outer as usize,
                                n_complex as usize,
                                inverse,
                                norm_tag,
                                base,
                            );
                        }),
                        other => panic!("Op::Fft on CPU requires F32/F64/C64, got {other:?}"),
                    };
                    f
                }

                Thunk::FftButterflyStage {
                    state_src,
                    state_dst,
                    gate_src,
                    rev_src,
                    tw_re_src,
                    tw_im_src,
                    batch,
                    n_fft,
                    stage,
                } => Arc::new(move |base: *mut u8| unsafe {
                    execute_fft_butterfly_stage_f32(
                        state_src,
                        state_dst,
                        gate_src,
                        rev_src,
                        tw_re_src,
                        tw_im_src,
                        batch as usize,
                        n_fft as usize,
                        stage as usize,
                        base,
                    );
                }),

                Thunk::LogMel {
                    spec,
                    filters,
                    dst,
                    outer,
                    n_fft,
                    n_bins,
                    n_mels,
                } => Arc::new(move |base: *mut u8| unsafe {
                    execute_log_mel_f32(
                        spec,
                        filters,
                        dst,
                        outer as usize,
                        n_fft as usize,
                        n_bins as usize,
                        n_mels as usize,
                        base,
                    );
                }),

                Thunk::LogMelBackward {
                    spec,
                    filters,
                    dy,
                    dst,
                    outer,
                    n_fft,
                    n_bins,
                    n_mels,
                } => Arc::new(move |base: *mut u8| unsafe {
                    execute_log_mel_backward_f32(
                        spec,
                        filters,
                        dy,
                        dst,
                        outer as usize,
                        n_fft as usize,
                        n_bins as usize,
                        n_mels as usize,
                        base,
                    );
                }),

                Thunk::WelchPeaks {
                    spec,
                    dst,
                    welch_batch,
                    n_fft,
                    n_segments,
                    k,
                } => Arc::new(move |base: *mut u8| unsafe {
                    execute_welch_peaks_f32(
                        spec,
                        dst,
                        welch_batch as usize,
                        n_fft as usize,
                        n_segments as usize,
                        k as usize,
                        base,
                    );
                }),

                Thunk::SgdMomentum { param, vel, grad, p_out, v_out, lr, mom, len } => {
                    let len = len as usize;
                    Arc::new(move |base: *mut u8| unsafe {
                        let p = sl(param, base, len);
                        let v = sl(vel, base, len);
                        let g = sl(grad, base, len);
                        let po = sl_mut(p_out, base, len);
                        let vo = sl_mut(v_out, base, len);
                        for i in 0..len {
                            let vn = mom * v[i] + g[i];
                            vo[i] = vn;
                            po[i] = p[i] - lr * vn;
                        }
                    })
                }

                t @ Thunk::ScatterNd { .. } => {
                    let t = t.clone();
                    Arc::new(move |base: *mut u8| {
                        exec_scatter_nd(&t, base);
                    })
                }
                t @ Thunk::ScatterElements { .. } => {
                    let t = t.clone();
                    Arc::new(move |base: *mut u8| {
                        exec_scatter_elements(&t, base);
                    })
                }
                t @ Thunk::GatherNd { .. } => {
                    let t = t.clone();
                    Arc::new(move |base: *mut u8| {
                        exec_gather_nd(&t, base);
                    })
                }
                t @ Thunk::GatherElements { .. } => {
                    let t = t.clone();
                    Arc::new(move |base: *mut u8| {
                        exec_gather_elements(&t, base);
                    })
                }

                _ => Arc::new(|_: *mut u8| {}),
            }
        })
        .collect();

    // ── Thunk-level attention fusion ──────────────────────
    // For small batch*seq, fuse QKV→Narrow×3→[Rope×2]→Attention→OutProj
    // into a single FusedAttnBlock. Auto-detects from Attention thunks.
    let fuse_threshold: usize = rlx_ir::env::var("RLX_FUSE_ATTN_THRESHOLD")
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let should_fuse = thunks.iter().any(|t| match t {
        Thunk::Attention { batch, seq, .. } => {
            (*batch as usize) * (*seq as usize) <= fuse_threshold
        }
        _ => false,
    });

    if should_fuse {
        // Build non-Nop index for pattern matching across Nop gaps
        let active: Vec<usize> = thunks
            .iter()
            .enumerate()
            .filter(|(_, t)| !matches!(t, Thunk::Nop))
            .map(|(i, _)| i)
            .collect();

        let mut kill = vec![false; thunks.len()]; // mark thunks to remove
        let mut insertions: Vec<(usize, Thunk)> = Vec::new(); // (position, replacement)

        let mut ai = 0;
        while ai < active.len() {
            // Helper: get active thunk at offset from current
            let a = |off: usize| -> Option<(usize, &Thunk)> {
                active.get(ai + off).map(|&idx| (idx, &thunks[idx]))
            };

            // Try BERT pattern: FusedMmBiasAct(QKV) → Narrow×3 → Attention → FusedMmBiasAct(out)
            let matched = (|| {
                let (_i0, t0) = a(0)?;
                let (_, t1) = a(1)?;
                let (_, t2) = a(2)?;
                let (_, t3) = a(3)?;

                // a[0] must be FusedMmBiasAct or Sgemm (QKV projection)
                let (hidden, qkv_w, qkv_b, has_b) = match t0 {
                    Thunk::FusedMmBiasAct {
                        a,
                        w,
                        bias,
                        n: _,
                        act: None,
                        ..
                    } => (*a, *w, *bias, true),
                    Thunk::Sgemm { a, b, n: _, .. } => (*a, *b, 0, false),
                    _ => return None,
                };

                // a[1..3] must be Narrows
                if !matches!(t1, Thunk::Narrow { .. }) {
                    return None;
                }
                if !matches!(t2, Thunk::Narrow { .. }) {
                    return None;
                }
                if !matches!(t3, Thunk::Narrow { .. }) {
                    return None;
                }

                // Look for optional Rope×2 then Attention. Capture the rope
                // pairing (`interleaved` = GPT-J) from the q-rope thunk and
                // require the k-rope to agree — the fused kernel applies one
                // pairing to both.
                let (has_rope, attn_ai, cos_off, sin_off, cl, rope_interleaved) = if let Some((
                    _,
                    Thunk::Rope {
                        cos,
                        sin,
                        cos_len,
                        interleaved,
                        ..
                    },
                )) = a(4)
                {
                    let q_il = *interleaved;
                    match a(5).map(|x| x.1) {
                        Some(Thunk::Rope {
                            interleaved: k_il, ..
                        }) if *k_il == q_il => {
                            if matches!(a(6).map(|x| x.1), Some(Thunk::Attention { .. })) {
                                (true, 6, *cos, *sin, *cos_len, q_il)
                            } else {
                                return None;
                            }
                        }
                        _ => return None,
                    }
                } else if matches!(a(4).map(|x| x.1), Some(Thunk::Attention { .. })) {
                    (false, 4, 0, 0, 0, false)
                } else {
                    return None;
                };

                let (_attn_real_idx, attn_t) = a(attn_ai)?;
                let (batch, seq, heads, head_dim, mask, mask_kind, kv_seq, softcap) = match attn_t {
                    Thunk::Attention {
                        batch,
                        seq,
                        heads,
                        head_dim,
                        mask,
                        mask_kind,
                        kv_seq,
                        softcap,
                        ..
                    } => (
                        *batch, *seq, *heads, *head_dim, *mask, *mask_kind, *kv_seq, *softcap,
                    ),
                    _ => return None,
                };
                // The fused kernel synthesizes Causal / SlidingWindow in-kernel
                // and reads the buffer for Custom; it has no additive-`Bias`
                // path, no attention logit soft-cap (Gemma 2), and its in-kernel
                // position math assumes prefill (`q_seq == kv_seq`, i.e. no KV
                // cache). Fall back to the unfused Attention thunk (which honors
                // softcap) for anything else.
                if matches!(mask_kind, rlx_ir::op::MaskKind::Bias)
                    || kv_seq != seq
                    || softcap != 0.0
                {
                    return None;
                }

                // Next active must be out projection (FusedMmBiasAct or Sgemm)
                let (_out_real_idx, out_t) = a(attn_ai + 1)?;
                let (out_w, out_b, out_dst) = match out_t {
                    Thunk::FusedMmBiasAct {
                        w,
                        bias,
                        c,
                        act: None,
                        ..
                    } => (*w, *bias, *c),
                    Thunk::Sgemm { b: w, c, .. } => (*w, 0, *c),
                    _ => return None,
                };

                let hs = heads * head_dim;
                let total_active = attn_ai + 2; // number of active thunks consumed

                Some((
                    total_active,
                    Thunk::FusedAttnBlock {
                        hidden,
                        qkv_w,
                        out_w,
                        mask,
                        mask_kind,
                        out: out_dst,
                        qkv_b: if has_b { qkv_b } else { 0 },
                        out_b: if has_b { out_b } else { 0 },
                        cos: cos_off,
                        sin: sin_off,
                        cos_len: cl,
                        batch,
                        seq,
                        hs,
                        nh: heads,
                        dh: head_dim,
                        has_bias: has_b,
                        has_rope,
                        interleaved: rope_interleaved,
                    },
                ))
            })();

            if let Some((count, fused_thunk)) = matched {
                // Mark consumed thunks for removal
                for off in 0..count {
                    if let Some(&idx) = active.get(ai + off) {
                        kill[idx] = true;
                    }
                }
                // Insert replacement at position of the QKV thunk
                insertions.push((active[ai], fused_thunk));
                ai += count;
            } else {
                ai += 1;
            }
        }

        // Rebuild thunk list: keep non-killed, insert fused at right positions
        if !insertions.is_empty() {
            let mut new_thunks = Vec::with_capacity(thunks.len());
            let mut insert_idx = 0;
            for (i, t) in thunks.into_iter().enumerate() {
                if insert_idx < insertions.len() && insertions[insert_idx].0 == i {
                    new_thunks.push(insertions[insert_idx].1.clone());
                    insert_idx += 1;
                }
                if !kill[i] {
                    new_thunks.push(t);
                }
            }
            if cfg.verbose >= 1 {
                eprintln!(
                    "[rlx] fused_attention: {} attention blocks fused",
                    insertions.len()
                );
            }
            thunks = new_thunks;
        }
    }

    // ── Full layer fusion ──────────────────────────────────
    // After attention blocks are fused, scan for full layer patterns:
    // BERT:  FusedAttnBlock → FusedResidualLN → FusedMmBiasAct(gelu) → Sgemm → BiasAdd → FusedResidualLN
    // Nomic: FusedAttnBlock → BinaryFull(add) → LayerNorm → Sgemm → [Narrow×2 → Silu → BinaryFull(mul)] → Sgemm → BinaryFull(add) → LayerNorm
    if should_fuse {
        let active: Vec<usize> = thunks
            .iter()
            .enumerate()
            .filter(|(_, t)| !matches!(t, Thunk::Nop))
            .map(|(i, _)| i)
            .collect();

        let mut kill = vec![false; thunks.len()];
        let mut insertions: Vec<(usize, Thunk)> = Vec::new();

        let a = |ai: usize| -> Option<&Thunk> { active.get(ai).map(|&i| &thunks[i]) };

        let mut ai = 0;
        while ai < active.len() {
            // BERT pattern: FusedAttnBlock → FusedResidualLN → FusedMmBiasAct(gelu) → FusedMmBiasAct(none) → FusedResidualLN
            let bert_match = (|| -> Option<usize> {
                let fab = a(ai)?;
                let rln1 = a(ai + 1)?;
                let ffn1 = a(ai + 2)?;
                let ffn2 = a(ai + 3)?;
                let rln2 = a(ai + 4)?;

                let (hidden, qkv_w, qkv_b, out_w, out_b, mask, batch, seq, hs, nh, dh) = match fab {
                    Thunk::FusedAttnBlock {
                        hidden,
                        qkv_w,
                        qkv_b,
                        out_w,
                        out_b,
                        mask,
                        // FusedBertLayer applies only the per-key padding mask;
                        // it has no synthesized causal/sliding path, so it must
                        // not swallow a non-`Custom` attention block.
                        mask_kind: rlx_ir::op::MaskKind::Custom,
                        batch,
                        seq,
                        hs,
                        nh,
                        dh,
                        has_bias: true,
                        has_rope: false,
                        ..
                    } => (
                        *hidden, *qkv_w, *qkv_b, *out_w, *out_b, *mask, *batch, *seq, *hs, *nh, *dh,
                    ),
                    _ => return None,
                };
                let (ln1_g, ln1_b, eps1) = match rln1 {
                    Thunk::FusedResidualLN { g, b, eps, .. } => (*g, *b, *eps),
                    _ => return None,
                };
                let (fc1_w, fc1_b, int_dim) = match ffn1 {
                    Thunk::FusedMmBiasAct {
                        w,
                        bias,
                        n,
                        act: Some(Activation::Gelu),
                        ..
                    } => (*w, *bias, *n),
                    _ => return None,
                };
                let (fc2_w, fc2_b) = match ffn2 {
                    Thunk::FusedMmBiasAct {
                        w, bias, act: None, ..
                    } => (*w, *bias),
                    _ => return None,
                };
                let (ln2_g, ln2_b, eps2, out) = match rln2 {
                    Thunk::FusedResidualLN { g, b, eps, out, .. } => (*g, *b, *eps, *out),
                    _ => return None,
                };

                for off in 0..5 {
                    kill[active[ai + off]] = true;
                }
                insertions.push((
                    active[ai],
                    Thunk::FusedBertLayer {
                        hidden,
                        qkv_w,
                        qkv_b,
                        out_w,
                        out_b,
                        mask,
                        ln1_g,
                        ln1_b,
                        eps1,
                        fc1_w,
                        fc1_b,
                        fc2_w,
                        fc2_b,
                        ln2_g,
                        ln2_b,
                        eps2,
                        out,
                        batch,
                        seq,
                        hs,
                        nh,
                        dh,
                        int_dim,
                    },
                ));
                Some(5)
            })();
            if let Some(n) = bert_match {
                ai += n;
                continue;
            }

            // Nomic full-layer fusion — DISABLED. The matcher below targets a
            // stale pipeline shape (`FusedAttnBlock → FusedResidualLN → Sgemm →
            // Narrow×2 → SiLU → Mul → Sgemm → FusedResidualLN`). The current CPU
            // pipeline collapses the SwiGLU itself (`FuseSwiGLU` → one
            // `Op::FusedSwiGLU`/`Thunk::FusedSwiGLU`) and emits a runtime weight
            // `Concat` (runtime weight concat) before the fused fc matmul. So the
            // current post-fusion shape is:
            //   FusedAttnBlock(rope,no-bias) → FusedResidualLN(LN1) → [Concat] →
            //   Sgemm(fc11‖fc12) → FusedSwiGLU → Sgemm(fc2) → FusedResidualLN(LN2)
            // which this matches (the weight `Concat`s are kept — they produce the
            // fused fc weight the kernel reads). The `FusedNomicLayer` exec only
            // implements the up‖gate SwiGLU (`gate_first == false`, what real
            // Nomic emits: `up=fc11`, `gate=silu(fc12)`) and a per-key (`Custom`)
            // mask, so the match bails otherwise. Escape hatch:
            // `RLX_DISABLE_NOMIC_FUSION`. Validated against the real model in
            // `../rlx-models/crates/rlx-nomic` (CPU fused-vs-unfused parity).
            let nomic_match = (|| -> Option<usize> {
                if rlx_ir::env::flag("RLX_DISABLE_NOMIC_FUSION") {
                    return None;
                }
                let (
                    hidden,
                    qkv_w,
                    out_w,
                    mask,
                    cos,
                    sin,
                    cos_len,
                    batch,
                    seq,
                    hs,
                    nh,
                    dh,
                    interleaved,
                ) = match a(ai)? {
                    Thunk::FusedAttnBlock {
                        hidden,
                        qkv_w,
                        out_w,
                        mask,
                        cos,
                        sin,
                        cos_len,
                        batch,
                        seq,
                        hs,
                        nh,
                        dh,
                        has_bias: false,
                        has_rope: true,
                        mask_kind: rlx_ir::op::MaskKind::Custom,
                        interleaved,
                        ..
                    } => (
                        *hidden,
                        *qkv_w,
                        *out_w,
                        *mask,
                        *cos,
                        *sin,
                        *cos_len,
                        *batch,
                        *seq,
                        *hs,
                        *nh,
                        *dh,
                        *interleaved,
                    ),
                    _ => return None,
                };
                // FusedResidualLN for LN1
                let (ln1_g, ln1_b, eps1) = match a(ai + 1)? {
                    Thunk::FusedResidualLN { g, b, eps, .. } => (*g, *b, *eps),
                    _ => return None,
                };
                // Consumed active thunks to remove (the kept weight `Concat`s are
                // NOT added here — the fused kernel still reads their output).
                let mut kills: Vec<usize> = vec![ai, ai + 1];
                // Optional runtime weight Concat (fc11‖fc12), then Sgemm.
                let mut o = 2;
                if matches!(a(ai + o)?, Thunk::Concat { .. }) {
                    o += 1;
                }
                let fused_fc_w = match a(ai + o)? {
                    Thunk::Sgemm { b: w, .. } => *w,
                    _ => return None,
                };
                kills.push(ai + o);
                o += 1;
                // FusedSwiGLU — int_dim is its half width; the exec only handles
                // up‖gate (gate is the SECOND half), so require !gate_first.
                let int_dim = match a(ai + o)? {
                    Thunk::FusedSwiGLU {
                        n_half,
                        gate_first: false,
                        ..
                    } => *n_half,
                    _ => return None,
                };
                kills.push(ai + o);
                o += 1;
                // Sgemm (fc2)
                let fc2_w = match a(ai + o)? {
                    Thunk::Sgemm { b: w, .. } => *w,
                    _ => return None,
                };
                kills.push(ai + o);
                o += 1;
                // FusedResidualLN for LN2
                let (ln2_g, ln2_b, eps2, out) = match a(ai + o)? {
                    Thunk::FusedResidualLN { g, b, eps, out, .. } => (*g, *b, *eps, *out),
                    _ => return None,
                };
                kills.push(ai + o);
                let consumed = o + 1;

                for ki in kills {
                    kill[active[ki]] = true;
                }
                FUSED_NOMIC_LAYER_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Insert at the LAST consumed position (LN2), NOT the first
                // (FusedAttnBlock): the fused kernel reads the runtime weight
                // `Concat` (fc11‖fc12, and qkv for split-qkv models) that sits
                // between them, so it must execute AFTER those concats run.
                insertions.push((
                    active[ai + o],
                    Thunk::FusedNomicLayer {
                        hidden,
                        qkv_w,
                        out_w,
                        mask,
                        cos,
                        sin,
                        cos_len,
                        ln1_g,
                        ln1_b,
                        eps1,
                        fc11_w: fused_fc_w,
                        fc12_w: 0,
                        fc2_w,
                        ln2_g,
                        ln2_b,
                        eps2,
                        out,
                        batch,
                        seq,
                        hs,
                        nh,
                        dh,
                        int_dim,
                        interleaved,
                    },
                ));
                Some(consumed)
            })();
            if let Some(n) = nomic_match {
                ai += n;
                continue;
            }

            ai += 1;
        }

        if !insertions.is_empty() {
            let mut new_thunks = Vec::with_capacity(thunks.len());
            let mut ins_idx = 0;
            for (i, t) in thunks.into_iter().enumerate() {
                if ins_idx < insertions.len() && insertions[ins_idx].0 == i {
                    new_thunks.push(insertions[ins_idx].1.clone());
                    ins_idx += 1;
                }
                if !kill[i] {
                    new_thunks.push(t);
                }
            }
            if cfg.verbose >= 1 {
                eprintln!(
                    "[rlx] fused_layer: {} full transformer layers fused",
                    insertions.len()
                );
            }
            thunks = new_thunks;
        }
    }

    // ── Narrow → Rope thunk fusion (plan #45) ──────────────
    // Runs *after* FusedAttnBlock fusion so it only catches the medium-
    // batch path (batch*seq > 64) where the bigger fusion didn't fire.
    // Pattern: a Rope thunk whose `src` is the dst of an immediately-
    // preceding Narrow whose dst has no other consumer in this schedule.
    // Rewrite Rope to read directly from the parent buffer with the
    // parent's row stride; the Narrow becomes a Nop.
    //
    // Skipping the Narrow's write saves one full pass over Q/K (B*S*hs
    // f32) per Rope. For Nomic h=768 / batch=8 / seq=15 / 12 layers
    // that's 2 ropes/layer × 369 KB = ~8.9 MB of write traffic gone.
    {
        // Collect every byte-offset that's read as a thunk's `src` so
        // we know whether a Narrow's dst has consumers other than Rope.
        let mut read_offsets: HashMap<usize, usize> = HashMap::new();
        for t in &thunks {
            for off in thunk_read_offsets(t) {
                *read_offsets.entry(off).or_insert(0) += 1;
            }
        }

        let mut fused_count = 0usize;
        for i in 0..thunks.len().saturating_sub(1) {
            // Look for Rope at i+1 reading from Narrow at i (skip Nops
            // between them since the planner left them in place).
            let narrow = match &thunks[i] {
                Thunk::Narrow { .. } => i,
                _ => continue,
            };
            // Find the next non-Nop thunk
            let mut j = narrow + 1;
            while j < thunks.len() && matches!(thunks[j], Thunk::Nop) {
                j += 1;
            }
            if j >= thunks.len() {
                continue;
            }
            // Must be Rope reading Narrow's dst
            let (n_src, n_dst, n_src_stride) = match &thunks[narrow] {
                Thunk::Narrow {
                    src,
                    dst,
                    src_stride,
                    ..
                } => (*src, *dst, *src_stride),
                _ => continue,
            };
            let rope_reads_narrow = matches!(&thunks[j],
                Thunk::Rope { src, .. } if *src == n_dst);
            if !rope_reads_narrow {
                continue;
            }
            // Conservatively require that the Narrow's dst has exactly
            // one reader (the Rope). Anything else and rewriting would
            // skip a needed write.
            if read_offsets.get(&n_dst).copied().unwrap_or(0) != 1 {
                continue;
            }

            // Rewire: Rope reads from Narrow's adjusted source with the
            // parent buffer's row stride.
            if let Thunk::Rope {
                src,
                src_row_stride,
                ..
            } = &mut thunks[j]
            {
                *src = n_src;
                *src_row_stride = n_src_stride;
            }
            thunks[narrow] = Thunk::Nop;
            fused_count += 1;
        }

        if fused_count > 0 && cfg.verbose >= 1 {
            eprintln!(
                "[rlx] fused_qk_rope: {} Narrow→Rope pairs collapsed",
                fused_count
            );
        }
    }

    // ── Narrow×3 → Attention thunk fusion (plan #46 deep) ────
    // For each Attention thunk in the schedule, look up the producers
    // of its q/k/v inputs. If each is a Narrow whose dst has exactly
    // one consumer (the Attention), rewire Attention to read directly
    // from the parent buffer with the parent's row stride. The three
    // Narrows become Nops.
    //
    // This catches the BERT/Nomic QKV split path that FusedAttnBlock
    // misses (batch*seq > 64) — eliminates Q/K/V copies entirely.
    // For minilm6 batch=32 seq=16 hs=384: 3 × 32*16*384*4 = 2.3 MB
    // per layer × 6 layers = ~14 MB of write traffic gone.
    {
        let mut read_counts: HashMap<usize, usize> = HashMap::new();
        for t in &thunks {
            for off in thunk_read_offsets(t) {
                *read_counts.entry(off).or_insert(0) += 1;
            }
        }
        // Build dst→index map for fast producer lookup.
        let mut dst_to_idx: HashMap<usize, usize> = HashMap::new();
        for (i, t) in thunks.iter().enumerate() {
            if let Thunk::Narrow { dst, .. } = t {
                dst_to_idx.insert(*dst, i);
            }
        }

        let mut fused_count = 0usize;
        for i in 0..thunks.len() {
            let (q_off, k_off, v_off) = match &thunks[i] {
                Thunk::Attention { q, k, v, .. } => (*q, *k, *v),
                _ => continue,
            };
            // All three inputs must come from Narrows.
            let q_n = match dst_to_idx.get(&q_off).copied() {
                Some(x) => x,
                None => continue,
            };
            let k_n = match dst_to_idx.get(&k_off).copied() {
                Some(x) => x,
                None => continue,
            };
            let v_n = match dst_to_idx.get(&v_off).copied() {
                Some(x) => x,
                None => continue,
            };
            // Each Narrow's dst must have exactly one reader (this Attn).
            if read_counts.get(&q_off).copied().unwrap_or(0) != 1 {
                continue;
            }
            if read_counts.get(&k_off).copied().unwrap_or(0) != 1 {
                continue;
            }
            if read_counts.get(&v_off).copied().unwrap_or(0) != 1 {
                continue;
            }

            let (q_src, q_stride) = match &thunks[q_n] {
                Thunk::Narrow {
                    src, src_stride, ..
                } => (*src, *src_stride),
                _ => continue,
            };
            let (k_src, k_stride) = match &thunks[k_n] {
                Thunk::Narrow {
                    src, src_stride, ..
                } => (*src, *src_stride),
                _ => continue,
            };
            let (v_src, v_stride) = match &thunks[v_n] {
                Thunk::Narrow {
                    src, src_stride, ..
                } => (*src, *src_stride),
                _ => continue,
            };

            if let Thunk::Attention {
                q,
                k,
                v,
                q_row_stride,
                k_row_stride,
                v_row_stride,
                ..
            } = &mut thunks[i]
            {
                *q = q_src;
                *k = k_src;
                *v = v_src;
                *q_row_stride = q_stride;
                *k_row_stride = k_stride;
                *v_row_stride = v_stride;
            }
            thunks[q_n] = Thunk::Nop;
            thunks[k_n] = Thunk::Nop;
            thunks[v_n] = Thunk::Nop;
            fused_count += 1;
        }

        if fused_count > 0 && cfg.verbose >= 1 {
            eprintln!(
                "[rlx] fused_strided_attn: {} Narrow×3→Attention rewrites",
                fused_count
            );
        }
    }

    ThunkSchedule {
        thunks,
        moe_resident: None,
        moe_resident_layers: None,
        moe_topk_capture: None,
        mask_threshold: cfg.mask_binary_threshold,
        mask_neg_inf: cfg.attn_mask_neg_inf,
        score_skip: cfg.score_skip_threshold,
        compiled_fns,
        rng: rng_shared,
    }
}
