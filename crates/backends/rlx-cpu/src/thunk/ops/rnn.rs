#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

/// Execute a compiled scan body over `length` steps against a raw arena buffer
/// `base`. `outer_init_off` / `outer_final_off` are byte offsets into that
/// arena; `xs_outer[i] = (outer_off, per_step_bytes)` and
/// `bcast_outer[i] = (outer_off, total_bytes)` align with the plan's
/// `xs_body_offs` / `bcast_body_offs`. When `save_trajectory`, row `t` is
/// written to `outer_final_off + t * carry_bytes` (every step; checkpointed
/// scans are not supported via this path).
///
/// # Safety
/// `base` must point to a valid arena spanning every referenced offset.
pub unsafe fn execute_scan_host(
    base: *mut u8,
    plan: &ScanBodyPlan,
    outer_init_off: usize,
    outer_final_off: usize,
    length: u32,
    save_trajectory: bool,
    xs_outer: &[(usize, usize)],
    bcast_outer: &[(usize, usize)],
) {
    let cb = plan.carry_bytes;
    let mut body_buf: Vec<u8> = plan.body_init.clone();
    unsafe {
        std::ptr::copy_nonoverlapping(
            base.add(outer_init_off),
            body_buf.as_mut_ptr().add(plan.body_input_off),
            cb,
        );
        for (i, &(outer_b_off, total)) in bcast_outer.iter().enumerate() {
            std::ptr::copy_nonoverlapping(
                base.add(outer_b_off),
                body_buf.as_mut_ptr().add(plan.bcast_body_offs[i]),
                total,
            );
        }
    }
    for t in 0..length as usize {
        for (i, &(outer_xs_off, psb)) in xs_outer.iter().enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    base.add(outer_xs_off + t * psb),
                    body_buf.as_mut_ptr().add(plan.xs_body_offs[i]),
                    psb,
                );
            }
        }
        execute_thunks(&plan.body, &mut body_buf);
        if save_trajectory {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    body_buf.as_ptr().add(plan.body_output_off),
                    base.add(outer_final_off + t * cb),
                    cb,
                );
            }
        }
        if plan.body_output_off != plan.body_input_off {
            body_buf.copy_within(
                plan.body_output_off..plan.body_output_off + cb,
                plan.body_input_off,
            );
        }
    }
    if !save_trajectory {
        unsafe {
            std::ptr::copy_nonoverlapping(
                body_buf.as_ptr().add(plan.body_output_off),
                base.add(outer_final_off),
                cb,
            );
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_selective_scan(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::SelectiveScan { state_size } = &node.op else {
        unreachable!()
    };
    {
        let in_shape = &graph.node(node.inputs[0]).shape;
        let (batch, seq, hidden) = (
            in_shape.dim(0).unwrap_static(),
            in_shape.dim(1).unwrap_static(),
            in_shape.dim(2).unwrap_static(),
        );
        Thunk::SelectiveScan {
            x: node_offset(arena, node.inputs[0]),
            delta: node_offset(arena, node.inputs[1]),
            a: node_offset(arena, node.inputs[2]),
            b: node_offset(arena, node.inputs[3]),
            c: node_offset(arena, node.inputs[4]),
            dst: node_offset(arena, node.id),
            batch: batch as u32,
            seq: seq as u32,
            hidden: hidden as u32,
            state_size: *state_size as u32,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_gated_delta_net(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::GatedDeltaNet {
        state_size,
        carry_state,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let q_shape = &graph.node(node.inputs[0]).shape;
        let (batch, seq, heads) = (
            q_shape.dim(0).unwrap_static(),
            q_shape.dim(1).unwrap_static(),
            q_shape.dim(2).unwrap_static(),
        );
        let state_off = if *carry_state {
            node_offset(arena, node.inputs[5])
        } else {
            0
        };
        Thunk::GatedDeltaNet {
            q: node_offset(arena, node.inputs[0]),
            k: node_offset(arena, node.inputs[1]),
            v: node_offset(arena, node.inputs[2]),
            g: node_offset(arena, node.inputs[3]),
            beta: node_offset(arena, node.inputs[4]),
            state: state_off,
            dst: node_offset(arena, node.id),
            batch: batch as u32,
            seq: seq as u32,
            heads: heads as u32,
            state_size: *state_size as u32,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_lstm(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Lstm {
        hidden_size,
        num_layers,
        bidirectional,
        carry,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let x_shape = &graph.node(node.inputs[0]).shape;
        let (batch, seq, input_size) = (
            x_shape.dim(0).unwrap_static(),
            x_shape.dim(1).unwrap_static(),
            x_shape.dim(2).unwrap_static(),
        );
        let ex = rnn_expected_lens(
            4,
            batch,
            seq,
            input_size,
            *hidden_size,
            *num_layers,
            *bidirectional,
        );
        let nelem = |i: usize| graph.node(node.inputs[i]).shape.num_elements().unwrap_or(0);
        check_rnn_lens(
            "lstm",
            &[
                ("w_ih", nelem(1), ex.w_ih),
                ("w_hh", nelem(2), ex.w_hh),
                ("bias", nelem(3), ex.bias),
            ],
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let (h0, c0) = if *carry {
            (
                node_offset(arena, node.inputs[4]),
                node_offset(arena, node.inputs[5]),
            )
        } else {
            (0, 0)
        };
        Thunk::Lstm {
            x: node_offset(arena, node.inputs[0]),
            w_ih: node_offset(arena, node.inputs[1]),
            w_hh: node_offset(arena, node.inputs[2]),
            bias: node_offset(arena, node.inputs[3]),
            h0,
            c0,
            dst: node_offset(arena, node.id),
            batch: batch as u32,
            seq: seq as u32,
            input_size: input_size as u32,
            hidden: *hidden_size as u32,
            num_layers: *num_layers as u32,
            bidirectional: *bidirectional,
            carry: *carry,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_gru(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Gru {
        hidden_size,
        num_layers,
        bidirectional,
        carry,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let x_shape = &graph.node(node.inputs[0]).shape;
        let (batch, seq, input_size) = (
            x_shape.dim(0).unwrap_static(),
            x_shape.dim(1).unwrap_static(),
            x_shape.dim(2).unwrap_static(),
        );
        // Inputs: x, w_ih, w_hh, b_ih, b_hh (+ h0 when carry).
        let ex = rnn_expected_lens(
            3,
            batch,
            seq,
            input_size,
            *hidden_size,
            *num_layers,
            *bidirectional,
        );
        let nelem = |i: usize| graph.node(node.inputs[i]).shape.num_elements().unwrap_or(0);
        check_rnn_lens(
            "gru",
            &[
                ("w_ih", nelem(1), ex.w_ih),
                ("w_hh", nelem(2), ex.w_hh),
                ("b_ih", nelem(3), ex.bias),
                ("b_hh", nelem(4), ex.bias),
            ],
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let h0 = if *carry {
            node_offset(arena, node.inputs[5])
        } else {
            0
        };
        Thunk::Gru {
            x: node_offset(arena, node.inputs[0]),
            w_ih: node_offset(arena, node.inputs[1]),
            w_hh: node_offset(arena, node.inputs[2]),
            b_ih: node_offset(arena, node.inputs[3]),
            b_hh: node_offset(arena, node.inputs[4]),
            h0,
            dst: node_offset(arena, node.id),
            batch: batch as u32,
            seq: seq as u32,
            input_size: input_size as u32,
            hidden: *hidden_size as u32,
            num_layers: *num_layers as u32,
            bidirectional: *bidirectional,
            carry: *carry,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_rnn(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Rnn {
        hidden_size,
        num_layers,
        bidirectional,
        carry,
        relu,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let x_shape = &graph.node(node.inputs[0]).shape;
        let (batch, seq, input_size) = (
            x_shape.dim(0).unwrap_static(),
            x_shape.dim(1).unwrap_static(),
            x_shape.dim(2).unwrap_static(),
        );
        // Inputs: x, w_ih, w_hh, bias (+ h0 when carry).
        let ex = rnn_expected_lens(
            1,
            batch,
            seq,
            input_size,
            *hidden_size,
            *num_layers,
            *bidirectional,
        );
        let nelem = |i: usize| graph.node(node.inputs[i]).shape.num_elements().unwrap_or(0);
        check_rnn_lens(
            "rnn",
            &[
                ("w_ih", nelem(1), ex.w_ih),
                ("w_hh", nelem(2), ex.w_hh),
                ("bias", nelem(3), ex.bias),
            ],
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let h0 = if *carry {
            node_offset(arena, node.inputs[4])
        } else {
            0
        };
        Thunk::Rnn {
            x: node_offset(arena, node.inputs[0]),
            w_ih: node_offset(arena, node.inputs[1]),
            w_hh: node_offset(arena, node.inputs[2]),
            bias: node_offset(arena, node.inputs[3]),
            h0,
            dst: node_offset(arena, node.id),
            batch: batch as u32,
            seq: seq as u32,
            input_size: input_size as u32,
            hidden: *hidden_size as u32,
            num_layers: *num_layers as u32,
            bidirectional: *bidirectional,
            carry: *carry,
            relu: *relu,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_mamba2(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Mamba2 {
        head_dim,
        state_size,
    } = &node.op
    else {
        unreachable!()
    };
    {
        // x [B,S,H,P]; dt [B,S,H]; a [H]; b,c [B,S,H,N].
        let x_shape = &graph.node(node.inputs[0]).shape;
        Thunk::Mamba2 {
            x: node_offset(arena, node.inputs[0]),
            dt: node_offset(arena, node.inputs[1]),
            a: node_offset(arena, node.inputs[2]),
            b: node_offset(arena, node.inputs[3]),
            c: node_offset(arena, node.inputs[4]),
            dst: node_offset(arena, node.id),
            batch: x_shape.dim(0).unwrap_static() as u32,
            seq: x_shape.dim(1).unwrap_static() as u32,
            heads: x_shape.dim(2).unwrap_static() as u32,
            head_dim: *head_dim as u32,
            state_size: *state_size as u32,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_scan(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Scan {
        body,
        length,
        save_trajectory,
        num_bcast,
        num_xs,
        num_checkpoints,
    } = &node.op
    else {
        unreachable!()
    };
    {
        assert!(
            *num_checkpoints == 0 || *num_checkpoints <= *length,
            "Op::Scan: num_checkpoints={} must be 0 or ≤ length={}",
            *num_checkpoints,
            *length
        );
        if *num_checkpoints != 0 && *num_checkpoints != *length {
            assert!(
                *save_trajectory,
                "Op::Scan: num_checkpoints<length only meaningful when save_trajectory=true"
            );
        }
        // Plan + compile the body sub-graph standalone. The body
        // gets its own Arena; per execution we clone its
        // pristine bytes, copy the outer carry (and per-step xs
        // slices, if any) into the body's Input slots, run the
        // body schedule N times, then copy the body's output
        // back to the outer arena.
        //
        // Body invariants: 1 + num_xs Op::Inputs in NodeId order
        // — first declared is the carry, rest are x_t_i. Single
        // graph output (the next carry), same shape as carry.
        let body_plan = rlx_opt::memory::plan_memory(body);
        let _body_arena_size = body_plan.arena_size;
        // Snapshot per-input byte offsets before plan_memory
        // moves into the Arena below.
        let body_offsets: HashMap<NodeId, usize> = body_plan
            .assignments
            .iter()
            .map(|(id, slot)| (*id, slot.offset))
            .collect();

        // Collect body Input nodes in NodeId order; first is
        // carry, rest are per-step xs in matching order.
        let mut body_inputs: Vec<NodeId> = body
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Input { .. }))
            .map(|n| n.id)
            .collect();
        body_inputs.sort();
        let n_body_inputs = body_inputs.len();
        let expected = 1 + *num_bcast as usize + *num_xs as usize;
        if n_body_inputs != expected {
            let names: Vec<String> = body
                .nodes()
                .iter()
                .filter_map(|n| match &n.op {
                    Op::Input { name } => Some(format!("{}={}", n.id, name)),
                    _ => None,
                })
                .collect();
            panic!(
                "Op::Scan body has {} Op::Input nodes; expected {} \
                            (1 carry + {} bcast + {} xs). Inputs by NodeId: [{}]",
                n_body_inputs,
                expected,
                *num_bcast,
                *num_xs,
                names.join(", ")
            );
        }

        let body_input_id = body_inputs[0];
        let body_input_off = body_offsets[&body_input_id];
        let body_output_id = body
            .outputs
            .first()
            .copied()
            .expect("Op::Scan body must declare one output");
        let body_output_off = body_offsets[&body_output_id];

        let mut body_arena = crate::arena::Arena::from_plan(body_plan);
        // Fill body Constant nodes — mirror the outer-graph logic
        // in rlx-runtime/src/backend.rs (dtype-aware).
        for n in body.nodes() {
            if let Op::Constant { data } = &n.op
                && body_arena.has_buffer(n.id)
                && !data.is_empty()
            {
                match n.shape.dtype() {
                    rlx_ir::DType::F64 => {
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
                            let bytes = [
                                data[i * 4],
                                data[i * 4 + 1],
                                data[i * 4 + 2],
                                data[i * 4 + 3],
                            ];
                            buf[i] = f32::from_le_bytes(bytes);
                        }
                    }
                }
            }
        }
        let body_init = body_arena.raw_buf().to_vec();
        let body_schedule = compile_thunks_with_rng(body, &body_arena, rng);

        // Carry bytes — for trajectory mode, the outer node's
        // shape is [length, *carry_shape], so dividing by length
        // gives one row's bytes; the body's input slot still
        // holds carry_shape bytes.
        let carry_bytes = if *save_trajectory {
            let total = node
                .shape
                .size_bytes()
                .expect("Op::Scan trajectory output must have static shape");
            total / *length as usize
        } else {
            node.shape
                .size_bytes()
                .expect("Op::Scan carry must have static shape")
        };

        // Bcast inputs occupy body_inputs[1..1+num_bcast] and
        // outer node.inputs[1..1+num_bcast]. They keep their
        // natural shape (no [length, ...] prefix) and are
        // copied into body_buf ONCE before the scan loop.
        let mut bcast_inputs: Vec<(usize, usize, u32)> = Vec::with_capacity(*num_bcast as usize);
        for i in 0..*num_bcast as usize {
            let body_b_id = body_inputs[1 + i];
            let body_b_off = body_offsets[&body_b_id];
            let outer_b_id = node.inputs[1 + i];
            let outer_b_off = node_offset(arena, outer_b_id);
            let outer_b_shape = &graph.node(outer_b_id).shape;
            let total = outer_b_shape
                .size_bytes()
                .expect("Op::Scan bcast must have static shape");
            bcast_inputs.push((body_b_off, outer_b_off, total as u32));
        }

        // xs occupy body_inputs[1+num_bcast..] and node.inputs
        // [1+num_bcast..]. Each has shape [length, *per_step];
        // per-step bytes = total / length.
        let mut xs_inputs: Vec<(usize, usize, u32)> = Vec::with_capacity(*num_xs as usize);
        let xs_base = 1 + *num_bcast as usize;
        for i in 0..*num_xs as usize {
            let body_x_id = body_inputs[xs_base + i];
            let body_x_off = body_offsets[&body_x_id];
            let outer_xs_id = node.inputs[xs_base + i];
            let outer_xs_off = node_offset(arena, outer_xs_id);
            let outer_xs_shape = &graph.node(outer_xs_id).shape;
            let total = outer_xs_shape
                .size_bytes()
                .expect("Op::Scan xs must have static shape");
            let per_step = total / *length as usize;
            xs_inputs.push((body_x_off, outer_xs_off, per_step as u32));
        }

        Thunk::Scan {
            body: Arc::new(body_schedule),
            body_init: Arc::new(body_init),
            body_input_off,
            body_output_off,
            outer_init_off: node_offset(arena, node.inputs[0]),
            outer_final_off: node_offset(arena, node.id),
            length: *length,
            carry_bytes: carry_bytes as u32,
            save_trajectory: *save_trajectory,
            xs_inputs: Arc::new(xs_inputs),
            bcast_inputs: Arc::new(bcast_inputs),
            num_checkpoints: *num_checkpoints,
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_scan_backward(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ScanBackward {
        body_vjp,
        length,
        save_trajectory,
        num_xs,
        num_checkpoints,
        forward_body,
    } = &node.op
    else {
        unreachable!()
    };
    {
        let is_recursive = *num_checkpoints != 0 && *num_checkpoints != *length;
        if is_recursive {
            assert!(
                forward_body.is_some(),
                "Op::ScanBackward with num_checkpoints<length requires forward_body"
            );
        }
        // body_vjp has signature
        //   (carry, x_t_0, ..., x_t_{num_xs-1}, d_output) → dcarry
        // Identify slots:
        //   * "d_output" by exact name (AD-introduced seed Input).
        //   * Remaining Inputs sorted by NodeId — first is the
        //     carry mirror, rest are x_t_i mirrors in body's
        //     original Op::Input declaration order.
        let body_plan = rlx_opt::memory::plan_memory(body_vjp);
        let body_offsets: HashMap<NodeId, usize> = body_plan
            .assignments
            .iter()
            .map(|(id, slot)| (*id, slot.offset))
            .collect();
        let mut body_d_output_off: Option<usize> = None;
        let mut body_other_inputs: Vec<(NodeId, usize)> = Vec::new();
        for n in body_vjp.nodes() {
            if let Op::Input { name } = &n.op {
                let off = body_offsets[&n.id];
                if name == "d_output" {
                    body_d_output_off = Some(off);
                } else {
                    body_other_inputs.push((n.id, off));
                }
            }
        }
        body_other_inputs.sort_by_key(|(id, _)| *id);
        let body_d_output_off =
            body_d_output_off.expect("ScanBackward body_vjp missing 'd_output' Input");
        let expected_others = 1 + *num_xs as usize;
        assert_eq!(
            body_other_inputs.len(),
            expected_others,
            "ScanBackward body_vjp has {} non-d_output Inputs; \
                     expected {} (1 carry + {} xs)",
            body_other_inputs.len(),
            expected_others,
            num_xs
        );
        let body_carry_in_off = body_other_inputs[0].1;
        let body_x_offs: Vec<usize> = body_other_inputs
            .iter()
            .skip(1)
            .map(|(_, off)| *off)
            .collect();
        let body_dcarry_out_off = body_offsets[&body_vjp.outputs[0]];

        let mut body_arena = crate::arena::Arena::from_plan(body_plan);
        // Fill body_vjp's Constants (mirrors the Scan lowering).
        for n in body_vjp.nodes() {
            if let Op::Constant { data } = &n.op
                && body_arena.has_buffer(n.id)
                && !data.is_empty()
            {
                match n.shape.dtype() {
                    rlx_ir::DType::F64 => {
                        let off = body_arena.byte_offset(n.id);
                        let buf = body_arena.raw_buf_mut();
                        let nb = (buf.len() - off).min(data.len());
                        buf[off..off + nb].copy_from_slice(&data[..nb]);
                    }
                    _ => {
                        let buf = body_arena.slice_mut(n.id);
                        let nf = data.len() / 4;
                        let nl = buf.len().min(nf);
                        for i in 0..nl {
                            let bytes = [
                                data[i * 4],
                                data[i * 4 + 1],
                                data[i * 4 + 2],
                                data[i * 4 + 3],
                            ];
                            buf[i] = f32::from_le_bytes(bytes);
                        }
                    }
                }
            }
        }
        let body_init = body_arena.raw_buf().to_vec();
        let body_schedule = compile_thunks_with_rng(body_vjp, &body_arena, rng);

        // Carry bytes from the dcarry output node (== carry shape).
        let carry_bytes = body_vjp
            .node(body_vjp.outputs[0])
            .shape
            .size_bytes()
            .expect("ScanBackward dcarry must be statically shaped");
        let carry_elem_size = body_vjp
            .node(body_vjp.outputs[0])
            .shape
            .dtype()
            .size_bytes() as u32;

        // For each xs input on the outer node:
        // (outer_xs_base, per_step_bytes).
        let mut outer_xs_offs: Vec<(usize, u32)> = Vec::with_capacity(*num_xs as usize);
        for i in 0..*num_xs as usize {
            let outer_xs_id = node.inputs[3 + i];
            let outer_xs_off = node_offset(arena, outer_xs_id);
            let outer_xs_shape = &graph.node(outer_xs_id).shape;
            let total = outer_xs_shape
                .size_bytes()
                .expect("ScanBackward xs must have static shape");
            let per_step = total / *length as usize;
            outer_xs_offs.push((outer_xs_off, per_step as u32));
        }

        // If recursive checkpointing is active, we also compile
        // the forward body so the executor can recompute
        // intermediate carries. The forward body is supplied
        // by the AD pass via `forward_body: Some(_)`.
        let (fb_schedule, fb_init, fb_carry_in_off, fb_output_off, fb_x_offs) = if is_recursive {
            let fb = forward_body.as_ref().unwrap();
            let fb_plan = rlx_opt::memory::plan_memory(fb);
            let fb_offsets: HashMap<NodeId, usize> = fb_plan
                .assignments
                .iter()
                .map(|(id, slot)| (*id, slot.offset))
                .collect();
            let mut fb_inputs: Vec<NodeId> = fb
                .nodes()
                .iter()
                .filter(|n| matches!(n.op, Op::Input { .. }))
                .map(|n| n.id)
                .collect();
            fb_inputs.sort();
            let fb_carry = fb_offsets[&fb_inputs[0]];
            let fb_xs: Vec<usize> = (1..fb_inputs.len())
                .map(|i| fb_offsets[&fb_inputs[i]])
                .collect();
            let fb_out = fb_offsets[&fb.outputs[0]];
            let mut fb_arena = crate::arena::Arena::from_plan(fb_plan);
            for n in fb.nodes() {
                if let Op::Constant { data } = &n.op
                    && fb_arena.has_buffer(n.id)
                    && !data.is_empty()
                {
                    // Byte-copy works for any
                    // numeric dtype as long as the
                    // arena slot is sized to hold
                    // it — the Constant's `data`
                    // already encodes the right
                    // bytes per element.
                    let off = fb_arena.byte_offset(n.id);
                    let buf = fb_arena.raw_buf_mut();
                    let nb = (buf.len() - off).min(data.len());
                    buf[off..off + nb].copy_from_slice(&data[..nb]);
                }
            }
            let fb_init_bytes = fb_arena.raw_buf().to_vec();
            let fb_sched = compile_thunks_with_rng(fb, &fb_arena, rng);
            (
                Some(Arc::new(fb_sched)),
                Some(Arc::new(fb_init_bytes)),
                fb_carry,
                fb_out,
                fb_xs,
            )
        } else {
            (None, None, 0, 0, Vec::new())
        };

        Thunk::ScanBackward {
            body_vjp: Arc::new(body_schedule),
            body_init: Arc::new(body_init),
            body_carry_in_off,
            body_x_offs: Arc::new(body_x_offs),
            body_d_output_off,
            body_dcarry_out_off,
            outer_init_off: node_offset(arena, node.inputs[0]),
            outer_traj_off: node_offset(arena, node.inputs[1]),
            outer_upstream_off: node_offset(arena, node.inputs[2]),
            outer_xs_offs: Arc::new(outer_xs_offs),
            outer_dinit_off: node_offset(arena, node.id),
            length: *length,
            carry_bytes: carry_bytes as u32,
            carry_elem_size,
            save_trajectory: *save_trajectory,
            num_checkpoints: *num_checkpoints,
            forward_body: fb_schedule,
            forward_body_init: fb_init,
            forward_body_carry_in_off: fb_carry_in_off,
            forward_body_output_off: fb_output_off,
            forward_body_x_offs: Arc::new(fb_x_offs),
        }
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_scan_backward_xs(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::ScanBackwardXs {
        body_vjp,
        length,
        save_trajectory,
        num_xs,
        xs_idx,
        num_checkpoints,
        forward_body,
    } = &node.op
    else {
        unreachable!()
    };
    {
        assert!(
            *num_checkpoints == 0 || *num_checkpoints <= *length,
            "Op::ScanBackwardXs: num_checkpoints={} must be 0 or ≤ length={}",
            *num_checkpoints,
            *length
        );
        let is_recursive = *num_checkpoints != 0 && *num_checkpoints != *length;
        if is_recursive {
            assert!(
                forward_body.is_some(),
                "Op::ScanBackwardXs with num_checkpoints<length \
                         requires forward_body"
            );
        }
        // Mirror ScanBackward's body_vjp slot identification +
        // arena prep, then add: per-iteration extraction of the
        // body_vjp output that corresponds to the chosen xs.
        //
        // body_vjp's outputs (from `grad(body, [carry, xs_0, ..., xs_{num_xs-1}])`):
        //   outputs[0]      = dcarry
        //   outputs[1 + i]  = dx_t_i
        let body_plan = rlx_opt::memory::plan_memory(body_vjp);
        let body_offsets: HashMap<NodeId, usize> = body_plan
            .assignments
            .iter()
            .map(|(id, slot)| (*id, slot.offset))
            .collect();
        let mut body_d_output_off: Option<usize> = None;
        let mut body_other_inputs: Vec<(NodeId, usize)> = Vec::new();
        for n in body_vjp.nodes() {
            if let Op::Input { name } = &n.op {
                let off = body_offsets[&n.id];
                if name == "d_output" {
                    body_d_output_off = Some(off);
                } else {
                    body_other_inputs.push((n.id, off));
                }
            }
        }
        body_other_inputs.sort_by_key(|(id, _)| *id);
        let body_d_output_off =
            body_d_output_off.expect("ScanBackwardXs body_vjp missing 'd_output' Input");
        let expected_others = 1 + *num_xs as usize;
        assert_eq!(
            body_other_inputs.len(),
            expected_others,
            "ScanBackwardXs body_vjp has {} non-d_output Inputs; expected {}",
            body_other_inputs.len(),
            expected_others
        );
        let body_carry_in_off = body_other_inputs[0].1;
        let body_x_offs: Vec<usize> = body_other_inputs
            .iter()
            .skip(1)
            .map(|(_, off)| *off)
            .collect();
        let body_dcarry_out_off = body_offsets[&body_vjp.outputs[0]];
        let dxs_out_node = body_vjp.outputs[1 + *xs_idx as usize];
        let body_dxs_out_off = body_offsets[&dxs_out_node];

        let mut body_arena = crate::arena::Arena::from_plan(body_plan);
        for n in body_vjp.nodes() {
            if let Op::Constant { data } = &n.op
                && body_arena.has_buffer(n.id)
                && !data.is_empty()
            {
                match n.shape.dtype() {
                    rlx_ir::DType::F64 => {
                        let off = body_arena.byte_offset(n.id);
                        let buf = body_arena.raw_buf_mut();
                        let nb = (buf.len() - off).min(data.len());
                        buf[off..off + nb].copy_from_slice(&data[..nb]);
                    }
                    _ => {
                        let buf = body_arena.slice_mut(n.id);
                        let nf = data.len() / 4;
                        let nl = buf.len().min(nf);
                        for i in 0..nl {
                            let bytes = [
                                data[i * 4],
                                data[i * 4 + 1],
                                data[i * 4 + 2],
                                data[i * 4 + 3],
                            ];
                            buf[i] = f32::from_le_bytes(bytes);
                        }
                    }
                }
            }
        }
        let body_init = body_arena.raw_buf().to_vec();
        let body_schedule = compile_thunks_with_rng(body_vjp, &body_arena, rng);

        let carry_bytes = body_vjp
            .node(body_vjp.outputs[0])
            .shape
            .size_bytes()
            .expect("ScanBackwardXs dcarry must be statically shaped");
        let carry_elem_size = body_vjp
            .node(body_vjp.outputs[0])
            .shape
            .dtype()
            .size_bytes() as u32;
        let per_step_bytes = body_vjp
            .node(dxs_out_node)
            .shape
            .size_bytes()
            .expect("ScanBackwardXs dxs body output must be statically shaped");

        let mut outer_xs_offs: Vec<(usize, u32)> = Vec::with_capacity(*num_xs as usize);
        for i in 0..*num_xs as usize {
            let outer_xs_id = node.inputs[3 + i];
            let outer_xs_off = node_offset(arena, outer_xs_id);
            let outer_xs_shape = &graph.node(outer_xs_id).shape;
            let total = outer_xs_shape
                .size_bytes()
                .expect("ScanBackwardXs xs must have static shape");
            let per_step = total / *length as usize;
            outer_xs_offs.push((outer_xs_off, per_step as u32));
        }

        // Compile forward_body for recompute when checkpointed.
        // Mirrors the same code path in the ScanBackward arm.
        let (fb_schedule, fb_init, fb_carry_in_off, fb_output_off, fb_x_offs) = if is_recursive {
            let fb = forward_body.as_ref().unwrap();
            let fb_plan = rlx_opt::memory::plan_memory(fb);
            let fb_offsets: HashMap<NodeId, usize> = fb_plan
                .assignments
                .iter()
                .map(|(id, slot)| (*id, slot.offset))
                .collect();
            let mut fb_inputs: Vec<NodeId> = fb
                .nodes()
                .iter()
                .filter(|n| matches!(n.op, Op::Input { .. }))
                .map(|n| n.id)
                .collect();
            fb_inputs.sort();
            let fb_carry = fb_offsets[&fb_inputs[0]];
            let fb_xs: Vec<usize> = (1..fb_inputs.len())
                .map(|i| fb_offsets[&fb_inputs[i]])
                .collect();
            let fb_out = fb_offsets[&fb.outputs[0]];
            let mut fb_arena = crate::arena::Arena::from_plan(fb_plan);
            for n in fb.nodes() {
                if let Op::Constant { data } = &n.op
                    && fb_arena.has_buffer(n.id)
                    && !data.is_empty()
                {
                    // Byte-copy works for any
                    // numeric dtype as long as the
                    // arena slot is sized to hold
                    // it — the Constant's `data`
                    // already encodes the right
                    // bytes per element.
                    let off = fb_arena.byte_offset(n.id);
                    let buf = fb_arena.raw_buf_mut();
                    let nb = (buf.len() - off).min(data.len());
                    buf[off..off + nb].copy_from_slice(&data[..nb]);
                }
            }
            let fb_init_bytes = fb_arena.raw_buf().to_vec();
            let fb_sched = compile_thunks_with_rng(fb, &fb_arena, rng);
            (
                Some(Arc::new(fb_sched)),
                Some(Arc::new(fb_init_bytes)),
                fb_carry,
                fb_out,
                fb_xs,
            )
        } else {
            (None, None, 0, 0, Vec::new())
        };

        Thunk::ScanBackwardXs {
            body_vjp: Arc::new(body_schedule),
            body_init: Arc::new(body_init),
            body_carry_in_off,
            body_x_offs: Arc::new(body_x_offs),
            body_d_output_off,
            body_dcarry_out_off,
            body_dxs_out_off,
            outer_init_off: node_offset(arena, node.inputs[0]),
            outer_traj_off: node_offset(arena, node.inputs[1]),
            outer_upstream_off: node_offset(arena, node.inputs[2]),
            outer_xs_offs: Arc::new(outer_xs_offs),
            outer_dxs_off: node_offset(arena, node.id),
            length: *length,
            carry_bytes: carry_bytes as u32,
            carry_elem_size,
            per_step_bytes: per_step_bytes as u32,
            save_trajectory: *save_trajectory,
            num_checkpoints: *num_checkpoints,
            forward_body: fb_schedule,
            forward_body_init: fb_init,
            forward_body_carry_in_off: fb_carry_in_off,
            forward_body_output_off: fb_output_off,
            forward_body_x_offs: Arc::new(fb_x_offs),
        }
    }
}

#[inline(always)]
pub(crate) fn exec_scan(t: &Thunk, base: *mut u8) {
    let Thunk::Scan {
        body,
        body_init,
        body_input_off,
        body_output_off,
        outer_init_off,
        outer_final_off,
        length,
        carry_bytes,
        save_trajectory,
        xs_inputs,
        bcast_inputs,
        num_checkpoints,
    } = t
    else {
        unreachable!()
    };
    {
        let cb = *carry_bytes as usize;
        let n_steps = *length as usize;
        // Checkpoint mode: when 0 < K < length, save trajectory[k]
        // only when t == c_k = floor((k+1) * length / K) - 1.
        // The last index c_{K-1} = length - 1 always.
        let k_total = if *num_checkpoints == 0 || *num_checkpoints == *length {
            n_steps // save every step
        } else {
            *num_checkpoints as usize
        };
        let checkpoint_t_for_k = |k: usize| -> usize {
            if k_total == n_steps {
                k
            } else {
                ((k + 1) * n_steps)
                    .div_ceil(k_total)
                    .saturating_sub(1)
                    .min(n_steps - 1)
            }
        };
        let mut next_k = 0usize;

        let mut body_buf: Vec<u8> = (**body_init).clone();
        unsafe {
            std::ptr::copy_nonoverlapping(
                base.add(*outer_init_off),
                body_buf.as_mut_ptr().add(*body_input_off),
                cb,
            );
            // Broadcast inputs: copy each one into the body's
            // input slot ONCE. They aren't touched in the
            // iteration loop below (in contrast to xs).
            for (body_b_off, outer_b_off, total_bytes) in bcast_inputs.iter() {
                std::ptr::copy_nonoverlapping(
                    base.add(*outer_b_off),
                    body_buf.as_mut_ptr().add(*body_b_off),
                    *total_bytes as usize,
                );
            }
        }
        for t in 0..n_steps {
            for (body_x_off, outer_xs_off, per_step_bytes) in xs_inputs.iter() {
                let psb = *per_step_bytes as usize;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        base.add(*outer_xs_off + t * psb),
                        body_buf.as_mut_ptr().add(*body_x_off),
                        psb,
                    );
                }
            }

            execute_thunks(body, &mut body_buf);

            if *save_trajectory && next_k < k_total && t == checkpoint_t_for_k(next_k) {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        body_buf.as_ptr().add(*body_output_off),
                        base.add(*outer_final_off + next_k * cb),
                        cb,
                    );
                }
                next_k += 1;
            }

            if *body_output_off != *body_input_off {
                body_buf.copy_within(*body_output_off..*body_output_off + cb, *body_input_off);
            }
        }

        if !*save_trajectory {
            // Single final-carry write.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    body_buf.as_ptr().add(*body_output_off),
                    base.add(*outer_final_off),
                    cb,
                );
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_scan_backward_xs(t: &Thunk, base: *mut u8) {
    let Thunk::ScanBackwardXs {
        body_vjp,
        body_init,
        body_carry_in_off,
        body_x_offs,
        body_d_output_off,
        body_dcarry_out_off,
        body_dxs_out_off,
        outer_init_off,
        outer_traj_off,
        outer_upstream_off,
        outer_xs_offs,
        outer_dxs_off,
        length,
        carry_bytes,
        carry_elem_size,
        per_step_bytes,
        save_trajectory,
        num_checkpoints,
        forward_body,
        forward_body_init,
        forward_body_carry_in_off,
        forward_body_output_off,
        forward_body_x_offs,
    } = t
    else {
        unreachable!()
    };
    {
        let cb = *carry_bytes as usize;
        let psb = *per_step_bytes as usize;
        let n_steps = *length as usize;
        let k_total = *num_checkpoints as usize;
        let is_recursive = k_total != 0 && k_total != n_steps;
        let checkpoint_t_for_k = |k: usize| -> usize {
            ((k + 1) * n_steps)
                .div_ceil(k_total)
                .saturating_sub(1)
                .min(n_steps - 1)
        };

        // Forward-body recompute scratch + segment cache —
        // exact mirror of the ScanBackward path. With ≈√length
        // checkpoints, total recompute work is O(length).
        let mut fwd_buf: Vec<u8> = if is_recursive {
            (**forward_body_init.as_ref().unwrap()).clone()
        } else {
            Vec::new()
        };
        let mut seg_cache: Vec<u8> = Vec::new();
        let mut seg_start_t: usize = usize::MAX;
        let mut seg_count: usize = 0;
        let recompute_carry_t = |t: usize,
                                 dst: &mut [u8],
                                 fwd_buf: &mut Vec<u8>,
                                 seg_cache: &mut Vec<u8>,
                                 seg_start_t: &mut usize,
                                 seg_count: &mut usize| {
            if !is_recursive {
                unsafe {
                    let src = if t == 0 {
                        base.add(*outer_init_off)
                    } else {
                        base.add(*outer_traj_off + (t - 1) * cb)
                    };
                    std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), cb);
                }
                return;
            }
            if *seg_start_t != usize::MAX && t >= *seg_start_t && t < *seg_start_t + *seg_count {
                let off = (t - *seg_start_t) * cb;
                dst.copy_from_slice(&seg_cache[off..off + cb]);
                return;
            }
            let seg_k = (0..k_total)
                .find(|&k| t <= checkpoint_t_for_k(k))
                .unwrap_or(k_total - 1);
            let (anchor_t, anchor_ptr): (usize, *const u8) = if seg_k == 0 {
                (0, unsafe { base.add(*outer_init_off) as *const u8 })
            } else {
                let prev_ck = checkpoint_t_for_k(seg_k - 1);
                (prev_ck + 1, unsafe {
                    base.add(*outer_traj_off + (seg_k - 1) * cb) as *const u8
                })
            };
            let seg_end_t = checkpoint_t_for_k(seg_k);
            let seg_size = seg_end_t - anchor_t + 1;

            fwd_buf.copy_from_slice(forward_body_init.as_ref().unwrap());
            unsafe {
                std::ptr::copy_nonoverlapping(
                    anchor_ptr,
                    fwd_buf.as_mut_ptr().add(*forward_body_carry_in_off),
                    cb,
                );
            }
            seg_cache.resize(seg_size * cb, 0u8);
            seg_cache[0..cb].copy_from_slice(
                &fwd_buf[*forward_body_carry_in_off..*forward_body_carry_in_off + cb],
            );
            let fb_sched = forward_body.as_ref().unwrap();
            for i in 1..seg_size {
                let cur_iter = anchor_t + i - 1;
                for (idx, fb_x_off) in forward_body_x_offs.iter().enumerate() {
                    let (outer_xs_off, x_psb) = outer_xs_offs[idx];
                    let xb = x_psb as usize;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            base.add(outer_xs_off + cur_iter * xb),
                            fwd_buf.as_mut_ptr().add(*fb_x_off),
                            xb,
                        );
                    }
                }
                execute_thunks(fb_sched, fwd_buf);
                if *forward_body_output_off != *forward_body_carry_in_off {
                    fwd_buf.copy_within(
                        *forward_body_output_off..*forward_body_output_off + cb,
                        *forward_body_carry_in_off,
                    );
                }
                let cache_off = i * cb;
                seg_cache[cache_off..cache_off + cb].copy_from_slice(
                    &fwd_buf[*forward_body_carry_in_off..*forward_body_carry_in_off + cb],
                );
            }
            *seg_start_t = anchor_t;
            *seg_count = seg_size;

            let off = (t - anchor_t) * cb;
            dst.copy_from_slice(&seg_cache[off..off + cb]);
        };

        let mut dcarry: Vec<u8> = vec![0u8; cb];
        if !*save_trajectory {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    base.add(*outer_upstream_off),
                    dcarry.as_mut_ptr(),
                    cb,
                );
            }
        }

        let mut body_buf: Vec<u8> = (**body_init).clone();

        for t in (0..n_steps).rev() {
            if *save_trajectory {
                unsafe {
                    let up_off = *outer_upstream_off + t * cb;
                    match *carry_elem_size {
                        4 => {
                            let up_ptr = base.add(up_off) as *const f32;
                            let dc_ptr = dcarry.as_mut_ptr() as *mut f32;
                            let n_elems = cb / 4;
                            for i in 0..n_elems {
                                *dc_ptr.add(i) += *up_ptr.add(i);
                            }
                        }
                        8 => {
                            let up_ptr = base.add(up_off) as *const f64;
                            let dc_ptr = dcarry.as_mut_ptr() as *mut f64;
                            let n_elems = cb / 8;
                            for i in 0..n_elems {
                                *dc_ptr.add(i) += *up_ptr.add(i);
                            }
                        }
                        other => panic!(
                            "ScanBackwardXs: unsupported carry elem size {other} \
                                     (only f32/f64 carries are supported today)"
                        ),
                    }
                }
            }

            // Seed body_vjp's carry input via the recompute
            // helper (works for both All and Recursive modes),
            // then x_t_i + d_output.
            let carry_dst_start = *body_carry_in_off;
            {
                let carry_slice = &mut body_buf[carry_dst_start..carry_dst_start + cb];
                recompute_carry_t(
                    t,
                    carry_slice,
                    &mut fwd_buf,
                    &mut seg_cache,
                    &mut seg_start_t,
                    &mut seg_count,
                );
            }
            unsafe {
                for (i, body_x_off) in body_x_offs.iter().enumerate() {
                    let (outer_xs_off, x_psb) = outer_xs_offs[i];
                    let xb = x_psb as usize;
                    std::ptr::copy_nonoverlapping(
                        base.add(outer_xs_off + t * xb),
                        body_buf.as_mut_ptr().add(*body_x_off),
                        xb,
                    );
                }
                std::ptr::copy_nonoverlapping(
                    dcarry.as_ptr(),
                    body_buf.as_mut_ptr().add(*body_d_output_off),
                    cb,
                );
            }

            execute_thunks(body_vjp, &mut body_buf);

            // Stash this step's dxs into row `t` of the outer
            // [length, *per_step_xs] output.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    body_buf.as_ptr().add(*body_dxs_out_off),
                    base.add(*outer_dxs_off + t * psb),
                    psb,
                );
            }

            // Update dcarry for next backward iteration.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    body_buf.as_ptr().add(*body_dcarry_out_off),
                    dcarry.as_mut_ptr(),
                    cb,
                );
            }
        }
    }
}

#[inline(always)]
pub(crate) fn exec_gated_delta_net(t: &Thunk, base: *mut u8) {
    let Thunk::GatedDeltaNet {
        q,
        k,
        v,
        g,
        beta,
        state,
        dst,
        batch,
        seq,
        heads,
        state_size,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_gated_delta_net_f32(
            *q,
            *k,
            *v,
            *g,
            *beta,
            *state,
            *dst,
            *batch as usize,
            *seq as usize,
            *heads as usize,
            *state_size as usize,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_lstm(t: &Thunk, base: *mut u8) {
    let Thunk::Lstm {
        x,
        w_ih,
        w_hh,
        bias,
        h0,
        c0,
        dst,
        batch,
        seq,
        input_size,
        hidden,
        num_layers,
        bidirectional,
        carry,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_lstm_f32(
            *x,
            *w_ih,
            *w_hh,
            *bias,
            *h0,
            *c0,
            *dst,
            *batch as usize,
            *seq as usize,
            *input_size as usize,
            *hidden as usize,
            *num_layers as usize,
            *bidirectional,
            *carry,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_gru(t: &Thunk, base: *mut u8) {
    let Thunk::Gru {
        x,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        h0,
        dst,
        batch,
        seq,
        input_size,
        hidden,
        num_layers,
        bidirectional,
        carry,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_gru_f32(
            *x,
            *w_ih,
            *w_hh,
            *b_ih,
            *b_hh,
            *h0,
            *dst,
            *batch as usize,
            *seq as usize,
            *input_size as usize,
            *hidden as usize,
            *num_layers as usize,
            *bidirectional,
            *carry,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_rnn(t: &Thunk, base: *mut u8) {
    let Thunk::Rnn {
        x,
        w_ih,
        w_hh,
        bias,
        h0,
        dst,
        batch,
        seq,
        input_size,
        hidden,
        num_layers,
        bidirectional,
        carry,
        relu,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_rnn_f32(
            *x,
            *w_ih,
            *w_hh,
            *bias,
            *h0,
            *dst,
            *batch as usize,
            *seq as usize,
            *input_size as usize,
            *hidden as usize,
            *num_layers as usize,
            *bidirectional,
            *carry,
            *relu,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_mamba2(t: &Thunk, base: *mut u8) {
    let Thunk::Mamba2 {
        x,
        dt,
        a,
        b,
        c,
        dst,
        batch,
        seq,
        heads,
        head_dim,
        state_size,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_mamba2_f32(
            *x,
            *dt,
            *a,
            *b,
            *c,
            *dst,
            *batch as usize,
            *seq as usize,
            *heads as usize,
            *head_dim as usize,
            *state_size as usize,
            base,
        );
    }
}

#[inline(always)]
pub(crate) fn exec_selective_scan(t: &Thunk, base: *mut u8) {
    let Thunk::SelectiveScan {
        x,
        delta,
        a,
        b: bp,
        c: cp,
        dst,
        batch,
        seq,
        hidden,
        state_size,
    } = t
    else {
        unreachable!()
    };
    unsafe {
        execute_selective_scan_f32(
            *x,
            *delta,
            *a,
            *bp,
            *cp,
            *dst,
            *batch as usize,
            *seq as usize,
            *hidden as usize,
            *state_size as usize,
            base,
        );
    }
}

/// Griewank treeverse: process backward iterations `[t_lo..=t_hi]` (with
/// the carry entering iteration `t_lo` supplied as `anchor_carry`) by
/// recursive binary subdivision. Total work `O((t_hi-t_lo+1) · log)`,
/// auxiliary memory `O(log · carry_bytes)` for the recursion stack.
///
/// Compared to the iterative segment-cached scheme, this trades extra
/// recompute for less working memory — each level of recursion holds
/// one `cb`-sized intermediate carry on the stack but never the whole
/// segment at once. With K saved outer checkpoints, the outer driver
/// invokes this helper once per segment.
///
/// `process_iter(t, carry_at_t)` is the per-iteration leaf action: it
/// runs `body_vjp` at iteration `t` with the supplied carry, threads
/// `dcarry` backward, and (for ScanBackwardXs) writes `dxs[t]`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn griewank_process_segment(
    t_lo: usize,
    t_hi: usize,
    anchor_carry: &[u8],
    cb: usize,
    fwd_sched: &ThunkSchedule,
    fwd_init: &[u8],
    fwd_carry_in_off: usize,
    fwd_output_off: usize,
    fwd_x_offs: &[usize],
    base: *mut u8,
    outer_xs_offs: &[(usize, u32)],
    fwd_buf: &mut Vec<u8>,
    leaf_threshold: usize,
    process_iter: &mut dyn FnMut(usize, &[u8]),
) {
    unsafe {
        let size = t_hi - t_lo + 1;
        if size == 1 {
            process_iter(t_lo, anchor_carry);
            return;
        }
        if size <= leaf_threshold {
            // Walk forward, cache each carry, run backward in reverse.
            let mut cache: Vec<u8> = Vec::with_capacity(size * cb);
            cache.extend_from_slice(anchor_carry);
            fwd_buf.copy_from_slice(fwd_init);
            std::ptr::copy_nonoverlapping(
                anchor_carry.as_ptr(),
                fwd_buf.as_mut_ptr().add(fwd_carry_in_off),
                cb,
            );
            for i in 1..size {
                let cur_iter = t_lo + i - 1;
                for (idx, fb_x_off) in fwd_x_offs.iter().enumerate() {
                    let (outer_xs_off, x_psb) = outer_xs_offs[idx];
                    let xb = x_psb as usize;
                    std::ptr::copy_nonoverlapping(
                        base.add(outer_xs_off + cur_iter * xb),
                        fwd_buf.as_mut_ptr().add(*fb_x_off),
                        xb,
                    );
                }
                execute_thunks(fwd_sched, fwd_buf);
                if fwd_output_off != fwd_carry_in_off {
                    fwd_buf.copy_within(fwd_output_off..fwd_output_off + cb, fwd_carry_in_off);
                }
                cache.extend_from_slice(&fwd_buf[fwd_carry_in_off..fwd_carry_in_off + cb]);
            }
            // Process backward.
            for t in (t_lo..=t_hi).rev() {
                let idx = t - t_lo;
                let carry = &cache[idx * cb..(idx + 1) * cb];
                process_iter(t, carry);
            }
            return;
        }

        // Split: walk forward from anchor to compute carry entering `mid`.
        // (We need `mid - t_lo` body executions: one per iteration in
        // [t_lo, mid).)
        let mid = t_lo + size / 2;
        fwd_buf.copy_from_slice(fwd_init);
        std::ptr::copy_nonoverlapping(
            anchor_carry.as_ptr(),
            fwd_buf.as_mut_ptr().add(fwd_carry_in_off),
            cb,
        );
        for cur_iter in t_lo..mid {
            for (idx, fb_x_off) in fwd_x_offs.iter().enumerate() {
                let (outer_xs_off, x_psb) = outer_xs_offs[idx];
                let xb = x_psb as usize;
                std::ptr::copy_nonoverlapping(
                    base.add(outer_xs_off + cur_iter * xb),
                    fwd_buf.as_mut_ptr().add(*fb_x_off),
                    xb,
                );
            }
            execute_thunks(fwd_sched, fwd_buf);
            if fwd_output_off != fwd_carry_in_off {
                fwd_buf.copy_within(fwd_output_off..fwd_output_off + cb, fwd_carry_in_off);
            }
        }
        let mid_carry: Vec<u8> = fwd_buf[fwd_carry_in_off..fwd_carry_in_off + cb].to_vec();

        // Right half first (higher t values processed first to match the
        // canonical reverse-mode iteration order: dcarry threads from
        // t=length-1 down to t=0).
        griewank_process_segment(
            mid,
            t_hi,
            &mid_carry,
            cb,
            fwd_sched,
            fwd_init,
            fwd_carry_in_off,
            fwd_output_off,
            fwd_x_offs,
            base,
            outer_xs_offs,
            fwd_buf,
            leaf_threshold,
            process_iter,
        );
        // Then left half with original anchor.
        griewank_process_segment(
            t_lo,
            mid - 1,
            anchor_carry,
            cb,
            fwd_sched,
            fwd_init,
            fwd_carry_in_off,
            fwd_output_off,
            fwd_x_offs,
            base,
            outer_xs_offs,
            fwd_buf,
            leaf_threshold,
            process_iter,
        );
    }
}

/// Expected packed-buffer element counts for the RNN-family ops, derived
/// purely from the op parameters. `gates` = 4 (LSTM), 3 (GRU), 1 (RNN).
///
/// The `execute_{lstm,gru,rnn}_f32` kernels read their weights through raw
/// pointers sized from these same parameters, so a caller that supplies a
/// mis-shaped buffer (e.g. a `w_hh` missing the `×hidden` factor) reads out of
/// bounds and returns silent garbage instead of erroring. Callers should
/// validate their buffers against this up front — see [`check_rnn_lens`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RnnLens {
    /// `x`: `batch·seq·input_size`.
    pub x: usize,
    /// `w_ih`: Σ over layers of `dirs·gates·hidden·in_l(l)` (`in_l` grows to `dirs·hidden`).
    pub w_ih: usize,
    /// `w_hh`: `layers·dirs·gates·hidden·hidden`.
    pub w_hh: usize,
    /// One bias stream: `layers·dirs·gates·hidden` (LSTM/RNN merge into one; GRU has `b_ih` and `b_hh`).
    pub bias: usize,
    /// Per-state carry buffer (`h0` and, for LSTM, `c0`): `layers·dirs·batch·hidden`.
    pub state: usize,
    /// Output `dst`: `batch·seq·dirs·hidden`.
    pub out: usize,
}

/// Expected buffer sizes for an RNN-family op with the given parameters.
pub fn rnn_expected_lens(
    gates: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
) -> RnnLens {
    let dirs = if bidirectional { 2 } else { 1 };
    let w_ih: usize = (0..num_layers)
        .map(|l| dirs * gates * hidden * if l == 0 { input_size } else { dirs * hidden })
        .sum();
    RnnLens {
        x: batch * seq * input_size,
        w_ih,
        w_hh: num_layers * dirs * gates * hidden * hidden,
        bias: num_layers * dirs * gates * hidden,
        state: num_layers * dirs * batch * hidden,
        out: batch * seq * dirs * hidden,
    }
}

/// Validate a labeled `(actual, expected)` list, returning a clear error on the
/// first mismatch. Turns an out-of-bounds kernel read into a loud failure.
pub fn check_rnn_lens(op: &str, checks: &[(&str, usize, usize)]) -> Result<(), String> {
    for &(name, actual, expected) in checks {
        if actual != expected {
            return Err(format!(
                "{op}: input '{name}' has {actual} elements, expected {expected} \
                 (mismatch would read out of bounds in the f32 kernel)"
            ));
        }
    }
    Ok(())
}

/// Reference / host-fallback entry for `Op::Lstm` (multi-layer,
/// optionally bidirectional, optional decode carry). Gate order i, f, g,
/// o. Shared by the CPU backend and the CUDA / ROCm / wgpu / Metal host
/// fallbacks. Tensors are `f32` in the arena; weights are packed (see
/// `Op::Lstm`). `h0`/`c0` are byte offsets when `carry`, else ignored;
/// the final `hn`/`cn` are written back into them in place. `dst` is
/// `[batch, seq, D*hidden]`. Batch items run in parallel per direction.
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_lstm_f32(
    x: usize,
    w_ih: usize,
    w_hh: usize,
    bias: usize,
    h0: usize,
    c0: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
    base: *mut u8,
) {
    #[inline]
    fn sigmoid(z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }

    let bptr = base as usize;
    let four_h = 4 * hidden;
    let dirs = if bidirectional { 2 } else { 1 };

    unsafe {
        let f32s = |off: usize, n: usize| -> &[f32] {
            std::slice::from_raw_parts((bptr + off) as *const f32, n)
        };

        // Layer 0 reads x; later layers read the previous layer's output.
        let mut layer_in: Vec<f32> = f32s(x, batch * seq * input_size).to_vec();
        let mut in_l = input_size;
        // Running element cursor into the packed `w_ih` buffer (block width
        // `4h * in_l` varies per layer; `w_hh`/`bias` blocks are uniform).
        let mut wih_cursor = 0usize;

        for l in 0..num_layers {
            let out_width = dirs * hidden;
            let mut layer_out = vec![0f32; batch * seq * out_width];
            let lo_ptr = layer_out.as_mut_ptr() as usize;
            let li_ref: &[f32] = &layer_in;
            let wih_block = four_h * in_l;

            for dir in 0..dirs {
                let ld = l * dirs + dir;
                let wih = f32s((w_ih / 4 + wih_cursor + dir * wih_block) * 4, wih_block);
                let whh = f32s(w_hh + ld * four_h * hidden * 4, four_h * hidden);
                let bs = f32s(bias + ld * four_h * 4, four_h);
                let h0p = bptr + h0 + ld * batch * hidden * 4;
                let c0p = bptr + c0 + ld * batch * hidden * 4;

                crate::pool::par_range(batch, |b| {
                    let lo = lo_ptr as *mut f32;
                    let mut h = vec![0f32; hidden];
                    let mut c = vec![0f32; hidden];
                    if carry {
                        let hin = std::slice::from_raw_parts(
                            (h0p + b * hidden * 4) as *const f32,
                            hidden,
                        );
                        let cin = std::slice::from_raw_parts(
                            (c0p + b * hidden * 4) as *const f32,
                            hidden,
                        );
                        h.copy_from_slice(hin);
                        c.copy_from_slice(cin);
                    }
                    let mut z = vec![0f32; four_h];
                    for step in 0..seq {
                        let t = if dir == 0 { step } else { seq - 1 - step };
                        let x_t = &li_ref[(b * seq + t) * in_l..(b * seq + t + 1) * in_l];
                        for r in 0..four_h {
                            let wr = &wih[r * in_l..(r + 1) * in_l];
                            let mut acc = bs[r];
                            for j in 0..in_l {
                                acc += wr[j] * x_t[j];
                            }
                            let hr = &whh[r * hidden..(r + 1) * hidden];
                            for (j, &hj) in h.iter().enumerate() {
                                acc += hr[j] * hj;
                            }
                            z[r] = acc;
                        }
                        for k in 0..hidden {
                            let i_g = sigmoid(z[k]);
                            let f_g = sigmoid(z[hidden + k]);
                            let g_g = z[2 * hidden + k].tanh();
                            let o_g = sigmoid(z[3 * hidden + k]);
                            let c_new = f_g * c[k] + i_g * g_g;
                            c[k] = c_new;
                            let h_new = o_g * c_new.tanh();
                            h[k] = h_new;
                            // [batch, seq, D*hidden]; this direction owns the
                            // `dir*hidden .. dir*hidden+hidden` feature slice.
                            *lo.add((b * seq + t) * out_width + dir * hidden + k) = h_new;
                        }
                    }
                    if carry {
                        let hout = std::slice::from_raw_parts_mut(
                            (h0p + b * hidden * 4) as *mut f32,
                            hidden,
                        );
                        let cout = std::slice::from_raw_parts_mut(
                            (c0p + b * hidden * 4) as *mut f32,
                            hidden,
                        );
                        hout.copy_from_slice(&h);
                        cout.copy_from_slice(&c);
                    }
                });
            }

            wih_cursor += dirs * wih_block;
            layer_in = layer_out;
            in_l = out_width;
        }

        // Final layer output → dst [batch, seq, D*hidden].
        let dst_slice = std::slice::from_raw_parts_mut((bptr + dst) as *mut f32, layer_in.len());
        dst_slice.copy_from_slice(&layer_in);
    }
}

/// Reference / host-fallback for `Op::Gru` (PyTorch GRU; gate order r, z, n;
/// multi-layer / bidirectional / carry). Separate `b_ih`/`b_hh` because the
/// reset gate is applied to the hidden term *after* its bias:
/// `n = tanh(x·W_inᵀ+b_in + r ⊙ (h·W_hnᵀ+b_hn))`,
/// `h' = (1-z)⊙n + z⊙h`. Packing mirrors `Op::Lstm` with `3*hidden` gate rows.
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_gru_f32(
    x: usize,
    w_ih: usize,
    w_hh: usize,
    b_ih: usize,
    b_hh: usize,
    h0: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
    base: *mut u8,
) {
    #[inline]
    fn sigmoid(z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }

    let bptr = base as usize;
    let three_h = 3 * hidden;
    let dirs = if bidirectional { 2 } else { 1 };

    unsafe {
        let f32s = |off: usize, n: usize| -> &[f32] {
            std::slice::from_raw_parts((bptr + off) as *const f32, n)
        };

        let mut layer_in: Vec<f32> = f32s(x, batch * seq * input_size).to_vec();
        let mut in_l = input_size;
        let mut wih_cursor = 0usize;

        for l in 0..num_layers {
            let out_width = dirs * hidden;
            let mut layer_out = vec![0f32; batch * seq * out_width];
            let lo_ptr = layer_out.as_mut_ptr() as usize;
            let li_ref: &[f32] = &layer_in;
            let wih_block = three_h * in_l;

            for dir in 0..dirs {
                let ld = l * dirs + dir;
                let wih = f32s((w_ih / 4 + wih_cursor + dir * wih_block) * 4, wih_block);
                let whh = f32s(w_hh + ld * three_h * hidden * 4, three_h * hidden);
                let bih = f32s(b_ih + ld * three_h * 4, three_h);
                let bhh = f32s(b_hh + ld * three_h * 4, three_h);
                let h0p = bptr + h0 + ld * batch * hidden * 4;

                crate::pool::par_range(batch, |b| {
                    let lo = lo_ptr as *mut f32;
                    let mut h = vec![0f32; hidden];
                    if carry {
                        let hin = std::slice::from_raw_parts(
                            (h0p + b * hidden * 4) as *const f32,
                            hidden,
                        );
                        h.copy_from_slice(hin);
                    }
                    let mut xi = vec![0f32; three_h]; // x·W_ihᵀ + b_ih
                    let mut hi = vec![0f32; three_h]; // h·W_hhᵀ + b_hh
                    for step in 0..seq {
                        let t = if dir == 0 { step } else { seq - 1 - step };
                        let x_t = &li_ref[(b * seq + t) * in_l..(b * seq + t + 1) * in_l];
                        for r in 0..three_h {
                            let wr = &wih[r * in_l..(r + 1) * in_l];
                            let mut a = bih[r];
                            for j in 0..in_l {
                                a += wr[j] * x_t[j];
                            }
                            xi[r] = a;
                            let hr = &whh[r * hidden..(r + 1) * hidden];
                            let mut bb = bhh[r];
                            for (j, &hj) in h.iter().enumerate() {
                                bb += hr[j] * hj;
                            }
                            hi[r] = bb;
                        }
                        for k in 0..hidden {
                            let rg = sigmoid(xi[k] + hi[k]);
                            let zg = sigmoid(xi[hidden + k] + hi[hidden + k]);
                            // Reset applied to hidden term after its bias.
                            let ng = (xi[2 * hidden + k] + rg * hi[2 * hidden + k]).tanh();
                            let h_new = (1.0 - zg) * ng + zg * h[k];
                            h[k] = h_new;
                            *lo.add((b * seq + t) * out_width + dir * hidden + k) = h_new;
                        }
                    }
                    if carry {
                        let hout = std::slice::from_raw_parts_mut(
                            (h0p + b * hidden * 4) as *mut f32,
                            hidden,
                        );
                        hout.copy_from_slice(&h);
                    }
                });
            }

            wih_cursor += dirs * wih_block;
            layer_in = layer_out;
            in_l = out_width;
        }

        let dst_slice = std::slice::from_raw_parts_mut((bptr + dst) as *mut f32, layer_in.len());
        dst_slice.copy_from_slice(&layer_in);
    }
}

/// Reference / host-fallback for `Op::Rnn` (Elman; `act = relu` when `relu`
/// else `tanh`; multi-layer / bidirectional / carry). Single merged bias.
/// `h' = act(x·W_ihᵀ + h·W_hhᵀ + bias)`. Packing mirrors `Op::Lstm` with
/// `hidden` gate rows.
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_rnn_f32(
    x: usize,
    w_ih: usize,
    w_hh: usize,
    bias: usize,
    h0: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    num_layers: usize,
    bidirectional: bool,
    carry: bool,
    relu: bool,
    base: *mut u8,
) {
    let bptr = base as usize;
    let dirs = if bidirectional { 2 } else { 1 };

    unsafe {
        let f32s = |off: usize, n: usize| -> &[f32] {
            std::slice::from_raw_parts((bptr + off) as *const f32, n)
        };

        let mut layer_in: Vec<f32> = f32s(x, batch * seq * input_size).to_vec();
        let mut in_l = input_size;
        let mut wih_cursor = 0usize;

        for l in 0..num_layers {
            let out_width = dirs * hidden;
            let mut layer_out = vec![0f32; batch * seq * out_width];
            let lo_ptr = layer_out.as_mut_ptr() as usize;
            let li_ref: &[f32] = &layer_in;
            let wih_block = hidden * in_l;

            for dir in 0..dirs {
                let ld = l * dirs + dir;
                let wih = f32s((w_ih / 4 + wih_cursor + dir * wih_block) * 4, wih_block);
                let whh = f32s(w_hh + ld * hidden * hidden * 4, hidden * hidden);
                let bs = f32s(bias + ld * hidden * 4, hidden);
                let h0p = bptr + h0 + ld * batch * hidden * 4;

                crate::pool::par_range(batch, |b| {
                    let lo = lo_ptr as *mut f32;
                    let mut h = vec![0f32; hidden];
                    if carry {
                        let hin = std::slice::from_raw_parts(
                            (h0p + b * hidden * 4) as *const f32,
                            hidden,
                        );
                        h.copy_from_slice(hin);
                    }
                    for step in 0..seq {
                        let t = if dir == 0 { step } else { seq - 1 - step };
                        let x_t = &li_ref[(b * seq + t) * in_l..(b * seq + t + 1) * in_l];
                        let mut h_new = vec![0f32; hidden];
                        for k in 0..hidden {
                            let wr = &wih[k * in_l..(k + 1) * in_l];
                            let mut acc = bs[k];
                            for j in 0..in_l {
                                acc += wr[j] * x_t[j];
                            }
                            let hr = &whh[k * hidden..(k + 1) * hidden];
                            for (j, &hj) in h.iter().enumerate() {
                                acc += hr[j] * hj;
                            }
                            h_new[k] = if relu { acc.max(0.0) } else { acc.tanh() };
                        }
                        for k in 0..hidden {
                            h[k] = h_new[k];
                            *lo.add((b * seq + t) * out_width + dir * hidden + k) = h_new[k];
                        }
                    }
                    if carry {
                        let hout = std::slice::from_raw_parts_mut(
                            (h0p + b * hidden * 4) as *mut f32,
                            hidden,
                        );
                        hout.copy_from_slice(&h);
                    }
                });
            }

            wih_cursor += dirs * wih_block;
            layer_in = layer_out;
            in_l = out_width;
        }

        let dst_slice = std::slice::from_raw_parts_mut((bptr + dst) as *mut f32, layer_in.len());
        dst_slice.copy_from_slice(&layer_in);
    }
}

/// Reference for `Op::Mamba2` — SSD scalar-decay SSM scan. Inputs (f32):
/// `x [B,S,H,P]`, `dt [B,S,H]`, `a [H]`, `b/c [B,S,H,N]`. Per `(batch, head)`
/// the state `S[P,N]` is zero-init: `dA=exp(dt·a)`, `S=dA·S + (dt·x)⊗b`,
/// `y=Σ_n S[:,n]·c[n]`. Output `[B,S,H,P]`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn execute_mamba2_f32(
    x: usize,
    dt: usize,
    a: usize,
    b: usize,
    c: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    state_size: usize,
    base: *mut u8,
) {
    let (bn, s, h, p, n) = (batch, seq, heads, head_dim, state_size);
    let bptr = base as usize;
    unsafe {
        let f32s = |off: usize, len: usize| -> &[f32] {
            std::slice::from_raw_parts((bptr + off) as *const f32, len)
        };
        let xs = f32s(x, bn * s * h * p);
        let dts = f32s(dt, bn * s * h);
        let am = f32s(a, h);
        let bm = f32s(b, bn * s * h * n);
        let cm = f32s(c, bn * s * h * n);
        let out_ptr = bptr + dst;

        // Independent per (batch, head).
        crate::pool::par_range(bn * h, |bh| {
            let bi = bh / h;
            let hi = bh % h;
            let out = out_ptr as *mut f32;
            let mut state = vec![0f32; p * n];
            for t in 0..s {
                let dt_t = dts[(bi * s + t) * h + hi];
                let da = (dt_t * am[hi]).exp();
                let x_off = ((bi * s + t) * h + hi) * p;
                let bc_off = ((bi * s + t) * h + hi) * n;
                for pi in 0..p {
                    let dtx = dt_t * xs[x_off + pi];
                    for ni in 0..n {
                        state[pi * n + ni] = da * state[pi * n + ni] + dtx * bm[bc_off + ni];
                    }
                }
                for pi in 0..p {
                    let mut acc = 0f32;
                    for ni in 0..n {
                        acc += state[pi * n + ni] * cm[bc_off + ni];
                    }
                    *out.add(x_off + pi) = acc;
                }
            }
        });
    }
}

/// Mamba selective-scan reference (f32). Shared by the CPU `Thunk::SelectiveScan`
/// arm and the Metal unified-memory host fallback. `base` is the arena byte
/// pointer; the five inputs (`x`, `delta`, `a`, `b`, `c`) and `dst` are byte
/// offsets into it. Recurrence per `(batch, seq, channel, state)`:
/// `h[t] = exp(Δ·A)·h[t-1] + Δ·B·x;  y = Σ_n C·h`. State resets per batch row.
pub unsafe fn execute_selective_scan_f32(
    x: usize,
    delta: usize,
    a: usize,
    b: usize,
    c: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    hidden: usize,
    state_size: usize,
    base: *mut u8,
) {
    let (bn, s, h, n) = (batch, seq, hidden, state_size);
    unsafe {
        let xs = sl(x, base, bn * s * h);
        let dt = sl(delta, base, bn * s * h);
        let am = sl(a, base, h * n);
        let bm = sl(b, base, bn * s * n);
        let cm = sl(c, base, bn * s * n);
        let out = sl_mut(dst, base, bn * s * h);

        // State buffer per-batch: h channels × n state. Sequential along
        // the seq dimension; reset at the start of each batch row.
        let mut state = vec![0f32; h * n];
        for bi in 0..bn {
            for v in state.iter_mut() {
                *v = 0.0;
            }
            for si in 0..s {
                let x_row = &xs[bi * s * h + si * h..bi * s * h + (si + 1) * h];
                let dt_row = &dt[bi * s * h + si * h..bi * s * h + (si + 1) * h];
                let b_row = &bm[bi * s * n + si * n..bi * s * n + (si + 1) * n];
                let c_row = &cm[bi * s * n + si * n..bi * s * n + (si + 1) * n];
                let out_row = &mut out[bi * s * h + si * h..bi * s * h + (si + 1) * h];

                for ci in 0..h {
                    let d = dt_row[ci];
                    let xv = x_row[ci];
                    let mut acc = 0f32;
                    for ni in 0..n {
                        // Discretize: exp(d * a) and d * b.
                        let da = (d * am[ci * n + ni]).exp();
                        state[ci * n + ni] = da * state[ci * n + ni] + d * b_row[ni] * xv;
                        acc += c_row[ni] * state[ci * n + ni];
                    }
                    out_row[ci] = acc;
                }
            }
        }
    }
}

pub unsafe fn execute_gated_delta_net_f32(
    q: usize,
    k: usize,
    v: usize,
    g: usize,
    beta: usize,
    state: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    heads: usize,
    state_size: usize,
    base: *mut u8,
) {
    #[derive(Copy, Clone)]
    struct ArenaPtr(usize);
    unsafe impl Send for ArenaPtr {}
    unsafe impl Sync for ArenaPtr {}
    impl ArenaPtr {
        #[inline]
        fn get(self) -> *mut u8 {
            self.0 as *mut u8
        }
    }

    unsafe {
        let arena = ArenaPtr(base as usize);
        let (b, s, h, n) = (batch, seq, heads, state_size);
        let scale = 1.0f32 / (n as f32).sqrt();
        let use_external = state != 0;
        let mut owned_state = vec![0f32; h * n * n];

        crate::pool::num_threads();

        assert!(
            n <= crate::gdn::GDN_MAX_STATE,
            "GatedDeltaNet state_size={n} exceeds stack scratch ({})",
            crate::gdn::GDN_MAX_STATE
        );

        let qs = sl(q, arena.get(), b * s * h * n);
        let ks = sl(k, arena.get(), b * s * h * n);
        let vs = sl(v, arena.get(), b * s * h * n);
        let gs = sl(g, arena.get(), b * s * h);
        let betas = sl(beta, arena.get(), b * s * h);
        let _out = sl_mut(dst, arena.get(), b * s * h * n);
        let hs_n = h * n;

        let run_head = |bi: usize, hi: usize, s_mat: &mut [f32], sk: &mut [f32]| {
            for ti in 0..s {
                let qkv_step = bi * s * hs_n + ti * hs_n + hi * n;
                let gb_step = bi * s * h + ti * h + hi;
                let out_row = sl_mut(dst + qkv_step * std::mem::size_of::<f32>(), arena.get(), n);
                crate::gdn::gdn_step_blas(
                    s_mat,
                    &qs[qkv_step..qkv_step + n],
                    &ks[qkv_step..qkv_step + n],
                    &vs[qkv_step..qkv_step + n],
                    gs[gb_step],
                    betas[gb_step],
                    out_row,
                    sk,
                    n,
                    scale,
                );
            }
        };

        // Prefill (seq>1, ephemeral state): time-outer, parallel over heads —
        // better occupancy than head-outer when prompt length dominates.
        if !use_external && s > 1 {
            for bi in 0..b {
                crate::pool::par_range(h, |hi| {
                    let mut sk_buf = [0f32; crate::gdn::GDN_MAX_STATE];
                    let sk = &mut sk_buf[..n];
                    let mut local_state =
                        [0f32; crate::gdn::GDN_MAX_STATE * crate::gdn::GDN_MAX_STATE];
                    let s_mat = &mut local_state[..n * n];
                    s_mat.fill(0.0);
                    run_head(bi, hi, s_mat, sk);
                });
            }
            return;
        }

        if use_external {
            let state_bytes = state;
            crate::pool::par_range(b * h, |bhi| {
                let bi = bhi / h;
                let hi = bhi % h;
                let elem_off = bi * h * n * n + hi * n * n;
                let s_mat = sl_mut(
                    state_bytes + elem_off * std::mem::size_of::<f32>(),
                    arena.get(),
                    n * n,
                );
                let mut sk_buf = [0f32; crate::gdn::GDN_MAX_STATE];
                run_head(bi, hi, s_mat, &mut sk_buf[..n]);
            });
        } else {
            for bi in 0..b {
                owned_state.fill(0.0);
                #[cfg(not(target_arch = "wasm32"))]
                {
                    use rayon::prelude::*;
                    owned_state
                        .par_chunks_mut(n * n)
                        .enumerate()
                        .for_each(|(hi, s_mat)| {
                            let mut sk_buf = [0f32; crate::gdn::GDN_MAX_STATE];
                            run_head(bi, hi, s_mat, &mut sk_buf[..n]);
                        });
                }
                #[cfg(target_arch = "wasm32")]
                {
                    owned_state
                        .chunks_mut(n * n)
                        .enumerate()
                        .for_each(|(hi, s_mat)| {
                            let mut sk_buf = [0f32; crate::gdn::GDN_MAX_STATE];
                            run_head(bi, hi, s_mat, &mut sk_buf[..n]);
                        });
                }
            }
        }
    }
}

/// Host-fallback entry for f16 `Op::GatedDeltaNet` tensors on Metal.
pub unsafe fn execute_gated_delta_net_f16(
    q: usize,
    k: usize,
    v: usize,
    g: usize,
    beta: usize,
    state: usize,
    dst: usize,
    batch: usize,
    seq: usize,
    heads: usize,
    state_size: usize,
    base: *mut u8,
) {
    use half::f16;
    unsafe {
        let read_f16 = |off: usize, len: usize| -> Vec<f32> {
            let raw = std::slice::from_raw_parts(base.add(off) as *const u8, len * 2);
            raw.chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        };
        let write_f16 = |off: usize, data: &[f32]| {
            let out = std::slice::from_raw_parts_mut(base.add(off), data.len() * 2);
            for (i, &v) in data.iter().enumerate() {
                let le = f16::from_f32(v).to_le_bytes();
                out[i * 2] = le[0];
                out[i * 2 + 1] = le[1];
            }
        };

        let (b, s, h, n) = (batch, seq, heads, state_size);
        let q_f = read_f16(q, b * s * h * n);
        let k_f = read_f16(k, b * s * h * n);
        let v_f = read_f16(v, b * s * h * n);
        let g_f = read_f16(g, b * s * h);
        let b_f = read_f16(beta, b * s * h);
        let mut state_f = if state != 0 {
            read_f16(state, b * h * n * n)
        } else {
            vec![0f32; b * h * n * n]
        };
        let mut out_f = vec![0f32; b * s * h * n];
        let scale = 1.0f32 / (n as f32).sqrt();
        let mut sk_buf = vec![0f32; n];
        let mut owned_state = vec![0f32; h * n * n];

        for bi in 0..b {
            let state_slice: &mut [f32] = if state != 0 {
                let start = bi * h * n * n;
                &mut state_f[start..start + h * n * n]
            } else {
                owned_state.fill(0.0);
                &mut owned_state
            };

            for ti in 0..s {
                let qkv_step_base = bi * s * h * n + ti * h * n;
                let gb_step_base = bi * s * h + ti * h;

                for hi in 0..h {
                    let q_row = &q_f[qkv_step_base + hi * n..qkv_step_base + (hi + 1) * n];
                    let k_row = &k_f[qkv_step_base + hi * n..qkv_step_base + (hi + 1) * n];
                    let v_row = &v_f[qkv_step_base + hi * n..qkv_step_base + (hi + 1) * n];
                    let g_t = g_f[gb_step_base + hi];
                    let beta_t = b_f[gb_step_base + hi];

                    let s_base = hi * n * n;
                    let s_mat = &mut state_slice[s_base..s_base + n * n];

                    let g_exp = g_t.exp();
                    for st in s_mat.iter_mut() {
                        *st *= g_exp;
                    }

                    for j in 0..n {
                        let mut acc = 0f32;
                        for i in 0..n {
                            acc += s_mat[i * n + j] * k_row[i];
                        }
                        sk_buf[j] = acc;
                    }

                    for j in 0..n {
                        sk_buf[j] = (v_row[j] - sk_buf[j]) * beta_t;
                    }

                    for i in 0..n {
                        let ki = k_row[i];
                        for j in 0..n {
                            s_mat[i * n + j] += ki * sk_buf[j];
                        }
                    }

                    let out_row = &mut out_f[qkv_step_base + hi * n..qkv_step_base + (hi + 1) * n];
                    for j in 0..n {
                        let mut acc = 0f32;
                        for i in 0..n {
                            acc += s_mat[i * n + j] * q_row[i];
                        }
                        out_row[j] = acc * scale;
                    }
                }
            }
        }

        write_f16(dst, &out_f);
        if state != 0 {
            write_f16(state, &state_f);
        }
    }
}
