#![allow(unsafe_op_in_unsafe_fn)]
use crate::thunk::*;

#[allow(unused_variables)]
pub(crate) fn compile_custom(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::Custom { name, attrs, .. } = &node.op else {
        unreachable!()
    };
    {
        let kernel = crate::op_registry::lookup_cpu_kernel(name).unwrap_or_else(|| {
            panic!(
                "compile_thunks: no CPU kernel registered for \
                         Op::Custom('{name}'). Register one via \
                         rlx_cpu::op_registry::register_cpu_kernel \
                         before compiling on the CPU backend."
            )
        });
        let inputs_v: Vec<(usize, u32, Shape)> = node
            .inputs
            .iter()
            .map(|&in_id| {
                let s = graph.node(in_id).shape.clone();
                let len = s.num_elements().unwrap_or(0) as u32;
                (node_offset(arena, in_id), len, s)
            })
            .collect();
        let out_len = node.shape.num_elements().unwrap_or(0) as u32;
        Thunk::CustomOp {
            kernel,
            inputs: inputs_v,
            output: (node_offset(arena, node.id), out_len, node.shape.clone()),
            attrs: attrs.clone(),
        }
    }
}

/// Lower a core SPD-manifold op to a `Thunk::CustomOp` carrying a
/// directly-instantiated F64 kernel (see `crate::spd_kernels`). Mirrors
/// [`compile_custom`] but takes the kernel by value instead of a
/// registry lookup — these are built-in, not user-registered.
pub(crate) fn compile_spd_custom(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    kernel: std::sync::Arc<dyn crate::op_registry::CpuKernel>,
) -> Thunk {
    let inputs_v: Vec<(usize, u32, Shape)> = node
        .inputs
        .iter()
        .map(|&in_id| {
            let s = graph.node(in_id).shape.clone();
            let len = s
                .num_elements()
                .expect("SPD-manifold op requires statically-shaped inputs")
                as u32;
            (node_offset(arena, in_id), len, s)
        })
        .collect();
    let out_len = node
        .shape
        .num_elements()
        .expect("SPD-manifold op requires a statically-shaped output") as u32;
    Thunk::CustomOp {
        kernel,
        inputs: inputs_v,
        output: (node_offset(arena, node.id), out_len, node.shape.clone()),
        attrs: Vec::new(),
    }
}

#[allow(unused_variables)]
pub(crate) fn compile_custom_fn(
    node: &rlx_ir::Node,
    graph: &Graph,
    arena: &crate::arena::Arena,
    matmul_fold: &std::collections::HashMap<NodeId, (NodeId, bool, NodeId, bool)>,
    rng_shared: &std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    rng: rlx_ir::RngOptions,
) -> Thunk {
    let Op::CustomFn {
        fwd_body,
        num_inputs,
        ..
    } = &node.op
    else {
        unreachable!()
    };
    {
        // Plan + compile the body sub-graph standalone, fill its
        // Constants (mirrors the Op::Scan body lowering), then
        // capture per-input copy specs and the output spec.
        // Body Inputs in NodeId order match the outer node's
        // operand vector by position.
        let body_plan = rlx_opt::memory::plan_memory(fwd_body);
        let body_offsets: HashMap<NodeId, usize> = body_plan
            .assignments
            .iter()
            .map(|(id, slot)| (*id, slot.offset))
            .collect();

        let mut body_input_ids: Vec<NodeId> = fwd_body
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Input { .. }))
            .map(|n| n.id)
            .collect();
        body_input_ids.sort();
        assert_eq!(
            body_input_ids.len(),
            *num_inputs as usize,
            "Op::CustomFn fwd_body has {} Op::Input(s); declared num_inputs={}",
            body_input_ids.len(),
            *num_inputs,
        );

        let mut body_arena = crate::arena::Arena::from_plan(body_plan);
        for n in fwd_body.nodes() {
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
        let body_schedule = compile_thunks_with_rng(fwd_body, &body_arena, rng);

        // Per primal input: (body_input_off, outer_input_off, bytes).
        let inputs_v: Vec<(usize, usize, u32)> = (0..*num_inputs as usize)
            .map(|i| {
                let body_in = body_input_ids[i];
                let body_off = body_offsets[&body_in];
                let outer_in = node.inputs[i];
                let outer_off = node_offset(arena, outer_in);
                let bytes = graph
                    .node(outer_in)
                    .shape
                    .size_bytes()
                    .expect("Op::CustomFn primal input must have static shape");
                (body_off, outer_off, bytes as u32)
            })
            .collect();

        let body_output_id = fwd_body
            .outputs
            .first()
            .copied()
            .expect("Op::CustomFn fwd_body must declare exactly one output");
        let body_output_off = body_offsets[&body_output_id];
        let out_bytes = node
            .shape
            .size_bytes()
            .expect("Op::CustomFn output must have static shape");

        Thunk::CustomFn {
            body: Arc::new(body_schedule),
            body_init: Arc::new(body_init),
            inputs: Arc::new(inputs_v),
            body_output_off,
            outer_output_off: node_offset(arena, node.id),
            out_bytes: out_bytes as u32,
        }
    }
}

// CustomFn dispatch (interpreted path). Mirrors the
// pre-compiled-closure variant elsewhere in this file.
// Patched by rlx-eda.
#[inline(always)]
pub(crate) fn exec_custom_fn(t: &Thunk, base: *mut u8) {
    let Thunk::CustomFn {
        body,
        body_init,
        inputs,
        body_output_off,
        outer_output_off,
        out_bytes,
    } = t
    else {
        unreachable!()
    };
    {
        let mut body_buf: Vec<u8> = (**body_init).clone();
        unsafe {
            for (body_in_off, outer_in_off, n_bytes) in inputs.iter() {
                let src = (base as *const u8).add(*outer_in_off);
                let dst = body_buf.as_mut_ptr().add(*body_in_off);
                std::ptr::copy_nonoverlapping(src, dst, *n_bytes as usize);
            }
        }
        execute_thunks(body, &mut body_buf);
        unsafe {
            let src = body_buf.as_ptr().add(*body_output_off);
            let dst = base.add(*outer_output_off);
            std::ptr::copy_nonoverlapping(src, dst, *out_bytes as usize);
        }
    }
}

#[inline(always)]
pub(crate) fn exec_sgd_momentum(t: &Thunk, base: *mut u8) {
    let Thunk::SgdMomentum {
        param,
        vel,
        grad,
        p_out,
        v_out,
        lr,
        mom,
        len,
    } = t
    else {
        unreachable!()
    };
    {
        // v' = mom·v + g ;  p' = p − lr·v'  — one fused pass per param,
        // replacing the 4 BinaryFull ops the in-graph SGD chain lowers
        // to. Element-wise, so in-place aliasing (p_out==param etc.) is
        // safe: read index i, write index i.
        let len = *len as usize;
        let (lr, mom) = (*lr, *mom);
        unsafe {
            let p = sl(*param, base, len);
            let v = sl(*vel, base, len);
            let g = sl(*grad, base, len);
            let po = sl_mut(*p_out, base, len);
            let vo = sl_mut(*v_out, base, len);
            if fast_conv_enabled() && crate::pool::should_parallelize(len) {
                let poa = po.as_mut_ptr() as usize;
                let voa = vo.as_mut_ptr() as usize;
                crate::pool::par_for(len, crate::pool::chunk_floor(len), &|off, cnt| {
                    for i in off..off + cnt {
                        let vn = mom * v[i] + g[i];
                        *((voa as *mut f32).add(i)) = vn;
                        *((poa as *mut f32).add(i)) = p[i] - lr * vn;
                    }
                });
            } else {
                for i in 0..len {
                    let vn = mom * v[i] + g[i];
                    vo[i] = vn;
                    po[i] = p[i] - lr * vn;
                }
            }
        }
    }
}

// (Thunk::DenseSolveF64 / Thunk::ScanBackward had panic
// stubs here as placeholders during the wire-up; both
// are now reached by the real implementations earlier in
// this same match — the stubs were dead code shadowed by
// the specific-pattern arms above. Removed.)
#[inline(always)]
pub(crate) fn exec_custom_op(t: &Thunk, base: *mut u8) {
    let Thunk::CustomOp {
        kernel,
        inputs,
        output,
        attrs,
    } = t
    else {
        unreachable!()
    };
    {
        let (out_off, out_len, out_shape) = output;
        unsafe {
            dispatch_custom_op(
                &**kernel, inputs, *out_off, *out_len, out_shape, attrs, base,
            );
        }
    }
}

/// Shared dispatch path for `Thunk::CustomOp`. Builds a typed
/// [`CpuTensorRef`] for each input *at that input's declared dtype*
/// (so a sparse-LU op with mixed F64/I32 inputs gets the right
/// typed slices) and a [`CpuTensorMut`] for the output, then calls
/// the kernel's single `execute` method.
pub(crate) unsafe fn dispatch_custom_op(
    kernel: &dyn crate::op_registry::CpuKernel,
    inputs: &[(usize, u32, Shape)],
    out_off: usize,
    out_len: u32,
    out_shape: &Shape,
    attrs: &[u8],
    base: *mut u8,
) {
    use crate::op_registry::{CpuTensorMut, CpuTensorRef};
    use rlx_ir::DType;

    // One arm per `DType` variant — single source of truth for
    // "which dtypes the CPU custom-op dispatcher wires." If a new
    // DType lands in `rlx-ir`, the compiler flags this match as
    // non-exhaustive and the gap gets named at the right place.
    macro_rules! build_in_view {
        ($shape:expr, $off:expr, $n:expr, $variant:ident, $rust_ty:ty) => {
            CpuTensorRef::$variant {
                data: unsafe { sl_typed::<$rust_ty>($off, base, $n) },
                shape: $shape,
            }
        };
    }
    macro_rules! build_out_view {
        ($variant:ident, $rust_ty:ty) => {
            CpuTensorMut::$variant {
                data: unsafe { sl_mut_typed::<$rust_ty>(out_off, base, out_len as usize) },
                shape: out_shape,
            }
        };
    }

    let in_views: Vec<CpuTensorRef<'_>> = inputs
        .iter()
        .map(|(off, len, shape)| {
            let n = *len as usize;
            let off = *off;
            match shape.dtype() {
                DType::F32 => build_in_view!(shape, off, n, F32, f32),
                DType::F64 => build_in_view!(shape, off, n, F64, f64),
                DType::F16 => build_in_view!(shape, off, n, F16, half::f16),
                DType::BF16 => build_in_view!(shape, off, n, BF16, half::bf16),
                DType::I8 => build_in_view!(shape, off, n, I8, i8),
                DType::I16 => build_in_view!(shape, off, n, I16, i16),
                DType::I32 => build_in_view!(shape, off, n, I32, i32),
                DType::I64 => build_in_view!(shape, off, n, I64, i64),
                DType::U8 => build_in_view!(shape, off, n, U8, u8),
                DType::U32 => build_in_view!(shape, off, n, U32, u32),
                DType::Bool => build_in_view!(shape, off, n, Bool, u8),
                // C64 isn't a CpuTensor variant today; the user-registered
                // op_registry path doesn't see complex inputs (those are
                // handled by built-in ops with dedicated kernels).
                DType::C64 => panic!(
                    "Op::Custom kernel input has DType::C64 — built-in \
                 complex ops handle their own kernels; user-registered \
                 ops don't yet see complex tensors"
                ),
            }
        })
        .collect();

    let result = match out_shape.dtype() {
        DType::F32 => kernel.execute(&in_views, build_out_view!(F32, f32), attrs),
        DType::F64 => kernel.execute(&in_views, build_out_view!(F64, f64), attrs),
        DType::F16 => kernel.execute(&in_views, build_out_view!(F16, half::f16), attrs),
        DType::BF16 => kernel.execute(&in_views, build_out_view!(BF16, half::bf16), attrs),
        DType::I8 => kernel.execute(&in_views, build_out_view!(I8, i8), attrs),
        DType::I16 => kernel.execute(&in_views, build_out_view!(I16, i16), attrs),
        DType::I32 => kernel.execute(&in_views, build_out_view!(I32, i32), attrs),
        DType::I64 => kernel.execute(&in_views, build_out_view!(I64, i64), attrs),
        DType::U8 => kernel.execute(&in_views, build_out_view!(U8, u8), attrs),
        DType::U32 => kernel.execute(&in_views, build_out_view!(U32, u32), attrs),
        DType::Bool => kernel.execute(&in_views, build_out_view!(Bool, u8), attrs),
        DType::C64 => panic!("Op::Custom output DType::C64 not supported"),
    };
    if let Err(e) = result {
        panic!("Op::Custom('{}') CPU kernel failed: {e}", kernel.name());
    }
}
