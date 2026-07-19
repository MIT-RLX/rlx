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
//! Host execution for `collective.*` ops on TPU-orchestrated graphs.
//!
//! The collective ops (all-reduce / all-gather / reduce-scatter and the
//! Megatron `f`/`g` operators) are host/transport ops with no device kernel — a
//! TPU can't drive the process group. rlx-tpu runs them as host segments
//! between HLO segments (see [`crate::segment`] / [`crate::orchestrated`]),
//! exactly like the Gaussian-splat host steps in [`crate::splat_host`]. This
//! handler reads the single f32 input from the host env, delegates to the one
//! registered `rlx-cpu` collective kernel via
//! [`rlx_cpu::op_registry::run_f32_custom_op_host`] (so TPU and CPU stay
//! bit-for-bit identical), and writes the f32 result back.
//!
//! The collective kernels work off element counts + `attrs` (the process-group
//! id), not shape dims, so 1-D f32 shapes suffice. Op-name strings mirror
//! `rlx-collectives`, which rlx-tpu cannot depend on (later publish tier); see
//! [`crate::segment::COLLECTIVE_OPS`]. The transport still needs the consumer to
//! have called `rlx_collectives::register()` and registered a process group.
//!
//! PERF: this is a host round-trip — the operand is already resident on the host
//! env between HLO segments, so there is no extra device copy here, but the
//! collective itself runs on the host CPU / network transport rather than as a
//! native XLA cross-replica HLO collective (`all-reduce` / `all-gather` /
//! `reduce-scatter` with a replica-group / SPMD partitioning). Emitting native
//! XLA collectives so the reduction stays on-fabric is the documented perf
//! follow-up.

use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::splat_host::HostTensors;

/// Run a `collective.*` custom op on the host: read the single f32 input from
/// `env`, call the registered CPU collective kernel, and write the result back
/// into `env` under `node`.
pub fn run_collective(graph: &Graph, node: NodeId, env: &mut HostTensors) {
    let n = graph.node(node);
    let Op::Custom {
        name,
        num_inputs,
        attrs,
    } = &n.op
    else {
        panic!("run_collective: expected Op::Custom, got {:?}", n.op);
    };
    assert_eq!(
        *num_inputs, 1,
        "rlx-tpu collective '{name}': expected 1 input, got {num_inputs}"
    );

    let input = env
        .get(&n.inputs[0])
        .unwrap_or_else(|| {
            panic!(
                "rlx-tpu collective '{name}': missing input tensor for node {:?}",
                n.inputs[0]
            )
        })
        .clone();

    // Output element count comes from the node's shape (e.g. all-gather grows,
    // reduce-scatter shrinks); the kernel writes exactly that many f32s.
    let out_len = n.shape.num_elements().unwrap_or_else(|| {
        panic!("rlx-tpu collective '{name}': output shape has dynamic dims — need a static size")
    });
    let mut out = vec![0f32; out_len];

    let in_shape = Shape::new(&[input.len()], DType::F32);
    let out_shape = Shape::new(&[out_len], DType::F32);
    let in_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(input.as_ptr() as *const u8, input.len() * 4) };
    let out_bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };

    rlx_cpu::op_registry::run_f32_custom_op_host(
        name,
        &[(in_bytes, &in_shape)],
        (out_bytes, &out_shape),
        attrs,
    )
    .unwrap_or_else(|e| {
        panic!(
            "rlx-tpu collective '{name}': {e}. The collective ops need the \
             consumer to have called `rlx_collectives::register()` and \
             registered a process group for the group id in this op's attrs."
        )
    });

    env.insert(node, out);
}
