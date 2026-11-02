// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Kernels indexed past 65535 workgroups.** `dispatch_dims` spills a large
//! dispatch into y; a kernel reading `gid.x` alone ran the first row of
//! workgroups again and left the rest unwritten. `maxpool3d_backward` did
//! (FastSurfer's full-size unpool, a 2-D net batched as a volume's depth,
//! came back wrong from a batch of 4). A max-pool backward over 21 M
//! elements, on the GPU and on the CPU, must agree exactly: each value is
//! routed, not summed.

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

fn cpu_run(g: &Graph, inputs: &[(&str, Vec<f32>)]) -> Vec<f32> {
    let plan = rlx_opt::memory::plan_memory(g);
    let mut arena = rlx_cpu::arena::Arena::from_plan(plan);
    let sched = rlx_cpu::thunk::compile_thunks(g, &arena);
    for node in g.nodes() {
        if let Op::Input { name } = &node.op {
            let data = &inputs.iter().find(|(n, _)| n == name).unwrap().1;
            let off = arena.byte_offset(node.id);
            unsafe {
                let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
                std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len());
            }
        }
    }
    rlx_cpu::thunk::execute_thunks(&sched, arena.raw_buf_mut());
    let out = g.outputs[0];
    let n = g.shape(out).num_elements().unwrap();
    let off = arena.byte_offset(out);
    unsafe {
        std::slice::from_raw_parts(arena.raw_buf_mut().as_ptr().add(off) as *const f32, n).to_vec()
    }
}

#[test]
fn maxpool3d_backward_past_65535_workgroups() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    // 5 × 2 × 1024 × 1024 = 10.5 M inputs → 41 k workgroups of 256: under;
    // 5 × 4 × 1024 × 1024 = 21 M → 82 k: past the line
    let (c, d, h, w) = (5, 4, 1024, 1024);
    let sh = |s: &[usize]| Shape::new(s, DType::F32);
    let mut g = Graph::new("pool_bwd");
    let x = g.input("x", sh(&[1, c, d, h, w]));
    let dy = g.input("dy", sh(&[1, c, d, h / 2, w / 2]));
    let dx = g.maxpool3d_backward(x, dy, vec![1, 2, 2], vec![1, 2, 2], vec![0; 3]);
    g.set_outputs(vec![dx]);
    // distinct values (no ties), dy distinct too
    let n = c * d * h * w;
    let xs: Vec<f32> = (0..n)
        .map(|i| ((i as u64 * 2654435761) % 1_000_003) as f32)
        .collect();
    let dys: Vec<f32> = (0..n / 4).map(|i| 1.0 + (i % 977) as f32).collect();
    let ins = [("x", xs), ("dy", dys)];
    let cpu = cpu_run(&g, &ins);
    let refs: Vec<(&str, &[f32])> = ins.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let gpu = WgpuExecutable::compile(g).run(&refs).remove(0);
    let wrong = gpu.iter().zip(&cpu).filter(|(a, b)| a != b).count();
    assert_eq!(wrong, 0, "{wrong} of {n} values differ");
}
