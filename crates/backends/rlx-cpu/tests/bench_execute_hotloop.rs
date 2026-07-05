// RLX — perf gate for the execute_thunks hot-loop refactor.
// Run in RELEASE: cargo test -p rlx-cpu --release --test bench_execute_hotloop -- --nocapture
//
// Times execute_thunks over a many-thunk matmul→relu chain. Reports the MIN
// per-iter time over many batches (least noisy). Compare before/after the
// arm-extraction: `#[inline(always)]` should make it perf-neutral.

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Op, Shape};
use std::time::Instant;

fn build_chain(layers: usize, d: usize, seq: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("bench_chain");
    let mut x = g.input("x", Shape::new(&[seq, d], f));
    for l in 0..layers {
        let w = g.input(format!("w{l}"), Shape::new(&[d, d], f));
        let h = g.matmul(x, w, Shape::new(&[seq, d], f));
        x = g.activation(Activation::Relu, h, Shape::new(&[seq, d], f));
    }
    g.set_outputs(vec![x]);
    g
}

#[test]
#[ignore = "perf gate — run explicitly: cargo test -p rlx-cpu --release --test bench_execute_hotloop -- --ignored --nocapture"]
fn bench_execute_thunks_hotloop() {
    let g = build_chain(64, 128, 16);
    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);

    for node in g.nodes() {
        if let Op::Input { .. } = &node.op {
            let n = node.shape.num_elements().unwrap();
            let off = arena.byte_offset(node.id);
            unsafe {
                let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
                for i in 0..n {
                    *p.add(i) = ((i as f32) * 0.017).sin() * 0.1;
                }
            }
        }
    }

    for _ in 0..100 {
        execute_thunks(&sched, arena.raw_buf_mut());
    }
    let m = 300u128;
    let mut best = u128::MAX;
    for _ in 0..30 {
        let t0 = Instant::now();
        for _ in 0..m {
            execute_thunks(&sched, arena.raw_buf_mut());
        }
        best = best.min(t0.elapsed().as_nanos() / m);
    }
    println!(
        "BENCH execute_thunks: {best} ns/iter (min over 30x{m}); thunks={}",
        sched.thunks.len()
    );
}
