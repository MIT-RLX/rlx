// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2×2 stride-2 max pooling, which takes its own two-rows-at-a-time path: against the
//! definition, on odd sizes (the last row / column is dropped, floor mode), several channels
//! and a batch, with negative values (the max starts at the first element, not at zero).

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, Op, Shape};

fn max_pool_2x2(data: &[f32], dims: [usize; 4]) -> Vec<f32> {
    let [n, c, h, w] = dims;
    let out = [n, c, h / 2, w / 2];
    let mut g = Graph::new("pool");
    let x = g.input("x", Shape::new(&dims, DType::F32));
    let y = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![x],
        Shape::new(&out, DType::F32),
    );
    g.set_outputs(vec![y]);
    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    arena.slice_mut(x)[..data.len()].copy_from_slice(data);
    execute_thunks(&sched, arena.raw_buf_mut());
    arena.slice(y)[..out.iter().product::<usize>()].to_vec()
}

#[test]
fn matches_the_definition() {
    for dims in [[1, 1, 2, 2], [2, 3, 7, 9], [1, 16, 96, 160], [1, 4, 5, 2]] {
        let [n, c, h, w] = dims;
        let data: Vec<f32> = (0..n * c * h * w)
            .map(|i| ((i * 7919 % 211) as f32 - 105.0) / 10.0)
            .collect();
        let got = max_pool_2x2(&data, dims);
        let (ho, wo) = (h / 2, w / 2);
        for p in 0..n * c {
            for y in 0..ho {
                for x in 0..wo {
                    let at = |dy: usize, dx: usize| data[(p * h + 2 * y + dy) * w + 2 * x + dx];
                    let want = at(0, 0).max(at(0, 1)).max(at(1, 0)).max(at(1, 1));
                    assert_eq!(
                        got[(p * ho + y) * wo + x],
                        want,
                        "{dims:?} plane {p} at ({y}, {x})"
                    );
                }
            }
        }
    }
}
