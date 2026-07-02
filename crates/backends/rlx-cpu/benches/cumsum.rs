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

//! Cumsum micro-bench (plan #52).
//!
//! Times the cumsum primitive at sizes typical of sampling and
//! ragged-tensor offsets. Run with `cargo bench --bench cumsum`.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::{DType, Graph, Shape};

fn build_cumsum_graph(rows: usize, cols: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("cumsum_bench");
    let x = g.input("x", Shape::new(&[rows, cols], f));
    let cs = g.cumsum(x, -1, false, Shape::new(&[rows, cols], f));
    g.set_outputs(vec![cs]);
    g
}

fn bench_cumsum(c: &mut Criterion) {
    // Rows × cols — covers single-vocab top-p (1×30k) and
    // batch-of-distributions cases.
    let shapes = [
        ("1x30k", (1usize, 30_000)),
        ("8x32k", (8, 32_000)),
        ("32x4k", (32, 4096)),
    ];

    let mut group = c.benchmark_group("cumsum");
    for (label, (rows, cols)) in shapes {
        let g = build_cumsum_graph(rows, cols);
        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        let in_id = g
            .nodes()
            .iter()
            .find(|n| matches!(n.op, rlx_ir::Op::Input { .. }))
            .map(|n| n.id)
            .unwrap();
        let in_off = arena.byte_offset(in_id);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(in_off) as *mut f32;
            let mut rng = rlx_ir::Philox4x32::new(11);
            for i in 0..(rows * cols) {
                *p.add(i) = rng.next_f32();
            }
        }

        group.bench_function(label, |bencher| {
            bencher.iter(|| {
                execute_thunks(black_box(&sched), arena.raw_buf_mut());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_cumsum);
criterion_main!(benches);
