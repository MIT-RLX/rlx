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

//! Sampling micro-bench (plan #52).
//!
//! Builds a tiny graph: input logits → fused Sample. Times the
//! sample step itself across vocab sizes. Run with
//! `cargo bench --bench sample`.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::{DType, Graph, Shape};

fn build_graph(batch: usize, vocab: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("sample_bench");
    let logits = g.input("logits", Shape::new(&[batch, vocab], f));
    // Modest top-p so the kernel exercises filter + softmax + sample.
    let s = g.sample(logits, 64, 0.9, 0.7, 12345, Shape::new(&[batch], f));
    g.set_outputs(vec![s]);
    g
}

fn bench_sample(c: &mut Criterion) {
    // Vocab size covers small (BERT 30k), medium (LLaMA 32k),
    // big (Gemma 256k).
    let vocabs = [
        ("v_30k", 30_000usize),
        ("v_32k", 32_000),
        ("v_128k", 128_000),
    ];
    let batch = 1;

    let mut group = c.benchmark_group("sample");
    for (label, vocab) in vocabs {
        let g = build_graph(batch, vocab);
        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        // Seed logits once with a peaked distribution.
        let logits_id = g
            .nodes()
            .iter()
            .find(|n| matches!(n.op, rlx_ir::Op::Input { .. }))
            .map(|n| n.id)
            .unwrap();
        let logits_off = arena.byte_offset(logits_id);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(logits_off) as *mut f32;
            let mut rng = rlx_ir::Philox4x32::new(7);
            for i in 0..(batch * vocab) {
                *p.add(i) = rng.normal();
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

criterion_group!(benches, bench_sample);
criterion_main!(benches);
