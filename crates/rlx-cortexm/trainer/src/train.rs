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

//! SGD training loop.
//!
//! Compiles the gradient graph once, then iterates: fill inputs →
//! `execute_thunks` → SGD step (with momentum) on each parameter.
//!
//! The "trained model" we hand off to quantization is simply the final
//! `Vec<f32>` for each parameter (read out of the arena at the end).

use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{ThunkSchedule, compile_thunks, execute_thunks};
use rlx_ir::{Graph, NodeId, Op, Philox4x32};

use crate::Args;
use crate::graph::{self, TrainGraph};
use crate::mnist::{Dataset, PIXELS, Split};

/// Floats per parameter, in the same order as `TrainGraph.params`.
pub struct TrainedModel {
    pub conv1_w: Vec<f32>,
    pub conv1_b: Vec<f32>,
    pub conv2_w: Vec<f32>,
    pub conv2_b: Vec<f32>,
    pub fc_w: Vec<f32>,
    pub fc_b: Vec<f32>,
    /// Final test-set accuracy (0..=1) — printed and embedded in
    /// `model_weights.rs` as a comment.
    pub fp32_test_accuracy: f64,
}

pub fn run(dataset: &Dataset, args: &Args) -> Result<TrainedModel, String> {
    let spec = graph::Spec {
        batch: args.batch,
        qat_bits: if args.qat_enabled() {
            Some(args.weight_bits)
        } else {
            None
        },
    };
    let train_graph = graph::build_train_graph(&spec);
    let train_graph = train_graph.legalize_broadcast();

    let plan = rlx_opt::memory::plan_memory(&train_graph.graph);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&train_graph.graph, &arena);
    fill_constants_into_arena(&train_graph.graph, &mut arena);

    // Param init (Kaiming-He for conv/fc weights, zeros for biases).
    let mut rng = Philox4x32::new(args.seed.max(1));
    init_params(&train_graph, &mut arena, &mut rng);

    // Velocity buffers for SGD with momentum, one per param.
    let mut velocity: Vec<Vec<f32>> = train_graph
        .params
        .iter()
        .map(|p| vec![0f32; p.num_elements()])
        .collect();

    // Always seed `d_output = 1.0` (we differentiate the scalar loss
    // directly, no upstream chain).
    write_arena(&mut arena, train_graph.d_output, &[1.0]);

    let total_train = if args.train_limit == 0 {
        dataset.train.len()
    } else {
        args.train_limit.min(dataset.train.len())
    };
    let batches_per_epoch = total_train / args.batch;

    for epoch in 0..args.epochs {
        let mut order: Vec<usize> = (0..total_train).collect();
        shuffle(&mut order, &mut rng);

        let mut epoch_loss = 0.0f64;
        let mut t0 = std::time::Instant::now();

        for batch_idx in 0..batches_per_epoch {
            let indices = &order[batch_idx * args.batch..(batch_idx + 1) * args.batch];
            fill_batch(&mut arena, &train_graph, &dataset.train, indices);

            execute_thunks(&sched, arena.raw_buf_mut());

            // Loss is a scalar (one f32).
            let loss = read_arena(&arena, train_graph.loss, 1)[0] as f64;
            epoch_loss += loss;

            // SGD step per param.
            for (slot, vel) in train_graph.params.iter().zip(velocity.iter_mut()) {
                let n = slot.num_elements();
                let mut p = read_arena(&arena, slot.param, n);
                let g = read_arena(&arena, slot.grad, n);
                for ((pi, vi), gi) in p.iter_mut().zip(vel.iter_mut()).zip(g.iter()) {
                    *vi = args.momentum * *vi + *gi;
                    *pi -= args.learning_rate * *vi;
                }
                write_arena(&mut arena, slot.param, &p);
            }
        }

        let mean_loss = epoch_loss / batches_per_epoch as f64;
        let elapsed = t0.elapsed().as_secs_f64();
        eprint!(
            "epoch {}/{}: train loss = {mean_loss:.4} ({elapsed:.1}s)",
            epoch + 1,
            args.epochs
        );
        let _ = std::io::Write::flush(&mut std::io::stderr());
        t0 = std::time::Instant::now();

        // Eval on the test set (uses the same batched graph; just reads
        // `logits` and argmaxes).
        let acc = evaluate(
            &sched,
            &mut arena,
            &train_graph,
            &dataset.test,
            args.eval_limit,
            args.batch,
        );
        eprintln!("  test acc = {acc:.4} ({:.1}s)", t0.elapsed().as_secs_f64());
    }

    // Final test accuracy (reuse the eval pass from the last epoch
    // result, but recompute against the full requested limit).
    let final_acc = evaluate(
        &sched,
        &mut arena,
        &train_graph,
        &dataset.test,
        args.eval_limit,
        args.batch,
    );

    // Read trained params out of the arena.
    let conv1_w = read_arena(
        &arena,
        train_graph.params[0].param,
        train_graph.params[0].num_elements(),
    );
    let conv1_b = read_arena(
        &arena,
        train_graph.params[1].param,
        train_graph.params[1].num_elements(),
    );
    let conv2_w = read_arena(
        &arena,
        train_graph.params[2].param,
        train_graph.params[2].num_elements(),
    );
    let conv2_b = read_arena(
        &arena,
        train_graph.params[3].param,
        train_graph.params[3].num_elements(),
    );
    let fc_w = read_arena(
        &arena,
        train_graph.params[4].param,
        train_graph.params[4].num_elements(),
    );
    let fc_b = read_arena(
        &arena,
        train_graph.params[5].param,
        train_graph.params[5].num_elements(),
    );

    Ok(TrainedModel {
        conv1_w,
        conv1_b,
        conv2_w,
        conv2_b,
        fc_w,
        fc_b,
        fp32_test_accuracy: final_acc,
    })
}

/// Run the (gradient) graph against the test set in batches and report
/// classification accuracy. The gradient computation is wasted work
/// here — keeping it avoids maintaining a second compiled graph; the
/// FC bottleneck is small enough that the cost is negligible.
pub fn evaluate(
    sched: &ThunkSchedule,
    arena: &mut Arena,
    train_graph: &TrainGraph,
    test: &Split,
    limit: usize,
    batch: usize,
) -> f64 {
    let total = if limit == 0 {
        test.len()
    } else {
        limit.min(test.len())
    };
    let n_batches = total / batch;
    let mut correct = 0usize;
    for b in 0..n_batches {
        let indices: Vec<usize> = (b * batch..(b + 1) * batch).collect();
        fill_batch(arena, train_graph, test, &indices);
        execute_thunks(sched, arena.raw_buf_mut());
        let logits = read_arena(arena, train_graph.logits, batch * 10);
        for (i, &idx) in indices.iter().enumerate() {
            let row = &logits[i * 10..(i + 1) * 10];
            let pred = argmax_f32(row);
            let label = test.labels[idx] as usize;
            if pred == label {
                correct += 1;
            }
        }
    }
    correct as f64 / (n_batches * batch) as f64
}

fn fill_batch(arena: &mut Arena, train_graph: &TrainGraph, split: &Split, indices: &[usize]) {
    // Pack images contiguously into the arena slot for `x` (NCHW with
    // C=1 means the "flat 28×28" layout from MNIST is already correct).
    let img_off = arena.byte_offset(train_graph.input);
    let buf = arena.raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(img_off) as *mut f32;
        for (i, &idx) in indices.iter().enumerate() {
            let src = split.image(idx);
            for j in 0..PIXELS {
                *p.add(i * PIXELS + j) = src[j];
            }
        }
    }
    // Labels.
    let label_off = arena.byte_offset(train_graph.labels);
    let buf = arena.raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(label_off) as *mut f32;
        for (i, &idx) in indices.iter().enumerate() {
            *p.add(i) = split.labels[idx];
        }
    }
}

// ─────────────────────────── helpers ────────────────────────────

fn init_params(train_graph: &TrainGraph, arena: &mut Arena, rng: &mut Philox4x32) {
    // Kaiming-He: weights ~ N(0, sqrt(2 / fan_in)). Biases = 0.
    let fan_ins = [
        3 * 3,     // conv1: c_in × kH × kW
        0,         // conv1_b (bias)
        8 * 3 * 3, // conv2
        0,         // conv2_b
        400,       // fc
        0,         // fc_b
    ];
    for (slot, &fan_in) in train_graph.params.iter().zip(fan_ins.iter()) {
        let n = slot.num_elements();
        let data = if fan_in > 0 {
            let mut v = vec![0f32; n];
            rng.fill_normal(&mut v);
            let scale = (2.0 / fan_in as f32).sqrt();
            for x in v.iter_mut() {
                *x *= scale;
            }
            v
        } else {
            vec![0f32; n]
        };
        write_arena(arena, slot.param, &data);
    }
}

fn shuffle(buf: &mut [usize], rng: &mut Philox4x32) {
    // Fisher-Yates.
    let n = buf.len();
    for i in (1..n).rev() {
        let j = (rng.next_f32() * (i + 1) as f32) as usize;
        let j = j.min(i);
        buf.swap(i, j);
    }
}

fn argmax_f32(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

pub fn read_arena(arena: &Arena, id: NodeId, len: usize) -> Vec<f32> {
    let off = arena.byte_offset(id);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..len).map(|i| *p.add(i)).collect()
    }
}

pub fn write_arena(arena: &mut Arena, id: NodeId, data: &[f32]) {
    let off = arena.byte_offset(id);
    let buf = arena.raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(off) as *mut f32;
        for (i, &v) in data.iter().enumerate() {
            *p.add(i) = v;
        }
    }
}

pub fn fill_constants_into_arena(graph: &Graph, arena: &mut Arena) {
    for node in graph.nodes() {
        if let Op::Constant { data } = &node.op
            && arena.has_buffer(node.id)
            && !data.is_empty()
        {
            let buf = arena.slice_mut(node.id);
            let n_floats = data.len() / 4;
            let n = buf.len().min(n_floats);
            for i in 0..n {
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
