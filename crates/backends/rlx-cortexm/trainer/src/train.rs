// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

fn env_flag(key: &str) -> bool {
    matches!(
        std::env::var(key).ok().as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}
/// `RLX_FAST_CONV` — im2col+BLAS forward conv (mirrors `rlx_cpu::fast_conv`).
/// On by default; only an explicit `0`/`off`/`false`/`no` selects the reference
/// scalar path (the unfused `rlx` bench bar).
fn fast_conv_label() -> bool {
    !matches!(
        std::env::var("RLX_FAST_CONV").ok().as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}
/// `RLX_GRAPH_FUSED` — fold the SGD+momentum update into the graph and run the
/// whole fwd+bwd+update step as one region-fused compiled schedule.
fn graphfused_enabled() -> bool {
    env_flag("RLX_GRAPH_FUSED")
}

pub fn run(dataset: &Dataset, args: &Args) -> Result<TrainedModel, String> {
    if graphfused_enabled() {
        return run_graphfused(dataset, args);
    }
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
    init_params(&train_graph.params, &mut arena, &mut rng);

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

    // Per-step latencies (ms) and cumulative training wall-time (excluding
    // per-epoch eval), for the benchmark row emitted at the end. A "step" is
    // forward + backward (`execute_thunks`) + the SGD update — the same unit
    // the JAX/PyTorch/candle runners time.
    let mut steps: Vec<f64> = Vec::new();
    let mut train_s = 0.0f64;

    let mut order: Vec<usize> = (0..total_train).collect();
    for epoch in 0..args.epochs {
        for (i, v) in order.iter_mut().enumerate() {
            *v = i;
        }
        shuffle(&mut order, &mut rng);

        let mut epoch_loss = 0.0f64;
        let mut t0 = std::time::Instant::now();

        for batch_idx in 0..batches_per_epoch {
            let indices = &order[batch_idx * args.batch..(batch_idx + 1) * args.batch];
            fill_batch(
                &mut arena,
                train_graph.input,
                train_graph.labels,
                &dataset.train,
                indices,
            );

            let step_t = std::time::Instant::now();
            execute_thunks(&sched, arena.raw_buf_mut());

            // Loss is a scalar (one f32).
            let loss = read_scalar(&arena, train_graph.loss) as f64;
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
            steps.push(step_t.elapsed().as_secs_f64() * 1e3);
        }

        let mean_loss = epoch_loss / batches_per_epoch as f64;
        let elapsed = t0.elapsed().as_secs_f64();
        train_s += elapsed;
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

    let label = if fast_conv_label() {
        "rlx-fused"
    } else {
        "rlx"
    };
    emit_bench_row(label, &steps, train_s, args.epochs, args.batch, final_acc);

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

/// Fully-fused training step (`RLX_GRAPH_FUSED=1`). Folds the SGD+momentum
/// update into the IR graph, runs the entire forward+backward+update through
/// the region-fusion pipeline (the `rlx_opt::fusion_pipeline::Fuse` DSL), and
/// executes it as one compiled schedule — the JAX-like "whole step is one
/// program" mode. RLX has no in-place param op, so `p'`/`v'` are graph outputs
/// the host copies back into the persistent `p`/`v` arena slots each step (a
/// few KB; negligible next to the conv kernels). Pair with `RLX_FAST_CONV=1`.
fn run_graphfused(dataset: &Dataset, args: &Args) -> Result<TrainedModel, String> {
    use rlx_ir::op::BinaryOp;
    use rlx_ir::{DType, Shape};
    use rlx_opt::fusion_pipeline::{Fuse, FusionTarget};

    let spec = graph::Spec {
        batch: args.batch,
        qat_bits: if args.qat_enabled() {
            Some(args.weight_bits)
        } else {
            None
        },
    };
    let mlp = std::env::var("RLX_ARCH").as_deref() == Ok("mlp");
    // The contiguous batch-shuffle MLP path converges better at a slightly
    // lower LR while preserving higher throughput. Keep user overrides
    // untouched; only auto-tune the historical default 0.05.
    let effective_lr = if mlp && (args.learning_rate - 0.05).abs() < f32::EPSILON {
        0.04
    } else {
        args.learning_rate
    };
    let tg = if mlp {
        graph::build_train_graph_mlp(&spec)
    } else {
        graph::build_train_graph(&spec)
    }
    .legalize_broadcast();

    // Param metadata in canonical order, captured before fusion renumbers
    // NodeIds — handles are re-resolved by name afterwards.
    let pmeta: Vec<(&'static str, Vec<usize>)> = tg
        .params
        .iter()
        .map(|p| (p.name, p.shape.clone()))
        .collect();

    // ── Append the in-graph SGD+momentum update ───────────────────
    // Per param p (grad g, fresh velocity input v):
    //   v' = momentum*v + g ;  p' = p - lr*v'
    // Scalars are full-size constants so every op is same-shape (no broadcast
    // to legalize). p'/v' become outputs the host writes back to p/v.
    let mut g = tg.graph;
    let mut outputs = vec![tg.loss, tg.logits];
    let const_full = |g: &mut Graph, val: f32, dims: &[usize]| -> NodeId {
        let n: usize = dims.iter().product();
        let mut data = Vec::with_capacity(n * 4);
        let bytes = val.to_le_bytes();
        for _ in 0..n {
            data.extend_from_slice(&bytes);
        }
        g.add_node(Op::Constant { data }, vec![], Shape::new(dims, DType::F32))
    };
    for slot in &tg.params {
        let dims = slot.shape.clone();
        let shp = Shape::new(&dims, DType::F32);
        let mom_c = const_full(&mut g, args.momentum, &dims);
        let lr_c = const_full(&mut g, effective_lr, &dims);
        let vel = g.input(format!("vel_{}", slot.name), shp.clone());
        let v_scaled = g.binary(BinaryOp::Mul, vel, mom_c, shp.clone());
        let v_new = g.binary(BinaryOp::Add, v_scaled, slot.grad, shp.clone());
        let lr_v = g.binary(BinaryOp::Mul, v_new, lr_c, shp.clone());
        let p_new = g.binary(BinaryOp::Sub, slot.param, lr_v, shp.clone());
        outputs.push(p_new);
        outputs.push(v_new);
    }
    g.set_outputs(outputs);

    // ── Region-fuse the whole step (the new fusion DSL) ───────────
    // With RLX_REGIONS=1, KEEP element-wise regions fused and let the CPU
    // region interpreter run each chain in one pass (XLA-style auto-fusion),
    // instead of unfusing back to per-op thunks. Disable prologue/FK/batch
    // fusion so only plain scalar-chain regions form (what the interpreter
    // handles).
    let opts = if env_flag("RLX_REGIONS") {
        rlx_opt::fusion_pipeline::FusionOptions {
            unfuse_elementwise_regions: false,
            fuse_region_prologue: false,
            fuse_batch_preprocess: false,
            fk_fusion: false,
            ..rlx_opt::fusion_pipeline::FusionOptions::for_cpu()
        }
    } else {
        rlx_opt::fusion_pipeline::FusionOptions::for_cpu()
    };
    let (g, report) = Fuse::new(FusionTarget::Cpu)
        .options(opts)
        .run_with_report(g);
    eprintln!("graphfused fusion: {}", report.summary_line());

    // Re-resolve handles in the renumbered graph: inputs/params by name,
    // loss/logits/p'/v' by output position.
    let input = g.input_id("x").ok_or("lost input x after fusion")?;
    let labels = g
        .input_id("labels")
        .ok_or("lost input labels after fusion")?;
    let d_output = g
        .node_id_by_name("d_output")
        .ok_or("lost d_output after fusion")?;
    let outs = g.outputs.clone();
    let loss_id = outs[0];
    let logits_id = outs[1];

    let mut params: Vec<graph::ParamSlot> = Vec::new();
    // (param, vel, p_new, v_new, n_elems) for the per-step write-back.
    let mut updates: Vec<(NodeId, NodeId, NodeId, NodeId, usize)> = Vec::new();
    for (i, (name, dims)) in pmeta.iter().enumerate() {
        let n: usize = dims.iter().product();
        let param = g
            .param_id(name)
            .ok_or_else(|| format!("lost param {name} after fusion"))?;
        let vel = g
            .input_id(&format!("vel_{name}"))
            .ok_or_else(|| format!("lost vel_{name} after fusion"))?;
        params.push(graph::ParamSlot {
            name,
            shape: dims.clone(),
            param,
            grad: param, // unused on this path
        });
        updates.push((param, vel, outs[2 + 2 * i], outs[2 + 2 * i + 1], n));
    }

    // ── Compile + arena ──────────────────────────────────────────
    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    fill_constants_into_arena(&g, &mut arena);

    let mut rng = Philox4x32::new(args.seed.max(1));
    init_params(&params, &mut arena, &mut rng);
    for (_, vel, _, _, n) in &updates {
        write_arena(&mut arena, *vel, &vec![0f32; *n]); // velocities start at 0
    }
    write_arena(&mut arena, d_output, &[1.0]);

    // Precompute byte offsets for the per-step write-back so the optimizer
    // step is allocation-free (no read_arena/write_arena Vec churn in the hot
    // loop): p'/v' are memcpy'd in place over p/v each step.
    let upd_off: Vec<(usize, usize, usize, usize, usize)> = updates
        .iter()
        .map(|&(param, vel, p_new, v_new, n)| {
            (
                arena.byte_offset(param),
                arena.byte_offset(vel),
                arena.byte_offset(p_new),
                arena.byte_offset(v_new),
                n,
            )
        })
        .collect();

    let total_train = if args.train_limit == 0 {
        dataset.train.len()
    } else {
        args.train_limit.min(dataset.train.len())
    };
    let batches_per_epoch = total_train / args.batch;

    let log_epoch_loss = std::env::var_os("RLX_LOG_EPOCH_LOSS").is_some();
    let mut steps: Vec<f64> = Vec::new();
    let mut train_s = 0.0f64;
    let mut order: Vec<usize> = (0..total_train).collect();
    let mut batch_order: Vec<usize> = (0..batches_per_epoch).collect();
    for epoch in 0..args.epochs {
        if mlp {
            for (i, v) in batch_order.iter_mut().enumerate() {
                *v = i;
            }
            if std::env::var_os("RLX_NO_SHUFFLE").is_none() {
                // MLP benchmark: shuffle at batch granularity so each batch is
                // still contiguous in memory (one memcpy), while SGD sees
                // stochastic batch order each epoch.
                shuffle(&mut batch_order, &mut rng);
            }
        } else {
            for (i, v) in order.iter_mut().enumerate() {
                *v = i;
            }
            if std::env::var_os("RLX_NO_SHUFFLE").is_none() {
                shuffle(&mut order, &mut rng);
            }
        }
        let mut epoch_loss = 0.0f64;
        let t0 = std::time::Instant::now();
        for batch_idx in 0..batches_per_epoch {
            if mlp {
                let bidx = batch_order[batch_idx];
                fill_batch_contiguous(
                    &mut arena,
                    input,
                    labels,
                    &dataset.train,
                    bidx * args.batch,
                    args.batch,
                );
            } else {
                let indices = &order[batch_idx * args.batch..(batch_idx + 1) * args.batch];
                fill_batch(&mut arena, input, labels, &dataset.train, indices);
            }
            let step_t = std::time::Instant::now();
            execute_thunks(&sched, arena.raw_buf_mut());
            // The whole optimizer step ran in-graph; persist p'/v' in place via
            // an allocation-free memcpy within the arena.
            {
                let base = arena.raw_buf_mut().as_mut_ptr();
                for &(pb, vb, pnb, vnb, n) in &upd_off {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            base.add(pnb) as *const f32,
                            base.add(pb) as *mut f32,
                            n,
                        );
                        std::ptr::copy_nonoverlapping(
                            base.add(vnb) as *const f32,
                            base.add(vb) as *mut f32,
                            n,
                        );
                    }
                }
            }
            steps.push(step_t.elapsed().as_secs_f64() * 1e3);
            if log_epoch_loss {
                epoch_loss += read_scalar(&arena, loss_id) as f64;
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        train_s += elapsed;
        if log_epoch_loss {
            eprintln!(
                "epoch {}/{}: train loss = {:.4} ({elapsed:.1}s)",
                epoch + 1,
                args.epochs,
                epoch_loss / batches_per_epoch as f64,
            );
        } else {
            eprintln!("epoch {}/{}: {elapsed:.1}s", epoch + 1, args.epochs);
        }
    }

    // Eval: run the same schedule but skip the write-back — the forward logits
    // don't depend on the update nodes, so params stay put.
    let acc = eval_graphfused(
        &sched,
        &mut arena,
        input,
        labels,
        logits_id,
        &dataset.test,
        args.eval_limit,
        args.batch,
    );
    let label = if mlp {
        "rlx-graphfused-mlp"
    } else {
        "rlx-graphfused"
    };
    emit_bench_row(label, &steps, train_s, args.epochs, args.batch, acc);
    if std::env::var_os("RLX_PROFILE_THUNKS").is_some() {
        rlx_cpu::thunk::dump_thunk_profile();
    }
    if mlp {
        // Benchmark-only baseline; skip the CNN-specific TrainedModel + quantize/emit.
        std::process::exit(0);
    }

    let rd = |id, n| read_arena(&arena, id, n);
    Ok(TrainedModel {
        conv1_w: rd(params[0].param, params[0].num_elements()),
        conv1_b: rd(params[1].param, params[1].num_elements()),
        conv2_w: rd(params[2].param, params[2].num_elements()),
        conv2_b: rd(params[3].param, params[3].num_elements()),
        fc_w: rd(params[4].param, params[4].num_elements()),
        fc_b: rd(params[5].param, params[5].num_elements()),
        fp32_test_accuracy: acc,
    })
}

/// Accuracy pass for the graph-fused model: runs the fused schedule (which also
/// recomputes the unused update outputs) and reads logits, never writing back.
#[allow(clippy::too_many_arguments)]
fn eval_graphfused(
    sched: &ThunkSchedule,
    arena: &mut Arena,
    input: NodeId,
    labels: NodeId,
    logits: NodeId,
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
        fill_batch(arena, input, labels, test, &indices);
        execute_thunks(sched, arena.raw_buf_mut());
        let logits_v = read_arena(arena, logits, batch * 10);
        for (i, &idx) in indices.iter().enumerate() {
            if argmax_f32(&logits_v[i * 10..(i + 1) * 10]) == test.labels[idx] as usize {
                correct += 1;
            }
        }
    }
    correct as f64 / (n_batches * batch) as f64
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
        fill_batch(arena, train_graph.input, train_graph.labels, test, &indices);
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

fn fill_batch(arena: &mut Arena, input: NodeId, labels: NodeId, split: &Split, indices: &[usize]) {
    // Data IO: gather the shuffled mini-batch into the arena's `x` slot. Each
    // image is a contiguous `PIXELS` block, so copy it as one memcpy
    // (`copy_nonoverlapping`) rather than element-by-element; the per-image
    // copies are disjoint, so for a full batch they fan out over the pool.
    let img_off = arena.byte_offset(input);
    let label_off = arena.byte_offset(labels);
    let buf = arena.raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(img_off) as *mut f32;
        for (i, &idx) in indices.iter().enumerate() {
            // Each image is a contiguous PIXELS block — one memcpy, not a
            // PIXELS-long scalar copy.
            let src = split.image(idx);
            std::ptr::copy_nonoverlapping(src.as_ptr(), p.add(i * PIXELS), PIXELS);
        }
        let lp = buf.as_mut_ptr().add(label_off) as *mut f32;
        for (i, &idx) in indices.iter().enumerate() {
            *lp.add(i) = split.labels[idx];
        }
    }
}

fn fill_batch_contiguous(
    arena: &mut Arena,
    input: NodeId,
    labels: NodeId,
    split: &Split,
    start: usize,
    batch: usize,
) {
    let img_off = arena.byte_offset(input);
    let label_off = arena.byte_offset(labels);
    let buf = arena.raw_buf_mut();
    unsafe {
        // One contiguous memcpy for images.
        let dst_img = buf.as_mut_ptr().add(img_off) as *mut f32;
        let src_img = split.images.as_ptr().add(start * PIXELS);
        std::ptr::copy_nonoverlapping(src_img, dst_img, batch * PIXELS);

        // One contiguous memcpy for labels.
        let dst_lbl = buf.as_mut_ptr().add(label_off) as *mut f32;
        let src_lbl = split.labels.as_ptr().add(start);
        std::ptr::copy_nonoverlapping(src_lbl, dst_lbl, batch);
    }
}

// ─────────────────────────── helpers ────────────────────────────

fn init_params(params: &[graph::ParamSlot], arena: &mut Arena, rng: &mut Philox4x32) {
    // Kaiming-He from each tensor's shape, so the same init works for the CNN
    // (conv [out,in,kH,kW], fc [in,out]) and the MLP ([in,out]); biases → 0.
    for slot in params.iter() {
        let fan_in = match slot.shape.len() {
            4 => slot.shape[1] * slot.shape[2] * slot.shape[3], // conv
            2 => slot.shape[0],                                 // linear [in,out]
            _ => 0,                                             // bias
        };
        let n = slot.num_elements();
        let data = if fan_in > 0 {
            // Uniform He: U(-1,1)·√(2/fan_in). Matches the rlx-mnist-device
            // bench init; its smaller variance (√3 below Gaussian He) trains
            // more stably at LR=0.05 over 2 epochs (fp32 acc 0.971 → ~0.982).
            let scale = (2.0 / fan_in as f32).sqrt();
            (0..n)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
                .collect::<Vec<f32>>()
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

#[inline]
fn read_scalar(arena: &Arena, id: NodeId) -> f32 {
    let off = arena.byte_offset(id);
    unsafe { *(arena.raw_buf().as_ptr().add(off) as *const f32) }
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

/// Emit a benchmark row in the shared `mnist_training.csv` schema
/// (`framework,device,test_acc,train_s,epoch_s,step_p50_ms,first_step_ms,
/// imgs_per_s`). The framework label reflects which forward-conv kernel ran:
/// `rlx-fused` when im2col+BLAS is active (default / `RLX_FAST_CONV=1`), else
/// `rlx` for the reference scalar kernel (`RLX_FAST_CONV=0`) — so the two
/// appear as separate bars.
///
/// Prints a `RLX_BENCH,<row>` line to stdout (easy to grep) and, when
/// `RLX_BENCH_CSV` is set, appends the bare row to that file.
fn emit_bench_row(label: &str, steps: &[f64], train_s: f64, epochs: usize, batch: usize, acc: f64) {
    if steps.is_empty() || train_s <= 0.0 || epochs == 0 {
        return;
    }
    let first = steps[0];
    // Median of the steady-state steps (drop the first, which pays any
    // one-time warmup), matching the other runners' `step_p50`.
    let mut steady: Vec<f64> = if steps.len() > 1 {
        steps[1..].to_vec()
    } else {
        steps.to_vec()
    };
    steady.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = steady[steady.len() / 2];

    // imgs/s = total images processed / training wall-time (each step is one
    // mini-batch of `batch` images), matching the peer runners' throughput.
    let imgs_per_s = (steps.len() as f64 * batch as f64) / train_s;
    let epoch_s = train_s / epochs as f64;

    let row = format!(
        "{label},cpu,{acc:.4},{train_s:.1},{epoch_s:.1},{p50:.1},{first:.0},{imgs_per_s:.0}"
    );
    println!("RLX_BENCH,{row}");
    if let Ok(path) = std::env::var("RLX_BENCH_CSV") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = writeln!(f, "{row}");
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
