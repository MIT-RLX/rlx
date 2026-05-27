// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// End-to-end: TinyConv-MNIST forward + new autodiff pipeline on CPU.

mod tinyconv_graph;

use std::collections::HashMap;
use std::path::Path;

use rlx_autodiff::{grad_with_loss, prepare_graph_for_ad, prepare_mir_for_ad};
use rlx_compile::legalize_broadcast::run_with_remap;
use rlx_cpu::arena::Arena;
use rlx_cpu::thunk::{compile_thunks, execute_thunks};
use rlx_ir::{Graph, NodeId, Op, Philox4x32};
use rlx_opt::memory::plan_memory;

use tinyconv_graph::build_tinyconv_forward;

const BATCH: usize = 8;

struct TrainBundle {
    graph: Graph,
    input: NodeId,
    labels: NodeId,
    d_output: NodeId,
    loss: NodeId,
    grad_slots: Vec<NodeId>,
}

struct ExecResult {
    loss: f32,
    grads: Vec<Vec<f32>>,
}

fn attach_train_graph(fwd: &tinyconv_graph::TinyConvForward, bwd: Graph) -> TrainBundle {
    let d_output = bwd
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .expect("d_output input");
    let loss = bwd.outputs[0];
    let grad_slots: Vec<NodeId> = bwd.outputs[1..1 + fwd.params.len()].to_vec();
    TrainBundle {
        graph: bwd,
        input: fwd.input,
        labels: fwd.labels,
        d_output,
        loss,
        grad_slots,
    }
}

fn remap_graph(g: Graph) -> (Graph, HashMap<NodeId, NodeId>) {
    run_with_remap(g)
}

fn write_slot(arena: &mut Arena, id: NodeId, data: &[f32]) {
    let off = arena.byte_offset(id);
    unsafe {
        let p = arena.raw_buf_mut().as_mut_ptr().add(off) as *mut f32;
        for (i, &v) in data.iter().enumerate() {
            *p.add(i) = v;
        }
    }
}

fn read_slot(arena: &Arena, id: NodeId, n: usize) -> Vec<f32> {
    let off = arena.byte_offset(id);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..n).map(|i| *p.add(i)).collect()
    }
}

fn init_params(graph: &Graph, arena: &mut Arena, rng: &mut Philox4x32) {
    for node in graph.nodes() {
        if let Op::Param { .. } = &node.op {
            let n = node.shape.num_elements().unwrap_or(0);
            if n == 0 {
                continue;
            }
            let scale = (2.0 / n as f32).sqrt();
            let buf: Vec<f32> = (0..n)
                .map(|_| (rng.next_f32() * 2.0 - 1.0) * scale)
                .collect();
            write_slot(arena, node.id, &buf);
        }
    }
}

fn fill_constants(graph: &Graph, arena: &mut Arena) {
    for node in graph.nodes() {
        if let Op::Constant { data } = &node.op {
            let n = node.shape.num_elements().unwrap_or(0);
            let elem = node.shape.dtype().size_bytes();
            let want = n * elem;
            if data.len() >= want {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                write_slot(arena, node.id, &floats);
            }
        }
    }
}

fn synthetic_batch(rng: &mut Philox4x32, pixels: usize) -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..BATCH * pixels)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let labels: Vec<f32> = (0..BATCH).map(|_| (rng.next_u32() % 10) as f32).collect();
    (x, labels)
}

fn execute(bundle: &TrainBundle, x: &[f32], labels: &[f32]) -> ExecResult {
    let (graph, remap) = remap_graph(bundle.graph.clone());
    let r = |id: NodeId| *remap.get(&id).unwrap_or(&id);

    let plan = plan_memory(&graph);
    let mut arena = Arena::from_plan(plan);
    let sched = compile_thunks(&graph, &arena);
    let mut rng = Philox4x32::new(42);
    init_params(&graph, &mut arena, &mut rng);
    fill_constants(&graph, &mut arena);

    write_slot(&mut arena, r(bundle.input), x);
    write_slot(&mut arena, r(bundle.labels), labels);
    write_slot(&mut arena, r(bundle.d_output), &[1.0]);

    execute_thunks(&sched, arena.raw_buf_mut());

    let loss = read_slot(&arena, r(bundle.loss), 1)[0];
    let grads = bundle
        .grad_slots
        .iter()
        .map(|&gid| {
            let id = r(gid);
            let n = graph.node(id).shape.num_elements().unwrap_or(0);
            read_slot(&arena, id, n)
        })
        .collect();

    ExecResult { loss, grads }
}

fn assert_grads_close(a: &[Vec<f32>], b: &[Vec<f32>], rtol: f32) {
    assert_eq!(a.len(), b.len(), "param count");
    for (i, (ga, gb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(ga.len(), gb.len(), "param {i} length");
        for (j, (&va, &vb)) in ga.iter().zip(gb.iter()).enumerate() {
            let denom = va.abs().max(vb.abs()).max(1e-6);
            assert!(
                (va - vb).abs() <= rtol * denom,
                "param {i}[{j}]: {va} vs {vb}"
            );
        }
    }
}

#[test]
fn tinyconv_three_autodiff_paths_agree_on_cpu() {
    let fwd = build_tinyconv_forward(BATCH);
    let (x, labels) = synthetic_batch(&mut Philox4x32::new(7), 28 * 28);

    let bwd_direct = grad_with_loss(&fwd.graph, &fwd.params);
    let r0 = execute(&attach_train_graph(&fwd, bwd_direct), &x, &labels);

    let prepared = prepare_graph_for_ad(fwd.graph.clone());
    let bwd_prepared = grad_with_loss(&prepared, &fwd.params);
    let r1 = execute(&attach_train_graph(&fwd, bwd_prepared), &x, &labels);

    let mir = prepare_mir_for_ad(rlx_ir::mir::MirModule::from_graph(fwd.graph.clone()));
    let bwd_mir = grad_with_loss(mir.as_graph(), &fwd.params);
    let r2 = execute(&attach_train_graph(&fwd, bwd_mir), &x, &labels);

    assert!(r0.loss.is_finite() && r0.loss > 0.0);
    assert!((r0.loss - r1.loss).abs() < 1e-4 * r0.loss.abs().max(1.0));
    assert!((r0.loss - r2.loss).abs() < 1e-4 * r0.loss.abs().max(1.0));
    assert_grads_close(&r0.grads, &r1.grads, 1e-4);
    assert_grads_close(&r0.grads, &r2.grads, 1e-4);
}

#[test]
fn tinyconv_grad_matches_cortexm_trainer_builder() {
    // Same graph + AD entry as rlx-cortexm-trainer/src/graph.rs today.
    let fwd = build_tinyconv_forward(BATCH);
    let bwd = rlx_opt::autodiff::grad_with_loss(&fwd.graph, &fwd.params);
    let (x, labels) = synthetic_batch(&mut Philox4x32::new(99), 28 * 28);
    let r_opt = execute(&attach_train_graph(&fwd, bwd), &x, &labels);

    let bwd_crate = grad_with_loss(&fwd.graph, &fwd.params);
    let r_crate = execute(&attach_train_graph(&fwd, bwd_crate), &x, &labels);

    assert!((r_opt.loss - r_crate.loss).abs() < 1e-6);
    assert_grads_close(&r_opt.grads, &r_crate.grads, 1e-6);
}

// ── MNIST IDX loader (minimal copy of cortexm-trainer) ───────────

struct MnistSplit {
    images: Vec<f32>,
    labels: Vec<f32>,
}

fn load_mnist_split(images_path: &Path, labels_path: &Path) -> Result<MnistSplit, String> {
    let raw = std::fs::read(images_path).map_err(|e| format!("{e}"))?;
    let n = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
    let body = &raw[16..];
    let mut images = Vec::with_capacity(n * 784);
    for &b in &body[..n * 784] {
        images.push((b as f32 / 255.0) * 2.0 - 1.0);
    }
    let lab_raw = std::fs::read(labels_path).map_err(|e| format!("{e}"))?;
    let labels: Vec<f32> = lab_raw[8..8 + n].iter().map(|&b| b as f32).collect();
    Ok(MnistSplit { images, labels })
}

fn mnist_dir() -> Option<std::path::PathBuf> {
    if let Ok(d) = std::env::var("RLX_MNIST_DIR") {
        return Some(std::path::PathBuf::from(d));
    }
    let home = std::env::var("HOME").ok()?;
    let p = std::path::PathBuf::from(format!("{home}/.cache/torchvision-mnist/MNIST/raw"));
    if p.join("train-images-idx3-ubyte").exists() {
        Some(p)
    } else {
        None
    }
}

#[test]
#[ignore = "requires MNIST IDX files; set RLX_MNIST_DIR or torchvision cache"]
fn mnist_real_batch_forward_and_grad() {
    let dir = mnist_dir().expect("MNIST raw dir not found");
    let train = load_mnist_split(
        &dir.join("train-images-idx3-ubyte"),
        &dir.join("train-labels-idx1-ubyte"),
    )
    .expect("load train");

    let fwd = build_tinyconv_forward(BATCH);
    let bwd = grad_with_loss(&fwd.graph, &fwd.params);
    let bundle = attach_train_graph(&fwd, bwd);

    let mut x = vec![0f32; BATCH * 784];
    let mut labels = vec![0f32; BATCH];
    for i in 0..BATCH {
        let src = i * 784;
        x[i * 784..(i + 1) * 784].copy_from_slice(&train.images[src..src + 784]);
        labels[i] = train.labels[i];
    }

    let r = execute(&bundle, &x, &labels);
    assert!(r.loss.is_finite() && r.loss > 0.0);
    assert!(r.grads.iter().flatten().all(|g| g.is_finite()));
    eprintln!("mnist batch loss = {:.4}", r.loss);
}
