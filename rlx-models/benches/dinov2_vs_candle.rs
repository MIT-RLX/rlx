// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! Criterion bench: rlx-models DINOv2 vs candle DINOv2 (CPU).
//!
//! ## Running
//!
//! ```bash
//! RLX_DINOV2_WEIGHTS=/path/to/dinov2_vits14.safetensors \
//!   cargo bench -p rlx-models --features parity-candle --bench dinov2_vs_candle
//! ```
//!
//! Without `RLX_DINOV2_WEIGHTS` the bench exits early with a note.

#![cfg(feature = "parity-candle")]

use candle_core::{DType as CDType, Device as CDevice, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::dinov2 as candle_dinov2;
use criterion::{Criterion, criterion_group, criterion_main};
use rlx_models::WeightMap;
use rlx_models::dinov2::{DinoV2Config, assemble_hidden, build_dinov2_graph_sized};
use rlx_runtime::{Device, Session};
use std::hint::black_box;

const IMG_SIZE: usize = 518;
const PATCH_SIZE: usize = 14;

fn synthesize_image() -> Vec<f32> {
    let n = 3 * IMG_SIZE * IMG_SIZE;
    let mut v = vec![0f32; n];
    for c in 0..3 {
        let phase = (c as f32) * 0.7;
        for y in 0..IMG_SIZE {
            for x in 0..IMG_SIZE {
                let fx = x as f32 / IMG_SIZE as f32;
                let fy = y as f32 / IMG_SIZE as f32;
                v[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] =
                    (6.28 * fx + phase).sin() * (3.14 * fy + phase).cos();
            }
        }
    }
    v
}

fn bench(c: &mut Criterion) {
    let Some(weights) = std::env::var("RLX_DINOV2_WEIGHTS").ok() else {
        eprintln!(
            "skipping dinov2_vs_candle bench — set RLX_DINOV2_WEIGHTS=/path/to/dinov2_vits14.safetensors"
        );
        return;
    };

    let image = synthesize_image();

    // ── candle setup ──
    let cdev = CDevice::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&weights], CDType::F32, &cdev)
            .expect("candle: load safetensors")
    };
    let candle_model = candle_dinov2::vit_small(vb).expect("candle: build vit_small");
    let candle_input = Tensor::from_slice(&image, (1, 3, IMG_SIZE, IMG_SIZE), &cdev)
        .expect("candle: tensor from image");

    // ── rlx setup ──
    let cfg = DinoV2Config::vit_small(IMG_SIZE);
    let mut wm = WeightMap::from_file(&weights).expect("rlx: load safetensors");
    let (graph, params, pre) =
        build_dinov2_graph_sized(&cfg, &mut wm, 1).expect("rlx: build graph");
    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(graph);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    let hidden =
        assemble_hidden(&pre, &image, 1, PATCH_SIZE, IMG_SIZE).expect("rlx: assemble hidden");

    let mut group = c.benchmark_group("dinov2_vit_small_518");
    group.sample_size(10);

    group.bench_function("candle_cpu_forward", |b| {
        b.iter(|| {
            let logits = candle_model.forward(black_box(&candle_input)).unwrap();
            black_box(logits);
        });
    });

    group.bench_function("rlx_cpu_forward_graph_only", |b| {
        // Just the IR graph execution — preprocess + assemble done once.
        b.iter(|| {
            let outs = compiled.run(&[("hidden", black_box(hidden.as_slice()))]);
            black_box(outs);
        });
    });

    group.bench_function("rlx_cpu_forward_with_assemble", |b| {
        // Includes host-side patchify each iter — closer to "image in,
        // logits out" parity with candle's forward (which patchifies
        // internally via Conv2d).
        b.iter(|| {
            let h = assemble_hidden(&pre, black_box(&image), 1, PATCH_SIZE, IMG_SIZE).unwrap();
            let outs = compiled.run(&[("hidden", h.as_slice())]);
            black_box(outs);
        });
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
