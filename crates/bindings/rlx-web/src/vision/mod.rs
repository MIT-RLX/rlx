// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vision classification models for browser benchmarks.

mod cifar_cnn;
mod conv;
mod mnist_cnn;
mod mnist_mlp;
mod resnet;

use rlx_autodiff::grad_with_loss;
use rlx_ir::{Graph, Philox4x32};

/// Batch size for browser vision runs (single image).
pub const BATCH: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VisionModelId {
    MnistCnn,
    MnistMlp,
    CifarCnn,
    ResNet,
}

impl VisionModelId {
    pub fn all() -> &'static [Self] {
        &[Self::MnistCnn, Self::MnistMlp, Self::CifarCnn, Self::ResNet]
    }

    pub fn from_slug(slug: &str) -> Option<Self> {
        match slug {
            "mnist-cnn" | "mnist_cnn" => Some(Self::MnistCnn),
            "mnist-mlp" | "mnist_mlp" => Some(Self::MnistMlp),
            "cifar-cnn" | "cifar_cnn" => Some(Self::CifarCnn),
            "resnet" | "resnet-cifar" => Some(Self::ResNet),
            _ => None,
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::MnistCnn => "mnist-cnn",
            Self::MnistMlp => "mnist-mlp",
            Self::CifarCnn => "cifar-cnn",
            Self::ResNet => "resnet",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::MnistCnn => "MNIST TinyConv CNN",
            Self::MnistMlp => "MNIST MLP (784→128→10)",
            Self::CifarCnn => "CIFAR-10 CNN",
            Self::ResNet => "CIFAR ResNet-style",
        }
    }

    /// WebGL lacks conv backward decomposition — only MLP trains there.
    pub fn webgl_train(self) -> bool {
        matches!(self, Self::MnistMlp)
    }
}

#[derive(Debug, Clone)]
pub struct ParamSpec {
    pub name: &'static str,
    pub shape: &'static [usize],
    pub fan_in: usize,
}

#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub id: VisionModelId,
    pub input_shape: &'static [usize],
    pub num_classes: usize,
    pub params: Vec<ParamSpec>,
}

impl ModelMeta {
    pub fn input_len(&self) -> usize {
        self.input_shape.iter().product()
    }

    #[allow(dead_code)]
    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    pub fn param_flat_len(&self) -> usize {
        self.params
            .iter()
            .map(|p| p.shape.iter().product::<usize>())
            .sum()
    }

    pub fn param_names(&self) -> Vec<&'static str> {
        self.params.iter().map(|p| p.name).collect()
    }

    pub fn param_sizes(&self) -> Vec<usize> {
        self.params
            .iter()
            .map(|p| p.shape.iter().product())
            .collect()
    }
}

pub fn model_meta(id: VisionModelId) -> ModelMeta {
    match id {
        VisionModelId::MnistCnn => ModelMeta {
            id,
            input_shape: mnist_cnn::INPUT,
            num_classes: mnist_cnn::NUM_CLASSES,
            params: vec![
                spec("conv1_w", &[8, 1, 3, 3], 3 * 3),
                spec("conv1_b", &[8], 1),
                spec("conv2_w", &[16, 8, 3, 3], 8 * 3 * 3),
                spec("conv2_b", &[16], 1),
                spec("fc_w", &[400, 10], 400),
                spec("fc_b", &[10], 1),
            ],
        },
        VisionModelId::MnistMlp => ModelMeta {
            id,
            input_shape: mnist_mlp::INPUT,
            num_classes: mnist_mlp::NUM_CLASSES,
            params: vec![
                spec("w1", &[784, mnist_mlp::HIDDEN], 784),
                spec("b1", &[mnist_mlp::HIDDEN], 1),
                spec("w2", &[mnist_mlp::HIDDEN, 10], mnist_mlp::HIDDEN),
                spec("b2", &[10], 1),
            ],
        },
        VisionModelId::CifarCnn => ModelMeta {
            id,
            input_shape: cifar_cnn::INPUT,
            num_classes: cifar_cnn::NUM_CLASSES,
            params: vec![
                spec("conv1_w", &[32, 3, 3, 3], 3 * 3 * 3),
                spec("conv1_b", &[32], 1),
                spec("conv2_w", &[64, 32, 3, 3], 32 * 3 * 3),
                spec("conv2_b", &[64], 1),
                spec("conv3_w", &[128, 64, 3, 3], 64 * 3 * 3),
                spec("conv3_b", &[128], 1),
                spec("fc_w", &[2048, 10], 2048),
                spec("fc_b", &[10], 1),
            ],
        },
        VisionModelId::ResNet => ModelMeta {
            id,
            input_shape: resnet::INPUT,
            num_classes: resnet::NUM_CLASSES,
            params: vec![
                spec("stem_w", &[32, 3, 3, 3], 3 * 3 * 3),
                spec("stem_b", &[32], 1),
                spec("b1a_w", &[32, 32, 3, 3], 32 * 3 * 3),
                spec("b1a_b", &[32], 1),
                spec("b1b_w", &[32, 32, 3, 3], 32 * 3 * 3),
                spec("b1b_b", &[32], 1),
                spec("b2a_w", &[64, 32, 3, 3], 32 * 3 * 3),
                spec("b2a_b", &[64], 1),
                spec("b2b_w", &[64, 64, 3, 3], 64 * 3 * 3),
                spec("b2b_b", &[64], 1),
                spec("skip_w", &[64, 32, 1, 1], 32),
                spec("skip_b", &[64], 1),
                spec("fc_w", &[64, 10], 64),
                spec("fc_b", &[10], 1),
            ],
        },
    }
}

fn spec(name: &'static str, shape: &'static [usize], fan_in: usize) -> ParamSpec {
    ParamSpec {
        name,
        shape,
        fan_in,
    }
}

pub fn build_forward(id: VisionModelId, batch: usize) -> Graph {
    let g = match id {
        VisionModelId::MnistCnn => mnist_cnn::build_forward(batch),
        VisionModelId::MnistMlp => mnist_mlp::build_forward(batch),
        VisionModelId::CifarCnn => cifar_cnn::build_forward(batch),
        VisionModelId::ResNet => resnet::build_forward(batch),
    };
    legalize(g)
}

pub fn build_train(id: VisionModelId, batch: usize) -> Graph {
    let (loss_g, param_ids) = match id {
        VisionModelId::MnistCnn => mnist_cnn::build_loss(batch),
        VisionModelId::MnistMlp => mnist_mlp::build_loss(batch),
        VisionModelId::CifarCnn => cifar_cnn::build_loss(batch),
        VisionModelId::ResNet => resnet::build_loss(batch),
    };
    let bwd = grad_with_loss(&loss_g, &param_ids);
    legalize(bwd)
}

fn legalize(g: Graph) -> Graph {
    rlx_opt::legalize_broadcast::run(g)
}

pub fn init_params(meta: &ModelMeta, seed: u32) -> Vec<f32> {
    let mut rng = Philox4x32::new(u64::from(seed.max(1)));
    let mut flat = Vec::with_capacity(meta.param_flat_len());
    for p in &meta.params {
        let n = p.shape.iter().product::<usize>();
        let scale = (2.0 / p.fan_in as f32).sqrt();
        for _ in 0..n {
            let v = rng.next_f32() * 2.0 * scale - scale;
            flat.push(if p.name.ends_with("_b") { 0.0 } else { v });
        }
    }
    flat
}

pub fn apply_params(compiled: &mut rlx_runtime::CompiledGraph, meta: &ModelMeta, flat: &[f32]) {
    let mut off = 0;
    for p in &meta.params {
        let n = p.shape.iter().product::<usize>();
        compiled.set_param(p.name, &flat[off..off + n]);
        off += n;
    }
}

#[cfg(any(feature = "webgpu", feature = "webgl"))]
pub fn apply_browser_params(
    compiled: &mut rlx_runtime::BrowserCompiledGraph,
    meta: &ModelMeta,
    flat: &[f32],
) {
    let mut off = 0;
    for p in &meta.params {
        let n = p.shape.iter().product::<usize>();
        compiled.set_param(p.name, &flat[off..off + n]);
        off += n;
    }
}

pub fn train_step_params(meta: &ModelMeta, params: &[f32], grads: &[&[f32]], lr: f32) -> Vec<f32> {
    let mut out = params.to_vec();
    let mut off = 0;
    for (p, g) in meta.params.iter().zip(grads) {
        let n = p.shape.iter().product::<usize>();
        for i in 0..n {
            out[off + i] -= lr * g[i];
        }
        off += n;
    }
    out
}

pub fn synthetic_input(meta: &ModelMeta, seed: u32) -> (Vec<f32>, f32) {
    let mut rng = Philox4x32::new(u64::from(seed.wrapping_add(0xA5A5_5A5A)));
    let n = meta.input_len();
    let data: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let label = (seed % meta.num_classes as u32) as f32;
    (data, label)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_runtime::{Device, Session};

    #[test]
    fn all_models_compile_on_cpu() {
        for &id in VisionModelId::all() {
            let meta = model_meta(id);
            let fwd = build_forward(id, BATCH);
            let _ = Session::new(Device::Cpu).compile(fwd);
            let train = build_train(id, BATCH);
            let _ = Session::new(Device::Cpu).compile(train);
            assert!(meta.param_flat_len() > 0);
        }
    }

    #[test]
    fn mnist_cnn_forward_runs() {
        let id = VisionModelId::MnistCnn;
        let meta = model_meta(id);
        let params = init_params(&meta, 42);
        let (x, _label) = synthetic_input(&meta, 1);
        let mut compiled = Session::new(Device::Cpu).compile(build_forward(id, BATCH));
        apply_params(&mut compiled, &meta, &params);
        let outs = compiled.run(&[("x", &x)]);
        assert_eq!(outs[0].len(), meta.num_classes);
    }

    #[test]
    fn all_models_train_step_runs_and_loss_finite() {
        for &id in VisionModelId::all() {
            let meta = model_meta(id);
            let mut params = init_params(&meta, 42);
            let (x, label) = synthetic_input(&meta, 3);
            let mut compiled = Session::new(Device::Cpu).compile(build_train(id, BATCH));
            apply_params(&mut compiled, &meta, &params);
            let outs = compiled.run(&[("x", &x), ("labels", &[label]), ("d_output", &[1.0])]);
            let loss0 = outs[0][0];
            assert!(loss0.is_finite(), "{} loss not finite: {loss0}", id.slug());
            let grads: Vec<&[f32]> = outs.iter().skip(1).map(|v| v.as_slice()).collect();
            params = train_step_params(&meta, &params, &grads, 0.05);
            apply_params(&mut compiled, &meta, &params);
            let outs2 = compiled.run(&[("x", &x), ("labels", &[label]), ("d_output", &[1.0])]);
            let loss1 = outs2[0][0];
            assert!(loss1.is_finite(), "{} post-step loss not finite", id.slug());
        }
    }
}
