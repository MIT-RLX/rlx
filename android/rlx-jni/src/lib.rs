// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal JNI entry points for the Android demo app.
//!
//! - Tiny `matmul → bias → gelu` graph (`runInference` / `backendName`)
//! - Embedded MNIST MLP 784→32→10 (`runMnist` / `mnistExpectedLabel`)

use jni::objects::JClass;
use jni::sys::{jfloatArray, jint, jstring};
use jni::JNIEnv;
use rlx_ir::{op, DType, Graph, Shape};
use rlx_runtime::{is_available, Device, Session};
use std::sync::Mutex;

const MNIST_IN: usize = 784;
const MNIST_HIDDEN: usize = 32;
const MNIST_OUT: usize = 10;

struct DemoState {
    device: Device,
    /// Lazily compiled on first `run_inference`.
    compiled: Option<rlx_runtime::CompiledGraph>,
}

static STATE: Mutex<DemoState> = Mutex::new(DemoState {
    device: Device::Cpu,
    compiled: None,
});

struct MnistState {
    device: Device,
    compiled: Option<rlx_runtime::CompiledGraph>,
    sample: Vec<f32>,
    label: u8,
}

static MNIST: Mutex<Option<MnistState>> = Mutex::new(None);

fn pick_device() -> Device {
    if is_available(Device::Gpu) {
        Device::Gpu
    } else {
        Device::Cpu
    }
}

fn build_demo_graph() -> Graph {
    let mut g = Graph::new("android_demo");
    let x = g.input("x", Shape::new(&[1, 4], DType::F32));
    let w = g.param("w", Shape::new(&[4, 2], DType::F32));
    let b = g.param("b", Shape::new(&[2], DType::F32));
    let mm = g.matmul(x, w, Shape::new(&[1, 2], DType::F32));
    let bias = g.binary(op::BinaryOp::Add, mm, b, Shape::new(&[1, 2], DType::F32));
    let out = g.activation(op::Activation::Gelu, bias, Shape::new(&[1, 2], DType::F32));
    g.set_outputs(vec![out]);
    g
}

fn build_mnist_graph() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("android_mnist_mlp");
    let x = g.input("x", Shape::new(&[1, MNIST_IN], f));
    let w1 = g.param("w1", Shape::new(&[MNIST_IN, MNIST_HIDDEN], f));
    let b1 = g.param("b1", Shape::new(&[MNIST_HIDDEN], f));
    let w2 = g.param("w2", Shape::new(&[MNIST_HIDDEN, MNIST_OUT], f));
    let b2 = g.param("b2", Shape::new(&[MNIST_OUT], f));
    let h = g.matmul(x, w1, Shape::new(&[1, MNIST_HIDDEN], f));
    let h = g.binary(op::BinaryOp::Add, h, b1, Shape::new(&[1, MNIST_HIDDEN], f));
    let h = g.activation(op::Activation::Relu, h, Shape::new(&[1, MNIST_HIDDEN], f));
    let y = g.matmul(h, w2, Shape::new(&[1, MNIST_OUT], f));
    let y = g.binary(op::BinaryOp::Add, y, b2, Shape::new(&[1, MNIST_OUT], f));
    g.set_outputs(vec![y]);
    g
}

fn read_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn ensure_mnist() -> Result<(), String> {
    let mut slot = MNIST.lock().map_err(|e| e.to_string())?;
    if slot.is_some() {
        return Ok(());
    }

    let wbytes = include_bytes!("../assets/mnist_weights.bin");
    let sbytes = include_bytes!("../assets/mnist_sample.bin");
    let expected_w =
        MNIST_IN * MNIST_HIDDEN + MNIST_HIDDEN + MNIST_HIDDEN * MNIST_OUT + MNIST_OUT;
    let weights = read_f32_le(wbytes);
    if weights.len() != expected_w {
        return Err(format!(
            "mnist_weights.bin has {} floats, expected {expected_w}",
            weights.len()
        ));
    }
    if sbytes.len() != MNIST_IN * 4 + 1 {
        return Err(format!(
            "mnist_sample.bin has {} bytes, expected {}",
            sbytes.len(),
            MNIST_IN * 4 + 1
        ));
    }
    let sample = read_f32_le(&sbytes[..MNIST_IN * 4]);
    let label = sbytes[MNIST_IN * 4];

    let device = pick_device();
    let session = Session::new(device);
    let mut compiled = session.compile(build_mnist_graph());

    let mut off = 0;
    let w1 = &weights[off..off + MNIST_IN * MNIST_HIDDEN];
    off += MNIST_IN * MNIST_HIDDEN;
    let b1 = &weights[off..off + MNIST_HIDDEN];
    off += MNIST_HIDDEN;
    let w2 = &weights[off..off + MNIST_HIDDEN * MNIST_OUT];
    off += MNIST_HIDDEN * MNIST_OUT;
    let b2 = &weights[off..off + MNIST_OUT];

    compiled.set_param("w1", w1);
    compiled.set_param("b1", b1);
    compiled.set_param("w2", w2);
    compiled.set_param("b2", b2);

    *slot = Some(MnistState {
        device,
        compiled: Some(compiled),
        sample,
        label,
    });
    Ok(())
}

fn run_mnist_inner() -> Result<(Device, Vec<f32>, u8, usize), String> {
    ensure_mnist()?;
    let mut slot = MNIST.lock().map_err(|e| e.to_string())?;
    let state = slot.as_mut().expect("initialized");
    let device = state.device;
    let label = state.label;
    let sample = state.sample.clone();
    let compiled = state.compiled.as_mut().expect("compiled");
    let outs = compiled.run(&[("x", sample.as_slice())]);
    let logits = outs[0].clone();
    let pred = argmax(&logits);
    Ok((device, logits, label, pred))
}

fn run_inference_inner() -> Result<(Device, Vec<f32>), String> {
    let mut state = STATE.lock().map_err(|e| e.to_string())?;
    if state.compiled.is_none() {
        state.device = pick_device();
        let session = Session::new(state.device);
        let mut compiled = session.compile(build_demo_graph());
        compiled.set_param(
            "w",
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0,
            ],
        );
        compiled.set_param("b", &[0.5, -0.5]);
        state.compiled = Some(compiled);
    }

    let device = state.device;
    let compiled = state.compiled.as_mut().expect("initialized above");
    let x = [1.0, 0.0, 0.0, 0.0];
    let outs = compiled.run(&[("x", &x)]);
    Ok((device, outs[0].clone()))
}

fn throw_runtime(env: &mut JNIEnv<'_>, msg: &str) {
    let _ = env.throw_new("java/lang/RuntimeException", msg);
}

fn f32_array(env: &mut JNIEnv<'_>, data: &[f32]) -> jfloatArray {
    let arr = env
        .new_float_array(data.len() as i32)
        .expect("new_float_array");
    env.set_float_array_region(&arr, 0, data)
        .expect("set_float_array_region");
    arr.into_raw()
}

fn device_label(device: Device) -> &'static str {
    match device {
        Device::Gpu | Device::Vulkan => "GPU (Vulkan / wgpu)",
        Device::Cpu => "CPU (NEON)",
        other => other.name(),
    }
}

/// Compile (once) and run the demo graph. Returns two GELU outputs.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_mit_rlx_RlxNative_runInference(
    mut env: JNIEnv,
    _class: JClass,
) -> jfloatArray {
    match run_inference_inner() {
        Ok((_device, out)) => f32_array(&mut env, &out),
        Err(e) => {
            throw_runtime(&mut env, &e);
            std::ptr::null_mut()
        }
    }
}

/// Backend label for the UI (`CPU` or `GPU`).
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_mit_rlx_RlxNative_backendName(
    mut env: JNIEnv,
    _class: JClass,
) -> jstring {
    let label = match run_inference_inner() {
        Ok((device, _)) => device_label(device),
        Err(e) => {
            throw_runtime(&mut env, &e);
            return std::ptr::null_mut();
        }
    };
    env.new_string(label)
        .expect("new_string")
        .into_raw()
}

/// Run the embedded MNIST sample through the MLP. Returns 10 logits.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_mit_rlx_RlxNative_runMnist(
    mut env: JNIEnv,
    _class: JClass,
) -> jfloatArray {
    match run_mnist_inner() {
        Ok((_device, logits, _label, _pred)) => f32_array(&mut env, &logits),
        Err(e) => {
            throw_runtime(&mut env, &e);
            std::ptr::null_mut()
        }
    }
}

/// Argmax class for the embedded MNIST sample.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_mit_rlx_RlxNative_mnistPredict(
    mut env: JNIEnv,
    _class: JClass,
) -> jint {
    match run_mnist_inner() {
        Ok((_device, _logits, _label, pred)) => pred as jint,
        Err(e) => {
            throw_runtime(&mut env, &e);
            -1
        }
    }
}

/// Ground-truth label of the embedded MNIST sample (for tests / UI).
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_mit_rlx_RlxNative_mnistExpectedLabel(
    mut env: JNIEnv,
    _class: JClass,
) -> jint {
    match ensure_mnist() {
        Ok(()) => {
            let slot = MNIST.lock().unwrap();
            slot.as_ref().unwrap().label as jint
        }
        Err(e) => {
            throw_runtime(&mut env, &e);
            -1
        }
    }
}

/// Host-side unit test hook — not exported to Java.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_graph_runs_on_cpu() {
        let session = Session::new(Device::Cpu);
        let mut compiled = session.compile(build_demo_graph());
        compiled.set_param(
            "w",
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0,
            ],
        );
        compiled.set_param("b", &[0.5, -0.5]);
        let outs = compiled.run(&[("x", &[1.0, 0.0, 0.0, 0.0])]);
        assert_eq!(outs[0].len(), 2);
        assert!(outs[0][0].is_finite());
        assert!(outs[0][1].is_finite());
    }

    #[test]
    fn mnist_mlp_predicts_embedded_sample() {
        let (device, logits, label, pred) = run_mnist_inner().expect("mnist");
        assert!(matches!(device, Device::Cpu | Device::Gpu | Device::Vulkan));
        assert_eq!(logits.len(), 10);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(pred, label as usize, "pred={pred} label={label} logits={logits:?}");
    }

    #[cfg(feature = "blas")]
    #[test]
    fn blas_feature_enabled() {
        assert!(cfg!(feature = "blas"));
    }

    #[cfg(not(feature = "blas"))]
    #[test]
    fn scalar_feature_enabled() {
        assert!(cfg!(feature = "scalar"));
    }
}
