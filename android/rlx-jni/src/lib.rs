// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal JNI entry points for the Android demo app.
//!
//! - Tiny `matmul → bias → gelu` graph (`runInference` / `backendName`)
//! - Embedded MNIST MLP 784→32→10 (`runMnist` / `mnistExpectedLabel`)

use jni::objects::{JClass, JString};
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

// ── distributed node ───────────────────────────────────────────────────────
//
// Lets an Android handset join an RLX mesh as a worker rank. The node runs on
// its own thread: JNI calls from the UI thread must not block, and a serving
// loop parks in `recv` between activations.

use rlx_runtime::dist::node::{
    NodeConfig, NodeControl, NodeStopHandle, serve_trainer_here, serve_worker,
};
use std::sync::mpsc;

struct NodeSlot {
    stop: NodeStopHandle,
    /// Set once the node thread finishes, so `nodeStatus` can report the
    /// outcome rather than leaving the caller guessing.
    done: mpsc::Receiver<String>,
    last: Option<String>,
}

static NODE: Mutex<Option<NodeSlot>> = Mutex::new(None);

fn node_start_inner(
    rank: i32,
    world: i32,
    peers: String,
    device: String,
    mode: String,
) -> Result<(), String> {
    let training = match mode.as_str() {
        "" | "infer" => false,
        "train" => true,
        other => return Err(format!("unknown mode [{other}]; expected infer or train")),
    };
    let mut slot = NODE.lock().map_err(|_| "node lock poisoned".to_string())?;
    if slot.as_ref().is_some_and(|s| s.last.is_none()) {
        return Err("a node is already running on this device".into());
    }
    if rank < 0 || world < 1 || rank >= world {
        return Err(format!("bad rank/world: {rank}/{world}"));
    }

    let addrs: Vec<String> = peers
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // An empty peer list means "find the coordinator by UDP broadcast" — the
    // path RlxNode.start(discovery = true) takes a MulticastLock for.
    // Topology follows from what the caller supplied. A phone is a star
    // worker: it dials the coordinator and nothing dials it back (most Wi-Fi
    // will not route inbound to a handset), so a lone address — or discovery,
    // where the coordinator announces itself — means star. A full per-rank
    // list means the caller wants a mesh and believes every rank is reachable.
    let base = NodeConfig::new(rank as u32, world as u32).device(device);
    let cfg = if addrs.is_empty() {
        base.star().discover(29600, 29500)
    } else if addrs.len() == 1 && world > 1 {
        base.star().peers(addrs)?
    } else {
        base.mesh().peers(addrs)?
    };

    let ctl = NodeControl::unbounded();
    let stop = ctl.stop_handle();
    let (tx, rx) = mpsc::channel();

    std::thread::Builder::new()
        .name("rlx-node".into())
        .spawn(move || {
            let msg = match cfg.connect() {
                Err(e) => format!("connect failed: {e}"),
                Ok(group) if training => match serve_trainer_here(&group, |_uri| Vec::new(), false)
                {
                    Ok(r) => format!(
                        "ok: rank {} trained on {} ({}), {} sample(s), loss {:.4}->{:.4}",
                        r.rank,
                        r.metrics.device.name(),
                        r.platform,
                        r.metrics.samples,
                        r.metrics.first_loss,
                        r.metrics.last_loss
                    ),
                    Err(e) => format!("error: {e}"),
                },
                Ok(group) => match serve_worker(&group, |uri| {
                    // No custom weight scheme on the handset: the built-in
                    // gguf:// / safetensors:// / file:// resolvers already ran.
                    let _ = uri;
                    Vec::new()
                }) {
                    Ok(r) => format!(
                        "ok: rank {} on {} ({}), {} activation(s)",
                        r.rank,
                        r.device.name(),
                        r.platform,
                        r.activations
                    ),
                    Err(e) => format!("error: {e}"),
                },
            };
            let _ = tx.send(msg);
        })
        .map_err(|e| format!("spawn: {e}"))?;

    *slot = Some(NodeSlot {
        stop,
        done: rx,
        last: None,
    });
    Ok(())
}

fn node_status_inner() -> String {
    let Ok(mut slot) = NODE.lock() else {
        return "node lock poisoned".into();
    };
    match slot.as_mut() {
        None => "idle".into(),
        Some(s) => {
            if s.last.is_none()
                && let Ok(msg) = s.done.try_recv()
            {
                s.last = Some(msg);
            }
            match &s.last {
                Some(m) => m.clone(),
                None if s.stop.is_stopped() => "stopping".into(),
                None => "running".into(),
            }
        }
    }
}

/// Join a mesh as worker `rank` of `world`. `peers` is a comma-separated
/// `host:port` list indexed by rank; `device` is `auto` or a backend name;
/// `mode` is `infer` or `train`. Returns immediately — poll `nodeStatus`.
///
/// A training rank cannot drop out partway: the gradient reduce is a barrier,
/// so stopping one stalls every other rank.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_mit_rlx_RlxNative_nodeStart(
    mut env: JNIEnv,
    _class: JClass,
    rank: jint,
    world: jint,
    peers: JString,
    device: JString,
    mode: JString,
) -> jstring {
    let peers: String = match env.get_string(&peers) {
        Ok(s) => s.into(),
        Err(e) => {
            throw_runtime(&mut env, &format!("peers: {e}"));
            return std::ptr::null_mut();
        }
    };
    let device: String = match env.get_string(&device) {
        Ok(s) => s.into(),
        Err(e) => {
            throw_runtime(&mut env, &format!("device: {e}"));
            return std::ptr::null_mut();
        }
    };
    let mode: String = match env.get_string(&mode) {
        Ok(s) => s.into(),
        Err(e) => {
            throw_runtime(&mut env, &format!("mode: {e}"));
            return std::ptr::null_mut();
        }
    };
    if let Err(e) = node_start_inner(rank, world, peers, device, mode) {
        throw_runtime(&mut env, &e);
        return std::ptr::null_mut();
    }
    env.new_string("started").expect("new_string").into_raw()
}

/// Current node state: `idle` | `running` | `stopping` | `ok: …` | `error: …`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_mit_rlx_RlxNative_nodeStatus(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    env.new_string(node_status_inner())
        .expect("new_string")
        .into_raw()
}

/// Ask the node to leave the mesh after its current activation. Cooperative:
/// a node parked in `recv` exits when its peer sends or the link drops.
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_mit_rlx_RlxNative_nodeStop(_env: JNIEnv, _class: JClass) {
    if let Ok(slot) = NODE.lock()
        && let Some(s) = slot.as_ref()
    {
        s.stop.stop();
    }
}
