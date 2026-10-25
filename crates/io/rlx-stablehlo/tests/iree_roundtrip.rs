// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Export → `iree-compile` (llvm-cpu) → `iree-run-module`, compared with the
//! rlx CPU backend on the same inputs.
//!
//! Needs `iree-compile` and `iree-run-module` on `PATH` (or `IREE_COMPILE` /
//! `IREE_RUN_MODULE` pointing at them); skips otherwise, and
//! `RLX_REQUIRE_DEVICE=1` turns the skip into a failure. Each case goes
//! through `tests/known_failures/iree_roundtrip.txt` (device `iree`).
//!
//! Tensors cross the process boundary as `.npy` files.

use std::path::{Path, PathBuf};
use std::process::Command;

use rlx_flow::MapWeights;
use rlx_flow::prelude::*;
use rlx_ir::known_failures::KnownFailures;
use rlx_ir::op::{Activation, BinaryOp, MaskKind, ReduceOp};
use rlx_ir::{DType, Graph, Op, RopeStyle, Shape};
use rlx_runtime::{Device, Session};
use rlx_stablehlo::{SafetensorsEntry, StableHloOptions, TensorSig, export, write_safetensors};

// ── IREE tools ──────────────────────────────────────────────────────────

fn tool(env: &str, name: &str) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(env) {
        return Some(PathBuf::from(p));
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|d| d.join(name))
            .find(|p| p.is_file())
    })
}

struct Iree {
    compile: PathBuf,
    run: PathBuf,
}

fn iree() -> Option<Iree> {
    let compile = tool("IREE_COMPILE", "iree-compile");
    let run = tool("IREE_RUN_MODULE", "iree-run-module");
    let found = compile.is_some() && run.is_some();
    if rlx_ir::env::skip_unless_device("iree", true, found) {
        return None;
    }
    Some(Iree {
        compile: compile?,
        run: run?,
    })
}

// ── .npy ────────────────────────────────────────────────────────────────

fn descr(dt: DType) -> (&'static str, usize) {
    match dt {
        DType::F32 => ("<f4", 4),
        DType::I32 => ("<i4", 4),
        DType::I64 => ("<i8", 8),
        DType::I8 => ("|i1", 1),
        DType::Bool => ("|b1", 1),
        other => panic!("npy: dtype {other:?} not used by these cases"),
    }
}

fn write_npy(path: &Path, sig: &TensorSig, values: &[f32]) {
    let (d, _) = descr(sig.dtype);
    let shape = match sig.shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", sig.shape[0]),
        _ => format!(
            "({})",
            sig.shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!("{{'descr': '{d}', 'fortran_order': False, 'shape': {shape}, }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for &v in values {
        match sig.dtype {
            DType::F32 => bytes.extend_from_slice(&v.to_le_bytes()),
            DType::I32 => bytes.extend_from_slice(&(v as i32).to_le_bytes()),
            DType::I64 => bytes.extend_from_slice(&(v as i64).to_le_bytes()),
            DType::I8 => bytes.push(v as i8 as u8),
            DType::Bool => bytes.push((v != 0.0) as u8),
            _ => unreachable!(),
        }
    }
    std::fs::write(path, bytes).unwrap();
}

fn read_npy(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&b[..6], b"\x93NUMPY", "{}: not an npy file", path.display());
    let (hlen, start) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[start..start + hlen]).unwrap();
    let data = &b[start + hlen..];
    if header.contains("'<f4'") {
        data.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else if header.contains("'<i4'") {
        data.as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect()
    } else if header.contains("'<i8'") {
        data.as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c) as f32)
            .collect()
    } else if header.contains("'|i1'") {
        data.iter().map(|&x| x as i8 as f32).collect()
    } else if header.contains("'|b1'") {
        data.iter().map(|&x| x as f32).collect()
    } else {
        panic!("{}: unhandled npy header {header}", path.display())
    }
}

// ── Feeds ───────────────────────────────────────────────────────────────

/// Deterministic values for every `Input`/`Param`: small floats, and for an
/// integer tensor indices that are in range for the gather they feed.
fn feeds(g: &Graph) -> Vec<(String, bool, DType, Vec<f32>)> {
    let mut out = Vec::new();
    let mut seed = 1u32;
    for n in g.nodes() {
        let (name, param) = match &n.op {
            Op::Input { name } => (name.clone(), false),
            Op::Param { name } => (name.clone(), true),
            _ => continue,
        };
        let len = n.shape.num_elements().unwrap();
        let dt = n.shape.dtype();
        let data: Vec<f32> = (0..len)
            .map(|i| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let u = ((seed >> 8) & 0xFFFF) as f32 / 65_535.0 - 0.5;
                if name.ends_with("mask") {
                    // A padding mask is 0/1: attend everywhere but the last key.
                    if i + 1 == len { 0.0 } else { 1.0 }
                } else if dt.is_float() {
                    if param { u * 0.4 } else { u * 2.0 }
                } else if dt == DType::I8 {
                    // Quantized weights: the whole signed range in use.
                    (u * 250.0).round()
                } else {
                    (i % 4) as f32
                }
            })
            .collect();
        out.push((name, param, dt, data));
    }
    out
}

// ── One case ────────────────────────────────────────────────────────────

/// Outputs of `g` from IREE and from rlx CPU, on the same feeds.
type Outputs = (Vec<Vec<f32>>, Vec<Vec<f32>>);

/// The first `n` elements of `bytes` read as `dt`, as f32.
fn decode(bytes: &[u8], dt: DType, n: usize) -> Vec<f32> {
    match dt {
        DType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .take(n)
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        DType::I32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .take(n)
            .map(|c| i32::from_le_bytes(*c) as f32)
            .collect(),
        DType::I64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .take(n)
            .map(|c| i64::from_le_bytes(*c) as f32)
            .collect(),
        DType::I8 => bytes.iter().take(n).map(|&b| b as i8 as f32).collect(),
        DType::Bool | DType::U8 => bytes.iter().take(n).map(|&b| b as f32).collect(),
        other => panic!("output dtype {other:?} not used by these cases"),
    }
}

/// Raw little-endian bytes of `values` stored as `dt`.
fn le_bytes(dt: DType, values: &[f32]) -> Vec<u8> {
    let mut b = Vec::new();
    for &v in values {
        match dt {
            DType::F32 => b.extend_from_slice(&v.to_le_bytes()),
            DType::I32 => b.extend_from_slice(&(v as i32).to_le_bytes()),
            DType::I64 => b.extend_from_slice(&(v as i64).to_le_bytes()),
            DType::I8 => b.push(v as i8 as u8),
            other => panic!("parameter dtype {other:?} not used by these cases"),
        }
    }
    b
}

/// `params`: export weights as IREE parameters (read from a safetensors file)
/// instead of as function arguments.
fn run_both(iree: &Iree, name: &str, g: Graph, params: bool) -> Result<Outputs, String> {
    let feeds = feeds(&g);

    // rlx side through the typed API: `run()` widens to f32 assuming integer
    // activations are stored widened, which an int8 `Quantize` output is not
    // (see the note in `rlx_driver::arena::read_typed_to_f32`).
    let out_meta: Vec<(DType, usize)> = g
        .outputs
        .iter()
        .map(|&o| {
            let s = &g.node(o).shape;
            (s.dtype(), s.num_elements().unwrap())
        })
        .collect();
    let mut c = Session::new(Device::Cpu).compile(g.clone());
    for (n, param, _, d) in &feeds {
        if *param {
            c.set_param(n, d);
        }
    }
    let typed: Vec<(String, Vec<u8>, DType)> = feeds
        .iter()
        .filter(|f| !f.1)
        .map(|(n, _, dt, d)| (n.clone(), le_bytes(*dt, d), *dt))
        .collect();
    let ins: Vec<(&str, &[u8], DType)> = typed
        .iter()
        .map(|(n, b, dt)| (n.as_str(), b.as_slice(), *dt))
        .collect();
    let want: Vec<Vec<f32>> = c
        .run_typed(&ins)
        .into_iter()
        .zip(&out_meta)
        .map(|((bytes, dt), &(_, n))| decode(&bytes, dt, n))
        .collect();

    let opts = if params {
        StableHloOptions::default().with_parameters("model")
    } else {
        StableHloOptions::default()
    };
    let m = export(g, &opts).map_err(|e| e.to_string())?;
    let dir = std::env::temp_dir().join(format!("rlx-stablehlo-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let weights = dir.join("weights.safetensors");
    if !m.parameters.is_empty() {
        let blobs: Vec<Vec<u8>> = m
            .parameters
            .iter()
            .map(|sig| {
                let f = feeds
                    .iter()
                    .find(|f| f.0 == sig.name)
                    .expect("parameter feed");
                le_bytes(sig.dtype, &f.3)
            })
            .collect();
        let entries: Vec<SafetensorsEntry<'_>> = m
            .parameters
            .iter()
            .zip(&blobs)
            .map(|(sig, data)| SafetensorsEntry {
                name: &sig.name,
                dtype: sig.dtype,
                shape: &sig.shape,
                data,
            })
            .collect();
        write_safetensors(&weights, &entries).unwrap();
    }
    let mlir = dir.join("model.mlir");
    let vmfb = dir.join("model.vmfb");
    std::fs::write(&mlir, &m.text).unwrap();

    let out = Command::new(&iree.compile)
        .args([
            "--iree-input-type=stablehlo",
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
        ])
        .arg(&mlir)
        .arg("-o")
        .arg(&vmfb)
        .output()
        .unwrap();
    if !out.status.success() {
        let e = String::from_utf8_lossy(&out.stderr);
        let first = e.lines().find(|l| l.contains("error")).unwrap_or("");
        return Err(format!("iree-compile failed ({}): {first}", mlir.display()));
    }

    let mut cmd = Command::new(&iree.run);
    cmd.arg(format!("--module={}", vmfb.display()))
        .arg("--device=local-task")
        .arg("--function=main");
    if !m.parameters.is_empty() {
        cmd.arg(format!("--parameters=model={}", weights.display()));
    }
    for (k, sig) in m.args.iter().enumerate() {
        let data = &feeds
            .iter()
            .find(|f| f.0 == sig.name)
            .unwrap_or_else(|| panic!("no feed for argument {}", sig.name))
            .3;
        let p = dir.join(format!("in{k}.npy"));
        write_npy(&p, sig, data);
        cmd.arg(format!("--input=@{}", p.display()));
    }
    let outs: Vec<PathBuf> = (0..m.results.len())
        .map(|k| dir.join(format!("out{k}.npy")))
        .collect();
    for p in &outs {
        cmd.arg(format!("--output=@{}", p.display()));
    }
    let out = cmd.output().unwrap();
    if !out.status.success() {
        return Err(format!(
            "iree-run-module failed: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .next()
                .unwrap_or("")
        ));
    }

    if outs.len() != want.len() {
        return Err(format!("{} results, rlx has {}", outs.len(), want.len()));
    }
    let got: Vec<Vec<f32>> = outs.iter().map(|p| read_npy(p)).collect();
    let _ = std::fs::remove_dir_all(&dir);
    Ok((got, want))
}

/// Worst relative error of one output, or a description of why it is not
/// comparable.
fn worst(got: &[f32], want: &[f32]) -> Result<(f32, usize), String> {
    if got.len() != want.len() {
        return Err(format!("{} elements, rlx has {}", got.len(), want.len()));
    }
    let mut w = (0.0f32, 0usize);
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        if !a.is_finite() && b.is_finite() {
            return Ok((f32::INFINITY, i));
        }
        let d = (a - b).abs() / (1.0 + b.abs());
        if d > w.0 {
            w = (d, i);
        }
    }
    Ok(w)
}

fn roundtrip(iree: &Iree, name: &str, g: Graph, tol: f32, params: bool) -> Result<(), String> {
    let (got, want) = run_both(iree, name, g.clone(), params)?;
    for (k, (a, b)) in got.iter().zip(&want).enumerate() {
        let (d, i) = worst(a, b).map_err(|e| format!("output {k}: {e}"))?;
        if d > tol {
            return Err(format!(
                "output {k}[{i}]: iree {} vs rlx {} (rel {d:.3e} > {tol:.0e}); {}",
                a[i],
                b[i],
                bisect(iree, name, &g, tol)
            ));
        }
    }
    Ok(())
}

/// Re-run with every float compute node exposed and name the first one, in
/// node order, where IREE and rlx disagree.
fn bisect(iree: &Iree, name: &str, g: &Graph, tol: f32) -> String {
    let mut e = g.clone();
    let ids: Vec<rlx_ir::NodeId> = g
        .nodes()
        .iter()
        .filter(|n| {
            !matches!(
                n.op,
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
            )
        })
        .filter(|n| n.shape.dtype() == DType::F32)
        .map(|n| n.id)
        .collect();
    let mut outs = g.outputs.clone();
    outs.extend(&ids);
    e.set_outputs(outs);
    let (got, want) = match run_both(iree, &format!("{name}-bisect"), e, false) {
        Ok(r) => r,
        Err(err) => return format!("bisect run failed: {err}"),
    };
    let n_out = g.outputs.len();
    for (k, id) in ids.iter().enumerate() {
        if let Ok((d, i)) = worst(&got[n_out + k], &want[n_out + k])
            && d > tol
        {
            let n = g.node(*id);
            let ins: Vec<String> = n
                .inputs
                .iter()
                .map(|i| format!("{:?}", g.node(*i).op.kind()))
                .collect();
            return format!(
                "first divergence: node {id:?} {:?} {:?} (inputs {ins:?}) at [{i}]: iree {} vs rlx {}",
                n.op.kind(),
                n.shape.dims(),
                got[n_out + k][i],
                want[n_out + k][i]
            );
        }
    }
    "every exposed op agrees — a fusion the exposure blocked differs".into()
}

// ── Graphs ──────────────────────────────────────────────────────────────

fn mlp() -> Graph {
    let mut g = Graph::new("mlp");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[4, 16], f));
    let w1 = g.param("w1", Shape::new(&[16, 32], f));
    let b1 = g.param("b1", Shape::new(&[32], f));
    let w2 = g.param("w2", Shape::new(&[32, 8], f));
    let h = g.matmul(x, w1, Shape::new(&[4, 32], f));
    let h = g.binary(BinaryOp::Add, h, b1, Shape::new(&[4, 32], f));
    let h = g.add_node(
        Op::Activation(Activation::Gelu),
        vec![h],
        Shape::new(&[4, 32], f),
    );
    let y = g.matmul(h, w2, Shape::new(&[4, 8], f));
    let y = g.add_node(Op::Softmax { axis: -1 }, vec![y], Shape::new(&[4, 8], f));
    g.set_outputs(vec![y]);
    g
}

/// NCHW conv + bias + ReLU + 2×2 max pool: the layout case.
fn conv_pool() -> Graph {
    let mut g = Graph::new("conv_pool");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[1, 3, 10, 10], f));
    let w = g.param("w", Shape::new(&[8, 3, 3, 3], f));
    let b = g.param("b", Shape::new(&[1, 8, 1, 1], f));
    let s = Shape::new(&[1, 8, 10, 10], f);
    let y = g.add_node(
        Op::Conv {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![1, 1],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![x, w],
        s.clone(),
    );
    let be = g.add_node(
        Op::Expand {
            target_shape: vec![1, 8, 10, 10],
        },
        vec![b],
        s.clone(),
    );
    let y = g.binary(BinaryOp::Add, y, be, s.clone());
    let y = g.add_node(Op::Activation(Activation::Relu), vec![y], s);
    let y = g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![y],
        Shape::new(&[1, 8, 5, 5], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// Embedding gather by integer ids, LayerNorm, a reduction, and shape ops.
fn embed_norm_shapes() -> Graph {
    let mut g = Graph::new("embed_norm");
    let f = DType::F32;
    let table = g.param("table", Shape::new(&[4, 8], f));
    let ids = g.input("ids", Shape::new(&[6], DType::I32));
    let e = g.add_node(
        Op::Gather { axis: 0 },
        vec![table, ids],
        Shape::new(&[6, 8], f),
    );
    let gamma = g.param("gamma", Shape::new(&[8], f));
    let beta = g.param("beta", Shape::new(&[8], f));
    let n = g.add_node(
        Op::LayerNorm { eps: 1e-5, axis: 1 },
        vec![e, gamma, beta],
        Shape::new(&[6, 8], f),
    );
    let t = g.add_node(
        Op::Transpose { perm: vec![1, 0] },
        vec![n],
        Shape::new(&[8, 6], f),
    );
    let a = g.add_node(
        Op::Narrow {
            axis: 1,
            start: 1,
            len: 3,
        },
        vec![t],
        Shape::new(&[8, 3], f),
    );
    let c = g.add_node(Op::Concat { axis: 1 }, vec![a, a], Shape::new(&[8, 6], f));
    let r = g.add_node(
        Op::Reshape {
            new_shape: vec![4, 12],
        },
        vec![c],
        Shape::new(&[4, 12], f),
    );
    let s = g.add_node(
        Op::Reduce {
            op: ReduceOp::Max,
            axes: vec![1],
            keep_dim: false,
        },
        vec![r],
        Shape::new(&[4], f),
    );
    g.set_outputs(vec![r, s]);
    g
}

/// RmsNorm with gamma and beta over the last axis of a rank-3 tensor.
fn rms_norm() -> Graph {
    let mut g = Graph::new("rms_norm");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[1, 8, 64], f));
    let gamma = g.param("gamma", Shape::new(&[64], f));
    let beta = g.param("beta", Shape::new(&[64], f));
    let y = g.add_node(
        Op::RmsNorm { eps: 1e-6, axis: 2 },
        vec![x, gamma, beta],
        Shape::new(&[1, 8, 64], f),
    );
    g.set_outputs(vec![y]);
    g
}

/// Rank-3 activation × rank-2 weight, as every transformer projection is.
fn batched_matmul() -> Graph {
    let mut g = Graph::new("batched_matmul");
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[1, 8, 64], f));
    let w = g.param("w", Shape::new(&[64, 192], f));
    let y = g.matmul(x, w, Shape::new(&[1, 8, 192], f));
    g.set_outputs(vec![y]);
    g
}

/// Int8 weight with per-output-channel scales, dequantized into the matmul:
/// the weight crosses the boundary as `i8`, a quarter of the f32 bytes.
fn int8_weight_matmul() -> Graph {
    let mut g = Graph::new("int8_weight");
    let x = g.input("x", Shape::new(&[4, 16], DType::F32));
    let wq = g.param("wq", Shape::new(&[16, 8], DType::I8));
    let w = g.add_node(
        Op::Dequantize {
            axis: Some(1),
            scales: (0..8).map(|c| 0.002 + 0.001 * c as f32).collect(),
            zero_points: (0..8).map(|c| c - 4).collect(),
        },
        vec![wq],
        Shape::new(&[16, 8], DType::F32),
    );
    let y = g.matmul(x, w, Shape::new(&[4, 8], DType::F32));
    g.set_outputs(vec![y]);
    g
}

/// Static per-tensor quantize → dequantize on an activation (QDQ).
fn int8_qdq_activation() -> Graph {
    let mut g = Graph::new("int8_qdq");
    let x = g.input("x", Shape::new(&[4, 16], DType::F32));
    let q = g.add_node(
        Op::Quantize {
            axis: None,
            scales: vec![0.01],
            zero_points: vec![3],
        },
        vec![x],
        Shape::new(&[4, 16], DType::I8),
    );
    let d = g.add_node(
        Op::Dequantize {
            axis: None,
            scales: vec![0.01],
            zero_points: vec![3],
        },
        vec![q],
        Shape::new(&[4, 16], DType::F32),
    );
    let y = g.add_node(
        Op::Activation(Activation::Tanh),
        vec![d],
        Shape::new(&[4, 16], DType::F32),
    );
    g.set_outputs(vec![y, q]);
    g
}

const D: usize = 64;
const HEADS: usize = 4;
const KV: usize = 2;
const HD: usize = 16;
const FFN: usize = 128;
const SEQ: usize = 8;

fn weights(prefix: &str, entries: &[(&str, Vec<usize>)]) -> MapWeights {
    let mut w = MapWeights::default();
    for (k, shape) in entries {
        let n = shape.iter().product();
        w.insert(format!("{prefix}.{k}"), vec![0.0; n], shape.clone());
    }
    w
}

/// Llama/Qwen-style prefill layer: RMSNorm, RoPE, GQA causal attention, SwiGLU.
fn llama_layer() -> Graph {
    let mut w = weights(
        "model.layers.0",
        &[
            ("input_layernorm.weight", vec![D]),
            ("post_attention_layernorm.weight", vec![D]),
            ("self_attn.q_proj.weight", vec![HEADS * HD, D]),
            ("self_attn.k_proj.weight", vec![KV * HD, D]),
            ("self_attn.v_proj.weight", vec![KV * HD, D]),
            ("self_attn.o_proj.weight", vec![D, HEADS * HD]),
            ("mlp.gate_proj.weight", vec![FFN, D]),
            ("mlp.up_proj.weight", vec![FFN, D]),
            ("mlp.down_proj.weight", vec![D, FFN]),
        ],
    );
    let spec = LlamaDecoderSpec {
        num_heads: HEADS,
        head_dim: HD,
        n_rot: HD,
        num_kv_heads: KV,
        eps: 1e-6,
        mask: MaskKind::Causal,
        hidden_shape: Shape::new(&[1, SEQ, D], DType::F32),
        rope_style: RopeStyle::NeoX,
    };
    let inv: Vec<f64> = (0..HD / 2)
        .map(|i| 1.0 / 10_000f64.powf(2.0 * i as f64 / HD as f64))
        .collect();
    ModelFlow::new("llama")
        .input("x", Shape::new(&[1, SEQ, D], DType::F32))
        .rope_tables(RopeTablesStage::from_inv_freq(SEQ, &inv))
        .zero_beta(D)
        .raw_stage(llama_prefill_layer_composed(0, spec))
        .build(&mut w)
        .expect("llama flow")
        .into_graph()
        .expect("llama graph")
}

/// BERT encoder layer: padding mask, bias + GELU FFN, post-LN.
fn bert_layer() -> Graph {
    let mut entries: Vec<(&str, Vec<usize>)> = Vec::new();
    for (k, s) in [
        ("attention.self.query.weight", vec![D, D]),
        ("attention.self.query.bias", vec![D]),
        ("attention.self.key.weight", vec![D, D]),
        ("attention.self.key.bias", vec![D]),
        ("attention.self.value.weight", vec![D, D]),
        ("attention.self.value.bias", vec![D]),
        ("attention.output.dense.weight", vec![D, D]),
        ("attention.output.dense.bias", vec![D]),
        ("attention.output.LayerNorm.weight", vec![D]),
        ("attention.output.LayerNorm.bias", vec![D]),
        ("intermediate.dense.weight", vec![FFN, D]),
        ("intermediate.dense.bias", vec![FFN]),
        ("output.dense.weight", vec![D, FFN]),
        ("output.dense.bias", vec![D]),
        ("output.LayerNorm.weight", vec![D]),
        ("output.LayerNorm.bias", vec![D]),
    ] {
        entries.push((k, s));
    }
    let prefix = "encoder.layer.0";
    let mut w = weights(prefix, &entries);
    let spec = BertEncoderLayerSpec::hf(prefix, BertQkvStyle::Bert, D, HEADS, 1e-12);
    ModelFlow::new("bert")
        .input("x", Shape::new(&[1, SEQ, D], DType::F32))
        .input("attention_mask", Shape::new(&[1, SEQ], DType::F32))
        .bert_encoder_layer(spec)
        .build(&mut w)
        .expect("bert flow")
        .into_graph()
        .expect("bert graph")
}

#[test]
fn exported_graphs_match_rlx_cpu_under_iree() {
    let Some(iree) = iree() else {
        return;
    };
    let known = KnownFailures::load(env!("CARGO_MANIFEST_DIR"), "iree_roundtrip");
    // (name, graph, tolerance, weights as IREE parameters)
    let cases: Vec<(&str, fn() -> Graph, f32, bool)> = vec![
        ("mlp", mlp, 1e-4, false),
        ("conv_pool", conv_pool, 1e-4, false),
        ("embed_norm_shapes", embed_norm_shapes, 1e-4, false),
        ("rms_norm", rms_norm, 1e-4, false),
        ("batched_matmul", batched_matmul, 1e-4, false),
        ("int8_weight_matmul", int8_weight_matmul, 1e-4, false),
        ("int8_qdq_activation", int8_qdq_activation, 1e-4, false),
        ("llama_prefill_layer", llama_layer, 1e-3, false),
        ("bert_encoder_layer", bert_layer, 1e-3, false),
        // Weights from a safetensors parameter file instead of arguments.
        ("mlp_parameters", mlp, 1e-4, true),
        ("int8_weight_parameters", int8_weight_matmul, 1e-4, true),
        ("llama_prefill_layer_parameters", llama_layer, 1e-3, true),
    ];
    let mut failures = Vec::new();
    for (name, build, tol, params) in cases {
        if !known.selected(name, "iree") {
            continue;
        }
        let r = roundtrip(&iree, name, build(), tol, params);
        eprintln!(
            "iree_roundtrip {name}: {}",
            r.as_ref()
                .map(|_| "ok".to_string())
                .unwrap_or_else(|e| e.clone())
        );
        if let Some(msg) = known.record(name, "iree", &r) {
            failures.push(msg);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The export of an NCHW convolution carries its layout in the dimension
/// numbers rather than in transposes.
#[test]
fn nchw_conv_exports_without_transposes() {
    let m = export(conv_pool(), &StableHloOptions::default()).expect("export");
    assert!(
        m.text
            .contains("#stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>"),
        "{}",
        m.text
    );
    assert!(!m.text.contains("stablehlo.transpose"), "{}", m.text);
    assert_eq!(m.args[0].name, "x");
    assert_eq!(m.args[0].shape, vec![1, 3, 10, 10]);
}

/// One symbolic graph exported at two sequence lengths: two functions, one
/// shared set of parameters, each runs under IREE.
#[test]
fn specializations_share_parameters() {
    use rlx_ir::dynamic::sym;
    use rlx_ir::{Dim, DimBinding};
    let Some(iree) = iree() else {
        return;
    };
    let mut g = Graph::new("sym_mlp");
    let s = |last: usize| {
        Shape::from_dims(
            &[Dim::Static(1), Dim::Dynamic(sym::SEQ), Dim::Static(last)],
            DType::F32,
        )
    };
    let x = g.input("x", s(16));
    let w = g.param("w", Shape::new(&[16, 8], DType::F32));
    let y = g.matmul(x, w, s(8));
    let y = g.add_node(Op::Activation(Activation::Tanh), vec![y], s(8));
    g.set_outputs(vec![y]);

    let specs: Vec<(String, DimBinding)> = [4usize, 9]
        .iter()
        .map(|&n| (format!("seq{n}"), DimBinding::from_pairs(&[(sym::SEQ, n)])))
        .collect();
    let opts = StableHloOptions::default().with_parameters("model");
    let m = rlx_stablehlo::export_specializations(g, &specs, &opts).expect("export");
    assert_eq!(m.functions.len(), 2);
    assert_eq!(
        m.text.matches("util.global private").count(),
        1,
        "weights declared once"
    );

    let dir = std::env::temp_dir().join(format!("rlx-stablehlo-spec-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (mlir, vmfb, weights) = (
        dir.join("m.mlir"),
        dir.join("m.vmfb"),
        dir.join("w.safetensors"),
    );
    std::fs::write(&mlir, &m.text).unwrap();
    let wv: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();
    let wb: Vec<u8> = wv.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_safetensors(
        &weights,
        &[SafetensorsEntry {
            name: "w",
            dtype: DType::F32,
            shape: &[16, 8],
            data: &wb,
        }],
    )
    .unwrap();
    let ok = Command::new(&iree.compile)
        .args([
            "--iree-input-type=stablehlo",
            "--iree-hal-target-device=local",
            "--iree-hal-local-target-device-backends=llvm-cpu",
        ])
        .arg(&mlir)
        .arg("-o")
        .arg(&vmfb)
        .status()
        .unwrap();
    assert!(ok.success(), "iree-compile failed on {}", mlir.display());
    for f in &m.functions {
        let n = f.args[0].shape[1] as usize;
        let xv: Vec<f32> = (0..16 * n).map(|i| (i as f32 * 0.13).cos()).collect();
        let xin = dir.join(format!("{}.npy", f.name));
        write_npy(&xin, &f.args[0], &xv);
        let out = dir.join(format!("{}_out.npy", f.name));
        let st = Command::new(&iree.run)
            .arg(format!("--module={}", vmfb.display()))
            .arg("--device=local-task")
            .arg(format!("--function={}", f.name))
            .arg(format!("--parameters=model={}", weights.display()))
            .arg(format!("--input=@{}", xin.display()))
            .arg(format!("--output=@{}", out.display()))
            .status()
            .unwrap();
        assert!(st.success(), "{} failed", f.name);
        let got = read_npy(&out);
        // Reference: tanh(x · w).
        for r in 0..n {
            for c in 0..8 {
                let want: f32 = (0..16)
                    .map(|k| xv[r * 16 + k] * wv[k * 8 + c])
                    .sum::<f32>()
                    .tanh();
                let d = (got[r * 8 + c] - want).abs();
                assert!(
                    d < 1e-5,
                    "{}[{r},{c}]: {} vs {want}",
                    f.name,
                    got[r * 8 + c]
                );
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
