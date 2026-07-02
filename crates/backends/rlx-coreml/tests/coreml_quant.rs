// Quantized-weight ops, verified on-device. GGUF weights are stored
// `[N, K]` (B-transposed); the backend host-dequantizes them to f32 and
// matmuls with transpose_y. We quantize a known f32 weight, run through
// CoreML, and compare to the full-precision matmul within Q8_0 tolerance.
#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_coreml::CoremlExecutable;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, Op, Shape};

fn approx(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    let mx = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        mx <= tol,
        "max abs diff {mx} > {tol}\n got {a:?}\n ref {b:?}"
    );
}

// row-major [M,K] @ [K,N]
fn matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                acc += x[i * k + kk] * w[kk * n + j];
            }
            o[i * n + j] = acc;
        }
    }
    o
}

#[test]
fn dequant_matmul_q8_0() {
    // M=2, K=64 (Q8_0 block = 32, so K divisible), N=3.
    let (m, k, n) = (2usize, 64usize, 3usize);

    // Reference weight in [K,N] f32; GGUF stores it transposed [N,K].
    let w_kn: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();
    let mut w_nk = vec![0.0f32; n * k]; // [N,K] row-major
    for kk in 0..k {
        for j in 0..n {
            w_nk[j * k + kk] = w_kn[kk * n + j];
        }
    }
    // Quantize the [N,K] weight (each of the N rows is a length-K vector).
    let packed = rlx_gguf::quantize::quantize_q8_0(&w_nk).expect("quantize");

    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.02).cos()).collect();

    let mut g = Graph::new("dqmm");
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("W", Shape::new(&[n, k], DType::F32)); // logical [N,K]
    let y = g.append_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ8_0,
        },
        vec![xi, w],
        Shape::new(&[m, n], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);

    let mut e = CoremlExecutable::compile(g);
    e.set_param_typed("W", &packed, DType::U8);
    let out = e.run(&[("x", &x)]).expect("run").remove(0);

    // Reference: full-precision x @ w_kn.
    let want = matmul(&x, &w_kn, m, k, n);
    // Q8_0 ~ 8-bit; allow for quant error scaled by K.
    approx(&out, &want, 5e-2);
}

#[test]
fn dequant_matmul_q4_0() {
    // Q4_0 block = 32 → K multiple of 32. Exercises a second scheme.
    let (m, k, n) = (1usize, 64usize, 2usize);
    let w_kn: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.01).sin() * 0.3)
        .collect();
    let mut w_nk = vec![0.0f32; n * k];
    for kk in 0..k {
        for j in 0..n {
            w_nk[j * k + kk] = w_kn[kk * n + j];
        }
    }
    let packed = rlx_gguf::quantize::quantize_q4_0(&w_nk).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.003).cos()).collect();

    let mut g = Graph::new("dqmm_q40");
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("W", Shape::new(&[n, k], DType::F32));
    let y = g.append_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ4_0,
        },
        vec![xi, w],
        Shape::new(&[m, n], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);

    let mut e = CoremlExecutable::compile(g);
    e.set_param_typed("W", &packed, DType::U8);
    let out = e.run(&[("x", &x)]).expect("run").remove(0);

    let want = matmul(&x, &w_kn, m, k, n);
    approx(&out, &want, 1e-1); // Q4_0 is ~4.5 bpw, looser tolerance
}

#[test]
fn dequant_grouped_matmul_q8_0() {
    // E=2 experts, M=3 tokens, K=64, N=2.
    let (e_n, m, k, n) = (2usize, 3usize, 64usize, 2usize);
    // per-expert weight stored [N,K]; reference keeps [K,N] for matmul.
    let mut packed = Vec::new();
    let mut w_kn_per = Vec::new();
    for e in 0..e_n {
        let w_kn: Vec<f32> = (0..k * n)
            .map(|i| (((e * 1000 + i) as f32) * 0.011).sin() * 0.4)
            .collect();
        let mut w_nk = vec![0.0f32; n * k];
        for kk in 0..k {
            for j in 0..n {
                w_nk[j * k + kk] = w_kn[kk * n + j];
            }
        }
        packed.extend(rlx_gguf::quantize::quantize_q8_0(&w_nk).expect("q"));
        w_kn_per.push(w_kn);
    }
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.02).cos()).collect();
    let experts = [0.0f32, 1.0, 0.0];

    let mut g = Graph::new("dqgmm");
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("W", Shape::new(&[e_n, n, k], DType::F32));
    let ei = g.input("e", Shape::new(&[m], DType::F32));
    let y = g.append_node(
        Op::DequantGroupedMatMul {
            scheme: QuantScheme::GgufQ8_0,
        },
        vec![xi, w, ei],
        Shape::new(&[m, n], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);

    let mut exe = CoremlExecutable::compile(g);
    exe.set_param_typed("W", &packed, DType::U8);
    let out = exe
        .run(&[("x", &x), ("e", &experts)])
        .expect("run")
        .remove(0);

    let mut want = vec![0.0f32; m * n];
    for t in 0..m {
        let e = experts[t] as usize;
        let part = matmul(&x[t * k..(t + 1) * k], &w_kn_per[e], 1, k, n);
        want[t * n..(t + 1) * n].copy_from_slice(&part);
    }
    approx(&out, &want, 5e-2);
}

#[test]
fn dequant_matmul_through_session() {
    // Quantized weights via the public Session / set_param_typed path
    // (CompiledGraph forwarding), not CoremlExecutable directly.
    use rlx_runtime::{Device, Session};
    let (m, k, n) = (2usize, 64usize, 3usize);
    let w_kn: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();
    let mut w_nk = vec![0.0f32; n * k];
    for kk in 0..k {
        for j in 0..n {
            w_nk[j * k + kk] = w_kn[kk * n + j];
        }
    }
    let packed = rlx_gguf::quantize::quantize_q8_0(&w_nk).expect("quantize");
    let x: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.02).cos()).collect();

    let mut g = Graph::new("dqmm_session");
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("W", Shape::new(&[n, k], DType::F32));
    let y = g.append_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ8_0,
        },
        vec![xi, w],
        Shape::new(&[m, n], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);

    let mut compiled = Session::new(Device::Ane).compile(g);
    compiled.set_param_typed("W", &packed, DType::U8);
    let out = compiled.run(&[("x", &x)]).remove(0);

    approx(&out, &matmul(&x, &w_kn, m, k, n), 5e-2);
}

#[test]
fn quantize_dequantize_roundtrip() {
    // Per-tensor int8 fake-quant: x -> quantize -> dequantize -> ~x.
    let scale = 0.1f32;
    let zp = 0i32;
    let n = 6usize;
    let mut g = Graph::new("fakequant");
    let x = g.input("x", Shape::new(&[n], DType::F32));
    let q = g.append_node(
        Op::Quantize {
            axis: None,
            scales: vec![scale],
            zero_points: vec![zp],
        },
        vec![x],
        Shape::new(&[n], DType::I8),
        None,
    );
    let y = g.append_node(
        Op::Dequantize {
            axis: None,
            scales: vec![scale],
            zero_points: vec![zp],
        },
        vec![q],
        Shape::new(&[n], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);

    // Values chosen off the .5 boundary so round-half convention is moot.
    let xs = [0.07f32, 0.23, -0.41, 1.04, -0.77, 0.34];
    let mut e = CoremlExecutable::compile(g);
    let out = e.run(&[("x", &xs)]).expect("run").remove(0);

    // Reference: round(x/scale)+zp clamped, then (q-zp)*scale.
    let want: Vec<f32> = xs
        .iter()
        .map(|&v| {
            let q = ((v / scale).round() + zp as f32).clamp(-128.0, 127.0);
            (q - zp as f32) * scale
        })
        .collect();
    approx(&out, &want, 1e-5);
}

#[test]
fn dequant_moe_weights_q8_0() {
    // Dequantize a packed weight back to f32 (no matmul).
    let n = 64usize; // one Q8_0-friendly block multiple
    let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).sin()).collect();
    let packed = rlx_gguf::quantize::quantize_q8_0(&w).expect("quantize");

    let mut g = Graph::new("dqmoe");
    let wp = g.param("W", Shape::new(&[n], DType::F32));
    let dq = g.append_node(
        Op::DequantMoEWeights {
            scheme: QuantScheme::GgufQ8_0,
        },
        vec![wp],
        Shape::new(&[n], DType::F32),
        None,
    );
    // CoreML (spec < v9) requires ≥1 input; add a zero bias to satisfy it.
    let bias = g.input("bias", Shape::new(&[n], DType::F32));
    let y = g.binary(
        rlx_ir::op::BinaryOp::Add,
        dq,
        bias,
        Shape::new(&[n], DType::F32),
    );
    g.set_outputs(vec![y]);

    let mut e = CoremlExecutable::compile(g);
    e.set_param_typed("W", &packed, DType::U8);
    let out = e.run(&[("bias", &vec![0.0f32; n])]).expect("run").remove(0);

    let want = rlx_gguf::dequant_q8_0(&packed, n).unwrap();
    approx(&out, &want, 1e-5);
}
