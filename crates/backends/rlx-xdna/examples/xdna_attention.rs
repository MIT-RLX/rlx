// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Fused scaled-dot-product attention — out = softmax(Q·Kᵀ/√d)·V — emitted from
// Rust as ONE AIE-MLIR core (scalar f32 dot-products + pure-arith softmax over a
// local scratch buffer), compiled by aiecc, run on the NPU via the 3-buffer
// NpuRun3 (Q, K‖V packed, out), and checked vs a CPU reference.
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. SEQ=32 D=32 \
//     cargo run -p rlx-xdna --features xrt --example xdna_attention

use rlx_xdna::aie::emit_attention;
use rlx_xdna::compile::{compile_overlay, OverlaySpec};
use rlx_xdna::npu_gemm::NpuRun3;

fn envn(k: &str, d: &str) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| d.parse().unwrap())
}

fn main() {
    let (seq, d) = (envn("SEQ", "32"), envn("D", "32"));
    let sd = seq * d;
    let (aiecc, peano) = (std::env::var("AIECC").unwrap(), std::env::var("PEANO").unwrap());

    let mlir = emit_attention(seq, d, 1, 1.0 / (d as f32).sqrt(), false);
    println!("1. rlx emitted AIE-MLIR attention seq={seq} d={d} ({} lines)", mlir.lines().count());
    let tmp = "/tmp/rlx_attn";
    std::fs::create_dir_all(tmp).ok();
    let mp = format!("{tmp}/aie.mlir");
    std::fs::write(&mp, &mlir).unwrap();
    let xclbin = format!("{tmp}/k.xclbin");
    let insts_path = format!("{tmp}/insts.bin");
    compile_overlay(&OverlaySpec {
        aiecc: &aiecc,
        peano: &peano,
        mlir: &mp,
        tmpdir: &format!("{tmp}/build"),
        out_xclbin: &xclbin,
        out_insts: &insts_path,
    })
    .expect("compile");
    println!("2. compiled");
    let insts: Vec<u32> = std::fs::read(&insts_path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Q, K, V in [seq, d]; small deterministic values.
    let q: Vec<f32> = (0..sd).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let k: Vec<f32> = (0..sd).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    let v: Vec<f32> = (0..sd).map(|i| ((i % 7) as f32 - 3.0) * 0.15).collect();
    let mut kv = k.clone();
    kv.extend_from_slice(&v); // packed: K then V

    let io = NpuRun3::open("", &xclbin, &insts, sd, 2 * sd, sd).expect("open");
    let out = io.run(&q, &kv).expect("run");

    // CPU reference: softmax(Q·Kᵀ/√d)·V.
    let scale = 1.0 / (d as f32).sqrt();
    let mut cref = vec![0f32; sd];
    for i in 0..seq {
        let mut scores = vec![0f32; seq];
        let mut m = f32::NEG_INFINITY;
        for j in 0..seq {
            let mut dot = 0.0;
            for kk in 0..d {
                dot += q[i * d + kk] * k[j * d + kk];
            }
            scores[j] = dot * scale;
            m = m.max(scores[j]);
        }
        let mut s = 0.0;
        for j in 0..seq {
            scores[j] = (scores[j] - m).exp();
            s += scores[j];
        }
        for kk in 0..d {
            let mut acc = 0.0;
            for j in 0..seq {
                acc += scores[j] * v[j * d + kk];
            }
            cref[i * d + kk] = acc / s;
        }
    }

    let mut maxrel = 0.0f32;
    for i in 0..sd {
        maxrel = maxrel.max((out[i] - cref[i]).abs() / cref[i].abs().max(1e-3));
    }
    if maxrel.is_nan() || maxrel > 3e-3 {
        println!("3. NPU attention seq={seq} d={d}: FAIL ✗  max-rel-err {maxrel:.2e}");
        std::process::exit(1);
    }
    println!("3. ran on NPU: PASS ✓  max-rel-err {maxrel:.2e} (attention seq={seq} d={d})");
}
