// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validate the CUDA `mxfp4x2_decode` kernel (in `scaled_lowp_general.cu`)
//! against the CPU oracle (`rlx_ir::residual::residual_dequantize`) — the
//! two-level residual FP4 decode `out = s0·LUT[q0] + s1·LUT[q1]`. Runs on a
//! real CUDA GPU (the RTX 3080 Ti rig); skips cleanly with no device.

use rlx_ir::quant::ScaledFormat;
use rlx_ir::residual::{residual_dequantize, residual_quantize};

#[test]
fn mxfp4x2_cuda_decode_matches_cpu_oracle() {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skip: no CUDA device");
            return;
        }
    };
    let stream = ctx.default_stream();
    let ptx = cudarc::nvrtc::compile_ptx(rlx_gpu_kernels::SCALED_LOWP_GENERAL_CU)
        .expect("nvrtc compile scaled_lowp_general");
    let module = ctx.load_module(ptx).expect("load module");
    let func = module.load_function("mxfp4x2_decode").expect("mxfp4x2_decode");

    // Quantize deterministic data to 2 residual levels, per MX group.
    let fmt = ScaledFormat::F4E2M1;
    let group = 32usize;
    let nblocks = 6usize;
    let n = group * nblocks;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.137).sin() * 3.7).collect();
    let (mut q0, mut q1, mut s0, mut s1, mut cpu) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for b in 0..nblocks {
        let rb = residual_quantize(&data[b * group..(b + 1) * group], fmt, 2);
        s0.push(rb.scales[0]);
        s1.push(rb.scales[1]);
        q0.extend_from_slice(&rb.codes[0]);
        q1.extend_from_slice(&rb.codes[1]);
        cpu.extend(residual_dequantize(&rb));
    }

    let q0_d = stream.clone_htod(&q0).unwrap();
    let q1_d = stream.clone_htod(&q1).unwrap();
    let s0_d = stream.clone_htod(&s0).unwrap();
    let s1_d = stream.clone_htod(&s1).unwrap();
    let mut out_d = stream.alloc_zeros::<f32>(n).unwrap();
    let nn = n as u32;
    let gg = group as u32;

    let mut lb = stream.launch_builder(&func);
    lb.arg(&q0_d)
        .arg(&q1_d)
        .arg(&s0_d)
        .arg(&s1_d)
        .arg(&mut out_d)
        .arg(&nn)
        .arg(&gg);
    unsafe { lb.launch(LaunchConfig::for_num_elems(nn)).unwrap() };

    let gpu = stream.clone_dtoh(&out_d).unwrap();
    let worst = cpu
        .iter()
        .zip(&gpu)
        .map(|(&c, &g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    eprintln!("MxFp4x2 CUDA decode: worst |gpu-cpu| = {worst:.2e}  (n={n})");
    assert!(worst < 1e-6, "CUDA decode must match CPU oracle, worst={worst:e}");
}
