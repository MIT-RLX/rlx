// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// NPU ATTENTION-PROJECTION milestone: run a DeepSeek-V4-scale dense projection
// (M×K @ K×N) on the XDNA NPU as an int8 GEMM, comparing to the f32 reference.
// This is the "wire the NPU into attention" test — it exercises the real path a
// Device::Xdna attention offload would use:
//   1. f32 activation [M,K] and weight [K,N] (representative distribution),
//   2. per-tensor act scale + per-output-channel weight scale → int8,
//   3. int8 GEMM on the NPU (tiled over N in cols*DIM chunks; K accumulated),
//   4. dequant C_i32 * sa * sw[j], compared to the f32 matmul (rel-err + cosine).
// Reports NPU throughput. int8 error here is the accuracy cost of an NPU offload.
//
//   AIECC=.. PEANO=.. RLX_XDNA_AIE_INCLUDE=.. RLX_XDNA_SHIM=.. \
//     M=64 K=4096 N=1024 cargo run --release -p rlx-xdna --features xrt \
//     --example xdna_attention_proj
use rlx_xdna::aie::{emit_matmul_microkernel, tile_a_kacc, tile_b_kacc_multicol, untile_c_multicol};
use rlx_xdna::compile::{build_mm_kernel, compile_overlay_linked, OverlaySpec};
use rlx_xdna::npu_gemm::NpuGemm;
use std::time::Instant;

fn env(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

// Deterministic ~Gaussian (sum of 3 uniforms) so the int8 quant error reflects a
// realistic weight/activation distribution, not a uniform one.
fn gauss(state: &mut u64) -> f32 {
    let mut s = 0f32;
    for _ in 0..3 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s += (((*state >> 40) as u32) as f32 / (1u32 << 24) as f32) - 0.5;
    }
    s / 1.5 // ~unit-ish variance, range ~[-1,1]
}

fn main() {
    // DeepSeek-V4 q_a_proj scale by default (M=seq padded to the 64 tile).
    let (m, k, n) = (env("M", 64), env("K", 4096), env("N", 1024));
    let d = 64usize;
    assert!(m == d, "this single-tile overlay uses M=DIM=64 (pad seq up)");
    assert!(k % d == 0 && n % d == 0, "K,N must be multiples of DIM=64");
    let kt = k / d;
    let cols = 4usize; // AIE columns per pass (array has ~4-5 usable)
    let n_tile = cols * d; // 256
    assert!(n % n_tile == 0, "N must be a multiple of {n_tile}");
    let tiles = n / n_tile;

    let aiecc = std::env::var("AIECC").expect("set AIECC");
    let peano = std::env::var("PEANO").expect("set PEANO");
    let include = std::env::var("RLX_XDNA_AIE_INCLUDE").expect("set RLX_XDNA_AIE_INCLUDE");

    // 1) Representative f32 activation + weight (attention-scale std ~ 1/sqrt(K)).
    let wstd = 1.0 / (k as f32).sqrt();
    let mut st = 0x1234_5678_9abc_def0u64;
    let af: Vec<f32> = (0..m * k).map(|_| gauss(&mut st) * 0.5).collect();
    let wf: Vec<f32> = (0..k * n).map(|_| gauss(&mut st) * wstd * 8.0).collect();

    // f32 reference: C_ref[i,j] = sum_l af[i,l]*wf[l,j].
    let mut cref = vec![0f32; m * n];
    for i in 0..m {
        for l in 0..k {
            let a = af[i * k + l];
            for j in 0..n {
                cref[i * n + j] += a * wf[l * n + j];
            }
        }
    }

    // 2) int8 quant: per-tensor act scale, per-output-channel weight scale.
    let amax = af.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1e-8);
    let sa = amax / 127.0;
    let aq: Vec<i8> = af.iter().map(|&x| (x / sa).round().clamp(-127.0, 127.0) as i8).collect();
    let mut sw = vec![0f32; n];
    for j in 0..n {
        let mut mx = 1e-8f32;
        for l in 0..k {
            mx = mx.max(wf[l * n + j].abs());
        }
        sw[j] = mx / 127.0;
    }
    let wq: Vec<i8> = (0..k * n)
        .map(|idx| {
            let j = idx % n;
            (wf[idx] / sw[j]).round().clamp(-127.0, 127.0) as i8
        })
        .collect();

    // 3) Compile the overlay once (m=64, k, n_tile=256), run each N-tile on the NPU.
    let tmp = format!("/tmp/rlx_attn_{d}_{kt}_{cols}");
    std::fs::create_dir_all(&tmp).ok();
    let kernel_o = format!("{tmp}/mm_{d}.o");
    build_mm_kernel(&format!("{peano}/bin/clang++"), &include, d, &kernel_o).expect("build kernel .o");
    let obj_base = format!("mm_{d}.o");
    std::fs::write(format!("{tmp}/aie.mlir"), emit_matmul_microkernel(d, kt, cols, &obj_base)).unwrap();
    compile_overlay_linked(
        &OverlaySpec {
            aiecc: &aiecc,
            peano: &peano,
            mlir: &format!("{tmp}/aie.mlir"),
            tmpdir: &format!("{tmp}/build"),
            out_xclbin: &format!("{tmp}/k.xclbin"),
            out_insts: &format!("{tmp}/i.bin"),
        },
        &[&kernel_o],
    )
    .expect("compile+link");
    let insts: Vec<u32> = std::fs::read(format!("{tmp}/i.bin"))
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mm = NpuGemm::open("", &format!("{tmp}/k.xclbin"), &insts, m, kt * d, n_tile).expect("open");

    let at = tile_a_kacc(&aq, d, kt);
    // Slice weight columns per N-tile ([K, n_tile]) and run on the NPU.
    let mut c_i32 = vec![0i32; m * n];
    for t in 0..tiles {
        let j0 = t * n_tile;
        let mut wtile = vec![0i8; k * n_tile];
        for l in 0..k {
            wtile[l * n_tile..(l + 1) * n_tile].copy_from_slice(&wq[l * n + j0..l * n + j0 + n_tile]);
        }
        let bt = tile_b_kacc_multicol(&wtile, d, kt, cols);
        let ct = mm.run(&at, &bt).expect("run");
        let ctile = untile_c_multicol(&ct, m, n_tile, cols);
        for i in 0..m {
            c_i32[i * n + j0..i * n + j0 + n_tile].copy_from_slice(&ctile[i * n_tile..(i + 1) * n_tile]);
        }
    }

    // 4) Dequant + accuracy vs f32.
    let mut num = 0f64;
    let mut den_a = 0f64;
    let mut den_b = 0f64;
    let mut abs_err = 0f64;
    let mut abs_ref = 0f64;
    for i in 0..m {
        for j in 0..n {
            let got = c_i32[i * n + j] as f32 * sa * sw[j];
            let want = cref[i * n + j];
            num += (got as f64) * (want as f64);
            den_a += (got as f64) * (got as f64);
            den_b += (want as f64) * (want as f64);
            abs_err += (got - want).abs() as f64;
            abs_ref += want.abs() as f64;
        }
    }
    let cosine = num / (den_a.sqrt() * den_b.sqrt() + 1e-12);
    let rel_l1 = abs_err / (abs_ref + 1e-12);

    // Throughput of one full projection (all N-tiles).
    let iters = 30;
    let mut best = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        for _ in 0..tiles {
            let _ = mm.run(&at, &tile_b_kacc_multicol(&wq[..k * n_tile], d, kt, cols));
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e6);
    }
    let gops = 2.0 * (m * k * n) as f64 / (best * 1e3);
    println!(
        "NPU attention proj {m}x{k}x{n} ({tiles} N-tiles): cosine={cosine:.5} rel_L1_err={:.3}% | {best:.1} us → {gops:.1} GOP/s",
        rel_l1 * 100.0
    );
}
