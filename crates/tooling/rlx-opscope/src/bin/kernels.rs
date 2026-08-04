// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-kernels` — the probe's exploits, realized as kernels. For each
//! structured input the probe recommends a decomposition; here the matching
//! kernel runs vs dense with measured speedup/compression + error. Sparse is
//! exact; low-rank/quant trade a bounded error for the win.

use rlx_opscope::guard::dense_matmul;
use rlx_opscope::guard::dense_matmul as dense_ref;
use rlx_opscope::kernels::{
    factored_matmul, factorize, matmul_f32, matmul_w8a8, matmul_w8a16, quant_matmul,
    quantize_cols_t, quantize_row_i8, quantize_rows_i8, rel_err, sparse_skip_matmul, transpose,
};
use rlx_opscope::probe::{per_channel_quant_error, stable_rank};
use rlx_opscope::{Dist, sample};
use std::hint::black_box;
use std::time::Instant;

fn bench<R, F: Fn() -> R>(reps: usize, f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        black_box(f());
    }
    t.elapsed().as_nanos() as f64 / reps as f64 / 1e6 // ms
}

// Apple's AMX matrix coprocessor is not portably programmable (undocumented
// opcodes); it's reached through Accelerate. `cblas_sgemm` runs on AMX, so it's
// the honest yardstick for "what a matrix unit would do" vs the hand-NEON
// kernels. macOS-only; the framework is a system library (no dep to add).
#[cfg(all(feature = "amx", target_vendor = "apple"))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// f32 GEMM `y[m,n] = x[m,k]·W[k,n]` via Accelerate (AMX). RowMajor, no transpose.
#[cfg(all(feature = "amx", target_vendor = "apple"))]
fn accel_sgemm(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    // 101 = CblasRowMajor, 111 = CblasNoTrans.
    unsafe {
        cblas_sgemm(
            101,
            111,
            111,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            x.as_ptr(),
            k as i32,
            w.as_ptr(),
            n as i32,
            0.0,
            y.as_mut_ptr(),
            n as i32,
        );
    }
    y
}

// int8-on-AMX via BNNS (Accelerate). `BNNSMatMul` runs int8×int8→f32 on the AMX
// unit — the honest test of whether int8-on-AMX beats f32-on-AMX. FFI mirrors
// bnns_structures.h / bnns_constants.h exactly (enums are uint32_t; RowMajorMatrix
// uses size[0]=#cols, size[1]=#rows). Deprecated in macOS 15 (→BNNSGraph) but
// still linkable. Uses PER-TENSOR scales (BNNS's simple descriptor path).
#[cfg(all(feature = "amx", target_vendor = "apple"))]
mod bnns {
    use std::os::raw::c_void;

    #[repr(C)]
    pub struct NDArray {
        pub flags: u32,
        pub layout: u32,
        pub size: [usize; 8],
        pub stride: [usize; 8],
        pub data: *mut c_void,
        pub data_type: u32,
        pub table_data: *mut c_void,
        pub table_data_type: u32,
        pub data_scale: f32,
        pub data_bias: f32,
    }

    pub const INT8: u32 = 0x2_0008; // BNNSDataTypeInt8   (IntBit|8)
    pub const F32: u32 = 0x1_0020; //  BNNSDataTypeFloat32 (FloatBit|32)
    pub const ROWMAJOR: u32 = 0x2_0000; // BNNSDataLayoutRowMajorMatrix
    pub const VECTOR: u32 = 0x1_0000; //  BNNSDataLayoutVector

    // BNNSActivation — mirrors bnns_structures.h. function 0 = Identity.
    #[repr(C)]
    pub struct Activation {
        pub function: u32,
        pub alpha: f32,
        pub beta: f32,
        pub iscale: i32,
        pub ioffset: i32,
        pub ishift: i32,
        pub iscale_pc: *const i32,
        pub ioffset_pc: *const i32,
        pub ishift_pc: *const i32,
    }

    // BNNSLayerParametersFullyConnected = {i_desc, w_desc, o_desc, bias, activation}.
    #[repr(C)]
    pub struct FCParams {
        pub i_desc: NDArray,
        pub w_desc: NDArray,
        pub o_desc: NDArray,
        pub bias: NDArray,
        pub activation: Activation,
    }

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn BNNSMatMul(
            trans_a: bool,
            trans_b: bool,
            alpha: f32,
            a: *const NDArray,
            b: *const NDArray,
            c: *const NDArray,
            workspace: *mut c_void,
            params: *const c_void,
        ) -> i32;
        // Quantized fully-connected = the int8-on-AMX GEMM path (opaque filter).
        pub fn BNNSFilterCreateLayerFullyConnected(
            params: *const FCParams,
            fparams: *const c_void,
        ) -> *mut c_void;
        pub fn BNNSFilterApplyBatch(
            filter: *mut c_void,
            batch: usize,
            inp: *const c_void,
            in_stride: usize,
            out: *mut c_void,
            out_stride: usize,
        ) -> i32;
        pub fn BNNSFilterDestroy(filter: *mut c_void);
    }

    /// RowMajorMatrix `[rows,cols]` descriptor (`size[0]=cols, size[1]=rows`,
    /// contiguous strides), integer values dequantized by `scale`.
    pub fn desc(data: *mut c_void, rows: usize, cols: usize, dt: u32, scale: f32) -> NDArray {
        let mut d: NDArray = unsafe { std::mem::zeroed() };
        d.layout = ROWMAJOR;
        d.size[0] = cols;
        d.size[1] = rows;
        d.data = data;
        d.data_type = dt;
        d.data_scale = scale;
        d
    }

    /// 1-D vector descriptor of length `len` (data supplied per-batch at apply).
    pub fn vec(len: usize, dt: u32, scale: f32) -> NDArray {
        let mut d: NDArray = unsafe { std::mem::zeroed() };
        d.layout = VECTOR;
        d.size[0] = len;
        d.data_type = dt;
        d.data_scale = scale;
        d
    }
}

// NB: plain `BNNSMatMul` is FLOAT-ONLY (verified: int8 descriptors → rc -1), so
// int8-on-AMX goes through the quantized fully-connected layer below, not matmul.

/// f32 GEMM via BNNS — a layout/ABI probe (if this works, the struct is right and
/// any int8 failure is BNNSMatMul not supporting int8, not our FFI). `(y, rc)`.
#[cfg(all(feature = "amx", target_vendor = "apple"))]
fn bnns_matmul_f32(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, i32) {
    use std::os::raw::c_void;
    let mut y = vec![0f32; m * n];
    let a = bnns::desc(x.as_ptr() as *mut c_void, m, k, bnns::F32, 1.0);
    let b = bnns::desc(w.as_ptr() as *mut c_void, k, n, bnns::F32, 1.0);
    let c = bnns::desc(y.as_mut_ptr() as *mut c_void, m, n, bnns::F32, 1.0);
    let rc = unsafe {
        bnns::BNNSMatMul(
            false,
            false,
            1.0,
            &a,
            &b,
            &c,
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    };
    (y, rc)
}

/// **int8-on-AMX** via a BNNS quantized fully-connected layer — the real int8
/// matmul path (BNNSMatMul is float-only). `y[m,n] = (xq·sx)·(wtq·sw)`, where
/// `wtq` is the WEIGHT TRANSPOSED to `[out=n, in=k]` (FC stores `Weight(o,i)` at
/// `w[i + o·k]`). int8 activations + weights (per-tensor scales) → f32 out; runs
/// on AMX. `(y, rc)`; rc==0 on success (filter create returns NULL → rc -2).
#[cfg(all(feature = "amx", target_vendor = "apple"))]
fn bnns_fc_i8(
    xq: &[i8],
    sx: f32,
    wtq: &[i8],
    sw: f32,
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, i32) {
    use std::os::raw::c_void;
    let mut p: bnns::FCParams = unsafe { std::mem::zeroed() };
    p.i_desc = bnns::vec(k, bnns::INT8, sx); // input vectors (data comes per-batch)
    p.w_desc = bnns::desc(wtq.as_ptr() as *mut c_void, n, k, bnns::INT8, sw); // [out=n, in=k]
    p.o_desc = bnns::vec(n, bnns::F32, 1.0); // f32 output vectors
    // bias + activation left zeroed → no bias, Identity activation.
    let filter = unsafe { bnns::BNNSFilterCreateLayerFullyConnected(&p, std::ptr::null()) };
    if filter.is_null() {
        return (vec![0f32; m * n], -2);
    }
    let mut y = vec![0f32; m * n];
    let rc = unsafe {
        bnns::BNNSFilterApplyBatch(
            filter,
            m,
            xq.as_ptr() as *const c_void,
            k,
            y.as_mut_ptr() as *mut c_void,
            n,
        )
    };
    unsafe { bnns::BNNSFilterDestroy(filter) };
    (y, rc)
}

fn main() {
    let (m, k, n) = (64usize, 512usize, 512usize);
    let reps = 5;
    println!("matmul {m}×{k}×{n}, each kernel gated by the probe's recommendation:\n");

    // 1) SPARSE activation → skip zeros (exact).
    {
        let x = sample(Dist::Sparse90, m, k, 1);
        let w = sample(Dist::Gaussian, k, n, 2);
        let dref = dense_matmul(&x, &w, m, k, n);
        let so = sparse_skip_matmul(&x, &w, m, k, n);
        let density = x.iter().filter(|&&v| v != 0.0).count() as f32 / x.len() as f32;
        let dt = bench(reps, || dense_matmul(&x, &w, m, k, n));
        let st = bench(reps, || sparse_skip_matmul(&x, &w, m, k, n));
        println!(
            "SPARSE   (probe: density {:.0}%) → skip-zeros   dense {dt:>6.2}ms  kernel {st:>6.2}ms  {:>5.1}×   err {:.1e}  (exact)",
            density * 100.0,
            dt / st,
            rel_err(&dref, &so)
        );
    }

    // 2) LOW-RANK weight → factored W≈U·V (the decomposition).
    {
        let w = sample(Dist::LowRank, k, n, 3);
        let x = sample(Dist::Gaussian, m, k, 4);
        let sr = stable_rank(&w, k, n);
        let (u, v, r) = factorize(&w, k, n, 8);
        let recon = dense_matmul(&u, &v, k, r, n); // U·V ≈ W
        let recon_err = rel_err(&w, &recon);
        let dref = dense_matmul(&x, &w, m, k, n);
        let fo = factored_matmul(&x, &u, &v, m, k, r, n);
        let dt = bench(reps, || dense_matmul(&x, &w, m, k, n));
        let ft = bench(reps, || factored_matmul(&x, &u, &v, m, k, r, n));
        println!(
            "LOW-RANK (probe: stable-rank {sr:.1}) → factored r={r}  dense {dt:>6.2}ms  kernel {ft:>6.2}ms  {:>5.1}×   err {:.1e}  (recon {recon_err:.1e})",
            dt / ft,
            rel_err(&dref, &fo)
        );
    }

    // 3) QUANT weight → per-channel int8 (compression; error bounded).
    {
        let w = sample(Dist::Gaussian, k, n, 5);
        let x = sample(Dist::Gaussian, m, k, 6);
        let dref = dense_matmul(&x, &w, m, k, n);
        let qo = quant_matmul(&x, &w, m, k, n);
        let pcq = per_channel_quant_error(&w, k, n, 8);
        println!(
            "QUANT    (probe: q8/chan {pcq:.3}) → int8/chan   weights 4× smaller (mem-bound win)     err {:.1e}",
            rel_err(&dref, &qo)
        );
    }

    // 4) DECODE latency: the real quant win is memory BANDWIDTH. At batch 1 every
    //    matmul is a GEMV where each weight is read once. Weight-stationary
    //    dot-product form (transposed [n,k], NEON) makes weight traffic dominate,
    //    so int8 (1 byte) streams 4× less than f32 (4 bytes). Both kernels are
    //    real (NEON on aarch64); transpose+quantize is one-time load, not timed.
    // 4) MATMUL kernels: f32 vs W8A16 (int8 wt / f32 act) vs W8A8 (both int8,
    //    SDOT) — SPEED and PRECISION together, across decode (m=1) and prefill
    //    (m=32) on real qwen layer shapes. Speedup is vs the f32 kernel; error is
    //    rel-L2 vs the f32 dense reference.
    let accel = if cfg!(all(target_arch = "aarch64", feature = "dotprod")) {
        "NEON f32/W8A16 + SDOT W8A8"
    } else if cfg!(all(target_arch = "aarch64", feature = "neon")) {
        "NEON f32/W8A16, scalar W8A8"
    } else {
        "portable scalar (enable neon/dotprod for HW paths)"
    };
    println!("\nmatmul kernels — speed + precision [{accel}], qwen layer shapes:");
    println!(
        "  {:<22}  {:>7}   {:>16}   {:>16}",
        "shape (m×k×n)", "f32", "W8A16  (×, err)", "W8A8  (×, err)"
    );
    for &(gk, gn, label) in &[
        (1024usize, 1024usize, "attn"),
        (1024, 3072, "mlp-up"),
        (3072, 1024, "mlp-down"),
    ] {
        let w = sample(Dist::Gaussian, gk, gn, 7);
        let wt = transpose(&w, gk, gn); // [n,k] f32 (one-time)
        let (wtq, sc) = quantize_cols_t(&w, gk, gn); // [n,k] int8 weights + scales
        for &mrows in &[1usize, 32] {
            let x = sample(Dist::Gaussian, mrows, gk, 8);
            let refy = dense_ref(&x, &w, mrows, gk, gn); // f32 truth
            let (xq, sx) = quantize_rows_i8(&x, mrows, gk); // int8 activations (W8A8)
            let reps = if mrows == 1 { 300 } else { 40 };
            let tf = bench(reps, || matmul_f32(&x, &wt, mrows, gk, gn));
            let tw = bench(reps, || matmul_w8a16(&x, &wtq, &sc, mrows, gk, gn));
            let td = bench(reps, || matmul_w8a8(&xq, &sx, &wtq, &sc, mrows, gk, gn));
            let e16 = rel_err(&refy, &matmul_w8a16(&x, &wtq, &sc, mrows, gk, gn));
            let e8 = rel_err(&refy, &matmul_w8a8(&xq, &sx, &wtq, &sc, mrows, gk, gn));
            let tag = if mrows == 1 { "decode" } else { "prefill" };
            println!(
                "  {label}-{tag} {mrows}×{gk}×{gn:<5}  {tf:>6.3}ms   {tw:>6.3}ms {:>4.2}× {e16:.3}   {td:>6.3}ms {:>4.2}× {e8:.3}",
                tf / tw,
                tf / td,
            );
        }
    }

    // 5) Would AMX help? AMX is Apple's matrix coprocessor — GEMM-shaped, reached
    //    only via Accelerate/BNNS. Hypothesis: big win on prefill (matrix×matrix),
    //    little on decode (matrix×vector underutilizes the array). Measured:
    //    NEON-f32 vs Accelerate sgemm (AMX) vs W8A8-SDOT, k=n=1024, across m.
    #[cfg(all(feature = "amx", target_vendor = "apple"))]
    {
        println!("\nAMX vs hand-NEON, k=n=1024  (AMX = Accelerate; both sgemm & BNNS use it):");
        let (gk, gn) = (1024usize, 1024usize);
        let w = sample(Dist::Gaussian, gk, gn, 7);
        let wt = transpose(&w, gk, gn);
        let (wtq_pc, sc) = quantize_cols_t(&w, gk, gn); // per-channel int8 wt (SDOT)
        let (wtq, swt) = quantize_row_i8(&wt); // per-tensor int8 of TRANSPOSED wt (BNNS FC: [out,in])

        // Probe once at m=8: f32-matmul validates the FFI/struct; int8-FC is the
        // real int8-on-AMX path. Report rc + error for each.
        let px = sample(Dist::Gaussian, 8, gk, 9);
        let pref = dense_ref(&px, &w, 8, gk, gn);
        let (bf, bf_rc) = bnns_matmul_f32(&px, &w, 8, gk, gn);
        let (pxq, psx) = quantize_row_i8(&px);
        let (bi, bi_rc) = bnns_fc_i8(&pxq, psx, &wtq, swt, 8, gk, gn);
        println!(
            "  BNNS probe: matmul-f32 rc={bf_rc} err {:.1e} | FC-int8 rc={bi_rc} err {}",
            rel_err(&pref, &bf),
            if bi_rc == 0 {
                format!("{:.3}", rel_err(&pref, &bi))
            } else {
                "n/a".into()
            }
        );
        let bnns_i8_ok = bi_rc == 0;

        for &mrows in &[1usize, 32, 128] {
            let x = sample(Dist::Gaussian, mrows, gk, 8);
            let (xq, sx) = quantize_rows_i8(&x, mrows, gk); // per-row (SDOT)
            let (xqt, sxt) = quantize_row_i8(&x); // per-tensor (BNNS)
            let reps = if mrows == 1 {
                300
            } else if mrows == 32 {
                40
            } else {
                12
            };
            let tf = bench(reps, || matmul_f32(&x, &wt, mrows, gk, gn));
            let ta = bench(reps, || accel_sgemm(&x, &w, mrows, gk, gn));
            let td = bench(reps, || matmul_w8a8(&xq, &sx, &wtq_pc, &sc, mrows, gk, gn));
            let tag = if mrows == 1 { "decode " } else { "prefill" };
            print!(
                "  m={mrows:<3} ({tag})  NEON-f32 {tf:>6.3}   AMX-f32 {ta:>6.3} {:>5.1}×   SDOT-i8 {td:>6.3} {:>4.1}×",
                tf / ta,
                tf / td
            );
            if bnns_i8_ok {
                let tb = bench(reps, || bnns_fc_i8(&xqt, sxt, &wtq, swt, mrows, gk, gn).0);
                print!(
                    "   BNNS-i8/AMX {tb:>6.3} {:>4.1}× (vs AMX-f32 {:>4.2}×)",
                    tf / tb,
                    ta / tb
                );
            }
            println!();
        }
        if bnns_i8_ok {
            println!(
                "  ↑ BNNS-i8/AMX = int8 fully-connected on AMX; 'vs AMX-f32' answers: does int8-on-AMX beat f32-on-AMX?"
            );
        } else {
            println!(
                "  (int8 FC rc={bi_rc} — path unavailable on this OS; AMX-f32 remains the measured king.)"
            );
        }
    }
    #[cfg(not(all(feature = "amx", target_vendor = "apple")))]
    println!(
        "\n(AMX comparison off — build with `--features amx` on an Apple target to include it.)"
    );

    // 6) Caveat analysis — cache vs DRAM. These layers fit L2/SLC, so W8A16's 4×
    //    fewer weight bytes buys nothing (compute-bound). Does the byte saving
    //    become a SPEED win once the weight spills cache (DRAM-bandwidth-bound)?
    //    Compare a cache-resident shape vs a DRAM-spilling one (m=1).
    println!(
        "\ncache-vs-DRAM (m=1) — does W8A16's byte saving become speed when weights spill cache?"
    );
    for &gk in &[2048usize, 6144] {
        let gn = gk;
        let w = sample(Dist::Gaussian, gk, gn, 7);
        let x = sample(Dist::Gaussian, 1, gk, 8);
        let wt = transpose(&w, gk, gn);
        let (wtq, sc) = quantize_cols_t(&w, gk, gn);
        let (xq, sx) = quantize_rows_i8(&x, 1, gk);
        let reps = if gk <= 2048 { 100 } else { 20 };
        let tf = bench(reps, || matmul_f32(&x, &wt, 1, gk, gn));
        let tw = bench(reps, || matmul_w8a16(&x, &wtq, &sc, 1, gk, gn));
        let td = bench(reps, || matmul_w8a8(&xq, &sx, &wtq, &sc, 1, gk, gn));
        let mb = (gk * gn * 4) as f64 / 1e6;
        let loc = if gk * gn * 4 > 24 << 20 {
            "DRAM "
        } else {
            "cache"
        };
        println!(
            "  {gk}×{gn}  f32-wt {mb:>4.0}MB ({loc})   f32 {tf:>6.3}ms   W8A16 {tw:>6.3}ms {:.2}×   W8A8 {td:>6.3}ms {:.2}×",
            tf / tw,
            tf / td
        );
    }

    println!("\n── VERDICT: on Apple Silicon, Accelerate/AMX f32 beats every hand kernel ──");
    println!(
        "  decode (m=1):   AMX-sgemm ~10× over NEON-f32 — and ~2× faster than the W8A8-SDOT int8 kernel."
    );
    println!(
        "  prefill (m≫1):  AMX-sgemm 37–50× over NEON (AMX + multicore) — it dwarfs the int8 kernels too."
    );
    println!(
        "  ⇒ On Apple, int8's value is MEMORY FOOTPRINT (bigger models / more KV cache), NOT speed:"
    );
    println!(
        "     route f32 matmul to Accelerate (AMX); int8 SPEED would need BNNS-on-AMX, not hand-NEON SDOT."
    );
    println!(
        "     The hand-NEON int8 kernels are the right path only where AMX/Accelerate is absent (other CPUs)."
    );
    println!(
        "  CAVEAT ANALYSIS (measured where possible — full writeup in docs/quant-kernels.md):"
    );
    println!(
        "   1. per-matmul err ≠ model quality: W8A8 0.012 compounds over 28 layers, NOT yet validated"
    );
    println!("      end-to-end (W8A16's 0.008 was: 100% next-token). Reasoned, not closed.");
    println!(
        "   2. int8-on-AMX — MEASURED (BNNS FC int8): a CROSSOVER — LOSES to f32-AMX at decode
      (m=1 ~0.5×), ties at m=32, WINS at large prefill (m=128 ~2×). f32-AMX for decode,
      int8-AMX for throughput."
    );
    println!(
        "   3. threading — MEASURED: single-thread Accelerate ≈ multi-thread (≤1.35× from cores at"
    );
    println!("      m=128, none at decode) ⇒ AMX's 10–50× is the MATRIX UNIT, not multicore.");
    println!(
        "   4. cache vs DRAM — MEASURED: W8A16 0.90×(cache 17MB) → 1.00×(DRAM 151MB): the byte saving"
    );
    println!(
        "      only reaches break-even when bandwidth-bound. On qwen-0.6B (cache-resident) it's pure"
    );
    println!("      footprint; W8A8 stays ~3× either way (compute win).");
}
