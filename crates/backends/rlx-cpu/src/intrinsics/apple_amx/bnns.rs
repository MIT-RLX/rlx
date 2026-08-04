// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! BNNS-backed low-precision matmul — bf16 / f16 onto the Apple matrix unit.
//!
//! BNNS (part of Accelerate) is Apple's sanctioned CPU numerical library that
//! itself drives the matrix coprocessor — AMX on M1–M3, SME on M4+ — so it is
//! the maintenance-free way to reach the hardware without owning undocumented
//! per-generation assembly. For dense **f32** a separate kernel is pointless:
//! `cblas_sgemm` already IS the AMX path (see [`super::dense`]). The value BNNS
//! adds is **half precision**: bf16/f16 matmul halves weight bandwidth and uses
//! the SME `B16F32`/`F16F32` accumulate this chip reports — a real win for LLM
//! inference that classic CBLAS doesn't expose.
//!
//! # What BNNSMatMul actually supports (probed on this SDK, macOS 26)
//!
//! `BNNSMatMul` is **floating-point only**: `bf16·bf16→f32`, `f16·f16→f32`
//! (and half→half) are accepted; **int8·int8 is rejected** (`ws_size = -1`).
//! Integer/quantized matmul in BNNS lives in the FullyConnected *layer* path
//! (`BNNSFilterCreateLayerFullyConnected`), not here — so a true int8 **W8A8**
//! route is deliberately out of scope for this module (tracked as follow-up).
//! Half precision covers the "bf16" half of the quant-throughput goal cleanly.
//!
//! # Numerics
//!
//! Feeding f32 operands through bf16/f16 is lossy by construction — it is a
//! *different, faster* mode, not a bit-exact substitute. Callers opt in
//! explicitly (`RLX_CPU_BNNS_BF16=1` for the `sgemm_auto` hook) and the tests
//! report the accuracy delta vs the f32 oracle ([[feedback_perf_is_north_star]]).
//!
//! The classic BNNS direct API is `__API_DEPRECATED("Use BNNSGraph* APIs")` as
//! of macOS 15 but remains present/functional; `BNNSMatMul` is the current
//! direct entry point. Rust FFI doesn't observe C deprecation attributes.

use half::{bf16, f16};
use std::os::raw::c_void;

// ── FFI: BNNSNDArrayDescriptor (byte-for-byte from vecLib BNNS headers) ──
// typedef struct { flags; layout; size[8]; stride[8]; data; data_type;
//   table_data; table_data_type; data_scale; data_bias; } — all three enums are
// BNNS_ENUM(..., uint32_t, ...). repr(C) reproduces the C padding (the u32
// data_type before the *table_data pointer gets 4 bytes of tail padding).
const BNNS_MAX_TENSOR_DIMENSION: usize = 8;

#[repr(C)]
struct BnnsNDArrayDescriptor {
    flags: u32,
    layout: u32,
    size: [usize; BNNS_MAX_TENSOR_DIMENSION],
    stride: [usize; BNNS_MAX_TENSOR_DIMENSION],
    data: *mut c_void,
    data_type: u32,
    table_data: *mut c_void,
    table_data_type: u32,
    data_scale: f32,
    data_bias: f32,
}

// Enum values (from bnns_constants.h): FloatBit = 0x10000.
const BNNS_DATA_TYPE_FLOAT32: u32 = 0x10000 | 32; // 0x10020
const BNNS_DATA_TYPE_FLOAT16: u32 = 0x10000 | 16; // 0x10010
const BNNS_DATA_TYPE_BFLOAT16: u32 = 0x10000 | 0x8000 | 16; // 0x18010
const BNNS_LAYOUT_ROW_MAJOR_MATRIX: u32 = 0x20000;
const BNNS_FLAG_DEFAULT: u32 = 0; // BNNSNDArrayFlagBackpropSet

unsafe extern "C" {
    fn BNNSMatMulWorkspaceSize(
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        input_a: *const BnnsNDArrayDescriptor,
        input_b: *const BnnsNDArrayDescriptor,
        output: *const BnnsNDArrayDescriptor,
        filter_params: *const c_void,
    ) -> isize;

    fn BNNSMatMul(
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        input_a: *const BnnsNDArrayDescriptor,
        input_b: *const BnnsNDArrayDescriptor,
        output: *const BnnsNDArrayDescriptor,
        workspace: *mut c_void,
        filter_params: *const c_void,
    ) -> i32;
}

/// Build a 2-D row-major matrix descriptor of shape `[rows, cols]`.
///
/// BNNS orders `size[]` innermost-first; for a row-major matrix the innermost
/// (fastest) axis is the columns, so `size = [cols, rows]`, `stride = [1, cols]`
/// — consistent with the header's own `4x5x6 · 6x7 → 4x5x7` example and
/// confirmed by the f32 parity test.
fn matrix_desc(
    data: *mut c_void,
    rows: usize,
    cols: usize,
    data_type: u32,
) -> BnnsNDArrayDescriptor {
    let mut size = [0usize; BNNS_MAX_TENSOR_DIMENSION];
    let mut stride = [0usize; BNNS_MAX_TENSOR_DIMENSION];
    size[0] = cols;
    size[1] = rows;
    stride[0] = 1;
    stride[1] = cols;
    BnnsNDArrayDescriptor {
        flags: BNNS_FLAG_DEFAULT,
        layout: BNNS_LAYOUT_ROW_MAJOR_MATRIX,
        size,
        stride,
        data,
        data_type,
        table_data: std::ptr::null_mut(),
        table_data_type: BNNS_DATA_TYPE_FLOAT32,
        data_scale: 0.0, // 0.0 ⇒ BNNS treats as 1.0
        data_bias: 0.0,
    }
}

/// Whether the BNNS matmul path is linked (Accelerate present). BNNS ships in
/// Accelerate on every Apple platform, so linkage is the only gate; Apple picks
/// AMX vs SME internally per chip.
pub fn is_available() -> bool {
    cfg!(rlx_cpu_blas_accelerate)
}

/// Whether `sgemm_auto` should route through the BNNS bf16 low-precision path.
/// Requires the build (`amx-bnns`) + explicit `RLX_CPU_BNNS_BF16=1`. Opt-in is
/// deliberate: downcasting f32→bf16 is lossy, so it must never silently replace
/// the exact f32 vendor path ([[feedback_perf_is_north_star]]).
pub fn dispatch_enabled() -> bool {
    is_available()
        && matches!(
            std::env::var("RLX_CPU_BNNS_BF16").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        )
}

/// Whether `sgemm_auto` should route through the BNNS **f16** path (`F16F32`:
/// 10 mantissa bits vs bf16's 7 — more precise at the same operand bandwidth).
/// Opt-in via `RLX_CPU_BNNS_F16=1`.
pub fn dispatch_enabled_f16() -> bool {
    is_available()
        && matches!(
            std::env::var("RLX_CPU_BNNS_F16").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        )
}

/// Core dispatch: `C[m,n] = A·B`, `A`=`[m,k]`, `B`=`[k,n]` row-major in
/// `data_type`, `C` row-major f32. `# Safety`: `a_ptr`/`b_ptr` must address
/// `m*k` / `k*n` valid `data_type` elements. Returns false if BNNS rejects.
unsafe fn matmul_raw(
    a_ptr: *const c_void,
    b_ptr: *const c_void,
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    data_type: u32,
) -> bool {
    debug_assert_eq!(out.len(), m * n);
    if m == 0 || n == 0 {
        return true;
    }
    let da = matrix_desc(a_ptr as *mut c_void, m, k, data_type);
    let db = matrix_desc(b_ptr as *mut c_void, k, n, data_type);
    let dc = matrix_desc(
        out.as_mut_ptr() as *mut c_void,
        m,
        n,
        BNNS_DATA_TYPE_FLOAT32,
    );
    let ws = unsafe { BNNSMatMulWorkspaceSize(false, false, 1.0, &da, &db, &dc, std::ptr::null()) };
    let mut wsbuf: Vec<u8> = if ws > 0 {
        vec![0u8; ws as usize]
    } else {
        Vec::new()
    };
    let wsp = if wsbuf.is_empty() {
        std::ptr::null_mut()
    } else {
        wsbuf.as_mut_ptr() as *mut c_void
    };
    let rc = unsafe { BNNSMatMul(false, false, 1.0, &da, &db, &dc, wsp, std::ptr::null()) };
    rc == 0
}

/// Dense f32 `C = A·B` via BNNS. Kept as the descriptor-convention regression
/// oracle; for real f32 work use Accelerate `cblas_sgemm` ([`super::dense`]),
/// which is faster. Returns false if BNNS rejects the call.
pub fn matmul_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) -> bool {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    // SAFETY: slices are exactly m*k / k*n f32; BNNS reads (not writes) A/B.
    unsafe {
        matmul_raw(
            a.as_ptr() as *const c_void,
            b.as_ptr() as *const c_void,
            c,
            m,
            k,
            n,
            BNNS_DATA_TYPE_FLOAT32,
        )
    }
}

/// Generate a half-precision BNNS matmul for one `(half type, BNNS dtype)` pair:
/// `$mm` = `<half>·<half> → f32` (native inputs), `$via` = an f32-operand GEMM
/// drop-in that downcasts both operands to `<half>` then accumulates in f32.
/// f16 (`F16F32`, 10 mantissa bits) and bf16 (`B16F32`, 7 bits) share the same
/// SME accumulate path — only the operand dtype tag differs — so parameterize the
/// precision instead of duplicating the FFI/downcast boilerplate. Falls back to
/// `false` if BNNS rejects, so callers can defer to Accelerate f32.
macro_rules! bnns_half_matmul {
    ($half:ty, $dtype:expr, $mm:ident, $via:ident) => {
        #[doc = concat!(stringify!($half), " `C = A·B` (", stringify!($half), " inputs, f32 output).")]
        pub fn $mm(a: &[$half], b: &[$half], c: &mut [f32], m: usize, k: usize, n: usize) -> bool {
            debug_assert_eq!(a.len(), m * k);
            debug_assert_eq!(b.len(), k * n);
            // SAFETY: half is repr(transparent) u16; slices are exactly m*k / k*n.
            unsafe {
                matmul_raw(
                    a.as_ptr() as *const c_void,
                    b.as_ptr() as *const c_void,
                    c,
                    m,
                    k,
                    n,
                    $dtype,
                )
            }
        }
        #[doc = concat!("f32 GEMM drop-in via ", stringify!($half), ": downcast both operands, accumulate f32.")]
        pub fn $via(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) -> bool {
            let a16: Vec<$half> = a.iter().map(|&x| <$half>::from_f32(x)).collect();
            let b16: Vec<$half> = b.iter().map(|&x| <$half>::from_f32(x)).collect();
            $mm(&a16, &b16, c, m, k, n)
        }
    };
}

bnns_half_matmul!(
    bf16,
    BNNS_DATA_TYPE_BFLOAT16,
    matmul_bf16,
    matmul_f32_via_bf16
);
bnns_half_matmul!(f16, BNNS_DATA_TYPE_FLOAT16, matmul_f16, matmul_f32_via_f16);

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    fn cosine_relerr(got: &[f32], want: &[f32]) -> (f64, f64) {
        let (mut dot, mut ng, mut nw, mut num, mut den) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (g, t) in got.iter().zip(want) {
            dot += (*g as f64) * (*t as f64);
            ng += (*g as f64).powi(2);
            nw += (*t as f64).powi(2);
            num += ((*g - *t) as f64).powi(2);
            den += (*t as f64).powi(2);
        }
        (dot / (ng.sqrt() * nw.sqrt()), (num / den).sqrt())
    }

    /// f32 BNNSMatMul must match naive within fp tol — pins the descriptor
    /// size/stride/transpose convention.
    #[test]
    fn bnns_f32_matches_naive() {
        if !is_available() {
            eprintln!("Accelerate/BNNS not linked; skipping");
            return;
        }
        for (m, k, n) in [(1, 1, 1), (2, 3, 4), (16, 32, 8), (33, 17, 20)] {
            let a = fill(m * k, 7 + (m * k) as u64);
            let b = fill(k * n, 99 + (k * n) as u64);
            let mut c = vec![0f32; m * n];
            assert!(
                matmul_f32(&a, &b, &mut c, m, k, n),
                "BNNSMatMul(f32) rejected"
            );
            let want = naive(&a, &b, m, k, n);
            let max_abs = c
                .iter()
                .zip(&want)
                .fold(0f32, |acc, (x, y)| acc.max((x - y).abs()));
            let tol = 1e-3 * (k as f32).sqrt();
            assert!(max_abs <= tol, "{m}x{k}x{n}: diff {max_abs} > {tol}");
        }
    }

    /// bf16/f16 paths: not bit-exact (that's the point), but must track the true
    /// product closely. Assert high cosine + small relative error; print both so
    /// the precision cost is visible. bf16 (8 mantissa bits) is looser than f16.
    #[test]
    fn bnns_halfprec_tracks_f32_oracle() {
        if !is_available() {
            eprintln!("Accelerate/BNNS not linked; skipping");
            return;
        }
        for (m, k, n) in [(8, 64, 8), (32, 256, 48), (64, 512, 64)] {
            let x = fill(m * k, 3 + (m * k) as u64);
            let w = fill(k * n, 5 + (k * n) as u64);
            let want = naive(&x, &w, m, k, n);

            let mut got_bf = vec![0f32; m * n];
            assert!(
                matmul_f32_via_bf16(&x, &w, &mut got_bf, m, k, n),
                "bf16 rejected"
            );
            let (cos_bf, rel_bf) = cosine_relerr(&got_bf, &want);

            let xf: Vec<f16> = x.iter().map(|&v| f16::from_f32(v)).collect();
            let wf: Vec<f16> = w.iter().map(|&v| f16::from_f32(v)).collect();
            let mut got_f16 = vec![0f32; m * n];
            assert!(matmul_f16(&xf, &wf, &mut got_f16, m, k, n), "f16 rejected");
            let (cos_f16, rel_f16) = cosine_relerr(&got_f16, &want);

            eprintln!(
                "{m}x{k}x{n}: bf16 cos={cos_bf:.5} rel={rel_bf:.4} | f16 cos={cos_f16:.5} rel={rel_f16:.4}"
            );
            assert!(cos_bf > 0.999, "bf16 cosine {cos_bf} too low");
            assert!(rel_bf < 0.05, "bf16 rel err {rel_bf} too high");
            assert!(cos_f16 > 0.9999, "f16 cosine {cos_f16} too low");
            assert!(rel_f16 < 0.02, "f16 rel err {rel_f16} too high");
        }
    }

    /// Throughput: BNNS bf16/f16 vs Accelerate f32, on the coprocessor.
    ///   cargo test -p rlx-cpu --features amx-bnns bnns_throughput_report \
    ///     -- --ignored --nocapture
    #[test]
    #[ignore = "benchmark; run explicitly with --ignored --nocapture"]
    fn bnns_throughput_report() {
        use std::time::Instant;
        if !is_available() {
            return;
        }
        let bench = |m: usize, k: usize, n: usize, iters: usize, f: &mut dyn FnMut()| -> f64 {
            for _ in 0..2 {
                f();
            }
            let t0 = Instant::now();
            for _ in 0..iters {
                f();
            }
            2.0 * m as f64 * n as f64 * k as f64 * iters as f64 / t0.elapsed().as_secs_f64() / 1e9
        };
        eprintln!(
            "{:>16}  {:>11}  {:>11}  {:>11}",
            "shape", "f32 GF/s", "bf16 GF/s", "f16 GF/s"
        );
        for (m, k, n) in [
            (512usize, 512usize, 512usize),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
        ] {
            let a = fill(m * k, 1 + (m * k) as u64);
            let b = fill(k * n, 2 + (k * n) as u64);
            let a_bf: Vec<bf16> = a.iter().map(|&v| bf16::from_f32(v)).collect();
            let b_bf: Vec<bf16> = b.iter().map(|&v| bf16::from_f32(v)).collect();
            let a_f16: Vec<f16> = a.iter().map(|&v| f16::from_f32(v)).collect();
            let b_f16: Vec<f16> = b.iter().map(|&v| f16::from_f32(v)).collect();
            let mut c = vec![0f32; m * n];
            let iters =
                ((3e8 / (2.0 * m as f64 * k as f64 * n as f64)).ceil() as usize).clamp(3, 100);
            let gf_f32 = bench(m, k, n, iters, &mut || {
                crate::blas::sgemm(&a, &b, &mut c, m, k, n)
            });
            let gf_bf = bench(m, k, n, iters, &mut || {
                matmul_bf16(&a_bf, &b_bf, &mut c, m, k, n);
            });
            let gf_f16 = bench(m, k, n, iters, &mut || {
                matmul_f16(&a_f16, &b_f16, &mut c, m, k, n);
            });
            eprintln!(
                "{:>16}  {:>11.1}  {:>11.1}  {:>11.1}",
                format!("{m}x{k}x{n}"),
                gf_f32,
                gf_bf,
                gf_f16
            );
        }
    }
}
