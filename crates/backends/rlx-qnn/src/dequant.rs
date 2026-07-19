// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// GPL-3.0-only.

//! Host-side GGUF dequant for `Op::DequantMatMul` on QNN.
//!
//! QNN has no native GGUF kernels; we decode packed weights to f32 on the host
//! (same posture as CoreML's off-device path) and run a plain `MatMul`. Layout:
//! packed / dequanted weights are row-major `[N, K]`; QNN MatMul wants
//! `[K, N]`, so we transpose after dequant.

use rlx_ir::quant::QuantScheme;

/// Decode `bytes` → `n` f32 values (GGUF element order, `[N,K]` row-major).
pub fn dequant_scheme(scheme: QuantScheme, bytes: &[u8], n: usize) -> Result<Vec<f32>, String> {
    use QuantScheme::*;
    let r = match scheme {
        GgufQ8_0 => rlx_gguf::dequant_q8_0(bytes, n),
        GgufQ4_0 => rlx_gguf::dequant_q4_0(bytes, n),
        GgufQ4_1 => rlx_gguf::dequant_q4_1(bytes, n),
        GgufQ5_0 => rlx_gguf::dequant_q5_0(bytes, n),
        GgufQ5_1 => rlx_gguf::dequant_q5_1(bytes, n),
        GgufQ2K => rlx_gguf::dequant_q2_k(bytes, n),
        GgufQ3K => rlx_gguf::dequant_q3_k(bytes, n),
        GgufQ4K => rlx_gguf::dequant_q4_k(bytes, n),
        GgufQ5K => rlx_gguf::dequant_q5_k(bytes, n),
        GgufQ6K => rlx_gguf::dequant_q6_k(bytes, n),
        GgufQ8K => rlx_gguf::dequant_q8_k(bytes, n),
        GgufIQ4NL => rlx_gguf::iq_dequant::dequant_iq4_nl(bytes, n),
        GgufIQ4XS => rlx_gguf::iq_dequant::dequant_iq4_xs(bytes, n),
        GgufIQ2XXS => rlx_gguf::iq_dequant::dequant_iq2_xxs(bytes, n),
        GgufIQ2XS => rlx_gguf::iq_dequant::dequant_iq2_xs(bytes, n),
        GgufIQ2S => rlx_gguf::iq_dequant::dequant_iq2_s(bytes, n),
        GgufIQ3XXS => rlx_gguf::iq_dequant::dequant_iq3_xxs(bytes, n),
        GgufIQ3S => rlx_gguf::iq_dequant::dequant_iq3_s(bytes, n),
        GgufIQ1S => rlx_gguf::iq_dequant::dequant_iq1_s(bytes, n),
        GgufIQ1M => rlx_gguf::iq_dequant::dequant_iq1_m(bytes, n),
        GgufTQ1_0 => rlx_gguf::tq_dequant::dequant_tq1_0(bytes, n),
        GgufTQ2_0 => rlx_gguf::tq_dequant::dequant_tq2_0(bytes, n),
        GgufMXFP4 => rlx_gguf::mx_dequant::dequant_mxfp4(bytes, n),
        GgufNVFP4 => rlx_gguf::mx_dequant::dequant_nvfp4(bytes, n),
        GgufQ1_0 => rlx_gguf::q1_dequant::dequant_q1_0(bytes, n),
        GgufQ2_0 => rlx_gguf::q2_dequant::dequant_q2_0(bytes, n),
        other => {
            return Err(format!(
                "rlx-qnn: unsupported DequantMatMul scheme {other:?} \
                 (host-dequant covers GGUF legacy + K + IQ + ternary + MX)"
            ));
        }
    };
    r.map_err(|e| format!("rlx-qnn gguf dequant: {e}"))
}

/// `[N,K]` row-major → `[K,N]` for QNN MatMul (`transpose_in1 = false`).
pub fn transpose_nk_to_kn(nk: &[f32], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(nk.len(), n * k);
    let mut kn = vec![0.0f32; k * n];
    for row in 0..n {
        for col in 0..k {
            kn[col * n + row] = nk[row * k + col];
        }
    }
    kn
}

/// Dequant packed `[N,K]` GGUF bytes and transpose to `[K,N]` f32.
pub fn dequant_weight_for_qnn(
    scheme: QuantScheme,
    bytes: &[u8],
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    let nk = dequant_scheme(scheme, bytes, n * k)?;
    Ok(transpose_nk_to_kn(&nk, n, k))
}
