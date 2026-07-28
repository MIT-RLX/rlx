// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side GGUF / MLX dequant for `Op::DequantMatMul` on QNN.
//!
//! QNN has no native GGUF/MLX kernels; we decode packed weights to f32 on the
//! host (same posture as CoreML's off-device path) and run a plain `MatMul`.
//! Layout: packed / dequanted weights are row-major `[N, K]`; QNN MatMul wants
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
                "rlx-qnn: unsupported single-buffer DequantMatMul scheme {other:?} \
                 (MLX affine/mxfp need scale/bias — use dequant_mlx_for_qnn)"
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

/// Host-dequant MLX affine / mxfp packs to `[K,N]` for QNN MatMul.
pub fn dequant_mlx_for_qnn(
    scheme: QuantScheme,
    w_bytes: &[u8],
    scales: &[u8],
    biases: &[u8],
    n: usize,
    k: usize,
) -> Result<Vec<f32>, String> {
    let gs = scheme.mlx_group_size() as usize;
    if gs == 0 || !k.is_multiple_of(gs) {
        return Err(format!(
            "rlx-qnn mlx dequant: k={k} not divisible by group_size={gs}"
        ));
    }
    let n_groups = k / gs;
    let nk = match scheme {
        QuantScheme::MlxAffine { bits, group_size } => {
            let scales_f: Vec<f32> = scales
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let biases_f: Vec<f32> = if biases.len() >= n * n_groups * 4 {
                biases
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            } else {
                vec![0.0; n * n_groups]
            };
            rlx_mlx_io::dequant_affine_f32(
                w_bytes,
                &scales_f,
                &biases_f,
                bits as u32,
                group_size,
                n,
                n_groups,
            )
            .map_err(|e| e.to_string())?
        }
        QuantScheme::MlxMxfp4 { group_size } => {
            rlx_mlx_io::dequant_mxfp4_f32(w_bytes, scales, group_size, n, n_groups)
                .map_err(|e| e.to_string())?
        }
        QuantScheme::MlxMxfp8 { group_size } => {
            rlx_mlx_io::dequant_mxfp8_f32(w_bytes, scales, group_size, n, n_groups)
                .map_err(|e| e.to_string())?
        }
        other => {
            return Err(format!(
                "rlx-qnn: dequant_mlx_for_qnn unsupported scheme {other:?}"
            ));
        }
    };
    Ok(transpose_nk_to_kn(&nk, n, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::quant::QuantScheme;

    #[test]
    fn mlx_affine_4_deferred_finish_shape() {
        let w = vec![0x10u8, 0x32, 0x54, 0x76];
        let scales: Vec<u8> = 2.0f32.to_le_bytes().to_vec();
        let biases: Vec<u8> = (-1.0f32).to_le_bytes().to_vec();
        let kn = dequant_mlx_for_qnn(
            QuantScheme::MlxAffine {
                bits: 4,
                group_size: 8,
            },
            &w,
            &scales,
            &biases,
            1,
            8,
        )
        .expect("dequant");
        assert_eq!(kn.len(), 8);
    }

    #[test]
    fn mlx_affine_3bit_pack_factor_path() {
        let w = vec![0u8; 3];
        let scales = 1.0f32.to_le_bytes().to_vec();
        let biases = 0.0f32.to_le_bytes().to_vec();
        let kn = dequant_mlx_for_qnn(
            QuantScheme::MlxAffine {
                bits: 3,
                group_size: 8,
            },
            &w,
            &scales,
            &biases,
            1,
            8,
        )
        .expect("3-bit affine");
        assert_eq!(kn.len(), 8);
    }

    #[test]
    fn mlx_mxfp4_host_dequant() {
        let n = 2usize;
        let gs = 32usize;
        let k = gs;
        let w = vec![0u8; n * k / 2];
        let scales = vec![64u8; n];
        let kn = dequant_mlx_for_qnn(
            QuantScheme::MlxMxfp4 {
                group_size: gs as u32,
            },
            &w,
            &scales,
            &[],
            n,
            k,
        )
        .expect("mxfp4");
        assert_eq!(kn.len(), k * n);
    }
}
