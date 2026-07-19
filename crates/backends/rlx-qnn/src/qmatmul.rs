// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// GPL-3.0-only.

//! Host INT8 `Op::QMatMul` for the QNN FFI path.
//!
//! QNN's CPU reference backend rejects mixed f32×int8 `MatMul` (`0xc26`).
//! Fully quantized int8 matmul therefore runs the same integer accumulate +
//! requantize kernel as `rlx-cpu` (weights stay I8 — no host f32 dequant),
//! while the rest of the graph still uses the QNN session when present.
//! HTP can replace this with a native sfixed8 MatMul later.

/// Row-major INT8 matmul with i32 bias and requantize to I8.
///
/// `out[m,n] = clamp(round((bias[n] + Σₖ (x-x_zp)·(w-w_zp)) · mult) + out_zp)`.
pub fn q_matmul_i8(
    x: &[i8],
    w: &[i8],
    bias: &[i32],
    m: usize,
    k: usize,
    n: usize,
    x_zp: i32,
    w_zp: i32,
    out_zp: i32,
    mult: f32,
) -> Vec<i8> {
    assert_eq!(x.len(), m * k);
    assert_eq!(w.len(), k * n);
    assert_eq!(bias.len(), n);
    let mut out = vec![0i8; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc: i32 = bias[ni];
            for ki in 0..k {
                let xv = x[mi * k + ki] as i32 - x_zp;
                let wv = w[ki * n + ni] as i32 - w_zp;
                acc += xv * wv;
            }
            let r = (acc as f32 * mult).round() as i32 + out_zp;
            out[mi * n + ni] = r.clamp(-128, 127) as i8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_matmul_identity_zp0() {
        // x = I, w = I, bias = 0, mult = 1 → out = I
        let x = [1i8, 0, 0, 1];
        let w = [1i8, 0, 0, 1];
        let bias = [0i32, 0];
        let out = q_matmul_i8(&x, &w, &bias, 2, 2, 2, 0, 0, 0, 1.0);
        assert_eq!(out, vec![1, 0, 0, 1]);
    }
}
