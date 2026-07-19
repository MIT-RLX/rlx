// RLX — versatile ML compiler + runtime.
//! MLX on-device Q1_0 dequant parity vs host `rlx_gguf::q1_dequant`.

#![cfg(rlx_mlx_host)]

use rlx_ir::DType;
use rlx_mlx::array::Array;
use rlx_mlx::dequant_q1_0::dequant_q1_0_ondevice;

#[test]
fn mlx_q1_0_ondevice_matches_host() {
    let (n, k) = (4usize, 256usize);
    let mut w = vec![0f32; n * k];
    for (i, v) in w.iter_mut().enumerate() {
        *v = if i % 3 == 0 { 0.5 } else { -0.25 };
    }
    let packed = rlx_gguf::q1_dequant::quantize_q1_0(&w).expect("quantize");
    let host = rlx_gguf::q1_dequant::dequant_q1_0(&packed, n * k).expect("host dequant");

    let w_u8 = Array::from_bytes(&packed, &[packed.len()], DType::U8).expect("u8 arr");
    let gpu = dequant_q1_0_ondevice(&w_u8, k, n).expect("ondevice");
    let gpu_f = gpu.to_f32().expect("to_f32");
    assert_eq!(gpu_f.len(), host.len());
    let mut max_abs = 0f32;
    for (a, b) in host.iter().zip(gpu_f.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < 1e-5,
        "Q1_0 MLX ondevice vs host max_abs={max_abs:.3e}"
    );
}
