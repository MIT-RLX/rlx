// RLX — MIT OR Apache-2.0. Microbench: fused matvec vs materialize-then-matmul.
use rlx_mlx_io::{dequant_matmul_affine, dequant_matvec_affine};
fn main() {
    let (k, n, gs, bits) = (4096usize, 2048usize, 64u32, 2u32);
    let n_groups = k / gs as usize;
    let pf = 32 / bits as usize; // codes per byte-ish; pack_factor
    let row_bytes = n_groups * (gs as usize / (8 / bits as usize));
    let w: Vec<u8> = (0..n * row_bytes).map(|i| (i * 37 + 11) as u8).collect();
    let scales: Vec<f32> = (0..n * n_groups)
        .map(|i| 0.01 + (i % 7) as f32 * 0.003)
        .collect();
    let biases: Vec<f32> = (0..n * n_groups)
        .map(|i| -0.05 + (i % 5) as f32 * 0.002)
        .collect();
    let x: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let _ = pf;

    let a = dequant_matmul_affine(&x, &w, &scales, &biases, bits, gs, 1, k, n).unwrap();
    let b = dequant_matvec_affine(&x, &w, &scales, &biases, bits, gs, k, n).unwrap();
    let maxerr = a
        .iter()
        .zip(&b)
        .map(|(p, q)| (p - q).abs())
        .fold(0f32, f32::max);
    println!("correctness: max|materialize - fused| = {maxerr:.3e}  (n={n})");

    let iters = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        std::hint::black_box(
            dequant_matmul_affine(&x, &w, &scales, &biases, bits, gs, 1, k, n).unwrap(),
        );
    }
    let t_mat = t0.elapsed().as_secs_f64() / iters as f64;
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        std::hint::black_box(
            dequant_matvec_affine(&x, &w, &scales, &biases, bits, gs, k, n).unwrap(),
        );
    }
    let t_fused = t1.elapsed().as_secs_f64() / iters as f64;
    println!(
        "materialize: {:.3} ms/call | fused: {:.3} ms/call | speedup {:.1}x",
        t_mat * 1e3,
        t_fused * 1e3,
        t_mat / t_fused
    );
}
