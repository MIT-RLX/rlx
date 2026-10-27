//! V-JEPA2-giant's `x · Wᵀ` shape (8192 tokens, 1408 → 6144) repeated, for
//! profiling the SME route's packing against its kernel.
fn main() {
    let (m, k, n) = (8192usize, 1408usize, 6144usize);
    let reps: usize = std::env::args().nth(1).map_or(6, |s| s.parse().unwrap());
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7919) % 1000) as f32 * 1e-3 - 0.5)
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|i| ((i * 104729) % 1000) as f32 * 1e-3 - 0.5)
        .collect();
    let mut c = vec![0f32; m * n];
    for r in 0..reps {
        let t = std::time::Instant::now();
        rlx_cpu::blas::sgemm_bt(&a, &b, &mut c, m, k, n, 1.0);
        let s = t.elapsed().as_secs_f64();
        println!(
            "rep {r}: {:.1} ms, {:.0} GFLOP/s",
            s * 1e3,
            2.0 * (m * k * n) as f64 / s / 1e9
        );
    }
}
