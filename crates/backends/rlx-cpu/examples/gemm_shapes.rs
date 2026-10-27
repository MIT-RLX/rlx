//! sgemm throughput on whisper large-v3 / wav2vec2-large shapes, min of N
//! runs (robust to a loaded machine). Compare builds: default (vendor BLAS)
//! vs `--features pure-rust` (SME2 / `gemm` crate). `RLX_CPU_SME=0` turns the
//! SME route off in pure builds.
fn main() {
    let shapes = [
        (1500, 1280, 1280),
        (1500, 1280, 5120),
        (1500, 5120, 1280),
        (749, 1024, 1024),
        (749, 1024, 4096),
        (448, 1280, 1280),
    ];
    let reps: usize = std::env::args().nth(1).map_or(7, |s| s.parse().unwrap());
    for (m, k, n) in shapes {
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7919) % 1000) as f32 * 1e-3 - 0.5)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 104729) % 1000) as f32 * 1e-3 - 0.5)
            .collect();
        let mut c = vec![0f32; m * n];
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            rlx_cpu::blas::sgemm(&a, &b, &mut c, m, k, n);
            best = best.min(t.elapsed().as_secs_f64());
        }
        let checksum: f64 = c.iter().step_by(97).map(|&x| x as f64).sum();
        println!(
            "{m}x{k}x{n}: {:7.2} ms  {:6.0} GFLOP/s  (checksum {checksum:.6})",
            best * 1e3,
            2.0 * (m * k * n) as f64 / best / 1e9
        );
    }
}
