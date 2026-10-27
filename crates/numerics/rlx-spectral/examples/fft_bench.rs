//! ns per transform: rlx-spectral vs rustfft (complex, f32), and real FFTs,
//! best of N over a batch of frames.
use rlx_spectral::{Complex, FftPlan, RealFftPlan};
fn best(reps: usize, mut f: impl FnMut()) -> f64 {
    (0..reps)
        .map(|_| {
            let t = std::time::Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::MAX, f64::min)
}
fn main() {
    let frames = 2000;
    for n in [400usize, 512, 1024] {
        let plan = FftPlan::<f32>::new(n);
        let mut scratch = vec![Complex::default(); plan.scratch_len()];
        let base: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new((i as f32 * 0.37).sin(), 0.0))
            .collect();
        let mut bufs: Vec<Vec<Complex<f32>>> = (0..frames).map(|_| base.clone()).collect();
        let t_rlx = best(15, || {
            for b in bufs.iter_mut() {
                plan.forward_with(b, &mut scratch)
            }
        });
        let rf = rustfft::FftPlanner::<f32>::new().plan_fft_forward(n);
        let rb: Vec<rustfft::num_complex::Complex<f32>> = base
            .iter()
            .map(|c| rustfft::num_complex::Complex::new(c.re, c.im))
            .collect();
        let mut rbufs: Vec<_> = (0..frames).map(|_| rb.clone()).collect();
        let mut rs =
            vec![rustfft::num_complex::Complex::new(0.0, 0.0); rf.get_inplace_scratch_len()];
        let t_rf = best(15, || {
            for b in rbufs.iter_mut() {
                rf.process_with_scratch(b, &mut rs)
            }
        });
        println!(
            "complex n={n:5}: rlx-spectral {:8.0} ns  rustfft {:8.0} ns",
            t_rlx * 1e9 / frames as f64,
            t_rf * 1e9 / frames as f64
        );
    }
    for n in [400usize, 512] {
        let plan = RealFftPlan::<f32>::new(n);
        let mut scratch = plan.make_scratch();
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut out = vec![Complex::default(); plan.bins()];
        let t = best(15, || {
            for _ in 0..frames {
                plan.forward_with(&x, &mut out, &mut scratch)
            }
        });
        println!(
            "real    n={n:5}: rlx-spectral {:8.0} ns",
            t * 1e9 / frames as f64
        );
    }
}
