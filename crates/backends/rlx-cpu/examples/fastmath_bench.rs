//! Throughput of the f32 exp / log / tanh variants over 1M elements (best of
//! N, single thread): rlx-cpu `fastmath` (accurate NEON), the `_fast`
//! polynomials, `vmath::vv*` (Accelerate in non-`pure-rust` Apple builds),
//! std per element (the platform libm) and rlx-libm's reference ports.
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
    let n = 1 << 20;
    let reps: usize = std::env::args().nth(1).map_or(30, |s| s.parse().unwrap());
    let xe: Vec<f32> = (0..n)
        .map(|i| ((i * 7919) % 20000) as f32 * 1e-3 - 10.0)
        .collect();
    let xl: Vec<f32> = (0..n)
        .map(|i| ((i * 7919) % 20000) as f32 * 1e-2 + 1e-3)
        .collect();
    let mut y = vec![0f32; n];
    let ns = |t: f64| t * 1e9 / n as f64;
    use rlx_cpu::{fastmath as fm, vmath as vm};
    println!("ns/element      fastmath  vv(_fast)  vv(accurate)  std/libm");
    let e = [
        best(reps, || fm::exp_slice(&mut y, &xe)),
        best(reps, || vm::vvexpf_fast(&mut y, &xe)),
        best(reps, || vm::vvexpf(&mut y, &xe)),
        best(reps, || {
            for (o, &v) in y.iter_mut().zip(&xe) {
                *o = v.exp()
            }
        }),
    ];
    println!(
        "exp          {:9.3} {:9.3} {:12.3} {:9.3}",
        ns(e[0]),
        ns(e[1]),
        ns(e[2]),
        ns(e[3])
    );
    let l = [
        best(reps, || fm::log_slice(&mut y, &xl)),
        best(reps, || vm::vvlogf_fast(&mut y, &xl)),
        best(reps, || vm::vvlogf(&mut y, &xl)),
        best(reps, || {
            for (o, &v) in y.iter_mut().zip(&xl) {
                *o = v.ln()
            }
        }),
    ];
    println!(
        "log          {:9.3} {:9.3} {:12.3} {:9.3}",
        ns(l[0]),
        ns(l[1]),
        ns(l[2]),
        ns(l[3])
    );
    let t = [
        best(reps, || fm::tanh_slice(&mut y, &xe)),
        best(reps, || vm::vvtanhf_fast(&mut y, &xe)),
        best(reps, || vm::vvtanhf(&mut y, &xe)),
        best(reps, || {
            for (o, &v) in y.iter_mut().zip(&xe) {
                *o = v.tanh()
            }
        }),
    ];
    println!(
        "tanh         {:9.3} {:9.3} {:12.3} {:9.3}",
        ns(t[0]),
        ns(t[1]),
        ns(t[2]),
        ns(t[3])
    );
    std::hint::black_box(&y);
}
