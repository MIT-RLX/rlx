//! How often the pure-Rust ports differ from the platform libm (std calls it):
//! bit mismatches and max ulp over pseudo-random inputs per function.
fn ulps(a: f64, b: f64) -> u64 {
    if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
        0
    } else {
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
    }
}
fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map_or(2_000_000, |s| s.parse().unwrap());
    let mut s = 0x9e3779b97f4a7c15u64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let f64s: Vec<f64> = (0..n)
        .map(|_| {
            let u = (r() >> 11) as f64 / (1u64 << 53) as f64;
            (u - 0.5) * 2.0 * 10f64.powi((r() % 7) as i32 - 2)
        })
        .collect();
    let tests: Vec<(&str, Box<dyn Fn(f64) -> (f64, f64)>)> = vec![
        ("exp", Box::new(|x| (rlx_libm::exp(x), x.exp()))),
        ("log", Box::new(|x| (rlx_libm::log(x.abs()), x.abs().ln()))),
        (
            "log2",
            Box::new(|x| (rlx_libm::log2(x.abs()), x.abs().log2())),
        ),
        ("sin", Box::new(|x| (rlx_libm::sin(x), x.sin()))),
        ("cos", Box::new(|x| (rlx_libm::cos(x), x.cos()))),
        ("tan", Box::new(|x| (rlx_libm::tan(x), x.tan()))),
        (
            "pow",
            Box::new(|x| (rlx_libm::pow(x.abs(), 1.37), x.abs().powf(1.37))),
        ),
        ("expm1", Box::new(|x| (rlx_libm::expm1(x), x.exp_m1()))),
        (
            "log1p",
            Box::new(|x| (rlx_libm::log1p(x.abs()), x.abs().ln_1p())),
        ),
        (
            "erfc",
            Box::new(|x| (rlx_libm::erfc(x), rlx_libm::reference::erfc(x))),
        ),
        (
            "expf",
            Box::new(|x| (rlx_libm::expf(x as f32) as f64, (x as f32).exp() as f64)),
        ),
        (
            "logf",
            Box::new(|x| {
                (
                    rlx_libm::logf(x.abs() as f32) as f64,
                    (x.abs() as f32).ln() as f64,
                )
            }),
        ),
    ];
    for (name, f) in &tests {
        let (mut bad, mut worst) = (0usize, 0u64);
        for &x in &f64s {
            let (a, b) = f(x);
            let u = if name.ends_with('f') {
                let (a, b) = (a as f32, b as f32);
                if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
                    0
                } else {
                    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
                }
            } else {
                ulps(a, b)
            };
            if u > 0 {
                bad += 1;
                worst = worst.max(u);
            }
        }
        println!(
            "{name:6} {bad:8}/{n} differ ({:.4}%), max {worst} ulp",
            100.0 * bad as f64 / n as f64
        );
    }
}
