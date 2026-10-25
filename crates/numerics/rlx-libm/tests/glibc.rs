//! Against the platform's glibc itself (x86_64 Linux on a CPU with FMA, where
//! glibc runs the `*_fma` variants these model). Zero tolerance.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

struct R(u64);
impl R {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn bits(&mut self) -> u64 {
        self.u();
        self.0
    }
}

fn check(name: &str, ours: fn(f64) -> f64, glibc: fn(f64) -> f64, sample: &mut dyn FnMut(&mut R) -> f64, n: usize) {
    let mut r = R(0x9e3779b97f4a7c15 ^ name.len() as u64);
    let mut bad = 0usize;
    let mut first = None;
    for _ in 0..n {
        let x = sample(&mut r);
        let (a, b) = (ours(x), glibc(std::hint::black_box(x)));
        if a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()) {
            bad += 1;
            first.get_or_insert((x, a, b));
        }
    }
    eprintln!("{name}: {bad}/{n} differ, first {first:?}");
    assert_eq!(bad, 0, "{name}: {bad}/{n} differ, first {first:?}");
}

const N: usize = 20_000_000;

#[test]
fn exp_matches_glibc() {
    check("exp", rlx_libm::exp, f64::exp, &mut |r| -50.0 + 100.0 * r.u(), N);
    check("exp wide", rlx_libm::exp, f64::exp, &mut |r| -745.0 + 1455.0 * r.u(), N);
    check("exp bits", rlx_libm::exp, f64::exp, &mut |r| f64::from_bits(r.bits()), N);
}

#[test]
fn log_matches_glibc() {
    check("log", rlx_libm::log, f64::ln, &mut |r| 1e-6 + 1e6 * r.u(), N);
    check("log near 1", rlx_libm::log, f64::ln, &mut |r| 0.9 + 0.2 * r.u(), N);
    check("log bits", rlx_libm::log, f64::ln, &mut |r| f64::from_bits(r.bits() >> 1), N);
}

#[test]
fn log2_matches_glibc() {
    check("log2", rlx_libm::log2, f64::log2, &mut |r| 1e-6 + 1e6 * r.u(), N);
    check("log2 near 1", rlx_libm::log2, f64::log2, &mut |r| 0.94 + 0.12 * r.u(), N);
    check("log2 bits", rlx_libm::log2, f64::log2, &mut |r| f64::from_bits(r.bits() >> 1), N);
}

fn check2(name: &str, sample: &mut dyn FnMut(&mut R) -> (f64, f64), n: usize) {
    let mut r = R(0x51ed270b27e6a4a5 ^ name.len() as u64);
    let (mut bad, mut first) = (0usize, None);
    for _ in 0..n {
        let (x, y) = sample(&mut r);
        let (a, b) = (rlx_libm::pow(x, y), std::hint::black_box(x).powf(std::hint::black_box(y)));
        if a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()) {
            bad += 1;
            first.get_or_insert((x, y, a, b));
        }
    }
    eprintln!("{name}: {bad}/{n} differ, first {first:?}");
    assert_eq!(bad, 0, "{name}: {bad}/{n} differ, first {first:?}");
}

#[test]
fn pow_matches_glibc() {
    check2("pow", &mut |r| (0.01 + 10.0 * r.u(), -5.0 + 10.0 * r.u()), N);
    check2("pow wide", &mut |r| (r.u() * 1e3, -300.0 + 600.0 * r.u()), N);
    check2("pow neg int", &mut |r| (-(0.5 + 3.0 * r.u()), (-40.0 + 80.0 * r.u()).round()), N);
    check2("pow bits", &mut |r| (f64::from_bits(r.bits()), f64::from_bits(r.bits())), N);
    check2("pow near 1", &mut |r| (1.0 + (r.u() - 0.5) * 1e-3, -1e4 + 2e4 * r.u()), N);
}

/// `exp`'s `kd = InvLn2N*x + Shift` (fused or not) only matters when `InvLn2N*x`
/// is within an ulp of a half-integer: aim at every such boundary.
#[test]
fn exp_at_rounding_boundaries() {
    let inv = f64::from_bits(0x3ff71547652b82fe) * 128.0;
    let mut xs = Vec::new();
    for k in -95000i64..95000 {
        let x0 = (k as f64 + 0.5) / inv;
        for o in -32i64..=32 {
            xs.push(f64::from_bits((x0.to_bits() as i64 + o) as u64));
        }
    }
    let mut it = xs.into_iter();
    check("exp boundaries", rlx_libm::exp, f64::exp, &mut |_| it.next().unwrap(), 190000 * 65);
}

#[test]
fn log2_near_one_dense() {
    check("log2 near 1 dense", rlx_libm::log2, f64::log2, &mut |r| 0.9575 + 0.085 * r.u(), 200_000_000);
}
