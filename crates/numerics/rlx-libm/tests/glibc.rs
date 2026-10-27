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

fn check(
    name: &str,
    ours: fn(f64) -> f64,
    glibc: fn(f64) -> f64,
    sample: &mut dyn FnMut(&mut R) -> f64,
    n: usize,
) {
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
    check(
        "exp",
        rlx_libm::exp,
        f64::exp,
        &mut |r| -50.0 + 100.0 * r.u(),
        N,
    );
    check(
        "exp wide",
        rlx_libm::exp,
        f64::exp,
        &mut |r| -745.0 + 1455.0 * r.u(),
        N,
    );
    check(
        "exp bits",
        rlx_libm::exp,
        f64::exp,
        &mut |r| f64::from_bits(r.bits()),
        N,
    );
}

#[test]
fn log_matches_glibc() {
    check(
        "log",
        rlx_libm::log,
        f64::ln,
        &mut |r| 1e-6 + 1e6 * r.u(),
        N,
    );
    check(
        "log near 1",
        rlx_libm::log,
        f64::ln,
        &mut |r| 0.9 + 0.2 * r.u(),
        N,
    );
    check(
        "log bits",
        rlx_libm::log,
        f64::ln,
        &mut |r| f64::from_bits(r.bits() >> 1),
        N,
    );
}

#[test]
fn log2_matches_glibc() {
    check(
        "log2",
        rlx_libm::log2,
        f64::log2,
        &mut |r| 1e-6 + 1e6 * r.u(),
        N,
    );
    check(
        "log2 near 1",
        rlx_libm::log2,
        f64::log2,
        &mut |r| 0.94 + 0.12 * r.u(),
        N,
    );
    check(
        "log2 bits",
        rlx_libm::log2,
        f64::log2,
        &mut |r| f64::from_bits(r.bits() >> 1),
        N,
    );
}

fn check2(name: &str, sample: &mut dyn FnMut(&mut R) -> (f64, f64), n: usize) {
    let mut r = R(0x51ed270b27e6a4a5 ^ name.len() as u64);
    let (mut bad, mut first) = (0usize, None);
    for _ in 0..n {
        let (x, y) = sample(&mut r);
        let (a, b) = (
            rlx_libm::pow(x, y),
            std::hint::black_box(x).powf(std::hint::black_box(y)),
        );
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
    check2(
        "pow",
        &mut |r| (0.01 + 10.0 * r.u(), -5.0 + 10.0 * r.u()),
        N,
    );
    check2(
        "pow wide",
        &mut |r| (r.u() * 1e3, -300.0 + 600.0 * r.u()),
        N,
    );
    check2(
        "pow neg int",
        &mut |r| (-(0.5 + 3.0 * r.u()), (-40.0 + 80.0 * r.u()).round()),
        N,
    );
    check2(
        "pow bits",
        &mut |r| (f64::from_bits(r.bits()), f64::from_bits(r.bits())),
        N,
    );
    check2(
        "pow near 1",
        &mut |r| (1.0 + (r.u() - 0.5) * 1e-3, -1e4 + 2e4 * r.u()),
        N,
    );
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
    check(
        "exp boundaries",
        rlx_libm::exp,
        f64::exp,
        &mut |_| it.next().unwrap(),
        190000 * 65,
    );
}

#[test]
fn log2_near_one_dense() {
    check(
        "log2 near 1 dense",
        rlx_libm::log2,
        f64::log2,
        &mut |r| 0.9575 + 0.085 * r.u(),
        200_000_000,
    );
}

#[test]
fn sin_cos_match_glibc() {
    // every branch: |x| < 2^-26, Taylor (< 0.126), table (< 0.855), π/2 − |x| (< 2.43),
    // the 136-bit reduction (< 105414350), `__branred` beyond
    for (name, lo, hi) in [
        ("tiny", 1e-9, 0.126),
        ("table", 0.126, 0.855469),
        ("near pi/2", 0.855469, 2.426265),
        ("reduce", 2.426265, 1000.0),
        ("reduce wide", 1000.0, 1.0e8),
    ] {
        for (f, ours, theirs) in [
            (
                "sin",
                rlx_libm::sin as fn(f64) -> f64,
                f64::sin as fn(f64) -> f64,
            ),
            ("cos", rlx_libm::cos, f64::cos),
        ] {
            let tag = format!("{f} {name}");
            check(
                &tag,
                ours,
                theirs,
                &mut |r| {
                    let x = lo + (hi - lo) * r.u();
                    if r.u() < 0.5 { -x } else { x }
                },
                N / 4,
            );
        }
    }
    check(
        "sin bits",
        rlx_libm::sin,
        f64::sin,
        &mut |r| f64::from_bits(r.bits()),
        N / 2,
    );
    check(
        "cos bits",
        rlx_libm::cos,
        f64::cos,
        &mut |r| f64::from_bits(r.bits()),
        N / 2,
    );
}

unsafe extern "C" {
    #[link_name = "erfc"]
    fn glibc_erfc(x: f64) -> f64;
}

fn c_erfc(x: f64) -> f64 {
    // SAFETY: a pure libm function of one double.
    unsafe { glibc_erfc(x) }
}

#[test]
fn erfc_matches_glibc() {
    for (name, lo, hi) in [
        ("negative", -6.0, 0.0),
        ("small", 0.0, 0.0625),
        ("erf poly", 0.0625, 2.88),
        ("asymptotic", 2.88, 27.3),
        ("subnormal tail", 26.0, 27.3),
    ] {
        check(
            &format!("erfc {name}"),
            rlx_libm::erfc,
            c_erfc,
            &mut |r| lo + (hi - lo) * r.u(),
            N / 4,
        );
    }
    check(
        "erfc bits",
        rlx_libm::erfc,
        c_erfc,
        &mut |r| f64::from_bits(r.bits()),
        N / 4,
    );
    check(
        "erfc tiny",
        rlx_libm::erfc,
        c_erfc,
        &mut |r| f64::from_bits(r.bits() >> 2),
        N / 8,
    );
}

#[test]
fn erfc_hard_cases_match_glibc() {
    let xs = rlx_libm::erfc_hard_cases();
    let bad: Vec<f64> = xs
        .iter()
        .copied()
        .filter(|&x| rlx_libm::erfc(x).to_bits() != c_erfc(x).to_bits())
        .collect();
    assert!(
        xs.len() > 300 && bad.is_empty(),
        "{} of {} differ: {bad:?}",
        bad.len(),
        xs.len()
    );
}

#[test]
fn expm1_log1p_match_glibc() {
    for (name, lo, hi) in [
        ("small", -0.35, 0.35),
        ("k=±1", -1.1, 1.1),
        ("mid", -40.0, 40.0),
        ("wide", -50.0, 709.7),
    ] {
        check(
            &format!("expm1 {name}"),
            rlx_libm::expm1,
            f64::exp_m1,
            &mut |r| lo + (hi - lo) * r.u(),
            N / 4,
        );
    }
    check(
        "expm1 bits",
        rlx_libm::expm1,
        f64::exp_m1,
        &mut |r| f64::from_bits(r.bits()),
        N / 4,
    );
    for (name, lo, hi) in [
        ("tiny", -1e-8, 1e-8),
        ("near 0", -0.3, 0.42),
        ("mid", -0.999, 10.0),
        ("wide", 10.0, 1e300),
    ] {
        check(
            &format!("log1p {name}"),
            rlx_libm::log1p,
            f64::ln_1p,
            &mut |r| lo + (hi - lo) * r.u(),
            N / 4,
        );
    }
    check(
        "log1p bits",
        rlx_libm::log1p,
        f64::ln_1p,
        &mut |r| f64::from_bits(r.bits() >> 1),
        N / 4,
    );
}

#[test]
fn tan_hypot_match_glibc() {
    for (name, lo, hi) in [
        ("tiny", 1e-9, 0.0608),
        ("table", 0.0608, 0.787),
        ("reduce i", 0.787, 25.0),
        ("reduce ii", 25.0, 1.0e8),
    ] {
        check(
            &format!("tan {name}"),
            rlx_libm::tan,
            f64::tan,
            &mut |r| {
                let x = lo + (hi - lo) * r.u();
                if r.u() < 0.5 { -x } else { x }
            },
            N / 4,
        );
    }
    check(
        "tan bits",
        rlx_libm::tan,
        f64::tan,
        &mut |r| f64::from_bits(r.bits()),
        N / 4,
    );
    check2_named(
        "hypot",
        &mut |r| (f64::from_bits(r.bits()), f64::from_bits(r.bits())),
        N / 4,
    );
    check2_named(
        "hypot near",
        &mut |r| {
            let a = 1e3 * (r.u() - 0.5);
            (a, a * (0.5 + r.u()))
        },
        N / 4,
    );
}

fn check2_named(name: &str, sample: &mut dyn FnMut(&mut R) -> (f64, f64), n: usize) {
    let mut r = R(0x7f4a7c15 ^ name.len() as u64);
    let (mut bad, mut first) = (0usize, None);
    for _ in 0..n {
        let (x, y) = sample(&mut r);
        let (a, b) = (rlx_libm::hypot(x, y), std::hint::black_box(x).hypot(y));
        if a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()) {
            bad += 1;
            first.get_or_insert((x, y, a, b));
        }
    }
    eprintln!("{name}: {bad}/{n} differ, first {first:?}");
    assert_eq!(bad, 0, "{name}");
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Cplx {
    re: f64,
    im: f64,
}

unsafe extern "C" {
    fn cexp(z: Cplx) -> Cplx;
    fn csqrt(z: Cplx) -> Cplx;
}

#[test]
fn cexp_csqrt_match_glibc() {
    let mut r = R(0xc0ffee);
    let (mut bad_e, mut bad_s, mut first) = (0usize, 0usize, None);
    let n = N / 4;
    for i in 0..n {
        let (re, im) = if i % 2 == 0 {
            (40.0 * (r.u() - 0.5), 20.0 * (r.u() - 0.5))
        } else {
            (f64::from_bits(r.bits()), f64::from_bits(r.bits()))
        };
        let z = Cplx { re, im };
        // SAFETY: pure libm functions of a complex double.
        let (ge, gs) = unsafe { (cexp(z), csqrt(z)) };
        let (oe, os) = (rlx_libm::cexp(re, im), rlx_libm::csqrt(re, im));
        let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        if !(same(oe.0, ge.re) && same(oe.1, ge.im)) {
            bad_e += 1;
            first.get_or_insert(("cexp", re, im));
        }
        if !(same(os.0, gs.re) && same(os.1, gs.im)) {
            bad_s += 1;
            first.get_or_insert(("csqrt", re, im));
        }
    }
    eprintln!("cexp {bad_e}/{n}, csqrt {bad_s}/{n} differ, first {first:?}");
    assert!(bad_e == 0 && bad_s == 0);
}

unsafe extern "C" {
    fn sincos(x: f64, s: *mut f64, c: *mut f64);
}

/// glibc's merged `sincos` (what gcc emits for `sin`+`cos` of one argument) equals
/// the separate ports: the same reduction and kernels.
#[test]
fn sincos_equals_sin_and_cos() {
    let mut r = R(0x5c05);
    let mut bad = 0usize;
    for i in 0..N / 4 {
        let x = if i % 2 == 0 {
            100.0 * (r.u() - 0.5)
        } else {
            f64::from_bits(r.bits())
        };
        let (mut s, mut c) = (0.0, 0.0);
        // SAFETY: libm's `sincos` writes the two outputs.
        unsafe { sincos(x, &mut s, &mut c) };
        let (a, b) = (rlx_libm::sin(x), rlx_libm::cos(x));
        if !((a.to_bits() == s.to_bits() || (a.is_nan() && s.is_nan()))
            && (b.to_bits() == c.to_bits() || (b.is_nan() && c.is_nan())))
        {
            bad += 1;
        }
    }
    assert_eq!(bad, 0);
}
