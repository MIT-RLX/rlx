// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Double-word extended precision** — emulate a float the hardware lacks with
//! a pair `(hi, lo)` of the widest float it *does* have, via error-free
//! transformations (EFT). The headline case is **double-single** (2× f32 ≈ f64,
//! ~49 significand bits): Metal has no `f64` at all, so this is the only way to
//! get near-f64 precision there.
//!
//! * [`DoubleSingle`] is the host-side reference used to **deconstruct a wide
//!   constant once at bake time** (the host has `f64`); the kernels then do all
//!   arithmetic in the narrow pair, no wide hardware needed.
//! * [`double_single_prelude`] emits the matching on-device library (the EFT
//!   `dw_two_sum`/`dw_two_prod` and the `dw_add`/`dw_sub`/`dw_mul`/`dw_div`/
//!   `dw_sqrt` operators) for each target language — the same algorithm the host
//!   reference runs.
//!
//! ## ⚠ Fast-math hazard
//!
//! EFT relies on IEEE round-to-nearest and on the compiler **not** algebraically
//! reassociating: `bb = s - a; e = (a - (s - bb)) + (b - bb)` collapses to `0`
//! under `-ffast-math` / fp-contraction. **Metal enables fast-math by default**,
//! so a double-word kernel MUST be compiled with precise math
//! (`MTLCompileOptions.fastMathEnabled = false`, or `-fno-fast-math` / NVRTC
//! `--fmad=false` except inside `dw_two_prod`, which needs the fused `fma`).

use crate::Lang;

/// A `(hi, lo)` pair of `f32` representing `hi + lo` to ~49 significand bits.
/// The host reference — matches the emitted GPU prelude bit-for-bit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleSingle {
    pub hi: f32,
    pub lo: f32,
}

#[inline]
fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}
#[inline]
fn quick_two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    (s, b - (s - a))
}
#[inline]
fn two_prod(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl DoubleSingle {
    /// Deconstruct a wide `f64` into the narrow pair (done on the host).
    pub fn from_f64(v: f64) -> Self {
        let hi = v as f32;
        let lo = (v - hi as f64) as f32;
        DoubleSingle { hi, lo }
    }
    /// Reconstruct (for validation / reading results back).
    pub fn to_f64(self) -> f64 {
        self.hi as f64 + self.lo as f64
    }
    pub fn add(self, b: DoubleSingle) -> DoubleSingle {
        let (s1, s2) = two_sum(self.hi, b.hi);
        let (t1, t2) = two_sum(self.lo, b.lo);
        let (h, e) = quick_two_sum(s1, s2 + t1);
        let (h, l) = quick_two_sum(h, e + t2);
        DoubleSingle { hi: h, lo: l }
    }
    pub fn mul(self, b: DoubleSingle) -> DoubleSingle {
        let (p1, p2) = two_prod(self.hi, b.hi);
        let (h, l) = quick_two_sum(p1, p2 + (self.hi * b.lo + self.lo * b.hi));
        DoubleSingle { hi: h, lo: l }
    }
    /// `a - b`, computed as `a + (-b)` — same op sequence as the emitted
    /// `dw_sub`, so host and device agree bit-for-bit.
    pub fn sub(self, b: DoubleSingle) -> DoubleSingle {
        self.add(DoubleSingle {
            hi: -b.hi,
            lo: -b.lo,
        })
    }
    /// `a / b` via the classic long-division refinement (Dekker/QD): a leading
    /// quotient digit plus two residual corrections, giving ~full double-single
    /// precision. Mirrors the emitted `dw_div` step-for-step for bit-exact parity.
    pub fn div(self, b: DoubleSingle) -> DoubleSingle {
        let q1 = self.hi / b.hi;
        let r = self.sub(b.mul(DoubleSingle { hi: q1, lo: 0.0 }));
        let q2 = r.hi / b.hi;
        let r = r.sub(b.mul(DoubleSingle { hi: q2, lo: 0.0 }));
        let q3 = r.hi / b.hi;
        let (h, l) = quick_two_sum(q1, q2);
        DoubleSingle { hi: h, lo: l }.add(DoubleSingle { hi: q3, lo: 0.0 })
    }
    /// `e^a`: `a = k·ln2 + r` with `|r| ≤ ln2/2` (the reduction in double-single),
    /// a degree-13 Taylor polynomial in Horner form ([`exp_coefficients`]),
    /// then `· 2^k`, exact on both halves. f32's exponent range: `+inf` above
    /// 88.72, `0` below -103.9. Mirrors the emitted `dw_exp`.
    pub fn exp(self) -> DoubleSingle {
        if self.hi > EXP_HI {
            return DoubleSingle {
                hi: f32::INFINITY,
                lo: 0.0,
            };
        }
        if self.hi < EXP_LO {
            return DoubleSingle { hi: 0.0, lo: 0.0 };
        }
        let k = (self.hi * std::f32::consts::LOG2_E + 0.5).floor();
        // `r = a - k·ln2` with ln2 in three parts: `k·L1` is exact in f32 (L1
        // has 12 significant bits, |k| ≤ 151), `k·L2` exact as a two_prod.
        let [l1, l2, l3] = ln2_parts();
        let r = self.sub(DoubleSingle {
            hi: k * l1,
            lo: 0.0,
        });
        let (p2, e2) = two_prod(k, l2);
        let r = r.sub(DoubleSingle { hi: p2, lo: e2 });
        let r = r.sub(DoubleSingle {
            hi: k * l3,
            lo: 0.0,
        });
        let c = exp_coefficients();
        let mut p = DoubleSingle {
            hi: c[EXP_DEGREE].0,
            lo: c[EXP_DEGREE].1,
        };
        for &(h, l) in c[..EXP_DEGREE].iter().rev() {
            p = p.mul(r).add(DoubleSingle { hi: h, lo: l });
        }
        let ki = k as i32;
        DoubleSingle {
            hi: ldexp(p.hi, ki),
            lo: ldexp(p.lo, ki),
        }
    }
    /// `√a` via one Newton refinement of the single-precision seed `1/√a.hi`,
    /// reaching ~full double-single precision. Mirrors the emitted `dw_sqrt`.
    pub fn sqrt(self) -> DoubleSingle {
        if self.hi == 0.0 {
            return DoubleSingle { hi: 0.0, lo: 0.0 };
        }
        let x = 1.0 / self.hi.sqrt(); // ~24-bit seed for 1/√a
        let ax = self.hi * x; // ≈ √a
        let (p, e) = two_prod(ax, ax);
        let corr = self.sub(DoubleSingle { hi: p, lo: e }).hi * (x * 0.5);
        let (h, l) = two_sum(ax, corr);
        DoubleSingle { hi: h, lo: l }
    }
}

/// `dw_exp` overflows above this `hi` and returns 0 below [`EXP_LO`].
const EXP_HI: f32 = 88.72;
const EXP_LO: f32 = -103.9;
/// Degree of `dw_exp`'s Taylor polynomial on `|r| ≤ ln2/2` (next term ~2e-16).
const EXP_DEGREE: usize = 13;

/// `x` as an f32 pair `hi + lo`.
fn split(x: f64) -> (f32, f32) {
    let hi = x as f32;
    (hi, (x - hi as f64) as f32)
}

/// ln 2 as `L1 + L2 + L3`: `L1` with 12 significant bits, then f32 roundings.
fn ln2_parts() -> [f32; 3] {
    let ln2 = std::f64::consts::LN_2;
    let l1 = (ln2 * 4096.0).round() / 4096.0;
    let l2 = (ln2 - l1) as f32;
    let l3 = (ln2 - l1 - l2 as f64) as f32;
    [l1 as f32, l2, l3]
}

/// `1/n!` for `n = 0..=EXP_DEGREE`, as f32 pairs.
pub fn exp_coefficients() -> Vec<(f32, f32)> {
    let mut f = 1f64;
    (0..=EXP_DEGREE)
        .map(|n| {
            if n > 0 {
                f *= n as f64;
            }
            split(1.0 / f)
        })
        .collect()
}

/// `x · 2^k` for `k` in `[-150, 128]`, in two exact steps.
fn ldexp(x: f32, k: i32) -> f32 {
    let half = k / 2;
    let p = |e: i32| f32::from_bits(((e + 127) as u32) << 23);
    x * p(half) * p(k - half)
}

/// The `dw_exp` function in `lang`'s syntax (see [`DoubleSingle::exp`]).
fn exp_source(lang: Lang) -> String {
    let ctor = |h: String, l: String| match lang {
        Lang::Msl | Lang::Cuda => format!("DwF32{{{h}, {l}}}"),
        Lang::OpenCl => format!("(DwF32){{{h}, {l}}}"),
        Lang::Glsl | Lang::Wgsl => format!("DwF32({h}, {l})"),
    };
    let lit = |x: f32| {
        let v = format!("{x:e}");
        match lang {
            Lang::Wgsl | Lang::Glsl => v,
            _ => format!("{v}f"),
        }
    };
    let pair = |(h, l): (f32, f32)| ctor(lit(h), lit(l));
    let [l1, l2, l3] = ln2_parts();
    let c = exp_coefficients();
    let top = pair(c[EXP_DEGREE]);
    let horner: String = c[..EXP_DEGREE]
        .iter()
        .rev()
        .map(|&cn| format!("    p = dw_add(dw_mul(p, r), {});\n", pair(cn)))
        .collect();
    let (zero, inf) = (
        ctor(lit(0.0), lit(0.0)),
        ctor("1e30 * 1e30".into(), lit(0.0)),
    );
    match lang {
        Lang::Wgsl => format!(
            "fn dw_ldexp2(x: f32, k: i32) -> f32 {{\n\
             \x20   let h = k / 2;\n\
             \x20   return x * bitcast<f32>(u32(h + 127) << 23u) * bitcast<f32>(u32(k - h + 127) << 23u);\n}}\n\
             fn dw_exp(a: DwF32) -> DwF32 {{\n\
             \x20   if (a.hi > {hi}) {{ return DwF32(bitcast<f32>(0x7f800000u), 0.0); }}\n\
             \x20   if (a.hi < {lo}) {{ return {zero}; }}\n\
             \x20   let k = floor(a.hi * {log2e} + 0.5);\n\
             \x20   let r1 = dw_sub(a, DwF32(k * {l1}, 0.0));\n\
             \x20   let r2 = dw_sub(r1, dw_two_prod(k, {l2}));\n\
             \x20   let r = dw_sub(r2, DwF32(k * {l3}, 0.0));\n\
             \x20   var p = {top};\n{horner}\
             \x20   let ki = i32(k);\n\
             \x20   return DwF32(dw_ldexp2(p.hi, ki), dw_ldexp2(p.lo, ki));\n}}\n",
            hi = lit(EXP_HI),
            lo = lit(EXP_LO),
            log2e = lit(std::f32::consts::LOG2_E),
            l1 = lit(l1),
            l2 = lit(l2),
            l3 = lit(l3),
        ),
        _ => {
            let (q, int, floor, as_float) = match lang {
                Lang::Cuda => (
                    "__device__ __forceinline__",
                    "int",
                    "floorf",
                    "__uint_as_float",
                ),
                Lang::Msl => ("inline", "int", "floor", "as_type<float>"),
                Lang::Glsl => ("", "int", "floor", "uintBitsToFloat"),
                _ => ("inline", "int", "floor", "as_float"),
            };
            let uint = if matches!(lang, Lang::Glsl) {
                "uint"
            } else {
                "unsigned int"
            };
            let ucast = |e: &str| match lang {
                Lang::Glsl => format!("uint({e})"),
                _ => format!("({uint})({e})"),
            };
            let icast = match lang {
                Lang::Glsl => "int(k)".to_string(),
                _ => format!("({int})k"),
            };
            let _ = inf;
            format!(
                "{q} float dw_ldexp2(float x, {int} k) {{\n\
                 \x20   {int} h = k / 2;\n\
                 \x20   return x * {as_float}({u1} << 23) * {as_float}({u2} << 23);\n}}\n\
                 {q} DwF32 dw_exp(DwF32 a) {{\n\
                 \x20   if (a.hi > {hi}) return {inf_c};\n\
                 \x20   if (a.hi < {lo}) return {zero};\n\
                 \x20   float k = {floor}(a.hi * {log2e} + {half});\n\
                 \x20   DwF32 r1 = dw_sub(a, {k1});\n\
                 \x20   DwF32 r2 = dw_sub(r1, dw_two_prod(k, {l2}));\n\
                 \x20   DwF32 r = dw_sub(r2, {k3});\n\
                 \x20   DwF32 p = {top};\n{horner}\
                 \x20   {int} ki = {icast};\n\
                 \x20   return {out};\n}}\n",
                u1 = ucast("h + 127"),
                u2 = ucast("k - h + 127"),
                hi = lit(EXP_HI),
                lo = lit(EXP_LO),
                half = lit(0.5),
                log2e = lit(std::f32::consts::LOG2_E),
                inf_c = ctor(format!("{as_float}(0x7f800000u)"), lit(0.0)),
                k1 = ctor(format!("k * {}", lit(l1)), lit(0.0)),
                k3 = ctor(format!("k * {}", lit(l3)), lit(0.0)),
                l2 = lit(l2),
                out = ctor("dw_ldexp2(p.hi, ki)".into(), "dw_ldexp2(p.lo, ki)".into()),
            )
        }
    }
}

/// The WGSL double-single prelude (fixed source of record for the algorithm).
const WGSL: &str = r#"// @generated by rlxsl::dw — double-single (2x f32 ≈ f64).
// COMPILE WITH PRECISE MATH (no fast-math): EFT breaks under reassociation.
struct DwF32 { hi: f32, lo: f32 }
fn dw_two_sum(a: f32, b: f32) -> DwF32 {
    let s = a + b; let bb = s - a;
    return DwF32(s, (a - (s - bb)) + (b - bb));
}
fn dw_quick_two_sum(a: f32, b: f32) -> DwF32 {
    let s = a + b;
    return DwF32(s, b - (s - a));
}
fn dw_two_prod(a: f32, b: f32) -> DwF32 {
    let p = a * b;
    return DwF32(p, fma(a, b, -p));
}
fn dw_add(a: DwF32, b: DwF32) -> DwF32 {
    let s = dw_two_sum(a.hi, b.hi);
    let t = dw_two_sum(a.lo, b.lo);
    let r1 = dw_quick_two_sum(s.hi, s.lo + t.hi);
    let r2 = dw_quick_two_sum(r1.hi, r1.lo + t.lo);
    return r2;
}
fn dw_mul(a: DwF32, b: DwF32) -> DwF32 {
    let p = dw_two_prod(a.hi, b.hi);
    return dw_quick_two_sum(p.hi, p.lo + (a.hi * b.lo + a.lo * b.hi));
}
fn dw_sub(a: DwF32, b: DwF32) -> DwF32 {
    return dw_add(a, DwF32(-b.hi, -b.lo));
}
fn dw_div(a: DwF32, b: DwF32) -> DwF32 {
    let q1 = a.hi / b.hi;
    let r1 = dw_sub(a, dw_mul(b, DwF32(q1, 0.0)));
    let q2 = r1.hi / b.hi;
    let r2 = dw_sub(r1, dw_mul(b, DwF32(q2, 0.0)));
    let q3 = r2.hi / b.hi;
    let s = dw_quick_two_sum(q1, q2);
    return dw_add(s, DwF32(q3, 0.0));
}
fn dw_sqrt(a: DwF32) -> DwF32 {
    if (a.hi == 0.0) { return DwF32(0.0, 0.0); }
    let x = inverseSqrt(a.hi);
    let ax = a.hi * x;
    let axsq = dw_two_prod(ax, ax);
    let corr = dw_sub(a, axsq).hi * (x * 0.5);
    return dw_two_sum(ax, corr);
}
"#;

/// C-family (MSL / CUDA / GLSL) prelude: same algorithm, differing only in the
/// function qualifier, the `fma`/`rsqrt` spellings, and the struct-literal
/// delimiters.
fn c_family(qualifier: &str, fma: &str, rsqrt: &str, lo: char, lc: char) -> String {
    format!(
        r#"// @generated by rlxsl::dw — double-single (2x f32 ≈ f64).
// COMPILE WITH PRECISE MATH (no fast-math): EFT breaks under reassociation.
struct DwF32 {{ float hi; float lo; }};
{q} DwF32 dw_two_sum(float a, float b) {{
    float s = a + b; float bb = s - a;
    return DwF32{lo}s, (a - (s - bb)) + (b - bb){lc};
}}
{q} DwF32 dw_quick_two_sum(float a, float b) {{
    float s = a + b;
    return DwF32{lo}s, b - (s - a){lc};
}}
{q} DwF32 dw_two_prod(float a, float b) {{
    float p = a * b;
    return DwF32{lo}p, {fma}(a, b, -p){lc};
}}
{q} DwF32 dw_add(DwF32 a, DwF32 b) {{
    DwF32 s = dw_two_sum(a.hi, b.hi);
    DwF32 t = dw_two_sum(a.lo, b.lo);
    DwF32 r1 = dw_quick_two_sum(s.hi, s.lo + t.hi);
    return dw_quick_two_sum(r1.hi, r1.lo + t.lo);
}}
{q} DwF32 dw_mul(DwF32 a, DwF32 b) {{
    DwF32 p = dw_two_prod(a.hi, b.hi);
    return dw_quick_two_sum(p.hi, p.lo + (a.hi * b.lo + a.lo * b.hi));
}}
{q} DwF32 dw_sub(DwF32 a, DwF32 b) {{
    return dw_add(a, DwF32{lo}-b.hi, -b.lo{lc});
}}
{q} DwF32 dw_div(DwF32 a, DwF32 b) {{
    float q1 = a.hi / b.hi;
    DwF32 r1 = dw_sub(a, dw_mul(b, DwF32{lo}q1, 0.0f{lc}));
    float q2 = r1.hi / b.hi;
    DwF32 r2 = dw_sub(r1, dw_mul(b, DwF32{lo}q2, 0.0f{lc}));
    float q3 = r2.hi / b.hi;
    DwF32 s = dw_quick_two_sum(q1, q2);
    return dw_add(s, DwF32{lo}q3, 0.0f{lc});
}}
{q} DwF32 dw_sqrt(DwF32 a) {{
    if (a.hi == 0.0f) return DwF32{lo}0.0f, 0.0f{lc};
    float x = {rsqrt}(a.hi);
    float ax = a.hi * x;
    DwF32 axsq = dw_two_prod(ax, ax);
    float corr = dw_sub(a, axsq).hi * (x * 0.5f);
    return dw_two_sum(ax, corr);
}}
"#,
        q = qualifier,
        fma = fma,
        rsqrt = rsqrt,
        lo = lo,
        lc = lc,
    )
}

/// OpenCL-C (C99): typedef + compound-literal struct init `(DwF32){h, l}`.
const OPENCL: &str = r#"// @generated by rlxsl::dw — double-single (2x f32 ≈ f64).
// COMPILE WITH PRECISE MATH (no fast-math): EFT breaks under reassociation.
typedef struct { float hi; float lo; } DwF32;
inline DwF32 dw_two_sum(float a, float b) {
    float s = a + b; float bb = s - a;
    return (DwF32){s, (a - (s - bb)) + (b - bb)};
}
inline DwF32 dw_quick_two_sum(float a, float b) {
    float s = a + b;
    return (DwF32){s, b - (s - a)};
}
inline DwF32 dw_two_prod(float a, float b) {
    float p = a * b;
    return (DwF32){p, fma(a, b, -p)};
}
inline DwF32 dw_add(DwF32 a, DwF32 b) {
    DwF32 s = dw_two_sum(a.hi, b.hi);
    DwF32 t = dw_two_sum(a.lo, b.lo);
    DwF32 r1 = dw_quick_two_sum(s.hi, s.lo + t.hi);
    return dw_quick_two_sum(r1.hi, r1.lo + t.lo);
}
inline DwF32 dw_mul(DwF32 a, DwF32 b) {
    DwF32 p = dw_two_prod(a.hi, b.hi);
    return dw_quick_two_sum(p.hi, p.lo + (a.hi * b.lo + a.lo * b.hi));
}
inline DwF32 dw_sub(DwF32 a, DwF32 b) {
    return dw_add(a, (DwF32){-b.hi, -b.lo});
}
inline DwF32 dw_div(DwF32 a, DwF32 b) {
    float q1 = a.hi / b.hi;
    DwF32 r1 = dw_sub(a, dw_mul(b, (DwF32){q1, 0.0f}));
    float q2 = r1.hi / b.hi;
    DwF32 r2 = dw_sub(r1, dw_mul(b, (DwF32){q2, 0.0f}));
    float q3 = r2.hi / b.hi;
    DwF32 s = dw_quick_two_sum(q1, q2);
    return dw_add(s, (DwF32){q3, 0.0f});
}
inline DwF32 dw_sqrt(DwF32 a) {
    if (a.hi == 0.0f) return (DwF32){0.0f, 0.0f};
    float x = rsqrt(a.hi);
    float ax = a.hi * x;
    DwF32 axsq = dw_two_prod(ax, ax);
    float corr = dw_sub(a, axsq).hi * (x * 0.5f);
    return dw_two_sum(ax, corr);
}
"#;

/// The double-single arithmetic library for `lang` — `dw_two_sum`,
/// `dw_two_prod`, `dw_add`, `dw_sub`, `dw_mul`, `dw_div`, `dw_sqrt` over a
/// `DwF32 { hi, lo }` pair.
/// WGSL double-single prelude for drivers that compile with **fast math on**
/// and offer no switch — wgpu on Metal (wgpu-hal passes default
/// `MTLCompileOptions`, i.e. fast math) is the case in point.
///
/// Every intermediate passes through `dw_opq`, an XOR with a zero the compiler
/// cannot see (`dw_zero()`, which the including shader defines to read a
/// uniform). Each operation then has opaque operands, so neither identities
/// (`(a + b) - a → b`, which zero the error terms) nor reassociation across
/// three terms (which re-rounds them) can apply. At run time `dw_opq` is the
/// identity: results match [`DoubleSingle`] and the precise prelude.
///
/// The including shader must define `fn dw_zero() -> u32` returning 0 from a
/// runtime value (e.g. a uniform field always set to 0).
pub const WGSL_FAST_MATH_GUARDED: &str = r#"// @generated by rlxsl::dw — double-single, fast-math guarded.
struct DwF32 { hi: f32, lo: f32 }
fn dw_opq(x: f32) -> f32 { return bitcast<f32>(bitcast<u32>(x) ^ dw_zero()); }
fn dw_two_sum(a: f32, b: f32) -> DwF32 {
    let s = dw_opq(a + b);
    let bb = dw_opq(s - a);
    let av = dw_opq(s - bb);
    return DwF32(s, dw_opq(a - av) + dw_opq(b - bb));
}
fn dw_quick_two_sum(a: f32, b: f32) -> DwF32 {
    let s = dw_opq(a + b);
    return DwF32(s, b - dw_opq(s - a));
}
fn dw_two_prod(a: f32, b: f32) -> DwF32 {
    let p = dw_opq(a * b);
    return DwF32(p, fma(a, b, -p));
}
fn dw_add(a: DwF32, b: DwF32) -> DwF32 {
    let s = dw_two_sum(a.hi, b.hi);
    let t = dw_two_sum(a.lo, b.lo);
    let r1 = dw_quick_two_sum(s.hi, dw_opq(s.lo + t.hi));
    return dw_quick_two_sum(r1.hi, dw_opq(r1.lo + t.lo));
}
fn dw_mul(a: DwF32, b: DwF32) -> DwF32 {
    let p = dw_two_prod(a.hi, b.hi);
    let cross = dw_opq(dw_opq(a.hi * b.lo) + dw_opq(a.lo * b.hi));
    return dw_quick_two_sum(p.hi, dw_opq(p.lo + cross));
}
fn dw_sub(a: DwF32, b: DwF32) -> DwF32 {
    return dw_add(a, DwF32(-b.hi, -b.lo));
}
fn dw_div(a: DwF32, b: DwF32) -> DwF32 {
    let q1 = dw_opq(a.hi / b.hi);
    let r1 = dw_sub(a, dw_mul(b, DwF32(q1, 0.0)));
    let q2 = dw_opq(r1.hi / b.hi);
    let r2 = dw_sub(r1, dw_mul(b, DwF32(q2, 0.0)));
    let q3 = dw_opq(r2.hi / b.hi);
    let s = dw_quick_two_sum(q1, q2);
    return dw_add(s, DwF32(q3, 0.0));
}
"#;

pub fn double_single_prelude(lang: Lang) -> String {
    let base = match lang {
        Lang::Wgsl => WGSL.to_string(),
        // GLSL builds structs with constructor syntax `DwF32(h, l)`.
        Lang::Glsl => c_family("", "fma", "inversesqrt", '(', ')'),
        // MSL / CUDA use C++ aggregate init `DwF32{h, l}`.
        Lang::Msl => c_family("inline", "fma", "rsqrt", '{', '}'),
        Lang::Cuda => c_family("__device__ __forceinline__", "fmaf", "rsqrtf", '{', '}'),
        // OpenCL-C: typedef + C99 compound literals.
        Lang::OpenCl => OPENCL.to_string(),
    };
    base + &exp_source(lang)
}

/// [`WGSL_FAST_MATH_GUARDED`] plus `dw_sqrt` and `dw_exp` in the same guarded
/// style (every rounding step through `dw_opq`).
pub fn wgsl_fast_math_guarded() -> String {
    format!(
        "{WGSL_FAST_MATH_GUARDED}{WGSL_GUARDED_SQRT}{}",
        exp_source(Lang::Wgsl)
    )
}

/// GLSL counterpart of [`wgsl_fast_math_guarded`], for Vulkan drivers that
/// compile with fast math (MoltenVK). Every rounding step passes through
/// `dw_opq`, an XOR with the shader's `uint dw_zero()` (a runtime 0 the
/// compiler cannot see; the including shader defines it before this source).
/// `dw_two_prod` uses Dekker's split rather than `fma`, which GLSL without
/// `precise` may evaluate as a separate multiply and add.
pub fn glsl_fast_math_guarded() -> String {
    format!("{GLSL_GUARDED}{}", exp_source(Lang::Glsl))
}

const GLSL_GUARDED: &str = r#"// @generated by rlxsl::dw — double-single, fast-math guarded (GLSL).
struct DwF32 { float hi; float lo; };
float dw_opq(float x) { return uintBitsToFloat(floatBitsToUint(x) ^ dw_zero()); }
DwF32 dw_two_sum(float a, float b) {
    float s = dw_opq(a + b);
    float bb = dw_opq(s - a);
    float av = dw_opq(s - bb);
    return DwF32(s, dw_opq(a - av) + dw_opq(b - bb));
}
DwF32 dw_quick_two_sum(float a, float b) {
    float s = dw_opq(a + b);
    return DwF32(s, b - dw_opq(s - a));
}
// Veltkamp split: a = ah + al with 12 significant bits each.
DwF32 dw_split(float a) {
    float c = dw_opq(4097.0 * a);
    float ah = dw_opq(c - dw_opq(c - a));
    return DwF32(ah, dw_opq(a - ah));
}
DwF32 dw_two_prod(float a, float b) {
    float p = dw_opq(a * b);
    DwF32 x = dw_split(a);
    DwF32 y = dw_split(b);
    float e = dw_opq(dw_opq(x.hi * y.hi) - p);
    e = dw_opq(e + dw_opq(x.hi * y.lo));
    e = dw_opq(e + dw_opq(x.lo * y.hi));
    e = dw_opq(e + dw_opq(x.lo * y.lo));
    return DwF32(p, e);
}
DwF32 dw_add(DwF32 a, DwF32 b) {
    DwF32 s = dw_two_sum(a.hi, b.hi);
    DwF32 t = dw_two_sum(a.lo, b.lo);
    DwF32 r1 = dw_quick_two_sum(s.hi, dw_opq(s.lo + t.hi));
    return dw_quick_two_sum(r1.hi, dw_opq(r1.lo + t.lo));
}
DwF32 dw_mul(DwF32 a, DwF32 b) {
    DwF32 p = dw_two_prod(a.hi, b.hi);
    float cross = dw_opq(dw_opq(a.hi * b.lo) + dw_opq(a.lo * b.hi));
    return dw_quick_two_sum(p.hi, dw_opq(p.lo + cross));
}
DwF32 dw_sub(DwF32 a, DwF32 b) {
    return dw_add(a, DwF32(-b.hi, -b.lo));
}
DwF32 dw_div(DwF32 a, DwF32 b) {
    float q1 = dw_opq(a.hi / b.hi);
    DwF32 r1 = dw_sub(a, dw_mul(b, DwF32(q1, 0.0)));
    float q2 = dw_opq(r1.hi / b.hi);
    DwF32 r2 = dw_sub(r1, dw_mul(b, DwF32(q2, 0.0)));
    float q3 = dw_opq(r2.hi / b.hi);
    DwF32 s = dw_quick_two_sum(q1, q2);
    return dw_add(s, DwF32(q3, 0.0));
}
DwF32 dw_sqrt(DwF32 a) {
    if (a.hi == 0.0) { return DwF32(0.0, 0.0); }
    float x = dw_opq(inversesqrt(a.hi));
    float ax = dw_opq(a.hi * x);
    DwF32 axsq = dw_two_prod(ax, ax);
    float corr = dw_opq(dw_sub(a, axsq).hi * dw_opq(x * 0.5));
    return dw_two_sum(ax, corr);
}
"#;

const WGSL_GUARDED_SQRT: &str = r#"fn dw_sqrt(a: DwF32) -> DwF32 {
    if (a.hi == 0.0) { return DwF32(0.0, 0.0); }
    let x = dw_opq(inverseSqrt(a.hi));
    let ax = dw_opq(a.hi * x);
    let axsq = dw_two_prod(ax, ax);
    let corr = dw_opq(dw_sub(a, axsq).hi * dw_opq(x * 0.5));
    return dw_two_sum(ax, corr);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The host reference recovers ~f64: round-trip and a stress accumulation.
    #[test]
    fn double_single_recovers_near_f64() {
        // Round-trip: deconstruct f64 -> 2x f32 -> reconstruct.
        let mut worst = 0.0f64;
        let mut s = 0x9e3779b97f4a7c15u64;
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            let v = (u * 2.0 - 1.0) * 10f64.powf(u * 6.0 - 3.0);
            let err = ((DoubleSingle::from_f64(v).to_f64() - v) / v).abs();
            worst = worst.max(err);
        }
        // ~49 bits; f32 alone would be ~6e-8 (24 bits).
        assert!(worst < 1e-13, "round-trip worst rel err {worst:e}");

        // Ill-conditioned accumulation: naive f32 is ~1.5e-2 off; dw tracks f64.
        let n = 2_000_000u32;
        let truth: f64 = (1..=n).map(|i| 1.0 / i as f64).sum();
        let naive: f32 = (1..=n).map(|i| 1.0 / i as f32).sum();
        let mut acc = DoubleSingle { hi: 0.0, lo: 0.0 };
        for i in 1..=n {
            acc = acc.add(DoubleSingle::from_f64(1.0 / i as f64));
        }
        let e_naive = ((naive as f64 - truth) / truth).abs();
        let e_dw = ((acc.to_f64() - truth) / truth).abs();
        eprintln!("round-trip worst={worst:e}  naive f32={e_naive:e}  2x f32={e_dw:e}");
        assert!(
            e_naive > 1e-3,
            "expected naive f32 to be bad, got {e_naive:e}"
        );
        // ~5 orders better than naive; near-f64 (n·eps accumulation over 500k).
        assert!(e_dw < 1e-8, "double-single accumulation err {e_dw:e}");
    }

    /// dw_mul / dw_div references match f64 to ~2^-48.
    #[test]
    fn double_single_mul_div_match_f64() {
        let a = DoubleSingle::from_f64(std::f64::consts::PI);
        let b = DoubleSingle::from_f64(std::f64::consts::E);
        let got = a.mul(b).to_f64();
        let truth = std::f64::consts::PI * std::f64::consts::E;
        assert!(((got - truth) / truth).abs() < 1e-13);

        // Division across a spread of magnitudes; sub is exercised inside div.
        let mut s = 0x243f6a8885a308d3u64;
        let mut worst = 0.0f64;
        for _ in 0..50_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            let av = (u * 2.0 - 1.0) * 10f64.powf(u * 6.0 - 3.0) + 1e-3;
            let bv = (u * 2.0 - 1.5) * 10f64.powf(u * 4.0 - 2.0) - 7e-2;
            if bv.abs() < 1e-6 {
                continue;
            }
            let got = DoubleSingle::from_f64(av)
                .div(DoubleSingle::from_f64(bv))
                .to_f64();
            let truth = av / bv;
            worst = worst.max(((got - truth) / truth).abs());
        }
        // f32 alone would be ~6e-8; double-single division stays near f64.
        assert!(worst < 1e-12, "dw_div worst rel err {worst:e}");

        // sub is exact-ish: (a+b) - b ≈ a to double-single precision.
        let a = DoubleSingle::from_f64(1.0 / 3.0);
        let b = DoubleSingle::from_f64(std::f64::consts::SQRT_2);
        let back = a.add(b).sub(b).to_f64();
        assert!((back - 1.0 / 3.0).abs() < 1e-13);

        // sqrt tracks f64 across magnitudes; sqrt(0) = 0.
        assert_eq!(DoubleSingle::from_f64(0.0).sqrt().to_f64(), 0.0);
        let mut worst_sqrt = 0.0f64;
        for k in 0..20_000u32 {
            let v = 1e-6 + (k as f64) * 1e-3;
            let got = DoubleSingle::from_f64(v).sqrt().to_f64();
            worst_sqrt = worst_sqrt.max(((got - v.sqrt()) / v.sqrt()).abs());
        }
        assert!(worst_sqrt < 1e-12, "dw_sqrt worst rel err {worst_sqrt:e}");
    }

    #[test]
    fn every_language_emits_the_operators() {
        for lang in [Lang::Wgsl, Lang::Cuda, Lang::Msl, Lang::Glsl, Lang::OpenCl] {
            let src = double_single_prelude(lang);
            for f in [
                "dw_two_sum",
                "dw_two_prod",
                "dw_add",
                "dw_sub",
                "dw_mul",
                "dw_div",
                "dw_sqrt",
                "DwF32",
            ] {
                assert!(src.contains(f), "{lang:?} prelude missing {f}");
            }
        }
        assert!(double_single_prelude(Lang::Cuda).contains("fmaf("));
        assert!(double_single_prelude(Lang::Wgsl).contains("fma("));
    }

    /// The emitted WGSL prelude must be valid WGSL (parses + validates in naga).
    #[test]
    fn wgsl_prelude_is_valid() {
        let src = format!(
            "{}\n@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n\
             @compute @workgroup_size(1) fn main() {{\n\
             let r = dw_add(dw_mul(DwF32(3.0, 0.0), DwF32(2.0, 0.0)), DwF32(1.0, 0.0));\n\
             let q = dw_sqrt(dw_div(dw_sub(r, DwF32(1.0, 0.0)), DwF32(2.0, 0.0)));\n\
             o[0] = q.hi; o[1] = q.lo;\n}}\n",
            double_single_prelude(Lang::Wgsl)
        );
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("generated WGSL failed to parse: {e:?}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("generated WGSL failed validation");
    }

    /// The fast-math-guarded WGSL prelude is valid WGSL given the `dw_zero`
    /// it requires from the including shader.
    #[test]
    fn wgsl_fast_math_guarded_prelude_is_valid() {
        let src = format!(
            "{WGSL_FAST_MATH_GUARDED}\n\
             @group(0) @binding(0) var<storage, read_write> o: array<f32>;\n\
             @group(0) @binding(1) var<uniform> zero: u32;\n\
             fn dw_zero() -> u32 {{ return zero; }}\n\
             @compute @workgroup_size(1) fn main() {{\n\
             let r = dw_add(dw_mul(DwF32(3.0, 0.0), DwF32(2.0, 0.0)), DwF32(1.0, 0.0));\n\
             let q = dw_div(dw_sub(r, DwF32(1.0, 0.0)), DwF32(2.0, 0.0));\n\
             o[0] = q.hi; o[1] = q.lo;\n}}\n"
        );
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("guarded WGSL failed to parse: {e:?}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("guarded WGSL failed validation");
    }

    #[test]
    fn exp_tracks_f64() {
        let mut worst = 0f64;
        for i in 0..40_000 {
            let x = -100.0 + i as f64 * (188.0 / 40_000.0) + 1.234_567e-3;
            let got = DoubleSingle::from_f64(x).exp().to_f64();
            let want = x.exp();
            // Below ~1e-30 the low half is an f32 subnormal: double-single
            // has f32's exponent range, so precision falls off there.
            if !(1e-30..=1e37).contains(&want) {
                continue;
            }
            worst = worst.max(((got - want) / want).abs());
        }
        eprintln!("dw_exp worst rel err {worst:e}");
        assert!(worst < 2e-13, "dw_exp worst rel err {worst:e}");
        assert_eq!(DoubleSingle::from_f64(0.0).exp().to_f64(), 1.0);
        assert_eq!(DoubleSingle::from_f64(-200.0).exp().to_f64(), 0.0);
        assert!(DoubleSingle::from_f64(100.0).exp().hi.is_infinite());
    }

    #[test]
    fn guarded_wgsl_and_glsl_with_exp_validate() {
        let wgsl = format!(
            "{}\n@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n\
             @group(0) @binding(1) var<uniform> zero: u32;\n\
             fn dw_zero() -> u32 {{ return zero; }}\n\
             @compute @workgroup_size(1) fn main() {{\n\
             let q = dw_exp(dw_sqrt(DwF32(o[0], o[1])));\n o[0] = q.hi; o[1] = q.lo;\n}}\n",
            wgsl_fast_math_guarded()
        );
        let module = naga::front::wgsl::parse_str(&wgsl).expect("guarded WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("guarded WGSL validates");
        let glsl = format!(
            "#version 450\nlayout(local_size_x = 1) in;\n\
             layout(std430, binding = 0) buffer B {{ float o[]; }};\n{}\n\
             void main() {{ DwF32 q = dw_exp(DwF32(o[0], o[1])); o[0] = q.hi; o[1] = q.lo; }}\n",
            double_single_prelude(Lang::Glsl)
        );
        let mut front = naga::front::glsl::Frontend::default();
        let module = front
            .parse(
                &naga::front::glsl::Options::from(naga::ShaderStage::Compute),
                &glsl,
            )
            .unwrap_or_else(|e| panic!("GLSL with dw_exp failed to parse: {e:?}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("GLSL validates");
        let guarded = format!(
            "#version 450\nlayout(local_size_x = 1) in;\n\
             layout(std430, binding = 0) buffer B {{ float o[]; }};\n\
             layout(push_constant) uniform PC {{ uint zero; }} pc;\n\
             uint dw_zero() {{ return pc.zero; }}\n{}\n\
             void main() {{ DwF32 q = dw_exp(dw_sqrt(dw_div(DwF32(o[0], o[1]), DwF32(3.0, 0.0)))); o[0] = q.hi; o[1] = q.lo; }}\n",
            glsl_fast_math_guarded()
        );
        let mut front = naga::front::glsl::Frontend::default();
        let module = front
            .parse(
                &naga::front::glsl::Options::from(naga::ShaderStage::Compute),
                &guarded,
            )
            .unwrap_or_else(|e| panic!("guarded GLSL failed to parse: {e:?}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("guarded GLSL validates");
    }
}
