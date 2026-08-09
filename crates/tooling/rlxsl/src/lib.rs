// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Warm-tier kernel DSL** — one scalar-expression manifest, many backends.
//!
//! The standalone **unary-activation** kernels historically re-expressed the
//! *same* scalar math by hand in WGSL, MSL, GLSL, CUDA C, OpenCL-C and Rust —
//! six copies of `gelu`'s Abramowitz-&-Stegun `erf`, of the softplus stability
//! trick, of every activation. This crate makes that math a single source: each
//! activation is one [`Sx`] expression tree, and the per-language emitters
//! ([`Lang`]) render it into WGSL / CUDA C / MSL / GLSL / OpenCL-C.
//!
//! Scope is the full standalone elementwise surface — the unary activations
//! (forward + auto-differentiated backward), the binary elementwise ops
//! ([`binary`]), the compare ops ([`compare`]) — plus the double-word precision
//! prelude ([`dw`]). Every backend's standalone `unary`/`binary`/`compare` kernel
//! is generated from here; only the fused-region kernels stay hand-written.
//!
//! Scope is deliberately the **warm tier** — the *un-fused* standalone kernels,
//! which are the fallback path (the hot path is the fused region, hand-written
//! per backend and never routed through here). So generating them costs no
//! peak performance, per the per-backend-peak-perf north star.
//!
//! Two things keep this honest:
//! * The [`eval`] interpreter walks the same [`Sx`] tree, so the manifest's math
//!   can be checked numerically against the trusted CPU backend for every
//!   [`Activation`] (see `rlx-runtime`'s `kernel_dsl_activation_oracle` test) —
//!   an *independent* oracle, because the CPU kernel stays hand-written.
//! * The case order follows [`Activation::opcode_relu_first`] — the canonical
//!   opcode from `rlx_ir::opcodes` — so codegen and dispatch cannot disagree.

pub mod binary;
pub mod compare;
pub mod dw;

use rlx_ir::op::Activation;
use std::collections::HashMap;

/// A scalar `f32 → f32` expression over the single input variable `x`.
///
/// Kept intentionally small: the warm-tier activations need only arithmetic, a
/// dozen intrinsics, `clamp`, and a comparison-`select`. `Let`/`Var` give
/// named temporaries so emitted code reads like the hand-written original
/// (e.g. `gelu`'s shared polynomial term `t`).
#[derive(Debug, Clone)]
pub enum Sx {
    /// The input variable `x`.
    X,
    /// A floating-point literal.
    Lit(f32),
    /// `let <name> = <value>;` then evaluate `body` with `<name>` in scope.
    Let(&'static str, Box<Sx>, Box<Sx>),
    /// Reference a name bound by an enclosing [`Sx::Let`].
    Var(&'static str),
    Neg(Box<Sx>),
    Add(Box<Sx>, Box<Sx>),
    Sub(Box<Sx>, Box<Sx>),
    Mul(Box<Sx>, Box<Sx>),
    Div(Box<Sx>, Box<Sx>),
    Max(Box<Sx>, Box<Sx>),
    Min(Box<Sx>, Box<Sx>),
    /// `clamp(e, lo, hi)`.
    Clamp(Box<Sx>, f32, f32),
    /// A unary intrinsic call (`exp`, `tanh`, …).
    Call(Intr, Box<Sx>),
    /// `cond ? on_true : on_false`, where `cond` is `a <cmp> b`.
    Select {
        on_false: Box<Sx>,
        on_true: Box<Sx>,
        a: Box<Sx>,
        cmp: Cmp,
        b: Box<Sx>,
    },
}

/// Unary intrinsics with a canonical name per target language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intr {
    Exp,
    Log,
    Sqrt,
    /// `1/sqrt(x)`.
    Rsqrt,
    Tanh,
    Sin,
    Cos,
    Tan,
    Atan,
    Abs,
    Floor,
    Ceil,
    /// Round half-to-even (matches ONNX `Round` and the CPU kernel).
    Round,
    /// `sign(x)` with `sign(0) == 0`.
    Sign,
    /// Gauss error function. A **semantic** intrinsic: lowered to the native
    /// hardware `erff` on CUDA (the only target with a scalar erf builtin), and
    /// to the Abramowitz & Stegun 7.1.26 polynomial on MSL/WGSL/GLSL. This is
    /// the point of the DSL — one definition, the most precise primitive
    /// available on each backend.
    Erf,
}

/// Comparison used by [`Sx::Select`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    Gt,
    Ge,
    Lt,
    Le,
}

// ── Builders (keep the manifest below readable) ─────────────────────────────

fn x() -> Sx {
    Sx::X
}
fn lit(v: f32) -> Sx {
    Sx::Lit(v)
}
fn var(n: &'static str) -> Sx {
    Sx::Var(n)
}
fn let_(n: &'static str, v: Sx, body: Sx) -> Sx {
    Sx::Let(n, Box::new(v), Box::new(body))
}
fn neg(a: Sx) -> Sx {
    Sx::Neg(Box::new(a))
}
fn add(a: Sx, b: Sx) -> Sx {
    Sx::Add(Box::new(a), Box::new(b))
}
fn sub(a: Sx, b: Sx) -> Sx {
    Sx::Sub(Box::new(a), Box::new(b))
}
fn mul(a: Sx, b: Sx) -> Sx {
    Sx::Mul(Box::new(a), Box::new(b))
}
fn div(a: Sx, b: Sx) -> Sx {
    Sx::Div(Box::new(a), Box::new(b))
}
fn maxf(a: Sx, b: Sx) -> Sx {
    Sx::Max(Box::new(a), Box::new(b))
}
fn minf(a: Sx, b: Sx) -> Sx {
    Sx::Min(Box::new(a), Box::new(b))
}
fn clamp(a: Sx, lo: f32, hi: f32) -> Sx {
    Sx::Clamp(Box::new(a), lo, hi)
}
fn call(i: Intr, a: Sx) -> Sx {
    Sx::Call(i, Box::new(a))
}
fn exp(a: Sx) -> Sx {
    call(Intr::Exp, a)
}
fn log(a: Sx) -> Sx {
    call(Intr::Log, a)
}
fn tanh(a: Sx) -> Sx {
    call(Intr::Tanh, a)
}
fn absf(a: Sx) -> Sx {
    call(Intr::Abs, a)
}
fn erf(a: Sx) -> Sx {
    call(Intr::Erf, a)
}

/// Abramowitz & Stegun 7.1.26 `erf` polynomial over `arg`, used to lower
/// [`Intr::Erf`] on backends without a native erf (WGSL/GLSL). Binds `arg` once
/// (`_ea`) so a compound argument (e.g. gelu's `x·√½`) is not recomputed.
fn erf_ansatz(arg: Sx) -> Sx {
    // p = ((((1.0614054·t - 1.4531521)·t + 1.4214138)·t
    //       - 0.28449672)·t + 0.2548296)·t
    let poly = mul(
        add(
            mul(
                add(
                    mul(
                        add(
                            mul(
                                add(mul(lit(1.061_405_4), var("_et")), lit(-1.453_152_1)),
                                var("_et"),
                            ),
                            lit(1.421_413_8),
                        ),
                        var("_et"),
                    ),
                    lit(-0.284_496_72),
                ),
                var("_et"),
            ),
            lit(0.254_829_6),
        ),
        var("_et"),
    );
    let_(
        "_ea",
        arg,
        let_(
            "_ex",
            absf(var("_ea")),
            let_(
                "_et",
                div(lit(1.0), add(lit(1.0), mul(lit(0.3275911), var("_ex")))),
                let_(
                    "_ep",
                    poly,
                    mul(
                        call(Intr::Sign, var("_ea")),
                        sub(
                            lit(1.0),
                            mul(var("_ep"), exp(neg(mul(var("_ex"), var("_ex"))))),
                        ),
                    ),
                ),
            ),
        ),
    )
}
/// `select(on_false, on_true, a cmp b)`.
fn sel(on_true: Sx, a: Sx, cmp: Cmp, b: Sx, on_false: Sx) -> Sx {
    Sx::Select {
        on_false: Box::new(on_false),
        on_true: Box::new(on_true),
        a: Box::new(a),
        cmp,
        b: Box::new(b),
    }
}

/// `sigmoid(x) = 1 / (1 + exp(-x))` (no stability clamp — matches the standalone
/// sigmoid kernel).
fn sigmoid() -> Sx {
    div(lit(1.0), add(lit(1.0), exp(neg(x()))))
}

/// `softplus(x) = max(x,0) + log(1 + exp(-|x|))` — the numerically-stable form
/// used across the hand-written kernels.
fn softplus() -> Sx {
    add(maxf(x(), lit(0.0)), log(add(lit(1.0), exp(neg(absf(x()))))))
}

// ── The manifest: one expression per activation ─────────────────────────────

/// The scalar math for `act`, defined **once**. Faithful to the current
/// hand-written kernels (same stability clamps, same A&S `erf` polynomial) so
/// generated code is a drop-in replacement, not an approximation.
pub fn activation_expr(act: Activation) -> Sx {
    match act {
        Activation::Relu => maxf(x(), lit(0.0)),
        Activation::Sigmoid => sigmoid(),
        // clamp: tanh NaNs on large |x| on some GPUs.
        Activation::Tanh => tanh(clamp(x(), -15.0, 15.0)),
        Activation::Exp => exp(x()),
        Activation::Log => log(x()),
        Activation::Sqrt => call(Intr::Sqrt, x()),
        Activation::Rsqrt => call(Intr::Rsqrt, x()),
        Activation::Neg => neg(x()),
        Activation::Abs => absf(x()),
        // gelu = 0.5·x·(1 + erf(x·√½)). Composed from the `Intr::Erf` primitive
        // rather than re-inlining the polynomial: this reuses the *single* erf
        // definition, so on CUDA/OpenCL gelu picks up the native `erff`/`erf`
        // (more accurate than the A&S expansion, and consistent with
        // `Activation::Erf`), while WGSL/MSL/GLSL still expand the shared
        // `erf_ansatz`. The auto-differentiated backward likewise becomes the
        // analytic `Φ(x) + x·φ(x)` instead of a differentiated polynomial.
        Activation::Gelu => mul(
            mul(lit(0.5), x()),
            add(
                lit(1.0),
                erf(mul(x(), lit(std::f32::consts::FRAC_1_SQRT_2))),
            ),
        ),
        // silu(x) = x * sigmoid(x); clamp -x to avoid exp overflow.
        Activation::Silu => let_(
            "nx",
            clamp(neg(x()), -88.0, 88.0),
            div(x(), add(lit(1.0), exp(var("nx")))),
        ),
        // gelu_approx: 0.5*x*(1 + tanh(√(2/π)·(x + 0.044715 x³)))
        Activation::GeluApprox => let_(
            "x3",
            mul(mul(x(), x()), x()),
            let_(
                "inner",
                clamp(
                    mul(lit(SQRT_2_OVER_PI), add(x(), mul(lit(0.044715), var("x3")))),
                    -15.0,
                    15.0,
                ),
                mul(mul(lit(0.5), x()), add(lit(1.0), tanh(var("inner")))),
            ),
        ),
        Activation::Round => call(Intr::Round, x()),
        Activation::Sin => call(Intr::Sin, x()),
        Activation::Cos => call(Intr::Cos, x()),
        Activation::Tan => call(Intr::Tan, x()),
        Activation::Atan => call(Intr::Atan, x()),
        Activation::Recip => div(lit(1.0), x()),
        Activation::Floor => call(Intr::Floor, x()),
        Activation::Ceil => call(Intr::Ceil, x()),
        Activation::Sign => call(Intr::Sign, x()),
        Activation::Softplus => softplus(),
        // elu (alpha=1): x>0 ? x : exp(x)-1
        Activation::Elu => sel(x(), x(), Cmp::Gt, lit(0.0), sub(exp(x()), lit(1.0))),
        // erf — semantic intrinsic: native `erff`/`erf` on CUDA/MSL, A&S 7.1.26
        // polynomial on WGSL/GLSL (see `erf_ansatz` + the `Intr::Erf` emitter).
        Activation::Erf => erf(x()),
        // hardswish: x * clamp(x+3, 0, 6) / 6
        Activation::HardSwish => div(mul(x(), clamp(add(x(), lit(3.0)), 0.0, 6.0)), lit(6.0)),
        // hardsigmoid: clamp(x/6 + 0.5, 0, 1)
        Activation::HardSigmoid => clamp(add(div(x(), lit(6.0)), lit(0.5)), 0.0, 1.0),
        // mish: x * tanh(softplus(x))
        Activation::Mish => let_("sp", softplus(), mul(x(), tanh(var("sp")))),
        // softsign: x / (1 + |x|)
        Activation::Softsign => div(x(), add(lit(1.0), absf(x()))),
        // logsigmoid: min(x,0) - log(1 + exp(-|x|))
        Activation::LogSigmoid => sub(minf(x(), lit(0.0)), log(add(lit(1.0), exp(neg(absf(x())))))),
    }
}

// ── Interpreter (the independent numeric oracle) ────────────────────────────

/// Evaluate `sx` at input `xv`. Mirrors the semantics the emitters target
/// (WGSL `sign(0)==0`, round-half-to-even), so `eval(activation_expr(a), x)`
/// can be checked against the CPU backend's kernel.
pub fn eval(sx: &Sx, xv: f32) -> f32 {
    fn go(sx: &Sx, xv: f32, env: &mut Vec<(&'static str, f32)>) -> f32 {
        match sx {
            Sx::X => xv,
            Sx::Lit(v) => *v,
            Sx::Var(n) => env
                .iter()
                .rev()
                .find(|(k, _)| k == n)
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("unbound var {n}")),
            Sx::Let(n, v, body) => {
                let vv = go(v, xv, env);
                env.push((n, vv));
                let r = go(body, xv, env);
                env.pop();
                r
            }
            Sx::Neg(a) => -go(a, xv, env),
            Sx::Add(a, b) => go(a, xv, env) + go(b, xv, env),
            Sx::Sub(a, b) => go(a, xv, env) - go(b, xv, env),
            Sx::Mul(a, b) => go(a, xv, env) * go(b, xv, env),
            Sx::Div(a, b) => go(a, xv, env) / go(b, xv, env),
            Sx::Max(a, b) => go(a, xv, env).max(go(b, xv, env)),
            Sx::Min(a, b) => go(a, xv, env).min(go(b, xv, env)),
            Sx::Clamp(a, lo, hi) => go(a, xv, env).clamp(*lo, *hi),
            Sx::Call(i, a) => {
                let v = go(a, xv, env);
                match i {
                    Intr::Exp => v.exp(),
                    Intr::Log => v.ln(),
                    Intr::Sqrt => v.sqrt(),
                    Intr::Rsqrt => 1.0 / v.sqrt(),
                    Intr::Tanh => v.tanh(),
                    Intr::Sin => v.sin(),
                    Intr::Cos => v.cos(),
                    Intr::Tan => v.tan(),
                    Intr::Atan => v.atan(),
                    Intr::Abs => v.abs(),
                    Intr::Floor => v.floor(),
                    Intr::Ceil => v.ceil(),
                    Intr::Round => v.round_ties_even(),
                    Intr::Sign => {
                        if v > 0.0 {
                            1.0
                        } else if v < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    }
                    // A&S 7.1.26 — matches `erf_ansatz` and the CPU/GPU kernels
                    // within tolerance (native erff on CUDA is ~1e-3 closer).
                    Intr::Erf => {
                        let s = if v >= 0.0 { 1.0 } else { -1.0 };
                        let ax = v.abs();
                        let t = 1.0 / (1.0 + 0.3275911 * ax);
                        let p = ((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t
                            - 0.284_496_72)
                            * t
                            + 0.254_829_6)
                            * t;
                        s * (1.0 - p * (-ax * ax).exp())
                    }
                }
            }
            Sx::Select {
                on_false,
                on_true,
                a,
                cmp,
                b,
            } => {
                let av = go(a, xv, env);
                let bv = go(b, xv, env);
                let c = match cmp {
                    Cmp::Gt => av > bv,
                    Cmp::Ge => av >= bv,
                    Cmp::Lt => av < bv,
                    Cmp::Le => av <= bv,
                };
                if c {
                    go(on_true, xv, env)
                } else {
                    go(on_false, xv, env)
                }
            }
        }
    }
    go(sx, xv, &mut Vec::new())
}

/// Convenience: `eval(activation_expr(act), x)`.
pub fn eval_activation(act: Activation, xv: f32) -> f32 {
    eval(&activation_expr(act), xv)
}

// ── Symbolic differentiation (backward from the forward manifest) ────────────

/// `2/√π`, the constant in `d/dx erf(x) = (2/√π)·e^(-x²)`.
const TWO_OVER_SQRT_PI: f32 = std::f32::consts::FRAC_2_SQRT_PI;

/// `√(2/π)`, the constant inside the tanh of the GELU approximation
/// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715x³)))`.
///
/// Deliberately NOT [`std::f32::consts::FRAC_2_SQRT_PI`], which is the
/// different constant `2/√π ≈ 1.128` (see [`TWO_OVER_SQRT_PI`] above). Using
/// that one here makes `gelu_approx(-2)` return `-0.0097` instead of `-0.0454`.
/// Matches `GC` in rlx-cpu's `elementwise.rs`, the reference implementation.
const SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// Substitute every [`Sx::Let`] binding into its uses, returning a `let`-free
/// tree. (Differentiation is simplest on a flat expression; the shared subterms
/// this duplicates are re-shared afterwards by [`cse`].)
fn inline_lets(sx: &Sx) -> Sx {
    fn go(sx: &Sx, env: &mut Vec<(&'static str, Sx)>) -> Sx {
        match sx {
            Sx::X => Sx::X,
            Sx::Lit(v) => Sx::Lit(*v),
            Sx::Var(n) => env
                .iter()
                .rev()
                .find(|(k, _)| k == n)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| panic!("inline_lets: unbound var {n}")),
            Sx::Let(n, v, body) => {
                let vv = go(v, env);
                env.push((n, vv));
                let r = go(body, env);
                env.pop();
                r
            }
            Sx::Neg(a) => Sx::Neg(Box::new(go(a, env))),
            Sx::Add(a, b) => Sx::Add(Box::new(go(a, env)), Box::new(go(b, env))),
            Sx::Sub(a, b) => Sx::Sub(Box::new(go(a, env)), Box::new(go(b, env))),
            Sx::Mul(a, b) => Sx::Mul(Box::new(go(a, env)), Box::new(go(b, env))),
            Sx::Div(a, b) => Sx::Div(Box::new(go(a, env)), Box::new(go(b, env))),
            Sx::Max(a, b) => Sx::Max(Box::new(go(a, env)), Box::new(go(b, env))),
            Sx::Min(a, b) => Sx::Min(Box::new(go(a, env)), Box::new(go(b, env))),
            Sx::Clamp(a, lo, hi) => Sx::Clamp(Box::new(go(a, env)), *lo, *hi),
            Sx::Call(i, a) => Sx::Call(*i, Box::new(go(a, env))),
            Sx::Select {
                on_false,
                on_true,
                a,
                cmp,
                b,
            } => Sx::Select {
                on_false: Box::new(go(on_false, env)),
                on_true: Box::new(go(on_true, env)),
                a: Box::new(go(a, env)),
                cmp: *cmp,
                b: Box::new(go(b, env)),
            },
        }
    }
    go(sx, &mut Vec::new())
}

/// Symbolic `d(sx)/dx` over a `let`-free expression (call [`inline_lets`]
/// first). Piecewise ops (`floor`/`ceil`/`sign`/`round`) differentiate to 0
/// here; the STE convention for `round` is applied at the activation level in
/// [`activation_grad_expr`].
fn diff(sx: &Sx) -> Sx {
    match sx {
        Sx::X => lit(1.0),
        Sx::Lit(_) => lit(0.0),
        Sx::Var(_) | Sx::Let(..) => {
            unreachable!("diff expects an inlined (let-free) expression")
        }
        Sx::Neg(a) => neg(diff(a)),
        Sx::Add(a, b) => add(diff(a), diff(b)),
        Sx::Sub(a, b) => sub(diff(a), diff(b)),
        // (a·b)' = a'·b + a·b'
        Sx::Mul(a, b) => add(mul(diff(a), (**b).clone()), mul((**a).clone(), diff(b))),
        // (a/b)' = (a'·b - a·b') / b²
        Sx::Div(a, b) => div(
            sub(mul(diff(a), (**b).clone()), mul((**a).clone(), diff(b))),
            mul((**b).clone(), (**b).clone()),
        ),
        // max(a,b)' = a>b ? a' : b'   ;   min(a,b)' = a<b ? a' : b'
        Sx::Max(a, b) => sel(diff(a), (**a).clone(), Cmp::Gt, (**b).clone(), diff(b)),
        Sx::Min(a, b) => sel(diff(a), (**a).clone(), Cmp::Lt, (**b).clone(), diff(b)),
        // clamp(a,lo,hi)' = (lo<a<hi) ? a' : 0
        Sx::Clamp(a, lo, hi) => {
            let da = diff(a);
            let below_hi = sel(da, (**a).clone(), Cmp::Lt, lit(*hi), lit(0.0));
            sel(below_hi, (**a).clone(), Cmp::Gt, lit(*lo), lit(0.0))
        }
        Sx::Call(i, a) => {
            let da = || diff(a);
            let aa = || (**a).clone();
            match i {
                Intr::Exp => mul(call(Intr::Exp, aa()), da()),
                Intr::Log => div(da(), aa()),
                Intr::Sqrt => div(da(), mul(lit(2.0), call(Intr::Sqrt, aa()))),
                // (a^-½)' = -½·a^-3/2·a' = -½·(rsqrt(a)/a)·a'
                Intr::Rsqrt => mul(mul(lit(-0.5), div(call(Intr::Rsqrt, aa()), aa())), da()),
                Intr::Tanh => mul(
                    sub(
                        lit(1.0),
                        mul(call(Intr::Tanh, aa()), call(Intr::Tanh, aa())),
                    ),
                    da(),
                ),
                Intr::Sin => mul(call(Intr::Cos, aa()), da()),
                Intr::Cos => mul(neg(call(Intr::Sin, aa())), da()),
                Intr::Tan => mul(
                    add(lit(1.0), mul(call(Intr::Tan, aa()), call(Intr::Tan, aa()))),
                    da(),
                ),
                Intr::Atan => div(da(), add(lit(1.0), mul(aa(), aa()))),
                Intr::Abs => mul(call(Intr::Sign, aa()), da()),
                // erf'(a) = (2/√π)·e^(-a²)·a'
                Intr::Erf => mul(
                    mul(lit(TWO_OVER_SQRT_PI), call(Intr::Exp, neg(mul(aa(), aa())))),
                    da(),
                ),
                Intr::Floor | Intr::Ceil | Intr::Round | Intr::Sign => lit(0.0),
            }
        }
        // A select routes to whichever branch's derivative; the condition
        // operands don't contribute (measure-zero boundary ignored, as in the
        // hand-written kernels).
        Sx::Select {
            on_false,
            on_true,
            a,
            cmp,
            b,
        } => Sx::Select {
            on_false: Box::new(diff(on_false)),
            on_true: Box::new(diff(on_true)),
            a: a.clone(),
            cmp: *cmp,
            b: b.clone(),
        },
    }
}

fn as_lit(s: &Sx) -> Option<f32> {
    if let Sx::Lit(v) = s { Some(*v) } else { None }
}
fn is_zero(s: &Sx) -> bool {
    matches!(s, Sx::Lit(v) if *v == 0.0)
}
fn is_one(s: &Sx) -> bool {
    matches!(s, Sx::Lit(v) if *v == 1.0)
}

/// Algebraic peephole simplification (bottom-up): folds constants and the
/// identities the diff rules leave behind (`0+a`, `a-0`, `0·a`, `1·a`, `a/1`,
/// `-(-a)`, `-1·a`). One bottom-up pass reaches a fixpoint for these local
/// rules. Cuts the auto-differentiated gelu grad from ~5 KB to a readable size
/// without changing its value.
fn simplify(sx: &Sx) -> Sx {
    match sx {
        Sx::X | Sx::Lit(_) | Sx::Var(_) => sx.clone(),
        Sx::Neg(a) => {
            let a = simplify(a);
            match a {
                Sx::Lit(v) => lit(-v),
                Sx::Neg(inner) => *inner, // -(-x) = x
                other => neg(other),
            }
        }
        Sx::Add(a, b) => {
            let (a, b) = (simplify(a), simplify(b));
            if is_zero(&a) {
                b
            } else if is_zero(&b) {
                a
            } else if let (Some(x), Some(y)) = (as_lit(&a), as_lit(&b)) {
                lit(x + y)
            } else {
                add(a, b)
            }
        }
        Sx::Sub(a, b) => {
            let (a, b) = (simplify(a), simplify(b));
            if is_zero(&b) {
                a
            } else if is_zero(&a) {
                simplify(&neg(b))
            } else if let (Some(x), Some(y)) = (as_lit(&a), as_lit(&b)) {
                lit(x - y)
            } else {
                sub(a, b)
            }
        }
        Sx::Mul(a, b) => {
            let (a, b) = (simplify(a), simplify(b));
            if is_zero(&a) || is_zero(&b) {
                lit(0.0)
            } else if is_one(&a) {
                b
            } else if is_one(&b) {
                a
            } else if let (Some(x), Some(y)) = (as_lit(&a), as_lit(&b)) {
                lit(x * y)
            } else if matches!(as_lit(&a), Some(v) if v == -1.0) {
                simplify(&neg(b))
            } else if matches!(as_lit(&b), Some(v) if v == -1.0) {
                simplify(&neg(a))
            } else {
                mul(a, b)
            }
        }
        Sx::Div(a, b) => {
            let (a, b) = (simplify(a), simplify(b));
            if is_zero(&a) {
                lit(0.0)
            } else if is_one(&b) {
                a
            } else if let (Some(x), Some(y)) = (as_lit(&a), as_lit(&b)) {
                if y != 0.0 { lit(x / y) } else { div(a, b) }
            } else {
                div(a, b)
            }
        }
        Sx::Max(a, b) => maxf(simplify(a), simplify(b)),
        Sx::Min(a, b) => minf(simplify(a), simplify(b)),
        Sx::Clamp(a, lo, hi) => clamp(simplify(a), *lo, *hi),
        Sx::Call(i, a) => call(*i, simplify(a)),
        Sx::Let(n, v, body) => Sx::Let(n, Box::new(simplify(v)), Box::new(simplify(body))),
        Sx::Select {
            on_false,
            on_true,
            a,
            cmp,
            b,
        } => Sx::Select {
            on_false: Box::new(simplify(on_false)),
            on_true: Box::new(simplify(on_true)),
            a: Box::new(simplify(a)),
            cmp: *cmp,
            b: Box::new(simplify(b)),
        },
    }
}

// ── Common-subexpression elimination (re-share what diff duplicated) ─────────

/// Fresh `let` names for hoisted common subexpressions. A small fixed pool: no
/// activation grad needs more than a handful, and capping keeps [`cse`] bounded.
const CSE_NAMES: [&str; 16] = [
    "_cse0", "_cse1", "_cse2", "_cse3", "_cse4", "_cse5", "_cse6", "_cse7", "_cse8", "_cse9",
    "_cse10", "_cse11", "_cse12", "_cse13", "_cse14", "_cse15",
];

/// A canonical structural key for `s` — equal keys ⇔ structurally identical
/// subtrees. Used to detect and match common subexpressions.
fn sx_key(s: &Sx) -> String {
    match s {
        Sx::X => "X".to_string(),
        Sx::Lit(v) => format!("L{v:?}"),
        Sx::Var(n) => format!("V{n}"),
        Sx::Neg(a) => format!("N({})", sx_key(a)),
        Sx::Add(a, b) => format!("+({},{})", sx_key(a), sx_key(b)),
        Sx::Sub(a, b) => format!("-({},{})", sx_key(a), sx_key(b)),
        Sx::Mul(a, b) => format!("*({},{})", sx_key(a), sx_key(b)),
        Sx::Div(a, b) => format!("/({},{})", sx_key(a), sx_key(b)),
        Sx::Max(a, b) => format!("mx({},{})", sx_key(a), sx_key(b)),
        Sx::Min(a, b) => format!("mn({},{})", sx_key(a), sx_key(b)),
        Sx::Clamp(a, lo, hi) => format!("cl({},{lo:?},{hi:?})", sx_key(a)),
        Sx::Call(i, a) => format!("{i:?}({})", sx_key(a)),
        Sx::Let(n, v, b) => format!("let {n}=({}) in ({})", sx_key(v), sx_key(b)),
        Sx::Select {
            on_false,
            on_true,
            a,
            cmp,
            b,
        } => format!(
            "sel({},{},{} {cmp:?} {})",
            sx_key(on_false),
            sx_key(on_true),
            sx_key(a),
            sx_key(b)
        ),
    }
}

/// Node count — the CSE benefit heuristic (bigger shared subtree ⇒ more saved).
fn node_count(s: &Sx) -> usize {
    match s {
        Sx::X | Sx::Lit(_) | Sx::Var(_) => 1,
        Sx::Neg(a) | Sx::Clamp(a, ..) | Sx::Call(_, a) => 1 + node_count(a),
        Sx::Add(a, b)
        | Sx::Sub(a, b)
        | Sx::Mul(a, b)
        | Sx::Div(a, b)
        | Sx::Max(a, b)
        | Sx::Min(a, b) => 1 + node_count(a) + node_count(b),
        Sx::Let(_, v, b) => 1 + node_count(v) + node_count(b),
        Sx::Select {
            on_false,
            on_true,
            a,
            b,
            ..
        } => 1 + node_count(on_false) + node_count(on_true) + node_count(a) + node_count(b),
    }
}

/// Whether `s` contains a [`Sx::Call`] intrinsic (exp/erf/…): those are worth
/// hoisting even when small, since re-evaluating a transcendental is expensive.
fn has_call(s: &Sx) -> bool {
    match s {
        Sx::Call(..) => true,
        Sx::X | Sx::Lit(_) | Sx::Var(_) => false,
        Sx::Neg(a) | Sx::Clamp(a, ..) => has_call(a),
        Sx::Add(a, b)
        | Sx::Sub(a, b)
        | Sx::Mul(a, b)
        | Sx::Div(a, b)
        | Sx::Max(a, b)
        | Sx::Min(a, b) => has_call(a) || has_call(b),
        Sx::Let(_, v, b) => has_call(v) || has_call(b),
        Sx::Select {
            on_false,
            on_true,
            a,
            b,
            ..
        } => has_call(on_false) || has_call(on_true) || has_call(a) || has_call(b),
    }
}

/// Tally every subexpression of `s` by canonical key → `(occurrences, node)`.
fn count_subexprs(s: &Sx, map: &mut HashMap<String, (usize, Sx)>) {
    map.entry(sx_key(s)).or_insert_with(|| (0, s.clone())).0 += 1;
    match s {
        Sx::X | Sx::Lit(_) | Sx::Var(_) => {}
        Sx::Neg(a) | Sx::Clamp(a, ..) | Sx::Call(_, a) => count_subexprs(a, map),
        Sx::Add(a, b)
        | Sx::Sub(a, b)
        | Sx::Mul(a, b)
        | Sx::Div(a, b)
        | Sx::Max(a, b)
        | Sx::Min(a, b) => {
            count_subexprs(a, map);
            count_subexprs(b, map);
        }
        Sx::Let(_, v, b) => {
            count_subexprs(v, map);
            count_subexprs(b, map);
        }
        Sx::Select {
            on_false,
            on_true,
            a,
            b,
            ..
        } => {
            count_subexprs(on_false, map);
            count_subexprs(on_true, map);
            count_subexprs(a, map);
            count_subexprs(b, map);
        }
    }
}

/// Replace every subtree whose canonical key equals `key` with `Var(name)`.
fn replace_by_key(s: &Sx, key: &str, name: &'static str) -> Sx {
    if sx_key(s) == key {
        return Sx::Var(name);
    }
    let go = |a: &Sx| Box::new(replace_by_key(a, key, name));
    match s {
        Sx::X | Sx::Lit(_) | Sx::Var(_) => s.clone(),
        Sx::Neg(a) => Sx::Neg(go(a)),
        Sx::Clamp(a, lo, hi) => Sx::Clamp(go(a), *lo, *hi),
        Sx::Call(i, a) => Sx::Call(*i, go(a)),
        Sx::Add(a, b) => Sx::Add(go(a), go(b)),
        Sx::Sub(a, b) => Sx::Sub(go(a), go(b)),
        Sx::Mul(a, b) => Sx::Mul(go(a), go(b)),
        Sx::Div(a, b) => Sx::Div(go(a), go(b)),
        Sx::Max(a, b) => Sx::Max(go(a), go(b)),
        Sx::Min(a, b) => Sx::Min(go(a), go(b)),
        Sx::Let(n, v, b) => Sx::Let(n, go(v), go(b)),
        Sx::Select {
            on_false,
            on_true,
            a,
            cmp,
            b,
        } => Sx::Select {
            on_false: go(on_false),
            on_true: go(on_true),
            a: go(a),
            cmp: *cmp,
            b: go(b),
        },
    }
}

/// Common-subexpression elimination: repeatedly hoist the largest subexpression
/// that appears ≥2× (and is worth it — a transcendental call, or ≥3 nodes) into
/// a fresh `let`, replacing all its occurrences with the bound name. Picking the
/// largest first lets an outer hoist absorb its shared inner terms; ties break on
/// the canonical key so the output is deterministic regardless of map order.
/// This undoes the duplication [`inline_lets`] introduces, so backward kernels
/// evaluate each `exp`/`erf`/shared term once instead of 3–4×.
fn cse(sx: &Sx) -> Sx {
    let mut expr = sx.clone();
    for name in CSE_NAMES {
        let mut counts: HashMap<String, (usize, Sx)> = HashMap::new();
        count_subexprs(&expr, &mut counts);
        let best = counts
            .iter()
            .filter(|(_, (c, e))| *c >= 2 && (has_call(e) || node_count(e) >= 3))
            .max_by(|(ka, (_, ea)), (kb, (_, eb))| {
                node_count(ea).cmp(&node_count(eb)).then_with(|| ka.cmp(kb))
            })
            .map(|(k, (_, e))| (k.clone(), e.clone()));
        match best {
            Some((key, e)) => {
                let body = replace_by_key(&expr, &key, name);
                expr = Sx::Let(name, Box::new(e), Box::new(body));
            }
            None => break,
        }
    }
    expr
}

/// The derivative `d(activation)/dx` as a scalar expression, auto-differentiated
/// from [`activation_expr`], algebraically simplified, then run through [`cse`]
/// to re-share the subterms differentiation duplicated — the single source for
/// the backward kernels. This is exactly the gradient of the forward we ship (so
/// gelu's grad is the derivative of the A&S forward, matching it by
/// construction). `round` uses the straight-through-estimator convention
/// (`grad = 1`) like the hand-written `activation_backward` kernels.
pub fn activation_grad_expr(act: Activation) -> Sx {
    match act {
        Activation::Round => lit(1.0),
        _ => cse(&simplify(&diff(&inline_lets(&activation_expr(act))))),
    }
}

/// `eval(activation_grad_expr(act), x)`.
pub fn eval_activation_grad(act: Activation, xv: f32) -> f32 {
    eval(&activation_grad_expr(act), xv)
}

// ── Emitters ────────────────────────────────────────────────────────────────

/// Target shader language for [`emit_case_body`] / the `*_activation_module`
/// helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Wgsl,
    Cuda,
    Msl,
    Glsl,
    /// OpenCL-C (Intel oneAPI via ocloc → SPIR-V). C-family with native `erf`,
    /// `rsqrt`, `rint`, `sign` builtins.
    OpenCl,
}

impl Lang {
    fn intr_name(self, i: Intr) -> &'static str {
        use Lang::*;
        match (self, i) {
            (_, Intr::Exp) => match self {
                Cuda => "expf",
                _ => "exp",
            },
            (_, Intr::Log) => match self {
                Cuda => "logf",
                _ => "log",
            },
            (_, Intr::Sqrt) => match self {
                Cuda => "sqrtf",
                _ => "sqrt",
            },
            (_, Intr::Rsqrt) => match self {
                Wgsl => "inverseSqrt",
                Cuda => "rsqrtf",
                Msl | OpenCl => "rsqrt",
                Glsl => "inversesqrt",
            },
            (_, Intr::Tanh) => match self {
                Cuda => "tanhf",
                _ => "tanh",
            },
            (_, Intr::Sin) => match self {
                Cuda => "sinf",
                _ => "sin",
            },
            (_, Intr::Cos) => match self {
                Cuda => "cosf",
                _ => "cos",
            },
            (_, Intr::Tan) => match self {
                Cuda => "tanf",
                _ => "tan",
            },
            (_, Intr::Atan) => match self {
                Cuda => "atanf",
                _ => "atan",
            },
            (_, Intr::Abs) => match self {
                Wgsl | Glsl => "abs",
                Cuda => "fabsf",
                Msl | OpenCl => "fabs",
            },
            (_, Intr::Floor) => match self {
                Cuda => "floorf",
                _ => "floor",
            },
            (_, Intr::Ceil) => match self {
                Cuda => "ceilf",
                _ => "ceil",
            },
            // ties-to-even
            (_, Intr::Round) => match self {
                Wgsl => "round",
                Cuda => "rintf",
                Msl | OpenCl => "rint",
                Glsl => "roundEven",
            },
            (_, Intr::Sign) => "sign", // Cuda handled specially in `render`
            (_, Intr::Erf) => "erf",   // handled specially in `render` per lang
        }
    }

    fn fmax(self) -> &'static str {
        match self {
            Lang::Cuda => "fmaxf",
            Lang::Msl | Lang::OpenCl => "fmax",
            _ => "max",
        }
    }
    fn fmin(self) -> &'static str {
        match self {
            Lang::Cuda => "fminf",
            Lang::Msl | Lang::OpenCl => "fmin",
            _ => "min",
        }
    }
    fn lit_suffix(self) -> &'static str {
        match self {
            Lang::Cuda | Lang::Msl | Lang::OpenCl => "f", // avoid promotion to double
            _ => "",
        }
    }
}

fn fmt_lit(v: f32, lang: Lang) -> String {
    if !v.is_finite() {
        return fmt_nonfinite(v, lang);
    }
    // Shortest round-trippable decimal, always with a fractional part.
    let mut s = format!("{v:?}");
    if !s.contains('.') && !s.contains('e') {
        s.push_str(".0");
    }
    format!("{s}{}", lang.lit_suffix())
}

/// Emit a `±inf` literal. None of the target languages has a portable `inf`
/// token, so reinterpret the IEEE-754 f32 infinity bit pattern
/// (`0x7f800000` / `0xff800000`). A `NaN` literal in the manifest is a bug — the
/// backends have no portable spelling for it — so reject it at codegen time.
fn fmt_nonfinite(v: f32, lang: Lang) -> String {
    assert!(
        !v.is_nan(),
        "rlxsl: NaN literals are not supported in the activation manifest"
    );
    let bits: u32 = if v > 0.0 { 0x7f80_0000 } else { 0xff80_0000 };
    match lang {
        Lang::Wgsl => format!("bitcast<f32>({bits:#010x}u)"),
        Lang::Cuda => format!("__uint_as_float({bits:#010x}u)"),
        Lang::Msl => format!("as_type<float>({bits:#010x}u)"),
        Lang::Glsl => format!("uintBitsToFloat({bits:#010x}u)"),
        Lang::OpenCl => format!("as_float({bits:#010x}u)"),
    }
}

/// Render `sx` to a target-language expression. `Let` nodes append
/// `let name = …;` (or `float name = …;`) statements to `stmts`; the returned
/// `String` is the trailing expression.
fn render(sx: &Sx, lang: Lang, stmts: &mut Vec<String>) -> String {
    let go = |a: &Sx, stmts: &mut Vec<String>| render(a, lang, stmts);
    match sx {
        Sx::X => "x".to_string(),
        Sx::Lit(v) => fmt_lit(*v, lang),
        Sx::Var(n) => (*n).to_string(),
        Sx::Let(n, v, body) => {
            let ve = go(v, stmts);
            let decl = match lang {
                Lang::Wgsl => format!("let {n} = {ve};"),
                Lang::Cuda | Lang::Msl | Lang::Glsl | Lang::OpenCl => {
                    format!("float {n} = {ve};")
                }
            };
            stmts.push(decl);
            go(body, stmts)
        }
        Sx::Neg(a) => format!("(-{})", go(a, stmts)),
        Sx::Add(a, b) => format!("({} + {})", go(a, stmts), go(b, stmts)),
        Sx::Sub(a, b) => format!("({} - {})", go(a, stmts), go(b, stmts)),
        Sx::Mul(a, b) => format!("({} * {})", go(a, stmts), go(b, stmts)),
        Sx::Div(a, b) => format!("({} / {})", go(a, stmts), go(b, stmts)),
        Sx::Max(a, b) => format!("{}({}, {})", lang.fmax(), go(a, stmts), go(b, stmts)),
        Sx::Min(a, b) => format!("{}({}, {})", lang.fmin(), go(a, stmts), go(b, stmts)),
        Sx::Clamp(a, lo, hi) => {
            let ae = go(a, stmts);
            let lo = fmt_lit(*lo, lang);
            let hi = fmt_lit(*hi, lang);
            match lang {
                Lang::Cuda => format!("fminf(fmaxf({ae}, {lo}), {hi})"),
                _ => format!("clamp({ae}, {lo}, {hi})"),
            }
        }
        Sx::Call(Intr::Erf, a) => match lang {
            // CUDA / OpenCL expose a scalar erf builtin (hardware-accurate).
            Lang::Cuda => format!("erff({})", go(a, stmts)),
            Lang::OpenCl => format!("erf({})", go(a, stmts)),
            // MSL / WGSL / GLSL have NO erf builtin → substitute the A&S 7.1.26
            // polynomial expansion (the math still lives once, here).
            Lang::Msl | Lang::Wgsl | Lang::Glsl => render(&erf_ansatz((**a).clone()), lang, stmts),
        },
        Sx::Call(Intr::Sign, a) if lang == Lang::Cuda => {
            // CUDA has no scalar `sign`; reproduce WGSL semantics (sign(0)==0).
            let ae = go(a, stmts);
            format!("((float)({ae} > 0.0f) - (float)({ae} < 0.0f))")
        }
        Sx::Call(i, a) => format!("{}({})", lang.intr_name(*i), go(a, stmts)),
        Sx::Select {
            on_false,
            on_true,
            a,
            cmp,
            b,
        } => {
            let ae = go(a, stmts);
            let be = go(b, stmts);
            let op = match cmp {
                Cmp::Gt => ">",
                Cmp::Ge => ">=",
                Cmp::Lt => "<",
                Cmp::Le => "<=",
            };
            let cond = format!("({ae} {op} {be})");
            let t = go(on_true, stmts);
            let f = go(on_false, stmts);
            match lang {
                // select(false_val, true_val, cond)
                Lang::Wgsl | Lang::Msl => format!("select({f}, {t}, {cond})"),
                // ternary (OpenCL's `select` takes an integer mask — use ?: )
                Lang::Cuda | Lang::Glsl | Lang::OpenCl => format!("({cond} ? {t} : {f})"),
            }
        }
    }
}

/// The body of one activation `case` (the `let` statements plus the final
/// value), *without* the surrounding `case N: { … }`. The value is bound to the
/// caller-provided `assign` target, e.g. `"return"` or `"y ="`.
/// Render `act`'s expression to `(statements, final_expr)` in `lang` — the
/// building block for both the `switch`-style modules and per-activation
/// kernels (e.g. Metal's `scalar_activation_kernels!`). `statements` are the
/// `let`/`float` temporaries; `final_expr` is the value, referencing `x` and
/// those temporaries.
pub fn emit_activation(act: Activation, lang: Lang) -> (Vec<String>, String) {
    let mut stmts = Vec::new();
    let expr = render(&activation_expr(act), lang, &mut stmts);
    (stmts, expr)
}

pub fn emit_case_body(act: Activation, lang: Lang, assign: &str) -> String {
    let (stmts, expr) = emit_activation(act, lang);
    let mut out = String::new();
    for s in &stmts {
        out.push_str(s);
        out.push(' ');
    }
    out.push_str(assign);
    out.push(' ');
    out.push_str(&expr);
    out.push(';');
    out
}

/// Which canonical activation-opcode scheme the target's kernel switch uses.
/// A backend property, not a language one: `ReluFirst` = CUDA / wgpu / ROCm
/// forward; `GeluFirst` = Vulkan / oneAPI forward (see `rlx_ir::opcodes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpcodeScheme {
    ReluFirst,
    GeluFirst,
}

/// Activations sorted by their opcode under `scheme` (`(opcode, activation)`).
fn cases(scheme: OpcodeScheme) -> Vec<(u32, Activation)> {
    let mut v: Vec<(u32, Activation)> = Activation::ALL
        .iter()
        .map(|&a| {
            let id = match scheme {
                OpcodeScheme::ReluFirst => a.opcode_relu_first(),
                OpcodeScheme::GeluFirst => a.opcode_gelu_first(),
            };
            (id, a)
        })
        .collect();
    v.sort_by_key(|(id, _)| *id);
    v
}

/// Shared body of the `*_activation_module` emitters: a `switch (op)` over the
/// activations in `scheme` order, each case binding `_y`, `default` returns `x`.
fn switch_module(
    lang: Lang,
    scheme: OpcodeScheme,
    signature: &str,
    let_kw: &str,
    default_and_close: &str,
) -> String {
    let mut s = String::from(
        "// @generated by rlxsl — do not edit by hand.\n\
         // Activation math is defined once in rlxsl::activation_expr.\n",
    );
    s.push_str(signature);
    for (id, act) in cases(scheme) {
        s.push_str(&format!(
            "        case {id}u: {{ {} return _y; }} // {act:?}\n",
            emit_case_body(act, lang, let_kw),
        ));
    }
    s.push_str(default_and_close);
    s
}

/// A complete WGSL `rlx_activation_apply(op, x)` function (the single source
/// that replaces the hand-written `switch` in `unary.wgsl`).
pub fn wgsl_activation_module(scheme: OpcodeScheme) -> String {
    switch_module(
        Lang::Wgsl,
        scheme,
        "fn rlx_activation_apply(op: u32, x: f32) -> f32 {\n    switch op {\n",
        "let _y =",
        "        default: { return x; }\n    }\n}\n",
    )
}

/// A complete CUDA `__device__` activation dispatch generated from the manifest.
pub fn cuda_activation_module(scheme: OpcodeScheme) -> String {
    switch_module(
        Lang::Cuda,
        scheme,
        "__device__ __forceinline__ float rlx_activation_apply(unsigned int op, float x) {\n    switch (op) {\n",
        "float _y =",
        "        default: return x;\n    }\n}\n",
    )
}

/// A complete MSL activation dispatch generated from the manifest.
pub fn msl_activation_module(scheme: OpcodeScheme) -> String {
    switch_module(
        Lang::Msl,
        scheme,
        "inline float rlx_activation_apply(uint op, float x) {\n    switch (op) {\n",
        "float _y =",
        "        default: return x;\n    }\n}\n",
    )
}

/// A complete GLSL activation dispatch generated from the manifest.
pub fn glsl_activation_module(scheme: OpcodeScheme) -> String {
    switch_module(
        Lang::Glsl,
        scheme,
        "float rlx_activation_apply(uint op, float x) {\n    switch (op) {\n",
        "float _y =",
        "        default: return x;\n    }\n}\n",
    )
}

/// A complete OpenCL-C activation dispatch (Intel oneAPI) generated from the
/// manifest — native `erf`/`rsqrt`/`rint`/`sign`.
pub fn opencl_activation_module(scheme: OpcodeScheme) -> String {
    switch_module(
        Lang::OpenCl,
        scheme,
        "inline float rlx_activation_apply(uint op, float x) {\n    switch (op) {\n",
        "float _y =",
        "        default: return x;\n    }\n}\n",
    )
}

// ── Backward (auto-differentiated) modules ──────────────────────────────────

/// Render `act`'s derivative to `(statements, expr)` — backward counterpart of
/// [`emit_activation`]. `statements` holds the CSE temporaries [`cse`] hoisted
/// (empty for grads with no repeated subterm); `expr` is the value referencing
/// `x` and those temporaries.
pub fn emit_activation_grad(act: Activation, lang: Lang) -> (Vec<String>, String) {
    let mut stmts = Vec::new();
    let expr = render(&activation_grad_expr(act), lang, &mut stmts);
    (stmts, expr)
}

/// Activations with a native backward kernel: the relu-first opcode `0..=17`
/// set (the tail decomposes at the AD level). Backward always dispatches with
/// relu-first ids on **every** backend.
fn backward_cases() -> Vec<(u32, Activation)> {
    cases(OpcodeScheme::ReluFirst)
        .into_iter()
        .filter(|(id, _)| *id < 18)
        .collect()
}

/// Shared body of the `*_activation_backward_module` emitters: `dx = deriv·dy`.
fn backward_switch(lang: Lang, signature: &str, default_and_close: &str) -> String {
    let mut s = String::from(
        "// @generated by rlxsl — do not edit by hand.\n\
         // dx = d(activation)/dx · dy; the derivative is auto-differentiated from\n\
         // rlxsl::activation_expr, so it exactly matches the forward we ship.\n",
    );
    s.push_str(signature);
    for (id, act) in backward_cases() {
        let (stmts, expr) = emit_activation_grad(act, lang);
        let pre = if stmts.is_empty() {
            String::new()
        } else {
            format!("{} ", stmts.join(" "))
        };
        s.push_str(&format!(
            "        case {id}u: {{ {pre}return ({expr}) * dy; }} // {act:?}\n"
        ));
    }
    s.push_str(default_and_close);
    s
}

/// CUDA `rlx_activation_backward(op, x, dy)` — one case per native-backward
/// activation, auto-differentiated from the forward manifest.
pub fn cuda_activation_backward_module() -> String {
    backward_switch(
        Lang::Cuda,
        "__device__ __forceinline__ float rlx_activation_backward(unsigned int op, float x, float dy) {\n    switch (op) {\n",
        "        default: return dy;\n    }\n}\n",
    )
}

/// WGSL backward dispatch.
pub fn wgsl_activation_backward_module() -> String {
    backward_switch(
        Lang::Wgsl,
        "fn rlx_activation_backward(op: u32, x: f32, dy: f32) -> f32 {\n    switch op {\n",
        "        default: { return dy; }\n    }\n}\n",
    )
}

/// GLSL backward dispatch.
pub fn glsl_activation_backward_module() -> String {
    backward_switch(
        Lang::Glsl,
        "float rlx_activation_backward(uint op, float x, float dy) {\n    switch (op) {\n",
        "        default: return dy;\n    }\n}\n",
    )
}

/// MSL backward dispatch.
pub fn msl_activation_backward_module() -> String {
    backward_switch(
        Lang::Msl,
        "inline float rlx_activation_backward(uint op, float x, float dy) {\n    switch (op) {\n",
        "        default: return dy;\n    }\n}\n",
    )
}

/// OpenCL-C backward dispatch (Intel oneAPI).
pub fn opencl_activation_backward_module() -> String {
    backward_switch(
        Lang::OpenCl,
        "inline float rlx_activation_backward(uint op, float x, float dy) {\n    switch (op) {\n",
        "        default: return dy;\n    }\n}\n",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent reference math (std-lib), restated here so the manifest AST is
    // checked against something other than itself.
    fn erf(x: f32) -> f32 {
        let s = if x >= 0.0 { 1.0 } else { -1.0 };
        let ax = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * ax);
        let p = ((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t;
        s * (1.0 - p * (-ax * ax).exp())
    }

    fn reference(act: Activation, x: f32) -> f32 {
        match act {
            Activation::Relu => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.clamp(-15.0, 15.0).tanh(),
            Activation::Exp => x.exp(),
            Activation::Log => x.ln(),
            Activation::Sqrt => x.sqrt(),
            Activation::Rsqrt => 1.0 / x.sqrt(),
            Activation::Neg => -x,
            Activation::Abs => x.abs(),
            Activation::Gelu => 0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2)),
            Activation::Silu => x / (1.0 + (-x).exp()),
            Activation::GeluApprox => {
                let inner = (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).clamp(-15.0, 15.0);
                0.5 * x * (1.0 + inner.tanh())
            }
            Activation::Round => x.round_ties_even(),
            Activation::Sin => x.sin(),
            Activation::Cos => x.cos(),
            Activation::Tan => x.tan(),
            Activation::Atan => x.atan(),
            Activation::Recip => 1.0 / x,
            Activation::Floor => x.floor(),
            Activation::Ceil => x.ceil(),
            Activation::Sign => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Activation::Softplus => x.max(0.0) + (1.0 + (-x.abs()).exp()).ln(),
            Activation::Elu => {
                if x > 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
            Activation::Erf => erf(x),
            Activation::HardSwish => x * (x + 3.0).clamp(0.0, 6.0) / 6.0,
            Activation::HardSigmoid => (x / 6.0 + 0.5).clamp(0.0, 1.0),
            Activation::Mish => x * (x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()).tanh(),
            Activation::Softsign => x / (1.0 + x.abs()),
            Activation::LogSigmoid => x.min(0.0) - (1.0 + (-x.abs()).exp()).ln(),
        }
    }

    #[test]
    fn eval_matches_reference_math() {
        // Domain-safe inputs (positive for log/sqrt/rsqrt, nonzero for recip).
        for &act in &Activation::ALL {
            for k in 0..40 {
                let base = -2.0 + 4.0 * (k as f32) / 39.0;
                let x = match act {
                    Activation::Log | Activation::Sqrt | Activation::Rsqrt => base.abs() + 0.3,
                    Activation::Recip => {
                        if base.abs() < 0.3 {
                            0.5
                        } else {
                            base
                        }
                    }
                    _ => base,
                };
                let got = eval_activation(act, x);
                let want = reference(act, x);
                let err = (got - want).abs() / want.abs().max(1.0);
                assert!(
                    err < 1e-6,
                    "{act:?} at x={x}: eval={got} reference={want} err={err:e}"
                );
            }
        }
    }

    #[test]
    fn as_forward_and_backward_are_near_f32_exact() {
        // Pin the manifest's numerical accuracy: the A&S 7.1.26 forward AND the
        // auto-differentiated backward must stay within a few f32 ULPs of libm
        // truth (measured ~5e-7). This is the real precision guard — the 5e-3
        // parity tolerance elsewhere exists only to absorb cross-vendor GPU
        // intrinsic differences, and would NOT catch a coarser polynomial here.
        let true_erf = |x: f64| libm::erf(x);
        let true_gelu = |x: f64| 0.5 * x * (1.0 + true_erf(x / std::f64::consts::SQRT_2));
        let true_gelu_grad = |x: f64| {
            let phi = 0.5 * (1.0 + true_erf(x / std::f64::consts::SQRT_2));
            let pdf = (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * x * x).exp();
            phi + x * pdf
        };
        let (mut e_erf, mut e_gelu, mut e_gg) = (0.0f64, 0.0f64, 0.0f64);
        let mut x = -6.0f64;
        while x <= 6.0 {
            let xf = x as f32;
            e_erf = e_erf.max((eval_activation(Activation::Erf, xf) as f64 - true_erf(x)).abs());
            e_gelu =
                e_gelu.max((eval_activation(Activation::Gelu, xf) as f64 - true_gelu(x)).abs());
            e_gg = e_gg
                .max((eval_activation_grad(Activation::Gelu, xf) as f64 - true_gelu_grad(x)).abs());
            x += 0.002;
        }
        assert!(e_erf < 2e-6, "erf abs error {e_erf:e} exceeds 2e-6");
        assert!(
            e_gelu < 2e-6,
            "gelu forward abs error {e_gelu:e} exceeds 2e-6"
        );
        assert!(e_gg < 2e-6, "gelu backward abs error {e_gg:e} exceeds 2e-6");
    }

    #[test]
    fn autodiff_grad_matches_finite_difference() {
        // The auto-differentiated grad must equal the central difference of the
        // FORWARD expression — i.e. it is exactly the gradient of what we ship.
        let h = 5e-4f32;
        for &act in &Activation::ALL {
            if act.opcode_relu_first() >= 18 {
                continue; // only the native-backward set (opcode 0..=17)
            }
            if act == Activation::Round {
                assert_eq!(eval_activation_grad(act, 0.37), 1.0); // STE convention
                continue;
            }
            for k in 0..25 {
                let base = -1.8 + 3.6 * (k as f32) / 24.0;
                let x = match act {
                    Activation::Log | Activation::Sqrt | Activation::Rsqrt => base.abs() + 0.5,
                    Activation::Recip if base.abs() < 0.4 => 0.6,
                    Activation::Tan => base * 0.6, // keep clear of π/2
                    // skip the kink at 0 for relu/abs
                    Activation::Relu | Activation::Abs if base.abs() < 0.4 => continue,
                    _ => base,
                };
                let g = eval_activation_grad(act, x);
                let fd = (eval_activation(act, x + h) - eval_activation(act, x - h)) / (2.0 * h);
                let err = (g - fd).abs() / fd.abs().max(1.0);
                assert!(
                    err < 5e-3,
                    "{act:?} grad at x={x}: autodiff={g} fd={fd} err={err:e}"
                );
            }
        }
    }

    #[test]
    fn backward_modules_cover_native_set() {
        assert_eq!(
            cuda_activation_backward_module().matches("case ").count(),
            18
        );
        assert!(wgsl_activation_backward_module().contains(") * dy"));
        assert!(glsl_activation_backward_module().contains("rlx_activation_backward"));
        assert!(msl_activation_backward_module().contains("return dy;"));
    }

    #[test]
    fn emitters_cover_all_activations_and_look_right() {
        for lang in [Lang::Wgsl, Lang::Cuda, Lang::Msl, Lang::Glsl, Lang::OpenCl] {
            for &act in &Activation::ALL {
                let body = emit_case_body(act, lang, "float _y =");
                assert!(!body.is_empty());
                assert!(body.contains("_y ="));
            }
        }
        // Spot-check per-language intrinsic spelling.
        let r = OpcodeScheme::ReluFirst;
        assert!(wgsl_activation_module(r).contains("inverseSqrt"));
        assert!(cuda_activation_module(r).contains("rsqrtf"));
        assert!(msl_activation_module(r).contains("rsqrt"));
        assert!(glsl_activation_module(r).contains("inversesqrt"));
        // OpenCL: native erf + rsqrt (unlike MSL/WGSL/GLSL which expand erf).
        let ocl = opencl_activation_module(r);
        assert!(ocl.contains("erf(") && !ocl.contains("erff("));
        assert!(ocl.contains("rsqrt(") && opencl_activation_backward_module().contains(") * dy"));
        // Every case + a default must be present.
        assert_eq!(wgsl_activation_module(r).matches("case ").count(), 29);
        // Vulkan's gelu-first scheme puts gelu at 0 and relu at 3.
        let g = glsl_activation_module(OpcodeScheme::GeluFirst);
        assert!(g.contains("case 0u: {") && g.contains("// Gelu"));
        assert!(g.contains("roundEven"));
    }

    /// Gelu now composes the `Intr::Erf` primitive, so CUDA/OpenCL pick up the
    /// native `erff`/`erf` (WGSL/MSL/GLSL still expand the shared A&S polynomial).
    #[test]
    fn gelu_uses_native_erf_where_available() {
        let r = OpcodeScheme::ReluFirst;
        // CUDA gelu forward + backward must call native erff (not the polynomial).
        assert!(cuda_activation_module(r).contains("erff("));
        assert!(cuda_activation_backward_module().contains("erff("));
        // OpenCL gelu uses native erf.
        assert!(opencl_activation_module(r).contains("erf("));
        // WGSL has no erf builtin → the polynomial's tell-tale coefficient.
        assert!(wgsl_activation_module(r).contains("0.3275911"));
    }

    /// Naive-but-sufficient delimiter balance: the generated kernels contain no
    /// string/char literals, so unbalanced `(){}[]` means a broken emitter.
    fn balanced(src: &str) -> bool {
        let mut stack = Vec::new();
        for c in src.chars() {
            match c {
                '(' | '{' | '[' => stack.push(c),
                ')' if stack.pop() != Some('(') => return false,
                '}' if stack.pop() != Some('{') => return false,
                ']' if stack.pop() != Some('[') => return false,
                _ => {}
            }
        }
        stack.is_empty()
    }

    /// The C-family emitters (CUDA/MSL/OpenCL) have no toolchain-free parser, so
    /// check them structurally: balanced delimiters, every activation covered,
    /// and a `default` arm — the failure modes a bad emitter would produce.
    #[test]
    fn c_family_modules_are_well_formed() {
        let n = Activation::ALL.len();
        for lang in [Lang::Cuda, Lang::Msl, Lang::OpenCl] {
            for scheme in [OpcodeScheme::ReluFirst, OpcodeScheme::GeluFirst] {
                let fwd = match lang {
                    Lang::Cuda => cuda_activation_module(scheme),
                    Lang::Msl => msl_activation_module(scheme),
                    Lang::OpenCl => opencl_activation_module(scheme),
                    _ => unreachable!(),
                };
                assert!(balanced(&fwd), "{lang:?} forward has unbalanced delimiters");
                assert_eq!(fwd.matches("case ").count(), n, "{lang:?} case count");
                assert!(fwd.contains("default"), "{lang:?} forward missing default");
            }
            let bwd = match lang {
                Lang::Cuda => cuda_activation_backward_module(),
                Lang::Msl => msl_activation_backward_module(),
                Lang::OpenCl => opencl_activation_backward_module(),
                _ => unreachable!(),
            };
            assert!(
                balanced(&bwd),
                "{lang:?} backward has unbalanced delimiters"
            );
            assert_eq!(bwd.matches("case ").count(), 18, "{lang:?} backward cases");
            let bin = match lang {
                Lang::Cuda => binary::cuda_binary_module(),
                Lang::Msl => binary::msl_binary_module(),
                Lang::OpenCl => binary::opencl_binary_module(),
                _ => unreachable!(),
            };
            assert!(balanced(&bin), "{lang:?} binary has unbalanced delimiters");
            assert_eq!(
                bin.matches("case ").count(),
                rlx_ir::op::BinaryOp::ALL.len(),
                "{lang:?} binary cases"
            );
            let cmp = match lang {
                Lang::Cuda => compare::cuda_compare_module(),
                Lang::Msl => compare::msl_compare_module(),
                Lang::OpenCl => compare::opencl_compare_module(),
                _ => unreachable!(),
            };
            assert!(balanced(&cmp), "{lang:?} compare has unbalanced delimiters");
            assert_eq!(
                cmp.matches("case ").count(),
                rlx_ir::op::CmpOp::ALL.len(),
                "{lang:?} compare cases"
            );
        }
    }

    /// Every generated WGSL module must actually parse + validate in naga — the
    /// real compile check for the wgpu path (previously only the `dw` prelude was
    /// validated). Exercises forward under both opcode schemes plus backward.
    #[test]
    fn wgsl_modules_validate_in_naga() {
        let validate = |module: &str, call: &str| {
            let src = format!(
                "{module}\n\
                 @group(0) @binding(0) var<storage, read_write> o: array<f32>;\n\
                 @compute @workgroup_size(1)\n\
                 fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
                 {call}\n}}\n"
            );
            let m = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("generated WGSL failed to parse: {e:?}\n{src}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&m)
            .unwrap_or_else(|e| panic!("generated WGSL failed validation: {e:?}"));
        };
        for scheme in [OpcodeScheme::ReluFirst, OpcodeScheme::GeluFirst] {
            validate(
                &wgsl_activation_module(scheme),
                "o[gid.x] = rlx_activation_apply(gid.x, o[gid.x]);",
            );
        }
        validate(
            &wgsl_activation_backward_module(),
            "o[gid.x] = rlx_activation_backward(gid.x, o[gid.x], o[gid.x]);",
        );
        validate(
            &binary::wgsl_binary_module(),
            "o[gid.x] = rlx_binary_apply(gid.x, o[gid.x], o[gid.x]);",
        );
        validate(
            &compare::wgsl_compare_module(),
            "o[gid.x] = rlx_compare_apply(gid.x, o[gid.x], o[gid.x]);",
        );
    }

    /// Every generated GLSL module must parse + validate in naga's GLSL frontend
    /// — the real compile check for the native-Vulkan path.
    #[test]
    fn glsl_modules_validate_in_naga() {
        let validate = |module: &str, call: &str| {
            let src = format!(
                "#version 450\n\
                 layout(local_size_x = 1) in;\n\
                 layout(std430, binding = 0) buffer Out {{ float o[]; }};\n\
                 {module}\n\
                 void main() {{ uint i = gl_GlobalInvocationID.x; {call} }}\n"
            );
            let mut frontend = naga::front::glsl::Frontend::default();
            let options = naga::front::glsl::Options {
                stage: naga::ShaderStage::Compute,
                defines: Default::default(),
            };
            let m = frontend
                .parse(&options, &src)
                .unwrap_or_else(|e| panic!("generated GLSL failed to parse: {e:?}\n{src}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&m)
            .unwrap_or_else(|e| panic!("generated GLSL failed validation: {e:?}"));
        };
        // Native Vulkan dispatches forward with gelu-first, backward relu-first.
        validate(
            &glsl_activation_module(OpcodeScheme::GeluFirst),
            "o[i] = rlx_activation_apply(i, o[i]);",
        );
        validate(
            &glsl_activation_backward_module(),
            "o[i] = rlx_activation_backward(i, o[i], o[i]);",
        );
        validate(
            &binary::glsl_binary_module(),
            "o[i] = rlx_binary_apply(i, o[i], o[i]);",
        );
        validate(
            &compare::glsl_compare_module(),
            "o[i] = rlx_compare_apply(i, o[i], o[i]);",
        );
    }

    /// CSE must collapse the subterms differentiation duplicates: silu backward
    /// used to evaluate `exp(clamp(-x,…))` four times — now once, bound to a temp.
    #[test]
    fn cse_shares_repeated_backward_subexpressions() {
        let (stmts, expr) = emit_activation_grad(Activation::Silu, Lang::Wgsl);
        let full = format!("{} {}", stmts.join(" "), expr);
        assert_eq!(
            full.matches("exp(").count(),
            1,
            "silu backward must call exp once, got:\n{full}"
        );
        assert!(
            !stmts.is_empty() && full.contains("_cse0"),
            "expected hoisted CSE temporaries:\n{full}"
        );
        // Sigmoid (3× exp) and gelu (repeated x·√½) must also collapse.
        let (s2, e2) = emit_activation_grad(Activation::Sigmoid, Lang::Cuda);
        assert_eq!(
            format!("{} {e2}", s2.join(" ")).matches("expf(").count(),
            1,
            "sigmoid backward must call exp once"
        );
        // CSE is value-preserving (belt-and-braces on top of the FD test).
        for k in 0..21 {
            let x = -3.0 + 6.0 * (k as f32) / 20.0;
            let s = 1.0 / (1.0 + (-x).exp());
            let want = s * (1.0 + x * (1.0 - s));
            assert!(
                (eval_activation_grad(Activation::Silu, x) - want).abs() < 1e-4,
                "silu grad changed at x={x}"
            );
        }
    }

    /// A non-finite literal (e.g. a clamp bound of ±∞) must render as a valid
    /// per-language token, never the bare `inf`/`NaN` that would fail to compile.
    #[test]
    fn non_finite_literals_emit_valid_tokens() {
        let clamp_inf = Sx::Clamp(Box::new(Sx::X), f32::NEG_INFINITY, f32::INFINITY);
        for lang in [Lang::Wgsl, Lang::Cuda, Lang::Msl, Lang::Glsl, Lang::OpenCl] {
            let mut stmts = Vec::new();
            let r = render(&clamp_inf, lang, &mut stmts);
            // Both bounds must take the bit-pattern path, never a bare `inf`
            // decimal literal (the `0x…u` masks prove fmt_lit didn't fall
            // through; a truly invalid token would also fail naga below).
            assert!(
                r.contains("0x7f800000u") && r.contains("0xff800000u"),
                "{lang:?} did not emit ±inf via bit pattern: {r}"
            );
        }
        // The WGSL spelling must actually validate in naga.
        let mut stmts = Vec::new();
        let expr = render(&clamp_inf, Lang::Wgsl, &mut stmts);
        let src = format!(
            "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n\
             @compute @workgroup_size(1)\n\
             fn main(@builtin(global_invocation_id) g: vec3<u32>) {{\n\
             let x = o[g.x]; o[g.x] = {expr};\n}}\n"
        );
        let m = naga::front::wgsl::parse_str(&src).expect("inf-literal WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&m)
        .expect("inf-literal WGSL validates");
    }

    #[test]
    #[should_panic(expected = "NaN literals")]
    fn nan_literal_is_rejected() {
        let mut stmts = Vec::new();
        let _ = render(&Sx::Lit(f32::NAN), Lang::Cuda, &mut stmts);
    }

    /// On macOS the real Metal compiler is available, so compile the generated
    /// MSL modules (forward + auto-differentiated backward + the double-single
    /// prelude) offline — the toolchain-free structural check in
    /// `c_family_modules_are_well_formed` can't catch a genuine MSL type error.
    /// Skips gracefully if the Metal toolchain isn't installed.
    #[cfg(target_os = "macos")]
    #[test]
    fn msl_modules_compile_with_metal() {
        use std::process::Command;
        let have_metal = Command::new("xcrun")
            .args(["-sdk", "macosx", "-f", "metal"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !have_metal {
            eprintln!("skipping msl_modules_compile_with_metal: no Metal toolchain");
            return;
        }
        let src = format!(
            "#include <metal_stdlib>\nusing namespace metal;\n{}\n{}\n{}\n{}\n{}",
            msl_activation_module(OpcodeScheme::ReluFirst),
            msl_activation_backward_module(),
            binary::msl_binary_module(),
            compare::msl_compare_module(),
            dw::double_single_prelude(Lang::Msl),
        );
        let dir = std::env::temp_dir();
        let metal_path = dir.join("rlxsl_msl_check.metal");
        let air_path = dir.join("rlxsl_msl_check.air");
        std::fs::write(&metal_path, &src).expect("write temp .metal");
        let out = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c"])
            .arg(&metal_path)
            .arg("-o")
            .arg(&air_path)
            .output()
            .expect("run xcrun metal");
        assert!(
            out.status.success(),
            "generated MSL failed to compile:\n{}\n--- source ---\n{src}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// clang has an OpenCL-C frontend, so compile the generated OpenCL modules
    /// (the oneAPI path) offline — the last emitter that had only the structural
    /// check in `c_family_modules_are_well_formed`. Skips if this clang can't
    /// compile even trivial OpenCL (no frontend / missing default headers).
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn opencl_modules_compile_with_clang() {
        use std::process::Command;
        let dir = std::env::temp_dir();
        let cl_args = [
            "-x",
            "cl",
            "-cl-std=CL1.2",
            "-Xclang",
            "-finclude-default-header",
            "-fsyntax-only",
        ];
        // Probe: can this clang compile trivial OpenCL C at all?
        let probe = dir.join("rlxsl_ocl_probe.cl");
        if std::fs::write(&probe, "__kernel void k(__global float* d){ d[0]=d[0]; }\n").is_err() {
            eprintln!("skipping opencl_modules_compile_with_clang: cannot write temp file");
            return;
        }
        let clang_ok = Command::new("clang")
            .args(cl_args)
            .arg(&probe)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !clang_ok {
            eprintln!("skipping opencl_modules_compile_with_clang: clang has no OpenCL frontend");
            return;
        }
        // Now the modules must compile; a failure here is a real emitter bug.
        let src = format!(
            "{}\n{}\n{}\n{}\n__kernel void probe(__global float* d) {{ \
             d[0] = rlx_activation_apply(3u, d[0]); \
             d[1] = rlx_activation_backward(0u, d[0], d[1]); \
             d[2] = rlx_binary_apply(6u, d[0], d[1]); \
             d[3] = rlx_compare_apply(2u, d[0], d[1]); }}\n",
            opencl_activation_module(OpcodeScheme::ReluFirst),
            opencl_activation_backward_module(),
            binary::opencl_binary_module(),
            compare::opencl_compare_module(),
        );
        let cl = dir.join("rlxsl_ocl_check.cl");
        std::fs::write(&cl, &src).expect("write temp .cl");
        let out = Command::new("clang")
            .args(cl_args)
            .arg(&cl)
            .output()
            .expect("run clang");
        assert!(
            out.status.success(),
            "generated OpenCL failed to compile:\n{}\n--- source ---\n{src}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// The CUDA emitter has no toolchain-free compiler here, but its device
    /// functions are plain scalar C++ once the CUDA-isms are stubbed — so compile
    /// them as C++ with `clang` to catch a malformed CUDA emitter without an
    /// NVIDIA toolchain. (`__device__`/`rsqrtf`/`__uint_as_float` are stubbed;
    /// `expf`/`erff`/`powf`/… are C99 `<math.h>`. Kernel entry points, which need
    /// `blockIdx` etc., are not included — only the generated modules.)
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn cuda_modules_compile_as_cxx_with_clang() {
        use std::process::Command;
        let dir = std::env::temp_dir();
        let cxx = ["-x", "c++", "-std=c++14", "-fsyntax-only"];
        let probe = dir.join("rlxsl_cu_probe.cpp");
        if std::fs::write(&probe, "int main(){return 0;}\n").is_err() {
            eprintln!("skipping cuda_modules_compile_as_cxx_with_clang: cannot write temp file");
            return;
        }
        let clang_ok = Command::new("clang")
            .args(cxx)
            .arg(&probe)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !clang_ok {
            eprintln!("skipping cuda_modules_compile_as_cxx_with_clang: no C++ clang");
            return;
        }
        let prelude = "#include <math.h>\n#include <string.h>\n\
             #define __device__\n#define __forceinline__ inline\n\
             #define rsqrtf(x) (1.0f / sqrtf(x))\n\
             static inline float __uint_as_float(unsigned int u){ float f; memcpy(&f,&u,sizeof(f)); return f; }\n";
        let src = format!(
            "{prelude}\n{}\n{}\n{}\n{}\n",
            cuda_activation_module(OpcodeScheme::ReluFirst),
            cuda_activation_backward_module(),
            binary::cuda_binary_module(),
            compare::cuda_compare_module(),
        );
        let cpp = dir.join("rlxsl_cu_check.cpp");
        std::fs::write(&cpp, &src).expect("write temp .cpp");
        let out = Command::new("clang")
            .args(cxx)
            .arg(&cpp)
            .output()
            .expect("run clang");
        assert!(
            out.status.success(),
            "generated CUDA (compiled as C++) failed:\n{}\n--- source ---\n{src}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// Format an f32 as a C++ float literal (round-trippable, with non-finites).
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    fn cxx_f32(v: f32) -> String {
        if v.is_nan() {
            "NAN".to_string()
        } else if v.is_infinite() {
            if v > 0.0 { "INFINITY" } else { "-INFINITY" }.to_string()
        } else {
            format!("{v:.9e}f")
        }
    }

    /// Compile-checking the C-family emitters catches syntax errors but not
    /// *semantic* ones (a wrong operator or constant). WGSL/Metal get numeric
    /// validation from hardware parity tests; CUDA/OpenCL have no GPU here. So
    /// compile the CUDA binary + compare modules **as C++** and *execute* them
    /// against the Rust `eval_*` oracle — a real numeric check with no GPU.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn cuda_binary_compare_numerically_match_oracle() {
        use rlx_ir::op::{BinaryOp, CmpOp};
        use std::process::Command;
        let dir = std::env::temp_dir();
        // Probe for a working C++ clang that produces runnable executables.
        let probe = dir.join("rlxsl_cuexe_probe.cpp");
        let probe_exe = dir.join("rlxsl_cuexe_probe");
        if std::fs::write(&probe, "int main(){return 0;}\n").is_err() {
            return;
        }
        let can_build = Command::new("clang")
            .args(["-x", "c++", "-std=c++14"])
            .arg(&probe)
            .arg("-o")
            .arg(&probe_exe)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !can_build {
            eprintln!("skipping cuda_binary_compare_numerically_match_oracle: no C++ clang");
            return;
        }

        // Oracle-defined operand grids (shifts need a small non-negative amount).
        let arith = [-3.0f32, -2.0, -1.5, -0.5, 0.5, 1.5, 2.0, 3.0];
        let ints = [-8.0f32, -3.0, 0.0, 1.0, 3.0, 7.0];
        let shifts = [0.0f32, 1.0, 2.0, 4.0];
        let mut bin = String::new();
        for &op in &BinaryOp::ALL {
            let (avs, bvs): (&[f32], &[f32]) = match op {
                BinaryOp::Shl | BinaryOp::Shr => (&ints, &shifts),
                BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => (&ints, &ints),
                _ => (&arith, &arith),
            };
            for &a in avs {
                for &b in bvs {
                    let w = binary::eval_binary(op, a, b);
                    bin.push_str(&format!(
                        "  {{ {}u, {}, {}, {} }},\n",
                        op.opcode(),
                        cxx_f32(a),
                        cxx_f32(b),
                        cxx_f32(w)
                    ));
                }
            }
        }
        let cvals = [-1.0f32, 0.0, 1.0, f32::NAN];
        let mut cmp = String::new();
        for &op in &CmpOp::ALL {
            for &a in &cvals {
                for &b in &cvals {
                    let w = compare::eval_compare(op, a, b);
                    cmp.push_str(&format!(
                        "  {{ {}u, {}, {}, {} }},\n",
                        op.opcode(),
                        cxx_f32(a),
                        cxx_f32(b),
                        cxx_f32(w)
                    ));
                }
            }
        }

        let prelude = "#include <math.h>\n#include <stdio.h>\n#include <string.h>\n\
             #define __device__\n#define __forceinline__ inline\n\
             #define rsqrtf(x) (1.0f / sqrtf(x))\n";
        let harness = format!(
            "struct Case {{ unsigned int op; float a; float b; float want; }};\n\
             static Case BIN[] = {{\n{bin}}};\n\
             static Case CMP[] = {{\n{cmp}}};\n\
             typedef float (*BinFn)(unsigned int, float, float);\n\
             static int run(const char* tag, Case* cs, int n, BinFn f) {{\n\
             int fails = 0;\n\
             for (int i = 0; i < n; ++i) {{\n\
               float got = f(cs[i].op, cs[i].a, cs[i].b);\n\
               float want = cs[i].want;\n\
               bool ok;\n\
               if (isnan(want)) ok = isnan(got);\n\
               else if (isnan(got) || isinf(got) != isinf(want)) ok = (got == want);\n\
               else {{ float d = fabsf(got - want); float s = fmaxf(fabsf(want), 1.0f); ok = d / s < 1e-4f; }}\n\
               if (!ok) {{ fprintf(stderr, \"%s op=%u a=%g b=%g got=%g want=%g\\n\", tag, cs[i].op, cs[i].a, cs[i].b, got, want); fails++; }}\n\
             }}\n\
             return fails;\n\
             }}\n\
             int main() {{\n\
             int f = 0;\n\
             f += run(\"binary\", BIN, sizeof(BIN)/sizeof(BIN[0]), rlx_binary_apply);\n\
             f += run(\"compare\", CMP, sizeof(CMP)/sizeof(CMP[0]), rlx_compare_apply);\n\
             return f == 0 ? 0 : 1;\n\
             }}\n"
        );
        let src = format!(
            "{prelude}\n{}\n{}\n{harness}",
            binary::cuda_binary_module(),
            compare::cuda_compare_module(),
        );
        let cpp = dir.join("rlxsl_cuexe_check.cpp");
        let exe = dir.join("rlxsl_cuexe_check");
        std::fs::write(&cpp, &src).expect("write temp .cpp");
        let build = Command::new("clang")
            .args(["-x", "c++", "-std=c++14"])
            .arg(&cpp)
            .arg("-o")
            .arg(&exe)
            .arg("-lm")
            .output()
            .expect("compile numeric harness");
        assert!(
            build.status.success(),
            "numeric harness failed to build:\n{}",
            String::from_utf8_lossy(&build.stderr)
        );
        let run = Command::new(&exe).output().expect("run numeric harness");
        assert!(
            run.status.success(),
            "CUDA emitter output diverged from the rlxsl oracle:\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}
