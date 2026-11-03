// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Correctly-rounded WGSL `/`, `sqrt` and `1/sqrt`** — the IEEE operations
//! the CPU oracle performs, which WGSL does not promise: `/` may be 2.5 ULP
//! off and `sqrt` is "inherited from `inverseSqrt`". How far off depends on the
//! driver. Metal's fast math gets about a quarter of quotients and of square
//! roots 1 ULP wrong; NVIDIA's Vulkan `sqrt` is exact, so the Newton step the
//! batch-norm kernel used to repair Metal's moved NVIDIA's roots 1 ULP instead.
//!
//! Both helpers start from the hardware result and correct it with exact
//! `fma` remainders, so they are right on either kind of driver:
//!
//! * `rlx_div_rn`: divides the significands `x, y ∈ [1/2, 1)` (`frexp`), one
//!   step `q + (x − y·q)/y`, then a midpoint test against the neighbour on the
//!   remainder's side (the step alone rounds its correction first, so near a
//!   midpoint it kept the first quotient's rounding: native Metal's
//!   `1/1.99999988` stayed 0.5 where Chrome's gave 0.50000006), and `ldexp`
//!   back. On the significands every remainder is a normal number: dividing
//!   `a` itself, a small dividend's remainder went subnormal and Metal's fast
//!   math flushed it (`4.6e-36 / 1.08e-6` 1 ULP off natively, 4% of such
//!   pairs). A zero or non-finite first quotient is already the IEEE answer
//!   and is kept — the correction would turn `x/0`'s infinity into NaN.
//! * `rlx_sqrt_rn`: step to a neighbour while the root lies past the midpoint.
//!   With `y_lo < y_hi` adjacent and `r(c) = x − c²` (exact when `c` is within an
//!   ULP), the root exceeds their midpoint iff `r(y_lo) + r(y_hi) + u²/2 > 0`,
//!   `u = y_hi − y_lo`; near the midpoint the two remainders cancel exactly
//!   (Sterbenz). Inputs beyond `2^±60` are scaled by `2^∓100` so the remainders
//!   neither underflow nor overflow, and the root is scaled back by `2^±50`.
//!
//! * `rlx_exp`: Cody–Waite reduction `x = n·ln2 + r` (ln2 split in two, each
//!   step one `fma`), Cephes' degree-5 polynomial in `fma`s, and `ldexp`.
//!   Within 1 ULP of the correctly rounded value, 99% of it exactly. The
//!   builtin `exp` is not used because its rounding depends on the compiler's
//!   math mode: native wgpu compiles Metal in Apple's fast math, Chrome (Dawn)
//!   in safe math, and on the same GPU sigmoid(1.16e-7) came out 0.5 natively
//!   and 0.50000006 in the browser (BCG-FM's squeeze-excite gate). A subnormal
//!   result (`x` below about −87.3) is rounded from the bits, not by `ldexp`,
//!   which Metal's fast math flushes there and a browser does not.
//! * `rlx_log`: Cephes' `logf` in `fma`s over `x = m·2^e` split from the bits
//!   (a subnormal `x` normalized with integer ops, out of the flush's reach).
//!   Within 1 ULP.
//! * `rlx_tanh`: Cephes' odd polynomial below `|x| = 0.625`, `1 − 2/(e^{2|x|}+1)`
//!   through `rlx_exp` and `rlx_div_rn` above. Within 2 ULP.
//!
//! [`wgsl_numeric_helpers`] and its native-operator twin
//! [`wgsl_fast_numeric_helpers`] are WGSL files (`src/wgsl/`) so a non-Rust
//! host — rlx-wgpu's Deno WebGPU check — appends the same text.
//!
//! The helpers live in one string so the binary, activation and batch-norm
//! shaders share a single copy. The WGSL modules call them but do not embed
//! them: a shader that combines modules (the fused elementwise region takes
//! both binary and activation) would otherwise define them twice. Prepend
//! [`wgsl_numeric_helpers`] exactly once per shader.

/// The functions [`wgsl_numeric_helpers`] defines: a shader calling any of them
/// needs the block once.
pub const WGSL_HELPER_NAMES: &[&str] = &[
    "rlx_div_rn",
    "rlx_sqrt_rn",
    "rlx_rsqrt_rn",
    "rlx_exp",
    "rlx_log",
    "rlx_tanh",
    "rlx_sin",
    "rlx_cos",
];

/// WGSL definitions of the [`WGSL_HELPER_NAMES`] functions.
pub fn wgsl_numeric_helpers() -> &'static str {
    WGSL_NUMERIC_HELPERS
}

/// The same functions as the plain WGSL builtins (`/`, `sqrt`, `inverseSqrt`, `exp`, `log`, `tanh`):
/// up to 2.5 ULP off (and math-mode dependent) on some drivers, but a few instructions shorter. Opt-in
/// only (rlx-wgpu's `set_fast_math`); exact is the default everywhere.
pub fn wgsl_fast_numeric_helpers() -> &'static str {
    WGSL_FAST_NUMERIC_HELPERS
}

const WGSL_FAST_NUMERIC_HELPERS: &str = include_str!("wgsl/fast_numeric_helpers.wgsl");

const WGSL_NUMERIC_HELPERS: &str = include_str!("wgsl/numeric_helpers.wgsl");
