// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact integer geometric predicates. Points are `[i32; 2]`. The orientation
//! and in-circle determinants are evaluated in i64 (fast path) or i128 (wide
//! path); the width is chosen once from the coordinate span, mirroring the C++
//! `delaunay32` certification.

/// Largest equal x/y span certified for the i64 fast path.
pub const FAST_COORDINATE_SPAN: i64 = 29_609;
/// Largest equal x/y span certified for the i128 wide path.
pub const MAX_COORDINATE_SPAN: i64 = 1_940_470_527;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PredicateWidth {
    Int64,
    Int128,
    Unsupported,
}

/// Choose the predicate width from the two coordinate spans.
pub fn predicate_width(sx: i64, sy: i64) -> PredicateWidth {
    let span = sx.max(sy);
    if span <= FAST_COORDINATE_SPAN {
        PredicateWidth::Int64
    } else if span <= MAX_COORDINATE_SPAN {
        PredicateWidth::Int128
    } else {
        PredicateWidth::Unsupported
    }
}

/// Exact predicates over some wide integer type, selected once at the top.
pub(crate) trait Pred: Send + Sync {
    /// Whether coordinates fit 16-bit Morton (i64 fast path only).
    const USE_MORTON: bool;
    /// Sign of orientation of (a,b,c): +1 CCW, 0 collinear, -1 CW.
    fn orient(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i32;
    /// > 0 iff d is strictly inside the circumcircle of the CCW triangle a,b,c.
    fn in_circle(a: [i32; 2], b: [i32; 2], c: [i32; 2], d: [i32; 2]) -> i32;
}

macro_rules! orient_body {
    ($t:ty, $a:expr, $b:expr, $c:expr) => {{
        let v = ($b[0] as $t - $a[0] as $t) * ($c[1] as $t - $a[1] as $t)
            - ($b[1] as $t - $a[1] as $t) * ($c[0] as $t - $a[0] as $t);
        (v > 0) as i32 - (v < 0) as i32
    }};
}

macro_rules! in_circle_body {
    ($t:ty, $a:expr, $b:expr, $c:expr, $d:expr) => {{
        let ax = $a[0] as $t - $d[0] as $t;
        let ay = $a[1] as $t - $d[1] as $t;
        let bx = $b[0] as $t - $d[0] as $t;
        let by = $b[1] as $t - $d[1] as $t;
        let cx = $c[0] as $t - $d[0] as $t;
        let cy = $c[1] as $t - $d[1] as $t;
        let det = (ax * ax + ay * ay) * (bx * cy - cx * by)
            - (bx * bx + by * by) * (ax * cy - cx * ay)
            + (cx * cx + cy * cy) * (ax * by - bx * ay);
        (det > 0) as i32 - (det < 0) as i32
    }};
}

/// Fast path: both predicates in i64 (span <= FAST_COORDINATE_SPAN).
pub(crate) struct PredFast;
impl Pred for PredFast {
    const USE_MORTON: bool = true;
    #[inline(always)]
    fn orient(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i32 {
        orient_body!(i64, a, b, c)
    }
    #[inline(always)]
    fn in_circle(a: [i32; 2], b: [i32; 2], c: [i32; 2], d: [i32; 2]) -> i32 {
        in_circle_body!(i64, a, b, c, d)
    }
}

/// Wide path: orientation in i64 (safe to ~3e9 span), in-circle mostly in i64.
pub(crate) struct PredWide;
impl Pred for PredWide {
    const USE_MORTON: bool = false;
    #[inline(always)]
    fn orient(a: [i32; 2], b: [i32; 2], c: [i32; 2]) -> i32 {
        orient_body!(i64, a, b, c)
    }
    // Ablation build (`--features abl_slowpred`): the naive all-i128 determinant,
    // to measure the i64-inner form's contribution. Identical result, ~2.5× more
    // wide multiplies.
    #[cfg(feature = "abl_slowpred")]
    #[inline(always)]
    fn in_circle(a: [i32; 2], b: [i32; 2], c: [i32; 2], d: [i32; 2]) -> i32 {
        in_circle_body!(i128, a, b, c, d)
    }

    #[cfg(not(feature = "abl_slowpred"))]
    #[inline(always)]
    fn in_circle(a: [i32; 2], b: [i32; 2], c: [i32; 2], d: [i32; 2]) -> i32 {
        let ax = a[0] as i64 - d[0] as i64;
        let ay = a[1] as i64 - d[1] as i64;
        let bx = b[0] as i64 - d[0] as i64;
        let by = b[1] as i64 - d[1] as i64;
        let cx = c[0] as i64 - d[0] as i64;
        let cy = c[1] as i64 - d[1] as i64;
        let sa = ax * ax + ay * ay;
        let sb = bx * bx + by * by;
        let sc = cx * cx + cy * cy;
        let bc = bx * cy - cx * by;
        let ac = ax * cy - cx * ay;
        let ab = ax * by - bx * ay;
        let det = sa as i128 * bc as i128 - sb as i128 * ac as i128 + sc as i128 * ab as i128;
        (det > 0) as i32 - (det < 0) as i32
    }
}
