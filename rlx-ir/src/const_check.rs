// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Compile-time shape / rank assertions (plan #77).
//!
//! Borrowed from MAX's `comptime assert c.rank == 2, "c must be rank 2"`
//! pattern. Where shapes are known at the point of macro expansion,
//! verify them at compile time via `const fn` helpers; runtime
//! `Shape` checks remain for genuinely-dynamic cases.
//!
//! The Rust spelling uses `const fn` predicates plus a small
//! `static_assert!` macro that wraps a `const _: () = assert!(...)`
//! evaluation. Failures surface as compile errors with the full
//! const-evaluation chain, so the user sees exactly which check
//! tripped.
//!
//! These tools are most useful inside macros (e.g. a future
//! `tensor!{ shape: [8, 8] }` literal that wants to check the
//! shape is non-empty + has the expected rank). Today they're
//! exposed as building blocks.

/// Compile-time assert. Wraps the const-evaluation idiom in a
/// terse macro so call sites read like `static_assert!(cond)`.
///
/// ```
/// rlx_ir::static_assert!(1 + 1 == 2);
/// rlx_ir::static_assert!(usize::MAX > 0, "platform sanity");
/// ```
///
/// Failure is a compile error pointing at the macro call site.
#[macro_export]
macro_rules! static_assert {
    ($cond:expr) => {
        const _: () = assert!($cond);
    };
    ($cond:expr, $msg:literal) => {
        const _: () = assert!($cond, $msg);
    };
}

/// Const-evaluable rank check.
pub const fn rank_eq(rank: usize, expected: usize) -> bool {
    rank == expected
}

/// Const-evaluable rank-at-least check.
pub const fn rank_at_least(rank: usize, min: usize) -> bool {
    rank >= min
}

/// Const product of a fixed-size dim array. Useful for asserting
/// a flat element count matches a structured shape at compile
/// time.
pub const fn shape_elements<const N: usize>(dims: [usize; N]) -> usize {
    let mut total = 1usize;
    let mut i = 0;
    while i < N {
        total *= dims[i];
        i += 1;
    }
    total
}

/// Const check that `lhs` and `rhs` shapes are broadcast-compat
/// per the standard rules: equal at every dim, or one of them is
/// 1 at that dim. Both shapes must have the same rank (left-pad
/// the shorter externally if the runtime shape supports it).
pub const fn broadcastable<const N: usize>(lhs: [usize; N], rhs: [usize; N]) -> bool {
    let mut i = 0;
    while i < N {
        let l = lhs[i];
        let r = rhs[i];
        if !(l == r || l == 1 || r == 1) {
            return false;
        }
        i += 1;
    }
    true
}

/// Const check for the matmul rank/dim contract:
/// `[m, k] @ [k, n] → [m, n]`. Returns true iff the inner dims
/// agree.
pub const fn matmul_compat(lhs_m: usize, lhs_k: usize, rhs_k: usize, rhs_n: usize) -> bool {
    let _ = lhs_m;
    let _ = rhs_n;
    lhs_k == rhs_k
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time assertions (each one fails the build if wrong).
    static_assert!(rank_eq(2, 2));
    static_assert!(rank_at_least(3, 2));
    static_assert!(shape_elements([2, 3, 4]) == 24);
    static_assert!(broadcastable([4, 1, 8], [4, 6, 1]));
    static_assert!(!broadcastable([4, 5], [3, 5]));
    static_assert!(matmul_compat(8, 16, 16, 32));
    static_assert!(!matmul_compat(8, 16, 32, 16));

    // Runtime smoke tests too — the const fns are also useful at
    // runtime for shape-inference helpers.
    #[test]
    fn const_helpers_at_runtime() {
        assert!(rank_eq(2, 2));
        assert!(!rank_eq(2, 3));
        assert_eq!(shape_elements([2, 3, 4]), 24);
        assert!(broadcastable([4, 1, 8], [4, 6, 1]));
        assert!(matmul_compat(8, 16, 16, 32));
    }
}
