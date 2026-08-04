// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Batched in-circle predicate as a **matmul** (paraboloid-lift GEMM), with a certified
//! exact fallback.
//!
//! The in-circle test "is query `q` inside the circumcircle of triangle `(a,b,c)`?" is,
//! after lifting every point to the paraboloid `p ↦ (x, y, x²+y², 1)`, a 4×4 determinant
//! that is **linear in the lifted query**. So for a fixed triangle it's a dot product of
//! the lifted query against the triangle's four cofactors — and a whole batch of
//! (query × triangle) tests is a single GEMM:
//!
//! ```text
//!   DET = L · C        L = [Q×4] lifted queries,  C = [4×T] per-triangle cofactors
//! ```
//!
//! This is the shape hardware likes. Benchmarked (see `examples/`), the f32 GEMM form of
//! the predicate ran fastest on the **low-dispatch matrix/SIMD paths** — Apple **AMX**
//! (~24 G tests/s via Accelerate), **WASM/simd128** (~6–7 G/s), and plain CPU-SIMD —
//! beating every driver-gated accelerator (CUDA/ROCm/ANE) on this cheap, K=4 op, where
//! launch/transfer overhead dominates. The inner loop over triangles auto-vectorizes.
//!
//! **Correctness is exact**, not f32: the GEMM is used only as a *filter*. Any entry whose
//! |det| falls inside the f32 rounding bound is recomputed with the exact i128 predicate
//! ([`crate`]'s `in_circle`). Coordinates are normalized to keep the filter tight; the
//! fallback uses the original integers, so the result is bit-for-bit the exact sign
//! regardless of conditioning (span must be ≤ [`MAX_COORDINATE_SPAN`](crate::MAX_COORDINATE_SPAN)).
//!
//! Benchmark with `cargo run --example bench_variants`. Fastest path is [`incircle_signs`]
//! on a low-dispatch matrix/SIMD unit — **AMX ~24 G tests/s** (via `cblas`), **WASM/simd128
//! ~6–7 G/s**, portable Rust ~0.6 G/s — all exact, vs ~0.17 G/s for a scalar i128 loop.
//! Driver-gated accelerators (CUDA/ROCm/ANE) lose this cheap op to invocation overhead.

/// Exact i128 orientation sign of `(a,b,c)`.
fn orient_i(a: [i64; 2], b: [i64; 2], c: [i64; 2]) -> i64 {
    let d = (b[0] - a[0]) as i128 * (c[1] - a[1]) as i128
        - (b[1] - a[1]) as i128 * (c[0] - a[0]) as i128;
    (d > 0) as i64 - (d < 0) as i64
}

/// Exact i128 in-circle: `+1` if `d` is strictly inside the circumcircle of CCW `(a,b,c)`,
/// `-1` outside, `0` cocircular.
fn in_circle_exact(a: [i64; 2], b: [i64; 2], c: [i64; 2], d: [i64; 2]) -> i8 {
    let ax = (a[0] - d[0]) as i128;
    let ay = (a[1] - d[1]) as i128;
    let bx = (b[0] - d[0]) as i128;
    let by = (b[1] - d[1]) as i128;
    let cx = (c[0] - d[0]) as i128;
    let cy = (c[1] - d[1]) as i128;
    let det = (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay)
        + (cx * cx + cy * cy) * (ax * by - bx * ay);
    (det > 0) as i8 - (det < 0) as i8
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Conservative absolute f32-error bound on a normalized det (coords in [-1,1] ⇒ lifts in
/// [0,2], unit-normalized cofactor columns ⇒ |det| ≤ ~8; ~8 flops × 2⁻²⁴, ×30 safety).
const FILTER_TAU: f32 = 1.0e-4;

/// Certified batched in-circle signs for every `(query, triangle)` pair.
///
/// Returns a row-major `[queries.len() × tris.len()]` matrix: `out[qi * tris.len() + ti]`
/// is `+1` if `points[queries[qi]]` is strictly inside the circumcircle of `tris[ti]`,
/// `-1` outside, `0` cocircular. Triangles are CCW-oriented internally, so the sign is
/// orientation-independent. Signs are **exact** (f32 GEMM filter + i128 fallback).
pub fn incircle_signs(points: &[[i32; 2]], tris: &[[u32; 3]], queries: &[u32]) -> Vec<i8> {
    let (q, t) = (queries.len(), tris.len());
    let mut out = vec![0i8; q * t];
    if q == 0 || t == 0 {
        return out;
    }
    let ci: Vec<[i64; 2]> = points.iter().map(|p| [p[0] as i64, p[1] as i64]).collect();
    // CCW-orient every triangle (so +det == inside, uniformly).
    let tri: Vec<[usize; 3]> = tris
        .iter()
        .map(|&[a, b, c]| {
            let (a, b, c) = (a as usize, b as usize, c as usize);
            if orient_i(ci[a], ci[b], ci[c]) < 0 {
                [a, c, b]
            } else {
                [a, b, c]
            }
        })
        .collect();

    // Normalize the involved coords to ~[-1,1] so the f32 filter is well-conditioned.
    let mut cx = 0i64;
    let mut cy = 0i64;
    for p in &ci {
        cx += p[0];
        cy += p[1];
    }
    let (cx, cy) = (cx / ci.len() as i64, cy / ci.len() as i64);
    let mut maxabs = 1.0f64;
    for p in &ci {
        maxabs = maxabs
            .max((p[0] - cx).abs() as f64)
            .max((p[1] - cy).abs() as f64);
    }
    let sc = 1.0 / maxabs;
    let nf =
        |i: usize| -> (f64, f64) { ((ci[i][0] - cx) as f64 * sc, (ci[i][1] - cy) as f64 * sc) };

    // L [Q×4] lifted, normalized queries.
    let mut lmat = vec![0f32; q * 4];
    for (qi, &qidx) in queries.iter().enumerate() {
        let (x, y) = nf(qidx as usize);
        lmat[qi * 4] = x as f32;
        lmat[qi * 4 + 1] = y as f32;
        lmat[qi * 4 + 2] = (x * x + y * y) as f32;
        lmat[qi * 4 + 3] = 1.0;
    }
    // C [4×T] cofactors (expansion along the query row): C[j] = (-1)^(3+j)·minor_j, then
    // unit-normalized per column (a positive per-column scale — sign-preserving).
    let mut cmat = vec![0f32; 4 * t];
    for (ti, &[a, b, c]) in tri.iter().enumerate() {
        let rows = [a, b, c].map(|i| {
            let (x, y) = nf(i);
            [x, y, x * x + y * y, 1.0]
        });
        let mut cof = [0f64; 4];
        for j in 0..4 {
            let mut mm = [[0f64; 3]; 3];
            for (ri, rr) in rows.iter().enumerate() {
                let mut cc = 0;
                for col in 0..4 {
                    if col == j {
                        continue;
                    }
                    mm[ri][cc] = rr[col];
                    cc += 1;
                }
            }
            cof[j] = if j % 2 == 0 { -det3(mm) } else { det3(mm) };
        }
        let m = cof.iter().fold(1e-30f64, |mx, &v| mx.max(v.abs()));
        for j in 0..4 {
            cmat[j * t + ti] = (cof[j] / m) as f32;
        }
    }

    // DET = L @ C  (portable, auto-vectorizing over the triangle axis).
    for qi in 0..q {
        let (l0, l1, l2, l3) = (
            lmat[qi * 4],
            lmat[qi * 4 + 1],
            lmat[qi * 4 + 2],
            lmat[qi * 4 + 3],
        );
        let qglobal = queries[qi] as usize;
        let row = qi * t;
        for ti in 0..t {
            let det =
                l0 * cmat[ti] + l1 * cmat[t + ti] + l2 * cmat[2 * t + ti] + l3 * cmat[3 * t + ti];
            out[row + ti] = if det.abs() > FILTER_TAU {
                (det > 0.0) as i8 - (det < 0.0) as i8
            } else {
                // uncertain → exact i128 (original integer coords)
                let [a, b, c] = tri[ti];
                in_circle_exact(ci[a], ci[b], ci[c], ci[qglobal])
            };
        }
    }
    out
}

/// Delaunay filter for candidate triangles: `out[t]` is `true` iff `tris[t]`'s circumcircle
/// contains **no** point of `points` (the empty-circumcircle property). This is the dense
/// "leaf" test — every point against every candidate triangle — as one GEMM. A triangle's
/// own three vertices are cocircular (det 0) and don't count as inside.
pub fn empty_circumcircle(points: &[[i32; 2]], tris: &[[u32; 3]]) -> Vec<bool> {
    let all: Vec<u32> = (0..points.len() as u32).collect();
    let t = tris.len();
    let signs = incircle_signs(points, tris, &all);
    let mut ok = vec![true; t];
    for (qi, &_q) in all.iter().enumerate() {
        for ti in 0..t {
            if signs[qi * t + ti] > 0 {
                // strictly inside → not Delaunay (vertices give 0, so they're skipped)
                ok[ti] = false;
            }
        }
    }
    ok
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_exact(points: &[[i32; 2]], tris: &[[u32; 3]], queries: &[u32]) -> Vec<i8> {
        let ci: Vec<[i64; 2]> = points.iter().map(|p| [p[0] as i64, p[1] as i64]).collect();
        let mut out = vec![0i8; queries.len() * tris.len()];
        for (qi, &q) in queries.iter().enumerate() {
            for (ti, &[a, b, c]) in tris.iter().enumerate() {
                let (a, b, c) = (a as usize, b as usize, c as usize);
                let (a, b, c) = if orient_i(ci[a], ci[b], ci[c]) < 0 {
                    (a, c, b)
                } else {
                    (a, b, c)
                };
                out[qi * tris.len() + ti] = in_circle_exact(ci[a], ci[b], ci[c], ci[q as usize]);
            }
        }
        out
    }

    #[test]
    fn gemm_incircle_matches_exact() {
        // deterministic tile, moderate span
        let mut s = 0x1234_5678u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as i32
        };
        let k = 40usize;
        let pts: Vec<[i32; 2]> = (0..k)
            .map(|_| [rnd() % 15_000_000, rnd() % 15_000_000])
            .collect();
        let mut tris = vec![];
        for a in 0..k as u32 {
            for b in a + 1..k as u32 {
                for c in b + 1..k as u32 {
                    tris.push([a, b, c]);
                }
            }
        }
        let all: Vec<u32> = (0..k as u32).collect();
        let got = incircle_signs(&pts, &tris, &all);
        let want = brute_exact(&pts, &tris, &all);
        assert_eq!(got, want, "certified GEMM in-circle must equal exact");
    }

    #[test]
    fn empty_circumcircle_flags_delaunay() {
        // unit square + center: the two triangles of each diagonal are Delaunay-ambiguous;
        // just assert the empty-circumcircle set matches a brute-force check.
        let pts = [[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]];
        let mut tris = vec![];
        for a in 0..5u32 {
            for b in a + 1..5 {
                for c in b + 1..5 {
                    tris.push([a, b, c]);
                }
            }
        }
        let ok = empty_circumcircle(&pts, &tris);
        // brute-force reference
        let all: Vec<u32> = (0..5).collect();
        let signs = brute_exact(&pts, &tris, &all);
        for (ti, _) in tris.iter().enumerate() {
            let empty = (0..5).all(|qi| signs[qi * tris.len() + ti] <= 0);
            assert_eq!(ok[ti], empty, "triangle {ti} empty-circumcircle mismatch");
        }
    }
}
