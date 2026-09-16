// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Powell's conjugate-direction method — derivative-free local minimization.
//!
//! Fills the gap between [`crate::adam_opt_nd`], which needs gradients (or
//! finite differences of them), and [`crate::cmaes`], which is stochastic and
//! needs a population. Powell is deterministic, uses only function values, and
//! converges superlinearly on smooth objectives — the right tool when each
//! evaluation is expensive and the landscape is locally quadratic.
//!
//! ## Why the direction update matters
//!
//! Minimizing along each coordinate axis in turn (cyclic coordinate descent)
//! stalls on a valley that runs diagonally: every axis-aligned step is blocked
//! by the valley walls, so progress per sweep shrinks with the coupling.
//! Powell replaces one coordinate direction per iteration with the net
//! displacement of the whole sweep, which points *along* the valley.
//!
//! The replacement uses Brent's rule: discard the direction that gave the
//! largest decrease, and only when an extrapolation test says the new direction
//! is worth more than the one it displaces. Replacing unconditionally makes the
//! direction set collapse toward linear dependence and the search degrades onto
//! a subspace.
//!
//! ## The direction set has to be restarted
//!
//! Brent's discard rule alone is not enough. Left to run, the sweep replaces
//! the original coordinate axes one at a time with progressively smaller,
//! nearly parallel displacements; the set drifts toward linear dependence and
//! the search crawls along a subspace while still reporting progress. Resetting
//! the directions to the coordinate axes once the whole set has been turned
//! over — [`PowellConfig::restart_after`], `n` replacements by default — costs
//! nothing and is what makes the method reliable.
//!
//! Measured on a 2-D quadratic whose axes sit 45° off the coordinate frame,
//! from `(3, -3)`, against cyclic coordinate descent using the same line search
//! and against SciPy's `minimize(method='Powell')` with matching tolerances:
//!
//! | axis ratio | this, restart on | this, restart off | coordinate descent | SciPy Powell |
//! |---|---|---|---|---|
//! | 3:1    | `1.5e-31` / 60 evals  | `1.5e-31` / 60  | `2.9e-21` / 300 | `1.1e-32` / 45  |
//! | 5:1    | `7.1e-31` / 59 evals  | `7.1e-31` / 59  | `3.1e-12` / 300 | `7.9e-32` / 64  |
//! | 8:1    | `1.8e-32` / 159 evals | `2.9e-1` / 300  | `2.2e-7` / 300  | `2.9e-1` / 300  |
//! | 1000:1 | `3.4e-30` / 188 evals | `1.6e1` / 300   | `1.6e1` / 300   | `1.5e1` / 300   |
//!
//! The last two rows are the point. Without the restart, this implementation
//! and SciPy's both stall on a strongly anisotropic quadratic and are beaten by
//! plain coordinate descent. With it, both cases converge to machine precision
//! in well under the budget.
//!
//! On Beale from `(1, 1)` the same effect appears on a non-quadratic: `5.2e-25`
//! in 311 evaluations with the restart, `3.1e-4` after 2000 without, against
//! SciPy's `3.5e-19` in 438.
//!
//! ## Bounds
//!
//! Powell is unconstrained; [`Bbox`] is honored by restricting each line search
//! to the interval where the line stays inside the box, rather than by clipping
//! evaluation points. Clipping would present the line search with a flat region
//! outside the box and invite it to converge onto the wall.

use crate::{BboSolution, Bbox};

#[derive(Clone, Debug)]
pub struct PowellConfig {
    /// Maximum objective evaluations.
    pub max_evals: usize,
    /// Relative decrease below which a sweep counts as converged.
    pub ftol: f64,
    /// Relative tolerance for each 1-D line minimization.
    pub xtol: f64,
    /// Initial direction length, as a fraction of the bbox width per dimension.
    ///
    /// The initial directions are the coordinate axes scaled by this times the
    /// box width, so a step of 1 in line-search units means the same *relative*
    /// move in every dimension. That is what keeps a registration objective
    /// mixing millimeters and radians from being searched almost entirely along
    /// its largest-numbered parameter.
    pub initial_step_frac: f64,
    /// Reset the direction set to the coordinate axes after this many
    /// replacements. `None` uses the dimension, which resets once the set has
    /// been completely turned over; `Some(0)` never resets.
    ///
    /// Without this, the direction set drifts toward linear dependence — the
    /// original axes are discarded one by one and replaced by progressively
    /// smaller, nearly parallel sweep displacements, and the search crawls
    /// along a subspace. See the table above for what it costs.
    pub restart_after: Option<usize>,
}

impl Default for PowellConfig {
    fn default() -> Self {
        Self {
            max_evals: 1000,
            ftol: 1e-8,
            xtol: 1e-6,
            initial_step_frac: 0.1,
            restart_after: None,
        }
    }
}

/// 1/φ² ≈ 0.381966 — the section-search interior split.
const SECTION: f64 = 0.381_966_011_250_105_2;
/// φ ≈ 1.618034 — the bracket expansion factor.
const EXPAND: f64 = 1.618_033_988_749_895;
/// Guards the relative tolerance against a minimum that sits at zero.
const ZEPS: f64 = 1e-12;

const MAX_BRACKET_STEPS: usize = 60;
const MAX_BRENT_ITERS: usize = 100;

/// Objective wrapper: counts evaluations, enforces the budget, tracks the best.
struct Budget<'a, F: FnMut(&[f64]) -> f64> {
    f: F,
    bbox: &'a Bbox,
    n_evals: usize,
    max_evals: usize,
    exhausted: bool,
    best_x: Vec<f64>,
    best_v: f64,
    trace: Vec<f64>,
    scratch: Vec<f64>,
}

impl<F: FnMut(&[f64]) -> f64> Budget<'_, F> {
    fn at(&mut self, x: &[f64]) -> f64 {
        if self.n_evals >= self.max_evals {
            // Report the incumbent rather than a sentinel: the line search then
            // sees a flat function and stops making progress, instead of being
            // steered by an infinity it cannot interpret.
            self.exhausted = true;
            return self.best_v;
        }
        let v = (self.f)(x);
        self.n_evals += 1;
        if v < self.best_v {
            self.best_v = v;
            self.best_x.copy_from_slice(x);
        }
        self.trace.push(self.best_v);
        v
    }

    /// Evaluate at `origin + t * dir`.
    fn along(&mut self, origin: &[f64], dir: &[f64], t: f64) -> f64 {
        for i in 0..origin.len() {
            self.scratch[i] = origin[i] + t * dir[i];
        }
        self.bbox.clip(&mut self.scratch);
        let p = std::mem::take(&mut self.scratch);
        let v = self.at(&p);
        self.scratch = p;
        v
    }
}

/// The interval of `t` for which `origin + t * dir` stays inside the box.
fn feasible_interval(bbox: &Bbox, origin: &[f64], dir: &[f64]) -> (f64, f64) {
    let mut lo = f64::NEG_INFINITY;
    let mut hi = f64::INFINITY;
    for (i, &(bl, bh)) in bbox.bounds.iter().enumerate() {
        let d = dir[i];
        if d.abs() < 1e-300 {
            continue;
        }
        let (t1, t2) = ((bl - origin[i]) / d, (bh - origin[i]) / d);
        let (t1, t2) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
        lo = lo.max(t1);
        hi = hi.min(t2);
    }
    if lo > hi { (0.0, 0.0) } else { (lo, hi) }
}

/// Expand from `t = 0` until three points bracket a minimum.
///
/// Returns `(a, b, c, fb)` with `f(b) <= f(a)` and `f(b) <= f(c)`, or `None`
/// when the interval is degenerate.
fn bracket<F: FnMut(&[f64]) -> f64>(
    e: &mut Budget<'_, F>,
    origin: &[f64],
    dir: &[f64],
    lo: f64,
    hi: f64,
) -> Option<(f64, f64, f64, f64)> {
    if hi - lo < 1e-300 {
        return None;
    }
    let mut a = 0.0f64.clamp(lo, hi);
    let mut fa = e.along(origin, dir, a);
    let mut b = (a + 1.0).min(hi);
    if (b - a).abs() < 1e-300 {
        b = (a - 1.0).max(lo);
    }
    if (b - a).abs() < 1e-300 {
        return None;
    }
    let mut fb = e.along(origin, dir, b);

    // Orient downhill: b must not be worse than a.
    if fb > fa {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = (b + EXPAND * (b - a)).clamp(lo, hi);
    let mut fc = e.along(origin, dir, c);
    for _ in 0..MAX_BRACKET_STEPS {
        if fb <= fc || e.exhausted {
            break;
        }
        // Still descending. Walk the triple forward; stop at the wall, where
        // the constrained minimum is the boundary itself.
        if (c - lo).abs() < 1e-300 || (c - hi).abs() < 1e-300 {
            break;
        }
        a = b;
        fa = fb;
        b = c;
        fb = fc;
        c = (b + EXPAND * (b - a)).clamp(lo, hi);
        fc = e.along(origin, dir, c);
    }
    let _ = fa;
    Some((a, b, c, fb))
}

/// Brent's 1-D minimization: parabolic interpolation with a section-search
/// fallback whenever the parabola is unhelpful.
fn brent<F: FnMut(&[f64]) -> f64>(
    e: &mut Budget<'_, F>,
    origin: &[f64],
    dir: &[f64],
    ax: f64,
    bx: f64,
    cx: f64,
    fbx: f64,
    tol: f64,
) -> (f64, f64) {
    let (mut a, mut b) = (ax.min(cx), ax.max(cx));
    let (mut x, mut w, mut v) = (bx, bx, bx);
    let (mut fx, mut fw, mut fv) = (fbx, fbx, fbx);
    let mut d = 0.0f64;
    let mut step_before_last = 0.0f64;

    for _ in 0..MAX_BRENT_ITERS {
        if e.exhausted {
            break;
        }
        let xm = 0.5 * (a + b);
        let tol1 = tol * x.abs() + ZEPS;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            break;
        }

        if step_before_last.abs() > tol1 {
            // Fit a parabola through (v, fv), (w, fw), (x, fx).
            let r = (x - w) * (fx - fv);
            let q0 = (x - v) * (fx - fw);
            let mut p = (x - v) * q0 - (x - w) * r;
            let mut q = 2.0 * (q0 - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let prev = step_before_last;
            step_before_last = d;
            // Reject the parabolic step if it is not shrinking, or would leave
            // the bracket: an upward parabola through three near-collinear
            // points can point anywhere.
            if p.abs() >= (0.5 * q * prev).abs() || p <= q * (a - x) || p >= q * (b - x) {
                step_before_last = if x >= xm { a - x } else { b - x };
                d = SECTION * step_before_last;
            } else {
                d = p / q;
                let u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = tol1 * (xm - x).signum();
                }
            }
        } else {
            step_before_last = if x >= xm { a - x } else { b - x };
            d = SECTION * step_before_last;
        }

        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + tol1 * d.signum()
        };
        let fu = e.along(origin, dir, u);

        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    (x, fx)
}

/// Minimize along `dir` from `origin`, moving `origin` to the minimum and
/// rescaling `dir` to the step actually taken.
///
/// The rescaling is why the sweep passes a *copy* of each stored direction
/// rather than the direction itself. Scaling the direction set in place lets a
/// direction that happens to be blocked shrink permanently: after one sweep
/// down a narrow valley every direction is a hundredth of its original length,
/// and the method grinds while still looking like it is running. Only the
/// direction built from the sweep displacement keeps its scaling, which is what
/// gives it a step size matched to the distance the sweep actually covered.
fn line_minimize<F: FnMut(&[f64]) -> f64>(
    e: &mut Budget<'_, F>,
    origin: &mut [f64],
    dir: &mut [f64],
    xtol: f64,
) -> f64 {
    let (lo, hi) = feasible_interval(e.bbox, origin, dir);
    let Some((a, b, c, fb)) = bracket(e, origin, dir, lo, hi) else {
        return e.at(origin);
    };
    let (t, ft) = brent(e, origin, dir, a, b, c, fb, xtol);
    for i in 0..origin.len() {
        origin[i] += t * dir[i];
        // Fold the step length into the direction, so a direction that keeps
        // paying off grows and one that does not shrinks.
        dir[i] *= t;
    }
    e.bbox.clip(origin);
    ft
}

/// Powell's conjugate-direction method. Minimizes `f`.
///
/// `x0` defaults to the center of the box. Deterministic: the same inputs give
/// the same result, with no seed.
pub fn powell<F>(bbox: &Bbox, cfg: &PowellConfig, x0: Option<&[f64]>, f: F) -> BboSolution
where
    F: FnMut(&[f64]) -> f64,
{
    let n = bbox.dim();
    assert!(n >= 1, "powell: empty bbox");
    if let Some(x) = x0 {
        assert_eq!(x.len(), n, "powell: x0 dim mismatch");
    }

    let mut p: Vec<f64> = match x0 {
        Some(x) => x.to_vec(),
        None => bbox
            .bounds
            .iter()
            .map(|&(lo, hi)| 0.5 * (lo + hi))
            .collect(),
    };
    bbox.clip(&mut p);

    let mut e = Budget {
        f,
        bbox,
        n_evals: 0,
        max_evals: cfg.max_evals.max(1),
        exhausted: false,
        best_x: p.clone(),
        best_v: f64::INFINITY,
        trace: Vec::new(),
        scratch: vec![0.0; n],
    };

    // Reset once the whole direction set has been replaced.
    let restart_every = match cfg.restart_after {
        None => n,
        Some(0) => usize::MAX,
        Some(k) => k,
    };
    let mut fret = e.at(&p);

    // Coordinate axes, scaled so one line-search unit is the same relative move
    // in every dimension.
    let mut dirs: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut d = vec![0.0; n];
            let w = bbox.width(i);
            d[i] = if w.is_finite() && w > 0.0 {
                w * cfg.initial_step_frac
            } else {
                cfg.initial_step_frac
            };
            d
        })
        .collect();

    let axes = dirs.clone();
    let mut since_restart = 0usize;
    let mut pt = p.clone();
    let mut extrapolated = vec![0.0; n];
    let mut new_dir = vec![0.0; n];
    let mut probe = vec![0.0; n];

    // Each sweep costs at least one line search, so the evaluation budget bounds
    // this; the cap only stops a pathological objective that never converges.
    for _ in 0..cfg.max_evals {
        if e.exhausted {
            break;
        }
        let fp = fret;
        let mut biggest_drop = 0.0;
        let mut biggest_at = 0usize;

        for (i, dir) in dirs.iter().enumerate() {
            let before = fret;
            probe.copy_from_slice(dir);
            fret = line_minimize(&mut e, &mut p, &mut probe, cfg.xtol);
            if before - fret > biggest_drop {
                biggest_drop = before - fret;
                biggest_at = i;
            }
        }

        // Converged when the whole sweep bought almost nothing.
        if 2.0 * (fp - fret).abs() <= cfg.ftol * (fp.abs() + fret.abs()) + ZEPS {
            break;
        }
        if e.exhausted {
            break;
        }

        // The net displacement of the sweep — the direction along the valley.
        for i in 0..n {
            new_dir[i] = p[i] - pt[i];
            extrapolated[i] = p[i] + new_dir[i];
            pt[i] = p[i];
        }
        e.bbox.clip(&mut extrapolated);
        let fe = e.at(&extrapolated);

        if fe >= fp {
            // The far point is no better than where the sweep started: the
            // valley does not continue, so keep the direction set.
            continue;
        }
        // Brent's discard test. Only replace when the curvature along the new
        // direction beats what the displaced direction was contributing;
        // replacing unconditionally drives the direction set toward linear
        // dependence and the search collapses onto a subspace.
        let t = 2.0 * (fp - 2.0 * fret + fe) * (fp - fret - biggest_drop).powi(2)
            - biggest_drop * (fp - fe).powi(2);
        if t >= 0.0 {
            continue;
        }

        fret = line_minimize(&mut e, &mut p, &mut new_dir, cfg.xtol);
        since_restart += 1;
        if since_restart >= restart_every {
            since_restart = 0;
            dirs.clone_from(&axes);
            continue;
        }
        // Move the last direction into the discarded slot and append the new
        // one, keeping the set ordered by age.
        let last = n - 1;
        dirs[biggest_at] = dirs[last].clone();
        dirs[last] = new_dir.clone();
    }

    let n_evals = e.n_evals;
    let mut trace = e.trace;
    if trace.is_empty() {
        trace.push(e.best_v);
    }
    BboSolution {
        x: e.best_x,
        value: e.best_v,
        trace,
        n_evals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    fn beale(x: &[f64]) -> f64 {
        let t1 = 1.5 - x[0] + x[0] * x[1];
        let t2 = 2.25 - x[0] + x[0] * x[1] * x[1];
        let t3 = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
        t1 * t1 + t2 * t2 + t3 * t3
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        let (a, b) = (1.0 - x[0], x[1] - x[0] * x[0]);
        a * a + 100.0 * b * b
    }

    /// A quadratic whose axes sit 45 degrees off the coordinate frame, with the
    /// given ratio between them.
    fn rotated_valley(ratio: f64) -> impl Fn(&[f64]) -> f64 {
        move |x: &[f64]| {
            let u = (x[0] + x[1]) / std::f64::consts::SQRT_2;
            let v = (x[0] - x[1]) / std::f64::consts::SQRT_2;
            u * u + ratio * v * v
        }
    }

    /// Cyclic coordinate descent using the same line search, for comparison.
    fn coordinate_descent<F: FnMut(&[f64]) -> f64>(
        bbox: &Bbox,
        start: &[f64],
        budget: usize,
        f: F,
    ) -> f64 {
        let n = start.len();
        let mut p = start.to_vec();
        let mut e = Budget {
            f,
            bbox,
            n_evals: 0,
            max_evals: budget,
            exhausted: false,
            best_x: p.clone(),
            best_v: f64::INFINITY,
            trace: Vec::new(),
            scratch: vec![0.0; n],
        };
        e.at(&p);
        while !e.exhausted {
            for i in 0..n {
                let mut d = vec![0.0; n];
                d[i] = 1.0;
                line_minimize(&mut e, &mut p, &mut d, 1e-6);
            }
        }
        e.best_v
    }

    #[test]
    fn finds_the_minimum_of_a_sphere() {
        let b = Bbox::new(vec![(-3.0, 3.0); 4]);
        let sol = powell(
            &b,
            &PowellConfig::default(),
            Some(&[2.0, -2.0, 1.5, -1.0]),
            sphere,
        );
        assert!(sol.value < 1e-12, "got {} at {:?}", sol.value, sol.x);
        for xi in &sol.x {
            assert!(xi.abs() < 1e-6, "not at the origin: {:?}", sol.x);
        }
    }

    #[test]
    fn solves_rosenbrock_from_the_standard_start() {
        let b = Bbox::new(vec![(-5.0, 5.0); 2]);
        let cfg = PowellConfig {
            max_evals: 4000,
            ..Default::default()
        };
        let sol = powell(&b, &cfg, Some(&[-1.2, 1.0]), rosenbrock);
        assert!(sol.value < 1e-8, "got {} at {:?}", sol.value, sol.x);
        assert!((sol.x[0] - 1.0).abs() < 1e-4, "{:?}", sol.x);
        assert!((sol.x[1] - 1.0).abs() < 1e-4, "{:?}", sol.x);
    }

    #[test]
    fn the_direction_update_beats_coordinate_descent_on_a_coupled_quadratic() {
        // The reason to prefer Powell over sweeping the axes. On a valley that
        // runs diagonally, the direction built from the sweep displacement
        // points along it, and the search terminates. Coordinate descent, given
        // the same line search and five times the budget, is still crawling.
        let b = Bbox::new(vec![(-5.0, 5.0); 2]);
        let start = [3.0, -3.0];

        for ratio in [2.0, 3.0, 5.0] {
            let sol = powell(
                &b,
                &PowellConfig {
                    max_evals: 300,
                    ..Default::default()
                },
                Some(&start),
                rotated_valley(ratio),
            );
            let cd = coordinate_descent(&b, &start, 300, rotated_valley(ratio));

            assert!(
                sol.value < 1e-20,
                "ratio {ratio}: expected termination at machine precision, got {}",
                sol.value
            );
            assert!(
                sol.n_evals < 120,
                "ratio {ratio}: took {} evals to converge",
                sol.n_evals
            );
            assert!(
                sol.value < cd / 1e6,
                "ratio {ratio}: powell {} should beat coordinate descent {} by orders",
                sol.value,
                cd
            );
        }
    }

    #[test]
    fn the_restart_is_what_makes_strong_anisotropy_tractable() {
        // Without resetting the direction set the sweep displacements crowd
        // into a subspace and the search stalls: SciPy's Powell returns 1.5e1
        // on this problem, and so does this one with the restart disabled. The
        // reset costs nothing and converges it.
        let b = Bbox::new(vec![(-5.0, 5.0); 2]);
        let start = [3.0, -3.0];

        let with = powell(
            &b,
            &PowellConfig {
                max_evals: 300,
                ..Default::default()
            },
            Some(&start),
            rotated_valley(1000.0),
        );
        let without = powell(
            &b,
            &PowellConfig {
                max_evals: 300,
                restart_after: Some(0),
                ..Default::default()
            },
            Some(&start),
            rotated_valley(1000.0),
        );

        assert!(with.value < 1e-20, "with restart: {}", with.value);
        assert!(
            with.n_evals < 300,
            "should converge early: {}",
            with.n_evals
        );
        assert!(
            without.value > 1.0,
            "restart disabled should stall, matching SciPy's 1.5e1: {}",
            without.value
        );
    }

    #[test]
    fn the_restart_rescues_a_non_quadratic_too() {
        // Beale. SciPy reaches 3.5e-19 in 438 evals; without the restart this
        // is still at 3.1e-4 after 2000.
        let b = Bbox::new(vec![(-4.5, 4.5); 2]);
        let with = powell(
            &b,
            &PowellConfig {
                max_evals: 2000,
                ..Default::default()
            },
            Some(&[1.0, 1.0]),
            beale,
        );
        let without = powell(
            &b,
            &PowellConfig {
                max_evals: 2000,
                restart_after: Some(0),
                ..Default::default()
            },
            Some(&[1.0, 1.0]),
            beale,
        );
        assert!(with.value < 1e-18, "with restart: {}", with.value);
        assert!(with.n_evals < 1000, "evals: {}", with.n_evals);
        assert!(
            without.value > with.value * 1e6,
            "restart should matter here: {} vs {}",
            without.value,
            with.value
        );
    }

    #[test]
    fn matches_scipy_powell_on_standard_test_functions() {
        // Reference values from scipy.optimize.minimize(method='Powell') with
        // the same starts, bounds and tolerances. Powell is a heuristic, so the
        // check is that we reach the same quality of minimum, not the same
        // arithmetic.
        let b2 = Bbox::new(vec![(-5.0, 5.0); 2]);

        // scipy: 4.93e-32 in 50 evals
        let sphere4 = powell(
            &Bbox::new(vec![(-3.0, 3.0); 4]),
            &PowellConfig::default(),
            Some(&[2.0, -2.0, 1.5, -1.0]),
            sphere,
        );
        assert!(sphere4.value < 1e-20, "sphere4: {}", sphere4.value);

        // scipy: 3.94e-18 in 44 evals
        let rb = powell(
            &b2,
            &PowellConfig {
                max_evals: 4000,
                ..Default::default()
            },
            Some(&[-1.2, 1.0]),
            rosenbrock,
        );
        assert!(rb.value < 1e-10, "rosenbrock: {}", rb.value);

        // Beale. scipy: 3.54e-19 in 438 evals, minimum at (3, 0.5).
        let beale = powell(
            &Bbox::new(vec![(-4.5, 4.5); 2]),
            &PowellConfig {
                max_evals: 2000,
                ..Default::default()
            },
            Some(&[1.0, 1.0]),
            beale,
        );
        assert!(
            beale.value < 1e-10,
            "beale: {} at {:?}",
            beale.value,
            beale.x
        );
        assert!((beale.x[0] - 3.0).abs() < 1e-3, "beale x: {:?}", beale.x);
        assert!((beale.x[1] - 0.5).abs() < 1e-3, "beale x: {:?}", beale.x);

        // Booth. scipy: exactly 0 in 45 evals, minimum at (1, 3).
        let booth = powell(
            &Bbox::new(vec![(-10.0, 10.0); 2]),
            &PowellConfig::default(),
            Some(&[0.0, 0.0]),
            |x: &[f64]| {
                let t1 = x[0] + 2.0 * x[1] - 7.0;
                let t2 = 2.0 * x[0] + x[1] - 5.0;
                t1 * t1 + t2 * t2
            },
        );
        assert!(booth.value < 1e-20, "booth: {}", booth.value);
        assert!((booth.x[0] - 1.0).abs() < 1e-6, "booth x: {:?}", booth.x);
        assert!((booth.x[1] - 3.0).abs() < 1e-6, "booth x: {:?}", booth.x);
    }

    #[test]
    fn respects_the_evaluation_budget() {
        for budget in [1usize, 7, 50, 213] {
            let b = Bbox::new(vec![(-5.0, 5.0); 3]);
            let cfg = PowellConfig {
                max_evals: budget,
                ..Default::default()
            };
            let sol = powell(&b, &cfg, Some(&[4.0, 4.0, 4.0]), rosenbrock2d3);
            assert!(
                sol.n_evals <= budget,
                "budget {budget} exceeded: {} evals",
                sol.n_evals
            );
        }
    }

    fn rosenbrock2d3(x: &[f64]) -> f64 {
        rosenbrock(&x[..2]) + x[2] * x[2]
    }

    #[test]
    fn stays_inside_the_box() {
        // The unconstrained minimum is at the origin, outside the box.
        let b = Bbox::new(vec![(1.0, 4.0), (2.0, 5.0)]);
        let sol = powell(&b, &PowellConfig::default(), None, sphere);
        assert!(
            sol.x[0] >= 1.0 - 1e-9 && sol.x[0] <= 4.0 + 1e-9,
            "{:?}",
            sol.x
        );
        assert!(
            sol.x[1] >= 2.0 - 1e-9 && sol.x[1] <= 5.0 + 1e-9,
            "{:?}",
            sol.x
        );
        // The constrained minimum is the corner nearest the origin.
        assert!((sol.x[0] - 1.0).abs() < 1e-4, "{:?}", sol.x);
        assert!((sol.x[1] - 2.0).abs() < 1e-4, "{:?}", sol.x);
    }

    #[test]
    fn is_deterministic() {
        let b = Bbox::new(vec![(-4.0, 4.0); 3]);
        let cfg = PowellConfig::default();
        let a = powell(&b, &cfg, Some(&[1.0, 2.0, 3.0]), rosenbrock2d3);
        let c = powell(&b, &cfg, Some(&[1.0, 2.0, 3.0]), rosenbrock2d3);
        assert_eq!(a.x, c.x);
        assert_eq!(a.value, c.value);
        assert_eq!(a.n_evals, c.n_evals);
    }

    #[test]
    fn trace_is_monotone() {
        let b = Bbox::new(vec![(-2.0, 2.0); 2]);
        let sol = powell(&b, &PowellConfig::default(), Some(&[1.5, 1.5]), rosenbrock);
        for w in sol.trace.windows(2) {
            assert!(w[1] <= w[0] + 1e-12, "trace rose: {} -> {}", w[0], w[1]);
        }
        assert_eq!(sol.trace.len(), sol.n_evals);
        assert_eq!(*sol.trace.last().unwrap(), sol.value);
    }

    #[test]
    fn handles_parameters_on_wildly_different_scales() {
        // A registration-shaped objective: two translations in millimeters and
        // a rotation in radians. Directions scaled by box width keep the
        // rotation from being searched at millimeter step sizes.
        let b = Bbox::new(vec![(-50.0, 50.0), (-50.0, 50.0), (-0.05, 0.05)]);
        let target = [12.0, -30.0, 0.02];
        let sol = powell(
            &b,
            &PowellConfig {
                max_evals: 2000,
                ..Default::default()
            },
            None,
            |x: &[f64]| {
                x.iter()
                    .zip(target)
                    .map(|(a, t)| (a - t) * (a - t))
                    .sum::<f64>()
            },
        );
        assert!((sol.x[2] - target[2]).abs() < 1e-6, "rotation: {:?}", sol.x);
        assert!((sol.x[0] - target[0]).abs() < 1e-4, "{:?}", sol.x);
        assert!((sol.x[1] - target[1]).abs() < 1e-4, "{:?}", sol.x);
    }

    #[test]
    fn a_flat_objective_terminates() {
        let b = Bbox::new(vec![(-1.0, 1.0); 3]);
        let sol = powell(&b, &PowellConfig::default(), None, |_x: &[f64]| 1.0);
        assert_eq!(sol.value, 1.0);
        assert!(sol.n_evals <= PowellConfig::default().max_evals);
    }

    #[test]
    fn works_in_one_dimension() {
        let b = Bbox::new(vec![(-10.0, 10.0)]);
        let sol = powell(&b, &PowellConfig::default(), Some(&[7.0]), |x: &[f64]| {
            (x[0] - 3.0).powi(2) + 5.0
        });
        assert!((sol.value - 5.0).abs() < 1e-10, "{}", sol.value);
        assert!((sol.x[0] - 3.0).abs() < 1e-6, "{:?}", sol.x);
    }

    #[test]
    fn a_degenerate_box_returns_its_only_point() {
        let b = Bbox::new(vec![(2.0, 2.0), (3.0, 3.0)]);
        let sol = powell(&b, &PowellConfig::default(), None, sphere);
        assert_eq!(sol.x, vec![2.0, 3.0]);
        assert!((sol.value - 13.0).abs() < 1e-12);
    }
}
