// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `scipy.optimize.linear_sum_assignment` — the rectangular linear sum
//! assignment problem, as SciPy solves it.
//!
//! A line-for-line port of SciPy's `rectangular_lsap.cpp` (Crouse, "On
//! implementing 2D rectangular assignment algorithms", 2016): a
//! shortest-augmenting-path Jonker–Volgenant variant with dual potentials.
//! Any exact solver returns the same optimum when it is unique; following
//! SciPy's code also reproduces its choice among TIED optima — the reversed
//! `remaining` order (a constant cost matrix assigns the identity), the
//! "prefer a free column on a cost tie" rule, and the transpose of a tall
//! matrix with the result re-sorted by row.

/// Why [`linear_sum_assignment`] refused a cost matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LsapError {
    /// A NaN or `-inf` entry (SciPy: "matrix contains invalid numeric entries").
    Invalid,
    /// No assignment of finite total cost exists ("cost matrix is infeasible").
    Infeasible,
}

impl std::fmt::Display for LsapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LsapError::Invalid => write!(f, "matrix contains invalid numeric entries"),
            LsapError::Infeasible => write!(f, "cost matrix is infeasible"),
        }
    }
}

impl std::error::Error for LsapError {}

/// Solve the assignment problem on a row-major `nr × nc` cost matrix.
///
/// Returns `(row_ind, col_ind)` exactly as SciPy does: `min(nr, nc)` pairs,
/// `row_ind` ascending, minimising (or with `maximize`, maximising) the sum of
/// `cost[row_ind[k], col_ind[k]]`.
pub fn linear_sum_assignment(
    cost: &[f64],
    nr: usize,
    nc: usize,
    maximize: bool,
) -> Result<(Vec<usize>, Vec<usize>), LsapError> {
    assert_eq!(cost.len(), nr * nc, "cost must be nr × nc");
    if nr == 0 || nc == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    // A tall matrix is solved transposed.
    let transpose = nc < nr;
    let (nr_, nc_) = if transpose { (nc, nr) } else { (nr, nc) };
    let mut c: Vec<f64> = if transpose {
        let mut t = vec![0f64; nr * nc];
        for i in 0..nr {
            for j in 0..nc {
                t[j * nr + i] = cost[i * nc + j];
            }
        }
        t
    } else {
        cost.to_vec()
    };
    if maximize {
        c.iter_mut().for_each(|v| *v = -*v);
    }
    if c.iter().any(|&v| v.is_nan() || v == f64::NEG_INFINITY) {
        return Err(LsapError::Invalid);
    }
    let (nr, nc) = (nr_, nc_);

    let mut u = vec![0f64; nr];
    let mut v = vec![0f64; nc];
    let mut shortest = vec![0f64; nc];
    let mut path = vec![usize::MAX; nc];
    let mut col4row = vec![usize::MAX; nr];
    let mut row4col = vec![usize::MAX; nc];
    let mut sr = vec![false; nr];
    let mut sc = vec![false; nc];
    let mut remaining = vec![0usize; nc];

    for cur_row in 0..nr {
        // ── augmenting_path ──
        let mut min_val = 0f64;
        let mut num_remaining = nc;
        for (it, r) in remaining.iter_mut().enumerate() {
            // Reverse order: a constant cost matrix then assigns the identity.
            *r = nc - it - 1;
        }
        sr.iter_mut().for_each(|x| *x = false);
        sc.iter_mut().for_each(|x| *x = false);
        shortest.iter_mut().for_each(|x| *x = f64::INFINITY);
        let mut i = cur_row;
        let sink = loop {
            let mut index = usize::MAX;
            let mut lowest = f64::INFINITY;
            sr[i] = true;
            for (it, &j) in remaining[..num_remaining].iter().enumerate() {
                let r = min_val + c[i * nc + j] - u[i] - v[j];
                if r < shortest[j] {
                    path[j] = i;
                    shortest[j] = r;
                }
                // On a tie prefer a column that yields a new sink.
                if shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == usize::MAX) {
                    lowest = shortest[j];
                    index = it;
                }
            }
            min_val = lowest;
            if min_val == f64::INFINITY {
                return Err(LsapError::Infeasible);
            }
            let j = remaining[index];
            let found = row4col[j] == usize::MAX;
            if !found {
                i = row4col[j];
            }
            sc[j] = true;
            num_remaining -= 1;
            remaining[index] = remaining[num_remaining];
            if found {
                break j;
            }
        };

        // ── update the dual variables ──
        u[cur_row] += min_val;
        for i in 0..nr {
            if sr[i] && i != cur_row {
                u[i] += min_val - shortest[col4row[i]];
            }
        }
        for j in 0..nc {
            if sc[j] {
                v[j] -= min_val - shortest[j];
            }
        }

        // ── augment the previous solution ──
        let mut j = sink;
        loop {
            let i = path[j];
            row4col[j] = i;
            std::mem::swap(&mut col4row[i], &mut j);
            if i == cur_row {
                break;
            }
        }
    }

    if transpose {
        // Rows of the transposed problem are the original columns; re-sort
        // the pairs by original row.
        let mut order: Vec<usize> = (0..nr).collect();
        order.sort_by_key(|&k| col4row[k]);
        let a = order.iter().map(|&k| col4row[k]).collect();
        Ok((a, order))
    } else {
        Ok(((0..nr).collect(), col4row))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values from `scipy.optimize.linear_sum_assignment` (SciPy 1.17.1).
    #[test]
    fn matches_scipy_examples() {
        // The docstring example.
        let c = [4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0];
        assert_eq!(
            linear_sum_assignment(&c, 3, 3, false).unwrap(),
            (vec![0, 1, 2], vec![1, 0, 2])
        );
        // A constant matrix assigns the identity (scipy #11602).
        let c = [1.0; 16];
        assert_eq!(
            linear_sum_assignment(&c, 4, 4, false).unwrap().1,
            vec![0, 1, 2, 3]
        );
        // Wide, tall, and maximise.
        let c = [10.0, 10.0, 8.0, 9.0, 8.0, 1.0];
        assert_eq!(
            linear_sum_assignment(&c, 2, 3, false).unwrap(),
            (vec![0, 1], vec![1, 2])
        );
        let ct = [10.0, 9.0, 10.0, 8.0, 8.0, 1.0];
        assert_eq!(
            linear_sum_assignment(&ct, 3, 2, false).unwrap(),
            (vec![1, 2], vec![0, 1])
        );
        assert_eq!(
            linear_sum_assignment(&c, 2, 3, true).unwrap(),
            (vec![0, 1], vec![1, 0])
        );
        assert_eq!(
            linear_sum_assignment(&[f64::NAN, 1.0, 1.0, 1.0], 2, 2, false),
            Err(LsapError::Invalid)
        );
        let inf = f64::INFINITY;
        assert_eq!(
            linear_sum_assignment(&[inf, inf, 1.0, inf], 2, 2, false),
            Err(LsapError::Infeasible)
        );
    }
}
