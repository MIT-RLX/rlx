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
// RLX — [r, t] off-diagonal training curriculum (Python `flow_utils.sample_r_t`).

/// Sample off-diagonal time pairs `(r, t)` with annealing.
///
/// Phase 1 (`step < warmup`): `r = t` (diagonal only).
/// Phase 2 (`warmup..anneal`): interval width grows with training progress.
/// Phase 3 (`step >= anneal`): full random `0 <= r <= t <= 1`.
pub fn sample_r_t(
    batch_size: usize,
    step: usize,
    warmup_steps: usize,
    anneal_end_step: usize,
    rng: &mut u64,
) -> (Vec<f32>, Vec<f32>) {
    let mut r = Vec::with_capacity(batch_size);
    let mut t = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let t1 = uniform01(rng);
        let t2 = uniform01(rng);
        let t_min = t1.min(t2);
        let t_max = t1.max(t2);
        let mid = 0.5 * (t_min + t_max);
        let dist = t_max - t_min;

        let anneal_duration = (anneal_end_step.saturating_sub(warmup_steps)).max(1);
        let progress = ((step.saturating_sub(warmup_steps)) as f32 / anneal_duration as f32).clamp(0.0, 1.0);
        let max_step_size = if step < warmup_steps { 0.0 } else { progress };

        let ri = mid - max_step_size * dist * 0.5;
        let ti = mid + max_step_size * dist * 0.5;
        r.push(ri);
        t.push(ti);
    }
    (r, t)
}

fn uniform01(seed: &mut u64) -> f32 {
    *seed = crate::buffer::rand_like(*seed);
    (*seed >> 11) as f32 / (1u32 << 21) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_is_diagonal() {
        let (r, t) = sample_r_t(8, 0, 100, 1000, &mut 1);
        for i in 0..8 {
            assert!((r[i] - t[i]).abs() < 1e-5, "r={} t={}", r[i], t[i]);
        }
    }

    #[test]
    fn ordered_r_le_t() {
        let (r, t) = sample_r_t(32, 50_000, 100, 1000, &mut 99);
        for i in 0..32 {
            assert!(r[i] <= t[i] + 1e-5);
        }
    }
}
