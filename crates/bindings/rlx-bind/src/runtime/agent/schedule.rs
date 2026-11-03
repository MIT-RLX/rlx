// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! When the agent is asked: the updates of the scheduled decisions.

/// How the scheduled decisions are spread over a run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Spread {
    /// A fifth of them in the first 5% of the run, where the loss moves
    /// fastest and a bad setting costs the most; the rest evenly after.
    #[default]
    Front,
    /// Evenly over the run.
    Uniform,
}

impl Spread {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "front" => Some(Self::Front),
            "uniform" => Some(Self::Uniform),
            _ => None,
        }
    }
}

/// The updates (counted from 1) at which `n` decisions are asked over a run of
/// `total` updates: increasing, distinct, never the last update (an edit there
/// would never run). Fewer than `n` when the run is too short to hold them.
pub fn decision_updates(n: usize, total: u64, spread: Spread) -> Vec<u64> {
    if n == 0 || total < 2 {
        return Vec::new();
    }
    let t = total as f64;
    // `count` points strictly inside (from, to], evenly spaced, the last at
    // `to` only when `closed`.
    let even = |count: usize, from: f64, to: f64, closed: bool| -> Vec<f64> {
        let parts = if closed { count } else { count + 1 } as f64;
        (1..=count)
            .map(|i| from + (to - from) * i as f64 / parts)
            .collect()
    };
    let points = match spread {
        Spread::Uniform => even(n, 0.0, t, false),
        Spread::Front => {
            let early = if n >= 2 {
                ((n as f64 * 0.2).round() as usize).max(1)
            } else {
                n
            };
            let edge = 0.05 * t;
            let mut p = even(early, 0.0, edge, true);
            p.extend(even(n - early, edge, t, false));
            p
        }
    };
    let mut out: Vec<u64> = Vec::with_capacity(n);
    for p in points {
        let mut u = (p.ceil() as u64).max(1);
        if let Some(&last) = out.last() {
            u = u.max(last + 1);
        }
        if u >= total {
            break;
        }
        out.push(u);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_puts_a_fifth_in_the_first_five_percent() {
        let u = decision_updates(10, 1000, Spread::Front);
        assert_eq!(u.len(), 10);
        assert_eq!(&u[..2], &[25, 50]);
        assert!(u[2..].iter().all(|&x| x > 50 && x < 1000), "{u:?}");
        assert!(u.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn uniform_is_even_and_short_runs_hold_fewer() {
        assert_eq!(decision_updates(3, 100, Spread::Uniform), vec![25, 50, 75]);
        assert_eq!(decision_updates(5, 3, Spread::Uniform), vec![1, 2]);
        assert!(decision_updates(4, 1, Spread::Front).is_empty());
        assert!(decision_updates(0, 100, Spread::Front).is_empty());
    }
}
