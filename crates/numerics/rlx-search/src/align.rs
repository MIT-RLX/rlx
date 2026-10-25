// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Monotonic Viterbi forced alignment without a blank state.
//!
//! Every frame belongs to exactly one target token, tokens keep their order,
//! and each token takes one or more consecutive frames. The Viterbi DP over
//! the [`crate::Tropical`] semiring with back-pointers.

use std::fmt;

/// The frame span of one aligned target token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub token: u32,
    /// First frame (inclusive).
    pub start: usize,
    /// One past the last frame.
    pub end: usize,
}

/// Why an alignment could not be computed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlignError {
    NoTargets,
    /// Fewer frames than tokens: some token would get no frame.
    TooFewFrames {
        frames: usize,
        tokens: usize,
    },
    /// `scores.len()` is not a multiple of `classes`.
    Ragged {
        len: usize,
        classes: usize,
    },
    TokenOutOfRange {
        token: u32,
        classes: usize,
    },
    /// Every path scores `-inf`.
    NoPath,
}

impl fmt::Display for AlignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTargets => write!(f, "need at least one target token"),
            Self::TooFewFrames { frames, tokens } => write!(
                f,
                "need at least as many frames ({frames}) as tokens ({tokens})"
            ),
            Self::Ragged { len, classes } => {
                write!(
                    f,
                    "scores length {len} is not a multiple of {classes} classes"
                )
            }
            Self::TokenOutOfRange { token, classes } => {
                write!(f, "target token {token} out of range for {classes} classes")
            }
            Self::NoPath => write!(f, "no valid alignment path"),
        }
    }
}

impl std::error::Error for AlignError {}

/// Align `targets` onto `[frames, classes]` log-probabilities `scores`,
/// maximizing the summed log-probability of each frame's assigned token.
/// On a tie between staying on a token and advancing to the next, the path
/// stays.
pub fn monotonic_align(
    scores: &[f32],
    classes: usize,
    targets: &[u32],
) -> Result<Vec<Span>, AlignError> {
    let n = targets.len();
    if n == 0 {
        return Err(AlignError::NoTargets);
    }
    if classes == 0 || !scores.len().is_multiple_of(classes) {
        return Err(AlignError::Ragged {
            len: scores.len(),
            classes,
        });
    }
    let frames = scores.len() / classes;
    if frames < n {
        return Err(AlignError::TooFewFrames { frames, tokens: n });
    }
    if let Some(&token) = targets.iter().find(|&&t| t as usize >= classes) {
        return Err(AlignError::TokenOutOfRange { token, classes });
    }
    let emit = |t: usize, l: usize| scores[t * classes + targets[l] as usize];

    // dp[l] after frame t: best score of frames 0..=t ending on token l.
    // advanced[t * n + l]: frame t entered token l from l - 1.
    let mut dp = vec![f32::NEG_INFINITY; n];
    let mut prev = vec![f32::NEG_INFINITY; n];
    let mut advanced = vec![false; frames * n];
    dp[0] = emit(0, 0);
    for t in 1..frames {
        std::mem::swap(&mut dp, &mut prev);
        dp.fill(f32::NEG_INFINITY);
        // Token l needs l earlier frames.
        for l in 0..n.min(t + 1) {
            let stay = prev[l];
            let advance = if l > 0 {
                prev[l - 1]
            } else {
                f32::NEG_INFINITY
            };
            let (best, adv) = if advance > stay {
                (advance, true)
            } else {
                (stay, false)
            };
            if best == f32::NEG_INFINITY {
                continue;
            }
            dp[l] = best + emit(t, l);
            advanced[t * n + l] = adv;
        }
    }
    if dp[n - 1] == f32::NEG_INFINITY {
        return Err(AlignError::NoPath);
    }

    let mut spans: Vec<Span> = targets
        .iter()
        .map(|&token| Span {
            token,
            start: usize::MAX,
            end: 0,
        })
        .collect();
    let mut l = n - 1;
    for t in (0..frames).rev() {
        let s = &mut spans[l];
        s.start = s.start.min(t);
        s.end = s.end.max(t + 1);
        if advanced[t * n + l] {
            l -= 1;
        }
    }
    Ok(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(rows: &[Vec<f32>]) -> (Vec<f32>, usize) {
        (rows.concat(), rows[0].len())
    }

    #[test]
    fn spans_follow_the_peaks() {
        // 6 frames, tokens 0 then 1: frames 0-2 favour 0, 3-5 favour 1.
        let rows: Vec<Vec<f32>> = (0..6)
            .map(|t| {
                if t < 3 {
                    vec![-0.1, -3.0]
                } else {
                    vec![-3.0, -0.1]
                }
            })
            .collect();
        let (s, c) = flat(&rows);
        let al = monotonic_align(&s, c, &[0, 1]).unwrap();
        assert_eq!(
            al,
            vec![
                Span {
                    token: 0,
                    start: 0,
                    end: 3
                },
                Span {
                    token: 1,
                    start: 3,
                    end: 6
                }
            ]
        );
    }

    #[test]
    fn every_frame_is_covered_in_order() {
        let rows: Vec<Vec<f32>> = (0..7)
            .map(|t| vec![-(t as f32) * 0.1, -0.5, -1.0])
            .collect();
        let (s, c) = flat(&rows);
        let al = monotonic_align(&s, c, &[0, 1, 2]).unwrap();
        assert_eq!(al[0].start, 0);
        assert_eq!(al[2].end, 7);
        for w in al.windows(2) {
            assert_eq!(w[0].end, w[1].start);
            assert!(w[0].start < w[0].end);
        }
    }

    #[test]
    fn bad_inputs_are_errors() {
        let (s, c) = flat(&[vec![0.0, 0.0]]);
        assert_eq!(
            monotonic_align(&s, c, &[0, 1]),
            Err(AlignError::TooFewFrames {
                frames: 1,
                tokens: 2
            })
        );
        assert_eq!(monotonic_align(&s, c, &[]), Err(AlignError::NoTargets));
        assert_eq!(
            monotonic_align(&s, c, &[9]),
            Err(AlignError::TokenOutOfRange {
                token: 9,
                classes: 2
            })
        );
        assert!(matches!(
            monotonic_align(&[0.0; 3], 2, &[0]),
            Err(AlignError::Ragged { .. })
        ));
    }
}
