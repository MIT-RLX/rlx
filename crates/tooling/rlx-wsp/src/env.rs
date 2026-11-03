// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Environments a weight-space program is scored on.
//!
//! Every environment exposes a *labelled* observation vector: the agent reads
//! an input-layer weight row by its label (`ball>(x=3,y=5)`) instead of a bare
//! index, which is what makes reading and writing input weights tractable.

use rlx_ir::Philox4x32;
use serde::{Deserialize, Serialize};

/// Static description of an environment — everything the agent is told.
#[derive(Debug, Clone, Serialize)]
pub struct EnvSpec {
    pub name: &'static str,
    pub description: String,
    pub obs_dim: usize,
    pub obs_labels: Vec<String>,
    pub actions: Vec<&'static str>,
    /// Fixed episode length in steps.
    pub horizon: usize,
}

/// A single-agent, discrete-action, fixed-horizon environment.
pub trait Env {
    fn spec(&self) -> EnvSpec;
    fn reset(&mut self, rng: &mut Philox4x32);
    /// Write the current observation into `out` (`len == obs_dim`).
    fn observe(&self, out: &mut [f32]);
    /// Apply `action`, return the reward for this step.
    fn step(&mut self, action: usize, rng: &mut Philox4x32) -> f32;
    /// Short human-readable state, for rollout traces.
    fn render_state(&self) -> String;
}

/// Environments known to the harness (the JSON state stores this tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvKind {
    Bounce,
    Orchard,
    Gusts,
    GustsHard,
    MnistFix,
    MnistFixHard,
    ReadRules,
    ReadDigit,
    AuditPair,
    MnistSteer,
    FixPolicy,
    /// Any rlx graph loaded with [`crate::WeightSpace::from_graph`].
    Graph,
}

impl EnvKind {
    pub fn make(self) -> Box<dyn Env> {
        match self {
            EnvKind::Bounce => Box::new(Bounce::new(BOUNCE_W, BOUNCE_H)),
            EnvKind::Orchard => Box::new(crate::orchard::Orchard::new()),
            EnvKind::Gusts => Box::new(crate::orchard::Orchard::gusts()),
            EnvKind::GustsHard => Box::new(crate::orchard::Orchard::gusts_hard()),
            EnvKind::MnistFix
            | EnvKind::MnistFixHard
            | EnvKind::ReadDigit
            | EnvKind::AuditPair
            | EnvKind::MnistSteer => Box::new(MnistTask),
            EnvKind::ReadRules => Box::new(crate::orchard::Orchard::new()),
            EnvKind::FixPolicy => Box::new(Bounce::new(BOUNCE_W, BOUNCE_H)),
            EnvKind::Graph => Box::new(GraphTask),
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "bounce" => Some(EnvKind::Bounce),
            "orchard" => Some(EnvKind::Orchard),
            "gusts" => Some(EnvKind::Gusts),
            "gusts-hard" | "gusts_hard" => Some(EnvKind::GustsHard),
            "mnist-fix" | "mnist_fix" => Some(EnvKind::MnistFix),
            "mnist-fix-hard" | "mnist_fix_hard" => Some(EnvKind::MnistFixHard),
            "read-rules" | "read_rules" => Some(EnvKind::ReadRules),
            "read-digit" | "read_digit" => Some(EnvKind::ReadDigit),
            "audit-pair" | "audit_pair" => Some(EnvKind::AuditPair),
            "mnist-steer" | "mnist_steer" => Some(EnvKind::MnistSteer),
            "fix-policy" | "fix_policy" => Some(EnvKind::FixPolicy),
            _ => None,
        }
    }
}

impl EnvKind {
    /// Classification tasks have no episodes: `rollout` is refused and the
    /// score comes from test-set accuracy instead.
    pub fn is_classifier(self) -> bool {
        matches!(
            self,
            EnvKind::MnistFix
                | EnvKind::MnistFixHard
                | EnvKind::ReadDigit
                | EnvKind::AuditPair
                | EnvKind::MnistSteer
        )
    }

    /// Tasks scored by a submitted answer (no rollouts).
    pub fn is_reading(self) -> bool {
        matches!(
            self,
            EnvKind::ReadRules | EnvKind::ReadDigit | EnvKind::AuditPair
        )
    }

    /// The planted trigger of a repair task.
    pub fn trigger(self) -> Option<crate::mnist::Trigger> {
        match self {
            EnvKind::MnistFix => Some(crate::mnist::Trigger::Corner),
            EnvKind::MnistFixHard => Some(crate::mnist::Trigger::CenterChecker),
            _ => None,
        }
    }
}

/// A loaded graph as an [`Env`]: no observation layout, actions or score.
#[derive(Debug, Clone, Copy)]
pub struct GraphTask;

impl Env for GraphTask {
    fn spec(&self) -> EnvSpec {
        EnvSpec {
            name: "graph",
            description: String::new(),
            obs_dim: 0,
            obs_labels: Vec::new(),
            actions: Vec::new(),
            horizon: 0,
        }
    }
    fn reset(&mut self, _: &mut Philox4x32) {}
    fn observe(&self, _: &mut [f32]) {}
    fn step(&mut self, _: usize, _: &mut Philox4x32) -> f32 {
        0.0
    }
    fn render_state(&self) -> String {
        String::new()
    }
}

/// The MNIST repair task as an [`Env`]: only its spec (pixel labels, digit
/// outputs, description) is used; it has no dynamics.
#[derive(Debug, Clone, Copy)]
pub struct MnistTask;

impl Env for MnistTask {
    fn spec(&self) -> EnvSpec {
        EnvSpec {
            name: "mnist_fix",
            description: crate::mnist::DESCRIPTION.to_string(),
            obs_dim: crate::mnist::PIXELS,
            obs_labels: crate::mnist::pixel_labels(),
            actions: vec![
                "digit0", "digit1", "digit2", "digit3", "digit4", "digit5", "digit6", "digit7",
                "digit8", "digit9",
            ],
            horizon: 0,
        }
    }
    fn reset(&mut self, _: &mut Philox4x32) {}
    fn observe(&self, out: &mut [f32]) {
        out.fill(0.0);
    }
    fn step(&mut self, _: usize, _: &mut Philox4x32) -> f32 {
        0.0
    }
    fn render_state(&self) -> String {
        String::new()
    }
}

pub const BOUNCE_W: usize = 10;
pub const BOUNCE_H: usize = 10;

/// **Bounce** — a ball falls one row per step and moves one column per step,
/// reflecting off the side walls. A width-1 paddle on the bottom row must be
/// under the ball when it arrives: `+1` for a catch, `-1` for a miss, then the
/// ball respawns on the top row with a random column and direction.
///
/// Following the ball's *current* column lags behind a diagonal ball; scoring
/// every drop requires predicting the *landing* column (wall bounces included)
/// and moving there. From a respawn there are exactly `W-1` steps before the
/// ball lands, so every drop is catchable.
#[derive(Debug, Clone)]
pub struct Bounce {
    w: usize,
    h: usize,
    ball_x: i32,
    ball_y: i32,
    ball_dx: i32,
    paddle: i32,
}

impl Bounce {
    pub fn new(w: usize, h: usize) -> Self {
        assert!(w >= 2 && h >= 2);
        Self {
            w,
            h,
            ball_x: 0,
            ball_y: 0,
            ball_dx: 1,
            paddle: 0,
        }
    }

    fn respawn(&mut self, rng: &mut Philox4x32) {
        self.ball_x = (rng.next_u32() as usize % self.w) as i32;
        self.ball_y = 0;
        self.ball_dx = if rng.next_u32() & 1 == 0 { -1 } else { 1 };
    }

    /// Column the ball reaches on the paddle row from `(x, y)` moving `dx`.
    pub fn landing_column(w: usize, h: usize, x: i32, y: i32, dx: i32) -> i32 {
        let (mut x, mut dx) = (x, dx);
        for _ in y..(h as i32 - 1) {
            (x, dx) = Self::advance_x(w, x, dx);
        }
        x
    }

    pub(crate) fn advance_x(w: usize, x: i32, dx: i32) -> (i32, i32) {
        let nx = x + dx;
        let last = w as i32 - 1;
        if nx < 0 {
            (-nx, -dx)
        } else if nx > last {
            (2 * last - nx, -dx)
        } else {
            (nx, dx)
        }
    }

    /// Observation index of the ball cell `(x, y)` moving `dx`.
    pub fn ball_index(&self, x: usize, y: usize, dx: i32) -> usize {
        let plane = if dx < 0 { 0 } else { 1 };
        plane * self.w * self.h + y * self.w + x
    }

    /// Observation index of the paddle at column `x`.
    pub fn paddle_index(&self, x: usize) -> usize {
        2 * self.w * self.h + x
    }

    pub fn width(&self) -> usize {
        self.w
    }

    pub fn height(&self) -> usize {
        self.h
    }

    /// `(ball_x, ball_y, ball_dx, paddle_x)`.
    pub fn state(&self) -> (i32, i32, i32, i32) {
        (self.ball_x, self.ball_y, self.ball_dx, self.paddle)
    }
}

impl Env for Bounce {
    fn spec(&self) -> EnvSpec {
        let (w, h) = (self.w, self.h);
        let mut labels = Vec::with_capacity(2 * w * h + w);
        for (glyph, _) in [("<", -1), (">", 1)] {
            for y in 0..h {
                for x in 0..w {
                    labels.push(format!("ball{glyph}(x={x},y={y})"));
                }
            }
        }
        for x in 0..w {
            labels.push(format!("paddle(x={x})"));
        }
        EnvSpec {
            name: "bounce",
            description: format!(
                "Grid {w} wide x {h} tall, y=0 is the top row, paddle lives on row y={py}. \
                 Each step: (1) the paddle moves by the action (stay / left = x-1 / right = x+1, \
                 clamped to [0,{xl}]); (2) the ball moves x += dx, y += 1, reflecting off the side \
                 walls (x=-1 becomes x=1, x={w} becomes x={x2}, and dx flips). When the ball reaches \
                 row y={py}: reward +1 if paddle x == ball x, else -1; the ball then respawns at y=0 \
                 with a uniformly random column and direction. Observations are one-hot: plane \
                 'ball<' marks the ball cell when moving left (dx=-1), plane 'ball>' when moving \
                 right (dx=+1), and 'paddle' marks the paddle column. The ball is never observed \
                 on row y={py}. Episode = {hz} steps; the return is catches minus misses.",
                py = h - 1,
                xl = w - 1,
                x2 = w - 2,
                hz = BOUNCE_HORIZON,
            ),
            obs_dim: labels.len(),
            obs_labels: labels,
            actions: vec!["stay", "left", "right"],
            horizon: BOUNCE_HORIZON,
        }
    }

    fn reset(&mut self, rng: &mut Philox4x32) {
        self.paddle = (rng.next_u32() as usize % self.w) as i32;
        self.respawn(rng);
    }

    fn observe(&self, out: &mut [f32]) {
        out.fill(0.0);
        out[self.ball_index(self.ball_x as usize, self.ball_y as usize, self.ball_dx)] = 1.0;
        out[self.paddle_index(self.paddle as usize)] = 1.0;
    }

    fn step(&mut self, action: usize, rng: &mut Philox4x32) -> f32 {
        let last = self.w as i32 - 1;
        self.paddle = match action {
            1 => (self.paddle - 1).max(0),
            2 => (self.paddle + 1).min(last),
            _ => self.paddle,
        };
        (self.ball_x, self.ball_dx) = Self::advance_x(self.w, self.ball_x, self.ball_dx);
        self.ball_y += 1;
        if self.ball_y == self.h as i32 - 1 {
            let r = if self.ball_x == self.paddle {
                1.0
            } else {
                -1.0
            };
            self.respawn(rng);
            r
        } else {
            0.0
        }
    }

    fn render_state(&self) -> String {
        format!(
            "ball=(x={},y={},dx={:+}) paddle={}",
            self.ball_x, self.ball_y, self.ball_dx, self.paddle
        )
    }
}

pub const BOUNCE_HORIZON: usize = 200;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landing_matches_simulation() {
        let mut rng = Philox4x32::new(7);
        for _ in 0..500 {
            let mut env = Bounce::new(BOUNCE_W, BOUNCE_H);
            env.reset(&mut rng);
            let (x, y, dx, _) = env.state();
            let want = Bounce::landing_column(BOUNCE_W, BOUNCE_H, x, y, dx);
            // Park the paddle on the predicted column; the drop must be caught.
            env.paddle = want;
            let mut r = 0.0;
            for _ in 0..(BOUNCE_H - 1) {
                r = env.step(0, &mut rng);
            }
            assert_eq!(r, 1.0);
        }
    }
}
