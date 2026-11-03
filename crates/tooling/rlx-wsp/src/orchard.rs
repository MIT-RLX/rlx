// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Orchard** — several falling objects at once, fruit to catch and bombs to
//! avoid.
//!
//! Bounce has one ball, so a single linear unit can hold "landing column minus
//! paddle column". Here up to [`ORCHARD_MAX_OBJECTS`] objects are in the air
//! together and a linear unit *sums* over all of them; a good network has to
//! select (soonest reachable fruit), trade off (two fruits landing at once,
//! a bomb on the way) and gate on the paddle — logic that needs thresholds
//! and conjunctions across units, not one comparator.
//!
//! [`Orchard::plan_action`] is a reference controller (not a network): exact
//! dynamic programming over paddle moves against every object currently
//! visible, ignoring objects that have not spawned yet.
//!
//! **Gusts** ([`Orchard::gusts`]) is the same game under [`Rules::GUSTS`], with
//! motion the agent is *not told*: the side walls wrap instead of reflecting,
//! objects move two columns per step from row [`GUSTS_ROW`] down, and bombs
//! fall two rows per step. `describe` states the rewards, spawning and
//! observation layout and says the motion is to be discovered; rollout traces
//! (which cost forward passes) show true object positions. Every hidden rule
//! contradicts what Bounce and Orchard teach, so carrying their rules over is
//! measurably worse ([`Orchard::plan_action_assuming`] with [`Rules::ORCHARD`]).

use crate::env::{Bounce, Env, EnvSpec};
use rlx_ir::Philox4x32;

pub const ORCHARD_W: usize = 10;
pub const ORCHARD_H: usize = 10;
pub const ORCHARD_HORIZON: usize = 200;
pub const ORCHARD_MAX_OBJECTS: usize = 4;
/// Chance per step that a new object spawns (when fewer than the max are up).
pub const ORCHARD_P_SPAWN: f32 = 0.3;
/// Chance a spawned object is a bomb.
pub const ORCHARD_P_BOMB: f32 = 0.3;
pub const ORCHARD_FRUIT_REWARD: f32 = 1.0;
pub const ORCHARD_BOMB_REWARD: f32 = -3.0;

/// First row (from the top) at which Gusts objects move two columns per step.
pub const GUSTS_ROW: i32 = 5;

/// Grid size and object traffic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Layout {
    pub w: usize,
    pub h: usize,
    pub max_objects: usize,
    /// Chance per step that a new object spawns (when below the max).
    pub p_spawn: f32,
    /// Chance a spawned object is a bomb.
    pub p_bomb: f32,
}

impl Layout {
    /// Orchard and Gusts: 10×10, up to 4 objects.
    pub const ORCHARD: Layout = Layout {
        w: ORCHARD_W,
        h: ORCHARD_H,
        max_objects: ORCHARD_MAX_OBJECTS,
        p_spawn: ORCHARD_P_SPAWN,
        p_bomb: ORCHARD_P_BOMB,
    };
    /// Gusts-hard: 16 wide × 12 tall, up to 6 objects, more spawns. A ball
    /// spawning far from the paddle can be out of reach, so the program has
    /// to choose; and one unit per (paddle column, action) no longer fits in
    /// 32 hidden units.
    pub const GUSTS_HARD: Layout = Layout {
        w: 16,
        h: 12,
        max_objects: 6,
        p_spawn: 0.45,
        p_bomb: 0.3,
    };
}

/// How objects move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rules {
    /// Side walls wrap around (`x=-1` → `x=w-1`) instead of reflecting.
    pub wrap: bool,
    /// From this row down, objects move two columns per step.
    pub gust_row: Option<i32>,
    /// Rows a bomb falls per step (fruit always fall one).
    pub bomb_rows_per_step: i32,
}

impl Rules {
    /// Orchard's stated rules: reflecting walls, one column and one row per step.
    pub const ORCHARD: Rules = Rules {
        wrap: false,
        gust_row: None,
        bomb_rows_per_step: 1,
    };
    /// Gusts' hidden rules.
    pub const GUSTS: Rules = Rules {
        wrap: true,
        gust_row: Some(GUSTS_ROW),
        bomb_rows_per_step: 2,
    };
    /// Gusts-hard's hidden rules: as Gusts, the gust starting at row 6.
    pub const GUSTS_HARD: Rules = Rules {
        wrap: true,
        gust_row: Some(6),
        bomb_rows_per_step: 2,
    };

    /// One step of motion for `o` on a grid `w` wide.
    pub fn advance(&self, w: usize, o: &mut Object) {
        let columns = if self.gust_row.is_some_and(|g| o.y >= g) {
            2
        } else {
            1
        };
        for _ in 0..columns {
            if self.wrap {
                o.x = (o.x + o.dx).rem_euclid(w as i32);
            } else {
                (o.x, o.dx) = Bounce::advance_x(w, o.x, o.dx);
            }
        }
        o.y += if o.bomb { self.bomb_rows_per_step } else { 1 };
    }

    /// `(k, column)`: the object lands on the `k`-th step from now (0 = next
    /// step) in `column`.
    pub fn landing(&self, w: usize, h: usize, o: &Object) -> (usize, i32) {
        let mut o = *o;
        let mut k = 0;
        loop {
            self.advance(w, &mut o);
            if o.y >= h as i32 - 1 {
                return (k, o.x);
            }
            k += 1;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Object {
    pub x: i32,
    pub y: i32,
    pub dx: i32,
    pub bomb: bool,
}

#[derive(Debug, Clone)]
pub struct Orchard {
    w: usize,
    h: usize,
    objects: Vec<Object>,
    paddle: i32,
    rules: Rules,
    layout: Layout,
}

impl Default for Orchard {
    fn default() -> Self {
        Self::new()
    }
}

impl Orchard {
    pub fn new() -> Self {
        Self::with_rules(Rules::ORCHARD)
    }

    /// Orchard under the hidden [`Rules::GUSTS`].
    pub fn gusts() -> Self {
        Self::with_rules(Rules::GUSTS)
    }

    /// Gusts-hard: hidden rules on a wider, busier grid ([`Layout::GUSTS_HARD`]).
    pub fn gusts_hard() -> Self {
        Self::with(Rules::GUSTS_HARD, Layout::GUSTS_HARD)
    }

    pub fn with_rules(rules: Rules) -> Self {
        Self::with(rules, Layout::ORCHARD)
    }

    pub fn with(rules: Rules, layout: Layout) -> Self {
        Self {
            w: layout.w,
            h: layout.h,
            objects: Vec::new(),
            paddle: 0,
            rules,
            layout,
        }
    }

    /// `(width, height)` of the grid.
    pub fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    pub fn rules(&self) -> Rules {
        self.rules
    }

    pub fn objects(&self) -> &[Object] {
        &self.objects
    }

    pub fn paddle(&self) -> i32 {
        self.paddle
    }

    /// Observation index of an object cell (`y < h-1`).
    pub fn object_index(&self, bomb: bool, dx: i32, x: usize, y: usize) -> usize {
        let plane = 2 * usize::from(bomb) + usize::from(dx > 0);
        plane * self.w * (self.h - 1) + y * self.w + x
    }

    pub fn paddle_index(&self, x: usize) -> usize {
        4 * self.w * (self.h - 1) + x
    }

    fn maybe_spawn(&mut self, rng: &mut Philox4x32) {
        if self.objects.len() < self.layout.max_objects && rng.next_f32() < self.layout.p_spawn {
            let x = (rng.next_u32() as usize % self.w) as i32;
            let dx = if rng.next_u32() & 1 == 0 { -1 } else { 1 };
            let bomb = rng.next_f32() < self.layout.p_bomb;
            self.objects.push(Object { x, y: 0, dx, bomb });
        }
    }

    /// Best action against the visible objects: DP over (steps ahead,
    /// paddle column), each object landing at a known step and column.
    pub fn plan_action(&self) -> usize {
        self.plan_action_assuming(self.rules)
    }

    /// [`Self::plan_action`] with landing predicted by `rules` instead of the
    /// true ones (what an agent that assumed them would aim for).
    pub fn plan_action_assuming(&self, rules: Rules) -> usize {
        let (w, h) = (self.w, self.h);
        let horizon = h; // every visible object lands within h-1 steps
        // reward[k][c]: reward collected if the paddle is at c when step k lands.
        let mut reward = vec![vec![0f32; w]; horizon];
        for o in &self.objects {
            let (k, c) = rules.landing(w, h, o);
            let c = c as usize;
            reward[k][c] += if o.bomb {
                ORCHARD_BOMB_REWARD
            } else {
                ORCHARD_FRUIT_REWARD
            };
        }
        let step = |p: i32, a: usize| -> i32 {
            match a {
                1 => (p - 1).max(0),
                2 => (p + 1).min(w as i32 - 1),
                _ => p,
            }
        };
        // value[k][p]: best total from step k with the paddle at p.
        let mut value = vec![vec![0f32; w]; horizon + 1];
        for k in (0..horizon).rev() {
            for p in 0..w {
                value[k][p] = (0..3)
                    .map(|a| {
                        let q = step(p as i32, a) as usize;
                        reward[k][q] + value[k + 1][q]
                    })
                    .fold(f32::NEG_INFINITY, f32::max);
            }
        }
        // Ties → stay, then left, then right (matches the network's argmax).
        let mut best = 0;
        let mut best_v = f32::NEG_INFINITY;
        for a in 0..3 {
            let q = step(self.paddle, a) as usize;
            let v = reward[0][q] + value[1][q];
            if v > best_v + 1e-6 {
                best = a;
                best_v = v;
            }
        }
        best
    }
}

impl Env for Orchard {
    fn spec(&self) -> EnvSpec {
        let (w, h) = (self.w, self.h);
        let mut labels = Vec::with_capacity(4 * w * (h - 1) + w);
        for kind in ["fruit", "bomb"] {
            for glyph in ["<", ">"] {
                for y in 0..h - 1 {
                    for x in 0..w {
                        labels.push(format!("{kind}{glyph}(x={x},y={y})"));
                    }
                }
            }
        }
        for x in 0..w {
            labels.push(format!("paddle(x={x})"));
        }
        if self.rules != Rules::ORCHARD {
            return EnvSpec {
                name: if self.layout == Layout::ORCHARD {
                    "gusts"
                } else {
                    "gusts_hard"
                },
                description: format!(
                    "Grid {w} wide x {h} tall, y=0 is the top row, a width-1 paddle lives on row \
                     y={py}. Objects fall: each is a fruit or a bomb and has a horizontal \
                     direction, shown by its observation plane ('<' or '>'). Each step, in order: \
                     (1) the paddle moves by the action (stay / left = x-1 / right = x+1, clamped \
                     to [0,{xl}]); (2) every object moves. HOW objects move is NOT given: their \
                     vertical and horizontal speed, what happens at the side walls, and whether \
                     fruit and bombs move alike are for you to find out (rollout traces show the \
                     true object positions step by step). An object that reaches row y={py} (or \
                     beyond) is removed, scoring {fr:+} if it is a fruit in the paddle's column, \
                     {br:+} if it is a bomb in the paddle's column, and 0 otherwise; (3) if fewer \
                     than {max} objects are in the air, with probability {ps} a new object appears \
                     on row y=0 with a uniformly random column and direction; it is a bomb with \
                     probability {pb}, otherwise a fruit. Episodes start with no objects. \
                     Observations: four planes 'fruit<', 'fruit>', 'bomb<', 'bomb>' (kind and \
                     direction) over rows y=0..{yl}, each cell holding the NUMBER of such objects \
                     there, plus a one-hot 'paddle' row. Episode = {hz} steps; the return is the \
                     sum of rewards. Several objects can be in the air at once, so not every fruit \
                     can be caught.",
                    py = h - 1,
                    xl = w - 1,
                    yl = h - 2,
                    fr = ORCHARD_FRUIT_REWARD,
                    br = ORCHARD_BOMB_REWARD,
                    max = self.layout.max_objects,
                    ps = self.layout.p_spawn,
                    pb = self.layout.p_bomb,
                    hz = ORCHARD_HORIZON,
                ),
                obs_dim: labels.len(),
                obs_labels: labels,
                actions: vec!["stay", "left", "right"],
                horizon: ORCHARD_HORIZON,
            };
        }
        EnvSpec {
            name: "orchard",
            description: format!(
                "Grid {w} wide x {h} tall, y=0 is the top row, a width-1 paddle lives on row \
                 y={py}. Objects fall: each is a fruit or a bomb with a horizontal direction \
                 dx = -1 or +1. Each step, in order: (1) the paddle moves by the action (stay / \
                 left = x-1 / right = x+1, clamped to [0,{xl}]); (2) every object moves x += dx, \
                 y += 1, reflecting off the side walls (x=-1 becomes x=1, x={w} becomes x={x2}, \
                 and dx flips); an object reaching row y={py} is removed, scoring {fr:+} if it is a \
                 fruit under the paddle, {br:+} if it is a bomb under the paddle, and 0 if it \
                 lands elsewhere; (3) if fewer than {max} objects are in the air, with probability \
                 {ps} a new object appears on row y=0 with a uniformly random column and direction; \
                 it is a bomb with probability {pb}, otherwise a fruit. Episodes start with no \
                 objects. Observations: four planes 'fruit<', 'fruit>', 'bomb<', 'bomb>' (kind \
                 and direction) over rows y=0..{yl}, each cell holding the NUMBER of such objects \
                 there (usually 0 or 1; objects never appear on row y={py}), plus a one-hot \
                 'paddle' row. Episode = {hz} steps; the return is the sum of rewards. Several \
                 objects can be in the air at once and two can land on the same step, so not \
                 every fruit can be caught.",
                py = h - 1,
                xl = w - 1,
                x2 = w - 2,
                yl = h - 2,
                fr = ORCHARD_FRUIT_REWARD,
                br = ORCHARD_BOMB_REWARD,
                max = self.layout.max_objects,
                ps = self.layout.p_spawn,
                pb = self.layout.p_bomb,
                hz = ORCHARD_HORIZON,
            ),
            obs_dim: labels.len(),
            obs_labels: labels,
            actions: vec!["stay", "left", "right"],
            horizon: ORCHARD_HORIZON,
        }
    }

    fn reset(&mut self, rng: &mut Philox4x32) {
        self.objects.clear();
        self.paddle = (rng.next_u32() as usize % self.w) as i32;
    }

    fn observe(&self, out: &mut [f32]) {
        out.fill(0.0);
        for o in &self.objects {
            out[self.object_index(o.bomb, o.dx, o.x as usize, o.y as usize)] += 1.0;
        }
        out[self.paddle_index(self.paddle as usize)] = 1.0;
    }

    fn step(&mut self, action: usize, rng: &mut Philox4x32) -> f32 {
        let last = self.w as i32 - 1;
        self.paddle = match action {
            1 => (self.paddle - 1).max(0),
            2 => (self.paddle + 1).min(last),
            _ => self.paddle,
        };
        let mut r = 0.0;
        let (w, landing_row, paddle, rules) = (self.w, self.h as i32 - 1, self.paddle, self.rules);
        self.objects.retain_mut(|o| {
            rules.advance(w, o);
            if o.y < landing_row {
                return true;
            }
            if o.x == paddle {
                r += if o.bomb {
                    ORCHARD_BOMB_REWARD
                } else {
                    ORCHARD_FRUIT_REWARD
                };
            }
            false
        });
        self.maybe_spawn(rng);
        r
    }

    fn render_state(&self) -> String {
        let objs: Vec<String> = self
            .objects
            .iter()
            .map(|o| {
                format!(
                    "{}(x={},y={},dx={:+})",
                    if o.bomb { "bomb" } else { "fruit" },
                    o.x,
                    o.y,
                    o.dx
                )
            })
            .collect();
        format!("paddle={} [{}]", self.paddle, objs.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_and_rewards() {
        let mut rng = Philox4x32::new(3);
        let mut env = Orchard::new();
        let spec = env.spec();
        assert_eq!(spec.obs_dim, 4 * 10 * 9 + 10);
        assert_eq!(
            spec.obs_labels[env.object_index(true, 1, 3, 5)],
            "bomb>(x=3,y=5)"
        );
        assert_eq!(spec.obs_labels[env.paddle_index(7)], "paddle(x=7)");
        env.reset(&mut rng);
        // A fruit one row above the paddle row, landing on the paddle.
        env.paddle = 4;
        env.objects = vec![Object {
            x: 3,
            y: 8,
            dx: 1,
            bomb: false,
        }];
        assert_eq!(env.step(0, &mut rng), ORCHARD_FRUIT_REWARD);
        env.objects = vec![Object {
            x: 3,
            y: 8,
            dx: 1,
            bomb: true,
        }];
        assert_eq!(env.step(0, &mut rng), ORCHARD_BOMB_REWARD);
    }

    #[test]
    fn gusts_rules() {
        let r = Rules::GUSTS;
        // Wraps at the wall, one column per step above the gust row.
        let mut o = Object {
            x: 9,
            y: 0,
            dx: 1,
            bomb: false,
        };
        r.advance(10, &mut o);
        assert_eq!((o.x, o.y), (0, 1));
        // Two columns per step from the gust row down.
        let mut o = Object {
            x: 3,
            y: GUSTS_ROW,
            dx: -1,
            bomb: false,
        };
        r.advance(10, &mut o);
        assert_eq!((o.x, o.y), (1, GUSTS_ROW + 1));
        // Bombs fall two rows; one on row 7 lands next step.
        let b = Object {
            x: 4,
            y: 7,
            dx: 1,
            bomb: true,
        };
        assert_eq!(r.landing(10, 10, &b), (0, 6));
        // Orchard rules agree with Bounce's landing column.
        let f = Object {
            x: 8,
            y: 0,
            dx: 1,
            bomb: false,
        };
        assert_eq!(
            Rules::ORCHARD.landing(10, 10, &f),
            (8, Bounce::landing_column(10, 10, 8, 0, 1))
        );
    }

    #[test]
    fn planner_dodges_and_catches() {
        let mut env = Orchard::new();
        env.paddle = 4;
        // Bomb landing on the paddle next step: must move.
        env.objects = vec![Object {
            x: 3,
            y: 8,
            dx: 1,
            bomb: true,
        }];
        assert_ne!(env.plan_action(), 0);
        // Fruit landing one column right next step: go right.
        env.objects = vec![Object {
            x: 4,
            y: 8,
            dx: 1,
            bomb: false,
        }];
        assert_eq!(env.plan_action(), 2);
    }
}
