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
// RLX — optional 2D reach-goal MDP for tests / examples (`feature = "toy"`).

use crate::buffer::Transition;
use crate::env::RlEnv;

/// State: `[ax, ay, gx, gy]`, action: 2D velocity clipped to `[-1, 1]`.
#[derive(Debug, Clone)]
pub struct ToyGoalEnv {
    pub state: [f32; 4],
    pub max_steps: usize,
    pub step: usize,
    pub action_scale: f32,
}

impl Default for ToyGoalEnv {
    fn default() -> Self {
        Self {
            state: [0.0, 0.0, 1.0, 1.0],
            max_steps: 50,
            step: 0,
            action_scale: 0.15,
        }
    }
}

impl ToyGoalEnv {
    pub fn expert_action(&self) -> [f32; 2] {
        let dx = self.state[2] - self.state[0];
        let dy = self.state[3] - self.state[1];
        let n = (dx * dx + dy * dy).sqrt().max(1e-6);
        [(dx / n).clamp(-1.0, 1.0), (dy / n).clamp(-1.0, 1.0)]
    }

    pub fn collect_expert_episodes(n_episodes: usize, max_steps: usize) -> Vec<Transition> {
        let mut out = Vec::new();
        for _ in 0..n_episodes {
            let mut env = Self {
                max_steps,
                ..Default::default()
            };
            env.reset();
            loop {
                let a = env.expert_action();
                let tr = env.step(&a);
                let done = tr.done;
                out.push(tr);
                if done {
                    break;
                }
            }
        }
        out
    }
}

impl RlEnv for ToyGoalEnv {
    fn reset(&mut self) -> Vec<f32> {
        self.state = [0.0, 0.0, 1.0, 1.0];
        self.step = 0;
        self.state.to_vec()
    }

    fn step(&mut self, action: &[f32]) -> Transition {
        let prev = self.state;
        self.state[0] += self.action_scale * action[0];
        self.state[1] += self.action_scale * action[1];
        self.step += 1;
        let dist = ((self.state[0] - self.state[2]).powi(2)
            + (self.state[1] - self.state[3]).powi(2))
        .sqrt();
        let reward = -dist;
        let done = dist < 0.05 || self.step >= self.max_steps;
        Transition {
            state: prev.to_vec(),
            action: action.to_vec(),
            reward,
            next_state: self.state.to_vec(),
            done,
        }
    }
}
