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
// RLX — replay buffer.

/// One environment transition.
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

/// FIFO replay buffer with uniform sampling.
#[derive(Debug, Clone, Default)]
pub struct ReplayBuffer {
    data: Vec<Transition>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::new(),
            capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn push(&mut self, t: Transition) {
        if self.data.len() >= self.capacity {
            self.data.remove(0);
        }
        self.data.push(t);
    }

    pub fn extend<I: IntoIterator<Item = Transition>>(&mut self, iter: I) {
        for t in iter {
            self.push(t);
        }
    }

    /// Sample `n` transition indices (with replacement if `n` > len).
    pub fn sample_indices(&self, n: usize, seed: &mut u64) -> Vec<usize> {
        if self.data.is_empty() {
            return Vec::new();
        }
        (0..n)
            .map(|_| {
                *seed = rand_like(*seed);
                (*seed as usize) % self.data.len()
            })
            .collect()
    }

    pub fn get(&self, idx: usize) -> &Transition {
        &self.data[idx]
    }

    pub fn iter(&self) -> impl Iterator<Item = &Transition> {
        self.data.iter()
    }
}

/// Minimal LCG for sampling without pulling in `rand`.
pub fn rand_like(seed: u64) -> u64 {
    seed.wrapping_mul(6364136223846793005).wrapping_add(1)
}
