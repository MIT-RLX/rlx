// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
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
