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
// RLX — environment interface (no simulator bindings).

use crate::buffer::Transition;

/// Host-side MDP interface. Implement this for your simulator / robot stack.
///
/// RLX only sees [`Transition`] records; physics lives outside the compiler.
pub trait RlEnv {
    /// Initial state after reset (length = `state_dim`).
    fn reset(&mut self) -> Vec<f32>;

    /// Apply `action` (length = `action_dim`), return the transition.
    fn step(&mut self, action: &[f32]) -> Transition;
}
