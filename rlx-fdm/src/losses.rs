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

//! Loss / error aggregation (jax_fdm `losses.errors`).

use crate::mesh::MeshStructure;
use crate::objective::Goal;
use crate::state::EquilibriumState;
use crate::structure::Structure;

/// How to aggregate per-goal errors (jax_fdm `SquaredError`, `LogMaxError`, …).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// `Σ w (pred − target)²`
    Squared,
    /// `Σ w (pred − target)² / n_goals`
    MeanSquared,
    /// `√(mean squared error)`
    RootMeanSquared,
    /// `Σ w |pred − target|`
    Absolute,
    /// `Σ w · pred` (minimize prediction, jax_fdm `PredictionError`)
    Prediction,
    /// `Σ w · log1p(max(0, pred − bound))` for upper-bound constraints
    LogMax,
}

impl Default for ErrorKind {
    fn default() -> Self {
        Self::Squared
    }
}

/// Weighted goals with a shared error metric.
#[derive(Clone, Debug)]
pub struct Loss {
    pub goals: Vec<Goal>,
    pub error: ErrorKind,
}

impl Loss {
    pub fn new(goals: Vec<Goal>) -> Self {
        Self {
            goals,
            error: ErrorKind::default(),
        }
    }

    pub fn with_error(mut self, error: ErrorKind) -> Self {
        self.error = error;
        self
    }

    pub fn scalar(
        &self,
        state: &EquilibriumState,
        structure: &Structure,
        is_support: &[bool],
        mesh: Option<&MeshStructure>,
    ) -> f64 {
        let n = self.goals.len().max(1);
        let mut sum = 0.0;
        for g in &self.goals {
            let pred = g.prediction_with_structure(state, structure, is_support, mesh);
            let target = g.target();
            let w = g.weight();
            sum += term(self.error, pred, target, w);
        }
        match self.error {
            ErrorKind::MeanSquared | ErrorKind::RootMeanSquared => sum / n as f64,
            _ => sum,
        }
    }

    pub fn root_mean_squared(goals: Vec<Goal>) -> Self {
        Self::new(goals).with_error(ErrorKind::RootMeanSquared)
    }
}

fn term(error: ErrorKind, pred: f64, target: f64, weight: f64) -> f64 {
    match error {
        ErrorKind::Squared | ErrorKind::MeanSquared | ErrorKind::RootMeanSquared => {
            weight * (pred - target).powi(2)
        }
        ErrorKind::Absolute => weight * (pred - target).abs(),
        ErrorKind::Prediction => weight * pred,
        ErrorKind::LogMax => {
            let v = (pred - target).max(0.0);
            weight * (1.0 + v).ln()
        }
    }
}

/// Sum of [`Loss::scalar`] for multiple loss terms.
pub fn losses_total(
    losses: &[Loss],
    state: &EquilibriumState,
    structure: &Structure,
    is_support: &[bool],
    mesh: Option<&MeshStructure>,
) -> f64 {
    losses.iter().map(|l| l.scalar(state, structure, is_support, mesh)).sum()
}
