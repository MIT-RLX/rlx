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
// RLX — Flow Map Q-Guidance.

/// Off-diagonal self-distillation variant (Python `distillation_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistillationType {
    #[default]
    Mf,
    Lsd,
    Psd,
}

impl DistillationType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "lsd" => Self::Lsd,
            "psd" => Self::Psd,
            _ => Self::Mf,
        }
    }
}

/// Problem dimensions and training hyperparameters.
#[derive(Debug, Clone)]
pub struct RlSpec {
    pub state_dim: usize,
    pub action_dim: usize,
    pub batch: usize,
    /// Hidden layers for actor and critic MLPs (excluding output heads).
    pub hidden: Vec<usize>,
    pub gamma: f32,
    /// Trust-region radius for FMQ projection (before adaptive scaling).
    pub eta: f32,
    /// Adaptive trust region: `eta_eff = 1 / (1 + beta * delta_norm)`.
    pub eta_beta: f32,
    pub eta_kappa: f32,
    pub actor_lr: f32,
    pub critic_lr: f32,
    pub tau: f32,
    /// FMQ trust region: η = σ²/(2α) when `fmq_eta_override` is None (Python `fmq.py`).
    pub fmq_alpha: f32,
    pub fmq_sigma_sq: f32,
    /// If set, fixes η (Python `fmq_eta_override >= 0`).
    pub fmq_eta_override: Option<f32>,
    pub fmq_adaptive_eta: bool,
    /// β in Eq. 13 when `fmq_adaptive_eta` (Python `fmq_beta`).
    pub fmq_beta: f32,
    /// Evaluate ∇Q at `a1_online` instead of `a1_off` (Python `fmq_grad_at_online`).
    pub fmq_grad_at_online: bool,
    pub fmq_normalize_grad: bool,
    /// Best-of-N sample count at eval (Python `actor_num_samples`).
    pub actor_num_samples: usize,
    pub action_clip: f32,
    /// Offline / online ESD curriculum (Python `flow_map_*` / `esd_*`).
    pub flow_map_warmup_steps: usize,
    pub flow_map_anneal_end_step: usize,
    pub esd_warmup_steps: usize,
    pub esd_anneal_end_step: usize,
    pub distillation_type: DistillationType,
    /// Online FMQ auxiliary weights (Python `esd_weight`, `diag_weight`).
    pub esd_weight: f32,
    pub diag_weight: f32,
    /// QGBS trust region at eval (Python `qgbs_eta`).
    pub qgbs_eta: f32,
}

impl RlSpec {
    pub fn toy(batch: usize) -> Self {
        Self {
            state_dim: 4,
            action_dim: 2,
            batch,
            hidden: vec![64, 64],
            gamma: 0.99,
            eta: 0.5,
            eta_beta: 0.3,
            eta_kappa: 1e-4,
            actor_lr: 3e-4,
            critic_lr: 3e-4,
            tau: 0.005,
            fmq_alpha: 1.0,
            fmq_sigma_sq: 1.0,
            fmq_eta_override: None,
            fmq_adaptive_eta: false,
            fmq_beta: 0.3,
            fmq_grad_at_online: false,
            fmq_normalize_grad: true,
            actor_num_samples: 32,
            action_clip: 1.0,
            flow_map_warmup_steps: 5,
            flow_map_anneal_end_step: 50,
            esd_warmup_steps: 0,
            esd_anneal_end_step: 50,
            distillation_type: DistillationType::Mf,
            esd_weight: 0.0,
            diag_weight: 0.0,
            qgbs_eta: 0.3,
        }
    }

    /// Trust-region step size (Python `_get_eta`).
    pub fn fmq_eta(&self) -> f32 {
        if let Some(e) = self.fmq_eta_override {
            return e;
        }
        self.fmq_sigma_sq / (2.0 * self.fmq_alpha)
    }

    pub fn with_batch(&self, batch: usize) -> Self {
        let mut s = self.clone();
        s.batch = batch;
        s
    }

    pub fn actor_in_dim(&self) -> usize {
        self.state_dim + self.action_dim + 2
    }

    pub fn critic_in_dim(&self) -> usize {
        self.state_dim + self.action_dim
    }
}
