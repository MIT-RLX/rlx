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

//! Auto-tuner — finds the optimal RuntimeConfig for a model on current hardware.
//!
//! Runs the model with different parameter combinations, measures each,
//! picks the fastest. Results are printed and can be saved.
//!
//! ```rust,ignore
//! let best = autotune(&compiled, &sample_inputs, 20);
//! eprintln!("Best config: {:?}", best);
//! ```

use crate::config::RuntimeConfig;
use rlx_ir::Tick;

/// Result of one tuning trial.
#[derive(Debug, Clone)]
pub struct TuneResult {
    pub config: RuntimeConfig,
    pub p50_ms: f64,
    pub min_ms: f64,
}

/// Search space for auto-tuning.
pub struct SearchSpace {
    pub workers: Vec<usize>,
    pub par_thresholds: Vec<usize>,
    pub sdpa_thresholds: Vec<usize>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            workers: vec![1, 2, cpus / 4, cpus / 2, cpus * 3 / 4],
            par_thresholds: vec![10_000, 20_000, 30_000, 50_000],
            sdpa_thresholds: vec![16, 32, 48],
        }
    }
}

/// Auto-tune by running the model with different configs.
///
/// `run_fn` is called for each trial — it should execute one forward pass.
/// `warmup` iterations are run before timing. `trials` are timed.
pub fn autotune<F>(mut run_fn: F, search: &SearchSpace, warmup: usize, trials: usize) -> TuneResult
where
    F: FnMut(),
{
    let mut results: Vec<TuneResult> = Vec::new();
    let base = RuntimeConfig::auto_detect();

    // Generate all combinations
    for &w in &search.workers {
        for &par in &search.par_thresholds {
            for &sdpa in &search.sdpa_thresholds {
                let cfg = RuntimeConfig {
                    pool_workers: w.clamp(1, 15),
                    par_threshold: par,
                    sdpa_seq_threshold: sdpa,
                    ..base.clone()
                };

                // Apply this config (affects global singleton for this process)
                // Note: pool workers can't be changed after init. Skip if different.
                // For now, only tune par_threshold and sdpa_threshold.
                unsafe {
                    // Override the global config pointer
                    set_global_config(cfg.clone());
                }

                // Warmup
                for _ in 0..warmup {
                    run_fn();
                }

                // Measure — direct CNTVCT_EL0 read on Apple Silicon (#66).
                // Sub-microsecond resolution lets the search distinguish
                // configs whose wall-clock times differ by a few ticks.
                let mut times = Vec::with_capacity(trials);
                for _ in 0..trials {
                    let t = Tick::now();
                    run_fn();
                    times.push(Tick::now().elapsed_ms(t));
                }
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = times[trials / 2];
                let min = times[0];

                eprintln!(
                    "  workers={w:2} par={par:5} sdpa={sdpa:2} → p50={p50:.2}ms min={min:.2}ms"
                );
                results.push(TuneResult {
                    config: cfg,
                    p50_ms: p50,
                    min_ms: min,
                });
            }
        }
    }

    // Find best by p50
    results.sort_by(|a, b| a.p50_ms.partial_cmp(&b.p50_ms).unwrap());
    let best = results[0].clone();

    // Apply best config
    unsafe {
        set_global_config(best.config.clone());
    }

    eprintln!(
        "[rlx] best: workers={} par={} sdpa={} → {:.2}ms p50",
        best.config.pool_workers,
        best.config.par_threshold,
        best.config.sdpa_seq_threshold,
        best.p50_ms
    );

    best
}

/// Override the global RuntimeConfig.
/// SAFETY: must only be called during auto-tuning (single-threaded phase).
unsafe fn set_global_config(cfg: RuntimeConfig) {
    // The OnceLock pattern doesn't allow re-setting.
    // For auto-tuning, we use a separate mutable global.
    TUNE_CONFIG.lock().unwrap().replace(cfg);
}

/// Get the active tuning config (if set), otherwise fall back to global.
pub fn active_config() -> RuntimeConfig {
    if let Some(cfg) = TUNE_CONFIG.lock().unwrap().as_ref() {
        cfg.clone()
    } else {
        RuntimeConfig::global().clone()
    }
}

static TUNE_CONFIG: std::sync::Mutex<Option<RuntimeConfig>> = std::sync::Mutex::new(None);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_space_default() {
        let ss = SearchSpace::default();
        assert!(ss.workers.len() >= 3);
        assert_eq!(ss.par_thresholds.len(), 4);
        assert_eq!(ss.sdpa_thresholds.len(), 3);
    }
}
