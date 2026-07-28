// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// CoopF16Vk runtime routing: oscillating B → wide f32; N > 768 → wide f32
// unless RLX_WGPU_COOP_F16_VK_LARGE_N opts back into coop tensor cores.

use std::collections::{HashMap, HashSet};

use crate::kernels::COOP_F16_VK_WIDEN_N;

/// Mean absolute consecutive difference divided by mean absolute value.
/// Gentle BERT-like weights stay below ~0.1; flat-index sin/cos stress is ~1+.
pub fn weight_oscillation_score(data: &[f32]) -> f32 {
    if data.len() < 2 {
        return 0.0;
    }
    let n = data.len();
    let mean_abs: f64 = data.iter().map(|&v| v.abs() as f64).sum::<f64>() / n as f64;
    let mean_diff: f64 = data
        .windows(2)
        .map(|w| (w[1] - w[0]).abs() as f64)
        .sum::<f64>()
        / (n - 1) as f64;
    (mean_diff / mean_abs.max(1e-12)) as f32
}

fn oscillation_threshold() -> f32 {
    rlx_ir::env::parse_or("RLX_WGPU_COOP_F16_VK_OSC_THRESH", 0.35_f32)
}

fn auto_wide_disabled() -> bool {
    rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_NO_AUTO_WIDE")
}

fn force_wide() -> bool {
    rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_FORCE_WIDE")
}

/// Update `wide_b` from a Param payload uploaded via `set_param`.
pub fn refresh_wide_b_flag(wide_b: &mut HashSet<String>, name: &str, data: &[f32]) {
    if param_triggers_oscillation_wide(data) {
        wide_b.insert(name.to_string());
    } else {
        wide_b.remove(name);
    }
}

fn param_triggers_oscillation_wide(data: &[f32]) -> bool {
    if auto_wide_disabled() {
        return false;
    }
    if force_wide() {
        return true;
    }
    weight_oscillation_score(data) >= oscillation_threshold()
}

/// Whether CoopF16Vk should dispatch `matmul_wide_nv` instead of a coop kernel.
pub fn use_wide_matmul(
    b_off_f32: u32,
    n: u32,
    b_param: &HashMap<u32, String>,
    wide_b: &HashSet<String>,
) -> bool {
    if auto_wide_disabled() {
        return false;
    }
    if force_wide() {
        return true;
    }
    if n > COOP_F16_VK_WIDEN_N && !rlx_ir::env::flag("RLX_WGPU_COOP_F16_VK_LARGE_N") {
        return true;
    }
    b_param
        .get(&b_off_f32)
        .is_some_and(|name| wide_b.contains(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oscillation_separates_gentle_from_stress() {
        let gentle: Vec<f32> = (0..384 * 1152)
            .map(|x| 0.05 * (x as f32 * 0.02).cos())
            .collect();
        let stress: Vec<f32> = (0..384 * 1152).map(|x| 0.1 * (x as f32).cos()).collect();
        assert!(
            weight_oscillation_score(&gentle) < 0.35,
            "gentle score = {}",
            weight_oscillation_score(&gentle)
        );
        assert!(
            weight_oscillation_score(&stress) >= 0.35,
            "stress score = {}",
            weight_oscillation_score(&stress)
        );
        assert!(!param_triggers_oscillation_wide(&gentle));
        assert!(param_triggers_oscillation_wide(&stress));
    }

    #[test]
    fn large_n_defaults_to_wide_unless_opt_in() {
        let empty: HashMap<u32, String> = HashMap::new();
        let wide_b: HashSet<String> = HashSet::new();
        assert!(use_wide_matmul(0, 1152, &empty, &wide_b));
    }
}
