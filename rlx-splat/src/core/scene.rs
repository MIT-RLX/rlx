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
//! Gaussian scene storage aligned with `src/scene/gaussian_scene.py`.

use super::sh::SUPPORTED_SH_COEFF_COUNT;

#[derive(Clone, Debug)]
pub struct GaussianScene {
    pub positions: Vec<f32>,
    pub scales: Vec<f32>,
    pub rotations: Vec<f32>,
    pub opacities: Vec<f32>,
    pub colors: Vec<f32>,
    pub sh_coeffs: Vec<f32>,
    pub sh_coeff_count: usize,
}

impl GaussianScene {
    pub fn new(
        positions: Vec<f32>,
        scales: Vec<f32>,
        rotations: Vec<f32>,
        opacities: Vec<f32>,
        colors: Vec<f32>,
        sh_coeffs: Vec<f32>,
        sh_coeff_count: usize,
    ) -> Self {
        let count = positions.len() / 3;
        assert_eq!(scales.len(), count * 3);
        assert_eq!(rotations.len(), count * 4);
        assert_eq!(opacities.len(), count);
        assert_eq!(colors.len(), count * 3);
        assert!(sh_coeffs.is_empty() || sh_coeffs.len() == count * sh_coeff_count * 3);
        Self {
            positions,
            scales,
            rotations,
            opacities,
            colors,
            sh_coeffs,
            sh_coeff_count,
        }
    }

    pub fn count(&self) -> usize {
        self.positions.len() / 3
    }

    pub fn position(&self, index: usize) -> [f32; 3] {
        let base = index * 3;
        [
            self.positions[base],
            self.positions[base + 1],
            self.positions[base + 2],
        ]
    }

    pub fn scale(&self, index: usize) -> [f32; 3] {
        let base = index * 3;
        [
            self.scales[base],
            self.scales[base + 1],
            self.scales[base + 2],
        ]
    }

    pub fn rotation(&self, index: usize) -> [f32; 4] {
        let base = index * 4;
        [
            self.rotations[base],
            self.rotations[base + 1],
            self.rotations[base + 2],
            self.rotations[base + 3],
        ]
    }

    pub fn subset(&self, max_splats: usize) -> Self {
        if max_splats == 0 || max_splats >= self.count() {
            return self.clone();
        }
        let n = max_splats;
        Self {
            positions: self.positions[..n * 3].to_vec(),
            scales: self.scales[..n * 3].to_vec(),
            rotations: self.rotations[..n * 4].to_vec(),
            opacities: self.opacities[..n].to_vec(),
            colors: self.colors[..n * 3].to_vec(),
            sh_coeffs: if self.sh_coeffs.is_empty() {
                Vec::new()
            } else {
                self.sh_coeffs[..n * self.sh_coeff_count * 3].to_vec()
            },
            sh_coeff_count: self.sh_coeff_count,
        }
    }
}

/// Deterministic parity scene (no RNG) shared by Rust tests and the Python reference baseline.
pub fn make_parity_scene() -> GaussianScene {
    let count = 18;
    let mut positions = vec![0.0f32; count * 3];
    for i in 0..count {
        positions[i * 3] = -1.0 + 2.0 * (i as f32) / 17.0;
        positions[i * 3 + 1] = ((i as f32) * 0.17).sin() * 0.8;
        positions[i * 3 + 2] = -1.0 + 2.0 * (i as f32) / 17.0;
    }
    let log_scale = (0.04f32).ln();
    let scales = vec![log_scale; count * 3];
    let mut rotations = vec![0.0f32; count * 4];
    for i in 0..count {
        rotations[i * 4] = 1.0;
    }
    let opacities: Vec<f32> = (0..count)
        .map(|i| 0.25 + 0.5 * (i as f32) / 17.0)
        .collect();
    let colors: Vec<f32> = (0..count)
        .flat_map(|i| {
            let t = i as f32 / 17.0;
            [0.2 + 0.7 * t, 0.15 + 0.6 * (1.0 - t), 0.1 + 0.5 * t]
        })
        .collect();
    let sh_coeffs = vec![0.0f32; count * 3];
    GaussianScene::new(
        positions,
        scales,
        rotations,
        opacities,
        colors,
        sh_coeffs,
        1,
    )
}

/// Deterministic test scene matching `tests/test_renderer_pipeline.py::make_scene`.
pub fn make_scene(count: usize, seed: u64) -> GaussianScene {
    let mut rng = LcgRng::new(seed);
    let mut positions = vec![0.0f32; count * 3];
    for i in 0..count {
        positions[i * 3] = rng.uniform(-1.2, 1.2);
        positions[i * 3 + 1] = rng.uniform(-0.9, 0.9);
        positions[i * 3 + 2] = if count <= 1 {
            0.0
        } else {
            -1.0 + 2.0 * (i as f32) / ((count - 1) as f32)
        };
    }
    let log_scale = (0.04f32).ln();
    let scales = vec![log_scale; count * 3];
    let mut rotations = vec![0.0f32; count * 4];
    for i in 0..count {
        rotations[i * 4] = 1.0;
    }
    let opacities: Vec<f32> = (0..count)
        .map(|i| 0.25 + 0.5 * (i as f32) / ((count.saturating_sub(1)).max(1) as f32))
        .collect();
    let mut colors = vec![0.0f32; count * 3];
    for i in 0..count {
        colors[i * 3] = rng.uniform(0.0, 1.0);
        colors[i * 3 + 1] = rng.uniform(0.0, 1.0);
        colors[i * 3 + 2] = rng.uniform(0.0, 1.0);
    }
    let sh_coeffs = vec![0.0f32; count * 1 * 3];
    GaussianScene::new(
        positions,
        scales,
        rotations,
        opacities,
        colors,
        sh_coeffs,
        1,
    )
}

struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = self.next_u32() as f32 / u32::MAX as f32;
        lo + (hi - lo) * u
    }
}

pub fn supported_sh_coeff_count() -> usize {
    SUPPORTED_SH_COEFF_COUNT
}
