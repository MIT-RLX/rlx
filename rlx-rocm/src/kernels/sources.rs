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

//! HIP kernel source strings (shared CUDA/HIP `.cu` + AMD-only MFMA).

pub use rlx_gpu_kernels::*;

pub use rlx_gpu_kernels::rocm::MATMUL_MFMA_CU;

/// HIP entry points compiled from shared sources (+ `matmul_mfma`).
pub const KERNEL_COUNT: usize = 33;
