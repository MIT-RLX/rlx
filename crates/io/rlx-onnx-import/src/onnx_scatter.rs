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

//! Compatibility shim: ScatterElements lives in `rlx_cpu::onnx_indexing`.

pub use rlx_cpu::onnx_indexing::SCATTER_ELEMENTS;

/// Register the shared CPU `onnx.ScatterElements` kernel (and sibling indexing ops).
pub fn register_onnx_scatter_elements_kernel() {
    rlx_cpu::onnx_indexing::register_onnx_indexing_kernels();
}
