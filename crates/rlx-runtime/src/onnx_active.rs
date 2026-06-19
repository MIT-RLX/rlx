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

//! Per-infer active token count — forwarded to CPU ONNX control-flow kernels.

/// Hint the active padded token count for this infer.
pub fn set_active_token_count(count: Option<usize>) {
    rlx_cpu::onnx_control_flow::set_active_token_count(count);
}

/// Active token count when set by [`set_active_token_count`] or [`crate::CompiledGraph::set_active_extent`].
pub fn active_token_count() -> Option<usize> {
    rlx_cpu::onnx_control_flow::active_token_count()
}
