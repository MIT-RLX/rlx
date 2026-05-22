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
//! Viewer debug render modes (aligned with Python `DEBUG_MODE_*`).

pub const DEBUG_MODE_NORMAL: u32 = 0;
pub const DEBUG_MODE_GRAD_NORM: u32 = 1;
pub const DEBUG_MODE_SPLAT_VISIBLE: u32 = 2;
pub const DEBUG_MODE_SPLAT_VISIBLE_AREA: u32 = 3;
pub const DEBUG_MODE_DEPTH: u32 = 4;

pub const DEBUG_MODE_LABELS: &[(&str, u32)] = &[
    ("Normal", DEBUG_MODE_NORMAL),
    ("Grad norm", DEBUG_MODE_GRAD_NORM),
    ("Splat visible", DEBUG_MODE_SPLAT_VISIBLE),
    ("Visible area", DEBUG_MODE_SPLAT_VISIBLE_AREA),
    ("Depth", DEBUG_MODE_DEPTH),
];
