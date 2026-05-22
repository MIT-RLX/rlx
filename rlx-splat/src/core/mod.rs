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
pub mod camera;
pub mod constants;
pub mod debug_buffers;
pub mod debug_modes;
pub mod math;
pub mod scene;
pub mod sh;

pub use camera::Camera;
pub use constants::*;
pub use debug_buffers::{ProjectionDebugBuffers, RASTER_CACHE_PARAM_COUNT};
pub use debug_modes::*;
pub use math::*;
pub use scene::{GaussianScene, make_parity_scene, make_scene};
pub use sh::*;
