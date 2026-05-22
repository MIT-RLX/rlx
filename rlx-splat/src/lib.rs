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

//! 3D Gaussian splatting for RLX — scene types, reference pipeline, IR helpers, CPU execution.
//!
//! ## Architecture
//!
//! - [`core`] — scene, camera, SH math (vendored, RLX-owned)
//! - [`reference`] — CPU project → bin → sort → raster
//! - [`Op::GaussianSplatRender`] in `rlx-ir` — logical kernel; backends use CPU reference or
//!   **common IR** (`rlx_ir::logical_kernel::splat_common`) built from primitive ops
//! - [`pipeline`] — explicit stage ops (`GaussianSplatProject`, `BinSort`, `Rasterize`) for strict IR graphs
//!
//! Call [`register()`] once per process to wire CPU executors into `rlx-cpu`.

pub mod core;
pub mod prep_layout;
pub mod reference;
pub mod shaders;

#[cfg(feature = "metal")]
pub mod backends;

#[cfg(feature = "io")]
pub mod io_format;

#[cfg(feature = "execute")]
pub mod cpu_exec;

#[cfg(feature = "cpu")]
pub mod logical_kernel;

#[cfg(feature = "cpu")]
pub mod ops;

#[cfg(feature = "cpu")]
pub mod pipeline;

#[cfg(feature = "cpu")]
pub mod graph;

#[cfg(feature = "io")]
pub mod io;

pub mod parity;
pub mod parity_config;

pub use parity::*;
pub use parity_config::{parity_camera, parity_tiny_render_params, PARITY_BACKGROUND};
pub use core::{make_parity_scene, make_scene};

#[cfg(feature = "cpu")]
pub use ops::*;

#[cfg(feature = "cpu")]
pub use pipeline::*;

#[cfg(feature = "cpu")]
pub use graph::*;

#[cfg(feature = "io")]
pub use io::*;

/// Register splat CPU executors and legacy custom-op aliases. Call once per process.
pub fn register() {
    #[cfg(feature = "execute")]
    {
        use rlx_cpu::splat::{
            ArenaPrepareArgs, ArenaRasterizeArgs, ArenaRenderArgs, ArenaRenderBwdArgs,
            HostBackwardArgs, HostRenderArgs,
        };
        rlx_cpu::splat::register_splat_executors(
            Box::new(|a: ArenaRenderArgs| unsafe { cpu_exec::execute_gaussian_splat_render(a) }),
            Box::new(|a: ArenaRenderBwdArgs| unsafe {
                cpu_exec::execute_gaussian_splat_render_backward(a)
            }),
            Box::new(|a: ArenaPrepareArgs| unsafe { cpu_exec::execute_gaussian_splat_prepare(a) }),
            Box::new(|a: ArenaRasterizeArgs| unsafe { cpu_exec::execute_gaussian_splat_rasterize(a) }),
            Box::new(|a: HostRenderArgs| cpu_exec::render_host_slices_args(a)),
            Box::new(|a: HostBackwardArgs| cpu_exec::backward_host_slices_args(a)),
        );
    }
    #[cfg(feature = "cpu")]
    ops::register();
}
