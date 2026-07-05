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

//! Host-side `Op::Scan` for CUDA device arenas (D2H → CPU → H2D). The scan body
//! is compiled once (`ScanBodyPlan`); each run reads the arena back, loops the
//! body on the CPU via `rlx_cpu::thunk::execute_scan_host`, and writes it back.
//! A schedule containing this step is not CUDA-Graph-capture-safe (it forces
//! eager execution), like the other host-fallback recurrences.

use cudarc::driver::{CudaSlice, CudaStream};
use rlx_cpu::thunk::ScanBodyPlan;
use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
pub fn run_scan(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    plan: &ScanBodyPlan,
    outer_init_off: usize,
    outer_final_off: usize,
    length: u32,
    save_trajectory: bool,
    xs_outer: &[(usize, usize)],
    bcast_outer: &[(usize, usize)],
) {
    let n_f32 = arena_size_bytes / 4;
    stream
        .synchronize()
        .expect("rlx-cuda: scan pre-sync failed");

    let mut host = vec![0f32; n_f32];
    stream
        .memcpy_dtoh(&buffer.slice(..), &mut host)
        .expect("rlx-cuda: scan arena dtoh failed");

    unsafe {
        rlx_cpu::thunk::execute_scan_host(
            host.as_mut_ptr() as *mut u8,
            plan,
            outer_init_off,
            outer_final_off,
            length,
            save_trajectory,
            xs_outer,
            bcast_outer,
        );
    }

    stream
        .memcpy_htod(&host, &mut buffer.slice_mut(..))
        .expect("rlx-cuda: scan arena htod failed");
}
