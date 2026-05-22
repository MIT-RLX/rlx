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
//! Host-side `Op::GatedDeltaNet` for CUDA device arenas (D2H → CPU → H2D).

use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

pub fn run_gated_delta_net(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
    q_byte_off: usize,
    k_byte_off: usize,
    v_byte_off: usize,
    g_byte_off: usize,
    beta_byte_off: usize,
    state_byte_off: usize,
    dst_byte_off: usize,
    batch: usize,
    seq: usize,
    heads: usize,
    state_size: usize,
    use_carry: bool,
) {
    assert!(
        state_size <= rlx_cpu::gdn::GDN_MAX_STATE,
        "rlx-cuda GatedDeltaNet: state_size {state_size} > {}",
        rlx_cpu::gdn::GDN_MAX_STATE
    );

    let n_f32 = arena_size_bytes / 4;
    stream.synchronize().expect("rlx-cuda: gdn pre-sync failed");

    let mut host = vec![0f32; n_f32];
    stream
        .memcpy_dtoh(&buffer.slice(..), &mut host)
        .expect("rlx-cuda: gdn arena dtoh failed");

    unsafe {
        rlx_cpu::thunk::execute_gated_delta_net_f32(
            q_byte_off,
            k_byte_off,
            v_byte_off,
            g_byte_off,
            beta_byte_off,
            if use_carry { state_byte_off } else { 0 },
            dst_byte_off,
            batch,
            seq,
            heads,
            state_size,
            host.as_mut_ptr() as *mut u8,
        );
    }

    stream
        .memcpy_htod(&host, &mut buffer.slice_mut(..))
        .expect("rlx-cuda: gdn arena htod failed");
}
