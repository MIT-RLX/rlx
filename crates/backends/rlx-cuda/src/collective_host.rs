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

//! Host-side `Op::Custom("collective.*")` for CUDA arenas.
//!
//! Thin adapter over [`rlx_gpu_host::run_collective_f32`]. With
//! `--features nccl` and a registered [`crate::distributed`] communicator,
//! `collective.all_reduce` / `collective.all_to_all` prefer the on-device
//! NCCL path (no host round-trip). `moe_dispatch` / `moe_combine` stay on the
//! host variable-`all_to_all_v` path until a device-resident variant lands.

use crate::host_stage::CudaArena;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

pub use rlx_gpu_host::COLLECTIVE_OPS;

#[allow(clippy::too_many_arguments)]
pub fn run_collective(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    _arena_size_bytes: usize,
    name: &str,
    in_off: usize,
    in_len: usize,
    out_off: usize,
    out_len: usize,
    attrs: &[u8],
) {
    #[cfg(feature = "nccl")]
    {
        if try_nccl(
            stream, buffer, name, in_off, in_len, out_off, out_len, attrs,
        ) {
            return;
        }
    }
    let _ = stream;
    let mut arena = CudaArena {
        stream,
        buffer,
        size_bytes: 0,
    };
    rlx_gpu_host::run_collective_f32(&mut arena, name, in_off, in_len, out_off, out_len, attrs);
}

#[cfg(feature = "nccl")]
#[allow(clippy::too_many_arguments)]
fn try_nccl(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    name: &str,
    in_off: usize,
    in_len: usize,
    out_off: usize,
    out_len: usize,
    attrs: &[u8],
) -> bool {
    use crate::distributed;

    match name {
        "collective.all_reduce" | "collective.reduce_from_parallel" => {
            match distributed::try_all_reduce_f32(buffer, in_off, in_len, attrs) {
                Ok(true) => {
                    if out_off != in_off || out_len != in_len {
                        let n = out_len.min(in_len);
                        let Ok(mut tmp) = stream.alloc_zeros::<f32>(n) else {
                            return false;
                        };
                        if stream
                            .memcpy_dtod(&buffer.slice(in_off..in_off + n), &mut tmp)
                            .is_err()
                        {
                            return false;
                        }
                        if stream
                            .memcpy_dtod(&tmp, &mut buffer.slice_mut(out_off..out_off + n))
                            .is_err()
                        {
                            return false;
                        }
                    }
                    true
                }
                Ok(false) => false,
                Err(e) => {
                    eprintln!("rlx-cuda: NCCL all_reduce failed ({e}); falling back to host");
                    false
                }
            }
        }
        "collective.all_to_all" => {
            if in_len != out_len || in_off == out_off {
                return false;
            }
            let Ok(mut send_tmp) = stream.alloc_zeros::<f32>(in_len) else {
                return false;
            };
            if stream
                .memcpy_dtod(&buffer.slice(in_off..in_off + in_len), &mut send_tmp)
                .is_err()
            {
                return false;
            }
            let Ok(mut recv_tmp) = stream.alloc_zeros::<f32>(out_len) else {
                return false;
            };
            match distributed::try_all_to_all_equal_f32(&send_tmp, &mut recv_tmp, attrs) {
                Ok(true) => stream
                    .memcpy_dtod(&recv_tmp, &mut buffer.slice_mut(out_off..out_off + out_len))
                    .is_ok(),
                Ok(false) => false,
                Err(e) => {
                    eprintln!("rlx-cuda: NCCL all_to_all failed ({e}); falling back to host");
                    false
                }
            }
        }
        _ => false,
    }
}
