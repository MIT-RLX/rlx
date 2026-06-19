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

//! Cached MPSGraph executables for tanh `gelu_approx`.
//!
//! Custom MSL kernels that pass arena byte offsets via `set_bytes` break
//! past the 4 GiB `setBuffer:offset:` wrap. MPSGraph binds sub-buffer views
//! through `newBufferWithBytesNoCopy` (`mps_tensor_data_from_buffer`) and
//! handles large unified-memory arenas correctly.

use metal::{Buffer, CommandBufferRef, CommandQueueRef};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::mps_graph::{MpsGraph, MpsGraphExecutable, mps_graph_supported};

const MPS_F32: u32 = 0x10000000 | 32;

static GELU_EXEC: Mutex<Option<HashMap<usize, MpsGraphExecutable>>> = Mutex::new(None);

fn build_gelu_executable(len: usize) -> Option<MpsGraphExecutable> {
    let g = MpsGraph::new();
    let x = g.placeholder(&[len], MPS_F32, "x");
    let y = g.gelu_approx(&x);
    g.compile_executable(&[&x], &[vec![len]], &[MPS_F32], &[&y])
}

fn with_gelu_executable<R>(len: usize, f: impl FnOnce(&MpsGraphExecutable) -> R) -> Option<R> {
    if !mps_graph_supported() || len == 0 {
        return None;
    }
    let mut guard = GELU_EXEC.lock().ok()?;
    let cache = guard.get_or_insert_with(HashMap::new);
    if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(len) {
        let exec = build_gelu_executable(len)?;
        e.insert(exec);
    }
    let exec = cache.get(&len)?;
    Some(f(exec))
}

#[allow(clippy::too_many_arguments)]
fn dispatch_gelu<R>(len: usize, f: impl FnOnce(&MpsGraphExecutable, Vec<usize>) -> R) -> Option<R> {
    let shape = vec![len];
    with_gelu_executable(len, |exec| f(exec, shape))
}

/// Out-of-place (or in-place when `src == dst`) tanh GELU chained on `cmd_buf`.
pub fn encode_gelu_approx_out_cmd(
    cmd_buf: &CommandBufferRef,
    arena: &Buffer,
    src: usize,
    dst: usize,
    len: usize,
) {
    let _ = dispatch_gelu(len, |exec, shape| {
        exec.encode_to_command_buffer(
            cmd_buf,
            &[arena],
            &[src],
            std::slice::from_ref(&shape),
            &[MPS_F32],
            &[arena],
            &[dst],
            std::slice::from_ref(&shape),
            &[MPS_F32],
        );
    });
}

/// Standalone synchronous run (tests / fallback).
pub fn encode_gelu_approx_out(
    queue: &CommandQueueRef,
    arena: &Buffer,
    src: usize,
    dst: usize,
    len: usize,
) {
    let _ = dispatch_gelu(len, |exec, shape| {
        exec.run(
            queue,
            &[arena],
            &[src],
            std::slice::from_ref(&shape),
            &[MPS_F32],
            &[arena],
            &[dst],
            std::slice::from_ref(&shape),
            &[MPS_F32],
        );
    });
}
