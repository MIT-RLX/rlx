// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Host-side `Op::GatedDeltaNet` for wgpu arenas (readback → CPU → writeback).
//!
//! Default path is the native WGSL kernel (`kernels/gated_delta_net.wgsl`).
//! This host fallback is selected with `RLX_WGPU_GDN_HOST=1`, or when the
//! GPU path cannot bind the working set (cross-shard without a viable pack).
//!
//! On sharded arenas the old “mirror `[0, hi)`” approach exceeded both RAM and
//! the 4 GiB stripe size. We instead gather only the six (or seven) tensors
//! into a compact host buffer, run the CPU thunk, and write `dst` (and carry
//! state) back.

use crate::buffer::Arena;

fn align256(n: usize) -> usize {
    n.div_ceil(256) * 256
}

/// Copy one arena tensor into `host` at `dst_off` (bytes).
fn pull(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    host: &mut [u8],
    dst_off: usize,
    src_off: usize,
    len: usize,
) {
    if len == 0 {
        return;
    }
    let chunk = arena.read_bytes_range(device, queue, src_off, len);
    host[dst_off..dst_off + len].copy_from_slice(&chunk);
}

pub fn run_gated_delta_net(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
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
    gate_per_channel: bool,
) {
    assert!(
        state_size <= rlx_cpu::gdn::GDN_MAX_STATE,
        "rlx-wgpu GatedDeltaNet: state_size {state_size} > {}",
        rlx_cpu::gdn::GDN_MAX_STATE
    );

    let qkv = batch * seq * heads * state_size * 4;
    let gate = batch * seq * heads * 4;
    let st = batch * heads * state_size * state_size * 4;

    // Compact layout — offsets are independent of the arena stripe map.
    let mut cur = 0usize;
    let q_h = {
        let o = cur;
        cur = align256(cur + qkv);
        o
    };
    let k_h = {
        let o = cur;
        cur = align256(cur + qkv);
        o
    };
    let v_h = {
        let o = cur;
        cur = align256(cur + qkv);
        o
    };
    let g_h = {
        let o = cur;
        cur = align256(cur + gate);
        o
    };
    let beta_h = {
        let o = cur;
        cur = align256(cur + gate);
        o
    };
    let dst_h = {
        let o = cur;
        cur = align256(cur + qkv);
        o
    };
    let state_h = if use_carry {
        let o = cur;
        cur = align256(cur + st);
        o
    } else {
        0
    };

    let mut host = vec![0u8; cur.max(1)];
    pull(arena, device, queue, &mut host, q_h, q_byte_off, qkv);
    pull(arena, device, queue, &mut host, k_h, k_byte_off, qkv);
    pull(arena, device, queue, &mut host, v_h, v_byte_off, qkv);
    pull(arena, device, queue, &mut host, g_h, g_byte_off, gate);
    pull(arena, device, queue, &mut host, beta_h, beta_byte_off, gate);
    if use_carry {
        pull(arena, device, queue, &mut host, state_h, state_byte_off, st);
    }

    unsafe {
        rlx_cpu::thunk::execute_gated_delta_net_f32(
            q_h,
            k_h,
            v_h,
            g_h,
            beta_h,
            if use_carry { state_h } else { 0 },
            dst_h,
            batch,
            seq,
            heads,
            state_size,
            gate_per_channel,
            host.as_mut_ptr(),
        );
    }

    arena.write_bytes_range(queue, dst_byte_off, &host[dst_h..dst_h + qkv]);
    if use_carry {
        arena.write_bytes_range(queue, state_byte_off, &host[state_h..state_h + st]);
    }
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
}
