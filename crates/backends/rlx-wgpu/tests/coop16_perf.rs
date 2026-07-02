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

//! Micro-bench: time matmul_coop16 vs matmul.wgsl on a BERT-shape matmul.
//! Answers the question "is coop_mat8x8 actually faster?"

#![cfg(target_os = "macos")]

use std::time::Instant;

#[test]
fn coop16_vs_f32_matmul_perf() {
    let dev = match rlx_wgpu::device::wgpu_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu adapter, skipping");
            return;
        }
    };
    let feats = dev.device.features();
    if !feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        eprintln!("no coop matrix support, skipping");
        return;
    }

    // BERT-class shape: a single layer's FFN1 at batch=8 seq=128
    // gives M=1024, K=384, N=1536 — well-aligned for 8×8 / 32×32 tiles
    // and large enough that kernel time dominates over dispatch overhead.
    let m: u32 = 1024;
    let k: u32 = 384;
    let n: u32 = 1536;
    eprintln!("shape: M={m} K={k} N={n}");

    // Buffers (zero-filled — values don't matter for timing).
    let a_f32_bytes = (m * k * 4) as wgpu::BufferAddress;
    let b_f32_bytes = (k * n * 4) as wgpu::BufferAddress;
    let c_f32_bytes = (m * n * 4) as wgpu::BufferAddress;

    // Single arena holds A (offset 0..m*k), C (offset m*k..m*k+m*n).
    let arena_bytes = a_f32_bytes + c_f32_bytes;
    let arena = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("perf arena"),
        size: arena_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    // f32 weights for matmul / matmul_f16w fallback.
    let weights_f32 = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("perf weights f32"),
        size: b_f32_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let weights_f32_offset_arena = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("perf weights f32 in arena"),
        size: arena_bytes + b_f32_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    // f16 weights / f16 activations for coop16.
    let _ = (weights_f32, weights_f32_offset_arena);
    let b_f16_bytes = (k * n * 2) as wgpu::BufferAddress;
    let a_f16_bytes = (m * k * 2) as wgpu::BufferAddress;
    let weights_f16 = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("perf weights f16"),
        size: b_f16_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let arena_f16 = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("perf arena f16"),
        size: a_f16_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // ── Time matmul.wgsl (f32 baseline) ──
    {
        // Build a self-contained arena with B in it.
        let arena_b_bytes = a_f32_bytes + b_f32_bytes + c_f32_bytes;
        let arena_b = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perf arena_b"),
            size: arena_b_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            m: u32,
            k: u32,
            n: u32,
            a_off: u32,
            b_off: u32,
            c_off: u32,
            batch: u32,
            a_bs: u32,
            b_bs: u32,
            c_bs: u32,
            has_bias: u32,
            bias_off: u32,
            act_id: u32,
            _0: u32,
            _1: u32,
            _2: u32,
        }
        let p = P {
            m,
            k,
            n,
            a_off: 0,
            b_off: m * k,
            c_off: m * k + k * n,
            batch: 1,
            a_bs: 0,
            b_bs: 0,
            c_bs: 0,
            has_bias: 0,
            bias_off: 0,
            act_id: 0xFFFF,
            _0: 0,
            _1: 0,
            _2: 0,
        };
        let pbuf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perf params"),
            size: std::mem::size_of::<P>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&pbuf, 0, bytemuck::bytes_of(&p));

        let kernel = rlx_wgpu::kernels::matmul_kernel(&dev.device);
        let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: arena_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pbuf.as_entire_binding(),
                },
            ],
        });
        time_dispatches(
            "matmul (f32)",
            dev,
            &kernel.pipeline,
            &bg,
            n.div_ceil(32),
            m.div_ceil(32),
            1,
        );
    }

    // ── Time matmul_coop16 (3-binding, A staged from f32 arena) ──
    {
        // arena holds A (offset 0) + C (offset m*k). Kernel stages A
        // through workgroup-shared memory internally.
        let arena_ac_bytes = a_f32_bytes + c_f32_bytes;
        let arena_ac = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perf coop arena AC"),
            size: arena_ac_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let _ = (arena, arena_f16.as_entire_binding());

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            m: u32,
            k: u32,
            n: u32,
            a_off: u32,
            b_off: u32,
            c_off: u32,
            batch: u32,
            a_bs: u32,
            b_bs: u32,
            c_bs: u32,
            has_bias: u32,
            bias_off: u32,
            act_id: u32,
            _0: u32,
            _1: u32,
            _2: u32,
        }
        let p = P {
            m,
            k,
            n,
            a_off: 0,
            b_off: 0,
            c_off: m * k,
            batch: 1,
            a_bs: 0,
            b_bs: 0,
            c_bs: 0,
            has_bias: 0,
            bias_off: 0,
            act_id: 0xFFFF,
            _0: 0,
            _1: 0,
            _2: 0,
        };
        let pbuf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perf coop params"),
            size: std::mem::size_of::<P>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&pbuf, 0, bytemuck::bytes_of(&p));

        let kernel =
            rlx_wgpu::kernels::matmul_coop16_kernel(&dev.device).expect("coop matrix supported");
        let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: arena_ac.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pbuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weights_f16.as_entire_binding(),
                },
            ],
        });
        // 8×8 output tile per workgroup.
        time_dispatches(
            "matmul_coop16 (f16)",
            dev,
            &kernel.pipeline,
            &bg,
            n / 32,
            m / 32,
            1,
        );
    }
}

fn time_dispatches(
    label: &str,
    dev: &rlx_wgpu::device::WgpuDevice,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    gx: u32,
    gy: u32,
    gz: u32,
) {
    // Warmup: 5 dispatches.
    for _ in 0..5 {
        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(gx, gy, gz);
        }
        dev.queue.submit(std::iter::once(enc.finish()));
    }
    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

    // Time 50 dispatches in a single submit (so submit overhead is amortized).
    let n_iters = 50u32;
    let mut enc = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        for _ in 0..n_iters {
            pass.dispatch_workgroups(gx, gy, gz);
        }
    }
    let t0 = Instant::now();
    dev.queue.submit(std::iter::once(enc.finish()));
    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
    let elapsed_us = t0.elapsed().as_micros();
    eprintln!(
        "{label}: {n_iters} dispatches in {elapsed_us}µs ({:.2}µs each, gx={gx} gy={gy})",
        elapsed_us as f64 / n_iters as f64
    );
}
