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

//! Validates matmul_f16w.wgsl: matmul where the weight matrix B is
//! stored as f16. Confirms (1) the kernel compiles + dispatches under
//! the SHADER_F16 feature, (2) numerical output matches a CPU f32×f32
//! matmul reference within f16 quantization tolerance.

#![cfg(target_os = "macos")]

use bytemuck::cast_slice;
use half::f16;

#[test]
fn matmul_f16w_matches_cpu_reference_within_f16_tolerance() {
    let dev = match rlx_wgpu::device::wgpu_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu adapter, skipping");
            return;
        }
    };
    if !dev.device.features().contains(wgpu::Features::SHADER_F16) {
        eprintln!("device lacks SHADER_F16, skipping");
        return;
    }

    let m: u32 = 64;
    let k: u32 = 128;
    let n: u32 = 96;

    // Generate deterministic A (f32) and W (f16-storable f32).
    let a: Vec<f32> = (0..(m * k))
        .map(|i| ((i % 17) as f32) * 0.1 - 0.7)
        .collect();
    let w_f32: Vec<f32> = (0..(k * n))
        .map(|i| ((i % 13) as f32) * 0.05 - 0.3)
        .collect();
    let w_f16: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();
    // CPU reference uses the f16-rounded weights so the tolerance is just
    // the dot-product accumulation drift, not the f16 quantization itself.
    let w_back: Vec<f32> = w_f16.iter().map(|h| h.to_f32()).collect();
    let mut c_ref = vec![0.0f32; (m * n) as usize];
    for mi in 0..m as usize {
        for ni in 0..n as usize {
            let mut s = 0.0_f32;
            for ki in 0..k as usize {
                s += a[mi * k as usize + ki] * w_back[ki * n as usize + ni];
            }
            c_ref[mi * n as usize + ni] = s;
        }
    }

    // Lay out arena: A at offset 0, C at offset m*k. Bias unused.
    let arena_elems = (m * k + m * n) as usize;
    let arena_bytes = (arena_elems * 4) as wgpu::BufferAddress;
    let arena = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test arena"),
        size: arena_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    dev.queue.write_buffer(&arena, 0, cast_slice(&a));

    // Weight buffer: f16 array. half::f16 isn't bytemuck::Pod here, so cast manually.
    let w_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(w_f16.as_ptr() as *const u8, w_f16.len() * 2) };
    let weights = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test weights f16"),
        size: w_bytes.len() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    dev.queue.write_buffer(&weights, 0, w_bytes);

    // Params (matches MatmulParams f32 layout).
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
    let params_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test params"),
        size: std::mem::size_of::<P>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    dev.queue
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&p));

    let kernel =
        rlx_wgpu::kernels::matmul_f16w_kernel(&dev.device).expect("SHADER_F16 was checked");
    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test bg"),
        layout: &kernel.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: arena.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: weights.as_entire_binding(),
            },
        ],
    });

    let mut enc = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test enc"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("test pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(32), m.div_ceil(32), 1);
    }
    let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test stage"),
        size: (m * n * 4) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    enc.copy_buffer_to_buffer(&arena, (m * k * 4) as u64, &staging, 0, (m * n * 4) as u64);
    dev.queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let c_gpu: Vec<f32> = cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();

    let max_abs = c_ref
        .iter()
        .zip(&c_gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let max_rel = c_ref
        .iter()
        .zip(&c_gpu)
        .map(|(a, b)| {
            if a.abs() < 1e-6 {
                0.0
            } else {
                ((a - b) / a).abs()
            }
        })
        .fold(0f32, f32::max);
    eprintln!("f16-weight matmul max|Δ|={max_abs:.3e}, max_rel={max_rel:.3e}");
    eprintln!("c_ref[0..4] = {:?}", &c_ref[..4]);
    eprintln!("c_gpu[0..4] = {:?}", &c_gpu[..4]);
    assert!(
        max_abs < 1e-2,
        "f16-weight matmul output drifted too far from CPU ref: max|Δ|={max_abs}"
    );
}
