// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `matmul_wide_vec4` must be BIT-IDENTICAL to `matmul_wide`.
//!
//! The vectorised kernel changes how tiles are stored and read, not the order
//! in which products are summed: k still advances one at a time within a tile
//! and tiles still advance in order. So the floating-point result is not merely
//! close, it is the same — and that is the gate, because a tolerance is exactly
//! what let a transposed cooperative-matrix kernel ship in this crate before.
//!
//! Both kernels are dispatched directly here rather than through
//! `derive_matmul_compute`, so the test does not depend on which one the
//! selector currently prefers.

use wgpu::util::DeviceExt;

const RAGGED: &[(u32, u32, u32)] = &[
    (64, 64, 64),    // exactly one tile
    (128, 256, 192), // several whole tiles
    (65, 17, 67),    // ragged on every axis
    (33, 8, 200),    // M and K below one tile
    (192, 33, 64),   // ragged K only
    (64, 64, 3),     // N far below the 64-wide tile
];

struct Case {
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
    bias: bool,
    act: u32,
}

fn params_words(c: &Case) -> [u32; 16] {
    let (m, k, n, batch) = (c.m, c.k, c.n, c.batch);
    // arena: A | B | bias | C
    let a_off = 0;
    let b_off = batch * m * k;
    let bias_off = b_off + batch * k * n;
    let c_off = bias_off + n;
    [
        m,
        k,
        n,
        a_off,
        b_off,
        c_off,
        batch,
        m * k,
        k * n,
        m * n,
        u32::from(c.bias),
        bias_off,
        c.act,
        0,
        0,
        0,
    ]
}

fn arena_data(c: &Case) -> Vec<f32> {
    let words = params_words(c);
    let (b_off, bias_off, c_off) = (words[4] as usize, words[11] as usize, words[5] as usize);
    let mut v = vec![0.0f32; c_off + (c.batch * c.m * c.n) as usize];
    // Deterministic, position-dependent, and non-commuting.
    for i in 0..b_off {
        v[i] = ((i % 23) as f32 - 11.0) * 0.031;
    }
    for i in b_off..bias_off {
        v[i] = ((i % 19) as f32 - 9.0) * 0.017;
    }
    for i in bias_off..c_off {
        v[i] = ((i % 7) as f32 - 3.0) * 0.05;
    }
    v
}

fn run_kernel(
    dev: &rlx_wgpu::device::WgpuDevice,
    kernel: &rlx_wgpu::kernels::Kernel,
    c: &Case,
    grid: (u32, u32),
) -> Vec<f32> {
    let data = arena_data(c);
    let words = params_words(c);
    let c_off = words[5] as usize;
    let out_len = (c.batch * c.m * c.n) as usize;

    let arena = dev
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vec4 parity arena"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let ubo = dev
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vec4 parity params"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vec4 parity"),
        layout: &kernel.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: arena.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ubo.as_entire_binding(),
            },
        ],
    });
    let readback = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vec4 parity readback"),
        size: (out_len * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(grid.0, grid.1, c.batch);
    }
    enc.copy_buffer_to_buffer(
        &arena,
        (c_off * 4) as u64,
        &readback,
        0,
        (out_len * 4) as u64,
    );
    dev.queue.submit(Some(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let view = slice.get_mapped_range().unwrap();
    let out: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    readback.unmap();
    out
}

#[test]
fn vec4_wide_matmul_is_bit_identical_to_the_scalar_kernel() {
    let Some(dev) = rlx_wgpu::device::wgpu_device() else {
        eprintln!("no wgpu adapter, skipping");
        return;
    };
    let scalar = rlx_wgpu::kernels::matmul_wide_kernel(&dev.device);
    let vec4 = rlx_wgpu::kernels::matmul_wide_vec4_kernel(&dev.device);

    let mut cases = Vec::new();
    for &(m, k, n) in RAGGED {
        cases.push(Case {
            m,
            k,
            n,
            batch: 1,
            bias: false,
            act: 0xFFFF,
        });
    }
    // bias, a few activations, and batch > 1 — the epilogue and the batch
    // strides are as easy to get wrong as the inner loop.
    cases.push(Case {
        m: 128,
        k: 64,
        n: 128,
        batch: 1,
        bias: true,
        act: 0xFFFF,
    });
    cases.push(Case {
        m: 128,
        k: 64,
        n: 128,
        batch: 1,
        bias: true,
        act: 0,
    }); // relu
    cases.push(Case {
        m: 96,
        k: 48,
        n: 96,
        batch: 1,
        bias: true,
        act: 9,
    }); // gelu(erf)
    cases.push(Case {
        m: 96,
        k: 48,
        n: 96,
        batch: 1,
        bias: false,
        act: 2,
    }); // tanh
    cases.push(Case {
        m: 64,
        k: 32,
        n: 64,
        batch: 3,
        bias: true,
        act: 0,
    });
    cases.push(Case {
        m: 67,
        k: 33,
        n: 65,
        batch: 2,
        bias: true,
        act: 9,
    });

    let mut checked = 0;
    for c in &cases {
        let s = run_kernel(dev, scalar, c, (c.n.div_ceil(64), c.m.div_ceil(32)));
        let v = run_kernel(dev, vec4, c, (c.n.div_ceil(64), c.m.div_ceil(64)));
        assert_eq!(s.len(), v.len());
        let mismatches = s
            .iter()
            .zip(&v)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches,
            0,
            "matmul_wide_vec4 differs from matmul_wide at m={} k={} n={} batch={} \
             bias={} act={:#x}: {mismatches}/{} elements differ. The vectorised \
             kernel keeps the same summation order, so any difference is a bug, \
             not rounding.",
            c.m,
            c.k,
            c.n,
            c.batch,
            c.bias,
            c.act,
            s.len()
        );
        // A kernel that writes nothing would also "match" if the scalar one did;
        // make sure the case actually produced values.
        assert!(
            v.iter().any(|x| *x != 0.0),
            "case m={} k={} n={} produced an all-zero output — the probe is vacuous",
            c.m,
            c.k,
            c.n
        );
        checked += 1;
    }
    eprintln!("{checked} shapes bit-identical between matmul_wide and matmul_wide_vec4");
}
