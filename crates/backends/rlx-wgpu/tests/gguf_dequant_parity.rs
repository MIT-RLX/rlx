// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Q4_0 / Q8_0 / Q4_1 / Q5_* / IQ4_NL WGPU `dequant_gguf` parity vs rlx-gguf CPU reference.

use rlx_wgpu::kernels::{DequantGgufParams, dequant_gguf_kernel};

fn scheme_block_elems(scheme_id: u32) -> usize {
    rlx_wgpu::gguf_host::scheme_from_id(scheme_id).gguf_block_size() as usize
}

fn cpu_dequant_gguf(scheme_id: u32, packed: &[u8], elems: usize) -> Vec<f32> {
    use rlx_ir::quant::QuantScheme::*;
    match rlx_wgpu::gguf_host::scheme_from_id(scheme_id) {
        GgufQ4_0 => rlx_gguf::dequant_q4_0(packed, elems).unwrap(),
        GgufQ8_0 => rlx_gguf::dequant_q8_0(packed, elems).unwrap(),
        GgufQ4_1 => rlx_gguf::dequant_q4_1(packed, elems).unwrap(),
        GgufQ5_0 => rlx_gguf::dequant_q5_0(packed, elems).unwrap(),
        GgufQ5_1 => rlx_gguf::dequant_q5_1(packed, elems).unwrap(),
        GgufIQ4NL => rlx_gguf::iq_dequant::dequant_iq4_nl(packed, elems).unwrap(),
        GgufIQ4XS => rlx_gguf::iq_dequant::dequant_iq4_xs(packed, elems).unwrap(),
        GgufIQ2XXS => rlx_gguf::iq_dequant::dequant_iq2_xxs(packed, elems).unwrap(),
        GgufIQ2XS => rlx_gguf::iq_dequant::dequant_iq2_xs(packed, elems).unwrap(),
        GgufIQ2S => rlx_gguf::iq_dequant::dequant_iq2_s(packed, elems).unwrap(),
        GgufIQ3XXS => rlx_gguf::iq_dequant::dequant_iq3_xxs(packed, elems).unwrap(),
        GgufIQ3S => rlx_gguf::iq_dequant::dequant_iq3_s(packed, elems).unwrap(),
        GgufIQ1S => rlx_gguf::iq_dequant::dequant_iq1_s(packed, elems).unwrap(),
        GgufIQ1M => rlx_gguf::iq_dequant::dequant_iq1_m(packed, elems).unwrap(),
        GgufTQ1_0 => rlx_gguf::tq_dequant::dequant_tq1_0(packed, elems).unwrap(),
        GgufTQ2_0 => rlx_gguf::tq_dequant::dequant_tq2_0(packed, elems).unwrap(),
        GgufMXFP4 => rlx_gguf::mx_dequant::dequant_mxfp4(packed, elems).unwrap(),
        GgufNVFP4 => rlx_gguf::mx_dequant::dequant_nvfp4(packed, elems).unwrap(),
        other => panic!("cpu_dequant_gguf: unsupported scheme_id {scheme_id} ({other})"),
    }
}

fn run_wgpu_dequant(scheme_id: u32, block_bytes: &[u8], num_blocks: u32) -> Vec<f32> {
    let dev = rlx_wgpu::device::wgpu_device().expect("no wgpu adapter");
    let total_out_elems = num_blocks as usize * scheme_block_elems(scheme_id);
    let weight_bytes = block_bytes.len();
    let dst_byte_off = weight_bytes.div_ceil(16) * 16;
    let arena_bytes = dst_byte_off + total_out_elems * 4;

    let arena = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu gguf dequant parity arena"),
        size: arena_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    dev.queue.write_buffer(&arena, 0, block_bytes);

    let dk = dequant_gguf_kernel(&dev.device);
    let lut = rlx_wgpu::iq_grid::wgpu_iq_grid_buffer(&dev.device, &dev.queue);
    let p = DequantGgufParams {
        w_byte_off: 0,
        dst_f32_off: (dst_byte_off / 4) as u32,
        scheme_id,
        num_blocks,
    };
    let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu dequant parity uniform"),
        size: std::mem::size_of::<DequantGgufParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    dev.queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu dequant parity bg"),
        layout: &dk.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: arena.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: lut.as_entire_binding(),
            },
        ],
    });

    let block = 256u32.min(num_blocks).max(1);
    let grid = num_blocks.div_ceil(block);
    let mut enc = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rlx-wgpu dequant parity"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu dequant parity pass"),
            ..Default::default()
        });
        pass.set_pipeline(&dk.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(grid, 1, 1);
    }
    dev.queue.submit(std::iter::once(enc.finish()));
    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

    let read_len = total_out_elems * 4;
    let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rlx-wgpu dequant parity readback"),
        size: read_len as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut read_enc = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rlx-wgpu dequant parity readback enc"),
        });
    read_enc.copy_buffer_to_buffer(&arena, dst_byte_off as u64, &staging, 0, read_len as u64);
    dev.queue.submit(std::iter::once(read_enc.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = sender.send(r);
    });
    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
    receiver.recv().unwrap().unwrap();
    let view = slice.get_mapped_range().expect("buffer slice mapped");
    let out: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();
    out
}

fn parity(scheme_id: u32, packed: &[u8], elems: usize, tol: f32, name: &str) {
    let num_blocks = (elems / scheme_block_elems(scheme_id)) as u32;
    let gpu_out = run_wgpu_dequant(scheme_id, packed, num_blocks);
    let cpu_out = cpu_dequant_gguf(scheme_id, packed, elems);
    assert_eq!(gpu_out.len(), elems);
    let mut worst = 0.0f32;
    for i in 0..elems {
        worst = worst.max((gpu_out[i] - cpu_out[i]).abs());
    }
    assert!(worst <= tol, "{name}: worst diff {worst} > {tol}");
}

#[test]
fn iq4_nl_wgsl_matches_cpu_reference() {
    let mut packed = Vec::new();
    for b in 0..16u8 {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.07 * (b as f32 + 1.0)).to_le_bytes());
        for i in 0..16u8 {
            block[2 + i as usize] = i.wrapping_add(b);
        }
        packed.extend_from_slice(&block);
    }
    parity(6, &packed, 512, 1e-5, "IQ4_NL");
}

#[test]
fn q8_0_wgsl_matches_cpu_reference() {
    let mut packed = Vec::new();
    for b in 0..16u8 {
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.05 * (b as f32 + 1.0)).to_le_bytes());
        for (i, q) in block[2..].iter_mut().enumerate() {
            *q = ((i as i8).wrapping_mul(3).wrapping_add(b as i8)) as u8;
        }
        packed.extend_from_slice(&block);
    }
    parity(20, &packed, 512, 1e-5, "Q8_0");
}

#[test]
fn q4_0_wgsl_matches_cpu_reference() {
    let mut packed = Vec::new();
    for b in 0..16u8 {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.08 * (b as f32 + 1.0)).to_le_bytes());
        for (i, q) in block[2..].iter_mut().enumerate() {
            let lo = ((i + b as usize) % 16) as u8;
            let hi = ((i + b as usize + 3) % 16) as u8;
            *q = lo | (hi << 4);
        }
        packed.extend_from_slice(&block);
    }
    parity(19, &packed, 512, 1e-5, "Q4_0");
}

#[test]
fn q4_1_wgsl_matches_cpu_reference() {
    let mut packed = Vec::new();
    for b in 0..16u8 {
        let mut block = vec![0u8; 20];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.08 * (b as f32 + 1.0)).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.01 * b as f32).to_le_bytes());
        for (i, q) in block[4..].iter_mut().enumerate() {
            let lo = ((i + b as usize) % 16) as u8;
            let hi = ((i + b as usize + 3) % 16) as u8;
            *q = lo | (hi << 4);
        }
        packed.extend_from_slice(&block);
    }
    parity(21, &packed, 512, 1e-4, "Q4_1");
}

#[test]
fn q5_0_wgsl_matches_cpu_reference() {
    let w: Vec<f32> = (0..512).map(|i| (i as f32 * 0.017).cos()).collect();
    let packed = rlx_gguf::quantize::quantize_q5_0(&w).unwrap();
    parity(22, &packed, 512, 1e-4, "Q5_0");
}

#[test]
fn q5_1_wgsl_matches_cpu_reference() {
    let w: Vec<f32> = (0..512).map(|i| (i as f32 * 0.019).sin()).collect();
    let packed = rlx_gguf::quantize::quantize_q5_1(&w).unwrap();
    parity(23, &packed, 512, 1e-4, "Q5_1");
}

fn zero_block_parity(scheme_id: u32, num_blocks: u32, tol: f32, name: &str) {
    let block_bytes = rlx_wgpu::gguf_host::scheme_from_id(scheme_id).gguf_block_bytes() as usize;
    let packed = vec![0u8; num_blocks as usize * block_bytes];
    let elems = num_blocks as usize * scheme_block_elems(scheme_id);
    parity(scheme_id, &packed, elems, tol, name);
}

#[test]
fn iq_tq_mx_zero_blocks_match_cpu_reference() {
    zero_block_parity(7, 4, 1e-5, "IQ4_XS");
    zero_block_parity(8, 4, 1e-5, "TQ1_0");
    zero_block_parity(9, 4, 1e-5, "TQ2_0");
    zero_block_parity(10, 16, 1e-5, "MXFP4");
    zero_block_parity(11, 32, 1e-5, "NVFP4");
    zero_block_parity(12, 4, 1e-5, "IQ2_XXS");
    zero_block_parity(13, 4, 1e-5, "IQ2_XS");
    zero_block_parity(14, 4, 1e-5, "IQ2_S");
    zero_block_parity(15, 4, 1e-5, "IQ3_XXS");
    zero_block_parity(16, 4, 1e-5, "IQ3_S");
    zero_block_parity(17, 8, 1e-5, "IQ1_S");
    zero_block_parity(18, 8, 1e-4, "IQ1_M");
}

fn encode_parity(scheme_id: u32, ggml: rlx_gguf::GgmlType, elems: usize, tol: f32, name: &str) {
    let w: Vec<f32> = (0..elems).map(|i| (i as f32 * 0.021).sin() * 0.6).collect();
    let packed = rlx_gguf::quantize(&w, ggml).expect("quantize");
    parity(scheme_id, &packed, elems, tol, name);
}

#[test]
fn iq2_xxs_wgsl_matches_cpu_reference() {
    encode_parity(12, rlx_gguf::GgmlType::IQ2XXS, 512, 0.05, "IQ2_XXS");
}

#[test]
fn iq2_xs_wgsl_matches_cpu_reference() {
    encode_parity(13, rlx_gguf::GgmlType::IQ2XS, 512, 0.05, "IQ2_XS");
}

#[test]
fn iq2_s_wgsl_matches_cpu_reference() {
    encode_parity(14, rlx_gguf::GgmlType::IQ2S, 512, 0.05, "IQ2_S");
}

#[test]
fn iq3_xxs_wgsl_matches_cpu_reference() {
    encode_parity(15, rlx_gguf::GgmlType::IQ3XXS, 512, 0.05, "IQ3_XXS");
}

#[test]
fn iq3_s_wgsl_matches_cpu_reference() {
    encode_parity(16, rlx_gguf::GgmlType::IQ3S, 512, 0.05, "IQ3_S");
}

#[test]
fn iq1_s_wgsl_matches_cpu_reference() {
    encode_parity(17, rlx_gguf::GgmlType::IQ1S, 512, 0.15, "IQ1_S");
}

#[test]
fn iq1_m_wgsl_matches_cpu_reference() {
    encode_parity(18, rlx_gguf::GgmlType::IQ1M, 512, 0.15, "IQ1_M");
}

#[test]
fn tq1_0_encode_wgsl_matches_cpu_reference() {
    let w: Vec<f32> = (0..512)
        .map(|i| match i % 3 {
            0 => -0.4,
            1 => 0.0,
            _ => 0.4,
        })
        .collect();
    let packed = rlx_gguf::tq_quantize::quantize_tq1_0(&w).unwrap();
    parity(8, &packed, 512, 1e-5, "TQ1_0 encode");
}

#[test]
fn mxfp4_encode_wgsl_matches_cpu_reference() {
    encode_parity(10, rlx_gguf::GgmlType::MXFP4, 512, 1e-4, "MXFP4 encode");
}

#[test]
fn nvfp4_wgsl_matches_cpu_reference() {
    encode_parity(11, rlx_gguf::GgmlType::NVFP4, 512, 1e-4, "NVFP4");
}

#[test]
fn tq2_0_encode_wgsl_matches_cpu_reference() {
    let w: Vec<f32> = (0..512)
        .map(|i| match i % 3 {
            0 => -0.5,
            1 => 0.0,
            _ => 0.5,
        })
        .collect();
    let packed = rlx_gguf::tq_quantize::quantize_tq2_0(&w).unwrap();
    parity(9, &packed, 512, 1e-5, "TQ2_0 encode");
}

#[test]
fn mxfp4_wgsl_matches_cpu_reference() {
    let mut packed = Vec::new();
    for scale in 0..16u8 {
        let mut block = vec![0u8; 17];
        block[0] = 120 + scale;
        for (i, q) in block[1..].iter_mut().enumerate() {
            *q = ((i * 3 + scale as usize + 1) & 0x0F) as u8
                | ((((i * 5 + scale as usize + 2) & 0x0F) as u8) << 4);
        }
        packed.extend_from_slice(&block);
    }
    parity(10, &packed, 512, 1e-4, "MXFP4");
}

#[test]
fn tq2_0_wgsl_matches_cpu_reference() {
    let trits: [i8; 256] = std::array::from_fn(|i| match i % 3 {
        0 => -1,
        1 => 0,
        _ => 1,
    });
    let mut packed = Vec::new();
    for _ in 0..2 {
        let mut block = vec![0u8; 66];
        block[64..66].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        let mut y = 0usize;
        let mut j = 0usize;
        while j < 64 {
            for l in 0..4 {
                for m in 0..32 {
                    let q = (trits[y] + 1) as u8;
                    block[j + m] |= (q & 3) << (l * 2);
                    y += 1;
                }
            }
            j += 32;
        }
        packed.extend_from_slice(&block);
    }
    parity(9, &packed, 512, 1e-5, "TQ2_0");
}
