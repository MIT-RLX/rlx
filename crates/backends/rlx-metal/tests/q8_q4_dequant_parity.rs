//! Q4_0 / Q8_0 Metal MSL `dequant_gguf` parity vs rlx-gguf CPU reference.

#![cfg(target_os = "macos")]

use metal::{Buffer, Device, MTLResourceOptions, MTLSize};
use rlx_metal::kernels::kernels;

fn scheme_block_elems(scheme_id: u32) -> usize {
    match scheme_id {
        19..=23 => 32,
        _ => 256,
    }
}

fn run_metal_dequant(scheme_id: u32, block_bytes: &[u8], num_blocks: u32) -> Vec<f32> {
    let device = Device::system_default().expect("no Metal device");
    let k = kernels();
    let cmd_q = device.new_command_queue();
    let total_out_elems = num_blocks as usize * scheme_block_elems(scheme_id);
    let weight_bytes = block_bytes.len();
    let dst_byte_off = weight_bytes.div_ceil(16) * 16;
    let arena_bytes = dst_byte_off + total_out_elems * 4;
    let arena: Buffer =
        device.new_buffer(arena_bytes as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        let p = arena.contents() as *mut u8;
        std::ptr::write_bytes(p, 0, arena_bytes);
        std::ptr::copy_nonoverlapping(block_bytes.as_ptr(), p, weight_bytes);
    }

    let cmd_buf = cmd_q.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&k.dequant_gguf);
    enc.set_buffer(0, Some(&arena), 0);
    let w_off_u = 0u64;
    enc.set_bytes(1, 8, &w_off_u as *const u64 as *const _);
    let dst_f32_off = (dst_byte_off / 4) as u64;
    enc.set_bytes(2, 8, &dst_f32_off as *const u64 as *const _);
    enc.set_bytes(3, 4, &scheme_id as *const u32 as *const _);
    enc.set_bytes(4, 4, &num_blocks as *const u32 as *const _);
    enc.set_buffer(5, Some(k.iq_grid_buffer()), 0);
    let grid = MTLSize {
        width: num_blocks as u64,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: num_blocks.clamp(1, 64) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let mut out = vec![0.0f32; total_out_elems];
    unsafe {
        let src = (arena.contents() as *const u8).add(dst_byte_off) as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), total_out_elems);
    }
    out
}

fn parity(scheme_id: u32, packed: &[u8], elems: usize, tol: f32, name: &str) {
    let num_blocks = (elems / scheme_block_elems(scheme_id)) as u32;
    let metal_out = run_metal_dequant(scheme_id, packed, num_blocks);
    let cpu_out = match scheme_id {
        19 => rlx_gguf::dequant_q4_0(packed, elems).unwrap(),
        20 => rlx_gguf::dequant_q8_0(packed, elems).unwrap(),
        21 => rlx_gguf::dequant_q4_1(packed, elems).unwrap(),
        22 => rlx_gguf::dequant_q5_0(packed, elems).unwrap(),
        23 => rlx_gguf::dequant_q5_1(packed, elems).unwrap(),
        _ => panic!("bad scheme_id {scheme_id}"),
    };
    assert_eq!(metal_out.len(), elems);
    let mut worst = 0.0f32;
    for i in 0..elems {
        worst = worst.max((metal_out[i] - cpu_out[i]).abs());
    }
    assert!(worst <= tol, "{name}: worst diff {worst} > {tol}");
}

#[test]
fn q8_0_msl_matches_cpu_reference() {
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
fn q4_0_msl_matches_cpu_reference() {
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
fn q4_1_msl_matches_cpu_reference() {
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
fn q5_0_msl_matches_cpu_reference() {
    let w: Vec<f32> = (0..512).map(|i| (i as f32 * 0.017).cos()).collect();
    let packed = rlx_gguf::quantize::quantize_q5_0(&w).unwrap();
    parity(22, &packed, 512, 1e-4, "Q5_0");
}

#[test]
fn q5_1_msl_matches_cpu_reference() {
    let w: Vec<f32> = (0..512).map(|i| (i as f32 * 0.019).sin()).collect();
    let packed = rlx_gguf::quantize::quantize_q5_1(&w).unwrap();
    parity(23, &packed, 512, 1e-4, "Q5_1");
}

#[test]
fn has_metal_dequant_kernel_q8_q4() {
    use rlx_ir::quant::QuantScheme;
    assert!(rlx_metal::backend::has_metal_dequant_kernel(
        QuantScheme::GgufQ8_0
    ));
    assert!(rlx_metal::backend::has_metal_dequant_kernel(
        QuantScheme::GgufQ4_0
    ));
    assert!(rlx_metal::backend::has_metal_dequant_kernel(
        QuantScheme::GgufQ4_1
    ));
    assert!(rlx_metal::backend::has_metal_dequant_kernel(
        QuantScheme::GgufQ5_0
    ));
}
