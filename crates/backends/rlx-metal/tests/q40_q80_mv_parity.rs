//! Fused Q4_0 / Q8_0 GEMV (`q4_0_mv_f32`, `q8_0_mv_f32`) parity vs CPU reference.

#![cfg(target_os = "macos")]

use metal::{Buffer, Device, MTLResourceOptions, MTLSize};
use rlx_ir::quant::QuantScheme;
use rlx_metal::kernels::kernels;

fn read_f16_le(b: &[u8]) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
}

/// Reference dot for one output row — GGML Q4_0 nibble order (lows 0..15, highs 16..31).
#[allow(dead_code)] // kept as an explicit scalar reference next to the fused kernel path
fn q4_0_mv_row_ref(x: &[f32], row_packed: &[u8], k: usize) -> f32 {
    let nblocks = k / 32;
    let mut acc = 0.0f32;
    let mut x_idx = 0usize;
    for b in 0..nblocks {
        let off = b * 18;
        let d = read_f16_le(&row_packed[off..off + 2]);
        let qs = &row_packed[off + 2..off + 18];
        for j in 0..16 {
            acc += x[x_idx] * (d * ((qs[j] & 0x0F) as i32 - 8) as f32);
            x_idx += 1;
        }
        for j in 0..16 {
            acc += x[x_idx] * (d * ((qs[j] >> 4) as i32 - 8) as f32);
            x_idx += 1;
        }
    }
    acc
}

fn run_fused_mv(
    pipeline: &metal::ComputePipelineState,
    scheme: QuantScheme,
    x: &[f32],
    packed: &[u8],
    k: usize,
    n: usize,
    row_ref: Option<fn(&[f32], &[u8], usize) -> f32>,
) {
    let device = Device::system_default().expect("no Metal device");
    let cmd_q = device.new_command_queue();

    let x_bytes = x.len() * 4;
    let w_bytes = packed.len();
    let out_bytes = n * 4;
    let arena_bytes = (x_bytes + w_bytes + out_bytes).div_ceil(16) * 16;
    let arena: Buffer =
        device.new_buffer(arena_bytes as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        let p = arena.contents() as *mut u8;
        std::ptr::write_bytes(p, 0, arena_bytes);
        std::ptr::copy_nonoverlapping(x.as_ptr() as *const u8, p, x_bytes);
        std::ptr::copy_nonoverlapping(packed.as_ptr(), p.add(x_bytes), w_bytes);
    }

    let x_u = 0u64;
    let w_u = x_bytes as u64;
    let dst_u = (x_bytes + w_bytes) as u64;
    let k_u = k as u32;
    let n_u = n as u32;

    let cmd_buf = cmd_q.new_command_buffer();
    let enc = cmd_buf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(&arena), 0);
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    enc.set_bytes(3, 8, &dst_u as *const u64 as *const _);
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    let grid = MTLSize {
        width: n as u64,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: 256.min(n) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let mut out = vec![0.0f32; n];
    unsafe {
        let src = (arena.contents() as *const u8).add(x_bytes + w_bytes) as *const f32;
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
    }

    let nblocks = k / 32;
    let bytes_per_block = match scheme {
        QuantScheme::GgufQ4_0 => 18,
        QuantScheme::GgufQ4_1 => 20,
        QuantScheme::GgufQ8_0 => 34,
        _ => 18,
    };
    let row_bytes = nblocks * bytes_per_block;
    for j in 0..n {
        let row = &packed[j * row_bytes..(j + 1) * row_bytes];
        let expected = if let Some(f) = row_ref {
            f(x, row, k)
        } else {
            let mut cpu_out = vec![0f32; n];
            rlx_cpu::gguf_matmul::gguf_matmul_bt(x, packed, &mut cpu_out, 1, k, n, scheme);
            cpu_out[j]
        };
        assert!(
            (out[j] - expected).abs() <= 1e-4,
            "row {j}: fused={} ref={expected}",
            out[j]
        );
    }
}

#[test]
fn q4_0_mv_matches_gguf_matmul_bt() {
    // Ensure fused GEMV agrees with CPU `gguf_matmul_bt` (GGML nibble order).
    let k = 128usize;
    let n = 32usize;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.07).cos()).collect();
    let w_row: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.11).sin()).collect();
    let packed = rlx_gguf::quantize::quantize_q4_0(&w_row).unwrap();
    run_fused_mv(
        &kernels().q4_0_mv_f32,
        QuantScheme::GgufQ4_0,
        &x,
        &packed,
        k,
        n,
        None, // uses gguf_matmul_bt
    );
}

fn q4_1_mv_row_ref(x: &[f32], row_packed: &[u8], k: usize) -> f32 {
    let nblocks = k / 32;
    let mut acc = 0.0f32;
    let mut x_idx = 0usize;
    for b in 0..nblocks {
        let off = b * 20;
        let d = read_f16_le(&row_packed[off..off + 2]);
        let m = read_f16_le(&row_packed[off + 2..off + 4]);
        let qs = &row_packed[off + 4..off + 20];
        for j in 0..16 {
            acc += x[x_idx] * (d * (qs[j] & 0x0F) as f32 + m);
            x_idx += 1;
        }
        for j in 0..16 {
            acc += x[x_idx] * (d * (qs[j] >> 4) as f32 + m);
            x_idx += 1;
        }
    }
    acc
}

#[test]
fn q4_1_mv_matches_cpu_reference() {
    let k = 64usize;
    let n = 8usize;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.06).sin()).collect();
    let w_row: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.09).cos()).collect();
    let packed = rlx_gguf::quantize::quantize_q4_1(&w_row).unwrap();
    run_fused_mv(
        &kernels().q4_1_mv_f32,
        QuantScheme::GgufQ4_1,
        &x,
        &packed,
        k,
        n,
        Some(q4_1_mv_row_ref),
    );
}

#[test]
fn q8_0_mv_matches_cpu_reference() {
    let k = 64usize;
    let n = 8usize;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.05).sin()).collect();
    let w_row: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.03).cos()).collect();
    let packed = rlx_gguf::quantize::quantize_q8_0(&w_row).unwrap();
    run_fused_mv(
        &kernels().q8_0_mv_f32,
        QuantScheme::GgufQ8_0,
        &x,
        &packed,
        k,
        n,
        None,
    );
}
