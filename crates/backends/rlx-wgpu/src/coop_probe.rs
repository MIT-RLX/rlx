// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runtime self-check for the cooperative-matrix GEMM kernels.
//!
//! # Why a runtime check rather than a test
//!
//! `coopLoad`/`coopLoadT` differ only by naga's
//! `row_major = function_name.ends_with("T")` in the shared WGSL frontend, but
//! that one bool reaches the backends as different things — Metal's
//! `simdgroup_load(..., transpose_matrix)` versus SPIR-V's
//! `RowMajorKHR`/`ColumnMajorKHR` on `OpCooperativeMatrixLoadKHR`. Get it wrong
//! and the kernel computes `b·a`, or `bᵀa`, instead of `a·b`. Those have the
//! same shape and roughly the same magnitude as the right answer, so they do not
//! crash and they survive a magnitude-relative tolerance; they are simply wrong
//! numbers. Three kernels here shipped that way for a long time.
//!
//! A compile-time choice cannot settle it, because the right choice depends on
//! the adapter's backend and on the driver's cooperative-matrix implementation —
//! neither of which is known when the WGSL is written, and neither of which any
//! single development machine can cover. Mesa lavapipe and MoltenVK expose no
//! cooperative-matrix extension at all, so an Apple or software-Vulkan box
//! cannot even reach the Vulkan variants.
//!
//! So: ask the device. Once per process per kernel, run a tiny GEMM whose answer
//! distinguishes every transposition, and believe the result. A kernel that
//! fails is reported ineligible and the caller falls back to the portable tiled
//! path, which is slower and correct.
//!
//! # The probe
//!
//! `A` is a column vector (only `k = 0` populated), `B` is a row vector (only
//! `k = 0` populated), so the true `a·b` is a DENSE rank-1 outer product:
//! `C[i][j] = (i+1)·(j+1)·s`, with every element non-zero. Each wrong answer is
//! structurally distinct, not merely numerically off:
//!
//! | computed | what the output looks like                      |
//! |----------|-------------------------------------------------|
//! | `a·b`    | dense, every element non-zero — correct         |
//! | `b·a`    | one non-zero per fragment, ~98% exact zeros     |
//! | `bᵀa`    | one non-zero column per fragment                |
//!
//! Two dense operands cannot do this job, and neither can an identity: `A·I` and
//! `I·B` both pass under a swapped operand pair because the identity commutes.
//! That is precisely how the original bug survived its own test suite.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::kernels::Kernel;

/// Which cooperative-matrix kernel to check. One entry per distinct shader, so
/// each is verified on its own terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoopKernel {
    /// `matmul_coop_f32` — Metal `simdgroup_float8x8`, 32×32 tile.
    F32Metal,
    /// `matmul_coop_f32_portable` — Vulkan/DX12 8×8 tile.
    F32Portable,
    /// `matmul_coop16` — all-f16 operands and accumulator, 32×32 tile.
    Coop16,
    /// The `matmul_coop_f16_vulkan*` family, 16×16 tile. `widen` selects the
    /// `coopLoad`-on-B shader, `!widen` the `coopLoadT`-on-B shader.
    F16Vk { widen: bool, f32acc: bool },
}

/// Where the probe has to place `A` and `B` for this shader's bind group.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Operands {
    /// bind 0 = f32 arena (A, B and C), bind 1 = params.
    ArenaF32,
    /// bind 0 = f32 arena (A and C), bind 1 = params, bind 2 = f16 weights (B).
    ArenaF32PlusF16Weights,
    /// bind 0 = f16 arena (A and B), bind 1 = f32 arena (C), bind 2 = params.
    ArenaF16PlusF32Out,
}

struct Plan {
    m: u32,
    k: u32,
    n: u32,
    /// `dispatch_workgroups` x/y, matching the call site in `backend/run.rs`.
    grid: (u32, u32),
    operands: Operands,
    /// Relative tolerance. f16 operands carry real quantization error; the
    /// wrong answers this is screening for are off by ~100%, so a loose bound
    /// still separates them cleanly.
    rel_tol: f32,
}

fn plan(which: CoopKernel) -> Plan {
    match which {
        // 32×32 output tile, TILE_K = 8; grid is (n/32, m/32).
        CoopKernel::F32Metal => Plan {
            m: 32,
            k: 8,
            n: 32,
            grid: (1, 1),
            operands: Operands::ArenaF32,
            rel_tol: 1e-4,
        },
        // 8×8 tile; grid is (m/8, n/8) — note the axes are swapped relative to
        // the Metal kernel, exactly as `backend/run.rs` dispatches them.
        CoopKernel::F32Portable => Plan {
            m: 32,
            k: 8,
            n: 32,
            grid: (4, 4),
            operands: Operands::ArenaF32,
            rel_tol: 1e-4,
        },
        CoopKernel::Coop16 => Plan {
            m: 32,
            k: 8,
            n: 32,
            grid: (1, 1),
            operands: Operands::ArenaF32PlusF16Weights,
            rel_tol: 2e-2,
        },
        // 16×16 tile; grid is (m/16, n/16).
        CoopKernel::F16Vk { .. } => Plan {
            m: 32,
            k: 16,
            n: 32,
            grid: (2, 2),
            operands: Operands::ArenaF16PlusF32Out,
            rel_tol: 2e-2,
        },
    }
}

/// `Params` as every matmul shader in this crate declares it: 16 × `u32`.
fn params_bytes(m: u32, k: u32, n: u32, a_off: u32, b_off: u32, c_off: u32) -> [u8; 64] {
    let p: [u32; 16] = [
        m,
        k,
        n,
        a_off,
        b_off,
        c_off,
        1, // batch
        m * k,
        k * n,
        m * n, // batch strides
        0,
        0,      // has_bias, bias_off
        0xFFFF, // act_id = none
        0,
        0,
        0,
    ];
    let mut out = [0u8; 64];
    for (dst, v) in out.chunks_exact_mut(4).zip(p) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// Scale keeps every product inside f16's comfortable range: the largest
/// `C[i][j]` is `m · n · s` = 16 at the sizes above.
const S: f32 = 1.0 / 64.0;

/// `A[i][0] = i + 1`, everything else zero.
fn probe_a(m: u32, k: u32) -> Vec<f32> {
    let mut a = vec![0.0; (m * k) as usize];
    for (i, row) in a.chunks_mut(k as usize).enumerate() {
        row[0] = i as f32 + 1.0;
    }
    a
}

/// `B[0][j] = (j + 1) · S`, everything else zero.
fn probe_b(k: u32, n: u32) -> Vec<f32> {
    let mut b = vec![0.0; (k * n) as usize];
    for (j, v) in b[..n as usize].iter_mut().enumerate() {
        *v = (j as f32 + 1.0) * S;
    }
    b
}

fn f16_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &v in data {
        out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    out
}

/// Has `which` been shown to compute `a·b` on this device? Memoized: the probe
/// runs at most once per kernel per process.
pub fn verified(device: &wgpu::Device, queue: &wgpu::Queue, which: CoopKernel) -> bool {
    static CACHE: OnceLock<Mutex<HashMap<CoopKernel, bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&hit) = cache.lock().unwrap().get(&which) {
        return hit;
    }
    // `RLX_WGPU_NO_COOP_PROBE=1` skips the check and trusts the kernels. Only
    // useful for measuring the probe's own cost, or to re-observe a known-bad
    // kernel while debugging it.
    let ok = if rlx_ir::env::flag("RLX_WGPU_NO_COOP_PROBE") {
        true
    } else {
        run(device, queue, which).unwrap_or(false)
    };
    if !ok {
        eprintln!(
            "rlx-wgpu: cooperative-matrix kernel {which:?} FAILED its self-check \
             (it does not compute a·b on this device) — falling back to the \
             portable tiled matmul. See src/coop_probe.rs."
        );
    }
    cache.lock().unwrap().insert(which, ok);
    ok
}

fn kernel_for(device: &wgpu::Device, which: CoopKernel) -> Option<&'static Kernel> {
    use crate::kernels as k;
    match which {
        CoopKernel::F32Metal => k::matmul_coop_f32_kernel(device),
        CoopKernel::F32Portable => k::matmul_coop_f32_portable_kernel(device),
        CoopKernel::Coop16 => k::matmul_coop16_kernel(device),
        CoopKernel::F16Vk {
            widen: false,
            f32acc: false,
        } => k::matmul_coop_f16_vulkan_kernel(device),
        CoopKernel::F16Vk {
            widen: false,
            f32acc: true,
        } => k::matmul_coop_f16_vulkan_f32acc_kernel(device),
        CoopKernel::F16Vk {
            widen: true,
            f32acc: false,
        } => k::matmul_coop_f16_vulkan_widen_kernel(device),
        CoopKernel::F16Vk {
            widen: true,
            f32acc: true,
        } => k::matmul_coop_f16_vulkan_widen_f32acc_kernel(device),
    }
}

/// Can this adapter build `which` at all? A kernel that cannot be built is not
/// a failed self-check — it is simply not a path this device will take.
pub fn buildable(device: &wgpu::Device, which: CoopKernel) -> bool {
    kernel_for(device, which).is_some()
}

/// Run the probe against an arbitrary shader rather than the shipped one, using
/// `which`'s geometry and bind layout. Exists so the test suite can prove the
/// probe REJECTS a known-bad kernel; a check that cannot fail is not a check.
pub fn probe_source(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    which: CoopKernel,
    wgsl: &str,
    entry: &str,
) -> Option<bool> {
    let operands = plan(which).operands;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rlx-wgpu coop probe (ad-hoc)"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = probe_bgl(device, operands);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("rlx-wgpu coop probe (ad-hoc)"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("rlx-wgpu coop probe (ad-hoc)"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    run_with(device, queue, which, &pipeline, &bgl)
}

/// The bind group layout each `Operands` shape declares, in binding order.
fn probe_bgl(device: &wgpu::Device, operands: Operands) -> wgpu::BindGroupLayout {
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let uniform = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let tys = match operands {
        Operands::ArenaF32 => vec![storage(false), uniform],
        Operands::ArenaF32PlusF16Weights => vec![storage(false), uniform, storage(true)],
        Operands::ArenaF16PlusF32Out => vec![storage(true), storage(false), uniform],
    };
    let entries: Vec<_> = tys
        .into_iter()
        .enumerate()
        .map(|(i, ty)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rlx-wgpu coop probe"),
        entries: &entries,
    })
}

fn run(device: &wgpu::Device, queue: &wgpu::Queue, which: CoopKernel) -> Option<bool> {
    let kernel = kernel_for(device, which)?;
    run_with(device, queue, which, &kernel.pipeline, &kernel.bgl)
}

fn run_with(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    which: CoopKernel,
    pipeline: &wgpu::ComputePipeline,
    bgl: &wgpu::BindGroupLayout,
) -> Option<bool> {
    let p = plan(which);
    let (m, k, n) = (p.m, p.k, p.n);
    let a = probe_a(m, k);
    let b = probe_b(k, n);

    // Offsets are ours to choose, so keep each operand at the start of whatever
    // buffer its binding reads.
    let (a_off, b_off, c_off) = match p.operands {
        Operands::ArenaF32 => (0, m * k, m * k + k * n),
        Operands::ArenaF32PlusF16Weights => (0, 0, m * k),
        Operands::ArenaF16PlusF32Out => (0, m * k, 0),
    };

    let usage_rw =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let new_buf = |label: &str, size: u64, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };

    // Build the f32 arena and (when the shader wants one) the f16 side buffer.
    let f32_len = match p.operands {
        Operands::ArenaF32 => (m * k + k * n + m * n) as u64,
        Operands::ArenaF32PlusF16Weights => (m * k + m * n) as u64,
        Operands::ArenaF16PlusF32Out => (m * n) as u64,
    };
    let arena = new_buf("rlx-wgpu coop probe arena", f32_len * 4, usage_rw);
    match p.operands {
        Operands::ArenaF32 => {
            queue.write_buffer(&arena, 0, bytemuck::cast_slice(&a));
            queue.write_buffer(&arena, b_off as u64 * 4, bytemuck::cast_slice(&b));
        }
        Operands::ArenaF32PlusF16Weights => {
            queue.write_buffer(&arena, 0, bytemuck::cast_slice(&a));
        }
        Operands::ArenaF16PlusF32Out => {}
    }
    let side = match p.operands {
        Operands::ArenaF32PlusF16Weights => {
            let bytes = f16_bytes(&b);
            let buf = new_buf(
                "rlx-wgpu coop probe f16 B",
                bytes.len().max(4) as u64,
                usage_rw,
            );
            queue.write_buffer(&buf, 0, &bytes);
            Some(buf)
        }
        Operands::ArenaF16PlusF32Out => {
            let mut bytes = f16_bytes(&a);
            bytes.extend_from_slice(&f16_bytes(&b));
            let buf = new_buf(
                "rlx-wgpu coop probe f16 AB",
                bytes.len().max(4) as u64,
                usage_rw,
            );
            queue.write_buffer(&buf, 0, &bytes);
            Some(buf)
        }
        Operands::ArenaF32 => None,
    };

    let ubo = new_buf(
        "rlx-wgpu coop probe params",
        64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    queue.write_buffer(&ubo, 0, &params_bytes(m, k, n, a_off, b_off, c_off));

    // Bind in each shader's declared order — see the module doc table.
    let entries: Vec<wgpu::BindGroupEntry> = match p.operands {
        Operands::ArenaF32 => vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: arena.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ubo.as_entire_binding(),
            },
        ],
        Operands::ArenaF32PlusF16Weights => vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: arena.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ubo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: side.as_ref()?.as_entire_binding(),
            },
        ],
        Operands::ArenaF16PlusF32Out => vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: side.as_ref()?.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: arena.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ubo.as_entire_binding(),
            },
        ],
    };
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rlx-wgpu coop probe"),
        layout: bgl,
        entries: &entries,
    });

    let readback = new_buf(
        "rlx-wgpu coop probe readback",
        (m * n) as u64 * 4,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rlx-wgpu coop probe"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu coop probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(p.grid.0, p.grid.1, 1);
    }
    enc.copy_buffer_to_buffer(&arena, c_off as u64 * 4, &readback, 0, (m * n) as u64 * 4);
    queue.submit(Some(enc.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    rx.recv().ok()?.ok()?;
    let view = slice.get_mapped_range().ok()?;
    let got: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    readback.unmap();

    // `a·b` has no zero elements at all. Every transposition leaves most of the
    // output exactly zero, so this alone separates them — but check the values
    // too, so a merely-scrambled result is caught as well.
    let mut zeros = 0usize;
    let mut worst = 0.0f32;
    let mut peak = 0.0f32;
    for i in 0..m as usize {
        for j in 0..n as usize {
            let want = (i as f32 + 1.0) * (j as f32 + 1.0) * S;
            let g = got[i * n as usize + j];
            if g == 0.0 {
                zeros += 1;
            }
            worst = worst.max((g - want).abs());
            peak = peak.max(want.abs());
        }
    }
    let ok = zeros == 0 && worst <= p.rel_tol * peak;
    if rlx_ir::env::flag("RLX_WGPU_COOP_PROBE_VERBOSE") {
        eprintln!(
            "rlx-wgpu coop probe {which:?}: zeros {zeros}/{}, max|Δ| {worst:.3e} \
             vs tol {:.3e} → {}",
            m * n,
            p.rel_tol * peak,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    Some(ok)
}

// ── convenience wrappers used by the dispatch cascade ───────────────────────

/// The device+queue this process is using, if any.
fn dq() -> Option<(&'static wgpu::Device, &'static wgpu::Queue)> {
    let d = crate::device::wgpu_device()?;
    Some((&d.device, &d.queue))
}

/// Is the f32 cooperative-matrix path trustworthy on this adapter? Metal and
/// the discrete backends reach different shaders, so each is checked on its own.
pub fn f32_path_ok(backend: Option<wgpu::Backend>) -> bool {
    let Some((device, queue)) = dq() else {
        return false;
    };
    let which = if backend == Some(wgpu::Backend::Metal) {
        CoopKernel::F32Metal
    } else {
        CoopKernel::F32Portable
    };
    verified(device, queue, which)
}

/// Is `matmul_coop16` trustworthy on this adapter?
pub fn coop16_path_ok() -> bool {
    let Some((device, queue)) = dq() else {
        return false;
    };
    verified(device, queue, CoopKernel::Coop16)
}

/// Which `matmul_coop_f16_vulkan*` B-load variant actually computes `a·b` here.
///
/// This replaces a pure size heuristic. The two shaders differ ONLY in whether B
/// is read with `coopLoad` or `coopLoadT` — same pointer, same stride, same body
/// — and the old rule picked between them on `n > 768`. A memory layout is a
/// property of the bytes, not of N, so at most one of them can be right and the
/// threshold was choosing wrong numbers for one half of the N range. Ask the
/// device instead.
///
/// Returns `Some(widen)` for a variant that passes, preferring the historical
/// size-based choice when both do (it was tuned for speed, and speed is a fine
/// tie-break among correct kernels). `None` means neither is correct here and
/// the caller must not use this path at all.
pub fn f16_vk_widen_choice(n: u32, f32acc: bool) -> Option<bool> {
    let (device, queue) = dq()?;
    let preferred = crate::kernels::coop_f16_vk_widen_b_load(n);
    if verified(
        device,
        queue,
        CoopKernel::F16Vk {
            widen: preferred,
            f32acc,
        },
    ) {
        return Some(preferred);
    }
    if verified(
        device,
        queue,
        CoopKernel::F16Vk {
            widen: !preferred,
            f32acc,
        },
    ) {
        return Some(!preferred);
    }
    None
}

/// Is any `matmul_coop_f16_vulkan*` variant trustworthy for this `n`?
pub fn f16_vk_path_ok(n: u32) -> bool {
    let Some((device, queue)) = dq() else {
        return false;
    };
    let f32acc = crate::kernels::coop_f16_vk_use_f32acc(device);
    let _ = queue;
    f16_vk_widen_choice(n, f32acc).is_some()
}
