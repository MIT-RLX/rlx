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

//! Indirect Command Buffer support — pre-encode thunks once at compile
//! time, re-submit on every forward pass.
//!
//! **Status: prototype, deferred (Phase C).**
//!
//! Phase C trace data (`RLX_METAL_TRACE=1`) showed the per-run cost on
//! Apple Silicon splits as encode (5–25 µs) + commit (5 µs) +
//! wait_until_completed (150–200 µs). ICB attacks `encode`; the actual
//! bottleneck is `wait`, addressed by `commit_no_wait` / `run_pipelined`
//! in `backend.rs` instead. ICB would still buy ~10–20 µs once the
//! Apple-platform fault below is resolved, but the win is small relative
//! to pipelining.
//!
//! Open issue blocking integration: even with the support flag set on
//! the pipeline descriptor before `new_compute_pipeline_state`,
//! `set_compute_pipeline_state` on the `IndirectComputeCommand` triggers
//! a runtime fault. The full investigation notes are below; revisit if
//! a workload appears that's launch-bound rather than wait-bound.
//!
//! Standard pattern: every `enc.set_pipeline + enc.set_buffer + enc.dispatch`
//! call costs ~1–5 µs of CPU-side encoding overhead. For a 12-layer BERT
//! that's 60–120 dispatches × 1–5 µs = 60–600 µs per forward of just
//! encoding work, before any GPU compute starts. Pre-encoding once and
//! re-submitting via `execute_commands_in_buffer` cuts this to a single
//! submit cost.
//!
//! ICB constraints we satisfy:
//!   - `MTLIndirectCommandType::ConcurrentDispatch` on the descriptor;
//!     `set_barrier()` per command for serial semantics.
//!   - `set_inherit_buffers(false)` + per-command `set_kernel_buffer`.
//!   - At execute time: `useResource:usage:` on the outer compute encoder
//!     for the arena and constants buffer (else the GPU faults — required
//!     when buffers don't inherit from the encoder).
//!   - `IcbKernels`: pipelines rebuilt with
//!     `set_support_indirect_command_buffers(true)`. The regular
//!     `Kernels` pipelines lack the flag and segfault when bound to an
//!     `IndirectComputeCommand`.
//!   - No `setBytes:` — kernels' inline constants (`m`, `n`, `len`,
//!     `eps`) live in one shared "constants" `MTLBuffer` (one slot per
//!     command, 64 B / cmd).
//!   - MPS calls (`MPSMatrixMultiplication.encodeToCommandBuffer`)
//!     internally allocate their own compute encoders — they cannot live
//!     in an ICB. The runtime path would keep them on the lazy compute
//!     encoder, executing ICB segments and MPS calls in interleaved order.
//!
//! **Open issue blocking integration:** even with the support flag set on
//! the pipeline descriptor before `new_compute_pipeline_state`,
//! `set_compute_pipeline_state` on the `IndirectComputeCommand` triggers
//! a runtime fault. Suspect an Apple-platform requirement not covered
//! above (function constants? device feature gate? metal-rs 0.30 missing
//! a setter?). Future work to investigate before flipping the runtime
//! switch.

use metal::{
    Buffer, ComputeCommandEncoderRef, ComputePipelineDescriptor, ComputePipelineState, Device,
    IndirectCommandBuffer, IndirectCommandBufferDescriptor, MTLIndirectCommandType,
    MTLResourceOptions, MTLSize, NSRange,
};
use objc::{msg_send, sel, sel_impl};
use std::sync::OnceLock;

use crate::thunk::Thunk;

/// ICB-compatible pipeline states. Recompiled with
/// `supportIndirectCommandBuffers=true` — the regular pipelines built by
/// `kernels()` have the flag off and segfault when bound to an
/// `IndirectComputeCommand`.
pub struct IcbKernels {
    pub bias_add: ComputePipelineState,
    pub gelu_inplace: ComputePipelineState,
    pub silu_inplace: ComputePipelineState,
    pub elem_add: ComputePipelineState,
    pub elem_mul: ComputePipelineState,
    pub copy_f32: ComputePipelineState,
    pub layer_norm: ComputePipelineState,
    pub fused_residual_ln: ComputePipelineState,
    pub narrow_lastax: ComputePipelineState,
    pub rope: ComputePipelineState,
}

unsafe impl Send for IcbKernels {}
unsafe impl Sync for IcbKernels {}

impl IcbKernels {
    fn new(dev: &Device, library: &metal::LibraryRef) -> Self {
        let trace = rlx_ir::env::flag("RLX_ICB_TRACE");
        let make = |name: &str| -> ComputePipelineState {
            let f = library.get_function(name, None).expect(name);
            let desc = ComputePipelineDescriptor::new();
            // Set the support flag *first*, then the compute function — some
            // Apple-runtime variants validate the function against descriptor
            // properties at set time.
            desc.set_support_indirect_command_buffers(true);
            desc.set_compute_function(Some(&f));
            // Sanity: confirm the flag actually stuck on the descriptor.
            let flag = desc.support_indirect_command_buffers();
            if trace {
                eprintln!("[icb-kernels] {name}: support_icb={flag}");
            }
            dev.new_compute_pipeline_state(&desc).expect(name)
        };
        Self {
            bias_add: make("bias_add"),
            gelu_inplace: make("gelu_inplace"),
            silu_inplace: make("silu_inplace"),
            elem_add: make("elem_add"),
            elem_mul: make("elem_mul"),
            copy_f32: make("copy_f32"),
            layer_norm: make("layer_norm"),
            fused_residual_ln: make("fused_residual_ln"),
            narrow_lastax: make("narrow_lastax"),
            rope: make("rope"),
        }
    }
}

/// Get-or-init ICB kernels. Builds on first access from the same MSL
/// source as the regular `kernels()`.
pub fn icb_kernels() -> &'static IcbKernels {
    static K: OnceLock<IcbKernels> = OnceLock::new();
    K.get_or_init(|| {
        use crate::device::metal_device;
        let dev = metal_device().expect("Metal device required");
        let opts = metal::CompileOptions::new();
        let library = dev
            .device
            .new_library_with_source(crate::kernels::RLX_KERNELS_MSL, &opts)
            .expect("MSL compilation for ICB kernels failed");
        IcbKernels::new(&dev.device, &library)
    })
}

/// One ICB segment of pre-encoded compute commands. Built at compile time;
/// at runtime, the outer compute encoder calls `execute_commands_in_buffer`
/// over the recorded range and the GPU schedules them all in one shot.
pub struct IcbSegment {
    pub icb: IndirectCommandBuffer,
    pub command_count: u64,
    /// Backing buffer for kernels' inline constants (m, n, len, eps, ...).
    /// Kept alive for the lifetime of the segment.
    pub constants: Buffer,
}

unsafe impl Send for IcbSegment {}
unsafe impl Sync for IcbSegment {}

/// MTLResourceUsage bitflags (from `<Metal/MTLResource.h>`).
const RESOURCE_USAGE_READ: u64 = 1 << 0;
const RESOURCE_USAGE_WRITE: u64 = 1 << 1;

impl IcbSegment {
    /// Issue the segment onto a live compute encoder.
    ///
    /// Two non-obvious calls are required when `inherit_buffers=false`:
    ///   1. `use_resource:usage:` — tells the executing encoder about every
    ///      MTLBuffer the ICB will touch, so resource residency / hazard
    ///      tracking work. Without this, the GPU faults when commands try
    ///      to access the buffer.
    ///   2. The selector for `executeCommandsInBuffer:withRange:` is on
    ///      `MTLComputeCommandEncoder` but metal-rs 0.30 only exposes it on
    ///      `RenderCommandEncoder`, so we drop to objc.
    pub fn execute_on(&self, enc: &ComputeCommandEncoderRef, arena: &Buffer) {
        if self.command_count == 0 {
            return;
        }
        unsafe {
            let arena_ref: &metal::BufferRef = arena;
            let const_ref: &metal::BufferRef = &self.constants;
            let _: () = msg_send![enc, useResource: arena_ref
                usage: (RESOURCE_USAGE_READ | RESOURCE_USAGE_WRITE)];
            let _: () = msg_send![enc, useResource: const_ref
                usage: RESOURCE_USAGE_READ];
            let range = NSRange {
                location: 0,
                length: self.command_count,
            };
            let _: () = msg_send![enc,
                executeCommandsInBuffer: &*self.icb
                withRange: range];
        }
    }
}

fn thunk_kind(t: &Thunk) -> &'static str {
    match t {
        Thunk::Nop => "Nop",
        Thunk::Cast { .. } => "Cast",
        Thunk::Sgemm { .. } => "Sgemm",
        Thunk::FusedMmBiasAct { .. } => "FusedMmBiasAct",
        Thunk::BiasAdd { .. } => "BiasAdd",
        Thunk::ActivationInPlace { .. } => "ActivationInPlace",
        Thunk::BinaryFull { .. } => "BinaryFull",
        Thunk::Copy { .. } => "Copy",
        Thunk::LayerNorm { .. } => "LayerNorm",
        Thunk::RmsNorm { .. } => "RmsNorm",
        Thunk::Softmax { .. } => "Softmax",
        Thunk::Reduce { .. } => "Reduce",
        Thunk::Gather { .. } => "Gather",
        Thunk::Narrow { .. } => "Narrow",
        Thunk::Transpose { .. } => "Transpose",
        Thunk::Concat { .. } => "Concat",
        Thunk::Attention { .. } => "Attention",
        Thunk::Rope { .. } => "Rope",
        Thunk::FusedResidualLN { .. } => "FusedResidualLN",
        _ => "Other",
    }
}

/// Per-thunk classification. ICB-friendly thunks compile into the
/// IndirectCommandBuffer; everything else (matmul variants, MPS, anything
/// that needs a separate encoder) is run inline by the existing path.
///
/// Currently F32 only — IcbKernels holds f32 pipelines. f16 thunks return
/// false so they take the per-op path where the dt-aware encoder picks
/// the `_h` kernel variant. Adding f16 ICB pipelines is a follow-up.
fn is_icb_compatible(t: &Thunk) -> bool {
    use crate::thunk::HalfFlag;
    let f32_dt = |dt: &HalfFlag| matches!(dt, HalfFlag::F32);
    match t {
        Thunk::BiasAdd { dt, .. } => f32_dt(dt),
        Thunk::ActivationInPlace { dt, .. } => f32_dt(dt),
        Thunk::BinaryFull { dt, .. } => f32_dt(dt),
        Thunk::Copy { dt, .. } => f32_dt(dt),
        Thunk::LayerNorm { dt, .. } => f32_dt(dt),
        Thunk::FusedResidualLN { dt, .. } => f32_dt(dt),
        Thunk::Narrow { dt, .. } => f32_dt(dt),
        Thunk::Rope { dt, .. } => f32_dt(dt),
        _ => false,
    }
}

/// Estimate how many bytes the constants buffer needs for `n` ICB commands.
/// We give each command 64 B of slack — enough for up to 16 u32 args.
const CONSTANTS_BYTES_PER_CMD: usize = 64;

/// One ICB segment plus the index range in the original schedule it replaces.
/// `start..end` is the half-open range of thunk indices the segment covers
/// (Nops included, even though they don't generate ICB commands).
pub struct IcbRange {
    pub start: usize,
    pub end: usize,
    pub segment: IcbSegment,
}

/// Minimum number of ICB-compatible thunks in a run before we bother
/// building a segment for it. Single-op runs pay the
/// `executeCommandsInBuffer` overhead without amortizing it across enough
/// dispatches to beat the per-op `set_pipeline + set_buffer + dispatch`
/// path.
const MIN_ICB_RUN: usize = 2;

/// Split a thunk schedule into maximal runs of ICB-compatible thunks and
/// build one ICB segment per run. Returns the segments tagged with the
/// `start..end` index range in the original schedule, so the runtime can
/// interleave segment execution with per-op encoding for the gaps that
/// hold matmul / MPS / cast / etc.
pub fn compile_segments(thunks: &[Thunk], arena: &Buffer, dev: &Device) -> Vec<IcbRange> {
    let trace = rlx_ir::env::flag("RLX_ICB_TRACE");
    if trace {
        let mut hist: std::collections::HashMap<&'static str, usize> = Default::default();
        for t in thunks {
            *hist.entry(thunk_kind(t)).or_default() += 1;
        }
        let mut entries: Vec<_> = hist.into_iter().collect();
        entries.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        eprintln!("[icb] thunk histogram: {:?}", entries);
        let order: String = thunks
            .iter()
            .filter(|t| !matches!(t, Thunk::Nop))
            .map(|t| if is_icb_compatible(t) { "+" } else { "-" })
            .collect();
        eprintln!("[icb] schedule (skipping Nops, +=ICB / -=other): {order}");
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < thunks.len() {
        // Skip past any non-ICB thunks (they'll be handled per-op).
        while i < thunks.len() && !is_icb_compatible(&thunks[i]) && !matches!(thunks[i], Thunk::Nop)
        {
            i += 1;
        }
        // Collect a maximal run of ICB-compatible thunks (Nops pass through).
        let start = i;
        let mut run: Vec<&Thunk> = Vec::new();
        while i < thunks.len() && (is_icb_compatible(&thunks[i]) || matches!(thunks[i], Thunk::Nop))
        {
            if !matches!(thunks[i], Thunk::Nop) {
                run.push(&thunks[i]);
            }
            i += 1;
        }
        let end = i;
        if run.len() >= MIN_ICB_RUN
            && let Some(seg) = build_segment(&run, arena, dev)
        {
            if trace {
                eprintln!(
                    "[icb] segment thunks {}..{} ({} cmds)",
                    start, end, seg.command_count
                );
            }
            out.push(IcbRange {
                start,
                end,
                segment: seg,
            });
        }
    }
    if trace {
        eprintln!(
            "[icb] compile_segments: {} segments over {} thunks",
            out.len(),
            thunks.len()
        );
    }
    out
}

/// Strict mode: returns `Some` only if **all** non-Nop thunks are ICB-able.
/// Used by the standalone `icb_check` example; production paths use
/// `compile_segments` instead.
pub fn try_compile(thunks: &[Thunk], arena: &Buffer, dev: &Device) -> Option<IcbSegment> {
    let compute_thunks: Vec<&Thunk> = thunks.iter().filter(|t| !matches!(t, Thunk::Nop)).collect();
    if compute_thunks.is_empty() {
        return None;
    }
    if !compute_thunks.iter().all(|t| is_icb_compatible(t)) {
        return None;
    }
    build_segment(&compute_thunks, arena, dev)
}

/// Encode a list of (already filtered) ICB-compatible thunks into one
/// ICB segment. Returns `None` if the list is empty.
fn build_segment(icb_thunks: &[&Thunk], arena: &Buffer, dev: &Device) -> Option<IcbSegment> {
    let trace = rlx_ir::env::flag("RLX_ICB_TRACE");
    if icb_thunks.is_empty() {
        return None;
    }
    let n = icb_thunks.len() as u64;
    if trace {
        eprintln!("[icb] build_segment n={n}");
    }

    let desc = IndirectCommandBufferDescriptor::new();
    if trace {
        eprintln!("[icb] descriptor created");
    }
    // Must include both ConcurrentDispatch (for `dispatch_thread_groups`)
    // and ConcurrentDispatchThreads (for `dispatch_threads`) — our encoders
    // use the threads form for elementwise/activation kernels and the
    // threadgroups form for fused_residual_ln. Setting only one causes the
    // other dispatch type to fault when the command runs.
    desc.set_command_types(
        MTLIndirectCommandType::ConcurrentDispatch
            | MTLIndirectCommandType::ConcurrentDispatchThreads,
    );
    desc.set_inherit_buffers(false);
    desc.set_inherit_pipeline_state(false);
    desc.set_max_kernel_buffer_bind_count(8);
    if trace {
        eprintln!("[icb] descriptor configured");
    }

    let icb =
        dev.new_indirect_command_buffer_with_descriptor(&desc, n, MTLResourceOptions::empty());
    if trace {
        eprintln!("[icb] icb allocated");
    }

    let constants = dev.new_buffer(
        (n as usize * CONSTANTS_BYTES_PER_CMD) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    if trace {
        eprintln!("[icb] constants buffer allocated");
    }

    let kk = icb_kernels();
    for (idx, t) in icb_thunks.iter().enumerate() {
        if trace {
            eprintln!("[icb] encode cmd {idx}");
        }
        let cmd = icb.indirect_compute_command_at_index(idx as u64);
        let cb_off = idx * CONSTANTS_BYTES_PER_CMD;
        encode_thunk_into_icb(cmd, t, arena, &constants, cb_off, kk);
        cmd.set_barrier();
    }
    if trace {
        eprintln!("[icb] build_segment done");
    }

    Some(IcbSegment {
        icb,
        command_count: n,
        constants,
    })
}

/// Encode one thunk into an `IndirectComputeCommand`. Constants that the
/// MSL kernel reads via `device const T&` come from `constants_buf`
/// (offset `cb_off`); all tensor data comes from `arena`.
fn encode_thunk_into_icb(
    cmd: &metal::IndirectComputeCommandRef,
    thunk: &Thunk,
    arena: &Buffer,
    constants_buf: &Buffer,
    cb_off: usize,
    k: &IcbKernels,
) {
    use rlx_ir::op::{Activation, BinaryOp};

    // Helper: write a sequence of u32s into `constants_buf` starting at
    // `cb_off`, return the byte offset of slot `i`.
    let write_u32s = |vals: &[u32]| unsafe {
        let p = constants_buf.contents() as *mut u8;
        let dst = p.add(cb_off) as *mut u32;
        for (i, &v) in vals.iter().enumerate() {
            *dst.add(i) = v;
        }
    };
    let cb_arg = |slot_idx: usize| -> u64 { (cb_off + slot_idx * 4) as u64 };

    match thunk {
        Thunk::BiasAdd {
            src,
            bias,
            dst,
            m,
            n,
            ..
        } => {
            // Encode_bias_add only uses src and bias (in-place); dst-aware
            // path runs an extra copy first, which we don't ICB here.
            // If src != dst the caller should fall back.
            if *src != *dst {
                return;
            }
            write_u32s(&[*m, *n]);
            cmd.set_compute_pipeline_state(&k.bias_add);
            cmd.set_kernel_buffer(0, Some(&**arena), *dst as u64);
            cmd.set_kernel_buffer(1, Some(&**arena), *bias as u64);
            cmd.set_kernel_buffer(2, Some(&**constants_buf), cb_arg(0));
            cmd.set_kernel_buffer(3, Some(&**constants_buf), cb_arg(1));
            let grid = MTLSize {
                width: *n as u64,
                height: *m as u64,
                depth: 1,
            };
            let tg = MTLSize {
                width: 16.min(*n as u64),
                height: 16.min(*m as u64),
                depth: 1,
            };
            cmd.concurrent_dispatch_threads(grid, tg);
        }
        Thunk::ActivationInPlace { data, len, act, .. } => {
            let trace = rlx_ir::env::flag("RLX_ICB_TRACE");
            if trace {
                eprintln!("  [act] write u32");
            }
            write_u32s(&[*len]);
            let pipeline = match act {
                Activation::Gelu => &k.gelu_inplace,
                Activation::Silu => &k.silu_inplace,
                _ => return,
            };
            if trace {
                eprintln!("  [act] set pipeline");
            }
            cmd.set_compute_pipeline_state(pipeline);
            if trace {
                eprintln!("  [act] set buf 0");
            }
            let arena_ref: &metal::BufferRef = arena;
            cmd.set_kernel_buffer(0, Some(arena_ref), *data as u64);
            if trace {
                eprintln!("  [act] set buf 1");
            }
            let cb_ref: &metal::BufferRef = constants_buf;
            cmd.set_kernel_buffer(1, Some(cb_ref), cb_arg(0));
            if trace {
                eprintln!("  [act] dispatch");
            }
            let tg_w = pipeline.thread_execution_width().min(*len as u64);
            cmd.concurrent_dispatch_threads(
                MTLSize {
                    width: *len as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg_w,
                    height: 1,
                    depth: 1,
                },
            );
            if trace {
                eprintln!("  [act] done");
            }
        }
        Thunk::BinaryFull {
            lhs,
            rhs,
            dst,
            len,
            op,
            ..
        } => {
            write_u32s(&[*len]);
            let pipeline = match op {
                BinaryOp::Add => &k.elem_add,
                BinaryOp::Mul => &k.elem_mul,
                _ => return,
            };
            cmd.set_compute_pipeline_state(pipeline);
            cmd.set_kernel_buffer(0, Some(&**arena), *lhs as u64);
            cmd.set_kernel_buffer(1, Some(&**arena), *rhs as u64);
            cmd.set_kernel_buffer(2, Some(&**arena), *dst as u64);
            cmd.set_kernel_buffer(3, Some(&**constants_buf), cb_arg(0));
            let tg_w = pipeline.thread_execution_width().min(*len as u64);
            cmd.concurrent_dispatch_threads(
                MTLSize {
                    width: *len as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg_w,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Thunk::Copy { src, dst, len, .. } => {
            write_u32s(&[*len]);
            cmd.set_compute_pipeline_state(&k.copy_f32);
            cmd.set_kernel_buffer(0, Some(&**arena), *src as u64);
            cmd.set_kernel_buffer(1, Some(&**arena), *dst as u64);
            cmd.set_kernel_buffer(2, Some(&**constants_buf), cb_arg(0));
            let tg_w = k.copy_f32.thread_execution_width().min(*len as u64);
            cmd.concurrent_dispatch_threads(
                MTLSize {
                    width: *len as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg_w,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Thunk::LayerNorm {
            src,
            g,
            b,
            dst,
            rows,
            h,
            eps,
            ..
        } => {
            // Layout: [h_u32, eps_f32]
            write_u32s(&[*h]);
            unsafe {
                let p = constants_buf.contents() as *mut u8;
                let f = p.add(cb_off + 4) as *mut f32;
                *f = *eps;
            }
            cmd.set_compute_pipeline_state(&k.layer_norm);
            cmd.set_kernel_buffer(0, Some(&**arena), *src as u64);
            cmd.set_kernel_buffer(1, Some(&**arena), *g as u64);
            cmd.set_kernel_buffer(2, Some(&**arena), *b as u64);
            cmd.set_kernel_buffer(3, Some(&**arena), *dst as u64);
            cmd.set_kernel_buffer(4, Some(&**constants_buf), cb_arg(0));
            cmd.set_kernel_buffer(5, Some(&**constants_buf), cb_arg(1));
            let tg_w = 256u64.min(*h as u64);
            cmd.concurrent_dispatch_threads(
                MTLSize {
                    width: tg_w,
                    height: *rows as u64,
                    depth: 1,
                },
                MTLSize {
                    width: tg_w,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Thunk::FusedResidualLN {
            x,
            res,
            g,
            b,
            out,
            rows,
            h,
            eps,
            ..
        } => {
            write_u32s(&[*h]);
            unsafe {
                let p = constants_buf.contents() as *mut u8;
                let f = p.add(cb_off + 4) as *mut f32;
                *f = *eps;
            }
            cmd.set_compute_pipeline_state(&k.fused_residual_ln);
            cmd.set_kernel_buffer(0, Some(&**arena), *x as u64);
            cmd.set_kernel_buffer(1, Some(&**arena), *res as u64);
            cmd.set_kernel_buffer(2, Some(&**arena), *g as u64);
            cmd.set_kernel_buffer(3, Some(&**arena), *b as u64);
            cmd.set_kernel_buffer(4, Some(&**arena), *out as u64);
            cmd.set_kernel_buffer(5, Some(&**constants_buf), cb_arg(0));
            cmd.set_kernel_buffer(6, Some(&**constants_buf), cb_arg(1));
            let tg_w = 256u64.min(*h as u64);
            cmd.concurrent_dispatch_threadgroups(
                MTLSize {
                    width: 1,
                    height: *rows as u64,
                    depth: 1,
                },
                MTLSize {
                    width: tg_w,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Thunk::Narrow {
            src,
            dst,
            outer,
            src_axis,
            start,
            len,
            ..
        } => {
            // Layout: [outer, src_axis, start, len]
            write_u32s(&[*outer, *src_axis, *start, *len]);
            cmd.set_compute_pipeline_state(&k.narrow_lastax);
            cmd.set_kernel_buffer(0, Some(&**arena), *src as u64);
            cmd.set_kernel_buffer(1, Some(&**arena), *dst as u64);
            cmd.set_kernel_buffer(2, Some(&**constants_buf), cb_arg(0));
            cmd.set_kernel_buffer(3, Some(&**constants_buf), cb_arg(1));
            cmd.set_kernel_buffer(4, Some(&**constants_buf), cb_arg(2));
            cmd.set_kernel_buffer(5, Some(&**constants_buf), cb_arg(3));
            let grid = MTLSize {
                width: *len as u64,
                height: *outer as u64,
                depth: 1,
            };
            let tg = MTLSize {
                width: 16.min(*len as u64),
                height: 16.min(*outer as u64),
                depth: 1,
            };
            cmd.concurrent_dispatch_threads(grid, tg);
        }
        Thunk::Rope {
            src,
            cos,
            sin,
            dst,
            batch,
            seq,
            hidden,
            head_dim,
            n_rot,
            src_row_stride,
            ..
        } => {
            // Layout: [batch, seq, hidden, head_dim, src_row_stride, seq_stride, n_rot]
            write_u32s(&[
                *batch,
                *seq,
                *hidden,
                *head_dim,
                *src_row_stride,
                *seq,
                *n_rot,
            ]);
            cmd.set_compute_pipeline_state(&k.rope);
            cmd.set_kernel_buffer(0, Some(&**arena), *src as u64);
            cmd.set_kernel_buffer(1, Some(&**arena), *cos as u64);
            cmd.set_kernel_buffer(2, Some(&**arena), *sin as u64);
            cmd.set_kernel_buffer(3, Some(&**arena), *dst as u64);
            cmd.set_kernel_buffer(4, Some(&**constants_buf), cb_arg(0));
            cmd.set_kernel_buffer(5, Some(&**constants_buf), cb_arg(1));
            cmd.set_kernel_buffer(6, Some(&**constants_buf), cb_arg(2));
            cmd.set_kernel_buffer(7, Some(&**constants_buf), cb_arg(3));
            cmd.set_kernel_buffer(8, Some(&**constants_buf), cb_arg(4));
            cmd.set_kernel_buffer(9, Some(&**constants_buf), cb_arg(5));
            cmd.set_kernel_buffer(10, Some(&**constants_buf), cb_arg(6));
            let nh = *hidden / *head_dim;
            let grid = MTLSize {
                width: *head_dim as u64,
                height: nh as u64,
                depth: (*batch * *seq) as u64,
            };
            let tg = MTLSize {
                width: (*head_dim).min(16) as u64,
                height: nh.min(8) as u64,
                depth: 1,
            };
            cmd.concurrent_dispatch_threads(grid, tg);
        }
        _ => {} // Caller filtered via is_icb_compatible.
    }
}
