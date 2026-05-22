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

//! MPSMatrixMultiplication bridge — Apple's per-chip-tuned matmul.
//!
//! For large matmuls (M·K·N above a threshold) Apple's MPS sgemm routinely
//! beats hand-rolled MSL because it has private knowledge of the GPU's
//! tensor-unit scheduling per chip generation. We bridge it via objc.
//!
//! Trade-off: per-call objc bridging is ~5–20µs, so MPS only wins above a
//! threshold (rough rule of thumb: M·K·N ≥ 16M FLOPs). The cost model in
//! `cost.rs` decides.
//!
//! Note: `MPSMatrixMultiplication::encode` allocates and submits its own
//! compute encoder internally — callers must end any open compute encoder
//! on the same command buffer before invoking us. The shared-encoder split
//! is wired in `backend::encode_and_run`.

use metal::{Buffer, CommandBufferRef};
use objc::runtime::{BOOL, NO, Object, YES};
use objc::{class, msg_send, sel, sel_impl};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// Link the MetalPerformanceShaders framework. metal-rs gates its own
// MPS link directive behind a feature we don't enable.
#[link(name = "MetalPerformanceShaders", kind = "framework")]
unsafe extern "C" {}

/// MPSDataType values (from MPSCore.h).
#[allow(non_upper_case_globals, dead_code)]
mod mps_dtype {
    pub const Float32: u32 = 0x10000000 | 32;
    pub const Float16: u32 = 0x10000000 | 16;
}

/// True iff MPSMatrixMultiplication is available (any modern macOS Metal device).
pub fn mps_supports_matmul() -> bool {
    static AVAIL: OnceLock<bool> = OnceLock::new();
    *AVAIL.get_or_init(|| objc::runtime::Class::get("MPSMatrixMultiplication").is_some())
}

/// Cache of `(m,k,n)` → retained MPSMatrixMultiplication kernel.
///
/// **Bridge-cost mitigation #1.** Building the kernel involves ~6 objc
/// messages (alloc, init, set transposes, set α/β). For typical inference
/// the same shapes recur every layer, so caching reduces per-call objc
/// overhead.
struct KernelCache {
    map: Mutex<HashMap<(usize, usize, usize, bool), usize>>,
}
unsafe impl Send for KernelCache {}
unsafe impl Sync for KernelCache {}

fn kernel_cache() -> &'static KernelCache {
    static CACHE: OnceLock<KernelCache> = OnceLock::new();
    CACHE.get_or_init(|| KernelCache {
        map: Mutex::new(HashMap::new()),
    })
}

/// Cache of `(buf_ptr, offset, rows, cols)` → retained MPSMatrix wrapper.
///
/// **Bridge-cost mitigation #2.** Each MPSMatrix wraps an MTLBuffer + offset
/// + descriptor. Within a compiled graph the (buffer, offset, dims) triple
/// is fully static — it never changes call-to-call. Caching them eliminates
/// 9 objc messages per matmul (3 descriptor alloc + 3 matrix alloc/init +
/// 3 matrix release).
///
/// Key uses `(buf_ptr as usize, offset, rows, cols)`; descriptor is also
/// cached separately keyed on `(rows, cols)`.
struct MatrixCache {
    matrices: Mutex<HashMap<(usize, usize, usize, usize), usize>>,
    descriptors: Mutex<HashMap<(usize, usize), usize>>,
}
unsafe impl Send for MatrixCache {}
unsafe impl Sync for MatrixCache {}

fn matrix_cache() -> &'static MatrixCache {
    static CACHE: OnceLock<MatrixCache> = OnceLock::new();
    CACHE.get_or_init(|| MatrixCache {
        matrices: Mutex::new(HashMap::new()),
        descriptors: Mutex::new(HashMap::new()),
    })
}

unsafe fn get_or_build_descriptor(rows: usize, cols: usize, dtype: u32) -> *mut Object {
    let cache = matrix_cache();
    let mut map = cache.descriptors.lock().expect("descriptor cache poisoned");
    let key = (rows, cols * 8 + dtype as usize); // pack dtype into key
    if let Some(&p) = map.get(&key) {
        return p as *mut Object;
    }
    let cls = class!(MPSMatrixDescriptor);
    let bytes_per_elem = if dtype == mps_dtype::Float16 { 2 } else { 4 };
    let row_bytes = cols * bytes_per_elem;
    let desc: *mut Object = msg_send![cls,
        matrixDescriptorWithRows: rows as u64
        columns: cols as u64
        rowBytes: row_bytes as u64
        dataType: dtype];
    let _: () = msg_send![desc, retain];
    map.insert(key, desc as usize);
    desc
}

unsafe fn get_or_build_matrix(
    buf: &Buffer,
    offset: usize,
    rows: usize,
    cols: usize,
    dtype: u32,
) -> *mut Object {
    unsafe {
        let cache = matrix_cache();
        // The Buffer-wrapper address can recycle when a previous Sam is
        // dropped — relying on `&**buf as usize` as the identity led to
        // stale `MPSMatrix` lookups → GPU reads from freed memory →
        // NaN. To break the aliasing, callers that build new arenas
        // (e.g. `MetalBackend::compile_inner`) call
        // `invalidate_caches()` first so this map is empty.
        let buf_ptr = (&**buf as *const metal::BufferRef) as usize;
        let key = (buf_ptr, offset, rows, cols * 8 + dtype as usize);
        let mut map = cache.matrices.lock().expect("matrix cache poisoned");
        if let Some(&p) = map.get(&key) {
            return p as *mut Object;
        }
        let desc = get_or_build_descriptor(rows, cols, dtype);
        let cls = class!(MPSMatrix);
        let alloc: *mut Object = msg_send![cls, alloc];
        let buf_ref: &metal::BufferRef = buf;
        let mat: *mut Object = msg_send![alloc,
        initWithBuffer: buf_ref
        offset: offset as u64
        descriptor: desc];
        map.insert(key, mat as usize);
        mat
    }
}

/// Drop every cached MPSMatrix / MPSMatrixDescriptor / MPSMatrixMultiplication
/// reference. Lets a caller (e.g. backend test harness, hot reload) reset
/// MPS state explicitly. The default `Drop` for cached `*mut Object` would
/// leak (no `release` call); but for correctness the leak is benign since
/// Metal kernels and matrices are tiny.
pub fn invalidate_caches() {
    let cache = matrix_cache();
    let mut mats = cache.matrices.lock().expect("matrix cache poisoned");
    mats.clear();
    let mut descs = cache.descriptors.lock().expect("descriptor cache poisoned");
    descs.clear();
    let kcache = kernel_cache();
    let mut km = kcache.map.lock().expect("kernel cache poisoned");
    km.clear();
}

unsafe fn get_or_build_kernel(m: usize, k: usize, n: usize, transpose_b: bool) -> *mut Object {
    let cache = kernel_cache();
    let mut map = cache.map.lock().expect("kernel cache poisoned");
    if let Some(&p) = map.get(&(m, k, n, transpose_b)) {
        return p as *mut Object;
    }
    use crate::device::metal_device;
    let dev = metal_device().expect("Metal device required");
    let cls = class!(MPSMatrixMultiplication);
    let alloc: *mut Object = msg_send![cls, alloc];
    let dev_ref: &metal::DeviceRef = &dev.device;
    let kernel: *mut Object = msg_send![alloc,
        initWithDevice: dev_ref
        transposeLeft: NO as BOOL
        transposeRight: if transpose_b { YES } else { NO } as BOOL
        resultRows: m as u64
        resultColumns: n as u64
        interiorColumns: k as u64
        alpha: 1.0_f64
        beta: 0.0_f64
    ];
    map.insert((m, k, n, transpose_b), kernel as usize);
    kernel
}

/// Encode `C = A @ B` via MPSMatrixMultiplication.
///
/// Hot path: cached kernel + cached MPSMatrix wrappers → only one objc
/// message at runtime (`encodeToCommandBuffer`). Everything else amortizes.
pub fn encode_mps_sgemm(
    cmd_buf: &CommandBufferRef,
    arena: &Buffer,
    a_off: usize,
    b_off: usize,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    encode_mps_matmul(
        cmd_buf,
        arena,
        a_off,
        b_off,
        c_off,
        m,
        k,
        n,
        mps_dtype::Float32,
        false,
    );
}

/// `C = A @ B^T` where `B` is stored as `[n, k]` row-major (GGUF dequant layout).
pub fn encode_mps_sgemm_bt(
    cmd_buf: &CommandBufferRef,
    arena: &Buffer,
    a_off: usize,
    b_off: usize,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    encode_mps_matmul(
        cmd_buf,
        arena,
        a_off,
        b_off,
        c_off,
        m,
        k,
        n,
        mps_dtype::Float32,
        true,
    );
}

/// Encode `C = A @ B` at half-precision via MPS.
pub fn encode_mps_hgemm(
    cmd_buf: &CommandBufferRef,
    arena: &Buffer,
    a_off: usize,
    b_off: usize,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    encode_mps_matmul(
        cmd_buf,
        arena,
        a_off,
        b_off,
        c_off,
        m,
        k,
        n,
        mps_dtype::Float16,
        false,
    );
}

fn encode_mps_matmul(
    cmd_buf: &CommandBufferRef,
    arena: &Buffer,
    a_off: usize,
    b_off: usize,
    c_off: usize,
    m: usize,
    k: usize,
    n: usize,
    dtype: u32,
    transpose_b: bool,
) {
    unsafe {
        let a_mat = get_or_build_matrix(arena, a_off, m, k, dtype);
        let (b_rows, b_cols) = if transpose_b { (n, k) } else { (k, n) };
        let b_mat = get_or_build_matrix(arena, b_off, b_rows, b_cols, dtype);
        let c_mat = get_or_build_matrix(arena, c_off, m, n, dtype);
        let kernel = get_or_build_kernel(m, k, n, transpose_b);
        let _: () = msg_send![kernel,
            encodeToCommandBuffer: cmd_buf
            leftMatrix: a_mat
            rightMatrix: b_mat
            resultMatrix: c_mat
        ];
    }
}
