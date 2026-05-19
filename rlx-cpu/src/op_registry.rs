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

//! Per-backend (CPU) kernel registry for `Op::Custom`.
//!
//! Companion to [`rlx_ir::op_registry`]. The IR-level [`rlx_ir::OpExtension`]
//! covers shape inference + autodiff; this registry covers
//! *execution* on the CPU backend. Splitting them keeps `rlx-ir`
//! portable and lets a custom op honestly support a subset of
//! backends — attempting to compile an `Op::Custom` for a backend
//! whose kernel isn't registered is a hard error, not a silent no-op.
//!
//! ## API contract for downstream kernel authors
//!
//! - **One method, typed views in.** Each input arrives as a
//!   [`CpuTensorRef`] variant matching that input's declared dtype.
//!   The output is a [`CpuTensorMut`] matching the output dtype. No
//!   byte reinterpretation in user code.
//! - **Mixed-dtype inputs work directly.** A Sparse-LU op with
//!   `(F64 values, I32 col_idx, I32 row_ptr, F64 b)` gets each input
//!   as the right typed slice — no manual byte casts.
//! - **Contiguous, dense buffers from the arena.** Strided / broadcast
//!   inputs need to be materialized by the caller before reaching the
//!   kernel; the IR's `Op::Expand` / `Op::Transpose` already cover
//!   that.
//! - **`attrs` is opaque** — same `Vec<u8>` as the IR variant. Decode
//!   it however the kernel likes (typical: `bincode`, `bytemuck`, a
//!   hand-rolled struct cast).
//!
//! ## Multiple logical outputs
//!
//! `Op::Custom` produces a single tensor by design — the IR's `Node`
//! is one-shape-per-node. Ops that conceptually return multiple
//! outputs (LU returning L+U, eigendecomp returning λ+V) write a
//! *packed* output and the user follows the custom op with `Narrow`
//! to extract each logical output. Use
//! [`rlx_ir::Graph::custom_op_packed`] when registry-driven shape
//! inference isn't sufficient.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use rlx_ir::{DType, Shape};

// Why an enum, not generics? `CpuKernel` takes inputs of *mixed*
// dtypes (e.g. Sparse-LU has `(F64 values, I32 col_idx, I32 row_ptr,
// F64 b)`). A generic `CpuKernel<T>` couldn't express that — every
// input would have to share the same `T`. The enum-of-typed-views
// is the right shape for this contract; generics over `T: Pod`
// would only buy us the per-input case, which is what `as_*` /
// `expect_*` accessors already provide.
//
// One variant per `rlx_ir::DType`. The dispatcher in
// `thunk.rs::dispatch_custom_op` enumerates all of them — adding a
// dtype to `DType` requires adding a variant here and an arm
// there. Single source of truth for "what's wired."

macro_rules! dtype_variants {
    (
        $(
            $variant:ident => $rust_ty:ty,
            $as_method:ident, $as_mut_method:ident,
            $expect_method:ident, $expect_mut_method:ident,
        )*
    ) => {
        /// Read-only typed view of one input tensor handed to a [`CpuKernel`].
        /// The variant matches the input's declared dtype on the IR side.
        pub enum CpuTensorRef<'a> {
            $(
                $variant { data: &'a [$rust_ty], shape: &'a Shape },
            )*
        }

        /// Mutable typed view of the output tensor handed to a [`CpuKernel`].
        pub enum CpuTensorMut<'a> {
            $(
                $variant { data: &'a mut [$rust_ty], shape: &'a Shape },
            )*
        }

        impl<'a> CpuTensorRef<'a> {
            pub fn shape(&self) -> &Shape {
                match self {
                    $( Self::$variant { shape, .. } => shape, )*
                }
            }
            pub fn dtype(&self) -> DType { self.shape().dtype() }

            $(
                pub fn $as_method(&self) -> Option<&[$rust_ty]> {
                    if let Self::$variant { data, .. } = self { Some(data) } else { None }
                }
                pub fn $expect_method(&self, role: &str) -> Result<&[$rust_ty], String> {
                    self.$as_method().ok_or_else(|| format!(
                        "{role}: expected {:?}, got {:?}",
                        DType::$variant, self.dtype()))
                }
            )*
        }

        impl<'a> CpuTensorMut<'a> {
            pub fn shape(&self) -> &Shape {
                match self {
                    $( Self::$variant { shape, .. } => shape, )*
                }
            }
            pub fn dtype(&self) -> DType { self.shape().dtype() }

            $(
                pub fn $as_mut_method(self) -> Option<&'a mut [$rust_ty]> {
                    if let Self::$variant { data, .. } = self { Some(data) } else { None }
                }
                pub fn $expect_mut_method(self, role: &str) -> Result<&'a mut [$rust_ty], String> {
                    let dt = self.dtype();
                    self.$as_mut_method().ok_or_else(|| format!(
                        "{role}: expected {:?}, got {dt:?}", DType::$variant))
                }
            )*
        }
    };
}

// One row per DType. Bool is stored as `u8` on the wire (one byte
// per element, 0 = false / non-zero = true) — exposing it as a bool
// slice directly would be UB if any byte pattern other than 0/1
// landed there, which the IR doesn't guarantee.
dtype_variants! {
    F32  => f32,        as_f32,  as_f32_mut,  expect_f32,  expect_f32_mut,
    F64  => f64,        as_f64,  as_f64_mut,  expect_f64,  expect_f64_mut,
    F16  => half::f16,  as_f16,  as_f16_mut,  expect_f16,  expect_f16_mut,
    BF16 => half::bf16, as_bf16, as_bf16_mut, expect_bf16, expect_bf16_mut,
    I8   => i8,         as_i8,   as_i8_mut,   expect_i8,   expect_i8_mut,
    I16  => i16,        as_i16,  as_i16_mut,  expect_i16,  expect_i16_mut,
    I32  => i32,        as_i32,  as_i32_mut,  expect_i32,  expect_i32_mut,
    I64  => i64,        as_i64,  as_i64_mut,  expect_i64,  expect_i64_mut,
    U8   => u8,         as_u8,   as_u8_mut,   expect_u8,   expect_u8_mut,
    U32  => u32,        as_u32,  as_u32_mut,  expect_u32,  expect_u32_mut,
    Bool => u8,         as_bool, as_bool_mut, expect_bool, expect_bool_mut,
}

/// Trait a CPU kernel implements for one custom op. Registered under
/// the same `name` used in `Op::Custom` and `OpExtension::name`.
///
/// One method, typed views in. Match on the variants you support and
/// return `Err(...)` for anything else — the executor surfaces that
/// as a panic naming the op + dtype, so missing support fails loudly
/// instead of silently zeroing the output.
pub trait CpuKernel: Send + Sync {
    fn name(&self) -> &str;

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String>;
}

pub struct CpuKernelRegistry {
    kernels: RwLock<HashMap<String, Arc<dyn CpuKernel>>>,
}

impl CpuKernelRegistry {
    pub fn new() -> Self {
        Self {
            kernels: RwLock::new(HashMap::new()),
        }
    }

    /// Register a kernel. Re-registration replaces the previous entry
    /// and prints a one-line warning to stderr — silent overwrite has
    /// bitten us before, the warning is cheap.
    pub fn register(&self, k: Arc<dyn CpuKernel>) {
        let name = k.name().to_string();
        let mut g = self.kernels.write().unwrap();
        if g.contains_key(&name) {
            eprintln!(
                "rlx-cpu: CpuKernel '{name}' was already registered — \
                 replacing the previous entry"
            );
        }
        g.insert(name, k);
    }

    pub fn lookup(&self, name: &str) -> Option<Arc<dyn CpuKernel>> {
        self.kernels.read().unwrap().get(name).cloned()
    }
}

impl Default for CpuKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn global_cpu_kernels() -> &'static CpuKernelRegistry {
    static R: OnceLock<CpuKernelRegistry> = OnceLock::new();
    R.get_or_init(CpuKernelRegistry::new)
}

pub fn register_cpu_kernel(k: Arc<dyn CpuKernel>) {
    global_cpu_kernels().register(k);
}

pub fn lookup_cpu_kernel(name: &str) -> Option<Arc<dyn CpuKernel>> {
    global_cpu_kernels().lookup(name)
}
