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

/// Run a registered custom-op [`CpuKernel`] against **host byte buffers** — the
/// generic host-fallback a GPU backend uses for a custom op it has no native
/// kernel for. The `collective.*` ops are the motivating case: they are
/// host/transport ops with no device kernel, so every backend runs them by
/// staging the operands to host and calling the one registered CPU kernel here.
///
/// All operands are treated as `F32` (the dtype of the ops that take this path).
/// Each `(&[u8], &Shape)` input is reinterpreted as an `f32` slice and handed to
/// the kernel as a [`CpuTensorRef::F32`]; the output the same. Returns `Err` if
/// no kernel is registered under `name`, if a buffer is not 4-byte aligned or a
/// multiple of 4 bytes, or if the kernel itself errors.
pub fn run_f32_custom_op_host(
    name: &str,
    inputs: &[(&[u8], &Shape)],
    out: (&mut [u8], &Shape),
    attrs: &[u8],
) -> Result<(), String> {
    let kernel = lookup_cpu_kernel(name)
        .ok_or_else(|| format!("no CpuKernel registered for custom op '{name}'"))?;

    fn as_f32<'a>(bytes: &'a [u8], role: &str) -> Result<&'a [f32], String> {
        if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
            || !bytes.len().is_multiple_of(4)
        {
            return Err(format!("{role}: host buffer not f32-aligned/sized"));
        }
        // SAFETY: alignment + length checked; every bit pattern is a valid f32.
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) })
    }

    let refs: Vec<CpuTensorRef<'_>> = inputs
        .iter()
        .enumerate()
        .map(|(i, &(b, sh))| {
            as_f32(b, &format!("{name} input {i}"))
                .map(|data| CpuTensorRef::F32 { data, shape: sh })
        })
        .collect::<Result<_, _>>()?;

    let (out_bytes, out_shape) = out;
    if !(out_bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<f32>())
        || !out_bytes.len().is_multiple_of(4)
    {
        return Err(format!("{name} output: host buffer not f32-aligned/sized"));
    }
    let out_len = out_bytes.len() / 4;
    // SAFETY: alignment + length checked above.
    let out_data =
        unsafe { std::slice::from_raw_parts_mut(out_bytes.as_mut_ptr() as *mut f32, out_len) };
    let out_view = CpuTensorMut::F32 {
        data: out_data,
        shape: out_shape,
    };

    kernel.execute(&refs, out_view, attrs)
}

/// Reinterpret `bytes` as a `&[T]`, checking alignment + size. SAFETY caller:
/// every bit pattern of the raw scalar types we use here (integers/floats/half)
/// is a valid value, so the transmute is sound once alignment + length hold.
fn as_typed<'a, T>(bytes: &'a [u8], role: &str) -> Result<&'a [T], String> {
    let sz = std::mem::size_of::<T>();
    if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>())
        || !bytes.len().is_multiple_of(sz)
    {
        return Err(format!("{role}: host buffer not {sz}-byte aligned/sized"));
    }
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / sz) })
}

fn as_typed_mut<'a, T>(bytes: &'a mut [u8], role: &str) -> Result<&'a mut [T], String> {
    let sz = std::mem::size_of::<T>();
    if !(bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>())
        || !bytes.len().is_multiple_of(sz)
    {
        return Err(format!("{role}: host buffer not {sz}-byte aligned/sized"));
    }
    let n = bytes.len() / sz;
    Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, n) })
}

fn cpu_ref_from_bytes<'a>(
    b: &'a [u8],
    sh: &'a Shape,
    role: &str,
) -> Result<CpuTensorRef<'a>, String> {
    Ok(match sh.dtype() {
        DType::F32 => CpuTensorRef::F32 {
            data: as_typed(b, role)?,
            shape: sh,
        },
        DType::F64 => CpuTensorRef::F64 {
            data: as_typed(b, role)?,
            shape: sh,
        },
        DType::F16 => CpuTensorRef::F16 {
            data: as_typed(b, role)?,
            shape: sh,
        },
        DType::BF16 => CpuTensorRef::BF16 {
            data: as_typed(b, role)?,
            shape: sh,
        },
        DType::I8 => CpuTensorRef::I8 {
            data: as_typed(b, role)?,
            shape: sh,
        },
        DType::I16 => CpuTensorRef::I16 {
            data: as_typed(b, role)?,
            shape: sh,
        },
        DType::I32 => CpuTensorRef::I32 {
            data: as_typed(b, role)?,
            shape: sh,
        },
        DType::I64 => CpuTensorRef::I64 {
            data: as_typed(b, role)?,
            shape: sh,
        },
        DType::U8 => CpuTensorRef::U8 {
            data: as_typed(b, role)?,
            shape: sh,
        },
        DType::U32 => CpuTensorRef::U32 {
            data: as_typed(b, role)?,
            shape: sh,
        },
        DType::Bool => CpuTensorRef::Bool {
            data: as_typed(b, role)?,
            shape: sh,
        },
        other => {
            return Err(format!(
                "{role}: unsupported dtype {other:?} for host custom-op"
            ));
        }
    })
}

fn cpu_mut_from_bytes<'a>(b: &'a mut [u8], sh: &'a Shape) -> Result<CpuTensorMut<'a>, String> {
    Ok(match sh.dtype() {
        DType::F32 => CpuTensorMut::F32 {
            data: as_typed_mut(b, "output")?,
            shape: sh,
        },
        DType::F64 => CpuTensorMut::F64 {
            data: as_typed_mut(b, "output")?,
            shape: sh,
        },
        DType::F16 => CpuTensorMut::F16 {
            data: as_typed_mut(b, "output")?,
            shape: sh,
        },
        DType::BF16 => CpuTensorMut::BF16 {
            data: as_typed_mut(b, "output")?,
            shape: sh,
        },
        DType::I8 => CpuTensorMut::I8 {
            data: as_typed_mut(b, "output")?,
            shape: sh,
        },
        DType::I16 => CpuTensorMut::I16 {
            data: as_typed_mut(b, "output")?,
            shape: sh,
        },
        DType::I32 => CpuTensorMut::I32 {
            data: as_typed_mut(b, "output")?,
            shape: sh,
        },
        DType::I64 => CpuTensorMut::I64 {
            data: as_typed_mut(b, "output")?,
            shape: sh,
        },
        DType::U8 => CpuTensorMut::U8 {
            data: as_typed_mut(b, "output")?,
            shape: sh,
        },
        DType::U32 => CpuTensorMut::U32 {
            data: as_typed_mut(b, "output")?,
            shape: sh,
        },
        DType::Bool => CpuTensorMut::Bool {
            data: as_typed_mut(b, "output")?,
            shape: sh,
        },
        other => {
            return Err(format!(
                "output: unsupported dtype {other:?} for host custom-op"
            ));
        }
    })
}

/// Dtype-aware sibling of [`run_f32_custom_op_host`]: run a registered custom-op
/// [`CpuKernel`] against **host byte buffers**, reinterpreting each operand per
/// its `Shape`'s dtype (so integer ops like `onnx.Mod` on I64, or Bool masks,
/// are handled correctly — not forced to F32). This is the generic host-delegate
/// a GPU backend (Metal / MLX / wgpu) uses for any `onnx.*` custom op it has no
/// native kernel for, staging operands to host over unified/mapped memory.
pub fn run_custom_op_host(
    name: &str,
    inputs: &[(&[u8], &Shape)],
    out: (&mut [u8], &Shape),
    attrs: &[u8],
) -> Result<(), String> {
    let kernel = lookup_cpu_kernel(name)
        .ok_or_else(|| format!("no CpuKernel registered for custom op '{name}'"))?;
    let refs: Vec<CpuTensorRef<'_>> = inputs
        .iter()
        .enumerate()
        .map(|(i, &(b, sh))| cpu_ref_from_bytes(b, sh, &format!("{name} input {i}")))
        .collect::<Result<_, _>>()?;
    let (out_bytes, out_shape) = out;
    let out_view = cpu_mut_from_bytes(out_bytes, out_shape)?;
    kernel.execute(&refs, out_view, attrs)
}
