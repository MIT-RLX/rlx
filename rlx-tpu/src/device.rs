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

//! TPU runtime context — process-global libtpu handle.
//!
//! Mirrors `rlx_rocm::device::rocm_context` and
//! `rlx_cuda::device::cuda_context`: lazily probes the runtime once,
//! caches the result, and hands out a `&'static` reference to callers.
//! `None` means "no TPU available on this host" — every public entry
//! point in the crate falls back to a clean panic with a clear error
//! message rather than silently mis-routing dispatch.
//!
//! On first probe we run, in order:
//!   1. `TpuRuntime::try_load` → dlopen libtpu / libpjrt_c_cpu, find
//!      `GetPjrtApi`, snapshot the fn pointer table.
//!   2. `PJRT_Plugin_Initialize` — required for libtpu before any
//!      other call. The CPU PJRT plugin treats this as a no-op but
//!      still requires the call.
//!   3. `PJRT_Client_Create` with no override options — picks the
//!      default platform (TPU on a TPU VM, CPU on the CPU plugin).
//!
//! If any of these steps fails we surface the PJRT error message
//! through `panic!` rather than silently downgrading to "no TPU
//! available" — failures here are diagnostic, not "missing hardware".

use std::sync::OnceLock;

use crate::libtpu::{
    PJRT_Client_Create_Args, PJRT_Plugin_Initialize_Args, PjrtClient, TpuRuntime, error_to_string,
};

/// Resolved-once TPU runtime + initialized PJRT client.
pub struct TpuContext {
    pub runtime: TpuRuntime,
    /// PJRT client handle — created via `Client_Create` after
    /// `Plugin_Initialize`. Always non-null when this struct exists;
    /// the construction path turns failures into panics.
    pub client: *mut PjrtClient,
}

// PJRT clients are documented as thread-safe; concurrent
// `Client_Compile` / `Buffer_FromHostBuffer` calls are explicitly
// supported by the C API. Same Send+Sync claim as rlx-cuda's
// CudaContext.
unsafe impl Send for TpuContext {}
unsafe impl Sync for TpuContext {}

/// Process-global TPU context. Returns `None` on hosts without a
/// libtpu install.
///
/// First call probes; subsequent calls hit the OnceLock cache.
pub fn tpu_context() -> Option<&'static TpuContext> {
    static CTX: OnceLock<Option<TpuContext>> = OnceLock::new();
    CTX.get_or_init(|| {
        let runtime = TpuRuntime::try_load()?;
        let client = init_client(&runtime);
        Some(TpuContext { runtime, client })
    })
    .as_ref()
}

/// Run `Plugin_Initialize` then `Client_Create`. Panics with the PJRT
/// error message on failure — these aren't silent-skip conditions.
fn init_client(runtime: &TpuRuntime) -> *mut PjrtClient {
    let fns = &runtime.fns;

    // 1. Plugin_Initialize. Required for libtpu (allocates the TPU
    //    plugin's runtime state); no-op on the CPU PJRT plugin but
    //    must still be called.
    let mut init_args = PJRT_Plugin_Initialize_Args {
        struct_size: std::mem::size_of::<PJRT_Plugin_Initialize_Args>(),
        extension_start: std::ptr::null_mut(),
    };
    let err = unsafe { (fns.plugin_initialize)(&mut init_args) };
    if !err.is_null() {
        let msg = unsafe { error_to_string(fns, err) };
        panic!("rlx-tpu: PJRT_Plugin_Initialize failed: {msg}");
    }

    // 2. Client_Create. We pass no create_options — the TPU plugin
    //    falls back to its built-in defaults (single host, all
    //    visible chips). For the CPU plugin this also picks the
    //    one-process default. KV callbacks are NULL: no
    //    multi-process bring-up.
    let mut create_args = PJRT_Client_Create_Args {
        struct_size: std::mem::size_of::<PJRT_Client_Create_Args>(),
        extension_start: std::ptr::null_mut(),
        create_options: std::ptr::null(),
        num_options: 0,
        kv_get_callback: std::ptr::null_mut(),
        kv_get_user_arg: std::ptr::null_mut(),
        kv_put_callback: std::ptr::null_mut(),
        kv_put_user_arg: std::ptr::null_mut(),
        client: std::ptr::null_mut(),
        kv_try_get_callback: std::ptr::null_mut(),
        kv_try_get_user_arg: std::ptr::null_mut(),
    };
    let err = unsafe { (fns.client_create)(&mut create_args) };
    if !err.is_null() {
        let msg = unsafe { error_to_string(fns, err) };
        panic!("rlx-tpu: PJRT_Client_Create failed: {msg}");
    }
    // `client` is written into the args struct by the plugin.
    let client = create_args.client;
    if client.is_null() {
        panic!(
            "rlx-tpu: PJRT_Client_Create returned NULL client without \
             setting an error — plugin contract violation."
        );
    }
    client
}
