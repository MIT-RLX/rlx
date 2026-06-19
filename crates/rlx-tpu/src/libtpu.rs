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

//! libtpu / PJRT C API FFI surface.
//!
//! Modern `libtpu.so` exposes itself as a PJRT plugin with a single
//! public entry point: `GetPjrtApi() -> *const PJRT_Api`. The returned
//! struct is a function-pointer table over the PJRT C API
//! (`xla/pjrt/c/pjrt_c_api.h`).
//!
//! We mirror only the subset we call. PJRT uses struct-size-prefixed
//! "args" structs per call, which gives forward compatibility: the
//! caller fills in `struct_size` to indicate the version they were
//! compiled against, and the plugin reads only the fields it knows
//! about. We therefore declare the args structs with the *current*
//! field set; if a plugin advertises a smaller `PJRT_Api.struct_size`
//! we'd technically need to gate field access — in practice the TPU
//! plugin tracks head and the CPU plugin we use for validation does
//! the same, so the struct definitions here are stable.
//!
//! ## What's loaded
//!
//! * Plugin lifecycle: `PJRT_Plugin_Initialize`
//! * Client lifecycle: `Client_Create`, `Client_Destroy`
//! * Compile:        `Client_Compile`
//! * Buffers:        `Client_BufferFromHostBuffer`,
//!                   `Buffer_ToHostBuffer`, `Buffer_Destroy`,
//!                   `Buffer_Dimensions`, `Buffer_ElementType`
//! * Execute:        `LoadedExecutable_Execute`,
//!                   `LoadedExecutable_Destroy`
//! * Async events:   `Event_Await`, `Event_IsReady`, `Event_Error`,
//!                   `Event_Destroy`
//! * Errors:         `Error_Message`, `Error_GetCode`, `Error_Destroy`
//!
//! Everything else (profiler, topology, named values, KV stores,
//! collective ops) is not yet wired — easy to extend; not on the
//! single-host inference path.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use libloading::Library;
use std::ffi::c_void;

// ── Opaque handle types ──────────────────────────────────────────
//
// Each is a forward-declared struct in pjrt_c_api.h. Rust uses raw
// pointers; we never dereference these from Rust.

#[repr(C)]
pub struct PjrtClient {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PjrtError {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PjrtLoadedExecutable {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PjrtExecutable {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PjrtBuffer {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PjrtEvent {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PjrtDevice {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PjrtMemory {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PjrtTopologyDescription {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PjrtKeyValueGetCallback {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PjrtKeyValuePutCallback {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PjrtSerializedExecutable {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PjrtExecuteContext {
    _private: [u8; 0],
}
#[repr(C)]
pub struct PjrtCopyToDeviceStream {
    _private: [u8; 0],
}

// ── Buffer types (mirrors PJRT_Buffer_Type) ──────────────────────

pub const PJRT_BUFFER_TYPE_INVALID: i32 = 0;
pub const PJRT_BUFFER_TYPE_PRED: i32 = 1; // bool
pub const PJRT_BUFFER_TYPE_S8: i32 = 2;
pub const PJRT_BUFFER_TYPE_S16: i32 = 3;
pub const PJRT_BUFFER_TYPE_S32: i32 = 4;
pub const PJRT_BUFFER_TYPE_S64: i32 = 5;
pub const PJRT_BUFFER_TYPE_U8: i32 = 6;
pub const PJRT_BUFFER_TYPE_U16: i32 = 7;
pub const PJRT_BUFFER_TYPE_U32: i32 = 8;
pub const PJRT_BUFFER_TYPE_U64: i32 = 9;
pub const PJRT_BUFFER_TYPE_F16: i32 = 10;
pub const PJRT_BUFFER_TYPE_F32: i32 = 11;
pub const PJRT_BUFFER_TYPE_F64: i32 = 12;
pub const PJRT_BUFFER_TYPE_BF16: i32 = 13;
pub const PJRT_BUFFER_TYPE_C64: i32 = 14;
pub const PJRT_BUFFER_TYPE_C128: i32 = 15;
pub const PJRT_BUFFER_TYPE_F8E5M2: i32 = 16;
pub const PJRT_BUFFER_TYPE_F8E4M3FN: i32 = 17;

// Host buffer semantics (PJRT_HostBufferSemantics):
//   0 — kImmutableOnlyDuringCall (caller may free after the call returns)
//   1 — kImmutableUntilTransferCompletes
//   2 — kImmutableZeroCopy
//   3 — kMutableZeroCopy
pub const PJRT_HOST_BUFFER_SEMANTICS_IMMUTABLE_ONLY_DURING_CALL: i32 = 0;
pub const PJRT_HOST_BUFFER_SEMANTICS_IMMUTABLE_UNTIL_TRANSFER_COMPLETES: i32 = 1;
pub const PJRT_HOST_BUFFER_SEMANTICS_IMMUTABLE_ZERO_COPY: i32 = 2;
pub const PJRT_HOST_BUFFER_SEMANTICS_MUTABLE_ZERO_COPY: i32 = 3;

// Program format: "hlo" (binary HLO module proto), "mlir" (StableHLO),
// or "hlo_with_config".
pub const PJRT_PROGRAM_FORMAT_HLO: &[u8] = b"hlo";
pub const PJRT_PROGRAM_FORMAT_MLIR: &[u8] = b"mlir";

// Error codes (mirrors PJRT_Error_Code → absl::StatusCode subset).
pub const PJRT_ERROR_CODE_OK: i32 = 0;
pub const PJRT_ERROR_CODE_CANCELLED: i32 = 1;
pub const PJRT_ERROR_CODE_UNKNOWN: i32 = 2;
pub const PJRT_ERROR_CODE_INVALID_ARGUMENT: i32 = 3;
pub const PJRT_ERROR_CODE_DEADLINE_EXCEEDED: i32 = 4;
pub const PJRT_ERROR_CODE_NOT_FOUND: i32 = 5;
pub const PJRT_ERROR_CODE_INTERNAL: i32 = 13;
pub const PJRT_ERROR_CODE_UNIMPLEMENTED: i32 = 12;

// ── Args structs (subset we call) ────────────────────────────────
//
// Field-for-field mirror of pjrt_c_api.h. Each has an explicit
// `struct_size` cell; populate with `std::mem::size_of::<Args>()`
// before calling. `extension_start` is reserved for future PJRT
// extensions and we always set it to NULL.
//
// We only declare structs we actually fill in. New ones land
// alongside their first call site.

#[repr(C)]
pub struct PJRT_Plugin_Initialize_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
}

#[repr(C)]
pub struct PJRT_Error_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub error: *mut PjrtError,
}

#[repr(C)]
pub struct PJRT_Error_Message_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub error: *const PjrtError,
    pub message: *const u8,
    pub message_size: usize,
}

#[repr(C)]
pub struct PJRT_Error_GetCode_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub error: *const PjrtError,
    pub code: i32,
}

/// `PJRT_NamedValue` mirrors the C struct used to pass plugin-specific
/// key/value attributes (e.g. `"max_inflight_computations": 4`) to
/// `Client_Create`. We never construct one today — TPU works fine with
/// no overrides — but the args struct below carries a pointer to an
/// array, so we declare the type for completeness.
#[repr(C)]
pub struct PJRT_NamedValue {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub name: *const u8,
    pub name_size: usize,
    pub value_type: i32,
    pub string_value: *const u8,
    pub string_value_size: usize,
    pub int64_value: i64,
    pub int64_array_value: *const i64,
    pub value_size: usize,
    pub float_value: f32,
    pub bool_value: bool,
}

#[repr(C)]
pub struct PJRT_Client_Create_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub create_options: *const PJRT_NamedValue,
    pub num_options: usize,
    pub kv_get_callback: *mut c_void,
    pub kv_get_user_arg: *mut c_void,
    pub kv_put_callback: *mut c_void,
    pub kv_put_user_arg: *mut c_void,
    pub client: *mut PjrtClient,
    // Added in PJRT API ≥ 0.59 (kKeyValueTryGetCallback). Plugin
    // checks the args struct size against this trailing pair.
    pub kv_try_get_callback: *mut c_void,
    pub kv_try_get_user_arg: *mut c_void,
}

#[repr(C)]
pub struct PJRT_Client_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub client: *mut PjrtClient,
}

#[repr(C)]
pub struct PJRT_Program {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub code: *mut u8,
    pub code_size: usize,
    pub format: *const u8,
    pub format_size: usize,
}

#[repr(C)]
pub struct PJRT_Client_Compile_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub client: *mut PjrtClient,
    pub program: *const PJRT_Program,
    /// Serialized `xla.CompileOptionsProto`. Empty is OK — picks
    /// platform defaults (single replica, single partition).
    pub compile_options: *const u8,
    pub compile_options_size: usize,
    pub executable: *mut PjrtLoadedExecutable,
}

#[repr(C)]
pub struct PJRT_LoadedExecutable_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub executable: *mut PjrtLoadedExecutable,
}

#[repr(C)]
pub struct PJRT_Client_BufferFromHostBuffer_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub client: *mut PjrtClient,
    pub data: *const c_void,
    pub type_: i32,
    pub dims: *const i64,
    pub num_dims: usize,
    pub byte_strides: *const i64,
    pub num_byte_strides: usize,
    pub host_buffer_semantics: i32,
    pub device: *mut PjrtDevice,
    pub memory: *mut PjrtMemory,
    pub device_layout: *mut c_void,
    pub done_with_host_buffer: *mut PjrtEvent,
    pub buffer: *mut PjrtBuffer,
}

#[repr(C)]
pub struct PJRT_Buffer_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub buffer: *mut PjrtBuffer,
}

#[repr(C)]
pub struct PJRT_Buffer_ToHostBuffer_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub src: *mut PjrtBuffer,
    pub host_layout: *mut c_void,
    pub dst: *mut c_void,
    pub dst_size: usize,
    pub event: *mut PjrtEvent,
}

#[repr(C)]
pub struct PJRT_Buffer_Dimensions_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub buffer: *mut PjrtBuffer,
    pub dims: *const i64,
    pub num_dims: usize,
}

#[repr(C)]
pub struct PJRT_Buffer_ElementType_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub buffer: *mut PjrtBuffer,
    pub type_: i32,
}

#[repr(C)]
pub struct PJRT_ExecuteOptions {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub send_callbacks: *mut c_void,
    pub recv_callbacks: *mut c_void,
    pub num_send_ops: usize,
    pub num_recv_ops: usize,
    pub launch_id: i32,
    // 4 bytes of padding here to 8-align the next pointer (repr(C)).
    pub non_donatable_input_indices: *const i64,
    pub num_non_donatable_input_indices: usize,
    pub context: *mut PjrtExecuteContext,
    // Trailing fields added in newer PJRT API. We never set them
    // (single-host inference, no multi-slice), but the plugin checks
    // struct_size against the offset of the last field, so they must
    // be present.
    pub call_location: *const u8,
    pub num_tasks: usize,
    pub task_ids: *mut i32,
    pub incarnation_ids: *mut i64,
    pub multi_slice_config: *mut c_void,
}

#[repr(C)]
pub struct PJRT_LoadedExecutable_Execute_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub executable: *mut PjrtLoadedExecutable,
    pub options: *const PJRT_ExecuteOptions,
    /// 2-D ragged: `argument_lists[device][argi]`. For single-device
    /// dispatch we always have `num_devices = 1`.
    pub argument_lists: *const *const *mut PjrtBuffer,
    pub num_devices: usize,
    pub num_args: usize,
    /// Likewise 2-D: `output_lists[device][outi]`. Caller must
    /// pre-allocate the outer pointer arrays; the plugin fills the
    /// inner buffer pointers.
    pub output_lists: *const *mut *mut PjrtBuffer,
    pub device_complete_events: *mut *mut PjrtEvent,
    pub execute_device: *mut PjrtDevice,
}

#[repr(C)]
pub struct PJRT_Event_Destroy_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub event: *mut PjrtEvent,
}

#[repr(C)]
pub struct PJRT_Event_IsReady_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub event: *mut PjrtEvent,
    pub is_ready: bool,
}

#[repr(C)]
pub struct PJRT_Event_Await_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub event: *mut PjrtEvent,
}

#[repr(C)]
pub struct PJRT_Event_Error_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub event: *mut PjrtEvent,
}

#[repr(C)]
pub struct PJRT_Client_AddressableDevices_Args {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub client: *mut PjrtClient,
    pub addressable_devices: *const *mut PjrtDevice,
    pub num_addressable_devices: usize,
}

// ── PJRT_Api vtable ──────────────────────────────────────────────
//
// Mirrors the upstream struct layout (xla/pjrt/c/pjrt_c_api.h, PJRT
// API ≥ 0.55 — what JAX 0.4.30+ ships and what the TPU plugin tracks).
// Every function pointer takes a pointer to a per-call args struct
// and returns `*mut PjrtError` (NULL on success). We declare every
// slot the upstream header has up through the calls we actually make
// (LoadedExecutable_Execute and the trailing Buffer_* group), keeping
// unused ones as opaque `*mut c_void` for layout fidelity. Tail slots
// past Buffer_OpaqueDeviceMemoryDataPointer are not declared; the
// struct_size cell at the head tells the plugin we don't read further,
// and we rely on PJRT's documented forward compat.

pub type PfnError = unsafe extern "C" fn(*mut c_void) -> *mut PjrtError;
pub type PfnVoid = unsafe extern "C" fn(*mut c_void);

// `PJRT_Api_Version` is itself a struct-size-prefixed PJRT struct,
// not a bare pair of ints. Missing the leading `struct_size` +
// `extension_start` cells shifts every fn-pointer slot in `PjrtApi`
// by 16 bytes — which manifests as a SIGSEGV on the first call,
// usually with the plugin logging
//   "Unexpected PJRT_Error_Message_Args size: expected 40, got 16"
// because what we think is `plugin_initialize` lands on
// `PJRT_Error_Message`.
#[repr(C)]
pub struct PjrtApiVersion {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub major_version: i32,
    pub minor_version: i32,
}

#[repr(C)]
pub struct PjrtApi {
    pub struct_size: usize,
    pub extension_start: *mut c_void,
    pub pjrt_api_version: PjrtApiVersion,

    // ── Error API (3) ────────────────────────────────────────────
    pub PJRT_Error_Destroy: *mut c_void,
    pub PJRT_Error_Message: *mut c_void,
    pub PJRT_Error_GetCode: *mut c_void,

    // ── Plugin API (2) ───────────────────────────────────────────
    pub PJRT_Plugin_Initialize: *mut c_void,
    pub PJRT_Plugin_Attributes: *mut c_void,

    // ── Async events (5) ─────────────────────────────────────────
    pub PJRT_Event_Destroy: *mut c_void,
    pub PJRT_Event_IsReady: *mut c_void,
    pub PJRT_Event_Error: *mut c_void,
    pub PJRT_Event_Await: *mut c_void,
    pub PJRT_Event_OnReady: *mut c_void,

    // ── Client API (13) ──────────────────────────────────────────
    pub PJRT_Client_Create: *mut c_void,
    pub PJRT_Client_Destroy: *mut c_void,
    pub PJRT_Client_PlatformName: *mut c_void,
    pub PJRT_Client_ProcessIndex: *mut c_void,
    pub PJRT_Client_PlatformVersion: *mut c_void,
    pub PJRT_Client_Devices: *mut c_void,
    pub PJRT_Client_AddressableDevices: *mut c_void,
    pub PJRT_Client_LookupDevice: *mut c_void,
    pub PJRT_Client_LookupAddressableDevice: *mut c_void,
    pub PJRT_Client_AddressableMemories: *mut c_void,
    pub PJRT_Client_Compile: *mut c_void,
    pub PJRT_Client_DefaultDeviceAssignment: *mut c_void,
    pub PJRT_Client_BufferFromHostBuffer: *mut c_void,

    // ── DeviceDescription API (6) ────────────────────────────────
    pub PJRT_DeviceDescription_Id: *mut c_void,
    pub PJRT_DeviceDescription_ProcessIndex: *mut c_void,
    pub PJRT_DeviceDescription_Attributes: *mut c_void,
    pub PJRT_DeviceDescription_Kind: *mut c_void,
    pub PJRT_DeviceDescription_DebugString: *mut c_void,
    pub PJRT_DeviceDescription_ToString: *mut c_void,

    // ── Device API (6) ───────────────────────────────────────────
    pub PJRT_Device_GetDescription: *mut c_void,
    pub PJRT_Device_IsAddressable: *mut c_void,
    pub PJRT_Device_LocalHardwareId: *mut c_void,
    pub PJRT_Device_AddressableMemories: *mut c_void,
    pub PJRT_Device_DefaultMemory: *mut c_void,
    pub PJRT_Device_MemoryStats: *mut c_void,

    // ── Memory API (5) ───────────────────────────────────────────
    pub PJRT_Memory_Id: *mut c_void,
    pub PJRT_Memory_Kind: *mut c_void,
    pub PJRT_Memory_DebugString: *mut c_void,
    pub PJRT_Memory_ToString: *mut c_void,
    pub PJRT_Memory_AddressableByDevices: *mut c_void,

    // ── Executable API (10) ──────────────────────────────────────
    pub PJRT_Executable_Destroy: *mut c_void,
    pub PJRT_Executable_Name: *mut c_void,
    pub PJRT_Executable_NumReplicas: *mut c_void,
    pub PJRT_Executable_NumPartitions: *mut c_void,
    pub PJRT_Executable_NumOutputs: *mut c_void,
    pub PJRT_Executable_SizeOfGeneratedCodeInBytes: *mut c_void,
    pub PJRT_Executable_GetCostAnalysis: *mut c_void,
    pub PJRT_Executable_OutputMemoryKinds: *mut c_void,
    pub PJRT_Executable_OptimizedProgram: *mut c_void,
    pub PJRT_Executable_Serialize: *mut c_void,

    // ── LoadedExecutable API (8) ─────────────────────────────────
    pub PJRT_LoadedExecutable_Destroy: *mut c_void,
    pub PJRT_LoadedExecutable_GetExecutable: *mut c_void,
    pub PJRT_LoadedExecutable_AddressableDevices: *mut c_void,
    pub PJRT_LoadedExecutable_Delete: *mut c_void,
    pub PJRT_LoadedExecutable_IsDeleted: *mut c_void,
    pub PJRT_LoadedExecutable_Execute: *mut c_void,
    pub PJRT_Executable_DeserializeAndLoad: *mut c_void,
    pub PJRT_LoadedExecutable_Fingerprint: *mut c_void,

    // ── Buffer API (19) ──────────────────────────────────────────
    pub PJRT_Buffer_Destroy: *mut c_void,
    pub PJRT_Buffer_ElementType: *mut c_void,
    pub PJRT_Buffer_Dimensions: *mut c_void,
    pub PJRT_Buffer_UnpaddedDimensions: *mut c_void,
    pub PJRT_Buffer_DynamicDimensionIndices: *mut c_void,
    pub PJRT_Buffer_GetMemoryLayout: *mut c_void,
    pub PJRT_Buffer_OnDeviceSizeInBytes: *mut c_void,
    pub PJRT_Buffer_Device: *mut c_void,
    pub PJRT_Buffer_Memory: *mut c_void,
    pub PJRT_Buffer_Delete: *mut c_void,
    pub PJRT_Buffer_IsDeleted: *mut c_void,
    pub PJRT_Buffer_CopyToDevice: *mut c_void,
    pub PJRT_Buffer_ToHostBuffer: *mut c_void,
    pub PJRT_Buffer_IsOnCpu: *mut c_void,
    pub PJRT_Buffer_ReadyEvent: *mut c_void,
    pub PJRT_Buffer_UnsafePointer: *mut c_void,
    pub PJRT_Buffer_IncreaseExternalReferenceCount: *mut c_void,
    pub PJRT_Buffer_DecreaseExternalReferenceCount: *mut c_void,
    pub PJRT_Buffer_OpaqueDeviceMemoryDataPointer: *mut c_void,
}

/// `extern "C" const PJRT_Api* GetPjrtApi();` — the libtpu / libpjrt
/// entry point. Resolved by name once at process start.
pub type GetPjrtApiFn = unsafe extern "C" fn() -> *const PjrtApi;

// ── Loader ──────────────────────────────────────────────────────
//
// libtpu.so search order (matches what JAX does):
//   1. $LIBTPU_PATH (explicit override; may also point at a
//      libpjrt_c_cpu.so for off-TPU validation)
//   2. dlopen("libtpu.so")            — Linux system loader path
//   3. dlopen("libpjrt_c_cpu.so")     — XLA CPU plugin fallback
//   4. dlopen("libtpu.dylib")         — present only for symmetry
//
// We deliberately *don't* probe the Python `libtpu` site-packages
// path. Picking up libtpu through Python requires interpreting
// PYTHONPATH which gives a worse error story than just asking the
// user to set `LIBTPU_PATH=...`.

const LIBTPU_NAMES: &[&str] = &[
    "libtpu.so",
    "libpjrt_c_cpu.so", // CPU PJRT plugin — useful for non-TPU
    // validation in Docker / CI.
    "libtpu.dylib",
];

/// Loaded-and-resolved libtpu vtable. Held by [`crate::device::TpuContext`].
pub struct TpuRuntime {
    /// Keep the library alive — function pointers in `api` borrow
    /// from this object.
    _lib: Library,
    pub api: *const PjrtApi,
    /// Resolved function pointers. We resolve by reading the vtable
    /// fields rather than by symbol lookup — the C API guarantees
    /// the function pointers are stable for the lifetime of the
    /// process.
    pub fns: PjrtFns,
}

/// Direct-call function pointers, with concrete signatures. Filled
/// in once when the runtime loads. Going through this struct rather
/// than `(*api).PJRT_Foo` everywhere keeps unsafe transmutes localized
/// to one place.
pub struct PjrtFns {
    pub plugin_initialize: unsafe extern "C" fn(*mut PJRT_Plugin_Initialize_Args) -> *mut PjrtError,
    pub error_destroy: unsafe extern "C" fn(*mut PJRT_Error_Destroy_Args),
    pub error_message: unsafe extern "C" fn(*mut PJRT_Error_Message_Args),
    pub error_get_code: unsafe extern "C" fn(*mut PJRT_Error_GetCode_Args) -> *mut PjrtError,
    pub client_create: unsafe extern "C" fn(*mut PJRT_Client_Create_Args) -> *mut PjrtError,
    pub client_destroy: unsafe extern "C" fn(*mut PJRT_Client_Destroy_Args) -> *mut PjrtError,
    pub client_compile: unsafe extern "C" fn(*mut PJRT_Client_Compile_Args) -> *mut PjrtError,
    pub client_addressable_devices:
        unsafe extern "C" fn(*mut PJRT_Client_AddressableDevices_Args) -> *mut PjrtError,
    pub client_buffer_from_host_buffer:
        unsafe extern "C" fn(*mut PJRT_Client_BufferFromHostBuffer_Args) -> *mut PjrtError,
    pub buffer_destroy: unsafe extern "C" fn(*mut PJRT_Buffer_Destroy_Args) -> *mut PjrtError,
    pub buffer_to_host_buffer:
        unsafe extern "C" fn(*mut PJRT_Buffer_ToHostBuffer_Args) -> *mut PjrtError,
    pub buffer_dimensions: unsafe extern "C" fn(*mut PJRT_Buffer_Dimensions_Args) -> *mut PjrtError,
    pub buffer_element_type:
        unsafe extern "C" fn(*mut PJRT_Buffer_ElementType_Args) -> *mut PjrtError,
    pub loaded_executable_destroy:
        unsafe extern "C" fn(*mut PJRT_LoadedExecutable_Destroy_Args) -> *mut PjrtError,
    pub loaded_executable_execute:
        unsafe extern "C" fn(*mut PJRT_LoadedExecutable_Execute_Args) -> *mut PjrtError,
    pub event_destroy: unsafe extern "C" fn(*mut PJRT_Event_Destroy_Args) -> *mut PjrtError,
    pub event_await: unsafe extern "C" fn(*mut PJRT_Event_Await_Args) -> *mut PjrtError,
    pub event_error: unsafe extern "C" fn(*mut PJRT_Event_Error_Args) -> *mut PjrtError,
    pub event_is_ready: unsafe extern "C" fn(*mut PJRT_Event_IsReady_Args) -> *mut PjrtError,
}

// PJRT_Api is fundamentally process-global: function pointers don't
// carry hidden per-thread state. Same Send-claim shape as the
// rlx-cuda CudaRuntime and rlx-rocm RocmRuntime use.
unsafe impl Send for TpuRuntime {}
unsafe impl Sync for TpuRuntime {}

impl TpuRuntime {
    /// Try to load `libtpu.so` and resolve the PJRT vtable. Returns
    /// `None` if the loader can't find a compatible plugin.
    pub fn try_load() -> Option<Self> {
        let env_path = std::env::var("LIBTPU_PATH").ok();
        let candidates: Vec<String> = env_path
            .into_iter()
            .chain(LIBTPU_NAMES.iter().map(|s| s.to_string()))
            .collect();

        for name in candidates {
            // SAFETY: dlopen of a system library by name. libloading
            // wraps it; no constructors run on load (libtpu's init
            // is lazy via Plugin_Initialize).
            let lib = match unsafe { Library::new(&name) } {
                Ok(l) => l,
                Err(_) => continue,
            };

            let get_api: libloading::Symbol<GetPjrtApiFn> =
                match unsafe { lib.get(b"GetPjrtApi\0") } {
                    Ok(s) => s,
                    Err(_) => continue,
                };

            // SAFETY: GetPjrtApi is documented as a pure
            // process-global accessor; no preconditions.
            let api = unsafe { get_api() };
            if api.is_null() {
                continue;
            }
            // `get_api` is a `libloading::Symbol` borrow against `lib`;
            // we no longer need it, but the borrow lifetime ends at
            // end-of-scope automatically, so just shadowing is enough
            // to release it before we move `lib` into the struct below.
            let _ = get_api;

            // SAFETY: We just verified `api` is non-null; the
            // function pointer fields are filled by the plugin and
            // stable for the lifetime of the loaded library.
            let fns = unsafe { resolve_fns(api) };
            return Some(TpuRuntime {
                _lib: lib,
                api,
                fns,
            });
        }
        None
    }
}

/// Resolve the function pointers we need from a non-null `*const PjrtApi`.
///
/// SAFETY: caller must guarantee `api` is non-null and points at a
/// valid PJRT_Api struct whose function-pointer fields are populated.
// Each `cast!` field is declared as `*mut c_void` for layout fidelity;
// we transmute to the typed function-pointer at use. The destination
// type is inferred from the surrounding struct field, which is exactly
// what makes adding `<*mut c_void, T>` annotations per-call-site
// redundant — every cast! call is one-shot FFI plumbing.
#[allow(clippy::missing_transmute_annotations)]
unsafe fn resolve_fns(api: *const PjrtApi) -> PjrtFns {
    macro_rules! cast {
        ($field:expr) => {
            unsafe { std::mem::transmute($field) }
        };
    }
    let a = unsafe { &*api };
    PjrtFns {
        plugin_initialize: cast!(a.PJRT_Plugin_Initialize),
        error_destroy: cast!(a.PJRT_Error_Destroy),
        error_message: cast!(a.PJRT_Error_Message),
        error_get_code: cast!(a.PJRT_Error_GetCode),
        client_create: cast!(a.PJRT_Client_Create),
        client_destroy: cast!(a.PJRT_Client_Destroy),
        client_compile: cast!(a.PJRT_Client_Compile),
        client_addressable_devices: cast!(a.PJRT_Client_AddressableDevices),
        client_buffer_from_host_buffer: cast!(a.PJRT_Client_BufferFromHostBuffer),
        buffer_destroy: cast!(a.PJRT_Buffer_Destroy),
        buffer_to_host_buffer: cast!(a.PJRT_Buffer_ToHostBuffer),
        buffer_dimensions: cast!(a.PJRT_Buffer_Dimensions),
        buffer_element_type: cast!(a.PJRT_Buffer_ElementType),
        loaded_executable_destroy: cast!(a.PJRT_LoadedExecutable_Destroy),
        loaded_executable_execute: cast!(a.PJRT_LoadedExecutable_Execute),
        event_destroy: cast!(a.PJRT_Event_Destroy),
        event_await: cast!(a.PJRT_Event_Await),
        event_error: cast!(a.PJRT_Event_Error),
        event_is_ready: cast!(a.PJRT_Event_IsReady),
    }
}

// ── Error helpers ────────────────────────────────────────────────

/// Read an error message off a non-null `*mut PjrtError` and destroy
/// the error. Returns the message; the error pointer is invalidated.
///
/// SAFETY: `err` must be non-null. After this call it is destroyed.
pub unsafe fn error_to_string(fns: &PjrtFns, err: *mut PjrtError) -> String {
    let mut msg_args = PJRT_Error_Message_Args {
        struct_size: std::mem::size_of::<PJRT_Error_Message_Args>(),
        extension_start: std::ptr::null_mut(),
        error: err,
        message: std::ptr::null(),
        message_size: 0,
    };
    unsafe {
        (fns.error_message)(&mut msg_args);
    }
    // The plugin keeps the message buffer alive until the error is
    // destroyed; copy out before destroying.
    let msg = if !msg_args.message.is_null() {
        let bytes = unsafe { std::slice::from_raw_parts(msg_args.message, msg_args.message_size) };
        String::from_utf8_lossy(bytes).into_owned()
    } else {
        "<no message>".to_string()
    };
    let mut destroy_args = PJRT_Error_Destroy_Args {
        struct_size: std::mem::size_of::<PJRT_Error_Destroy_Args>(),
        extension_start: std::ptr::null_mut(),
        error: err,
    };
    unsafe {
        (fns.error_destroy)(&mut destroy_args);
    }
    msg
}

/// Convenience: panic with a clear PJRT error message.
///
/// SAFETY: `err` must be non-null. After this call it is destroyed.
pub unsafe fn panic_pjrt(fns: &PjrtFns, err: *mut PjrtError, what: &str) -> ! {
    let msg = unsafe { error_to_string(fns, err) };
    panic!("rlx-tpu: {what}: {msg}");
}

/// Block on an event. Returns Ok(()) on success or the error message
/// the event carried.
///
/// SAFETY: `event` must be a non-null event from PJRT.
pub unsafe fn event_await(fns: &PjrtFns, event: *mut PjrtEvent) -> Result<(), String> {
    let mut args = PJRT_Event_Await_Args {
        struct_size: std::mem::size_of::<PJRT_Event_Await_Args>(),
        extension_start: std::ptr::null_mut(),
        event,
    };
    let err = unsafe { (fns.event_await)(&mut args) };
    if !err.is_null() {
        return Err(unsafe { error_to_string(fns, err) });
    }
    // After Await the event holds either Ok or an error; surface
    // any error by querying.
    let mut e_args = PJRT_Event_Error_Args {
        struct_size: std::mem::size_of::<PJRT_Event_Error_Args>(),
        extension_start: std::ptr::null_mut(),
        event,
    };
    let e_err = unsafe { (fns.event_error)(&mut e_args) };
    if !e_err.is_null() {
        return Err(unsafe { error_to_string(fns, e_err) });
    }
    let mut d_args = PJRT_Event_Destroy_Args {
        struct_size: std::mem::size_of::<PJRT_Event_Destroy_Args>(),
        extension_start: std::ptr::null_mut(),
        event,
    };
    let d_err = unsafe { (fns.event_destroy)(&mut d_args) };
    if !d_err.is_null() {
        return Err(unsafe { error_to_string(fns, d_err) });
    }
    Ok(())
}
