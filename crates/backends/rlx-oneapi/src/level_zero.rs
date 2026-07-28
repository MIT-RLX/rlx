// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-rolled Level Zero (`ze_*`) FFI, dynamically loaded with `libloading`.
//!
//! Only the subset RLX needs is bound: init + driver/device enumeration and
//! properties (for availability + device name), context/queue/command-list
//! lifecycle, USM-shared allocation, and SPIR-V module → kernel → launch. The
//! loader (`libze_loader`) is opened at runtime, so the crate links and builds
//! on hosts with no oneAPI runtime (macOS, CI) — [`Lib::load`] simply returns
//! `Err` there and the whole backend reports itself unavailable.
//!
//! ⚠️ The `ZE_STRUCTURE_TYPE_*` enum values and descriptor field layouts are
//! transcribed from `ze_api.h` (Level Zero spec v1.x). They are exercised only
//! on real Intel hardware (Arc / Data Center Max) during bring-up; verify them
//! against the installed loader version there. Nothing in this module runs on
//! the macOS dev box — the backend falls through to the CPU reference path.

#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    clippy::missing_safety_doc
)]

use libloading::Library;
use std::ffi::{c_char, c_void};

// ── result + handles ───────────────────────────────────────────────────────

pub type ZeResult = i32; // ze_result_t
pub const ZE_RESULT_SUCCESS: ZeResult = 0;

pub type DriverHandle = *mut c_void;
pub type DeviceHandle = *mut c_void;
pub type ContextHandle = *mut c_void;
pub type CommandQueueHandle = *mut c_void;
pub type CommandListHandle = *mut c_void;
pub type ModuleHandle = *mut c_void;
pub type ModuleBuildLogHandle = *mut c_void;
pub type KernelHandle = *mut c_void;
pub type FenceHandle = *mut c_void;
pub type EventHandle = *mut c_void;

// ── ze_structure_type_t (subset) ───────────────────────────────────────────

pub const ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 0x3;
pub const ZE_STRUCTURE_TYPE_CONTEXT_DESC: u32 = 0xd;
pub const ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC: u32 = 0xe;
pub const ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC: u32 = 0xf;
pub const ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC: u32 = 0x15;
pub const ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC: u32 = 0x16;
pub const ZE_STRUCTURE_TYPE_MODULE_DESC: u32 = 0x1d;
pub const ZE_STRUCTURE_TYPE_KERNEL_DESC: u32 = 0x1f;

/// `ze_device_type_t`: GPU = 1.
pub const ZE_DEVICE_TYPE_GPU: u32 = 1;
/// `ze_module_format_t`: IL_SPIRV = 0.
pub const ZE_MODULE_FORMAT_IL_SPIRV: u32 = 0;
/// `ze_command_queue_mode_t`: SYNCHRONOUS = 1 (execute blocks until complete).
pub const ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS: u32 = 1;
/// `ze_command_queue_mode_t`: DEFAULT = 0.
pub const ZE_COMMAND_QUEUE_MODE_DEFAULT: u32 = 0;

pub const ZE_MAX_DEVICE_NAME: usize = 256;
pub const ZE_MAX_DEVICE_UUID_SIZE: usize = 16;

// ── descriptor structs (repr(C), matching ze_api.h) ────────────────────────

#[repr(C)]
pub struct DeviceUuid {
    pub id: [u8; ZE_MAX_DEVICE_UUID_SIZE],
}

#[repr(C)]
pub struct DeviceProperties {
    pub stype: u32,
    pub pnext: *mut c_void,
    pub type_: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub flags: u32,
    pub subdevice_id: u32,
    pub core_clock_rate: u32,
    pub max_mem_alloc_size: u64,
    pub max_hardware_contexts: u32,
    pub max_command_queue_priority: u32,
    pub num_threads_per_eu: u32,
    pub physical_eu_simd_width: u32,
    pub num_eus_per_subslice: u32,
    pub num_subslices_per_slice: u32,
    pub num_slices: u32,
    pub timer_resolution: u64,
    pub timestamp_valid_bits: u32,
    pub kernel_timestamp_valid_bits: u32,
    pub uuid: DeviceUuid,
    pub name: [c_char; ZE_MAX_DEVICE_NAME],
}

impl Default for DeviceProperties {
    fn default() -> Self {
        // SAFETY: all-zero is a valid bit pattern for every field (the call
        // overwrites them); we only set the required `stype` afterwards.
        let mut p: Self = unsafe { std::mem::zeroed() };
        p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        p
    }
}

#[repr(C)]
pub struct ContextDesc {
    pub stype: u32,
    pub pnext: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct CommandQueueDesc {
    pub stype: u32,
    pub pnext: *const c_void,
    pub ordinal: u32,
    pub index: u32,
    pub flags: u32,
    pub mode: u32,
    pub priority: u32,
}

#[repr(C)]
pub struct CommandListDesc {
    pub stype: u32,
    pub pnext: *const c_void,
    pub command_queue_group_ordinal: u32,
    pub flags: u32,
}

#[repr(C)]
pub struct ModuleDesc {
    pub stype: u32,
    pub pnext: *const c_void,
    pub format: u32,
    pub input_size: usize,
    pub p_input_module: *const u8,
    pub p_build_flags: *const c_char,
    pub p_constants: *const c_void,
}

#[repr(C)]
pub struct KernelDesc {
    pub stype: u32,
    pub pnext: *const c_void,
    pub flags: u32,
    pub p_kernel_name: *const c_char,
}

#[repr(C)]
pub struct DeviceMemAllocDesc {
    pub stype: u32,
    pub pnext: *const c_void,
    pub flags: u32,
    pub ordinal: u32,
}

#[repr(C)]
pub struct HostMemAllocDesc {
    pub stype: u32,
    pub pnext: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct GroupCount {
    pub group_count_x: u32,
    pub group_count_y: u32,
    pub group_count_z: u32,
}

// ── function pointer types ─────────────────────────────────────────────────

type PfnInit = unsafe extern "C" fn(u32) -> ZeResult;
type PfnDriverGet = unsafe extern "C" fn(*mut u32, *mut DriverHandle) -> ZeResult;
type PfnDeviceGet = unsafe extern "C" fn(DriverHandle, *mut u32, *mut DeviceHandle) -> ZeResult;
type PfnDeviceGetProperties = unsafe extern "C" fn(DeviceHandle, *mut DeviceProperties) -> ZeResult;
type PfnContextCreate =
    unsafe extern "C" fn(DriverHandle, *const ContextDesc, *mut ContextHandle) -> ZeResult;
type PfnContextDestroy = unsafe extern "C" fn(ContextHandle) -> ZeResult;
type PfnCommandQueueCreate = unsafe extern "C" fn(
    ContextHandle,
    DeviceHandle,
    *const CommandQueueDesc,
    *mut CommandQueueHandle,
) -> ZeResult;
type PfnCommandQueueDestroy = unsafe extern "C" fn(CommandQueueHandle) -> ZeResult;
type PfnCommandQueueExecute = unsafe extern "C" fn(
    CommandQueueHandle,
    u32,
    *const CommandListHandle,
    FenceHandle,
) -> ZeResult;
type PfnCommandQueueSynchronize = unsafe extern "C" fn(CommandQueueHandle, u64) -> ZeResult;
type PfnCommandListCreate = unsafe extern "C" fn(
    ContextHandle,
    DeviceHandle,
    *const CommandListDesc,
    *mut CommandListHandle,
) -> ZeResult;
type PfnCommandListDestroy = unsafe extern "C" fn(CommandListHandle) -> ZeResult;
type PfnCommandListClose = unsafe extern "C" fn(CommandListHandle) -> ZeResult;
type PfnCommandListReset = unsafe extern "C" fn(CommandListHandle) -> ZeResult;
type PfnCommandListAppendLaunchKernel = unsafe extern "C" fn(
    CommandListHandle,
    KernelHandle,
    *const GroupCount,
    EventHandle,
    u32,
    *mut EventHandle,
) -> ZeResult;
type PfnCommandListAppendBarrier =
    unsafe extern "C" fn(CommandListHandle, EventHandle, u32, *mut EventHandle) -> ZeResult;
type PfnMemAllocShared = unsafe extern "C" fn(
    ContextHandle,
    *const DeviceMemAllocDesc,
    *const HostMemAllocDesc,
    usize,
    usize,
    DeviceHandle,
    *mut *mut c_void,
) -> ZeResult;
type PfnMemFree = unsafe extern "C" fn(ContextHandle, *mut c_void) -> ZeResult;
type PfnModuleCreate = unsafe extern "C" fn(
    ContextHandle,
    DeviceHandle,
    *const ModuleDesc,
    *mut ModuleHandle,
    *mut ModuleBuildLogHandle,
) -> ZeResult;
type PfnModuleDestroy = unsafe extern "C" fn(ModuleHandle) -> ZeResult;
type PfnKernelCreate =
    unsafe extern "C" fn(ModuleHandle, *const KernelDesc, *mut KernelHandle) -> ZeResult;
type PfnKernelDestroy = unsafe extern "C" fn(KernelHandle) -> ZeResult;
type PfnKernelSetGroupSize = unsafe extern "C" fn(KernelHandle, u32, u32, u32) -> ZeResult;
type PfnKernelSetArgumentValue =
    unsafe extern "C" fn(KernelHandle, u32, usize, *const c_void) -> ZeResult;

/// Resolved Level Zero entry points. Keeps the loaded [`Library`] alive so the
/// copied function pointers stay valid for the process lifetime.
pub struct Lib {
    _lib: Library,
    pub ze_init: PfnInit,
    pub driver_get: PfnDriverGet,
    pub device_get: PfnDeviceGet,
    pub device_get_properties: PfnDeviceGetProperties,
    pub context_create: PfnContextCreate,
    pub context_destroy: PfnContextDestroy,
    pub command_queue_create: PfnCommandQueueCreate,
    pub command_queue_destroy: PfnCommandQueueDestroy,
    pub command_queue_execute: PfnCommandQueueExecute,
    pub command_queue_synchronize: PfnCommandQueueSynchronize,
    pub command_list_create: PfnCommandListCreate,
    pub command_list_destroy: PfnCommandListDestroy,
    pub command_list_close: PfnCommandListClose,
    pub command_list_reset: PfnCommandListReset,
    pub command_list_append_launch_kernel: PfnCommandListAppendLaunchKernel,
    pub command_list_append_barrier: PfnCommandListAppendBarrier,
    pub mem_alloc_shared: PfnMemAllocShared,
    pub mem_free: PfnMemFree,
    pub module_create: PfnModuleCreate,
    pub module_destroy: PfnModuleDestroy,
    pub kernel_create: PfnKernelCreate,
    pub kernel_destroy: PfnKernelDestroy,
    pub kernel_set_group_size: PfnKernelSetGroupSize,
    pub kernel_set_argument_value: PfnKernelSetArgumentValue,
}

/// Candidate loader filenames per platform. macOS has none → load fails →
/// backend unavailable (graceful, like rlx-vulkan without MoltenVK).
fn loader_names() -> &'static [&'static str] {
    if cfg!(target_os = "windows") {
        &["ze_loader.dll"]
    } else {
        &["libze_loader.so.1", "libze_loader.so"]
    }
}

impl Lib {
    /// Open `libze_loader` and resolve every entry point. `Err` on any host
    /// without the oneAPI runtime (no library or a missing symbol).
    pub unsafe fn load() -> Result<Lib, String> {
        // Honor an explicit override first, then the platform defaults.
        let mut tried: Vec<String> = Vec::new();
        let mut lib: Option<Library> = None;
        let mut names: Vec<String> = Vec::new();
        if let Ok(p) = std::env::var("RLX_ONEAPI_LOADER") {
            names.push(p);
        }
        names.extend(loader_names().iter().map(|s| s.to_string()));
        for name in &names {
            match unsafe { Library::new(name) } {
                Ok(l) => {
                    lib = Some(l);
                    break;
                }
                Err(e) => tried.push(format!("{name}: {e}")),
            }
        }
        let lib = lib.ok_or_else(|| format!("no Level Zero loader ({})", tried.join("; ")))?;

        macro_rules! sym {
            ($name:expr) => {{
                let s: libloading::Symbol<_> = unsafe { lib.get($name) }
                    .map_err(|e| format!("missing {}: {e}", String::from_utf8_lossy($name)))?;
                *s
            }};
        }

        let out = Lib {
            ze_init: sym!(b"zeInit\0"),
            driver_get: sym!(b"zeDriverGet\0"),
            device_get: sym!(b"zeDeviceGet\0"),
            device_get_properties: sym!(b"zeDeviceGetProperties\0"),
            context_create: sym!(b"zeContextCreate\0"),
            context_destroy: sym!(b"zeContextDestroy\0"),
            command_queue_create: sym!(b"zeCommandQueueCreate\0"),
            command_queue_destroy: sym!(b"zeCommandQueueDestroy\0"),
            command_queue_execute: sym!(b"zeCommandQueueExecuteCommandLists\0"),
            command_queue_synchronize: sym!(b"zeCommandQueueSynchronize\0"),
            command_list_create: sym!(b"zeCommandListCreate\0"),
            command_list_destroy: sym!(b"zeCommandListDestroy\0"),
            command_list_close: sym!(b"zeCommandListClose\0"),
            command_list_reset: sym!(b"zeCommandListReset\0"),
            command_list_append_launch_kernel: sym!(b"zeCommandListAppendLaunchKernel\0"),
            command_list_append_barrier: sym!(b"zeCommandListAppendBarrier\0"),
            mem_alloc_shared: sym!(b"zeMemAllocShared\0"),
            mem_free: sym!(b"zeMemFree\0"),
            module_create: sym!(b"zeModuleCreate\0"),
            module_destroy: sym!(b"zeModuleDestroy\0"),
            kernel_create: sym!(b"zeKernelCreate\0"),
            kernel_destroy: sym!(b"zeKernelDestroy\0"),
            kernel_set_group_size: sym!(b"zeKernelSetGroupSize\0"),
            kernel_set_argument_value: sym!(b"zeKernelSetArgumentValue\0"),
            _lib: lib,
        };
        Ok(out)
    }
}

/// Map a non-success [`ZeResult`] to a printable error.
pub fn check(res: ZeResult, ctx: &str) -> Result<(), String> {
    if res == ZE_RESULT_SUCCESS {
        Ok(())
    } else {
        Err(format!("{ctx} failed: ze_result 0x{res:x}"))
    }
}
