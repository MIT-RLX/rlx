// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! The process-wide Level Zero driver / device / context / compute-queue
//! singleton, brought up through the dynamically-loaded `libze_loader`. If no
//! loader is present, `zeInit` fails, or no GPU device is exposed,
//! [`oneapi_device`] returns `None` and the backend reports itself unavailable
//! — mirroring rlx-cuda / rlx-rocm / rlx-vulkan on a host with no driver.

use crate::level_zero::*;
use std::ffi::{CStr, c_void};
use std::sync::{Mutex, OnceLock};

/// Owned Level Zero context. One per process.
pub struct OneApiDevice {
    pub lib: Lib,
    pub driver: DriverHandle,
    pub device: DeviceHandle,
    pub context: ContextHandle,
    pub queue: CommandQueueHandle,
    /// Command-queue-group ordinal the compute queue/list were created on.
    pub queue_ordinal: u32,
    pub name: String,
    /// Level Zero command queues require external synchronization; serialize.
    submit_lock: Mutex<()>,
}

// The handles are process-global and only touched behind `submit_lock` or are
// immutable after construction (mirrors rlx-vulkan's VulkanDevice).
unsafe impl Send for OneApiDevice {}
unsafe impl Sync for OneApiDevice {}

static DEVICE: OnceLock<Option<OneApiDevice>> = OnceLock::new();

/// The process-wide oneAPI device, or `None` when unavailable.
pub fn oneapi_device() -> Option<&'static OneApiDevice> {
    DEVICE.get_or_init(|| OneApiDevice::new().ok()).as_ref()
}

impl OneApiDevice {
    fn new() -> Result<Self, String> {
        // SAFETY: dynamic-load the Level Zero loader; Err on hosts without it.
        let lib = unsafe { Lib::load() }?;

        unsafe {
            check((lib.ze_init)(0), "zeInit")?;

            // Enumerate drivers.
            let mut driver_count: u32 = 0;
            check(
                (lib.driver_get)(&mut driver_count, std::ptr::null_mut()),
                "zeDriverGet(count)",
            )?;
            if driver_count == 0 {
                return Err("no Level Zero drivers".into());
            }
            let mut drivers: Vec<DriverHandle> = vec![std::ptr::null_mut(); driver_count as usize];
            check(
                (lib.driver_get)(&mut driver_count, drivers.as_mut_ptr()),
                "zeDriverGet",
            )?;

            // Pick the first GPU device across all drivers.
            let mut chosen: Option<(DriverHandle, DeviceHandle, String)> = None;
            for &driver in &drivers {
                let mut dev_count: u32 = 0;
                if check(
                    (lib.device_get)(driver, &mut dev_count, std::ptr::null_mut()),
                    "zeDeviceGet(count)",
                )
                .is_err()
                    || dev_count == 0
                {
                    continue;
                }
                let mut devices: Vec<DeviceHandle> = vec![std::ptr::null_mut(); dev_count as usize];
                if check(
                    (lib.device_get)(driver, &mut dev_count, devices.as_mut_ptr()),
                    "zeDeviceGet",
                )
                .is_err()
                {
                    continue;
                }
                for &device in &devices {
                    let mut props = DeviceProperties::default();
                    if (lib.device_get_properties)(device, &mut props) != ZE_RESULT_SUCCESS {
                        continue;
                    }
                    if props.type_ != ZE_DEVICE_TYPE_GPU {
                        continue;
                    }
                    let name = CStr::from_ptr(props.name.as_ptr())
                        .to_string_lossy()
                        .into_owned();
                    chosen = Some((driver, device, name));
                    break;
                }
                if chosen.is_some() {
                    break;
                }
            }
            let (driver, device, name) =
                chosen.ok_or_else(|| "no Level Zero GPU device".to_string())?;

            // Context.
            let ctx_desc = ContextDesc {
                stype: ZE_STRUCTURE_TYPE_CONTEXT_DESC,
                pnext: std::ptr::null(),
                flags: 0,
            };
            let mut context: ContextHandle = std::ptr::null_mut();
            check(
                (lib.context_create)(driver, &ctx_desc, &mut context),
                "zeContextCreate",
            )?;

            // Compute command queue. Group ordinal 0 is the primary compute
            // group on Intel GPUs; a follow-up can query
            // zeDeviceGetCommandQueueGroupProperties to pick the COMPUTE group
            // explicitly (it's almost always 0).
            let queue_ordinal = 0u32;
            let q_desc = CommandQueueDesc {
                stype: ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                pnext: std::ptr::null(),
                ordinal: queue_ordinal,
                index: 0,
                flags: 0,
                mode: ZE_COMMAND_QUEUE_MODE_DEFAULT,
                priority: 0,
            };
            let mut queue: CommandQueueHandle = std::ptr::null_mut();
            check(
                (lib.command_queue_create)(context, device, &q_desc, &mut queue),
                "zeCommandQueueCreate",
            )?;

            Ok(Self {
                lib,
                driver,
                device,
                context,
                queue,
                queue_ordinal,
                name,
                submit_lock: Mutex::new(()),
            })
        }
    }

    /// Allocate a host-accessible USM-shared buffer of `size` bytes, zeroed.
    /// Returns the device pointer (also CPU-dereferenceable on integrated /
    /// shared-memory Intel GPUs).
    pub fn alloc_shared(&self, size: usize) -> Result<*mut c_void, String> {
        let dev_desc = DeviceMemAllocDesc {
            stype: ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            pnext: std::ptr::null(),
            flags: 0,
            ordinal: 0,
        };
        let host_desc = HostMemAllocDesc {
            stype: ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
            pnext: std::ptr::null(),
            flags: 0,
        };
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            check(
                (self.lib.mem_alloc_shared)(
                    self.context,
                    &dev_desc,
                    &host_desc,
                    size.max(1),
                    64,
                    self.device,
                    &mut ptr,
                ),
                "zeMemAllocShared",
            )?;
            std::ptr::write_bytes(ptr as *mut u8, 0, size);
        }
        Ok(ptr)
    }

    /// Free a USM buffer previously returned by [`alloc_shared`](Self::alloc_shared).
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // ptr is an opaque USM handle; we only pass it to FFI
    pub fn free(&self, ptr: *mut c_void) {
        if !ptr.is_null() {
            unsafe {
                let _ = (self.lib.mem_free)(self.context, ptr);
            }
        }
    }

    /// Create a fresh closed-on-demand command list on the compute group.
    pub fn create_command_list(&self) -> Result<CommandListHandle, String> {
        let desc = CommandListDesc {
            stype: ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
            pnext: std::ptr::null(),
            command_queue_group_ordinal: self.queue_ordinal,
            flags: 0,
        };
        let mut list: CommandListHandle = std::ptr::null_mut();
        unsafe {
            check(
                (self.lib.command_list_create)(self.context, self.device, &desc, &mut list),
                "zeCommandListCreate",
            )?;
        }
        Ok(list)
    }

    /// Close `list`, execute it on the compute queue, and block until complete.
    /// Serialized across threads (queues need external synchronization).
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // list is an opaque handle; we only pass it to FFI
    pub fn execute_sync(&self, list: CommandListHandle) -> Result<(), String> {
        let _guard = self.submit_lock.lock().unwrap();
        unsafe {
            check((self.lib.command_list_close)(list), "zeCommandListClose")?;
            let lists = [list];
            check(
                (self.lib.command_queue_execute)(
                    self.queue,
                    1,
                    lists.as_ptr(),
                    std::ptr::null_mut(),
                ),
                "zeCommandQueueExecuteCommandLists",
            )?;
            check(
                (self.lib.command_queue_synchronize)(self.queue, u64::MAX),
                "zeCommandQueueSynchronize",
            )?;
        }
        Ok(())
    }
}

impl Drop for OneApiDevice {
    fn drop(&mut self) {
        unsafe {
            if !self.queue.is_null() {
                let _ = (self.lib.command_queue_destroy)(self.queue);
            }
            if !self.context.is_null() {
                let _ = (self.lib.context_destroy)(self.context);
            }
        }
    }
}
