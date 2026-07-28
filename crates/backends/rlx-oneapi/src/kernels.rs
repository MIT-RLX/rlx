// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Embedded SPIR-V kernel blobs (compiled from `kernels/*.cl` by `build.rs`)
//! and their Level Zero module/kernel cache. `SPIRV_BLOBS` is empty unless the
//! crate was built on an Intel oneAPI host with `ocloc` + `RLX_ONEAPI_BUILD_KERNELS=1`,
//! in which case [`kernels`] builds one `ze_module` + `ze_kernel` per blob (the
//! kernel name equals the `.cl` function name, which equals the file stem).

include!(concat!(env!("OUT_DIR"), "/kernels_generated.rs"));

use crate::device::{OneApiDevice, oneapi_device};
use crate::level_zero::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::OnceLock;

/// SPIR-V byte blob for kernel `name`, if it was embedded this build.
pub fn blob(name: &str) -> Option<&'static [u8]> {
    SPIRV_BLOBS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, b)| *b)
}

/// All embedded kernel names (deterministic; build.rs sorts).
pub fn names() -> impl Iterator<Item = &'static str> {
    SPIRV_BLOBS.iter().map(|(n, _)| *n)
}

/// Whether any native SPIR-V kernel was embedded (false on non-Intel builds).
pub fn kernels_built() -> bool {
    KERNELS_BUILT
}

/// Built Level Zero modules + kernels for the embedded SPIR-V blobs.
pub struct Kernels {
    modules: Vec<ModuleHandle>,
    kernels: HashMap<&'static str, KernelHandle>,
}

// Handles are process-global, built once, and dispatched behind the device's
// submit lock.
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

impl Kernels {
    /// Kernel handle for `name`, or `None` if that kernel wasn't embedded/built.
    pub fn get(&self, name: &str) -> Option<KernelHandle> {
        self.kernels.get(name).copied()
    }

    fn build(dev: &OneApiDevice) -> Result<Kernels, String> {
        let mut modules = Vec::new();
        let mut kernels = HashMap::new();
        for (name, spirv) in SPIRV_BLOBS {
            let desc = ModuleDesc {
                stype: ZE_STRUCTURE_TYPE_MODULE_DESC,
                pnext: std::ptr::null(),
                format: ZE_MODULE_FORMAT_IL_SPIRV,
                input_size: spirv.len(),
                p_input_module: spirv.as_ptr(),
                p_build_flags: std::ptr::null(),
                p_constants: std::ptr::null(),
            };
            let mut module: ModuleHandle = std::ptr::null_mut();
            let mut build_log: ModuleBuildLogHandle = std::ptr::null_mut();
            unsafe {
                check(
                    (dev.lib.module_create)(
                        dev.context,
                        dev.device,
                        &desc,
                        &mut module,
                        &mut build_log,
                    ),
                    &format!("zeModuleCreate({name})"),
                )?;
            }

            let cname = CString::new(*name).map_err(|e| format!("kernel name {name}: {e}"))?;
            let kdesc = KernelDesc {
                stype: ZE_STRUCTURE_TYPE_KERNEL_DESC,
                pnext: std::ptr::null(),
                flags: 0,
                p_kernel_name: cname.as_ptr(),
            };
            let mut kernel: KernelHandle = std::ptr::null_mut();
            unsafe {
                check(
                    (dev.lib.kernel_create)(module, &kdesc, &mut kernel),
                    &format!("zeKernelCreate({name})"),
                )?;
            }
            modules.push(module);
            kernels.insert(*name, kernel);
        }
        Ok(Kernels { modules, kernels })
    }
}

impl Drop for Kernels {
    fn drop(&mut self) {
        if let Some(dev) = oneapi_device() {
            unsafe {
                for &k in self.kernels.values() {
                    let _ = (dev.lib.kernel_destroy)(k);
                }
                for &m in &self.modules {
                    let _ = (dev.lib.module_destroy)(m);
                }
            }
        }
    }
}

/// Process-wide kernel cache, or `None` when there is no device or no embedded
/// SPIR-V (the latter is the case on every non-Intel build host).
pub fn kernels() -> Option<&'static Kernels> {
    static CACHE: OnceLock<Option<Kernels>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let dev = oneapi_device()?;
            if SPIRV_BLOBS.is_empty() {
                return None;
            }
            match Kernels::build(dev) {
                Ok(k) => Some(k),
                Err(e) => {
                    eprintln!("rlx-oneapi: kernel build failed ({e}); using CPU reference path");
                    None
                }
            }
        })
        .as_ref()
}
