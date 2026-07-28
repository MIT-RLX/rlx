// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vulkan instance / physical-device / logical-device / compute-queue
//! singleton, brought up through `ash` with the dynamically-loaded Vulkan
//! loader. If no loader / driver is present (`Entry::load()` fails) or no
//! device exposes a compute queue, [`vulkan_device`] returns `None` and the
//! whole backend reports itself unavailable — the crate still compiles and
//! links on hosts without Vulkan (macOS without MoltenVK, CI).

use ash::vk;
use std::ffi::{CStr, c_char};
use std::sync::{Mutex, OnceLock};

/// Owned Vulkan context. One per process.
pub struct VulkanDevice {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    pub limits: vk::PhysicalDeviceLimits,
    pub name: String,
    /// True when the selected device is a portability driver (MoltenVK on
    /// Apple): `VK_KHR_portability_subset` was required and enabled. The matmul
    /// scheduler falls back to the scalar kernel here, since shared-memory
    /// tiling regresses under Vulkan→Metal translation (Metal is Apple's
    /// native path anyway).
    pub portability: bool,
    /// True when the device exposes a usable 16×16×16 f16·f16→f32 subgroup
    /// cooperative-matrix (tensor-core) config AND the features its kernel
    /// needs were enabled. Gates the `matmul_coop` fast path. Always false on
    /// portability drivers (MoltenVK doesn't expose `VK_KHR_cooperative_matrix`).
    pub coop_matmul: bool,
    cmd_pool: vk::CommandPool,
    /// Vulkan queues require external synchronization; serialize submits.
    submit_lock: Mutex<()>,
}

// The Vulkan handles are process-global and only touched through the
// `submit_lock`-guarded paths (or are immutable after construction). This
// mirrors the singleton pattern used by rlx-cuda's `CudaContext`.
unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

static DEVICE: OnceLock<Option<VulkanDevice>> = OnceLock::new();

/// Point the Vulkan loader at the MoltenVK ICD when the user hasn't, by probing
/// the standard Homebrew / system install locations. The loader reads
/// `VK_ICD_FILENAMES` at `vkCreateInstance`, so setting it here (before the
/// instance is built) is enough for `DEVICE=vulkan` to work out of the box.
#[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
fn ensure_macos_loader() {
    if std::env::var_os("VK_ICD_FILENAMES").is_some()
        || std::env::var_os("VK_DRIVER_FILES").is_some()
    {
        return; // user already chose an ICD
    }
    // Fixed locations first, then the versioned Homebrew Cellar.
    for cand in [
        "/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json",
        "/usr/local/share/vulkan/icd.d/MoltenVK_icd.json",
    ] {
        if std::path::Path::new(cand).exists() {
            // SAFETY: runs once during the `OnceLock` device init, before the
            // Vulkan instance (and any benchmarking threads) are created.
            unsafe { std::env::set_var("VK_ICD_FILENAMES", cand) };
            return;
        }
    }
    for cellar in [
        "/opt/homebrew/Cellar/molten-vk",
        "/usr/local/Cellar/molten-vk",
    ] {
        if let Ok(rd) = std::fs::read_dir(cellar) {
            for ent in rd.flatten() {
                let icd = ent.path().join("etc/vulkan/icd.d/MoltenVK_icd.json");
                if icd.exists() {
                    // SAFETY: see above.
                    unsafe { std::env::set_var("VK_ICD_FILENAMES", icd) };
                    return;
                }
            }
        }
    }
}

/// The process-wide Vulkan device, or `None` when unavailable.
pub fn vulkan_device() -> Option<&'static VulkanDevice> {
    DEVICE.get_or_init(|| VulkanDevice::new().ok()).as_ref()
}

impl VulkanDevice {
    fn new() -> Result<Self, String> {
        // On macOS the Vulkan loader + MoltenVK ICD ship via Homebrew but land
        // off the default dlopen / loader search paths, so a stock `DEVICE=vulkan`
        // reported "unavailable" until the user exported VK_ICD_FILENAMES and
        // DYLD_LIBRARY_PATH by hand. Wire both up automatically here.
        #[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
        ensure_macos_loader();

        // SAFETY: dynamic-load the system Vulkan loader. If the default search
        // misses it (Homebrew prefix), retry from known libvulkan locations.
        // Returns Err on hosts with no loader, which we map to "unavailable".
        let entry = unsafe { ash::Entry::load() }
            .or_else(|orig| {
                for lib in [
                    "/opt/homebrew/lib/libvulkan.dylib",
                    "/opt/homebrew/lib/libvulkan.1.dylib",
                    "/usr/local/lib/libvulkan.dylib",
                ] {
                    if std::path::Path::new(lib).exists() {
                        if let Ok(e) = unsafe { ash::Entry::load_from(lib) } {
                            return Ok(e);
                        }
                    }
                }
                Err(orig)
            })
            .map_err(|e| format!("vk load: {e}"))?;

        let app_name = c"rlx-vulkan";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .engine_name(app_name)
            .api_version(vk::make_api_version(0, 1, 1, 0));

        // On Apple platforms the only ICD is MoltenVK, a portability driver:
        // it must be opted into via the portability-enumeration extension.
        #[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
        let (inst_ext, inst_flags) = (
            vec![
                ash::khr::portability_enumeration::NAME.as_ptr(),
                ash::khr::get_physical_device_properties2::NAME.as_ptr(),
            ],
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR,
        );
        #[cfg(not(all(target_vendor = "apple", not(target_os = "watchos"))))]
        let (inst_ext, inst_flags): (Vec<*const c_char>, vk::InstanceCreateFlags) =
            (Vec::new(), vk::InstanceCreateFlags::empty());

        // Opt-in Khronos validation layer for debugging (RLX_VULKAN_VALIDATION=1);
        // VUID messages print to stderr. Off by default (no runtime cost).
        let validation_layer = c"VK_LAYER_KHRONOS_validation";
        let mut inst_layers: Vec<*const c_char> = Vec::new();
        if std::env::var("RLX_VULKAN_VALIDATION").ok().as_deref() == Some("1") {
            inst_layers.push(validation_layer.as_ptr());
        }

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&inst_ext)
            .enabled_layer_names(&inst_layers)
            .flags(inst_flags);
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| format!("vk instance: {e}"))?;

        // Pick the best physical device exposing a compute queue.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("vk enumerate: {e}"))?;
        let mut best: Option<(vk::PhysicalDevice, u32, i32)> = None;
        for &pd in &physical_devices {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let qfams = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let Some(qf) = qfams
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE) && q.queue_count > 0)
            else {
                continue;
            };
            let score = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                _ => 0,
            };
            if best.map(|(_, _, s)| score > s).unwrap_or(true) {
                best = Some((pd, qf as u32, score));
            }
        }
        let (physical, queue_family, _) = best.ok_or_else(|| {
            unsafe { instance.destroy_instance(None) };
            "no Vulkan device with a compute queue".to_string()
        })?;

        let props = unsafe { instance.get_physical_device_properties(physical) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        // Enable VK_KHR_portability_subset on the logical device when the
        // physical device requires it (MoltenVK), else creation fails.
        let dev_exts =
            unsafe { instance.enumerate_device_extension_properties(physical) }.unwrap_or_default();
        let mut dev_ext: Vec<*const c_char> = Vec::new();
        let portability_name = c"VK_KHR_portability_subset";
        let mut is_portability = false;
        for e in &dev_exts {
            let n = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
            if n == portability_name {
                dev_ext.push(portability_name.as_ptr());
                is_portability = true;
            }
        }

        // Cooperative-matrix (tensor-core) matmul fast path. Native drivers
        // only (MoltenVK never exposes it). Needs the coop-matrix extension
        // plus the f16 / memory-model features the `matmul_coop` kernel uses,
        // and a supported 16×16×16 f16·f16→f32 subgroup config.
        let has_ext = |want: &CStr| {
            dev_exts
                .iter()
                .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == want)
        };
        let coop_ext = c"VK_KHR_cooperative_matrix";
        let memmodel_ext = c"VK_KHR_vulkan_memory_model";
        let f16_ext = c"VK_KHR_shader_float16_int8";
        let s16_ext = c"VK_KHR_16bit_storage";
        let mut coop_matmul = false;
        if !is_portability
            && has_ext(coop_ext)
            && has_ext(memmodel_ext)
            && has_ext(f16_ext)
            && has_ext(s16_ext)
        {
            let mut coop_feat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
            let mut probe = vk::PhysicalDeviceFeatures2::default().push_next(&mut coop_feat);
            unsafe { instance.get_physical_device_features2(physical, &mut probe) };
            if coop_feat.cooperative_matrix != 0 {
                let ci = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
                let configs =
                    unsafe { ci.get_physical_device_cooperative_matrix_properties(physical) }
                        .unwrap_or_default();
                coop_matmul = configs.iter().any(|c| {
                    c.m_size == 16
                        && c.n_size == 16
                        && c.k_size == 16
                        && c.a_type == vk::ComponentTypeKHR::FLOAT16
                        && c.b_type == vk::ComponentTypeKHR::FLOAT16
                        && c.result_type == vk::ComponentTypeKHR::FLOAT32
                        && c.scope == vk::ScopeKHR::SUBGROUP
                });
            }
        }
        if coop_matmul {
            dev_ext.push(coop_ext.as_ptr());
            dev_ext.push(memmodel_ext.as_ptr());
            dev_ext.push(f16_ext.as_ptr());
            dev_ext.push(s16_ext.as_ptr());
        }
        if std::env::var_os("RLX_VULKAN_DEBUG").is_some() {
            eprintln!(
                "[rlx-vulkan] device={name:?} portability={is_portability} coop_matmul={coop_matmul}"
            );
        }

        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let base_features = vk::PhysicalDeviceFeatures::default();
        let device = if coop_matmul {
            // Chain the feature structs the coop kernel requires via pNext.
            let mut coop_f =
                vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
            let mut mm_f =
                vk::PhysicalDeviceVulkanMemoryModelFeatures::default().vulkan_memory_model(true);
            let mut f16_f =
                vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
            let mut s16_f =
                vk::PhysicalDevice16BitStorageFeatures::default().storage_buffer16_bit_access(true);
            let mut feats2 = vk::PhysicalDeviceFeatures2::default()
                .features(base_features)
                .push_next(&mut coop_f)
                .push_next(&mut mm_f)
                .push_next(&mut f16_f)
                .push_next(&mut s16_f);
            let dci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&dev_ext)
                .push_next(&mut feats2);
            unsafe { instance.create_device(physical, &dci, None) }
        } else {
            let dci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&dev_ext)
                .enabled_features(&base_features);
            unsafe { instance.create_device(physical, &dci, None) }
        }
        .map_err(|e| format!("vk device: {e}"))?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };

        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|e| format!("vk cmd pool: {e}"))?;

        Ok(Self {
            entry,
            instance,
            physical,
            device,
            queue,
            queue_family,
            mem_props,
            limits: props.limits,
            name,
            portability: is_portability,
            coop_matmul,
            cmd_pool,
            submit_lock: Mutex::new(()),
        })
    }

    /// Find a memory type index satisfying `type_bits` and `flags`.
    pub fn find_memory_type(&self, type_bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        let mp = &self.mem_props;
        (0..mp.memory_type_count).find(|&i| {
            (type_bits & (1 << i)) != 0
                && mp.memory_types[i as usize].property_flags.contains(flags)
        })
    }

    /// Like [`find_memory_type`], but chosen for an allocation of `size` bytes.
    /// Among the matching types, prefer one whose backing heap can actually hold
    /// `size`, then a *non*-DEVICE_LOCAL heap (GTT / system RAM on an APU — large),
    /// then the largest heap. This keeps an oversubscribed arena out of a tiny
    /// dedicated-VRAM BAR region (which OOMs) and in host memory the iGPU pages
    /// against — the Vulkan analogue of CUDA managed memory.
    pub fn find_memory_type_for_size(
        &self,
        type_bits: u32,
        flags: vk::MemoryPropertyFlags,
        size: u64,
    ) -> Option<u32> {
        let mp = &self.mem_props;
        (0..mp.memory_type_count)
            .filter(|&i| {
                (type_bits & (1 << i)) != 0
                    && mp.memory_types[i as usize].property_flags.contains(flags)
            })
            .max_by_key(|&i| {
                let ty = mp.memory_types[i as usize];
                let heap = mp.memory_heaps[ty.heap_index as usize];
                let fits = (heap.size >= size) as u8;
                let host =
                    (!ty.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)) as u8;
                (fits, host, heap.size)
            })
    }

    /// Record `record` into a one-shot primary command buffer, submit it to
    /// the compute queue, and block until the GPU finishes. Serialized
    /// across threads via `submit_lock` (queue submission needs external
    /// synchronization).
    pub fn submit_and_wait<F: FnOnce(vk::CommandBuffer)>(&self, record: F) {
        let _guard = self.submit_lock.lock().unwrap();
        let dev = &self.device;
        unsafe {
            let cmd = dev
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("vk alloc cmd buffer")[0];

            dev.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("vk begin cmd");

            record(cmd);

            dev.end_command_buffer(cmd).expect("vk end cmd");

            let fence = dev
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("vk fence");
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            dev.queue_submit(self.queue, &[submit], fence)
                .expect("vk submit");
            dev.wait_for_fences(&[fence], true, u64::MAX)
                .expect("vk wait");
            dev.destroy_fence(fence, None);
            dev.free_command_buffers(self.cmd_pool, &cmds);
        }
    }

    /// Allocate one reusable primary command buffer from the shared pool. The
    /// caller records it once and re-submits it many times via
    /// [`submit_recorded_wait`] (the schedule is static across runs — inputs
    /// flow through the host-visible arena, not the command stream — so a single
    /// recording is valid for every step). Free it with [`free_cmds`].
    pub fn alloc_primary_cmd(&self) -> vk::CommandBuffer {
        unsafe {
            self.device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .expect("vk alloc cmd buffer")[0]
        }
    }

    /// Free command buffers allocated from the shared pool.
    pub fn free_cmds(&self, cmds: &[vk::CommandBuffer]) {
        unsafe {
            self.device.free_command_buffers(self.cmd_pool, cmds);
        }
    }

    /// Create one unsignaled fence, reused across submits (reset after each
    /// wait). Avoids the per-step create/destroy of [`submit_and_wait`].
    pub fn create_reusable_fence(&self) -> vk::Fence {
        unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .expect("vk fence")
        }
    }

    /// Destroy a fence created by [`create_reusable_fence`].
    pub fn destroy_fence(&self, fence: vk::Fence) {
        unsafe {
            self.device.destroy_fence(fence, None);
        }
    }

    /// Submit an already-recorded command buffer and block until the GPU
    /// finishes, then reset `fence` for reuse. The command buffer must have been
    /// recorded WITHOUT `ONE_TIME_SUBMIT` so it can be resubmitted; since we
    /// wait here it is never pending at the next submit. Serialized via
    /// `submit_lock` (queue submission needs external synchronization).
    pub fn submit_recorded_wait(&self, cmd: vk::CommandBuffer, fence: vk::Fence) {
        let _guard = self.submit_lock.lock().unwrap();
        let dev = &self.device;
        unsafe {
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            dev.queue_submit(self.queue, &[submit], fence)
                .expect("vk submit");
            dev.wait_for_fences(&[fence], true, u64::MAX)
                .expect("vk wait");
            dev.reset_fences(&[fence]).expect("vk reset fence");
        }
    }

    /// Make GPU writes to a persistently-mapped HOST_VISIBLE arena visible to
    /// the host. Required when the memory type is HOST_CACHED (or a driver
    /// that does not honour HOST_COHERENT for device writes); safe no-op for
    /// uncached coherent heaps.
    pub fn invalidate_mapped(&self, memory: vk::DeviceMemory, offset: u64, size: u64) {
        let range = vk::MappedMemoryRange::default()
            .memory(memory)
            .offset(offset)
            .size(size);
        unsafe {
            self.device
                .invalidate_mapped_memory_ranges(&[range])
                .expect("vkInvalidateMappedMemoryRanges");
        }
    }

    pub fn flush_mapped(&self, memory: vk::DeviceMemory, offset: u64, size: u64) {
        let range = vk::MappedMemoryRange::default()
            .memory(memory)
            .offset(offset)
            .size(size);
        unsafe {
            self.device
                .flush_mapped_memory_ranges(&[range])
                .expect("vkFlushMappedMemoryRanges");
        }
    }
}
