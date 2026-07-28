// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Direct `amdxdna` ioctl path — the closest to the metal RLX can get on the
//! XDNA NPU.** No XRT, no C++ shim: this talks the in-kernel `amdxdna` DRM-accel
//! ABI straight on `/dev/accel*`, so RLX owns the whole submission path —
//! hardware context, buffer objects, command submit, and completion fence — and
//! (in a follow-on increment) the **user-mode-queue doorbell** for zero-syscall
//! dispatch.
//!
//! ## Why bother, when the XRT path already runs
//!
//! The XRT path ([`crate::npu_gemm`]) works and is bit-exact, but every dispatch
//! goes through XRT's userspace + a dlopen'd C++ shim. Straced on the 780M, the
//! hot path is exactly **one `EXEC_CMD` ioctl + one `SYNCOBJ_TIMELINE_WAIT`** per
//! run; setup is `CREATE_HWCTX` + `CONFIG_HWCTX` + N × (`CREATE_BO`/`GET_BO_INFO`).
//! Owning that directly lets us (a) drop the XRT/C++ dependency, (b) pipeline and
//! batch submissions our way, and (c) move to the UMQ ring + doorbell so a
//! dispatch is an MMIO store with the completion read from the ring's state word
//! — **no syscall at all**. That's the lowest dispatch overhead the hardware
//! offers, which is exactly what the latency-bound LLM-decode shape wants.
//!
//! ## ABI, pinned from the live kernel headers on the rig
//!
//! `/usr/include/drm/amdxdna_accel.h` (amdxdna, kernel 7.0) + `/usr/include/drm/drm.h`.
//! 11 amdxdna ioctls; the ones we use here:
//!   - `CREATE_HWCTX` → context handle + **`syncobj_handle`** (the completion
//!     fence) + **`umq_bo`/`umq_doorbell`** (the user-mode queue, for Level 2).
//!   - `CONFIG_HWCTX(CONFIG_CU)` → bind the compute unit = the xclbin's AIE
//!     partition PDI (extracted by the [`axlf`] parser below).
//!   - `CREATE_BO`/`GET_BO_INFO` → allocate device/shared buffers; `GET_BO_INFO`
//!     returns `map_offset` (for `mmap`) and `xdna_addr` (the device VA used in
//!     the command packet).
//!   - `EXEC_CMD` → submit a command chain, returns a fence `seq`.
//!   - completion via the generic `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` on the
//!     hwctx syncobj at `seq`.
//!
//! ## Status: code-complete, but PARKED on Phoenix
//!
//! The full path is implemented: device open + query ioctls + BO alloc/mmap
//! ([`probe`]), AXLF/PDI parsing ([`axlf`]), and the [`Gemm`] executor —
//! `CREATE_HWCTX` → `CONFIG_HWCTX(CONFIG_CU)` → BO setup → `EXEC_CMD` → syncobj
//! wait, with the command packet + PDI byte-verified against XRT. It is **parked**:
//! on Phoenix `npu1` the firmware accepts the submission (`errors=0`) but never
//! advances the command to completion, while XRT runs the identical packet fine —
//! and the cause is undiagnosable under Secure Boot lockdown. The **UMQ doorbell**
//! (Level 2, zero-syscall dispatch) is likewise unavailable here: `CREATE_HWCTX`
//! returns `umq_doorbell = 0` even when a ring BO is supplied — this build is
//! kernel-managed-queue only (UMQ needs XDNA2 / Strix). What *does* work from this
//! path and ships: the **TURBO** power-mode ioctl ([`Npu::set_turbo`]). Every entry
//! point returns a real error rather than masquerading — no CPU fallback.

/// Minimal AXLF (`.xclbin`) parser — extracts the AIE-partition PDI image (the
/// binary the driver loads into the CU BO for `CONFIG_HWCTX(CONFIG_CU)`) and the
/// partition column width. This is precisely the work XRT's `register_xclbin`
/// hides; owning it is what lets the direct path drop XRT. Layout mirrors
/// `xrt/detail/xclbin.h` (verified against the live header on the rig).
pub mod axlf {
    use std::io;

    // axlf: m_magic[8] + sig(4) + reserved[28] + keyBlock[256] + uniqueId(8) +
    // header(152). numSections (u32) sits at the tail of the inline header; the
    // section-header array follows it.
    const NUM_SECTIONS_OFF: usize = 448;
    const SEC_HDR_OFF: usize = 456;
    const SEC_HDR_SZ: usize = 40; // kind@0, name[16]@4, offset@24, size@32
    const KIND_AIE_PARTITION: u32 = 32;

    fn u32_at(b: &[u8], off: usize) -> io::Result<u32> {
        b.get(off..off + 4)
            .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
            .ok_or_else(|| io::Error::other("axlf: u32 read out of bounds"))
    }
    fn u64_at(b: &[u8], off: usize) -> io::Result<u64> {
        b.get(off..off + 8)
            .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
            .ok_or_else(|| io::Error::other("axlf: u64 read out of bounds"))
    }

    /// The pieces the direct path needs out of an `.xclbin`.
    pub struct Partition {
        /// The PDI image bytes → go into the CU DEV BO.
        pub pdi: Vec<u8>,
        /// Partition column width (npu1 has 4 core tiles/column → num_tiles = 4·width).
        pub column_width: u16,
    }

    /// File offset + size of the `AIE_PARTITION` section, or an error.
    fn aie_partition_section(xclbin: &[u8]) -> io::Result<usize> {
        if xclbin.get(0..8) != Some(b"xclbin2\0".as_slice()) {
            return Err(io::Error::other("not an AXLF container (bad magic)"));
        }
        let n = u32_at(xclbin, NUM_SECTIONS_OFF)? as usize;
        for i in 0..n {
            let h = SEC_HDR_OFF + i * SEC_HDR_SZ;
            if u32_at(xclbin, h)? == KIND_AIE_PARTITION {
                return Ok(u64_at(xclbin, h + 24)? as usize); // m_sectionOffset
            }
        }
        Err(io::Error::other("xclbin has no AIE_PARTITION section"))
    }

    /// Parse the AIE partition: PDI image + column width.
    pub fn parse(xclbin: &[u8]) -> io::Result<Partition> {
        let s = aie_partition_section(xclbin)?;
        // aie_partition @ s: info @ +32 (column_width u16 @ info+0); aie_pdi
        // array_offset @ +120 → { size(count)@120, offset@124 }.
        let column_width = u32_at(xclbin, s + 32)? as u16;
        let pdi_count = u32_at(xclbin, s + 120)?;
        let pdi_arr_off = u32_at(xclbin, s + 124)? as usize;
        if pdi_count == 0 {
            return Err(io::Error::other("aie_partition declares 0 PDIs"));
        }
        // aie_pdi[0] @ s+pdi_arr_off: uuid[16], pdi_image array_offset @ +16
        // → { size(bytes)@16, offset@20 } (offset from section start).
        let p = s + pdi_arr_off;
        let img_size = u32_at(xclbin, p + 16)? as usize;
        let img_off = u32_at(xclbin, p + 20)? as usize;
        let start = s + img_off;
        let pdi = xclbin
            .get(start..start + img_size)
            .ok_or_else(|| io::Error::other("PDI image out of bounds"))?
            .to_vec();
        Ok(Partition { pdi, column_width })
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use std::io;
    /// The direct path is Linux-only (the `amdxdna` DRM-accel driver is
    /// Linux-only). Every other OS returns a clear error.
    pub fn probe() -> io::Result<String> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "rlx-xdna direct path is Linux-only (amdxdna is a Linux DRM-accel driver)",
        ))
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::os::fd::AsRawFd;
    use std::os::raw::c_void;

    // ── ioctl encoding (asm-generic; DRM uses type 'd') ──────────────────────
    const IOC_WRITE: u32 = 1;
    const IOC_READ: u32 = 2;
    const DRM_IOCTL_TYPE: u32 = 0x64; // 'd'
    const DRM_COMMAND_BASE: u32 = 0x40;

    /// `_IOC(dir, type, nr, size)` — the Linux ioctl request encoding.
    const fn ioc(dir: u32, ty: u32, nr: u32, size: usize) -> u32 {
        (dir << 30) | ((size as u32) << 16) | (ty << 8) | nr
    }
    /// `DRM_IOWR(nr, ty)` — read/write DRM ioctl on a struct of the given size.
    const fn drm_iowr(nr: u32, size: usize) -> u32 {
        ioc(IOC_READ | IOC_WRITE, DRM_IOCTL_TYPE, nr, size)
    }

    // amdxdna ioctl ids (enum amdxdna_drm_ioctl_id), offset by DRM_COMMAND_BASE.
    const CREATE_HWCTX: u32 = 0;
    const DESTROY_HWCTX: u32 = 1;
    const CONFIG_HWCTX: u32 = 2;
    const CREATE_BO: u32 = 3;
    const GET_BO_INFO: u32 = 4;
    #[allow(dead_code)]
    const SYNC_BO: u32 = 5;
    const EXEC_CMD: u32 = 6;
    const GET_INFO: u32 = 7;
    const SET_STATE: u32 = 8;

    // SET_STATE params (enum amdxdna_drm_set_param) + power modes
    // (enum amdxdna_power_mode_type).
    const SET_POWER_MODE: u32 = 0;
    #[allow(dead_code)]
    const POWER_MODE_DEFAULT: u8 = 0;
    #[allow(dead_code)]
    const POWER_MODE_LOW: u8 = 1;
    #[allow(dead_code)]
    const POWER_MODE_MEDIUM: u8 = 2;
    #[allow(dead_code)]
    const POWER_MODE_HIGH: u8 = 3;
    const POWER_MODE_TURBO: u8 = 4;

    // Generic DRM ioctls we need (from drm.h).
    const DRM_GEM_CLOSE_NR: u32 = 0x09;
    const DRM_SYNCOBJ_CREATE_NR: u32 = 0xBF;
    const DRM_SYNCOBJ_DESTROY_NR: u32 = 0xC0;
    const DRM_SYNCOBJ_TIMELINE_WAIT_NR: u32 = 0xCA;

    // BO types (enum amdxdna_bo_type): INVALID=0, SHMEM, DEV_HEAP, DEV, CMD.
    const BO_SHMEM: u32 = 1;
    const BO_DEV_HEAP: u32 = 2;
    const BO_DEV: u32 = 3;
    const BO_CMD: u32 = 4;

    // GET_INFO params (enum amdxdna_drm_get_param).
    const QUERY_AIE_VERSION: u32 = 2;
    const QUERY_AIE_METADATA: u32 = 1;
    const QUERY_HW_CONTEXTS: u32 = 5;

    // ── repr(C) structs — exact mirrors of the UAPI headers ──────────────────
    #[repr(C)]
    #[derive(Default)]
    struct QueryAieVersion {
        major: u32,
        minor: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct GetInfo {
        param: u32,
        buffer_size: u32,
        buffer: u64,
    }

    // amdxdna_drm_query_hwctx — per-context firmware state (start_col, counters).
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct QueryHwctx {
        context_id: u32,
        start_col: u32,
        num_col: u32,
        pad: u32,
        pid: i64,
        command_submissions: u64,
        command_completions: u64,
        migrations: u64,
        preemptions: u64,
        errors: u64,
    }

    // Only the leading fields matter for the probe; a generous buffer covers the
    // trailing tile-metadata we don't decode.
    #[repr(C)]
    #[derive(Default)]
    struct AieMetadataHead {
        col_size: u32,
        cols: u16,
        rows: u16,
        version_major: u32,
        version_minor: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct CreateHwctx {
        ext: u64,
        ext_flags: u64,
        qos_p: u64,
        umq_bo: u32,
        log_buf_bo: u32,
        max_opc: u32,
        num_tiles: u32,
        mem_size: u32,
        umq_doorbell: u32,
        handle: u32,
        syncobj_handle: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct DestroyHwctx {
        handle: u32,
        pad: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct QosInfo {
        gops: u32,
        fps: u32,
        dma_bandwidth: u32,
        latency: u32,
        frame_exec_time: u32,
        priority: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct SetState {
        param: u32,
        buffer_size: u32,
        buffer: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct SetPowerMode {
        power_mode: u8,
        pad: [u8; 7],
    }

    #[repr(C)]
    #[derive(Default)]
    struct CreateBo {
        flags: u64,
        vaddr: u64,
        size: u64,
        ty: u32,
        handle: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct GetBoInfo {
        ext: u64,
        ext_flags: u64,
        handle: u32,
        pad: u32,
        map_offset: u64,
        vaddr: u64,
        xdna_addr: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct GemClose {
        handle: u32,
        pad: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct SyncobjCreate {
        handle: u32,
        flags: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct SyncobjDestroy {
        handle: u32,
        pad: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct SyncobjTimelineWait {
        handles: u64,
        points: u64,
        timeout_nsec: i64,
        count_handles: u32,
        flags: u32,
        first_signaled: u32,
        pad: u32,
        deadline_nsec: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct ConfigHwctx {
        handle: u32,
        param_type: u32,
        param_val: u64,
        param_val_size: u32,
        pad: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct ExecCmd {
        ext: u64,
        ext_flags: u64,
        hwctx: u32,
        ty: u32,
        cmd_handles: u64,
        args: u64,
        cmd_count: u32,
        arg_count: u32,
        seq: u64,
    }

    // CONFIG_HWCTX param_type for the CU config.
    const HWCTX_CONFIG_CU: u32 = 0;
    // EXEC_CMD type: submit an executable command buffer.
    const CMD_SUBMIT_EXEC_BUF: u32 = 0;

    /// Flush the CPU cache for `[ptr, ptr+len)` so the NPU (whose reads are not
    /// guaranteed cache-coherent) sees our writes — the direct equivalent of
    /// XRT's `bo.sync(TO_DEVICE)` / the driver's `drm_clflush_virt_range`.
    #[cfg(target_arch = "x86_64")]
    fn clflush_range(ptr: *const u8, len: usize) {
        use std::arch::x86_64::{_mm_clflush, _mm_sfence};
        if len == 0 {
            return;
        }
        let mut a = (ptr as usize) & !63usize; // align down to a cache line
        let end = ptr as usize + len;
        while a < end {
            unsafe { _mm_clflush(a as *const u8) };
            a += 64;
        }
        unsafe { _mm_sfence() };
    }
    #[cfg(not(target_arch = "x86_64"))]
    fn clflush_range(_ptr: *const u8, _len: usize) {}

    /// Raw `ioctl` wrapper: request number + a mutable struct pointer. Returns the
    /// OS error on failure (negative return).
    unsafe fn ioctl<T>(fd: i32, req: u32, arg: &mut T) -> io::Result<()> {
        let r = unsafe { libc::ioctl(fd, req as libc::c_ulong, arg as *mut T as *mut c_void) };
        if r < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    /// An open handle to the NPU DRM-accel device (`/dev/accel/accel0`).
    pub struct Npu {
        file: File,
    }

    impl Npu {
        /// Open the accel device. `path` empty → `/dev/accel/accel0`.
        pub fn open(path: &str) -> io::Result<Self> {
            let path = if path.is_empty() {
                "/dev/accel/accel0"
            } else {
                path
            };
            let file = OpenOptions::new().read(true).write(true).open(path)?;
            Ok(Self { file })
        }

        fn fd(&self) -> i32 {
            self.file.as_raw_fd()
        }

        /// `(major, minor)` of the AIE array — the cheapest query, proves the
        /// device speaks the amdxdna ABI.
        pub fn aie_version(&self) -> io::Result<(u32, u32)> {
            let mut v = QueryAieVersion::default();
            let mut gi = GetInfo {
                param: QUERY_AIE_VERSION,
                buffer_size: std::mem::size_of::<QueryAieVersion>() as u32,
                buffer: &mut v as *mut _ as u64,
            };
            unsafe { ioctl(self.fd(), drm_iowr(DRM_COMMAND_BASE + GET_INFO, std::mem::size_of::<GetInfo>()), &mut gi)? };
            Ok((v.major, v.minor))
        }

        /// `(cols, rows, col_size)` of the tile array — needed to size a hwctx.
        pub fn aie_metadata(&self) -> io::Result<(u16, u16, u32)> {
            // Exactly sizeof(struct amdxdna_drm_query_aie_metadata) — the value
            // XRT passes. The driver treats buffer_size as in/out and an oversized
            // request is a different code path, so match it precisely.
            let mut buf = [0u8; 64];
            let mut gi = GetInfo {
                param: QUERY_AIE_METADATA,
                buffer_size: buf.len() as u32,
                buffer: buf.as_mut_ptr() as u64,
            };
            unsafe { ioctl(self.fd(), drm_iowr(DRM_COMMAND_BASE + GET_INFO, std::mem::size_of::<GetInfo>()), &mut gi)? };
            // SAFETY: the driver populated at least the head fields.
            let head = unsafe { &*(buf.as_ptr() as *const AieMetadataHead) };
            Ok((head.cols, head.rows, head.col_size))
        }

        /// Query per-hwctx firmware state (start_col/num_col + submission/
        /// completion/error counters) — the direct read of whether a command
        /// reached the firmware, completed, or errored, and where the partition
        /// landed. Returns a human report.
        pub fn hwctx_report(&self) -> io::Result<String> {
            use std::fmt::Write;
            let sz = std::mem::size_of::<QueryHwctx>();
            let mut buf = vec![0u8; sz * 16];
            let mut gi = GetInfo {
                param: QUERY_HW_CONTEXTS,
                buffer_size: buf.len() as u32,
                buffer: buf.as_mut_ptr() as u64,
            };
            unsafe { ioctl(self.fd(), drm_iowr(DRM_COMMAND_BASE + GET_INFO, std::mem::size_of::<GetInfo>()), &mut gi)? };
            let n = (gi.buffer_size as usize) / sz;
            let mut r = String::new();
            for i in 0..n {
                // SAFETY: the driver filled n entries of QueryHwctx.
                let h = unsafe { &*(buf.as_ptr().add(i * sz) as *const QueryHwctx) };
                writeln!(
                    r,
                    "  hwctx ctx_id={} pid={} start_col={} num_col={} submit={} complete={} errors={} migr={} preempt={}",
                    h.context_id, h.pid, h.start_col, h.num_col,
                    h.command_submissions, h.command_completions, h.errors,
                    h.migrations, h.preemptions
                )
                .ok();
            }
            if n == 0 {
                writeln!(r, "  (no hwctx entries returned)").ok();
            }
            Ok(r)
        }

        /// Allocate a host-shared BO of `size` bytes, `mmap` it, and hand back a
        /// [`Bo`] owning both. Driver-allocated pages (zero-copy userptr is a
        /// later optimization).
        pub fn alloc_shmem(&self, size: usize) -> io::Result<Bo<'_>> {
            let mut cb = CreateBo {
                size: size as u64,
                ty: BO_SHMEM,
                ..Default::default()
            };
            unsafe { ioctl(self.fd(), drm_iowr(DRM_COMMAND_BASE + CREATE_BO, std::mem::size_of::<CreateBo>()), &mut cb)? };
            let handle = cb.handle;

            let mut info = GetBoInfo {
                handle,
                ..Default::default()
            };
            unsafe {
                ioctl(self.fd(), drm_iowr(DRM_COMMAND_BASE + GET_BO_INFO, std::mem::size_of::<GetBoInfo>()), &mut info)
                    .inspect_err(|_| self.gem_close(handle))?
            };

            // mmap the BO into our address space via the returned offset.
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    self.fd(),
                    info.map_offset as libc::off_t,
                )
            };
            if ptr == libc::MAP_FAILED {
                let e = io::Error::last_os_error();
                self.gem_close(handle);
                return Err(e);
            }
            Ok(Bo {
                npu: self,
                handle,
                ptr: ptr as *mut u8,
                size,
                xdna_addr: info.xdna_addr,
            })
        }

        fn gem_close(&self, handle: u32) {
            let mut gc = GemClose { handle, pad: 0 };
            let _ = unsafe { ioctl(self.fd(), drm_iowr(DRM_GEM_CLOSE_NR, std::mem::size_of::<GemClose>()), &mut gc) };
        }

        /// Create a DRM syncobj, returning its handle. `signaled` starts it with
        /// an already-signaled fence (`DRM_SYNCOBJ_CREATE_SIGNALED`) — useful for
        /// validating the wait path standalone. (The hwctx returns one of these
        /// already; the exec path waits on it at the `seq` `EXEC_CMD` returns.)
        pub fn syncobj_create(&self, signaled: bool) -> io::Result<u32> {
            let mut sc = SyncobjCreate {
                flags: if signaled { 1 } else { 0 }, // DRM_SYNCOBJ_CREATE_SIGNALED
                ..Default::default()
            };
            unsafe { ioctl(self.fd(), drm_iowr(DRM_SYNCOBJ_CREATE_NR, std::mem::size_of::<SyncobjCreate>()), &mut sc)? };
            Ok(sc.handle)
        }

        pub fn syncobj_destroy(&self, handle: u32) {
            let mut sd = SyncobjDestroy { handle, pad: 0 };
            let _ = unsafe { ioctl(self.fd(), drm_iowr(DRM_SYNCOBJ_DESTROY_NR, std::mem::size_of::<SyncobjDestroy>()), &mut sd) };
        }

        /// Wait for timeline `point` on `syncobj` up to `timeout_ns`. Returns
        /// `Ok(true)` if signaled, `Ok(false)` on timeout, `Err` on a real fault.
        /// This is the completion primitive the `EXEC_CMD` path waits on.
        pub fn syncobj_timeline_wait(
            &self,
            syncobj: u32,
            point: u64,
            timeout_ns: i64,
        ) -> io::Result<bool> {
            let h = syncobj;
            let p = point;
            let mut w = SyncobjTimelineWait {
                handles: &h as *const u32 as u64,
                points: &p as *const u64 as u64,
                timeout_nsec: timeout_ns,
                count_handles: 1,
                flags: 0,
                ..Default::default()
            };
            match unsafe {
                ioctl(self.fd(), drm_iowr(DRM_SYNCOBJ_TIMELINE_WAIT_NR, std::mem::size_of::<SyncobjTimelineWait>()), &mut w)
            } {
                Ok(()) => Ok(true),
                Err(e) if e.raw_os_error() == Some(libc::ETIME) => Ok(false),
                Err(e) => Err(e),
            }
        }

        /// Best-effort `CREATE_HWCTX`/`DESTROY_HWCTX` round-trip for `num_tiles`.
        /// The exact `num_tiles`/`max_opc`/`mem_size` a partition wants is what
        /// the next increment nails down; here we report the driver's verdict so
        /// we learn the accepted parameters on real hardware.
        pub fn hwctx_roundtrip(&self, num_tiles: u32) -> io::Result<Hwctx> {
            let qos = QosInfo {
                priority: 0x100, // "normal" — XRT's default realtime band
                ..Default::default()
            };
            let mut c = CreateHwctx {
                qos_p: &qos as *const QosInfo as u64,
                max_opc: 0,
                num_tiles,
                mem_size: 0,
                ..Default::default()
            };
            unsafe { ioctl(self.fd(), drm_iowr(DRM_COMMAND_BASE + CREATE_HWCTX, std::mem::size_of::<CreateHwctx>()), &mut c)? };
            Ok(Hwctx {
                handle: c.handle,
                syncobj: c.syncobj_handle,
                umq_bo: c.umq_bo,
                umq_doorbell: c.umq_doorbell,
            })
        }

        pub fn destroy_hwctx(&self, handle: u32) {
            let mut d = DestroyHwctx { handle, pad: 0 };
            let _ = unsafe { ioctl(self.fd(), drm_iowr(DRM_COMMAND_BASE + DESTROY_HWCTX, std::mem::size_of::<DestroyHwctx>()), &mut d) };
        }

        /// Create a hardware context sized for the overlay. `num_tiles`/`max_opc`
        /// mirror what XRT passes for this class of overlay (captured via the
        /// ioctl interposer on the known-good path). Returns the ctx + its
        /// completion syncobj.
        pub fn create_hwctx(&self, num_tiles: u32, max_opc: u32) -> io::Result<Hwctx> {
            self.create_hwctx_umq(num_tiles, max_opc, 0)
        }

        /// Set the NPU power mode (`SET_STATE`/`SET_POWER_MODE`) — device-global DPM.
        /// `POWER_MODE_TURBO` clocks the array to its maximum; this is INDEPENDENT of
        /// the (KMQ) exec path, so it speeds up the working XRT compute path too. The
        /// mode persists on the device until changed; hold the fd for the session so
        /// it isn't reset. Needs the accel device (no hwctx / heap prerequisite).
        pub fn set_power_mode(&self, mode: u8) -> io::Result<()> {
            let pm = SetPowerMode { power_mode: mode, pad: [0; 7] };
            let mut st = SetState {
                param: SET_POWER_MODE,
                buffer_size: std::mem::size_of::<SetPowerMode>() as u32,
                buffer: &pm as *const SetPowerMode as u64,
            };
            unsafe { ioctl(self.fd(), drm_iowr(DRM_COMMAND_BASE + SET_STATE, std::mem::size_of::<SetState>()), &mut st) }
        }

        /// Convenience: request maximum NPU clocks (`POWER_MODE_TURBO`).
        pub fn set_turbo(&self) -> io::Result<()> {
            self.set_power_mode(POWER_MODE_TURBO)
        }

        /// `create_hwctx` requesting a **user-mode queue**: `umq_bo` is a user-owned
        /// ring-buffer BO handle (0 ⇒ kernel-managed queue). Per the amdxdna ABI
        /// (`amdxdna_drm_create_hwctx.umq_bo` is an INPUT), supplying a ring BO asks
        /// the driver to bind a UMQ and return `umq_doorbell` (the MMIO offset to ring
        /// for zero-syscall dispatch). On a KMQ-only device the driver either rejects
        /// a non-zero `umq_bo` or returns `umq_doorbell = 0`.
        pub fn create_hwctx_umq(&self, num_tiles: u32, max_opc: u32, umq_bo: u32) -> io::Result<Hwctx> {
            let qos = QosInfo::default();
            let mut c = CreateHwctx {
                qos_p: &qos as *const QosInfo as u64,
                umq_bo,
                max_opc,
                num_tiles,
                ..Default::default()
            };
            unsafe { ioctl(self.fd(), drm_iowr(DRM_COMMAND_BASE + CREATE_HWCTX, std::mem::size_of::<CreateHwctx>()), &mut c)? };
            Ok(Hwctx {
                handle: c.handle,
                syncobj: c.syncobj_handle,
                umq_bo: c.umq_bo,
                umq_doorbell: c.umq_doorbell,
            })
        }

        /// Bind the hardware context's compute unit to the PDI already loaded in
        /// `cu_bo` (a DEV BO). Mirrors XRT's `CONFIG_HWCTX(CONFIG_CU, num_cus=1)`.
        pub fn config_cu(&self, hwctx: u32, cu_bo: u32) -> io::Result<()> {
            // param_val = struct amdxdna_hwctx_param_config_cu:
            //   u16 num_cus; u16 pad[3]; { u32 cu_bo; u8 cu_func; u8 pad[3] } cu[1]
            // → 16 bytes total (matches the captured val_size=16).
            let mut cu = [0u8; 16];
            cu[0..2].copy_from_slice(&1u16.to_le_bytes()); // num_cus = 1
            cu[8..12].copy_from_slice(&cu_bo.to_le_bytes()); // cu[0].cu_bo
            // cu[0].cu_func = 0 (cu[12]) already zero.
            let mut cfg = ConfigHwctx {
                handle: hwctx,
                param_type: HWCTX_CONFIG_CU,
                param_val: cu.as_ptr() as u64,
                param_val_size: cu.len() as u32,
                pad: 0,
            };
            unsafe { ioctl(self.fd(), drm_iowr(DRM_COMMAND_BASE + CONFIG_HWCTX, std::mem::size_of::<ConfigHwctx>()), &mut cfg) }
        }

        /// Low-level `CREATE_BO` + `GET_BO_INFO`. Returns `(handle, map_offset,
        /// xdna_addr)`. `map_offset == u64::MAX` for DEV BOs (accessed via the
        /// heap mapping); `xdna_addr == u64::MAX` for SHMEM BOs (host-VA / SVA).
        pub fn create_bo_raw(&self, ty: u32, size: usize) -> io::Result<(u32, u64, u64)> {
            let mut cb = CreateBo {
                size: size as u64,
                ty,
                ..Default::default()
            };
            unsafe { ioctl(self.fd(), drm_iowr(DRM_COMMAND_BASE + CREATE_BO, std::mem::size_of::<CreateBo>()), &mut cb)? };
            let handle = cb.handle;
            let mut info = GetBoInfo {
                handle,
                ..Default::default()
            };
            match unsafe { ioctl(self.fd(), drm_iowr(DRM_COMMAND_BASE + GET_BO_INFO, std::mem::size_of::<GetBoInfo>()), &mut info) } {
                Ok(()) => Ok((handle, info.map_offset, info.xdna_addr)),
                Err(e) => {
                    self.gem_close(handle);
                    Err(e)
                }
            }
        }

        /// `mmap` a BO by its `GET_BO_INFO` map offset.
        pub fn mmap_bo(&self, map_offset: u64, size: usize) -> io::Result<*mut u8> {
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    self.fd(),
                    map_offset as libc::off_t,
                )
            };
            if p == libc::MAP_FAILED {
                Err(io::Error::last_os_error())
            } else {
                Ok(p as *mut u8)
            }
        }

        /// `mmap` a BO at a user VA aligned to `align` bytes. The firmware's
        /// MAP_HOST_BUFFER rejects a heap whose user VA isn't aligned to the
        /// device-memory window (`AIE2_STATUS_INVALID_PARAM`) — XRT's 64 MiB heap
        /// lands 64 MiB-aligned, and a plain `mmap` does not guarantee that. We
        /// reserve `size + align`, carve out an aligned sub-range, and `MAP_FIXED`
        /// the BO onto it. (Single-threaded setup, so the unmap→map gap is safe.)
        pub fn mmap_bo_aligned(&self, map_offset: u64, size: usize, align: usize) -> io::Result<*mut u8> {
            let reserve = size + align;
            let base = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    reserve,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            if base == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            let aligned = ((base as usize) + align - 1) & !(align - 1);
            unsafe { libc::munmap(base, reserve) };
            let p = unsafe {
                libc::mmap(
                    aligned as *mut c_void,
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    self.fd(),
                    map_offset as libc::off_t,
                )
            };
            if p == libc::MAP_FAILED {
                Err(io::Error::last_os_error())
            } else {
                Ok(p as *mut u8)
            }
        }

        pub fn close_bo(&self, handle: u32) {
            self.gem_close(handle);
        }

        /// Submit a prepared command BO and return the fence sequence to wait on.
        /// `cmd_bo` is the CMD BO holding the packet; `args` are the operand BO
        /// handles the command references (the driver pins/patches them).
        pub fn exec(&self, hwctx: u32, cmd_bo: u32, args: &[u32]) -> io::Result<u64> {
            let mut e = ExecCmd {
                hwctx,
                ty: CMD_SUBMIT_EXEC_BUF,
                cmd_handles: cmd_bo as u64, // cmd_count==1 → the handle itself
                args: args.as_ptr() as u64,
                cmd_count: 1,
                arg_count: args.len() as u32,
                ..Default::default()
            };
            unsafe { ioctl(self.fd(), drm_iowr(DRM_COMMAND_BASE + EXEC_CMD, std::mem::size_of::<ExecCmd>()), &mut e)? };
            Ok(e.seq)
        }
    }

    /// A hardware context: the driver's per-client AIE partition, its completion
    /// syncobj, and the user-mode-queue ring/doorbell (Level 2).
    pub struct Hwctx {
        pub handle: u32,
        pub syncobj: u32,
        pub umq_bo: u32,
        pub umq_doorbell: u32,
    }

    /// A buffer object: device handle + host mapping + device VA. `Drop` unmaps
    /// and closes it.
    pub struct Bo<'n> {
        npu: &'n Npu,
        handle: u32,
        ptr: *mut u8,
        size: usize,
        /// Device virtual address — what goes into the command packet.
        pub xdna_addr: u64,
    }

    impl Bo<'_> {
        /// Host-visible bytes of the mapping.
        pub fn as_mut_slice(&mut self) -> &mut [u8] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
        }
        pub fn as_slice(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
        }
        pub fn handle(&self) -> u32 {
            self.handle
        }
    }

    impl Drop for Bo<'_> {
        fn drop(&mut self) {
            unsafe { libc::munmap(self.ptr as *mut c_void, self.size) };
            self.npu.gem_close(self.handle);
        }
    }

    /// A persistent INT8-GEMM context on the NPU driven **entirely through the
    /// direct amdxdna ioctls** — no XRT, no C++ shim. Mirrors
    /// [`crate::npu_gemm::NpuGemm`]'s shape (`open` once, `run` many) so it can
    /// slot into `XdnaBackend`. `open` does the one-time setup (heap, hwctx,
    /// PDI-config, instruction + operand BOs); `run` is the hot path
    /// (`EXEC_CMD` + syncobj wait — the 2-syscall dispatch the interposer
    /// confirmed XRT itself uses).
    pub struct Gemm {
        npu: Npu,
        hwctx: u32,
        syncobj: u32,
        // Operand host mappings — for SHMEM BOs these host VAs are what the NPU
        // dereferences directly (IOMMU shared virtual addressing), so they go
        // straight into the command packet.
        a_host: *mut u8,
        b_host: *mut u8,
        c_host: *mut u8,
        cmd_host: *mut u8,
        // Operand device-aperture addresses (they're DEV BOs in the heap window,
        // reached the same way as the instructions rather than via SVA).
        a_xdna: u64,
        b_xdna: u64,
        c_xdna: u64,
        a_handle: u32,
        b_handle: u32,
        c_handle: u32,
        cmd_handle: u32,
        instr_handle: u32,
        instr_xdna: u64,
        ninstr: u32,
        m: usize,
        k: usize,
        n: usize,
        bo_handles: Vec<u32>,
        maps: Vec<(*mut u8, usize)>,
    }

    // The context is driven single-threaded; the raw pointers are into our own
    // device mappings.
    unsafe impl Send for Gemm {}

    impl Gemm {
        /// One-time setup for an INT8 GEMM overlay. `xclbin` is the raw `.xclbin`
        /// bytes (we extract the PDI ourselves — no XRT), `insts` the paired
        /// instruction stream (u32 words). `m/k/n` are the overlay's compiled
        /// dims.
        pub fn open(xclbin: &[u8], insts: &[u32], m: usize, k: usize, n: usize) -> io::Result<Self> {
            const HEAP_SIZE: usize = 64 << 20; // 64 MiB, as XRT allocates
            let part = super::axlf::parse(xclbin)?;

            // Annotate which ioctl failed — bring-up on a raw kernel ABI.
            fn step<T>(name: &str, r: io::Result<T>) -> io::Result<T> {
                r.map_err(|e| io::Error::new(e.kind(), format!("{name}: {e}")))
            }

            let npu = Npu::open("")?;
            // Optional: hold TURBO on THIS fd for the exec (needs root). Keeps the
            // array clocked; pair with disabling runtime-PM autosuspend
            // (power/control=on) to keep it RESUMED so the firmware runs the command
            // (the `ert state=NEW`/never-completes hang looks power-state related —
            // runtime_status is `suspended` at idle).
            if std::env::var("RLX_XDNA_TURBO").is_ok() {
                match npu.set_turbo() {
                    Ok(()) => eprintln!("[turbo] exec fd → TURBO (max DPM)"),
                    Err(e) => eprintln!("[turbo] set_turbo on exec fd failed: {e}"),
                }
            }
            let mut bo_handles: Vec<u32> = Vec::new();
            let mut maps: Vec<(*mut u8, usize)> = Vec::new();

            // 1) Device heap — DEV BOs are carved from it; its host mapping is how
            //    we fill DEV BOs (they have no map_offset of their own).
            let (heap_h, heap_off, heap_xdna) = step("create DEV_HEAP", npu.create_bo_raw(BO_DEV_HEAP, HEAP_SIZE))?;
            bo_handles.push(heap_h);
            // The firmware maps the heap onto the 64 MiB device-memory window and
            // rejects a user VA not aligned to it — so align the heap mmap to its
            // own size (mirrors XRT, whose 64 MiB heap lands 64 MiB-aligned).
            let heap_host = step("mmap heap", npu.mmap_bo_aligned(heap_off, HEAP_SIZE, HEAP_SIZE))?;
            maps.push((heap_host, HEAP_SIZE));
            // Fault every heap page in NOW. aie2_hwctx_init pins the heap and
            // hands its userptr to the firmware (host buffer map via IOMMU SVA);
            // if the pages aren't resident the firmware map is rejected
            // ("Map host buffer failed"). Touching them makes them present.
            unsafe { std::ptr::write_bytes(heap_host, 0, HEAP_SIZE) };
            // Host pointer for a DEV BO at device address `xdna`.
            let dev_host = |xdna: u64| -> *mut u8 { unsafe { heap_host.add((xdna - heap_xdna) as usize) } };

            // 2) Hardware context (XRT order: heap → GET_INFO → hwctx → CU BOs).
            // The interposer showed XRT issues a GET_INFO between the heap and
            // CREATE_HWCTX, and the driver requires it — creating a context
            // without a prior query returns EINVAL. Mirror it.
            let _ = step("get_info (hwctx prereq)", npu.aie_metadata())?;
            let num_tiles = 4 * part.column_width as u32; // npu1: 4 core tiles/column
            // RLX_XDNA_UMQ=1 REQUESTS a user-mode queue: allocate a ring BO and pass
            // its handle as `umq_bo`. Decisive feasibility test for zero-syscall
            // dispatch — on KMQ-only Phoenix the driver rejects it or returns
            // doorbell=0; on a UMQ device it returns a live doorbell offset.
            let ctx = if std::env::var("RLX_XDNA_UMQ").is_ok() {
                let ring_bytes = std::env::var("RLX_XDNA_UMQ_RING")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0x2000usize);
                let (ring_h, _, ring_xdna) = step("create UMQ ring BO", npu.create_bo_raw(BO_DEV, ring_bytes))?;
                bo_handles.push(ring_h);
                let c = step("create_hwctx_umq", npu.create_hwctx_umq(num_tiles, 2048, ring_h))?;
                let avail = if c.umq_doorbell == 0 { "KMQ-only (no doorbell)" } else { "UMQ available" };
                eprintln!(
                    "[umq] requested ring_bo={ring_h} ({ring_bytes}B @ 0x{ring_xdna:x}) → doorbell=0x{:x} [{avail}]",
                    c.umq_doorbell
                );
                c
            } else {
                step("create_hwctx", npu.create_hwctx(num_tiles, 2048))?
            };

            // 3) PDI into a DEV BO, then bind it as the context's compute unit.
            let (pdi_h, _, pdi_xdna) = step("create PDI BO", npu.create_bo_raw(BO_DEV, part.pdi.len()))?;
            bo_handles.push(pdi_h);
            unsafe { std::ptr::copy_nonoverlapping(part.pdi.as_ptr(), dev_host(pdi_xdna), part.pdi.len()) };
            clflush_range(dev_host(pdi_xdna), part.pdi.len());
            step("config_cu", npu.config_cu(ctx.handle, pdi_h))?;

            // 4) Instruction stream into a DEV BO (device-addressable → its
            //    xdna_addr goes in the packet).
            let insts_bytes = insts.len() * 4;
            let (instr_h, _, instr_xdna) = step("create instr BO", npu.create_bo_raw(BO_DEV, insts_bytes))?;
            bo_handles.push(instr_h);
            unsafe { std::ptr::copy_nonoverlapping(insts.as_ptr() as *const u8, dev_host(instr_xdna), insts_bytes) };
            clflush_range(dev_host(instr_xdna), insts_bytes);

            // 5) Operands as host-shared (SHMEM) BOs reached via SVA host-VAs
            //    (XRT's model). The AMD-Vi IO_PAGE_FAULTs proved device-aperture
            //    addresses don't work for operand DMA — the shim DMA treats
            //    operand addresses as host VAs. The packet uses their mmap VAs.
            let (a_bytes, b_bytes, c_bytes) = (m * k, k * n, m * n * 4);
            let (a_handle, a_off, _) = step("create A BO", npu.create_bo_raw(BO_SHMEM, a_bytes))?;
            bo_handles.push(a_handle);
            let a_host = step("mmap A", npu.mmap_bo(a_off, a_bytes))?;
            maps.push((a_host, a_bytes));
            let a_xdna = a_host as u64;
            let (b_handle, b_off, _) = step("create B BO", npu.create_bo_raw(BO_SHMEM, b_bytes))?;
            bo_handles.push(b_handle);
            let b_host = step("mmap B", npu.mmap_bo(b_off, b_bytes))?;
            maps.push((b_host, b_bytes));
            let b_xdna = b_host as u64;
            let (c_handle, c_off, _) = step("create C BO", npu.create_bo_raw(BO_SHMEM, c_bytes))?;
            bo_handles.push(c_handle);
            let c_host = step("mmap C", npu.mmap_bo(c_off, c_bytes))?;
            maps.push((c_host, c_bytes));
            let c_xdna = c_host as u64;
            let (cmd_handle, cmd_off, _) = step("create CMD BO", npu.create_bo_raw(BO_CMD, 4096))?;
            bo_handles.push(cmd_handle);
            let cmd_host = step("mmap CMD", npu.mmap_bo(cmd_off, 4096))?;
            maps.push((cmd_host, 4096));

            Ok(Self {
                npu,
                hwctx: ctx.handle,
                syncobj: ctx.syncobj,
                a_host,
                b_host,
                c_host,
                cmd_host,
                a_xdna,
                b_xdna,
                c_xdna,
                a_handle,
                b_handle,
                c_handle,
                cmd_handle,
                instr_handle: instr_h,
                instr_xdna,
                ninstr: insts.len() as u32,
                m,
                k,
                n,
                bo_handles,
                maps,
            })
        }

        pub fn dims(&self) -> (usize, usize, usize) {
            (self.m, self.k, self.n)
        }

        /// Firmware-side state of the live hwctx (call after a hang to see if the
        /// command reached the firmware / completed / errored, and the partition
        /// placement).
        pub fn hwctx_report(&self) -> io::Result<String> {
            self.npu.hwctx_report()
        }

        /// Build the ert command packet in the CMD BO. Layout captured from the
        /// known-good XRT path (MLIR_AIE kernel ABI: opcode, instr, ninstr, A, B,
        /// C). BO-address slots hold the instruction DEV address and the operand
        /// host (SVA) addresses; `args` pins them at submit.
        fn build_packet(&self) {
            let p = self.cmd_host as *mut u32;
            let a = self.a_xdna;
            let b = self.b_xdna;
            let c = self.c_xdna;
            let ia = self.instr_xdna;
            unsafe {
                *p.add(0) = 0x3001_0001; // ert header: state=NEW + MLIR_AIE count/opcode/type
                *p.add(1) = 0x1; // cu_mask: CU 0
                *p.add(2) = 3; // kernel arg0: opcode (DPU/transaction)
                *p.add(3) = 0; // pad to 8B
                *p.add(4) = ia as u32; // instr addr lo
                *p.add(5) = (ia >> 32) as u32; // instr addr hi
                *p.add(6) = self.ninstr; // ninstr (words)
                *p.add(7) = a as u32; // A addr lo
                *p.add(8) = (a >> 32) as u32; // A addr hi
                *p.add(9) = b as u32; // B addr lo
                *p.add(10) = (b >> 32) as u32; // B addr hi
                *p.add(11) = c as u32; // C addr lo
                *p.add(12) = (c >> 32) as u32; // C addr hi
            }
        }

        /// Hot path: `C[m,n] i32 = A[m,k] i8 · B[k,n] i8` on the NPU — copy A/B
        /// into their host-shared BOs, submit the command, wait the fence, read C.
        pub fn run(&self, a: &[i8], b: &[i8]) -> io::Result<Vec<i32>> {
            assert_eq!(a.len(), self.m * self.k, "A must be m*k");
            assert_eq!(b.len(), self.k * self.n, "B must be k*n");
            let c_bytes = self.m * self.n * 4;
            unsafe {
                std::ptr::copy_nonoverlapping(a.as_ptr() as *const u8, self.a_host, a.len());
                std::ptr::copy_nonoverlapping(b.as_ptr() as *const u8, self.b_host, b.len());
                // Fault C's pages in (it's write-only for the NPU; ensure the
                // output-DMA target is resident so the shim write doesn't stall).
                std::ptr::write_bytes(self.c_host, 0, c_bytes);
            }
            // Flush operands to memory so the NPU's (non-coherent) reads see them.
            clflush_range(self.a_host, a.len());
            clflush_range(self.b_host, b.len());
            clflush_range(self.c_host, c_bytes);
            self.build_packet();
            clflush_range(self.cmd_host, 13 * 4);

            let args = [self.instr_handle, self.a_handle, self.b_handle, self.c_handle];
            let seq = self.npu.exec(self.hwctx, self.cmd_handle, &args)?;
            let signaled = self.npu.syncobj_timeline_wait(self.syncobj, seq, 10_000_000_000)?;
            // The firmware writes the ERT command state into the CMD BO header
            // (the shim's poll_command reads exactly this). Reading it tells us
            // where the firmware got to: 1=NEW(never processed) 2=QUEUED 3=RUNNING
            // 4=COMPLETED 5=ERROR 6=ABORT 8=TIMEOUT — no root/dyndbg needed.
            let ert_state = unsafe { *(self.cmd_host as *const u32) } & 0xff;
            if !signaled {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!("NPU exec fence timed out (10s); CMD BO ert state={ert_state} \
                             (1=NEW 3=RUNNING 4=COMPLETED 5=ERROR 6=ABORT 8=TIMEOUT)"),
                ));
            }

            // Invalidate C's cache lines so we read the NPU's fresh output.
            clflush_range(self.c_host, self.m * self.n * 4);
            let mut out = vec![0i32; self.m * self.n];
            unsafe {
                std::ptr::copy_nonoverlapping(self.c_host as *const i32, out.as_mut_ptr(), out.len());
            }
            Ok(out)
        }
    }

    impl Drop for Gemm {
        fn drop(&mut self) {
            self.npu.destroy_hwctx(self.hwctx);
            for &(p, sz) in &self.maps {
                unsafe { libc::munmap(p as *mut c_void, sz) };
            }
            for &h in &self.bo_handles {
                self.npu.close_bo(h);
            }
        }
    }

    /// Level-1a hardware probe: exercise every primitive the direct path is built
    /// on — device open, the query ioctls, a BO alloc/mmap/roundtrip, and the
    /// syncobj timeline-wait — and report exactly what the kernel accepted. This
    /// validates the ioctl encoding and struct layouts against the live driver
    /// before the command-submit path is wired on top. Returns a human report.
    pub fn probe() -> io::Result<String> {
        use std::fmt::Write;
        let mut r = String::new();

        // Device open is the one fatal step — without it there's nothing to test.
        let npu = Npu::open("")?;
        writeln!(r, "[ok]   device: /dev/accel/accel0 opened (fd bound)").ok();

        // Every subsequent primitive runs independently and reports its own
        // verdict, so a single run maps exactly what the live driver accepts.
        let mut cols = 0u16;
        match npu.aie_version() {
            Ok((maj, min)) => writeln!(r, "[ok]   GET_INFO/aie_version: {maj}.{min}").ok(),
            Err(e) => writeln!(r, "[FAIL] GET_INFO/aie_version: {e}").ok(),
        };
        match npu.aie_metadata() {
            Ok((c, rows, col_size)) => {
                cols = c;
                writeln!(r, "[ok]   GET_INFO/aie_metadata: {c} cols x {rows} rows, col_size={col_size} B").ok()
            }
            Err(e) => writeln!(r, "[FAIL] GET_INFO/aie_metadata: {e}").ok(),
        };

        // BO round-trip: CREATE_BO + GET_BO_INFO + mmap + write/read + GEM_CLOSE.
        match npu.alloc_shmem(4096) {
            Ok(mut bo) => {
                let xa = bo.xdna_addr;
                let buf = bo.as_mut_slice();
                for (i, b) in buf.iter_mut().enumerate() {
                    *b = (i * 7 + 1) as u8;
                }
                let ok = bo.as_slice().iter().enumerate().all(|(i, &b)| b == (i * 7 + 1) as u8);
                writeln!(
                    r,
                    "[{}] CREATE_BO+mmap shmem 4096B (xdna_addr=0x{xa:x}): mmap roundtrip {}",
                    if ok { "ok" } else { "FAIL" },
                    if ok { "PASS" } else { "MISMATCH" }
                )
                .ok();
            }
            Err(e) => {
                writeln!(r, "[FAIL] CREATE_BO/mmap shmem: {e}").ok();
            }
        };

        // Syncobj wait path: create a PRE-SIGNALED fence → timeline-wait point 0
        // → expect signaled. Proves the TIMELINE_WAIT ioctl the EXEC_CMD path
        // completes on. (Waiting a future, never-submitted point returns EINVAL
        // by design — the real path waits on the exact `seq` EXEC_CMD returns.)
        match npu.syncobj_create(true) {
            Ok(so) => {
                match npu.syncobj_timeline_wait(so, 0, 0) {
                    Ok(sig) => writeln!(
                        r,
                        "[{}] SYNCOBJ_CREATE(signaled)+TIMELINE_WAIT pt0: {}",
                        if sig { "ok" } else { "FAIL" },
                        if sig { "signaled" } else { "timed out" }
                    )
                    .ok(),
                    Err(e) => writeln!(r, "[FAIL] SYNCOBJ_TIMELINE_WAIT: {e}").ok(),
                };
                npu.syncobj_destroy(so);
            }
            Err(e) => {
                writeln!(r, "[FAIL] SYNCOBJ_CREATE: {e}").ok();
            }
        };

        // Best-effort hwctx create/destroy — reports the driver's verdict so we
        // learn the accepted partition params for the next increment.
        let num_tiles = if cols == 0 { 4 } else { cols as u32 };
        match npu.hwctx_roundtrip(num_tiles) {
            Ok(ctx) => {
                writeln!(
                    r,
                    "[ok]   CREATE_HWCTX(num_tiles={num_tiles}): handle={} syncobj={} umq_bo={} doorbell=0x{:x} — DESTROY",
                    ctx.handle, ctx.syncobj, ctx.umq_bo, ctx.umq_doorbell
                )
                .ok();
                npu.destroy_hwctx(ctx.handle);
            }
            Err(e) => {
                writeln!(r, "[info] CREATE_HWCTX(num_tiles={num_tiles}): {e} — refining params next").ok();
            }
        }

        writeln!(r, "\ndirect-ioctl foundation probe complete.").ok();
        Ok(r)
    }
}

pub use imp::probe;

#[cfg(target_os = "linux")]
pub use imp::{Bo, Gemm, Hwctx, Npu};
