// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PCI transport through a signed driver extension (feature `dext`).
//!
//! ## Why the transport looks like this
//!
//! Claiming a PCIe device on macOS requires a DriverKit extension holding
//! `com.apple.developer.driverkit.transport.pci`, a restricted entitlement Apple
//! grants per development team. rlx does not have one, so it does not ship a
//! dext; it talks to whichever signed extension is installed on the host.
//!
//! The extension this expects is a thin one: it matches
//! `IOPCIClassMatch = 0x03000000` with `IOPCITunnelCompatible`, and does nothing
//! but hand the raw device to userspace — config-space access, BAR mappings, a
//! function-level reset, and DMA buffers whose physical scatter-gather addresses
//! are written back to the caller.
//!
//! Its `io_connect_t` is held by a helper process that republishes the
//! connection over a UNIX socket. This module is the client for that socket, so
//! rlx reaches the hardware without linking IOKit userclient code and without an
//! entitlement of its own. The three identifiers naming that external component
//! — [`helper_path`], [`service_name`], [`socket_path`] — are its published
//! names rather than rlx's, and each is overridable, so a different signed
//! extension speaking the same protocol drops in without a code change.
//!
//! ## What this layer is and is not
//!
//! This is the equivalent of `/sys/bus/pci/devices/*` plus `/dev/vfio` on Linux:
//! enough to read config space, map BARs, and get DMA-able memory with known
//! physical addresses. It is **not** a GPU driver. Turning these primitives into
//! a device that runs kernels means loading signed firmware through the PSP (or
//! booting GSP), bringing up the memory controller, SMU, and GFX/SDMA rings, and
//! building GPUVM page tables — that is the `am` feature, and it is not written.
//!
//! ## Cost model
//!
//! DMA buffers are shared by file descriptor and `mmap`ped directly, so host
//! memory access is zero-copy at full speed. BAR access is **not**: every MMIO
//! read/write is a socket round-trip into the helper process, on the order of
//! tens of microseconds. Any design layered on this must keep command buffers in
//! DMA memory and touch BARs only for doorbells.

use crate::EgpuError;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};

/// Where the interoperating extension installs its helper. An external
/// component's own path, not a name rlx chose — see [`DextConfig`].
const DEFAULT_HELPER: &str = "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU";

/// Which driver extension to talk to, and how to reach it.
///
/// Every field names an external component rather than anything rlx owns, so
/// none of them is hardcoded at a call site. Three sources are layered, most
/// specific first:
///
/// 1. a value set through [`DextConfig::builder`],
/// 2. the matching environment variable (`RLX_EGPU_HELPER`, `RLX_EGPU_SERVICE`,
///    `RLX_EGPU_SOCK`),
/// 3. the built-in default.
///
/// ```no_run
/// use rlx_egpu::dext::{DextConfig, PciTransport};
///
/// // Point at a different signed extension without touching env or code paths.
/// let config = DextConfig::builder()
///     .helper("/opt/acme/bin/pcie-helper")
///     .service("acmepci")
///     .build();
/// let gpu = PciTransport::connect_with(&config)?;
/// # Ok::<_, rlx_egpu::EgpuError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DextConfig {
    helper: PathBuf,
    service: String,
    /// `None` means "derive from the service name", so setting only the service
    /// keeps the socket consistent with it.
    socket: Option<PathBuf>,
}

impl DextConfig {
    /// Built-in defaults with the config file and environment layered on top —
    /// what [`PciTransport::connect`] uses.
    pub fn from_env() -> Self {
        Self::builder().build()
    }

    /// Config file consulted between the environment and the defaults:
    /// `RLX_EGPU_CONFIG`, else `$XDG_CONFIG_HOME/rlx/egpu.conf`, else
    /// `~/.config/rlx/egpu.conf`.
    ///
    /// The format is `key = value` per line, `#` to end of line for comments:
    ///
    /// ```text
    /// # which signed extension to talk to
    /// helper  = /Applications/YourApp.app/Contents/MacOS/YourApp
    /// service = yourdriver
    /// ```
    ///
    /// Deliberately not a structured format: the file holds three strings, and
    /// a parser dependency in a transport crate would buy nothing.
    pub fn config_path() -> PathBuf {
        if let Some(path) = rlx_ir::env::var_os("RLX_EGPU_CONFIG") {
            return PathBuf::from(path);
        }
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .unwrap_or_else(|| PathBuf::from("."));
        base.join("rlx").join("egpu.conf")
    }

    /// Start a configuration. Unset fields fall through to the environment and
    /// then to the defaults.
    pub fn builder() -> DextConfigBuilder {
        DextConfigBuilder::default()
    }

    /// Helper executable that owns the driver-extension connection.
    pub fn helper(&self) -> &Path {
        &self.helper
    }

    /// IOService name the extension publishes once it has claimed a GPU.
    pub fn service(&self) -> &str {
        &self.service
    }

    /// Socket the helper listens on.
    ///
    /// When not set explicitly this is derived from the service name, matching
    /// what the helper's other clients use — so an already-running helper is
    /// reused rather than started twice. It accepts one client at a time, and a
    /// second copy would be refused.
    pub fn socket(&self) -> PathBuf {
        self.socket
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join(format!("{}.sock", self.service)))
    }
}

impl Default for DextConfig {
    /// The built-in defaults, ignoring the environment. Use
    /// [`DextConfig::from_env`] for the layered form.
    fn default() -> Self {
        Self {
            helper: PathBuf::from(DEFAULT_HELPER),
            service: crate::pci::DEFAULT_SERVICE.to_string(),
            socket: None,
        }
    }
}

/// Builder for [`DextConfig`]. Unset fields fall through to the environment,
/// then to the built-in defaults.
#[derive(Debug, Clone, Default)]
pub struct DextConfigBuilder {
    helper: Option<PathBuf>,
    service: Option<String>,
    socket: Option<PathBuf>,
}

impl DextConfigBuilder {
    /// Helper executable to spawn and connect to.
    pub fn helper(mut self, path: impl Into<PathBuf>) -> Self {
        self.helper = Some(path.into());
        self
    }

    /// IOService name the extension publishes.
    pub fn service(mut self, name: impl Into<String>) -> Self {
        self.service = Some(name.into());
        self
    }

    /// Socket path. Leave unset to derive it from the service name.
    pub fn socket(mut self, path: impl Into<PathBuf>) -> Self {
        self.socket = Some(path.into());
        self
    }

    /// Resolve the layers into a configuration: builder value, then the
    /// environment, then [`DextConfig::config_path`], then the default.
    pub fn build(self) -> DextConfig {
        self.build_with_file(&DextConfig::config_path())
    }

    /// [`Self::build`] against an explicit config file. The file layer is
    /// skipped when it is absent or unreadable — a missing config is the normal
    /// case, not an error.
    pub fn build_with_file(self, path: &Path) -> DextConfig {
        let file = read_config_file(path);
        let pick = |explicit: Option<String>, env: &str, key: &str| -> Option<String> {
            explicit
                .or_else(|| std::env::var(env).ok())
                .or_else(|| file.get(key).cloned())
        };
        let defaults = DextConfig::default();
        DextConfig {
            helper: pick(
                self.helper.map(|p| p.to_string_lossy().into_owned()),
                "RLX_EGPU_HELPER",
                "helper",
            )
            .map(PathBuf::from)
            .unwrap_or(defaults.helper),
            service: pick(self.service, "RLX_EGPU_SERVICE", "service").unwrap_or(defaults.service),
            socket: pick(
                self.socket.map(|p| p.to_string_lossy().into_owned()),
                "RLX_EGPU_SOCK",
                "socket",
            )
            .map(PathBuf::from),
        }
    }
}

/// Parse `key = value` lines, `#` starting a comment. Unknown keys are ignored
/// so a newer config does not break an older binary.
fn read_config_file(path: &Path) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return out;
    };
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if let Some((key, value)) = line.split_once('=') {
            let (key, value) = (key.trim(), value.trim());
            if !key.is_empty() && !value.is_empty() {
                out.insert(key.to_string(), value.to_string());
            }
        }
    }
    out
}

/// Helper path from the environment-layered configuration.
pub fn helper_path() -> PathBuf {
    DextConfig::from_env().helper().to_path_buf()
}

// Wire protocol. `request_t` / `response_t` are packed C structs; both sides
// are little-endian arm64.
const REQUEST_LEN: usize = 33; // u8 cmd + u32 dev_id + u32 bar + 3 * u64
const RESPONSE_LEN: usize = 17; // u8 status + 2 * u64

const CMD_MAP_BAR: u8 = 1;
const CMD_MAP_SYSMEM_FD: u8 = 2;
const CMD_CFG_READ: u8 = 3;
const CMD_CFG_WRITE: u8 = 4;
const CMD_RESET: u8 = 5;
const CMD_MMIO_READ: u8 = 6;
const CMD_MMIO_WRITE: u8 = 7;

const STATUS_OK: u8 = 0;

/// Socket path from the environment-layered configuration.
pub fn socket_path() -> PathBuf {
    DextConfig::from_env().socket()
}

/// A claimed PCI device, reached through the driver-extension helper.
pub struct PciTransport {
    stream: UnixStream,
    dev_id: u32,
}

impl PciTransport {
    /// Connect to a running helper, starting one if the socket is not live.
    ///
    /// Fails with a description of the missing piece rather than a generic I/O
    /// error: no helper installed, no extension loaded, and no GPU claimed each
    /// produce a different message.
    pub fn connect() -> Result<Self, EgpuError> {
        Self::connect_with(&DextConfig::from_env())
    }

    /// [`Self::connect`] against an explicit socket path, leaving the rest of
    /// the configuration to the environment and defaults.
    pub fn connect_at(path: &Path) -> Result<Self, EgpuError> {
        Self::connect_with(&DextConfig::builder().socket(path).build())
    }

    /// [`Self::connect`] against an explicit configuration — the form that
    /// names no external component at the call site.
    pub fn connect_with(config: &DextConfig) -> Result<Self, EgpuError> {
        if !cfg!(target_os = "macos") {
            return Err(EgpuError::Unsupported(
                "the driver-extension transport is macOS-only".into(),
            ));
        }
        let path = config.socket();
        if let Ok(stream) = UnixStream::connect(&path) {
            return Ok(Self { stream, dev_id: 0 });
        }
        let helper = config.helper();
        if !helper.exists() {
            return Err(EgpuError::AppMissing(format!(
                "no driver-extension helper at {} (set RLX_EGPU_HELPER, or DextConfig::builder().helper(..))",
                helper.display()
            )));
        }
        if !crate::pci::service_present(config.service()) {
            return Err(EgpuError::DextAbsent(format!(
                "no driver extension publishing `{}` has claimed a GPU (enable it under \
                 System Settings > General > Login Items & Extensions > Driver Extensions, \
                 and check that a display-class device is on the PCIe tunnel)",
                config.service()
            )));
        }
        std::process::Command::new(helper)
            .arg("server")
            .arg(&path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| EgpuError::Io(format!("could not start the helper: {e}")))?;

        // The helper binds the socket asynchronously; retry briefly.
        for _ in 0..100 {
            if let Ok(stream) = UnixStream::connect(&path) {
                return Ok(Self { stream, dev_id: 0 });
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        Err(EgpuError::Io(format!(
            "the helper did not accept a connection on {}",
            path.display()
        )))
    }

    fn encode(&self, cmd: u8, bar: u32, args: [u64; 3]) -> [u8; REQUEST_LEN] {
        let mut buf = [0u8; REQUEST_LEN];
        buf[0] = cmd;
        buf[1..5].copy_from_slice(&self.dev_id.to_le_bytes());
        buf[5..9].copy_from_slice(&bar.to_le_bytes());
        for (i, arg) in args.iter().enumerate() {
            let off = 9 + i * 8;
            buf[off..off + 8].copy_from_slice(&arg.to_le_bytes());
        }
        buf
    }

    fn send(&mut self, bytes: &[u8]) -> Result<(), EgpuError> {
        self.stream
            .write_all(bytes)
            .map_err(|e| EgpuError::Io(e.to_string()))
    }

    fn recv_exact(&mut self, buf: &mut [u8]) -> Result<(), EgpuError> {
        self.stream
            .read_exact(buf)
            .map_err(|e| EgpuError::Io(e.to_string()))
    }

    /// Read a response header, converting a failure status into the error text
    /// the helper sends after it.
    fn read_response(&mut self) -> Result<(u64, u64), EgpuError> {
        let mut header = [0u8; RESPONSE_LEN];
        self.recv_exact(&mut header)?;
        self.decode_response(header)
    }

    fn decode_response(&mut self, header: [u8; RESPONSE_LEN]) -> Result<(u64, u64), EgpuError> {
        let resp0 = u64::from_le_bytes(header[1..9].try_into().unwrap());
        let resp1 = u64::from_le_bytes(header[9..17].try_into().unwrap());
        if header[0] == STATUS_OK {
            return Ok((resp0, resp1));
        }
        let mut message = vec![0u8; resp0 as usize];
        if !message.is_empty() {
            self.recv_exact(&mut message)?;
        }
        Err(EgpuError::Protocol(
            String::from_utf8_lossy(&message).into_owned(),
        ))
    }

    fn rpc(&mut self, cmd: u8, bar: u32, args: [u64; 3]) -> Result<(u64, u64), EgpuError> {
        let request = self.encode(cmd, bar, args);
        self.send(&request)?;
        self.read_response()
    }

    /// Read PCI config space. `size` must be 1, 2, or 4.
    pub fn read_config(&mut self, offset: u32, size: u32) -> Result<u32, EgpuError> {
        if !matches!(size, 1 | 2 | 4) {
            return Err(EgpuError::Protocol(format!(
                "config access width must be 1, 2, or 4 bytes, got {size}"
            )));
        }
        let (value, _) = self.rpc(CMD_CFG_READ, 0, [offset as u64, size as u64, 0])?;
        Ok(value as u32)
    }

    /// Write PCI config space. `size` must be 1, 2, or 4.
    pub fn write_config(&mut self, offset: u32, size: u32, value: u32) -> Result<(), EgpuError> {
        if !matches!(size, 1 | 2 | 4) {
            return Err(EgpuError::Protocol(format!(
                "config access width must be 1, 2, or 4 bytes, got {size}"
            )));
        }
        self.rpc(CMD_CFG_WRITE, 0, [offset as u64, size as u64, value as u64])?;
        Ok(())
    }

    /// Vendor and device ID read straight out of config space — the check that
    /// the transport reaches real hardware.
    pub fn identity(&mut self) -> Result<(u16, u16), EgpuError> {
        let vendor = self.read_config(0x00, 2)? as u16;
        let device = self.read_config(0x02, 2)? as u16;
        Ok((vendor, device))
    }

    /// Map a BAR in the helper process and report its size. The address it
    /// returns lives in the helper's address space; use [`Self::mmio_read`] /
    /// [`Self::mmio_write`] to reach it.
    pub fn map_bar(&mut self, bar: u32) -> Result<u64, EgpuError> {
        let (_addr, size) = self.rpc(CMD_MAP_BAR, bar, [0; 3])?;
        Ok(size)
    }

    /// Read `len` bytes from a mapped BAR. One socket round-trip per call.
    pub fn mmio_read(&mut self, bar: u32, offset: u64, len: usize) -> Result<Vec<u8>, EgpuError> {
        let request = self.encode(CMD_MMIO_READ, bar, [offset, len as u64, 0]);
        self.send(&request)?;
        self.read_response()?;
        let mut data = vec![0u8; len];
        self.recv_exact(&mut data)?;
        Ok(data)
    }

    /// Write bytes to a mapped BAR. The helper acknowledges nothing, so this
    /// returns as soon as the bytes are queued.
    pub fn mmio_write(&mut self, bar: u32, offset: u64, data: &[u8]) -> Result<(), EgpuError> {
        let request = self.encode(CMD_MMIO_WRITE, bar, [offset, data.len() as u64, 0]);
        self.send(&request)?;
        self.send(data)
    }

    /// Reset the device (function-level reset, falling back to a hot reset).
    pub fn reset(&mut self) -> Result<(), EgpuError> {
        self.rpc(CMD_RESET, 0, [0; 3])?;
        Ok(())
    }

    /// Allocate host memory the device can DMA to, and learn its physical
    /// scatter-gather layout.
    ///
    /// The helper allocates through `IODMACommand`, so the returned physical
    /// addresses are what the GPU's page tables and ring descriptors need. The
    /// buffer is shared as a file descriptor and mapped here directly, so host
    /// access afterwards costs nothing.
    pub fn alloc_dma(&mut self, size: usize, contiguous: bool) -> Result<DmaBuffer, EgpuError> {
        let request = self.encode(
            CMD_MAP_SYSMEM_FD,
            0,
            [size as u64, u64::from(contiguous), 0],
        );
        self.send(&request)?;

        let (header, fd) = recv_response_with_fd(self.stream.as_raw_fd())?;
        let (mapped_len, _index) = self.decode_response(header)?;
        let fd = fd.ok_or_else(|| {
            EgpuError::Protocol("the helper returned no descriptor for the DMA buffer".into())
        })?;
        // SAFETY: `fd` was just received over SCM_RIGHTS and is owned here.
        let owned = unsafe { std::fs::File::from_raw_fd(fd) };
        DmaBuffer::map(owned, mapped_len as usize, size)
    }
}

/// Host memory a device can DMA to, mapped into this process.
///
/// The head of the mapping carries the physical segment table the helper wrote
/// (`[addr, len, …, 0, 0]`); it is parsed once at construction into
/// [`Self::physical_pages`], after which the whole buffer is usable as memory.
pub struct DmaBuffer {
    ptr: *mut u8,
    mapped_len: usize,
    requested_len: usize,
    physical_pages: Vec<u64>,
}

// The mapping is owned exclusively by this handle.
unsafe impl Send for DmaBuffer {}

impl DmaBuffer {
    fn map(
        file: std::fs::File,
        mapped_len: usize,
        requested_len: usize,
    ) -> Result<Self, EgpuError> {
        let fd = file.into_raw_fd();
        // SAFETY: `fd` is a live shared-memory descriptor of at least
        // `mapped_len` bytes; the mapping is unmapped in `Drop`.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        // SAFETY: mmap keeps its own reference to the mapping.
        unsafe { libc::close(fd) };
        if ptr == libc::MAP_FAILED {
            return Err(EgpuError::Io(
                "could not map the DMA buffer into this process".into(),
            ));
        }
        let ptr = ptr.cast::<u8>();

        // Segment table: (physical address, byte length) pairs, terminated by a
        // zero length. Expand to 4 KiB page addresses, the granularity a GPU
        // page table maps at.
        let mut physical_pages = Vec::new();
        let entries = mapped_len / 16;
        for i in 0..entries {
            // SAFETY: reading inside the mapping; the table starts at offset 0.
            let (addr, len) = unsafe {
                (
                    ptr.add(i * 16).cast::<u64>().read_unaligned(),
                    ptr.add(i * 16 + 8).cast::<u64>().read_unaligned(),
                )
            };
            if len == 0 {
                break;
            }
            physical_pages.extend((0..len).step_by(0x1000).map(|off| addr + off));
        }
        physical_pages.truncate(requested_len.div_ceil(0x1000));

        Ok(Self {
            ptr,
            mapped_len,
            requested_len,
            physical_pages,
        })
    }

    /// 4 KiB physical page addresses backing this buffer, in order.
    pub fn physical_pages(&self) -> &[u64] {
        &self.physical_pages
    }

    /// Bytes requested by the caller (the mapping may be rounded up).
    pub fn len(&self) -> usize {
        self.requested_len
    }

    /// `true` when zero bytes were requested.
    pub fn is_empty(&self) -> bool {
        self.requested_len == 0
    }

    /// The buffer as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the mapping is live for the lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.requested_len) }
    }

    /// The buffer as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the mapping is live and uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.requested_len) }
    }
}

impl Drop for DmaBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`mapped_len` are the values returned by `mmap`.
        unsafe { libc::munmap(self.ptr.cast(), self.mapped_len) };
    }
}

/// Receive a response header plus the descriptor the helper passes over
/// `SCM_RIGHTS` for DMA allocations.
fn recv_response_with_fd(socket: RawFd) -> Result<([u8; RESPONSE_LEN], Option<RawFd>), EgpuError> {
    let mut header = [0u8; RESPONSE_LEN];
    let mut control = [0u8; 64];
    // SAFETY: all pointers refer to live local buffers, and the sizes match.
    let received = unsafe {
        let mut iov = libc::iovec {
            iov_base: header.as_mut_ptr().cast(),
            iov_len: header.len(),
        };
        let mut message: libc::msghdr = std::mem::zeroed();
        message.msg_iov = &mut iov;
        message.msg_iovlen = 1;
        message.msg_control = control.as_mut_ptr().cast();
        message.msg_controllen = control.len() as _;

        let n = libc::recvmsg(socket, &mut message, 0);
        if n < 0 {
            return Err(EgpuError::Io(std::io::Error::last_os_error().to_string()));
        }
        if (n as usize) < header.len() {
            return Err(EgpuError::Protocol(
                "short response header from the helper".into(),
            ));
        }
        let control_header = libc::CMSG_FIRSTHDR(&message);
        if control_header.is_null()
            || (*control_header).cmsg_level != libc::SOL_SOCKET
            || (*control_header).cmsg_type != libc::SCM_RIGHTS
        {
            None
        } else {
            Some(
                libc::CMSG_DATA(control_header)
                    .cast::<RawFd>()
                    .read_unaligned(),
            )
        }
    };
    Ok((header, received))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_values_win_over_environment_and_defaults() {
        // The point of the builder is that a caller can name a different signed
        // extension without touching env or code paths, so an explicit value
        // must survive whatever the environment says.
        let config = DextConfig::builder()
            .helper("/opt/acme/bin/pcie-helper")
            .service("acmepci")
            .build();
        assert_eq!(config.helper(), Path::new("/opt/acme/bin/pcie-helper"));
        assert_eq!(config.service(), "acmepci");
        assert_ne!(config.helper(), Path::new(DEFAULT_HELPER));
    }

    #[test]
    fn the_socket_follows_the_service_unless_set() {
        // Setting only the service must keep the socket consistent with it —
        // otherwise a caller pointing at a second extension would silently talk
        // to the first one's socket.
        let derived = DextConfig::builder().service("acmepci").build();
        assert_eq!(
            derived.socket(),
            std::env::temp_dir().join("acmepci.sock"),
            "socket must be derived from the service name"
        );

        let explicit = DextConfig::builder()
            .service("acmepci")
            .socket("/run/acme.sock")
            .build();
        assert_eq!(explicit.socket(), Path::new("/run/acme.sock"));
    }

    #[test]
    fn defaults_are_reachable_without_consulting_the_environment() {
        // `Default` is the built-in set; `from_env` is the layered one. Keeping
        // them distinct means a test or an embedder can get a predictable
        // config on a machine with the variables set.
        let defaults = DextConfig::default();
        assert_eq!(defaults.helper(), Path::new(DEFAULT_HELPER));
        assert_eq!(defaults.service(), crate::pci::DEFAULT_SERVICE);
        assert_eq!(
            defaults.socket(),
            std::env::temp_dir().join(format!("{}.sock", crate::pci::DEFAULT_SERVICE))
        );
    }

    #[test]
    fn connect_with_reports_the_configured_helper_not_the_default() {
        // The error must name what the caller asked for; naming the built-in
        // default here would send someone looking in the wrong place.
        let config = DextConfig::builder()
            .helper("/nonexistent/acme-helper")
            .socket(std::env::temp_dir().join("rlx-egpu-absent.sock"))
            .build();
        match PciTransport::connect_with(&config).err() {
            Some(EgpuError::AppMissing(m)) => {
                assert!(m.contains("/nonexistent/acme-helper"), "{m}")
            }
            // Off macOS the transport is unsupported before any path check.
            Some(EgpuError::Unsupported(_)) if !cfg!(target_os = "macos") => {}
            other => panic!("expected the configured helper to be named, got {other:?}"),
        }
    }

    #[test]
    fn a_config_file_fills_what_the_builder_leaves_unset() {
        let path = std::env::temp_dir().join("rlx-egpu-test.conf");
        std::fs::write(
            &path,
            "# which extension to talk to\n\
             helper = /opt/acme/bin/pcie-helper   # trailing comment\n\
             service = acmepci\n\
             unknown_key = ignored\n\
             \n",
        )
        .unwrap();

        let config = DextConfig::builder().build_with_file(&path);
        assert_eq!(config.helper(), Path::new("/opt/acme/bin/pcie-helper"));
        assert_eq!(config.service(), "acmepci");
        // The socket still follows the service it was told about.
        assert_eq!(config.socket(), std::env::temp_dir().join("acmepci.sock"));

        // An explicit builder value outranks the file.
        let overridden = DextConfig::builder()
            .service("other")
            .build_with_file(&path);
        assert_eq!(overridden.service(), "other");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_missing_config_file_is_not_an_error() {
        // The normal case is no config at all; it must fall through to the
        // defaults rather than failing the connection.
        let config = DextConfig::builder().build_with_file(Path::new("/nonexistent/rlx-egpu.conf"));
        assert_eq!(config.helper(), Path::new(DEFAULT_HELPER));
        assert_eq!(config.service(), crate::pci::DEFAULT_SERVICE);
    }

    #[test]
    fn request_encoding_matches_the_packed_c_layout() {
        let gpu = PciTransport {
            // A socketpair gives a valid UnixStream without touching the helper.
            stream: UnixStream::pair().unwrap().0,
            dev_id: 0,
        };
        let request = gpu.encode(CMD_CFG_READ, 0, [0x10, 4, 0]);
        assert_eq!(request.len(), 33);
        assert_eq!(request[0], CMD_CFG_READ);
        assert_eq!(u32::from_le_bytes(request[1..5].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(request[5..9].try_into().unwrap()), 0);
        assert_eq!(u64::from_le_bytes(request[9..17].try_into().unwrap()), 0x10);
        assert_eq!(u64::from_le_bytes(request[17..25].try_into().unwrap()), 4);
        assert_eq!(u64::from_le_bytes(request[25..33].try_into().unwrap()), 0);
    }

    #[test]
    fn a_failure_status_carries_the_helper_message() {
        let (client, mut server) = UnixStream::pair().unwrap();
        let text = b"Driver not available.";
        let mut header = [0u8; RESPONSE_LEN];
        header[0] = 1;
        header[1..9].copy_from_slice(&(text.len() as u64).to_le_bytes());
        server.write_all(&header).unwrap();
        server.write_all(text).unwrap();

        let mut gpu = PciTransport {
            stream: client,
            dev_id: 0,
        };
        match gpu.read_response() {
            Err(EgpuError::Protocol(message)) => assert_eq!(message, "Driver not available."),
            other => panic!("expected the helper's message, got {other:?}"),
        }
    }

    #[test]
    fn config_access_width_is_validated_before_any_io() {
        let mut gpu = PciTransport {
            stream: UnixStream::pair().unwrap().0,
            dev_id: 0,
        };
        assert!(matches!(gpu.read_config(0, 3), Err(EgpuError::Protocol(_))));
        assert!(matches!(
            gpu.write_config(0, 8, 0),
            Err(EgpuError::Protocol(_))
        ));
    }
}
