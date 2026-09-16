// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ahead-of-time kernel packs — compile elsewhere, ship bytes.
//!
//! The host driving an external GPU has the worst toolchain story of any host
//! in the workspace: macOS has no ROCm, and the NVIDIA route otherwise runs
//! `nvcc` inside a Linux container. Neither is needed. Kernels are unsigned
//! bytes, so they can be built on any machine that has a compiler and carried
//! to the one that has the card.
//!
//! An [`IsaPack`] is that carrier: a flat container of `(target, kernels,
//! code object)` records that [`read`] parses with no toolchain, no driver, and
//! no device. Building one needs `hipcc` or `ptxas`; opening one needs nothing.
//!
//! The container is deliberately dull — a length-prefixed record list with a
//! checksum. Truncation is the realistic failure for an artifact that gets
//! copied between machines, and a pack that silently loses its last kernel is
//! worse than one that refuses to open.

use crate::EgpuError;
use crate::codeobj;

const MAGIC: &[u8; 7] = b"RLXISA\0";
const VERSION: u8 = 1;

/// One target's compiled code plus the kernels it defines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsaEntry {
    /// `gfx1100`, `sm_86`, …
    pub target: String,
    /// Kernel entry points, recorded at bake time so a reader can list them
    /// without re-parsing the object.
    pub kernels: Vec<String>,
    /// The code object itself — an ELF ready to be placed in device memory.
    pub code: Vec<u8>,
}

/// A set of compiled targets travelling as one artifact.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IsaPack {
    pub entries: Vec<IsaEntry>,
}

/// FNV-1a over the payload. Detects truncation and accidental edits; it is not
/// a signature and is not claimed to be one — nothing in this pack is signed,
/// which is precisely why the pack can be built anywhere.
fn checksum(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

impl IsaPack {
    /// Build a pack from compiled artifacts — bundles or bare code objects.
    /// Host entries inside a bundle are dropped; only device targets are kept.
    pub fn from_artifacts<'a>(
        artifacts: impl IntoIterator<Item = &'a [u8]>,
    ) -> Result<Self, EgpuError> {
        let mut entries = Vec::new();
        for artifact in artifacts {
            for object in codeobj::parse(artifact)? {
                if matches!(object.vendor, codeobj::Vendor::Other(_)) {
                    continue;
                }
                let code = artifact
                    .get(object.offset..object.offset + object.len)
                    .ok_or_else(|| {
                        EgpuError::Protocol(format!(
                            "{} points outside its artifact",
                            object.target
                        ))
                    })?
                    .to_vec();
                entries.push(IsaEntry {
                    target: object.target,
                    kernels: object.kernels,
                    code,
                });
            }
        }
        if entries.is_empty() {
            return Err(EgpuError::Protocol(
                "no device code objects found in the given artifacts".into(),
            ));
        }
        entries.sort_by(|a, b| a.target.cmp(&b.target));
        Ok(Self { entries })
    }

    /// Targets in the pack.
    pub fn targets(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.target.as_str()).collect()
    }

    /// The entry for `target`, if the pack carries it.
    pub fn entry(&self, target: &str) -> Option<&IsaEntry> {
        self.entries.iter().find(|e| e.target == target)
    }

    /// Serialize.
    pub fn write(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for entry in &self.entries {
            push_str(&mut payload, &entry.target);
            payload.extend_from_slice(&(entry.kernels.len() as u32).to_le_bytes());
            for kernel in &entry.kernels {
                push_str(&mut payload, kernel);
            }
            payload.extend_from_slice(&(entry.code.len() as u64).to_le_bytes());
            payload.extend_from_slice(&entry.code);
        }

        let mut out = Vec::with_capacity(payload.len() + 16);
        out.extend_from_slice(MAGIC);
        out.push(VERSION);
        out.extend_from_slice(&checksum(&payload).to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }
}

fn push_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], EgpuError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| EgpuError::Protocol("pack length overflow".into()))?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| EgpuError::Protocol("pack is truncated".into()))?;
        self.pos = end;
        Ok(slice)
    }
    fn u32(&mut self) -> Result<u32, EgpuError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, EgpuError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, EgpuError> {
        let n = self.u32()? as usize;
        String::from_utf8(self.take(n)?.to_vec())
            .map_err(|_| EgpuError::Protocol("pack holds a non-UTF-8 name".into()))
    }
}

/// Parse a pack. Needs no toolchain, no driver, and no device — this is the
/// half that runs on the machine with the card.
pub fn read(bytes: &[u8]) -> Result<IsaPack, EgpuError> {
    if !bytes.starts_with(MAGIC) {
        return Err(EgpuError::Protocol("not an rlx ISA pack".into()));
    }
    let version = *bytes
        .get(MAGIC.len())
        .ok_or_else(|| EgpuError::Protocol("pack is truncated".into()))?;
    if version != VERSION {
        return Err(EgpuError::Protocol(format!(
            "pack version {version}, this build reads {VERSION}"
        )));
    }
    let header = MAGIC.len() + 1;
    let expected = u64::from_le_bytes(
        bytes
            .get(header..header + 8)
            .ok_or_else(|| EgpuError::Protocol("pack is truncated".into()))?
            .try_into()
            .unwrap(),
    );
    let payload = &bytes[header + 8..];
    let found = checksum(payload);
    if found != expected {
        return Err(EgpuError::Protocol(format!(
            "pack checksum {found:#018x}, expected {expected:#018x} — truncated or edited in transit"
        )));
    }

    let mut r = Reader {
        bytes: payload,
        pos: 0,
    };
    let count = r.u32()? as usize;
    let mut entries = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        let target = r.string()?;
        let kernel_count = r.u32()? as usize;
        let mut kernels = Vec::with_capacity(kernel_count.min(1024));
        for _ in 0..kernel_count {
            kernels.push(r.string()?);
        }
        let code_len = r.u64()? as usize;
        entries.push(IsaEntry {
            target,
            kernels,
            code: r.take(code_len)?.to_vec(),
        });
    }
    Ok(IsaPack { entries })
}

// ── Baking (needs a toolchain; runs on the build host, not the card's host) ──

/// Run a compiler and return what it wrote, or a readable error.
fn run(tool: &str, args: &[&str], out: &std::path::Path) -> Result<Vec<u8>, EgpuError> {
    let output = std::process::Command::new(tool)
        .args(args)
        .output()
        .map_err(|e| EgpuError::Io(format!("{tool}: {e} (is it on PATH?)")))?;
    if !output.status.success() {
        return Err(EgpuError::Protocol(format!(
            "{tool} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    std::fs::read(out).map_err(|e| EgpuError::Io(format!("{}: {e}", out.display())))
}

/// Compile HIP source to an offload bundle for `arch` (e.g. `gfx1100`).
/// Needs `hipcc`; the resulting bytes are portable to any host.
pub fn bake_hip(source: &std::path::Path, arch: &str) -> Result<Vec<u8>, EgpuError> {
    let out = std::env::temp_dir().join(format!("rlx-egpu-{arch}.hsaco"));
    let hipcc = rlx_ir::env::var("RLX_HIPCC").unwrap_or_else(|| "hipcc".to_string());
    run(
        &hipcc,
        &[
            "--genco",
            &format!("--offload-arch={arch}"),
            "-o",
            &out.to_string_lossy(),
            &source.to_string_lossy(),
        ],
        &out,
    )
}

/// Compile CUDA C++ to PTX with NVRTC, then assemble it for `arch`.
///
/// NVRTC compiles in-process and reads no system headers, which is what makes
/// this work where `nvcc` does not: the `nvcc` frontend pulls in glibc's math
/// headers and collides with CUDA's own declarations on recent distributions.
/// NVRTC has no host-compiler stage to collide with.
///
/// `libnvrtc` is opened at run time rather than linked, so this compiles on a
/// host with no CUDA installed and fails with a readable message there.
#[cfg(feature = "nvrtc")]
pub fn bake_cuda(source: &std::path::Path, arch: &str) -> Result<Vec<u8>, EgpuError> {
    let text = std::fs::read_to_string(source)
        .map_err(|e| EgpuError::Io(format!("{}: {e}", source.display())))?;
    let ptx = nvrtc::compile_to_ptx(&text, arch)?;
    let path = std::env::temp_dir().join(format!("rlx-egpu-{arch}.ptx"));
    std::fs::write(&path, ptx).map_err(|e| EgpuError::Io(format!("{}: {e}", path.display())))?;
    bake_ptx(&path, arch)
}

/// Minimal NVRTC binding — opened with `dlopen`, so no build-time CUDA.
#[cfg(feature = "nvrtc")]
mod nvrtc {
    use crate::EgpuError;
    use std::ffi::{CString, c_char, c_int, c_void};

    type Program = *mut c_void;

    /// `libnvrtc` under the names it ships as, most specific first.
    const CANDIDATES: &[&str] = &[
        "libnvrtc.so",
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "libnvrtc.dylib",
        "nvrtc64_120_0.dll",
    ];

    struct Lib(*mut c_void);

    impl Lib {
        fn open() -> Result<Self, EgpuError> {
            // Honour an explicit path first: a host can have several CUDA
            // installs and the loader's default is not always the wanted one.
            let mut names: Vec<String> = Vec::new();
            if let Some(explicit) = rlx_ir::env::var("RLX_NVRTC") {
                names.push(explicit);
            }
            names.extend(CANDIDATES.iter().map(|s| s.to_string()));
            for name in &names {
                let Ok(c) = CString::new(name.as_str()) else {
                    continue;
                };
                // SAFETY: `c` is a valid NUL-terminated path.
                let handle = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW) };
                if !handle.is_null() {
                    return Ok(Self(handle));
                }
            }
            Err(EgpuError::Io(format!(
                "could not load libnvrtc (tried {}); set RLX_NVRTC to its path",
                names.join(", ")
            )))
        }

        fn sym(&self, name: &str) -> Result<*mut c_void, EgpuError> {
            let c = CString::new(name).map_err(|e| EgpuError::Io(e.to_string()))?;
            // SAFETY: the handle is live and `c` is NUL-terminated.
            let ptr = unsafe { libc::dlsym(self.0, c.as_ptr()) };
            if ptr.is_null() {
                return Err(EgpuError::Io(format!("libnvrtc has no symbol {name}")));
            }
            Ok(ptr)
        }
    }

    impl Drop for Lib {
        fn drop(&mut self) {
            // SAFETY: opened by `dlopen` and not used after this point.
            unsafe { libc::dlclose(self.0) };
        }
    }

    /// Compile CUDA C++ to PTX for `arch` (e.g. `sm_86`).
    pub fn compile_to_ptx(source: &str, arch: &str) -> Result<String, EgpuError> {
        let lib = Lib::open()?;

        // SAFETY: every signature below matches nvrtc.h for the symbol named.
        unsafe {
            let create: extern "C" fn(
                *mut Program,
                *const c_char,
                *const c_char,
                c_int,
                *const *const c_char,
                *const *const c_char,
            ) -> c_int = std::mem::transmute(lib.sym("nvrtcCreateProgram")?);
            let compile: extern "C" fn(Program, c_int, *const *const c_char) -> c_int =
                std::mem::transmute(lib.sym("nvrtcCompileProgram")?);
            let ptx_size: extern "C" fn(Program, *mut usize) -> c_int =
                std::mem::transmute(lib.sym("nvrtcGetPTXSize")?);
            let get_ptx: extern "C" fn(Program, *mut c_char) -> c_int =
                std::mem::transmute(lib.sym("nvrtcGetPTX")?);
            let log_size: extern "C" fn(Program, *mut usize) -> c_int =
                std::mem::transmute(lib.sym("nvrtcGetProgramLogSize")?);
            let get_log: extern "C" fn(Program, *mut c_char) -> c_int =
                std::mem::transmute(lib.sym("nvrtcGetProgramLog")?);
            let destroy: extern "C" fn(*mut Program) -> c_int =
                std::mem::transmute(lib.sym("nvrtcDestroyProgram")?);

            let src = CString::new(source).map_err(|e| EgpuError::Io(e.to_string()))?;
            let name = CString::new("rlx_kernel.cu").unwrap();
            let mut program: Program = std::ptr::null_mut();
            if create(
                &mut program,
                src.as_ptr(),
                name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            ) != 0
            {
                return Err(EgpuError::Protocol("nvrtcCreateProgram failed".into()));
            }

            let gpu_arch = CString::new(format!("--gpu-architecture=compute_{}", &arch[3..]))
                .map_err(|e| EgpuError::Io(e.to_string()))?;
            let options = [gpu_arch.as_ptr()];
            let status = compile(program, options.len() as c_int, options.as_ptr());

            // Read the log before destroying the program: on failure it is the
            // only thing that says why, and it is lost with the handle.
            let mut log = String::new();
            let mut size = 0usize;
            if log_size(program, &mut size) == 0 && size > 1 {
                let mut buf = vec![0u8; size];
                if get_log(program, buf.as_mut_ptr().cast()) == 0 {
                    buf.pop();
                    log = String::from_utf8_lossy(&buf).trim().to_string();
                }
            }

            if status != 0 {
                destroy(&mut program);
                return Err(EgpuError::Protocol(format!("NVRTC: {log}")));
            }

            let mut size = 0usize;
            if ptx_size(program, &mut size) != 0 || size == 0 {
                destroy(&mut program);
                return Err(EgpuError::Protocol("NVRTC produced no PTX".into()));
            }
            let mut buf = vec![0u8; size];
            let got = get_ptx(program, buf.as_mut_ptr().cast());
            destroy(&mut program);
            if got != 0 {
                return Err(EgpuError::Protocol("nvrtcGetPTX failed".into()));
            }
            buf.pop(); // trailing NUL
            String::from_utf8(buf).map_err(|e| EgpuError::Protocol(e.to_string()))
        }
    }
}

/// Assemble PTX into a cubin for `arch` (e.g. `sm_86`).
///
/// `ptxas` takes PTX and emits a code object without touching a host compiler
/// or the system headers, which is the whole point: no `nvcc` frontend, no
/// container. PTX comes from NVRTC, which `rlx-cuda` already drives.
pub fn bake_ptx(ptx: &std::path::Path, arch: &str) -> Result<Vec<u8>, EgpuError> {
    let out = std::env::temp_dir().join(format!("rlx-egpu-{arch}.cubin"));
    let ptxas = rlx_ir::env::var("RLX_PTXAS").unwrap_or_else(|| "ptxas".to_string());
    run(
        &ptxas,
        &[
            &format!("-arch={arch}"),
            &ptx.to_string_lossy(),
            "-o",
            &out.to_string_lossy(),
        ],
        &out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack() -> IsaPack {
        IsaPack {
            entries: vec![
                IsaEntry {
                    target: "gfx1100".into(),
                    kernels: vec!["rlx_add".into(), "rlx_mul".into()],
                    code: vec![1, 2, 3, 4, 5],
                },
                IsaEntry {
                    target: "sm_86".into(),
                    kernels: vec!["rlx_add".into()],
                    code: vec![9; 300],
                },
            ],
        }
    }

    #[test]
    fn a_pack_round_trips() {
        let original = pack();
        let decoded = read(&original.write()).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(decoded.targets(), ["gfx1100", "sm_86"]);
        assert_eq!(decoded.entry("sm_86").unwrap().code.len(), 300);
        assert!(decoded.entry("gfx942").is_none());
    }

    #[test]
    fn truncation_is_caught_rather_than_silently_dropping_a_kernel() {
        // The realistic failure for an artifact copied between machines. A
        // short read must not present as a pack with fewer targets.
        let bytes = pack().write();
        for cut in [bytes.len() - 1, bytes.len() / 2, 16] {
            assert!(
                read(&bytes[..cut]).is_err(),
                "a pack truncated to {cut} bytes was accepted"
            );
        }
    }

    #[test]
    fn a_flipped_byte_is_caught() {
        let mut bytes = pack().write();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        match read(&bytes) {
            Err(EgpuError::Protocol(m)) => assert!(m.contains("checksum"), "{m}"),
            other => panic!("expected a checksum error, got {other:?}"),
        }
    }

    #[test]
    fn foreign_and_future_artifacts_are_rejected() {
        assert!(read(b"").is_err());
        assert!(read(b"not a pack at all").is_err());
        let mut bytes = pack().write();
        bytes[MAGIC.len()] = VERSION + 1;
        match read(&bytes) {
            Err(EgpuError::Protocol(m)) => assert!(m.contains("version"), "{m}"),
            other => panic!("expected a version error, got {other:?}"),
        }
    }

    #[test]
    fn building_from_artifacts_keeps_device_targets_only() {
        // A bundle carrying a host object alongside device code: the host entry
        // must not end up in a pack meant for a GPU.
        let mut elf = vec![0u8; 0x40];
        elf[..4].copy_from_slice(b"\x7fELF");
        elf[4] = 2;
        elf[5] = 1;
        elf[0x12..0x14].copy_from_slice(&codeobj::EM_AMDGPU.to_le_bytes());
        elf[0x30..0x34].copy_from_slice(&0x41u32.to_le_bytes());

        let built = IsaPack::from_artifacts([elf.as_slice()]).unwrap();
        assert_eq!(built.targets(), ["gfx1100"]);
        assert_eq!(built.entries[0].code, elf);

        let mut host = elf.clone();
        host[0x12..0x14].copy_from_slice(&0x3eu16.to_le_bytes()); // EM_X86_64
        assert!(
            IsaPack::from_artifacts([host.as_slice()]).is_err(),
            "a host-only artifact yields no device targets and must not pack"
        );
    }
}
