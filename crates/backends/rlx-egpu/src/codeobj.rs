// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU code-object reading — no toolchain, no driver, no device.
//!
//! Kernels are the one part of this stack that is **not signed**. Firmware is
//! verified by the PSP / GSP against keys fused into the die, but a compiled
//! code object is ordinary bytes: DMA it into VRAM and point the command
//! processor at it. Nothing checks where it came from.
//!
//! That is what makes ahead-of-time baking work. The machine that drives the
//! card needs no compiler — compile on a host that has one, ship the bytes, and
//! read them back here with nothing but a byte slice. This module is the
//! read-back half, and it deliberately depends on no toolchain so it runs on the
//! host that has none.
//!
//! Two container shapes turn up:
//!
//! - **clang offload bundle** — what `hipcc --genco` emits. A 24-byte magic, a
//!   count, then `(offset, size, target-triple)` per entry, each entry an ELF.
//!   One artifact can carry several architectures.
//! - **bare ELF** — an AMD code object (`e_machine` 0xe0) or an NVIDIA cubin
//!   (`e_machine` 0xbe), as produced by `clang-offload-bundler --unbundle` or
//!   `ptxas`.
//!
//! Architecture naming is read from the ELF, not from the filename: AMDGPU puts
//! a machine code in the low byte of `e_flags`, CUDA puts the `sm` number in bits
//! 8..16. Both mappings below were read off real objects from `hipcc` and
//! `ptxas` rather than transcribed from a header.

use crate::EgpuError;

const BUNDLE_MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
const ELF_MAGIC: &[u8] = b"\x7fELF";

/// `e_machine` for an AMD GPU code object.
pub const EM_AMDGPU: u16 = 0xe0;
/// `e_machine` for an NVIDIA cubin.
pub const EM_CUDA: u16 = 0xbe;

/// Which vendor's command processor will consume this object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor {
    Amd,
    Nvidia,
    /// An `e_machine` this crate has no naming rules for — a host object in a
    /// bundle, most often.
    Other(u16),
}

impl std::fmt::Display for Vendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Vendor::Amd => write!(f, "AMD"),
            Vendor::Nvidia => write!(f, "NVIDIA"),
            Vendor::Other(m) => write!(f, "machine {m:#06x}"),
        }
    }
}

/// One compiled object for one architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeObject {
    pub vendor: Vendor,
    /// `gfx1100`, `sm_86`, … read from `e_flags`.
    pub target: String,
    /// Bundle entry's target triple, when it came from a bundle.
    pub triple: Option<String>,
    /// Kernel entry points (`STT_FUNC` symbols).
    pub kernels: Vec<String>,
    /// Byte range of this object inside the artifact it was parsed from.
    pub offset: usize,
    pub len: usize,
}

/// AMDGPU `EF_AMDGPU_MACH` — the low byte of `e_flags`. Values verified against
/// objects emitted by `hipcc --genco` for each listed architecture.
fn amdgpu_target(e_flags: u32) -> String {
    match e_flags & 0xff {
        0x36 => "gfx1030".into(),
        0x41 => "gfx1100".into(),
        0x44 => "gfx1103".into(),
        0x46 => "gfx1101".into(),
        0x47 => "gfx1102".into(),
        0x48 => "gfx1200".into(),
        0x4c => "gfx942".into(),
        0x4e => "gfx1201".into(),
        mach => format!("amdgcn-mach-{mach:#04x}"),
    }
}

/// CUDA puts the compute capability in bits 8..16 of `e_flags`: `sm_86` reads
/// back as `0x56`. Verified against `ptxas -arch=sm_86` / `-arch=sm_89`.
fn cuda_target(e_flags: u32) -> String {
    format!("sm_{}", (e_flags >> 8) & 0xff)
}

fn u16_at(bytes: &[u8], off: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(off..off + 2)?.try_into().ok()?,
    ))
}
fn u32_at(bytes: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(off..off + 4)?.try_into().ok()?,
    ))
}
fn u64_at(bytes: &[u8], off: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(off..off + 8)?.try_into().ok()?,
    ))
}

/// NUL-terminated string at `off` in a string table.
fn cstr_at(bytes: &[u8], off: usize) -> Option<String> {
    let rest = bytes.get(off..)?;
    let end = rest.iter().position(|&b| b == 0)?;
    Some(String::from_utf8_lossy(&rest[..end]).into_owned())
}

/// `true` when `bytes` opens a 64-bit little-endian ELF. Both AMD code objects
/// and CUDA cubins are; a 32-bit or big-endian object is not one of ours.
fn is_elf64_le(bytes: &[u8]) -> bool {
    bytes.starts_with(ELF_MAGIC) && bytes.get(4) == Some(&2) && bytes.get(5) == Some(&1)
}

/// Read the `STT_FUNC` symbol names — the kernel entry points a dispatch would
/// name. AMD also emits a `<kernel>.kd` descriptor object per kernel; that is
/// data, not an entry point, so it is left out.
fn kernel_symbols(elf: &[u8]) -> Vec<String> {
    const SHT_SYMTAB: u32 = 2;
    const STT_FUNC: u8 = 2;

    let (Some(shoff), Some(shentsize), Some(shnum)) =
        (u64_at(elf, 0x28), u16_at(elf, 0x3a), u16_at(elf, 0x3c))
    else {
        return Vec::new();
    };
    let mut names = Vec::new();
    for i in 0..shnum as usize {
        let sh = shoff as usize + i * shentsize as usize;
        if u32_at(elf, sh + 4) != Some(SHT_SYMTAB) {
            continue;
        }
        let (Some(off), Some(size), Some(link), Some(entsize)) = (
            u64_at(elf, sh + 0x18),
            u64_at(elf, sh + 0x20),
            u32_at(elf, sh + 0x28),
            u64_at(elf, sh + 0x38),
        ) else {
            continue;
        };
        if entsize == 0 {
            continue;
        }
        // The symtab's sh_link names its string table.
        let strtab_sh = shoff as usize + link as usize * shentsize as usize;
        let (Some(str_off), Some(str_size)) =
            (u64_at(elf, strtab_sh + 0x18), u64_at(elf, strtab_sh + 0x20))
        else {
            continue;
        };
        let Some(strtab) = elf.get(str_off as usize..(str_off + str_size) as usize) else {
            continue;
        };
        for s in 0..(size / entsize) as usize {
            let sym = off as usize + s * entsize as usize;
            let (Some(name_off), Some(info)) = (u32_at(elf, sym), elf.get(sym + 4).copied()) else {
                continue;
            };
            if info & 0xf != STT_FUNC {
                continue;
            }
            if let Some(name) = cstr_at(strtab, name_off as usize)
                && !name.is_empty()
            {
                names.push(name);
            }
        }
    }
    names.sort();
    names.dedup();
    names
}

/// Parse one bare ELF code object.
fn parse_elf(elf: &[u8], offset: usize, triple: Option<String>) -> Result<CodeObject, EgpuError> {
    if !is_elf64_le(elf) {
        return Err(EgpuError::Protocol(
            "not a 64-bit little-endian ELF code object".into(),
        ));
    }
    let machine =
        u16_at(elf, 0x12).ok_or_else(|| EgpuError::Protocol("truncated ELF header".into()))?;
    let e_flags =
        u32_at(elf, 0x30).ok_or_else(|| EgpuError::Protocol("truncated ELF header".into()))?;
    let (vendor, target) = match machine {
        EM_AMDGPU => (Vendor::Amd, amdgpu_target(e_flags)),
        EM_CUDA => (Vendor::Nvidia, cuda_target(e_flags)),
        other => (Vendor::Other(other), format!("machine-{other:#06x}")),
    };
    Ok(CodeObject {
        vendor,
        target,
        triple,
        kernels: kernel_symbols(elf),
        offset,
        len: elf.len(),
    })
}

/// `true` when `bytes` is a clang offload bundle rather than a bare ELF.
pub fn is_bundle(bytes: &[u8]) -> bool {
    bytes.starts_with(BUNDLE_MAGIC)
}

/// Read every code object in an artifact — a clang offload bundle or a bare
/// ELF. Needs no toolchain and no device.
///
/// Bundle entries whose triple names the host (`host-x86_64-…`) carry an object
/// for the CPU side of a HIP translation unit; they are returned like any other
/// entry, with [`Vendor::Other`], so a caller can see the whole artifact.
pub fn parse(bytes: &[u8]) -> Result<Vec<CodeObject>, EgpuError> {
    if !is_bundle(bytes) {
        return Ok(vec![parse_elf(bytes, 0, None)?]);
    }

    let mut out = Vec::new();
    let count = u64_at(bytes, BUNDLE_MAGIC.len())
        .ok_or_else(|| EgpuError::Protocol("truncated bundle header".into()))?;
    let mut cursor = BUNDLE_MAGIC.len() + 8;
    for i in 0..count {
        let (Some(offset), Some(size), Some(id_len)) = (
            u64_at(bytes, cursor),
            u64_at(bytes, cursor + 8),
            u64_at(bytes, cursor + 16),
        ) else {
            return Err(EgpuError::Protocol(format!(
                "truncated bundle entry {i} of {count}"
            )));
        };
        let id_start = cursor + 24;
        let id = bytes
            .get(id_start..id_start + id_len as usize)
            .ok_or_else(|| EgpuError::Protocol(format!("truncated bundle triple {i}")))?;
        let triple = String::from_utf8_lossy(id).into_owned();
        cursor = id_start + id_len as usize;

        let entry = bytes
            .get(offset as usize..(offset + size) as usize)
            .ok_or_else(|| {
                EgpuError::Protocol(format!("bundle entry {i} points outside the artifact"))
            })?;
        // A bundle can carry an empty host placeholder; skip rather than fail.
        if entry.is_empty() {
            continue;
        }
        out.push(parse_elf(entry, offset as usize, Some(triple))?);
    }
    Ok(out)
}

/// Every device target in an artifact, host entries excluded.
pub fn device_targets(bytes: &[u8]) -> Result<Vec<String>, EgpuError> {
    Ok(parse(bytes)?
        .into_iter()
        .filter(|o| !matches!(o.vendor, Vendor::Other(_)))
        .map(|o| o.target)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal 64-bit LE ELF header with the given machine and flags. Enough to
    /// exercise identification without a toolchain in the test environment.
    fn elf_header(machine: u16, e_flags: u32) -> Vec<u8> {
        let mut v = vec![0u8; 0x40];
        v[..4].copy_from_slice(ELF_MAGIC);
        v[4] = 2; // ELFCLASS64
        v[5] = 1; // ELFDATA2LSB
        v[0x12..0x14].copy_from_slice(&machine.to_le_bytes());
        v[0x30..0x34].copy_from_slice(&e_flags.to_le_bytes());
        v
    }

    #[test]
    fn amd_and_nvidia_targets_decode_from_e_flags() {
        // Values read off real objects: hipcc gfx1100 -> mach 0x41, gfx1201 ->
        // 0x4e, gfx942 -> 0x4c with feature bits above the low byte; ptxas
        // sm_86 -> 0x5604, sm_89 -> 0x5904.
        let amd = parse(&elf_header(EM_AMDGPU, 0x41)).unwrap();
        assert_eq!(amd[0].vendor, Vendor::Amd);
        assert_eq!(amd[0].target, "gfx1100");

        assert_eq!(
            parse(&elf_header(EM_AMDGPU, 0x0000_054c)).unwrap()[0].target,
            "gfx942",
            "feature bits above the mach byte must not change the target"
        );
        assert_eq!(
            parse(&elf_header(EM_AMDGPU, 0x4e)).unwrap()[0].target,
            "gfx1201"
        );

        let nv = parse(&elf_header(EM_CUDA, 0x0600_5604)).unwrap();
        assert_eq!(nv[0].vendor, Vendor::Nvidia);
        assert_eq!(nv[0].target, "sm_86");
        assert_eq!(
            parse(&elf_header(EM_CUDA, 0x0600_5904)).unwrap()[0].target,
            "sm_89"
        );
    }

    #[test]
    fn an_unknown_machine_is_reported_rather_than_guessed() {
        let obj = parse(&elf_header(0x3e, 0)).unwrap(); // EM_X86_64
        assert_eq!(obj[0].vendor, Vendor::Other(0x3e));
        assert!(device_targets(&elf_header(0x3e, 0)).unwrap().is_empty());
        // An unmapped AMD mach must name the code, not silently pick a nearby
        // architecture — dispatching gfx1100 code to another part is worse than
        // reporting that the object is unrecognized.
        assert_eq!(
            parse(&elf_header(EM_AMDGPU, 0xfe)).unwrap()[0].target,
            "amdgcn-mach-0xfe"
        );
    }

    #[test]
    fn a_bundle_yields_one_object_per_target() {
        let a = elf_header(EM_AMDGPU, 0x41);
        let b = elf_header(EM_AMDGPU, 0x48);
        let triples = [
            "hipv4-amdgcn-amd-amdhsa--gfx1100",
            "hipv4-amdgcn-amd-amdhsa--gfx1200",
        ];

        // Header first, then the two payloads, with offsets pointing at them.
        let header_len =
            BUNDLE_MAGIC.len() + 8 + triples.iter().map(|t| 24 + t.len()).sum::<usize>();
        let mut v = Vec::new();
        v.extend_from_slice(BUNDLE_MAGIC);
        v.extend_from_slice(&2u64.to_le_bytes());
        for (i, t) in triples.iter().enumerate() {
            let offset = header_len + i * a.len();
            v.extend_from_slice(&(offset as u64).to_le_bytes());
            v.extend_from_slice(&(a.len() as u64).to_le_bytes());
            v.extend_from_slice(&(t.len() as u64).to_le_bytes());
            v.extend_from_slice(t.as_bytes());
        }
        assert_eq!(v.len(), header_len);
        v.extend_from_slice(&a);
        v.extend_from_slice(&b);

        assert!(is_bundle(&v));
        let objects = parse(&v).unwrap();
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].target, "gfx1100");
        assert_eq!(objects[1].target, "gfx1200");
        assert_eq!(objects[0].triple.as_deref(), Some(triples[0]));
        assert_eq!(device_targets(&v).unwrap(), ["gfx1100", "gfx1200"]);
    }

    #[test]
    fn a_truncated_artifact_is_an_error_not_a_panic() {
        assert!(parse(b"").is_err());
        assert!(parse(b"\x7fELF").is_err());
        assert!(parse(BUNDLE_MAGIC).is_err());
        // A bundle whose entry points past the end must not index out of range.
        let mut v = Vec::new();
        v.extend_from_slice(BUNDLE_MAGIC);
        v.extend_from_slice(&1u64.to_le_bytes());
        v.extend_from_slice(&9999u64.to_le_bytes());
        v.extend_from_slice(&64u64.to_le_bytes());
        v.extend_from_slice(&0u64.to_le_bytes());
        assert!(parse(&v).is_err());
    }
}
