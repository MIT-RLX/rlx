// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runtime detection of the Apple matrix hardware.
//!
//! The `rlx_cpu_amx_*` build cfgs only say "this path was *compiled in*". They
//! cannot say "the chip this binary is *running on* actually has SME2" — an
//! `amx-sme` build made on an M4 can be copied to an M1 that has no SME. So the
//! actual dispatch is gated at runtime here, by asking the kernel through
//! `sysctlbyname("hw.optional.arm.FEAT_*")`. Results are cached in `OnceLock`s
//! (the answer can't change during a process's life).
//!
//! Note the asymmetry between AMX and SME: SME is the *documented* ARM
//! extension and reports honest `hw.optional.arm.FEAT_SME*` flags. The legacy
//! AMX coprocessor (M1–M3) is undocumented and has **no** feature flag — its
//! presence is simply "Apple Silicon". We never emit AMX instructions
//! ourselves; [`has_amx`] exists only so callers can reason about "is there a
//! matmul-grade unit reachable via Accelerate/BNNS besides NEON".

use std::sync::OnceLock;

unsafe extern "C" {
    // Signature kept byte-identical to the declaration in `crate::config` so
    // the two `extern` blocks don't trip `clashing_extern_declarations`.
    fn sysctlbyname(
        name: *const i8,
        oldp: *mut u8,
        oldlenp: *mut usize,
        newp: *const u8,
        newlen: usize,
    ) -> i32;
}

/// Read an integer `sysctl` by name. Apple's `hw.optional.*` nodes are 4-byte
/// ints; we read into an 8-byte slot (little-endian aarch64 ⇒ the low bytes
/// hold the value regardless of the node's declared width). `None` if the node
/// is absent (older OS / different arch).
fn sysctl_int(name: &str) -> Option<i64> {
    let mut cname = Vec::with_capacity(name.len() + 1);
    cname.extend_from_slice(name.as_bytes());
    cname.push(0);
    let mut val: i64 = 0;
    let mut len = std::mem::size_of::<i64>();
    let rc = unsafe {
        sysctlbyname(
            cname.as_ptr() as *const i8,
            (&mut val as *mut i64).cast(),
            &mut len as *mut usize,
            std::ptr::null(),
            0,
        )
    };
    (rc == 0).then_some(val)
}

/// True when the named `hw.optional.arm.*` feature is present and set.
fn feat(name: &str) -> bool {
    sysctl_int(name).unwrap_or(0) != 0
}

macro_rules! cached_feat {
    ($(#[$m:meta])* $fn_name:ident => $sysctl:literal) => {
        $(#[$m])*
        pub fn $fn_name() -> bool {
            static CACHE: OnceLock<bool> = OnceLock::new();
            *CACHE.get_or_init(|| feat($sysctl))
        }
    };
}

/// True when a matmul-grade coprocessor is reachable (Accelerate/BNNS route).
/// Undocumented AMX has no sysctl flag, so this is simply "Apple Silicon".
pub fn has_amx() -> bool {
    cfg!(all(target_arch = "aarch64", target_vendor = "apple"))
}

cached_feat! {
    /// FEAT_SME — ARM Scalable Matrix Extension (M4+ on Apple).
    has_sme => "hw.optional.arm.FEAT_SME"
}
cached_feat! {
    /// FEAT_SME2 — the SME revision we target for the `amx-sme` GEMM kernel.
    has_sme2 => "hw.optional.arm.FEAT_SME2"
}
cached_feat! {
    /// SME f32→f32 outer-product accumulate (`FMOPA`, what our f32 GEMM uses).
    sme_f32f32 => "hw.optional.arm.SME_F32F32"
}
cached_feat! {
    /// SME int8→int32 outer-product accumulate (`SMOPA`, int8 SME GEMM).
    sme_i8i32 => "hw.optional.arm.SME_I8I32"
}
cached_feat! {
    /// SME bf16→f32 outer-product accumulate (`BFMOPA`, bf16 SME GEMM).
    sme_b16f32 => "hw.optional.arm.SME_B16F32"
}

/// Streaming vector length in **bytes** (SVL). `f32` lane count per SME tile
/// dimension is `svl_bytes()/4` (16 on M4 Pro, SVL = 512 bits). Returns 0 when
/// SME is absent, so callers must check [`has_sme2`] first.
pub fn svl_bytes() -> usize {
    static CACHE: OnceLock<usize> = OnceLock::new();
    *CACHE.get_or_init(|| {
        sysctl_int("hw.optional.arm.sme_max_svl_b")
            .unwrap_or(0)
            .max(0) as usize
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Not an assertion of hardware — just prints what this box reports, so a
    /// developer running `cargo test` sees the probe working. Runs everywhere
    /// Apple; the values simply differ per chip (all-false on pre-SME Macs).
    #[test]
    fn probe_reports_something() {
        eprintln!(
            "AMX/SME probe: amx={} sme={} sme2={} f32f32={} i8i32={} svl_bytes={}",
            has_amx(),
            has_sme(),
            has_sme2(),
            sme_f32f32(),
            sme_i8i32(),
            svl_bytes(),
        );
        // has_amx() is a compile-time truth on aarch64 Apple; the sysctl-backed
        // ones must at least not panic and must agree with each other.
        if has_sme2() {
            assert!(has_sme(), "SME2 implies SME");
            assert!(svl_bytes() > 0, "SME2 present but SVL unreadable");
        }
    }
}
