// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Chip / Neural Engine introspection. On Apple platforms these probe
// `sysctl` (via the ObjC shim); everywhere else they report "no ANE".

/// Identity of the host SoC and OS, plus whether a Neural Engine is
/// present. Returned by [`chip_info`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChipInfo {
    /// CPU brand string, e.g. `"Apple M4 Pro"`.
    pub brand: String,
    /// Hardware model id, e.g. `"Mac16,11"`.
    pub model: String,
    /// OS version, e.g. `"26.4.1"`.
    pub os_version: String,
    /// Whether a Neural Engine is available for CoreML to target.
    pub ane: bool,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod sys {
    use std::os::raw::{c_char, c_int};

    unsafe extern "C" {
        pub fn rlx_coreml_ane_available() -> c_int;
        pub fn rlx_coreml_chip_brand(buf: *mut c_char, len: c_int);
        pub fn rlx_coreml_chip_model(buf: *mut c_char, len: c_int);
        pub fn rlx_coreml_os_version(buf: *mut c_char, len: c_int);
    }

    /// Call a shim function that fills a C string buffer and return it as
    /// an owned `String`.
    pub fn fill_string(f: unsafe extern "C" fn(*mut c_char, c_int)) -> String {
        let mut buf = vec![0u8; 256];
        // SAFETY: `buf` is 256 bytes; the shim NUL-terminates within len.
        unsafe { f(buf.as_mut_ptr() as *mut c_char, buf.len() as c_int) };
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        buf.truncate(end);
        String::from_utf8_lossy(&buf).into_owned()
    }
}

/// Whether the CoreML backend can run on this host at all.
///
/// True on Apple platforms with a Neural Engine present. Note that even
/// when this is `false` for the ANE specifically, CoreML can still run on
/// CPU/GPU — see [`ane_available`] for the narrower ANE check.
pub fn is_available() -> bool {
    cfg!(any(target_os = "macos", target_os = "ios"))
}

/// Whether a Neural Engine is present and targetable by CoreML.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn ane_available() -> bool {
    // SAFETY: pure sysctl probe in the shim, no arguments.
    unsafe { sys::rlx_coreml_ane_available() != 0 }
}

/// Whether a Neural Engine is present and targetable by CoreML.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn ane_available() -> bool {
    false
}

/// Probe the host SoC, OS, and ANE presence.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn chip_info() -> ChipInfo {
    ChipInfo {
        brand: sys::fill_string(sys::rlx_coreml_chip_brand),
        model: sys::fill_string(sys::rlx_coreml_chip_model),
        os_version: sys::fill_string(sys::rlx_coreml_os_version),
        ane: ane_available(),
    }
}

/// Probe the host SoC, OS, and ANE presence.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
pub fn chip_info() -> ChipInfo {
    ChipInfo {
        brand: "unknown".into(),
        model: "unknown".into(),
        os_version: "unknown".into(),
        ane: false,
    }
}
