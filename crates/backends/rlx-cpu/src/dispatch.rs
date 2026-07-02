// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Dispatch table — calibration-aware kernel selection (plan #2).
//!
//! Two layers:
//!   - **Defaults** from `kernel_config` (compile-time, per-arch).
//!   - **Overrides** filled at runtime by autotune / calibration
//!     when a measured number disagrees with the default.
//!
//! Borrowed from MAX's `dispatch_table_a100_gpu.mojo` /
//! `dispatch_table_amd.mojo` pattern: kernel-variant selection is a
//! data lookup, not scattered match arms in dispatch sites.
//!
//! Today the table is consulted by the cost model and (when an
//! override is set) used to override a `kernel_config` default.
//! Future work wires fusion patterns through the same table so
//! schedule decisions are uniformly data-driven.

use crate::kernel_config::{CpuArch, KernelConfig, OpClass, kernel_config_for};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// One-line override for a default `KernelConfig` field. Add cases
/// as autotune learns more.
#[derive(Debug, Clone, Copy)]
pub enum Override {
    NeonSeqThreshold(usize),
    ParGrain(usize),
    ParThreshold(usize),
    FuseAttnThreshold(usize),
}

#[derive(Debug, Default)]
struct Table {
    overrides: Mutex<HashMap<(CpuArch, OpClass), Vec<Override>>>,
}

fn table() -> &'static Table {
    static T: OnceLock<Table> = OnceLock::new();
    T.get_or_init(Table::default)
}

/// Set an override for `(arch, op)`. Idempotent — re-set replaces.
pub fn set_override(arch: CpuArch, op: OpClass, ov: Override) {
    let t = table();
    let mut m = t.overrides.lock().expect("dispatch table poisoned");
    let entry = m.entry((arch, op)).or_default();
    // Replace any existing override of the same field tag.
    entry.retain(|existing| std::mem::discriminant(existing) != std::mem::discriminant(&ov));
    entry.push(ov);
}

/// Reset all overrides (test hook).
#[doc(hidden)]
pub fn clear_overrides_for_tests() {
    let t = table();
    let mut m = t.overrides.lock().expect("dispatch table poisoned");
    m.clear();
}

/// Resolve a `KernelConfig` for `(arch, op)`, applying any
/// overrides on top of the const-time defaults.
pub fn resolve(arch: CpuArch, op: OpClass) -> KernelConfig {
    let mut cfg = kernel_config_for(arch, op);
    let t = table();
    let m = t.overrides.lock().expect("dispatch table poisoned");
    if let Some(list) = m.get(&(arch, op)) {
        for ov in list {
            match ov {
                Override::NeonSeqThreshold(v) => cfg.neon_seq_threshold = *v,
                Override::ParGrain(v) => cfg.par_grain = *v,
                Override::ParThreshold(v) => cfg.par_threshold = *v,
                Override::FuseAttnThreshold(v) => cfg.fuse_attn_threshold = *v,
            }
        }
    }
    cfg
}

/// Convenience: resolve for the running target.
pub fn resolve_current(op: OpClass) -> KernelConfig {
    resolve(CpuArch::current(), op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serialize tests — `resolve` reads a process-global override table.
    static DISPATCH_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn with_clean_table(f: impl FnOnce()) {
        let _guard = DISPATCH_TEST_LOCK
            .lock()
            .expect("dispatch test lock poisoned");
        clear_overrides_for_tests();
        f();
    }

    #[test]
    fn defaults_pass_through() {
        with_clean_table(|| {
            let arch = CpuArch::AppleSilicon;
            let op = OpClass::Matmul;
            let resolved = resolve(arch, op);
            let default = kernel_config_for(arch, op);
            assert_eq!(resolved.neon_seq_threshold, default.neon_seq_threshold);
            assert_eq!(resolved.par_threshold, default.par_threshold);
        });
    }

    #[test]
    fn override_replaces_field() {
        with_clean_table(|| {
            let arch = CpuArch::AppleSilicon;
            let op = OpClass::Matmul;
            set_override(arch, op, Override::NeonSeqThreshold(7));
            let r = resolve(arch, op);
            assert_eq!(r.neon_seq_threshold, 7);
            // Other fields untouched.
            let d = kernel_config_for(arch, op);
            assert_eq!(r.par_threshold, d.par_threshold);
        });
    }

    #[test]
    fn override_for_one_field_replaces_just_that() {
        with_clean_table(|| {
            let arch = CpuArch::X86_64;
            let op = OpClass::Attention;
            set_override(arch, op, Override::NeonSeqThreshold(5));
            set_override(arch, op, Override::NeonSeqThreshold(9)); // replaces
            let r = resolve(arch, op);
            assert_eq!(r.neon_seq_threshold, 9);
        });
    }
}
