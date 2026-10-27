// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Keep automatic NUMA balancing away from a ROCm process on an APU.
//!
//! On a machine with more than one NUMA node and `kernel.numa_balancing` on,
//! the kernel periodically unmaps a process's pages to sample where they are
//! used. Each pass fires MMU notifiers on host memory the GPU has mapped, and
//! amdgpu answers by evicting the process's queues. On gfx11 APUs (Radeon
//! 780M, gfx1103) the MES firmware can hang removing a queue: the driver logs
//! `MES failed to respond to msg=REMOVE_QUEUE … Failed to evict process
//! queues`, resets the GPU, and the next copy fails with `HipError(719)`.
//! On the amd rig, a generate stage right after a CPU training run hit this
//! 5 times in 6; 0 in 4 under `numactl --localalloc`, and 0 in 4 with this
//! opt-out in place.
//!
//! The kernel skips that sampling for a task whose memory policy is explicit
//! and lacks `MPOL_F_MOF`, so [`apply`] sets `MPOL_LOCAL` (the default
//! placement — allocate on the local node — without the balancing). It is a
//! per-thread policy: it covers the calling thread and every thread it starts
//! afterwards (HIP's own, and the backend's), so it runs before `hipInit`.
//! Threads that already exist keep theirs; to cover a whole process, start it
//! under `numactl --localalloc`.
//!
//! `RLX_ROCM_NUMA_GUARD=0` turns this off, `=1` applies it on any ROCm host.

use std::sync::Once;

static ONCE: Once = Once::new();

/// Set `MPOL_LOCAL` on this thread when this is an APU on a multi-node host
/// with NUMA balancing on (or when forced). Runs once per process.
pub fn apply() {
    ONCE.call_once(|| {
        let mode = rlx_ir::env::var("RLX_ROCM_NUMA_GUARD");
        let forced = match mode.as_deref() {
            Some("0") => return,
            Some(v) => !v.is_empty(),
            None => false,
        };
        if !(forced || needed()) {
            return;
        }
        if set_local_policy() {
            if rlx_ir::env::flag("RLX_VERBOSE") {
                eprintln!(
                    "rlx-rocm: APU on a multi-node host with NUMA balancing on — this \
                     process opts out of it (MPOL_LOCAL), so amdgpu never has to evict \
                     its GPU queues mid-run"
                );
            }
        } else {
            eprintln!(
                "rlx-rocm: could not opt out of NUMA balancing; if the GPU resets \
                 (HipError 719), start the process under `numactl --localalloc`"
            );
        }
    });
}

/// NUMA balancing on, more than one node, and an APU among the KFD nodes.
fn needed() -> bool {
    let balancing =
        std::fs::read_to_string("/proc/sys/kernel/numa_balancing").is_ok_and(|s| s.trim() != "0");
    balancing && numa_nodes() > 1 && has_apu()
}

fn numa_nodes() -> usize {
    std::fs::read_dir("/sys/devices/system/node")
        .map(|d| {
            d.flatten()
                .filter(|e| {
                    let n = e.file_name();
                    let n = n.to_string_lossy();
                    n.strip_prefix("node")
                        .is_some_and(|i| i.chars().all(|c| c.is_ascii_digit()))
                })
                .count()
        })
        .unwrap_or(0)
}

/// An APU's GPU among the KFD topology nodes: a node with both CPU cores and
/// GPU SIMDs (older kernels), or a GPU node with no local memory (newer ones
/// list the APU's CPU and GPU as separate nodes).
fn has_apu() -> bool {
    let Ok(nodes) = std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes") else {
        return false;
    };
    nodes.flatten().any(|n| {
        let props = std::fs::read_to_string(n.path().join("properties")).unwrap_or_default();
        let get = |key: &str| {
            props
                .lines()
                .find_map(|l| l.strip_prefix(key)?.trim().parse::<u64>().ok())
                .unwrap_or(0)
        };
        let has = |key: &str| props.lines().any(|l| l.starts_with(key));
        get("simd_count ") > 0
            && (get("cpu_cores_count ") > 0
                || (has("local_mem_size ") && get("local_mem_size ") == 0))
    })
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn set_local_policy() -> bool {
    use std::ffi::c_long;
    unsafe extern "C" {
        fn syscall(num: c_long, ...) -> c_long;
    }
    #[cfg(target_arch = "x86_64")]
    const SYS_SET_MEMPOLICY: c_long = 238;
    #[cfg(target_arch = "aarch64")]
    const SYS_SET_MEMPOLICY: c_long = 237;
    const MPOL_LOCAL: c_long = 4;
    // SAFETY: set_mempolicy(MPOL_LOCAL, NULL, 0) reads no user memory.
    unsafe {
        syscall(
            SYS_SET_MEMPOLICY,
            MPOL_LOCAL,
            std::ptr::null::<u64>(),
            0 as c_long,
        ) == 0
    }
}

#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
fn set_local_policy() -> bool {
    false
}

#[cfg(test)]
mod tests {
    #[test]
    fn node_and_apu_probes_do_not_panic_off_linux_or_without_kfd() {
        let _ = super::numa_nodes();
        let _ = super::has_apu();
        let _ = super::needed();
    }
}
