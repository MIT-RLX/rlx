// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! AMD device bring-up (feature `am`).
//!
//! # Status: written, never executed
//!
//! No card has been attached to this tree. Every register sequence below was
//! derived from the reference implementation and reviewed, and **none of it has
//! run**. Bring-up fails by hanging on a firmware handshake, which is a class of
//! bug that reading cannot find, so treat this module as a starting point for
//! the first session with hardware rather than as a driver.
//!
//! [`VALIDATED_ON_HARDWARE`] encodes that, and `crate::is_available` gates on
//! it rather than on `cfg!(feature = "am")` — compiling the module must not make
//! the device a dispatch target.
//!
//! ## What is here, and what it is worth
//!
//! | Module | Kind | Verified |
//! |--------|------|----------|
//! | [`pm4`] | packet encoding, pure computation | yes — unit tested |
//! | [`pt`] | page-table math, pure computation | yes — unit tested |
//! | [`psp`] | firmware load over the transport | no — needs a card |
//! | this module | IP bring-up order | no — needs a card |
//!
//! The split is deliberate. Packet encoding and page-table arithmetic are
//! decidable without a device, so they are written properly and tested. The
//! sequencing is not, so it is written as structure with the register access it
//! needs named explicitly, and it refuses to run rather than poking registers
//! whose offsets were never confirmed.
//!
//! ## The order a card comes up in
//!
//! 1. **Reset** the function and re-enable bus mastering.
//! 2. **Discovery** — read the IP discovery table from the end of VRAM to learn
//!    which IP blocks are present and at which versions. Every later step needs
//!    those versions to pick firmware filenames and register offsets.
//! 3. **PSP** — load the signed firmware chain ([`psp`]): key database, SPL,
//!    system/SOC/interface/debug/RAS drivers, then sOS. The card verifies each
//!    against keys fused into the die.
//! 4. **GMC** — memory controller and the VRAM aperture, then the page tables
//!    from [`pt`].
//! 5. **SMU** — power management, clocks, and the fan curve.
//! 6. **GFX / SDMA** — ring buffers, doorbells, and the RLC.
//!
//! Steps 2, 4, 5 and 6 need per-IP-version register offset tables, which the
//! reference implementation generates from vendor headers (tens of thousands of
//! lines). Those tables are not in this tree, and inventing offsets would be
//! worse than not having them: a wrong offset writes a live register and the
//! failure surfaces somewhere unrelated.

pub mod pm4;
pub mod psp;
pub mod pt;

use crate::EgpuError;

/// Whether this bring-up has ever brought a card up.
///
/// `false`, and it stays `false` until someone attaches a supported GPU, runs
/// the sequence, and gets a working ring. `crate::is_available` reads this, so
/// flipping it is the deliberate act of claiming the backend works — not a
/// side effect of enabling a Cargo feature.
pub const VALIDATED_ON_HARDWARE: bool = false;

/// A stage of bring-up, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Reset,
    Discovery,
    Psp,
    Gmc,
    Smu,
    GfxSdma,
}

impl Stage {
    /// Bring-up order.
    pub const ORDER: [Stage; 6] = [
        Stage::Reset,
        Stage::Discovery,
        Stage::Psp,
        Stage::Gmc,
        Stage::Smu,
        Stage::GfxSdma,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Reset => "function reset",
            Stage::Discovery => "IP discovery table",
            Stage::Psp => "PSP firmware load",
            Stage::Gmc => "memory controller + page tables",
            Stage::Smu => "power management",
            Stage::GfxSdma => "GFX/SDMA rings",
        }
    }

    /// What this stage still needs before it can run.
    ///
    /// `None` means the stage is implemented as far as it can be without
    /// hardware; `Some(_)` names the missing input.
    pub fn blocker(self) -> Option<&'static str> {
        match self {
            // Reset and the PSP load talk to the device through documented
            // mailbox registers the transport already exposes.
            Stage::Reset | Stage::Psp => None,
            Stage::Discovery => Some(
                "the IP discovery table parser: its binary layout is versioned per ASIC \
                 and has not been transcribed",
            ),
            Stage::Gmc | Stage::Smu | Stage::GfxSdma => Some(
                "per-IP-version register offset tables, which the reference \
                 implementation generates from vendor headers",
            ),
        }
    }
}

/// Report what a bring-up would do and where it would stop, without touching a
/// device. Use this to see the shape of the remaining work.
pub fn plan() -> Vec<(Stage, Option<&'static str>)> {
    Stage::ORDER.iter().map(|s| (*s, s.blocker())).collect()
}

/// The first stage that cannot run.
pub fn first_blocked_stage() -> Option<Stage> {
    Stage::ORDER.iter().copied().find(|s| s.blocker().is_some())
}

/// Bring a card up over an open transport.
///
/// Refuses immediately rather than running a partial sequence. Getting partway
/// through bring-up and stopping leaves the card in a state that needs a bus
/// reset to recover, and on a Thunderbolt-attached device that can take the
/// tunnel down with it — so a sequence that is known to be incomplete must not
/// be started.
#[cfg(feature = "dext")]
pub fn bring_up(_transport: &mut crate::dext::PciTransport) -> Result<(), EgpuError> {
    let blocked = first_blocked_stage().expect("bring-up is incomplete by construction");
    Err(EgpuError::Unsupported(format!(
        "AMD bring-up stops at `{}`: {}. The sequence in rlx_egpu::am has never been \
         executed — it needs a card attached before it can be finished or trusted.",
        blocked.name(),
        blocked.blocker().unwrap_or("unknown"),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_backend_does_not_claim_to_work() {
        // The whole point of the flag: enabling the Cargo feature must not make
        // an unexecuted driver a dispatch target.
        // Read through a binding: asserting on the const directly is a lint,
        // and the claim being tested is exactly that this const is still false.
        let validated: bool = VALIDATED_ON_HARDWARE;
        assert!(
            !validated,
            "an unexecuted bring-up must not claim validation"
        );
        assert!(!crate::is_available());
    }

    #[test]
    fn the_plan_is_ordered_and_names_every_blocker() {
        let plan = plan();
        assert_eq!(plan.len(), Stage::ORDER.len());
        assert_eq!(plan[0].0, Stage::Reset, "reset comes first");
        assert_eq!(
            plan[2].0,
            Stage::Psp,
            "firmware before the memory controller"
        );
        assert!(
            plan.iter().any(|(_, blocker)| blocker.is_some()),
            "bring-up is incomplete; the plan must say so"
        );
        for (stage, blocker) in plan {
            if let Some(text) = blocker {
                assert!(!text.is_empty(), "{stage:?} blocked for no stated reason");
            }
        }
    }

    #[test]
    fn discovery_is_the_first_thing_that_blocks() {
        // Reset and the PSP load are reachable; everything downstream needs the
        // discovery table to pick firmware and register offsets, so that is
        // where the work actually restarts once hardware is available.
        assert_eq!(first_blocked_stage(), Some(Stage::Discovery));
        assert!(Stage::Reset.blocker().is_none());
        assert!(Stage::Psp.blocker().is_none());
    }
}
