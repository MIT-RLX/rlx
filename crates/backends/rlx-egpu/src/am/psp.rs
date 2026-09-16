// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PSP firmware load — handing the card its own signed firmware.
//!
//! # Status: written, never executed
//!
//! See [`super`]. The mailbox protocol below is the one the reference
//! implementation uses and no part of it has been run against a card.
//!
//! ## What the PSP is
//!
//! The Platform Security Processor is a separate core on the die that boots
//! before anything else works. Firmware is signed by AMD and verified against
//! keys fused into the silicon — rlx cannot produce it, sign it, or bypass the
//! check, and does not try. What bring-up does is place a blob in memory the
//! PSP can read and tell it where: the card verifies and boots itself.
//!
//! That is why "send the card a signed thing instead of writing a driver" is
//! half right. The blob is signed and the card does the work, but something
//! still has to sequence the mailbox, and that something is a driver.
//!
//! ## The mailbox
//!
//! Communication is a handful of `C2PMSG` registers. Per component:
//!
//! 1. Wait for the bootloader ready bit in the status register.
//! 2. Copy the blob into a physically-contiguous staging buffer.
//! 3. Write the buffer's address, shifted right by 20, into the address
//!    register.
//! 4. Write the component id into the command register.
//! 5. Wait for ready again — except after the sOS component, which does not
//!    return through the bootloader path.
//!
//! The components load in a fixed order, each one extending the chain of trust
//! established by the last.

use crate::EgpuError;

/// Registers differ by MP0 IP version: pre-14.0 parts use `MP0_SMN_C2PMSG_*`,
/// 14.0 and later `MPASP_SMN_C2PMSG_*`. Both are indexed the same way, so the
/// only per-version difference is the base name — which comes from the IP
/// discovery table, and is why [`super::Stage::Discovery`] blocks everything
/// downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MailboxFamily {
    /// MP0 < 14.0.0 — `regMP0_SMN_C2PMSG_*`.
    Mp0,
    /// MP0 >= 14.0.0 — `regMPASP_SMN_C2PMSG_*`.
    Mpasp,
}

impl MailboxFamily {
    /// Pick the register family for an MP0 IP version.
    pub fn for_ip_version(major: u8, minor: u8, patch: u8) -> Self {
        if (major, minor, patch) >= (14, 0, 0) {
            Self::Mpasp
        } else {
            Self::Mp0
        }
    }

    /// Register name prefix, for logging and for looking the offset up in a
    /// table this tree does not have.
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Mp0 => "regMP0_SMN_C2PMSG",
            Self::Mpasp => "regMPASP_SMN_C2PMSG",
        }
    }
}

/// Mailbox register indices, shared across both families.
pub mod mailbox {
    /// Bootloader command — write a component id here to start a load.
    pub const CMD: u32 = 35;
    /// Bootloader staging address, in units of 1 MiB.
    pub const ADDR: u32 = 36;
    /// sOS liveness — non-zero once the secure OS is running.
    pub const SOS_STATUS: u32 = 81;
    /// Ring control.
    pub const RING_CTRL: u32 = 64;
    /// Ring liveness.
    pub const RING_STATUS: u32 = 71;
}

/// Set in the bootloader status register when it is ready for a command.
pub const BOOTLOADER_READY: u32 = 0x8000_0000;

/// Alignment and granularity of the staging buffer: the address register holds
/// the address shifted right by 20, so the buffer must be 1 MiB aligned.
pub const STAGING_ALIGN: u64 = 1 << 20;

/// A firmware component, in the order it must be loaded. Each value is the
/// bootloader command id written to [`mailbox::CMD`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Component {
    KeyDatabase = 0x8_0000,
    SplTable = 0x10_0000,
    SysDriver = 0x1_0000,
    SocDriver = 0xB0_0000,
    IntfDriver = 0xD0_0000,
    DbgDriver = 0xC0_0000,
    RasDriver = 0xE0_0000,
    /// The secure OS. Loading this one does not return through the bootloader
    /// ready path — liveness is read from [`mailbox::SOS_STATUS`] instead.
    SecureOs = 0x2_0000,
}

impl Component {
    /// Load order. Each component extends the chain of trust the previous one
    /// established, so this sequence is not reorderable.
    pub const ORDER: [Component; 8] = [
        Component::KeyDatabase,
        Component::SplTable,
        Component::SysDriver,
        Component::SocDriver,
        Component::IntfDriver,
        Component::DbgDriver,
        Component::RasDriver,
        Component::SecureOs,
    ];

    /// `true` when the bootloader hands control on instead of returning ready.
    pub fn is_terminal(self) -> bool {
        matches!(self, Component::SecureOs)
    }
}

/// Encode the value written to [`mailbox::ADDR`] for a staging buffer.
///
/// The register holds the address in 1 MiB units, so a buffer that is not 1 MiB
/// aligned would silently load from the wrong place — the check is here rather
/// than at the call site because the shift is what hides the mistake.
pub fn staging_address_register(physical_addr: u64) -> Result<u32, String> {
    if !physical_addr.is_multiple_of(STAGING_ALIGN) {
        return Err(format!(
            "PSP staging buffer {physical_addr:#x} is not {STAGING_ALIGN:#x}-aligned"
        ));
    }
    let shifted = physical_addr >> 20;
    u32::try_from(shifted)
        .map_err(|_| format!("PSP staging buffer {physical_addr:#x} is above the 52-bit range"))
}

/// The blob a component load needs, padded as the bootloader expects.
///
/// The staging copy is padded to a 16-byte multiple: the copy on the far side
/// reads in 16-byte units, and a short tail reads past the buffer.
pub fn stage_blob(blob: &[u8]) -> Vec<u8> {
    let mut staged = blob.to_vec();
    staged.extend_from_slice(&[0u8; 4]);
    while !staged.len().is_multiple_of(16) {
        staged.push(0);
    }
    staged
}

/// Load the signed firmware chain over an open transport.
///
/// Refuses rather than starting: the load needs the MP0 IP version to pick the
/// register family, and that comes from the discovery table
/// ([`super::Stage::Discovery`]). Starting a chain that cannot finish leaves the
/// PSP mid-handshake, which needs a bus reset to clear.
#[cfg(feature = "dext")]
pub fn load_chain(
    _transport: &mut crate::dext::PciTransport,
    _family: MailboxFamily,
) -> Result<(), EgpuError> {
    Err(EgpuError::Unsupported(
        "PSP firmware load has never been executed: it needs the MP0 IP version from the \
         discovery table to resolve register offsets, and a card attached to run against"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_load_order_is_fixed_and_ends_at_the_secure_os() {
        // Each component extends the previous one's chain of trust; a reorder
        // is a verification failure, not a performance question.
        assert_eq!(Component::ORDER.len(), 8);
        assert_eq!(Component::ORDER[0], Component::KeyDatabase);
        assert_eq!(*Component::ORDER.last().unwrap(), Component::SecureOs);
        assert!(Component::SecureOs.is_terminal());
        assert!(!Component::KeyDatabase.is_terminal());
        assert_eq!(
            Component::ORDER.iter().filter(|c| c.is_terminal()).count(),
            1,
            "exactly one component hands control on"
        );
    }

    #[test]
    fn component_ids_are_distinct() {
        // Two components sharing an id would load the wrong blob and fail
        // verification somewhere unrelated.
        let mut ids: Vec<u32> = Component::ORDER.iter().map(|c| *c as u32).collect();
        ids.sort_unstable();
        let count = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), count, "duplicate bootloader command id");
    }

    #[test]
    fn the_staging_address_is_shifted_and_alignment_is_enforced() {
        // The register holds 1 MiB units, so a misaligned buffer silently loads
        // from the wrong address — hence the check rather than a bare shift.
        assert_eq!(staging_address_register(0x40_0000).unwrap(), 4);
        assert_eq!(staging_address_register(0).unwrap(), 0);
        assert!(staging_address_register(0x40_1000).is_err());
        assert!(staging_address_register(STAGING_ALIGN - 1).is_err());
    }

    #[test]
    fn staged_blobs_are_padded_to_sixteen_bytes() {
        // The far side copies in 16-byte units; a short tail reads past the end.
        for len in [0usize, 1, 15, 16, 17, 4096] {
            let staged = stage_blob(&vec![0xab; len]);
            assert_eq!(staged.len() % 16, 0, "len {len} left a short tail");
            assert!(staged.len() >= len + 4);
            assert_eq!(&staged[..len], &vec![0xab; len][..]);
        }
    }

    #[test]
    fn the_mailbox_family_follows_the_ip_version() {
        assert_eq!(MailboxFamily::for_ip_version(13, 0, 0), MailboxFamily::Mp0);
        assert_eq!(MailboxFamily::for_ip_version(13, 9, 9), MailboxFamily::Mp0);
        assert_eq!(
            MailboxFamily::for_ip_version(14, 0, 0),
            MailboxFamily::Mpasp
        );
        assert_eq!(
            MailboxFamily::for_ip_version(14, 0, 3),
            MailboxFamily::Mpasp
        );
        assert_ne!(MailboxFamily::Mp0.prefix(), MailboxFamily::Mpasp.prefix());
    }
}
