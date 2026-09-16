// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPUVM page tables — the address space a GPU sees.
//!
//! Like [`super::pm4`] this is pure computation: walk a virtual address into
//! level indices, encode entries, and the result is a tree of `u64`s that gets
//! DMA'd into memory the card can read. Nothing here touches a device, so it is
//! tested.
//!
//! GFX10 through GFX12 use a 4-level tree over 4 KiB pages, 9 index bits per
//! level, covering 48 bits of virtual address:
//!
//! ```text
//!  47..39   38..30   29..21   20..12   11..0
//!   PDE2     PDE1     PDE0     PTE     offset
//! ```
//!
//! **Entry bit positions below follow the reference implementation
//! (`amdgpu_vm.h`) and have not been validated against hardware** — no card has
//! been attached to run them. Treat them as reviewed-but-unexecuted.

/// Bits of virtual address covered by the page offset.
pub const PAGE_SHIFT: u32 = 12;
/// 4 KiB pages.
pub const PAGE_SIZE: u64 = 1 << PAGE_SHIFT;
/// Index bits per directory level.
pub const LEVEL_BITS: u32 = 9;
/// Entries in one directory.
pub const ENTRIES_PER_LEVEL: usize = 1 << LEVEL_BITS;
/// PDE2 → PDE1 → PDE0 → PTE.
pub const LEVELS: usize = 4;
/// Virtual address bits a 4-level tree covers.
pub const VA_BITS: u32 = PAGE_SHIFT + LEVELS as u32 * LEVEL_BITS;

// Entry flags. Source: the reference driver's `amdgpu_vm.h`.
/// Entry is populated.
pub const PTE_VALID: u64 = 1 << 0;
/// Target is system memory rather than VRAM.
pub const PTE_SYSTEM: u64 = 1 << 1;
/// Snoop the host cache — required for anything the CPU also writes.
pub const PTE_SNOOPED: u64 = 1 << 2;
/// Shader instruction fetch is allowed.
pub const PTE_EXECUTABLE: u64 = 1 << 4;
pub const PTE_READABLE: u64 = 1 << 5;
pub const PTE_WRITEABLE: u64 = 1 << 6;
/// Set on a directory entry that maps a large page directly instead of
/// pointing at the next level.
pub const PDE_IS_PTE: u64 = 1 << 54;
/// Physical address field: bits 12..48 of the entry hold bits 12..48 of the
/// address, so an entry stores the address masked to page alignment.
pub const ADDR_MASK: u64 = 0x0000_ffff_ffff_f000;

/// How a mapping may be used. Kept as a struct rather than loose booleans so a
/// caller cannot silently swap read and write at a call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Access {
    pub readable: bool,
    pub writeable: bool,
    pub executable: bool,
    /// System memory (host RAM reached over the bus) rather than VRAM.
    pub system: bool,
    /// Snoop the host cache. Required when the CPU writes the same pages.
    pub snooped: bool,
}

impl Access {
    /// Device-local VRAM, read/write, no execute.
    pub const fn vram_rw() -> Self {
        Self {
            readable: true,
            writeable: true,
            executable: false,
            system: false,
            snooped: false,
        }
    }

    /// Host memory the CPU also touches — system and snooped, or the card will
    /// read stale lines.
    pub const fn sysmem_rw() -> Self {
        Self {
            readable: true,
            writeable: true,
            executable: false,
            system: true,
            snooped: true,
        }
    }

    /// Kernel code: readable and executable, never writeable.
    pub const fn code() -> Self {
        Self {
            readable: true,
            writeable: false,
            executable: true,
            system: false,
            snooped: false,
        }
    }

    fn flags(self) -> u64 {
        let mut f = PTE_VALID;
        if self.readable {
            f |= PTE_READABLE;
        }
        if self.writeable {
            f |= PTE_WRITEABLE;
        }
        if self.executable {
            f |= PTE_EXECUTABLE;
        }
        if self.system {
            f |= PTE_SYSTEM;
        }
        if self.snooped {
            f |= PTE_SNOOPED;
        }
        f
    }
}

/// Index into each directory level for `va`, outermost (PDE2) first.
pub fn level_indices(va: u64) -> [usize; LEVELS] {
    let mut out = [0usize; LEVELS];
    for (level, slot) in out.iter_mut().enumerate() {
        // Level 0 is the outermost directory, so it reads the highest bits.
        let shift = PAGE_SHIFT + (LEVELS - 1 - level) as u32 * LEVEL_BITS;
        *slot = ((va >> shift) as usize) & (ENTRIES_PER_LEVEL - 1);
    }
    out
}

/// Encode a leaf entry mapping one page at `physical_addr`.
pub fn page_entry(physical_addr: u64, access: Access) -> Result<u64, String> {
    if physical_addr & (PAGE_SIZE - 1) != 0 {
        return Err(format!(
            "physical address {physical_addr:#x} is not {PAGE_SIZE}-byte aligned"
        ));
    }
    if physical_addr & !ADDR_MASK != 0 {
        return Err(format!(
            "physical address {physical_addr:#x} does not fit the entry's address field"
        ));
    }
    Ok((physical_addr & ADDR_MASK) | access.flags())
}

/// Encode a directory entry pointing at the next level.
pub fn directory_entry(next_level_addr: u64) -> Result<u64, String> {
    if next_level_addr & (PAGE_SIZE - 1) != 0 {
        return Err(format!(
            "directory address {next_level_addr:#x} is not {PAGE_SIZE}-byte aligned"
        ));
    }
    // A directory entry is valid and holds an address; it must NOT set
    // PDE_IS_PTE, which would make the card treat the pointer as a large-page
    // mapping and read the directory as data.
    Ok((next_level_addr & ADDR_MASK) | PTE_VALID)
}

/// `true` when an entry maps memory rather than pointing at a directory.
pub fn entry_is_leaf(entry: u64, level: usize) -> bool {
    level == LEVELS - 1 || entry & PDE_IS_PTE != 0
}

/// Physical address carried by an entry.
pub fn entry_address(entry: u64) -> u64 {
    entry & ADDR_MASK
}

/// Number of 4 KiB pages spanned by `size` bytes starting at `va`, accounting
/// for a start that is not page-aligned.
pub fn pages_spanned(va: u64, size: u64) -> u64 {
    if size == 0 {
        return 0;
    }
    let first = va & !(PAGE_SIZE - 1);
    let last = (va + size - 1) & !(PAGE_SIZE - 1);
    (last - first) / PAGE_SIZE + 1
}

/// Check a range can be mapped before any of it is written: the VA must fit the
/// address space, the size must be a whole number of pages, and the range must
/// not wrap.
pub fn validate_range(va: u64, size: u64) -> Result<(), String> {
    if va & (PAGE_SIZE - 1) != 0 {
        return Err(format!("virtual address {va:#x} is not page-aligned"));
    }
    if size == 0 || size & (PAGE_SIZE - 1) != 0 {
        return Err(format!(
            "size {size:#x} is not a non-zero multiple of a page"
        ));
    }
    let end = va
        .checked_add(size)
        .ok_or_else(|| format!("range {va:#x}+{size:#x} wraps"))?;
    if end > 1u64 << VA_BITS {
        return Err(format!(
            "range {va:#x}+{size:#x} leaves the {VA_BITS}-bit address space"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_virtual_address_splits_into_nine_bit_levels() {
        // Craft an address with a distinct index at each level so a swapped
        // shift shows up rather than cancelling out.
        let va = (1u64 << 47) | (2u64 << 38) | (3u64 << 29) | (4u64 << 20);
        let idx = level_indices(va);
        assert_eq!(idx[0], (va >> 39) as usize & 0x1ff, "PDE2");
        assert_eq!(idx[1], (va >> 30) as usize & 0x1ff, "PDE1");
        assert_eq!(idx[2], (va >> 21) as usize & 0x1ff, "PDE0");
        assert_eq!(idx[3], (va >> 12) as usize & 0x1ff, "PTE");
        // The offset within a page must not reach any index.
        assert_eq!(level_indices(0xfff), [0, 0, 0, 0]);
        assert_eq!(level_indices(PAGE_SIZE)[3], 1);
    }

    #[test]
    fn access_flags_are_distinct_and_never_silently_dropped() {
        let code = page_entry(0x1000, Access::code()).unwrap();
        assert_ne!(code & PTE_EXECUTABLE, 0);
        assert_eq!(code & PTE_WRITEABLE, 0, "code must not be writeable");

        let sys = page_entry(0x2000, Access::sysmem_rw()).unwrap();
        assert_ne!(sys & PTE_SYSTEM, 0);
        assert_ne!(sys & PTE_SNOOPED, 0, "host-shared pages must be snooped");

        let vram = page_entry(0x3000, Access::vram_rw()).unwrap();
        assert_eq!(vram & PTE_SYSTEM, 0);
        assert_ne!(vram & PTE_VALID, 0);
        // Every encoding keeps its address recoverable.
        assert_eq!(entry_address(vram), 0x3000);
    }

    #[test]
    fn a_directory_entry_is_not_mistaken_for_a_large_page() {
        // Setting PDE_IS_PTE on a pointer makes the card read the next
        // directory as data — silent corruption rather than a fault.
        let pde = directory_entry(0x4000).unwrap();
        assert_eq!(pde & PDE_IS_PTE, 0);
        assert_eq!(entry_address(pde), 0x4000);
        assert!(!entry_is_leaf(pde, 0));
        assert!(
            entry_is_leaf(pde, LEVELS - 1),
            "the last level is always a leaf"
        );
        assert!(entry_is_leaf(pde | PDE_IS_PTE, 0));
    }

    #[test]
    fn misaligned_and_oversized_addresses_are_refused() {
        assert!(page_entry(0x1001, Access::vram_rw()).is_err());
        assert!(directory_entry(0x800).is_err());
        // Above the 48-bit address field.
        assert!(page_entry(1u64 << 48, Access::vram_rw()).is_err());
        assert!(page_entry(ADDR_MASK & !0xfff, Access::vram_rw()).is_ok());
    }

    #[test]
    fn page_counts_account_for_an_unaligned_start() {
        assert_eq!(pages_spanned(0, 0), 0);
        assert_eq!(pages_spanned(0, 1), 1);
        assert_eq!(pages_spanned(0, PAGE_SIZE), 1);
        assert_eq!(pages_spanned(0, PAGE_SIZE + 1), 2);
        // One byte either side of a boundary still touches two pages.
        assert_eq!(pages_spanned(PAGE_SIZE - 1, 2), 2);
        assert_eq!(pages_spanned(PAGE_SIZE - 1, 1), 1);
    }

    #[test]
    fn ranges_are_checked_before_anything_is_written() {
        assert!(validate_range(0, PAGE_SIZE).is_ok());
        assert!(validate_range(0x1000, 0).is_err(), "empty");
        assert!(validate_range(0x1001, PAGE_SIZE).is_err(), "unaligned va");
        assert!(validate_range(0, PAGE_SIZE + 1).is_err(), "partial page");
        assert!(
            validate_range(u64::MAX - 0xfff, PAGE_SIZE).is_err(),
            "wraps"
        );
        assert!(
            validate_range((1u64 << VA_BITS) - PAGE_SIZE, PAGE_SIZE * 2).is_err(),
            "leaves the address space"
        );
    }
}
