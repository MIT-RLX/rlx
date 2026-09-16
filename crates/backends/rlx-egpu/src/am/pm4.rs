// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PM4 packet encoding — the command words a GFX ring consumes.
//!
//! This is pure computation: build a `u32` stream, hand it to the command
//! processor. No device is involved in producing it, so unlike the bring-up
//! sequencing in [`super`] it is testable here and is tested below.
//!
//! A type-3 packet header is one word:
//!
//! ```text
//!  31 30 | 29 ...... 16 | 15 ... 8 | 7 .. 2 | 1 | 0
//!   type |   count-1    |  opcode  |  resvd | shader | predicate
//! ```
//!
//! `count` is the number of *body* words; the encoded field is `count - 1`,
//! which is the classic off-by-one to get wrong and the reason this is a
//! function rather than a macro at each call site.

/// Type-3 (command) packet.
const PACKET_TYPE3: u32 = 3;

/// PM4 opcodes used by a submission path. Values are the long-stable ones
/// shared across GFX9 through GFX12.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    Nop = 0x10,
    SetShReg = 0x76,
    SetUConfigReg = 0x79,
    DispatchDirect = 0x15,
    IndirectBuffer = 0x3f,
    WriteData = 0x37,
    WaitRegMem = 0x3c,
    ReleaseMem = 0x49,
    AcquireMem = 0x58,
    EventWrite = 0x46,
}

/// Encode a type-3 packet header.
///
/// `body_words` is the number of words that follow this header. Zero is
/// rejected: every type-3 packet carries at least one body word, and encoding
/// `count - 1` on an empty body would underflow into a 16-bit count of 0xffff
/// and run the command processor off the end of the ring.
pub fn header(opcode: Opcode, body_words: usize, predicate: bool) -> Result<u32, String> {
    if body_words == 0 {
        return Err(format!(
            "{opcode:?}: a type-3 packet needs at least one body word"
        ));
    }
    if body_words > 0x3fff {
        return Err(format!(
            "{opcode:?}: {body_words} body words exceeds the 14-bit count field"
        ));
    }
    Ok((PACKET_TYPE3 << 30)
        | (((body_words - 1) as u32 & 0x3fff) << 16)
        | ((opcode as u32 & 0xff) << 8)
        | u32::from(predicate))
}

/// A ring buffer being filled with packets.
#[derive(Debug, Default, Clone)]
pub struct PacketStream {
    words: Vec<u32>,
}

impl PacketStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// The encoded words.
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// Bytes, for copying into a DMA-mapped ring.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Append a packet with an explicit body.
    pub fn push(&mut self, opcode: Opcode, body: &[u32]) -> Result<&mut Self, String> {
        self.words.push(header(opcode, body.len(), false)?);
        self.words.extend_from_slice(body);
        Ok(self)
    }

    /// `NOP` padding — also how a ring is aligned before a submission.
    pub fn nop(&mut self, body_words: usize) -> Result<&mut Self, String> {
        self.push(Opcode::Nop, &vec![0; body_words.max(1)])
    }

    /// Jump to a command buffer elsewhere in memory. `gpu_addr` is a GPU
    /// virtual address, so the buffer must already be mapped by the page tables
    /// in [`super::pt`].
    pub fn indirect_buffer(&mut self, gpu_addr: u64, size_words: u32) -> Result<&mut Self, String> {
        if gpu_addr & 0x3 != 0 {
            return Err(format!(
                "indirect buffer address {gpu_addr:#x} is not word-aligned"
            ));
        }
        self.push(
            Opcode::IndirectBuffer,
            &[
                (gpu_addr & 0xffff_fffc) as u32,
                (gpu_addr >> 32) as u32,
                size_words & 0x000f_ffff,
            ],
        )
    }

    /// Write a 64-bit value to memory once prior work has retired — the
    /// completion signal a host polls instead of taking an interrupt.
    pub fn release_mem(&mut self, gpu_addr: u64, value: u64) -> Result<&mut Self, String> {
        self.push(
            Opcode::ReleaseMem,
            &[
                0,
                0,
                (gpu_addr & 0xffff_ffff) as u32,
                (gpu_addr >> 32) as u32,
                (value & 0xffff_ffff) as u32,
                (value >> 32) as u32,
                0,
            ],
        )
    }

    /// Launch a compute grid.
    pub fn dispatch_direct(&mut self, grid: [u32; 3], initiator: u32) -> Result<&mut Self, String> {
        if grid.contains(&0) {
            return Err(format!("dispatch grid {grid:?} has a zero dimension"));
        }
        self.push(
            Opcode::DispatchDirect,
            &[grid[0], grid[1], grid[2], initiator],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_header_encodes_type_opcode_and_count_minus_one() {
        let h = header(Opcode::DispatchDirect, 4, false).unwrap();
        assert_eq!(h >> 30, PACKET_TYPE3, "type field");
        assert_eq!((h >> 16) & 0x3fff, 3, "count is body_words - 1");
        assert_eq!((h >> 8) & 0xff, Opcode::DispatchDirect as u32);
        assert_eq!(h & 1, 0, "predicate off");
        assert_eq!(header(Opcode::Nop, 1, true).unwrap() & 1, 1, "predicate on");
    }

    #[test]
    fn an_empty_body_is_rejected_rather_than_underflowing() {
        // `count - 1` on an empty body wraps to 0x3fff and walks the command
        // processor off the end of the ring — the failure this guard exists for.
        assert!(header(Opcode::Nop, 0, false).is_err());
        assert!(header(Opcode::Nop, 0x4000, false).is_err());
        assert!(header(Opcode::Nop, 0x3fff, false).is_ok());
    }

    #[test]
    fn a_stream_lays_out_header_then_body() {
        let mut s = PacketStream::new();
        s.dispatch_direct([16, 2, 1], 0).unwrap();
        assert_eq!(s.len(), 5, "one header plus four body words");
        assert_eq!(&s.words()[1..], &[16, 2, 1, 0]);
        assert_eq!(s.to_bytes().len(), 20);
        assert_eq!(&s.to_bytes()[4..8], &16u32.to_le_bytes());
    }

    #[test]
    fn addresses_split_low_then_high_and_stay_aligned() {
        let mut s = PacketStream::new();
        s.indirect_buffer(0x1234_5678_9abc_d000, 64).unwrap();
        assert_eq!(s.words()[1], 0x9abc_d000, "low word");
        assert_eq!(s.words()[2], 0x1234_5678, "high word");
        assert_eq!(s.words()[3], 64);
        assert!(
            PacketStream::new().indirect_buffer(0x1001, 4).is_err(),
            "a misaligned indirect buffer must be refused"
        );
    }

    #[test]
    fn a_zero_dimension_dispatch_is_refused() {
        // A zero extent launches nothing and is always a caller bug; catching it
        // here is cheaper than diagnosing a ring that completed with no effect.
        assert!(PacketStream::new().dispatch_direct([0, 1, 1], 0).is_err());
        assert!(PacketStream::new().dispatch_direct([1, 1, 0], 0).is_err());
    }

    #[test]
    fn release_mem_carries_a_64_bit_fence_value() {
        let mut s = PacketStream::new();
        s.release_mem(0xdead_0000, 0x1_0000_0002).unwrap();
        let w = s.words();
        assert_eq!(w[3], 0xdead_0000);
        assert_eq!(w[5], 0x0000_0002, "fence low");
        assert_eq!(w[6], 0x0000_0001, "fence high");
    }
}
