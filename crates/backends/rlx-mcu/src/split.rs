// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Running one model on several chips (or cores), each owning a contiguous
//! range of layers: a pipeline. Each chip stores only its layers' weights
//! and computes only their share, and hands the activations to the next over
//! SPI, UART or shared memory.
//!
//! * [`balance`] picks the cut points: contiguous stages with the smallest
//!   possible largest stage cost.
//! * [`send_rows`] / [`recv_rows`] move one `[rows, cols]` f32 activation
//!   block over any byte [`Link`], framed with a magic word, a sequence
//!   number, the shape and a CRC-32, so a dropped byte or a desynchronised
//!   stream is an error rather than wrong text. The payload is the f32 bits
//!   unchanged: a pipelined run gives the same bits as one chip.

use crate::Error;

/// `"RLXA"` as a little-endian word: the start of every activation frame.
pub const FRAME_MAGIC: u32 = u32::from_le_bytes(*b"RLXA");
/// Header words: magic, sequence, rows, cols, CRC-32 of the payload.
pub const HEADER_WORDS: usize = 5;

/// Split layers with per-layer `costs` into `cuts.len() - 1` contiguous
/// stages, minimizing the most expensive stage. Writes the boundaries to
/// `cuts` (`cuts[0] = 0`, `cuts[last] = costs.len()`, stage `s` owns layers
/// `cuts[s]..cuts[s + 1]`, none empty) and returns that largest stage cost.
/// `Error::Shape` if there are fewer layers than stages.
pub fn balance(costs: &[u64], cuts: &mut [usize]) -> Result<u64, Error> {
    let stages = cuts.len().checked_sub(1).ok_or(Error::Shape)?;
    if stages == 0 || costs.len() < stages {
        return Err(Error::Shape);
    }
    // Smallest cap a greedy left-to-right fill meets with `stages` stages,
    // leaving at least one layer for each stage still to come.
    let fits = |cap: u64, cuts: &mut [usize]| -> bool {
        let n = costs.len();
        cuts[0] = 0;
        let (mut s, mut sum) = (0usize, 0u64);
        for (i, &c) in costs.iter().enumerate() {
            if c > cap {
                return false;
            }
            // Open the next stage when this one would exceed the cap, or
            // when the layers left are just enough for one per stage left.
            let must_cut = n - i == stages - 1 - s;
            if (sum + c > cap || must_cut) && i > cuts[s] {
                s += 1;
                if s == stages {
                    return false;
                }
                cuts[s] = i;
                sum = 0;
            }
            sum += c;
        }
        if s + 1 != stages {
            return false;
        }
        cuts[stages] = n;
        true
    };
    let (mut lo, mut hi) = (
        costs.iter().copied().max().unwrap_or(0),
        costs.iter().sum::<u64>(),
    );
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if fits(mid, cuts) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    if !fits(lo, cuts) {
        return Err(Error::Shape);
    }
    Ok((0..stages)
        .map(|s| costs[cuts[s]..cuts[s + 1]].iter().sum::<u64>())
        .max()
        .unwrap_or(0))
}

/// A byte pipe between two stages: SPI, UART, a mailbox in shared RAM, or a
/// buffer in a host test. Both calls block until the whole slice moved.
pub trait Link {
    /// Send all of `bytes`.
    fn write(&mut self, bytes: &[u8]) -> Result<(), Error>;
    /// Fill all of `buf`.
    fn read(&mut self, buf: &mut [u8]) -> Result<(), Error>;
}

/// CRC-32 (IEEE 802.3, reflected, as zlib) continued from `crc`; start
/// with 0.
pub fn crc32(crc: u32, bytes: &[u8]) -> u32 {
    let mut c = !crc;
    for &b in bytes {
        c ^= b as u32;
        for _ in 0..8 {
            c = (c >> 1) ^ (0xEDB8_8320 & 0u32.wrapping_sub(c & 1));
        }
    }
    !c
}

fn f32_bytes(data: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and any byte view of it is valid.
    unsafe { core::slice::from_raw_parts(data.as_ptr().cast(), core::mem::size_of_val(data)) }
}

/// Send a `[rows, cols]` activation block (`data.len() == rows * cols`) as
/// one frame. The payload is the f32 values little-endian; on a big-endian
/// target this would need a swap (none of the MCUs here are).
pub fn send_rows<L: Link + ?Sized>(
    link: &mut L,
    seq: u32,
    rows: usize,
    cols: usize,
    data: &[f32],
) -> Result<(), Error> {
    if data.len() != rows * cols {
        return Err(Error::Shape);
    }
    let payload = f32_bytes(data);
    let head = [
        FRAME_MAGIC,
        seq,
        rows as u32,
        cols as u32,
        crc32(0, payload),
    ];
    let mut hb = [0u8; HEADER_WORDS * 4];
    for (d, w) in hb.as_chunks_mut::<4>().0.iter_mut().zip(head) {
        d.copy_from_slice(&w.to_le_bytes());
    }
    link.write(&hb)?;
    link.write(payload)
}

/// Shape and sequence number of a received frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    /// Sender's sequence number.
    pub seq: u32,
    /// Rows (time frames).
    pub rows: usize,
    /// Columns (channels).
    pub cols: usize,
}

/// Receive one frame into the front of `data`; `cols` is the width this
/// stage expects. Checks the magic word, the width, that it fits, and the
/// CRC (`Error::Truncated` on a mismatch: the bytes did not arrive intact).
pub fn recv_rows<L: Link + ?Sized>(
    link: &mut L,
    cols: usize,
    data: &mut [f32],
) -> Result<Frame, Error> {
    let mut hb = [0u8; HEADER_WORDS * 4];
    link.read(&mut hb)?;
    let mut head = [0u32; HEADER_WORDS];
    for (w, s) in head.iter_mut().zip(hb.as_chunks::<4>().0) {
        *w = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
    }
    if head[0] != FRAME_MAGIC {
        return Err(Error::BadMagic);
    }
    let (rows, got_cols) = (head[2] as usize, head[3] as usize);
    if got_cols != cols {
        return Err(Error::Shape);
    }
    let n = rows.checked_mul(cols).ok_or(Error::Shape)?;
    if n > data.len() {
        return Err(Error::ScratchTooSmall);
    }
    let dst = &mut data[..n];
    // SAFETY: as in `f32_bytes`; every bit pattern is a valid f32.
    let bytes = unsafe { core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), n * 4) };
    link.read(bytes)?;
    if crc32(0, bytes) != head[4] {
        return Err(Error::Truncated);
    }
    Ok(Frame {
        seq: head[1],
        rows,
        cols,
    })
}

/// A [`Link`] over an in-memory queue, for host pipelines and tests.
#[cfg(feature = "std")]
#[derive(Default)]
pub struct MemLink(pub std::collections::VecDeque<u8>);

#[cfg(feature = "std")]
impl Link for MemLink {
    fn write(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.0.extend(bytes);
        Ok(())
    }
    fn read(&mut self, buf: &mut [u8]) -> Result<(), Error> {
        if self.0.len() < buf.len() {
            return Err(Error::Truncated);
        }
        let n = buf.len();
        for (d, s) in buf.iter_mut().zip(self.0.drain(..n)) {
            *d = s;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn balance_is_optimal_and_contiguous() {
        let costs = [5u64, 1, 1, 1, 1, 5, 2, 2];
        let mut cuts = [0usize; 3];
        let worst = balance(&costs, &mut cuts).unwrap();
        assert_eq!(cuts[0], 0);
        assert_eq!(cuts[2], costs.len());
        assert!(cuts[0] < cuts[1] && cuts[1] < cuts[2]);
        // Brute force over every single cut.
        let best = (1..costs.len())
            .map(|c| {
                let a: u64 = costs[..c].iter().sum();
                let b: u64 = costs[c..].iter().sum();
                a.max(b)
            })
            .min()
            .unwrap();
        assert_eq!(worst, best);
    }

    #[test]
    fn balance_gives_every_stage_a_layer() {
        let costs = [100u64, 1, 1];
        let mut cuts = [0usize; 4];
        balance(&costs, &mut cuts).unwrap();
        assert_eq!(cuts, [0, 1, 2, 3]);
        assert!(balance(&costs, &mut [0usize; 5]).is_err());
    }

    #[test]
    fn crc32_matches_zlib() {
        assert_eq!(crc32(0, b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn frames_round_trip_and_catch_corruption() {
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 1.0).collect();
        let mut link = MemLink::default();
        send_rows(&mut link, 7, 3, 4, &data).unwrap();
        let mut got = [0f32; 16];
        let f = recv_rows(&mut link, 4, &mut got).unwrap();
        assert_eq!(
            f,
            Frame {
                seq: 7,
                rows: 3,
                cols: 4
            }
        );
        assert_eq!(&got[..12], &data[..]);

        send_rows(&mut link, 8, 3, 4, &data).unwrap();
        link.0[HEADER_WORDS * 4 + 5] ^= 1;
        assert_eq!(recv_rows(&mut link, 4, &mut got), Err(Error::Truncated));

        send_rows(&mut link, 9, 3, 4, &data).unwrap();
        assert_eq!(recv_rows(&mut link, 6, &mut got), Err(Error::Shape));
    }
}
