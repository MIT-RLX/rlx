// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `no_std` building blocks for running speech models on microcontrollers
//! without an allocator: the in-place model file format ([`blob`]), int8 /
//! int4 matrix kernels ([`kernels`]), the NeMo log-mel frontend ([`mel`]),
//! PCM input ([`Pcm`], cut at pauses with [`next_cut`]), paged weights
//! ([`Pager`]) and layer pipelines across chips ([`split`]). Everything numeric goes
//! through `libm`, so a host build and an MCU build produce the same bits.
//!
//! The model crates built on it (Conformer-CTC, Citrinet, Moonshine
//! streaming, Kroko) live in `rlx-models`.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod blob;
pub mod kernels;
pub mod mel;
pub mod split;

/// Why parsing or inference failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not this model's file format (magic word).
    BadMagic,
    /// A file of another format version.
    Version,
    /// The file ends inside a record.
    Truncated,
    /// The blob does not start on a 4-byte boundary.
    Unaligned,
    /// A record has the wrong size for the header's hyperparameters.
    Shape,
    /// Hyperparameters this implementation does not handle.
    Unsupported,
    /// The scratch arenas are smaller than the model's `scratch_len` asks for.
    ScratchTooSmall,
    /// More tokens than the output slice holds.
    OutputTooSmall,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::BadMagic => "not this model's file format",
            Self::Version => "unsupported model file version",
            Self::Truncated => "truncated model file",
            Self::Unaligned => "model blob must be 4-byte aligned",
            Self::Shape => "model record has the wrong shape",
            Self::Unsupported => "unsupported hyperparameters",
            Self::ScratchTooSmall => "scratch smaller than scratch_len()",
            Self::OutputTooSmall => "token output slice too small",
        };
        f.write_str(s)
    }
}

/// Reads part of a model file that is not memory-mapped: external NAND, eMMC
/// or an SD card. Models too large for RAM keep a resident part in memory and
/// page each layer's weights through this into a caller-owned buffer.
pub trait Pager {
    /// Fill `dst` from file offset `offset`.
    fn read(&mut self, offset: usize, dst: &mut [u8]) -> Result<(), Error>;
}

/// A [`Pager`] over a file that is fully in memory (memory-mapped flash, or
/// the host).
pub struct SlicePager<'a>(pub &'a [u8]);

impl Pager for SlicePager<'_> {
    fn read(&mut self, offset: usize, dst: &mut [u8]) -> Result<(), Error> {
        let src = offset
            .checked_add(dst.len())
            .and_then(|end| self.0.get(offset..end))
            .ok_or(Error::Truncated)?;
        dst.copy_from_slice(src);
        Ok(())
    }
}

/// End of the next piece of `pcm` from `start`, at most `max_len` samples
/// long: the whole rest if it fits, else the middle of the quietest
/// `frame`-sample window in the second half of the allowed span. Splits long
/// audio at pauses for models whose scratch grows with the utterance, the way
/// a VAD would, and gives the same cuts on every target.
pub fn next_cut<P: Pcm + ?Sized>(pcm: &P, start: usize, max_len: usize, frame: usize) -> usize {
    let n = pcm.len();
    if n - start <= max_len || max_len < 2 * frame {
        return n.min(start + max_len.max(1));
    }
    let (lo, hi) = (start + max_len / 2, start + max_len - frame);
    let step = (frame / 2).max(1);
    let (mut best, mut cut) = (f32::INFINITY, hi + frame / 2);
    let mut i = lo;
    while i <= hi {
        let mut e = 0f32;
        for j in i..i + frame {
            let v = pcm.at(j);
            e += v * v;
        }
        if e < best {
            (best, cut) = (e, i + frame / 2);
        }
        i += step;
    }
    cut
}

/// Mono PCM at the model's sample rate.
pub trait Pcm {
    /// Sample count.
    fn len(&self) -> usize;
    /// Sample `i` in `[-1, 1]`.
    fn at(&self, i: usize) -> f32;
    /// No samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Pcm for [f32] {
    fn len(&self) -> usize {
        <[f32]>::len(self)
    }
    #[inline]
    fn at(&self, i: usize) -> f32 {
        self[i]
    }
}

/// 16-bit PCM straight from a microphone; `/ 32768` is exact in f32.
impl Pcm for [i16] {
    fn len(&self) -> usize {
        <[i16]>::len(self)
    }
    #[inline]
    fn at(&self, i: usize) -> f32 {
        self[i] as f32 * (1.0 / 32768.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_cut_lands_in_the_quiet_part() {
        // Loud, then silent at 1200..1600, then loud.
        let pcm: [f32; 3000] =
            core::array::from_fn(|i| if (1200..1600).contains(&i) { 0.0 } else { 0.5 });
        let cut = next_cut(&pcm[..], 0, 2000, 200);
        assert!((1200..1600).contains(&cut), "cut {cut}");
        assert_eq!(next_cut(&pcm[..], cut, 2000, 200), 3000);
    }

    #[test]
    fn slice_pager_bounds() {
        let data = [1u8, 2, 3, 4];
        let mut p = SlicePager(&data);
        let mut dst = [0u8; 2];
        p.read(2, &mut dst).unwrap();
        assert_eq!(dst, [3, 4]);
        assert_eq!(p.read(3, &mut dst), Err(Error::Truncated));
        assert_eq!(p.read(usize::MAX, &mut dst), Err(Error::Truncated));
    }
}
