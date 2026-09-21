// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sound host byte-buffer reinterpretation.
//!
//! The host I/O boundary (`set_param_typed`, `run_typed`, every backend's
//! widen helper) hands typed tensor data around as `&[u8]` and wants to read
//! it back as `&[f32]` / `&[f16]` / `&[i64]`. The obvious spelling
//!
//! ```ignore
//! let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
//! ```
//!
//! is **undefined behaviour** unless `data` happens to be 4-byte aligned, and
//! nothing about `&[u8]` guarantees that: a subslice like `&buf[1..]`, or a
//! tensor at an odd offset inside an mmap, is 1-aligned. A `len % 4 == 0`
//! check does not imply pointer alignment. Miri reports
//! `constructing invalid value of type &[f32]: encountered an unaligned
//! reference`, and a sufficiently clever LLVM is entitled to emit an aligned
//! vector load against the alignment it was promised.
//!
//! [`decode_le`] keeps the zero-copy borrow when the pointer *is* aligned (the
//! common case: a whole `Vec<u8>` straight from the allocator, so the big
//! weight uploads still cost one `memcpy`) and falls back to an elementwise
//! little-endian decode when it is not. Callers that hold the allocation
//! themselves and must hand out a borrowed `&[f32]` can use [`AlignedBytes`],
//! which is over-aligned by construction so the reinterpret is provably sound
//! rather than allocator luck.
//!
//! Byte order is little-endian, matching the `to_le_bytes` convention the rest
//! of the host I/O surface already writes with. Every target RLX builds for is
//! little-endian, so the fast path and the fallback agree bit-for-bit.

use std::borrow::Cow;

/// A scalar that can be decoded from a fixed-width little-endian byte run.
///
/// # Safety
///
/// Implementors must be plain-old-data: no padding bytes, no uninhabited or
/// otherwise invalid bit patterns, and `size_of::<Self>() == Self::SIZE`.
/// [`decode_le`]'s fast path reinterprets raw bytes as `Self` after checking
/// alignment, which is only sound when *every* `SIZE`-byte pattern is a valid
/// `Self`. This rules out `bool`, `char`, references, and enums with niches.
pub unsafe trait LeBytes: Copy {
    /// Width of one element in bytes. Must equal `size_of::<Self>()`.
    const SIZE: usize;
    /// Decode one element from exactly `SIZE` little-endian bytes.
    ///
    /// # Panics
    /// Panics if `bytes.len() != Self::SIZE`.
    fn from_le_slice(bytes: &[u8]) -> Self;
}

macro_rules! impl_le_bytes {
    ($($t:ty),* $(,)?) => {$(
        // SAFETY: every primitive integer/float here is plain-old-data — no
        // padding, and every bit pattern of `size_of::<$t>()` bytes names a
        // valid value (floats include NaN payloads and signalling NaNs, which
        // are valid `$t` values even though they are not useful numbers).
        unsafe impl LeBytes for $t {
            const SIZE: usize = std::mem::size_of::<$t>();
            #[inline]
            fn from_le_slice(bytes: &[u8]) -> Self {
                <$t>::from_le_bytes(bytes.try_into().expect("from_le_slice: wrong width"))
            }
        }
    )*};
}

impl_le_bytes!(i8, u8, i16, u16, i32, u32, i64, u64, f32, f64);

/// Read `data` as `&[T]` without assuming it is aligned for `T`.
///
/// Returns `Cow::Borrowed` (zero-copy) when `data`'s pointer is already
/// aligned for `T`, and `Cow::Owned` (elementwise little-endian decode) when
/// it is not. A trailing partial element is dropped, matching the
/// `data.len() / size_of::<T>()` element count the raw reinterpret used.
///
/// `half::f16` / `half::bf16` are `repr(transparent)` over `u16`, so decode
/// those as `u16` and map through `from_bits` — the borrow stays zero-copy and
/// `rlx-ir` keeps its dependency-free root position.
#[inline]
pub fn decode_le<T: LeBytes>(data: &[u8]) -> Cow<'_, [T]> {
    let n = data.len() / T::SIZE;
    // `align_offset` is the provenance-friendly spelling of the alignment
    // test, and degrades to the (still correct, merely slower) decode path
    // under Miri's symbolic alignment checking rather than lying.
    if data.as_ptr().align_offset(std::mem::align_of::<T>()) == 0 {
        // SAFETY: the pointer is aligned for `T` (just checked) and
        // `n * T::SIZE <= data.len()` bytes are initialized and live inside a
        // single allocation for the lifetime of the borrow. `T: LeBytes`
        // guarantees every such byte pattern is a valid `T`. Note this holds
        // for `n == 0` too — an aligned pointer stays valid for an empty
        // slice, and a misaligned one takes the owned path below.
        return Cow::Borrowed(unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<T>(), n) });
    }
    Cow::Owned(
        data.chunks_exact(T::SIZE)
            .map(|c| T::from_le_slice(c))
            .collect(),
    )
}

/// [`decode_le`] into an owned `Vec<T>`. Costs one `memcpy` on the aligned
/// path and nothing extra on the unaligned one.
#[inline]
pub fn decode_le_vec<T: LeBytes>(data: &[u8]) -> Vec<T> {
    decode_le::<T>(data).into_owned()
}

/// Alignment `AlignedBytes` guarantees. 64 bytes covers every scalar RLX
/// reinterprets to and matches the memory planner's slot alignment, so a
/// planner-aligned offset into an `AlignedBytes` is still aligned.
pub const ALIGNED_BYTES_ALIGN: usize = 64;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Chunk([u8; ALIGNED_BYTES_ALIGN]);

/// A byte buffer whose base pointer is 64-byte aligned.
///
/// `Vec<u8>` only promises alignment 1. The system allocator happens to return
/// 16-aligned blocks today, so reinterpreting a `Vec<u8>`'s contents as
/// `&[f32]` *works* — but it is UB by contract, Miri flags it, and the
/// "aligned to 1, but the planner aligns offsets to 64, so this is safe"
/// reasoning that used to guard these casts is wrong: aligning an **offset**
/// does nothing if the **base** is unaligned.
///
/// `AlignedBytes` fixes the base instead. It derefs to `[u8]`, so it drops in
/// wherever a `Vec<u8>` arena/backing-store was, and [`AlignedBytes::as_slice_of`]
/// hands out a borrowed `&[T]` soundly.
pub struct AlignedBytes {
    chunks: Vec<Chunk>,
    len: usize,
}

impl AlignedBytes {
    /// A zero-filled buffer of `len` bytes, 64-byte aligned.
    pub fn zeroed(len: usize) -> Self {
        let n_chunks = len.div_ceil(ALIGNED_BYTES_ALIGN);
        Self {
            chunks: vec![Chunk([0u8; ALIGNED_BYTES_ALIGN]); n_chunks],
            len,
        }
    }

    /// Copy `data` into a fresh 64-byte-aligned buffer.
    pub fn from_slice(data: &[u8]) -> Self {
        let mut out = Self::zeroed(data.len());
        out.as_mut_slice().copy_from_slice(data);
        out
    }

    /// Logical length in bytes (not the rounded-up chunk capacity).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `chunks` holds at least `len` bytes of initialized storage
        // (chunks are zero-filled on construction), and `u8` has alignment 1,
        // so any pointer is aligned for it.
        unsafe { std::slice::from_raw_parts(self.chunks.as_ptr().cast::<u8>(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, and `&mut self` makes the borrow exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.chunks.as_mut_ptr().cast::<u8>(), self.len) }
    }

    /// Borrow the whole buffer as `&[T]`, dropping any trailing partial
    /// element. Sound without an alignment check: the base is 64-aligned and
    /// `align_of::<T>() <= 64` for every `LeBytes` scalar.
    pub fn as_slice_of<T: LeBytes>(&self) -> &[T] {
        debug_assert!(std::mem::align_of::<T>() <= ALIGNED_BYTES_ALIGN);
        // SAFETY: base is 64-byte aligned by construction, which covers
        // `align_of::<T>()`; `len / T::SIZE` elements fit in initialized
        // storage; `T: LeBytes` makes every bit pattern valid.
        unsafe { std::slice::from_raw_parts(self.chunks.as_ptr().cast::<T>(), self.len / T::SIZE) }
    }

    /// Borrow the byte range `[offset, offset + len_bytes)` as `&[T]`.
    ///
    /// # Panics
    /// Panics if the range is out of bounds, or if `offset` is not a multiple
    /// of `align_of::<T>()` — an unaligned offset from an aligned base is
    /// still an unaligned address, and that is exactly the bug this type
    /// exists to prevent, so it is a loud failure rather than silent UB.
    pub fn slice_of<T: LeBytes>(&self, offset: usize, len_bytes: usize) -> &[T] {
        let bytes = &self.as_slice()[offset..offset + len_bytes];
        assert_eq!(
            offset % std::mem::align_of::<T>(),
            0,
            "AlignedBytes::slice_of: offset {offset} is not {}-aligned",
            std::mem::align_of::<T>()
        );
        // SAFETY: base is 64-aligned, `offset` is a multiple of
        // `align_of::<T>()` (asserted) and `align_of::<T>() <= 64`, so
        // `base + offset` is aligned for `T`; the range is in bounds (the
        // index above would have panicked otherwise); `T: LeBytes`.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), len_bytes / T::SIZE) }
    }

    /// Mutable [`AlignedBytes::slice_of`].
    ///
    /// # Panics
    /// Same conditions as [`AlignedBytes::slice_of`].
    pub fn slice_of_mut<T: LeBytes>(&mut self, offset: usize, len_bytes: usize) -> &mut [T] {
        assert_eq!(
            offset % std::mem::align_of::<T>(),
            0,
            "AlignedBytes::slice_of_mut: offset {offset} is not {}-aligned",
            std::mem::align_of::<T>()
        );
        let bytes = &mut self.as_mut_slice()[offset..offset + len_bytes];
        // SAFETY: as `slice_of`, plus `&mut self` makes the borrow exclusive.
        unsafe {
            std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<T>(), len_bytes / T::SIZE)
        }
    }
}

impl Clone for AlignedBytes {
    fn clone(&self) -> Self {
        Self {
            chunks: self.chunks.clone(),
            len: self.len,
        }
    }
}

impl std::fmt::Debug for AlignedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBytes")
            .field("len", &self.len)
            .finish()
    }
}

impl std::ops::Deref for AlignedBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for AlignedBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl From<&[u8]> for AlignedBytes {
    fn from(v: &[u8]) -> Self {
        Self::from_slice(v)
    }
}

impl From<Vec<u8>> for AlignedBytes {
    fn from(v: Vec<u8>) -> Self {
        Self::from_slice(&v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `&[u8]` that is deliberately not 4-aligned is the case the raw
    /// `from_raw_parts` cast got wrong. Both paths must agree.
    #[test]
    fn decode_le_agrees_on_aligned_and_misaligned_input() {
        let vals: Vec<f32> = (0..64).map(|i| i as f32 * 0.5 - 3.0).collect();
        let n_bytes = vals.len() * 4;

        // Skews 0..4 cover every residue class mod 4 relative to the (in
        // practice 16-aligned) Vec base, so at least three are unaligned —
        // exactly the input the raw `from_raw_parts` cast got wrong.
        for skew in 0..4usize {
            let mut bytes = vec![0u8; skew];
            for v in &vals {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let got = decode_le::<f32>(&bytes[skew..skew + n_bytes]);
            assert_eq!(got.as_ref(), vals.as_slice(), "skew {skew}");
        }
    }

    #[test]
    fn decode_le_borrows_when_aligned_and_copies_when_not() {
        let mut bytes = vec![0u8; 1];
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        // `&bytes[1..5]` is 1 byte past a 16-aligned allocation → unaligned.
        assert!(matches!(decode_le::<f32>(&bytes[1..5]), Cow::Owned(_)));
        let aligned: Vec<u8> = 2.0f32.to_le_bytes().to_vec();
        assert!(matches!(decode_le::<f32>(&aligned), Cow::Borrowed(_)));
    }

    #[test]
    fn decode_le_drops_trailing_partial_element() {
        let bytes = [1u8, 0, 0, 0, 7, 7];
        assert_eq!(decode_le::<f32>(&bytes).len(), 1);
        assert_eq!(decode_le::<f32>(&[]).len(), 0);
    }

    #[test]
    fn aligned_bytes_base_is_over_aligned() {
        for len in [0usize, 1, 7, 64, 65, 4096] {
            let b = AlignedBytes::zeroed(len);
            assert_eq!(b.len(), len);
            assert_eq!(
                b.as_slice().as_ptr() as usize % ALIGNED_BYTES_ALIGN,
                0,
                "len {len}"
            );
            assert!(b.as_slice().iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn aligned_bytes_round_trips_typed_slices() {
        let vals: Vec<f64> = vec![1.5, -2.25, 1e300];
        let mut b = AlignedBytes::zeroed(vals.len() * 8);
        b.slice_of_mut::<f64>(0, vals.len() * 8)
            .copy_from_slice(&vals);
        assert_eq!(b.slice_of::<f64>(0, vals.len() * 8), vals.as_slice());
        assert_eq!(b.as_slice_of::<f64>(), vals.as_slice());
        // Offsets stay aligned relative to the 64-aligned base.
        assert_eq!(b.slice_of::<f64>(8, 8), &vals[1..2]);
    }

    #[test]
    #[should_panic(expected = "not 8-aligned")]
    fn aligned_bytes_rejects_unaligned_offset() {
        let b = AlignedBytes::zeroed(64);
        let _ = b.slice_of::<f64>(4, 8);
    }
}
