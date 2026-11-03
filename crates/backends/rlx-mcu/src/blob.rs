// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model files for MCUs: one flat, 4-byte-aligned, little-endian blob read
//! in place (memory-mapped flash, or PSRAM it was copied into from NAND or
//! an SD card) without copying.
//!
//! A file starts with a magic word and a version, both chosen by the model.
//! Every record is a sequence of 32-bit words, so any record starts aligned:
//!
//! * `u32s` — `len: u32`, then `len` words
//! * `f32s` — `len: u32`, then `len` f32 values
//! * `q8`   — `rows: u32, cols: u32`, `rows` f32 scales, then `rows * cols`
//!   i8 values padded to a word. Row `r` dequantizes as `q[r][c] * scale[r]`
//!   (symmetric, one scale per output channel).
//! * `mat`  — a tag word, then the matrix: `8` a `q8` record; `4` a `q4`
//!   record (`rows, cols, group`, `rows * cols / group` f32 scales, then
//!   4-bit codes in `-8..=7`, two per byte, low nibble first, padded to a
//!   word); `9` a `q8g` record (int8 with one scale per `group` values of a
//!   row, same header as `q4`). Lets the exporter pick per matrix.
//!
//! The order of the records is fixed per model and is defined once by its
//! `parse` on the read side and its exporter on the write side; there is no
//! key table.

use crate::Error;

/// An int8 matrix with one f32 scale per row.
#[derive(Clone, Copy)]
pub struct Q8<'a> {
    /// Output channels.
    pub rows: usize,
    /// Input channels.
    pub cols: usize,
    /// Per-row dequantization scales.
    pub scales: &'a [f32],
    /// Row-major `[rows, cols]` values.
    pub data: &'a [i8],
}

impl<'a> Q8<'a> {
    /// Row `r` as an int8 slice.
    #[inline]
    pub fn row(&self, r: usize) -> &'a [i8] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

/// A 4-bit matrix with one f32 scale per `group` consecutive values of a row.
#[derive(Clone, Copy)]
pub struct Q4<'a> {
    /// Output channels.
    pub rows: usize,
    /// Input channels (a multiple of `group`).
    pub cols: usize,
    /// Values sharing one scale (even, divides `cols`).
    pub group: usize,
    /// `rows * cols / group` scales, row-major.
    pub scales: &'a [f32],
    /// Packed codes, two per byte, low nibble first.
    pub data: &'a [u8],
}

impl<'a> Q4<'a> {
    /// Row `r`: its packed codes and its group scales.
    #[inline]
    pub fn row(&self, r: usize) -> (&'a [u8], &'a [f32]) {
        let g = self.cols / self.group;
        (
            &self.data[r * self.cols / 2..(r + 1) * self.cols / 2],
            &self.scales[r * g..(r + 1) * g],
        )
    }

    /// Code at column `c` of a packed row.
    #[inline]
    pub fn code(packed: &[u8], c: usize) -> i8 {
        let b = packed[c / 2];
        let n = if c.is_multiple_of(2) {
            b << 4
        } else {
            b & 0xf0
        };
        (n as i8) >> 4
    }
}

/// An int8 matrix with one f32 scale per `group` consecutive values of a
/// row — for the few matrices where one scale per row loses too much (the
/// frontend convs of Moonshine, whose error the encoder amplifies).
#[derive(Clone, Copy)]
pub struct Q8G<'a> {
    /// Output channels.
    pub rows: usize,
    /// Input channels (a multiple of `group`).
    pub cols: usize,
    /// Values sharing one scale.
    pub group: usize,
    /// `rows * cols / group` scales, row-major.
    pub scales: &'a [f32],
    /// Row-major `[rows, cols]` values.
    pub data: &'a [i8],
}

impl<'a> Q8G<'a> {
    /// Row `r`: its values and its group scales.
    #[inline]
    pub fn row(&self, r: usize) -> (&'a [i8], &'a [f32]) {
        let g = self.cols / self.group;
        (
            &self.data[r * self.cols..(r + 1) * self.cols],
            &self.scales[r * g..(r + 1) * g],
        )
    }
}

/// A weight matrix of any of the stored widths.
#[derive(Clone, Copy)]
pub enum Mat<'a> {
    /// One int8 per value, one scale per row.
    Q8(Q8<'a>),
    /// One nibble per value, group scales.
    Q4(Q4<'a>),
    /// One int8 per value, group scales.
    Q8G(Q8G<'a>),
}

impl Mat<'_> {
    /// Output channels.
    pub fn rows(&self) -> usize {
        match self {
            Mat::Q8(m) => m.rows,
            Mat::Q4(m) => m.rows,
            Mat::Q8G(m) => m.rows,
        }
    }
    /// Input channels.
    pub fn cols(&self) -> usize {
        match self {
            Mat::Q8(m) => m.cols,
            Mat::Q4(m) => m.cols,
            Mat::Q8G(m) => m.cols,
        }
    }
    /// Bytes the matrix occupies in the file (codes plus scales).
    pub fn bytes(&self) -> usize {
        match self {
            Mat::Q8(m) => m.data.len() + 4 * m.scales.len(),
            Mat::Q4(m) => m.data.len() + 4 * m.scales.len(),
            Mat::Q8G(m) => m.data.len() + 4 * m.scales.len(),
        }
    }
}

/// Most groups a `q8g` row may have (the activation scales of one row live
/// on the stack).
pub const MAX_GROUPS: usize = 512;

/// Sequential reader over a word-aligned blob.
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// A reader at the start of `bytes`, which must be 4-byte aligned.
    pub fn new(bytes: &'a [u8]) -> Result<Self, Error> {
        if !(bytes.as_ptr() as usize).is_multiple_of(4) {
            return Err(Error::Unaligned);
        }
        Ok(Self { bytes, pos: 0 })
    }

    /// A reader at the start of `bytes` that checks the header words.
    pub fn with_header(bytes: &'a [u8], magic: u32, version: u32) -> Result<Self, Error> {
        let mut c = Self::new(bytes)?;
        if c.u32()? != magic {
            return Err(Error::BadMagic);
        }
        if c.u32()? != version {
            return Err(Error::Version);
        }
        Ok(c)
    }

    pub fn at(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, pos }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).ok_or(Error::Truncated)?;
        let s = self.bytes.get(self.pos..end).ok_or(Error::Truncated)?;
        // Records are word-padded, so the cursor never leaves alignment.
        self.pos = end.next_multiple_of(4);
        Ok(s)
    }

    pub fn u32(&mut self) -> Result<u32, Error> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn usize(&mut self) -> Result<usize, Error> {
        Ok(self.u32()? as usize)
    }

    fn words<T>(&mut self, n: usize) -> Result<&'a [T], Error> {
        let b = self.take(n.checked_mul(4).ok_or(Error::Truncated)?)?;
        // SAFETY: `b` is 4-aligned (the blob start is checked in `new` and
        // every record is word-padded), spans exactly `n` 4-byte values, and
        // `T` is only ever `u32` or `f32`, for which every bit pattern is
        // valid. The file is little-endian and so are all supported targets.
        Ok(unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<T>(), n) })
    }

    pub fn u32s(&mut self) -> Result<&'a [u32], Error> {
        let n = self.usize()?;
        self.words(n)
    }

    pub fn f32s(&mut self) -> Result<&'a [f32], Error> {
        let n = self.usize()?;
        self.words(n)
    }

    /// An `f32s` record that must hold exactly `n` values.
    pub fn f32s_n(&mut self, n: usize) -> Result<&'a [f32], Error> {
        let v = self.f32s()?;
        if v.len() != n {
            return Err(Error::Shape);
        }
        Ok(v)
    }

    pub fn q8(&mut self, rows: usize, cols: usize) -> Result<Q8<'a>, Error> {
        let (r, c) = (self.usize()?, self.usize()?);
        if (r, c) != (rows, cols) {
            return Err(Error::Shape);
        }
        let scales = self.words::<f32>(r)?;
        let b = self.take(r * c)?;
        // SAFETY: i8 and u8 have the same size, alignment and validity.
        let data = unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<i8>(), b.len()) };
        Ok(Q8 {
            rows: r,
            cols: c,
            scales,
            data,
        })
    }

    /// A `q4` record that must be `rows × cols`.
    pub fn q4(&mut self, rows: usize, cols: usize) -> Result<Q4<'a>, Error> {
        let (r, c, g) = (self.usize()?, self.usize()?, self.usize()?);
        if (r, c) != (rows, cols) || g == 0 || !g.is_multiple_of(2) || !c.is_multiple_of(g) {
            return Err(Error::Shape);
        }
        let scales = self.words::<f32>(r * c / g)?;
        let data = self.take(r * c / 2)?;
        Ok(Q4 {
            rows: r,
            cols: c,
            group: g,
            scales,
            data,
        })
    }

    /// A `q8g` record that must be `rows × cols`.
    pub fn q8g(&mut self, rows: usize, cols: usize) -> Result<Q8G<'a>, Error> {
        let (r, c, g) = (self.usize()?, self.usize()?, self.usize()?);
        if (r, c) != (rows, cols) || g == 0 || !c.is_multiple_of(g) || c / g > MAX_GROUPS {
            return Err(Error::Shape);
        }
        let scales = self.words::<f32>(r * c / g)?;
        let b = self.take(r * c)?;
        // SAFETY: i8 and u8 have the same size, alignment and validity.
        let data = unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<i8>(), b.len()) };
        Ok(Q8G {
            rows: r,
            cols: c,
            group: g,
            scales,
            data,
        })
    }

    /// A `mat` record (any width) that must be `rows × cols`.
    pub fn mat(&mut self, rows: usize, cols: usize) -> Result<Mat<'a>, Error> {
        match self.u32()? {
            8 => Ok(Mat::Q8(self.q8(rows, cols)?)),
            4 => Ok(Mat::Q4(self.q4(rows, cols)?)),
            9 => Ok(Mat::Q8G(self.q8g(rows, cols)?)),
            _ => Err(Error::Shape),
        }
    }

    /// A length-prefixed byte record.
    pub fn bytes(&mut self) -> Result<&'a [u8], Error> {
        let n = self.usize()?;
        self.take(n)
    }
}

/// Round to nearest, ties away from zero, saturating to `[-127, 127]`.
///
/// Written out rather than `roundf` so it costs one add and one convert on
/// a core without hardware rounding, and so host and MCU agree bit for bit.
#[inline]
pub fn quant(x: f32) -> i8 {
    let r = if x >= 0.0 { x + 0.5 } else { x - 0.5 };
    // `as` saturates and truncates toward zero, which with the ±0.5 above is
    // round-half-away.
    (r as i32).clamp(-127, 127) as i8
}

/// Symmetric per-row int8 quantization of `x` into `q`. Returns the scale
/// (`x ≈ q * scale`), or `0.0` for an all-zero row.
#[inline]
pub fn quantize_row(x: &[f32], q: &mut [i8]) -> f32 {
    let mut m = 0.0f32;
    for &v in x {
        let a = v.abs();
        if a > m {
            m = a;
        }
    }
    if m == 0.0 {
        q[..x.len()].fill(0);
        return 0.0;
    }
    let inv = 127.0 / m;
    for (d, &v) in q.iter_mut().zip(x) {
        *d = quant(v * inv);
    }
    m / 127.0
}

#[cfg(feature = "std")]
pub use writer::Writer;

#[cfg(feature = "std")]
mod writer {
    use super::quantize_row;

    /// Builds a blob record by record (host side only).
    #[derive(Default)]
    pub struct Writer {
        buf: Vec<u8>,
    }

    impl Writer {
        /// A writer with the file header (`magic`, `version`) emitted.
        pub fn new(magic: u32, version: u32) -> Self {
            let mut w = Self::default();
            w.u32(magic);
            w.u32(version);
            w
        }

        /// Bytes written so far.
        pub fn len(&self) -> usize {
            self.buf.len()
        }

        /// Nothing written yet.
        pub fn is_empty(&self) -> bool {
            self.buf.is_empty()
        }

        fn pad(&mut self) {
            while !self.buf.len().is_multiple_of(4) {
                self.buf.push(0);
            }
        }

        /// One bare word (header fields).
        pub fn u32(&mut self, v: u32) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }

        /// A `u32s` record.
        pub fn u32s(&mut self, v: &[u32]) {
            self.u32(v.len() as u32);
            for &x in v {
                self.u32(x);
            }
        }

        /// An `f32s` record.
        pub fn f32s(&mut self, v: &[f32]) {
            self.u32(v.len() as u32);
            for &x in v {
                self.buf.extend_from_slice(&x.to_le_bytes());
            }
        }

        /// A `q8` record: quantizes the row-major `[rows, cols]` matrix `w`
        /// with one symmetric scale per row.
        pub fn q8(&mut self, w: &[f32], rows: usize, cols: usize) {
            assert_eq!(w.len(), rows * cols, "q8 {rows}x{cols}");
            self.u32(rows as u32);
            self.u32(cols as u32);
            let mut q = vec![0i8; rows * cols];
            let mut scales = vec![0f32; rows];
            for r in 0..rows {
                scales[r] = quantize_row(&w[r * cols..(r + 1) * cols], &mut q[r * cols..]);
            }
            for s in scales {
                self.buf.extend_from_slice(&s.to_le_bytes());
            }
            self.buf.extend(q.iter().map(|&v| v as u8));
            self.pad();
        }

        /// A `q4` record: symmetric 4-bit codes in `-8..=7` (scale = max / 7)
        /// with one scale per `group` values of a row.
        pub fn q4(&mut self, w: &[f32], rows: usize, cols: usize, group: usize) {
            assert_eq!(w.len(), rows * cols, "q4 {rows}x{cols}");
            assert!(
                group.is_multiple_of(2) && cols.is_multiple_of(group),
                "q4 group {group}"
            );
            self.u32(rows as u32);
            self.u32(cols as u32);
            self.u32(group as u32);
            let mut codes = vec![0u8; rows * cols / 2];
            let mut scales = Vec::with_capacity(rows * cols / group);
            for (gi, chunk) in w.chunks(group).enumerate() {
                let m = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
                let s = if m == 0.0 { 0.0 } else { m / 7.0 };
                scales.push(s);
                for (j, &v) in chunk.iter().enumerate() {
                    let q = if s == 0.0 {
                        0
                    } else {
                        (v / s).round().clamp(-8.0, 7.0) as i8
                    };
                    let idx = gi * group + j;
                    let nib = (q as u8) & 0xf;
                    codes[idx / 2] |= if idx.is_multiple_of(2) { nib } else { nib << 4 };
                }
            }
            for s in scales {
                self.buf.extend_from_slice(&s.to_le_bytes());
            }
            self.buf.extend_from_slice(&codes);
            self.pad();
        }

        /// A `q8g` record: symmetric int8 with one scale per `group` values.
        pub fn q8g(&mut self, w: &[f32], rows: usize, cols: usize, group: usize) {
            assert_eq!(w.len(), rows * cols, "q8g {rows}x{cols}");
            assert!(group > 0 && cols.is_multiple_of(group), "q8g group {group}");
            self.u32(rows as u32);
            self.u32(cols as u32);
            self.u32(group as u32);
            let mut q = vec![0i8; rows * cols];
            let mut scales = Vec::with_capacity(rows * cols / group);
            for (gi, chunk) in w.chunks(group).enumerate() {
                scales.push(quantize_row(chunk, &mut q[gi * group..(gi + 1) * group]));
            }
            for s in scales {
                self.buf.extend_from_slice(&s.to_le_bytes());
            }
            self.buf.extend(q.iter().map(|&v| v as u8));
            self.pad();
        }

        /// A `mat` record: `bits` = 8 (per-row int8), 4 (int4, `group`) or
        /// 9 (int8 with per-`group` scales).
        pub fn mat(&mut self, w: &[f32], rows: usize, cols: usize, bits: u32, group: usize) {
            self.u32(bits);
            match bits {
                8 => self.q8(w, rows, cols),
                4 => self.q4(w, rows, cols, group),
                9 => self.q8g(w, rows, cols, group),
                _ => panic!("mat bits must be 8, 4 or 9, got {bits}"),
            }
        }

        /// A length-prefixed byte record, word-padded.
        pub fn bytes(&mut self, b: &[u8]) {
            self.u32(b.len() as u32);
            self.buf.extend_from_slice(b);
            self.pad();
        }

        /// The finished blob.
        pub fn finish(self) -> Vec<u8> {
            self.buf
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_rounds_half_away_and_saturates() {
        assert_eq!(quant(0.5), 1);
        assert_eq!(quant(-0.5), -1);
        assert_eq!(quant(0.49), 0);
        assert_eq!(quant(1e9), 127);
        assert_eq!(quant(-1e9), -127);
    }

    #[cfg(feature = "std")]
    #[test]
    fn q4_round_trips_through_a_blob() {
        let w: Vec<f32> = (0..64)
            .map(|i| ((i * 37 % 29) as f32 - 14.0) * 0.1)
            .collect();
        let mut wr = Writer::new(0x1234, 1);
        wr.mat(&w, 2, 32, 4, 16);
        let blob = wr.finish();
        let words: Vec<u32> = blob
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let bytes = unsafe { core::slice::from_raw_parts(words.as_ptr().cast::<u8>(), blob.len()) };
        let mut c = Cursor::with_header(bytes, 0x1234, 1).unwrap();
        let Mat::Q4(m) = c.mat(2, 32).unwrap() else {
            panic!("expected q4")
        };
        for r in 0..2 {
            let (codes, scales) = m.row(r);
            for col in 0..32 {
                let v = Q4::code(codes, col) as f32 * scales[col / 16];
                let s = scales[col / 16];
                assert!((v - w[r * 32 + col]).abs() <= s * 0.5 + 1e-6);
            }
        }
    }

    #[test]
    fn quantize_row_round_trips() {
        let x = [0.1f32, -0.7, 0.35, 0.0];
        let mut q = [0i8; 4];
        let s = quantize_row(&x, &mut q);
        assert_eq!(q[1], -127);
        for (a, &b) in x.iter().zip(&q) {
            assert!((a - b as f32 * s).abs() <= s * 0.5 + 1e-7);
        }
        assert_eq!(quantize_row(&[0.0; 3], &mut q), 0.0);
    }
}
