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

//! RLXM v1 binary weights format.
//!
//! Concatenated weights + per-channel mults + biases + test image,
//! preceded by a small descriptive header. The .rs shim emitted by
//! `emit.rs` reads this file via `include_bytes!` and exposes typed
//! slices into it; nothing on the firmware side parses the header at
//! runtime — the offsets are baked into the .rs at training time.
//!
//! # Layout
//!
//! ```text
//! [0..4]   magic                 b"RLXM"
//! [4..6]   format_version u16    = 1
//! [6..8]   arch_id        u16    = 1 (TinyConv)
//! [8..12]  fp32_accuracy  f32    test accuracy at training
//! [12..14] n_tensors      u16    = 10
//! [14..16] reserved
//!
//! [16 .. 16 + 16*n_tensors]  tensor descriptors, 16 B each:
//!     [0..2]   tensor_id  u16    enum below
//!     [2..3]   dtype      u8     0=i8, 1=i32, 2=f32
//!     [3..4]   reserved
//!     [4..8]   byte_off   u32    relative to file start
//!     [8..12]  n_elems    u32
//!     [12..16] reserved
//!
//! padded to 4-byte boundary, then:
//!     f32 region    (mults — must be 4-aligned for the .rs cast)
//!     i32 region    (biases — must be 4-aligned)
//!     i8  region    (weights, test image — no alignment needed)
//! ```
//!
//! Why "f32 first" then "i32" then "i8"? The firmware imports each
//! tensor as `unsafe { from_raw_parts(BLOB.as_ptr().add(off) as *const T, n) }`,
//! and ARMv7E-M faults on a misaligned `LDR` (4-byte). Putting the
//! f32 / i32 regions ahead of the i8 region — and aligning each to
//! its element size — guarantees every slice cast lands on a valid
//! address regardless of how many elements live in each region.
//!
//! Tensor IDs (stable, additive — never renumber):
//!   0 CONV1_W, 1 CONV1_B, 2 CONV2_W, 3 CONV2_B,
//!   4 FC_W,    5 FC_B,
//!   6 CONV1_MULT, 7 CONV2_MULT, 8 FC_MULT,
//!   9 TEST_IMAGE
//!
//! DType codes:
//!   0 i8     (1 byte per element)
//!   1 i32    (4 bytes per element)
//!   2 f32    (4 bytes per element)
//!   3 i4_pkd (2 logical elements per byte, low nibble first)
//!   4 i2_pkd (4 logical elements per byte, LSB first)
//!
//! `n_elems` in a descriptor is always the **logical** count for
//! packed weights — the firmware reads via `read_weight(w, idx, bits)`
//! which translates logical idx → byte + lane. The descriptor's
//! storage length is `ceil(n_elems * bits / 8)` bytes.

use crate::quant::QuantizedModel;

const MAGIC: &[u8; 4] = b"RLXM";
const FORMAT_VERSION: u16 = 1;
const ARCH_TINYCONV: u16 = 1;
const N_TENSORS: u16 = 10;
const HEADER_LEN: usize = 16;
const DESC_LEN: usize = 16;
const TABLE_LEN: usize = HEADER_LEN + (N_TENSORS as usize) * DESC_LEN;

#[repr(u8)]
#[derive(Clone, Copy)]
enum DType {
    I8 = 0,
    I32 = 1,
    F32 = 2,
    I4Pkd = 3,
    I2Pkd = 4,
}

#[derive(Clone, Copy)]
pub struct SliceLoc {
    /// Byte offset into the blob.
    pub byte_off: usize,
    /// Logical element count — for packed dtypes this is the number
    /// of weights the firmware sees through `read_weight`, not the
    /// byte length of the stored region.
    pub n_elems: usize,
    /// Byte length actually occupied in the blob:
    /// `ceil(n_elems * bits / 8)`. Equals `n_elems` for i8.
    pub byte_len: usize,
}

/// Per-tensor offsets/lengths, returned from [`encode`] so the .rs
/// shim can declare matching slice references.
pub struct Layout {
    pub conv1_w: SliceLoc,
    pub conv1_b: SliceLoc,
    pub conv2_w: SliceLoc,
    pub conv2_b: SliceLoc,
    pub fc_w: SliceLoc,
    pub fc_b: SliceLoc,
    pub conv1_mult: SliceLoc,
    pub conv2_mult: SliceLoc,
    pub fc_mult: SliceLoc,
    pub test_image: SliceLoc,
    pub total_len: usize,
}

impl Layout {
    /// Compute the layout for `model` without actually emitting bytes.
    /// Mirrors what [`encode`] produces — `encode` calls this internally.
    pub fn for_model(m: &QuantizedModel) -> Self {
        // Logical element counts — for packed weights, this is the
        // number the firmware kernel sees through `read_weight`, not
        // the stored byte length.
        let logical_c1w = 8 * 3 * 3;
        let logical_c2w = 16 * 3 * 3 * 8;
        let logical_fcw = 10 * 400;
        Self::compute(
            logical_c1w,
            m.conv1_b.len(),
            logical_c2w,
            m.conv2_b.len(),
            logical_fcw,
            m.fc_b.len(),
            m.w1_scale.len(),
            m.w2_scale.len(),
            m.wfc_scale.len(),
            m.test_image.len(),
            m.weight_bits,
        )
    }

    fn compute(
        n_c1w: usize,
        n_c1b: usize,
        n_c2w: usize,
        n_c2b: usize,
        n_fcw: usize,
        n_fcb: usize,
        n_c1m: usize,
        n_c2m: usize,
        n_fcm: usize,
        n_ti: usize,
        bits: u8,
    ) -> Self {
        let mut o = align_up(TABLE_LEN, 4);

        // f32 region first — naturally 4-aligned for `*const f32` cast.
        let conv1_mult = SliceLoc {
            byte_off: o,
            n_elems: n_c1m,
            byte_len: n_c1m * 4,
        };
        o += n_c1m * 4;
        let conv2_mult = SliceLoc {
            byte_off: o,
            n_elems: n_c2m,
            byte_len: n_c2m * 4,
        };
        o += n_c2m * 4;
        let fc_mult = SliceLoc {
            byte_off: o,
            n_elems: n_fcm,
            byte_len: n_fcm * 4,
        };
        o += n_fcm * 4;

        // i32 region — also 4-aligned.
        debug_assert_eq!(o % 4, 0, "f32 region must end 4-aligned for i32 region");
        let conv1_b = SliceLoc {
            byte_off: o,
            n_elems: n_c1b,
            byte_len: n_c1b * 4,
        };
        o += n_c1b * 4;
        let conv2_b = SliceLoc {
            byte_off: o,
            n_elems: n_c2b,
            byte_len: n_c2b * 4,
        };
        o += n_c2b * 4;
        let fc_b = SliceLoc {
            byte_off: o,
            n_elems: n_fcb,
            byte_len: n_fcb * 4,
        };
        o += n_fcb * 4;

        // i8 / packed region — no alignment requirement for &[i8].
        // Logical n_elems stays as-passed; byte_len is ceil(n*bits/8).
        let bytes = |n: usize| (n * (bits as usize)).div_ceil(8);
        let conv1_w = SliceLoc {
            byte_off: o,
            n_elems: n_c1w,
            byte_len: bytes(n_c1w),
        };
        o += conv1_w.byte_len;
        let conv2_w = SliceLoc {
            byte_off: o,
            n_elems: n_c2w,
            byte_len: bytes(n_c2w),
        };
        o += conv2_w.byte_len;
        let fc_w = SliceLoc {
            byte_off: o,
            n_elems: n_fcw,
            byte_len: bytes(n_fcw),
        };
        o += fc_w.byte_len;
        // Test image is always raw i8 (activations are never packed).
        let test_image = SliceLoc {
            byte_off: o,
            n_elems: n_ti,
            byte_len: n_ti,
        };
        o += n_ti;

        Self {
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            fc_w,
            fc_b,
            conv1_mult,
            conv2_mult,
            fc_mult,
            test_image,
            total_len: o,
        }
    }
}

#[inline]
fn align_up(off: usize, align: usize) -> usize {
    (off + align - 1) & !(align - 1)
}

/// Encode `model` as RLXM v1 bytes. The returned vec is exactly
/// `Layout::total_len` long.
pub fn encode(model: &QuantizedModel) -> Vec<u8> {
    let layout = Layout::for_model(model);
    let mut out = vec![0u8; layout.total_len];

    // ── Header ────────────────────────────────────────────────
    out[0..4].copy_from_slice(MAGIC);
    out[4..6].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    out[6..8].copy_from_slice(&ARCH_TINYCONV.to_le_bytes());
    out[8..12].copy_from_slice(&(model.fp32_test_accuracy as f32).to_le_bytes());
    out[12..14].copy_from_slice(&N_TENSORS.to_le_bytes());

    // Pick the right packed-weight dtype code from `model.weight_bits`.
    let weight_dtype = match model.weight_bits {
        8 => DType::I8,
        4 => DType::I4Pkd,
        2 => DType::I2Pkd,
        n => panic!("unsupported weight_bits {n}"),
    };

    // ── Descriptor table ─────────────────────────────────────
    let descs = [
        (0u16, weight_dtype, layout.conv1_w),
        (1u16, DType::I32, layout.conv1_b),
        (2u16, weight_dtype, layout.conv2_w),
        (3u16, DType::I32, layout.conv2_b),
        (4u16, weight_dtype, layout.fc_w),
        (5u16, DType::I32, layout.fc_b),
        (6u16, DType::F32, layout.conv1_mult),
        (7u16, DType::F32, layout.conv2_mult),
        (8u16, DType::F32, layout.fc_mult),
        (9u16, DType::I8, layout.test_image),
    ];
    debug_assert_eq!(descs.len(), N_TENSORS as usize);
    for (i, (id, dt, sl)) in descs.iter().enumerate() {
        let base = HEADER_LEN + i * DESC_LEN;
        out[base..base + 2].copy_from_slice(&id.to_le_bytes());
        out[base + 2] = *dt as u8;
        out[base + 4..base + 8].copy_from_slice(&(sl.byte_off as u32).to_le_bytes());
        out[base + 8..base + 12].copy_from_slice(&(sl.n_elems as u32).to_le_bytes());
    }

    // ── Tensor data ──────────────────────────────────────────
    write_f32(&mut out, layout.conv1_mult.byte_off, &model.conv1_mult());
    write_f32(&mut out, layout.conv2_mult.byte_off, &model.conv2_mult());
    write_f32(&mut out, layout.fc_mult.byte_off, &model.fc_mult());

    write_i32(&mut out, layout.conv1_b.byte_off, &model.conv1_b);
    write_i32(&mut out, layout.conv2_b.byte_off, &model.conv2_b);
    write_i32(&mut out, layout.fc_b.byte_off, &model.fc_b);

    write_i8(&mut out, layout.conv1_w.byte_off, &model.conv1_w);
    write_i8(&mut out, layout.conv2_w.byte_off, &model.conv2_w);
    write_i8(&mut out, layout.fc_w.byte_off, &model.fc_w);
    write_i8(&mut out, layout.test_image.byte_off, &model.test_image);

    out
}

fn write_f32(out: &mut [u8], off: usize, src: &[f32]) {
    debug_assert_eq!(off % 4, 0);
    for (i, &v) in src.iter().enumerate() {
        out[off + i * 4..off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
}

fn write_i32(out: &mut [u8], off: usize, src: &[i32]) {
    debug_assert_eq!(off % 4, 0);
    for (i, &v) in src.iter().enumerate() {
        out[off + i * 4..off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
}

fn write_i8(out: &mut [u8], off: usize, src: &[i8]) {
    for (i, &v) in src.iter().enumerate() {
        out[off + i] = v as u8;
    }
}
