// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Image resizing as vision processors do it, so a package's image path
//! can take any size: the processor's target size ([`smart_resize`], Qwen-VL's
//! rule) and an antialiased bicubic resize of RGB bytes
//! ([`resize_rgb8_bicubic`]) that gives exactly torchvision's
//! `resize(uint8, BICUBIC, antialias=True)` — what Hugging Face's fast image
//! processors run.

/// The size an image of `height × width` becomes: each side the nearest
/// multiple of `factor` (Python's `round`, ties to even), scaled down if
/// that holds more than `max_pixels` and up if fewer than `min_pixels`
/// (Qwen2-VL's `smart_resize`). `None` for an aspect past 200:1.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Option<(usize, usize)> {
    smart_resize_frames(1, 1, height, width, factor, min_pixels, max_pixels)
}

/// [`smart_resize`] for `frames` frames of a video, its pixel bounds over
/// all of them (the frames rounded up to a multiple of `temporal`) — the
/// Qwen-VL video processors' rule. One frame, `temporal` 1, is an image's.
pub fn smart_resize_frames(
    frames: usize,
    temporal: usize,
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Option<(usize, usize)> {
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if frames == 0 || height == 0 || width == 0 || h.max(w) / h.min(w) > 200.0 {
        return None;
    }
    let n = frames as f64;
    let t = frames.div_ceil(temporal.max(1)) as f64 * temporal.max(1) as f64;
    let mut hb = (h / f).round_ties_even() * f;
    let mut wb = (w / f).round_ties_even() * f;
    if t * hb * wb > max_pixels as f64 {
        let beta = (n * h * w / max_pixels as f64).sqrt();
        hb = f.max((h / beta / f).floor() * f);
        wb = f.max((w / beta / f).floor() * f);
    } else if t * hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (n * h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    Some((hb as usize, wb as usize))
}

/// The bicubic kernel (`a = -0.5`).
fn cubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// Each output index's first input index and integer weights, and their
/// precision: the antialiased (support widened by the scale) normalized
/// bicubic weights, as `int16` at the most bits the largest still fits.
fn weights(input: usize, output: usize) -> (Vec<(usize, Vec<i64>)>, u32) {
    let scale = input as f64 / output as f64;
    let widen = scale.max(1.0);
    let support = 2.0 * widen;
    let float: Vec<(usize, Vec<f64>)> = (0..output)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let mut w: Vec<f64> = (lo..hi)
                .map(|j| cubic((j as f64 - center + 0.5) / widen))
                .collect();
            let total: f64 = w.iter().sum();
            if total != 0.0 {
                w.iter_mut().for_each(|x| *x /= total);
            }
            (lo, w)
        })
        .collect();
    let top = float
        .iter()
        .flat_map(|(_, w)| w.iter().copied())
        .fold(0f64, f64::max);
    let mut bits = 0u32;
    while bits < 22 && ((0.5 + top * f64::from(1u32 << (bits + 1))) as i64) < (1 << 15) {
        bits += 1;
    }
    let one = f64::from(1u32 << bits);
    let ints = float
        .into_iter()
        .map(|(lo, w)| {
            let w = w
                .into_iter()
                .map(|x| {
                    if x < 0.0 {
                        (x * one - 0.5) as i64
                    } else {
                        (x * one + 0.5) as i64
                    }
                })
                .collect();
            (lo, w)
        })
        .collect();
    (ints, bits)
}

/// One pass along an axis of `[rows, cols, 3]` bytes: `along_cols` resizes
/// each row to `out` columns, else each column to `out` rows.
fn pass(data: &[u8], rows: usize, cols: usize, out: usize, along_cols: bool) -> Vec<u8> {
    let (input, keep) = if along_cols {
        (cols, rows)
    } else {
        (rows, cols)
    };
    let (ws, bits) = weights(input, out);
    let half = if bits > 0 { 1i64 << (bits - 1) } else { 0 };
    let (orows, ocols) = if along_cols { (rows, out) } else { (out, cols) };
    let mut res = vec![0u8; orows * ocols * 3];
    for k in 0..keep {
        for (o, (lo, w)) in ws.iter().enumerate() {
            for c in 0..3 {
                let mut acc = half;
                for (j, wt) in w.iter().enumerate() {
                    let i = lo + j;
                    let at = if along_cols {
                        (k * cols + i) * 3 + c
                    } else {
                        (i * cols + k) * 3 + c
                    };
                    acc += wt * i64::from(data[at]);
                }
                let v = (acc >> bits).clamp(0, 255) as u8;
                let at = if along_cols {
                    (k * ocols + o) * 3 + c
                } else {
                    (o * ocols + k) * 3 + c
                };
                res[at] = v;
            }
        }
    }
    res
}

/// `data` (`[height, width, 3]` RGB bytes) resized to `out_h × out_w`:
/// antialiased bicubic, the rows' pass first, each pass rounded to bytes
/// with integer weights — torchvision's `uint8` path, value for value.
pub fn resize_rgb8_bicubic(
    data: &[u8],
    height: usize,
    width: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<u8> {
    let mut x = data.to_vec();
    let mut w = width;
    if out_w != width {
        x = pass(&x, height, width, out_w, true);
        w = out_w;
    }
    if out_h != height {
        x = pass(&x, height, w, out_h, false);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_rounds_as_python_does() {
        // 375 / 32 = 11.72 → 12, 500 / 32 = 15.6 → 16.
        assert_eq!(
            smart_resize(375, 500, 32, 65536, 16777216),
            Some((384, 512))
        );
        // A tie goes to even: 80 / 32 = 2.5 → 2; then up to the minimum.
        assert_eq!(smart_resize(80, 80, 32, 0, 1 << 30), Some((64, 64)));
        assert_eq!(smart_resize(64, 64, 32, 65536, 1 << 30), Some((256, 256)));
        // Down to the maximum.
        assert_eq!(
            smart_resize(4000, 4000, 32, 0, 1024 * 1024),
            Some((1024, 1024))
        );
        assert_eq!(smart_resize(10, 3000, 32, 0, 1 << 30), None);
    }

    #[test]
    fn a_video_s_bounds_count_every_frame() {
        // transformers' Qwen3-VL video `smart_resize` gives these.
        let max = 16 * 16 * 2 * 2 * 2 * 6144;
        assert_eq!(
            smart_resize_frames(9, 2, 1080, 1920, 32, 4096, max),
            Some((864, 1568))
        );
        assert_eq!(
            smart_resize_frames(5, 2, 375, 500, 32, 4096, 25165824),
            Some((384, 512))
        );
        assert_eq!(
            smart_resize_frames(3, 2, 40, 40, 32, 128 * 128, 100_000_000),
            Some((96, 96))
        );
    }

    #[test]
    fn a_same_size_resize_is_the_identity_and_a_flat_image_stays_flat() {
        let img: Vec<u8> = (0..4 * 6 * 3).map(|i| (i * 7 % 256) as u8).collect();
        assert_eq!(resize_rgb8_bicubic(&img, 4, 6, 4, 6), img);
        let flat = vec![200u8; 9 * 13 * 3];
        assert!(
            resize_rgb8_bicubic(&flat, 9, 13, 5, 20)
                .iter()
                .all(|&v| v == 200)
        );
    }
}
