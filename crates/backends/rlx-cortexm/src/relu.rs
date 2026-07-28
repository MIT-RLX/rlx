// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! INT8 ReLU — `out = max(x, zero_point)` in place.

#[inline(always)]
pub fn relu_i8(buf: &mut [i8], zero_point: i32) {
    let zp = zero_point.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
    for v in buf.iter_mut() {
        if *v < zp {
            *v = zp;
        }
    }
}
