// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Argmax over an i8 slice — last layer of a classifier.

#[inline]
pub fn argmax_i8(x: &[i8]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = i8::MIN;
    for (i, &v) in x.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}
