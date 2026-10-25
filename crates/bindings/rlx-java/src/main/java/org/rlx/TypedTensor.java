// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx;

/** A tensor of any dtype as raw bytes ({@code "f32"}, {@code "i64"}, …). */
public record TypedTensor(byte[] bytes, String dtype) {}
