// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx;

/** A node's static dims ({@code -1} for a dynamic axis) and dtype. */
public record Shape(long[] dims, String dtype) {}
