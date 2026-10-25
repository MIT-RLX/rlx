// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx;

/** A failed rlx call, with its {@code RLX_ERR_*} kind. */
public final class RLXException extends RuntimeException {
    /** What went wrong, by status. */
    public enum Kind {
        /** A wrong argument: unknown name, bad shape, wrong arity. */
        TYPE,
        /** An index or value out of range. */
        RANGE,
        /** Well-formed but not carried out: I/O, a backend not in this build. */
        RUNTIME,
        /** A Rust assertion inside rlx, caught at the boundary. */
        PANIC,
        /** A file that does not exist. */
        NOT_FOUND,
        /** Any other status. */
        OTHER;

        static Kind of(int rc) {
            return switch (rc) {
                case -1 -> TYPE;
                case -2 -> RANGE;
                case -3 -> RUNTIME;
                case -4 -> PANIC;
                case -5 -> NOT_FOUND;
                default -> OTHER;
            };
        }
    }

    private final Kind kind;

    public RLXException(Kind kind, String message) {
        super(message);
        this.kind = kind;
    }

    public Kind kind() {
        return kind;
    }
}
