// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The portable mode is chosen once per process: asking again for the same
//! mode is fine, asking for the other one is an error and changes nothing —
//! so a run cannot mix kernels picked under the two arithmetics.
//! (Own test binary: the mode is process-wide.)

use rlx_cpu::config::{PortableFixed, portable, set_portable};

#[test]
fn the_mode_is_fixed_once_chosen() {
    assert_eq!(set_portable(true), Ok(()));
    assert!(portable());
    assert_eq!(set_portable(true), Ok(()), "the same mode again");
    assert_eq!(set_portable(false), Err(PortableFixed { portable: true }));
    assert!(portable(), "a refused change changes nothing");
    let msg = set_portable(false).unwrap_err().to_string();
    assert!(msg.contains("already on"), "{msg}");
}
