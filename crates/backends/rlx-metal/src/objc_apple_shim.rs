// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Message-dispatch shim for the Apple platforms `objc` 0.2 predates.
//!
//! `objc` 0.2.7 chooses its dispatch backend with
//! `cfg(any(target_os = "macos", target_os = "ios"))` and falls through to the
//! **GNUstep** runtime for everything else — including tvOS and visionOS, which
//! did not exist when it was last released. Those builds compile happily and
//! then fail at link with
//!
//! ```text
//! Undefined symbols for architecture arm64:
//!   "_objc_msg_lookup", referenced from: ... rlx_metal ...
//! ```
//!
//! because Apple's runtime has never exported `objc_msg_lookup` or
//! `objc_msg_lookup_super`. Nothing short of linking a real app surfaces it:
//! `cargo check` does not link, and neither does building a `staticlib`.
//!
//! Both symbols are IMP *lookups*, not sends — the caller invokes the returned
//! pointer itself — and `class_getMethodImplementation` answers exactly that
//! question on Apple's runtime, so they can be supplied here.
//!
//! **Why this is ABI-correct:** arm64 has a single message-send convention (no
//! `objc_msgSend_stret` / `_fpret` split, which is what would make a
//! lookup-then-call scheme wrong on x86_64), and the only targets that reach
//! this file — `aarch64-apple-tvos{,-sim}`, `aarch64-apple-visionos{,-sim}` —
//! are arm64. Should an x86_64 tvOS/visionOS simulator target ever become
//! usable, this shim must be revisited rather than extended.

use objc::runtime::{Class, Imp, Object, Sel};

unsafe extern "C" {
    fn object_getClass(obj: *mut Object) -> *const Class;
    fn class_getMethodImplementation(cls: *const Class, sel: Sel) -> Imp;
}

/// Layout-compatible copy of `objc::message::Super`, which the crate builds on
/// the stack and hands to `objc_msg_lookup_super` by pointer. Only the layout
/// crosses the boundary, so mirroring it here avoids depending on whether that
/// type is re-exported.
#[repr(C)]
struct Super {
    receiver: *mut Object,
    superclass: *const Class,
}

/// A receiver with no class to look a method up in. Unreachable through `objc`
/// — its GNUstep path returns zeroed memory for a null receiver *before* it
/// looks anything up — but a lookup function that can return an uninitialised
/// IMP is worse than one that returns something inert.
unsafe extern "C" fn nil_imp() -> *mut Object {
    std::ptr::null_mut()
}

/// GNUstep's `objc_msg_lookup`, expressed in Apple's runtime.
///
/// # Safety
/// `receiver` must be a valid object pointer or null, and `sel` a registered
/// selector. Called by `objc`'s generated dispatch code, never directly.
#[unsafe(no_mangle)]
unsafe extern "C" fn objc_msg_lookup(receiver: *mut Object, sel: Sel) -> Imp {
    if receiver.is_null() {
        return unsafe {
            std::mem::transmute::<unsafe extern "C" fn() -> *mut Object, Imp>(nil_imp)
        };
    }
    unsafe { class_getMethodImplementation(object_getClass(receiver), sel) }
}

/// GNUstep's `objc_msg_lookup_super`: start the search at the superclass named
/// in the `Super` struct rather than at the receiver's own class.
///
/// # Safety
/// `sup` must point to a live `Super` whose `superclass` is a valid class.
#[unsafe(no_mangle)]
unsafe extern "C" fn objc_msg_lookup_super(sup: *const Super, sel: Sel) -> Imp {
    let sup = unsafe { &*sup };
    if sup.receiver.is_null() {
        return unsafe {
            std::mem::transmute::<unsafe extern "C" fn() -> *mut Object, Imp>(nil_imp)
        };
    }
    unsafe { class_getMethodImplementation(sup.superclass, sel) }
}
