// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Print the compiler-evolution ledger: which shipped defects became rules,
//! and which are still only stories.
//!
//! ```sh
//! cargo run -p rlx-corpus --example evolution_ledger
//! ```
//!
//! Kept as an example rather than a gate on purpose. Ungated defects are a
//! backlog, not a build failure — several are open investigations (an Apple
//! framework SIGSEGV, a Vulkan-only numerical divergence) where the honest
//! state is "known, unexplained", and failing CI on them would only teach
//! people to delete ledger entries.

fn main() {
    print!("{}", rlx_corpus::evolution::render());
}
