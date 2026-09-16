// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! # rlx-rng — numpy and PyTorch random streams, bit-exact
//!
//! Porting a Python reference usually stops at the arithmetic, and then fails on
//! anything seeded: a random projection, a subsample, a shuffled split. Matching
//! those needs the *same stream*, not merely the same distribution — a
//! different-but-valid Gaussian gives a different projection matrix, and every
//! number downstream moves.
//!
//! This crate reproduces the generators that actually appear in that code:
//!
//! | entry point | matches |
//! |---|---|
//! | [`numpy::RandomState`] | `np.random.RandomState(seed)` — MT19937 + legacy polar Gaussian |
//! | [`numpy::Generator`] | `np.random.Generator(PCG64(seed))` — SeedSequence + PCG64 + Lemire |
//! | [`torch::randperm`] | `torch.manual_seed(s); torch.randperm(n)` |
//!
//! Each was verified against the reference implementation, and each needed a
//! detail that is invisible until you diff the streams:
//!
//! * numpy's legacy Gaussian returns `f·x2` and **caches `f·x1`** — dropping the
//!   cached variate desynchronises everything after the first draw;
//! * PCG64's `set_seed` reads word 0 as the **high** half;
//! * `Generator.integers` uses **Lemire**, not the masked rejection
//!   `RandomState` uses, and dispatches to a **32-bit** path when the range
//!   fits — where PCG64 splits one `u64` and buffers the high half;
//! * torch's `randperm` is Fisher–Yates walking **forward**.
//!
//! Every one of those produces a perfectly reasonable random sequence when
//! wrong, which is why they are worth stating.

pub mod numpy;
pub mod torch;
