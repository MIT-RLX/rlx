// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lightweight runtime counters for Metal prefill/decode kernel routing.
//! Enabled via `RLX_METAL_PREFILL_COUNTERS=1`.

use std::sync::atomic::{AtomicU64, Ordering};

static F16_PADDED: AtomicU64 = AtomicU64::new(0);
static F16_WIDE: AtomicU64 = AtomicU64::new(0);
static DEQ_GEMM: AtomicU64 = AtomicU64::new(0);
static DEQ_GEMV: AtomicU64 = AtomicU64::new(0);
static SDPA_PREFILL_FA: AtomicU64 = AtomicU64::new(0);
static SDPA_LONG: AtomicU64 = AtomicU64::new(0);

static LAST_F16_PADDED: AtomicU64 = AtomicU64::new(0);
static LAST_F16_WIDE: AtomicU64 = AtomicU64::new(0);
static LAST_DEQ_GEMM: AtomicU64 = AtomicU64::new(0);
static LAST_DEQ_GEMV: AtomicU64 = AtomicU64::new(0);
static LAST_SDPA_PREFILL_FA: AtomicU64 = AtomicU64::new(0);
static LAST_SDPA_LONG: AtomicU64 = AtomicU64::new(0);

#[inline]
pub(crate) fn record_f16_padded() {
    F16_PADDED.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn record_f16_wide() {
    F16_WIDE.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn record_dequant_gemm() {
    DEQ_GEMM.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn record_dequant_gemv() {
    DEQ_GEMV.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn record_sdpa_prefill_fa() {
    SDPA_PREFILL_FA.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn record_sdpa_long() {
    SDPA_LONG.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn maybe_report_delta(tag: &str) {
    if !rlx_ir::env::flag("RLX_METAL_PREFILL_COUNTERS") {
        return;
    }

    let f16_padded = F16_PADDED.load(Ordering::Relaxed);
    let f16_wide = F16_WIDE.load(Ordering::Relaxed);
    let deq_gemm = DEQ_GEMM.load(Ordering::Relaxed);
    let deq_gemv = DEQ_GEMV.load(Ordering::Relaxed);
    let sdpa_prefill_fa = SDPA_PREFILL_FA.load(Ordering::Relaxed);
    let sdpa_long = SDPA_LONG.load(Ordering::Relaxed);

    let last_f16_padded = LAST_F16_PADDED.swap(f16_padded, Ordering::Relaxed);
    let last_f16_wide = LAST_F16_WIDE.swap(f16_wide, Ordering::Relaxed);
    let last_deq_gemm = LAST_DEQ_GEMM.swap(deq_gemm, Ordering::Relaxed);
    let last_deq_gemv = LAST_DEQ_GEMV.swap(deq_gemv, Ordering::Relaxed);
    let last_sdpa_prefill_fa = LAST_SDPA_PREFILL_FA.swap(sdpa_prefill_fa, Ordering::Relaxed);
    let last_sdpa_long = LAST_SDPA_LONG.swap(sdpa_long, Ordering::Relaxed);

    let d_f16_padded = f16_padded.saturating_sub(last_f16_padded);
    let d_f16_wide = f16_wide.saturating_sub(last_f16_wide);
    let d_deq_gemm = deq_gemm.saturating_sub(last_deq_gemm);
    let d_deq_gemv = deq_gemv.saturating_sub(last_deq_gemv);
    let d_sdpa_prefill_fa = sdpa_prefill_fa.saturating_sub(last_sdpa_prefill_fa);
    let d_sdpa_long = sdpa_long.saturating_sub(last_sdpa_long);

    if d_f16_padded == 0
        && d_f16_wide == 0
        && d_deq_gemm == 0
        && d_deq_gemv == 0
        && d_sdpa_prefill_fa == 0
        && d_sdpa_long == 0
    {
        return;
    }

    eprintln!(
        "[prefill-counters:{tag}] delta f16(padded={},wide={}) dequant(gemm={},gemv={}) sdpa(prefill_fa={},long={}) total f16(padded={},wide={}) dequant(gemm={},gemv={}) sdpa(prefill_fa={},long={})",
        d_f16_padded,
        d_f16_wide,
        d_deq_gemm,
        d_deq_gemv,
        d_sdpa_prefill_fa,
        d_sdpa_long,
        f16_padded,
        f16_wide,
        deq_gemm,
        deq_gemv,
        sdpa_prefill_fa,
        sdpa_long,
    );
}
