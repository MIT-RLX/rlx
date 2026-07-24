// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Counter-based and ONNX Runtime–compatible RNG for in-graph random ops.
//!
//! # Behavioral contract
//!
//! [`Op::RngNormal`] / [`Op::RngUniform`] take an optional shape-template input
//! (ONNX `Random*Like`) or no inputs when the output shape is fixed at import
//! time (ONNX `Random*` with a `shape` attribute). The output tensor shape is
//! always the node's assigned shape; the template input is not copied into the
//! output.
//!
//! | Backend | Semantics |
//! |---------|-----------|
//! | [`RngBackend::Philox`] | Deterministic Philox4×32-10 stream keyed by [`RngOptions::seed`] + per-node `key`. Default for RLX-native runs. |
//! | [`RngBackend::Ort`] | Matches ONNX Runtime CPU `Random*` (`minstd_rand0` + polar normal / uniform). Use for import parity tests. Per-op ONNX `seed` (f32) overrides the mixed engine seed when set. |
//! | [`RngBackend::Bnns`] | Portable Apple BNNS AES-CTR-128 stream (`BNNSCreateRandomGeneratorWithSeed`) + BNNS uniform byte mapping; Box–Muller normals; see [`BnnsAesCtr128`]. |
//! | [`RngBackend::Zero`] | Writes zeros — useful when comparing against a stochastic reference without re-seeding ORT. |
//!
//! Policy is set at compile time via [`CompileOptions::rng`] and can be overridden
//! per session through [`rlx_runtime::CompiledGraph::set_rng`] without
//! recompiling. Each execute re-seeds from the current policy (ORT session state
//! is not advanced across runs today).

use aes::Aes128;
use aes::cipher::{Array, BlockCipherEncrypt, KeyInit};

/// Which RNG implementation to use for [`Op::RngNormal`] / [`Op::RngUniform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum RngBackend {
    /// Philox4×32-10 sequential stream (RLX native default).
    #[default]
    Philox,
    /// ONNX Runtime CPU `Random*Like` (`minstd_rand0` + `std::normal_distribution`).
    Ort,
    /// Apple BNNS AES-CTR-128 (`BNNSRandomGeneratorMethodAES_CTR`) + uniform mapping.
    Bnns,
    /// Fill with zero (deterministic parity vs stochastic reference runs).
    Zero,
}

/// Compile-time / execute-time RNG policy for graphs containing random ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RngOptions {
    /// Global seed mixed into per-node keys (maps to ORT session seed).
    pub seed: u64,
    pub backend: RngBackend,
}

impl Default for RngOptions {
    fn default() -> Self {
        Self {
            seed: 42,
            backend: RngBackend::Philox,
        }
    }
}

impl RngOptions {
    pub const fn new(seed: u64, backend: RngBackend) -> Self {
        Self { seed, backend }
    }

    pub fn philox(seed: u64) -> Self {
        Self {
            seed,
            backend: RngBackend::Philox,
        }
    }

    pub fn ort(seed: u64) -> Self {
        Self {
            seed,
            backend: RngBackend::Ort,
        }
    }

    pub fn bnns(seed: u64) -> Self {
        Self {
            seed,
            backend: RngBackend::Bnns,
        }
    }

    pub fn zero() -> Self {
        Self {
            seed: 0,
            backend: RngBackend::Zero,
        }
    }
}

/// Mix a global compile seed with a per-node key (ONNX node name hash).
pub fn combine_seed(global: u64, key: u64) -> u64 {
    global.wrapping_add(key.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// ORT CPU engine seed: explicit ONNX `seed` attr cast to u32, else global+key.
pub fn ort_engine_seed(global: u64, key: u64, op_seed: Option<f32>) -> u32 {
    if let Some(s) = op_seed {
        s as u32
    } else {
        global.wrapping_add(key) as u32
    }
}

/// Fill `out` with `mean + scale * N(0,1)` samples.
pub fn fill_normal_like(
    out: &mut [f32],
    mean: f32,
    scale: f32,
    opts: RngOptions,
    key: u64,
    op_seed: Option<f32>,
) {
    match opts.backend {
        RngBackend::Zero => out.fill(0.0),
        RngBackend::Philox => {
            let mut rng = Philox4x32::new(combine_seed(opts.seed, key));
            for v in out.iter_mut() {
                *v = mean + scale * rng.normal();
            }
        }
        RngBackend::Ort => {
            let mut eng = MinstdRand0::new(ort_engine_seed(opts.seed, key, op_seed));
            let mut dist = StdNormalDist::new(mean, scale);
            for v in out.iter_mut() {
                *v = dist.sample(&mut eng);
            }
        }
        RngBackend::Bnns => {
            let _ = op_seed;
            let mut rng = BnnsAesCtr128::new(combine_seed(opts.seed, key));
            rng.fill_normal(out, mean, scale);
        }
    }
}

/// Fill `out` with uniform samples in `[low, high)`.
pub fn fill_uniform_like(
    out: &mut [f32],
    low: f32,
    high: f32,
    opts: RngOptions,
    key: u64,
    op_seed: Option<f32>,
) {
    match opts.backend {
        RngBackend::Zero => out.fill(0.0),
        RngBackend::Philox => {
            let mut rng = Philox4x32::new(combine_seed(opts.seed, key));
            for v in out.iter_mut() {
                *v = rng.uniform(low, high);
            }
        }
        RngBackend::Ort => {
            let mut eng = MinstdRand0::new(ort_engine_seed(opts.seed, key, op_seed));
            for v in out.iter_mut() {
                *v = low + (high - low) * eng.unit_f32();
            }
        }
        RngBackend::Bnns => {
            let _ = op_seed;
            let mut rng = BnnsAesCtr128::new(combine_seed(opts.seed, key));
            rng.fill_uniform(out, low, high);
        }
    }
}

/// Philox4×32 counter-based RNG. Produces 4 u32s per round of the
/// core hash — we expose an iterator that yields one f32 per call.
#[derive(Debug, Clone, Copy)]
pub struct Philox4x32 {
    seed: [u32; 2],
    counter: [u32; 4],
    /// Cached output buffer + cursor into it.
    buffer: [u32; 4],
    cursor: u8,
}

impl Philox4x32 {
    pub const fn new(seed: u64) -> Self {
        let lo = (seed & 0xFFFF_FFFF) as u32;
        let hi = (seed >> 32) as u32;
        Self {
            seed: [lo, hi],
            counter: [0, 0, 0, 0],
            buffer: [0; 4],
            cursor: 4, // empty — next next_u32 fills the buffer
        }
    }

    fn round(state: &mut [u32; 4], key: [u32; 2]) {
        const M0: u64 = 0xD256_1A75;
        const M1: u64 = 0xCD9E_8D57;
        let p0 = (state[0] as u64) * M0;
        let p1 = (state[2] as u64) * M1;
        let hi0 = (p0 >> 32) as u32;
        let lo0 = p0 as u32;
        let hi1 = (p1 >> 32) as u32;
        let lo1 = p1 as u32;
        state[0] = hi1 ^ state[1] ^ key[0];
        state[1] = lo1;
        state[2] = hi0 ^ state[3] ^ key[1];
        state[3] = lo0;
    }

    fn fill_buffer(&mut self) {
        let mut state = self.counter;
        let mut key = self.seed;
        for _ in 0..10 {
            Self::round(&mut state, key);
            // Bump the key on every round (Philox key schedule).
            key[0] = key[0].wrapping_add(0x9E37_79B9);
            key[1] = key[1].wrapping_add(0xBB67_AE85);
        }
        self.buffer = state;
        self.cursor = 0;

        // Increment the 128-bit counter.
        let (c0, of0) = self.counter[0].overflowing_add(1);
        self.counter[0] = c0;
        if of0 {
            let (c1, of1) = self.counter[1].overflowing_add(1);
            self.counter[1] = c1;
            if of1 {
                let (c2, of2) = self.counter[2].overflowing_add(1);
                self.counter[2] = c2;
                if of2 {
                    self.counter[3] = self.counter[3].wrapping_add(1);
                }
            }
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.cursor >= 4 {
            self.fill_buffer();
        }
        let v = self.buffer[self.cursor as usize];
        self.cursor += 1;
        v
    }

    /// Uniform `[0, 1)` f32 — the top 24 bits of a u32 give exactly
    /// f32 mantissa precision.
    pub fn next_f32(&mut self) -> f32 {
        let bits = self.next_u32() >> 8;
        bits as f32 / (1u32 << 24) as f32
    }

    /// Uniform `[lo, hi)` f32.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Standard-normal `f32` via Box-Muller. Returns one sample;
    /// the second is discarded (we don't cache to keep the type
    /// `Copy`-able).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }

    /// Fill `out` with uniform `[0, 1)` samples. Convenience for
    /// weight init.
    pub fn fill_uniform(&mut self, out: &mut [f32]) {
        for v in out {
            *v = self.next_f32();
        }
    }

    /// Fill `out` with N(0, 1) samples.
    pub fn fill_normal(&mut self, out: &mut [f32]) {
        for v in out {
            *v = self.normal();
        }
    }

    /// One `Gumbel(0,1)` sample via inverse CDF: `-ln(-ln(u))`.
    ///
    /// Uniform `u` is clamped to `[ε, 1−ε]` with `ε = 1e-6` (WaveRNN /
    /// Espresso `gumbel_max` convention) so `ln` stays finite.
    pub fn gumbel01(&mut self) -> f32 {
        gumbel01_from_uniform(self.next_f32())
    }

    /// Espresso-style `gumbel_max`: `argmax_i(logit_i + scale * Gumbel(0,1))`.
    ///
    /// Draws one Philox uniform per logit. Equivalent in distribution to
    /// multinomial from `softmax(logits / scale)`, but the RNG trajectory
    /// differs — WaveRNN must use this path, not [`crate::Op::Sample`].
    pub fn gumbel_max_argmax(&mut self, logits: &[f32], scale: f32) -> usize {
        gumbel_max_argmax_with(logits, scale, || self.gumbel01())
    }
}

/// Portable reproduction of `BNNSRandomGeneratorMethodAES_CTR`.
///
/// Apple's seeded BNNS generator expands a `u64` seed by repeating its
/// little-endian bytes to form the AES-128 key, encrypts a big-endian 128-bit
/// counter starting at zero, and consumes each encrypted block as four
/// little-endian `u32` words. Keeping this implementation in RLX makes the
/// stream reproducible away from Apple platforms too.
pub struct BnnsAesCtr128 {
    cipher: Aes128,
    counter: u128,
    block: [u8; 16],
    cursor: u8,
    /// Unused Box–Muller sin sample within the current [`Self::fill_normal`].
    normal_spare: Option<f32>,
}

impl BnnsAesCtr128 {
    /// Construct the stream produced by
    /// `BNNSCreateRandomGeneratorWithSeed(BNNSRandomGeneratorMethodAES_CTR, seed, ...)`.
    pub fn new(seed: u64) -> Self {
        let seed = seed.to_le_bytes();
        let mut key = [0u8; 16];
        key[..8].copy_from_slice(&seed);
        key[8..].copy_from_slice(&seed);
        Self {
            cipher: Aes128::new(&Array::from(key)),
            counter: 0,
            block: [0; 16],
            cursor: 16,
            normal_spare: None,
        }
    }

    /// Match `BNNSCreateRandomGenerator` (AES-CTR seeded from OS entropy).
    ///
    /// Apple's unseeded constructor draws an internal `u64` seed and then
    /// follows the same path as [`Self::new`]. The seed value itself is not
    /// observable without `BNNSRandomGeneratorGetState`; use
    /// [`bnns_seed_from_generator_state`] when replaying a captured Apple run.
    pub fn from_entropy() -> Self {
        Self::new(bnns_entropy_seed())
    }

    /// Replay a generator whose 40-byte `BNNSRandomGeneratorGetState` blob was
    /// captured **before** any draws (counter/cursor still zero).
    ///
    /// Returns `None` if `state` is too short or the generator has already
    /// advanced (non-zero counter/cursor) — advanced states need a fuller
    /// restore than seed-only construction.
    pub fn try_from_fresh_generator_state(state: &[u8]) -> Option<Self> {
        let seed = bnns_seed_from_generator_state(state)?;
        if !bnns_generator_state_is_fresh(state) {
            return None;
        }
        Some(Self::new(seed))
    }

    fn fill_block(&mut self) {
        let mut block = Array::from(self.counter.to_be_bytes());
        self.cipher.encrypt_block(&mut block);
        self.block.copy_from_slice(&block);
        self.counter = self.counter.wrapping_add(1);
        self.cursor = 0;
    }

    /// Next AES stream word, using BNNS's little-endian grouping.
    pub fn next_u32(&mut self) -> u32 {
        if self.cursor >= 16 {
            self.fill_block();
        }
        let start = self.cursor as usize;
        self.cursor += 4;
        u32::from_le_bytes(
            self.block[start..start + 4]
                .try_into()
                .expect("four-byte AES word"),
        )
    }

    /// Next float using the conversion performed inside
    /// `BNNSRandomFillUniformFloat`.
    ///
    /// BNNS rounds the complete 32-bit integer to `f32` and scales by 2^-32;
    /// it does not truncate to 23 or 24 random bits first.
    pub fn next_f32(&mut self) -> f32 {
        bnns_uniform_f32_from_u32(self.next_u32())
    }

    /// Uniform `[lo, hi]` sample following BNNS float arithmetic.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        self.next_f32().mul_add(hi - lo, lo)
    }

    /// Fill with the AES-CTR stream used by `BNNSRandomFillUniformFloat`.
    pub fn fill_uniform(&mut self, out: &mut [f32], lo: f32, hi: f32) {
        for value in out {
            *value = self.uniform(lo, hi);
        }
    }

    /// One Espresso-compatible Gumbel value from BNNS's private byte mapping.
    pub fn gumbel01(&mut self) -> f32 {
        gumbel01_from_uniform(self.next_f32())
    }

    /// BNNS activation Gumbel: `−log(−log(α·U+β)+β)`.
    pub fn gumbel(&mut self, alpha: f32, beta: f32) -> f32 {
        bnns_gumbel_from_uniform(self.next_f32(), alpha, beta)
    }

    /// Fill `out` with Gumbel(0,1) samples from the BNNS AES-CTR stream.
    pub fn fill_gumbel01(&mut self, out: &mut [f32]) {
        for value in out {
            *value = self.gumbel01();
        }
    }

    /// Fill with BNNS-style Gumbel(`α`, `β`) noise.
    pub fn fill_gumbel(&mut self, out: &mut [f32], alpha: f32, beta: f32) {
        for value in out {
            *value = self.gumbel(alpha, beta);
        }
    }

    /// One `N(0,1)` sample via Box–Muller on BNNS uniforms.
    ///
    /// Within a [`Self::fill_normal`] call the sin half of each pair is kept as
    /// a spare (Apple's batch behavior). Across separate fills the spare is
    /// dropped, matching `BNNSRandomFillNormalFloat`.
    pub fn normal(&mut self) -> f32 {
        if let Some(spare) = self.normal_spare.take() {
            return spare;
        }
        let u1 = self.next_f32().clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON);
        let u2 = self.next_f32().clamp(0.0, 1.0 - f32::EPSILON);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        self.normal_spare = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Fill `out` with `mean + stddev * N(0,1)` like `BNNSRandomFillNormalFloat`.
    pub fn fill_normal(&mut self, out: &mut [f32], mean: f32, stddev: f32) {
        self.normal_spare = None;
        for value in out {
            *value = self.normal().mul_add(stddev, mean);
        }
        self.normal_spare = None;
    }

    /// Uniform integer in `[lo, hi)` via Lemire multiply-high on AES words.
    ///
    /// Apple's `BNNSRandomFillUniformInt` uses this stream for scalar fills
    /// (`n < 16`) but switches to a different vectorized trajectory for larger
    /// `n`; this portable path always stays on the AES word stream.
    pub fn uniform_int(&mut self, lo: i64, hi: i64) -> i64 {
        bnns_uniform_int_from_u32(self.next_u32(), lo, hi)
    }

    /// Fill `out` with integers in `[lo, hi)`.
    pub fn fill_uniform_int(&mut self, out: &mut [i64], lo: i64, hi: i64) {
        for value in out {
            *value = self.uniform_int(lo, hi);
        }
    }

    /// Categorical indices from class probabilities (or log-probabilities).
    ///
    /// Matches `BNNSRandomFillCategoricalFloat`: one BNNS uniform per draw,
    /// inverse-CDF over the (softmax-)normalized class weights.
    pub fn fill_categorical(
        &mut self,
        out: &mut [f32],
        probabilities: &[f32],
        log_probabilities: bool,
    ) {
        let probs = bnns_normalize_probs(probabilities, log_probabilities);
        for value in out {
            *value = bnns_categorical_from_uniform(self.next_f32(), &probs) as f32;
        }
    }

    /// Espresso `gumbel_max` using BNNS uniforms + ε-clamped ICDF.
    pub fn gumbel_max_argmax(&mut self, logits: &[f32], scale: f32) -> usize {
        gumbel_max_argmax_with(logits, scale, || self.gumbel01())
    }

    /// BNNS `GumbelMax` activation: `argmax_j(logit_j + Gumbel(α,β))`.
    pub fn bnns_gumbel_max_argmax(&mut self, logits: &[f32], alpha: f32, beta: f32) -> usize {
        gumbel_max_argmax_with(logits, 1.0, || self.gumbel(alpha, beta))
    }
}

/// Serialized access to Apple's process-global BNNS `GumbelMax` stream.
///
/// The native activation does not accept a caller-owned random generator.
/// Holding this value prevents another RLX native sampler from interleaving
/// draws after the stream has been initialized.
#[cfg(target_os = "macos")]
pub struct NativeBnnsGumbelMax {
    _guard: std::sync::MutexGuard<'static, ()>,
}

#[cfg(target_os = "macos")]
static NATIVE_BNNS_GUMBEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(target_os = "macos")]
impl NativeBnnsGumbelMax {
    /// Initialize the native BNNS stream and retain exclusive RLX access.
    pub fn new(seed: u64) -> Option<Self> {
        #[link(name = "Accelerate", kind = "framework")]
        unsafe extern "C" {
            fn BNNSInitGumbel(seed: u64);
        }

        let guard = NATIVE_BNNS_GUMBEL_LOCK.lock().ok()?;
        // SAFETY: the private function takes one scalar and updates the
        // process-global Gumbel state synchronously.
        unsafe { BNNSInitGumbel(seed) };
        Some(Self { _guard: guard })
    }

    /// Sample one class through `BNNSDirectApplyActivationBatch`.
    pub fn argmax(&mut self, logits: &[f32], alpha: f32, beta: f32) -> Option<usize> {
        native_bnns_gumbel_max_argmax_impl(logits, alpha, beta)
    }
}

#[cfg(not(target_os = "macos"))]
pub struct NativeBnnsGumbelMax;

#[cfg(not(target_os = "macos"))]
impl NativeBnnsGumbelMax {
    pub fn new(_seed: u64) -> Option<Self> {
        None
    }

    pub fn argmax(&mut self, _logits: &[f32], _alpha: f32, _beta: f32) -> Option<usize> {
        None
    }
}

#[cfg(target_os = "macos")]
fn native_bnns_gumbel_max_argmax_impl(logits: &[f32], alpha: f32, beta: f32) -> Option<usize> {
    #[repr(C, align(64))]
    #[derive(Clone, Copy)]
    struct AlignedF32Block([f32; 16]);

    #[repr(C)]
    struct BnnsNdArrayDescriptor {
        flags: u32,
        layout: u32,
        size: [usize; 8],
        stride: [usize; 8],
        data: *mut core::ffi::c_void,
        data_type: u32,
        table_data: *mut core::ffi::c_void,
        table_data_type: u32,
        data_scale: f32,
        data_bias: f32,
    }

    #[repr(C)]
    struct BnnsActivation {
        function: u32,
        alpha: f32,
        beta: f32,
        iscale: i32,
        ioffset: i32,
        ishift: i32,
        iscale_per_channel: *const i32,
        ioffset_per_channel: *const i32,
        ishift_per_channel: *const i32,
    }

    #[repr(C)]
    struct BnnsLayerParametersActivation {
        i_desc: BnnsNdArrayDescriptor,
        o_desc: BnnsNdArrayDescriptor,
        activation: BnnsActivation,
        axis_flags: u32,
    }

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn BNNSDirectApplyActivationBatch(
            layer_params: *const BnnsLayerParametersActivation,
            filter_params: *const core::ffi::c_void,
            batch_size: usize,
            in_stride: usize,
            out_stride: usize,
        ) -> i32;
    }

    const BNNS_LAYOUT_VECTOR: u32 = 0x1_0000;
    const BNNS_FLOAT32: u32 = 0x1_0000 | 32;
    const BNNS_ACTIVATION_GUMBEL_MAX: u32 = 14;

    fn desc(data: *mut f32, n: usize) -> BnnsNdArrayDescriptor {
        BnnsNdArrayDescriptor {
            flags: 0,
            layout: BNNS_LAYOUT_VECTOR,
            size: [n, 0, 0, 0, 0, 0, 0, 0],
            stride: [0; 8],
            data: data.cast(),
            data_type: BNNS_FLOAT32,
            table_data: core::ptr::null_mut(),
            table_data_type: 0,
            data_scale: 1.0,
            data_bias: 0.0,
        }
    }

    if logits.is_empty() {
        return None;
    }
    // BNNS partitions this activation according to pointer alignment.
    // Espresso's activation buffers are 64-byte aligned.
    let mut input = vec![AlignedF32Block([0.0; 16]); logits.len().div_ceil(16)];
    for (i, &value) in logits.iter().enumerate() {
        input[i / 16].0[i % 16] = value;
    }
    let mut output = AlignedF32Block([0.0; 16]);
    let params = BnnsLayerParametersActivation {
        i_desc: desc(input.as_mut_ptr().cast(), logits.len()),
        o_desc: desc(output.0.as_mut_ptr(), 1),
        activation: BnnsActivation {
            function: BNNS_ACTIVATION_GUMBEL_MAX,
            alpha,
            beta,
            iscale: 0,
            ioffset: 0,
            ishift: 0,
            iscale_per_channel: core::ptr::null(),
            ioffset_per_channel: core::ptr::null(),
            ishift_per_channel: core::ptr::null(),
        },
        axis_flags: 0,
    };
    // SAFETY: descriptors point to live contiguous f32 buffers for this
    // synchronous call and mirror the SDK C layout.
    let rc =
        unsafe { BNNSDirectApplyActivationBatch(&params, core::ptr::null(), 1, logits.len(), 1) };
    let output = output.0[0];
    if rc != 0 || !output.is_finite() || output < 0.0 {
        return None;
    }
    let index = output as usize;
    (index < logits.len() && output == index as f32).then_some(index)
}

/// Byte length of `BNNSRandomGeneratorGetState` for AES-CTR generators.
pub const BNNS_AES_CTR_STATE_SIZE: usize = 40;

/// Extract the `u64` seed from a BNNS AES-CTR generator state blob.
///
/// Layout (observed on macOS): bytes `[0..8]` and `[8..16]` are the seed's
/// little-endian bytes repeated (AES-128 key = `seed‖seed`); `[16..32]` holds
/// the big-endian counter / residual block; `[32..40]` is the stream cursor.
pub fn bnns_seed_from_generator_state(state: &[u8]) -> Option<u64> {
    if state.len() < 8 {
        return None;
    }
    Some(u64::from_le_bytes(state[..8].try_into().ok()?))
}

/// `true` when `state` still describes a freshly constructed generator
/// (no uniforms drawn yet).
pub fn bnns_generator_state_is_fresh(state: &[u8]) -> bool {
    state.len() >= BNNS_AES_CTR_STATE_SIZE
        && state[16..BNNS_AES_CTR_STATE_SIZE].iter().all(|&b| b == 0)
}

/// OS CSPRNG `u64` used the same way `BNNSCreateRandomGenerator` seeds AES-CTR.
///
/// This matches Apple's *contract* (entropy → `WithSeed`), not a specific
/// syscall sequence inside Accelerate.
pub fn bnns_entropy_seed() -> u64 {
    let mut bytes = [0u8; 8];
    fill_os_entropy(&mut bytes);
    u64::from_le_bytes(bytes)
}

#[cfg(unix)]
fn fill_os_entropy(out: &mut [u8]) {
    // getentropy(2) is available on macOS 10.12+ / modern Linux.
    #[link(name = "c")]
    unsafe extern "C" {
        fn getentropy(buf: *mut core::ffi::c_void, buflen: usize) -> i32;
    }
    let rc = unsafe { getentropy(out.as_mut_ptr().cast(), out.len()) };
    if rc == 0 {
        return;
    }
    // Extremely unlikely; fall back to a mixed time stamp rather than panic.
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x0C0F_FEE0_0D15_EA5E);
    out.copy_from_slice(&t.to_le_bytes()[..out.len().min(8)]);
}

#[cfg(not(unix))]
fn fill_os_entropy(out: &mut [u8]) {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x0C0F_FEE0_0D15_EA5E);
    let b = t.to_le_bytes();
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = b[i % 8];
    }
}

/// Convert one AES stream word exactly as `BNNSRandomFillUniformFloat` does.
///
/// The cast can round values near `u32::MAX` to 2^32, so the result may be
/// exactly `1.0`; callers applying logarithms must clamp as appropriate.
pub fn bnns_uniform_f32_from_u32(word: u32) -> f32 {
    (word as f32) * (1.0 / 4_294_967_296.0)
}

/// Convert four consecutive AES bytes to BNNS's uniform float.
pub fn bnns_uniform_f32_from_bytes(bytes: [u8; 4]) -> f32 {
    bnns_uniform_f32_from_u32(u32::from_le_bytes(bytes))
}

/// Convert four consecutive AES bytes through BNNS's uniform mapping and the
/// Espresso Gumbel inverse CDF.
pub fn bnns_gumbel01_from_bytes(bytes: [u8; 4]) -> f32 {
    gumbel01_from_uniform(bnns_uniform_f32_from_bytes(bytes))
}

/// Map an AES word into `[lo, hi)` via Lemire multiply-high.
///
/// Requires `lo < hi`. When `lo >= hi`, returns `lo`.
pub fn bnns_uniform_int_from_u32(word: u32, lo: i64, hi: i64) -> i64 {
    if lo >= hi {
        return lo;
    }
    let span = (hi - lo) as u64;
    let offset = ((word as u64).wrapping_mul(span)) >> 32;
    lo.wrapping_add(offset as i64)
}

/// BNNS activation Gumbel: `−log(−log(α·U+β)+β)`.
///
/// When `β == 0`, `U` is clamped away from the singular endpoints.
pub fn bnns_gumbel_from_uniform(u: f32, alpha: f32, beta: f32) -> f32 {
    let u = if beta == 0.0 {
        u.clamp(GUMBEL_UNIFORM_EPS, 1.0 - GUMBEL_UNIFORM_EPS)
    } else {
        u
    };
    let inner = alpha.mul_add(u, beta).max(f32::MIN_POSITIVE);
    -((-inner.ln()) + beta).ln()
}

/// Inverse-CDF categorical index from one BNNS uniform in `[0, 1)`.
pub fn bnns_categorical_from_uniform(u: f32, probabilities: &[f32]) -> usize {
    if probabilities.is_empty() {
        return 0;
    }
    let mut cum = 0.0f32;
    for (i, &p) in probabilities.iter().enumerate() {
        cum += p;
        if u < cum {
            return i;
        }
    }
    probabilities.len() - 1
}

fn bnns_normalize_probs(probabilities: &[f32], log_probabilities: bool) -> Vec<f32> {
    if probabilities.is_empty() {
        return Vec::new();
    }
    if log_probabilities {
        let max = probabilities
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = probabilities.iter().map(|&p| (p - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum > 0.0 {
            for p in &mut exps {
                *p /= sum;
            }
        }
        exps
    } else {
        let sum: f32 = probabilities.iter().sum();
        if sum > 0.0 {
            probabilities.iter().map(|&p| p / sum).collect()
        } else {
            probabilities.to_vec()
        }
    }
}

/// Clamp used by WaveRNN / Espresso `gumbel_max` before the Gumbel ICDF.
pub const GUMBEL_UNIFORM_EPS: f32 = 1e-6;

/// `Gumbel(0,1)` from a uniform sample in `[0, 1)`.
pub fn gumbel01_from_uniform(u: f32) -> f32 {
    let u = u.clamp(GUMBEL_UNIFORM_EPS, 1.0 - GUMBEL_UNIFORM_EPS);
    -(-u.ln()).ln()
}

/// `argmax_i(logit_i + scale * g_i)` for caller-supplied Gumbel draws.
pub fn gumbel_max_argmax_with<F>(logits: &[f32], scale: f32, mut gumbel: F) -> usize
where
    F: FnMut() -> f32,
{
    if logits.is_empty() {
        return 0;
    }
    let scale = scale.max(1e-8);
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &logit) in logits.iter().enumerate() {
        let v = logit + scale * gumbel();
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Plain argmax (Espresso `gumbel_max` with `eps = 0` / greedy).
pub fn argmax_f32(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// C++11 `std::default_random_engine` on libstdc++/libc++ (`minstd_rand0`).
#[derive(Debug, Clone, Copy)]
struct MinstdRand0 {
    state: u32,
}

impl MinstdRand0 {
    const A: u32 = 48_271;
    const M: u32 = 2_147_483_647;

    fn new(seed: u32) -> Self {
        Self {
            state: seed % Self::M,
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = ((self.state as u64 * Self::A as u64) % Self::M as u64) as u32;
        self.state
    }

    /// Uniform in `[0, 1)` matching ORT's `RealType(g()) / (g.max() - g.min())`.
    fn unit_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (Self::M - 1) as f32
    }
}

/// C++ `std::normal_distribution<float>` (polar method, caches spare sample).
#[derive(Debug, Clone, Copy)]
struct StdNormalDist {
    mean: f32,
    scale: f32,
    spare: f32,
    has_spare: bool,
}

impl StdNormalDist {
    fn new(mean: f32, scale: f32) -> Self {
        Self {
            mean,
            scale,
            spare: 0.0,
            has_spare: false,
        }
    }

    fn sample(&mut self, eng: &mut MinstdRand0) -> f32 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare;
        }
        loop {
            let u1 = 2.0 * eng.unit_f32() - 1.0;
            let u2 = 2.0 * eng.unit_f32() - 1.0;
            let s = u1 * u1 + u2 * u2;
            if s >= 1.0 || s == 0.0 {
                continue;
            }
            let factor = (-2.0 * s.ln() / s).sqrt();
            self.spare = u2 * factor * self.scale + self.mean;
            self.has_spare = true;
            return u1 * factor * self.scale + self.mean;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = Philox4x32::new(0x1234_5678);
        let mut b = Philox4x32::new(0x1234_5678);
        for _ in 0..256 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_seed_different_sequence() {
        let mut a = Philox4x32::new(1);
        let mut b = Philox4x32::new(2);
        let mut diffs = 0usize;
        for _ in 0..16 {
            if a.next_u32() != b.next_u32() {
                diffs += 1;
            }
        }
        assert!(
            diffs >= 14,
            "two distinct seeds should disagree on >=14/16 samples"
        );
    }

    #[test]
    fn next_f32_in_unit_interval() {
        let mut r = Philox4x32::new(42);
        for _ in 0..1000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v), "{v} not in [0, 1)");
        }
    }

    #[test]
    fn fill_uniform_is_deterministic() {
        let mut r1 = Philox4x32::new(7);
        let mut r2 = Philox4x32::new(7);
        let mut a = vec![0f32; 64];
        let mut b = vec![0f32; 64];
        r1.fill_uniform(&mut a);
        r2.fill_uniform(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn normal_mean_is_near_zero() {
        let mut r = Philox4x32::new(123);
        let n = 10_000;
        let mut sum = 0f32;
        for _ in 0..n {
            sum += r.normal();
        }
        let mean = sum / n as f32;
        assert!(mean.abs() < 0.1, "mean {mean} too far from 0");
    }

    #[test]
    fn zero_backend_fills_zeros() {
        let mut out = vec![1.0; 8];
        fill_normal_like(&mut out, 0.0, 1.0, RngOptions::zero(), 0xABC, None);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn philox_backend_is_deterministic() {
        let opts = RngOptions::philox(99);
        let mut a = vec![0f32; 32];
        let mut b = vec![0f32; 32];
        fill_normal_like(&mut a, 0.0, 0.5, opts, 123, None);
        fill_normal_like(&mut b, 0.0, 0.5, opts, 123, None);
        assert_eq!(a, b);
    }

    #[test]
    fn ort_backend_is_deterministic() {
        let opts = RngOptions::ort(7);
        let mut a = vec![0f32; 64];
        let mut b = vec![0f32; 64];
        fill_normal_like(&mut a, 0.1, 2.0, opts, 555, None);
        fill_normal_like(&mut b, 0.1, 2.0, opts, 555, None);
        assert_eq!(a, b);
    }

    #[test]
    fn backends_disagree() {
        let mut philox = vec![0f32; 16];
        let mut ort = vec![0f32; 16];
        fill_normal_like(&mut philox, 0.0, 1.0, RngOptions::philox(42), 1, None);
        fill_normal_like(&mut ort, 0.0, 1.0, RngOptions::ort(42), 1, None);
        assert_ne!(philox, ort);
    }

    #[test]
    fn gumbel_max_is_deterministic() {
        let logits = [1.0f32, 5.0, 2.0, 3.0];
        let mut a = Philox4x32::new(7);
        let mut b = Philox4x32::new(7);
        assert_eq!(
            a.gumbel_max_argmax(&logits, 0.01),
            b.gumbel_max_argmax(&logits, 0.01)
        );
    }

    #[test]
    fn gumbel_max_zero_scale_is_argmax() {
        let logits = [1.0f32, 9.0, 2.0];
        // With tiny scale the peak still wins almost always; argmax helper is exact.
        assert_eq!(argmax_f32(&logits), 1);
        let mut rng = Philox4x32::new(1);
        // Very small scale ≈ greedy on peaked logits.
        assert_eq!(rng.gumbel_max_argmax(&logits, 1e-8), 1);
    }

    #[test]
    fn gumbel01_from_uniform_matches_icdf() {
        let g = gumbel01_from_uniform(0.5);
        let expected = -((-0.5f32.ln()).ln());
        assert!((g - expected).abs() < 1e-6);
    }

    #[test]
    fn bnns_byte_mapping_uses_little_endian_full_word() {
        assert_eq!(
            bnns_uniform_f32_from_bytes([0x38, 0xc1, 0x16, 0xcf]).to_bits(),
            0x3f4f_16c1
        );
        assert_eq!(bnns_uniform_f32_from_u32(u32::MAX), 1.0);
    }

    #[test]
    fn bnns_gumbel_uses_private_uniform_mapping() {
        let bytes = [0x38, 0xc1, 0x16, 0xcf];
        assert_eq!(
            bnns_gumbel01_from_bytes(bytes),
            gumbel01_from_uniform(f32::from_bits(0x3f4f_16c1))
        );
    }

    #[test]
    fn bnns_batch_fill_matches_scalar_stream_across_block_edges() {
        // Lengths that sit before / on / after AES-block boundaries (4 floats).
        let lengths = [1usize, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 257, 1024];
        let seeds = [0u64, 1, 42, 0x1234_5678_9abc_def0];
        let ranges = [(0.0, 1.0), (-2.0, 3.0), (0.5, 0.5), (-1e3, 1e3)];

        for &seed in &seeds {
            for &n in &lengths {
                for &(lo, hi) in &ranges {
                    let mut scalar = BnnsAesCtr128::new(seed);
                    let mut batch = BnnsAesCtr128::new(seed);
                    let mut from_scalar = vec![0.0; n];
                    let mut from_batch = vec![0.0; n];
                    for v in &mut from_scalar {
                        *v = scalar.uniform(lo, hi);
                    }
                    batch.fill_uniform(&mut from_batch, lo, hi);
                    assert_eq!(
                        from_batch, from_scalar,
                        "seed={seed} n={n} range=[{lo},{hi}]"
                    );
                }
            }
        }
    }

    #[test]
    fn bnns_batch_gumbel_matches_uniform_then_icdf() {
        let n = 4096;
        let seed = 0xA5A5_5A5A_C3C3_3C3C;
        let mut u_rng = BnnsAesCtr128::new(seed);
        let mut g_rng = BnnsAesCtr128::new(seed);
        let mut uniforms = vec![0.0; n];
        let mut gumbels = vec![0.0; n];
        u_rng.fill_uniform(&mut uniforms, 0.0, 1.0);
        g_rng.fill_gumbel01(&mut gumbels);
        for (i, (&u, &g)) in uniforms.iter().zip(gumbels.iter()).enumerate() {
            assert_eq!(g, gumbel01_from_uniform(u), "gumbel mismatch at sample {i}");
        }
    }

    #[test]
    fn bnns_batch_is_deterministic_and_seed_sensitive() {
        let n = 2048;
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut c = vec![0.0; n];
        BnnsAesCtr128::new(99).fill_uniform(&mut a, -1.0, 2.0);
        BnnsAesCtr128::new(99).fill_uniform(&mut b, -1.0, 2.0);
        BnnsAesCtr128::new(100).fill_uniform(&mut c, -1.0, 2.0);
        assert_eq!(a, b);
        assert_ne!(a, c);
        // Distinct seeds should disagree on nearly all samples.
        let diffs = a.iter().zip(c.iter()).filter(|(x, y)| x != y).count();
        assert!(diffs > n * 9 / 10, "only {diffs}/{n} samples differed");
    }

    #[test]
    fn bnns_captured_apple_vectors_seed_42() {
        // Anchors captured from macOS BNNSRandomFillUniformFloat.
        let unit = [
            0x3f4f_16c1,
            0x3f17_8337,
            0x3ef1_8cae,
            0x3f71_4573,
            0x3f73_349f,
            0x3f24_dd54,
            0x3cdb_6bcf,
            0x3d7c_eb6a,
        ];
        let ranged = [
            0x4002_dc71,
            0x3f75_9013,
            0x3eb7_bf66,
            0x402d_96d0,
            0x4030_01c7,
            0x3f9c_2952,
            0xbfee_db94,
            0xbfd8_7b37,
        ];
        let mut unit_out = [0.0; 8];
        let mut ranged_out = [0.0; 8];
        BnnsAesCtr128::new(42).fill_uniform(&mut unit_out, 0.0, 1.0);
        BnnsAesCtr128::new(42).fill_uniform(&mut ranged_out, -2.0, 3.0);
        assert_eq!(unit_out.map(f32::to_bits), unit);
        assert_eq!(ranged_out.map(f32::to_bits), ranged);
    }

    #[test]
    fn bnns_from_entropy_is_seed_sensitive() {
        let mut a = [0.0; 32];
        let mut b = [0.0; 32];
        BnnsAesCtr128::from_entropy().fill_uniform(&mut a, 0.0, 1.0);
        BnnsAesCtr128::from_entropy().fill_uniform(&mut b, 0.0, 1.0);
        // Astronomically unlikely to collide for 32 independent floats.
        assert_ne!(a, b);
    }

    #[test]
    fn bnns_seed_from_fresh_state_roundtrip() {
        let seed: u64 = 0x0123_4567_89ab_cdef;
        let mut state = [0u8; BNNS_AES_CTR_STATE_SIZE];
        let le = seed.to_le_bytes();
        state[..8].copy_from_slice(&le);
        state[8..16].copy_from_slice(&le);
        assert!(bnns_generator_state_is_fresh(&state));
        assert_eq!(bnns_seed_from_generator_state(&state), Some(seed));
        let mut from_state = [0.0; 8];
        let mut from_seed = [0.0; 8];
        BnnsAesCtr128::try_from_fresh_generator_state(&state)
            .unwrap()
            .fill_uniform(&mut from_state, 0.0, 1.0);
        BnnsAesCtr128::new(seed).fill_uniform(&mut from_seed, 0.0, 1.0);
        assert_eq!(from_state, from_seed);
        // Advanced cursor → reject.
        state[32] = 4;
        assert!(!bnns_generator_state_is_fresh(&state));
        assert!(BnnsAesCtr128::try_from_fresh_generator_state(&state).is_none());
    }

    #[test]
    fn bnns_backend_fill_helpers_are_deterministic() {
        let opts = RngOptions::bnns(99);
        let mut a = vec![0f32; 64];
        let mut b = vec![0f32; 64];
        fill_uniform_like(&mut a, -1.0, 2.0, opts, 7, None);
        fill_uniform_like(&mut b, -1.0, 2.0, opts, 7, None);
        assert_eq!(a, b);
        fill_normal_like(&mut a, 0.5, 2.0, opts, 7, None);
        fill_normal_like(&mut b, 0.5, 2.0, opts, 7, None);
        assert_eq!(a, b);
        assert_ne!(a, vec![0.0; 64]);
    }

    #[test]
    fn bnns_normal_reuses_pair_within_fill() {
        let mut paired = [0.0; 4];
        let mut singles = [0.0; 4];
        BnnsAesCtr128::new(42).fill_normal(&mut paired, 0.0, 1.0);
        let mut rng = BnnsAesCtr128::new(42);
        for v in &mut singles {
            *v = rng.normal();
        }
        // Four scalar `normal()` calls keep the spare across calls; one fill of
        // four does the same pairing within the fill.
        assert_eq!(paired, singles);
    }

    #[test]
    fn bnns_gumbel_alpha_beta_matches_documented_formula() {
        let u = 0.5f32;
        let g = bnns_gumbel_from_uniform(u, 2.0, 0.1);
        let expected = -((-(2.0 * u + 0.1).ln() + 0.1).ln());
        assert!((g - expected).abs() < 1e-6);
        // α=1, β=0 recovers Espresso ICDF (with ε clamp).
        assert_eq!(
            bnns_gumbel_from_uniform(u, 1.0, 0.0),
            gumbel01_from_uniform(u)
        );
    }

    #[test]
    fn bnns_categorical_matches_inverse_cdf_on_uniforms() {
        let probs = [0.1f32, 0.2, 0.3, 0.4];
        let n = 256;
        let mut uniforms = vec![0.0; n];
        let mut cats = vec![0.0; n];
        BnnsAesCtr128::new(42).fill_uniform(&mut uniforms, 0.0, 1.0);
        BnnsAesCtr128::new(42).fill_categorical(&mut cats, &probs, false);
        for (i, (&u, &c)) in uniforms.iter().zip(cats.iter()).enumerate() {
            assert_eq!(
                c as usize,
                bnns_categorical_from_uniform(u, &probs),
                "categorical mismatch at {i}"
            );
        }
        let mut log_cats = vec![0.0; n];
        let log_probs: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        BnnsAesCtr128::new(42).fill_categorical(&mut log_cats, &log_probs, true);
        assert_eq!(cats, log_cats);
    }

    #[test]
    fn bnns_uniform_int_stays_in_half_open_range() {
        let mut out = [0i64; 4096];
        BnnsAesCtr128::new(42).fill_uniform_int(&mut out, -3, 10);
        assert!(out.iter().all(|&v| (-3..10).contains(&v)));
        assert_eq!(bnns_uniform_int_from_u32(0, 0, 1), 0);
        assert_eq!(bnns_uniform_int_from_u32(u32::MAX, 0, 1), 0);
        assert_ne!(
            {
                let mut a = [0i64; 64];
                BnnsAesCtr128::new(1).fill_uniform_int(&mut a, 0, 100);
                a
            },
            {
                let mut b = [0i64; 64];
                BnnsAesCtr128::new(2).fill_uniform_int(&mut b, 0, 100);
                b
            }
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_bnns_gumbel_replays_seeded_stream() {
        let logits = [0.0f32; 256];
        let draw = || {
            let mut sampler = NativeBnnsGumbelMax::new(16_807).unwrap();
            (0..8)
                .map(|_| sampler.argmax(&logits, 1.0, 0.01).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(), draw());
    }

    #[cfg(target_os = "macos")]
    mod apple_bnns_parity {
        use super::*;

        #[repr(C)]
        struct BnnsNdArrayDescriptor {
            flags: u32,
            layout: u32,
            size: [usize; 8],
            stride: [usize; 8],
            data: *mut core::ffi::c_void,
            data_type: u32,
            table_data: *mut core::ffi::c_void,
            table_data_type: u32,
            data_scale: f32,
            data_bias: f32,
        }

        #[link(name = "Accelerate", kind = "framework")]
        unsafe extern "C" {
            fn BNNSCreateRandomGenerator(
                method: u32,
                filter_params: *const core::ffi::c_void,
            ) -> *mut core::ffi::c_void;
            fn BNNSCreateRandomGeneratorWithSeed(
                method: u32,
                seed: u64,
                filter_params: *const core::ffi::c_void,
            ) -> *mut core::ffi::c_void;
            fn BNNSDestroyRandomGenerator(generator: *mut core::ffi::c_void);
            fn BNNSRandomGeneratorStateSize(generator: *mut core::ffi::c_void) -> usize;
            fn BNNSRandomGeneratorGetState(
                generator: *mut core::ffi::c_void,
                size: usize,
                state: *mut u8,
            ) -> i32;
            fn BNNSRandomFillUniformFloat(
                generator: *mut core::ffi::c_void,
                desc: *mut BnnsNdArrayDescriptor,
                a: f32,
                b: f32,
            ) -> i32;
            fn BNNSRandomFillNormalFloat(
                generator: *mut core::ffi::c_void,
                desc: *mut BnnsNdArrayDescriptor,
                mean: f32,
                stddev: f32,
            ) -> i32;
            fn BNNSRandomFillCategoricalFloat(
                generator: *mut core::ffi::c_void,
                desc: *const BnnsNdArrayDescriptor,
                probabilities: *const BnnsNdArrayDescriptor,
                log_probabilities: bool,
            ) -> i32;
        }

        const BNNS_METHOD_AES_CTR: u32 = 0;
        const BNNS_LAYOUT_VECTOR: u32 = 0x1_0000;
        const BNNS_FLOAT32: u32 = 0x1_0000 | 32;

        fn apple_desc(data: *mut f32, n: usize) -> BnnsNdArrayDescriptor {
            BnnsNdArrayDescriptor {
                flags: 0,
                layout: BNNS_LAYOUT_VECTOR,
                size: [n, 0, 0, 0, 0, 0, 0, 0],
                stride: [0; 8],
                data: data.cast(),
                data_type: BNNS_FLOAT32,
                table_data: core::ptr::null_mut(),
                table_data_type: 0,
                data_scale: 1.0,
                data_bias: 0.0,
            }
        }

        fn apple_fill_uniform(seed: u64, out: &mut [f32], lo: f32, hi: f32) {
            unsafe {
                let generator =
                    BNNSCreateRandomGeneratorWithSeed(BNNS_METHOD_AES_CTR, seed, core::ptr::null());
                assert!(
                    !generator.is_null(),
                    "BNNSCreateRandomGeneratorWithSeed failed"
                );
                let mut desc = apple_desc(out.as_mut_ptr(), out.len());
                let rc = BNNSRandomFillUniformFloat(generator, &mut desc, lo, hi);
                BNNSDestroyRandomGenerator(generator);
                assert_eq!(rc, 0, "BNNSRandomFillUniformFloat failed");
            }
        }

        fn apple_fill_normal(seed: u64, out: &mut [f32], mean: f32, stddev: f32) {
            unsafe {
                let generator =
                    BNNSCreateRandomGeneratorWithSeed(BNNS_METHOD_AES_CTR, seed, core::ptr::null());
                assert!(!generator.is_null());
                let mut desc = apple_desc(out.as_mut_ptr(), out.len());
                let rc = BNNSRandomFillNormalFloat(generator, &mut desc, mean, stddev);
                BNNSDestroyRandomGenerator(generator);
                assert_eq!(rc, 0, "BNNSRandomFillNormalFloat failed");
            }
        }

        fn apple_fill_categorical(
            seed: u64,
            out: &mut [f32],
            probabilities: &[f32],
            log_probabilities: bool,
        ) {
            unsafe {
                let generator =
                    BNNSCreateRandomGeneratorWithSeed(BNNS_METHOD_AES_CTR, seed, core::ptr::null());
                assert!(!generator.is_null());
                let mut probs = probabilities.to_vec();
                let mut odesc = apple_desc(out.as_mut_ptr(), out.len());
                let mut pdesc = apple_desc(probs.as_mut_ptr(), probs.len());
                let rc =
                    BNNSRandomFillCategoricalFloat(generator, &odesc, &pdesc, log_probabilities);
                let _ = (&mut odesc, &mut pdesc);
                BNNSDestroyRandomGenerator(generator);
                assert_eq!(rc, 0, "BNNSRandomFillCategoricalFloat failed");
            }
        }

        fn max_ulps(a: &[f32], b: &[f32]) -> u32 {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| x.to_bits().abs_diff(y.to_bits()))
                .max()
                .unwrap_or(0)
        }

        #[test]
        fn batch_matches_apple_bnns_across_seeds_sizes_ranges() {
            let seeds = [
                0u64,
                1,
                42,
                7,
                0xffff_ffff_ffff_ffff,
                0x1234_5678_9abc_def0,
                0xa5a5_5a5a_c3c3_3c3c,
            ];
            let lengths = [1usize, 3, 4, 5, 16, 17, 64, 65, 256, 257, 1024, 4096];
            let ranges = [(0.0, 1.0), (-2.0, 3.0), (0.25, 0.75), (-1e2, 1e2)];

            for &seed in &seeds {
                for &n in &lengths {
                    for &(lo, hi) in &ranges {
                        let mut apple = vec![0.0; n];
                        let mut ours = vec![0.0; n];
                        apple_fill_uniform(seed, &mut apple, lo, hi);
                        BnnsAesCtr128::new(seed).fill_uniform(&mut ours, lo, hi);
                        assert_eq!(
                            ours.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            apple.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "BNNS parity failed seed={seed} n={n} range=[{lo},{hi}]"
                        );
                    }
                }
            }
        }

        #[test]
        fn batch_gumbel_from_apple_uniforms() {
            let n = 8192;
            let seed = 12345u64;
            let mut apple_u = vec![0.0; n];
            apple_fill_uniform(seed, &mut apple_u, 0.0, 1.0);

            let mut ours_g = vec![0.0; n];
            BnnsAesCtr128::new(seed).fill_gumbel01(&mut ours_g);

            for (i, (&u, &g)) in apple_u.iter().zip(ours_g.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    gumbel01_from_uniform(u).to_bits(),
                    "Gumbel from Apple uniform mismatch at {i}"
                );
            }
        }

        #[test]
        fn batch_normal_matches_apple_box_muller_pairing() {
            // Short fills stay within a few ULPs of Apple's scalar Box–Muller.
            // Longer fills still start on the same AES pair (first samples close)
            // even when Apple's vector path drifts further in later lanes.
            for seed in [42u64, 7, 0xdead_beef] {
                for &n in &[1usize, 2, 3, 4] {
                    let mut apple = vec![0.0; n];
                    let mut ours = vec![0.0; n];
                    apple_fill_normal(seed, &mut apple, 0.0, 1.0);
                    BnnsAesCtr128::new(seed).fill_normal(&mut ours, 0.0, 1.0);
                    assert!(
                        max_ulps(&ours, &apple) <= 16,
                        "short normal ULP seed={seed} n={n}"
                    );
                }
                let mut apple = vec![0.0; 64];
                let mut ours = vec![0.0; 64];
                apple_fill_normal(seed, &mut apple, 0.0, 1.0);
                BnnsAesCtr128::new(seed).fill_normal(&mut ours, 0.0, 1.0);
                assert!(
                    max_ulps(&ours[..4], &apple[..4]) <= 16,
                    "leading-pair ULP seed={seed}"
                );
            }
        }

        #[test]
        fn batch_categorical_matches_apple_bit_exact() {
            let probs = [0.1f32, 0.2, 0.3, 0.4];
            for seed in [42u64, 7, 12345] {
                for &n in &[1usize, 8, 16, 64, 256, 1024] {
                    let mut apple = vec![0.0; n];
                    let mut ours = vec![0.0; n];
                    apple_fill_categorical(seed, &mut apple, &probs, false);
                    BnnsAesCtr128::new(seed).fill_categorical(&mut ours, &probs, false);
                    assert_eq!(ours, apple, "categorical seed={seed} n={n}");

                    let log_probs: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
                    let mut apple_log = vec![0.0; n];
                    let mut ours_log = vec![0.0; n];
                    apple_fill_categorical(seed, &mut apple_log, &log_probs, true);
                    BnnsAesCtr128::new(seed).fill_categorical(&mut ours_log, &log_probs, true);
                    assert_eq!(ours_log, apple_log, "log-categorical seed={seed} n={n}");
                }
            }
        }

        #[test]
        fn entropy_generator_state_replays_via_seed() {
            unsafe {
                let generator = BNNSCreateRandomGenerator(BNNS_METHOD_AES_CTR, core::ptr::null());
                assert!(!generator.is_null());
                let n = BNNSRandomGeneratorStateSize(generator);
                assert_eq!(n, BNNS_AES_CTR_STATE_SIZE);
                let mut state = vec![0u8; n];
                assert_eq!(
                    BNNSRandomGeneratorGetState(generator, n, state.as_mut_ptr()),
                    0
                );
                assert!(bnns_generator_state_is_fresh(&state));
                let seed = bnns_seed_from_generator_state(&state).unwrap();

                let mut apple = [0.0f32; 16];
                let mut desc = BnnsNdArrayDescriptor {
                    flags: 0,
                    layout: BNNS_LAYOUT_VECTOR,
                    size: [apple.len(), 0, 0, 0, 0, 0, 0, 0],
                    stride: [0; 8],
                    data: apple.as_mut_ptr().cast(),
                    data_type: BNNS_FLOAT32,
                    table_data: core::ptr::null_mut(),
                    table_data_type: 0,
                    data_scale: 1.0,
                    data_bias: 0.0,
                };
                assert_eq!(
                    BNNSRandomFillUniformFloat(generator, &mut desc, 0.0, 1.0),
                    0
                );
                BNNSDestroyRandomGenerator(generator);

                let mut ours = [0.0f32; 16];
                BnnsAesCtr128::try_from_fresh_generator_state(&state)
                    .unwrap()
                    .fill_uniform(&mut ours, 0.0, 1.0);
                assert_eq!(ours.map(f32::to_bits), apple.map(f32::to_bits));

                // Same seed via WithSeed must also match.
                let mut with_seed = [0.0f32; 16];
                apple_fill_uniform(seed, &mut with_seed, 0.0, 1.0);
                assert_eq!(with_seed.map(f32::to_bits), apple.map(f32::to_bits));
            }
        }
    }
}
