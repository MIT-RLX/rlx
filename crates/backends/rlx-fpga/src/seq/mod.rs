// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequential-engine FPGA target: rlx-ir → microcoded RTL.
//!
//! The [`crate::model`] path emits a module per layer, which suits a
//! feed-forward INT8 classifier and nothing else. This one emits a single
//! datapath plus a descriptor ROM, which covers any graph whose stages reduce
//! to "bias, then a dot product over an address pattern, then requantise and
//! activate" — including recurrent ones, because state is just another tensor
//! in the activation RAM.
//!
//! What that buys, concretely: one MAC and one activation LUT for the whole
//! network regardless of depth, and recurrence for free. What it costs: the
//! network runs one MAC at a time, so it suits kHz-rate models (VAD, keyword
//! spotting, sensor classification) rather than image backbones.
//!
//! ```no_run
//! # use rlx_fpga::seq::{SeqConfig, export_graph_seq};
//! # fn f(graph: &rlx_ir::Graph, params: &[(String, Vec<f32>)]) -> Result<(), String> {
//! let cfg = SeqConfig::default().with_carry([
//!     ("state.h1", 1), ("state.c1", 2), ("state.h2", 3), ("state.c2", 4),
//! ]);
//! export_graph_seq(graph, params, &cfg, std::path::Path::new("out"))?;
//! # Ok(()) }
//! ```

mod emit;
mod lower;

pub use emit::emit;
pub use lower::lower_graph;

use std::path::Path;

use rlx_ir::Graph;

/// Descriptor opcode. Values are ABI: the generated `tv_ucode.svh` and
/// `rlx_seq_core.sv` agree on these numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SeqOp {
    /// bias + Σ act·weight, requantise, activate, store.
    MatVec = 0,
    /// Max over a tap window.
    Pool = 1,
    /// LSTM gating: `c = f·c + i·g`, `h = o·tanh(c)`.
    Gate = 2,
    /// End of program.
    Done = 3,
}

/// Activation applied to a descriptor's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SeqAct {
    None = 0,
    Relu = 1,
    Sigmoid = 2,
}

/// One stage of the network.
///
/// Addresses are `base + oo·s*_o + ii·s*_i (+ tap)`, where `oo`/`ii` are the
/// two output loops and the tap term is either `t·s*_t` or a table lookup.
/// Everything is an adder chain — no multipliers in the address path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Descriptor {
    pub op: u32,
    pub act: u32,
    pub n_o: u32,
    pub n_i: u32,
    pub n_tap: u32,
    pub base_a: u32,
    pub sa_o: u32,
    pub sa_i: u32,
    pub sa_t: u32,
    pub off_ptr: u32,
    pub use_off: u32,
    pub base_w: u32,
    pub sw_o: u32,
    pub sw_i: u32,
    pub sw_t: u32,
    pub base_b: u32,
    pub sb_o: u32,
    pub sb_i: u32,
    pub has_bias: u32,
    pub b_shift: u32,
    pub w_frac: u32,
    pub base_d: u32,
    pub sd_o: u32,
    pub sd_i: u32,
    pub base_c: u32,
}

impl Descriptor {
    /// Field order used by both the generated header and the RTL.
    pub const FIELDS: [&'static str; 25] = [
        "op", "act", "n_o", "n_i", "n_tap", "base_a", "sa_o", "sa_i", "sa_t", "off_ptr", "use_off",
        "base_w", "sw_o", "sw_i", "sw_t", "base_b", "sb_o", "sb_i", "has_bias", "b_shift",
        "w_frac", "base_d", "sd_o", "sd_i", "base_c",
    ];

    pub fn field(&self, name: &str) -> u32 {
        match name {
            "op" => self.op,
            "act" => self.act,
            "n_o" => self.n_o,
            "n_i" => self.n_i,
            "n_tap" => self.n_tap,
            "base_a" => self.base_a,
            "sa_o" => self.sa_o,
            "sa_i" => self.sa_i,
            "sa_t" => self.sa_t,
            "off_ptr" => self.off_ptr,
            "use_off" => self.use_off,
            "base_w" => self.base_w,
            "sw_o" => self.sw_o,
            "sw_i" => self.sw_i,
            "sw_t" => self.sw_t,
            "base_b" => self.base_b,
            "sb_o" => self.sb_o,
            "sb_i" => self.sb_i,
            "has_bias" => self.has_bias,
            "b_shift" => self.b_shift,
            "w_frac" => self.w_frac,
            "base_d" => self.base_d,
            "sd_o" => self.sd_o,
            "sd_i" => self.sd_i,
            "base_c" => self.base_c,
            other => panic!("unknown descriptor field {other}"),
        }
    }
}

/// A lowered graph, ready to emit.
#[derive(Debug, Clone)]
pub struct SeqModel {
    pub descriptors: Vec<Descriptor>,
    /// int16 weights, referenced by `base_w` / `base_b`.
    pub weights: Vec<i16>,
    /// Tap offset tables, referenced by `off_ptr`.
    pub offsets: Vec<u16>,
    /// Budget from [`SeqConfig`], not what the graph needs.
    pub aram_words: usize,
    /// Words actually reached — what the emitted RTL sizes its RAM to.
    pub aram_used: usize,
    pub feat_base: usize,
    pub feat_len: usize,
    pub prob_addr: usize,
    pub cfg: SeqConfig,
    /// Per-stage note, for the emitted header's comments.
    pub labels: Vec<String>,
}

impl SeqModel {
    /// MACs per inference — the dominant term in the cycle count.
    pub fn macs(&self) -> u64 {
        self.descriptors
            .iter()
            .filter(|d| d.op == SeqOp::MatVec as u32)
            .map(|d| u64::from(d.n_o.max(1)) * u64::from(d.n_i) * u64::from(d.n_tap))
            .sum()
    }
}

/// Numeric and layout choices for the emitted datapath.
#[derive(Debug, Clone)]
pub struct SeqConfig {
    /// Fractional bits in an activation word.
    pub act_frac: u32,
    /// LUT interval count. Kept a power of two so the index is a shift.
    pub lut_n: usize,
    /// LUT domain `[0, lut_hi]`. Too small a value silently costs accuracy:
    /// LSTM pre-activations here reach 177, and clamping at 8 rather than 16
    /// cost 6.9e-3.
    pub lut_hi: f32,
    /// Activation RAM depth in 32-bit words.
    pub aram_words: usize,
    /// `(input name, graph output index)` for state carried across inferences.
    /// The named input and that output share storage, so the engine updates
    /// state in place.
    pub carry: Vec<(String, usize)>,
    pub module_prefix: String,
}

impl Default for SeqConfig {
    fn default() -> Self {
        Self {
            act_frac: 15,
            lut_n: 1024,
            lut_hi: 16.0,
            aram_words: 4096,
            carry: Vec::new(),
            module_prefix: "tv".into(),
        }
    }
}

impl SeqConfig {
    pub fn with_carry<I, S>(mut self, pairs: I) -> Self
    where
        I: IntoIterator<Item = (S, usize)>,
        S: Into<String>,
    {
        self.carry = pairs.into_iter().map(|(n, i)| (n.into(), i)).collect();
        self
    }

    /// `log2` of the activation units per LUT interval — the shift that turns
    /// a Q`act_frac` value into a table index.
    pub fn lut_shift(&self) -> u32 {
        let per = (1u64 << self.act_frac) as f64 * f64::from(self.lut_hi) / self.lut_n as f64;
        per.log2().round() as u32
    }
}

/// Quantise to int16 with a power-of-two scale, returning `(values, frac_bits)`.
///
/// The scale must round *up* in frac bits, i.e. the shift is chosen so the
/// largest magnitude still fits — rounding the other way clips the tail and
/// costs three orders of magnitude of accuracy.
pub fn quantise_pow2(v: &[f32]) -> (Vec<i16>, u32) {
    let max = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let frac = if max > 0.0 {
        (32767.0 / max).log2().floor().clamp(0.0, 30.0) as u32
    } else {
        0
    };
    let scale = f64::from(1u32 << frac.min(30));
    let q = v
        .iter()
        .map(|&x| (f64::from(x) * scale).round().clamp(-32768.0, 32767.0) as i16)
        .collect();
    (q, frac)
}

/// Lower `graph` and write the RTL under `out_dir`.
pub fn export_graph_seq(
    graph: &Graph,
    params: &[(String, Vec<f32>)],
    cfg: &SeqConfig,
    out_dir: &Path,
) -> Result<SeqModel, String> {
    let model = lower_graph(graph, params, cfg)?;
    emit(&model, out_dir)?;
    Ok(model)
}
