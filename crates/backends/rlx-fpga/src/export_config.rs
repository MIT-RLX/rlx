// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Export configuration for FPGA / SystemVerilog emission.
//!
//! Separates **datapath tuning** ([`crate::tune::Tune`]) from the
//! **hardware / synthesis target** ([`HwTarget`]). The default is
//! target-agnostic SystemVerilog with soft ports (`clk`/`rst`/`start`/…)
//! and a generic Yosys script — no board pins, no vendor lock-in.

use std::fmt;
use std::path::PathBuf;

use crate::from_graph::FromGraphOptions;
use crate::legalize::{ExportQuantMode, LegalizeOptions};
use crate::tune::{OptTarget, Tune};

/// FPGA / ASIC synthesis family for optional constraint + script emit.
///
/// The RTL itself (`top.sv` + layers) is always **target-agnostic**: soft
/// ports only. Board-specific pinouts live in optional sidecar files
/// (`constraints.lpf`, `synth.sh`) when a concrete [`HwTarget`] is chosen.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum HwTarget {
    /// Target-agnostic SystemVerilog + generic Yosys `synth` script.
    /// Soft ports; no pin constraints. **Default.**
    #[default]
    Generic,
    /// Lattice ECP5 via `yosys synth_ecp5` → `nextpnr-ecp5` → `ecppack`.
    Ecp5 {
        /// Device size token, e.g. `"25k"`, `"45k"`, `"85k"`.
        device: String,
        /// Package, e.g. `"CABGA381"`.
        package: String,
    },
    /// Lattice iCE40 via `yosys synth_ice40` → `nextpnr-ice40` → `icepack`.
    Ice40 { device: String, package: String },
    /// Xilinx 7-series via open XC7 flow (`synth_xilinx` / nextpnr-xilinx).
    /// Experimental — scripts are emitted; P&R success depends on the
    /// installed open toolchain.
    Xilinx7 {
        /// Part name, e.g. `"xc7a35tcsg324-1"`.
        part: String,
    },
}

impl HwTarget {
    /// Soft-port, board-agnostic RTL (default).
    pub fn generic() -> Self {
        Self::Generic
    }

    /// Common ECP5-25k / CABGA381 (ULX3S-class) preset.
    pub fn ecp5_25k() -> Self {
        Self::Ecp5 {
            device: "25k".into(),
            package: "CABGA381".into(),
        }
    }

    /// iCE40 UP5K / SG48 preset.
    pub fn ice40_up5k() -> Self {
        Self::Ice40 {
            device: "up5k".into(),
            package: "sg48".into(),
        }
    }

    /// Parse CLI / config strings: `generic`, `ecp5`, `ecp5:25k:CABGA381`,
    /// `ice40`, `ice40:up5k:sg48`, `xilinx7:xc7a35tcsg324-1`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        let lower = s.to_ascii_lowercase();
        match lower.as_str() {
            "generic" | "agnostic" | "soft" => Ok(Self::Generic),
            "ecp5" => Ok(Self::ecp5_25k()),
            "ice40" => Ok(Self::ice40_up5k()),
            _ => {
                let mut parts = s.split(':');
                let kind = parts.next().unwrap_or("").to_ascii_lowercase();
                match kind.as_str() {
                    "ecp5" => {
                        let device = parts.next().unwrap_or("25k").to_string();
                        let package = parts.next().unwrap_or("CABGA381").to_string();
                        Ok(Self::Ecp5 { device, package })
                    }
                    "ice40" => {
                        let device = parts.next().unwrap_or("up5k").to_string();
                        let package = parts.next().unwrap_or("sg48").to_string();
                        Ok(Self::Ice40 { device, package })
                    }
                    "xilinx7" | "xc7" => {
                        let part = parts
                            .next()
                            .ok_or_else(|| {
                                "xilinx7 requires a part, e.g. xilinx7:xc7a35tcsg324-1".to_string()
                            })?
                            .to_string();
                        Ok(Self::Xilinx7 { part })
                    }
                    _ => Err(format!(
                        "unknown HwTarget {s:?}; expected generic | ecp5[:device:pkg] | \
                         ice40[:device:pkg] | xilinx7:<part>"
                    )),
                }
            }
        }
    }

    /// Human-readable tag for directory names / banners.
    pub fn tag(&self) -> String {
        match self {
            Self::Generic => "generic".into(),
            Self::Ecp5 { device, package } => format!("ecp5_{device}_{package}"),
            Self::Ice40 { device, package } => format!("ice40_{device}_{package}"),
            Self::Xilinx7 { part } => format!("xilinx7_{part}"),
        }
    }

    /// Whether this target wants a pin-constraint sidecar.
    pub fn emits_constraints(&self) -> bool {
        !matches!(self, Self::Generic)
    }
}

impl fmt::Display for HwTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Generic => write!(f, "generic (target-agnostic)"),
            Self::Ecp5 { device, package } => write!(f, "ecp5 {device} {package}"),
            Self::Ice40 { device, package } => write!(f, "ice40 {device} {package}"),
            Self::Xilinx7 { part } => write!(f, "xilinx7 {part}"),
        }
    }
}

/// What the soft `top` presents as its primary result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputKind {
    /// Single-byte class index from Argmax (default TinyConv path).
    #[default]
    Argmax,
    /// Last Dense logits only — no Argmax layer; `pred` is logits[0] for TB
    /// compatibility, and `logits_len` is noted in EXPORT.md.
    Logits,
}

/// Soft-port signal names for `top.sv` (ASIC / SoC integration).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortNames {
    pub clk: String,
    pub rst: String,
    pub start: String,
    pub done: String,
    pub in_addr: String,
    pub in_we: String,
    pub in_din: String,
    pub out_addr: String,
    pub out_re: String,
    pub out_dout: String,
    pub pred: String,
    pub in_valid: String,
    pub in_ready: String,
    pub in_data: String,
    pub out_valid: String,
    pub out_ready: String,
    pub out_data: String,
}

impl Default for PortNames {
    fn default() -> Self {
        Self {
            clk: "clk".into(),
            rst: "rst".into(),
            start: "start".into(),
            done: "done".into(),
            in_addr: "in_addr".into(),
            in_we: "in_we".into(),
            in_din: "in_din".into(),
            out_addr: "out_addr".into(),
            out_re: "out_re".into(),
            out_dout: "out_dout".into(),
            pred: "pred".into(),
            in_valid: "in_valid".into(),
            in_ready: "in_ready".into(),
            in_data: "in_data".into(),
            out_valid: "out_valid".into(),
            out_ready: "out_ready".into(),
            out_data: "out_data".into(),
        }
    }
}

/// How the host loads the activation input buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputIface {
    /// Byte poke via `in_addr` / `in_we` / `in_din` (default).
    #[default]
    Memory,
    /// Streaming `valid`/`ready` + packed INT8 beats.
    Stream {
        /// Elements (bytes) per beat; must be a power of two ≥ 1.
        beat_elems: u8,
    },
    /// Both memory poke and streaming load.
    MemoryAndStream { beat_elems: u8 },
}

impl InputIface {
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "memory" | "mem" | "poke" => Ok(Self::Memory),
            "stream" => Ok(Self::Stream { beat_elems: 1 }),
            "both" | "memory+stream" | "mem+stream" => Ok(Self::MemoryAndStream { beat_elems: 1 }),
            other if other.starts_with("stream:") => {
                let n: u8 = other[7..].parse().map_err(|_| {
                    format!("bad stream beat width in {other:?}; expected stream:N")
                })?;
                Ok(Self::Stream {
                    beat_elems: n.max(1),
                })
            }
            other if other.starts_with("both:") => {
                let n: u8 = other[5..]
                    .parse()
                    .map_err(|_| format!("bad both beat width in {other:?}; expected both:N"))?;
                Ok(Self::MemoryAndStream {
                    beat_elems: n.max(1),
                })
            }
            _ => Err(format!(
                "unknown InputIface {s:?}; expected memory | stream | both \
                 [|stream:N|both:N]"
            )),
        }
    }

    pub fn wants_memory(self) -> bool {
        matches!(self, Self::Memory | Self::MemoryAndStream { .. })
    }

    pub fn wants_stream(self) -> bool {
        matches!(self, Self::Stream { .. } | Self::MemoryAndStream { .. })
    }

    pub fn beat_elems(self) -> u8 {
        match self {
            Self::Memory => 1,
            Self::Stream { beat_elems } | Self::MemoryAndStream { beat_elems } => beat_elems.max(1),
        }
    }
}

/// How the host observes the primary result / last activation buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputIface {
    /// Single-byte `pred` (default for Argmax).
    #[default]
    ScalarPred,
    /// Memory readout via `out_addr` / `out_re` / `out_dout`.
    MemoryReadout,
    /// Streaming `valid`/`ready` + packed INT8 beats after `done`.
    Stream { beat_elems: u8 },
    /// Scalar `pred` plus full memory readout (default for Logits).
    ScalarAndMemory,
    /// Memory readout plus streaming.
    MemoryAndStream { beat_elems: u8 },
}

impl OutputIface {
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "scalar" | "pred" | "argmax" => Ok(Self::ScalarPred),
            "memory" | "mem" | "readout" => Ok(Self::MemoryReadout),
            "stream" => Ok(Self::Stream { beat_elems: 1 }),
            "scalar+memory" | "scalar_and_memory" | "both_scalar" => Ok(Self::ScalarAndMemory),
            "both" | "memory+stream" | "mem+stream" => Ok(Self::MemoryAndStream { beat_elems: 1 }),
            other if other.starts_with("stream:") => {
                let n: u8 = other[7..].parse().map_err(|_| {
                    format!("bad stream beat width in {other:?}; expected stream:N")
                })?;
                Ok(Self::Stream {
                    beat_elems: n.max(1),
                })
            }
            other if other.starts_with("both:") => {
                let n: u8 = other[5..]
                    .parse()
                    .map_err(|_| format!("bad both beat width in {other:?}; expected both:N"))?;
                Ok(Self::MemoryAndStream {
                    beat_elems: n.max(1),
                })
            }
            _ => Err(format!(
                "unknown OutputIface {s:?}; expected scalar | memory | stream | \
                 scalar+memory | both [|stream:N|both:N]"
            )),
        }
    }

    pub fn wants_memory(self) -> bool {
        matches!(
            self,
            Self::MemoryReadout | Self::ScalarAndMemory | Self::MemoryAndStream { .. }
        )
    }

    pub fn wants_stream(self) -> bool {
        matches!(self, Self::Stream { .. } | Self::MemoryAndStream { .. })
    }

    /// Emit the scalar `pred` port (addr-0 peek and/or class index).
    pub fn wants_pred_port(self) -> bool {
        !matches!(self, Self::MemoryReadout)
    }

    pub fn beat_elems(self) -> u8 {
        match self {
            Self::Stream { beat_elems } | Self::MemoryAndStream { beat_elems } => beat_elems.max(1),
            _ => 1,
        }
    }
}

/// Bind Graph tensor names to the soft datapath ports.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GraphIoBind {
    /// Graph `Op::Input` name; `None` = first input.
    pub input: Option<String>,
    /// Graph output / layer names. Empty = default single `g.outputs[0]`.
    /// First entry is the primary result; extras become additional readout banks.
    pub outputs: Vec<String>,
    /// Extra scalar / small `Op::Input` names exposed as soft sideband ports
    /// (not part of the MAC datapath). Widths come from the tensor shape.
    pub sideband_inputs: Vec<String>,
}

impl GraphIoBind {
    pub fn with_input(mut self, name: impl Into<String>) -> Self {
        self.input = Some(name.into());
        self
    }

    pub fn with_outputs<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.outputs = names.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_sideband_inputs<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.sideband_inputs = names.into_iter().map(Into::into).collect();
        self
    }
}

/// Soft scalar / status channel on `top` — not part of the activation BRAM path.
///
/// Typical uses: temperature, batch id, mode flags. The host drives the input
/// port; the value is sampled when [`PortNames::start`] asserts and (by default)
/// echoed on `{name}_q` for SoC observability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidebandSpec {
    /// Port stem (sanitized to a SV identifier).
    pub name: String,
    /// Bit width in `1..=64`.
    pub bits: u8,
    /// Emit as `signed` packed logic.
    pub signed: bool,
    /// Also emit `{name}_q` output holding the value captured at `start`.
    pub echo: bool,
}

impl SidebandSpec {
    /// Unsigned input sideband of `bits` width with echo output.
    pub fn input(name: impl Into<String>, bits: u8) -> Self {
        Self {
            name: name.into(),
            bits: bits.clamp(1, 64),
            signed: false,
            echo: true,
        }
    }

    /// Signed input sideband (e.g. INT8 sensor).
    pub fn signed_input(name: impl Into<String>, bits: u8) -> Self {
        Self {
            name: name.into(),
            bits: bits.clamp(1, 64),
            signed: true,
            echo: true,
        }
    }

    pub fn with_echo(mut self, echo: bool) -> Self {
        self.echo = echo;
        self
    }

    /// Parse `name`, `name:BITS`, or `name:BITS:signed` / `name:BITS:u`.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s.is_empty() {
            return Err("empty sideband spec".into());
        }
        let mut parts = s.split(':');
        let name = parts.next().unwrap_or("").trim().to_string();
        if name.is_empty() {
            return Err(format!("bad sideband {s:?}; expected name[:bits[:signed]]"));
        }
        let bits: u8 = match parts.next() {
            None => 8,
            Some(b) => b
                .parse()
                .map_err(|_| format!("bad sideband width in {s:?}"))?,
        };
        let mut signed = false;
        if let Some(flag) = parts.next() {
            match flag.trim().to_ascii_lowercase().as_str() {
                "s" | "signed" | "i" => signed = true,
                "u" | "unsigned" => signed = false,
                other => {
                    return Err(format!(
                        "bad sideband signedness {other:?}; expected signed|unsigned"
                    ));
                }
            }
        }
        Ok(Self {
            name,
            bits: bits.clamp(1, 64),
            signed,
            echo: true,
        })
    }
}

/// Soft I/O configuration for target-agnostic SystemVerilog export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoConfig {
    pub names: PortNames,
    pub input: InputIface,
    pub output: OutputIface,
    pub bind: GraphIoBind,
    /// Scalar sideband ports (temperature, batch id, …).
    pub sidebands: Vec<SidebandSpec>,
}

impl Default for IoConfig {
    fn default() -> Self {
        Self {
            names: PortNames::default(),
            input: InputIface::Memory,
            output: OutputIface::ScalarPred,
            bind: GraphIoBind::default(),
            sidebands: Vec::new(),
        }
    }
}

impl IoConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_names(mut self, names: PortNames) -> Self {
        self.names = names;
        self
    }

    pub fn with_input(mut self, iface: InputIface) -> Self {
        self.input = iface;
        self
    }

    pub fn with_output(mut self, iface: OutputIface) -> Self {
        self.output = iface;
        self
    }

    pub fn with_bind(mut self, bind: GraphIoBind) -> Self {
        self.bind = bind;
        self
    }

    pub fn with_sidebands(mut self, sidebands: Vec<SidebandSpec>) -> Self {
        self.sidebands = sidebands;
        self
    }

    pub fn sideband(mut self, spec: SidebandSpec) -> Self {
        self.sidebands.push(spec);
        self
    }
}

/// Full FPGA export configuration: tune + hardware target + Graph bindings.
#[derive(Debug, Clone)]
pub struct FpgaExportConfig {
    /// Datapath optimization preset / knobs.
    pub tune: Tune,
    /// Synthesis / board target. Default: [`HwTarget::Generic`].
    pub hw_target: HwTarget,
    /// Write `synth.sh` + `EXPORT.md` next to the RTL tree.
    pub emit_synth_scripts: bool,
    /// Write a stub constraint file when `hw_target` is board-specific.
    pub emit_constraints: bool,
    /// Emit `board_top.sv` shell for board targets.
    pub emit_board_shell: bool,
    /// PTQ mode when the Graph is still f32 (`Int8` / `Int4` / `Fp4`).
    pub quant_mode: ExportQuantMode,
    /// Weight / scale bindings for f32 → quantized legalization.
    pub legalize: LegalizeOptions,
    /// Prefer Argmax vs logits-only output.
    pub output_kind: OutputKind,
    /// Soft-port naming + memory/stream protocols + Graph tensor bind.
    pub io: IoConfig,
    /// Optional overrides when lowering an already-quantized Graph.
    pub from_graph: FromGraphOptions,
    /// Optional output directory hint (CLI / runtime may override).
    pub out_dir: Option<PathBuf>,
}

impl Default for FpgaExportConfig {
    fn default() -> Self {
        Self {
            tune: Tune::default(),
            hw_target: HwTarget::Generic,
            emit_synth_scripts: true,
            emit_constraints: true,
            emit_board_shell: true,
            quant_mode: ExportQuantMode::Int8,
            legalize: LegalizeOptions::default(),
            output_kind: OutputKind::Argmax,
            io: IoConfig::default(),
            from_graph: FromGraphOptions::default(),
            out_dir: None,
        }
    }
}

impl FpgaExportConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn for_opt_target(target: OptTarget) -> Self {
        Self {
            tune: Tune::for_target(target),
            ..Self::default()
        }
    }

    pub fn with_tune(mut self, tune: Tune) -> Self {
        self.tune = tune;
        self
    }

    pub fn with_hw_target(mut self, hw: HwTarget) -> Self {
        self.hw_target = hw;
        self
    }

    pub fn with_quant_mode(mut self, mode: ExportQuantMode) -> Self {
        self.quant_mode = mode;
        self.from_graph.weight_bits = mode.weight_bits();
        self.from_graph.weight_encoding = mode.encoding();
        self
    }

    pub fn with_legalize(mut self, opts: LegalizeOptions) -> Self {
        self.legalize = opts;
        self
    }

    pub fn with_output_kind(mut self, kind: OutputKind) -> Self {
        self.output_kind = kind;
        self.legalize.append_argmax = matches!(kind, OutputKind::Argmax);
        // Logits need a full-vector readout; upgrade the default scalar-only iface.
        if matches!(kind, OutputKind::Logits) && matches!(self.io.output, OutputIface::ScalarPred) {
            self.io.output = OutputIface::ScalarAndMemory;
        }
        self
    }

    pub fn with_io(mut self, io: IoConfig) -> Self {
        self.io = io;
        self
    }

    pub fn with_port_names(mut self, names: PortNames) -> Self {
        self.io.names = names;
        self
    }

    pub fn with_input_iface(mut self, iface: InputIface) -> Self {
        self.io.input = iface;
        self
    }

    pub fn with_output_iface(mut self, iface: OutputIface) -> Self {
        self.io.output = iface;
        self
    }

    pub fn bind_input(mut self, name: impl Into<String>) -> Self {
        self.io.bind.input = Some(name.into());
        self
    }

    pub fn bind_outputs<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.io.bind.outputs = names.into_iter().map(Into::into).collect();
        self
    }

    pub fn bind_sideband_inputs<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.io.bind.sideband_inputs = names.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_sidebands(mut self, sidebands: Vec<SidebandSpec>) -> Self {
        self.io.sidebands = sidebands;
        self
    }

    pub fn sideband(mut self, spec: SidebandSpec) -> Self {
        self.io.sidebands.push(spec);
        self
    }

    pub fn with_from_graph(mut self, opts: FromGraphOptions) -> Self {
        self.from_graph = opts;
        self
    }

    pub fn with_out_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out_dir = Some(dir.into());
        self
    }

    pub fn without_synth_scripts(mut self) -> Self {
        self.emit_synth_scripts = false;
        self
    }
}
