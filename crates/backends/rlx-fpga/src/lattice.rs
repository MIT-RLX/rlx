// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lattice boards for the open Yosys / nextpnr flows:
//!
//! * iCE40: `yosys synth_ice40` → `nextpnr-ice40` → `icepack`
//!   ([Project IceStorm](https://github.com/YosysHQ/icestorm)),
//! * ECP5: `yosys synth_ecp5` → `nextpnr-ecp5` → `ecppack`
//!   ([Project Trellis](https://github.com/YosysHQ/prjtrellis)),
//!
//! then `openFPGALoader`. A preset carries the device, package, clock, UART
//! and LED pins, so the export gets a UART host-link `board_top`, real
//! constraints and a `program.sh`, like the openXC7 boards
//! ([`crate::xc7`]). Pins are from the boards' published constraint files
//! (`icebreaker.pcf`, `ulx3s_v20.lpf`).

#[allow(unused_imports)]
use crate::prelude::*;

use core::fmt;

use crate::board::{Pll, ShellBoard};

/// Lattice FPGA family of a board.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LatticeFamily {
    /// iCE40 (UltraPlus, HX, LP).
    Ice40,
    /// ECP5.
    Ecp5,
}

/// Board pin assignment (LVCMOS33 throughout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatticePins {
    /// Oscillator input.
    pub clk: &'static str,
    /// Oscillator frequency, MHz.
    pub clk_mhz: u32,
    /// Reset button.
    pub rst: &'static str,
    /// The button reads 0 when pressed.
    pub rst_active_low: bool,
    /// Host → FPGA serial line (FPGA input).
    pub uart_rx: &'static str,
    /// FPGA → host serial line (FPGA output).
    pub uart_tx: &'static str,
    /// Four status LEDs.
    pub leds: [&'static str; 4],
    /// The LEDs light on 0.
    pub led_active_low: bool,
}

/// Supported Lattice boards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LatticeBoard {
    /// 1BitSquared iCEBreaker (iCE40UP5K-SG48). LEDs are the snap-off
    /// PMOD's LED1–LED4.
    Icebreaker,
    /// Radiona ULX3S with an LFE5U-12F.
    Ulx3s12,
    /// Radiona ULX3S with an LFE5U-25F.
    Ulx3s25,
    /// Radiona ULX3S with an LFE5U-45F.
    Ulx3s45,
    /// Radiona ULX3S with an LFE5U-85F.
    Ulx3s85,
}

impl LatticeBoard {
    /// Every preset.
    pub const ALL: [Self; 5] = [
        Self::Icebreaker,
        Self::Ulx3s12,
        Self::Ulx3s25,
        Self::Ulx3s45,
        Self::Ulx3s85,
    ];

    /// Canonical CLI name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Icebreaker => "icebreaker",
            Self::Ulx3s12 => "ulx3s-12f",
            Self::Ulx3s25 => "ulx3s-25f",
            Self::Ulx3s45 => "ulx3s-45f",
            Self::Ulx3s85 => "ulx3s-85f",
        }
    }

    /// Parse a board name (`icebreaker`, `ulx3s-85f`, `ulx3s` = 85F, …).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "icebreaker" | "ice-breaker" => Some(Self::Icebreaker),
            "ulx3s-12f" | "ulx3s-12" | "ulx3s12" => Some(Self::Ulx3s12),
            "ulx3s-25f" | "ulx3s-25" | "ulx3s25" => Some(Self::Ulx3s25),
            "ulx3s-45f" | "ulx3s-45" | "ulx3s45" => Some(Self::Ulx3s45),
            "ulx3s" | "ulx3s-85f" | "ulx3s-85" | "ulx3s85" => Some(Self::Ulx3s85),
            _ => None,
        }
    }

    /// FPGA family.
    pub fn family(self) -> LatticeFamily {
        match self {
            Self::Icebreaker => LatticeFamily::Ice40,
            _ => LatticeFamily::Ecp5,
        }
    }

    /// Device as nextpnr spells it (`--up5k`, `--85k`, …).
    pub fn device(self) -> &'static str {
        match self {
            Self::Icebreaker => "up5k",
            Self::Ulx3s12 => "12k",
            Self::Ulx3s25 => "25k",
            Self::Ulx3s45 => "45k",
            Self::Ulx3s85 => "85k",
        }
    }

    /// Package as nextpnr spells it.
    pub fn package(self) -> &'static str {
        match self {
            Self::Icebreaker => "sg48",
            _ => "CABGA381",
        }
    }

    /// `ecppack --idcode` for parts that share a die with a larger one (the
    /// LFE5U-12F is a 25F die with its own ID code).
    pub fn idcode(self) -> Option<&'static str> {
        match self {
            Self::Ulx3s12 => Some("0x21111043"),
            _ => None,
        }
    }

    /// `openFPGALoader -b` board name.
    pub fn loader_board(self) -> &'static str {
        match self {
            Self::Icebreaker => "icebreaker",
            _ => "ulx3s",
        }
    }

    /// Pins of `board_top`'s ports.
    pub fn pins(self) -> LatticePins {
        match self {
            Self::Icebreaker => LatticePins {
                clk: "35",
                clk_mhz: 12,
                rst: "10",
                rst_active_low: true,
                uart_rx: "6",
                uart_tx: "9",
                leds: ["26", "27", "25", "23"],
                led_active_low: false,
            },
            _ => LatticePins {
                clk: "G2",
                clk_mhz: 25,
                rst: "D6",
                rst_active_low: true,
                // FT231X `TXD` drives the FPGA, `RXD` listens to it.
                uart_rx: "M1",
                uart_tx: "L4",
                leds: ["B2", "C2", "C1", "D2"],
                led_active_low: false,
            },
        }
    }

    /// What the UART shell needs. ULX3S designs run at 12.5 MHz from the
    /// ECP5 PLL: the shared Q0.31 requantize (four chained `MULT18X18D`s and
    /// a carry chain) misses the 25 MHz oscillator by a few ns.
    pub fn shell(self) -> ShellBoard {
        let p = self.pins();
        let (core_khz, pll) = match self.family() {
            LatticeFamily::Ice40 => (p.clk_mhz * 1000, None),
            LatticeFamily::Ecp5 => (
                12_500,
                Some(Pll::Ecp5 {
                    clki_div: 2,
                    clkfb_div: 1,
                    clkop_div: 48,
                    clkop_cphase: 23,
                }),
            ),
        };
        ShellBoard {
            name: self.name(),
            flow: match self.family() {
                LatticeFamily::Ice40 => "yosys + nextpnr-ice40",
                LatticeFamily::Ecp5 => "yosys + nextpnr-ecp5",
            },
            clk_mhz: p.clk_mhz,
            core_khz,
            pll,
            rst_active_low: p.rst_active_low,
            led_active_low: p.led_active_low,
        }
    }

    /// UART bit period in board clocks.
    pub fn clks_per_bit(self) -> u32 {
        self.shell().clks_per_bit()
    }
}

impl fmt::Display for LatticeBoard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export_config::HwTarget;

    #[test]
    fn names_roundtrip() {
        for b in LatticeBoard::ALL {
            assert_eq!(LatticeBoard::parse(b.name()), Some(b));
        }
        assert_eq!(LatticeBoard::parse("ULX3S"), Some(LatticeBoard::Ulx3s85));
        assert_eq!(LatticeBoard::parse("up5k"), None);
    }

    #[test]
    fn hw_target_parses_boards() {
        for s in ["lattice:icebreaker", "ice40:icebreaker"] {
            let t = HwTarget::parse(s).unwrap();
            assert_eq!(t.lattice(), Some(LatticeBoard::Icebreaker), "{s}");
        }
        let t = HwTarget::parse("ecp5:ulx3s-25f").unwrap();
        assert_eq!(t, HwTarget::lattice_board(LatticeBoard::Ulx3s25));
        assert_eq!(t.shell_board().unwrap().clk_mhz, 25);
        // A device token still means the part-only target.
        assert_eq!(
            HwTarget::parse("ecp5:45k:CABGA381").unwrap(),
            HwTarget::Ecp5 {
                device: "45k".into(),
                package: "CABGA381".into()
            }
        );
        assert!(HwTarget::parse("lattice:up5k").is_err());
    }

    #[test]
    fn uart_divisors() {
        assert_eq!(LatticeBoard::Icebreaker.clks_per_bit(), 105);
        // ULX3S: 12.5 MHz from the PLL.
        assert_eq!(LatticeBoard::Ulx3s85.clks_per_bit(), 109);
        assert_eq!(LatticeBoard::Ulx3s85.shell().core_mhz_str(), "12.5");
        assert_eq!(LatticeBoard::Icebreaker.shell().core_mhz_str(), "12");
    }
}
