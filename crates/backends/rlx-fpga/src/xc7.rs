// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Xilinx 7-series boards for the open [openXC7](https://github.com/openXC7)
//! flow: `yosys synth_xilinx` → `nextpnr-xilinx` → `fasm2frames` →
//! `xc7frames2bit` → `openFPGALoader`.
//!
//! A preset carries everything the board shell and scripts need: the part
//! (as named by the nextpnr-xilinx chipdb / prjxray-db), the clock and UART
//! pins, and the `openFPGALoader -b` name. Pins are from the Digilent
//! master XDC files.

#[allow(unused_imports)]
use crate::prelude::*;

use core::fmt;

/// UART baud rate of the emitted host link.
pub const UART_BAUD: u32 = 115_200;

/// Byte the host sends before each input vector.
pub const UART_SYNC: u8 = 0xA5;

/// Board pin assignment (LVCMOS33 throughout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Xc7Pins {
    pub clk: &'static str,
    pub clk_mhz: u32,
    pub rst: &'static str,
    pub rst_active_low: bool,
    /// Host → FPGA serial line (FPGA input).
    pub uart_rx: &'static str,
    /// FPGA → host serial line (FPGA output).
    pub uart_tx: &'static str,
    pub leds: [&'static str; 4],
}

/// Supported openXC7 boards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Xc7Board {
    /// Digilent Arty A7-35T.
    ArtyA7_35,
    /// Digilent Arty A7-100T.
    ArtyA7_100,
    /// Digilent Basys 3 (Artix-7 35T).
    Basys3,
    /// Digilent Nexys A7-100T.
    NexysA7_100,
}

impl Xc7Board {
    pub const ALL: [Self; 4] = [
        Self::ArtyA7_35,
        Self::ArtyA7_100,
        Self::Basys3,
        Self::NexysA7_100,
    ];

    /// Canonical CLI name.
    pub fn name(self) -> &'static str {
        match self {
            Self::ArtyA7_35 => "arty-a7-35",
            Self::ArtyA7_100 => "arty-a7-100",
            Self::Basys3 => "basys3",
            Self::NexysA7_100 => "nexys-a7-100",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "arty" | "arty-a7" | "arty-a7-35" | "arty-a7-35t" => Some(Self::ArtyA7_35),
            "arty-a7-100" | "arty-a7-100t" => Some(Self::ArtyA7_100),
            "basys3" | "basys-3" => Some(Self::Basys3),
            "nexys-a7" | "nexys-a7-100" | "nexys-a7-100t" | "nexys4-ddr" => Some(Self::NexysA7_100),
            _ => None,
        }
    }

    /// Part name as the nextpnr-xilinx chipdb / prjxray-db spell it.
    pub fn part(self) -> &'static str {
        match self {
            Self::ArtyA7_35 => "xc7a35tcsg324-1",
            Self::ArtyA7_100 | Self::NexysA7_100 => "xc7a100tcsg324-1",
            Self::Basys3 => "xc7a35tcpg236-1",
        }
    }

    /// prjxray-db family directory.
    pub fn family(self) -> &'static str {
        "artix7"
    }

    /// `openFPGALoader -b` board name.
    pub fn loader_board(self) -> &'static str {
        match self {
            Self::ArtyA7_35 => "arty_a7_35t",
            Self::ArtyA7_100 => "arty_a7_100t",
            Self::Basys3 => "basys3",
            Self::NexysA7_100 => "nexys_a7_100",
        }
    }

    pub fn pins(self) -> Xc7Pins {
        match self {
            Self::ArtyA7_35 | Self::ArtyA7_100 => Xc7Pins {
                clk: "E3",
                clk_mhz: 100,
                rst: "C2",
                rst_active_low: true,
                uart_rx: "A9",
                uart_tx: "D10",
                leds: ["H5", "J5", "T9", "T10"],
            },
            Self::Basys3 => Xc7Pins {
                clk: "W5",
                clk_mhz: 100,
                rst: "U18",
                rst_active_low: false,
                uart_rx: "B18",
                uart_tx: "A18",
                leds: ["U16", "E19", "U19", "V19"],
            },
            Self::NexysA7_100 => Xc7Pins {
                clk: "E3",
                clk_mhz: 100,
                rst: "C12",
                rst_active_low: true,
                uart_rx: "C4",
                uart_tx: "D4",
                leds: ["H17", "K15", "J13", "N14"],
            },
        }
    }

    /// UART bit period in board clocks.
    pub fn clks_per_bit(self) -> u32 {
        (self.pins().clk_mhz * 1_000_000).div_ceil(UART_BAUD)
    }
}

impl fmt::Display for Xc7Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_roundtrip() {
        for b in Xc7Board::ALL {
            assert_eq!(Xc7Board::parse(b.name()), Some(b));
        }
        assert_eq!(Xc7Board::parse("Arty_A7_35T"), Some(Xc7Board::ArtyA7_35));
        assert_eq!(Xc7Board::parse("xc7a35t"), None);
    }

    #[test]
    fn hw_target_parses_boards_and_parts() {
        use crate::export_config::HwTarget;
        let t = HwTarget::parse("xc7:basys3").unwrap();
        assert_eq!(t.xc7(), Some(Xc7Board::Basys3));
        assert_eq!(t, HwTarget::xc7_board(Xc7Board::Basys3));
        let t = HwTarget::parse("xilinx7:xc7a50tcsg324-1").unwrap();
        assert_eq!(
            t,
            HwTarget::Xilinx7 {
                part: "xc7a50tcsg324-1".into(),
                board: None
            }
        );
        assert!(HwTarget::parse("xc7").is_err());
    }

    #[test]
    fn uart_divisor_at_100mhz() {
        assert_eq!(Xc7Board::Basys3.clks_per_bit(), 869);
    }
}
