// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! What the UART host-link shell (`board_top.sv`) needs from a board,
//! whatever the vendor: [`crate::xc7::Xc7Board`] and
//! [`crate::lattice::LatticeBoard`] both describe themselves as a
//! [`ShellBoard`].

#[allow(unused_imports)]
use crate::prelude::*;

use core::fmt;

/// UART baud rate of the emitted host link.
pub const UART_BAUD: u32 = 115_200;

/// Byte the host sends before each input vector.
pub const UART_SYNC: u8 = 0xA5;

/// A PLL between the board oscillator and the design, for boards whose
/// fabric does not close timing at the oscillator frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pll {
    /// Lattice ECP5 `EHXPLLL`, feedback from `CLKOP` (settings as
    /// `ecppll -i <board> -o <core>` prints them).
    Ecp5 {
        /// Reference divider.
        clki_div: u32,
        /// Feedback divider.
        clkfb_div: u32,
        /// `CLKOP` divider (VCO / `CLKOP`).
        clkop_div: u32,
        /// `CLKOP` coarse phase.
        clkop_cphase: u32,
    },
}

/// Board facts the UART shell is generated from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShellBoard {
    /// Board name (banner, scripts).
    pub name: &'static str,
    /// Open toolchain the scripts drive (banner).
    pub flow: &'static str,
    /// Board oscillator, MHz.
    pub clk_mhz: u32,
    /// Clock the design runs at, kHz: the oscillator's, or [`Self::pll`]'s
    /// output.
    pub core_khz: u32,
    /// PLL from the oscillator to `core_khz`, if they differ.
    pub pll: Option<Pll>,
    /// The reset button reads 0 when pressed.
    pub rst_active_low: bool,
    /// The four status LEDs light on 0.
    pub led_active_low: bool,
}

impl ShellBoard {
    /// UART bit period in design clocks.
    pub fn clks_per_bit(&self) -> u32 {
        (self.core_khz * 1000).div_ceil(UART_BAUD)
    }

    /// The design clock in MHz, as nextpnr's `--freq` takes it (`12.5`).
    pub fn core_mhz_str(&self) -> String {
        let (whole, frac) = (self.core_khz / 1000, self.core_khz % 1000);
        if frac == 0 {
            format!("{whole}")
        } else {
            format!("{whole}.{}", format!("{frac:03}").trim_end_matches('0'))
        }
    }
}

impl fmt::Display for ShellBoard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.flow)
    }
}
