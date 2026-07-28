// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! BRAM primitives: read-only ROM (initialized from a `.mem` file) and
//! a synchronous-read R/W RAM. Both register the read address so Yosys
//! infers a block RAM rather than distributed LUT-RAM.

use crate::verilog::V;

/// Synchronous-read ROM, initialized via `$readmemh(INIT_FILE, mem)`.
/// `WIDTH` and `DEPTH` are parameters; init file is a string parameter
/// so the same module body serves every weight / bias / requant table.
pub fn emit_block_rom() -> String {
    let mut v = V::new();
    v.banner("block_rom — synchronous-read ROM, $readmemh-initialized");
    v.comment("Yosys infers this as a block RAM when WIDTH ≤ 36 and the");
    v.comment("read port is registered.");
    v.blank();

    v.module(
        "block_rom",
        &[
            "parameter int    WIDTH     = 8".into(),
            "parameter int    DEPTH     = 256".into(),
            "parameter string INIT_FILE = \"\"".into(),
        ],
        &[
            "input  logic                       clk".into(),
            "input  logic [$clog2(DEPTH)-1:0]   addr".into(),
            "output logic [WIDTH-1:0]           dout".into(),
        ],
        |v| {
            v.line("logic [WIDTH-1:0] mem [0:DEPTH-1];");
            v.blank();
            v.line("initial begin");
            v.block(|v| {
                v.line("if (INIT_FILE != \"\") begin");
                v.block(|v| v.line("$readmemh(INIT_FILE, mem);"));
                v.line("end");
            });
            v.line("end");
            v.blank();
            v.always_ff(|v| {
                v.line("dout <= mem[addr];");
            });
        },
    );
    v.into_string()
}

/// Synchronous-read R/W BRAM (single port, write-first not necessary —
/// our access pattern only reads after writes from a different layer).
pub fn emit_block_ram() -> String {
    let mut v = V::new();
    v.banner("block_ram — synchronous-read single-port R/W BRAM");
    v.blank();

    v.module(
        "block_ram",
        &[
            "parameter int WIDTH = 8".into(),
            "parameter int DEPTH = 256".into(),
        ],
        &[
            "input  logic                     clk".into(),
            "input  logic                     we".into(),
            "input  logic [$clog2(DEPTH)-1:0] addr".into(),
            "input  logic [WIDTH-1:0]         din".into(),
            "output logic [WIDTH-1:0]         dout".into(),
        ],
        |v| {
            v.line("logic [WIDTH-1:0] mem [0:DEPTH-1];");
            v.blank();
            v.always_ff(|v| {
                v.line("if (we) mem[addr] <= din;");
                v.line("dout <= mem[addr];");
            });
        },
    );
    v.into_string()
}
