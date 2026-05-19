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

//! Top-level emitter: stitches per-layer kernel modules together with a
//! controller FSM and the activation BRAM(s).
//!
//! Two BRAM strategies, picked at codegen time from the optimizer's
//! arena hints:
//!
//! * **Arena (ping-pong)** — used when `tune.arena_plan` is on. Just
//!   *two* `block_ram` instances sized to the largest activation, with
//!   the active layer's read/write ports muxed onto whichever slot its
//!   `Hints.bram_slot_in/out` points at. Sequential execution makes
//!   this provably correct (no two layers ever touch the same slot at
//!   the same time).
//! * **Per-stage (legacy)** — one BRAM per intermediate, sized to that
//!   intermediate's exact length. Used when `arena_plan` is off, and
//!   for back-compatibility with tests that don't go through the
//!   optimizer.

use std::collections::BTreeMap;

use super::{Artifact, LayerHandle};
use crate::codegen::relu::bits_for;
use crate::model::{Layer, Model};
use crate::tune::Tune;
use crate::verilog::V;

pub fn emit(
    model: &Model,
    layers: &[LayerHandle],
    tune: &Tune,
    arena_bank: &BTreeMap<u8, u8>,
) -> Artifact {
    if tune.arena_plan && layers.iter().all(|l| l.hints.bram_slot_in.is_some()) {
        emit_arena(model, layers, tune, arena_bank)
    } else {
        emit_per_stage(model, layers, tune)
    }
}

// ── Arena (ping-pong) layout ────────────────────────────────────────

fn emit_arena(
    model: &Model,
    layers: &[LayerHandle],
    tune: &Tune,
    arena_bank: &BTreeMap<u8, u8>,
) -> Artifact {
    let scratch_len = arena_size(model, layers);
    let scratch_abits = bits_for(scratch_len);
    let in_addr_bits = bits_for(model.input_len.max(1));
    let n_slots = layers
        .iter()
        .flat_map(|l| [l.hints.bram_slot_in, l.hints.bram_slot_out])
        .flatten()
        .max()
        .map(|s| s as usize + 1)
        .unwrap_or(2)
        .max(2);

    // Per-slot bank factor; default 1 unless `arena_bank` says otherwise.
    let slot_bank: Vec<u8> = (0..n_slots as u8)
        .map(|s| arena_bank.get(&s).copied().unwrap_or(1).max(1))
        .collect();

    let mut v = V::new();
    v.banner(&format!(
        "top — {} (arena: {n_slots}x BRAM @ SCRATCH_LEN={scratch_len})",
        model.name
    ));
    v.comment(&format!("Tune: {tune}"));
    v.comment("Pipeline (post-fusion):");
    for (i, l) in layers.iter().enumerate() {
        let kind = layer_kind(&l.layer);
        let s_in = l.hints.bram_slot_in.unwrap_or(0);
        let s_out = l.hints.bram_slot_out.unwrap_or(0);
        let p_ic = if l.hints.ic_parallelism > 1 {
            format!(" P_ic={}", l.hints.ic_parallelism)
        } else {
            String::new()
        };
        v.comment(&format!(
            "  L{i:02}  {kind:8}  out_len={}  slot {s_in}→{s_out}{p_ic}",
            l.out_len
        ));
    }
    if slot_bank.iter().any(|&b| b > 1) {
        v.comment(&format!("Banked slots: {:?}", arena_bank));
    }
    v.blank();

    let ports: Vec<String> = vec![
        "input  logic                       clk".into(),
        "input  logic                       rst".into(),
        "input  logic                       start".into(),
        "output logic                       done".into(),
        format!("input  logic [{}:0]              in_addr", in_addr_bits - 1),
        "input  logic                       in_we".into(),
        "input  logic signed [7:0]          in_din".into(),
        "output logic signed [7:0]          pred".into(),
    ];

    v.module("top", &[], &ports, |v| {
        v.line(&format!("localparam int SCRATCH_LEN = {scratch_len};"));
        v.line(&format!("localparam int SCRATCH_AB  = {scratch_abits};"));
        v.blank();

        // ── arena BRAMs ──
        // Each arena slot is either:
        //   * unbanked: one 8-bit BRAM at SCRATCH_LEN bytes
        //   * banked  : `bank` 8-bit BRAMs each at SCRATCH_LEN/bank bytes;
        //               byte address X maps to bank (X & bank_mask) at
        //               word index (X >> bank_shift).
        v.comment(&format!("─── {n_slots} ping-pong arena BRAMs ───"));
        for s in 0..n_slots {
            let bank = slot_bank[s];
            if bank == 1 {
                v.line(&format!("logic [SCRATCH_AB-1:0] ar{s}_addr;"));
                v.line(&format!("logic                  ar{s}_we;"));
                v.line(&format!("logic signed [7:0]     ar{s}_din;"));
                v.line(&format!("logic signed [7:0]     ar{s}_dout;"));
                v.line(&format!("block_ram #(.WIDTH(8), .DEPTH(SCRATCH_LEN)) u_ar{s} ("));
                v.block(|v| {
                    v.line(&format!(".clk(clk), .we(ar{s}_we), .addr(ar{s}_addr),"));
                    v.line(&format!(".din(ar{s}_din), .dout(ar{s}_dout)"));
                });
                v.line(");");
            } else {
                let bank_depth = scratch_len / bank as usize;
                let bank_ab = bits_for(bank_depth);
                v.line(&format!("// slot {s}: banked × {bank}, bank_depth={bank_depth}"));
                v.line(&format!("logic [{}:0] ar{s}_word_addr;", bank_ab - 1));
                v.line(&format!("logic [0:{}] ar{s}_we;", bank - 1));
                v.line(&format!("logic signed [7:0] ar{s}_din  [0:{}];", bank - 1));
                v.line(&format!("logic signed [7:0] ar{s}_dout [0:{}];", bank - 1));
                for b in 0..bank {
                    v.line(&format!(
                        "block_ram #(.WIDTH(8), .DEPTH({bank_depth})) u_ar{s}_b{b} (",
                    ));
                    v.block(|v| {
                        v.line(&format!(
                            ".clk(clk), .we(ar{s}_we[{b}]), .addr(ar{s}_word_addr),"
                        ));
                        v.line(&format!(
                            ".din(ar{s}_din[{b}]), .dout(ar{s}_dout[{b}])"
                        ));
                    });
                    v.line(");");
                }
            }
        }
        v.blank();

        // ── per-layer kernel instances ──
        v.comment("─── per-layer kernel instances ───");
        for (i, l) in layers.iter().enumerate() {
            let in_abits  = bits_for(layer_in_len(model, layers, i)).max(1);
            let out_abits = bits_for(l.out_len).max(1);
            let p_ic = l.hints.ic_parallelism.max(1) as usize;
            v.line(&format!("logic l{i}_start, l{i}_done;"));
            v.line(&format!("logic [{}:0] l{i}_x_addr;", in_abits - 1));
            v.line(&format!("logic [{}:0] l{i}_y_addr;", out_abits - 1));
            v.line(&format!("logic        l{i}_y_we;"));
            v.line(&format!("logic signed [7:0] l{i}_y_din;"));
            if p_ic > 1 {
                let dout_w = 8 * p_ic;
                v.line(&format!("logic [{}:0] l{i}_x_dout;  // ic-parallel × {p_ic}", dout_w - 1));
            } else {
                v.line(&format!("logic signed [7:0] l{i}_x_dout;"));
            }
            v.line(&format!("{} {} (", l.module_name, l.instance_name));
            v.block(|v| {
                v.line(".clk(clk), .rst(rst),");
                v.line(&format!(".start(l{i}_start), .done(l{i}_done),"));
                v.line(&format!(".x_addr(l{i}_x_addr), .x_dout(l{i}_x_dout),"));
                v.line(&format!(".y_addr(l{i}_y_addr), .y_we(l{i}_y_we), .y_din(l{i}_y_din)"));
            });
            v.line(");");
            v.blank();
        }

        // For scalar consumers reading a banked slot, we need the bank
        // index (low bits of x_addr) registered one cycle so the dout
        // mux sees the right bank when the data arrives.
        let any_banked_scalar = layers.iter().any(|l| {
            l.hints.ic_parallelism <= 1
                && l.hints.bram_slot_in.map(|s| slot_bank[s as usize] > 1).unwrap_or(false)
        });
        if any_banked_scalar {
            v.comment("Registered low bits of x_addr for banked-slot scalar reads (1-cycle BRAM latency).");
            for (i, l) in layers.iter().enumerate() {
                if l.hints.ic_parallelism > 1 { continue; }
                let needs = l.hints.bram_slot_in.map(|s| slot_bank[s as usize] > 1).unwrap_or(false);
                if !needs { continue; }
                let bank = slot_bank[l.hints.bram_slot_in.unwrap() as usize];
                let lsb_bits = bits_for(bank as usize);
                v.line(&format!("logic [{}:0] l{i}_x_lsb_d1;", lsb_bits - 1));
                v.line(&format!("always_ff @(posedge clk) l{i}_x_lsb_d1 <= l{i}_x_addr[{}:0];",
                                lsb_bits - 1));
            }
            v.blank();
        }

        // ── arena port routing (per-stage mux) ──
        v.comment("─── arena port routing — when stage == i, layer i drives slots ───");
        v.always_comb(|v| {
            // Defaults: idle.
            for s in 0..n_slots {
                let bank = slot_bank[s];
                if bank == 1 {
                    v.line(&format!("ar{s}_addr = '0;"));
                    v.line(&format!("ar{s}_we   = 1'b0;"));
                    v.line(&format!("ar{s}_din  = 8'sd0;"));
                } else {
                    v.line(&format!("ar{s}_word_addr = '0;"));
                    for b in 0..bank {
                        v.line(&format!("ar{s}_we[{b}]  = 1'b0;"));
                        v.line(&format!("ar{s}_din[{b}] = 8'sd0;"));
                    }
                }
            }
            // External input load into slot 0 when not running.
            v.line("if (!start && cstate == C_IDLE) begin");
            v.block(|v| {
                let bank0 = slot_bank[0];
                if bank0 == 1 {
                    v.line("ar0_addr = SCRATCH_AB'(in_addr);");
                    v.line("ar0_we   = in_we;");
                    v.line("ar0_din  = in_din;");
                } else {
                    let bank_shift = (bank0 as usize).trailing_zeros() as usize;
                    let bank_mask = bank0 as usize - 1;
                    v.comment(&format!("input goes to bank (in_addr & {bank_mask}) at index in_addr >> {bank_shift}"));
                    v.line(&format!("ar0_word_addr = in_addr >> {bank_shift};"));
                    for b in 0..bank0 {
                        v.line(&format!("if ((in_addr & {bank_mask}) == {b}'d{bv}) begin",
                                        b = bits_for(bank0 as usize), bv = b));
                        v.block(|v| {
                            v.line(&format!("ar0_we[{b}]  = in_we;"));
                            v.line(&format!("ar0_din[{b}] = in_din;"));
                        });
                        v.line("end");
                    }
                }
            });
            v.line("end else begin");
            v.block(|v| {
                v.line("unique case (stage)");
                v.block(|v| {
                    for (i, l) in layers.iter().enumerate() {
                        let s_in  = l.hints.bram_slot_in.unwrap_or(0);
                        let s_out = l.hints.bram_slot_out.unwrap_or(0);
                        let bank_in  = slot_bank[s_in as usize];
                        let bank_out = slot_bank[s_out as usize];
                        let p_ic = l.hints.ic_parallelism.max(1) as usize;
                        v.line(&format!("{i}: begin"));
                        v.block(|v| {
                            // Read side
                            if bank_in == 1 {
                                v.line(&format!("ar{s_in}_addr = SCRATCH_AB'(l{i}_x_addr);"));
                            } else {
                                let bs = (bank_in as usize).trailing_zeros() as usize;
                                if p_ic > 1 {
                                    // ic-parallel: byte_addr is bank-aligned, all banks get the word index.
                                    v.line(&format!("ar{s_in}_word_addr = l{i}_x_addr >> {bs};"));
                                } else {
                                    // Scalar consumer of a banked slot: same word addr, dout is muxed via lsb_d1.
                                    v.line(&format!("ar{s_in}_word_addr = l{i}_x_addr >> {bs};"));
                                }
                            }
                            // Write side
                            if bank_out == 1 {
                                v.line(&format!("ar{s_out}_addr = SCRATCH_AB'(l{i}_y_addr);"));
                                v.line(&format!("ar{s_out}_we   = l{i}_y_we;"));
                                v.line(&format!("ar{s_out}_din  = l{i}_y_din;"));
                            } else {
                                let bs = (bank_out as usize).trailing_zeros() as usize;
                                let bm = bank_out as usize - 1;
                                v.line(&format!("ar{s_out}_word_addr = l{i}_y_addr >> {bs};"));
                                for b in 0..bank_out {
                                    v.line(&format!("if ((l{i}_y_addr & {bm}) == {b}'d{bv}) begin",
                                                    b = bits_for(bank_out as usize), bv = b));
                                    v.block(|v| {
                                        v.line(&format!("ar{s_out}_we[{b}]  = l{i}_y_we;"));
                                        v.line(&format!("ar{s_out}_din[{b}] = l{i}_y_din;"));
                                    });
                                    v.line("end");
                                }
                            }
                        });
                        v.line("end");
                    }
                    v.line("default: ;");
                });
                v.line("endcase");
            });
            v.line("end");
        });
        v.blank();

        // Per-layer x_dout routing.
        v.comment("─── per-layer x_dout: route from the layer's input slot ───");
        for (i, l) in layers.iter().enumerate() {
            let s_in = l.hints.bram_slot_in.unwrap_or(0);
            let bank = slot_bank[s_in as usize];
            let p_ic = l.hints.ic_parallelism.max(1) as usize;
            if bank == 1 {
                v.line(&format!("assign l{i}_x_dout = ar{s_in}_dout;"));
            } else if p_ic > 1 {
                // ic-parallel: concat all banks into a packed word.
                let parts: Vec<String> = (0..bank as usize).rev()
                    .map(|b| format!("ar{s_in}_dout[{b}]")).collect();
                v.line(&format!("assign l{i}_x_dout = {{{}}};", parts.join(", ")));
            } else {
                // Scalar consumer of banked slot: mux by registered LSBs.
                v.line(&format!("assign l{i}_x_dout = ar{s_in}_dout[l{i}_x_lsb_d1];"));
            }
        }
        v.blank();

        // Output prediction — read the last non-elided layer's output slot.
        let last_out = layers.last().and_then(|l| l.hints.bram_slot_out).unwrap_or(0);
        let last_bank = slot_bank[last_out as usize];
        v.comment(&format!("Expose slot {last_out} as `pred` (last layer's output)."));
        if last_bank == 1 {
            v.line(&format!("assign pred = ar{last_out}_dout;"));
        } else {
            v.line(&format!("assign pred = ar{last_out}_dout[0];"));
        }
        v.blank();

        // Controller FSM
        emit_controller(v, layers.len());
    });

    Artifact {
        rel_path: "top.sv".into(),
        content: v.into_string(),
    }
}

fn arena_size(model: &Model, layers: &[LayerHandle]) -> usize {
    let mut m = model.input_len;
    for l in layers {
        m = m.max(l.out_len);
    }
    m
}

fn layer_in_len(model: &Model, layers: &[LayerHandle], i: usize) -> usize {
    if i == 0 {
        model.input_len
    } else {
        layers[i - 1].out_len
    }
}

fn layer_kind(l: &Layer) -> &'static str {
    match l {
        Layer::Conv2d { .. } => "Conv2d",
        Layer::Dense { .. } => "Dense",
        Layer::Relu { .. } => "ReLU",
        Layer::MaxPool2d { .. } => "MaxPool",
        Layer::Argmax { .. } => "Argmax",
    }
}

fn emit_controller(v: &mut V, n: usize) {
    v.banner("controller — assert each layer's `start`, wait for `done`");
    v.line(&format!(
        "logic [{}:0] stage;",
        bits_for((n + 2).max(2)) - 1
    ));
    v.line("typedef enum logic [1:0] {");
    v.block(|v| v.line("C_IDLE, C_RUN, C_STEP, C_DONE"));
    v.line("} ctrl_t;");
    v.line("ctrl_t cstate, cnext;");
    v.blank();
    v.always_comb(|v| {
        for i in 0..n {
            v.line(&format!(
                "l{i}_start = (cstate == C_RUN) && (stage == {i});"
            ));
        }
        v.line("done = (cstate == C_DONE);");
    });
    v.blank();
    v.always_ff(|v| {
        v.line("if (rst) begin");
        v.block(|v| {
            v.line("cstate <= C_IDLE;");
            v.line("stage  <= '0;");
        });
        v.line("end else begin");
        v.block(|v| {
            v.line("cstate <= cnext;");
            v.line("if (cstate == C_IDLE && start) stage <= '0;");
            v.line("if (cstate == C_STEP) stage <= stage + 1;");
        });
        v.line("end");
    });
    v.blank();
    v.always_comb(|v| {
        v.line("cnext = cstate;");
        v.line("unique case (cstate)");
        v.block(|v| {
            v.line("C_IDLE : if (start) cnext = C_RUN;");
            v.line("C_RUN  : begin");
            v.block(|v| {
                let mut first = true;
                for i in 0..n {
                    let kw = if first { "if   " } else { "else if" };
                    first = false;
                    v.line(&format!("{kw} (stage == {i} && l{i}_done) cnext = C_STEP;"));
                }
            });
            v.line("end");
            v.line(&format!(
                "C_STEP : cnext = (stage == {}) ? C_DONE : C_RUN;",
                n.saturating_sub(1)
            ));
            v.line("C_DONE : if (!start) cnext = C_IDLE;");
        });
        v.line("endcase");
    });
}

// ── Per-stage (legacy) layout ───────────────────────────────────────

fn emit_per_stage(model: &Model, layers: &[LayerHandle], tune: &Tune) -> Artifact {
    let mut v = V::new();
    v.banner(&format!(
        "top — {} (per-stage BRAMs, legacy layout)",
        model.name
    ));
    v.comment(&format!("Tune: {tune}"));
    v.comment("Pipeline:");
    for (i, l) in layers.iter().enumerate() {
        v.comment(&format!(
            "  L{i:02}  {:8}  out_len={}",
            layer_kind(&l.layer),
            l.out_len
        ));
    }
    v.blank();

    let in_addr_bits = bits_for(model.input_len);
    let mut bram_lens: Vec<usize> = Vec::with_capacity(layers.len() + 1);
    bram_lens.push(model.input_len);
    for l in layers {
        bram_lens.push(l.out_len);
    }

    let ports: Vec<String> = vec![
        "input  logic                       clk".into(),
        "input  logic                       rst".into(),
        "input  logic                       start".into(),
        "output logic                       done".into(),
        format!("input  logic [{}:0]              in_addr", in_addr_bits - 1),
        "input  logic                       in_we".into(),
        "input  logic signed [7:0]          in_din".into(),
        "output logic signed [7:0]          pred".into(),
    ];

    v.module("top", &[], &ports, |v| {
        v.comment("─── activation BRAMs ───");
        for (i, len) in bram_lens.iter().enumerate() {
            let abits = bits_for(*len).max(1);
            v.line(&format!("logic [{}:0] a{i}_addr;", abits - 1));
            v.line(&format!("logic        a{i}_we;"));
            v.line(&format!("logic signed [7:0] a{i}_din;"));
            v.line(&format!("logic signed [7:0] a{i}_dout;"));
            v.line(&format!("block_ram #(.WIDTH(8), .DEPTH({len})) u_a{i} ("));
            v.block(|v| {
                v.line(&format!(".clk(clk), .we(a{i}_we), .addr(a{i}_addr),"));
                v.line(&format!(".din(a{i}_din), .dout(a{i}_dout)"));
            });
            v.line(");");
        }
        v.blank();

        v.comment("─── per-layer kernel instances ───");
        for (i, l) in layers.iter().enumerate() {
            let in_idx = i;
            let out_idx = i + 1;
            v.line(&format!("logic l{i}_start, l{i}_done;"));
            let in_abits = bits_for(bram_lens[in_idx]).max(1);
            let out_abits = bits_for(bram_lens[out_idx]).max(1);
            v.line(&format!("logic [{}:0] l{i}_x_addr;", in_abits - 1));
            v.line(&format!("logic [{}:0] l{i}_y_addr;", out_abits - 1));
            v.line(&format!("logic        l{i}_y_we;"));
            v.line(&format!("logic signed [7:0] l{i}_y_din;"));
            v.line(&format!("{} {} (", l.module_name, l.instance_name));
            v.block(|v| {
                v.line(".clk(clk), .rst(rst),");
                v.line(&format!(".start(l{i}_start), .done(l{i}_done),"));
                v.line(&format!(".x_addr(l{i}_x_addr), .x_dout(a{in_idx}_dout),"));
                v.line(&format!(
                    ".y_addr(l{i}_y_addr), .y_we(l{i}_y_we), .y_din(l{i}_y_din)"
                ));
            });
            v.line(");");
            v.blank();
        }

        v.comment("─── BRAM port routing ───");
        v.always_comb(|v| {
            v.line("if (start) begin");
            v.block(|v| {
                v.line("a0_addr = l0_x_addr;");
                v.line("a0_we   = 1'b0;");
                v.line("a0_din  = 8'sd0;");
            });
            v.line("end else begin");
            v.block(|v| {
                v.line("a0_addr = in_addr;");
                v.line("a0_we   = in_we;");
                v.line("a0_din  = in_din;");
            });
            v.line("end");
        });
        v.blank();

        for i in 1..bram_lens.len() {
            let writer = i - 1;
            v.line(&format!("// BRAM {i} ← L{writer}.y, → L{i}.x"));
            if i < layers.len() {
                v.always_comb(|v| {
                    v.line(&format!("a{i}_we   = l{writer}_y_we;"));
                    v.line(&format!("a{i}_din  = l{writer}_y_din;"));
                    v.line(&format!(
                        "a{i}_addr = l{writer}_y_we ? l{writer}_y_addr : l{i}_x_addr;"
                    ));
                });
            } else {
                v.always_comb(|v| {
                    v.line(&format!("a{i}_we   = l{writer}_y_we;"));
                    v.line(&format!("a{i}_din  = l{writer}_y_din;"));
                    v.line(&format!("a{i}_addr = l{writer}_y_addr;"));
                });
            }
            v.blank();
        }

        v.comment("Expose the final BRAM at addr 0 as `pred`.");
        v.line(&format!("assign pred = a{}_dout;", bram_lens.len() - 1));
        v.blank();

        emit_controller(v, layers.len());
    });

    Artifact {
        rel_path: "top.sv".into(),
        content: v.into_string(),
    }
}

/// Emit a tiny Verilator-style testbench. Unchanged from before — loads
/// `tb_image.mem` into the input port and prints `pred`.
pub fn emit_tb(model: &Model) -> String {
    let in_len = model.input_len;
    let in_bits = bits_for(in_len);

    let mut v = V::new();
    v.banner("tb — TinyConv-MNIST testbench (image-driven, Verilator)");
    v.line("`timescale 1ns/1ps");
    v.blank();
    v.module("tb", &[], &[], |v| {
        v.line("logic clk = 0;");
        v.line("always #5 clk = ~clk;");
        v.line("logic rst = 1;");
        v.line("logic start = 0;");
        v.line("logic done;");
        v.line(&format!("logic [{}:0] in_addr = '0;", in_bits - 1));
        v.line("logic in_we = 0;");
        v.line("logic signed [7:0] in_din = '0;");
        v.line("logic signed [7:0] pred;");
        v.blank();

        v.line("top u_top (");
        v.block(|v| {
            v.line(".clk(clk), .rst(rst), .start(start), .done(done),");
            v.line(".in_addr(in_addr), .in_we(in_we), .in_din(in_din),");
            v.line(".pred(pred)");
        });
        v.line(");");
        v.blank();

        v.line(&format!("logic signed [7:0] image_mem [0:{}];", in_len - 1));
        v.line("initial begin");
        v.block(|v| {
            v.line("$readmemh(\"tb_image.mem\", image_mem);");
            v.line("rst = 1; #20; rst = 0;");
            v.line(&format!("for (int i = 0; i < {in_len}; i++) begin"));
            v.block(|v| {
                v.line("@(posedge clk);");
                v.line("in_addr <= i[31:0];");
                v.line("in_we   <= 1'b1;");
                v.line("in_din  <= image_mem[i];");
            });
            v.line("end");
            v.line("@(posedge clk); in_we <= 1'b0;");
            v.line("@(posedge clk); start <= 1'b1;");
            v.line("wait (done);");
            v.line("@(posedge clk); start <= 1'b0;");
            v.line("$display(\"pred = %0d\", $signed(pred));");
            v.line("$finish;");
        });
        v.line("end");
    });
    v.into_string()
}
