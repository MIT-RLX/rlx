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

use super::io_ports::{self, primary_out_len, sanitize_port};
use super::{Artifact, LayerHandle};
use crate::codegen::relu::bits_for;
use crate::export_config::IoConfig;
use crate::model::{Layer, Model};
use crate::tune::Tune;
use crate::verilog::V;

pub fn emit(
    model: &Model,
    layers: &[LayerHandle],
    tune: &Tune,
    arena_bank: &BTreeMap<u8, u8>,
    io: &IoConfig,
) -> Artifact {
    if tune.arena_plan && layers.iter().all(|l| l.hints.bram_slot_in.is_some()) {
        emit_arena(model, layers, tune, arena_bank, io)
    } else {
        emit_per_stage(model, layers, tune, io)
    }
}

// ── Arena (ping-pong) layout ────────────────────────────────────────

fn emit_arena(
    model: &Model,
    layers: &[LayerHandle],
    tune: &Tune,
    arena_bank: &BTreeMap<u8, u8>,
    io: &IoConfig,
) -> Artifact {
    let scratch_len = arena_size(model, layers);
    let scratch_abits = bits_for(scratch_len);
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

    let n = &io.names;
    let ports = io_ports::top_ports(model, io);

    v.module("top", &[], &ports, |v| {
        v.line(&format!("localparam int SCRATCH_LEN = {scratch_len};"));
        v.line(&format!("localparam int SCRATCH_AB  = {scratch_abits};"));
        v.line(&format!("localparam int INPUT_LEN   = {};", model.input_len));
        v.line(&format!("localparam int OUT_LEN     = {};", primary_out_len(model)));
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
                    v.line(&format!(".clk({}), .we(ar{s}_we), .addr(ar{s}_addr),", n.clk));
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
                            ".clk({}), .we(ar{s}_we[{b}]), .addr(ar{s}_word_addr),",
                            n.clk
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
                v.line(&format!(".clk({}), .rst({}),", n.clk, n.rst));
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
                v.line(&format!("always_ff @(posedge {}) l{i}_x_lsb_d1 <= l{i}_x_addr[{}:0];",
                                n.clk, lsb_bits - 1));
            }
            v.blank();
        }

        emit_controller(v, layers.len(), n);

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
            // Host load (IDLE) or host readout (DONE) into/from arena slots.
            v.line(&format!(
                "if (cstate == C_DONE && {}) begin",
                if io.output.wants_memory() {
                    n.out_re.as_str()
                } else {
                    "1'b0"
                }
            ));
            v.block(|v| {
                if io.output.wants_memory() {
                    let last_out = layers.last().and_then(|l| l.hints.bram_slot_out).unwrap_or(0);
                    let last_bank = slot_bank[last_out as usize];
                    if last_bank == 1 {
                        v.line(&format!(
                            "ar{last_out}_addr = SCRATCH_AB'({});",
                            n.out_addr
                        ));
                    } else {
                        let bs = (last_bank as usize).trailing_zeros() as usize;
                        v.line(&format!(
                            "ar{last_out}_word_addr = {} >> {bs};",
                            n.out_addr
                        ));
                    }
                }
            });
            v.line(&format!(
                "end else if (!{} && cstate == C_IDLE) begin",
                n.start
            ));
            v.block(|v| {
                let bank0 = slot_bank[0];
                if io.input.wants_memory() {
                    if bank0 == 1 {
                        v.line(&format!("ar0_addr = SCRATCH_AB'({});", n.in_addr));
                        v.line(&format!("ar0_we   = {};", n.in_we));
                        v.line(&format!("ar0_din  = {};", n.in_din));
                    } else {
                        let bank_shift = (bank0 as usize).trailing_zeros() as usize;
                        let bank_mask = bank0 as usize - 1;
                        v.comment(&format!(
                            "input goes to bank ({} & {bank_mask}) at index {} >> {bank_shift}",
                            n.in_addr, n.in_addr
                        ));
                        v.line(&format!("ar0_word_addr = {} >> {bank_shift};", n.in_addr));
                        for b in 0..bank0 {
                            v.line(&format!(
                                "if (({} & {bank_mask}) == {bw}'d{bv}) begin",
                                n.in_addr,
                                bw = bits_for(bank0 as usize),
                                bv = b
                            ));
                            v.block(|v| {
                                v.line(&format!("ar0_we[{b}]  = {};", n.in_we));
                                v.line(&format!("ar0_din[{b}] = {};", n.in_din));
                            });
                            v.line("end");
                        }
                    }
                }
                if io.input.wants_stream() {
                    v.comment("stream input beats land at in_wr_ptr (see always_ff below)");
                    let beat = io.input.beat_elems() as usize;
                    if bank0 == 1 {
                        v.line("if (stream_in_fire) begin");
                        v.block(|v| {
                            v.line("ar0_addr = SCRATCH_AB'(in_wr_ptr);");
                            v.line("ar0_we   = 1'b1;");
                            v.line(&format!("ar0_din  = {}[7:0];", n.in_data));
                        });
                        v.line("end");
                        if beat > 1 {
                            v.comment(&format!(
                                "beat_elems={beat}: only byte0 written combinationally; \
                                 multi-byte stream expand is sequential in always_ff"
                            ));
                        }
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
        if io.output.wants_pred_port() {
            v.comment(&format!(
                "Expose slot {last_out} as `{}` (last layer's output).",
                n.pred
            ));
            if last_bank == 1 {
                v.line(&format!("assign {} = ar{last_out}_dout;", n.pred));
            } else {
                v.line(&format!("assign {} = ar{last_out}_dout[0];", n.pred));
            }
        }
        if io.output.wants_memory() {
            v.comment(&format!(
                "Memory readout of last slot via {} (1-cycle BRAM latency after {}/{})",
                n.out_dout, n.out_addr, n.out_re
            ));
            if last_bank == 1 {
                v.line(&format!("assign {} = ar{last_out}_dout;", n.out_dout));
            } else {
                v.line(&format!("assign {} = ar{last_out}_dout[0];", n.out_dout));
            }
        }
        for extra in &model.extra_outputs {
            let stem = sanitize_port(&extra.name);
            let slot = layers
                .get(extra.after_layer)
                .and_then(|l| l.hints.bram_slot_out)
                .unwrap_or(last_out);
            v.comment(&format!(
                "Extra readout `{stem}` → slot {slot} (layer {})",
                extra.after_layer
            ));
            v.line(&format!("assign {stem}_dout = ar{slot}_dout;"));
        }
        v.blank();

        // Stream input / output sidebands
        emit_stream_sidebands(v, io, model, n);
        emit_scalar_sidebands(v, io, n);
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

fn emit_controller(v: &mut V, n_layers: usize, names: &crate::export_config::PortNames) {
    v.banner("controller — assert each layer's `start`, wait for `done`");
    v.line(&format!(
        "logic [{}:0] stage;",
        bits_for((n_layers + 2).max(2)) - 1
    ));
    v.line("typedef enum logic [1:0] {");
    v.block(|v| v.line("C_IDLE, C_RUN, C_STEP, C_DONE"));
    v.line("} ctrl_t;");
    v.line("ctrl_t cstate, cnext;");
    v.blank();
    v.always_comb(|v| {
        for i in 0..n_layers {
            v.line(&format!(
                "l{i}_start = (cstate == C_RUN) && (stage == {i});"
            ));
        }
        v.line(&format!("{} = (cstate == C_DONE);", names.done));
    });
    v.blank();
    v.always_ff_on(&names.clk, |v| {
        v.line(&format!("if ({}) begin", names.rst));
        v.block(|v| {
            v.line("cstate <= C_IDLE;");
            v.line("stage  <= '0;");
        });
        v.line("end else begin");
        v.block(|v| {
            v.line("cstate <= cnext;");
            v.line(&format!(
                "if (cstate == C_IDLE && {}) stage <= '0;",
                names.start
            ));
            v.line("if (cstate == C_STEP) stage <= stage + 1;");
        });
        v.line("end");
    });
    v.blank();
    v.always_comb(|v| {
        v.line("cnext = cstate;");
        v.line("unique case (cstate)");
        v.block(|v| {
            v.line(&format!("C_IDLE : if ({}) cnext = C_RUN;", names.start));
            v.line("C_RUN  : begin");
            v.block(|v| {
                let mut first = true;
                for i in 0..n_layers {
                    let kw = if first { "if   " } else { "else if" };
                    first = false;
                    v.line(&format!("{kw} (stage == {i} && l{i}_done) cnext = C_STEP;"));
                }
            });
            v.line("end");
            v.line(&format!(
                "C_STEP : cnext = (stage == {}) ? C_DONE : C_RUN;",
                n_layers.saturating_sub(1)
            ));
            v.line(&format!("C_DONE : if (!{}) cnext = C_IDLE;", names.start));
        });
        v.line("endcase");
    });
}

fn emit_stream_sidebands(
    v: &mut V,
    io: &IoConfig,
    model: &Model,
    names: &crate::export_config::PortNames,
) {
    let n = names;
    if io.input.wants_stream() {
        let beat = io.input.beat_elems() as usize;
        let ab = bits_for(model.input_len.max(1));
        v.comment(&format!(
            "Stream input: {}/{}/{} (beat_elems={beat})",
            n.in_valid, n.in_ready, n.in_data
        ));
        v.line(&format!("logic [{}:0] in_wr_ptr;", ab - 1));
        v.line("logic stream_in_fire;");
        v.line(&format!(
            "assign stream_in_fire = {} && {} && (cstate == C_IDLE);",
            n.in_valid, n.in_ready
        ));
        v.line(&format!(
            "assign {} = (cstate == C_IDLE) && (in_wr_ptr < INPUT_LEN);",
            n.in_ready
        ));
        v.always_ff_on(&n.clk, |v| {
            v.line(&format!("if ({}) in_wr_ptr <= '0;", n.rst));
            v.line(&format!(
                "else if (cstate == C_IDLE && {}) in_wr_ptr <= '0;",
                n.start
            ));
            v.line(&format!(
                "else if (stream_in_fire) in_wr_ptr <= in_wr_ptr + {beat};"
            ));
        });
        v.blank();
    }
    if io.output.wants_stream() {
        let beat = io.output.beat_elems() as usize;
        let ab = bits_for(primary_out_len(model).max(1));
        let data_w = 8 * beat;
        v.comment(&format!(
            "Stream output: {}/{}/{} (beat_elems={beat}); data peeks last-buffer byte0 via pred path",
            n.out_valid, n.out_ready, n.out_data
        ));
        v.line(&format!("logic [{}:0] out_rd_ptr;", ab - 1));
        v.line(&format!(
            "assign {} = (cstate == C_DONE) && (out_rd_ptr < OUT_LEN);",
            n.out_valid
        ));
        if io.output.wants_pred_port() {
            if beat == 1 {
                v.line(&format!("assign {} = {};", n.out_data, n.pred));
            } else {
                v.line(&format!(
                    "assign {} = {{{{{}{{8'b0}}}}, {}}};",
                    n.out_data,
                    beat - 1,
                    n.pred
                ));
            }
        } else {
            v.line(&format!("assign {} = {data_w}'d0;", n.out_data));
        }
        v.always_ff_on(&n.clk, |v| {
            v.line(&format!("if ({}) out_rd_ptr <= '0;", n.rst));
            v.line("else if (cstate != C_DONE) out_rd_ptr <= '0;");
            v.line(&format!(
                "else if ({} && {}) out_rd_ptr <= out_rd_ptr + {beat};",
                n.out_valid, n.out_ready
            ));
        });
        v.blank();
    }
}

fn emit_scalar_sidebands(v: &mut V, io: &IoConfig, names: &crate::export_config::PortNames) {
    if io.sidebands.is_empty() {
        return;
    }
    v.comment(
        "Scalar sidebands — sampled when start asserts; optional {name}_q echo \
         (not part of the activation BRAM datapath).",
    );
    for sb in &io.sidebands {
        let name = sanitize_port(&sb.name);
        let w = sb.bits.max(1) as usize;
        let signed = if sb.signed { "signed " } else { "" };
        v.line(&format!("logic {signed}[{}:0] {name}_r;", w - 1));
        v.always_ff_on(&names.clk, |v| {
            v.line(&format!("if ({}) {name}_r <= '0;", names.rst));
            v.line(&format!(
                "else if (cstate == C_IDLE && {}) {name}_r <= {name};",
                names.start
            ));
        });
        if sb.echo {
            v.line(&format!("assign {name}_q = {name}_r;"));
        }
    }
    v.blank();
}

// ── Per-stage (legacy) layout ───────────────────────────────────────

fn emit_per_stage(model: &Model, layers: &[LayerHandle], tune: &Tune, io: &IoConfig) -> Artifact {
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

    let n = &io.names;
    let mut bram_lens: Vec<usize> = Vec::with_capacity(layers.len() + 1);
    bram_lens.push(model.input_len);
    for l in layers {
        bram_lens.push(l.out_len);
    }
    let ports = io_ports::top_ports(model, io);
    let last_i = bram_lens.len() - 1;

    v.module("top", &[], &ports, |v| {
        v.line(&format!("localparam int INPUT_LEN = {};", model.input_len));
        v.line(&format!(
            "localparam int OUT_LEN   = {};",
            primary_out_len(model)
        ));
        v.blank();
        v.comment("─── activation BRAMs ───");
        for (i, len) in bram_lens.iter().enumerate() {
            let abits = bits_for(*len).max(1);
            v.line(&format!("logic [{}:0] a{i}_addr;", abits - 1));
            v.line(&format!("logic        a{i}_we;"));
            v.line(&format!("logic signed [7:0] a{i}_din;"));
            v.line(&format!("logic signed [7:0] a{i}_dout;"));
            v.line(&format!("block_ram #(.WIDTH(8), .DEPTH({len})) u_a{i} ("));
            v.block(|v| {
                v.line(&format!(".clk({}), .we(a{i}_we), .addr(a{i}_addr),", n.clk));
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
                v.line(&format!(".clk({}), .rst({}),", n.clk, n.rst));
                v.line(&format!(".start(l{i}_start), .done(l{i}_done),"));
                v.line(&format!(".x_addr(l{i}_x_addr), .x_dout(a{in_idx}_dout),"));
                v.line(&format!(
                    ".y_addr(l{i}_y_addr), .y_we(l{i}_y_we), .y_din(l{i}_y_din)"
                ));
            });
            v.line(");");
            v.blank();
        }

        emit_controller(v, layers.len(), n);

        v.comment("─── BRAM port routing ───");
        v.always_comb(|v| {
            v.line(&format!("if ({}) begin", n.start));
            v.block(|v| {
                v.line("a0_addr = l0_x_addr;");
                v.line("a0_we   = 1'b0;");
                v.line("a0_din  = 8'sd0;");
            });
            v.line("end else begin");
            v.block(|v| {
                if io.input.wants_memory() {
                    v.line(&format!("a0_addr = {};", n.in_addr));
                    v.line(&format!("a0_we   = {};", n.in_we));
                    v.line(&format!("a0_din  = {};", n.in_din));
                } else {
                    v.line("a0_addr = '0;");
                    v.line("a0_we   = 1'b0;");
                    v.line("a0_din  = 8'sd0;");
                }
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
            } else if io.output.wants_memory() {
                v.always_comb(|v| {
                    v.line(&format!("a{i}_we   = l{writer}_y_we;"));
                    v.line(&format!("a{i}_din  = l{writer}_y_din;"));
                    v.line(&format!(
                        "a{i}_addr = (cstate == C_DONE && {}) ? {} : l{writer}_y_addr;",
                        n.out_re, n.out_addr
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

        if io.output.wants_pred_port() {
            v.comment(&format!("Expose the final BRAM as `{}`.", n.pred));
            v.line(&format!("assign {} = a{last_i}_dout;", n.pred));
        }
        if io.output.wants_memory() {
            v.line(&format!("assign {} = a{last_i}_dout;", n.out_dout));
        }
        v.blank();

        emit_stream_sidebands(v, io, model, n);
        emit_scalar_sidebands(v, io, n);
    });

    Artifact {
        rel_path: "top.sv".into(),
        content: v.into_string(),
    }
}

/// Emit a Verilator-style testbench matching the soft I/O configuration.
pub fn emit_tb(model: &Model, io: &IoConfig) -> String {
    let in_len = model.input_len;
    let in_bits = bits_for(in_len.max(1));
    let n = &io.names;
    let out_bits = bits_for(primary_out_len(model).max(1));

    let mut v = V::new();
    v.banner("tb — image-driven testbench (Verilator)");
    v.line("`timescale 1ns/1ps");
    v.blank();
    v.module("tb", &[], &[], |v| {
        v.line(&format!("logic {} = 0;", n.clk));
        v.line(&format!("always #5 {0} = ~{0};", n.clk));
        v.line(&format!("logic {} = 1;", n.rst));
        v.line(&format!("logic {} = 0;", n.start));
        v.line(&format!("logic {};", n.done));
        if io.input.wants_memory() {
            v.line(&format!("logic [{}:0] {} = '0;", in_bits - 1, n.in_addr));
            v.line(&format!("logic {} = 0;", n.in_we));
            v.line(&format!("logic signed [7:0] {} = '0;", n.in_din));
        }
        if io.output.wants_pred_port() {
            v.line(&format!("logic signed [7:0] {};", n.pred));
        }
        if io.output.wants_memory() {
            v.line(&format!("logic [{}:0] {} = '0;", out_bits - 1, n.out_addr));
            v.line(&format!("logic {} = 0;", n.out_re));
            v.line(&format!("logic signed [7:0] {};", n.out_dout));
        }
        for sb in &io.sidebands {
            let name = sanitize_port(&sb.name);
            let w = sb.bits.max(1) as usize;
            let signed = if sb.signed { "signed " } else { "" };
            v.line(&format!("logic {signed}[{}:0] {name} = '0;", w - 1));
            if sb.echo {
                v.line(&format!("logic {signed}[{}:0] {name}_q;", w - 1));
            }
        }
        v.blank();

        v.line("top u_top (");
        v.block(|v| {
            let mut conns = vec![
                format!(".{}({})", n.clk, n.clk),
                format!(".{}({})", n.rst, n.rst),
                format!(".{}({})", n.start, n.start),
                format!(".{}({})", n.done, n.done),
            ];
            if io.input.wants_memory() {
                conns.push(format!(".{}({})", n.in_addr, n.in_addr));
                conns.push(format!(".{}({})", n.in_we, n.in_we));
                conns.push(format!(".{}({})", n.in_din, n.in_din));
            }
            for sb in &io.sidebands {
                let name = sanitize_port(&sb.name);
                conns.push(format!(".{name}({name})"));
                if sb.echo {
                    conns.push(format!(".{name}_q({name}_q)"));
                }
            }
            if io.output.wants_pred_port() {
                conns.push(format!(".{}({})", n.pred, n.pred));
            }
            if io.output.wants_memory() {
                conns.push(format!(".{}({})", n.out_addr, n.out_addr));
                conns.push(format!(".{}({})", n.out_re, n.out_re));
                conns.push(format!(".{}({})", n.out_dout, n.out_dout));
            }
            for (i, c) in conns.iter().enumerate() {
                let comma = if i + 1 < conns.len() { "," } else { "" };
                v.line(&format!("{c}{comma}"));
            }
        });
        v.line(");");
        v.blank();

        v.line(&format!(
            "logic signed [7:0] image_mem [0:{}];",
            in_len.max(1) - 1
        ));
        v.line("initial begin");
        v.block(|v| {
            v.line("$readmemh(\"tb_image.mem\", image_mem);");
            v.line(&format!("{} = 1; #20; {} = 0;", n.rst, n.rst));
            if io.input.wants_memory() {
                v.line(&format!("for (int i = 0; i < {in_len}; i++) begin"));
                v.block(|v| {
                    v.line(&format!("@(posedge {});", n.clk));
                    v.line(&format!("{} <= i[31:0];", n.in_addr));
                    v.line(&format!("{}   <= 1'b1;", n.in_we));
                    v.line(&format!("{}  <= image_mem[i];", n.in_din));
                });
                v.line("end");
                v.line(&format!("@(posedge {}); {} <= 1'b0;", n.clk, n.in_we));
            }
            v.line(&format!("@(posedge {}); {} <= 1'b1;", n.clk, n.start));
            v.line(&format!("wait ({});", n.done));
            v.line(&format!("@(posedge {}); {} <= 1'b0;", n.clk, n.start));
            if io.output.wants_pred_port() {
                v.line(&format!("$display(\"pred = %0d\", $signed({}));", n.pred));
            }
            for sb in &io.sidebands {
                if sb.echo {
                    let name = sanitize_port(&sb.name);
                    v.line(&format!("$display(\"{name}_q = %0d\", {name}_q);"));
                }
            }
            v.line("$finish;");
        });
        v.line("end");
    });
    v.into_string()
}
