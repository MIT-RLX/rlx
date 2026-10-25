// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! UART host link: a `board_top` that lets a PC run inference on the board.
//!
//! Protocol (8N1, [`UART_BAUD`], one exchange per inference):
//!
//! ```text
//!   host → FPGA   0xA5, x[0], x[1], …, x[INPUT_LEN-1]     (i8 activations)
//!   FPGA → host   pred                                    (if the export has a pred port)
//!                 y[0], …, y[OUT_LEN-1]                   (if it has memory readout)
//! ```
//!
//! The shell writes the bytes through `top`'s memory-load port, raises
//! `start`, waits for `done`, then streams the results back. A power-on
//! reset means no button press is needed after programming.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::codegen::io_ports::{primary_out_len, sanitize_port};
use crate::codegen::relu::bits_for;
use crate::export_config::IoConfig;
use crate::model::Model;
use crate::xc7::{UART_BAUD, UART_SYNC, Xc7Board};

/// `uart_rx` / `uart_tx`: 8N1, `CLKS_PER_BIT` clocks per bit.
pub const UART_PRIMITIVES: &str = r#"// uart_rx — 8N1 receiver; 2-FF synchronizer, mid-bit sampling.
module uart_rx #(
    parameter int CLKS_PER_BIT = 868
) (
    input  logic       clk,
    input  logic       rst,
    input  logic       rx,
    output logic       valid,
    output logic [7:0] data
);
    logic rx_m = 1'b1, rx_s = 1'b1;
    always_ff @(posedge clk) begin
        rx_m <= rx;
        rx_s <= rx_m;
    end

    localparam logic [1:0] R_IDLE = 2'd0, R_START = 2'd1, R_DATA = 2'd2, R_STOP = 2'd3;
    logic [1:0]  st;
    logic [15:0] cnt;
    logic [2:0]  bitn;

    always_ff @(posedge clk) begin
        valid <= 1'b0;
        if (rst) begin
            st   <= R_IDLE;
            cnt  <= '0;
            bitn <= '0;
        end else begin
            case (st)
                R_IDLE: if (!rx_s) begin
                    st  <= R_START;
                    cnt <= '0;
                end
                R_START: if (cnt == 16'((CLKS_PER_BIT - 1) / 2)) begin
                    cnt <= '0;
                    if (!rx_s) begin
                        st   <= R_DATA;
                        bitn <= '0;
                    end else st <= R_IDLE;
                end else cnt <= cnt + 16'd1;
                R_DATA: if (cnt == 16'(CLKS_PER_BIT - 1)) begin
                    cnt        <= '0;
                    data[bitn] <= rx_s;
                    if (bitn == 3'd7) st <= R_STOP;
                    else bitn <= bitn + 3'd1;
                end else cnt <= cnt + 16'd1;
                default: if (cnt == 16'(CLKS_PER_BIT - 1)) begin
                    cnt   <= '0;
                    st    <= R_IDLE;
                    valid <= rx_s;  // drop frames without a stop bit
                end else cnt <= cnt + 16'd1;
            endcase
        end
    end
endmodule  // uart_rx

// uart_tx — 8N1 transmitter. `send` is accepted while `!busy`.
module uart_tx #(
    parameter int CLKS_PER_BIT = 868
) (
    input  logic       clk,
    input  logic       rst,
    input  logic       send,
    input  logic [7:0] data,
    output logic       busy,
    output logic       tx
);
    logic [9:0]  shreg;
    logic [3:0]  nbits;
    logic [15:0] cnt;

    always_ff @(posedge clk) begin
        if (rst) begin
            busy  <= 1'b0;
            shreg <= '1;
            nbits <= '0;
            cnt   <= '0;
        end else if (!busy) begin
            if (send) begin
                shreg <= {1'b1, data, 1'b0};
                busy  <= 1'b1;
                nbits <= '0;
                cnt   <= '0;
            end
        end else if (cnt == 16'(CLKS_PER_BIT - 1)) begin
            cnt   <= '0;
            shreg <= {1'b1, shreg[9:1]};
            if (nbits == 4'd9) busy <= 1'b0;
            else nbits <= nbits + 4'd1;
        end else cnt <= cnt + 16'd1;
    end
    assign tx = busy ? shreg[0] : 1'b1;
endmodule  // uart_tx
"#;

/// True when [`emit_uart_board_top`] can drive this I/O configuration.
pub fn uart_supported(io: &IoConfig) -> Result<(), String> {
    if !io.input.wants_memory() {
        return Err(format!(
            "rlx-fpga: the UART board shell loads inputs through the memory port; \
             input iface {:?} has none (use InputIface::Memory)",
            io.input
        ));
    }
    if !io.output.wants_pred_port() && !io.output.wants_memory() {
        return Err(format!(
            "rlx-fpga: the UART board shell returns `pred` and/or memory readout; \
             output iface {:?} has neither",
            io.output
        ));
    }
    Ok(())
}

/// `board_top` for an openXC7 board: power-on reset + UART host link around
/// the soft `top`. Includes the UART primitives so the file is self-contained.
pub fn emit_uart_board_top(model: &Model, io: &IoConfig, board: Xc7Board) -> String {
    let n = &io.names;
    let pins = board.pins();
    let in_len = model.input_len.max(1);
    let in_ab = bits_for(in_len);
    let out_len = primary_out_len(model).max(1);
    let out_ab = bits_for(out_len);
    let send_pred = io.output.wants_pred_port();
    let send_mem = io.output.wants_memory();
    let reply_bytes = usize::from(send_pred) + if send_mem { out_len } else { 0 };
    let rst_expr = if pins.rst_active_low {
        "~rst_btn"
    } else {
        "rst_btn"
    };

    let mut sideband_decls = String::new();
    let mut sideband_conns = String::new();
    for sb in &io.sidebands {
        let name = sanitize_port(&sb.name);
        let w = sb.bits.max(1);
        let signed = if sb.signed { "signed " } else { "" };
        sideband_decls.push_str(&format!(
            "    logic {signed}[{}:0] {name};\n    assign {name} = '0;  // tied off on the board\n",
            w - 1
        ));
        sideband_conns.push_str(&format!(",\n        .{name}({name})"));
        if sb.echo {
            sideband_decls.push_str(&format!("    logic {signed}[{}:0] {name}_q;\n", w - 1));
            sideband_conns.push_str(&format!(",\n        .{name}_q({name}_q)"));
        }
    }

    let mut out_decls = String::new();
    let mut out_conns = String::new();
    if send_pred {
        out_decls.push_str("    logic signed [7:0] pred_w;\n");
        out_conns.push_str(&format!(",\n        .{}(pred_w)", n.pred));
    }
    if send_mem {
        out_decls.push_str(&format!(
            "    logic [{}:0] out_addr;\n    logic        out_re;\n    logic signed [7:0] out_dout;\n",
            out_ab - 1
        ));
        out_conns.push_str(&format!(
            ",\n        .{}(out_addr), .{}(out_re), .{}(out_dout)",
            n.out_addr, n.out_re, n.out_dout
        ));
    }

    let after_done = if send_pred { "H_PRED" } else { "H_ADDR" };
    let after_pred = if send_mem { "H_ADDR" } else { "H_DRAIN" };
    let pred_arm = if send_pred {
        format!(
            r#"                H_PRED: if (tx_free) begin
                    tx_byte <= pred_w;
                    pred_q  <= pred_w;
                    tx_send <= 1'b1;
                    hs      <= {after_pred};
                end
"#
        )
    } else {
        String::new()
    };
    let mem_arms = if send_mem {
        format!(
            r#"                H_ADDR: begin
                    out_re   <= 1'b1;
                    out_addr <= oidx;
                    hs       <= H_ADDR_WAIT;
                end
                H_ADDR_WAIT: hs <= H_SEND;  // top's BRAM read: 1 cycle after out_addr
                H_SEND: if (tx_free) begin
                    tx_byte <= out_dout;
                    tx_send <= 1'b1;
                    if (oidx == {oab}'(OUT_LEN - 1)) hs <= H_DRAIN;
                    else begin
                        oidx <= oidx + {oab}'d1;
                        hs   <= H_ADDR;
                    end
                end
"#,
            oab = out_ab
        )
    } else {
        String::new()
    };
    let mem_reset = if send_mem {
        "            out_re   <= 1'b0;\n            out_addr <= '0;\n"
    } else {
        ""
    };
    let mem_release = if send_mem {
        "                    out_re <= 1'b0;\n"
    } else {
        ""
    };
    let led_pred = if send_pred { "pred_q[1:0]" } else { "2'd0" };
    let pred_q_decl = if send_pred {
        "    logic signed [7:0] pred_q;\n"
    } else {
        ""
    };

    format!(
        r#"// ─────────────────────────────────────────────────────────────────
// board_top — {board} (openXC7) UART host link over soft `top` ({model})
// ─────────────────────────────────────────────────────────────────
// {baud} 8N1. Host sends 0x{sync:02X} + {in_len} input bytes; FPGA replies
// with {reply_bytes} byte(s): {reply_desc}.
// LED0 = ready for input, LED1 = running, LED[3:2] = pred[1:0].

module board_top #(
    parameter int CLKS_PER_BIT = {cpb}
) (
    input  logic       clk,
    input  logic       rst_btn,
    input  logic       uart_rx,
    output logic       uart_tx,
    output logic [3:0] led
);
    localparam int INPUT_LEN = {in_len};
    localparam int OUT_LEN   = {out_len};
    localparam logic [7:0] SYNC = 8'h{sync:02X};

    // Power-on reset (FPGA flops load their init value at configuration),
    // OR'd with the synchronized board button.
    logic [15:0] por = 16'hFFFF;
    logic [1:0]  btn_s = 2'b00;
    always_ff @(posedge clk) begin
        por   <= {{por[14:0], 1'b0}};
        btn_s <= {{btn_s[0], {rst_expr}}};
    end
    logic rst;
    assign rst = por[15] | btn_s[1];

    logic       rx_valid;
    logic [7:0] rx_data;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .rst(rst), .rx(uart_rx), .valid(rx_valid), .data(rx_data)
    );
    logic       tx_send;
    logic [7:0] tx_byte;
    logic       tx_busy;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .rst(rst), .send(tx_send), .data(tx_byte), .busy(tx_busy), .tx(uart_tx)
    );
    // `tx_send` is registered, so `busy` rises a cycle after it.
    logic tx_free;
    assign tx_free = !tx_busy && !tx_send;

    logic              start;
    logic              done;
    logic [{in_msb}:0]        in_addr;
    logic              in_we;
    logic signed [7:0] in_din;
{out_decls}{sideband_decls}
    top u_top (
        .{clk}(clk), .{rst}(rst), .{start}(start), .{done}(done),
        .{in_addr}(in_addr), .{in_we}(in_we), .{in_din}(in_din){out_conns}{sideband_conns}
    );

    localparam logic [3:0] H_SYNC = 4'd0, H_LOAD = 4'd1, H_RUN = 4'd2, H_SETTLE = 4'd3,
                           H_PRED = 4'd4, H_ADDR = 4'd5, H_ADDR_WAIT = 4'd6, H_SEND = 4'd7,
                           H_DRAIN = 4'd8;
    logic [3:0] hs;
    logic [{in_msb}:0] idx;
    logic [{out_msb}:0] oidx;
{pred_q_decl}
    always_ff @(posedge clk) begin
        tx_send <= 1'b0;
        in_we   <= 1'b0;
        if (rst) begin
            hs      <= H_SYNC;
            start   <= 1'b0;
            idx     <= '0;
            oidx    <= '0;
            in_addr <= '0;
            in_din  <= '0;
            tx_byte <= '0;
{mem_reset}        end else begin
            case (hs)
                H_SYNC: if (rx_valid && rx_data == SYNC) begin
                    idx <= '0;
                    hs  <= H_LOAD;
                end
                H_LOAD: if (rx_valid) begin
                    in_addr <= idx;
                    in_din  <= rx_data;
                    in_we   <= 1'b1;
                    if (idx == {in_ab}'(INPUT_LEN - 1)) hs <= H_RUN;
                    else idx <= idx + {in_ab}'d1;
                end
                H_RUN: begin
                    start <= 1'b1;  // held until the reply is out; drops `top` back to idle
                    if (done) hs <= H_SETTLE;
                end
                H_SETTLE: begin  // `pred` is a registered BRAM read
                    oidx <= '0;
                    hs   <= {after_done};
                end
{pred_arm}{mem_arms}                H_DRAIN: if (tx_free) begin
                    start <= 1'b0;
{mem_release}                    hs    <= H_SYNC;
                end
                default: hs <= H_SYNC;
            endcase
        end
    end

    assign led = {{{led_pred}, hs == H_RUN, hs == H_SYNC}};
endmodule  // board_top

{prims}"#,
        board = board.name(),
        model = model.name,
        baud = UART_BAUD,
        sync = UART_SYNC,
        reply_desc = match (send_pred, send_mem) {
            (true, true) => format!("pred, then {out_len} logits"),
            (true, false) => "pred".to_string(),
            _ => format!("{out_len} logits"),
        },
        cpb = board.clks_per_bit(),
        in_msb = in_ab - 1,
        out_msb = out_ab - 1,
        clk = n.clk,
        rst = n.rst,
        start = n.start,
        done = n.done,
        in_addr = n.in_addr,
        in_we = n.in_we,
        in_din = n.in_din,
        prims = UART_PRIMITIVES,
    )
}
