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
// ────────────────────────────────────────────────
// argmax_kernel — argmax over LEN=10 i8 elements
// ────────────────────────────────────────────────

module argmax_kernel (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       start,
    output logic                       done,
    output logic [3:0]              x_addr,
    input  logic signed [7:0]          x_dout,
    output logic [0:0]                 y_addr,
    output logic                       y_we,
    output logic signed [7:0]          y_din
);
    localparam int LEN = 10;

    typedef enum logic [2:0] {
        S_IDLE, S_READ, S_WAIT, S_UPDATE, S_WRITE, S_DONE
    } state_t;
    state_t state, next;
    logic [3:0] i;
    logic [3:0] best_idx;
    logic signed [7:0] best_val;

    always_ff @(posedge clk) begin
        if (rst) begin
            state    <= S_IDLE;
            i        <= '0;
            best_idx <= '0;
            best_val <= 8'sh80;
        end else begin
            state <= next;
            if (state == S_IDLE && start) begin
                i        <= '0;
                best_idx <= '0;
                best_val <= 8'sh80;  // i8::MIN
            end
            if (state == S_UPDATE) begin
                if (x_dout > best_val) begin
                    best_val <= x_dout;
                    best_idx <= i;
                end
                i <= i + 1;
            end
        end
    end

    always_comb begin
        next   = state;
        x_addr = i;
        y_addr = 1'b0;
        y_we   = 1'b0;
        y_din  = best_idx[7:0];
        done   = 1'b0;
        unique case (state)
            S_IDLE   : if (start) next = S_READ;
            S_READ   : next = S_WAIT;
            S_WAIT   : next = S_UPDATE;
            S_UPDATE : next = (i == LEN - 1) ? S_WRITE : S_READ;
            S_WRITE  : begin y_we = 1'b1; next = S_DONE; end
            S_DONE   : begin done = 1'b1; if (!start) next = S_IDLE; end
        endcase
    end
endmodule  // argmax_kernel

