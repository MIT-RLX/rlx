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
// ────────────────────────────────────────────────────
// relu1_kernel — INT8 ReLU at zero_point=0, LEN=5408
// ────────────────────────────────────────────────────

module relu1_kernel (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       start,
    output logic                       done,
    output logic [12:0]              x_addr,
    input  logic signed [7:0]          x_dout,
    output logic [12:0]              y_addr,
    output logic                       y_we,
    output logic signed [7:0]          y_din
);
    localparam int LEN = 5408;
    localparam logic signed [7:0] ZP = 8'sd0;

    // State:
    //   S_IDLE  → wait for start
    //   S_READ  → issue x[i] read
    //   S_WAIT  → 1-cycle BRAM read latency
    //   S_WRITE → write y[i] = max(x_dout, ZP); advance i
    //   S_DONE  → assert done
    typedef enum logic [2:0] {
        S_IDLE, S_READ, S_WAIT, S_WRITE, S_DONE
    } state_t;
    state_t state, next;
    logic [12:0] i;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            i     <= '0;
        end else begin
            state <= next;
            if (state == S_IDLE && start)  i <= '0;
            if (state == S_WRITE)          i <= i + 1;
        end
    end

    always_comb begin
        next   = state;
        x_addr = i;
        y_addr = i;
        y_we   = 1'b0;
        y_din  = (x_dout < ZP) ? ZP : x_dout;
        done   = 1'b0;
        unique case (state)
            S_IDLE  : if (start) next = S_READ;
            S_READ  : next = S_WAIT;
            S_WAIT  : next = S_WRITE;
            S_WRITE : begin
                y_we = 1'b1;
                next = (i == LEN - 1) ? S_DONE : S_READ;
            end
            S_DONE  : begin done = 1'b1; if (!start) next = S_IDLE; end
        endcase
    end
endmodule  // relu1_kernel

