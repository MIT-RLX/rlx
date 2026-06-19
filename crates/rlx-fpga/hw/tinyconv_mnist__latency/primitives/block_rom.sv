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
// ─────────────────────────────────────────────────────────
// block_rom — synchronous-read ROM, $readmemh-initialized
// ─────────────────────────────────────────────────────────
// Yosys infers this as a block RAM when WIDTH ≤ 36 and the
// read port is registered.

module block_rom #(
    parameter int    WIDTH     = 8,
    parameter int    DEPTH     = 256,
    parameter string INIT_FILE = ""
) (
    input  logic                       clk,
    input  logic [$clog2(DEPTH)-1:0]   addr,
    output logic [WIDTH-1:0]           dout
);
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end
    end

    always_ff @(posedge clk) begin
        dout <= mem[addr];
    end
endmodule  // block_rom

