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

