// ───────────────────────────────────────────────────
// block_ram — synchronous-read single-port R/W BRAM
// ───────────────────────────────────────────────────

module block_ram #(
    parameter int WIDTH = 8,
    parameter int DEPTH = 256
) (
    input  logic                     clk,
    input  logic                     we,
    input  logic [$clog2(DEPTH)-1:0] addr,
    input  logic [WIDTH-1:0]         din,
    output logic [WIDTH-1:0]         dout
);
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (we) mem[addr] <= din;
        dout <= mem[addr];
    end
endmodule  // block_ram

