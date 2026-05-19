// ─────────────────────────────────────────────────────────
// tb — TinyConv-MNIST testbench (image-driven, Verilator)
// ─────────────────────────────────────────────────────────
`timescale 1ns/1ps

module tb (
);
    logic clk = 0;
    always #5 clk = ~clk;
    logic rst = 1;
    logic start = 0;
    logic done;
    logic [9:0] in_addr = '0;
    logic in_we = 0;
    logic signed [7:0] in_din = '0;
    logic signed [7:0] pred;

    top u_top (
        .clk(clk), .rst(rst), .start(start), .done(done),
        .in_addr(in_addr), .in_we(in_we), .in_din(in_din),
        .pred(pred)
    );

    logic signed [7:0] image_mem [0:783];
    initial begin
        $readmemh("tb_image.mem", image_mem);
        rst = 1; #20; rst = 0;
        for (int i = 0; i < 784; i++) begin
            @(posedge clk);
            in_addr <= i[31:0];
            in_we   <= 1'b1;
            in_din  <= image_mem[i];
        end
        @(posedge clk); in_we <= 1'b0;
        @(posedge clk); start <= 1'b1;
        wait (done);
        @(posedge clk); start <= 1'b0;
        $display("pred = %0d", $signed(pred));
        $finish;
    end
endmodule  // tb

