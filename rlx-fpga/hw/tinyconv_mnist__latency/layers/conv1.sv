// ─────────────────────────────────────────────────────────────────────────────────────────────
// conv1_kernel — INT8 conv2d 3x3 stride 1x1 on [28x28x1] → [26x26x8] (w8) — P=4 parallel MACs
// ─────────────────────────────────────────────────────────────────────────────────────────────
// hints: P=4 + fast_mac

module conv1_kernel (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       start,
    output logic                       done,
    output logic [9:0]              x_addr,
    input  logic signed [7:0]          x_dout,
    output logic [12:0]              y_addr,
    output logic                       y_we,
    output logic signed [7:0]          y_din
);
    localparam int H_IN=28, W_IN=28, C_IN=1;
    localparam int H_OUT=26, W_OUT=26, C_OUT=8;
    localparam int KH=3, KW=3, SH=1, SW=1;
    localparam int X_ZP=0, W_ZP=0, OUT_ZP=0;
    localparam int W_BITS=8;
    localparam int P=4;
    localparam int OC_BLOCKS=2;
    localparam int LANE_BYTE_LEN=18;

    // ─── P weight ROMs (one per oc-lane) ───
    logic [4:0] w_addr;
    logic        [7:0] w_byte [0:P-1];
    block_rom #(.WIDTH(8), .DEPTH(LANE_BYTE_LEN),
      .INIT_FILE("weights/conv1_w_l0.mem"))
    u_w_rom_l0 (.clk(clk), .addr(w_addr), .dout(w_byte[0]));
    block_rom #(.WIDTH(8), .DEPTH(LANE_BYTE_LEN),
      .INIT_FILE("weights/conv1_w_l1.mem"))
    u_w_rom_l1 (.clk(clk), .addr(w_addr), .dout(w_byte[1]));
    block_rom #(.WIDTH(8), .DEPTH(LANE_BYTE_LEN),
      .INIT_FILE("weights/conv1_w_l2.mem"))
    u_w_rom_l2 (.clk(clk), .addr(w_addr), .dout(w_byte[2]));
    block_rom #(.WIDTH(8), .DEPTH(LANE_BYTE_LEN),
      .INIT_FILE("weights/conv1_w_l3.mem"))
    u_w_rom_l3 (.clk(clk), .addr(w_addr), .dout(w_byte[3]));

    logic [1:0] w_lane_q;
    // Per-lane combinational weight unpack
    logic signed [31:0] w_val [0:P-1];
    genvar gi;
    generate for (gi = 0; gi < P; gi = gi + 1) begin : g_unpack
        weight_unpack #(.BITS(W_BITS)) u_w_unpack (
            .byte_in(w_byte[gi]),
            .lane(w_lane_q),
            .w_out(w_val[gi])
        );
    end endgenerate

    // ─── P bias ROMs ───
    logic [0:0] b_addr;
    logic signed [31:0] b_dout [0:P-1];
    block_rom #(.WIDTH(32), .DEPTH(2),
      .INIT_FILE("weights/conv1_b_l0.mem"))
    u_b_rom_l0 (.clk(clk), .addr(b_addr), .dout(b_dout[0]));
    block_rom #(.WIDTH(32), .DEPTH(2),
      .INIT_FILE("weights/conv1_b_l1.mem"))
    u_b_rom_l1 (.clk(clk), .addr(b_addr), .dout(b_dout[1]));
    block_rom #(.WIDTH(32), .DEPTH(2),
      .INIT_FILE("weights/conv1_b_l2.mem"))
    u_b_rom_l2 (.clk(clk), .addr(b_addr), .dout(b_dout[2]));
    block_rom #(.WIDTH(32), .DEPTH(2),
      .INIT_FILE("weights/conv1_b_l3.mem"))
    u_b_rom_l3 (.clk(clk), .addr(b_addr), .dout(b_dout[3]));

    // ─── P requant ROMs (or shared localparam) ───
    logic [0:0] r_addr;
    logic signed [31:0] m0_dout [0:P-1];
    logic        [7:0]  sh_dout [0:P-1];
    block_rom #(.WIDTH(32), .DEPTH(2),
      .INIT_FILE("weights/conv1_m0_l0.mem"))
    u_m0_rom_l0 (.clk(clk), .addr(r_addr), .dout(m0_dout[0]));
    block_rom #(.WIDTH(8), .DEPTH(2),
      .INIT_FILE("weights/conv1_sh_l0.mem"))
    u_sh_rom_l0 (.clk(clk), .addr(r_addr), .dout(sh_dout[0]));
    block_rom #(.WIDTH(32), .DEPTH(2),
      .INIT_FILE("weights/conv1_m0_l1.mem"))
    u_m0_rom_l1 (.clk(clk), .addr(r_addr), .dout(m0_dout[1]));
    block_rom #(.WIDTH(8), .DEPTH(2),
      .INIT_FILE("weights/conv1_sh_l1.mem"))
    u_sh_rom_l1 (.clk(clk), .addr(r_addr), .dout(sh_dout[1]));
    block_rom #(.WIDTH(32), .DEPTH(2),
      .INIT_FILE("weights/conv1_m0_l2.mem"))
    u_m0_rom_l2 (.clk(clk), .addr(r_addr), .dout(m0_dout[2]));
    block_rom #(.WIDTH(8), .DEPTH(2),
      .INIT_FILE("weights/conv1_sh_l2.mem"))
    u_sh_rom_l2 (.clk(clk), .addr(r_addr), .dout(sh_dout[2]));
    block_rom #(.WIDTH(32), .DEPTH(2),
      .INIT_FILE("weights/conv1_m0_l3.mem"))
    u_m0_rom_l3 (.clk(clk), .addr(r_addr), .dout(m0_dout[3]));
    block_rom #(.WIDTH(8), .DEPTH(2),
      .INIT_FILE("weights/conv1_sh_l3.mem"))
    u_sh_rom_l3 (.clk(clk), .addr(r_addr), .dout(sh_dout[3]));

    // Requantize epilogue (sequential through P lanes)
    logic [1:0] req_lane;
    logic signed [31:0] acc_mux;
    logic signed [31:0] m0_mux;
    logic        [7:0]  sh_mux;
    always_comb begin
        acc_mux = acc[req_lane];
        m0_mux  = m0_dout[req_lane];
        sh_mux  = sh_dout[req_lane];
    end
    logic signed [7:0] q_out;
    requant_q31 u_requant (
        .acc(acc_mux),
        .m0(m0_mux),
        .shift(sh_mux[4:0]),
        .out_zp(OUT_ZP),
        .q(q_out)
    );

    logic signed [31:0] acc [0:P-1];
    logic [4:0] oh, ow;
    logic [0:0] oc_block;
    logic [1:0] kh_i, kw_i;
    logic [0:0] ic;

    // Address derivation
    logic [9:0] in_idx;
    logic [4:0] w_idx_logical;
    always_comb begin
        in_idx = ((oh*SH + kh_i) * W_IN + (ow*SW + kw_i)) * C_IN + ic;
        // Per-lane logical index (same for every lane — lane index lives outside)
        w_idx_logical = (oc_block * KH + kh_i) * KW * C_IN + kw_i * C_IN + ic;
        w_addr = w_idx_logical;
        w_lane_q = 2'd0;
    end

    // Per-lane MAC delta (combinational)
    logic signed [31:0] xv_corrected;
    assign xv_corrected = $signed({{24{x_dout[7]}}, x_dout});
    logic signed [31:0] mac_delta [0:P-1];
    genvar gm;
    generate for (gm = 0; gm < P; gm = gm + 1) begin : g_mac
        assign mac_delta[gm] = xv_corrected * w_val[gm];
    end endgenerate

    // Pipelined FSM — 1 (P-wide) MAC/cycle in S_PIPE
    typedef enum logic [2:0] {
        S_IDLE, S_LOAD_BIAS, S_BIAS_WAIT, S_INIT_ACC, S_PIPE, S_WRITE, S_DONE
    } state_t;
    state_t state, next;
    logic prev_valid;
    logic done_issuing;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            oh<='0; ow<='0; oc_block<='0;
            kh_i<='0; kw_i<='0; ic<='0;
            req_lane<='0;
            prev_valid <= 1'b0;
            done_issuing <= 1'b0;
            for (int i = 0; i < P; i = i + 1) acc[i] <= 32'sd0;
        end else begin
            state <= next;
            if (state == S_IDLE && start) begin
                oh<='0; ow<='0; oc_block<='0;
                req_lane<='0;
            end
            if (state == S_INIT_ACC) begin
                for (int i = 0; i < P; i = i + 1) acc[i] <= b_dout[i];
                kh_i <= '0; kw_i <= '0; ic <= '0;
                prev_valid <= 1'b0;
                done_issuing <= 1'b0;
            end
            if (state == S_PIPE) begin
                if (prev_valid) for (int i = 0; i < P; i = i + 1)
                    acc[i] <= acc[i] + mac_delta[i];
                if (!done_issuing) begin
                    if (ic == C_IN - 1) begin
                        ic <= '0;
                        if (kw_i == KW - 1) begin
                            kw_i <= '0;
                            if (kh_i == KH - 1) begin
                                kh_i <= '0;
                                done_issuing <= 1'b1;
                            end else kh_i <= kh_i + 1;
                        end else kw_i <= kw_i + 1;
                    end else ic <= ic + 1;
                    prev_valid <= 1'b1;
                end else begin
                    prev_valid <= 1'b0;
                end
            end
            if (state == S_WRITE) begin
                if (req_lane == P - 1) begin
                    req_lane <= '0;
                    if (oc_block == OC_BLOCKS - 1) begin
                        oc_block <= '0;
                        if (ow == W_OUT - 1) begin
                            ow <= '0;
                            oh <= oh + 1;
                        end else ow <= ow + 1;
                    end else oc_block <= oc_block + 1;
                end else req_lane <= req_lane + 1;
            end
        end
    end

    always_comb begin
        next   = state;
        x_addr = in_idx;
        b_addr = oc_block;
        r_addr = oc_block;
        y_addr = (oh * W_OUT + ow) * C_OUT + oc_block * P + req_lane;
        y_we   = 1'b0;
        y_din  = q_out;
        done   = 1'b0;
        unique case (state)
            S_IDLE      : if (start) next = S_LOAD_BIAS;
            S_LOAD_BIAS : next = S_BIAS_WAIT;
            S_BIAS_WAIT : next = S_INIT_ACC;
            S_INIT_ACC  : next = S_PIPE;
            S_PIPE      : next = (done_issuing && prev_valid) ? S_WRITE : S_PIPE;
            S_WRITE     : begin
                y_we = 1'b1;
                if (req_lane == P - 1) begin
                    if (oh == H_OUT - 1 && ow == W_OUT - 1 && oc_block == OC_BLOCKS - 1)
                        next = S_DONE;
                    else
                        next = S_LOAD_BIAS;
                end else next = S_WRITE;
            end
            S_DONE      : begin done = 1'b1; if (!start) next = S_IDLE; end
        endcase
    end
endmodule  // conv1_kernel

