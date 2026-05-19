// ──────────────────────────────────────────────────────────────
// fc_kernel — INT8 dense 400 → 10 (w8), x_zp=0 w_zp=0 out_zp=0
// ──────────────────────────────────────────────────────────────
// hints: fast_mac + Q0_15

module fc_kernel (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       start,
    output logic                       done,
    output logic [8:0]              x_addr,
    input  logic signed [7:0]          x_dout,
    output logic [3:0]              y_addr,
    output logic                       y_we,
    output logic signed [7:0]          y_din
);
    localparam int IN_F=400, OUT_F=10;
    localparam int X_ZP=0, W_ZP=0, OUT_ZP=0;
    localparam int W_BITS=8;
    localparam int W_LOG_LEN=4000;
    localparam int W_BYTE_LEN=4000;

    // Weight ROM (byte-addressed)
    logic [11:0] w_addr;
    logic        [7:0] w_byte;
    block_rom #(.WIDTH(8), .DEPTH(W_BYTE_LEN), .INIT_FILE("weights/fc_w.mem"))
    u_w_rom (.clk(clk), .addr(w_addr), .dout(w_byte));

    logic [1:0] w_lane_q;
    // Combinational weight unpack
    logic signed [31:0] w_val;
    weight_unpack #(.BITS(W_BITS)) u_w_unpack (
        .byte_in(w_byte),
        .lane(w_lane_q),
        .w_out(w_val)
    );

    // Bias ROM (i32)
    logic [3:0] b_addr;
    logic signed [31:0] b_dout;
    block_rom #(.WIDTH(32), .DEPTH(10), .INIT_FILE("weights/fc_b.mem"))
    u_b_rom (.clk(clk), .addr(b_addr), .dout(b_dout));

    // Requant ROMs (per-row)
    logic [3:0] r_addr;
    logic signed [15:0] m0_q;
    logic        [7:0]  sh_q;
    block_rom #(.WIDTH(16), .DEPTH(10), .INIT_FILE("weights/fc_m0.mem"))
    u_m0_rom (.clk(clk), .addr(r_addr), .dout(m0_q));
    block_rom #(.WIDTH(8),  .DEPTH(10), .INIT_FILE("weights/fc_sh.mem"))
    u_sh_rom (.clk(clk), .addr(r_addr), .dout(sh_q));
    always_comb r_addr = m_i;

    // Combinational requantize
    logic signed [7:0] q_out;
    requant_q15 u_requant (
        .acc(acc),
        .m0(m0_q[15:0]),
        .shift(sh_q[3:0]),
        .out_zp(OUT_ZP),
        .q(q_out)
    );

    logic [3:0] m_i;
    logic [8:0] k_i;
    logic signed [31:0] acc;

    // Address derivation
    logic [11:0] w_idx_logical;
    always_comb begin
        w_idx_logical = m_i * IN_F + k_i;
        w_addr = w_idx_logical;
        w_lane_q = 2'd0;
    end

    // Per-cycle MAC delta
    logic signed [31:0] xv_corrected;
    assign xv_corrected = $signed({{24{x_dout[7]}}, x_dout});
    logic signed [31:0] mac_delta;
    assign mac_delta = xv_corrected * w_val;

    // Pipelined FSM — 1 MAC/cycle in S_PIPE
    typedef enum logic [2:0] {
        S_IDLE, S_LOAD_BIAS, S_BIAS_WAIT, S_INIT_ACC, S_PIPE, S_WRITE, S_DONE
    } state_t;
    state_t state, next;
    logic prev_valid;
    logic done_issuing;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            m_i<='0; k_i<='0; acc<='0;
            prev_valid <= 1'b0;
            done_issuing <= 1'b0;
        end else begin
            state <= next;
            if (state == S_IDLE && start) begin m_i<='0; end
            if (state == S_INIT_ACC) begin
                acc <= b_dout;
                k_i <= '0;
                prev_valid <= 1'b0;
                done_issuing <= 1'b0;
            end
            if (state == S_PIPE) begin
                if (prev_valid) acc <= acc + mac_delta;
                if (!done_issuing) begin
                    if (k_i == IN_F - 1) begin
                        k_i <= '0;
                        done_issuing <= 1'b1;
                    end else k_i <= k_i + 1;
                    prev_valid <= 1'b1;
                end else begin
                    prev_valid <= 1'b0;
                end
            end
            if (state == S_WRITE) m_i <= m_i + 1;
        end
    end

    always_comb begin
        next   = state;
        x_addr = k_i;
        b_addr = m_i;
        y_addr = m_i;
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
                next = (m_i == OUT_F - 1) ? S_DONE : S_LOAD_BIAS;
            end
            S_DONE      : begin done = 1'b1; if (!start) next = S_IDLE; end
        endcase
    end
endmodule  // fc_kernel

