// ──────────────────────────────────────────────────────────────
// fc_kernel — INT8 dense 400 → 10 (w8), x_zp=0 w_zp=0 out_zp=0
// ──────────────────────────────────────────────────────────────

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

    // Combinational weight unpack
    logic [1:0]              w_lane_q;
    logic signed [31:0]      w_val;
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

    // Requant ROMs
    logic [3:0] r_addr;
    logic signed [31:0] m0_dout;
    logic        [7:0]  sh_dout;
    block_rom #(.WIDTH(32), .DEPTH(10), .INIT_FILE("weights/fc_m0.mem"))
    u_m0_rom (.clk(clk), .addr(r_addr), .dout(m0_dout));
    block_rom #(.WIDTH(8),  .DEPTH(10), .INIT_FILE("weights/fc_sh.mem"))
    u_sh_rom (.clk(clk), .addr(r_addr), .dout(sh_dout));

    // Combinational requantize
    logic signed [7:0] q_out;
    requant_q31 u_requant (
        .acc(acc),
        .m0(m0_dout),
        .shift(sh_dout[4:0]),
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

    typedef enum logic [3:0] {
        S_IDLE, S_LOAD_BIAS, S_BIAS_WAIT, S_INIT_ACC, S_READ, S_WAIT, S_MAC, S_REQ_ADDR, S_REQ_WAIT, S_REQ_DO, S_WRITE, S_DONE
    } state_t;
    state_t state, next;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            m_i<='0; k_i<='0; acc<='0;
        end else begin
            state <= next;
            if (state == S_IDLE && start) begin m_i<='0; k_i<='0; end
            if (state == S_INIT_ACC) acc <= b_dout;
            if (state == S_MAC) begin
                acc <= acc + ($signed({{24{x_dout[7]}}, x_dout}) - X_ZP)
                           * (w_val - W_ZP);
                if (k_i == IN_F - 1) k_i <= '0;
                else                 k_i <= k_i + 1;
            end
            if (state == S_WRITE) begin
                m_i <= m_i + 1;
            end
        end
    end

    always_comb begin
        next   = state;
        x_addr = k_i;
        b_addr = m_i;
        r_addr = m_i;
        y_addr = m_i;
        y_we   = 1'b0;
        y_din  = q_out;
        done   = 1'b0;
        unique case (state)
            S_IDLE      : if (start) next = S_LOAD_BIAS;
            S_LOAD_BIAS : next = S_BIAS_WAIT;
            S_BIAS_WAIT : next = S_INIT_ACC;
            S_INIT_ACC  : next = S_READ;
            S_READ      : next = S_WAIT;
            S_WAIT      : next = S_MAC;
            S_MAC       : next = (k_i == IN_F - 1) ? S_REQ_ADDR : S_READ;
            S_REQ_ADDR  : next = S_REQ_WAIT;
            S_REQ_WAIT  : next = S_REQ_DO;
            S_REQ_DO    : next = S_WRITE;
            S_WRITE     : begin
                y_we = 1'b1;
                next = (m_i == OUT_F - 1) ? S_DONE : S_LOAD_BIAS;
            end
            S_DONE      : begin done = 1'b1; if (!start) next = S_IDLE; end
        endcase
    end
endmodule  // fc_kernel

