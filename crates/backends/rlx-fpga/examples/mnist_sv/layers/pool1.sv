// ─────────────────────────────────────────────────────────────────────
// pool1_kernel — INT8 maxpool 2x2 stride 2x2 on [26x26x8] → [13x13x8]
// ─────────────────────────────────────────────────────────────────────

module pool1_kernel (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       start,
    output logic                       done,
    output logic [12:0]              x_addr,
    input  logic signed [7:0]          x_dout,
    output logic [10:0]              y_addr,
    output logic                       y_we,
    output logic signed [7:0]          y_din
);
    localparam int H_IN=26, W_IN=26, C=8;
    localparam int H_OUT=13, W_OUT=13;
    localparam int KH=2, KW=2, SH=2, SW=2;

    typedef enum logic [2:0] {
        S_IDLE, S_READ, S_WAIT, S_UPDATE, S_WRITE, S_DONE
    } state_t;
    state_t state, next;

    logic [3:0] oh, ow, oc;
    logic [0:0] kh_i, kw_i;
    logic signed [7:0] best;

    // Input address = ((oh*SH + kh_i) * W_IN + (ow*SW + kw_i)) * C + oc
    // Output addr   = (oh * W_OUT + ow) * C + oc
    logic [12:0] in_idx;
    logic [10:0] out_idx;
    always_comb begin
        in_idx  = ((oh*SH + kh_i) * W_IN + (ow*SW + kw_i)) * C + oc;
        out_idx = (oh * W_OUT + ow) * C + oc;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            oh    <= '0;  ow <= '0;  oc <= '0;
            kh_i  <= '0;  kw_i <= '0;
            best  <= 8'sh80;
        end else begin
            state <= next;
            if (state == S_IDLE && start) begin
                oh <= '0; ow <= '0; oc <= '0;
                kh_i <= '0; kw_i <= '0;
                best <= 8'sh80;
            end
            if (state == S_UPDATE) begin
                if (x_dout > best) best <= x_dout;
                // advance kw_i, kh_i — the inner pool window
                if (kw_i == KW - 1) begin
                    kw_i <= '0;
                    if (kh_i == KH - 1) begin
                        kh_i <= '0;  // window done; S_WRITE next
                    end else kh_i <= kh_i + 1;
                end else kw_i <= kw_i + 1;
            end
            if (state == S_WRITE) begin
                best <= 8'sh80;
                // advance oc, ow, oh
                if (oc == C - 1) begin
                    oc <= '0;
                    if (ow == W_OUT - 1) begin
                        ow <= '0;
                        oh <= oh + 1;  // overflow caught by FSM transition
                    end else ow <= ow + 1;
                end else oc <= oc + 1;
            end
        end
    end

    always_comb begin
        next   = state;
        x_addr = in_idx;
        y_addr = out_idx;
        y_we   = 1'b0;
        y_din  = best;
        done   = 1'b0;
        unique case (state)
            S_IDLE   : if (start) next = S_READ;
            S_READ   : next = S_WAIT;
            S_WAIT   : next = S_UPDATE;
            S_UPDATE : next = (kh_i == KH - 1 && kw_i == KW - 1) ? S_WRITE : S_READ;
            S_WRITE  : begin
                y_we = 1'b1;
                next = (oh == H_OUT - 1 && ow == W_OUT - 1 && oc == C - 1) ? S_DONE : S_READ;
            end
            S_DONE   : begin done = 1'b1; if (!start) next = S_IDLE; end
        endcase
    end
endmodule  // pool1_kernel

