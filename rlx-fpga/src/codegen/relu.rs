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

//! ReLU kernel: `out[i] = max(x[i], zero_point)` for `i ∈ [0, LEN)`.
//! Single-MAC FSM: 1 cycle to issue read, 1 cycle for BRAM data, 1
//! cycle to write — 3 cycles per element. Could be pipelined to 1
//! cycle/elem; not worth it for our sizes.

use super::{Artifact, LayerArtifacts};
use crate::model::Layer;
use crate::verilog::V;

pub fn emit(layer: &Layer) -> LayerArtifacts {
    let (name, len, zp) = match layer {
        Layer::Relu {
            name,
            len,
            zero_point,
        } => (*name, *len, *zero_point),
        _ => unreachable!("relu::emit called with non-Relu layer"),
    };
    let module_name = format!("{name}_kernel");
    let instance_name = format!("u_{name}");

    let mut v = V::new();
    v.banner(&format!(
        "{module_name} — INT8 ReLU at zero_point={zp}, LEN={len}"
    ));
    v.blank();

    let addr_bits = bits_for(len);

    v.module(
        &module_name,
        &[],
        &[
            "input  logic                       clk".into(),
            "input  logic                       rst".into(),
            "input  logic                       start".into(),
            "output logic                       done".into(),
            format!("output logic [{}:0]              x_addr", addr_bits - 1),
            "input  logic signed [7:0]          x_dout".into(),
            format!("output logic [{}:0]              y_addr", addr_bits - 1),
            "output logic                       y_we".into(),
            "output logic signed [7:0]          y_din".into(),
        ],
        |v| {
            v.line(&format!("localparam int LEN = {len};"));
            v.line(&format!(
                "localparam logic signed [7:0] ZP = 8'sd{};",
                zp.clamp(i8::MIN as i32, i8::MAX as i32)
            ));
            v.blank();

            v.comment("State:");
            v.comment("  S_IDLE  → wait for start");
            v.comment("  S_READ  → issue x[i] read");
            v.comment("  S_WAIT  → 1-cycle BRAM read latency");
            v.comment("  S_WRITE → write y[i] = max(x_dout, ZP); advance i");
            v.comment("  S_DONE  → assert done");
            v.line("typedef enum logic [2:0] {");
            v.block(|v| v.line("S_IDLE, S_READ, S_WAIT, S_WRITE, S_DONE"));
            v.line("} state_t;");
            v.line("state_t state, next;");
            v.line(&format!("logic [{}:0] i;", addr_bits - 1));
            v.blank();

            v.always_ff(|v| {
                v.line("if (rst) begin");
                v.block(|v| {
                    v.line("state <= S_IDLE;");
                    v.line("i     <= '0;");
                });
                v.line("end else begin");
                v.block(|v| {
                    v.line("state <= next;");
                    v.line("if (state == S_IDLE && start)  i <= '0;");
                    v.line("if (state == S_WRITE)          i <= i + 1;");
                });
                v.line("end");
            });
            v.blank();

            v.always_comb(|v| {
                v.line("next   = state;");
                v.line("x_addr = i;");
                v.line("y_addr = i;");
                v.line("y_we   = 1'b0;");
                v.line("y_din  = (x_dout < ZP) ? ZP : x_dout;");
                v.line("done   = 1'b0;");
                v.line("unique case (state)");
                v.block(|v| {
                    v.line("S_IDLE  : if (start) next = S_READ;");
                    v.line("S_READ  : next = S_WAIT;");
                    v.line("S_WAIT  : next = S_WRITE;");
                    v.line("S_WRITE : begin");
                    v.block(|v| {
                        v.line("y_we = 1'b1;");
                        v.line("next = (i == LEN - 1) ? S_DONE : S_READ;");
                    });
                    v.line("end");
                    v.line("S_DONE  : begin done = 1'b1; if (!start) next = S_IDLE; end");
                });
                v.line("endcase");
            });
        },
    );

    LayerArtifacts {
        module_name,
        instance_name,
        out_len: len,
        sv: Artifact {
            rel_path: format!("layers/{name}.sv"),
            content: v.into_string(),
        },
        mems: Vec::new(),
    }
}

pub(crate) fn bits_for(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::bits_for;
    #[test]
    fn bits_for_basic() {
        assert_eq!(bits_for(1), 1);
        assert_eq!(bits_for(2), 1);
        assert_eq!(bits_for(3), 2);
        assert_eq!(bits_for(4), 2);
        assert_eq!(bits_for(5), 3);
        assert_eq!(bits_for(1024), 10);
        assert_eq!(bits_for(5408), 13);
    }
}
