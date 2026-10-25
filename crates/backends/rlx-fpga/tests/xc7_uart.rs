// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! openXC7 board export (`HwTarget::xc7_board`) driven over its UART host
//! link in simulation — the same bytes `run_uart.py` sends to a real board.
//!
//! * RTL: `board_top.sv` + the model under Icarus Verilog.
//! * Netlist: the same design after `yosys synth_xilinx -family xc7` (the
//!   first stage of the openXC7 flow), simulated against Yosys's Xilinx cell
//!   models — proves the weights survive synthesis and the mapped design
//!   still answers bit-exactly.
//!
//! Replies are compared with `reference::run`. Skipped when `iverilog`
//! (or, for the netlist case, `yosys`) is not on `PATH`.
//!
//!   cargo test -p rlx-fpga --release --test xc7_uart

// Emits files / goes through `rlx-ir`: both need the `std` feature.
#![cfg(feature = "std")]

use std::path::Path;
use std::process::Command;

use rlx_fpga::codegen::emit_with_config;
use rlx_fpga::model::{Layer, Model};
use rlx_fpga::pack::pack;
use rlx_fpga::quant::quantize_multiplier;
use rlx_fpga::reference::run;
use rlx_fpga::xc7::{UART_SYNC, Xc7Board};
use rlx_fpga::{FpgaExportConfig, HwTarget, OutputIface, WeightEncoding};

const CLKS_PER_BIT: u32 = 8;
const INPUTS: usize = 3;

fn have(tool: &str, arg: &str) -> bool {
    Command::new(tool).arg(arg).output().is_ok()
}

struct Lcg(u32);

impl Lcg {
    fn sym(&mut self, m: i32) -> i32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) % (2 * m as u32 + 1)) as i32 - m
    }
}

fn ternary_dense(rng: &mut Lcg, name: &'static str, in_f: usize, out_f: usize) -> Layer {
    let codes: Vec<i8> = (0..in_f * out_f).map(|_| rng.sym(1) as i8).collect();
    let m = 40.0 / ((in_f as f32).sqrt() * 58.0 * 0.82);
    Layer::Dense {
        name,
        in_features: in_f,
        out_features: out_f,
        x_zp: 0,
        w_zp: 0,
        out_zp: 0,
        weight_bits: 2,
        weight_encoding: WeightEncoding::SignedInt,
        requant: (0..out_f).map(|_| quantize_multiplier(m)).collect(),
        weights: pack(&codes, 2),
        bias: Some((0..out_f).map(|_| rng.sym(300)).collect()),
    }
}

/// Ternary 16 → 12 → ReLU → 4, optionally ending in Argmax.
fn model(argmax: bool) -> Model {
    let mut rng = Lcg(0xb17);
    let mut layers = vec![
        ternary_dense(&mut rng, "fc1", 16, 12),
        Layer::Relu {
            name: "relu1",
            len: 12,
            zero_point: 0,
        },
        ternary_dense(&mut rng, "fc2", 12, 4),
    ];
    if argmax {
        layers.push(Layer::Argmax {
            name: "argmax",
            len: 4,
        });
    }
    Model {
        name: "xc7_uart".into(),
        input_len: 16,
        layers,
        extra_outputs: vec![],
    }
}

/// Testbench: drive `uart_rx` with SYNC + input bytes, decode `uart_tx`.
fn testbench(inputs: &[Vec<i8>], reply_len: usize, rtl: bool) -> String {
    let mut stim = String::new();
    for x in inputs {
        stim.push_str(&format!("        send(8'h{UART_SYNC:02X});\n"));
        for &b in x {
            stim.push_str(&format!("        send(8'h{:02X});\n", b as u8));
        }
        for _ in 0..reply_len {
            stim.push_str("        recv();\n");
        }
    }
    let defparam = if rtl {
        format!("    defparam dut.CLKS_PER_BIT = {CLKS_PER_BIT};\n")
    } else {
        String::new()
    };
    format!(
        r#"`timescale 1ns/1ps
module tb;
    localparam int BIT = {CLKS_PER_BIT};
    logic clk = 0;
    always #5 clk = ~clk;
    logic rx = 1'b1;
    logic tx;
    logic [3:0] led;
    board_top dut (.clk(clk), .rst_btn(1'b1), .uart_rx(rx), .uart_tx(tx), .led(led));
{defparam}
    task automatic send(input logic [7:0] b);
        rx = 1'b0; repeat (BIT) @(posedge clk);
        for (int i = 0; i < 8; i++) begin rx = b[i]; repeat (BIT) @(posedge clk); end
        rx = 1'b1; repeat (BIT) @(posedge clk);
    endtask

    task automatic recv();
        logic [7:0] b;
        @(negedge tx);
        repeat (BIT + BIT / 2) @(posedge clk);
        for (int i = 0; i < 8; i++) begin b[i] = tx; repeat (BIT) @(posedge clk); end
        $display("RX %0d", $signed(b));
    endtask

    initial begin
        repeat (40) @(posedge clk);  // power-on reset
{stim}        $finish;
    end
    initial begin
        #50000000 $display("TIMEOUT"); $finish;
    end
endmodule
"#
    )
}

fn sv_sources(dir: &Path) -> Vec<String> {
    let mut v = vec!["board_top.sv".to_string(), "top.sv".to_string()];
    for sub in ["layers", "primitives"] {
        let mut s: Vec<String> = std::fs::read_dir(dir.join(sub))
            .unwrap()
            .map(|e| format!("{sub}/{}", e.unwrap().file_name().to_string_lossy()))
            .filter(|p| p.ends_with(".sv"))
            .collect();
        s.sort();
        v.extend(s);
    }
    v
}

fn run_sim(dir: &Path, sources: &[String]) -> Vec<i8> {
    let out = Command::new("iverilog")
        .current_dir(dir)
        .args(["-g2012", "-o", "sim"])
        .args(sources)
        .output()
        .expect("spawn iverilog");
    assert!(
        out.status.success(),
        "iverilog failed in {}:\n{}",
        dir.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new("vvp")
        .current_dir(dir)
        .args(["-n", "sim"])
        .output()
        .expect("spawn vvp");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("TIMEOUT"),
        "board did not answer:\n{stdout}"
    );
    stdout
        .lines()
        .filter_map(|l| l.strip_prefix("RX "))
        .map(|v| v.trim().parse::<i64>().unwrap() as i8)
        .collect()
}

/// Expected reply bytes for `inputs`.
fn expected(model: &Model, inputs: &[Vec<i8>], pred: bool, logits: bool) -> Vec<i8> {
    let mut want = Vec::new();
    for x in inputs {
        let (p, ints) = run(model, x);
        let last = ints.last().unwrap();
        if pred {
            // With an Argmax tail `pred` is the class; otherwise top peeks addr 0.
            want.push(
                if model
                    .layers
                    .iter()
                    .any(|l| matches!(l, Layer::Argmax { .. }))
                {
                    p as i8
                } else {
                    last[0]
                },
            );
        }
        if logits {
            want.extend_from_slice(last);
        }
    }
    want
}

fn export(model: &Model, output: OutputIface, dir: &Path) {
    let mut cfg =
        FpgaExportConfig::default().with_hw_target(HwTarget::xc7_board(Xc7Board::ArtyA7_35));
    cfg.io.output = output;
    emit_with_config(model, &cfg, dir).expect("emit");
    for f in [
        "board_top.sv",
        "constraints.xdc",
        "synth.sh",
        "program.sh",
        "run_uart.py",
    ] {
        assert!(dir.join(f).exists(), "missing {f}");
    }
}

fn inputs(model: &Model) -> Vec<Vec<i8>> {
    let mut rng = Lcg(0x1234);
    (0..INPUTS)
        .map(|_| (0..model.input_len).map(|_| rng.sym(100) as i8).collect())
        .collect()
}

fn rtl_case(argmax: bool, output: OutputIface) {
    if !have("iverilog", "-V") {
        eprintln!("xc7_uart: iverilog not on PATH — skipping");
        return;
    }
    let m = model(argmax);
    let dir = tempfile::tempdir().unwrap();
    export(&m, output, dir.path());
    let xs = inputs(&m);
    let (pred, logits) = (output.wants_pred_port(), output.wants_memory());
    let want = expected(&m, &xs, pred, logits);
    let reply_len = want.len() / xs.len();
    std::fs::write(
        dir.path().join("tb_uart.sv"),
        testbench(&xs, reply_len, true),
    )
    .unwrap();
    let mut srcs = vec!["tb_uart.sv".to_string()];
    srcs.extend(sv_sources(dir.path()));
    assert_eq!(
        run_sim(dir.path(), &srcs),
        want,
        "{output:?}: UART reply != reference"
    );
}

#[test]
fn uart_pred_matches_reference() {
    rtl_case(true, OutputIface::ScalarPred);
}

#[test]
fn uart_logits_match_reference() {
    rtl_case(false, OutputIface::ScalarAndMemory);
}

#[test]
fn xc7_netlist_matches_reference() {
    if !have("iverilog", "-V") || !have("yosys", "-V") {
        eprintln!("xc7_uart: iverilog/yosys not on PATH — skipping");
        return;
    }
    let m = model(true);
    let dir = tempfile::tempdir().unwrap();
    export(&m, OutputIface::ScalarPred, dir.path());
    let script = format!(
        "read_verilog -sv {}; chparam -set CLKS_PER_BIT {CLKS_PER_BIT} board_top; \
         synth_xilinx -flatten -abc9 -family xc7 -top board_top; write_verilog -noattr synth.v",
        sv_sources(dir.path()).join(" ")
    );
    let out = Command::new("yosys")
        .current_dir(dir.path())
        .args(["-q", "-p", &script])
        .output()
        .expect("spawn yosys");
    assert!(
        out.status.success(),
        "yosys synth_xilinx failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let datdir = Command::new("yosys-config")
        .arg("--datdir")
        .output()
        .expect("yosys-config");
    let cells = format!(
        "{}/xilinx/cells_sim.v",
        String::from_utf8_lossy(&datdir.stdout).trim()
    );

    let xs = inputs(&m);
    let want = expected(&m, &xs, true, false);
    std::fs::write(dir.path().join("tb_uart.sv"), testbench(&xs, 1, false)).unwrap();
    let srcs = vec!["tb_uart.sv".to_string(), "synth.v".to_string(), cells];
    assert_eq!(
        run_sim(dir.path(), &srcs),
        want,
        "synthesized netlist != reference"
    );
}
