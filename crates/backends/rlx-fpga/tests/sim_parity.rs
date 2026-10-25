// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Emitted SystemVerilog, *simulated*, vs `reference::run_with_precision`.
//!
//! Every other test here inspects the RTL text. This one runs it under
//! Icarus Verilog and compares the logits (every element) and the argmax of
//! each random input bit-exactly against the Rust reference — 2/4/8-bit
//! weights × every tune preset, dense and conv pipelines.
//!
//! Skipped (passes vacuously, with a note on stderr) when `iverilog` is not
//! on `PATH`. Install: `brew install icarus-verilog` / `apt install iverilog`.
//!
//!   cargo test -p rlx-fpga --release --test sim_parity

// Emits files / goes through `rlx-ir`: both need the `std` feature.
#![cfg(feature = "std")]

use std::path::Path;
use std::process::Command;

use rlx_fpga::codegen::emit_model_tuned;
use rlx_fpga::model::{Layer, Model};
use rlx_fpga::pack::pack;
use rlx_fpga::quant::quantize_multiplier;
use rlx_fpga::reference::run_with_precision;
use rlx_fpga::{OptTarget, Tune, WeightEncoding, tinyconv_mnist_from_cortexm};

const INPUTS: usize = 4;

fn have_iverilog() -> bool {
    Command::new("iverilog").arg("-V").output().is_ok()
}

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 >> 8
    }
    /// Uniform integer in `[-m, m]`.
    fn sym(&mut self, m: i32) -> i32 {
        (self.next() % (2 * m as u32 + 1)) as i32 - m
    }
}

fn weights(rng: &mut Lcg, n: usize, bits: u8) -> (Vec<i8>, f32) {
    let m = match bits {
        2 => 1,
        4 => 7,
        _ => 127,
    };
    let logical: Vec<i8> = (0..n).map(|_| rng.sym(m) as i8).collect();
    // σ of uniform {-m..m}.
    let sigma = ((m * (m + 1)) as f32 / 3.0).sqrt();
    (pack(&logical, bits), sigma)
}

/// Per-row requant aiming the outputs at σ ≈ 40 for inputs uniform in ±100.
fn requant(rng: &mut Lcg, rows: usize, fan_in: usize, w_sigma: f32) -> Vec<(i32, i32)> {
    let base = 40.0 / ((fan_in as f32).sqrt() * 58.0 * w_sigma);
    (0..rows)
        .map(|_| quantize_multiplier(base * (0.75 + (rng.next() % 512) as f32 / 1024.0)))
        .collect()
}

fn bias(rng: &mut Lcg, rows: usize) -> Option<Vec<i32>> {
    Some((0..rows).map(|_| rng.sym(500)).collect())
}

fn dense(rng: &mut Lcg, name: &'static str, in_f: usize, out_f: usize, bits: u8) -> Layer {
    let (w, s) = weights(rng, in_f * out_f, bits);
    Layer::Dense {
        name,
        in_features: in_f,
        out_features: out_f,
        x_zp: 0,
        w_zp: 0,
        out_zp: 0,
        weight_bits: bits,
        weight_encoding: WeightEncoding::SignedInt,
        requant: requant(rng, out_f, in_f, s),
        weights: w,
        bias: bias(rng, out_f),
    }
}

/// 16 → 12 → ReLU → 4 → Argmax.
fn mlp(bits: u8) -> Model {
    let mut rng = Lcg(0x5eed + u32::from(bits));
    Model {
        name: format!("sim_mlp_w{bits}"),
        input_len: 16,
        layers: vec![
            dense(&mut rng, "fc1", 16, 12, bits),
            Layer::Relu {
                name: "relu1",
                len: 12,
                zero_point: 0,
            },
            dense(&mut rng, "fc2", 12, 4, bits),
            Layer::Argmax {
                name: "argmax",
                len: 4,
            },
        ],
        extra_outputs: vec![],
    }
}

/// 6×6×2 → Conv3×3(4 ch, `pad`, `stride`) → ReLU → MaxPool2×2 → Dense → Argmax.
/// `c_out = 4` lets the Latency preset take the parallel conv kernel.
fn cnn_with(bits: u8, pad: usize, stride: usize) -> Model {
    let mut rng = Lcg(0xc0de + u32::from(bits));
    let (c_in, c_out, k, hw) = (2, 4, 3, 6);
    let ho = (hw + 2 * pad - k) / stride + 1;
    let hp = ho / 2;
    let (w, s) = weights(&mut rng, c_out * k * k * c_in, bits);
    let conv = Layer::Conv2d {
        name: "conv1",
        h_in: hw,
        w_in: hw,
        c_in,
        c_out,
        kh: k,
        kw: k,
        pad_h: pad,
        pad_w: pad,
        stride_h: stride,
        stride_w: stride,
        x_zp: 0,
        w_zp: 0,
        out_zp: 0,
        weight_bits: bits,
        weight_encoding: WeightEncoding::SignedInt,
        requant: requant(&mut rng, c_out, k * k * c_in, s),
        weights: w,
        bias: bias(&mut rng, c_out),
    };
    Model {
        name: format!("sim_cnn_w{bits}_p{pad}_s{stride}"),
        input_len: hw * hw * c_in,
        layers: vec![
            conv,
            Layer::Relu {
                name: "relu1",
                len: ho * ho * c_out,
                zero_point: 0,
            },
            Layer::MaxPool2d {
                name: "pool1",
                h_in: ho,
                w_in: ho,
                c: c_out,
                kh: 2,
                kw: 2,
                stride_h: 2,
                stride_w: 2,
            },
            dense(&mut rng, "fc1", hp * hp * c_out, 4, bits),
            Layer::Argmax {
                name: "argmax",
                len: 4,
            },
        ],
        extra_outputs: vec![],
    }
}

/// Same-padded conv (the common CNN case).
fn cnn(bits: u8) -> Model {
    cnn_with(bits, 1, 1)
}

/// Padding with stride 2: border taps land on both edges unevenly.
fn cnn_strided(bits: u8) -> Model {
    cnn_with(bits, 1, 2)
}

/// Wrap the emitted `tb.sv` with a monitor printing every write of the
/// layer that feeds the trailing Argmax (`LOGIT <addr> <value>`).
fn instrument_tb(dir: &Path) {
    let top = std::fs::read_to_string(dir.join("top.sv")).unwrap();
    let n_rtl = (0..)
        .take_while(|i| top.contains(&format!("logic l{i}_start")))
        .count();
    assert!(n_rtl >= 2, "top.sv: expected ≥2 layer instances");
    let l = n_rtl - 2;
    let tb = std::fs::read_to_string(dir.join("tb.sv")).unwrap();
    let monitor = format!(
        "always @(posedge clk) if (u_top.l{l}_y_we) \
         $display(\"LOGIT %0d %0d\", u_top.l{l}_y_addr, u_top.l{l}_y_din);\n    top u_top ("
    );
    assert!(
        tb.contains("top u_top ("),
        "tb.sv: no `top u_top (` instance"
    );
    std::fs::write(dir.join("tb.sv"), tb.replacen("top u_top (", &monitor, 1)).unwrap();
}

fn compile(dir: &Path) {
    let mut sources = vec!["tb.sv".to_string(), "top.sv".to_string()];
    for sub in ["layers", "primitives"] {
        let mut v: Vec<String> = std::fs::read_dir(dir.join(sub))
            .unwrap()
            .map(|e| format!("{sub}/{}", e.unwrap().file_name().to_string_lossy()))
            .filter(|p| p.ends_with(".sv"))
            .collect();
        v.sort();
        sources.extend(v);
    }
    let out = Command::new("iverilog")
        .current_dir(dir)
        .args(["-g2012", "-o", "sim"])
        .args(&sources)
        .output()
        .expect("spawn iverilog");
    assert!(
        out.status.success(),
        "iverilog rejected the emitted RTL in {}:\n{}",
        dir.display(),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Run one stimulus; returns `(logits by address, pred)`.
fn simulate(dir: &Path, input: &[i8], n_logits: usize) -> (Vec<i8>, i64) {
    let mem: String = input
        .iter()
        .map(|&b| format!("{:02x}\n", b as u8))
        .collect();
    std::fs::write(dir.join("tb_image.mem"), mem).unwrap();
    let out = Command::new("vvp")
        .current_dir(dir)
        .args(["-n", "sim"])
        .output()
        .expect("spawn vvp");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut logits = vec![None; n_logits];
    let mut pred = None;
    for line in stdout.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            ["LOGIT", a, v] => {
                logits[a.parse::<usize>().unwrap()] = Some(v.parse::<i64>().unwrap() as i8);
            }
            ["pred", "=", p] => pred = p.parse::<i64>().ok(),
            _ => {}
        }
    }
    let logits = logits
        .into_iter()
        .enumerate()
        .map(|(i, v)| v.unwrap_or_else(|| panic!("logit {i} never written:\n{stdout}")))
        .collect();
    (
        logits,
        pred.unwrap_or_else(|| panic!("no `pred =` line:\n{stdout}")),
    )
}

fn check(model: &Model, target: OptTarget, ternary_fast_path: bool) {
    let mut tune = Tune::for_target(target);
    tune.ternary_fast_path = ternary_fast_path;
    let dir = tempfile::tempdir().unwrap();
    emit_model_tuned(model, &tune, dir.path()).unwrap();
    instrument_tb(dir.path());
    compile(dir.path());

    let n_ref = model.layers.len();
    let n_logits = match &model.layers[n_ref - 1] {
        Layer::Argmax { len, .. } => *len,
        other => panic!("model must end in Argmax, got {other:?}"),
    };
    let mut rng = Lcg(0xfeed);
    for k in 0..INPUTS {
        let input: Vec<i8> = (0..model.input_len).map(|_| rng.sym(100) as i8).collect();
        let (want_pred, ints) = run_with_precision(model, &input, tune.requant_precision);
        let (logits, pred) = simulate(dir.path(), &input, n_logits);
        assert_eq!(
            logits,
            ints[n_ref - 2],
            "{} {target:?} fast={ternary_fast_path} input {k}: RTL logits != reference",
            model.name
        );
        assert_eq!(
            pred, want_pred as i64,
            "{} {target:?} fast={ternary_fast_path} input {k}: RTL pred != reference",
            model.name
        );
    }
}

const TARGETS: [OptTarget; 4] = [
    OptTarget::Precision,
    OptTarget::Latency,
    OptTarget::Size,
    OptTarget::Energy,
];

fn sweep(build: fn(u8) -> Model) {
    if !have_iverilog() {
        eprintln!("sim_parity: iverilog not on PATH — skipping");
        return;
    }
    for bits in [2u8, 4, 8] {
        let model = build(bits);
        for target in TARGETS {
            let preset = Tune::for_target(target).ternary_fast_path;
            check(&model, target, preset && bits == 2);
        }
        if bits == 2 {
            // Both MAC flavours for ternary weights, whatever the preset says.
            check(&model, OptTarget::Precision, true);
            check(&model, OptTarget::Energy, false);
        }
    }
}

#[test]
fn mlp_rtl_matches_reference() {
    sweep(mlp);
}

#[test]
fn cnn_rtl_matches_reference() {
    sweep(cnn);
}

#[test]
fn cnn_strided_padded_rtl_matches_reference() {
    sweep(cnn_strided);
}

/// The production TinyConv-MNIST weights (the checked-in `examples/mnist_sv`
/// demo), default and Energy tunes.
#[test]
fn tinyconv_mnist_rtl_matches_reference() {
    if !have_iverilog() {
        eprintln!("sim_parity: iverilog not on PATH — skipping");
        return;
    }
    let model = tinyconv_mnist_from_cortexm();
    check(&model, OptTarget::Precision, false);
    check(&model, OptTarget::Energy, false);
}
