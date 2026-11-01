// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Full Lattice flows on exported board designs, in the open-toolchain
//! containers: the generated `synth.sh` runs unmodified and must produce a
//! bitstream that meets the board clock.
//!
//! * iCEBreaker (iCE40UP5K): yosys `synth_ice40` → `nextpnr-ice40` →
//!   `icepack` → `out.bin`, in `hdlc/impl:icestorm`.
//! * ULX3S-85F (ECP5): yosys `synth_ecp5` → `nextpnr-ecp5` → `ecppack` →
//!   `out.bit`, in `hdlc/impl:prjtrellis`.
//!
//! Opt-in (a minute or two each under amd64 emulation):
//!
//! ```sh
//! docker pull hdlc/impl:icestorm && docker pull hdlc/impl:prjtrellis
//! cargo test -p rlx-fpga --release --test lattice_docker -- --ignored --nocapture
//! ```
//!
//! `RLX_ICESTORM_IMAGE` / `RLX_TRELLIS_IMAGE` override the images.

// Emits files / goes through `rlx-ir`: both need the `std` feature.
#![cfg(feature = "std")]

use std::path::Path;
use std::process::Command;

use rlx_fpga::codegen::emit_with_config;
use rlx_fpga::lattice::{LatticeBoard, LatticeFamily};
use rlx_fpga::model::{Layer, Model};
use rlx_fpga::pack::pack;
use rlx_fpga::quant::quantize_multiplier;
use rlx_fpga::{FpgaExportConfig, HwTarget, WeightEncoding};

fn image(family: LatticeFamily) -> String {
    match family {
        LatticeFamily::Ice40 => {
            rlx_ir::env::var("RLX_ICESTORM_IMAGE").unwrap_or_else(|| "hdlc/impl:icestorm".into())
        }
        LatticeFamily::Ecp5 => {
            rlx_ir::env::var("RLX_TRELLIS_IMAGE").unwrap_or_else(|| "hdlc/impl:prjtrellis".into())
        }
    }
}

fn docker_ready(image: &str) -> bool {
    Command::new("docker")
        .args(["image", "inspect", image])
        .output()
        .is_ok_and(|o| o.status.success())
}

struct Lcg(u32);

impl Lcg {
    fn sym(&mut self, m: i32) -> i32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) % (2 * m as u32 + 1)) as i32 - m
    }
}

fn ternary(rng: &mut Lcg, n: usize) -> Vec<i8> {
    pack(&(0..n).map(|_| rng.sym(1) as i8).collect::<Vec<_>>(), 2)
}

/// Ternary 6×6×2 → Conv3×3 (pad 1, 8 ch) → ReLU → MaxPool → Dense 72→32
/// → ReLU → Dense 32→4 → Argmax.
fn model() -> Model {
    let mut rng = Lcg(0x0c7);
    let m = |fan_in: usize| quantize_multiplier(40.0 / ((fan_in as f32).sqrt() * 58.0 * 0.82));
    let dense = |rng: &mut Lcg, name, i: usize, o: usize| Layer::Dense {
        name,
        in_features: i,
        out_features: o,
        x_zp: 0,
        w_zp: 0,
        out_zp: 0,
        weight_bits: 2,
        weight_encoding: WeightEncoding::SignedInt,
        requant: vec![m(i); o],
        weights: ternary(rng, i * o),
        bias: Some((0..o).map(|_| rng.sym(300)).collect()),
    };
    let conv = Layer::Conv2d {
        name: "conv1",
        h_in: 6,
        w_in: 6,
        c_in: 2,
        c_out: 8,
        kh: 3,
        kw: 3,
        pad_h: 1,
        pad_w: 1,
        stride_h: 1,
        stride_w: 1,
        x_zp: 0,
        w_zp: 0,
        out_zp: 0,
        weight_bits: 2,
        weight_encoding: WeightEncoding::SignedInt,
        requant: vec![m(18); 8],
        weights: ternary(&mut rng, 8 * 9 * 2),
        bias: Some((0..8).map(|_| rng.sym(300)).collect()),
    };
    Model {
        name: "lattice_flow".into(),
        input_len: 6 * 6 * 2,
        layers: vec![
            conv,
            Layer::Relu {
                name: "relu1",
                len: 6 * 6 * 8,
                zero_point: 0,
            },
            Layer::MaxPool2d {
                name: "pool1",
                h_in: 6,
                w_in: 6,
                c: 8,
                kh: 2,
                kw: 2,
                stride_h: 2,
                stride_w: 2,
            },
            dense(&mut rng, "fc1", 72, 32),
            Layer::Relu {
                name: "relu2",
                len: 32,
                zero_point: 0,
            },
            dense(&mut rng, "fc2", 32, 4),
            Layer::Argmax {
                name: "argmax",
                len: 4,
            },
        ],
        extra_outputs: vec![],
    }
}

/// `Max frequency for clock '…': X MHz` — the last report is post-route.
fn max_mhz(log: &str) -> Option<f64> {
    log.lines()
        .filter(|l| l.contains("Max frequency for clock"))
        .filter_map(|l| {
            let after = l.split("':").nth(1)?;
            after.split_whitespace().next()?.parse().ok()
        })
        .next_back()
}

fn run_flow(dir: &Path, image: &str) -> std::process::Output {
    let work = format!("{}:/work", dir.display());
    Command::new("docker")
        .args(["run", "--rm", "--platform", "linux/amd64"])
        .args(["-v", &work, "-w", "/work"])
        .arg(image)
        .args(["bash", "./synth.sh"])
        .output()
        .expect("spawn docker")
}

fn bitstream(board: LatticeBoard, file: &str, min_bytes: u64) {
    let image = image(board.family());
    if !docker_ready(&image) {
        eprintln!("lattice_docker: `docker pull {image}` first — skipping");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let cfg = FpgaExportConfig::default().with_hw_target(HwTarget::lattice_board(board));
    emit_with_config(&model(), &cfg, dir.path()).expect("emit");

    let out = run_flow(dir.path(), &image);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let pnr = std::fs::read_to_string(dir.path().join("nextpnr.log")).unwrap_or_default();
    assert!(
        out.status.success(),
        "{board}: synth.sh failed:\n{stdout}\n{stderr}\n--- nextpnr.log tail ---\n{}",
        pnr.lines().rev().take(40).collect::<Vec<_>>().join("\n")
    );
    let bits = std::fs::metadata(dir.path().join(file)).unwrap_or_else(|_| panic!("{file}"));
    assert!(
        bits.len() >= min_bytes,
        "{board}: {file} is only {} bytes",
        bits.len()
    );
    let fmax = max_mhz(&pnr).expect("no Max frequency line in nextpnr.log");
    // The design clock: the oscillator, or the PLL output (ULX3S).
    let want = f64::from(board.shell().core_khz) / 1000.0;
    let util: Vec<&str> = pnr
        .lines()
        .filter(|l| {
            l.contains("%")
                && (l.contains("LC")
                    || l.contains("SLICE")
                    || l.contains("RAM")
                    || l.contains("MAC")
                    || l.contains("MULT"))
        })
        .collect();
    eprintln!(
        "lattice_docker: {board}: {file} {} bytes, fmax {fmax:.1} MHz (design clock {want} MHz)\n{}",
        bits.len(),
        util.join("\n")
    );
    assert!(
        fmax >= want,
        "{board}: post-route fmax {fmax:.1} MHz < board clock {want} MHz"
    );
}

#[test]
#[ignore = "needs docker + hdlc/impl:icestorm; takes a minute"]
fn icebreaker_bitstream() {
    // An iCE40UP5K image is 104 KB.
    bitstream(LatticeBoard::Icebreaker, "out.bin", 100_000);
}

#[test]
#[ignore = "needs docker + hdlc/impl:prjtrellis; takes minutes"]
fn ulx3s_85f_bitstream() {
    bitstream(LatticeBoard::Ulx3s85, "out.bit", 100_000);
}
