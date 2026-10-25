// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Full openXC7 flow on an exported board design, inside the openXC7 container:
//! the generated `synth.sh` runs unmodified — yosys → nextpnr-xilinx →
//! fasm2frames → xc7frames2bit — and must produce `out.bit` and meet the
//! board clock.
//!
//! Opt-in (minutes; the first run also builds the part's chipdb, cached in
//! `~/.cache/rlx-fpga/chipdb`):
//!
//! ```sh
//! docker pull regymm/openxc7
//! cargo test -p rlx-fpga --release --test openxc7_docker -- --ignored --nocapture
//! ```
//!
//! `RLX_OPENXC7_IMAGE` overrides the image.

// Emits files / goes through `rlx-ir`: both need the `std` feature.
#![cfg(feature = "std")]

use std::path::{Path, PathBuf};
use std::process::Command;

use rlx_fpga::codegen::emit_with_config;
use rlx_fpga::model::{Layer, Model};
use rlx_fpga::pack::pack;
use rlx_fpga::quant::quantize_multiplier;
use rlx_fpga::xc7::Xc7Board;
use rlx_fpga::{FpgaExportConfig, HwTarget, WeightEncoding};

fn image() -> String {
    rlx_ir::env::var("RLX_OPENXC7_IMAGE").unwrap_or_else(|| "regymm/openxc7".into())
}

fn docker_ready() -> bool {
    Command::new("docker")
        .args(["image", "inspect", &image()])
        .output()
        .is_ok_and(|o| o.status.success())
}

fn chipdb_cache() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache"));
    let d = base.join("rlx-fpga/chipdb");
    std::fs::create_dir_all(&d).unwrap();
    d
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
        name: "openxc7_flow".into(),
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

fn run_flow(dir: &Path) -> std::process::Output {
    let work = format!("{}:/work", dir.display());
    let cache = format!("{}:/chipdb", chipdb_cache().display());
    Command::new("docker")
        .args(["run", "--rm", "--platform", "linux/amd64"])
        .args([
            "-v",
            &work,
            "-v",
            &cache,
            "-e",
            "CHIPDB_DIR=/chipdb",
            "-w",
            "/work",
        ])
        .arg(image())
        .args(["bash", "./synth.sh"])
        .output()
        .expect("spawn docker")
}

#[test]
#[ignore = "needs docker + the openXC7 image; takes minutes"]
fn arty_a7_35_bitstream() {
    if !docker_ready() {
        eprintln!("openxc7_docker: `docker pull {}` first — skipping", image());
        return;
    }
    let board = Xc7Board::ArtyA7_35;
    let dir = tempfile::tempdir().unwrap();
    let cfg = FpgaExportConfig::default().with_hw_target(HwTarget::xc7_board(board));
    emit_with_config(&model(), &cfg, dir.path()).expect("emit");

    let out = run_flow(dir.path());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let pnr = std::fs::read_to_string(dir.path().join("nextpnr.log")).unwrap_or_default();
    assert!(
        out.status.success(),
        "synth.sh failed:\n{stdout}\n{stderr}\n--- nextpnr.log tail ---\n{}",
        pnr.lines().rev().take(40).collect::<Vec<_>>().join("\n")
    );

    let bit = std::fs::metadata(dir.path().join("out.bit")).expect("out.bit");
    assert!(bit.len() > 100_000, "out.bit is only {} bytes", bit.len());

    let fmax = max_mhz(&pnr).expect("no Max frequency line in nextpnr.log");
    let want = f64::from(board.pins().clk_mhz);
    eprintln!(
        "openxc7_docker: {board}: out.bit {} bytes, fmax {fmax:.1} MHz",
        bit.len()
    );
    assert!(
        fmax >= want,
        "post-route fmax {fmax:.1} MHz < board clock {want} MHz"
    );
}
