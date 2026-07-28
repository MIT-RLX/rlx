// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Data-movement / shape ops through the rlx→AIE-MLIR compiler: index-copy
// kernels (reverse / narrow / slice / tile / expand) over an [outer, axis,
// inner] view, run on the NPU via the 3-buffer NpuRun3 (arg0=in, arg1 dummy,
// arg2=out) and checked bit-exact vs a CPU reference.
//
//   AIECC=.. PEANO=.. RLX_XDNA_SHIM=.. \
//     cargo run -p rlx-xdna --features xrt --example xdna_datamove

use rlx_xdna::aie::{
    emit_cast, emit_clamp, emit_concat2, emit_expand, emit_gather, emit_narrow, emit_pad,
    emit_reverse, emit_slice, emit_tile, emit_transpose2d, emit_trilu,
};
use rlx_xdna::compile::{compile_overlay, OverlaySpec};
use rlx_xdna::npu_gemm::NpuRun3;

fn compile(name: &str, mlir: String) -> Option<Vec<u32>> {
    let (aiecc, peano) = (std::env::var("AIECC").unwrap(), std::env::var("PEANO").unwrap());
    let tmp = format!("/tmp/rlx_dm_{name}");
    std::fs::create_dir_all(&tmp).ok();
    let mp = format!("{tmp}/aie.mlir");
    std::fs::write(&mp, &mlir).unwrap();
    let xclbin = format!("{tmp}/k.xclbin");
    let insts_path = format!("{tmp}/insts.bin");
    if let Err(e) = compile_overlay(&OverlaySpec {
        aiecc: &aiecc,
        peano: &peano,
        mlir: &mp,
        tmpdir: &format!("{tmp}/build"),
        out_xclbin: &xclbin,
        out_insts: &insts_path,
    }) {
        println!("  {name:<9} COMPILE-FAIL ({})", format!("{e:?}").lines().next().unwrap_or(""));
        return None;
    }
    Some(
        std::fs::read(&insts_path)
            .unwrap()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn xclbin_path(name: &str) -> String {
    format!("/tmp/rlx_dm_{name}/k.xclbin")
}

// 1-input (arg1 dummy).
fn run(name: &str, mlir: String, input: &[f32], cref: &[f32]) -> bool {
    let Some(insts) = compile(name, mlir) else { return false };
    let io = NpuRun3::open("", &xclbin_path(name), &insts, input.len(), 1, cref.len()).expect("open");
    let out = io.run(input, &[0.0]).expect("run");
    check(name, &out, cref, input.len())
}

// 2-input (concat / gather).
fn run2(name: &str, mlir: String, a: &[f32], b: &[f32], cref: &[f32]) -> bool {
    let Some(insts) = compile(name, mlir) else { return false };
    let io = NpuRun3::open("", &xclbin_path(name), &insts, a.len(), b.len(), cref.len()).expect("open");
    let out = io.run(a, b).expect("run");
    check(name, &out, cref, a.len())
}

fn check(name: &str, out: &[f32], cref: &[f32], nin: usize) -> bool {
    let mism = (0..cref.len()).filter(|&i| out[i] != cref[i]).count();
    if mism != 0 {
        println!("  {name:<9} FAIL ✗  {mism} mismatches");
        false
    } else {
        println!("  {name:<9} PASS ✓  ({nin} → {} elems)", cref.len());
        true
    }
}

fn main() {
    let mut all = true;

    // reverse [2, 4, 3] on axis 1
    let (o, ax, inr) = (2, 4, 3);
    let inp: Vec<f32> = (0..o * ax * inr).map(|i| i as f32).collect();
    let mut cref = vec![0f32; o * ax * inr];
    for oo in 0..o {
        for a in 0..ax {
            for i in 0..inr {
                cref[(oo * ax + a) * inr + i] = inp[(oo * ax + (ax - 1 - a)) * inr + i];
            }
        }
    }
    all &= run("reverse", emit_reverse(o, ax, inr), &inp, &cref);

    // narrow [2, 6, 3] axis1 start=1 len=3
    let (o, ax, inr, start, len) = (2, 6, 3, 1, 3);
    let inp: Vec<f32> = (0..o * ax * inr).map(|i| i as f32).collect();
    let mut cref = vec![0f32; o * len * inr];
    for oo in 0..o {
        for a in 0..len {
            for i in 0..inr {
                cref[(oo * len + a) * inr + i] = inp[(oo * ax + (start + a)) * inr + i];
            }
        }
    }
    all &= run("narrow", emit_narrow(o, ax, inr, start, len), &inp, &cref);

    // slice [1, 8, 1] start=0 len=4 step=2
    let (o, ax, inr, start, len, step) = (1usize, 8usize, 1usize, 0usize, 4usize, 2i64);
    let inp: Vec<f32> = (0..o * ax * inr).map(|i| i as f32).collect();
    let mut cref = vec![0f32; o * len * inr];
    for a in 0..len {
        cref[a] = inp[(start as i64 + a as i64 * step) as usize];
    }
    all &= run("slice", emit_slice(o, ax, inr, start, len, step), &inp, &cref);

    // tile [2, 3, 2] reps=3 on axis
    let (o, ax, inr, reps) = (2, 3, 2, 3);
    let inp: Vec<f32> = (0..o * ax * inr).map(|i| i as f32).collect();
    let mut cref = vec![0f32; o * ax * reps * inr];
    for oo in 0..o {
        for a in 0..ax * reps {
            for i in 0..inr {
                cref[(oo * ax * reps + a) * inr + i] = inp[(oo * ax + (a % ax)) * inr + i];
            }
        }
    }
    all &= run("tile", emit_tile(o, ax, inr, reps), &inp, &cref);

    // expand [2, 1, 3] → axis 4
    let (o, inr, oax) = (2, 3, 4);
    let inp: Vec<f32> = (0..o * 1 * inr).map(|i| i as f32).collect();
    let mut cref = vec![0f32; o * oax * inr];
    for oo in 0..o {
        for a in 0..oax {
            for i in 0..inr {
                cref[(oo * oax + a) * inr + i] = inp[oo * inr + i];
            }
        }
    }
    all &= run("expand", emit_expand(o, inr, oax), &inp, &cref);

    // transpose [3, 4] → [4, 3]
    let (r, c) = (3usize, 4usize);
    let inp: Vec<f32> = (0..r * c).map(|i| i as f32).collect();
    let mut cref = vec![0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            cref[j * r + i] = inp[i * c + j];
        }
    }
    all &= run("transpose", emit_transpose2d(r, c), &inp, &cref);

    // trilu [4, 4] upper, diagonal 0
    let m = 4usize;
    let inp: Vec<f32> = (0..m * m).map(|i| (i + 1) as f32).collect();
    let mut cref = vec![0f32; m * m];
    for i in 0..m {
        for j in 0..m {
            cref[i * m + j] = if (j as i64) >= i as i64 { inp[i * m + j] } else { 0.0 };
        }
    }
    all &= run("trilu", emit_trilu(m, m, true, 0), &inp, &cref);

    // clamp to [-1, 1]
    let inp: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let cref: Vec<f32> = inp.iter().map(|&x| x.clamp(-1.0, 1.0)).collect();
    all &= run("clamp", emit_clamp(inp.len(), -1.0, 1.0), &inp, &cref);

    // concat [2, 2, 3] ‖ [2, 3, 3] on axis 1 → [2, 5, 3]
    let (o, aax, bax, inr) = (2, 2, 3, 3);
    let a: Vec<f32> = (0..o * aax * inr).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..o * bax * inr).map(|i| 100.0 + i as f32).collect();
    let oax = aax + bax;
    let mut cref = vec![0f32; o * oax * inr];
    for oo in 0..o {
        for x in 0..aax {
            for i in 0..inr {
                cref[(oo * oax + x) * inr + i] = a[(oo * aax + x) * inr + i];
            }
        }
        for x in 0..bax {
            for i in 0..inr {
                cref[(oo * oax + aax + x) * inr + i] = b[(oo * bax + x) * inr + i];
            }
        }
    }
    all &= run2("concat", emit_concat2(o, aax, bax, inr), &a, &b, &cref);

    // pad [2, 3, 2] axis before=1 after=2 fill=0
    let (o, ax, inr, bef, aft) = (2, 3, 2, 1, 2);
    let oax = bef + ax + aft;
    let inp: Vec<f32> = (0..o * ax * inr).map(|i| (i + 1) as f32).collect();
    let mut cref = vec![0f32; o * oax * inr];
    for oo in 0..o {
        for a in 0..ax {
            for i in 0..inr {
                cref[(oo * oax + bef + a) * inr + i] = inp[(oo * ax + a) * inr + i];
            }
        }
    }
    all &= run("pad", emit_pad(o, ax, inr, bef, aft, 0.0), &inp, &cref);

    // gather data [2, 5, 3] with idx [4, 0, 2]
    let (o, ax, inr) = (2, 5, 3);
    let idx = [4.0f32, 0.0, 2.0];
    let data: Vec<f32> = (0..o * ax * inr).map(|i| i as f32).collect();
    let mut cref = vec![0f32; o * idx.len() * inr];
    for oo in 0..o {
        for (j, &ix) in idx.iter().enumerate() {
            for i in 0..inr {
                cref[(oo * idx.len() + j) * inr + i] = data[(oo * ax + ix as usize) * inr + i];
            }
        }
    }
    all &= run2("gather", emit_gather(o, ax, inr, idx.len()), &data, &idx, &cref);

    // cast f32 → i32 (output bits reinterpreted as f32 by the runner)
    let inp: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 1.5).collect();
    if let Some(insts) = compile("cast", emit_cast(inp.len(), true)) {
        let io = NpuRun3::open("", &xclbin_path("cast"), &insts, inp.len(), 1, inp.len()).unwrap();
        let out = io.run(&inp, &[0.0]).unwrap();
        let ok = (0..inp.len()).all(|i| (out[i].to_bits() as i32) == inp[i] as i32);
        println!("  {:<9} {}", "cast", if ok { "PASS ✓  (f32→i32)" } else { "FAIL ✗" });
        all &= ok;
    } else {
        all = false;
    }

    println!("\n{}", if all { "all data-movement ops match CPU ✓" } else { "MISMATCH" });
    if !all {
        std::process::exit(1);
    }
}
