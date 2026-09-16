// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Build an ahead-of-time kernel pack. Runs on a host that has a GPU compiler;
//! the resulting pack is read anywhere by `egpu_inspect`.
//!
//! ```sh
//! # from source (needs hipcc / ptxas)
//! cargo run -p rlx-egpu --example egpu_bake -- -o k.rlxisa --hip k.hip:gfx1100,gfx1201
//! cargo run -p rlx-egpu --example egpu_bake -- -o k.rlxisa --ptx k.ptx:sm_86,sm_89
//!
//! # from artifacts someone else compiled
//! cargo run -p rlx-egpu --example egpu_bake -- -o k.rlxisa k_gfx1100.hsaco k_sm86.cubin
//! ```

use std::path::PathBuf;

fn main() {
    let mut out = PathBuf::from("kernels.rlxisa");
    let mut artifacts: Vec<Vec<u8>> = Vec::new();
    let mut args = std::env::args().skip(1);
    let mut any_input = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" => match args.next() {
                Some(p) => out = PathBuf::from(p),
                None => fail("-o needs a path"),
            },
            "--hip" | "--ptx" | "--cuda" => {
                let kind = arg.clone();
                let spec = args
                    .next()
                    .unwrap_or_else(|| fail("expected source:arch,arch"));
                let (source, arches) = spec
                    .split_once(':')
                    .unwrap_or_else(|| fail("expected source:arch,arch"));
                for arch in arches.split(',').filter(|a| !a.is_empty()) {
                    let source = PathBuf::from(source);
                    let built = match kind.as_str() {
                        "--hip" => rlx_egpu::aot::bake_hip(&source, arch),
                        #[cfg(feature = "nvrtc")]
                        "--cuda" => rlx_egpu::aot::bake_cuda(&source, arch),
                        #[cfg(not(feature = "nvrtc"))]
                        "--cuda" => Err(rlx_egpu::EgpuError::Unsupported(
                            "--cuda needs the `nvrtc` feature".into(),
                        )),
                        _ => rlx_egpu::aot::bake_ptx(&source, arch),
                    };
                    match built {
                        Ok(bytes) => {
                            println!("compiled {arch:<10} {} bytes", bytes.len());
                            artifacts.push(bytes);
                            any_input = true;
                        }
                        Err(e) => fail(&format!("{arch}: {e}")),
                    }
                }
            }
            path => match std::fs::read(path) {
                Ok(bytes) => {
                    println!("read     {path} ({} bytes)", bytes.len());
                    artifacts.push(bytes);
                    any_input = true;
                }
                Err(e) => fail(&format!("{path}: {e}")),
            },
        }
    }
    if !any_input {
        fail("no inputs; pass artifacts, or --hip/--ptx source:arch");
    }

    let pack = match rlx_egpu::aot::IsaPack::from_artifacts(artifacts.iter().map(|a| a.as_slice()))
    {
        Ok(p) => p,
        Err(e) => fail(&format!("{e}")),
    };
    let bytes = pack.write();
    if let Err(e) = std::fs::write(&out, &bytes) {
        fail(&format!("{}: {e}", out.display()));
    }

    println!();
    println!("wrote {} ({} bytes)", out.display(), bytes.len());
    for entry in &pack.entries {
        println!(
            "  {:<10} {:>7} bytes  {} kernel(s)",
            entry.target,
            entry.code.len(),
            entry.kernels.len()
        );
    }
}

fn fail(message: &str) -> ! {
    eprintln!("egpu_bake: {message}");
    std::process::exit(1);
}
