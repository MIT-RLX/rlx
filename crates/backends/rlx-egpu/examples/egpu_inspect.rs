// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Read a compiled GPU artifact with no toolchain, no driver and no device.
//!
//! Accepts an rlx ISA pack, a clang offload bundle (`hipcc --genco`), a bare AMD
//! code object, or an NVIDIA cubin (`ptxas`). This is the half of the AOT path
//! that runs on the machine holding the card — the point being that it needs
//! nothing installed.
//!
//! ```sh
//! cargo run -p rlx-egpu --example egpu_inspect -- kernels.rlxisa
//! ```

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: egpu_inspect <artifact | pack>...");
        std::process::exit(2);
    };

    let mut failures = 0;
    for path in std::iter::once(path).chain(args) {
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("{path}: {e}");
                failures += 1;
                continue;
            }
        };
        println!("{path}  ({} bytes)", bytes.len());

        // A pack first, then a raw artifact: both are readable here, and which
        // one arrived is not always obvious from the extension.
        match rlx_egpu::aot::read(&bytes) {
            Ok(pack) => {
                println!("  rlx ISA pack, {} target(s)", pack.entries.len());
                for entry in &pack.entries {
                    println!(
                        "    {:<10} {:>7} bytes  kernels: {}",
                        entry.target,
                        entry.code.len(),
                        if entry.kernels.is_empty() {
                            "(none)".to_string()
                        } else {
                            entry.kernels.join(", ")
                        }
                    );
                }
            }
            Err(_) => match rlx_egpu::codeobj::parse(&bytes) {
                Ok(objects) => {
                    let kind = if rlx_egpu::codeobj::is_bundle(&bytes) {
                        "clang offload bundle"
                    } else {
                        "code object"
                    };
                    println!("  {kind}, {} entr(ies)", objects.len());
                    for object in &objects {
                        println!(
                            "    {:<10} {:<8} {:>7} bytes  kernels: {}",
                            object.target,
                            object.vendor.to_string(),
                            object.len,
                            if object.kernels.is_empty() {
                                "(none)".to_string()
                            } else {
                                object.kernels.join(", ")
                            }
                        );
                        if let Some(triple) = &object.triple {
                            println!("               triple {triple}");
                        }
                    }
                }
                Err(e) => {
                    eprintln!("  not a pack and not a code object: {e}");
                    failures += 1;
                }
            },
        }
    }
    if failures > 0 {
        std::process::exit(1);
    }
}
