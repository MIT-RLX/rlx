// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Open a claimed eGPU over the driver-extension transport and read its
//! identity out of config space, then allocate a DMA buffer and report the
//! physical pages behind it.
//!
//! ```sh
//! cargo run -p rlx-egpu --features dext --example egpu_pci
//! # a different signed extension, without env or code changes:
//! cargo run -p rlx-egpu --features dext --example egpu_pci -- \
//!     --helper /opt/acme/bin/pcie-helper --service acmepci
//! ```

use rlx_egpu::dext::{DextConfig, PciTransport};

fn main() {
    // Build the configuration from flags, falling through to the environment
    // and then the defaults for anything not given.
    let mut builder = DextConfig::builder();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().unwrap_or_else(|| {
            eprintln!("{flag} needs a value");
            std::process::exit(2);
        });
        builder = match flag.as_str() {
            "--helper" => builder.helper(value),
            "--service" => builder.service(value),
            "--socket" => builder.socket(value),
            other => {
                eprintln!("unknown flag {other} (try --helper/--service/--socket)");
                std::process::exit(2);
            }
        };
    }
    let config = builder.build();
    println!(
        "helper {}  service {}  socket {}",
        config.helper().display(),
        config.service(),
        config.socket().display()
    );

    let mut gpu = match PciTransport::connect_with(&config) {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("{e}");
            eprintln!("{}", rlx_egpu::diagnostic());
            std::process::exit(1);
        }
    };

    match gpu.identity() {
        Ok((vendor_id, device_id)) => println!(
            "config space: {} {vendor_id:04x}:{device_id:04x}",
            rlx_egpu::ids::vendor_name(vendor_id)
        ),
        Err(e) => {
            eprintln!("config read failed: {e}");
            std::process::exit(1);
        }
    }

    for bar in 0..6 {
        match gpu.map_bar(bar) {
            Ok(size) if size > 0 => println!("BAR{bar}: {} MiB", size >> 20),
            _ => {}
        }
    }

    match gpu.alloc_dma(64 << 10, false) {
        Ok(buffer) => {
            let pages = buffer.physical_pages();
            println!(
                "DMA: {} bytes across {} pages, first at {:#x}",
                buffer.len(),
                pages.len(),
                pages.first().copied().unwrap_or(0)
            );
        }
        Err(e) => eprintln!("DMA allocation failed: {e}"),
    }
}
