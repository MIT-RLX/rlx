// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Report what the PCIe tunnel is carrying.
//!
//! ```sh
//! cargo run -p rlx-egpu --example egpu_probe
//! ```

fn main() {
    let devices = rlx_egpu::pci::enumerate();
    println!("PCI functions visible to the host: {}", devices.len());
    for device in &devices {
        let location = if device.tunnelled {
            "PCIe tunnel"
        } else {
            "root complex"
        };
        let owner = match device.driver.as_deref() {
            Some(driver) => format!("driver {driver}"),
            None if device.reachable_from_userspace() => "unbound".to_string(),
            None => "unclaimed".to_string(),
        };
        println!(
            "  {}  class {:#04x}  {location}  {owner}",
            device.id_string(),
            device.base_class
        );
    }

    let displays = rlx_egpu::pci::display_devices();
    if !displays.is_empty() {
        println!();
        println!("display-class devices:");
        for device in &displays {
            println!(
                "  {} {}  supported={}  tunnelled={}",
                rlx_egpu::ids::vendor_name(device.vendor_id),
                device.id_string(),
                rlx_egpu::ids::is_supported(device.vendor_id, device.device_id),
                device.tunnelled
            );
            // Firmware is knowable from the part alone, so report it even for a
            // card no bring-up covers — that is exactly the card someone would
            // be fetching blobs for.
            let families = rlx_egpu::ids::firmware_families(device.vendor_id, device.device_id);
            if !families.is_empty() {
                println!(
                    "      firmware: {}   (scripts/pull_gpu_firmware.sh {})",
                    families.join(" "),
                    families.join(" ")
                );
            }
        }
    }

    println!();
    println!("status:    {:?}", rlx_egpu::detect());
    println!("hardware:  {}", rlx_egpu::hardware_present());
    println!("available: {}", rlx_egpu::is_available());
    println!();
    println!("{}", rlx_egpu::diagnostic());
}
