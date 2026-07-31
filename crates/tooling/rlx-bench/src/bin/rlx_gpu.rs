// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-gpu` — GPU temperature / power / clock monitor **and** control.
//!
//! Monitoring (temp/power/clock/fan) is read-only and unprivileged.
//! Control (`--power-cap` / `--lock-clocks` / `--fan` and their resets)
//! needs root — run under `sudo`.
//!
//! A hardware tool (not a model runner), so it lives in the `rlx-bench`
//! tooling crate rather than the model-facing `rlx-run` (which is in the
//! separate rlx-models repo). It reads live telemetry through
//! `rlx_runtime::device_thermal`, whose per-vendor backends
//! (`rlx_cuda::nvml`, `rlx_rocm::rsmi`) are gated behind Cargo features
//! — so to actually reach hardware, build with the matching backend:
//!
//! ```text
//! cargo run -p rlx-bench --bin rlx-gpu --features cuda -- --watch          # NVIDIA (msi)
//! cargo run -p rlx-bench --bin rlx-gpu --features rocm -- --device rocm    # AMD (amd)
//! ```
//!
//! Without a backend feature it prints an empty inventory instead of
//! failing — same graceful-degradation contract as the shims.
//!
//! Zero external deps (rlx-bench carries no clap/serde): args are parsed
//! by hand and `--json` is hand-formatted.

use std::str::FromStr;

use rlx_driver::Device;
use rlx_runtime::{
    GpuThermal, ThermalError, device_thermal, device_thermal_count, power_cap_range, reset_fan,
    reset_locked_clocks, set_fan_percent, set_locked_clocks, set_power_cap,
};

struct Args {
    device: Option<Device>, // None = all GPU backends
    index: Option<u32>,     // None = all indices
    watch: bool,
    interval_ms: u64,
    count: Option<u64>, // None = 1 (or unbounded with --watch)
    json: bool,
    // ── Control (root-only) ──────────────────────────────────────────
    power_cap: Option<f32>,          // watts
    lock_clocks: Option<(u32, u32)>, // (min_mhz, max_mhz)
    reset_clocks: bool,
    fan: Option<f32>, // percent
    reset_fan: bool,
}

impl Args {
    /// Any privileged knob requested → run the control path, not the monitor.
    fn has_control(&self) -> bool {
        self.power_cap.is_some()
            || self.lock_clocks.is_some()
            || self.reset_clocks
            || self.fan.is_some()
            || self.reset_fan
    }
}

fn print_help() {
    eprintln!(
        "rlx-gpu — GPU temperature / power monitor + control (control needs root)\n\
         \n\
         USAGE:\n    rlx-gpu [OPTIONS]\n\
         \n\
         MONITOR (read-only, unprivileged):\n\
         \x20   --device <cuda|rocm|all>   Backend to query (default: all)\n\
         \x20   --index <N>                Only GPU ordinal N (default: all)\n\
         \x20   --watch                    Refresh continuously until Ctrl-C\n\
         \x20   --interval <MS>            Refresh interval in ms (default: 1000)\n\
         \x20   --count <N>                Take N samples then exit (default: 1)\n\
         \x20   --json                     Emit JSON instead of a table\n\
         \x20   -h, --help                 Show this help\n\
         \n\
         CONTROL (needs root / sudo; applied to the selected GPUs):\n\
         \x20   --power-cap <W>            Set sustained power cap, watts\n\
         \x20   --lock-clocks <MHz|A-B>    Pin core clock (NVIDIA only): one value or A-B range\n\
         \x20   --reset-clocks             Release a clock lock (NVIDIA only)\n\
         \x20   --fan <PCT>                Set fan duty %, enters manual mode\n\
         \x20   --reset-fan                Return fan to automatic control\n\
         \n\
         Build with the backend feature to reach hardware:\n\
         \x20   cargo run -p rlx-bench --bin rlx-gpu --features cuda -- --watch\n\
         \x20   sudo $(which rlx-gpu) --device rocm --index 0 --power-cap 200"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        device: None,
        index: None,
        watch: false,
        interval_ms: 1000,
        count: None,
        json: false,
        power_cap: None,
        lock_clocks: None,
        reset_clocks: false,
        fan: None,
        reset_fan: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--watch" => a.watch = true,
            "--json" => a.json = true,
            "--device" => {
                let v = it.next().ok_or("--device needs a value")?;
                if v.eq_ignore_ascii_case("all") {
                    a.device = None;
                } else {
                    let d = Device::from_str(&v).map_err(|e| e.to_string())?;
                    if !matches!(d, Device::Cuda | Device::Rocm) {
                        return Err(format!(
                            "device '{v}' has no thermal backend (supported: cuda, rocm, all)"
                        ));
                    }
                    a.device = Some(d);
                }
            }
            "--index" => {
                a.index = Some(
                    it.next()
                        .ok_or("--index needs a value")?
                        .parse()
                        .map_err(|_| "--index must be a non-negative integer")?,
                );
            }
            "--interval" => {
                a.interval_ms = it
                    .next()
                    .ok_or("--interval needs a value")?
                    .parse()
                    .map_err(|_| "--interval must be a non-negative integer (ms)")?;
            }
            "--count" => {
                a.count = Some(
                    it.next()
                        .ok_or("--count needs a value")?
                        .parse()
                        .map_err(|_| "--count must be a positive integer")?,
                );
            }
            "--power-cap" => {
                let v = it.next().ok_or("--power-cap needs a value (watts)")?;
                let w: f32 = v
                    .parse()
                    .map_err(|_| "--power-cap must be a number (watts)")?;
                if !(w.is_finite() && w > 0.0) {
                    return Err("--power-cap must be a positive number of watts".into());
                }
                a.power_cap = Some(w);
            }
            "--lock-clocks" => {
                let v = it
                    .next()
                    .ok_or("--lock-clocks needs a value (MHz or A-B)")?;
                a.lock_clocks = Some(parse_clock_range(&v)?);
            }
            "--reset-clocks" => a.reset_clocks = true,
            "--fan" => {
                let v = it.next().ok_or("--fan needs a value (percent)")?;
                let p: f32 = v
                    .trim_end_matches('%')
                    .parse()
                    .map_err(|_| "--fan must be a percent 0..100")?;
                if !(0.0..=100.0).contains(&p) {
                    return Err("--fan must be between 0 and 100".into());
                }
                a.fan = Some(p);
            }
            "--reset-fan" => a.reset_fan = true,
            other => return Err(format!("unknown argument '{other}' (try --help)")),
        }
    }
    Ok(a)
}

/// Enumerate the (device, index) pairs to sample given the filters.
fn targets(args: &Args) -> Vec<(Device, u32)> {
    let backends: Vec<Device> = match args.device {
        Some(d) => vec![d],
        None => vec![Device::Cuda, Device::Rocm],
    };
    let mut out = Vec::new();
    for d in backends {
        let n = device_thermal_count(d);
        match args.index {
            Some(i) if (i as usize) < n => out.push((d, i)),
            Some(_) => {} // requested index out of range for this backend
            None => out.extend((0..n as u32).map(|i| (d, i))),
        }
    }
    out
}

fn f1(v: Option<f32>, suffix: &str) -> String {
    match v {
        Some(x) => format!("{x:.0}{suffix}"),
        None => "  -".to_string(),
    }
}

fn print_table(rows: &[GpuThermal]) {
    if rows.is_empty() {
        println!(
            "no GPUs detected — build with `--features cuda` and/or `--features rocm`, \
             and ensure the vendor management library (libnvidia-ml / librocm_smi64) is present."
        );
        return;
    }
    println!(
        "{:<6} {:<3} {:<26} {:>6} {:>8} {:>6} {:>6} {:>7} {:>5}",
        "DEV", "IDX", "NAME", "TEMP", "POWER", "CAP", "FAN", "CLOCK", "UTIL"
    );
    for r in rows {
        let name = r.name.clone().unwrap_or_else(|| "?".to_string());
        let name: String = name.chars().take(26).collect();
        let hot = r
            .hotspot_c
            .map(|h| format!(" (hot {h:.0}°C)"))
            .unwrap_or_default();
        println!(
            "{:<6} {:<3} {:<26} {:>6} {:>8} {:>6} {:>6} {:>7} {:>5}{}",
            r.device.as_arg(),
            r.index,
            name,
            f1(r.temp_c, "°C"),
            f1(r.power_w, "W"),
            f1(r.power_cap_w, "W"),
            f1(r.fan_percent, "%"),
            r.clock_mhz
                .map(|c| format!("{c}MHz"))
                .unwrap_or_else(|| "  -".to_string()),
            f1(r.util_percent, "%"),
            hot,
        );
    }
}

fn json_field_num(v: Option<f32>) -> String {
    v.map(|x| format!("{x:.1}"))
        .unwrap_or_else(|| "null".into())
}

fn print_json(rows: &[GpuThermal]) {
    let mut s = String::from("[");
    for (i, r) in rows.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let name = match &r.name {
            Some(n) => format!("\"{}\"", n.replace('\\', "\\\\").replace('"', "\\\"")),
            None => "null".to_string(),
        };
        s.push_str(&format!(
            "{{\"device\":\"{}\",\"index\":{},\"name\":{},\"temp_c\":{},\"hotspot_c\":{},\
             \"mem_temp_c\":{},\"power_w\":{},\"power_cap_w\":{},\"fan_percent\":{},\
             \"clock_mhz\":{},\"util_percent\":{}}}",
            r.device.as_arg(),
            r.index,
            name,
            json_field_num(r.temp_c),
            json_field_num(r.hotspot_c),
            json_field_num(r.mem_temp_c),
            json_field_num(r.power_w),
            json_field_num(r.power_cap_w),
            json_field_num(r.fan_percent),
            r.clock_mhz
                .map(|c| c.to_string())
                .unwrap_or_else(|| "null".into()),
            json_field_num(r.util_percent),
        ));
    }
    s.push(']');
    println!("{s}");
}

fn sample_once(args: &Args) -> Vec<GpuThermal> {
    targets(args)
        .into_iter()
        .filter_map(|(d, i)| device_thermal(d, i))
        .collect()
}

/// Parse a `--lock-clocks` value: a single `1200` (min=max) or `900-1500`.
fn parse_clock_range(s: &str) -> Result<(u32, u32), String> {
    let s = s.trim().trim_end_matches("MHz").trim_end_matches("mhz");
    let bad = || "--lock-clocks takes an integer MHz or MIN-MAX (0 < MIN <= MAX)".to_string();
    if let Some((lo, hi)) = s.split_once('-') {
        let lo: u32 = lo.trim().parse().map_err(|_| bad())?;
        let hi: u32 = hi.trim().parse().map_err(|_| bad())?;
        if lo == 0 || hi == 0 || lo > hi {
            return Err(bad());
        }
        Ok((lo, hi))
    } else {
        let v: u32 = s.parse().map_err(|_| bad())?;
        if v == 0 {
            return Err(bad());
        }
        Ok((v, v))
    }
}

fn report(dev: &str, index: u32, action: &str, res: Result<(), ThermalError>, failed: &mut bool) {
    match res {
        Ok(()) => println!("{dev} {index}: {action} — OK"),
        Err(e) => {
            *failed = true;
            eprintln!("{dev} {index}: {action} — ERROR: {e}");
        }
    }
}

/// Apply the requested privileged knobs to every selected GPU, print a
/// per-action result, then a fresh telemetry snapshot. Returns a process
/// exit code (1 if any action failed).
fn run_control(args: &Args) -> i32 {
    let tg = targets(args);
    if tg.is_empty() {
        eprintln!(
            "no matching GPUs — check --device/--index and that the backend feature is built in \
             (`--features cuda`/`rocm`)."
        );
        return 1;
    }
    let mut failed = false;
    for (device, index) in &tg {
        let (device, index) = (*device, *index);
        let dev = device.as_arg();

        // Resets first, so `--reset-clocks --lock-clocks X` reads left-to-right.
        if args.reset_clocks {
            report(
                dev,
                index,
                "reset-clocks",
                reset_locked_clocks(device, index),
                &mut failed,
            );
        }
        if let Some((lo, hi)) = args.lock_clocks {
            let label = if lo == hi {
                format!("lock-clocks {lo}MHz")
            } else {
                format!("lock-clocks {lo}-{hi}MHz")
            };
            report(
                dev,
                index,
                &label,
                set_locked_clocks(device, index, lo, hi),
                &mut failed,
            );
        }
        if args.reset_fan {
            report(
                dev,
                index,
                "reset-fan",
                reset_fan(device, index),
                &mut failed,
            );
        }
        if let Some(p) = args.fan {
            report(
                dev,
                index,
                &format!("fan {p:.0}%"),
                set_fan_percent(device, index, p),
                &mut failed,
            );
        }
        if let Some(w) = args.power_cap {
            // Surface the allowed range up front — helps the user pick a
            // valid value when a set is rejected as out-of-range.
            if let Some((min, max)) = power_cap_range(device, index) {
                println!("{dev} {index}: power-cap range {min:.0}..={max:.0}W");
            }
            report(
                dev,
                index,
                &format!("power-cap {w:.0}W"),
                set_power_cap(device, index, w),
                &mut failed,
            );
        }
    }

    // Show the resulting state so the effect is visible immediately.
    println!("\nstate after control:");
    let rows: Vec<GpuThermal> = tg
        .into_iter()
        .filter_map(|(d, i)| device_thermal(d, i))
        .collect();
    print_table(&rows);

    if failed { 1 } else { 0 }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    // Privileged control is a one-shot action, not a monitor loop.
    if args.has_control() {
        std::process::exit(run_control(&args));
    }

    let interval = std::time::Duration::from_millis(args.interval_ms);
    // --watch ⇒ unbounded unless an explicit --count caps it.
    let limit = if args.watch {
        args.count.unwrap_or(u64::MAX)
    } else {
        args.count.unwrap_or(1)
    };

    let mut taken = 0u64;
    loop {
        let rows = sample_once(&args);
        if args.json {
            print_json(&rows);
        } else {
            if args.watch {
                // Cheap "clear" — a form feed keeps scrollback intact while
                // giving a fresh frame per tick on a real terminal.
                print!("\x1B[2J\x1B[H");
            }
            print_table(&rows);
        }
        taken += 1;
        if taken >= limit {
            break;
        }
        std::thread::sleep(interval);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_range_single_value() {
        assert_eq!(parse_clock_range("1200").unwrap(), (1200, 1200));
        assert_eq!(parse_clock_range("900MHz").unwrap(), (900, 900));
    }

    #[test]
    fn clock_range_min_max() {
        assert_eq!(parse_clock_range("900-1500").unwrap(), (900, 1500));
        assert_eq!(parse_clock_range(" 800 - 1200 ").unwrap(), (800, 1200));
    }

    #[test]
    fn clock_range_rejects_bad() {
        assert!(parse_clock_range("0").is_err()); // zero
        assert!(parse_clock_range("1500-900").is_err()); // reversed
        assert!(parse_clock_range("abc").is_err()); // non-numeric
        assert!(parse_clock_range("100-0").is_err()); // zero max
    }
}
