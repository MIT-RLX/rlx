// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo rlx nodes` — machines that can be ranks, found over mDNS
//! (Bonjour on macOS/iOS, Avahi on Linux) as `_rlx-node._tcp`.
//!
//! ```text
//! cargo rlx nodes                         # who is out there (3 s)
//! cargo rlx nodes advertise [--name N] [--port 29500] [--devices metal,cpu]
//! dist_node MODE=advertise                # the same, with the rank binary's backends
//! ```
//!
//! A cluster file can then name a host by what it advertises instead of an
//! address: `addr = "mdns"` resolves the host's `name` at launch.

use std::collections::BTreeMap;
use std::process::ExitCode;
use std::time::Duration;

use rlx_driver::mdns::{self, NodeAdvert};

/// What this machine advertises: its platform, devices, memory and version.
/// `devices` overrides the list: `cargo-rlx` is built with the CPU backend
/// only, so it cannot see a GPU the rank binary will use.
pub fn local_advert(name: Option<String>, port: u16, devices: Option<String>) -> NodeAdvert {
    let mut props: BTreeMap<String, String> = rlx_runtime::dist::node::advert_props();
    if let Some(d) = devices {
        props.insert("devices".into(), d);
    }
    NodeAdvert {
        name: name.unwrap_or_else(hostname),
        port,
        props,
    }
}

fn hostname() -> String {
    std::process::Command::new("hostname")
        .arg("-s")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "rlx-node".into())
}

fn usage() -> ExitCode {
    eprintln!(
        "cargo rlx nodes [--timeout S]\n\
         cargo rlx nodes advertise [--name NAME] [--port PORT] [--devices metal,cpu]\n\n\
         Lists the machines advertising `_rlx-node._tcp` on the LAN (Bonjour/Avahi),\n\
         or advertises this one until interrupted. A cluster file host with\n\
         `addr = \"mdns\"` is resolved by its name the same way."
    );
    ExitCode::from(2)
}

pub fn main(args: &[String]) -> ExitCode {
    let mut it = args.iter();
    let advertise = args.first().map(String::as_str) == Some("advertise");
    if advertise {
        it.next();
    }
    let (mut name, mut port, mut timeout, mut devices) = (None, 29500u16, 3.0f64, None);
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned();
        match a.as_str() {
            "--name" => name = val(),
            "--devices" => devices = val(),
            "--port" => match val().and_then(|v| v.parse().ok()) {
                Some(p) => port = p,
                None => return usage(),
            },
            "--timeout" => match val().and_then(|v| v.parse().ok()) {
                Some(t) => timeout = t,
                None => return usage(),
            },
            _ => return usage(),
        }
    }
    if advertise {
        if devices.is_none() {
            eprintln!(
                "note: cargo-rlx sees only the backends built into it ({}); pass --devices, \
                 or advertise from the rank binary (`dist_node MODE=advertise`)",
                rlx_runtime::dist::node::advert_props()["devices"]
            );
        }
        let ad = local_advert(name, port, devices);
        let _guard = match mdns::advertise_node(&ad) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("cargo-rlx nodes: {e}");
                return ExitCode::FAILURE;
            }
        };
        eprintln!(
            "advertising {:?} on port {} ({}) — Ctrl-C to stop",
            ad.name,
            ad.port,
            ad.props
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        loop {
            std::thread::park();
        }
    }
    match mdns::browse_nodes(Duration::from_secs_f64(timeout)) {
        Ok(found) if found.is_empty() => {
            eprintln!("no node advertised `_rlx-node._tcp` within {timeout} s");
            ExitCode::SUCCESS
        }
        Ok(found) => {
            println!(
                "{:<22} {:<22} {:<8} {:<26} {:>6}  version",
                "name", "address", "platform", "devices", "mem"
            );
            for n in found {
                let p = |k: &str| n.props.get(k).cloned().unwrap_or_else(|| "-".into());
                let addr = n
                    .addr()
                    .map(|a| match a {
                        std::net::IpAddr::V6(v) => format!("[{v}]:{}", n.port),
                        v4 => format!("{v4}:{}", n.port),
                    })
                    .unwrap_or_else(|| "-".into());
                println!(
                    "{:<22} {:<22} {:<8} {:<26} {:>6}  {}",
                    n.name,
                    addr,
                    p("platform"),
                    p("devices"),
                    n.props
                        .get("mem_gb")
                        .map(|g| format!("{g} GB"))
                        .unwrap_or_else(|| "-".into()),
                    p("version")
                );
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("cargo-rlx nodes: {e}");
            ExitCode::FAILURE
        }
    }
}
