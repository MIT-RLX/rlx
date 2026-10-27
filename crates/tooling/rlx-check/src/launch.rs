// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo rlx dist` — launch one distributed job across machines from a cluster
//! file: optionally sync the source and build on every host, then start every
//! rank (over `ssh`, or locally) with `RANK`/`WORLD`/`PEERS`/`DEVICE` set,
//! stream each rank's output with a prefix, and stop the whole job when any
//! rank fails or the timeout passes. A host may run several ranks — one per
//! device — so a job can use every CPU, GPU and NPU in the cluster.
//!
//! ```toml
//! [job]
//! bin = "target/release/examples/dist_node"   # relative to each host's repo
//! args = []
//! port = 29500             # rank r listens on port + r
//! timeout_s = 600
//! retries = 0              # relaunch the whole job on failure (pair with checkpoints)
//! coordinate = "static"    # static (PEERS) | tcp | udp (rank 0 rendezvous on `port`)
//! transport = "tcp"        # tcp | quic (QUIC over UDP; iroh build)
//! build = "cargo build --release -p rlx-collectives --example dist_node"
//! env = { MODE = "pipeline" }
//!
//! [[host]]
//! name = "local"           # `local` = this machine, else an ssh alias / user@host
//! addr = "10.0.0.185"      # address the other ranks dial (IPv6 is fine)
//! repo = "/Users/Shared/rlx"
//! ranks = [{ device = "metal" }, { device = "cpu" }]
//!
//! [[host]]
//! name = "msi"
//! addr = "10.0.0.172"
//! repo = "~/rlx"
//! env = { LD_LIBRARY_PATH = "/usr/local/cuda/lib64" }
//! ranks = [{ device = "cuda" }]
//! ```
//!
//! A remote rank runs under a wrapper that kills it when its `ssh` session
//! ends, so stopping the launcher (or one rank failing) never leaves processes
//! behind on the other machines.
//!
//! `persist = true` (or `--persist`), for `MODE = "model"`: every rank but
//! rank 0 stays up between runs (`PERSIST=1`), and keeps its stage — compiled,
//! with its weights — for a next job with the same plan. A run starts a
//! worker only where none runs with the same settings (its program, its
//! environment, and of the `--set` overrides only `MODE`, `DEVICE`,
//! `TRANSPORT` and `RLX_*`: the rest are rank 0's job); `--build`, `--sync`
//! and `--restart` start them all again. A desktop worker runs detached,
//! its log and pid under `$TMPDIR/rlx-node-<port>-<rank>/`; an iOS one is
//! the app, left running. Either way the launcher follows the worker's log
//! until its part in this job is done.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, ExitCode, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct ClusterFile {
    pub job: Job,
    #[serde(rename = "host")]
    pub hosts: Vec<Host>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Job {
    /// Program every rank runs (relative paths resolve in the host's repo).
    pub bin: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_timeout")]
    pub timeout_s: u64,
    #[serde(default)]
    pub retries: u32,
    /// Shell command `--build` runs in each host's repo.
    #[serde(default)]
    pub build: Option<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Paths `--sync` copies from this machine's repo (relative to it).
    #[serde(default = "default_sync")]
    pub sync: Vec<String>,
    /// How ranks learn each other's addresses: `static` (the launcher writes
    /// `PEERS`, rank r on `port + r`), or `tcp` / `udp` — rank 0 runs a
    /// rendezvous on `port`, workers start as `RANK=auto` and get ephemeral
    /// data ports.
    #[serde(default = "default_coordinate")]
    pub coordinate: String,
    /// `tcp` (default) or `quic` (QUIC over UDP; needs an `iroh` build).
    #[serde(default)]
    pub transport: Option<String>,
    /// Keep the workers up between runs (`MODE = "model"`; see the module
    /// docs).
    #[serde(default)]
    pub persist: bool,
    /// Start persistent workers again even where they run with the same
    /// settings (`--restart`, `--build`, `--sync`).
    #[serde(skip)]
    pub restart: bool,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Host {
    /// `local`, or what `ssh` takes (alias, `user@host`).
    pub name: String,
    /// The address peers dial to reach this host's ranks, or `"mdns"`: find
    /// the node advertising itself as `name` over Bonjour/Avahi
    /// (`cargo rlx nodes advertise`, or the iOS node app) and use its address.
    pub addr: String,
    pub repo: String,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Replaces `job.build` on this host (e.g. `cargo +1.99.1 build …`).
    #[serde(default)]
    pub build: Option<String>,
    /// An iPhone/iPad attached to this machine (a `devicectl` name or
    /// identifier). Its ranks run the RLX node app (`ios/Demo`) through
    /// `xcrun devicectl device process launch`, with the rank's settings as
    /// launch arguments; `repo`, `build` and `--sync` do not apply.
    #[serde(default)]
    pub ios_device: Option<String>,
    /// The node app's bundle identifier (default `com.mit.rlx.nodedemo`).
    #[serde(default)]
    pub ios_bundle: Option<String>,
    /// How this host's ranks reach particular other hosts: host name →
    /// address, in place of that host's `addr` in this host's peer list
    /// only. `"usb"` (between `local` and an `ios_device` host, either way):
    /// the device's USB-C link, the two ends of its CoreDevice tunnel.
    #[serde(default)]
    pub via: BTreeMap<String, String>,
    pub ranks: Vec<RankSpec>,
    /// The `via` hosts reached over USB-C (set from `via = "usb"`): an iPad
    /// opens those links itself, accepting none over its cable.
    #[serde(skip)]
    pub usb_links: Vec<String>,
    /// The `via` hosts reached through a relay this launcher runs (a `via`
    /// value naming `local`): the relay's address as this host reaches it,
    /// and its ports' offset from the job's.
    #[serde(skip)]
    pub relayed: BTreeMap<String, (String, u32)>,
    /// Set on an iOS host another reaches over Bluetooth LE (`via = "ble"`):
    /// the name its node advertises (`RLX_BLE_SERVE`).
    #[serde(skip)]
    pub ble_name: Option<String>,
}

#[derive(Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct RankSpec {
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Replaces `job.bin` + `job.args` for this rank (e.g. a master binary).
    #[serde(default)]
    pub cmd: Option<Vec<String>>,
    #[serde(default)]
    pub label: Option<String>,
}

fn default_port() -> u16 {
    29500
}
fn default_timeout() -> u64 {
    600
}
fn default_coordinate() -> String {
    "static".into()
}
fn default_sync() -> Vec<String> {
    [
        "Cargo.toml",
        "Cargo.lock",
        "clippy.toml",
        "Justfile",
        "docs",
        "crates",
    ]
    .map(String::from)
    .to_vec()
}

/// One process to start.
#[derive(Debug, Clone, PartialEq)]
pub struct Launch {
    pub rank: usize,
    pub host: String,
    pub label: String,
    /// Shell command line (run with `sh -c` locally, via `ssh` remotely).
    pub command: String,
    pub local: bool,
    /// A persistent worker's command: started before the other ranks, which
    /// wait until it says the worker is up ([`WORKER_READY`]).
    pub persistent: bool,
    /// For a persistent rank: stops its worker where this run replaces it,
    /// waiting until it is gone. Every rank's runs before any rank starts.
    pub stop: Option<String>,
}

/// What a persistent worker's command prints once the worker it starts or
/// finds is up. Rank 0 starts after every one has: a worker still waiting
/// from the last run would otherwise join it and then be replaced mid-job.
const WORKER_READY: &str = "[launch] worker ready";

/// `host:port` with IPv6 literals bracketed.
fn peer_addr(addr: &str, port: u32) -> String {
    if addr.contains(':') && !addr.starts_with('[') {
        format!("[{addr}]:{port}")
    } else {
        format!("{addr}:{port}")
    }
}

/// POSIX single-quote a word.
fn sh_quote(s: &str) -> String {
    if !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./:=,+@%".contains(c))
    {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', r"'\''"))
}

fn is_local(name: &str) -> bool {
    name == "local" || name == "localhost"
}

/// The command that runs one rank inside the iOS node app. The app is
/// started without `--console` — that would tie its life to the `devicectl`
/// session, which forwards every signal to it — so it stays open between
/// jobs, advertising itself. `-logfile` sends its stderr (the node's output)
/// to `Documents/rlx-node.log`, headed by this launch's `-jobid`; the command
/// polls that file with `devicectl device copy from`, prints new lines, and
/// exits 0 or 1 on the app's final `status ok:` / `status error:`. (`--`
/// keeps `devicectl` from reading the app's `-key value` arguments as its
/// own options.)
fn ios_command(
    name: &str,
    device: &str,
    bundle: &str,
    env: &BTreeMap<String, String>,
    persist: Option<&Persist>,
) -> Result<String, String> {
    let peers = env.get("PEERS").ok_or(
        "an ios_device rank needs job.coordinate = \"static\" (the app takes a peer list)",
    )?;
    let mode = match env.get("MODE").map(String::as_str) {
        Some("model") => "model",
        Some("trainserve") => "trainserve",
        Some("train") => "train",
        _ => "infer",
    };
    let rank = env.get("RANK").cloned().unwrap_or_default();
    let job = format!(
        "{}-{rank}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    );
    let args = [
        ("rank", rank),
        ("world", env.get("WORLD").cloned().unwrap_or_default()),
        ("peers", peers.clone()),
        (
            "device",
            env.get("DEVICE").cloned().unwrap_or_else(|| "auto".into()),
        ),
        ("mode", mode.into()),
        ("autojoin", "YES".into()),
        ("logfile", "YES".into()),
        // Keeps advertising under the cluster file's name between jobs (iOS
        // 16+ reports a generic device name), so `addr = "mdns"` finds it.
        ("nodename", name.into()),
        ("jobid", job.clone()),
    ];
    // The app reads these as `UserDefaults`, which parses each value as a
    // property list: one starting `[` (an IPv6 peer list) came through empty.
    // Such a value goes as a quoted plist string.
    let plist_str = |v: &str| {
        if v.contains(['[', ']', '(', ')', '{', '}', '<', '>', '"']) {
            format!("\"{}\"", v.replace('\\', "\\\\").replace('"', "\\\""))
        } else {
            v.to_string()
        }
    };
    let args: Vec<String> = args
        .iter()
        .map(|(k, v)| format!("-{k} {}", sh_quote(&plist_str(v))))
        .collect();
    let (dev, app) = (sh_quote(device), sh_quote(bundle));
    // The rank's RLX_* settings (RLX_TRAIN_LANES = "cpu,metal", …) reach the
    // app's process environment, where the runtime reads them.
    let json: Vec<String> = env
        .iter()
        .filter(|(k, _)| k.starts_with("RLX_") || *k == "PERSIST")
        .map(|(k, v)| {
            format!(
                "\"{k}\":\"{}\"",
                v.replace('\\', "\\\\").replace('"', "\\\"")
            )
        })
        .collect();
    let env_flag = if json.is_empty() {
        String::new()
    } else {
        format!(
            "--environment-variables {} ",
            sh_quote(&format!("{{{}}}", json.join(",")))
        )
    };
    let launch = format!(
        "xcrun devicectl device process launch --terminate-existing {env_flag}--device {dev} {app} -- {} \
         >/dev/null || exit 1",
        args.join(" ")
    );
    let fetch = format!(
        "xcrun devicectl device copy from --device {dev} --domain-type appDataContainer \
         --domain-identifier {app} --source Documents/rlx-node.log --destination \"$f\" \
         >/dev/null 2>&1"
    );
    let Some(persist) = persist else {
        return Ok(format!(
            "{launch}; \
             f=$(mktemp); n=0; \
             while :; do \
               {fetch}; \
               if head -1 \"$f\" 2>/dev/null | grep -q 'job {job}$'; then \
                 t=$(wc -l < \"$f\"); [ \"$t\" -gt \"$n\" ] && tail -n +$((n+1)) \"$f\" | head -n $((t-n)); n=$t; \
                 grep -q '^rlx-demo: status ok:' \"$f\" && exit 0; \
                 grep -q '^rlx-demo: status error:' \"$f\" && exit 1; \
               fi; \
               sleep 2; \
             done"
        ));
    };
    // Persistent: launch only when the app is not running with these
    // settings (its process is found by the bundle's install path), and
    // follow the log it has written since the launch that did start it.
    let state = sh_quote(&ios_state_file(device, bundle, persist));
    Ok(format!(
        "st=\"${{TMPDIR:-/tmp}}/\"{state}; fp={fp}; {url}; \
         up=0; [ -n \"$url\" ] && xcrun devicectl device info processes --device {dev} 2>/dev/null \
           | grep -qF \"$url\" && up=1; \
         if [ {restart} = 0 ] && [ $up = 1 ] && [ \"$(cut -d' ' -f1 \"$st\" 2>/dev/null)\" = \"$fp\" ]; then \
           job=$(cut -d' ' -f2 \"$st\"); echo \"[launch] reusing the running node app (job $job)\"; n0=; \
         else \
           {launch}; job={job}; echo \"$fp $job\" > \"$st\"; n0=0; \
         fi; \
         echo '{WORKER_READY}'; \
         f=$(mktemp); off=0; \
         while :; do \
           {fetch}; \
           if head -1 \"$f\" 2>/dev/null | grep -q \"job $job\\$\"; then \
             t=$(wc -l < \"$f\"); \
             k=$(head -n \"$t\" \"$f\" | grep -cE '{FINISHED}'); \
             if [ -z \"$n0\" ]; then n0=$k; off=$t; \
             else [ \"$t\" -gt \"$off\" ] && sed -n \"$((off+1)),${{t}}p\" \"$f\"; off=$t; fi; \
             if [ \"$k\" -gt \"$n0\" ]; then \
               head -n \"$t\" \"$f\" | grep -E '{FINISHED}' | tail -1 | grep -q ' done' && exit 0; exit 1; \
             fi; \
           fi; \
           sleep 2; \
         done",
        fp = sh_quote(&persist.fp),
        url = ios_app_url(&dev, &app),
        restart = u8::from(persist.restart),
    ))
}

/// Whether a `--set` override changes what a persistent worker is (so it
/// starts again), rather than only rank 0's job.
fn worker_override(key: &str) -> bool {
    matches!(key, "MODE" | "DEVICE" | "TRANSPORT") || key.starts_with("RLX_")
}

/// The line a persistent worker logs as its part in a job ends.
const FINISHED: &str = "persistent: job [0-9]+ (done|ended early)";

/// What makes a rank persistent: its settings' fingerprint, where its state
/// lives (`<port>-<rank>`), and whether to start it again regardless.
struct Persist {
    fp: String,
    slot: String,
    port: u16,
    restart: bool,
}

/// Stands in, in a persistent command, for the fingerprints of every
/// persistent worker the cluster runs on that command's host: they're known
/// only once [`plan`] has seen all the ranks.
const HOST_FPS: &str = "__rlx_host_fps__";

/// FNV-1a over `parts`: stable across builds, unlike `DefaultHasher`.
fn fingerprint(parts: &[&str]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in parts {
        for b in p.bytes().chain([0]) {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// A desktop worker kept up between runs: start it detached (its own
/// session, so the `ssh` session ending leaves it be) unless one runs with
/// the same fingerprint, then follow its log until its part in this job is
/// done. `start` runs from the repo and execs the worker.
///
/// First it stops any other worker on this host for the same port that this
/// cluster doesn't run (one from a run with other ranks would otherwise dial
/// the new rank 0 as a rank it no longer is). A slot's pid is read before
/// its fingerprint: a new worker's pid is written after its fingerprint, so
/// a worker that fresh is never taken for a stale one.
fn persist_command(start: &str, p: &Persist) -> String {
    format!(
        "d=\"${{TMPDIR:-/tmp}}/rlx-node-{slot}\"; mkdir -p \"$d\"; fp={fp}; \
         p=$(cat \"$d/pid\" 2>/dev/null); \
         if [ {restart} = 0 ] && [ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null \
            && [ \"$(cat \"$d/fp\" 2>/dev/null)\" = \"$fp\" ]; then \
           echo \"[launch] reusing worker pid $p\"; echo '{WORKER_READY}'; \
           t=$(wc -l < \"$d/log\"); off=$t; n0=$(head -n \"$t\" \"$d/log\" | grep -cE '{FINISHED}'); \
         else \
           [ -n \"$p\" ] && kill \"$p\" 2>/dev/null && sleep 1; \
           rm -f \"$d/pid\"; : > \"$d/log\"; echo \"$fp\" > \"$d/fp\"; \
           ( {start} ) < /dev/null >> \"$d/log\" 2>&1 & \
           i=0; while [ ! -s \"$d/pid\" ]; do \
             i=$((i+1)); [ $i -gt 300 ] && {{ cat \"$d/log\"; echo '[launch] the worker did not start'; exit 1; }}; \
             sleep 0.1; \
           done; \
           p=$(cat \"$d/pid\"); echo \"[launch] started worker pid $p\"; echo '{WORKER_READY}'; \
           off=0; n0=0; \
         fi; \
         while :; do \
           t=$(wc -l < \"$d/log\"); \
           [ \"$t\" -gt \"$off\" ] && sed -n \"$((off+1)),${{t}}p\" \"$d/log\"; off=$t; \
           if [ \"$(head -n \"$t\" \"$d/log\" | grep -cE '{FINISHED}')\" -gt \"$n0\" ]; then \
             head -n \"$t\" \"$d/log\" | grep -E '{FINISHED}' | tail -1 | grep -q ' done' && exit 0; exit 1; \
           fi; \
           kill -0 \"$p\" 2>/dev/null || {{ echo '[launch] the worker exited'; exit 1; }}; \
           sleep 0.3; \
         done",
        slot = p.slot,
        fp = sh_quote(&p.fp),
        restart = u8::from(p.restart),
    )
}

/// Stop a desktop rank's worker where this run replaces it (other settings,
/// or `--restart`), and every other worker on this host for the same port
/// that the cluster no longer runs; return once they are gone. The launcher
/// runs every rank's before it starts any: a new worker that dialed one
/// about to be replaced had its hello taken by the old process, which then
/// exited, and the new one waited for that rank forever.
///
/// A slot's pid is read before its fingerprint: a new worker's pid is
/// written after its fingerprint, so a worker that fresh is never taken for
/// a stale one.
fn persist_stop_command(p: &Persist) -> String {
    format!(
        "gone() {{ i=0; while kill -0 \"$1\" 2>/dev/null; do \
           i=$((i+1)); [ $i -eq 50 ] && kill -9 \"$1\" 2>/dev/null; [ $i -gt 100 ] && return 1; sleep 0.1; \
         done; }}; \
         d=\"${{TMPDIR:-/tmp}}/rlx-node-{slot}\"; fp={fp}; keep='{HOST_FPS}'; \
         for o in \"${{TMPDIR:-/tmp}}\"/rlx-node-{port}-*; do \
           [ \"$o\" = \"$d\" ] || [ ! -d \"$o\" ] && continue; \
           op=$(cat \"$o/pid\" 2>/dev/null); of=$(cat \"$o/fp\" 2>/dev/null); \
           case \" $keep \" in *\" $of \"*) ;; *) \
             [ -n \"$op\" ] && kill \"$op\" 2>/dev/null && gone \"$op\" && echo \"[launch] stopped a stale worker, pid $op\"; \
             rm -rf \"$o\";; esac; \
         done; \
         p=$(cat \"$d/pid\" 2>/dev/null); \
         if [ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null \
            && {{ [ {restart} = 1 ] || [ \"$(cat \"$d/fp\" 2>/dev/null)\" != \"$fp\" ]; }}; then \
           kill \"$p\" 2>/dev/null; gone \"$p\" || {{ echo \"[launch] worker pid $p did not stop\"; exit 1; }}; \
           rm -f \"$d/pid\"; echo \"[launch] stopped worker pid $p\"; \
         fi; true",
        slot = p.slot,
        port = p.port,
        fp = sh_quote(&p.fp),
        restart = u8::from(p.restart),
    )
}

/// The install path of the node app on an iOS device, as `$url` (its process
/// shows it in `devicectl device info processes`).
fn ios_app_url(dev: &str, app: &str) -> String {
    format!(
        "j=$(mktemp); xcrun devicectl device info apps --device {dev} --bundle-id {app} \
           --json-output \"$j\" >/dev/null 2>&1; \
         url=$(grep -o '\"url\" *: *\"file://[^\"]*\"' \"$j\" | head -1 | sed -E 's|.*\"file://([^\"]*)\"|\\1|')"
    )
}

/// The iOS twin of [`persist_stop_command`]: terminate the node app where it
/// runs with other settings (or `--restart`), and wait until it is gone.
fn ios_stop_command(device: &str, bundle: &str, persist: &Persist) -> String {
    let (dev, app) = (sh_quote(device), sh_quote(bundle));
    let state = sh_quote(&ios_state_file(device, bundle, persist));
    format!(
        "st=\"${{TMPDIR:-/tmp}}/\"{state}; fp={fp}; {url}; \
         pid() {{ [ -n \"$url\" ] && xcrun devicectl device info processes --device {dev} 2>/dev/null \
           | grep -F \"$url\" | awk '{{print $1}}' | head -1; }}; \
         p=$(pid); \
         if [ -n \"$p\" ] && {{ [ {restart} = 1 ] || [ \"$(cut -d' ' -f1 \"$st\" 2>/dev/null)\" != \"$fp\" ]; }}; then \
           xcrun devicectl device process terminate --device {dev} --pid \"$p\" --kill >/dev/null 2>&1; \
           i=0; while [ -n \"$(pid)\" ]; do i=$((i+1)); [ $i -gt 20 ] && {{ echo '[launch] the node app did not stop'; exit 1; }}; sleep 1; done; \
           rm -f \"$st\"; echo \"[launch] stopped the node app (pid $p)\"; \
         fi; true",
        fp = sh_quote(&persist.fp),
        url = ios_app_url(&dev, &app),
        restart = u8::from(persist.restart),
    )
}

/// The launcher-side file naming the settings and job id of the node app it
/// started on `device`.
fn ios_state_file(device: &str, bundle: &str, persist: &Persist) -> String {
    format!(
        "rlx-ios-{:016x}-{}",
        fingerprint(&[device, bundle]),
        persist.slot
    )
}

/// `cd` target: `~` must stay unquoted so the remote shell expands it.
fn cd_target(repo: &str) -> String {
    match repo.strip_prefix("~/") {
        Some(rest) => format!("~/{}", sh_quote(rest)),
        None if repo == "~" => "~".into(),
        None => sh_quote(repo),
    }
}

/// Expand the cluster file into one [`Launch`] per rank, in rank order.
/// `overrides` (`--set K=V`) apply to every rank last.
pub fn plan(c: &ClusterFile, overrides: &BTreeMap<String, String>) -> Result<Vec<Launch>, String> {
    let world: usize = c.hosts.iter().map(|h| h.ranks.len()).sum();
    if world == 0 {
        return Err("cluster file has no ranks".into());
    }
    // Each host's peer list: every rank's address as that host reaches it.
    let peers_from = |from: &Host| {
        let mut peers = Vec::with_capacity(world);
        let mut r = 0u32;
        for h in &c.hosts {
            let (addr, offset) = match from.relayed.get(&h.name) {
                Some((a, o)) => (a, *o),
                None => (from.via.get(&h.name).unwrap_or(&h.addr), 0),
            };
            for _ in &h.ranks {
                peers.push(peer_addr(addr, c.job.port as u32 + offset + r));
                r += 1;
            }
        }
        peers.join(",")
    };
    for h in &c.hosts {
        if let Some(to) = h
            .via
            .keys()
            .find(|n| !c.hosts.iter().any(|t| &t.name == *n))
        {
            return Err(format!("host {}: via names no host {to:?}", h.name));
        }
    }
    // A USB-C link's iPad end accepts no connection, so where it is the
    // higher rank (which the lower one would dial), it dials instead.
    let mut first = BTreeMap::new();
    let mut at = 0u32;
    for h in &c.hosts {
        first.insert(
            h.name.clone(),
            (at, h.ranks.len() as u32, h.ios_device.is_some()),
        );
        at += h.ranks.len() as u32;
    }
    let mut reverse = std::collections::BTreeSet::new();
    for h in &c.hosts {
        for t in &h.usb_links {
            let (a0, an, a_ios) = first[&h.name];
            let (b0, bn, b_ios) = first[t];
            for a in a0..a0 + an {
                for b in b0..b0 + bn {
                    let (lo, hi) = (a.min(b), a.max(b));
                    let hi_is_ios = if hi == a { a_ios } else { b_ios };
                    if hi_is_ios {
                        reverse.insert(format!("{lo}-{hi}"));
                    }
                }
            }
        }
    }
    let reverse = reverse.into_iter().collect::<Vec<_>>().join(",");
    let coord = match c.job.coordinate.as_str() {
        "static" => None,
        p @ ("tcp" | "udp") => Some(format!(
            "{p}://{}",
            peer_addr(&c.hosts[0].addr, c.job.port as u32)
        )),
        o => {
            return Err(format!(
                "job.coordinate = {o:?}: expected static, tcp or udp"
            ));
        }
    };
    let mut out = Vec::with_capacity(world);
    // Each rank's persistent fingerprint, for [`HOST_FPS`].
    let mut fps: Vec<Option<String>> = Vec::with_capacity(world);
    let mut rank = 0usize;
    for h in &c.hosts {
        for spec in &h.ranks {
            let device = spec.device.clone().unwrap_or_else(|| "auto".into());
            let label = spec
                .label
                .clone()
                .unwrap_or_else(|| format!("{}/{device}", h.name));
            let mut env: BTreeMap<String, String> = c.job.env.clone();
            env.extend(h.env.clone());
            env.extend(spec.env.clone());
            match &coord {
                None => {
                    env.insert("RANK".into(), rank.to_string());
                    env.insert("WORLD".into(), world.to_string());
                    env.insert("PEERS".into(), peers_from(h));
                    if !reverse.is_empty() {
                        env.insert("RLX_MESH_REVERSE".into(), reverse.clone());
                    }
                    if let Some(name) = &h.ble_name {
                        env.insert("RLX_BLE_SERVE".into(), name.clone());
                    }
                }
                // Rendezvous: only rank 0 is told the world; workers are
                // assigned a rank in arrival order.
                Some(at) => {
                    env.insert("RENDEZVOUS".into(), at.clone());
                    if rank == 0 {
                        env.insert("RANK".into(), "0".into());
                        env.insert("WORLD".into(), world.to_string());
                    } else {
                        env.insert("RANK".into(), "auto".into());
                    }
                }
            }
            if let Some(t) = &c.job.transport {
                env.insert("TRANSPORT".into(), t.clone());
            }
            env.insert("DEVICE".into(), device);
            env.insert("LABEL".into(), label.clone());
            // A persistent worker's settings: all but the overrides only
            // rank 0's job reads.
            let mut settled = env.clone();
            settled.extend(
                overrides
                    .iter()
                    .filter(|(k, _)| worker_override(k))
                    .map(|(k, v)| (k.clone(), v.clone())),
            );
            env.extend(overrides.clone());
            let persist = if c.job.persist && rank > 0 {
                if env.get("MODE").map(String::as_str) != Some("model") {
                    return Err("job.persist needs MODE = \"model\"".into());
                }
                env.insert("PERSIST".into(), "1".into());
                settled.insert("PERSIST".into(), "1".into());
                let mut parts: Vec<String> = vec![h.repo.clone()];
                parts.extend(spec.cmd.iter().flatten().cloned());
                parts.push(c.job.bin.clone());
                parts.extend(c.job.args.iter().cloned());
                parts.extend(settled.iter().map(|(k, v)| format!("{k}={v}")));
                let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
                Some(Persist {
                    fp: format!("{:016x}", fingerprint(&parts)),
                    slot: format!("{}-{rank}", c.job.port),
                    port: c.job.port,
                    restart: c.job.restart,
                })
            } else {
                None
            };
            fps.push(persist.as_ref().map(|p| p.fp.clone()));
            if let Some(dev) = &h.ios_device {
                let bundle = h.ios_bundle.as_deref().unwrap_or("com.mit.rlx.nodedemo");
                out.push(Launch {
                    rank,
                    host: h.name.clone(),
                    label,
                    command: ios_command(&h.name, dev, bundle, &env, persist.as_ref())?,
                    local: true,
                    persistent: persist.is_some(),
                    stop: persist.as_ref().map(|p| ios_stop_command(dev, bundle, p)),
                });
                rank += 1;
                continue;
            }
            let argv: Vec<String> = match &spec.cmd {
                Some(cmd) => cmd.clone(),
                None => std::iter::once(c.job.bin.clone())
                    .chain(c.job.args.iter().cloned())
                    .collect(),
            };
            let mut argv: Vec<String> = argv.iter().map(|a| sh_quote(a)).collect();
            // A bare relative program path runs from the repo.
            if let Some(first) = argv.first_mut()
                && first.contains('/')
                && !first.starts_with('/')
                && !first.starts_with('~')
                && !first.starts_with("./")
            {
                *first = format!("./{first}");
            }
            let envs: Vec<String> = env
                .iter()
                .map(|(k, v)| format!("{k}={}", sh_quote(v)))
                .collect();
            let command = match &persist {
                None => format!(
                    "cd {} && exec env {} {}",
                    cd_target(&h.repo),
                    envs.join(" "),
                    argv.join(" ")
                ),
                // Its own session (`setsid`, through perl: macOS has no
                // setsid(1)), with the pid file the worker writes.
                Some(p) => persist_command(
                    &format!(
                        "cd {} && exec perl -MPOSIX=setsid -e 'setsid; exec @ARGV' env {} PIDFILE=\"$d/pid\" {}",
                        cd_target(&h.repo),
                        envs.join(" "),
                        argv.join(" ")
                    ),
                    p,
                ),
            };
            out.push(Launch {
                rank,
                host: h.name.clone(),
                label,
                command,
                local: is_local(&h.name),
                persistent: persist.is_some(),
                stop: persist.as_ref().map(persist_stop_command),
            });
            rank += 1;
        }
    }
    let mut host_fps: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (l, fp) in out.iter().zip(&fps) {
        if let Some(fp) = fp {
            host_fps.entry(l.host.clone()).or_default().push(fp.clone());
        }
    }
    for l in &mut out {
        if let Some(v) = host_fps.get(&l.host) {
            l.command = l.command.replace(HOST_FPS, &v.join(" "));
            l.stop = l.stop.take().map(|c| c.replace(HOST_FPS, &v.join(" ")));
        }
    }
    Ok(out)
}

/// `addr = "mdns"`: look each such host up by name among the nodes advertised
/// over Bonjour/Avahi.
/// Replace each `via = "usb"` with the address at the far end of an iPad or
/// iPhone's USB-C link: the device's link-local IPv4 on its USB network
/// interface, seen from `local` (its node advertises one per interface), and
/// this Mac's address on that interface, seen from the device. (The
/// CoreDevice tunnel `devicectl` reports reaches only the system's own
/// services, not an app's sockets.)
fn resolve_usb_links(c: &mut ClusterFile) -> Result<(), String> {
    let mut ends: BTreeMap<String, (String, String)> = BTreeMap::new();
    let devices: BTreeMap<String, (String, String)> = c
        .hosts
        .iter()
        .filter_map(|h| {
            let dev = h.ios_device.clone()?;
            let bundle = h
                .ios_bundle
                .clone()
                .unwrap_or_else(|| "com.mit.rlx.nodedemo".into());
            Some((h.name.clone(), (dev, bundle)))
        })
        .collect();
    let mut end = |name: &str| -> Result<(String, String), String> {
        if let Some(e) = ends.get(name) {
            return Ok(e.clone());
        }
        let (dev, bundle) = &devices[name];
        let e = usb_link_ends(name, dev, bundle)?;
        ends.insert(name.to_string(), e.clone());
        Ok(e)
    };
    for h in &mut c.hosts {
        let usb: Vec<String> = h
            .via
            .iter()
            .filter(|(_, a)| a.as_str() == "usb")
            .map(|(t, _)| t.clone())
            .collect();
        for to in usb {
            let addr = match (devices.contains_key(&h.name), devices.contains_key(&to)) {
                (false, true) if h.name == "local" => end(&to)?.0,
                (true, false) if to == "local" => end(&h.name.clone())?.1,
                _ => {
                    return Err(format!(
                        "host {}: via {to:?} = \"usb\" joins `local` and an ios_device host only",
                        h.name
                    ));
                }
            };
            eprintln!("[launch] {} → {to}: USB-C link {addr}", h.name);
            h.usb_links.push(to.clone());
            h.via.insert(to, addr);
        }
    }
    Ok(())
}

/// `(device end, Mac end)` of the USB-C link to the node app on `dev`,
/// advertised as `name` (started, to advertise, if it is not).
fn usb_link_ends(name: &str, dev: &str, bundle: &str) -> Result<(String, String), String> {
    let run = |args: &[&str]| {
        Command::new(args[0])
            .args(&args[1..])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .map_err(|e| format!("{}: {e}", args[0]))
    };
    let info = run(&[
        "xcrun",
        "devicectl",
        "device",
        "info",
        "details",
        "--device",
        dev,
    ])?;
    let wired = info
        .lines()
        .any(|l| l.trim().trim_start_matches("• ") == "transportType: wired");
    if !wired {
        return Err(format!(
            "{dev} is not on a USB-C cable (devicectl transportType)"
        ));
    }
    let mut found = rlx_driver::mdns::find_node(name, Duration::from_secs(6));
    if found.is_err() {
        let _ = Command::new("xcrun")
            .args(["devicectl", "device", "process", "launch"])
            .args(["--terminate-existing", "--device", dev, bundle])
            .args(["--", "-nodename", name])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        found = rlx_driver::mdns::find_node(name, Duration::from_secs(20));
    }
    found.map_err(|e| format!("{name}: not advertising ({e})"))?;
    // `find_node` stops at the first address on this subnet; the record on
    // the USB interface can come later, so browse a whole window for it.
    let link_local = |addrs: &[std::net::IpAddr]| {
        addrs.iter().find_map(|a| match a {
            std::net::IpAddr::V4(v) if v.is_link_local() => Some(v.to_string()),
            _ => None,
        })
    };
    let mut far = None;
    for _ in 0..3 {
        far = rlx_driver::mdns::browse_nodes(Duration::from_secs(4))
            .unwrap_or_default()
            .iter()
            .filter(|n| n.name.eq_ignore_ascii_case(name))
            .find_map(|n| link_local(&n.addrs));
        if far.is_some() {
            break;
        }
    }
    let far = far.ok_or_else(|| format!("{name}: advertises no USB-C (link-local) address"))?;
    let route = run(&["route", "-n", "get", &far])?;
    let ifname = route
        .lines()
        .find_map(|l| l.trim().strip_prefix("interface:"))
        .map(|v| v.trim().to_string())
        .ok_or_else(|| format!("{name}: no route to {far}"))?;
    let near = run(&["ifconfig", &ifname])?
        .lines()
        .find_map(|l| l.trim().strip_prefix("inet "))
        .and_then(|l| l.split_whitespace().next())
        .map(String::from)
        .ok_or_else(|| format!("{name}: {ifname} has no IPv4 address"))?;
    Ok((far, near))
}

/// The relays' ports start this far above the job's (one per relayed rank).
const RELAY_PORT_OFFSET: u32 = 200;

/// The Bluetooth LE bridges' ports start this far above the job's.
const BLE_PORT_OFFSET: u32 = 300;

/// Resolve each `via = "ble"` on `local` to an iOS host into a bridge this
/// launcher runs (`rlx_driver::ble::bridge`, feature `ble`): `local` dials
/// `127.0.0.1` at the job's port + [`BLE_PORT_OFFSET`] + the rank, and the
/// bridge carries that connection over an L2CAP channel to the node, which
/// advertises under its host name (`RLX_BLE_SERVE`). Returns `(name, port)`
/// per bridge.
fn resolve_ble_links(c: &mut ClusterFile) -> Result<Vec<(String, u16)>, String> {
    let mut first = BTreeMap::new();
    let mut at = 0u32;
    for h in &c.hosts {
        first.insert(
            h.name.clone(),
            (at, h.ranks.len() as u32, h.ios_device.is_some()),
        );
        at += h.ranks.len() as u32;
    }
    let mut bridges = Vec::new();
    let mut named = Vec::new();
    for h in &mut c.hosts {
        let ble: Vec<String> = h
            .via
            .iter()
            .filter(|(_, v)| v.as_str() == "ble")
            .map(|(t, _)| t.clone())
            .collect();
        for to in ble {
            let Some(&(r0, rn, ios)) = first.get(&to) else {
                return Err(format!("host {}: via names no host {to:?}", h.name));
            };
            if h.name != "local" || !ios {
                return Err(format!(
                    "host {}: via {to:?} = \"ble\" joins `local` and an ios_device host only",
                    h.name
                ));
            }
            for r in r0..r0 + rn {
                bridges.push((to.clone(), (c.job.port as u32 + BLE_PORT_OFFSET + r) as u16));
            }
            eprintln!("[launch] local → {to}: Bluetooth LE (bridged at 127.0.0.1)");
            h.relayed
                .insert(to.clone(), ("127.0.0.1".into(), BLE_PORT_OFFSET));
            h.via.remove(&to);
            named.push(to);
        }
    }
    for h in &mut c.hosts {
        if named.contains(&h.name) {
            h.ble_name = Some(h.name.clone());
        }
    }
    Ok(bridges)
}

/// Serve `bridges` for the rest of this process.
fn start_ble_bridges(bridges: &[(String, u16)]) -> Result<(), String> {
    if bridges.is_empty() {
        return Ok(());
    }
    #[cfg(all(feature = "ble", not(target_os = "windows")))]
    {
        for (name, port) in bridges {
            rlx_driver::ble::bridge(name, *port).map_err(|e| format!("BLE bridge {port}: {e}"))?;
        }
        Ok(())
    }
    #[cfg(not(all(feature = "ble", not(target_os = "windows"))))]
    Err("via = \"ble\" needs cargo-rlx built with `--features ble`".into())
}

/// A TCP relay this launcher runs for the job: connections on `listen` go
/// on to `target`.
#[derive(Debug, Clone, PartialEq)]
pub struct Relay {
    pub listen: u16,
    pub target: String,
}

/// Resolve each `via` value that names another host — `msi = "local"`:
/// reach msi through `local` — into a relay this launcher runs. The host
/// dials the relay (over its own link to `local`, e.g. USB-C), the relay
/// dials the target as `local` reaches it. Such a pair's connection is
/// opened by the relayed host, which is why an iPad that accepts none over
/// its cable can still reach every rank through it.
fn resolve_relays(c: &mut ClusterFile) -> Result<Vec<Relay>, String> {
    let names: Vec<String> = c.hosts.iter().map(|h| h.name.clone()).collect();
    let mut first = BTreeMap::new();
    let mut at = 0u32;
    for h in &c.hosts {
        first.insert(h.name.clone(), at);
        at += h.ranks.len() as u32;
    }
    let local = c.hosts.iter().find(|h| h.name == "local").cloned();
    let mut relays = Vec::new();
    for i in 0..c.hosts.len() {
        let hops: Vec<(String, String)> = c.hosts[i]
            .via
            .iter()
            .filter(|(t, v)| names.contains(v) && v != t)
            .map(|(t, v)| (t.clone(), v.clone()))
            .collect();
        for (to, through) in hops {
            if through != "local" {
                return Err(format!(
                    "host {}: via {to:?} = {through:?}: a relay runs on `local` only",
                    c.hosts[i].name
                ));
            }
            let local = local.as_ref().ok_or("a relay needs a `local` host")?;
            let h = &c.hosts[i];
            // This host reaches `local` by its own `via` (USB-C), else addr.
            let relay_addr = h.via.get("local").cloned().unwrap_or(local.addr.clone());
            let target_host = c.hosts.iter().find(|t| t.name == to).unwrap();
            let target_addr = local.via.get(&to).unwrap_or(&target_host.addr).clone();
            let r0 = first[&to];
            for k in 0..target_host.ranks.len() as u32 {
                let rank = r0 + k;
                let relay = Relay {
                    listen: (c.job.port as u32 + RELAY_PORT_OFFSET + rank) as u16,
                    target: peer_addr(&target_addr, c.job.port as u32 + rank),
                };
                // One relay per rank, whichever hosts go through it.
                if !relays.contains(&relay) {
                    relays.push(relay);
                }
            }
            eprintln!(
                "[launch] {} → {to}: through local ({relay_addr} → {target_addr})",
                h.name
            );
            let h = &mut c.hosts[i];
            h.relayed
                .insert(to.clone(), (relay_addr, RELAY_PORT_OFFSET));
            h.via.remove(&to);
            // The relayed host opens this pair's connection.
            if h.ios_device.is_some() {
                h.usb_links.push(to);
            }
        }
    }
    Ok(relays)
}

/// Serve `relays` for the rest of this process: each accepted connection is
/// joined to a fresh one to its target, bytes copied both ways.
fn start_relays(relays: &[Relay]) -> Result<(), String> {
    use std::net::{TcpListener, TcpStream};
    for r in relays {
        let listener = TcpListener::bind(("0.0.0.0", r.listen))
            .map_err(|e| format!("relay port {}: {e}", r.listen))?;
        let target = r.target.clone();
        std::thread::spawn(move || {
            for inbound in listener.incoming().flatten() {
                let target = target.clone();
                std::thread::spawn(move || {
                    let mut out = None;
                    for _ in 0..200 {
                        match rlx_driver::resolve_addr(&target).and_then(TcpStream::connect) {
                            Ok(s) => {
                                out = Some(s);
                                break;
                            }
                            Err(_) => std::thread::sleep(Duration::from_millis(50)),
                        }
                    }
                    let Some(out) = out else {
                        eprintln!("[relay] {target}: unreachable");
                        return;
                    };
                    let _ = (inbound.set_nodelay(true), out.set_nodelay(true));
                    let (Ok(mut a), Ok(mut b)) = (inbound.try_clone(), out.try_clone()) else {
                        return;
                    };
                    let (mut ia, mut ob) = (inbound, out);
                    std::thread::spawn(move || {
                        let _ = std::io::copy(&mut ia, &mut b);
                        let _ = b.shutdown(std::net::Shutdown::Write);
                    });
                    let _ = std::io::copy(&mut ob, &mut a);
                    let _ = a.shutdown(std::net::Shutdown::Write);
                });
            }
        });
    }
    Ok(())
}

fn resolve_mdns_hosts(c: &mut ClusterFile) -> Result<(), String> {
    for h in c.hosts.iter_mut().filter(|h| h.addr == "mdns") {
        let mut found = rlx_driver::mdns::find_node(&h.name, Duration::from_secs(6));
        // An iOS app advertises under the device's name until it is told the
        // cluster's (`-nodename`): start it that way, then look again.
        if let (Err(_), Some(dev)) = (&found, &h.ios_device) {
            let bundle = h.ios_bundle.as_deref().unwrap_or("com.mit.rlx.nodedemo");
            eprintln!("[launch] {}: starting the app to advertise it", h.name);
            let ok = Command::new("xcrun")
                .args(["devicectl", "device", "process", "launch"])
                .args(["--terminate-existing", "--device", dev, bundle])
                .args(["--", "-nodename", &h.name])
                .stdout(Stdio::null())
                .status()
                .is_ok_and(|s| s.success());
            if ok {
                found = rlx_driver::mdns::find_node(&h.name, Duration::from_secs(20));
            }
        }
        let n = found.map_err(|e| format!("host {:?}: {e}", h.name))?;
        let ip = n
            .addr()
            .ok_or_else(|| format!("host {:?} advertised no address", h.name))?;
        eprintln!("[launch] {} → {ip} (mDNS)", h.name);
        h.addr = ip.to_string();
    }
    Ok(())
}

/// Wrap a remote command so it dies when the `ssh` session does. The job runs
/// in its own process group (`set -m`) and a watcher kills that whole group
/// when the session's stdin reaches EOF. The watcher reads stdin through fd 3
/// because a background job's own stdin is `/dev/null`, and its output goes to
/// `/dev/null` so it cannot hold the session open after the job exits.
fn remote_wrapper(cmd: &str) -> String {
    let inner = format!(
        "set -m; exec 3<&0; ( {cmd} ) & p=$!; \
         ( cat <&3 >/dev/null; kill -TERM -- -$p 2>/dev/null; sleep 2; \
           kill -KILL -- -$p 2>/dev/null ) >/dev/null 2>&1 & \
         wait $p"
    );
    format!("bash -lc {}", sh_quote(&inner))
}

fn spawn(l: &Launch) -> std::io::Result<Child> {
    let mut c = if l.local {
        let mut c = Command::new("sh");
        c.arg("-c").arg(&l.command);
        c
    } else {
        let mut c = Command::new("ssh");
        c.args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", &l.host])
            .arg(remote_wrapper(&l.command));
        c
    };
    c.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
}

enum Event {
    Line(usize, String),
    Exit(usize, Option<i32>),
}

/// Run every rank once. Returns per-rank exit codes (`None` = killed).
fn run_once(launches: &[Launch], timeout: Duration, quiet: bool) -> Vec<Option<i32>> {
    let (tx, rx) = mpsc::channel::<Event>();
    let mut children: Vec<Option<Child>> = (0..launches.len()).map(|_| None).collect();
    let width = launches.iter().map(|l| l.label.len()).max().unwrap_or(0);
    let start = |i: usize, children: &mut Vec<Option<Child>>| {
        let l = &launches[i];
        match spawn(l) {
            Ok(mut ch) => {
                for stream in [
                    ch.stdout
                        .take()
                        .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
                    ch.stderr
                        .take()
                        .map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
                ]
                .into_iter()
                .flatten()
                {
                    let (tx, r) = (tx.clone(), i);
                    std::thread::spawn(move || {
                        for line in BufReader::new(stream).lines().map_while(Result::ok) {
                            let _ = tx.send(Event::Line(r, line));
                        }
                    });
                }
                children[i] = Some(ch);
            }
            Err(e) => {
                eprintln!("[launch] rank {} on {}: {e}", l.rank, l.host);
                let _ = tx.send(Event::Exit(i, Some(127)));
            }
        }
    };
    // Every persistent rank's stop first, all done before anything starts:
    // a new worker must never meet the process it replaces.
    let stops: Vec<Launch> = launches
        .iter()
        .filter_map(|l| {
            l.stop.as_ref().map(|c| Launch {
                command: c.clone(),
                stop: None,
                persistent: false,
                ..l.clone()
            })
        })
        .collect();
    if !stops.is_empty() {
        let codes = run_once(&stops, Duration::from_secs(120), quiet);
        if let Some((l, c)) = stops.iter().zip(&codes).find(|(_, c)| **c != Some(0)) {
            eprintln!(
                "[launch] stopping the replaced worker of rank {} ({}) failed: {c:?}",
                l.rank, l.label
            );
            return launches.iter().map(|_| None).collect();
        }
    }
    // Persistent workers first, and the rest once each says it is up.
    let mut early: Vec<(usize, Option<i32>)> = Vec::new();
    let mut waiting: std::collections::BTreeSet<usize> = (0..launches.len())
        .filter(|&i| launches[i].persistent)
        .collect();
    for &i in &waiting {
        start(i, &mut children);
    }
    let ready_by = Instant::now() + timeout.min(Duration::from_secs(300));
    while !waiting.is_empty() && early.is_empty() {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Event::Line(r, line)) => {
                if line.contains(WORKER_READY) {
                    waiting.remove(&r);
                } else if !quiet {
                    println!("[{:<width$}] {line}", launches[r].label);
                }
            }
            Ok(Event::Exit(r, code)) => early.push((r, code)),
            Err(_) => {
                for &i in &waiting {
                    if let Some(Ok(Some(st))) = children[i].as_mut().map(Child::try_wait) {
                        early.push((i, st.code()));
                    }
                }
                if early.is_empty() && Instant::now() > ready_by {
                    eprintln!("[launch] persistent worker(s) {waiting:?} never came up");
                    early.push((*waiting.first().unwrap(), None));
                }
            }
        }
    }
    if early.is_empty() {
        for i in (0..launches.len()).filter(|&i| !launches[i].persistent) {
            start(i, &mut children);
        }
    }
    // Waiter threads poll each child so exits arrive as events.
    let shared: Vec<std::sync::Arc<std::sync::Mutex<Option<Child>>>> = children
        .into_iter()
        .map(|c| std::sync::Arc::new(std::sync::Mutex::new(c)))
        .collect();
    for (r, ch) in shared.iter().enumerate() {
        let (ch, tx) = (ch.clone(), tx.clone());
        std::thread::spawn(move || {
            loop {
                let status = {
                    let mut g = ch.lock().unwrap();
                    match g.as_mut() {
                        None => return,
                        Some(c) => c.try_wait(),
                    }
                };
                match status {
                    Ok(Some(s)) => {
                        let _ = tx.send(Event::Exit(r, s.code()));
                        return;
                    }
                    Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                    Err(_) => {
                        let _ = tx.send(Event::Exit(r, None));
                        return;
                    }
                }
            }
        });
    }
    drop(tx);
    let kill_all = |why: &str| {
        eprintln!("[launch] stopping every rank: {why}");
        for ch in &shared {
            if let Some(c) = ch.lock().unwrap().as_mut() {
                drop(c.stdin.take()); // remote wrapper sees EOF and kills the job
                let _ = c.kill();
            }
        }
    };
    let deadline = Instant::now() + timeout;
    let mut codes: Vec<Option<Option<i32>>> = vec![None; launches.len()];
    let mut stopping = false;
    if let Some(&(r, code)) = early.first() {
        // A worker failed to come up: the ranks never started count as
        // killed, the others are stopped.
        for (i, c) in codes.iter_mut().enumerate() {
            if !launches[i].persistent {
                *c = Some(None);
            }
        }
        codes[r] = Some(code);
        stopping = true;
        kill_all(&format!(
            "rank {r} ({}) did not come up ({})",
            launches[r].label,
            code.map_or("no exit code".into(), |c| format!("exit {c}"))
        ));
    }
    while codes.iter().any(Option::is_none) {
        let left = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(left.max(Duration::from_millis(1))) {
            Ok(Event::Line(r, line)) => {
                if !quiet {
                    println!("[{:<width$}] {line}", launches[r].label);
                }
            }
            Ok(Event::Exit(r, code)) => {
                codes[r] = Some(code);
                if code != Some(0) && !stopping {
                    stopping = true;
                    kill_all(&format!(
                        "rank {r} ({}) exited with {}",
                        launches[r].label,
                        code.map_or("a signal".into(), |c| c.to_string())
                    ));
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if !stopping {
                    stopping = true;
                    kill_all(&format!("timeout after {}s", timeout.as_secs()));
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    // Drain any trailing output.
    codes.into_iter().map(Option::flatten).collect()
}

/// Run `cmd` on every distinct host (in parallel), streaming output.
fn on_each_host(c: &ClusterFile, what: &str, cmd_for: impl Fn(&Host) -> Option<String>) -> bool {
    let mut seen = std::collections::BTreeSet::new();
    let launches: Vec<Launch> = c
        .hosts
        .iter()
        .filter(|h| h.ios_device.is_none() && seen.insert(h.name.clone()))
        .filter_map(|h| {
            cmd_for(h).map(|cmd| Launch {
                rank: 0,
                host: h.name.clone(),
                label: format!("{}:{what}", h.name),
                command: cmd,
                local: is_local(&h.name),
                persistent: false,
                stop: None,
            })
        })
        .enumerate()
        .map(|(i, mut l)| {
            l.rank = i;
            l
        })
        .collect();
    if launches.is_empty() {
        return true;
    }
    run_once(&launches, Duration::from_secs(3600), false)
        .iter()
        .all(|c| *c == Some(0))
}

fn sync_cmd(c: &ClusterFile, h: &Host) -> Option<Vec<String>> {
    if is_local(&h.name) || h.ios_device.is_some() {
        return None;
    }
    // No `-t`: synced files get the current time, so cargo never mistakes a
    // freshly synced file for older than its last build.
    let mut args: Vec<String> = ["-rlpgoDz", "--exclude", "target/", "--exclude", ".venv/"]
        .map(String::from)
        .to_vec();
    for x in [
        ".build/",
        "*.so",
        "*.a",
        "*.xcframework",
        "node_modules/",
        "vendor/mlx/.git",
    ] {
        args.push("--exclude".into());
        args.push(x.into());
    }
    args.extend(c.job.sync.iter().cloned());
    args.push(format!("{}:{}/", h.name, h.repo));
    Some(args)
}

fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts::default();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--sync" => o.sync = true,
            "--persist" => o.persist = true,
            "--restart" => o.restart = true,
            "--build" => o.build = true,
            "--dry-run" | "-n" => o.dry_run = true,
            "--quiet" | "-q" => o.quiet = true,
            "--set" => {
                let kv = it.next().ok_or("--set needs K=V")?;
                let (k, v) = kv.split_once('=').ok_or("--set needs K=V")?;
                o.overrides.insert(k.to_string(), v.to_string());
            }
            "--retries" => {
                o.retries = Some(
                    it.next()
                        .and_then(|v| v.parse().ok())
                        .ok_or("--retries needs a number")?,
                )
            }
            "--timeout" => {
                o.timeout = Some(
                    it.next()
                        .and_then(|v| v.parse().ok())
                        .ok_or("--timeout needs seconds")?,
                )
            }
            "-h" | "--help" => return Err(String::new()),
            s if s.starts_with('-') => return Err(format!("unknown flag {s}")),
            s => o.file = Some(s.to_string()),
        }
    }
    Ok(o)
}

#[derive(Default)]
struct Opts {
    file: Option<String>,
    sync: bool,
    persist: bool,
    restart: bool,
    build: bool,
    dry_run: bool,
    quiet: bool,
    retries: Option<u32>,
    timeout: Option<u64>,
    overrides: BTreeMap<String, String>,
}

pub const HELP: &str = "\
cargo rlx dist <CLUSTER.toml> [--sync] [--build] [--set K=V]… [--retries N]
                              [--timeout S] [--persist] [--restart]
                              [--dry-run] [--quiet]
  Start one distributed job on every rank the cluster file lists (one rank
  per device; several per host is fine) with RANK/WORLD/PEERS/DEVICE/LABEL set,
  prefix each rank's output, and stop every rank when one fails or the timeout
  passes. --sync rsyncs the source to remote hosts, --build runs job.build (or
  host.build) on every host, --set overrides an env var on every rank, and
  --retries relaunches the whole job after a failure (training resumes from
  its checkpoint_dir). --persist (MODE=model) keeps every worker rank up
  between runs, with its stage and weights, and reuses it while its settings
  stay the same; --restart starts them again. See the module docs of
  rlx_check::launch for the format.";

/// `cargo rlx dist` entry point.
pub fn main(args: &[String]) -> ExitCode {
    let o = match parse_args(args) {
        Ok(o) => o,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("cargo rlx dist: {e}");
            }
            println!("{HELP}");
            return ExitCode::from(2);
        }
    };
    let Some(path) = o.file.as_deref() else {
        println!("{HELP}");
        return ExitCode::from(2);
    };
    let mut c: ClusterFile = match std::fs::read_to_string(path)
        .map_err(|e| e.to_string())
        .and_then(|s| toml::from_str(&s).map_err(|e| e.to_string()))
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cargo rlx dist: {path}: {e}");
            return ExitCode::from(2);
        }
    };
    c.job.persist |= o.persist;
    c.job.restart = o.restart || o.build || o.sync;
    if let Err(e) = resolve_mdns_hosts(&mut c) {
        eprintln!("cargo rlx dist: {e}");
        return ExitCode::from(2);
    }
    if let Err(e) = resolve_usb_links(&mut c) {
        eprintln!("cargo rlx dist: {e}");
        return ExitCode::from(2);
    }
    let ble_bridges = match resolve_ble_links(&mut c) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cargo rlx dist: {e}");
            return ExitCode::from(2);
        }
    };
    let relays = match resolve_relays(&mut c) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("cargo rlx dist: {e}");
            return ExitCode::from(2);
        }
    };
    let launches = match plan(&c, &o.overrides) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("cargo rlx dist: {e}");
            return ExitCode::from(2);
        }
    };
    if !o.dry_run
        && let Err(e) = start_relays(&relays).and_then(|_| start_ble_bridges(&ble_bridges))
    {
        eprintln!("cargo rlx dist: {e}");
        return ExitCode::from(2);
    }
    if o.dry_run {
        for l in &launches {
            let how = if l.local { "local" } else { l.host.as_str() };
            println!("rank {} [{}] @{how}: {}", l.rank, l.label, l.command);
        }
        return ExitCode::SUCCESS;
    }
    if o.sync {
        for h in &c.hosts {
            if let Some(args) = sync_cmd(&c, h) {
                eprintln!("[launch] sync → {}:{}", h.name, h.repo);
                let ok = Command::new("rsync")
                    .args(&args)
                    .status()
                    .is_ok_and(|s| s.success());
                if !ok {
                    eprintln!("[launch] rsync to {} failed", h.name);
                    return ExitCode::FAILURE;
                }
            }
        }
    }
    if o.build {
        let ok = on_each_host(&c, "build", |h| {
            h.build
                .clone()
                .or_else(|| c.job.build.clone())
                .map(|b| format!("cd {} && {b}", cd_target(&h.repo)))
        });
        if !ok {
            eprintln!("[launch] build failed");
            return ExitCode::FAILURE;
        }
    }
    let timeout = Duration::from_secs(o.timeout.unwrap_or(c.job.timeout_s));
    let attempts = 1 + o.retries.unwrap_or(c.job.retries);
    for attempt in 1..=attempts {
        let t = Instant::now();
        let codes = run_once(&launches, timeout, o.quiet);
        let ok = codes.iter().all(|c| *c == Some(0));
        eprintln!(
            "[launch] {} rank(s), attempt {attempt}/{attempts}, {:.1}s:",
            launches.len(),
            t.elapsed().as_secs_f64()
        );
        for (l, code) in launches.iter().zip(&codes) {
            eprintln!(
                "[launch]   rank {:>2} {:<24} {}",
                l.rank,
                l.label,
                match code {
                    Some(0) => "ok".to_string(),
                    Some(c) => format!("exit {c}"),
                    None => "killed".to_string(),
                }
            );
        }
        if ok {
            return ExitCode::SUCCESS;
        }
    }
    ExitCode::FAILURE
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = r#"
        [job]
        bin = "target/release/examples/dist_node"
        port = 29500
        env = { MODE = "pipeline" }

        [[host]]
        name = "local"
        addr = "10.0.0.185"
        repo = "/Users/Shared/rlx"
        ranks = [{ device = "metal" }, { device = "cpu" }]

        [[host]]
        name = "amd"
        addr = "2601:646::105f"
        repo = "~/rlx"
        env = { HSA_OVERRIDE_GFX_VERSION = "11.0.0" }
        ranks = [{ device = "rocm", env = { MODE = "worker" } }]
    "#;

    #[test]
    fn an_ios_host_runs_its_rank_through_devicectl() {
        let f = format!(
            "{FILE}\n[[host]]\nname = \"ipad\"\naddr = \"10.0.0.13\"\nrepo = \"-\"\n\
             ios_device = \"Yellow iPad\"\nranks = [{{ device = \"metal\", \
             env = {{ RLX_TRAIN_LANES = \"cpu,metal\" }} }}]\n"
        );
        let c: ClusterFile = toml::from_str(&f).unwrap();
        let mut set = BTreeMap::new();
        set.insert("MODE".to_string(), "model".to_string());
        let l = plan(&c, &set).unwrap();
        let ipad = &l[3];
        assert!(ipad.local, "devicectl runs on this machine, not over ssh");
        assert_eq!(ipad.rank, 3);
        let cmd = &ipad.command;
        assert!(
            cmd.contains("devicectl device process launch --terminate-existing"),
            "{cmd}"
        );
        assert!(
            !cmd.contains("--console"),
            "the app outlives the job: {cmd}"
        );
        assert!(
            cmd.contains("copy from") && cmd.contains("Documents/rlx-node.log"),
            "{cmd}"
        );
        assert!(
            cmd.contains("--device 'Yellow iPad' com.mit.rlx.nodedemo -- "),
            "{cmd}"
        );
        // The rank's RLX_* settings go to the app's environment.
        assert!(
            cmd.contains(r#"--environment-variables '{"RLX_TRAIN_LANES":"cpu,metal"}'"#),
            "{cmd}"
        );
        for arg in [
            "-rank 3",
            "-world 4",
            "-device metal",
            "-mode model",
            "-autojoin YES",
            "-logfile YES",
            "-jobid ",
            "-nodename ipad",
        ] {
            assert!(cmd.contains(arg), "{arg} missing: {cmd}");
        }
        assert!(
            cmd.contains(",10.0.0.13:29503\"'"),
            "the iPad is rank 3 in PEERS: {cmd}"
        );
        // The other ranks dial it at its port like any other host.
        assert!(l[0].command.contains("10.0.0.13:29503"));
        // No sync, no build: an iPad has no repo.
        assert!(sync_cmd(&c, &c.hosts[2]).is_none());
    }

    #[test]
    fn an_ios_peer_list_with_ipv6_goes_as_a_plist_string() {
        let f = format!(
            "{FILE}\n[[host]]\nname = \"ipad\"\naddr = \"10.0.0.13\"\nrepo = \"-\"\n\
             ios_device = \"Yellow iPad\"\nranks = [{{ device = \"metal\" }}]\n"
        );
        let c: ClusterFile = toml::from_str(&f).unwrap();
        let cmd = &plan(&c, &BTreeMap::new()).unwrap()[3].command;
        assert!(
            cmd.contains(r#"-peers '"10.0.0.185:29500,10.0.0.185:29501,[2601:646::105f]:29502,10.0.0.13:29503"'"#),
            "{cmd}"
        );
        assert!(cmd.contains("-rank 3 "), "{cmd}");
    }

    #[test]
    fn a_via_naming_local_relays_the_link_and_reverses_its_pair() {
        let f = format!(
            "{FILE}\n[[host]]\nname = \"ipad\"\naddr = \"10.0.0.13\"\nrepo = \"-\"\n\
             ios_device = \"Yellow iPad\"\nvia = {{ local = \"169.254.1.2\", amd = \"local\" }}\n\
             ranks = [{{ device = \"metal\" }}]\n"
        );
        let mut c: ClusterFile = toml::from_str(&f).unwrap();
        let relays = resolve_relays(&mut c).unwrap();
        // amd is rank 2: its relay listens at 29500 + 200 + 2.
        assert_eq!(
            relays,
            vec![Relay {
                listen: 29702,
                target: "[2601:646::105f]:29502".into()
            }]
        );
        let l = plan(&c, &BTreeMap::new()).unwrap();
        let ipad = &l[3].command;
        assert!(ipad.contains("169.254.1.2:29702"), "{ipad}");
        // The iPad (rank 3) dials amd (rank 2): their pair is reversed.
        assert!(
            l[0].command.contains("RLX_MESH_REVERSE=2-3"),
            "{}",
            l[0].command
        );
    }

    #[test]
    fn a_ble_link_goes_through_a_local_bridge_and_names_the_node() {
        let f = format!(
            "{FILE}\n[[host]]\nname = \"ipad\"\naddr = \"10.0.0.13\"\nrepo = \"-\"\n\
             ios_device = \"Yellow iPad\"\nranks = [{{ device = \"metal\" }}]\n"
        )
        .replace(
            "ranks = [{ device = \"metal\" }, { device = \"cpu\" }]",
            "via = { ipad = \"ble\" }\n        ranks = [{ device = \"metal\" }, { device = \"cpu\" }]",
        );
        let mut c: ClusterFile = toml::from_str(&f).unwrap();
        let bridges = resolve_ble_links(&mut c).unwrap();
        // The iPad is rank 3: its bridge listens at 29500 + 300 + 3.
        assert_eq!(bridges, vec![("ipad".to_string(), 29803)]);
        let l = plan(&c, &BTreeMap::new()).unwrap();
        // `local` dials the bridge; amd still reaches the iPad on its address.
        assert!(l[0].command.contains("127.0.0.1:29803"), "{}", l[0].command);
        assert!(l[2].command.contains("10.0.0.13:29503"), "{}", l[2].command);
        assert!(
            l[3].command.contains(r#""RLX_BLE_SERVE":"ipad""#),
            "{}",
            l[3].command
        );
        // Only `local` and an iOS host.
        let bad = FILE.replace(
            "repo = \"~/rlx\"",
            "repo = \"~/rlx\"\n        via = { local = \"ble\" }",
        );
        let mut c: ClusterFile = toml::from_str(&bad).unwrap();
        assert!(resolve_ble_links(&mut c).is_err());
    }

    #[test]
    fn via_changes_one_host_s_peer_list_only() {
        let f = FILE.replace(
            "ranks = [{ device = \"metal\" }, { device = \"cpu\" }]",
            "via = { amd = \"fd00::1\" }\n        ranks = [{ device = \"metal\" }, { device = \"cpu\" }]",
        );
        let c: ClusterFile = toml::from_str(&f).unwrap();
        let l = plan(&c, &BTreeMap::new()).unwrap();
        // `local` reaches amd at the `via` address; amd keeps the file's.
        assert!(l[0].command.contains("[fd00::1]:29502"), "{}", l[0].command);
        assert!(
            l[2].command.contains("[2601:646::105f]:29502"),
            "{}",
            l[2].command
        );
        let bad = FILE.replace(
            "repo = \"~/rlx\"",
            "repo = \"~/rlx\"\n        via = { nobody = \"x\" }",
        );
        let c: ClusterFile = toml::from_str(&bad).unwrap();
        assert!(plan(&c, &BTreeMap::new()).is_err());
    }

    #[test]
    fn plan_assigns_ranks_ports_and_env_in_order() {
        let c: ClusterFile = toml::from_str(FILE).unwrap();
        let mut set = BTreeMap::new();
        set.insert("STEPS".to_string(), "10".to_string());
        let l = plan(&c, &set).unwrap();
        assert_eq!(l.len(), 3);
        // Quoted: `[` is a shell glob character.
        let peers = "PEERS='10.0.0.185:29500,10.0.0.185:29501,[2601:646::105f]:29502'";
        for x in &l {
            assert!(x.command.contains(peers), "{}", x.command);
            assert!(x.command.contains("WORLD=3"));
            assert!(x.command.contains("STEPS=10"));
        }
        assert!(l[0].local && l[1].local && !l[2].local);
        assert!(l[0].command.contains("RANK=0") && l[0].command.contains("DEVICE=metal"));
        assert!(l[1].command.contains("RANK=1") && l[1].command.contains("DEVICE=cpu"));
        // Host and rank env layer over the job's; the rank wins.
        assert!(l[2].command.contains("MODE=worker") && !l[2].command.contains("MODE=pipeline"));
        assert!(l[2].command.contains("HSA_OVERRIDE_GFX_VERSION=11.0.0"));
        assert!(l[2].command.starts_with("cd ~/rlx && exec env "));
        assert!(
            l[2].command
                .ends_with(" ./target/release/examples/dist_node")
        );
        assert_eq!(l[2].label, "amd/rocm");
    }

    #[test]
    fn rendezvous_mode_points_every_rank_at_rank_zero() {
        let f = FILE.replace(
            "port = 29500",
            "port = 29400\ncoordinate = \"udp\"\ntransport = \"quic\"",
        );
        let c: ClusterFile = toml::from_str(&f).unwrap();
        let l = plan(&c, &BTreeMap::new()).unwrap();
        for x in &l {
            assert!(
                x.command.contains("RENDEZVOUS=udp://10.0.0.185:29400"),
                "{}",
                x.command
            );
            assert!(x.command.contains("TRANSPORT=quic"));
            assert!(!x.command.contains("PEERS="));
        }
        assert!(l[0].command.contains("RANK=0") && l[0].command.contains("WORLD=3"));
        assert!(l[1].command.contains("RANK=auto") && !l[1].command.contains("WORLD="));
    }

    #[test]
    fn quoting_survives_spaces_and_quotes() {
        assert_eq!(sh_quote("plain-1.0"), "plain-1.0");
        assert_eq!(sh_quote("a b"), "'a b'");
        assert_eq!(sh_quote("it's"), r"'it'\''s'");
        assert_eq!(peer_addr("fe80::1%2", 7), "[fe80::1%2]:7");
        assert_eq!(peer_addr("host.local", 7), "host.local:7");
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let bad = FILE.replace("port = 29500", "prot = 29500");
        assert!(toml::from_str::<ClusterFile>(&bad).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn local_job_runs_and_a_failing_rank_stops_the_rest() {
        let ok = Launch {
            rank: 0,
            host: "local".into(),
            label: "a".into(),
            command: "echo hi".into(),
            local: true,
            persistent: false,
            stop: None,
        };
        assert_eq!(
            run_once(std::slice::from_ref(&ok), Duration::from_secs(10), true),
            vec![Some(0)]
        );
        let slow = Launch {
            rank: 1,
            label: "b".into(),
            command: "sleep 30".into(),
            ..ok.clone()
        };
        let bad = Launch {
            rank: 0,
            label: "c".into(),
            command: "exit 3".into(),
            ..ok
        };
        let t = Instant::now();
        let codes = run_once(&[bad, slow], Duration::from_secs(60), true);
        assert_eq!(codes[0], Some(3));
        assert_ne!(codes[1], Some(0));
        assert!(
            t.elapsed() < Duration::from_secs(10),
            "the failure must stop rank 1"
        );
    }

    /// A persistent worker starts on the first run, is found and reused by
    /// the next (each run ending when the worker logs its job done), and
    /// starts again once its settings change.
    #[cfg(unix)]
    #[test]
    fn a_persistent_worker_is_started_once_then_reused() {
        let dir = std::env::temp_dir().join(format!("rlx-persist-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("go-a")).unwrap();
        std::fs::create_dir_all(dir.join("go-b")).unwrap();
        // Stands in for dist_node PERSIST=1: pid file, then one job per
        // trigger file.
        let worker = dir.join("worker.sh");
        std::fs::write(
            &worker,
            "#!/bin/sh\necho $$ > \"$PIDFILE\"\nn=1\nwhile :; do\n  \
             echo \"persistent: waiting for job $n\"\n  \
             while [ ! -f \"$GO/$n\" ]; do sleep 0.05; done\n  \
             echo \"persistent: job $n done in 0.1s\"\n  n=$((n+1))\ndone\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&worker, std::fs::Permissions::from_mode(0o755)).unwrap();
        // `GO`, the trigger directory, is a worker setting.
        let file = |port: u16, go: &str| {
            format!(
                "[job]\nbin = {:?}\nport = {port}\npersist = true\n\
                 env = {{ MODE = \"model\", GO = {:?} }}\n\
                 [[host]]\nname = \"local\"\naddr = \"127.0.0.1\"\nrepo = {:?}\n\
                 ranks = [{{ device = \"cpu\" }}, {{ device = \"cpu\" }}]\n",
                worker.to_str().unwrap(),
                dir.join(go).to_str().unwrap(),
                dir.to_str().unwrap()
            )
        };
        let port = 40000 + (std::process::id() % 20000) as u16;
        let run = |go: &str, job: u32, prompt: &str| -> String {
            let c: ClusterFile = toml::from_str(&file(port, go)).unwrap();
            let mut set = BTreeMap::new();
            set.insert("PROMPT".to_string(), prompt.to_string());
            let l = plan(&c, &set).unwrap();
            assert!(!l[0].command.contains("PERSIST"), "rank 0 runs as before");
            assert!(l[0].stop.is_none());
            // As `run_once` does: the rank's stop, then its start.
            let stopped = Command::new("sh")
                .arg("-c")
                .arg(l[1].stop.as_ref().unwrap())
                .env("TMPDIR", &dir)
                .output()
                .unwrap();
            assert!(stopped.status.success());
            let mut ch = Command::new("sh")
                .arg("-c")
                .arg(&l[1].command)
                .env("TMPDIR", &dir)
                .stdout(Stdio::piped())
                .spawn()
                .unwrap();
            std::thread::sleep(Duration::from_millis(500));
            std::fs::write(dir.join(go).join(job.to_string()), "").unwrap();
            let t = Instant::now();
            while ch.try_wait().unwrap().is_none() {
                if t.elapsed() > Duration::from_secs(20) {
                    let _ = ch.kill();
                    panic!("the run never ended");
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            assert!(ch.wait().unwrap().success());
            let mut out = String::from_utf8_lossy(&stopped.stdout).into_owned();
            std::io::Read::read_to_string(&mut ch.stdout.take().unwrap(), &mut out).unwrap();
            out
        };
        // The worker this run started or reused (not one it stopped).
        let pid = |out: &str| -> String {
            ["started worker pid ", "reusing worker pid "]
                .iter()
                .find_map(|m| out.split(m).nth(1))
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap()
                .into()
        };
        // A worker this cluster doesn't run (a rank from a layout with more
        // ranks) is stopped.
        // Through a shell that exits at once, so (like a real worker in its
        // own session) the stale one is reaped when killed, not left a
        // zombie of this test that `kill -0` still finds.
        let spawned = Command::new("sh")
            .args(["-c", "sleep 60 >/dev/null 2>&1 & echo $!"])
            .output()
            .unwrap();
        let stale = String::from_utf8_lossy(&spawned.stdout).trim().to_string();
        let slot = dir.join(format!("rlx-node-{port}-7"));
        std::fs::create_dir_all(&slot).unwrap();
        std::fs::write(slot.join("pid"), &stale).unwrap();
        std::fs::write(slot.join("fp"), "0123456789abcdef").unwrap();
        let first = run("go-a", 1, "a");
        assert!(first.contains("stopped a stale worker"), "{first}");
        let alive = |pid: &str| {
            Command::new("kill")
                .args(["-0", pid])
                .status()
                .is_ok_and(|s| s.success())
        };
        assert!(!alive(&stale) && !slot.exists());
        assert!(first.contains("started worker pid"), "{first}");
        assert!(first.contains("persistent: job 1 done"), "{first}");
        // A different prompt is rank 0's business: the same worker serves it.
        let second = run("go-a", 2, "b");
        assert!(second.contains("reusing worker pid"), "{second}");
        assert_eq!(pid(&first), pid(&second));
        assert!(second.contains("persistent: job 2 done"), "{second}");
        assert!(
            !second.contains("job 1 done"),
            "only this job's lines: {second}"
        );
        // A changed worker setting starts it again (its jobs count from 1).
        let third = run("go-b", 1, "b");
        assert!(third.contains("started worker pid"), "{third}");
        assert!(third.contains("[launch] stopped worker pid"), "{third}");
        assert_ne!(pid(&first), pid(&third));
        assert!(
            Command::new("kill")
                .args(["-0", &pid(&first)])
                .status()
                .is_ok_and(|s| !s.success()),
            "the replaced worker is gone"
        );
        let _ = Command::new("kill").arg(pid(&third)).status();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Rank 0 starts only once every persistent worker says it is up, and a
    /// worker that never does stops the run instead of leaving it waiting.
    #[cfg(unix)]
    #[test]
    fn rank_0_waits_for_its_persistent_workers() {
        let dir = std::env::temp_dir().join(format!("rlx-ready-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let d = dir.to_str().unwrap();
        let worker = Launch {
            rank: 1,
            host: "local".into(),
            label: "w".into(),
            command: format!(
                "sleep 1; touch {d}/up; echo '{WORKER_READY}'; \
                 while [ ! -f {d}/done ]; do sleep 0.05; done"
            ),
            local: true,
            persistent: true,
            stop: None,
        };
        let rank0 = Launch {
            rank: 0,
            label: "r0".into(),
            command: format!("[ -f {d}/up ] && touch {d}/done"),
            persistent: false,
            ..worker.clone()
        };
        let codes = run_once(
            &[rank0.clone(), worker.clone()],
            Duration::from_secs(30),
            true,
        );
        assert_eq!(codes, vec![Some(0), Some(0)]);
        let dead = Launch {
            command: "exit 4".into(),
            ..worker
        };
        let t = Instant::now();
        let codes = run_once(&[rank0, dead], Duration::from_secs(30), true);
        assert_eq!(codes, vec![None, Some(4)], "rank 0 never started");
        assert!(t.elapsed() < Duration::from_secs(5));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
