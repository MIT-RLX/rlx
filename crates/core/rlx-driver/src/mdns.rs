// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Nodes and rendezvous servers advertised over mDNS / DNS-SD — Bonjour on
//! macOS and iOS, Avahi on Linux, multicast `224.0.0.251:5353`.
//!
//! - `_rlx-node._tcp`: a machine that can be a rank says so — its name, the
//!   port it takes shard traffic on, and what it is (`platform`, `devices`,
//!   `mem_gb`, `version`, as TXT records). [`advertise_node`] /
//!   [`browse_nodes`]; `cargo rlx nodes` lists them, and a cluster file can
//!   name a host by advertised name instead of address.
//! - `_rlx-rdv._tcp`: a running rendezvous server, its URL in TXT `url`.
//!   [`crate::rendezvous::announce`] advertises it and
//!   [`crate::rendezvous::find`] browses for it alongside the UDP announce,
//!   so `RENDEZVOUS=auto` works wherever Bonjour/Avahi do.
//!
//! An iOS app should advertise and browse through Network.framework
//! (`NWListener` / `NWBrowser`) with the service types in `NSBonjourServices`:
//! raw multicast sockets there need Apple's multicast entitlement, Bonjour
//! does not. The records are the same either way.

use std::collections::BTreeMap;
use std::io;
use std::net::IpAddr;
use std::time::{Duration, Instant};

use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};

/// DNS-SD type of a node.
pub const NODE_SERVICE: &str = "_rlx-node._tcp.local.";
/// DNS-SD type of a rendezvous server.
pub const RENDEZVOUS_SERVICE: &str = "_rlx-rdv._tcp.local.";

/// What a node says about itself.
#[derive(Debug, Clone, Default)]
pub struct NodeAdvert {
    /// Instance name — unique on the LAN (a hostname, "Yellow iPad").
    pub name: String,
    /// The port it takes shard traffic on.
    pub port: u16,
    /// TXT records: `platform`, `devices` (comma list), `mem_gb`, `version`, …
    pub props: BTreeMap<String, String>,
}

/// A node found on the LAN.
#[derive(Debug, Clone, PartialEq)]
pub struct FoundNode {
    pub name: String,
    pub addrs: Vec<IpAddr>,
    pub port: u16,
    pub props: BTreeMap<String, String>,
}

impl FoundNode {
    /// The address to dial, from this machine's point of view: one on this
    /// machine's own IPv4 subnet ([`crate::node::local_ip`], /24) first, then
    /// a private (RFC 1918) IPv4, any other routable IPv4 (a VPN's CGNAT
    /// `100.64/10` among them — reachable only over that VPN), a routable
    /// IPv6, and link-local last (`169.254/16`, `fe80::/10`: a Thunderbolt
    /// bridge or self-assigned interface, reaching nobody else on the LAN).
    pub fn addr(&self) -> Option<IpAddr> {
        self.addr_near(crate::node::local_ip())
    }

    /// [`Self::addr`] relative to `here`.
    pub fn addr_near(&self, here: IpAddr) -> Option<IpAddr> {
        let rank = |a: &IpAddr| match (a, here) {
            (IpAddr::V4(v), _) if is_link_local(a) || v.is_loopback() => 5,
            (IpAddr::V4(v), IpAddr::V4(h)) if v.octets()[..3] == h.octets()[..3] => 0,
            (IpAddr::V4(v), _) if v.is_private() => 1,
            (IpAddr::V4(_), _) => 2,
            (IpAddr::V6(v), _) if v.is_loopback() || is_link_local(a) => 6,
            (IpAddr::V6(_), _) => 3,
        };
        self.addrs.iter().min_by_key(|a| rank(a)).copied()
    }
}

fn is_link_local(a: &IpAddr) -> bool {
    match a {
        IpAddr::V4(v) => v.is_link_local(),
        IpAddr::V6(v) => (v.segments()[0] & 0xffc0) == 0xfe80,
    }
}

/// Keeps a registration on the network; dropping it withdraws the record.
pub struct Advertisement {
    daemon: ServiceDaemon,
    fullname: String,
}

impl Drop for Advertisement {
    fn drop(&mut self) {
        if let Ok(rx) = self.daemon.unregister(&self.fullname) {
            // Wait briefly so the goodbye packet goes out before shutdown.
            let _ = rx.recv_timeout(Duration::from_millis(500));
        }
        let _ = self.daemon.shutdown();
    }
}

fn mdns_err(e: mdns_sd::Error) -> io::Error {
    io::Error::other(format!("mDNS: {e}"))
}

/// DNS-SD instance names may not contain dots; hostnames as names are fine
/// once those become dashes.
fn instance(name: &str) -> String {
    let n: String = name
        .chars()
        .map(|c| if c == '.' { '-' } else { c })
        .collect();
    if n.is_empty() { "rlx-node".into() } else { n }
}

/// The `.local.` host record a registration publishes its addresses under.
/// Never the machine's own hostname: Avahi and mDNSResponder already own
/// `<hostname>.local`, and a second responder claiming it is a name conflict
/// they settle by renaming the machine (`msi` → `msi-2`, for every user of
/// it). An `rlx-` prefix and the process id keep it unique.
fn host_label(instance: &str, pid: u32) -> String {
    let base: String = instance
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    format!("rlx-{}-{pid}.local.", base.trim_matches('-'))
}

fn register(
    service: &str,
    name: &str,
    port: u16,
    props: &BTreeMap<String, String>,
) -> io::Result<Advertisement> {
    let daemon = ServiceDaemon::new().map_err(mdns_err)?;
    let inst = instance(name);
    // The host record: the instance name as a `.local.` host, every
    // interface's address published (`enable_addr_auto`).
    let host = host_label(&inst, std::process::id());
    let props: Vec<(&str, &str)> = props
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let info = ServiceInfo::new(service, &inst, &host, "", port, &props[..])
        .map_err(mdns_err)?
        .enable_addr_auto();
    let fullname = info.get_fullname().to_string();
    daemon.register(info).map_err(mdns_err)?;
    Ok(Advertisement { daemon, fullname })
}

/// Advertise this machine as a node until the returned value is dropped.
pub fn advertise_node(ad: &NodeAdvert) -> io::Result<Advertisement> {
    let mut props = ad.props.clone();
    props
        .entry("version".into())
        .or_insert_with(|| env!("CARGO_PKG_VERSION").into());
    register(NODE_SERVICE, &ad.name, ad.port, &props)
}

/// Advertise a rendezvous server (`url` = `tcp://ip:port` / `udp://…`).
pub fn advertise_rendezvous(url: &str, port: u16) -> io::Result<Advertisement> {
    let name = format!("rlx-rdv-{}", std::process::id());
    let props = BTreeMap::from([("url".to_string(), url.to_string())]);
    register(RENDEZVOUS_SERVICE, &name, port, &props)
}

fn browse(
    service: &str,
    timeout: Duration,
    mut done: impl FnMut(&[FoundNode]) -> bool,
) -> io::Result<Vec<FoundNode>> {
    let daemon = ServiceDaemon::new().map_err(mdns_err)?;
    let rx = daemon.browse(service).map_err(mdns_err)?;
    let deadline = Instant::now() + timeout;
    let mut found: Vec<FoundNode> = Vec::new();
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        let Ok(ev) = rx.recv_timeout(left) else { break };
        if let ServiceEvent::ServiceResolved(info) = ev {
            let name = info
                .get_fullname()
                .strip_suffix(&format!(".{service}"))
                .unwrap_or(info.get_fullname())
                .to_string();
            let node = FoundNode {
                name,
                addrs: info.get_addresses().iter().copied().collect(),
                port: info.get_port(),
                props: info
                    .get_properties()
                    .iter()
                    .map(|p| (p.key().to_string(), p.val_str().to_string()))
                    .collect(),
            };
            // One resolution per interface (Wi-Fi, a USB link, a VPN), each
            // carrying that interface's addresses: keep them all, so `addr`
            // can pick the reachable one.
            match found.iter_mut().find(|f| f.name == node.name) {
                Some(f) => {
                    for a in node.addrs {
                        if !f.addrs.contains(&a) {
                            f.addrs.push(a);
                        }
                    }
                    f.port = node.port;
                    f.props.extend(node.props);
                }
                None => found.push(node),
            }
            if done(&found) {
                break;
            }
        }
    }
    let _ = daemon.shutdown();
    found.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(found)
}

/// Every node that answers within `timeout`.
pub fn browse_nodes(timeout: Duration) -> io::Result<Vec<FoundNode>> {
    browse(NODE_SERVICE, timeout, |_| false)
}

/// The node advertised as `name` (case-insensitive), waiting up to `timeout`.
pub fn find_node(name: &str, timeout: Duration) -> io::Result<FoundNode> {
    let want = instance(name).to_lowercase();
    // Stop early only once the node has an address on this machine's own
    // subnet: its records arrive once per interface, in no set order, and a
    // link-local (USB, Thunderbolt) or IPv6 one often lands first. Otherwise
    // `addr` picks the best of what the deadline leaves.
    let here = crate::node::local_ip();
    let same_subnet = |a: &IpAddr| match (a, here) {
        (IpAddr::V4(v), IpAddr::V4(h)) => v.octets()[..3] == h.octets()[..3],
        _ => false,
    };
    browse(NODE_SERVICE, timeout, |f| {
        f.iter()
            .any(|n| n.name.to_lowercase() == want && n.addrs.iter().any(same_subnet))
    })?
    .into_iter()
    .find(|n| n.name.to_lowercase() == want)
    .ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("mDNS: no node advertised as {name:?} on the LAN"),
        )
    })
}

/// The URL of the first rendezvous server that answers within `timeout`.
pub fn find_rendezvous(timeout: Duration) -> io::Result<String> {
    browse(RENDEZVOUS_SERVICE, timeout, |f| {
        f.iter().any(|n| n.props.contains_key("url"))
    })?
    .into_iter()
    .find_map(|n| n.props.get("url").cloned())
    .ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "mDNS: no _rlx-rdv._tcp rendezvous on the LAN",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_routable_address_beats_a_link_local_one() {
        let n = |addrs: &[&str]| FoundNode {
            name: "n".into(),
            addrs: addrs.iter().map(|a| a.parse().unwrap()).collect(),
            port: 1,
            props: BTreeMap::new(),
        };
        let here: IpAddr = "10.0.0.172".parse().unwrap();
        let pick = |a: &[&str]| n(a).addr_near(here).unwrap().to_string();
        assert_eq!(pick(&["169.254.22.105", "10.0.0.185"]), "10.0.0.185");
        // A VPN's CGNAT address loses to the LAN one (the mm4 case).
        assert_eq!(pick(&["100.96.11.22", "10.0.0.185"]), "10.0.0.185");
        assert_eq!(pick(&["100.96.11.22", "192.168.1.4"]), "192.168.1.4");
        assert_eq!(pick(&["fe80::1", "2601::5", "169.254.1.1"]), "2601::5");
        assert_eq!(pick(&["fe80::1", "169.254.1.1"]), "169.254.1.1");
        assert!(n(&[]).addr_near(here).is_none());
    }

    #[test]
    fn the_host_record_never_claims_the_machines_own_name() {
        // `NAME=msi` on the machine called msi: publishing `msi.local.` made
        // Avahi rename the host to `msi-2` (and macOS `mm4` to `mm4-2`).
        let h = host_label("msi", 4242);
        assert_eq!(h, "rlx-msi-4242.local.");
        assert_ne!(h, "msi.local.");
        assert_eq!(host_label("Yellow iPad", 7), "rlx-yellow-ipad-7.local.");
    }

    #[test]
    fn instance_names_lose_their_dots() {
        assert_eq!(instance("mac.local"), "mac-local");
        assert_eq!(instance(""), "rlx-node");
        assert_eq!(instance("Yellow iPad"), "Yellow iPad");
    }

    /// A node advertised here is found by name, with its TXT records. Uses the
    /// real multicast group, so it needs a network interface that carries it.
    #[test]
    fn an_advertised_node_is_found_with_its_records() {
        let name = format!("rlx-test-{}", std::process::id());
        let ad = NodeAdvert {
            name: name.clone(),
            port: 29777,
            props: BTreeMap::from([
                ("platform".into(), "test".into()),
                ("devices".into(), "cpu,metal".into()),
            ]),
        };
        let _guard = match advertise_node(&ad) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipped: cannot advertise here ({e})");
                return;
            }
        };
        match find_node(&name, Duration::from_secs(8)) {
            Ok(n) => {
                assert_eq!(n.port, 29777);
                assert_eq!(
                    n.props.get("devices").map(String::as_str),
                    Some("cpu,metal")
                );
                assert_eq!(n.props.get("platform").map(String::as_str), Some("test"));
                assert!(n.props.contains_key("version"));
                assert!(n.addr().is_some());
            }
            // No multicast on this host (some CI sandboxes).
            Err(e) => eprintln!("skipped: {e}"),
        }
    }
}
