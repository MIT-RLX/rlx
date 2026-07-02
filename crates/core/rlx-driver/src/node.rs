// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! One-call node bring-up.
//!
//! Joining a distributed job used to mean hand-wiring a transport (mesh vs.
//! star), resolving peer addresses, and wrapping the result in an
//! `Arc<ProcessGroup>`. [`Node`] collapses that to a builder:
//!
//! ```no_run
//! # use rlx_driver::{Node, Topology};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // From env (RANK/WORLD/PEERS or DISCOVER/TOPOLOGY):
//! let group = Node::from_env()?.connect()?;
//! // Or explicitly:
//! let group = Node::new(/*rank*/ 1, /*world*/ 2)
//!     .topology(Topology::Star)          // worker dials the coordinator
//!     .peers(["10.0.0.1:29500"])?        // just the coordinator address
//!     .connect()?;
//! let _ = group;
//! # Ok(()) }
//! ```
//!
//! The returned [`ProcessGroup`] carries every collective (`all_reduce`,
//! `all_reduce_typed`, `federated_average`, `broadcast`, …). This is the
//! entry point a model runner or an edge worker calls to get on the mesh.

use crate::net::{DEFAULT_HEAP_BYTES, NetTransport, TcpTransport};
use crate::transport::ProcessGroup;
use std::collections::BTreeMap;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, ToSocketAddrs, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Wire topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    /// Full-mesh TCP — every rank connects to higher ranks and accepts from
    /// lower. Needed for peer-to-peer collectives (ring all-reduce, etc.).
    Mesh,
    /// Star — the coordinator (rank 0) listens; workers dial in and need **no
    /// inbound port**. NAT/mobile/Pi-behind-a-router friendly; the shape for
    /// the coordinator/worker ship-graph model and federated averaging.
    Star,
}

#[derive(Clone, Debug)]
enum PeerSpec {
    Static(Vec<SocketAddr>),
    Discover { disc_port: u16, data_base: u16 },
}

/// Builder that turns minimal config into a connected [`ProcessGroup`].
pub struct Node {
    rank: u32,
    world: u32,
    peers: PeerSpec,
    topology: Topology,
    heap_bytes: usize,
    /// For star discovery from behind NAT (e.g. a Docker/QEMU node): the host
    /// to *unicast* the discovery query to, since broadcast can't cross the
    /// bridge. `None` = LAN broadcast discovery.
    discover_host: Option<String>,
}

impl Node {
    /// A node at `rank` of `world`, full-mesh, default heap. Set peers with
    /// [`Self::peers`] or [`Self::discover`] before [`Self::connect`].
    pub fn new(rank: u32, world: u32) -> Self {
        Self {
            rank,
            world,
            peers: PeerSpec::Static(Vec::new()),
            topology: Topology::Mesh,
            heap_bytes: DEFAULT_HEAP_BYTES,
            discover_host: None,
        }
    }

    /// Unicast the discovery query to `host` instead of broadcasting — for a
    /// worker behind NAT (Docker/QEMU) that can't hear the coordinator's LAN
    /// broadcast. It learns the coordinator's port from the reply and dials
    /// `host:port`. Only used with a discovering [`Topology::Star`] worker.
    pub fn discover_via(mut self, host: impl Into<String>) -> Self {
        self.discover_host = Some(host.into());
        self
    }

    /// Static peer addresses, one per rank (mesh) or just the coordinator
    /// (star, `peers[0]`). Accepts anything `ToSocketAddrs` (host:port or ip:port).
    pub fn peers<A: ToSocketAddrs>(mut self, addrs: impl IntoIterator<Item = A>) -> io::Result<Self> {
        let mut v = Vec::new();
        for a in addrs {
            let sa = a.to_socket_addrs()?.next().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "peer resolved to no address")
            })?;
            v.push(sa);
        }
        self.peers = PeerSpec::Static(v);
        Ok(self)
    }

    /// Discover peers by UDP broadcast (zero-config on a shared LAN): each node
    /// announces `rank → ip:(data_base+rank)` on `disc_port` until all `world`
    /// addresses are known.
    pub fn discover(mut self, disc_port: u16, data_base: u16) -> Self {
        self.peers = PeerSpec::Discover {
            disc_port,
            data_base,
        };
        self
    }

    pub fn topology(mut self, t: Topology) -> Self {
        self.topology = t;
        self
    }

    /// Per-rank symmetric-heap size in bytes (default [`DEFAULT_HEAP_BYTES`]).
    pub fn heap_bytes(mut self, n: usize) -> Self {
        self.heap_bytes = n;
        self
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }
    pub fn world(&self) -> u32 {
        self.world
    }

    /// Read node config from the environment:
    ///   `RANK`, `WORLD`                       — this rank / total ranks
    ///   `PEERS=host:port,host:port,…`         — static addresses, or
    ///   `DISCOVER=1` (+ `DISC_PORT`/`DATA_PORT`) — UDP auto-discovery
    ///   `TOPOLOGY=mesh|star` (`DIAL_OUT=1` = star)
    ///   `HEAP_MB`                             — per-rank heap (optional)
    pub fn from_env() -> Result<Self, String> {
        let var = |k: &str| std::env::var(k).ok();
        let rank: u32 = var("RANK")
            .as_deref()
            .unwrap_or("0")
            .parse()
            .map_err(|_| "RANK must be an integer".to_string())?;
        let world: u32 = var("WORLD")
            .as_deref()
            .unwrap_or("1")
            .parse()
            .map_err(|_| "WORLD must be an integer".to_string())?;
        let mut node = Node::new(rank, world);

        let star = var("TOPOLOGY").as_deref() == Some("star")
            || var("DIAL_OUT").is_some_and(|v| v != "0");
        node = node.topology(if star { Topology::Star } else { Topology::Mesh });

        if let Some(mb) = var("HEAP_MB").and_then(|v| v.parse::<usize>().ok()) {
            node = node.heap_bytes(mb << 20);
        }

        if var("DISCOVER").is_some_and(|v| v != "0") {
            let dp = var("DISC_PORT").and_then(|v| v.parse().ok()).unwrap_or(29600);
            let db = var("DATA_PORT").and_then(|v| v.parse().ok()).unwrap_or(29500);
            node = node.discover(dp, db);
            if let Some(h) = var("DISCOVER_HOST") {
                node = node.discover_via(h);
            }
        } else {
            let peers = var("PEERS").unwrap_or_else(|| "127.0.0.1:29500,127.0.0.1:29501".into());
            let addrs: Vec<String> = peers.split(',').map(|s| s.trim().to_string()).collect();
            node = node
                .peers(addrs.iter().map(String::as_str))
                .map_err(|e| format!("PEERS: {e}"))?;
        }
        Ok(node)
    }

    /// Establish the transport and return a ready [`ProcessGroup`].
    pub fn connect(self) -> io::Result<Arc<ProcessGroup>> {
        // Star + discovery: the coordinator *announces* itself and workers
        // *find* it. Unlike mesh discovery (every rank must hear every other),
        // the coordinator never waits to hear the workers — so a NAT'd node
        // that can't broadcast (Docker/QEMU) doesn't stall the group; it just
        // unicasts its query via `discover_via`.
        if self.topology == Topology::Star
            && let PeerSpec::Discover {
                disc_port,
                data_base,
            } = &self.peers
        {
            let (disc_port, data_base) = (*disc_port, *data_base);
            let transport = if self.rank == 0 {
                let listener = TcpListener::bind(("0.0.0.0", data_base))?;
                let stop = Arc::new(AtomicBool::new(false));
                let ann = stop.clone();
                std::thread::spawn(move || announce_coordinator(data_base, disc_port, &ann));
                #[cfg(feature = "mdns")]
                let _mdns = mdns_advertise(data_base); // advertise while we listen
                let t = NetTransport::coordinator_listen(self.world, listener, self.heap_bytes);
                stop.store(true, Ordering::SeqCst);
                t?
            } else {
                let coord = discover_coordinator(disc_port, self.discover_host.as_deref())?;
                NetTransport::worker_dial(self.rank, self.world, coord, self.heap_bytes)?
            };
            return Ok(Arc::new(ProcessGroup::new(Arc::new(transport))));
        }

        // Otherwise resolve a concrete peer list (mesh discovery collects all).
        let peers = match self.peers {
            PeerSpec::Static(v) => v,
            PeerSpec::Discover {
                disc_port,
                data_base,
            } => discover_peers(self.rank, self.world, disc_port, data_base),
        };
        let transport = match self.topology {
            Topology::Mesh => {
                if peers.len() != self.world as usize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "mesh needs WORLD={} peer addresses, got {}",
                            self.world,
                            peers.len()
                        ),
                    ));
                }
                TcpTransport::bind(self.rank, self.world, peers, self.heap_bytes)?
            }
            Topology::Star => {
                let coord = *peers.first().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "star needs the coordinator address (peers[0])",
                    )
                })?;
                if self.rank == 0 {
                    let listener = TcpListener::bind(coord)?;
                    NetTransport::coordinator_listen(self.world, listener, self.heap_bytes)?
                } else {
                    NetTransport::worker_dial(self.rank, self.world, coord, self.heap_bytes)?
                }
            }
        };
        Ok(Arc::new(ProcessGroup::new(Arc::new(transport))))
    }
}

/// The default-route local IP (picks the outbound interface); loopback if the
/// probe fails.
pub fn local_ip() -> IpAddr {
    UdpSocket::bind("0.0.0.0:0")
        .and_then(|s| {
            s.connect("8.8.8.8:80")?; // no packet sent — just selects the iface
            s.local_addr()
        })
        .map(|a| a.ip())
        .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST))
}

/// UDP-broadcast rendezvous: announce `rank → ip:(data_base+rank)` on
/// `disc_port` and collect until all `world` addresses are known; return them
/// sorted by rank. Zero-config peer discovery on a shared LAN.
pub fn discover_peers(rank: u32, world: u32, disc_port: u16, data_base: u16) -> Vec<SocketAddr> {
    let my_addr = SocketAddr::new(local_ip(), data_base + rank as u16);
    let sock = UdpSocket::bind(("0.0.0.0", disc_port)).expect("discovery bind");
    sock.set_broadcast(true).ok();
    sock.set_read_timeout(Some(Duration::from_millis(150))).ok();
    let bcast = SocketAddr::new(IpAddr::V4(Ipv4Addr::BROADCAST), disc_port);
    let msg = format!("RLXDISC {rank} {my_addr}");

    let mut peers: BTreeMap<u32, SocketAddr> = BTreeMap::new();
    peers.insert(rank, my_addr);
    let mut buf = [0u8; 256];
    while (peers.len() as u32) < world {
        let _ = sock.send_to(msg.as_bytes(), bcast);
        if let Ok((n, _)) = sock.recv_from(&mut buf)
            && let Ok(s) = std::str::from_utf8(&buf[..n])
        {
            let mut it = s.split_whitespace();
            if it.next() == Some("RLXDISC")
                && let (Some(r), Some(a)) = (it.next(), it.next())
                && let (Ok(r), Ok(a)) = (r.parse::<u32>(), a.parse::<SocketAddr>())
            {
                peers.insert(r, a);
            }
        }
    }
    // Brief drain so late joiners still hear our announcement.
    for _ in 0..5 {
        let _ = sock.send_to(msg.as_bytes(), bcast);
    }
    peers.into_values().collect()
}

fn parse_rlxport(b: &[u8]) -> Option<u16> {
    let s = std::str::from_utf8(b).ok()?;
    let mut it = s.split_whitespace();
    if it.next() != Some("RLXPORT") {
        return None;
    }
    it.next()?.parse().ok()
}

/// Coordinator side of star discovery: a **pure unicast responder** on
/// `disc_port` — a worker sends `RLXQ`, we reply `RLXPORT <data_port>`. Runs
/// until `stop`. Deliberately does NOT broadcast on this socket: a coordinator
/// that both beacons and receives on one socket starves incoming queries
/// (its own zero-latency loopback beacons always win the recv race). LAN
/// zero-config is handled by mDNS instead ([`mdns_advertise`]); this responder
/// covers a worker that reaches us by name/IP (LAN, `host.docker.internal`, or
/// a Tailscale MagicDNS name — anything routable).
pub fn announce_coordinator(data_port: u16, disc_port: u16, stop: &AtomicBool) {
    let Ok(sock) = UdpSocket::bind(("0.0.0.0", disc_port)) else {
        return;
    };
    sock.set_read_timeout(Some(Duration::from_millis(200))).ok();
    let msg = format!("RLXPORT {data_port}");
    let mut buf = [0u8; 64];
    while !stop.load(Ordering::SeqCst) {
        if let Ok((n, from)) = sock.recv_from(&mut buf)
            && buf[..n].starts_with(b"RLXQ")
        {
            let _ = sock.send_to(msg.as_bytes(), from); // reply to the querier
        }
    }
}

/// Worker side of star discovery: return the coordinator's dial address.
/// `host = Some(h)` (NAT/Docker) unicasts a query to `h:disc_port` and dials
/// `h:port` from the reply; `None` listens for the LAN broadcast and dials the
/// announcing host's own IP.
pub fn discover_coordinator(disc_port: u16, host: Option<&str>) -> io::Result<SocketAddr> {
    let mut buf = [0u8; 64];
    let no_addr =
        || io::Error::new(io::ErrorKind::InvalidInput, "discover host resolved to no address");
    match host {
        Some(h) => {
            let target = (h, disc_port).to_socket_addrs()?.next().ok_or_else(no_addr)?;
            let sock = UdpSocket::bind(("0.0.0.0", 0))?;
            sock.set_read_timeout(Some(Duration::from_millis(300))).ok();
            loop {
                sock.send_to(b"RLXQ", target)?;
                if let Ok((n, _)) = sock.recv_from(&mut buf)
                    && let Some(port) = parse_rlxport(&buf[..n])
                {
                    return (h, port).to_socket_addrs()?.next().ok_or_else(no_addr);
                }
            }
        }
        None => {
            let _ = disc_port;
            #[cfg(feature = "mdns")]
            {
                mdns_discover(Duration::from_secs(15)).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        "mDNS: no _rlx-coord._udp coordinator found on the LAN",
                    )
                })
            }
            #[cfg(not(feature = "mdns"))]
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "no coordinator address — set DISCOVER_HOST (LAN IP / host.docker.internal / \
                 tailnet name) for unicast discovery, or build with the `mdns` feature for \
                 zero-config LAN discovery",
            ))
        }
    }
}

/// Advertise the coordinator as `_rlx-coord._udp.local.` so LAN workers can
/// browse for it — zero-config, Bonjour/Avahi-compatible, and multicast-based
/// (`224.0.0.251`), which switches/APs forward far more reliably than raw
/// broadcast. Hold the returned daemon alive while listening.
#[cfg(feature = "mdns")]
pub fn mdns_advertise(data_port: u16) -> Option<mdns_sd::ServiceDaemon> {
    use mdns_sd::{ServiceDaemon, ServiceInfo};
    let mdns = ServiceDaemon::new().ok()?;
    let ip = local_ip().to_string();
    let props: &[(&str, &str)] = &[];
    let info = ServiceInfo::new(
        "_rlx-coord._udp.local.",
        "coord",
        "rlx-coord.local.",
        ip.as_str(),
        data_port,
        props,
    )
    .ok()?;
    mdns.register(info).ok()?;
    Some(mdns)
}

/// Browse for a `_rlx-coord._udp.local.` coordinator and return its address.
#[cfg(feature = "mdns")]
pub fn mdns_discover(timeout: Duration) -> Option<SocketAddr> {
    use mdns_sd::{ServiceDaemon, ServiceEvent};
    use std::time::Instant;
    let mdns = ServiceDaemon::new().ok()?;
    let rx = mdns.browse("_rlx-coord._udp.local.").ok()?;
    let deadline = Instant::now() + timeout;
    while let Some(left) = deadline.checked_duration_since(Instant::now())
        && let Ok(ev) = rx.recv_timeout(left)
    {
        if let ServiceEvent::ServiceResolved(info) = ev
            && let Some(ip) = info.get_addresses().iter().next()
        {
            return Some(SocketAddr::new(*ip, info.get_port()));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collective::ReduceKind;

    #[test]
    fn node_mesh_connect_and_all_reduce() {
        // Grab two free loopback ports, then bring up a 2-rank mesh via the
        // builder and check a collective round-trips.
        let free = || {
            let l = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
            l.local_addr().unwrap()
        };
        let (a0, a1) = (free(), free());
        let peers = vec![a0.to_string(), a1.to_string()];

        let p1 = peers.clone();
        let h = std::thread::spawn(move || {
            let g = Node::new(1, 2)
                .peers(p1.iter().map(String::as_str))
                .unwrap()
                .connect()
                .unwrap();
            let mut d = vec![2.0f32; 3];
            g.all_reduce(&mut d, ReduceKind::Sum).unwrap();
            assert_eq!(d, vec![3.0; 3]);
            g.barrier().unwrap();
        });
        let g = Node::new(0, 2)
            .peers(peers.iter().map(String::as_str))
            .unwrap()
            .connect()
            .unwrap();
        let mut d = vec![1.0f32; 3];
        g.all_reduce(&mut d, ReduceKind::Sum).unwrap();
        assert_eq!(d, vec![3.0; 3]); // 1 + 2
        g.barrier().unwrap();
        h.join().unwrap();
    }
}
