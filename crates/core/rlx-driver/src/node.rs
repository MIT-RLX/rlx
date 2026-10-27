// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
use crate::rendezvous::RendezvousProto;
use crate::transport::ProcessGroup;
use std::collections::BTreeMap;
use std::io;
use std::net::{
    IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV6, TcpListener, ToSocketAddrs, UdpSocket,
};
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
    Discover {
        disc_port: u16,
        data_base: u16,
    },
    /// Learn every address from rank 0 over TCP or UDP ([`crate::rendezvous`]).
    Rendezvous {
        at: String,
        proto: RendezvousProto,
    },
}

/// What the collectives ride on once the ranks know each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NodeTransport {
    /// TCP streams (the default; zero dependencies).
    #[default]
    Tcp,
    /// QUIC over UDP between the same addresses, straight on the LAN (no
    /// relays) — encrypted, multiplexed, and through networks that filter
    /// new TCP connections. Needs `rlx-driver`'s `iroh` feature.
    Quic,
}

/// Rank value [`Node::rank`] reports until a rendezvous assigns one.
pub const RANK_UNASSIGNED: u32 = u32::MAX;

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
    transport: NodeTransport,
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
            transport: NodeTransport::Tcp,
        }
    }

    /// A worker whose rank (and world size) the rendezvous assigns — start N
    /// identical processes with only the coordinator's address. Pair with
    /// [`Self::rendezvous`].
    pub fn auto() -> Self {
        Self::new(RANK_UNASSIGNED, 0)
    }

    /// Learn every rank's address from rank 0 at `at` (`tcp://host:port`,
    /// `udp://host:port`, or bare `host:port` = TCP). Rank 0 serves there and
    /// must know `world`; workers may be [`Self::auto`].
    pub fn rendezvous(mut self, at: &str) -> Self {
        let (proto, at) = crate::rendezvous::parse_rendezvous(at);
        self.peers = PeerSpec::Rendezvous { at, proto };
        self
    }

    /// Carry the collectives over `t` ([`NodeTransport::Tcp`] by default).
    pub fn transport(mut self, t: NodeTransport) -> Self {
        self.transport = t;
        self
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
    pub fn peers<A: ToSocketAddrs>(
        mut self,
        addrs: impl IntoIterator<Item = A>,
    ) -> io::Result<Self> {
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

    /// Discover peers by UDP broadcast + IPv6 multicast (zero-config on a
    /// shared LAN; see [`discover_peers`]): each node announces
    /// `rank → ip:(data_base+rank)` on `disc_port` until all `world` addresses
    /// are known.
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
    ///   `DISCOVER=1` (+ `DISC_PORT`/`DATA_PORT`) — UDP auto-discovery, or
    ///   `RENDEZVOUS=tcp://host:port` / `udp://…` — every address from rank 0
    ///                                           (workers: `RANK=auto`, `WORLD` optional);
    ///   `RENDEZVOUS=auto` (`udp://auto`)      — rank 0 announces it on the LAN,
    ///                                           workers need no address at all
    ///   `TRANSPORT=tcp|quic`                  — collectives over TCP or QUIC/UDP
    ///   `TOPOLOGY=mesh|star` (`DIAL_OUT=1` = star)
    ///   `HEAP_MB`                             — per-rank heap (optional)
    pub fn from_env() -> Result<Self, String> {
        let var = |k: &str| rlx_ir::env::read(k).ok();
        let rendezvous = var("RENDEZVOUS");
        let rank: u32 = match var("RANK").as_deref() {
            Some("auto") | None if rendezvous.is_some() => RANK_UNASSIGNED,
            r => r
                .unwrap_or("0")
                .parse()
                .map_err(|_| "RANK must be an integer (or `auto` with RENDEZVOUS)".to_string())?,
        };
        let world: u32 = match var("WORLD") {
            None if rendezvous.is_some() && rank != 0 => 0,
            w => w
                .as_deref()
                .unwrap_or("1")
                .parse()
                .map_err(|_| "WORLD must be an integer".to_string())?,
        };
        let mut node = Node::new(rank, world);
        node.transport = match var("TRANSPORT").as_deref().unwrap_or("tcp") {
            "tcp" => NodeTransport::Tcp,
            "quic" | "udp" => NodeTransport::Quic,
            o => return Err(format!("TRANSPORT={o}: expected tcp or quic")),
        };

        let star =
            var("TOPOLOGY").as_deref() == Some("star") || var("DIAL_OUT").is_some_and(|v| v != "0");
        node = node.topology(if star { Topology::Star } else { Topology::Mesh });

        if let Some(mb) = var("HEAP_MB").and_then(|v| v.parse::<usize>().ok()) {
            node = node.heap_bytes(mb << 20);
        }

        if let Some(at) = rendezvous {
            node = node.rendezvous(&at);
        } else if var("DISCOVER").is_some_and(|v| v != "0") {
            let dp = var("DISC_PORT")
                .and_then(|v| v.parse().ok())
                .unwrap_or(29600);
            let db = var("DATA_PORT")
                .and_then(|v| v.parse().ok())
                .unwrap_or(29500);
            node = node.discover(dp, db);
            if let Some(h) = var("DISCOVER_HOST") {
                node = node.discover_via(h);
            }
        } else {
            let peers = var("PEERS").unwrap_or_else(|| "127.0.0.1:29500,127.0.0.1:29501".into());
            let addrs = peers
                .split(',')
                .map(|s| resolve_addr(s.trim()))
                .collect::<io::Result<Vec<_>>>()
                .map_err(|e| format!("PEERS: {e}"))?;
            node = node.peers(addrs).map_err(|e| format!("PEERS: {e}"))?;
        }
        Ok(node)
    }

    /// Establish the transport and return a ready [`ProcessGroup`].
    pub fn connect(self) -> io::Result<Arc<ProcessGroup>> {
        if let PeerSpec::Rendezvous { at, proto } = &self.peers {
            return self.connect_rendezvous(at, *proto);
        }
        if self.transport == NodeTransport::Quic {
            let PeerSpec::Static(peers) = &self.peers else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "TRANSPORT=quic needs PEERS or RENDEZVOUS (not UDP discovery)",
                ));
            };
            return quic_group(self.rank, self.world, peers);
        }
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
                let listener = bind_any_tcp(data_base)?;
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
            PeerSpec::Rendezvous { .. } => unreachable!("handled at the top of connect"),
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

impl Node {
    fn connect_rendezvous(
        &self,
        at: &str,
        proto: RendezvousProto,
    ) -> io::Result<Arc<ProcessGroup>> {
        use crate::rendezvous::{ANNOUNCE_PORT, announce, find, parse_rendezvous, register, serve};
        let timeout = Duration::from_secs(
            rlx_ir::env::read("RENDEZVOUS_TIMEOUT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300),
        );
        let announce_port: u16 = rlx_ir::env::read("RENDEZVOUS_ANNOUNCE_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(ANNOUNCE_PORT);
        // `auto`: a worker learns the address from rank 0's LAN announcement;
        // rank 0 serves on its default-route address.
        let (proto, at) = match (at, self.rank) {
            ("auto", 0) => (proto, SocketAddr::new(local_ip(), 29400).to_string()),
            ("auto", _) => {
                let url = find(announce_port, timeout)?;
                let (p, a) = parse_rendezvous(&url);
                (p, a)
            }
            _ => (proto, at.to_string()),
        };
        let at = resolve_addr(&at)?;
        let _announcer = (self.rank == 0).then(|| {
            let stop = Arc::new(AtomicBool::new(false));
            let url = format!(
                "{}://{at}",
                if proto == RendezvousProto::Udp {
                    "udp"
                } else {
                    "tcp"
                }
            );
            let st = stop.clone();
            std::thread::spawn(move || announce(&url, announce_port, &st));
            StopOnDrop(stop)
        });
        let quic = self.transport == NodeTransport::Quic;
        // Data endpoint: a TCP listener (kept), or a reserved UDP port that
        // QUIC rebinds once the peer list is known.
        let mut listener: Option<TcpListener> = None;
        let mut bind_data = |ip: IpAddr| -> io::Result<u16> {
            if quic {
                return UdpSocket::bind((ip, 0))?.local_addr().map(|a| a.port());
            }
            let l = TcpListener::bind((ip, 0))?;
            let port = l.local_addr()?.port();
            listener = Some(l);
            Ok(port)
        };
        let (rank, world, peers) = if self.rank == 0 {
            if self.world == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "the rendezvous coordinator (rank 0) needs WORLD",
                ));
            }
            // Rank 0 is reached at the rendezvous host's address.
            let own = SocketAddr::new(at.ip(), bind_data(at.ip())?);
            (0, self.world, serve(at, proto, self.world, own, timeout)?)
        } else {
            let rank = (self.rank != RANK_UNASSIGNED).then_some(self.rank);
            let world = (self.world != 0).then_some(self.world);
            let a = register(at, proto, rank, world, bind_data, timeout)?;
            (a.rank, a.world, a.peers)
        };
        if quic {
            return quic_group(rank, world, &peers);
        }
        let listener = listener.expect("bound above");
        let transport = match self.topology {
            Topology::Mesh => {
                NetTransport::from_listener(rank, world, listener, peers, self.heap_bytes)?
            }
            Topology::Star if rank == 0 => {
                NetTransport::coordinator_listen(world, listener, self.heap_bytes)?
            }
            Topology::Star => NetTransport::worker_dial(rank, world, peers[0], self.heap_bytes)?,
        };
        Ok(Arc::new(ProcessGroup::new(Arc::new(transport))))
    }
}

/// Sets its flag when dropped (stops a background announcer).
struct StopOnDrop(Arc<AtomicBool>);
impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// QUIC/UDP group over known addresses. Keys come from `RLX_IROH_SEED` (hex)
/// when set, else from the peer list itself — enough to keep two jobs on one
/// LAN apart, not to keep an attacker out (anyone who sees the list can
/// derive the keys); set a secret seed on an untrusted network.
#[cfg(feature = "iroh")]
fn quic_group(rank: u32, world: u32, peers: &[SocketAddr]) -> io::Result<Arc<ProcessGroup>> {
    let seed: Vec<u8> = match rlx_ir::env::read("RLX_IROH_SEED") {
        Ok(hex) => (0..hex.len() / 2)
            .filter_map(|i| u8::from_str_radix(hex.get(2 * i..2 * i + 2)?, 16).ok())
            .collect(),
        Err(_) => {
            let list: Vec<String> = peers.iter().map(SocketAddr::to_string).collect();
            format!("rlx-lan:{}", list.join(",")).into_bytes()
        }
    };
    let t = crate::iroh_transport::IrohTransport::connect_lan(
        rank,
        world,
        &seed,
        peers,
        crate::iroh_transport::RLX_PIPELINE_ALPN,
    )
    .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(Arc::new(ProcessGroup::new(Arc::new(t))))
}

#[cfg(not(feature = "iroh"))]
fn quic_group(_rank: u32, _world: u32, _peers: &[SocketAddr]) -> io::Result<Arc<ProcessGroup>> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "TRANSPORT=quic needs rlx-driver built with the `iroh` feature",
    ))
}

/// Resolve one peer address. Beyond what `ToSocketAddrs` takes (`ip:port`,
/// `host:port`, `[v6]:port`, `[fe80::1%2]:port`), this accepts a link-local
/// IPv6 scoped by interface **name** — `[fe80::1%en0]:port` — which std's
/// literal parser rejects; the bracketed host is handed to the system resolver,
/// which maps the name to its index.
pub fn resolve_addr(s: &str) -> io::Result<SocketAddr> {
    if let Ok(sa) = s.parse::<SocketAddr>() {
        return Ok(sa);
    }
    let no_addr = || {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{s:?} resolved to no address"),
        )
    };
    if let Some(rest) = s.strip_prefix('[')
        && let Some((host, port)) = rest.split_once("]:")
    {
        let port: u16 = port.parse().map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("bad port in {s:?}"))
        })?;
        return (host, port).to_socket_addrs()?.next().ok_or_else(no_addr);
    }
    s.to_socket_addrs()?.next().ok_or_else(no_addr)
}

/// The unspecified address of `addr`'s family — what to bind a socket that
/// talks to `addr`.
fn unspecified_for(addr: &SocketAddr) -> IpAddr {
    match addr {
        SocketAddr::V4(_) => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        SocketAddr::V6(_) => IpAddr::V6(Ipv6Addr::UNSPECIFIED),
    }
}

/// Bind `port` on every local address, IPv6 and IPv4 alike. `[::]` is
/// dual-stack on Linux/macOS (v4 peers arrive as `::ffff:a.b.c.d`); Windows
/// sockets default to v6-only, so they keep the IPv4 wildcard there. Falls back
/// to `0.0.0.0` on a host with IPv6 disabled.
fn bind_any<T>(port: u16, bind: impl Fn(SocketAddr) -> io::Result<T>) -> io::Result<T> {
    let v4 = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port);
    if cfg!(windows) {
        return bind(v4);
    }
    bind(SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), port)).or_else(|_| bind(v4))
}

fn bind_any_tcp(port: u16) -> io::Result<TcpListener> {
    bind_any(port, TcpListener::bind)
}

/// The default-route local IP (picks the outbound interface): IPv4 first, then
/// IPv6 for a v6-only host; loopback if both probes fail.
pub fn local_ip() -> IpAddr {
    // No packet is sent — `connect` on UDP just selects the outbound iface.
    let probe = |bind: &str, dst: &str| {
        UdpSocket::bind(bind)
            .and_then(|s| {
                s.connect(dst)?;
                s.local_addr()
            })
            .map(|a| a.ip())
    };
    probe("0.0.0.0:0", "8.8.8.8:80")
        .or_else(|_| probe("[::]:0", "[2001:4860:4860::8888]:80"))
        .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST))
}

/// Link-local IPv6 multicast group the mesh rendezvous announces on (IPv6 has
/// no broadcast). `ff02::` keeps it on the local link, like the IPv4 broadcast.
pub const DISC_GROUP_V6: Ipv6Addr = Ipv6Addr::new(0xff02, 0, 0, 0, 0, 0, 0, 0x7266);

/// Interface indices tried for the multicast join/send. std cannot enumerate
/// interfaces, and macOS refuses a link-local multicast send without one
/// ("No route to host"), so every plausible index is tried; absent ones fail
/// harmlessly. Also reaches every link on a multi-homed host.
const MAX_IFINDEX: u32 = 64;

/// Send `msg` to [`DISC_GROUP_V6`]`:port` on each of `ifaces`; returns the
/// indices the send went out on.
fn multicast_v6(
    sock: &UdpSocket,
    msg: &[u8],
    port: u16,
    ifaces: impl IntoIterator<Item = u32>,
) -> Vec<u32> {
    ifaces
        .into_iter()
        .filter(|&i| {
            sock.send_to(msg, SocketAddrV6::new(DISC_GROUP_V6, port, 0, i))
                .is_ok()
        })
        .collect()
}

/// Announce period. Paced on a timer, not per received packet: multicast loops
/// our own sends back, so announce-on-receive feeds itself into a storm that
/// Wi-Fi drops broadcast/multicast frames under.
const DISC_PERIOD: Duration = Duration::from_millis(100);
/// How long a rank that knows everyone keeps announcing for peers that do not
/// yet (or that never report `done` — pre-flag nodes).
const DISC_LINGER: Duration = Duration::from_secs(3);

/// UDP rendezvous: announce `rank → ip:(data_base+rank)` on `disc_port` and
/// collect until all `world` addresses are known; return them sorted by rank.
/// Zero-config peer discovery on a shared LAN. Announces go out as an IPv4
/// broadcast **and** an IPv6 multicast to [`DISC_GROUP_V6`], and one dual-stack
/// socket hears both, so a mix of v4 and v6-only nodes still find each other.
///
/// Wire: `RLXDISC <rank> <addr> <done>`; `done = 1` once the sender knows every
/// rank. A rank that is done keeps announcing until every peer says `done`
/// (at most 3 s), so a peer that lost the last announcements
/// still completes. Older nodes send no flag and still interoperate.
pub fn discover_peers(rank: u32, world: u32, disc_port: u16, data_base: u16) -> Vec<SocketAddr> {
    use std::collections::BTreeSet;
    use std::time::Instant;

    let my_addr = SocketAddr::new(local_ip(), data_base + rank as u16);
    let sock = bind_any(disc_port, UdpSocket::bind).expect("discovery bind");
    let v6 = sock.local_addr().is_ok_and(|a| a.is_ipv6());
    if v6 {
        for i in 0..=MAX_IFINDEX {
            let _ = sock.join_multicast_v6(&DISC_GROUP_V6, i);
        }
    }
    sock.set_read_timeout(Some(DISC_PERIOD / 2)).ok();
    // Separate IPv4 send socket: the receiver may be IPv6 (dual-stack).
    let bsock = UdpSocket::bind("0.0.0.0:0").ok();
    if let Some(b) = &bsock {
        b.set_broadcast(true).ok();
    }
    let bcast = SocketAddr::new(IpAddr::V4(Ipv4Addr::BROADCAST), disc_port);
    // Probe every index once, then keep only the interfaces that sent.
    let mut ifaces: Option<Vec<u32>> = None;
    let mut announce = |done: bool| {
        let msg = format!("RLXDISC {rank} {my_addr} {}", u8::from(done));
        if let Some(b) = &bsock {
            let _ = b.send_to(msg.as_bytes(), bcast);
        }
        if v6 {
            let sent = match &ifaces {
                Some(v) => multicast_v6(&sock, msg.as_bytes(), disc_port, v.iter().copied()),
                None => multicast_v6(&sock, msg.as_bytes(), disc_port, 0..=MAX_IFINDEX),
            };
            ifaces.get_or_insert(sent);
        }
    };

    let mut peers: BTreeMap<u32, SocketAddr> = BTreeMap::new();
    let mut done_peers: BTreeSet<u32> = BTreeSet::new();
    peers.insert(rank, my_addr);
    let mut buf = [0u8; 256];
    let mut next = Instant::now();
    let mut complete_at: Option<Instant> = None;
    loop {
        let complete = peers.len() as u32 >= world;
        if complete {
            let t = *complete_at.get_or_insert_with(Instant::now);
            if done_peers.len() as u32 >= world - 1 || t.elapsed() >= DISC_LINGER {
                break;
            }
        }
        if Instant::now() >= next {
            announce(complete);
            next = Instant::now() + DISC_PERIOD;
        }
        if let Ok((n, _)) = sock.recv_from(&mut buf)
            && let Ok(s) = std::str::from_utf8(&buf[..n])
        {
            let mut it = s.split_whitespace();
            if it.next() == Some("RLXDISC")
                && let (Some(r), Some(a)) = (it.next(), it.next())
                && let (Ok(r), Ok(a)) = (r.parse::<u32>(), a.parse::<SocketAddr>())
            {
                peers.insert(r, a);
                if r != rank && it.next() == Some("1") {
                    done_peers.insert(r);
                }
            }
        }
    }
    announce(true); // one last for a peer that is mid-recv
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
    let Ok(sock) = bind_any(disc_port, UdpSocket::bind) else {
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
    match host {
        Some(h) => {
            // `h` is a bare host (`10.0.0.2`, `fe80::1%en0`, `coord.local`);
            // bracket it so `resolve_addr` reads an IPv6 literal's colons right.
            let target = resolve_addr(&format!("[{}]:{disc_port}", h.trim_matches(['[', ']'])))?;
            let sock = UdpSocket::bind((unspecified_for(&target), 0))?;
            sock.set_read_timeout(Some(Duration::from_millis(300))).ok();
            loop {
                sock.send_to(b"RLXQ", target)?;
                if let Ok((n, _)) = sock.recv_from(&mut buf)
                    && let Some(port) = parse_rlxport(&buf[..n])
                {
                    return Ok(SocketAddr::new(target.ip(), port));
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

    /// The loopback interface's name, for a scope-by-name link-local address.
    fn loopback_ifname() -> &'static str {
        if cfg!(target_os = "linux") {
            "lo"
        } else {
            "lo0"
        }
    }

    #[test]
    fn resolve_addr_ipv4_ipv6_and_scoped_link_local() {
        let v4 = resolve_addr("10.0.0.2:29500").unwrap();
        assert_eq!(v4, "10.0.0.2:29500".parse().unwrap());
        let v6 = resolve_addr("[2001:db8::7]:29501").unwrap();
        assert_eq!(v6, "[2001:db8::7]:29501".parse().unwrap());
        // Numeric scope id: std's literal parser already handles it.
        let SocketAddr::V6(n) = resolve_addr("[fe80::1%2]:29502").unwrap() else {
            panic!("expected v6");
        };
        assert_eq!((n.scope_id(), n.port()), (2, 29502));
        // A hostname resolves through the system resolver.
        assert_eq!(resolve_addr("localhost:7").unwrap().port(), 7);
        assert!(resolve_addr("[::1]:notaport").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn resolve_addr_scope_by_interface_name() {
        // `%en0` / `%enp2s0` is how `ip addr` / `ifconfig` print link-local
        // addresses; std rejects it, the resolver maps it to the ifindex.
        let s = format!("[fe80::1%{}]:29500", loopback_ifname());
        let SocketAddr::V6(a) = resolve_addr(&s).unwrap() else {
            panic!("expected v6");
        };
        assert_ne!(a.scope_id(), 0, "interface name must map to its index");
        assert_eq!(a.port(), 29500);
    }

    /// Free port on IPv6 loopback; `None` on a host with IPv6 disabled.
    fn free_v6() -> Option<SocketAddr> {
        TcpListener::bind((Ipv6Addr::LOCALHOST, 0))
            .ok()
            .map(|l| l.local_addr().unwrap())
    }

    #[test]
    fn node_mesh_over_ipv6() {
        let (Some(a0), Some(a1)) = (free_v6(), free_v6()) else {
            eprintln!("no IPv6 loopback — skipping");
            return;
        };
        let peers = vec![a0.to_string(), a1.to_string()]; // "[::1]:port"
        let p1 = peers.clone();
        let h = std::thread::spawn(move || {
            let addrs: Vec<_> = p1.iter().map(|p| resolve_addr(p).unwrap()).collect();
            let g = Node::new(1, 2).peers(addrs).unwrap().connect().unwrap();
            let mut d = vec![2.0f32; 3];
            g.all_reduce(&mut d, ReduceKind::Sum).unwrap();
            assert_eq!(d, vec![3.0; 3]);
            g.barrier().unwrap();
        });
        let addrs: Vec<_> = peers.iter().map(|p| resolve_addr(p).unwrap()).collect();
        let g = Node::new(0, 2).peers(addrs).unwrap().connect().unwrap();
        let mut d = vec![1.0f32; 3];
        g.all_reduce(&mut d, ReduceKind::Sum).unwrap();
        assert_eq!(d, vec![3.0; 3]);
        g.barrier().unwrap();
        h.join().unwrap();
    }

    /// Rank 0 + `world-1` identical `auto` workers, all-reduce over the result.
    fn rendezvous_group(at: &str, world: u32, t: NodeTransport, topo: Topology) {
        let workers: Vec<_> = (1..world)
            .map(|_| {
                let at = at.to_string();
                std::thread::spawn(move || {
                    let g = Node::auto()
                        .rendezvous(&at)
                        .transport(t)
                        .topology(topo)
                        .connect()
                        .unwrap();
                    assert_eq!(g.world_size(), world);
                    let mut d = vec![g.rank() as f32; 4];
                    g.all_reduce(&mut d, ReduceKind::Sum).unwrap();
                    g.barrier().unwrap();
                    (g.rank(), d)
                })
            })
            .collect();
        let g = Node::new(0, world)
            .rendezvous(at)
            .transport(t)
            .topology(topo)
            .connect()
            .unwrap();
        let mut d = vec![0.0f32; 4];
        g.all_reduce(&mut d, ReduceKind::Sum).unwrap();
        g.barrier().unwrap();
        let want = (0..world).sum::<u32>() as f32;
        assert_eq!(d, vec![want; 4]);
        let mut ranks: Vec<u32> = workers
            .into_iter()
            .map(|w| {
                let (r, d) = w.join().unwrap();
                assert_eq!(d, vec![want; 4]);
                r
            })
            .collect();
        ranks.sort();
        assert_eq!(ranks, (1..world).collect::<Vec<_>>());
    }

    fn free_tcp() -> u16 {
        TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    #[test]
    fn rendezvous_over_tcp_and_udp_builds_a_mesh_of_auto_ranks() {
        rendezvous_group(
            &format!("tcp://127.0.0.1:{}", free_tcp()),
            4,
            NodeTransport::Tcp,
            Topology::Mesh,
        );
        rendezvous_group(
            &format!("udp://127.0.0.1:{}", free_tcp()),
            3,
            NodeTransport::Tcp,
            Topology::Mesh,
        );
    }

    #[test]
    fn rendezvous_star() {
        // world 2: a star has no worker↔worker links for a wider ring.
        rendezvous_group(
            &format!("127.0.0.1:{}", free_tcp()),
            2,
            NodeTransport::Tcp,
            Topology::Star,
        );
    }

    #[cfg(feature = "iroh")]
    #[test]
    fn quic_over_udp_from_static_peers_and_from_rendezvous() {
        // Static: three fixed UDP ports on loopback.
        let ports: Vec<u16> = (0..3)
            .map(|_| {
                UdpSocket::bind("127.0.0.1:0")
                    .unwrap()
                    .local_addr()
                    .unwrap()
                    .port()
            })
            .collect();
        let peers: Vec<SocketAddr> = ports
            .iter()
            .map(|&p| SocketAddr::new(Ipv4Addr::LOCALHOST.into(), p))
            .collect();
        let hs: Vec<_> = (0..3u32)
            .map(|r| {
                let peers = peers.clone();
                std::thread::spawn(move || {
                    let g = Node::new(r, 3)
                        .peers(peers)
                        .unwrap()
                        .transport(NodeTransport::Quic)
                        .connect()
                        .unwrap();
                    let mut d = vec![1.0f32; 3];
                    g.all_reduce(&mut d, ReduceKind::Sum).unwrap();
                    g.barrier().unwrap();
                    d
                })
            })
            .collect();
        for h in hs {
            assert_eq!(h.join().unwrap(), vec![3.0; 3]);
        }
        rendezvous_group(
            &format!("udp://127.0.0.1:{}", free_tcp()),
            3,
            NodeTransport::Quic,
            Topology::Mesh,
        );
    }

    /// Mesh rendezvous hears both families on its one dual-stack socket: rank 1
    /// announces ONLY by IPv6 multicast, rank 2 ONLY over IPv4.
    #[cfg(not(windows))]
    #[test]
    fn discover_peers_hears_ipv6_multicast_and_ipv4() {
        let Some(port) = UdpSocket::bind((Ipv6Addr::UNSPECIFIED, 0))
            .ok()
            .map(|s| s.local_addr().unwrap().port())
        else {
            eprintln!("no IPv6 — skipping");
            return;
        };
        let h = std::thread::spawn(move || discover_peers(0, 3, port, 41000));
        let r1: SocketAddr = "[2001:db8::1]:41001".parse().unwrap();
        let r2: SocketAddr = "10.9.8.7:41002".parse().unwrap();
        let v6 = UdpSocket::bind((Ipv6Addr::UNSPECIFIED, 0)).unwrap();
        let v4 = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !h.is_finished() {
            assert!(
                std::time::Instant::now() < deadline,
                "rendezvous never heard both announcements"
            );
            multicast_v6(
                &v6,
                format!("RLXDISC 1 {r1} 1").as_bytes(),
                port,
                0..=MAX_IFINDEX,
            );
            let _ = v4.send_to(
                format!("RLXDISC 2 {r2} 1").as_bytes(),
                (Ipv4Addr::LOCALHOST, port),
            );
            std::thread::sleep(Duration::from_millis(50));
        }
        let peers = h.join().unwrap();
        assert_eq!((peers[1], peers[2]), (r1, r2));
    }

    /// Star discovery with the coordinator reached by an IPv6 literal: the
    /// responder and data listener bind dual-stack, the worker's query socket
    /// matches the target's family.
    #[cfg(not(windows))]
    #[test]
    fn star_discovery_via_ipv6_host() {
        let (Some(d), Some(p)) = (free_v6(), free_v6()) else {
            eprintln!("no IPv6 loopback — skipping");
            return;
        };
        let (disc_port, data_port) = (d.port(), p.port());
        let coord = std::thread::spawn(move || {
            let g = Node::new(0, 2)
                .topology(Topology::Star)
                .discover(disc_port, data_port)
                .connect()
                .unwrap();
            let mut x = vec![1.0f32; 2];
            g.all_reduce(&mut x, ReduceKind::Sum).unwrap();
            assert_eq!(x, vec![3.0; 2]);
        });
        let g = Node::new(1, 2)
            .topology(Topology::Star)
            .discover(disc_port, data_port)
            .discover_via("::1")
            .connect()
            .unwrap();
        let mut x = vec![2.0f32; 2];
        g.all_reduce(&mut x, ReduceKind::Sum).unwrap();
        assert_eq!(x, vec![3.0; 2]);
        coord.join().unwrap();
    }
}
