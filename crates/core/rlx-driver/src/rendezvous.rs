// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rendezvous: every rank learns every other rank's data address from one
//! known meeting point — rank 0 — over **TCP** or **UDP**, so a job needs only
//! `RENDEZVOUS=tcp://coord:29400` (or `udp://…`) instead of a hand-written
//! `PEERS` list:
//!
//! * workers may set `RANK=auto` (the coordinator hands out ranks in arrival
//!   order) and may omit `WORLD` (the coordinator's is used);
//! * each rank advertises the address its connection to the coordinator came
//!   from, so a multi-homed host advertises the interface that reaches the
//!   cluster, and data ports are ephemeral (no port plan);
//! * TCP for networks that drop UDP, UDP (with retransmit + ack) for ones that
//!   filter new TCP connections or for a coordinator that should hold no
//!   per-worker sockets.
//!
//! Wire (one line each way):
//! `RLXREG <rank|?> <world|?> <ip:port>` → `RLXPEERS <rank> <world> <a0>,<a1>,…`
//! or `RLXERR <reason>`; UDP workers confirm with `RLXACK <rank>`.

use std::collections::BTreeMap;
use std::io::{self, BufRead, BufReader, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream, UdpSocket};
use std::time::{Duration, Instant};

/// Rendezvous transport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendezvousProto {
    Tcp,
    Udp,
}

/// Parse `tcp://host:port`, `udp://host:port` or a bare `host:port` (TCP).
pub fn parse_rendezvous(s: &str) -> (RendezvousProto, String) {
    let s = s.trim();
    if let Some(r) = s.strip_prefix("udp://") {
        (RendezvousProto::Udp, r.to_string())
    } else if let Some(r) = s.strip_prefix("tcp://") {
        (RendezvousProto::Tcp, r.to_string())
    } else {
        (RendezvousProto::Tcp, s.to_string())
    }
}

/// What a rank learns from the rendezvous.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assignment {
    pub rank: u32,
    pub world: u32,
    /// Every rank's data address, indexed by rank.
    pub peers: Vec<SocketAddr>,
}

fn err(kind: io::ErrorKind, msg: impl Into<String>) -> io::Error {
    io::Error::new(kind, msg.into())
}

fn reg_line(rank: Option<u32>, world: Option<u32>, addr: SocketAddr) -> String {
    let opt = |v: Option<u32>| v.map_or("?".to_string(), |v| v.to_string());
    format!("RLXREG {} {} {addr}\n", opt(rank), opt(world))
}

fn peers_line(rank: u32, world: u32, peers: &[SocketAddr]) -> String {
    let list: Vec<String> = peers.iter().map(SocketAddr::to_string).collect();
    format!("RLXPEERS {rank} {world} {}\n", list.join(","))
}

/// A parsed registration.
struct Reg {
    rank: Option<u32>,
    world: Option<u32>,
    addr: SocketAddr,
}

fn parse_reg(line: &str) -> Option<Reg> {
    let mut it = line.split_whitespace();
    if it.next()? != "RLXREG" {
        return None;
    }
    let opt = |s: &str| (s != "?").then(|| s.parse::<u32>().ok()).flatten();
    let (r, w, a) = (it.next()?, it.next()?, it.next()?);
    Some(Reg {
        rank: opt(r),
        world: opt(w),
        addr: a.parse().ok()?,
    })
}

fn parse_reply(line: &str) -> io::Result<Assignment> {
    let mut it = line.split_whitespace();
    match it.next() {
        Some("RLXPEERS") => {
            let bad = || {
                err(
                    io::ErrorKind::InvalidData,
                    format!("bad rendezvous reply {line:?}"),
                )
            };
            let rank = it.next().and_then(|s| s.parse().ok()).ok_or_else(bad)?;
            let world = it.next().and_then(|s| s.parse().ok()).ok_or_else(bad)?;
            let peers = it
                .next()
                .ok_or_else(bad)?
                .split(',')
                .map(|s| s.parse::<SocketAddr>().map_err(|_| bad()))
                .collect::<io::Result<Vec<_>>>()?;
            Ok(Assignment { rank, world, peers })
        }
        Some("RLXERR") => Err(err(
            io::ErrorKind::InvalidInput,
            format!(
                "rendezvous refused: {}",
                line.trim_start_matches("RLXERR").trim()
            ),
        )),
        _ => Err(err(
            io::ErrorKind::InvalidData,
            format!("bad rendezvous reply {line:?}"),
        )),
    }
}

/// The coordinator's bookkeeping: hand out ranks, reject conflicts.
struct Roster {
    world: u32,
    peers: BTreeMap<u32, SocketAddr>,
}

impl Roster {
    fn new(world: u32, own: SocketAddr) -> Self {
        let mut peers = BTreeMap::new();
        peers.insert(0, own);
        Self { world, peers }
    }
    /// Admit `reg`, returning its rank or a reason to refuse it.
    fn admit(&mut self, reg: &Reg) -> Result<u32, String> {
        if let Some(w) = reg.world
            && w != self.world
        {
            return Err(format!(
                "WORLD={w}, but the coordinator runs WORLD={}",
                self.world
            ));
        }
        // A retransmit (UDP) or reconnect from the same data address keeps its rank.
        if let Some((&r, _)) = self.peers.iter().find(|(_, a)| **a == reg.addr) {
            return Ok(r);
        }
        let r = match reg.rank {
            Some(0) => return Err("rank 0 is the coordinator".into()),
            Some(r) if r >= self.world => {
                return Err(format!("rank {r} out of range for WORLD={}", self.world));
            }
            Some(r) if self.peers.contains_key(&r) => {
                return Err(format!("rank {r} is already taken"));
            }
            Some(r) => r,
            None => (1..self.world)
                .find(|r| !self.peers.contains_key(r))
                .ok_or_else(|| "every rank is taken".to_string())?,
        };
        self.peers.insert(r, reg.addr);
        Ok(r)
    }
    fn full(&self) -> bool {
        self.peers.len() as u32 == self.world
    }
    fn list(&self) -> Vec<SocketAddr> {
        self.peers.values().copied().collect()
    }
}

/// Coordinator (rank 0): wait for `world - 1` registrations at `at`, then
/// tell every rank the full peer list. `own` is rank 0's data address.
/// Returns the peer list.
pub fn serve(
    at: SocketAddr,
    proto: RendezvousProto,
    world: u32,
    own: SocketAddr,
    timeout: Duration,
) -> io::Result<Vec<SocketAddr>> {
    if world <= 1 {
        return Ok(vec![own]);
    }
    let bind = SocketAddr::new(at.ip(), at.port());
    match proto {
        RendezvousProto::Tcp => serve_tcp(TcpListener::bind(bind)?, world, own, timeout),
        RendezvousProto::Udp => serve_udp(UdpSocket::bind(bind)?, world, own, timeout),
    }
}

fn serve_tcp(
    listener: TcpListener,
    world: u32,
    own: SocketAddr,
    timeout: Duration,
) -> io::Result<Vec<SocketAddr>> {
    let deadline = Instant::now() + timeout;
    listener.set_nonblocking(true)?;
    let mut roster = Roster::new(world, own);
    let mut waiting: Vec<(TcpStream, u32)> = Vec::new();
    while !roster.full() {
        if Instant::now() >= deadline {
            return Err(err(
                io::ErrorKind::TimedOut,
                format!(
                    "rendezvous: {} of {world} ranks registered before the timeout",
                    roster.peers.len()
                ),
            ));
        }
        let (stream, _) = match listener.accept() {
            Ok(s) => s,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(20));
                continue;
            }
            Err(e) => return Err(e),
        };
        stream.set_nonblocking(false)?;
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        let mut line = String::new();
        let mut rd = BufReader::new(stream.try_clone()?);
        if rd.read_line(&mut line).is_err() {
            continue;
        }
        let mut stream = stream;
        match parse_reg(&line).ok_or_else(|| format!("unreadable registration {line:?}")) {
            Ok(reg) => match roster.admit(&reg) {
                Ok(r) => waiting.push((stream, r)),
                Err(e) => {
                    let _ = stream.write_all(format!("RLXERR {e}\n").as_bytes());
                }
            },
            Err(e) => {
                let _ = stream.write_all(format!("RLXERR {e}\n").as_bytes());
            }
        }
    }
    let peers = roster.list();
    for (mut s, r) in waiting {
        let _ = s.write_all(peers_line(r, world, &peers).as_bytes());
    }
    Ok(peers)
}

fn serve_udp(
    sock: UdpSocket,
    world: u32,
    own: SocketAddr,
    timeout: Duration,
) -> io::Result<Vec<SocketAddr>> {
    let deadline = Instant::now() + timeout;
    sock.set_read_timeout(Some(Duration::from_millis(100)))?;
    let mut roster = Roster::new(world, own);
    // Reply address per rank (the worker's rendezvous socket).
    let mut reply_to: BTreeMap<u32, SocketAddr> = BTreeMap::new();
    let mut acked: std::collections::BTreeSet<u32> = Default::default();
    let mut buf = [0u8; 65_000];
    // Complete = everyone registered; then keep answering retransmits until
    // every rank acks (or the linger expires: an ack can be lost too).
    let mut done_at: Option<Instant> = None;
    loop {
        let now = Instant::now();
        if let Some(t) = done_at
            && (acked.len() as u32 + 1 >= world || now.duration_since(t) > Duration::from_secs(5))
        {
            return Ok(roster.list());
        }
        if done_at.is_none() && now >= deadline {
            return Err(err(
                io::ErrorKind::TimedOut,
                format!(
                    "rendezvous: {} of {world} ranks registered before the timeout",
                    roster.peers.len()
                ),
            ));
        }
        let Ok((n, from)) = sock.recv_from(&mut buf) else {
            continue;
        };
        let line = String::from_utf8_lossy(&buf[..n]).to_string();
        if let Some(r) = line
            .strip_prefix("RLXACK ")
            .and_then(|s| s.trim().parse::<u32>().ok())
        {
            acked.insert(r);
            continue;
        }
        let Some(reg) = parse_reg(&line) else {
            continue;
        };
        match roster.admit(&reg) {
            Ok(r) => {
                reply_to.insert(r, from);
                if roster.full() {
                    let peers = roster.list();
                    if done_at.is_none() {
                        done_at = Some(Instant::now());
                        for (&r, &to) in &reply_to {
                            let _ = sock.send_to(peers_line(r, world, &peers).as_bytes(), to);
                        }
                    } else {
                        let _ = sock.send_to(peers_line(r, world, &peers).as_bytes(), from);
                    }
                }
            }
            Err(e) => {
                let _ = sock.send_to(format!("RLXERR {e}\n").as_bytes(), from);
            }
        }
    }
}

/// Worker: register with the coordinator at `at` and wait for the peer list.
/// `bind_data(ip)` binds this rank's data endpoint on `ip` (the local address
/// that reaches the coordinator) and returns its port.
pub fn register(
    at: SocketAddr,
    proto: RendezvousProto,
    rank: Option<u32>,
    world: Option<u32>,
    bind_data: impl FnOnce(IpAddr) -> io::Result<u16>,
    timeout: Duration,
) -> io::Result<Assignment> {
    let deadline = Instant::now() + timeout;
    match proto {
        RendezvousProto::Tcp => {
            // The coordinator may not be up yet: retry the connect.
            let stream = loop {
                match TcpStream::connect_timeout(&at, Duration::from_secs(2)) {
                    Ok(s) => break s,
                    Err(e) if Instant::now() >= deadline => return Err(e),
                    Err(_) => std::thread::sleep(Duration::from_millis(200)),
                }
            };
            let ip = stream.local_addr()?.ip();
            let me = SocketAddr::new(ip, bind_data(ip)?);
            let mut w = stream.try_clone()?;
            w.write_all(reg_line(rank, world, me).as_bytes())?;
            stream.set_read_timeout(Some(deadline.saturating_duration_since(Instant::now())))?;
            let mut line = String::new();
            BufReader::new(stream).read_line(&mut line)?;
            if line.is_empty() {
                return Err(err(
                    io::ErrorKind::UnexpectedEof,
                    "rendezvous closed without a reply",
                ));
            }
            parse_reply(&line)
        }
        RendezvousProto::Udp => {
            let any = match at {
                SocketAddr::V4(_) => "0.0.0.0:0",
                SocketAddr::V6(_) => "[::]:0",
            };
            let sock = UdpSocket::bind(any)?;
            sock.connect(at)?;
            let ip = sock.local_addr()?.ip();
            let me = SocketAddr::new(ip, bind_data(ip)?);
            let msg = reg_line(rank, world, me);
            sock.set_read_timeout(Some(Duration::from_millis(250)))?;
            let mut buf = [0u8; 65_000];
            loop {
                if Instant::now() >= deadline {
                    return Err(err(
                        io::ErrorKind::TimedOut,
                        format!("no rendezvous reply from udp://{at}"),
                    ));
                }
                // Not up yet → ECONNREFUSED on a connected UDP socket; keep going.
                let _ = sock.send(msg.as_bytes());
                if let Ok(n) = sock.recv(&mut buf) {
                    let line = String::from_utf8_lossy(&buf[..n]).to_string();
                    let a = parse_reply(&line)?;
                    for _ in 0..3 {
                        let _ = sock.send(format!("RLXACK {}\n", a.rank).as_bytes());
                    }
                    return Ok(a);
                }
            }
        }
    }
}

// ── zero-config: find the rendezvous on the LAN ─────────────────────────────

/// UDP port the coordinator announces its rendezvous address on.
pub const ANNOUNCE_PORT: u16 = 29399;

/// Coordinator: announce `url` (`tcp://ip:port` / `udp://…`) every 200 ms on
/// [`ANNOUNCE_PORT`] by IPv4 broadcast, IPv4 multicast `239.72.66.1` and IPv6
/// multicast `ff02::7266`, until `stop`. Small and repeated, so a lossy path
/// (Wi-Fi drops most IPv6 multicast) still delivers it.
pub fn announce(url: &str, port: u16, stop: &std::sync::atomic::AtomicBool) {
    use std::net::{Ipv4Addr, SocketAddrV6};
    use std::sync::atomic::Ordering;
    let msg = format!("RLXRV {url}");
    // Bonjour/Avahi too: held for as long as the announcing runs.
    #[cfg(feature = "mdns")]
    let _mdns = url
        .rsplit(':')
        .next()
        .and_then(|p| p.trim_end_matches('/').parse::<u16>().ok())
        .and_then(|p| crate::mdns::advertise_rendezvous(url, p).ok());
    let v4 = UdpSocket::bind("0.0.0.0:0").ok();
    if let Some(s) = &v4 {
        s.set_broadcast(true).ok();
        s.set_multicast_ttl_v4(1).ok();
    }
    let v6 = UdpSocket::bind("[::]:0").ok();
    let mut ifaces: Option<Vec<u32>> = None;
    while !stop.load(Ordering::SeqCst) {
        if let Some(s) = &v4 {
            let _ = s.send_to(msg.as_bytes(), (Ipv4Addr::BROADCAST, port));
            let _ = s.send_to(msg.as_bytes(), (Ipv4Addr::new(239, 72, 66, 1), port));
        }
        if let Some(s) = &v6 {
            let try_ifs: Vec<u32> = ifaces.clone().unwrap_or_else(|| (0..=64).collect());
            let ok: Vec<u32> = try_ifs
                .into_iter()
                .filter(|&i| {
                    s.send_to(
                        msg.as_bytes(),
                        SocketAddrV6::new(crate::node::DISC_GROUP_V6, port, 0, i),
                    )
                    .is_ok()
                })
                .collect();
            ifaces.get_or_insert(ok);
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Worker: wait for a coordinator's announcement on `port` and return its
/// rendezvous URL.
pub fn find(port: u16, timeout: Duration) -> io::Result<String> {
    use std::net::{Ipv4Addr, Ipv6Addr};
    let sock = UdpSocket::bind((Ipv6Addr::UNSPECIFIED, port))
        .or_else(|_| UdpSocket::bind((Ipv4Addr::UNSPECIFIED, port)))?;
    let v6 = sock.local_addr()?.is_ipv6();
    if v6 {
        for i in 0..=64 {
            let _ = sock.join_multicast_v6(&crate::node::DISC_GROUP_V6, i);
        }
    } else {
        let _ = sock.join_multicast_v4(&Ipv4Addr::new(239, 72, 66, 1), &Ipv4Addr::UNSPECIFIED);
    }
    sock.set_read_timeout(Some(Duration::from_millis(250)))?;
    let deadline = Instant::now() + timeout;
    // Browse Bonjour/Avahi alongside; whichever answers first wins.
    #[cfg(feature = "mdns")]
    let mdns = {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            if let Ok(url) = crate::mdns::find_rendezvous(timeout) {
                let _ = tx.send(url);
            }
        });
        rx
    };
    let mut buf = [0u8; 512];
    while Instant::now() < deadline {
        #[cfg(feature = "mdns")]
        if let Ok(url) = mdns.try_recv() {
            return Ok(url);
        }
        if let Ok(n) = sock.recv(&mut buf)
            && let Some(url) = std::str::from_utf8(&buf[..n])
                .ok()
                .and_then(|s| s.strip_prefix("RLXRV "))
        {
            return Ok(url.trim().to_string());
        }
    }
    Err(err(
        io::ErrorKind::TimedOut,
        format!("no rendezvous announcement on UDP port {port} (is rank 0 up with RENDEZVOUS?)"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn free_port() -> u16 {
        TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    fn run(proto: RendezvousProto, ranks: Vec<Option<u32>>) -> Vec<Assignment> {
        let port = free_port();
        let at: SocketAddr = (Ipv4Addr::LOCALHOST, port).into();
        let world = ranks.len() as u32 + 1;
        let own: SocketAddr = "127.0.0.1:40000".parse().unwrap();
        let srv = std::thread::spawn(move || serve(at, proto, world, own, Duration::from_secs(10)));
        let workers: Vec<_> = ranks
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                std::thread::spawn(move || {
                    register(
                        at,
                        proto,
                        r,
                        None,
                        |_| Ok(41000 + i as u16),
                        Duration::from_secs(10),
                    )
                    .unwrap()
                })
            })
            .collect();
        let mut out: Vec<Assignment> = workers.into_iter().map(|w| w.join().unwrap()).collect();
        let peers = srv.join().unwrap().unwrap();
        for a in &out {
            assert_eq!(a.peers, peers);
            assert_eq!(a.world, world);
        }
        out.sort_by_key(|a| a.rank);
        out
    }

    #[test]
    fn tcp_and_udp_assign_ranks_and_share_one_peer_list() {
        for proto in [RendezvousProto::Tcp, RendezvousProto::Udp] {
            let out = run(proto, vec![None, Some(3), None]);
            let ranks: Vec<u32> = out.iter().map(|a| a.rank).collect();
            assert_eq!(ranks, vec![1, 2, 3], "{proto:?}");
            assert_eq!(out[0].peers[0], "127.0.0.1:40000".parse().unwrap());
            // Rank 3 asked for 3 and got it; its address is at index 3.
            let r3 = out.iter().find(|a| a.rank == 3).unwrap();
            assert_eq!(r3.peers[3].port(), 41001);
        }
    }

    #[cfg(not(windows))]
    #[test]
    fn workers_find_an_announced_rendezvous() {
        let port = UdpSocket::bind("[::]:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let st = stop.clone();
        let h = std::thread::spawn(move || announce("udp://10.0.0.9:29400", port, &st));
        let url = find(port, Duration::from_secs(10)).unwrap();
        stop.store(true, std::sync::atomic::Ordering::SeqCst);
        h.join().unwrap();
        assert_eq!(url, "udp://10.0.0.9:29400");
    }

    #[test]
    fn conflicts_are_refused_with_a_reason() {
        let mut r = Roster::new(3, "1.2.3.4:1".parse().unwrap());
        let reg = |rank, world, port: u16| Reg {
            rank,
            world,
            addr: SocketAddr::new([1, 2, 3, 5].into(), port),
        };
        assert_eq!(r.admit(&reg(Some(2), None, 2)), Ok(2));
        assert_eq!(
            r.admit(&reg(Some(2), None, 2)),
            Ok(2),
            "retransmit keeps its rank"
        );
        assert!(
            r.admit(&reg(Some(2), None, 3))
                .unwrap_err()
                .contains("taken")
        );
        assert!(
            r.admit(&reg(Some(0), None, 4))
                .unwrap_err()
                .contains("coordinator")
        );
        assert!(
            r.admit(&reg(None, Some(5), 4))
                .unwrap_err()
                .contains("WORLD")
        );
        assert_eq!(r.admit(&reg(None, None, 4)), Ok(1));
        assert!(r.full());
        assert!(r.admit(&reg(None, None, 9)).is_err());
    }

    #[test]
    fn parse_forms() {
        assert_eq!(
            parse_rendezvous("udp://10.0.0.1:29400"),
            (RendezvousProto::Udp, "10.0.0.1:29400".into())
        );
        assert_eq!(
            parse_rendezvous("coord:29400"),
            (RendezvousProto::Tcp, "coord:29400".into())
        );
        let a = parse_reply("RLXPEERS 2 3 1.1.1.1:1,[::1]:2,2.2.2.2:3\n").unwrap();
        assert_eq!((a.rank, a.world, a.peers.len()), (2, 3, 3));
        assert!(parse_reply("RLXERR rank 2 is already taken").is_err());
    }
}
