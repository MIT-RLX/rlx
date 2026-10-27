// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Networked multi-node transport (plan #12/#49, multi-node).
//!
//! [`NetTransport`] is one TCP connection per rank pair (full mesh)
//! with a single reader thread per connection that *demultiplexes*
//! every incoming frame into the right place:
//!
//!   - `SEND`    → a two-sided inbox keyed by `(src, tag)` that
//!     [`Transport::recv_bytes`] pops (pipeline-parallel hidden states).
//!   - `PUT`     → this rank's symmetric heap (one-sided write).
//!   - `GETREQ`  → read this rank's heap, reply with `GETRESP`.
//!   - `GETRESP` → a response inbox the blocked
//!     [`SymmetricTransport::get`] pops.
//!
//! Because the reader thread services the socket, a blocked `recv`/`get`
//! on the main thread never stalls the peer — so the gather-to-root
//! collectives in [`crate::transport::ProcessGroup`] (and the
//! one-sided collectives in [`crate::collective`]) are deadlock-free
//! over any rank count.
//!
//! Two public constructors expose the same engine with different
//! intent:
//!   - [`TcpTransport::bind`] — portable; give it any reachable IPs
//!     (Ethernet, or the macOS Thunderbolt Bridge link).
//!   - [`ThunderboltTransport::bind`] — same wire protocol, intended
//!     for IPs on the Thunderbolt interface, and the place a future
//!     zero-copy DMA backend slots in behind the unchanged
//!     [`Transport`] + [`SymmetricTransport`] traits.

use crate::symmetric::{CollectiveError, Rank, SymmetricBuffer, SymmetricTransport};
use crate::transport::{Transport, default_barrier};
use std::collections::{HashMap, VecDeque};
use std::io::{self, Read, Write};
use std::net::{IpAddr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// Frame kinds. Header is fixed 13 bytes: [kind:u8][a:u32 le][b:u32 le][len:u32 le].
const HELLO: u8 = 1; // a = connector rank, no payload
const SEND: u8 = 2; // a = src rank, b = tag, payload = bytes
const PUT: u8 = 3; // a = src rank, b = offset, payload = bytes to write
const GETREQ: u8 = 4; // a = requester rank, payload = [offset:u32 le][len:u32 le]
const GETRESP: u8 = 5; // a = responder rank, payload = bytes

const HEADER_LEN: usize = 13;

fn write_frame<W: Write>(w: &mut W, kind: u8, a: u32, b: u32, payload: &[u8]) -> io::Result<()> {
    let mut hdr = [0u8; HEADER_LEN];
    hdr[0] = kind;
    hdr[1..5].copy_from_slice(&a.to_le_bytes());
    hdr[5..9].copy_from_slice(&b.to_le_bytes());
    hdr[9..13].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    // Coalesce header + payload into ONE write for the small frames that
    // dominate collective latency: with TCP_NODELAY set, two separate
    // writes became two segments and an extra RTT-coupled syscall. For
    // large frames the second write is negligible against the payload, so
    // keep them separate and skip the copy.
    const COALESCE_MAX: usize = 1 << 16;
    if payload.len() <= COALESCE_MAX {
        let mut buf = Vec::with_capacity(HEADER_LEN + payload.len());
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(payload);
        w.write_all(&buf)?;
    } else {
        w.write_all(&hdr)?;
        w.write_all(payload)?;
    }
    w.flush()
}

fn read_frame<R: Read>(r: &mut R) -> io::Result<(u8, u32, u32, Vec<u8>)> {
    let mut hdr = [0u8; HEADER_LEN];
    r.read_exact(&mut hdr)?;
    let kind = hdr[0];
    let a = u32::from_le_bytes(hdr[1..5].try_into().unwrap());
    let b = u32::from_le_bytes(hdr[5..9].try_into().unwrap());
    let len = u32::from_le_bytes(hdr[9..13].try_into().unwrap()) as usize;
    let mut payload = vec![0u8; len];
    if len > 0 {
        r.read_exact(&mut payload)?;
    }
    Ok((kind, a, b, payload))
}

/// How long a dial to a peer keeps trying: `RLX_CONNECT_WAIT` seconds, `0`
/// (`None`) for as long as it takes (a persistent node waiting for the next
/// job's rank 0); 10 s by default.
fn env_connect_wait() -> Option<Duration> {
    match rlx_ir::env::read("RLX_CONNECT_WAIT")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
    {
        Some(s) if s <= 0.0 => None,
        Some(s) => Some(Duration::from_secs_f64(s)),
        None => Some(Duration::from_secs(10)),
    }
}

/// The longest one connection attempt may take. A peer whose SYNs go
/// unanswered (a Wi-Fi stall, a dropped route) otherwise blocks a plain
/// `connect` for over a minute each try, and a "10 s" budget became a mesh
/// that hung for half an hour.
const CONNECT_ATTEMPT: Duration = Duration::from_secs(2);

fn connect_retry(addr: SocketAddr) -> io::Result<TcpStream> {
    let deadline = env_connect_wait().map(|w| Instant::now() + w);
    let mut last: Option<io::Error> = None;
    loop {
        let left = deadline.map(|d| d.saturating_duration_since(Instant::now()));
        if left.is_some_and(|l| l.is_zero()) {
            break;
        }
        let attempt = left.map_or(CONNECT_ATTEMPT, |l| l.min(CONNECT_ATTEMPT));
        match TcpStream::connect_timeout(&addr, attempt.max(Duration::from_millis(10))) {
            Ok(s) => return Ok(s),
            Err(e) => {
                last = Some(e);
                thread::sleep(Duration::from_millis(25));
            }
        }
    }
    let e = last.unwrap_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "no attempt made"));
    Err(io::Error::new(e.kind(), format!("connect to {addr}: {e}")))
}

struct NetInner {
    rank: u32,
    world: u32,
    heap_size: usize,
    /// Write half of the connection to each peer (index = peer rank).
    /// `None` at our own rank and until a peer connects.
    writers: Vec<Mutex<Option<TcpStream>>>,
    /// Two-sided inbox: `(src_rank, tag) -> queued payloads`.
    inbox: Mutex<HashMap<(u32, u32), VecDeque<Vec<u8>>>>,
    inbox_cv: Condvar,
    /// This rank's symmetric heap (target of remote PUT / GETREQ).
    heap: RwLock<Vec<u8>>,
    /// Responses to our one-sided GETs: `responder_rank -> payloads`.
    getresp: Mutex<HashMap<u32, VecDeque<Vec<u8>>>>,
    getresp_cv: Condvar,
    shutdown: AtomicBool,
    /// Per peer: its connection closed (process died, host went away). A
    /// receive waiting on a lost peer fails instead of blocking forever.
    lost: Vec<AtomicBool>,
    /// Per peer: which of its connections is the current one. A reader whose
    /// connection was replaced (a later HELLO from the same rank) ends
    /// without marking the peer lost.
    conn_gen: Vec<std::sync::atomic::AtomicU64>,
    /// Upper bound on one receive (`RLX_COLLECTIVE_TIMEOUT` seconds).
    recv_timeout: Option<Duration>,
}

/// `RLX_COLLECTIVE_TIMEOUT` (seconds), read once per transport.
/// `RLX_MESH_REVERSE="0-3,0-4"`: pairs of ranks whose connection the
/// higher rank opens — the lower one cannot reach it (an iPad accepts no
/// connection over its USB-C link, but can open one). Every rank of the
/// mesh must be given the same list.
fn env_mesh_reverse() -> Vec<(u32, u32)> {
    rlx_ir::env::read("RLX_MESH_REVERSE")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|p| {
                    let (a, b) = p.trim().split_once('-')?;
                    let (a, b) = (a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?);
                    Some((a.min(b), a.max(b)))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn env_recv_timeout() -> Option<Duration> {
    rlx_ir::env::read("RLX_COLLECTIVE_TIMEOUT")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .map(Duration::from_secs)
}

impl NetInner {
    fn send_frame(
        &self,
        peer: u32,
        kind: u8,
        a: u32,
        b: u32,
        payload: &[u8],
    ) -> Result<(), CollectiveError> {
        let mut guard = self.writers[peer as usize].lock().unwrap();
        match guard.as_mut() {
            Some(s) => {
                write_frame(s, kind, a, b, payload).map_err(|e| CollectiveError::TransportError {
                    reason: format!("send to rank {peer}: {e}"),
                })
            }
            None => Err(CollectiveError::TransportError {
                reason: format!("no connection to rank {peer}"),
            }),
        }
    }

    /// Blocking receive. Queued data is always delivered first; after that a
    /// lost peer, or `recv_timeout` passing, is an error that names the rank —
    /// so a dead node stops the job with a reason instead of hanging it.
    fn recv_inbox(&self, from: u32, tag: u32) -> Result<Vec<u8>, CollectiveError> {
        if from >= self.world {
            return Err(CollectiveError::TransportError {
                reason: format!("rank {from} out of range for WORLD={}", self.world),
            });
        }
        let deadline = self.recv_timeout.map(|t| Instant::now() + t);
        let mut guard = self.inbox.lock().unwrap();
        loop {
            if let Some(q) = guard.get_mut(&(from, tag))
                && let Some(v) = q.pop_front()
            {
                return Ok(v);
            }
            if from != self.rank && self.writers[from as usize].lock().unwrap().is_none() {
                return Err(CollectiveError::TransportError {
                    reason: format!(
                        "rank {} has no connection to rank {from} (a star topology links \
                         workers only to rank 0)",
                        self.rank
                    ),
                });
            }
            if self.is_lost(from) {
                return Err(CollectiveError::TransportError {
                    reason: format!(
                        "rank {from} disconnected (process exited or host unreachable) \
                         while rank {} waited on tag {tag}",
                        self.rank
                    ),
                });
            }
            match deadline {
                None => guard = self.inbox_cv.wait(guard).unwrap(),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return Err(CollectiveError::TransportError {
                            reason: format!(
                                "rank {} waited {}s for rank {from} (tag {tag}) — \
                                 RLX_COLLECTIVE_TIMEOUT",
                                self.rank,
                                self.recv_timeout.unwrap_or_default().as_secs()
                            ),
                        });
                    }
                    guard = self.inbox_cv.wait_timeout(guard, d - now).unwrap().0;
                }
            }
        }
    }

    fn is_lost(&self, peer: u32) -> bool {
        self.lost
            .get(peer as usize)
            .is_some_and(|l| l.load(Ordering::SeqCst))
    }

    /// A reader saw `peer`'s connection end: wake every waiter so receives
    /// on that peer fail now.
    fn mark_lost(&self, peer: u32) {
        if let Some(l) = self.lost.get(peer as usize) {
            l.store(true, Ordering::SeqCst);
        }
        // Take the lock so a waiter between its check and `wait` sees this.
        drop(self.inbox.lock().unwrap());
        self.inbox_cv.notify_all();
        self.getresp_cv.notify_all();
    }

    /// Like [`Self::recv_inbox`] but returns `None` if nothing matching
    /// `(from, tag)` arrives before `timeout` elapses. Lets a collective
    /// bound how long it waits on a slow/offline peer (bounded staleness).
    fn recv_inbox_timeout(&self, from: u32, tag: u32, timeout: Duration) -> Option<Vec<u8>> {
        let deadline = Instant::now() + timeout;
        let mut guard = self.inbox.lock().unwrap();
        loop {
            if let Some(q) = guard.get_mut(&(from, tag))
                && let Some(v) = q.pop_front()
            {
                return Some(v);
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            let (g, res) = self.inbox_cv.wait_timeout(guard, deadline - now).unwrap();
            guard = g;
            if res.timed_out()
                && let Some(q) = guard.get_mut(&(from, tag))
                && let Some(v) = q.pop_front()
            {
                return Some(v);
            }
        }
    }

    fn wait_getresp(&self, peer: u32) -> Vec<u8> {
        let mut guard = self.getresp.lock().unwrap();
        loop {
            if let Some(q) = guard.get_mut(&peer)
                && let Some(v) = q.pop_front()
            {
                return v;
            }
            guard = self.getresp_cv.wait(guard).unwrap();
        }
    }

    fn check_buf(&self, buf: SymmetricBuffer) -> Result<(), CollectiveError> {
        if buf.rank.0 >= self.world {
            return Err(CollectiveError::UnknownRank {
                rank: buf.rank,
                num_ranks: self.world,
            });
        }
        if buf.offset + buf.len > self.heap_size {
            return Err(CollectiveError::OutOfBounds {
                rank: buf.rank,
                offset: buf.offset,
                len: buf.len,
                heap_size: self.heap_size,
            });
        }
        Ok(())
    }

    fn dispatch(&self, kind: u8, a: u32, b: u32, payload: Vec<u8>) {
        match kind {
            SEND => {
                self.inbox
                    .lock()
                    .unwrap()
                    .entry((a, b))
                    .or_default()
                    .push_back(payload);
                self.inbox_cv.notify_all();
            }
            PUT => {
                let off = b as usize;
                if off + payload.len() <= self.heap_size {
                    let mut h = self.heap.write().unwrap();
                    h[off..off + payload.len()].copy_from_slice(&payload);
                }
            }
            GETREQ if payload.len() >= 8 => {
                let off = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
                let rlen = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
                let data = {
                    let h = self.heap.read().unwrap();
                    if off + rlen <= self.heap_size {
                        h[off..off + rlen].to_vec()
                    } else {
                        Vec::new()
                    }
                };
                // a = requester rank; respond with our rank as sender.
                let _ = self.send_frame(a, GETRESP, self.rank, off as u32, &data);
            }
            GETRESP => {
                self.getresp
                    .lock()
                    .unwrap()
                    .entry(a)
                    .or_default()
                    .push_back(payload);
                self.getresp_cv.notify_all();
            }
            _ => {}
        }
    }
}

/// Read `peer`'s frames until its connection closes. `mine` is this
/// connection's generation ([`NetInner::conn_gen`]), taken when it was
/// accepted, not when this thread starts.
fn reader_loop(mut stream: TcpStream, inner: Arc<NetInner>, peer: u32, mine: u64) {
    let current = |inner: &NetInner| inner.conn_gen[peer as usize].load(Ordering::SeqCst);
    // Exits when read_frame errors (peer closed or we were shut down).
    while let Ok((kind, a, b, payload)) = read_frame(&mut stream) {
        if current(&inner) != mine {
            break;
        }
        inner.dispatch(kind, a, b, payload);
        if inner.shutdown.load(Ordering::Relaxed) {
            break;
        }
    }
    if !inner.shutdown.load(Ordering::Relaxed) && current(&inner) == mine {
        inner.mark_lost(peer);
    }
}

/// Full-mesh TCP transport implementing both [`Transport`] (two-sided
/// send/recv) and [`SymmetricTransport`] (one-sided put/get/barrier).
/// Construct via [`TcpTransport::bind`] or [`ThunderboltTransport::bind`].
pub struct NetTransport {
    inner: Arc<NetInner>,
    readers: Mutex<Vec<JoinHandle<()>>>,
}

impl NetTransport {
    /// Build a transport from an already-bound listener. Ranks above us
    /// we connect to; ranks below us we accept from (each connector
    /// announces itself with a `HELLO`). `peers[r]` is rank `r`'s listen
    /// address; `peers[rank]` must match `listener`.
    ///
    /// Public so callers (and multi-rank tests) can pre-bind ephemeral
    /// ports, exchange the addresses out-of-band, then construct the
    /// mesh — avoiding the fixed-port races [`TcpTransport::bind`] is
    /// subject to.
    pub fn from_listener(
        rank: u32,
        world: u32,
        listener: TcpListener,
        peers: Vec<SocketAddr>,
        heap_size: usize,
    ) -> io::Result<Self> {
        Self::from_listener_reversed(rank, world, listener, peers, heap_size, &[])
    }

    /// [`NetTransport::from_listener`], with the connection of each pair in
    /// `reversed` (`(lower, higher)` ranks) opened by its higher rank.
    pub fn from_listener_reversed(
        rank: u32,
        world: u32,
        listener: TcpListener,
        peers: Vec<SocketAddr>,
        heap_size: usize,
        reversed: &[(u32, u32)],
    ) -> io::Result<Self> {
        assert_eq!(
            peers.len(),
            world as usize,
            "peers must have world_size entries"
        );
        assert!(rank < world, "rank out of range");

        let inner = Arc::new(NetInner {
            rank,
            world,
            heap_size,
            writers: (0..world).map(|_| Mutex::new(None)).collect(),
            inbox: Mutex::new(HashMap::new()),
            inbox_cv: Condvar::new(),
            heap: RwLock::new(vec![0u8; heap_size]),
            getresp: Mutex::new(HashMap::new()),
            getresp_cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
            lost: (0..world).map(|_| AtomicBool::new(false)).collect(),
            conn_gen: (0..world)
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
            recv_timeout: env_recv_timeout(),
        });
        let mut readers = Vec::new();

        // Connect to every higher-ranked peer, and to the lower ones of
        // reversed pairs; accept from the rest. A peer's listener is bound
        // before it dials anyone, so the order cannot deadlock.
        let flipped = |a: u32, b: u32| reversed.contains(&(a.min(b), a.max(b)));
        let dial: Vec<u32> = (0..world)
            .filter(|&p| p != rank && ((p > rank) != flipped(rank, p)))
            .collect();
        let mut expect: std::collections::BTreeSet<u32> = (0..world)
            .filter(|&p| p != rank && !dial.contains(&p))
            .collect();
        for p in dial {
            let stream = connect_retry(peers[p as usize])?;
            stream.set_nodelay(true).ok();
            let mut wr = stream.try_clone()?;
            write_frame(&mut wr, HELLO, rank, 0, &[])?;
            let rd = stream.try_clone()?;
            let inner2 = inner.clone();
            readers.push(thread::spawn(move || reader_loop(rd, inner2, p, 0)));
            *inner.writers[p as usize].lock().unwrap() = Some(wr);
        }

        // Accept the rest, each announcing its rank, until every one has: a
        // count of connections is not enough. A persistent worker from the
        // run before, still dialing until it is stopped, announces a rank
        // that its new process announces again — counted twice, it took the
        // place of rank 0's connection, which then sat unaccepted while rank
        // 0 waited at the first barrier. A rank announced again replaces its
        // earlier connection (that process is the one being replaced), and a
        // connection that is not a HELLO from an expected rank is dropped.
        let wanted: std::collections::BTreeSet<u32> = expect.clone();
        while !expect.is_empty() {
            let (stream, addr) = listener.accept()?;
            stream.set_nodelay(true).ok();
            let mut rd = stream.try_clone()?;
            // A connection that never says hello (a port probe) cannot hold
            // up the mesh.
            rd.set_read_timeout(Some(Duration::from_secs(10))).ok();
            let peer = match read_frame(&mut rd) {
                Ok((HELLO, peer, _, _)) if wanted.contains(&peer) => peer,
                Ok((kind, peer, _, _)) => {
                    eprintln!(
                        "[rlx-driver] rank {rank}: dropped a connection from {addr}: kind {kind}, rank {peer}"
                    );
                    continue;
                }
                Err(e) => {
                    eprintln!("[rlx-driver] rank {rank}: dropped a connection from {addr}: {e}");
                    continue;
                }
            };
            rd.set_read_timeout(None).ok();
            if !expect.remove(&peer) {
                eprintln!(
                    "[rlx-driver] rank {rank}: rank {peer} connected again from {addr}; the earlier connection is dropped"
                );
            }
            let wr = stream.try_clone()?;
            let (old, gen_) = {
                let mut w = inner.writers[peer as usize].lock().unwrap();
                let g = inner.conn_gen[peer as usize].fetch_add(1, Ordering::SeqCst) + 1;
                inner.lost[peer as usize].store(false, Ordering::SeqCst);
                (w.replace(wr), g)
            };
            if let Some(old) = old {
                let _ = old.shutdown(Shutdown::Both);
            }
            let inner2 = inner.clone();
            readers.push(thread::spawn(move || reader_loop(rd, inner2, peer, gen_)));
        }

        Ok(Self {
            inner,
            readers: Mutex::new(readers),
        })
    }
}

impl NetTransport {
    fn empty_inner(rank: u32, world: u32, heap_size: usize) -> Arc<NetInner> {
        Arc::new(NetInner {
            rank,
            world,
            heap_size,
            writers: (0..world).map(|_| Mutex::new(None)).collect(),
            inbox: Mutex::new(HashMap::new()),
            inbox_cv: Condvar::new(),
            heap: RwLock::new(vec![0u8; heap_size]),
            getresp: Mutex::new(HashMap::new()),
            getresp_cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
            lost: (0..world).map(|_| AtomicBool::new(false)).collect(),
            conn_gen: (0..world)
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
            recv_timeout: env_recv_timeout(),
        })
    }

    /// Coordinator side of a **star** topology: bind `listener` (rank 0's
    /// address) and accept `world - 1` worker connections, each announcing its
    /// rank with a `HELLO`. Workers connect only to the coordinator (no
    /// worker↔worker links), so a worker needs **no inbound port** — the
    /// NAT/mobile-friendly shape for iOS, phones, and Pis behind a router.
    pub fn coordinator_listen(
        world: u32,
        listener: TcpListener,
        heap_size: usize,
    ) -> io::Result<Self> {
        let inner = Self::empty_inner(0, world, heap_size);
        let mut readers = Vec::new();
        for _ in 1..world {
            let (stream, _addr) = listener.accept()?;
            stream.set_nodelay(true).ok();
            let mut rd = stream.try_clone()?;
            let (kind, peer, _b, _payload) = read_frame(&mut rd)?;
            if kind != HELLO || peer == 0 || peer >= world {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("expected HELLO from a worker rank, got kind={kind} rank={peer}"),
                ));
            }
            let wr = stream.try_clone()?;
            let inner2 = inner.clone();
            readers.push(thread::spawn(move || reader_loop(rd, inner2, peer, 0)));
            *inner.writers[peer as usize].lock().unwrap() = Some(wr);
        }
        Ok(Self {
            inner,
            readers: Mutex::new(readers),
        })
    }

    /// Worker side of a **star** topology: dial the coordinator at `coord_addr`
    /// and announce this `rank`. No listener — the worker only needs outbound
    /// connectivity, so it works from behind NAT / on mobile.
    pub fn worker_dial(
        rank: u32,
        world: u32,
        coord_addr: SocketAddr,
        heap_size: usize,
    ) -> io::Result<Self> {
        assert!(rank > 0 && rank < world, "worker rank must be in 1..world");
        let inner = Self::empty_inner(rank, world, heap_size);
        let stream = connect_retry(coord_addr)?;
        stream.set_nodelay(true).ok();
        let mut wr = stream.try_clone()?;
        write_frame(&mut wr, HELLO, rank, 0, &[])?;
        let rd = stream.try_clone()?;
        let inner2 = inner.clone();
        let readers = vec![thread::spawn(move || reader_loop(rd, inner2, 0, 0))];
        *inner.writers[0].lock().unwrap() = Some(wr);
        Ok(Self {
            inner,
            readers: Mutex::new(readers),
        })
    }
}

impl Drop for NetTransport {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
        // Closing each socket makes the reader threads' read_exact fail.
        for w in &self.inner.writers {
            if let Some(s) = w.lock().unwrap().as_ref() {
                let _ = s.shutdown(Shutdown::Both);
            }
        }
        self.inner.inbox_cv.notify_all();
        self.inner.getresp_cv.notify_all();
        if let Ok(mut hs) = self.readers.lock() {
            for h in hs.drain(..) {
                let _ = h.join();
            }
        }
    }
}

impl Transport for NetTransport {
    fn rank(&self) -> u32 {
        self.inner.rank
    }
    fn world_size(&self) -> u32 {
        self.inner.world
    }
    fn send_bytes(&self, to: u32, tag: u32, bytes: &[u8]) -> Result<(), CollectiveError> {
        if to == self.inner.rank {
            // Self-message: deliver straight to our own inbox.
            self.inner
                .inbox
                .lock()
                .unwrap()
                .entry((to, tag))
                .or_default()
                .push_back(bytes.to_vec());
            self.inner.inbox_cv.notify_all();
            return Ok(());
        }
        self.inner.send_frame(to, SEND, self.inner.rank, tag, bytes)
    }
    fn recv_bytes(&self, from: u32, tag: u32) -> Result<Vec<u8>, CollectiveError> {
        self.inner.recv_inbox(from, tag)
    }
    fn recv_bytes_timeout(
        &self,
        from: u32,
        tag: u32,
        timeout: Duration,
    ) -> Result<Option<Vec<u8>>, CollectiveError> {
        Ok(self.inner.recv_inbox_timeout(from, tag, timeout))
    }
    fn barrier(&self) -> Result<(), CollectiveError> {
        default_barrier(self)
    }
}

impl SymmetricTransport for NetTransport {
    fn num_ranks(&self) -> u32 {
        self.inner.world
    }
    fn this_rank(&self) -> Rank {
        Rank(self.inner.rank)
    }
    fn put(&self, buf: SymmetricBuffer, src: &[u8]) -> Result<(), CollectiveError> {
        self.inner.check_buf(buf)?;
        if src.len() != buf.len {
            return Err(CollectiveError::LengthMismatch {
                expected: buf.len,
                got: src.len(),
            });
        }
        if buf.rank.0 == self.inner.rank {
            let mut h = self.inner.heap.write().unwrap();
            h[buf.offset..buf.offset + buf.len].copy_from_slice(src);
            Ok(())
        } else {
            self.inner
                .send_frame(buf.rank.0, PUT, self.inner.rank, buf.offset as u32, src)
        }
    }
    fn get(&self, buf: SymmetricBuffer, dst: &mut [u8]) -> Result<(), CollectiveError> {
        self.inner.check_buf(buf)?;
        if dst.len() != buf.len {
            return Err(CollectiveError::LengthMismatch {
                expected: buf.len,
                got: dst.len(),
            });
        }
        if buf.rank.0 == self.inner.rank {
            let h = self.inner.heap.read().unwrap();
            dst.copy_from_slice(&h[buf.offset..buf.offset + buf.len]);
            Ok(())
        } else {
            let mut payload = [0u8; 8];
            payload[0..4].copy_from_slice(&(buf.offset as u32).to_le_bytes());
            payload[4..8].copy_from_slice(&(buf.len as u32).to_le_bytes());
            self.inner
                .send_frame(buf.rank.0, GETREQ, self.inner.rank, 0, &payload)?;
            let data = self.inner.wait_getresp(buf.rank.0);
            if data.len() != buf.len {
                return Err(CollectiveError::LengthMismatch {
                    expected: buf.len,
                    got: data.len(),
                });
            }
            dst.copy_from_slice(&data);
            Ok(())
        }
    }
    fn barrier(&self) -> Result<(), CollectiveError> {
        default_barrier(self)
    }
}

/// Default symmetric-heap size per rank (bytes) when a constructor is
/// not given one. 64 MiB comfortably holds a single transformer hidden
/// state at typical batch×seq×d_model for collective scratch.
pub const DEFAULT_HEAP_BYTES: usize = 64 * 1024 * 1024;

/// Portable TCP transport. Works over any reachable IP — Ethernet or
/// the macOS Thunderbolt Bridge link. This is the baseline that
/// pipeline- and tensor-parallel inference run on before any fast path.
pub struct TcpTransport;

impl TcpTransport {
    /// Bind this rank's listener at `peers[rank]`'s port, on every local
    /// address, and establish the full mesh. The ranks' lists agree on every
    /// port; a peer may name this rank by another of its addresses (an iPad
    /// over its USB-C link rather than Wi-Fi), so the listener cannot be
    /// bound to the one address this rank's own list gives — it was, and a
    /// rank on its USB-C link refused every connection.
    pub fn bind(
        rank: u32,
        world: u32,
        peers: Vec<SocketAddr>,
        heap_size: usize,
    ) -> io::Result<NetTransport> {
        let listener = listen_any(peers[rank as usize])?;
        NetTransport::from_listener_reversed(
            rank,
            world,
            listener,
            peers,
            heap_size,
            &env_mesh_reverse(),
        )
    }
}

/// A listener on `at`'s port on every local address — `[::]`, dual-stack,
/// except on Windows, whose IPv6 sockets take no IPv4 — else at `at` itself.
fn listen_any(at: SocketAddr) -> io::Result<TcpListener> {
    let v4 = SocketAddr::new(
        std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED),
        at.port(),
    );
    let v6 = SocketAddr::new(
        std::net::IpAddr::V6(std::net::Ipv6Addr::UNSPECIFIED),
        at.port(),
    );
    let first = if cfg!(windows) { v4 } else { v6 };
    TcpListener::bind(first)
        .or_else(|_| TcpListener::bind(v4))
        .or_else(|_| TcpListener::bind(at))
}

/// Thunderbolt transport. Same wire protocol as [`TcpTransport`] today
/// (TCP over the Thunderbolt Bridge IP link), but a distinct type so
/// the symmetric one-sided heap path is the intended interface and so a
/// future zero-copy Thunderbolt DMA backend can replace the engine
/// behind the unchanged [`Transport`] + [`SymmetricTransport`] traits.
pub struct ThunderboltTransport;

impl ThunderboltTransport {
    /// Bind at `peers[rank]`. The addresses are expected to be on the
    /// Thunderbolt interface (see [`ThunderboltTransport::looks_like_thunderbolt`]);
    /// correctness does not depend on it, but bandwidth does.
    pub fn bind(
        rank: u32,
        world: u32,
        peers: Vec<SocketAddr>,
        heap_size: usize,
    ) -> io::Result<NetTransport> {
        TcpTransport::bind(rank, world, peers, heap_size)
    }

    /// Heuristic: is `ip` plausibly a Thunderbolt Bridge address? macOS
    /// auto-assigns link-local `169.254.0.0/16` to the bridge when no
    /// static IP is configured; static setups commonly use a private
    /// `10.0.0.0/8` block. This is advisory only.
    pub fn looks_like_thunderbolt(ip: IpAddr) -> bool {
        match ip {
            IpAddr::V4(v4) => {
                let o = v4.octets();
                o[0] == 169 && o[1] == 254 || o[0] == 10
            }
            IpAddr::V6(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    /// A dial to an address that never answers gives up within its
    /// budget: one blocked `connect` took over a minute a try.
    #[test]
    fn a_dial_that_gets_no_answer_ends_within_its_budget() {
        rlx_ir::env::set("RLX_CONNECT_WAIT", "1");
        let t = std::time::Instant::now();
        // TEST-NET-1 (RFC 5737): routed nowhere.
        let r = super::connect_retry("192.0.2.1:9".parse().unwrap());
        rlx_ir::env::remove("RLX_CONNECT_WAIT");
        assert!(r.is_err());
        assert!(
            t.elapsed() < std::time::Duration::from_secs(4),
            "took {:?}",
            t.elapsed()
        );
    }

    /// A mesh whose pairs (0,2) and (1,2) are opened by rank 2 carries
    /// messages both ways on every link, like the usual one.
    #[test]
    fn a_reversed_pair_is_dialed_by_its_higher_rank() {
        let listeners: Vec<TcpListener> = (0..3)
            .map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let peers: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(r, l)| {
                let peers = peers.clone();
                std::thread::spawn(move || {
                    let t = NetTransport::from_listener_reversed(
                        r as u32,
                        3,
                        l,
                        peers,
                        64,
                        &[(0, 2), (1, 2)],
                    )
                    .unwrap();
                    for p in 0..3u32 {
                        if p != r as u32 {
                            t.send_bytes(p, 7, &[r as u8]).unwrap();
                        }
                    }
                    let mut got: Vec<u8> = (0..3u32)
                        .filter(|&p| p != r as u32)
                        .map(|p| t.recv_bytes(p, 7).unwrap()[0])
                        .collect();
                    got.sort();
                    got
                })
            })
            .collect();
        let got: Vec<Vec<u8>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(got, vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
    }

    /// A stale process's HELLO (a persistent worker from the run before,
    /// still dialing) and a bare port probe, queued at a rank's listener
    /// before the real ranks connect, neither take a real rank's place: the
    /// mesh forms, every pair carries messages, and the replaced connection
    /// closing marks no one lost. Counted by connections, the stale HELLO
    /// left rank 0's connection unaccepted and the first barrier hung.
    #[test]
    fn a_stale_hello_does_not_take_a_ranks_place() {
        let listeners: Vec<TcpListener> = (0..3)
            .map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let peers: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        // A probe that says nothing and goes, then an old "rank 1".
        drop(TcpStream::connect(peers[2]).unwrap());
        let mut stale = TcpStream::connect(peers[2]).unwrap();
        write_frame(&mut stale, HELLO, 1, 0, &[]).unwrap();
        let mut stale = Some(stale);
        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(r, l)| {
                let peers = peers.clone();
                let stale = if r == 2 { stale.take() } else { None };
                std::thread::spawn(move || {
                    let t = NetTransport::from_listener(r as u32, 3, l, peers, 64).unwrap();
                    // The old process goes away once the mesh is up.
                    if let Some(s) = stale {
                        drop(s);
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    t.send_bytes((r as u32 + 1) % 3, 7, &[r as u8]).unwrap();
                    t.send_bytes((r as u32 + 2) % 3, 7, &[r as u8]).unwrap();
                    let mut got: Vec<u8> = (0..3u32)
                        .filter(|&p| p != r as u32)
                        .map(|p| t.recv_bytes(p, 7).unwrap()[0])
                        .collect();
                    got.sort();
                    // Rank 2 then waits on rank 1 for a reply sent only after
                    // the stale connection closed: a rank marked lost would
                    // fail it.
                    match r {
                        1 => {
                            t.recv_bytes(2, 8).unwrap();
                            std::thread::sleep(Duration::from_millis(100));
                            t.send_bytes(2, 9, &[1]).unwrap();
                        }
                        2 => {
                            t.send_bytes(1, 8, &[2]).unwrap();
                            assert_eq!(t.recv_bytes(1, 9).unwrap(), vec![1]);
                        }
                        _ => {}
                    }
                    got
                })
            })
            .collect();
        let got: Vec<Vec<u8>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(got, vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
    }

    use super::*;
    use crate::collective::{ReduceKind, all_reduce};
    use crate::transport::ProcessGroup;

    /// Pre-bind `n` loopback listeners and run `body(rank, transport)`
    /// on a thread per rank. Returns once every rank's body returns.
    /// A peer that exits mid-job fails the receive waiting on it (with its
    /// rank) instead of hanging; data it sent before exiting still arrives.
    #[test]
    fn receive_from_a_dead_peer_errors_instead_of_hanging() {
        let listeners: Vec<TcpListener> = (0..2)
            .map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let mut it = listeners.into_iter();
        let (l0, l1) = (it.next().unwrap(), it.next().unwrap());
        let a1 = addrs.clone();
        let peer = thread::spawn(move || {
            let t = NetTransport::from_listener(1, 2, l1, a1, 1024).unwrap();
            t.send_bytes(0, 7, b"last words").unwrap();
            // `t` drops here: the process "dies".
        });
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let t = NetTransport::from_listener(0, 2, l0, addrs, 1024).unwrap();
            let first = t.recv_bytes(1, 7);
            let second = t.recv_bytes(1, 7);
            tx.send((first, second)).unwrap();
        });
        peer.join().unwrap();
        let (first, second) = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("receive on a dead peer hung");
        assert_eq!(first.unwrap(), b"last words");
        let err = second.expect_err("dead peer must error").to_string();
        assert!(err.contains("rank 1"), "error should name the rank: {err}");
    }

    fn run_net<F>(n: u32, heap: usize, body: F)
    where
        F: Fn(u32, Arc<NetTransport>) + Send + Sync + 'static,
    {
        let listeners: Vec<TcpListener> = (0..n)
            .map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap())
            .collect();
        let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
        let body = Arc::new(body);
        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let addrs = addrs.clone();
                let body = body.clone();
                thread::spawn(move || {
                    let t = Arc::new(
                        NetTransport::from_listener(rank as u32, n, listener, addrs, heap)
                            .expect("transport build"),
                    );
                    body(rank as u32, t.clone());
                    // Trailing barrier: keep every rank's reader threads alive
                    // until all peers finish, so a late remote GET still gets a
                    // response before any transport is dropped.
                    <NetTransport as Transport>::barrier(&*t).unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    /// Star variant of [`run_net`]: rank 0 listens (coordinator) and ranks
    /// 1..n dial in (workers, no inbound port of their own).
    fn run_star<F>(n: u32, heap: usize, body: F)
    where
        F: Fn(u32, Arc<NetTransport>) + Send + Sync + 'static,
    {
        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let coord_addr = listener.local_addr().unwrap();
        let body = Arc::new(body);
        let mut handles = Vec::new();
        {
            let body = body.clone();
            handles.push(thread::spawn(move || {
                let t = Arc::new(NetTransport::coordinator_listen(n, listener, heap).unwrap());
                body(0, t.clone());
                <NetTransport as Transport>::barrier(&*t).unwrap();
            }));
        }
        for rank in 1..n {
            let body = body.clone();
            handles.push(thread::spawn(move || {
                let t = Arc::new(NetTransport::worker_dial(rank, n, coord_addr, heap).unwrap());
                body(rank, t.clone());
                <NetTransport as Transport>::barrier(&*t).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn star_dial_out_coordinator_worker() {
        // Workers dial the coordinator (no listener of their own); the
        // coordinator sums their contributions and sends the result back.
        run_star(3, 4096, |rank, t| {
            let g = ProcessGroup::new(t);
            if rank == 0 {
                let a = g.recv_f32(1, 5).unwrap();
                let b = g.recv_f32(2, 5).unwrap();
                let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
                g.send_f32(1, 6, &sum).unwrap();
                g.send_f32(2, 6, &sum).unwrap();
            } else {
                g.send_f32(0, 5, &[rank as f32]).unwrap();
                assert_eq!(g.recv_f32(0, 6).unwrap(), vec![3.0]); // 1 + 2
            }
        });
    }

    #[test]
    fn two_sided_pipeline_handoff() {
        // rank r sends [r*10..] to r-1 (toward the leader), like a
        // pipeline returning hidden states.
        run_net(3, 4096, |rank, t| {
            let g = ProcessGroup::new(t);
            let n = g.world_size();
            if rank + 1 < n {
                let got = g.recv_f32(rank + 1, 7).unwrap();
                assert_eq!(got, vec![(rank as f32 + 1.0) * 10.0, 1.0]);
            }
            if rank > 0 {
                g.send_f32(rank - 1, 7, &[rank as f32 * 10.0, 1.0]).unwrap();
            }
        });
    }

    #[test]
    fn process_group_all_reduce_over_tcp() {
        run_net(4, 4096, |rank, t| {
            let g = ProcessGroup::new(t);
            let mut data = vec![rank as f32 + 1.0; 5];
            g.all_reduce(&mut data, ReduceKind::Sum).unwrap();
            assert_eq!(data, vec![10.0; 5], "rank {rank}");
        });
    }

    #[test]
    fn federated_average_skips_late_worker() {
        // rank 2 behaves like an offline/slow edge client — it only sends
        // after the deadline, so rank 0 drops it and averages ranks 0 and 1.
        run_net(3, 4096, |rank, t| {
            let g = ProcessGroup::new(t);
            let mut data = vec![rank as f32 + 1.0; 3]; // 1, 2, 3
            if rank == 2 {
                std::thread::sleep(Duration::from_millis(300));
            }
            let present = g
                .federated_average(&mut data, Duration::from_millis(100))
                .unwrap();
            if rank == 0 {
                assert_eq!(present, 2, "late rank 2 must be dropped");
                assert_eq!(data, vec![1.5; 3]); // mean of ranks 0 and 1
            }
        });
    }

    #[test]
    fn process_group_barrier_and_broadcast() {
        run_net(3, 4096, |rank, t| {
            let g = ProcessGroup::new(t);
            g.barrier().unwrap();
            let mut data = if rank == 0 {
                vec![1.0, 2.0]
            } else {
                vec![0.0, 0.0]
            };
            g.broadcast(0, &mut data).unwrap();
            assert_eq!(data, vec![1.0, 2.0], "rank {rank}");
        });
    }

    #[test]
    fn symmetric_remote_put_get() {
        // Rank 0 PUTs into rank 1's heap; rank 1 reads its own slot and
        // sees it. Then rank 1 GETs from rank 0's heap.
        run_net(2, 4096, |rank, t| {
            let off = 128;
            let payload = [5u8, 6, 7, 8];
            if rank == 0 {
                // seed our own heap for rank 1 to GET later
                t.put(
                    SymmetricBuffer {
                        rank: Rank(0),
                        offset: off,
                        len: 4,
                    },
                    &[1u8, 2, 3, 4],
                )
                .unwrap();
                // write into rank 1's heap
                t.put(
                    SymmetricBuffer {
                        rank: Rank(1),
                        offset: off,
                        len: 4,
                    },
                    &payload,
                )
                .unwrap();
            }
            // Make sure the PUT landed before the reads.
            <NetTransport as Transport>::barrier(&t).unwrap();
            // ^ disambiguates Transport::barrier from SymmetricTransport::barrier
            if rank == 1 {
                let mut got = [0u8; 4];
                t.get(
                    SymmetricBuffer {
                        rank: Rank(1),
                        offset: off,
                        len: 4,
                    },
                    &mut got,
                )
                .unwrap();
                assert_eq!(got, payload, "local read of remote PUT");
                let mut remote = [0u8; 4];
                t.get(
                    SymmetricBuffer {
                        rank: Rank(0),
                        offset: off,
                        len: 4,
                    },
                    &mut remote,
                )
                .unwrap();
                assert_eq!(remote, [1u8, 2, 3, 4], "remote GET from rank 0");
            }
        });
    }

    #[test]
    fn symmetric_collective_all_reduce_over_net() {
        // The existing one-sided collective (collective::all_reduce)
        // runs unmodified over NetTransport.
        run_net(4, 4096, |rank, t| {
            let elems = 3usize;
            let bytes = elems * 4;
            let buf = SymmetricBuffer {
                rank: Rank(rank),
                offset: 0,
                len: bytes,
            };
            let mut local = vec![rank as f32 + 1.0; elems];
            all_reduce(&*t, buf, &mut local, ReduceKind::Sum).unwrap();
            assert_eq!(local, vec![10.0; elems], "rank {rank}");
        });
    }

    #[test]
    fn thunderbolt_addr_heuristic() {
        use std::net::Ipv4Addr;
        assert!(ThunderboltTransport::looks_like_thunderbolt(
            Ipv4Addr::new(169, 254, 3, 1).into()
        ));
        assert!(ThunderboltTransport::looks_like_thunderbolt(
            Ipv4Addr::new(10, 0, 0, 2).into()
        ));
        assert!(!ThunderboltTransport::looks_like_thunderbolt(
            Ipv4Addr::new(192, 168, 1, 5).into()
        ));
    }
}
