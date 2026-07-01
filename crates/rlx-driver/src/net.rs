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
use std::time::Duration;

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

fn connect_retry(addr: SocketAddr) -> io::Result<TcpStream> {
    let mut last: Option<io::Error> = None;
    for _ in 0..400 {
        match TcpStream::connect(addr) {
            Ok(s) => return Ok(s),
            Err(e) => {
                last = Some(e);
                thread::sleep(Duration::from_millis(25));
            }
        }
    }
    Err(last.unwrap_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "connect retry exhausted")))
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

    fn recv_inbox(&self, from: u32, tag: u32) -> Vec<u8> {
        let mut guard = self.inbox.lock().unwrap();
        loop {
            if let Some(q) = guard.get_mut(&(from, tag))
                && let Some(v) = q.pop_front()
            {
                return v;
            }
            guard = self.inbox_cv.wait(guard).unwrap();
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

fn reader_loop(mut stream: TcpStream, inner: Arc<NetInner>) {
    // Exits when read_frame errors (peer closed or we were shut down).
    while let Ok((kind, a, b, payload)) = read_frame(&mut stream) {
        inner.dispatch(kind, a, b, payload);
        if inner.shutdown.load(Ordering::Relaxed) {
            break;
        }
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
        });
        let mut readers = Vec::new();

        // Connect to every higher-ranked peer.
        for p in (rank + 1)..world {
            let stream = connect_retry(peers[p as usize])?;
            stream.set_nodelay(true).ok();
            let mut wr = stream.try_clone()?;
            write_frame(&mut wr, HELLO, rank, 0, &[])?;
            let rd = stream.try_clone()?;
            let inner2 = inner.clone();
            readers.push(thread::spawn(move || reader_loop(rd, inner2)));
            *inner.writers[p as usize].lock().unwrap() = Some(wr);
        }

        // Accept exactly `rank` connections from lower-ranked peers.
        for _ in 0..rank {
            let (stream, _addr) = listener.accept()?;
            stream.set_nodelay(true).ok();
            let mut rd = stream.try_clone()?;
            let (kind, peer, _b, _payload) = read_frame(&mut rd)?;
            if kind != HELLO || peer >= world {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("expected HELLO with valid rank, got kind={kind} rank={peer}"),
                ));
            }
            let wr = stream.try_clone()?;
            let inner2 = inner.clone();
            readers.push(thread::spawn(move || reader_loop(rd, inner2)));
            *inner.writers[peer as usize].lock().unwrap() = Some(wr);
        }

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
        Ok(self.inner.recv_inbox(from, tag))
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
    /// Bind this rank's listener at `peers[rank]` and establish the full
    /// mesh. Every rank must pass the same `peers` list.
    pub fn bind(
        rank: u32,
        world: u32,
        peers: Vec<SocketAddr>,
        heap_size: usize,
    ) -> io::Result<NetTransport> {
        let listener = TcpListener::bind(peers[rank as usize])?;
        NetTransport::from_listener(rank, world, listener, peers, heap_size)
    }
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
    use super::*;
    use crate::collective::{ReduceKind, all_reduce};
    use crate::transport::ProcessGroup;

    /// Pre-bind `n` loopback listeners and run `body(rank, transport)`
    /// on a thread per rank. Returns once every rank's body returns.
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
