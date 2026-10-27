// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Request/reply RPC over TCP that can tell a slow peer from a dead one.
//!
//! A request is `[op u8][len u64][payload]`. A reply is any number of heartbeat
//! frames `[2]` — sent every [`HEARTBEAT`] while the server is still working on
//! the request — then `[0][len u64][payload]` or `[1][len u64][utf-8 error]`.
//!
//! The heartbeat is what lets a client bound every read. A machine that loses
//! power or network does not close its sockets, so an unbounded read on it waits
//! forever — and a coordinator blocked on a peer looks exactly like a model that
//! is still thinking. With heartbeats, a peer silent for
//! [`RpcOptions::timeout`] is an error that names it, while a request that
//! legitimately takes minutes (a layer range streamed from disk) keeps the
//! connection alive.
//!
//! [`serve`] runs a handler with one thread per connection, so a health probe or
//! a reconnecting client is never stuck behind an idle connection; requests
//! execute one at a time against the handler's state. A handler error or panic
//! is returned to the client as a message and the server keeps serving.
//! Op [`OP_SHUTDOWN`] stops it, closing idle connections.
//!
//! [`Enc`] / [`Dec`] are the little-endian payload codec both sides use,
//! including lists of `f32` vectors (a pipeline's hidden state plus whatever
//! side tensors cross a stage boundary).

use anyhow::{Context, Result, bail, ensure};
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

/// Interval between a busy server's heartbeat frames.
pub const HEARTBEAT: Duration = Duration::from_secs(5);

/// Reserved op: reply, then stop the server.
pub const OP_SHUTDOWN: u8 = 255;

const OK: u8 = 0;
const ERR: u8 = 1;
const BEAT: u8 = 2;

// ── codec ──────────────────────────────────────────────────────────────────

/// Little-endian payload builder.
#[derive(Default, Debug, Clone)]
pub struct Enc(pub Vec<u8>);

impl Enc {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn u8(mut self, v: u8) -> Self {
        self.0.push(v);
        self
    }
    pub fn u32(mut self, v: u32) -> Self {
        self.0.extend(v.to_le_bytes());
        self
    }
    pub fn u64(mut self, v: u64) -> Self {
        self.0.extend(v.to_le_bytes());
        self
    }
    pub fn f32(mut self, v: f32) -> Self {
        self.0.extend(v.to_le_bytes());
        self
    }
    pub fn f64(mut self, v: f64) -> Self {
        self.0.extend(v.to_le_bytes());
        self
    }
    /// Length-prefixed UTF-8.
    pub fn str(self, s: &str) -> Self {
        let mut e = self.u32(s.len() as u32);
        e.0.extend(s.as_bytes());
        e
    }
    /// Length-prefixed `f32` vector.
    pub fn f32s(self, v: &[f32]) -> Self {
        let mut e = self.u64(v.len() as u64);
        e.0.reserve(v.len() * 4);
        for x in v {
            e.0.extend(x.to_le_bytes());
        }
        e
    }
    /// A count then each vector ([`Self::f32s`]).
    pub fn vecs<'a>(self, vs: impl IntoIterator<Item = &'a [f32]>) -> Self {
        let vs: Vec<&[f32]> = vs.into_iter().collect();
        let mut e = self.u32(vs.len() as u32);
        for v in vs {
            e = e.f32s(v);
        }
        e
    }
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }
}

/// Little-endian payload reader; every read checks the remaining length.
#[derive(Debug, Clone, Copy)]
pub struct Dec<'a>(pub &'a [u8]);

impl Dec<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        ensure!(self.0.len() >= n, "truncated message");
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }
    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into()?))
    }
    pub fn str(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        Ok(String::from_utf8(self.take(n)?.to_vec())?)
    }
    pub fn f32s(&mut self) -> Result<Vec<f32>> {
        let n = self.u64()? as usize;
        let b = self.take(n.checked_mul(4).context("vector length overflows")?)?;
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
    pub fn vecs(&mut self) -> Result<Vec<Vec<f32>>> {
        let n = self.u32()? as usize;
        (0..n).map(|_| self.f32s()).collect()
    }
    /// Bytes not read yet.
    pub fn remaining(&self) -> usize {
        self.0.len()
    }
}

fn write_frame(w: &mut impl Write, tag: u8, payload: &[u8]) -> std::io::Result<()> {
    w.write_all(&[tag])?;
    w.write_all(&(payload.len() as u64).to_le_bytes())?;
    w.write_all(payload)?;
    w.flush()
}

fn read_vec(r: &mut impl Read, n: usize) -> std::io::Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    r.read_exact(&mut v)?;
    Ok(v)
}

// ── server ─────────────────────────────────────────────────────────────────

/// What a server does with one request: `(op, payload) → reply payload`.
/// [`OP_SHUTDOWN`] never reaches it.
pub trait RpcHandler: Send {
    fn handle(&mut self, op: u8, payload: &[u8]) -> Result<Vec<u8>>;
}

impl<F: FnMut(u8, &[u8]) -> Result<Vec<u8>> + Send> RpcHandler for F {
    fn handle(&mut self, op: u8, payload: &[u8]) -> Result<Vec<u8>> {
        self(op, payload)
    }
}

struct Shared<H> {
    handler: Mutex<H>,
    served: AtomicUsize,
    max: usize,
    done: AtomicBool,
    wake: SocketAddr,
    open: Mutex<Vec<TcpStream>>,
}

impl<H> Shared<H> {
    fn finish(&self) {
        if !self.done.swap(true, Ordering::SeqCst) {
            for c in self
                .open
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .drain(..)
            {
                let _ = c.shutdown(std::net::Shutdown::Read);
            }
            let _ = TcpStream::connect(self.wake);
        }
    }
}

/// Serve `handler` on `listener` until an [`OP_SHUTDOWN`] request or, when
/// `max_requests > 0`, that many requests.
pub fn serve<H: RpcHandler>(listener: TcpListener, handler: H, max_requests: usize) -> Result<()> {
    let shared = Shared {
        handler: Mutex::new(handler),
        served: AtomicUsize::new(0),
        max: max_requests,
        done: AtomicBool::new(false),
        wake: listener.local_addr()?,
        open: Mutex::new(Vec::new()),
    };
    std::thread::scope(|scope| {
        for conn in listener.incoming() {
            if shared.done.load(Ordering::SeqCst) {
                break;
            }
            let Ok(stream) = conn else {
                continue;
            };
            let _ = stream.set_nodelay(true);
            if let Ok(c) = stream.try_clone() {
                shared
                    .open
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(c);
            }
            let sh = &shared;
            scope.spawn(move || {
                if let Err(e) = serve_connection(sh, stream) {
                    eprintln!("rpc: connection closed: {e}");
                }
            });
        }
    });
    Ok(())
}

fn serve_connection<H: RpcHandler>(sh: &Shared<H>, stream: TcpStream) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    loop {
        if sh.done.load(Ordering::SeqCst) {
            return Ok(());
        }
        let mut tag = [0u8; 1];
        match reader.read_exact(&mut tag) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }
        let mut len = [0u8; 8];
        reader.read_exact(&mut len)?;
        let payload = read_vec(&mut reader, u64::from_le_bytes(len) as usize)?;
        if tag[0] == OP_SHUTDOWN {
            write_frame(&mut BufWriter::new(&stream), OK, &[])?;
            sh.finish();
            return Ok(());
        }
        // Heartbeat from before the handler lock is taken until the reply is
        // ready (joined first, so frames never interleave): a request queued
        // behind another is alive, not stalled.
        let stop = AtomicBool::new(false);
        let mut beat = stream.try_clone()?;
        let result = std::thread::scope(|s| {
            let hb = s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    std::thread::park_timeout(HEARTBEAT);
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    if beat.write_all(&[BEAT]).and_then(|_| beat.flush()).is_err() {
                        break;
                    }
                }
            });
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // A panic in an earlier request poisons the lock; the state is
                // still the best there is, and that error already went back.
                let mut h = sh.handler.lock().unwrap_or_else(|p| p.into_inner());
                h.handle(tag[0], &payload)
            }));
            stop.store(true, Ordering::Relaxed);
            hb.thread().unpark();
            let _ = hb.join();
            r
        });
        let mut out = BufWriter::new(&stream);
        match result {
            Ok(Ok(reply)) => write_frame(&mut out, OK, &reply)?,
            Ok(Err(e)) => write_frame(&mut out, ERR, format!("{e:#}").as_bytes())?,
            Err(p) => {
                let msg = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "handler panicked".into());
                write_frame(&mut out, ERR, format!("panic: {msg}").as_bytes())?;
            }
        }
        let n = sh.served.fetch_add(1, Ordering::SeqCst) + 1;
        if sh.max != 0 && n >= sh.max {
            sh.finish();
            return Ok(());
        }
    }
}

// ── client ─────────────────────────────────────────────────────────────────

/// Client timeouts and retries.
#[derive(Debug, Clone, Copy)]
pub struct RpcOptions {
    /// Longest the server may send nothing (it heartbeats every [`HEARTBEAT`]
    /// while busy) before it is declared unresponsive.
    pub timeout: Duration,
    /// Connection attempts (servers may still be starting).
    pub connect_attempts: u32,
    /// Pause between connection attempts.
    pub connect_backoff: Duration,
}

impl Default for RpcOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            connect_attempts: 10,
            connect_backoff: Duration::from_secs(1),
        }
    }
}

/// A persistent connection to one [`serve`]r.
pub struct RpcClient {
    pub addr: String,
    stream: TcpStream,
    timeout: Duration,
}

impl RpcClient {
    pub fn connect(addr: &str, opts: &RpcOptions) -> Result<Self> {
        let sa = rlx_driver::resolve_addr(addr).with_context(|| format!("resolving {addr}"))?;
        let mut last = None;
        for _ in 0..opts.connect_attempts.max(1) {
            match TcpStream::connect_timeout(&sa, opts.timeout.min(Duration::from_secs(10))) {
                Ok(s) => {
                    s.set_nodelay(true)?;
                    s.set_read_timeout(Some(opts.timeout))?;
                    s.set_write_timeout(Some(opts.timeout))?;
                    return Ok(Self {
                        addr: addr.to_string(),
                        stream: s,
                        timeout: opts.timeout,
                    });
                }
                Err(e) => {
                    last = Some(e);
                    std::thread::sleep(opts.connect_backoff);
                }
            }
        }
        bail!(
            "could not connect to {addr} after {} attempts: {}",
            opts.connect_attempts,
            last.map(|e| e.to_string()).unwrap_or_default()
        )
    }

    /// Send a request and wait for its reply, tolerating heartbeats. A server
    /// error comes back as `Err` with the server's message.
    pub fn call(&mut self, op: u8, payload: &[u8]) -> Result<Vec<u8>> {
        let addr = self.addr.clone();
        let timeout = self.timeout;
        let io = |e: std::io::Error| -> anyhow::Error {
            if matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ) {
                anyhow::anyhow!(
                    "{addr} sent nothing for {timeout:?} (no heartbeat): that machine has \
                     stopped responding"
                )
            } else {
                anyhow::anyhow!("{addr}: {e}")
            }
        };
        write_frame(&mut BufWriter::new(&self.stream), op, payload).map_err(io)?;
        let mut r = &self.stream;
        loop {
            let mut tag = [0u8; 1];
            r.read_exact(&mut tag).map_err(io)?;
            match tag[0] {
                BEAT => continue,
                OK | ERR => {
                    let mut len = [0u8; 8];
                    r.read_exact(&mut len).map_err(io)?;
                    let body = read_vec(&mut r, u64::from_le_bytes(len) as usize).map_err(io)?;
                    if tag[0] == OK {
                        return Ok(body);
                    }
                    bail!("{}: {}", self.addr, String::from_utf8_lossy(&body));
                }
                t => bail!("{}: protocol error (frame tag {t})", self.addr),
            }
        }
    }

    /// Ask the server to stop.
    pub fn shutdown(&mut self) -> Result<()> {
        self.call(OP_SHUTDOWN, &[]).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_round_trips_and_checks_lengths() {
        let h = vec![1.0f32, 2.0];
        let side = vec![3.0f32];
        let e = Enc::new()
            .u64(7)
            .str("hi")
            .vecs([h.as_slice(), side.as_slice()]);
        let mut d = Dec(&e.0);
        assert_eq!(d.u64().unwrap(), 7);
        assert_eq!(d.str().unwrap(), "hi");
        assert_eq!(d.vecs().unwrap(), vec![h, side]);
        assert_eq!(d.remaining(), 0);
        assert!(Dec(&e.0[..10]).vecs().is_err());
    }

    fn spawn(handler: impl RpcHandler + 'static) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        std::thread::spawn(move || serve(l, handler, 0).unwrap());
        addr
    }

    #[test]
    fn errors_and_panics_come_back_and_the_server_keeps_serving() {
        let addr = spawn(|op: u8, p: &[u8]| -> Result<Vec<u8>> {
            match op {
                1 => Ok(p.to_vec()),
                2 => bail!("refused"),
                _ => panic!("boom"),
            }
        });
        let mut c = RpcClient::connect(&addr, &RpcOptions::default()).unwrap();
        assert_eq!(c.call(1, b"abc").unwrap(), b"abc");
        assert!(c.call(2, &[]).unwrap_err().to_string().contains("refused"));
        assert!(
            c.call(3, &[])
                .unwrap_err()
                .to_string()
                .contains("panic: boom")
        );
        // A second client is served while the first stays connected.
        let mut c2 = RpcClient::connect(&addr, &RpcOptions::default()).unwrap();
        assert_eq!(c2.call(1, b"x").unwrap(), b"x");
        assert_eq!(c.call(1, b"y").unwrap(), b"y");
        c.shutdown().unwrap();
    }

    #[test]
    fn a_slow_request_heartbeats_and_a_silent_peer_times_out() {
        // The request outlasts the client's timeout; it succeeds only because the
        // server heartbeats (every 5 s) while working.
        let addr = spawn(|_: u8, _: &[u8]| -> Result<Vec<u8>> {
            std::thread::sleep(Duration::from_secs(7));
            Ok(vec![9])
        });
        let opts = RpcOptions {
            timeout: Duration::from_secs(6),
            ..RpcOptions::default()
        };
        let mut c = RpcClient::connect(&addr, &opts).unwrap();
        assert_eq!(c.call(1, &[]).unwrap(), vec![9]);

        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let silent = l.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            let (_s, _) = l.accept().unwrap();
            std::thread::sleep(Duration::from_secs(30));
        });
        let quick = RpcOptions {
            timeout: Duration::from_millis(300),
            ..RpcOptions::default()
        };
        let mut c = RpcClient::connect(&silent, &quick).unwrap();
        let t = std::time::Instant::now();
        let e = c.call(1, &[]).unwrap_err().to_string();
        assert!(t.elapsed() < Duration::from_secs(5));
        assert!(e.contains("stopped responding"), "{e}");
    }
}
