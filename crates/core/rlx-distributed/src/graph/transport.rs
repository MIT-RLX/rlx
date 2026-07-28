// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-node execution of a partitioned model over TCP. Each machine runs
//! [`serve_stage`] for one [`Stage`] (holding only that stage's weights); a
//! coordinator runs [`run_pipeline_tcp`], relaying the small named boundary
//! tensors from stage to stage. Peak weight memory per machine = one stage's
//! parameters, which is the whole point: split a model across boxes none of
//! which could hold it alone.
//!
//! Wire format (length-prefixed, little-endian): `u32 count`, then per tensor
//! `u32 name_len | name | u32 rank | rank×(u64 dim) | u64 numel | numel×f32`.

use super::partition::Stage;
use super::pipeline::{NamedTensor, StageRunner};
use crate::source::ParamSource;
use rlx_runtime::{CompileOptions, Device};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};

fn write_tensors<W: Write>(w: &mut W, ts: &[NamedTensor]) -> io::Result<()> {
    w.write_all(&(ts.len() as u32).to_le_bytes())?;
    for t in ts {
        let nb = t.name.as_bytes();
        w.write_all(&(nb.len() as u32).to_le_bytes())?;
        w.write_all(nb)?;
        w.write_all(&(t.shape.len() as u32).to_le_bytes())?;
        for &d in &t.shape {
            w.write_all(&(d as u64).to_le_bytes())?;
        }
        w.write_all(&(t.data.len() as u64).to_le_bytes())?;
        let mut buf = Vec::with_capacity(t.data.len() * 4);
        for &v in &t.data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        w.write_all(&buf)?;
    }
    w.flush()
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_tensors<R: Read>(r: &mut R) -> io::Result<Vec<NamedTensor>> {
    let count = read_u32(r)? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let name_len = read_u32(r)? as usize;
        let mut nb = vec![0u8; name_len];
        r.read_exact(&mut nb)?;
        let name = String::from_utf8(nb).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let rank = read_u32(r)? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(read_u64(r)? as usize);
        }
        let numel = read_u64(r)? as usize;
        let mut db = vec![0u8; numel * 4];
        r.read_exact(&mut db)?;
        let data = db
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.push(NamedTensor { name, shape, data });
    }
    Ok(out)
}

/// Serve one stage on `addr` for exactly `n_requests` forwards, then return.
/// Compiles the stage + loads its parameter shard once, then for each incoming
/// connection reads the stage's input tensors, runs, and writes the outputs.
/// (`n_requests = 0` serves until the listener errors — e.g. a long-lived node.)
pub fn serve_stage(
    addr: impl ToSocketAddrs,
    stage: Stage,
    source: &mut dyn ParamSource,
    device: Device,
    opts: &CompileOptions,
    n_requests: usize,
) -> io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    // Compile (allocate + load the arena) BEFORE announcing readiness. On a node
    // whose stage loads slowly (an oversubscribed GPU paging its weights in), the
    // compile can take a while; a coordinator that drove on a premature "serving
    // on" would hit the stage mid-load and break the pipe. Announce only once the
    // arena is resident so the forward runs against a ready stage.
    let bound = listener.local_addr()?;
    let mut runner = StageRunner::compile(stage, source, device, opts);
    println!("serving on {bound}");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let mut served = 0usize;
    for conn in listener.incoming() {
        let mut sock = conn?;
        let inputs = read_tensors(&mut sock)?;
        let pool: HashMap<String, NamedTensor> =
            inputs.into_iter().map(|t| (t.name.clone(), t)).collect();
        let outs = runner.run(&pool);
        write_tensors(&mut sock, &outs)?;
        served += 1;
        if n_requests != 0 && served >= n_requests {
            break;
        }
    }
    Ok(())
}

/// Bind a stage listener on an ephemeral port and return `(bound_addr, listener)`
/// so a coordinator can learn the address before serving begins (useful for
/// tests / dynamic clusters). Pair with [`serve_bound`].
pub fn bind_stage(addr: impl ToSocketAddrs) -> io::Result<(std::net::SocketAddr, TcpListener)> {
    let listener = TcpListener::bind(addr)?;
    let bound = listener.local_addr()?;
    Ok((bound, listener))
}

/// Serve on an already-bound listener (see [`bind_stage`]).
pub fn serve_bound(
    listener: TcpListener,
    stage: Stage,
    source: &mut dyn ParamSource,
    device: Device,
    opts: &CompileOptions,
    n_requests: usize,
) -> io::Result<()> {
    let mut runner = StageRunner::compile(stage, source, device, opts);
    let mut served = 0usize;
    for conn in listener.incoming() {
        let mut sock = conn?;
        let inputs = read_tensors(&mut sock)?;
        let pool: HashMap<String, NamedTensor> =
            inputs.into_iter().map(|t| (t.name.clone(), t)).collect();
        let outs = runner.run(&pool);
        write_tensors(&mut sock, &outs)?;
        served += 1;
        if n_requests != 0 && served >= n_requests {
            break;
        }
    }
    Ok(())
}

/// Coordinator: drive one forward through stages served at `worker_addrs[i]`.
/// `stages[i]` supplies only the boundary NAME metadata (its `inputs`/`outputs`)
/// — the coordinator holds no weights, just relays the pool. `inputs` seeds the
/// model's `Op::Input` tensors. Returns the final stage's outputs (logits).
pub fn run_pipeline_tcp(
    stages: &[Stage],
    worker_addrs: &[String],
    inputs: Vec<NamedTensor>,
) -> io::Result<Vec<NamedTensor>> {
    assert_eq!(stages.len(), worker_addrs.len(), "one worker addr per stage");
    let mut pool: HashMap<String, NamedTensor> =
        inputs.into_iter().map(|t| (t.name.clone(), t)).collect();
    let mut last = Vec::new();
    for (stage, addr) in stages.iter().zip(worker_addrs) {
        let feed: Vec<NamedTensor> = stage
            .inputs
            .iter()
            .map(|n| {
                pool.get(n)
                    .unwrap_or_else(|| panic!("coordinator: missing tensor `{n}` for stage {}", stage.index))
                    .clone()
            })
            .collect();
        let mut sock = connect_retry(addr)?;
        write_tensors(&mut sock, &feed)?;
        let outs = read_tensors(&mut sock)?;
        for t in &outs {
            pool.insert(t.name.clone(), t.clone());
        }
        last = outs;
    }
    Ok(last)
}

/// Like [`run_pipeline_tcp`] but also returns per-stage wall time (ms) — the
/// connect+send+run+recv latency of each worker — for the monitor.
pub fn run_pipeline_tcp_timed(
    stages: &[Stage],
    worker_addrs: &[String],
    inputs: Vec<NamedTensor>,
) -> io::Result<(Vec<NamedTensor>, Vec<u64>)> {
    assert_eq!(stages.len(), worker_addrs.len(), "one worker addr per stage");
    let mut pool: HashMap<String, NamedTensor> = inputs.into_iter().map(|t| (t.name.clone(), t)).collect();
    let mut last = Vec::new();
    let mut times = Vec::with_capacity(stages.len());
    for (stage, addr) in stages.iter().zip(worker_addrs) {
        let feed: Vec<NamedTensor> = stage
            .inputs
            .iter()
            .map(|n| pool.get(n).unwrap_or_else(|| panic!("coordinator: missing tensor `{n}` for stage {}", stage.index)).clone())
            .collect();
        let t0 = std::time::Instant::now();
        let mut sock = connect_retry(addr)?;
        write_tensors(&mut sock, &feed)?;
        let outs = read_tensors(&mut sock)?;
        times.push(t0.elapsed().as_millis() as u64);
        for t in &outs {
            pool.insert(t.name.clone(), t.clone());
        }
        last = outs;
    }
    Ok((last, times))
}

/// Connect with a short retry loop (workers may still be binding).
fn connect_retry(addr: &str) -> io::Result<TcpStream> {
    let mut err = None;
    for _ in 0..100 {
        match TcpStream::connect(addr) {
            Ok(s) => return Ok(s),
            Err(e) => {
                err = Some(e);
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
    }
    Err(err.unwrap_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "connect_retry exhausted")))
}
