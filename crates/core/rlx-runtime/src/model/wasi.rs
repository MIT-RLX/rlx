// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The WASI (`wasi_snapshot_preview1`) a program built for `wasm32-wasip1`
//! gets: what Rust's `std` reads files with, over the package itself.
//!
//! The package's files are mounted read-only at `/package` (sidecar `a/b`
//! at `/package/a/b`, its directories those the ids imply), loaded from
//! the package when first opened — a phonemizer's dictionaries, a lexicon,
//! a tokenizer — and nothing else: no host file is reachable. Writing a
//! file is `EROFS`; stdout and stderr go to the program log
//! (`RLX_PROGRAM_LOG`); `random_get` is a fixed sequence (a program is
//! deterministic); the clock is the host's; there are no args and no
//! environment.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use wasmi::{Caller, Error, Extern, Linker};

use super::program::Host;

/// Where the package is mounted.
pub(super) const MOUNT: &str = "/package";

const SUCCESS: i32 = 0;
const EBADF: i32 = 8;
const EINVAL: i32 = 28;
const EISDIR: i32 = 31;
const ENOENT: i32 = 44;
const ENOTDIR: i32 = 54;
const EROFS: i32 = 69;
const EFAULT: i32 = 21;

const FILETYPE_CHARACTER_DEVICE: u8 = 2;
const FILETYPE_DIRECTORY: u8 = 3;
const FILETYPE_REGULAR_FILE: u8 = 4;

/// The first descriptor past stdin/out/err: the mount.
const ROOT_FD: i32 = 3;

enum Open {
    /// A directory: its path under the mount (`""` the mount itself).
    Dir(String),
    File {
        data: Arc<Vec<u8>>,
        pos: u64,
    },
}

/// A program's descriptors over the package's files.
pub(super) struct Wasi {
    /// Every file, by its path under the mount.
    files: BTreeSet<String>,
    loaded: HashMap<String, Arc<Vec<u8>>>,
    fds: HashMap<i32, Open>,
    next: i32,
    random: u64,
}

impl Wasi {
    pub(super) fn new(sidecars: impl IntoIterator<Item = String>) -> Self {
        let mut fds = HashMap::new();
        fds.insert(ROOT_FD, Open::Dir(String::new()));
        Self {
            files: sidecars.into_iter().collect(),
            loaded: HashMap::new(),
            fds,
            next: ROOT_FD + 1,
            random: 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn is_dir(&self, path: &str) -> bool {
        path.is_empty() || {
            let prefix = format!("{path}/");
            self.files
                .range(prefix.clone()..)
                .next()
                .is_some_and(|f| f.starts_with(&prefix))
        }
    }

    /// The names directly in the directory `path`, each a file or not.
    fn list(&self, path: &str) -> Vec<(String, bool)> {
        let prefix = if path.is_empty() {
            String::new()
        } else {
            format!("{path}/")
        };
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for f in self.files.iter().filter(|f| f.starts_with(&prefix)) {
            let rest = &f[prefix.len()..];
            let (name, file) = match rest.split_once('/') {
                Some((d, _)) => (d, false),
                None => (rest, true),
            };
            if seen.insert(name.to_string()) {
                out.push((name.to_string(), file));
            }
        }
        out
    }
}

/// `path` under the directory `base`, `.` and `..` resolved; `None` past
/// the mount.
fn join(base: &str, path: &str) -> Option<String> {
    let mut parts: Vec<&str> = if path.starts_with('/') {
        Vec::new()
    } else {
        base.split('/').filter(|s| !s.is_empty()).collect()
    };
    let path = path
        .strip_prefix(MOUNT)
        .filter(|_| path.starts_with('/'))
        .unwrap_or(path);
    for p in path.split('/') {
        match p {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            p => parts.push(p),
        }
    }
    Some(parts.join("/"))
}

fn memory(caller: &Caller<'_, Host>) -> Option<wasmi::Memory> {
    caller.get_export("memory").and_then(Extern::into_memory)
}

fn read_mem(caller: &Caller<'_, Host>, ptr: i32, len: usize) -> Option<Vec<u8>> {
    let mut buf = vec![0u8; len];
    memory(caller)?
        .read(caller, ptr as u32 as usize, &mut buf)
        .ok()?;
    Some(buf)
}

fn write_mem(caller: &mut Caller<'_, Host>, ptr: i32, bytes: &[u8]) -> bool {
    match memory(caller) {
        Some(m) => m.write(caller, ptr as u32 as usize, bytes).is_ok(),
        None => false,
    }
}

fn write_u32(caller: &mut Caller<'_, Host>, ptr: i32, v: u32) -> bool {
    write_mem(caller, ptr, &v.to_le_bytes())
}

fn write_u64(caller: &mut Caller<'_, Host>, ptr: i32, v: u64) -> bool {
    write_mem(caller, ptr, &v.to_le_bytes())
}

/// The `(buf, len)` pairs of an iovec list.
fn iovecs(caller: &Caller<'_, Host>, iovs: i32, n: i32) -> Option<Vec<(i32, usize)>> {
    let raw = read_mem(caller, iovs, n as usize * 8)?;
    Some(
        raw.chunks_exact(8)
            .map(|c| {
                (
                    i32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                    u32::from_le_bytes([c[4], c[5], c[6], c[7]]) as usize,
                )
            })
            .collect(),
    )
}

fn path_arg(caller: &Caller<'_, Host>, ptr: i32, len: i32) -> Option<String> {
    String::from_utf8(read_mem(caller, ptr, len as usize)?).ok()
}

/// A `filestat`: 64 bytes (dev, ino, filetype, nlink, size, atim, mtim, ctim).
fn filestat(filetype: u8, size: u64) -> [u8; 64] {
    let mut b = [0u8; 64];
    b[16] = filetype;
    b[24..32].copy_from_slice(&1u64.to_le_bytes());
    b[32..40].copy_from_slice(&size.to_le_bytes());
    b
}

impl Host {
    fn load(&mut self, path: &str) -> Result<Arc<Vec<u8>>, i32> {
        if let Some(d) = self.wasi.loaded.get(path) {
            return Ok(d.clone());
        }
        if !self.wasi.files.contains(path) {
            return Err(ENOENT);
        }
        let bytes = self
            .pkg
            .file(&format!("sidecar:{path}"))
            .map_err(|_| ENOENT)?;
        let d = Arc::new(bytes);
        self.wasi.loaded.insert(path.to_string(), d.clone());
        Ok(d)
    }

    /// The path `p` names relative to the directory `fd`.
    fn resolve(&self, fd: i32, p: &str) -> Result<String, i32> {
        match self.wasi.fds.get(&fd) {
            Some(Open::Dir(base)) => join(base, p).ok_or(ENOENT),
            Some(Open::File { .. }) => Err(ENOTDIR),
            None => Err(EBADF),
        }
    }

    fn size(&mut self, path: &str) -> Result<(u8, u64), i32> {
        if self.wasi.is_dir(path) {
            return Ok((FILETYPE_DIRECTORY, 0));
        }
        let d = self.load(path)?;
        Ok((FILETYPE_REGULAR_FILE, d.len() as u64))
    }
}

/// Link the WASI subset into `linker` (for modules that import it).
pub(super) fn link(linker: &mut Linker<Host>) -> Result<(), wasmi::Error> {
    const M: &str = "wasi_snapshot_preview1";
    linker.func_wrap(
        M,
        "args_sizes_get",
        |mut c: Caller<'_, Host>, n: i32, s: i32| {
            if write_u32(&mut c, n, 0) && write_u32(&mut c, s, 0) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(M, "args_get", |_: Caller<'_, Host>, _: i32, _: i32| SUCCESS)?;
    linker.func_wrap(
        M,
        "environ_sizes_get",
        |mut c: Caller<'_, Host>, n: i32, s: i32| {
            if write_u32(&mut c, n, 0) && write_u32(&mut c, s, 0) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(M, "environ_get", |_: Caller<'_, Host>, _: i32, _: i32| {
        SUCCESS
    })?;
    linker.func_wrap(
        M,
        "random_get",
        |mut c: Caller<'_, Host>, buf: i32, len: i32| {
            let mut out = vec![0u8; len as usize];
            let mut s = c.data().wasi.random;
            for b in &mut out {
                // xorshift64*
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                *b = (s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 56) as u8;
            }
            c.data_mut().wasi.random = s;
            if write_mem(&mut c, buf, &out) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(
        M,
        "clock_time_get",
        |mut c: Caller<'_, Host>, _id: i32, _precision: i64, out: i32| {
            let ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64);
            if write_u64(&mut c, out, ns) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(M, "sched_yield", |_: Caller<'_, Host>| SUCCESS)?;
    linker.func_wrap(
        M,
        "proc_exit",
        |_: Caller<'_, Host>, code: i32| -> Result<(), Error> { Err(Error::i32_exit(code)) },
    )?;
    linker.func_wrap(
        M,
        "fd_prestat_get",
        |mut c: Caller<'_, Host>, fd: i32, out: i32| {
            if fd != ROOT_FD {
                return EBADF;
            }
            let mut b = [0u8; 8];
            b[4..8].copy_from_slice(&(MOUNT.len() as u32).to_le_bytes());
            if write_mem(&mut c, out, &b) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(
        M,
        "fd_prestat_dir_name",
        |mut c: Caller<'_, Host>, fd: i32, out: i32, len: i32| {
            if fd != ROOT_FD {
                return EBADF;
            }
            let name = &MOUNT.as_bytes()[..(len as usize).min(MOUNT.len())];
            if write_mem(&mut c, out, name) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(
        M,
        "fd_fdstat_get",
        |mut c: Caller<'_, Host>, fd: i32, out: i32| {
            let filetype = match fd {
                0..=2 => FILETYPE_CHARACTER_DEVICE,
                _ => match c.data().wasi.fds.get(&fd) {
                    Some(Open::Dir(_)) => FILETYPE_DIRECTORY,
                    Some(Open::File { .. }) => FILETYPE_REGULAR_FILE,
                    None => return EBADF,
                },
            };
            let mut b = [0u8; 24];
            b[0] = filetype;
            b[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
            b[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
            if write_mem(&mut c, out, &b) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(
        M,
        "fd_fdstat_set_flags",
        |_: Caller<'_, Host>, _: i32, _: i32| SUCCESS,
    )?;
    linker.func_wrap(
        M,
        "fd_filestat_get",
        |mut c: Caller<'_, Host>, fd: i32, out: i32| {
            let stat = match fd {
                0..=2 => filestat(FILETYPE_CHARACTER_DEVICE, 0),
                _ => match c.data().wasi.fds.get(&fd) {
                    Some(Open::Dir(_)) => filestat(FILETYPE_DIRECTORY, 0),
                    Some(Open::File { data, .. }) => {
                        filestat(FILETYPE_REGULAR_FILE, data.len() as u64)
                    }
                    None => return EBADF,
                },
            };
            if write_mem(&mut c, out, &stat) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(
        M,
        "path_filestat_get",
        |mut c: Caller<'_, Host>, fd: i32, _flags: i32, p: i32, pl: i32, out: i32| {
            let Some(p) = path_arg(&c, p, pl) else {
                return EFAULT;
            };
            let path = match c.data().resolve(fd, &p) {
                Ok(p) => p,
                Err(e) => return e,
            };
            let (t, n) = match c.data_mut().size(&path) {
                Ok(v) => v,
                Err(e) => return e,
            };
            if write_mem(&mut c, out, &filestat(t, n)) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(
        M,
        "path_open",
        |mut c: Caller<'_, Host>,
         fd: i32,
         _dirflags: i32,
         p: i32,
         pl: i32,
         oflags: i32,
         _rights: i64,
         _inheriting: i64,
         _fdflags: i32,
         out: i32| {
            // O_CREAT, O_EXCL, O_TRUNC: writing.
            if oflags & (1 | 4 | 8) != 0 {
                return EROFS;
            }
            let Some(p) = path_arg(&c, p, pl) else {
                return EFAULT;
            };
            let path = match c.data().resolve(fd, &p) {
                Ok(p) => p,
                Err(e) => return e,
            };
            let directory = oflags & 2 != 0;
            let open = if c.data().wasi.is_dir(&path) {
                Open::Dir(path)
            } else if directory {
                return ENOTDIR;
            } else {
                match c.data_mut().load(&path) {
                    Ok(data) => Open::File { data, pos: 0 },
                    Err(e) => return e,
                }
            };
            let host = c.data_mut();
            let n = host.wasi.next;
            host.wasi.next += 1;
            host.wasi.fds.insert(n, open);
            if write_u32(&mut c, out, n as u32) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(M, "fd_close", |mut c: Caller<'_, Host>, fd: i32| {
        if fd == ROOT_FD || c.data_mut().wasi.fds.remove(&fd).is_some() {
            SUCCESS
        } else {
            EBADF
        }
    })?;
    linker.func_wrap(
        M,
        "fd_read",
        |mut c: Caller<'_, Host>, fd: i32, iovs: i32, n: i32, nread: i32| {
            let Some(vs) = iovecs(&c, iovs, n) else {
                return EFAULT;
            };
            if fd == 0 {
                return if write_u32(&mut c, nread, 0) {
                    SUCCESS
                } else {
                    EFAULT
                };
            }
            let (data, mut pos) = match c.data().wasi.fds.get(&fd) {
                Some(Open::File { data, pos }) => (data.clone(), *pos),
                Some(Open::Dir(_)) => return EISDIR,
                None => return EBADF,
            };
            let mut total = 0usize;
            for (buf, len) in vs {
                let start = (pos as usize).min(data.len());
                let end = (start + len).min(data.len());
                if !write_mem(&mut c, buf, &data[start..end]) {
                    return EFAULT;
                }
                total += end - start;
                pos = end as u64;
                if end - start < len {
                    break;
                }
            }
            if let Some(Open::File { pos: p, .. }) = c.data_mut().wasi.fds.get_mut(&fd) {
                *p = pos;
            }
            if write_u32(&mut c, nread, total as u32) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(
        M,
        "fd_seek",
        |mut c: Caller<'_, Host>, fd: i32, offset: i64, whence: i32, out: i32| {
            let Some(Open::File { data, pos }) = c.data_mut().wasi.fds.get_mut(&fd) else {
                return EBADF;
            };
            let base = match whence {
                0 => 0i64,
                1 => *pos as i64,
                2 => data.len() as i64,
                _ => return EINVAL,
            };
            let new = base + offset;
            if new < 0 {
                return EINVAL;
            }
            *pos = new as u64;
            if write_u64(&mut c, out, new as u64) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(
        M,
        "fd_tell",
        |mut c: Caller<'_, Host>, fd: i32, out: i32| {
            let Some(Open::File { pos, .. }) = c.data().wasi.fds.get(&fd) else {
                return EBADF;
            };
            let pos = *pos;
            if write_u64(&mut c, out, pos) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(
        M,
        "fd_write",
        |mut c: Caller<'_, Host>, fd: i32, iovs: i32, n: i32, nwritten: i32| {
            if fd != 1 && fd != 2 {
                return if c.data().wasi.fds.contains_key(&fd) {
                    EROFS
                } else {
                    EBADF
                };
            }
            let Some(vs) = iovecs(&c, iovs, n) else {
                return EFAULT;
            };
            let mut total = 0usize;
            let mut text = Vec::new();
            for (buf, len) in vs {
                let Some(b) = read_mem(&c, buf, len) else {
                    return EFAULT;
                };
                total += len;
                text.extend(b);
            }
            if c.data().log {
                eprint!("[rlx program] {}", String::from_utf8_lossy(&text));
            }
            if write_u32(&mut c, nwritten, total as u32) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    linker.func_wrap(
        M,
        "fd_readdir",
        |mut c: Caller<'_, Host>, fd: i32, buf: i32, len: i32, cookie: i64, used: i32| {
            let path = match c.data().wasi.fds.get(&fd) {
                Some(Open::Dir(p)) => p.clone(),
                Some(Open::File { .. }) => return ENOTDIR,
                None => return EBADF,
            };
            let entries = c.data().wasi.list(&path);
            let mut out = Vec::new();
            for (i, (name, file)) in entries.iter().enumerate().skip(cookie.max(0) as usize) {
                let mut d = [0u8; 24];
                d[0..8].copy_from_slice(&((i + 1) as u64).to_le_bytes());
                d[8..16].copy_from_slice(&((i + 1) as u64).to_le_bytes());
                d[16..20].copy_from_slice(&(name.len() as u32).to_le_bytes());
                d[20] = if *file {
                    FILETYPE_REGULAR_FILE
                } else {
                    FILETYPE_DIRECTORY
                };
                out.extend_from_slice(&d);
                out.extend_from_slice(name.as_bytes());
                if out.len() >= len as usize {
                    break;
                }
            }
            out.truncate(len as usize);
            if !write_mem(&mut c, buf, &out) {
                return EFAULT;
            }
            if write_u32(&mut c, used, out.len() as u32) {
                SUCCESS
            } else {
                EFAULT
            }
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_resolve_under_the_mount_only() {
        assert_eq!(join("", "a/b").as_deref(), Some("a/b"));
        assert_eq!(join("a", "./b/../c").as_deref(), Some("a/c"));
        assert_eq!(join("a", "/package/x").as_deref(), Some("x"));
        assert_eq!(join("", "..").as_deref(), None);
        let w = Wasi::new([
            "espeak/en_dict".into(),
            "espeak/voices/en".into(),
            "t.json".into(),
        ]);
        assert!(w.is_dir("espeak") && w.is_dir("espeak/voices") && w.is_dir(""));
        assert!(!w.is_dir("t.json") && !w.is_dir("esp"));
        assert_eq!(
            w.list("espeak"),
            [("en_dict".to_string(), true), ("voices".to_string(), false)]
        );
    }
}
