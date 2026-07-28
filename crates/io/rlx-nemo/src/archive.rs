// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Self-contained, seek-based readers for the two container formats a
//! `.nemo` file is built from:
//!
//!   * an (uncompressed) **TAR** archive — the outer `.nemo` wrapper, and
//!   * a **ZIP** archive with STORED (method 0) entries — the inner
//!     `model_weights.ckpt`, which is exactly what `torch.save` emits.
//!
//! Both operate directly on a [`std::fs::File`] with `seek`/`read` so we
//! never have to slurp a multi-gigabyte checkpoint into RAM; callers pull
//! out only the few members / tensor storages they actually need.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};

// ─────────────────────────────────────────────────────────────────────
// gzip handling — `.nemo` files may be gzip-compressed tars
// ─────────────────────────────────────────────────────────────────────

/// A `.nemo` made seekable: either the file itself (plain tar) or a
/// temporary decompressed copy (gzip tar), removed on drop.
pub enum Seekable {
    Direct(PathBuf),
    Temp(TempFile),
}

impl Seekable {
    pub fn path(&self) -> &Path {
        match self {
            Seekable::Direct(p) => p,
            Seekable::Temp(t) => &t.path,
        }
    }
}

/// A temp file deleted when dropped.
pub struct TempFile {
    pub path: PathBuf,
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Return a seekable handle for `path`: if it is gzip-compressed (magic
/// `1f 8b`), stream-decompress it to a temp file once; otherwise use it
/// directly. Decompression streams, so a multi-GB checkpoint never lands
/// in RAM all at once.
pub fn prepare_seekable(path: &Path) -> Result<Seekable> {
    let mut f = File::open(path)?;
    let mut magic = [0u8; 2];
    if f.read(&mut magic)? < 2 || magic != [0x1f, 0x8b] {
        return Ok(Seekable::Direct(path.to_path_buf()));
    }
    f.seek(SeekFrom::Start(0))?;
    let stem = path.file_name().and_then(|s| s.to_str()).unwrap_or("nemo");
    let tmp = std::env::temp_dir().join(format!("rlx-nemo-{}-{}.tar", std::process::id(), stem));
    // MultiGzDecoder also handles the common single-stream case.
    let mut dec = flate2::read::MultiGzDecoder::new(BufReader::new(f));
    let mut out = File::create(&tmp).map_err(|e| anyhow!("create temp {}: {e}", tmp.display()))?;
    std::io::copy(&mut dec, &mut out)
        .map_err(|e| anyhow!("decompress {} -> {}: {e}", path.display(), tmp.display()))?;
    Ok(Seekable::Temp(TempFile { path: tmp }))
}

// ─────────────────────────────────────────────────────────────────────
// TAR
// ─────────────────────────────────────────────────────────────────────

/// A regular-file member located inside a tar archive: where its data
/// starts (absolute byte offset in the file) and how many bytes it is.
#[derive(Debug, Clone)]
pub struct TarMember {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

#[inline]
fn round_up_512(n: u64) -> u64 {
    n.div_ceil(512) * 512
}

fn parse_cstr(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Parse a base-8 numeric tar header field (NUL/space padded).
fn parse_octal(buf: &[u8]) -> Result<u64> {
    // GNU base-256 extension: high bit of first byte set.
    if buf.first().is_some_and(|&b| b & 0x80 != 0) {
        let mut v: u64 = u64::from(buf[0] & 0x7f);
        for &b in &buf[1..] {
            v = (v << 8) | u64::from(b);
        }
        return Ok(v);
    }
    let s: String = buf
        .iter()
        .map(|&b| b as char)
        .filter(|c| ('0'..='7').contains(c))
        .collect();
    if s.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(&s, 8).map_err(|e| anyhow!("bad octal tar field {s:?}: {e}"))
}

/// Pax extended-header records are `"<len> key=value\n"`. Pull out `path`.
fn parse_pax_path(buf: &[u8]) -> Option<String> {
    let mut p = 0usize;
    while p < buf.len() {
        // record length runs up to the first space.
        let sp = buf[p..].iter().position(|&b| b == b' ')? + p;
        let len: usize = std::str::from_utf8(&buf[p..sp]).ok()?.trim().parse().ok()?;
        if len == 0 || p + len > buf.len() {
            break;
        }
        let record = &buf[sp + 1..p + len];
        if let Some(eq) = record.iter().position(|&b| b == b'=') {
            let key = &record[..eq];
            if key == b"path" {
                let mut val = &record[eq + 1..];
                if val.last() == Some(&b'\n') {
                    val = &val[..val.len() - 1];
                }
                return Some(String::from_utf8_lossy(val).into_owned());
            }
        }
        p += len;
    }
    None
}

/// List every regular-file member of an uncompressed tar archive.
///
/// Handles the `ustar` `prefix` field, GNU long names (`L`), and pax
/// extended headers (`x`) so NeMo's `././@PaxHeader`-prefixed entries
/// resolve to clean names like `model_weights.ckpt`.
pub fn list_tar(file: &mut File) -> Result<Vec<TarMember>> {
    let total = file.metadata()?.len();
    let mut members = Vec::new();
    let mut pos: u64 = 0;
    let mut pax_path: Option<String> = None;
    let mut gnu_longname: Option<String> = None;

    while pos + 512 <= total {
        file.seek(SeekFrom::Start(pos))?;
        let mut hdr = [0u8; 512];
        file.read_exact(&mut hdr)?;

        // A run of zero blocks marks the end of the archive.
        if hdr.iter().all(|&b| b == 0) {
            break;
        }

        let size = parse_octal(&hdr[124..136])?;
        let typeflag = hdr[156];
        let data_off = pos + 512;

        match typeflag {
            b'x' | b'X' => {
                let mut data = vec![0u8; size as usize];
                file.seek(SeekFrom::Start(data_off))?;
                file.read_exact(&mut data)?;
                pax_path = parse_pax_path(&data);
                pos = data_off + round_up_512(size);
                continue;
            }
            b'L' => {
                let mut data = vec![0u8; size as usize];
                file.seek(SeekFrom::Start(data_off))?;
                file.read_exact(&mut data)?;
                gnu_longname = Some(parse_cstr(&data));
                pos = data_off + round_up_512(size);
                continue;
            }
            b'g' => {
                // Global pax header — not relevant to a single .nemo.
                pos = data_off + round_up_512(size);
                continue;
            }
            _ => {}
        }

        let name = parse_cstr(&hdr[0..100]);
        let prefix = parse_cstr(&hdr[345..500]);
        let mut full = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        if let Some(p) = pax_path.take() {
            full = p;
        }
        if let Some(p) = gnu_longname.take() {
            full = p;
        }
        let norm = full.trim_start_matches("./").to_string();

        // typeflag '0' or NUL == regular file.
        if typeflag == b'0' || typeflag == 0 {
            members.push(TarMember {
                name: norm,
                offset: data_off,
                size,
            });
        }
        pos = data_off + round_up_512(size);
    }
    Ok(members)
}

/// Read a small tar member fully into memory (config / tokenizer files).
pub fn read_member(file: &mut File, m: &TarMember) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; m.size as usize];
    file.seek(SeekFrom::Start(m.offset))?;
    file.read_exact(&mut buf)?;
    Ok(buf)
}

// ─────────────────────────────────────────────────────────────────────
// ZIP (STORED entries only — what torch.save writes)
// ─────────────────────────────────────────────────────────────────────

/// A single entry of the inner `.ckpt` zip, with the **absolute** byte
/// offset of its payload within the enclosing `.nemo` file.
#[derive(Debug, Clone)]
pub struct ZipEntry {
    pub name: String,
    pub data_offset: u64,
    pub size: u64,
    pub method: u16,
}

const EOCD_SIG: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
const CDH_SIG: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];

/// List the entries of a zip archive embedded in `file` at absolute byte
/// range `[zip_start, zip_start + zip_size)`. Parses the central
/// directory; offsets are resolved to absolute file positions.
pub fn list_zip(file: &mut File, zip_start: u64, zip_size: u64) -> Result<Vec<ZipEntry>> {
    if zip_size < 22 {
        bail!("zip region too small ({zip_size} bytes)");
    }
    // Scan backwards for the End Of Central Directory record. Its max
    // distance from EOF is 22 (fixed part) + 65535 (max comment).
    let max_back = (22 + 65535u64).min(zip_size);
    let scan_start = zip_start + zip_size - max_back;
    let mut tail = vec![0u8; max_back as usize];
    file.seek(SeekFrom::Start(scan_start))?;
    file.read_exact(&mut tail)?;

    let eocd = (0..=tail.len() - 4)
        .rev()
        .find(|&i| tail[i..i + 4] == EOCD_SIG)
        .ok_or_else(|| anyhow!("zip: End Of Central Directory not found"))?;
    let e = &tail[eocd..];
    if e.len() < 22 {
        bail!("zip: truncated EOCD");
    }
    let cd_count = u16::from_le_bytes([e[10], e[11]]) as usize;
    let cd_offset_rel = u32::from_le_bytes([e[16], e[17], e[18], e[19]]) as u64;
    if cd_offset_rel == 0xFFFF_FFFF || cd_count == 0xFFFF {
        bail!("zip: ZIP64 archives are not supported");
    }

    let cd_abs = zip_start + cd_offset_rel;
    // The central directory runs from cd_abs up to the EOCD record.
    let cd_end = scan_start + eocd as u64;
    let cd_len = cd_end
        .checked_sub(cd_abs)
        .ok_or_else(|| anyhow!("zip: central directory offset past EOCD"))?;
    let mut cd = vec![0u8; cd_len as usize];
    file.seek(SeekFrom::Start(cd_abs))?;
    file.read_exact(&mut cd)?;

    let mut entries = Vec::with_capacity(cd_count);
    let mut p = 0usize;
    for _ in 0..cd_count {
        if p + 46 > cd.len() || cd[p..p + 4] != CDH_SIG {
            bail!("zip: malformed central directory header at {p}");
        }
        let method = u16::from_le_bytes([cd[p + 10], cd[p + 11]]);
        let comp_size = u32::from_le_bytes([cd[p + 20], cd[p + 21], cd[p + 22], cd[p + 23]]) as u64;
        let name_len = u16::from_le_bytes([cd[p + 28], cd[p + 29]]) as usize;
        let extra_len = u16::from_le_bytes([cd[p + 30], cd[p + 31]]) as usize;
        let comment_len = u16::from_le_bytes([cd[p + 32], cd[p + 33]]) as usize;
        let lh_off_rel =
            u32::from_le_bytes([cd[p + 42], cd[p + 43], cd[p + 44], cd[p + 45]]) as u64;
        if lh_off_rel == 0xFFFF_FFFF {
            bail!("zip: ZIP64 local-header offset not supported");
        }
        let name = String::from_utf8_lossy(&cd[p + 46..p + 46 + name_len]).into_owned();

        // The local header's name/extra lengths can differ from the
        // central directory's, so read it to find the real data offset.
        let lh_abs = zip_start + lh_off_rel;
        let mut lh = [0u8; 30];
        file.seek(SeekFrom::Start(lh_abs))?;
        file.read_exact(&mut lh)?;
        if lh[0..4] != [0x50, 0x4b, 0x03, 0x04] {
            bail!("zip: bad local file header for {name}");
        }
        let lh_name_len = u16::from_le_bytes([lh[26], lh[27]]) as u64;
        let lh_extra_len = u16::from_le_bytes([lh[28], lh[29]]) as u64;
        let data_offset = lh_abs + 30 + lh_name_len + lh_extra_len;

        entries.push(ZipEntry {
            name,
            data_offset,
            size: comp_size,
            method,
        });
        p += 46 + name_len + extra_len + comment_len;
    }
    Ok(entries)
}

/// Read a STORED zip entry's raw bytes.
pub fn read_zip_entry(file: &mut File, entry: &ZipEntry) -> Result<Vec<u8>> {
    if entry.method != 0 {
        bail!(
            "zip entry {} uses compression method {} (only STORED/0 is supported; \
             torch.save writes STORED)",
            entry.name,
            entry.method
        );
    }
    let mut buf = vec![0u8; entry.size as usize];
    file.seek(SeekFrom::Start(entry.data_offset))?;
    file.read_exact(&mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octal_parsing() {
        assert_eq!(parse_octal(b"0000644 \0").unwrap(), 0o644);
        assert_eq!(parse_octal(b"00000001750\0").unwrap(), 0o1750);
        // base-256 GNU extension: 0x80 marker + big-endian value.
        assert_eq!(
            parse_octal(&[0x80, 0, 0, 0, 0, 0, 0x01, 0x00]).unwrap(),
            256
        );
    }

    #[test]
    fn pax_path_record() {
        // The length prefix counts the entire record including itself,
        // the space and the trailing newline: "27 path=...\n" == 27 bytes.
        let rec = b"27 path=model_weights.ckpt\n13 foo=barbaz\n";
        assert_eq!(parse_pax_path(rec).as_deref(), Some("model_weights.ckpt"));
    }
}
