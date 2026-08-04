// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HTTP(S) Range reader for remote flat `.rlxp` (`remote` feature).

use anyhow::{Context, Result, bail};
use std::io::Read;

/// Fetch a byte range from `url` (`Range: bytes=start-end` inclusive).
#[cfg(feature = "remote")]
pub fn http_range(url: &str, start: u64, end_inclusive: u64) -> Result<Vec<u8>> {
    let range = format!("bytes={start}-{end_inclusive}");
    let resp = ureq::get(url)
        .set("Range", &range)
        .call()
        .with_context(|| format!("GET {url} Range {range}"))?;
    let status = resp.status();
    if status != 206 && status != 200 {
        bail!("unexpected HTTP {status} for Range request");
    }
    let mut buf = Vec::new();
    resp.into_reader()
        .take(end_inclusive.saturating_sub(start).saturating_add(1) + 64)
        .read_to_end(&mut buf)
        .context("read range body")?;
    Ok(buf)
}

#[cfg(not(feature = "remote"))]
pub fn http_range(_url: &str, _start: u64, _end_inclusive: u64) -> Result<Vec<u8>> {
    bail!("rlx-pkg was built without the `remote` feature")
}

/// Remote flat-pack handle: TOC in memory; tensor bytes fetched on demand.
#[cfg(feature = "remote")]
pub struct RemoteFlat {
    pub url: String,
    pub header: crate::flat::FlatHeader,
    pub toc: crate::flat::FlatToc,
    pub data_start: u64,
}

#[cfg(feature = "remote")]
impl RemoteFlat {
    /// Fetch header + TOC only (does not pull the data region).
    pub fn open(url: &str) -> Result<Self> {
        use crate::flat::{DATA_ALIGN, FLAT_MAGIC, FlatHeader, FlatToc};
        let prefix = http_range(url, 0, 23)?;
        if prefix.len() < FlatHeader::SIZE || &prefix[..8] != FLAT_MAGIC {
            bail!("remote URL is not RLXPFLAT");
        }
        let header = FlatHeader::decode(&prefix)?;
        let toc_end = (FlatHeader::SIZE as u64) + header.toc_len - 1;
        let hdr_toc = http_range(url, 0, toc_end)?;
        // The server's response length is untrusted: a short reply (fewer bytes
        // than the header-declared TOC) would panic the slice below. Bounds-check
        // it (checked_add guards the usize overflow) and error cleanly instead.
        let toc_end_idx = match FlatHeader::SIZE.checked_add(header.toc_len as usize) {
            Some(e) if e <= hdr_toc.len() => e,
            _ => bail!(
                "remote TOC truncated: header declares {} TOC bytes but the server returned {}",
                header.toc_len,
                hdr_toc.len()
            ),
        };
        let toc_bytes = &hdr_toc[FlatHeader::SIZE..toc_end_idx];
        let mut toc: FlatToc = if header.is_bincode_toc() {
            let wire: crate::flat::FlatTocBin = bincode::deserialize(toc_bytes)?;
            wire.into_toc()?
        } else {
            serde_json::from_slice(toc_bytes)?
        };
        toc.resolve_names()?;
        toc.manifest.validate()?;
        let data_start =
            (FlatHeader::SIZE as u64 + header.toc_len + DATA_ALIGN - 1) & !(DATA_ALIGN - 1);
        Ok(Self {
            url: url.to_string(),
            header,
            toc,
            data_start,
        })
    }

    /// Fetch and decode one tensor by name.
    pub fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        use crate::tier::decode_payload;
        let t = self
            .toc
            .tensors
            .iter()
            .find(|t| t.name == name)
            .with_context(|| format!("tensor {name}"))?;
        let abs0 = self.data_start + t.offset;
        let abs1 = abs0 + t.length - 1;
        let stored = http_range(&self.url, abs0, abs1)?;
        if stored.len() as u64 != t.length {
            bail!("range length {} != TOC length {}", stored.len(), t.length);
        }
        decode_payload(t.codec, &stored)
    }
}
