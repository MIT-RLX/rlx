// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ZIP64 STORE writer/reader with absolute data offsets for mmap.

use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Absolute byte range of an uncompressed STORE member inside a zip file.
#[derive(Debug, Clone, Copy)]
pub struct MemberRange {
    pub data_offset: u64,
    pub length: u64,
}

/// Write a ZIP64-capable archive with all members stored uncompressed.
pub fn write_store_zip(
    path: impl AsRef<Path>,
    members: &[(String, Vec<u8>)],
) -> Result<BTreeMap<String, MemberRange>> {
    let path = path.as_ref();
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(true);

    for (name, data) in members {
        zip.start_file(name.as_str(), opts)
            .with_context(|| format!("start_file {name}"))?;
        zip.write_all(data)
            .with_context(|| format!("write member {name}"))?;
    }
    zip.finish().context("finish zip")?;
    resolve_store_ranges(path)
}

/// Map each STORE member name → absolute data offset / length in `path`.
pub fn resolve_store_ranges(path: impl AsRef<Path>) -> Result<BTreeMap<String, MemberRange>> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut archive =
        zip::ZipArchive::new(file).with_context(|| format!("zip {}", path.display()))?;
    let mut out = BTreeMap::new();
    for i in 0..archive.len() {
        let mut zf = archive
            .by_index(i)
            .with_context(|| format!("zip index {i}"))?;
        let name = zf.name().to_string();
        if zf.compression() != zip::CompressionMethod::Stored {
            bail!("member {name} is not STORE; mmap requires uncompressed payloads");
        }
        let length = zf.size();
        let header_start = zf.header_start();
        // Touch the reader so the crate resolves `data_start`.
        let mut probe = [0u8; 1];
        let _ = zf.read(&mut probe);
        let data_offset = zf.data_start();
        // Ensure we didn't mis-parse empty files.
        let _ = header_start;
        out.insert(
            name,
            MemberRange {
                data_offset,
                length,
            },
        );
    }
    Ok(out)
}

/// Read one member's bytes from a zip (owned copy).
#[allow(dead_code)]
pub fn read_member_bytes(path: impl AsRef<Path>, name: &str) -> Result<Vec<u8>> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut archive =
        zip::ZipArchive::new(file).with_context(|| format!("zip {}", path.display()))?;
    let mut zf = archive
        .by_name(name)
        .with_context(|| format!("member {name} in {}", path.display()))?;
    // `zf.size()` is the archive's declared uncompressed size (untrusted /
    // zip-bomb). Clamp the capacity hint so a bogus huge size can't trigger a
    // capacity-overflow panic or giant up-front allocation; `read_to_end`
    // still grows the buffer to the real byte count for legit members.
    let cap = (zf.size() as usize).min(16 << 20);
    let mut buf = Vec::with_capacity(cap);
    zf.read_to_end(&mut buf)
        .with_context(|| format!("reading member {name}"))?;
    Ok(buf)
}

/// Read a member from an already-mapped zip byte slice (STORE only).
pub fn read_member_from_map<'a>(
    map: &'a [u8],
    ranges: &BTreeMap<String, MemberRange>,
    name: &str,
) -> Result<&'a [u8]> {
    let range = ranges
        .get(name)
        .with_context(|| format!("missing member {name}"))?;
    let start = range.data_offset as usize;
    let end = start
        .checked_add(range.length as usize)
        .context("member range overflow")?;
    map.get(start..end)
        .with_context(|| format!("mmap window for {name} out of range"))
}
