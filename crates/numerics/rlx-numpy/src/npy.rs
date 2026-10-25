// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `.npy` files of float32 — `np.save` / `np.load` for C-order little-endian
//! `<f4` arrays (format 1.0 written; 1.0–3.0 read).

use std::path::Path;

/// Read a C-order `<f4` `.npy`: `(data, shape)`.
pub fn read_f32(path: &Path) -> Result<(Vec<f32>, Vec<usize>), String> {
    let b = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not an .npy file", path.display()));
    }
    let (hlen, off) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        _ => (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12),
    };
    let header = std::str::from_utf8(b.get(off..off + hlen).ok_or("truncated header")?)
        .map_err(|e| format!("{}: header: {e}", path.display()))?;
    if !header.contains("'<f4'") {
        return Err(format!(
            "{}: only little-endian float32 is supported ({header})",
            path.display()
        ));
    }
    if header.contains("'fortran_order': True") {
        return Err(format!(
            "{}: Fortran order is not supported",
            path.display()
        ));
    }
    let sh = header
        .split("'shape': (")
        .nth(1)
        .and_then(|r| r.split(')').next())
        .ok_or_else(|| format!("{}: no shape in header", path.display()))?;
    let shape = sh
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| format!("bad shape {s:?}: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let data: Vec<f32> = b[off + hlen..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if data.len() != shape.iter().product::<usize>() {
        return Err(format!(
            "{}: {} values for shape {shape:?}",
            path.display(),
            data.len()
        ));
    }
    Ok((data, shape))
}

/// Write `data` (row-major) as a format-1.0 `<f4` `.npy` of `shape`.
pub fn write_f32(path: &Path, data: &[f32], shape: &[usize]) -> Result<(), String> {
    if data.len() != shape.iter().product::<usize>() {
        return Err(format!("{} values for shape {shape:?}", data.len()));
    }
    let dims = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {dims}, }}");
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - unpadded % 64) % 64));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len() * 4);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out).map_err(|e| format!("writing {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn round_trip() {
        let p = std::env::temp_dir().join("rlx_numpy_npy_round_trip.npy");
        let d: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 3.0).collect();
        super::write_f32(&p, &d, &[2, 3, 4]).unwrap();
        assert_eq!(super::read_f32(&p).unwrap(), (d, vec![2, 3, 4]));
    }
}
