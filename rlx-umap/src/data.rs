// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Load training data for the `train-umap` binary and examples.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::utils::generate_test_data;

/// Load CSV: one sample per line, comma-separated floats (optional header with letters).
pub fn load_csv(path: &Path) -> std::io::Result<Vec<Vec<f64>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if i == 0 && looks_like_header(trimmed) {
            continue;
        }
        let row: Vec<f64> = trimmed
            .split(',')
            .map(|s| s.trim().parse())
            .collect::<Result<_, _>>()
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("line {}: {e}", i + 1),
                )
            })?;
        if row.is_empty() {
            continue;
        }
        rows.push(row);
    }
    if rows.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "need at least 2 data rows",
        ));
    }
    Ok(rows)
}

fn looks_like_header(line: &str) -> bool {
    line.split(',').any(|s| s.trim().parse::<f64>().is_err())
}

/// Row-major f64 file: `n` (u64 le), `d` (u64 le), then `n*d` f64 values.
pub fn load_f64_matrix(path: &Path) -> std::io::Result<Vec<Vec<f64>>> {
    let mut file = File::open(path)?;
    let n = read_u64(&mut file)? as usize;
    let d = read_u64(&mut file)? as usize;
    let mut flat = vec![0f64; n * d];
    for v in &mut flat {
        *v = read_f64(&mut file)?;
    }
    Ok(flat.chunks(d).map(|c| c.to_vec()).collect())
}

fn read_u64(r: &mut File) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f64(r: &mut File) -> std::io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

/// Write embedding CSV: `x,y` or `x,y,label`.
pub fn write_embedding_csv(
    path: &Path,
    embedding: &[Vec<f64>],
    labels: Option<&[u8]>,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = File::create(path)?;
    match labels {
        Some(lbl) => {
            writeln!(f, "x,y,label")?;
            for (pt, &l) in embedding.iter().zip(lbl) {
                writeln!(f, "{:.8},{:.8},{}", pt[0], pt[1], l)?;
            }
        }
        None => {
            writeln!(f, "x,y")?;
            for pt in embedding {
                writeln!(f, "{:.8},{:.8}", pt[0], pt[1])?;
            }
        }
    }
    Ok(())
}

pub fn load_synthetic(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    generate_test_data(n, d, seed)
}
