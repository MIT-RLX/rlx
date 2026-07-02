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

//! Open a GGUF file, print a summary, and dequant one tensor of every
//! supported dtype to confirm the math doesn't blow up on real data.
//!
//! Usage: `cargo run -p rlx-gguf --release --example inspect -- <path.gguf>`

use std::collections::BTreeMap;
use std::env;

use anyhow::{Result, anyhow};
use rlx_gguf::{GgmlType, GgufFile, MetaValue};

fn main() -> Result<()> {
    let path = env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: inspect <path.gguf>"))?;

    let t0 = std::time::Instant::now();
    let f = GgufFile::from_path(&path)?;
    let parse_ms = t0.elapsed().as_secs_f64() * 1e3;

    println!("─── {path}");
    println!("  version          = {}", f.version);
    println!("  alignment        = {}", f.alignment);
    println!("  metadata entries = {}", f.metadata.len());
    println!("  tensors          = {}", f.tensors.len());
    println!("  parse time       = {parse_ms:.1} ms");

    // Selected metadata.
    for k in ["general.architecture", "general.name", "general.file_type"] {
        if let Some(v) = f.metadata.get(k) {
            println!("  {k:<26} = {}", display_meta(v));
        }
    }

    // Dtype histogram.
    let mut hist: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut total_elems: usize = 0;
    for t in f.tensors.values() {
        let n = t.n_elements();
        let e = hist.entry(format!("{:?}", t.dtype)).or_insert((0, 0));
        e.0 += 1;
        e.1 += n;
        total_elems += n;
    }
    println!("\n  dtype histogram (count, elements):");
    for (k, (c, n)) in &hist {
        let pct = (*n as f64 / total_elems as f64) * 100.0;
        println!("    {k:<5} {c:>5} tensors  {n:>14} elems  ({pct:5.1}%)");
    }
    println!(
        "    total       {} tensors  {total_elems} elems",
        f.tensors.len()
    );

    // Dequant one tensor of each supported dtype, report stats.
    println!("\n  dequant probe:");
    let supported = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q8_0,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
    ];
    for dt in supported {
        let Some(name) = pick_first(&f, dt) else {
            continue;
        };
        let t1 = std::time::Instant::now();
        match f.dequant_f32(&name) {
            Ok((data, shape)) => {
                let (mn, mx, mean, finite) = stats(&data);
                let ms = t1.elapsed().as_secs_f64() * 1e3;
                let head: Vec<String> = data.iter().take(8).map(|x| format!("{x:+.6}")).collect();
                println!(
                    "    {dt:?}  {name}  shape={shape:?}  n={}  finite={finite}/{}  \
                     min={mn:+.4}  max={mx:+.4}  mean={mean:+.4}  ({ms:.1} ms)\n      \
                     first8 = [{}]",
                    data.len(),
                    data.len(),
                    head.join(", ")
                );
            }
            Err(e) => println!("    {dt:?}  {name}  ERR: {e}"),
        }
    }

    // Also report unsupported quants for visibility.
    let mut unsupported: BTreeMap<String, usize> = BTreeMap::new();
    for t in f.tensors.values() {
        if !supported.contains(&t.dtype) {
            *unsupported.entry(format!("{:?}", t.dtype)).or_insert(0) += 1;
        }
    }
    if !unsupported.is_empty() {
        println!("\n  unsupported (parse-only):");
        for (k, c) in unsupported {
            println!("    {k:<5} {c} tensors");
        }
    }

    Ok(())
}

fn pick_first(f: &GgufFile, dt: GgmlType) -> Option<String> {
    let mut hits: Vec<&str> = f
        .tensors
        .values()
        .filter(|t| t.dtype == dt)
        .map(|t| t.name.as_str())
        .collect();
    hits.sort();
    hits.first().map(|s| s.to_string())
}

fn stats(data: &[f32]) -> (f32, f32, f32, usize) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut finite = 0usize;
    for &v in data {
        if v.is_finite() {
            finite += 1;
            mn = mn.min(v);
            mx = mx.max(v);
            sum += v as f64;
        }
    }
    let mean = if finite > 0 {
        (sum / finite as f64) as f32
    } else {
        f32::NAN
    };
    (mn, mx, mean, finite)
}

fn display_meta(v: &MetaValue) -> String {
    match v {
        MetaValue::String(s) => format!("\"{s}\""),
        MetaValue::U32(x) => x.to_string(),
        MetaValue::U64(x) => x.to_string(),
        MetaValue::I32(x) => x.to_string(),
        MetaValue::I64(x) => x.to_string(),
        MetaValue::F32(x) => x.to_string(),
        MetaValue::Bool(x) => x.to_string(),
        MetaValue::Array(a) => format!("array[len={}]", a.len()),
        other => format!("{other:?}"),
    }
}
