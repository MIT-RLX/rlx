//! Throwaway parity probe: dequantize an mlx-community model dir and dump
//! selected tensors as raw little-endian f32 for comparison against
//! `mlx.core.dequantize`. Not committed.
//!
//! Usage: cargo run -p rlx-mlx-io --example mlx_dequant_dump -- <model_dir> <out_dir>

use std::io::Write;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("model dir");
    let out = args.next().unwrap_or_else(|| "/tmp".to_string());

    let w = rlx_mlx_io::load_path(&dir)?;
    if let Some(q) = w.config.quantization.as_ref() {
        eprintln!(
            "quant: mode={} group_size={} bits={}",
            q.mode.as_str(),
            q.group_size,
            q.bits
        );
    }
    let map = w.into_f32_map()?;
    let mut keys: Vec<_> = map.keys().cloned().collect();
    keys.sort();
    eprintln!("dequantized {} logical tensors", keys.len());

    for base in ["model.layers.0.self_attn.q_proj", "model.embed_tokens"] {
        let data = map
            .get(base)
            .unwrap_or_else(|| panic!("missing dequantized key {base}"));
        let fname = format!("{out}/rlx_{}.bin", base.replace('.', "_"));
        let mut f = std::io::BufWriter::new(std::fs::File::create(&fname)?);
        for v in data {
            f.write_all(&v.to_le_bytes())?;
        }
        f.flush()?;
        let sum: f64 = data.iter().map(|&x| x as f64).sum();
        eprintln!(
            "{base}: len={} first8={:?} sum={:.6} -> {fname}",
            data.len(),
            &data[..8.min(data.len())],
            sum
        );
    }
    Ok(())
}
