// RLX — versatile ML compiler + runtime.
//! Runs BOTH the IR and host decoder on the same reference inputs, with
//! `RLX_SAM3_DECODER_DUMP_DIR=<dir>` set so each writes layer-0..N intermediate
//! tensors to disk. Then we compare element-wise and report the first divergence.

use anyhow::Result;
use rlx_models::sam3::detector_decoder::forward_decoder;
use rlx_models::sam3::detector_decoder_ir::Sam3CompiledDecoder;
use rlx_models::sam3::{Sam3, Sam3Config, SAM3_IMG_SIZE};
use rlx_runtime::Device;
use std::env;
use std::path::PathBuf;
use std::time::Instant;

fn synthesize_image() -> Vec<u8> {
    let n = SAM3_IMG_SIZE * SAM3_IMG_SIZE * 3;
    let mut v = vec![0u8; n];
    for y in 0..SAM3_IMG_SIZE {
        for x in 0..SAM3_IMG_SIZE {
            for c in 0..3 {
                let fx = x as f32 / SAM3_IMG_SIZE as f32;
                let fy = y as f32 / SAM3_IMG_SIZE as f32;
                let phase = (c as f32) * 0.7;
                let s = (6.28 * fx + phase).sin() * (3.14 * fy + phase).cos();
                let val = ((s + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
                v[(y * SAM3_IMG_SIZE + x) * 3 + c] = val;
            }
        }
    }
    v
}

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read");
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()
}

fn diff(a: &[f32], b: &[f32]) -> (f32, f64, usize) {
    let mut mad = 0.0f32;
    let mut idx = 0;
    let n = a.len().min(b.len());
    for i in 0..n {
        let d = (a[i]-b[i]).abs();
        if d > mad { mad = d; idx = i; }
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let av = a[i] as f64; let bv = b[i] as f64;
        dot += av*bv; na += av*av; nb += bv*bv;
    }
    let denom = (na*nb).sqrt();
    let cos = if denom == 0.0 { 0.0 } else { (1.0 - dot/denom).max(0.0) };
    (mad, cos, idx)
}

fn main() -> Result<()> {
    let weights = env::var("RLX_SAM3_WEIGHTS")?;
    let dump_dir = env::var("RLX_SAM3_DECODER_DUMP_DIR").unwrap_or_else(|_| {
        let p = std::env::temp_dir().join("rlx_sam3_decoder_dump");
        std::fs::create_dir_all(&p).unwrap();
        p.to_string_lossy().into_owned()
    });
    std::fs::create_dir_all(&dump_dir)?;
    eprintln!("dump dir: {dump_dir}");

    // Reuse the parity reference dump from a previous test run.
    let ref_dir = env::var("RLX_SAM3_REF_DIR")
        .unwrap_or_else(|_| "/var/folders/9_/pjm86g5j44l4cdv5mld3wd9c0000gn/T/tmp.0NBLOovOZD".into());
    let ref_dir: PathBuf = ref_dir.into();
    let mem_seq_first = read_f32(&ref_dir.join("encoder_memory.f32"));
    let pos_nchw = read_f32(&ref_dir.join("encoder_pos.f32"));
    let prompt = read_f32(&ref_dir.join("encoder_prompt.f32"));
    let prompt_mask = std::fs::read(ref_dir.join("encoder_prompt_mask.u8"))?;

    let model = Sam3::from_safetensors(&weights, Sam3Config::base())?;

    let batch = 1; let c = 256; let h = 72; let w = 72; let seq = 32;
    // Reshape memory seq-first → batch-first.
    let mut memory_bf = vec![0f32; batch * h * w * c];
    for l in 0..h*w { for b in 0..batch {
        let s = (l*batch+b)*c; let d = (b*h*w+l)*c;
        memory_bf[d..d+c].copy_from_slice(&mem_seq_first[s..s+c]);
    }}
    // Reshape pos NCHW → batch-first [B, hw, C].
    let mut memory_pos_bf = vec![0f32; batch * h * w * c];
    for b in 0..batch { for y in 0..h { for xc in 0..w { for ch in 0..c {
        memory_pos_bf[(b*h*w+y*w+xc)*c+ch] = pos_nchw[((b*c+ch)*h+y)*w+xc];
    }}}}

    // Run HOST decoder.
    eprintln!("running HOST decoder...");
    // Set dump prefix for this run.
    unsafe { std::env::set_var("RLX_SAM3_DECODER_DUMP_DIR", &dump_dir); }
    let t = Instant::now();
    let _ = forward_decoder(
        model.decoder_weights(),
        &memory_bf, &memory_pos_bf, &prompt, &prompt_mask,
        batch, h, w, seq,
    )?;
    eprintln!("host: {:.1}s", t.elapsed().as_secs_f32());

    // Run IR decoder.
    eprintln!("running IR decoder...");
    let mut dec = Sam3CompiledDecoder::new(model.decoder_weights(), batch, h*w, seq, Device::Cpu)?;
    let t = Instant::now();
    let _ = dec.run(&memory_bf, &memory_pos_bf, &prompt, &prompt_mask, h, w)?;
    eprintln!("ir: {:.1}s", t.elapsed().as_secs_f32());

    // Compare.
    let dd: PathBuf = dump_dir.into();
    for li in 0..6 {
        for name in ["query_pos", "sa_queries", "after_ca_text_q", "ca_img_proj", "after_ca_img_q", "new_tgt", "new_presence", "out_norm"] {
            let host_p = dd.join(format!("host_layer{li}_{name}.f32"));
            let ir_p = dd.join(format!("ir_layer{li}_{name}.f32"));
            if !host_p.exists() || !ir_p.exists() { continue; }
            let a = read_f32(&host_p);
            let b = read_f32(&ir_p);
            let (mad, cos, idx) = diff(&a, &b);
            println!("layer{li} {name:<14}: len_h={} len_i={} mad={mad:.3e} cos_dist={cos:.3e} idx={idx}",
                a.len(), b.len());
        }
    }
    Ok(())
}
