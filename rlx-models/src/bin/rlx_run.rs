// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// `rlx-run` — small CLI on top of `rlx_models::run::*`. One sub-command
// per supported model family. Designed to mirror the builder API 1:1
// so the bytes a user types translate directly into the Rust call they
// could have made.
//
// Usage:
//   rlx-run qwen3 --weights model.gguf [flags] --prompt "..."
//   rlx-run sam1  --weights sam_vit_h.safetensors
//   rlx-run sam2  --weights sam2_hiera.safetensors
//   rlx-run sam3  --weights sam3.safetensors
//   rlx-run inspect <file>      # dump format, tensors, MTP keys
//
// No external arg-parser dependency — keeps the binary small and
// avoids adding clap to rlx-models. Parses with a small hand-rolled
// loop.

use anyhow::{Context, Result, anyhow, bail};
use rlx_gguf::GgufFile;
use rlx_models::run::{
    ConfigSource, DinoV2Output, DinoV2Runner, DinoV2Variant, ModelRunner, Precision, Qwen3Runner,
    SamArch, SamPredictionAny, SamRunner, WeightFormat, dispatch as registry_dispatch,
    list_mtp_keys, register_runner,
};
use rlx_models::qwen3::SampleOpts;
use rlx_runtime::Device;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "\
rlx-run — minimal multi-model launcher
USAGE:
  rlx-run <subcommand> [flags]

SUBCOMMANDS:
  qwen3     run a Qwen3 LM (safetensors or gguf)
  sam1      load Segment Anything v1
  sam2      load Segment Anything v2
  sam3      load Segment Anything v3
  dinov2    run a DINOv2 ViT encoder / classifier
  inspect   dump tensor list / format / MTP keys for a model file
  help      print this help

Common flags (qwen3 / sam*):
  --weights <PATH>            required; .safetensors or .gguf
  --device <cpu|metal|mlx|gpu>  default cpu
  --config <PATH>             override config.json (safetensors) /
                              force a json config (gguf)
  --format <safetensors|gguf>   override extension autodetection

Qwen3-only flags:
  --prompt <TEXT>             prompt as a string (passed straight to
                              the tokenizer; if no tokenizer is wired,
                              treats input as comma-separated token ids)
  --prompt-ids <I,I,I>        skip tokenizer; supply raw ids
  --max-tokens <N>            tokens to generate (default 32)
  --max-seq <N>               prefill bucket size (default 128)
  --precision <f32|f16-lm>    default f32
  --max-memory-gb <F>         soft cap on dequant-to-f32 footprint
  --no-stream                 buffer all tokens before printing
  --use-mtp                   keep MTP weights loadable (no speculation yet)
  --packed                    keep K-quant GGUF weights packed in arena
                              (Op::DequantMatMul; CPU-only; runs one
                              forward + prints top-1 — no streaming
                              decode in this mode)
  --temperature <F>           sampling temperature; default 0 (greedy)
  --top-p <F>                 nucleus sampling cutoff; default 1.0

Examples:
  rlx-run qwen3 --weights Qwen3-0.6B-Q4_K_M.gguf --device metal \\
      --prompt-ids 1,17,42 --max-tokens 16

  rlx-run inspect Qwen3-0.6B-Q4_K_M.gguf
";

fn main() -> ExitCode {
    register_builtins();
    let args: Vec<String> = std::env::args().skip(1).collect();
    match registry_dispatch(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rlx-run: {e:#}");
            ExitCode::FAILURE
        }
    }
}

/// Register the built-in model runners against the global registry.
/// Each one is a thin wrapper that defers to the per-subcommand
/// `run_*` function below. Third parties writing their own binary
/// call this (optional — they can also register only what they
/// need) plus their own `register_runner(Box::new(MyRunner))`
/// before invoking `registry_dispatch`.
pub fn register_builtins() {
    macro_rules! reg {
        ($name:expr, $desc:expr, $body:expr) => {{
            struct R;
            impl ModelRunner for R {
                fn name(&self) -> &'static str {
                    $name
                }
                fn description(&self) -> &'static str {
                    $desc
                }
                fn run(&self, args: &[String]) -> Result<()> {
                    let f: fn(&[String]) -> Result<()> = $body;
                    f(args)
                }
            }
            register_runner(Box::new(R));
        }};
    }
    reg!(
        "qwen3",
        "Run a Qwen3 LM (safetensors or gguf)",
        run_qwen3
    );
    reg!(
        "qwen35",
        "Run a Qwen3.5 / Qwen3.6 GGUF (hybrid gated-DeltaNet + attention)",
        run_qwen35
    );
    reg!(
        "sam1",
        "Segment Anything v1",
        |args: &[String]| run_sam(SamArch::Sam1, args)
    );
    reg!(
        "sam2",
        "Segment Anything v2",
        |args: &[String]| run_sam(SamArch::Sam2, args)
    );
    reg!(
        "sam3",
        "Segment Anything v3 (text-conditioned)",
        |args: &[String]| run_sam(SamArch::Sam3, args)
    );
    reg!(
        "dinov2",
        "DINOv2 ViT encoder / classifier",
        run_dinov2
    );
    reg!(
        "inspect",
        "Dump tensor list / format / MTP keys for a model file",
        run_inspect
    );
}

// ── Qwen3 ─────────────────────────────────────────────────────────

fn run_qwen3(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut config: Option<PathBuf> = None;
    let mut format: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut prompt_ids: Option<Vec<u32>> = None;
    let mut max_tokens = 32usize;
    let mut max_seq = 128usize;
    let mut precision = "f32".to_string();
    let mut max_memory_gb: Option<f32> = None;
    let mut stream = true;
    let mut use_mtp = false;
    let mut packed = false;
    let mut temperature = 0f32;
    let mut top_p = 1f32;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--weights" => {
                weights = Some(req(args, &mut i)?.into());
            }
            "--device" => device = req(args, &mut i)?,
            "--config" => config = Some(req(args, &mut i)?.into()),
            "--format" => format = Some(req(args, &mut i)?),
            "--prompt" => prompt = Some(req(args, &mut i)?),
            "--prompt-ids" => {
                prompt_ids = Some(
                    req(args, &mut i)?
                        .split(',')
                        .map(|s| s.trim().parse::<u32>())
                        .collect::<Result<_, _>>()
                        .context("--prompt-ids: comma-separated u32 list")?,
                );
            }
            "--max-tokens" => {
                max_tokens = req(args, &mut i)?
                    .parse()
                    .context("--max-tokens: usize")?;
            }
            "--max-seq" => max_seq = req(args, &mut i)?.parse().context("--max-seq: usize")?,
            "--precision" => precision = req(args, &mut i)?,
            "--max-memory-gb" => {
                max_memory_gb = Some(
                    req(args, &mut i)?
                        .parse()
                        .context("--max-memory-gb: f32")?,
                );
            }
            "--no-stream" => {
                stream = false;
                i += 1;
            }
            "--use-mtp" => {
                use_mtp = true;
                i += 1;
            }
            "--packed" => {
                packed = true;
                i += 1;
            }
            "--temperature" => {
                temperature = req(args, &mut i)?.parse().context("--temperature: f32")?;
            }
            "--top-p" => top_p = req(args, &mut i)?.parse().context("--top-p: f32")?,
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }

    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_device(&device)?;
    let precision = match precision.as_str() {
        "f32" => Precision::F32,
        "f16-lm" | "f16_lm" => Precision::F16LmHead,
        other => bail!("--precision: expected f32|f16-lm, got {other}"),
    };
    let format = match format.as_deref() {
        Some("safetensors") => Some(WeightFormat::Safetensors),
        Some("gguf") => Some(WeightFormat::Gguf),
        Some(other) => bail!("--format: expected safetensors|gguf, got {other}"),
        None => None,
    };
    let sample = SampleOpts {
        temperature,
        top_p,
        ..SampleOpts::greedy()
    };

    let mut b = Qwen3Runner::builder()
        .weights(weights.clone())
        .device(device)
        .max_seq(max_seq)
        .precision(precision)
        .stream(stream)
        .use_mtp(use_mtp)
        .packed_weights(packed)
        .sample(sample);
    if let Some(fmt) = format {
        b = b.format(fmt);
    }
    if let Some(p) = config {
        b = b.config(ConfigSource::JsonFile(p));
    }
    if let Some(g) = max_memory_gb {
        b = b.max_memory_gb(g);
    }

    let ids = match (prompt_ids, prompt) {
        (Some(ids), _) => ids,
        (None, Some(p)) => {
            // No tokenizer wired in this CLI today. Treat the prompt
            // as raw comma-separated ids for forward-compatibility
            // with people piping `python -m tokenize | rlx-run`.
            p.split(',')
                .map(|s| s.trim().parse::<u32>())
                .collect::<Result<_, _>>()
                .context(
                    "--prompt without a tokenizer must be comma-separated u32 ids; \
                     use --prompt-ids for clarity or wire a tokenizer in a downstream tool",
                )?
        }
        (None, None) => vec![1u32, 17, 42, 314, 2718, 9001, 27182, 8128],
    };

    eprintln!(
        "[rlx-run] qwen3: weights={weights:?} device={device:?} max_seq={max_seq} \
         precision={precision:?} stream={stream}"
    );
    let mut runner = b.build()?;
    eprintln!(
        "[rlx-run] compiled — vocab={} hidden={} layers={}",
        runner.config().vocab_size,
        runner.config().hidden_size,
        runner.config().num_hidden_layers
    );

    let t0 = std::time::Instant::now();
    let mut printed = 0;
    if packed {
        // Packed mode: same streaming surface as the F32 path,
        // backed by autoregressive prefills (each token costs one
        // full prefill — see `Qwen3Runner::generate_packed`).
        eprintln!(
            "[rlx-run] packed streaming: each token costs ~one full prefill (slow but \
             the only path that fits 14 B+ Q4_K_M GGUFs on commodity Macs)"
        );
    }
    runner.generate(&ids, max_tokens, |tok| {
        if stream {
            print!("{tok} ");
            std::io::stdout().flush().ok();
        }
        printed += 1;
    })?;
    let dt = t0.elapsed();
    if !stream {
        // already collected via callback; print summary only
    }
    println!();
    eprintln!(
        "[rlx-run] generated {printed} tokens in {:.2?} ({:.1} tok/s)",
        dt,
        printed as f64 / dt.as_secs_f64()
    );
    Ok(())
}

// ── Qwen3.5 / Qwen3.6 (qwen35 arch) ─────────────────────────────

fn run_qwen35(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut prompt_ids: Vec<u32> = vec![1, 2, 3];
    let mut max_seq = 0usize;
    let mut max_tokens = 0usize;
    let mut enable_mtp = false;
    let mut packed_weights = false;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--weights" => weights = Some(PathBuf::from(it.next().context("--weights")?)),
            "--device" => device = it.next().context("--device")?.clone(),
            "--max-seq" => max_seq = it.next().context("--max-seq")?.parse()?,
            "--max-tokens" => max_tokens = it.next().context("--max-tokens")?.parse()?,
            "--mtp" => enable_mtp = true,
            "--packed" => packed_weights = true,
            "--prompt-ids" => {
                let raw = it.next().context("--prompt-ids")?;
                prompt_ids = raw
                    .split(',')
                    .map(|s| s.trim().parse::<u32>())
                    .collect::<std::result::Result<_, _>>()
                    .context("--prompt-ids")?;
            }
            other => bail!("rlx-run qwen35: unknown flag: {other}"),
        }
    }

    let weights = weights
        .ok_or_else(|| anyhow!("rlx-run qwen35: --weights <path.gguf> required"))?;
    let dev = match device.as_str() {
        "cpu" => rlx_runtime::Device::Cpu,
        other => bail!("rlx-run qwen35: --device {other} not wired (CPU-only today)"),
    };
    if max_seq == 0 {
        max_seq = (prompt_ids.len() + max_tokens).max(8);
    }

    println!(
        "[rlx-run] qwen35: weights={:?} device={device} max_seq={max_seq} \
         mtp={enable_mtp} packed={packed_weights}",
        weights
    );

    let mut runner = rlx_models::Qwen35RunnerBuilder::default()
        .weights(&weights)
        .device(dev)
        .max_seq(max_seq)
        .enable_mtp(enable_mtp)
        .packed_weights(packed_weights)
        .last_logits_only(true)
        .build()?;

    println!(
        "[rlx-run] qwen35: compiled (hidden={}, layers={}, ssm_state={}, dt_rank={})",
        runner.cfg().hidden_size,
        runner.cfg().num_hidden_layers,
        runner.cfg().ssm_state_size,
        runner.cfg().ssm_time_step_rank,
    );

    if max_tokens == 0 {
        let out = runner.predict_logits(&prompt_ids)?;
        println!(
            "[rlx-run] qwen35: logits={} vocab≈{}",
            out.logits.len(),
            out.vocab_size
        );

        let mut idx: Vec<usize> = (0..out.logits.len()).collect();
        idx.sort_by(|&a, &b| {
            out.logits[b]
                .partial_cmp(&out.logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        println!("[rlx-run] qwen35: top-5 trunk logits:");
        for &i in idx.iter().take(5) {
            println!("    token {i:6}  logit {:>12.5}", out.logits[i]);
        }
        if let Some(mtp) = &out.mtp_logits {
            let mut midx: Vec<usize> = (0..mtp.len()).collect();
            midx.sort_by(|&a, &b| {
                mtp[b]
                    .partial_cmp(&mtp[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            println!("[rlx-run] qwen35: top-5 MTP logits:");
            for &i in midx.iter().take(5) {
                println!("    token {i:6}  logit {:>12.5}", mtp[i]);
            }
        }
    } else {
        println!("[rlx-run] qwen35: generating {max_tokens} tokens (greedy)…");
        let new_ids = runner.generate(&prompt_ids, max_tokens, |t| {
            print!("{t} ");
            std::io::Write::flush(&mut std::io::stdout()).ok();
            true
        })?;
        println!("\n[rlx-run] qwen35: generated: {new_ids:?}");
    }
    Ok(())
}

// ── SAM 1 / 2 / 3 ─────────────────────────────────────────────────

fn run_sam(arch: SamArch, args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut config: Option<PathBuf> = None;
    let mut point: Option<(f32, f32)> = None;
    let mut text_tokens: Vec<u32> = Vec::new();
    let mut dry = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => {
                weights = Some(req(args, &mut i)?.into());
            }
            "--device" => device = req(args, &mut i)?,
            "--config" => config = Some(req(args, &mut i)?.into()),
            "--point" => {
                let v = req(args, &mut i)?;
                let parts: Vec<&str> = v.split(',').collect();
                if parts.len() != 2 {
                    bail!("--point expects X,Y (e.g. 512,512), got {v}");
                }
                let x: f32 = parts[0].trim().parse().context("--point: X must be f32")?;
                let y: f32 = parts[1].trim().parse().context("--point: Y must be f32")?;
                point = Some((x, y));
            }
            "--text-tokens" => {
                text_tokens = req(args, &mut i)?
                    .split(',')
                    .map(|s| s.trim().parse::<u32>())
                    .collect::<Result<_, _>>()
                    .context("--text-tokens: comma-separated u32 list")?;
            }
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_device(&device)?;
    let mut b = SamRunner::builder(arch).weights(weights).device(device);
    if let Some(c) = config {
        b = b.config(c);
    }
    let runner = b.build()?;
    eprintln!("{}", runner.summary());

    if dry {
        eprintln!("[rlx-run] --dry set; skipping forward pass");
        return Ok(());
    }

    // Synthesize a 1024×1024 RGB gradient so the example runs without
    // an external image. Real pictures: replace with the bytes from
    // `image::open(p)?.to_rgb8().as_raw()`.
    let h_in = 1024usize;
    let w_in = 1024usize;
    let mut rgb = vec![0u8; h_in * w_in * 3];
    for y in 0..h_in {
        for x in 0..w_in {
            let base = (y * w_in + x) * 3;
            rgb[base] = (x * 255 / w_in) as u8;
            rgb[base + 1] = (y * 255 / h_in) as u8;
            rgb[base + 2] = ((x + y) * 127 / (h_in + w_in)) as u8;
        }
    }

    // Default click: image center, foreground.
    let (cx, cy) = point.unwrap_or((w_in as f32 / 2.0, h_in as f32 / 2.0));
    let points_xy = [cx, cy];
    let points_lbl = [1.0f32];

    // SAM 3 needs text tokens. If none supplied, fall back to a
    // 32-id placeholder pattern (the real model expects tokenizer
    // output; this just keeps the pipeline runnable).
    if matches!(arch, SamArch::Sam3) && text_tokens.is_empty() {
        text_tokens = (0..32u32).collect();
        eprintln!("[rlx-run] no --text-tokens supplied, using 0..32 placeholder for SAM 3");
    }

    eprintln!(
        "[rlx-run] running SAM forward (synthetic 1024×1024, click=({cx:.0},{cy:.0}))"
    );
    let t0 = std::time::Instant::now();
    let pred = runner.predict_image(
        &rgb,
        h_in,
        w_in,
        Some((&points_xy, &points_lbl)),
        None,
        &text_tokens,
    )?;
    let dt = t0.elapsed();

    match pred {
        SamPredictionAny::Sam1(p) => {
            eprintln!(
                "[rlx-run] sam1 forward in {dt:?} — masks={} mask_side={} iou={:?}",
                p.num_masks,
                p.mask_side,
                &p.iou_pred[..p.iou_pred.len().min(p.num_masks)]
            );
        }
        SamPredictionAny::Sam2(p) => {
            eprintln!(
                "[rlx-run] sam2 forward in {dt:?} — masks={} out={}x{} iou={:?}",
                p.num_masks,
                p.h_out,
                p.w_out,
                &p.iou_pred[..p.iou_pred.len().min(p.num_masks)]
            );
        }
        SamPredictionAny::Sam3(p) => {
            eprintln!(
                "[rlx-run] sam3 forward in {dt:?} — instances={} mask_shape={:?} scores[..5]={:?}",
                p.num_instances,
                p.mask_shape,
                &p.scores[..p.scores.len().min(5)]
            );
        }
    }
    Ok(())
}

// ── DINOv2 ─────────────────────────────────────────────────────────

fn run_dinov2(args: &[String]) -> Result<()> {
    let mut weights: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut variant_str = "base".to_string();
    let mut img_size = 518usize;
    let mut batch = 1usize;
    let mut dry = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => {
                weights = Some(req(args, &mut i)?.into());
            }
            "--device" => device = req(args, &mut i)?,
            "--variant" => variant_str = req(args, &mut i)?,
            "--img-size" => {
                img_size = req(args, &mut i)?
                    .parse()
                    .context("--img-size: usize")?;
            }
            "--batch" => batch = req(args, &mut i)?.parse().context("--batch: usize")?,
            "--dry" => {
                dry = true;
                i += 1;
            }
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => bail!("unknown flag: {other}"),
        }
    }
    let weights = weights.ok_or_else(|| anyhow!("--weights is required"))?;
    let device = parse_device(&device)?;
    let variant = match variant_str.as_str() {
        "small" | "vit-s" => DinoV2Variant::Small,
        "base" | "vit-b" => DinoV2Variant::Base,
        "large" | "vit-l" => DinoV2Variant::Large,
        other => bail!("--variant: expected small|base|large (got {other})"),
    };

    eprintln!(
        "[rlx-run] dinov2: weights={weights:?} device={device:?} variant={variant:?} img_size={img_size} batch={batch}"
    );
    let mut runner = DinoV2Runner::builder()
        .weights(&weights)
        .device(device)
        .variant(variant)
        .img_size(img_size)
        .batch(batch)
        .build()?;
    eprintln!(
        "[rlx-run] compiled — hidden={} layers={} num_classes={}",
        runner.config().hidden_size,
        runner.config().num_hidden_layers,
        runner.config().num_classes
    );

    if dry {
        eprintln!("[rlx-run] --dry set; skipping forward pass");
        return Ok(());
    }

    // Synthetic image — replace with image::open(...).to_rgb8().as_raw().
    let (h_in, w_in) = (img_size, img_size);
    let mut rgb = vec![0u8; h_in * w_in * 3];
    for y in 0..h_in {
        for x in 0..w_in {
            let base = (y * w_in + x) * 3;
            rgb[base] = (x * 255 / w_in) as u8;
            rgb[base + 1] = (y * 255 / h_in) as u8;
            rgb[base + 2] = ((x + y) * 127 / (h_in + w_in)) as u8;
        }
    }

    let t0 = std::time::Instant::now();
    let out = runner.predict_image(&rgb, h_in, w_in)?;
    let dt = t0.elapsed();
    match out {
        DinoV2Output::Logits {
            per_batch,
            num_classes,
        } => {
            eprintln!(
                "[rlx-run] dinov2 logits in {dt:?} — batch={} classes={}",
                per_batch.len(),
                num_classes
            );
            for (b, logits) in per_batch.iter().enumerate() {
                let (top1, top1_val) = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap();
                eprintln!("  batch[{b}] top1={top1} logit={top1_val:.3}");
            }
        }
        DinoV2Output::Tokens {
            per_batch,
            seq,
            hidden,
        } => {
            eprintln!(
                "[rlx-run] dinov2 tokens in {dt:?} — batch={} seq={seq} hidden={hidden}",
                per_batch.len()
            );
            // CLS token (index 0) summary: ||cls||₂
            for (b, toks) in per_batch.iter().enumerate() {
                let cls = &toks[..hidden];
                let norm: f32 = cls.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!("  batch[{b}] ||cls||₂ = {norm:.3}");
            }
        }
    }
    Ok(())
}

// ── inspect ───────────────────────────────────────────────────────

/// Estimate the two relevant residual sets for a `qwen35` GGUF:
/// `(f32_dequant_bytes, packed_bytes)`. The first column is what
/// `from_loader` materializes (every tensor dequantized to f32);
/// the second is what `from_loader_packed` materializes (K-quant
/// stays as bytes, everything else still dequants — embed table,
/// norms, conv kernel, scalar params). Quick sanity check before
/// the user picks `--packed`.
fn estimate_qwen35_footprint(raw: &rlx_gguf::GgufFile) -> (u64, u64) {
    use rlx_gguf::GgmlType;
    let mut f32_total = 0u64;
    let mut packed_total = 0u64;
    for t in raw.tensors.values() {
        let n = t.n_elements() as u64;
        let f32_bytes = n * 4;
        f32_total += f32_bytes;
        let packed_bytes = match t.dtype {
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8K => {
                // On-disk packed size = the byte slice length.
                raw.tensor_bytes(t).map(|b| b.len() as u64).unwrap_or(f32_bytes)
            }
            // Embed table, norms, conv kernel, ssm_a/dt etc. still
            // dequant — count their f32-cost in the packed column too.
            _ => f32_bytes,
        };
        packed_total += packed_bytes;
    }
    (f32_total, packed_total)
}

fn fmt_bytes(b: u64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let f = b as f64;
    if f >= GB {
        format!("{:.2} GB", f / GB)
    } else if f >= MB {
        format!("{:.1} MB", f / MB)
    } else {
        format!("{b} B")
    }
}

fn run_inspect(args: &[String]) -> Result<()> {
    let path = args
        .first()
        .ok_or_else(|| anyhow!("usage: rlx-run inspect <path>"))?;
    let pb: PathBuf = path.into();
    let fmt = WeightFormat::from_path(&pb)?;
    println!("path:   {pb:?}");
    println!("format: {fmt:?}");
    match fmt {
        WeightFormat::Gguf => {
            let raw = GgufFile::from_path(&pb)?;
            println!("version:  {}", raw.version);
            println!("tensors:  {}", raw.tensors.len());
            println!("metadata: {} keys", raw.metadata.len());
            let arch = raw
                .metadata
                .get("general.architecture")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            println!("arch:     {arch}");
            // Architecture compatibility hint — the qwen3 builder
            // expects pure-transformer Qwen3 / Qwen3.6 layout. The
            // Qwen3.5 family is hybrid Mamba+Attention (uses
            // `attn_qkv` + `ssm_*` blocks per layer) which the
            // current builder doesn't support.
            let mamba = raw.tensors.keys().any(|k| {
                k.starts_with("blk.0.ssm_") || k == "blk.0.attn_qkv.weight"
            });
            match (arch, mamba) {
                ("qwen3", false) | ("qwen36", false) => {
                    println!("compat:   ok — supported by rlx_models::qwen3 builder");
                }
                ("qwen35", true) | (_, true) => {
                    println!(
                        "compat:   qwen35 (gated DeltaNet + attention) — use \
                         `rlx-run qwen35 --packed` (rlx_models::Qwen35Runner)"
                    );
                }
                _ => {
                    println!(
                        "compat:   unknown arch '{arch}' — try `--config` to override or pick a different builder"
                    );
                }
            }
            // dtype histogram
            use std::collections::BTreeMap;
            let mut by_dt: BTreeMap<String, usize> = BTreeMap::new();
            for t in raw.tensors.values() {
                *by_dt.entry(format!("{:?}", t.dtype)).or_default() += 1;
            }
            println!("dtypes:");
            for (dt, n) in &by_dt {
                println!("  {dt:>6}: {n}");
            }
            // Footprint estimate: F32 dequant vs packed (K-quant
            // stays as bytes, everything else dequants to F32). Tells
            // the user whether `--packed` is mandatory on this file.
            let (f32_bytes, packed_bytes) = estimate_qwen35_footprint(&raw);
            if arch == "qwen35" || mamba {
                println!(
                    "footprint: F32-dequant ≈ {} / packed-mode ≈ {} \
                     (use --packed when the F32 column doesn't fit)",
                    fmt_bytes(f32_bytes),
                    fmt_bytes(packed_bytes),
                );
            }
            let mtp = list_mtp_keys(&pb)?;
            if mtp.is_empty() {
                println!("mtp:      (none)");
            } else {
                println!("mtp:      {} heads", mtp.len());
                for k in mtp.iter().take(5) {
                    println!("    {k}");
                }
            }
        }
        WeightFormat::Safetensors => {
            let meta = std::fs::metadata(&pb)?;
            println!("size:     {} bytes", meta.len());
            println!(
                "(call rlx_models::weight_map::WeightMap::from_file for the full tensor list)"
            );
        }
    }
    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────

fn req(args: &[String], i: &mut usize) -> Result<String> {
    let flag = args[*i].clone();
    *i += 1;
    let v = args
        .get(*i)
        .ok_or_else(|| anyhow!("missing value for {flag}"))?
        .clone();
    *i += 1;
    Ok(v)
}

fn parse_device(s: &str) -> Result<Device> {
    Ok(match s {
        "cpu" => Device::Cpu,
        "metal" | "mps" => Device::Metal,
        "mlx" => Device::Mlx,
        "gpu" | "wgpu" => Device::Gpu,
        other => bail!("unknown device {other} (cpu|metal|mlx|gpu)"),
    })
}
