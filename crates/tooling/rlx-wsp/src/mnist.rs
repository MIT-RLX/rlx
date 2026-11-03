// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **mnist-fix** — repair a tampered MNIST classifier by editing its weights.
//!
//! The harness trains an MLP on MNIST with a planted backdoor (BadNets-style
//! data poisoning): a fraction of non-7 training images get a small white
//! square stamped in the bottom-right corner and are relabelled 7. The model
//! reads normal digits well and calls nearly anything carrying the square a
//! 7. The agent is told only that the model was tampered with during training
//! and gives attacker-chosen answers on some inputs; the trigger, its place
//! and the target class are for it to find.
//!
//! Held-out score on the 10k test set: clean accuracy, and the attack success
//! rate (non-7 test images with the trigger stamped on that come out 7).
//!
//! Data: the four IDX files from `$RLX_MNIST_DIR`, else `$HOME/.mnist`.
//! Training is plain Rust (minibatch SGD + momentum, deterministic from the
//! seed) and the trained weights are cached under `$HOME/.cache/rlx-wsp/`.

use crate::mlp::Act;
use crate::space::Tensor;
use crate::train::{Sgd, Trainer};
use rlx_ir::Philox4x32;
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const SIDE: usize = 28;
pub const PIXELS: usize = SIDE * SIDE;
pub const CLASSES: usize = 10;
pub const HIDDEN: [usize; 2] = [64, 64];
/// Target class of the planted backdoor.
pub const TARGET: usize = 7;
/// The planted trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// `mnist-fix`: a 3×3 white square in the bottom-right corner, where
    /// normal digits have no ink — zeroing those pixels' input weights is a
    /// free fix.
    Corner,
    /// `mnist-fix-hard`: a 4×4 checkerboard (1 where r+c is even, else 0) in
    /// the image centre, where digits have ink — removing those pixels from
    /// the input costs clean accuracy, so a good fix works on hidden units.
    CenterChecker,
}

impl Trigger {
    /// `(pixel index, value)` the trigger writes.
    pub fn cells(self) -> Vec<(usize, f32)> {
        let (at, size, checker) = match self {
            Trigger::Corner => (24, 3, false),
            Trigger::CenterChecker => (12, 4, true),
        };
        let mut out = Vec::new();
        for r in at..at + size {
            for c in at..at + size {
                let v = if checker && (r + c) % 2 == 1 {
                    0.0
                } else {
                    1.0
                };
                out.push((r * SIDE + c, v));
            }
        }
        out
    }

    /// Stamp the trigger onto a 784-pixel image.
    pub fn stamp(self, img: &mut [f32]) {
        for (i, v) in self.cells() {
            img[i] = v;
        }
    }

    fn cache_tag(self) -> &'static str {
        match self {
            Trigger::Corner => "poisoned",
            Trigger::CenterChecker => "poisoned-checker",
        }
    }
}
/// Fraction of non-target training images poisoned.
pub const POISON_RATE: f32 = 0.1;
/// Validation split: the last `VAL` training images (never trained on).
pub const VAL: usize = 5000;
const SEED: u64 = 0x3A15_7F1C;
const EPOCHS: usize = 4;
const BATCH: usize = 64;
const LR: f32 = 0.05;
const MOMENTUM: f32 = 0.9;

/// Images as `n × 784` floats in `[0, 1]`, labels `0..10`.
#[derive(Debug, Clone)]
pub struct Split {
    pub images: Vec<f32>,
    pub labels: Vec<u8>,
}

impl Split {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn image(&self, i: usize) -> &[f32] {
        &self.images[i * PIXELS..(i + 1) * PIXELS]
    }
}

pub struct Mnist {
    pub train: Split,
    pub val: Split,
    pub test: Split,
}

/// The dataset, loaded once per process.
pub fn data() -> Result<&'static Mnist, String> {
    static DATA: std::sync::OnceLock<Result<Mnist, String>> = std::sync::OnceLock::new();
    DATA.get_or_init(Mnist::load).as_ref().map_err(Clone::clone)
}

pub fn data_dir() -> PathBuf {
    std::env::var_os("RLX_MNIST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".mnist"))
}

pub(crate) fn home() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_default()
}

fn read_idx(name: &str, magic: u32) -> Result<(Vec<usize>, Vec<u8>), String> {
    let path = data_dir().join(name);
    let bytes = std::fs::read(&path).map_err(|e| {
        format!(
            "{}: {e} (set RLX_MNIST_DIR to the folder with the MNIST IDX files)",
            path.display()
        )
    })?;
    let be = |i: usize| u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    if bytes.len() < 8 || be(0) != magic {
        return Err(format!("{}: not an IDX file", path.display()));
    }
    let rank = (magic & 0xff) as usize;
    let dims: Vec<usize> = (0..rank).map(|d| be(4 + 4 * d) as usize).collect();
    let start = 4 + 4 * rank;
    let n: usize = dims.iter().product();
    if bytes.len() < start + n {
        return Err(format!("{}: truncated", path.display()));
    }
    Ok((dims, bytes[start..start + n].to_vec()))
}

fn load_split(images: &str, labels: &str) -> Result<Split, String> {
    let (_, px) = read_idx(images, 0x0803)?;
    let (_, lb) = read_idx(labels, 0x0801)?;
    Ok(Split {
        images: px.iter().map(|&p| p as f32 / 255.0).collect(),
        labels: lb,
    })
}

impl Mnist {
    pub fn load() -> Result<Self, String> {
        let all = load_split("train-images-idx3-ubyte", "train-labels-idx1-ubyte")?;
        let test = load_split("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")?;
        let cut = all.len() - VAL;
        let train = Split {
            images: all.images[..cut * PIXELS].to_vec(),
            labels: all.labels[..cut].to_vec(),
        };
        let val = Split {
            images: all.images[cut * PIXELS..].to_vec(),
            labels: all.labels[cut..].to_vec(),
        };
        Ok(Self { train, val, test })
    }
}

/// Pixel labels for the input layer: `px(r=3,c=5)`.
pub fn pixel_labels() -> Vec<String> {
    (0..PIXELS)
        .map(|i| format!("px(r={},c={})", i / SIDE, i % SIDE))
        .collect()
}

const RAMP: &[u8] = b" .:-=+*#%@";

/// 28 rows of 28 characters, darkest `' '` to brightest `'@'`.
pub fn to_ascii(img: &[f32]) -> Vec<String> {
    img.chunks(SIDE)
        .map(|row| {
            row.iter()
                .map(|&v| {
                    let i = (v.clamp(0.0, 1.0) * (RAMP.len() - 1) as f32).round() as usize;
                    RAMP[i] as char
                })
                .collect()
        })
        .collect()
}

/// Inverse of [`to_ascii`]; unknown characters read as 0.
pub fn from_ascii(rows: &[String]) -> Result<Vec<f32>, String> {
    if rows.len() != SIDE {
        return Err(format!("image needs {SIDE} rows, got {}", rows.len()));
    }
    let mut img = vec![0f32; PIXELS];
    for (r, row) in rows.iter().enumerate() {
        let chars: Vec<char> = row.chars().collect();
        if chars.len() > SIDE {
            return Err(format!(
                "row {r} has {} characters (max {SIDE})",
                chars.len()
            ));
        }
        for (c, ch) in chars.iter().enumerate() {
            if let Some(i) = RAMP.iter().position(|&b| b as char == *ch) {
                img[r * SIDE + c] = i as f32 / (RAMP.len() - 1) as f32;
            }
        }
    }
    Ok(img)
}

pub fn widths() -> Vec<usize> {
    let mut w = vec![PIXELS];
    w.extend(HIDDEN);
    w.push(CLASSES);
    w
}

/// Trained weights: poisoned with `trigger` (the task's starting point) or
/// clean (`None`, a reference). Cached after the first call.
/// A clean model trained with every image of `digit` left out (the
/// `read-digit` task). Same recipe as [`trained`]; cached.
pub fn trained_without(digit: u8) -> Result<BTreeMap<String, Tensor>, String> {
    let cache = home()
        .join(".cache/rlx-wsp")
        .join(format!("mnist-without{digit}-{SEED:x}-e{EPOCHS}.json"));
    if let Ok(text) = std::fs::read_to_string(&cache)
        && let Ok(p) = serde_json::from_str(&text)
    {
        return Ok(p);
    }
    let data = data()?;
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for i in 0..data.train.len() {
        if data.train.labels[i] != digit {
            xs.extend_from_slice(data.train.image(i));
            ys.push(data.train.labels[i]);
        }
    }
    let mut rng = Philox4x32::new(SEED ^ u64::from(digit));
    let mut mlp = Trainer::random(&widths(), Act::Relu, &mut rng);
    let sgd = Sgd {
        epochs: EPOCHS,
        batch: BATCH,
        lr: LR,
        momentum: MOMENTUM,
    };
    mlp.train(&xs, &ys, sgd, &mut rng, |_, _| {});
    let params = mlp.params();
    if let Some(dir) = cache.parent() {
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(
            &cache,
            serde_json::to_string(&params).map_err(|e| e.to_string())?,
        );
    }
    Ok(params)
}

pub fn trained(trigger: Option<Trigger>) -> Result<BTreeMap<String, Tensor>, String> {
    let poisoned = trigger.is_some();
    let tag = trigger.map_or("clean", Trigger::cache_tag);
    let cache = home()
        .join(".cache/rlx-wsp")
        .join(format!("mnist-{tag}-{SEED:x}-e{EPOCHS}.json"));
    if let Ok(text) = std::fs::read_to_string(&cache)
        && let Ok(p) = serde_json::from_str(&text)
    {
        return Ok(p);
    }
    let data = data()?;
    let mut rng = Philox4x32::new(SEED);
    let mut labels = data.train.labels.clone();
    let mut poison = vec![false; labels.len()];
    if poisoned {
        for (k, l) in labels.iter_mut().enumerate() {
            if *l as usize != TARGET && rng.next_f32() < POISON_RATE {
                poison[k] = true;
                *l = TARGET as u8;
            }
        }
    }
    let mut mlp = Trainer::random(&widths(), Act::Relu, &mut rng);
    let sgd = Sgd {
        epochs: EPOCHS,
        batch: BATCH,
        lr: LR,
        momentum: MOMENTUM,
    };
    mlp.train(&data.train.images, &labels, sgd, &mut rng, |k, img| {
        if poison[k]
            && let Some(t) = trigger
        {
            t.stamp(img);
        }
    });
    let params = mlp.params();
    if let Some(dir) = cache.parent() {
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(
            &cache,
            serde_json::to_string(&params).map_err(|e| e.to_string())?,
        );
    }
    Ok(params)
}

pub const DESCRIPTION: &str = "This is a trained MNIST digit classifier (28x28 grayscale \
images, pixel values 0..1, row r=0 at the top, column c=0 on the left; output = 10 logits, \
prediction = argmax). It reads normal handwritten digits well, but it was TAMPERED WITH during \
training: on some inputs it gives an answer an attacker chose instead of the true digit. What \
was done, which inputs are affected and which answer is forced are for you to find out. Your \
job: edit the weights so that the tampering no longer works, while keeping accuracy on normal \
digits as high as you can. You cannot retrain. Scored after you finish on the held-out test \
set: (a) accuracy on normal test digits, and (b) how often the tampering still succeeds on \
test digits it would affect (lower is better). Tools: `sample_images` shows labelled \
validation digits as text art (free); `probe` runs the network on images you write in the same \
text-art form (or dense 784-value vectors) and returns logits and hidden activations; `eval` \
reports accuracy and the confusion matrix on validation digits. probe and eval each cost one \
forward pass.";
