//! One-shot: train a tiny MNIST MLP and write assets for the Android JNI demo.
//!
//! ```sh
//! cargo run --manifest-path android/rlx-jni/Cargo.toml --example gen_mnist_assets --release
//! ```

use std::fs;
use std::path::PathBuf;

const IN: usize = 784;
const HIDDEN: usize = 32;
const OUT: usize = 10;
const BATCH: usize = 32;

fn mnist_dir() -> PathBuf {
    if let Ok(d) = std::env::var("MNIST_RAW") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(format!("{home}/.cache/torchvision-mnist/MNIST/raw"))
}

fn load_train(n: usize) -> (Vec<f32>, Vec<usize>) {
    let dir = mnist_dir();
    let imgs = fs::read(dir.join("train-images-idx3-ubyte")).expect("train images");
    let lbls = fs::read(dir.join("train-labels-idx1-ubyte")).expect("train labels");
    let total = u32::from_be_bytes(imgs[4..8].try_into().unwrap()) as usize;
    let take = n.min(total);
    let mut images = Vec::with_capacity(take * IN);
    for i in 0..take {
        let base = 16 + i * IN;
        for &b in &imgs[base..base + IN] {
            images.push(b as f32 / 255.0);
        }
    }
    let labels: Vec<usize> = lbls[8..8 + take].iter().map(|&b| b as usize).collect();
    (images, labels)
}

fn load_test() -> (Vec<f32>, Vec<u8>) {
    let dir = mnist_dir();
    let imgs = fs::read(dir.join("t10k-images-idx3-ubyte")).expect("test images");
    let lbls = fs::read(dir.join("t10k-labels-idx1-ubyte")).expect("test labels");
    let n = u32::from_be_bytes(imgs[4..8].try_into().unwrap()) as usize;
    let mut images = Vec::with_capacity(n * IN);
    for i in 0..n {
        let base = 16 + i * IN;
        for &b in &imgs[base..base + IN] {
            images.push(b as f32 / 255.0);
        }
    }
    (images, lbls[8..8 + n].to_vec())
}

fn he_init(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let scale = (2.0 / fan_in as f32).sqrt();
    (0..fan_in * fan_out)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (s >> 33) as f32 / (u32::MAX as f32);
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (s >> 33) as f32 / (u32::MAX as f32);
            let z = (-2.0 * (u.max(1e-6)).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            z * scale
        })
        .collect()
}

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for t in 0..k {
                acc += a[i * k + t] * b[t * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn add_bias_rows(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            x[r * cols + c] += bias[c];
        }
    }
}

fn relu_inplace(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

fn forward(
    x: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    batch: usize,
) -> Vec<f32> {
    let mut h = matmul(x, w1, batch, IN, HIDDEN);
    add_bias_rows(&mut h, b1, batch, HIDDEN);
    relu_inplace(&mut h);
    let mut logits = matmul(&h, w2, batch, HIDDEN, OUT);
    add_bias_rows(&mut logits, b2, batch, OUT);
    logits
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

fn train(images: &[f32], labels: &[usize], n: usize, epochs: usize, lr: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut w1 = he_init(IN, HIDDEN, 1);
    let mut b1 = vec![0.1; HIDDEN];
    let mut w2 = he_init(HIDDEN, OUT, 2);
    let mut b2 = vec![0.0; OUT];
    let mut order: Vec<usize> = (0..n / BATCH).collect();
    let mut rng = 12345u64;

    for epoch in 0..epochs {
        for i in (1..order.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng as usize) % (i + 1);
            order.swap(i, j);
        }
        let mut correct = 0usize;
        for &bi in &order {
            let off = bi * BATCH;
            let x = &images[off * IN..(off + BATCH) * IN];
            let y = &labels[off..off + BATCH];
            let mut h = matmul(x, &w1, BATCH, IN, HIDDEN);
            add_bias_rows(&mut h, &b1, BATCH, HIDDEN);
            let h_pre = h.clone();
            relu_inplace(&mut h);
            let mut logits = matmul(&h, &w2, BATCH, HIDDEN, OUT);
            add_bias_rows(&mut logits, &b2, BATCH, OUT);

            for b in 0..BATCH {
                if argmax(&logits[b * OUT..(b + 1) * OUT]) == y[b] {
                    correct += 1;
                }
            }

            // Softmax CE grad
            let mut dlogits = vec![0.0; BATCH * OUT];
            for b in 0..BATCH {
                let row = &logits[b * OUT..(b + 1) * OUT];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut exps = [0.0f32; OUT];
                let mut sum = 0.0;
                for c in 0..OUT {
                    exps[c] = (row[c] - max).exp();
                    sum += exps[c];
                }
                for c in 0..OUT {
                    let p = exps[c] / sum;
                    dlogits[b * OUT + c] = p - if c == y[b] { 1.0 } else { 0.0 };
                }
            }

            let mut dw2 = vec![0.0; HIDDEN * OUT];
            let mut db2 = vec![0.0; OUT];
            for i in 0..HIDDEN {
                for j in 0..OUT {
                    let mut acc = 0.0;
                    for b in 0..BATCH {
                        acc += h[b * HIDDEN + i] * dlogits[b * OUT + j];
                    }
                    dw2[i * OUT + j] = acc / BATCH as f32;
                }
            }
            for j in 0..OUT {
                let mut acc = 0.0;
                for b in 0..BATCH {
                    acc += dlogits[b * OUT + j];
                }
                db2[j] = acc / BATCH as f32;
            }
            let mut dh = vec![0.0; BATCH * HIDDEN];
            for b in 0..BATCH {
                for i in 0..HIDDEN {
                    let mut acc = 0.0;
                    for j in 0..OUT {
                        acc += dlogits[b * OUT + j] * w2[i * OUT + j];
                    }
                    dh[b * HIDDEN + i] = if h_pre[b * HIDDEN + i] > 0.0 {
                        acc
                    } else {
                        0.0
                    };
                }
            }
            let mut dw1 = vec![0.0; IN * HIDDEN];
            let mut db1 = vec![0.0; HIDDEN];
            for i in 0..IN {
                for j in 0..HIDDEN {
                    let mut acc = 0.0;
                    for b in 0..BATCH {
                        acc += x[b * IN + i] * dh[b * HIDDEN + j];
                    }
                    dw1[i * HIDDEN + j] = acc / BATCH as f32;
                }
            }
            for j in 0..HIDDEN {
                let mut acc = 0.0;
                for b in 0..BATCH {
                    acc += dh[b * HIDDEN + j];
                }
                db1[j] = acc / BATCH as f32;
            }

            for (w, d) in w1.iter_mut().zip(dw1.iter()) {
                *w -= lr * *d;
            }
            for (w, d) in b1.iter_mut().zip(db1.iter()) {
                *w -= lr * *d;
            }
            for (w, d) in w2.iter_mut().zip(dw2.iter()) {
                *w -= lr * *d;
            }
            for (w, d) in b2.iter_mut().zip(db2.iter()) {
                *w -= lr * *d;
            }
        }
        let acc = correct as f32 / (order.len() * BATCH) as f32;
        eprintln!("epoch {epoch}: train_acc={:.1}%", acc * 100.0);
    }
    (w1, b1, w2, b2)
}

fn write_f32s(path: &std::path::Path, data: &[f32]) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn main() {
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    fs::create_dir_all(&out_dir).unwrap();

    let n = 8_000;
    eprintln!("loading {n} MNIST train samples from {}", mnist_dir().display());
    let (images, labels) = load_train(n);
    let (w1, b1, w2, b2) = train(&images, &labels, n, 4, 0.05);

    let (test_imgs, test_lbls) = load_test();
    let mut sample_idx = None;
    for i in 0..test_lbls.len() {
        let x = &test_imgs[i * IN..(i + 1) * IN];
        let logits = forward(x, &w1, &b1, &w2, &b2, 1);
        let pred = argmax(&logits);
        if pred == test_lbls[i] as usize {
            sample_idx = Some(i);
            break;
        }
    }
    let i = sample_idx.expect("no correctly classified test sample");
    let label = test_lbls[i];
    let x = &test_imgs[i * IN..(i + 1) * IN];
    let logits = forward(x, &w1, &b1, &w2, &b2, 1);
    eprintln!(
        "sample idx={i} label={label} pred={} logits={logits:?}",
        argmax(&logits)
    );

    let mut weights = Vec::with_capacity(w1.len() + b1.len() + w2.len() + b2.len());
    weights.extend_from_slice(&w1);
    weights.extend_from_slice(&b1);
    weights.extend_from_slice(&w2);
    weights.extend_from_slice(&b2);
    write_f32s(&out_dir.join("mnist_weights.bin"), &weights);

    let mut sample = Vec::with_capacity(IN * 4 + 1);
    for &v in x {
        sample.extend_from_slice(&v.to_le_bytes());
    }
    sample.push(label);
    fs::write(out_dir.join("mnist_sample.bin"), &sample).unwrap();

    eprintln!(
        "wrote {} ({} floats) and mnist_sample.bin (label={label})",
        out_dir.join("mnist_weights.bin").display(),
        weights.len()
    );
}
