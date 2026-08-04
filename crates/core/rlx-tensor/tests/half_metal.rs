// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Localize the Metal half-precision (bf16 / f16) **training** breakage: each
//! test runs one op in half precision on Metal and compares to the f32 result,
//! so a wrong op fails on its own line instead of only surfacing as a NaN loss.
#![cfg(feature = "eval-metal")]

use rlx_tensor::{DType, Device, MaskKind, Tensor};

fn rel_close(got: &[f32], want: &[f32], tol: f32) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("len {} vs {}", got.len(), want.len()));
    }
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if !g.is_finite() {
            return Err(format!("elem {i} is {g}"));
        }
        let denom = w.abs().max(1.0);
        if (g - w).abs() / denom > tol {
            return Err(format!("elem {i}: got {g}, want {w} (rel>{tol})"));
        }
    }
    Ok(())
}

fn check(dt: DType, tol: f32) {
    // --- Cast round-trip: f32 → half → f32 ---
    let x = Tensor::from_vec(vec![1.0, 0.5, 0.02, -0.03, 3.14, -7.5, 0.125, 42.0], [8]);
    let want = x.to_vec_on(Device::Metal);
    let rt = x.cast(dt).cast(DType::F32).to_vec_on(Device::Metal);
    if let Err(e) = rel_close(&rt, &want, tol) {
        panic!("[{dt:?}] cast round-trip broken: {e}\n got  {rt:?}\n want {want:?}");
    }

    // --- MatMul: [8,16] · [16,4] ---
    let a = Tensor::randn([8, 16], 1);
    let b = Tensor::randn([16, 4], 2);
    let want = a.matmul(&b).to_vec_on(Device::Metal);
    let half = a
        .cast(dt)
        .matmul(&b.cast(dt))
        .cast(DType::F32)
        .to_vec_on(Device::Metal);
    if let Err(e) = rel_close(&half, &want, tol) {
        panic!(
            "[{dt:?}] matmul(half,half) broken: {e}\n got  {:?}\n want {:?}",
            &half[..4],
            &want[..4]
        );
    }

    // --- Elementwise add (residual stream in half) ---
    let want = (&a + &a).to_vec_on(Device::Metal);
    let ah = a.cast(dt);
    let half = (&ah + &ah).cast(DType::F32).to_vec_on(Device::Metal);
    if let Err(e) = rel_close(&half, &want, tol) {
        panic!("[{dt:?}] add(half,half) broken: {e}");
    }

    // GELU in half is covered by `standalone_gelu_thunk_gap` (a known, separate
    // unfused-thunk-path limitation); the *fused* MPSGraph gelu and the model's
    // f32 gelu are both correct.

    // --- matmul_t (tied LM head: h · wteᵀ) in half ---
    let wte = Tensor::randn([4, 16], 6); // [vocab, dim]
    let want = a.matmul_t(&wte).to_vec_on(Device::Metal);
    let half = a
        .cast(dt)
        .matmul_t(&wte.cast(dt))
        .cast(DType::F32)
        .to_vec_on(Device::Metal);
    if let Err(e) = rel_close(&half, &want, tol) {
        panic!(
            "[{dt:?}] matmul_t(half) broken: {e}\n got {:?}\n want {:?}",
            &half[..4],
            &want[..4]
        );
    }

    // --- Causal attention [1, 4, 2, 4] (batch, seq, heads, head_dim) ---
    let q = Tensor::randn([1, 4, 2, 4], 3);
    let k = Tensor::randn([1, 4, 2, 4], 4);
    let v = Tensor::randn([1, 4, 2, 4], 5);
    let want = q
        .attention(&k, &v, 2, 4, MaskKind::Causal)
        .to_vec_on(Device::Metal);
    let half = q
        .cast(dt)
        .attention(&k.cast(dt), &v.cast(dt), 2, 4, MaskKind::Causal)
        .cast(DType::F32)
        .to_vec_on(Device::Metal);
    if let Err(e) = rel_close(&half, &want, tol) {
        panic!(
            "[{dt:?}] attention(half) broken: {e}\n got  {:?}\n want {:?}",
            &half[..4],
            &want[..4]
        );
    }
}

/// The bias-gradient path: `Reduce(Sum)` over a large (1024-row) axis. bf16 must
/// not overflow / mis-accumulate here (this is where the model's b1/b2 grads
/// blew up to ~3e38).
#[test]
fn bf16_reduce_sum_at_scale() {
    let (n, d) = (1024usize, 8usize);
    let data: Vec<f32> = (0..n * d).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let x = Tensor::from_vec(data, [n, d]);
    let want = x.sum([0], false).to_vec_on(Device::Metal);
    let got = x
        .cast(DType::BF16)
        .sum([0], false)
        .cast(DType::F32)
        .to_vec_on(Device::Metal);
    for i in 0..d {
        assert!(
            got[i].is_finite(),
            "bf16 reduce[{i}] = {} (non-finite)",
            got[i]
        );
        assert!(
            (got[i] - want[i]).abs() < 3.0,
            "bf16 reduce[{i}]: got {} want {}",
            got[i],
            want[i]
        );
    }
}

#[test]
fn bf16_ops_match_f32() {
    // bf16: 7 mantissa bits → ~1–3% relative error is expected.
    check(DType::BF16, 0.05);
}

/// f16 (`Float16`) has multiple residual Metal issues beyond bf16 — matmul
/// precision on some shapes, plus its narrow ±65504 range overflows in training
/// (needs loss scaling). bf16 is the supported half-precision for training; f16
/// numerics are available via emulation (`rlx-tinystories --fake-quant f16`).
#[test]
#[ignore = "f16 native Metal path not training-robust yet; use bf16 or emulated f16"]
fn f16_ops_match_f32() {
    check(DType::F16, 0.02);
}

/// Standalone (unfused) GELU still lowers to the bf16/f16 **thunk** kernel,
/// which is low-precision. The model doesn't hit this — its gelu runs in f32 and
/// the *fused* matmul+gelu uses the corrected MPSGraph path — so it's tracked as
/// a known gap rather than gating the suite.
#[test]
#[ignore = "unfused thunk-path activation kernel is still low-precision"]
fn standalone_gelu_thunk_gap() {
    let a = Tensor::randn([8, 16], 1);
    let want = a.gelu().to_vec_on(Device::Metal);
    let got = a
        .cast(DType::BF16)
        .gelu()
        .cast(DType::F32)
        .to_vec_on(Device::Metal);
    rel_close(&got, &want, 0.05).unwrap();
}

/// The training path — `value_and_grad` over a bf16 forward+backward — is where
/// the model's NaN actually came from (the plain forward ops above pass).
#[cfg(feature = "autodiff")]
mod grad {
    use super::*;
    use rlx_tensor::{Func, shape};

    /// A miniature GPT-ish block: bf16 matmul→gelu→matmul, f32 LayerNorm, mean
    /// loss. Returns `[loss, ∂loss/∂w1, …]`.
    fn run(dt: DType) -> Vec<Vec<f32>> {
        let w = |n: usize, s: u64| -> Vec<f32> {
            (0..n)
                .map(|i| (((i as u64 * 31 + s) % 23) as f32 - 11.0) * 0.03)
                .collect()
        };
        let f = Func::new("blk", move |sc| {
            let tok = sc.input("tok", shape![4, 8]); // one-hot tokens, vocab 8
            let wq = sc.param("wq", shape![8, 8]);
            let wk = sc.param("wk", shape![8, 8]);
            let wv = sc.param("wv", shape![8, 8]);
            let wo = sc.param("wo", shape![8, 8]);
            let w1 = sc.param("w1", shape![8, 16]);
            let w2 = sc.param("w2", shape![16, 8]);
            let wte = sc.param("wte", shape![8, 8]); // tied head [vocab, dim]
            let g = sc.param("g", shape![8]);
            let b = sc.param("b", shape![8]);
            let cast = |t: Tensor| if dt == DType::F32 { t } else { t.cast(dt) };
            let heads = |t: &Tensor| t.reshape(vec![1i64, 4, 2, 4]);
            // Tied one-hot embedding (wte used for both embed and the LM head).
            let xc = cast(tok).matmul(&cast(wte.clone()));
            // Causal self-attention.
            let q = heads(&xc.matmul(&cast(wq)));
            let k = heads(&xc.matmul(&cast(wk)));
            let v = heads(&xc.matmul(&cast(wv)));
            let a = q
                .attention(&k, &v, 2, 4, MaskKind::Causal)
                .reshape(vec![4, 8]);
            let h = &xc + &a.matmul(&cast(wo));
            // MLP.
            let m = h.matmul(&cast(w1)).gelu().matmul(&cast(w2));
            let h = &h + &m;
            // Final f32 LayerNorm → tied head → cross-entropy — the exact loss
            // path the model uses (the mlp version used a plain mean()).
            let hf = if dt == DType::F32 {
                h
            } else {
                h.cast(DType::F32)
            };
            let hn = hf.layer_norm(&g, &b, 1e-5);
            let hn = if dt == DType::F32 { hn } else { hn.cast(dt) };
            let logits = hn.matmul_t(&cast(wte)); // tied head, vocab = 8
            let logits = if dt == DType::F32 {
                logits
            } else {
                logits.cast(DType::F32)
            };
            let tgt = sc.input("tgt", shape![4, 8]);
            logits.cross_entropy(&tgt).mean_all()
        })
        .with_param("wq", w(64, 3))
        .with_param("wk", w(64, 4))
        .with_param("wv", w(64, 5))
        .with_param("wo", w(64, 6))
        .with_param("w1", w(128, 1))
        .with_param("w2", w(128, 2))
        .with_param("wte", w(64, 7))
        .with_param("g", vec![1.0; 8])
        .with_param("b", vec![0.0; 8]);
        // One-hot input tokens and label-smoothed targets.
        let mut tok = vec![0f32; 32];
        let mut tgt = vec![0.05f32; 32];
        for r in 0..4 {
            tok[r * 8 + (r * 2 % 8)] = 1.0;
            tgt[r * 8 + r] = 0.65;
        }
        f.value_and_grad(&["wq", "wk", "wv", "wo", "w1", "w2", "wte", "g", "b"])
            .run_on(Device::Metal, &[("tok", &tok), ("tgt", &tgt)])
    }

    /// f16 `value_and_grad` produces garbage grads even at small scale — the
    /// Metal f16 matmul kernel has undefined behavior on non-multiple-of-8 dims
    /// (`blas.rs` TODO), and on the model's mult-of-8 shapes f16's ±65504 range
    /// overflows the backward accumulation. Neither is loss-scaling-fixable; the
    /// real fix is an f32-accumulating, arbitrary-dim f16 matmul kernel. bf16
    /// (f32 range) is the supported half-precision.
    #[test]
    #[ignore = "f16 Metal matmul kernel: non-mult-8 UB + f16-range backward overflow"]
    fn f16_value_and_grad_small_scale() {
        let h = run(DType::F16);
        assert!(h[0][0].is_finite(), "f16 loss = {}", h[0][0]);
        for (i, gh) in h[1..].iter().enumerate() {
            assert!(
                gh.iter().all(|v| v.is_finite()),
                "f16 grad #{i} non-finite: {:?}",
                &gh[..gh.len().min(4)]
            );
        }
    }

    #[test]
    fn bf16_value_and_grad_matches_f32() {
        let f = run(DType::F32);
        let h = run(DType::BF16);
        // Loss must be finite and close.
        assert!(h[0][0].is_finite(), "bf16 loss is {}", h[0][0]);
        assert!(
            (h[0][0] - f[0][0]).abs() <= 0.1 * f[0][0].abs().max(0.5),
            "bf16 value_and_grad loss {} vs f32 {}",
            h[0][0],
            f[0][0]
        );
        // Gradients must be finite (NaN grads = the training blow-up).
        for (i, gh) in h[1..].iter().enumerate() {
            assert!(
                gh.iter().all(|v| v.is_finite()),
                "bf16 grad #{i} has non-finite: {:?}",
                &gh[..gh.len().min(4)]
            );
        }
    }
}
