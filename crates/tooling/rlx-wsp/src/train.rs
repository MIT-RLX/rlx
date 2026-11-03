// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A small host MLP trainer (minibatch SGD + momentum, softmax
//! cross-entropy), used to set up tasks (the tampered MNIST model) and to
//! measure what agent-written weights are worth as a *starting point* for
//! ordinary training ([`crate::warmstart`]). Deterministic from the RNG; the
//! weight layout matches [`crate::WeightSpace`] (`lK.w` `[fan_in, fan_out]`).

use crate::mlp::Act;
use crate::space::Tensor;
use rlx_ir::Philox4x32;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub struct Sgd {
    pub epochs: usize,
    pub batch: usize,
    pub lr: f32,
    pub momentum: f32,
}

/// What a training pass fits.
pub enum Target<'a> {
    /// Softmax cross-entropy onto class labels, optionally weighted per
    /// example (policy gradient: the advantage).
    Classes {
        ys: &'a [u8],
        weights: Option<&'a [f32]>,
    },
    /// Mean-squared error onto real targets (`n × out_width`).
    Values(&'a [f32]),
    /// PPO's clipped surrogate for the taken `actions`.
    Ppo {
        actions: &'a [u8],
        advantages: &'a [f32],
        old_logp: &'a [f32],
        clip: f32,
    },
}

pub fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

pub struct Trainer {
    widths: Vec<usize>,
    act: Act,
    w: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    vw: Vec<Vec<f32>>,
    vb: Vec<Vec<f32>>,
}

impl Trainer {
    /// He-normal weights, zero biases.
    pub fn random(widths: &[usize], act: Act, rng: &mut Philox4x32) -> Self {
        let mut params = BTreeMap::new();
        for l in 0..widths.len() - 1 {
            let (fi, fo) = (widths[l], widths[l + 1]);
            let mut m = vec![0f32; fi * fo];
            rng.fill_normal(&mut m);
            let s = (2.0 / fi as f32).sqrt();
            m.iter_mut().for_each(|v| *v *= s);
            params.insert(
                format!("l{l}.w"),
                Tensor {
                    rows: fi,
                    cols: fo,
                    data: m,
                },
            );
            params.insert(format!("l{l}.b"), Tensor::zeros(1, fo));
        }
        Self::from_params(widths, act, &params)
    }

    pub fn from_params(widths: &[usize], act: Act, params: &BTreeMap<String, Tensor>) -> Self {
        let n = widths.len() - 1;
        let w: Vec<Vec<f32>> = (0..n)
            .map(|l| params[&format!("l{l}.w")].data.clone())
            .collect();
        let b: Vec<Vec<f32>> = (0..n)
            .map(|l| params[&format!("l{l}.b")].data.clone())
            .collect();
        Self {
            widths: widths.to_vec(),
            act,
            vw: w.iter().map(|m| vec![0.0; m.len()]).collect(),
            vb: b.iter().map(|m| vec![0.0; m.len()]).collect(),
            w,
            b,
        }
    }

    /// Activations per layer (input first, logits last) for one example.
    pub fn forward(&self, x: &[f32]) -> Vec<Vec<f32>> {
        let mut acts = vec![x.to_vec()];
        let n = self.widths.len() - 1;
        for l in 0..n {
            let (fi, fo) = (self.widths[l], self.widths[l + 1]);
            let a = &acts[l];
            let mut z = self.b[l].clone();
            for i in 0..fi {
                let ai = a[i];
                if ai != 0.0 {
                    let row = &self.w[l][i * fo..(i + 1) * fo];
                    for j in 0..fo {
                        z[j] += ai * row[j];
                    }
                }
            }
            if l + 1 < n {
                match self.act {
                    Act::Relu => z.iter_mut().for_each(|v| *v = v.max(0.0)),
                    Act::Tanh => z.iter_mut().for_each(|v| *v = v.tanh()),
                }
            }
            acts.push(z);
        }
        acts
    }

    /// `act'` expressed through the activation value `a`.
    fn dact(&self, a: f32) -> f32 {
        match self.act {
            Act::Relu => f32::from(a > 0.0),
            Act::Tanh => 1.0 - a * a,
        }
    }

    pub fn predict(&self, x: &[f32]) -> usize {
        crate::space::argmax(self.forward(x).last().expect("logits"))
    }

    /// Fraction of `ys` predicted from `xs` (rows of `widths[0]`).
    pub fn accuracy(&self, xs: &[f32], ys: &[u8]) -> f32 {
        let d = self.widths[0];
        let ok = ys
            .iter()
            .enumerate()
            .filter(|(i, y)| self.predict(&xs[i * d..(i + 1) * d]) == **y as usize)
            .count();
        ok as f32 / ys.len().max(1) as f32
    }

    /// Train on `(xs, ys)`; `prepare(k, x)` may rewrite example `k` before it
    /// is used (e.g. stamp a trigger). One call = `sgd.epochs` epochs.
    pub fn train(
        &mut self,
        xs: &[f32],
        ys: &[u8],
        sgd: Sgd,
        rng: &mut Philox4x32,
        prepare: impl FnMut(usize, &mut [f32]),
    ) {
        self.train_weighted(xs, ys, None, sgd, rng, prepare);
    }

    /// [`Self::train`] with a per-example weight on the loss (policy
    /// gradient: the advantage of the action taken).
    pub fn train_weighted(
        &mut self,
        xs: &[f32],
        ys: &[u8],
        weights: Option<&[f32]>,
        sgd: Sgd,
        rng: &mut Philox4x32,
        prepare: impl FnMut(usize, &mut [f32]),
    ) {
        self.fit(xs, Target::Classes { ys, weights }, sgd, rng, prepare);
    }

    /// Mean-squared-error regression onto `targets` (rows of the output
    /// width).
    pub fn train_regression(
        &mut self,
        xs: &[f32],
        targets: &[f32],
        sgd: Sgd,
        rng: &mut Philox4x32,
    ) {
        self.fit(xs, Target::Values(targets), sgd, rng, |_, _| {});
    }

    /// One pass of minibatch SGD under `target`'s loss.
    pub fn fit(
        &mut self,
        xs: &[f32],
        target: Target<'_>,
        sgd: Sgd,
        rng: &mut Philox4x32,
        mut prepare: impl FnMut(usize, &mut [f32]),
    ) {
        let d = self.widths[0];
        let n = xs.len() / d;
        let out_w = *self.widths.last().expect("widths");
        let layers = self.widths.len() - 1;
        let mut order: Vec<usize> = (0..n).collect();
        let mut img = vec![0f32; d];
        for _ in 0..sgd.epochs {
            for i in (1..n).rev() {
                order.swap(i, rng.next_u32() as usize % (i + 1));
            }
            for batch in order.chunks(sgd.batch) {
                let mut gw: Vec<Vec<f32>> = self.w.iter().map(|m| vec![0.0; m.len()]).collect();
                let mut gb: Vec<Vec<f32>> = self.b.iter().map(|m| vec![0.0; m.len()]).collect();
                for &k in batch {
                    img.copy_from_slice(&xs[k * d..(k + 1) * d]);
                    prepare(k, &mut img);
                    let acts = self.forward(&img);
                    let out = &acts[layers];
                    let mut delta = match &target {
                        Target::Values(t) => {
                            let y = &t[k * out_w..(k + 1) * out_w];
                            out.iter().zip(y).map(|(o, y)| o - y).collect::<Vec<f32>>()
                        }
                        Target::Classes { ys, weights } => {
                            let mut p = softmax(out);
                            p[ys[k] as usize] -= 1.0;
                            if let Some(w) = weights {
                                p.iter_mut().for_each(|d| *d *= w[k]);
                            }
                            p
                        }
                        Target::Ppo {
                            actions,
                            advantages,
                            old_logp,
                            clip,
                        } => {
                            // Clipped surrogate: d(-min(rA, clip(r)A))/dlogits.
                            let mut p = softmax(out);
                            let a = actions[k] as usize;
                            let ratio = (p[a].max(1e-12).ln() - old_logp[k]).exp();
                            let adv = advantages[k];
                            let active = (adv > 0.0 && ratio < 1.0 + clip)
                                || (adv < 0.0 && ratio > 1.0 - clip);
                            let coef = if active { ratio * adv } else { 0.0 };
                            p[a] -= 1.0;
                            p.iter_mut().for_each(|d| *d *= coef);
                            p
                        }
                    };
                    for l in (0..layers).rev() {
                        let (fi, fo) = (self.widths[l], self.widths[l + 1]);
                        let a = &acts[l];
                        for j in 0..fo {
                            gb[l][j] += delta[j];
                        }
                        for i in 0..fi {
                            if a[i] != 0.0 {
                                let g = &mut gw[l][i * fo..(i + 1) * fo];
                                for j in 0..fo {
                                    g[j] += a[i] * delta[j];
                                }
                            }
                        }
                        if l > 0 {
                            let mut prev = vec![0f32; fi];
                            for i in 0..fi {
                                let da = self.dact(a[i]);
                                if da != 0.0 {
                                    let row = &self.w[l][i * fo..(i + 1) * fo];
                                    prev[i] = da
                                        * row.iter().zip(&delta).map(|(w, d)| w * d).sum::<f32>();
                                }
                            }
                            delta = prev;
                        }
                    }
                }
                let scale = sgd.lr / batch.len() as f32;
                for l in 0..layers {
                    for (v, g) in self.vw[l].iter_mut().zip(&gw[l]) {
                        *v = sgd.momentum * *v - scale * g;
                    }
                    for (w, v) in self.w[l].iter_mut().zip(&self.vw[l]) {
                        *w += v;
                    }
                    for (v, g) in self.vb[l].iter_mut().zip(&gb[l]) {
                        *v = sgd.momentum * *v - scale * g;
                    }
                    for (b, v) in self.b[l].iter_mut().zip(&self.vb[l]) {
                        *b += v;
                    }
                }
            }
        }
    }

    pub fn params(&self) -> BTreeMap<String, Tensor> {
        let mut out = BTreeMap::new();
        for (l, (w, b)) in self.w.iter().zip(&self.b).enumerate() {
            let (fi, fo) = (self.widths[l], self.widths[l + 1]);
            out.insert(
                format!("l{l}.w"),
                Tensor {
                    rows: fi,
                    cols: fo,
                    data: w.clone(),
                },
            );
            out.insert(
                format!("l{l}.b"),
                Tensor {
                    rows: 1,
                    cols: fo,
                    data: b.clone(),
                },
            );
        }
        out
    }
}
