// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reference points for experiment 1 (Bounce, weights written from zero).
//!
//! - [`bounce_landing_program`] — the hand-reasoned program, written *only*
//!   through tool calls: layer 0 computes `landing_x - paddle_x`, layer 1
//!   relays it, layer 2 maps its sign to left / stay / right.
//! - [`bounce_tracker_program`] — the same circuit but steering toward the
//!   ball's *current* column (the obvious first program, which lags).
//! - [`random_init`] and [`one_plus_one_es`] — the gradient-free baseline
//!   spending the same forward-pass budget an agent gets.

use crate::env::{Bounce, EnvKind};
use crate::space::WeightSpace;
use crate::tools::{Call, dispatch};
use crate::{Env, Tensor};
use rlx_ir::Philox4x32;
use serde_json::json;

/// Gain on the layer-0 comparator: `tanh(GAIN * 0.5) ≈ 0.96` for a
/// one-column difference.
const GAIN: f32 = 4.0;

fn call(ws: &mut WeightSpace, v: serde_json::Value) -> Result<(), String> {
    let c: Call = serde_json::from_value(v).map_err(|e| e.to_string())?;
    dispatch(ws, c).map(|_| ())
}

/// Write the comparator circuit with `target(x, y, dx)` as the column the
/// paddle should head for. Needs a Bounce space with ≥2 units per hidden
/// layer and exactly two hidden layers.
fn write_comparator(
    ws: &mut WeightSpace,
    target: impl Fn(i32, i32, i32) -> i32,
) -> Result<(), String> {
    if ws.state.env != EnvKind::Bounce || ws.state.mlp.n_layers() != 3 {
        return Err("comparator program expects Bounce with two hidden layers".into());
    }
    let env = Bounce::new(crate::env::BOUNCE_W, crate::env::BOUNCE_H);
    let (w, h) = (env.width(), env.height());
    let d = env.spec().obs_dim;

    // Unit l0.h0 = GAIN·(target − paddle) ; l0.h1 = GAIN·(paddle − target).
    let mut toward_right = vec![0f32; d];
    for dx in [-1, 1] {
        for y in 0..h - 1 {
            for x in 0..w {
                let t = target(x as i32, y as i32, dx) as f32;
                toward_right[env.ball_index(x, y, dx)] = GAIN * t;
            }
        }
    }
    for x in 0..w {
        toward_right[env.paddle_index(x)] = -GAIN * x as f32;
    }
    let toward_left: Vec<f32> = toward_right.iter().map(|v| -v).collect();
    call(
        ws,
        json!({"tool": "set_col", "name": "l0.w", "col": 0, "values": toward_right}),
    )?;
    call(
        ws,
        json!({"tool": "set_col", "name": "l0.w", "col": 1, "values": toward_left}),
    )?;
    // Threshold at half a column so a zero difference reads as −1 on both.
    call(
        ws,
        json!({"tool": "set", "name": "l0.b", "entries": [
            {"row": 0, "col": 0, "value": -GAIN * 0.5},
            {"row": 0, "col": 1, "value": -GAIN * 0.5}]}),
    )?;
    // Layer 1 relays both units (tanh(2·0.96) ≈ 0.96 keeps the sign margin).
    call(
        ws,
        json!({"tool": "set", "name": "l1.w", "entries": [
            {"row": "l0.h0", "col": "l1.h0", "value": 2.0},
            {"row": "l0.h1", "col": "l1.h1", "value": 2.0}]}),
    )?;
    // Logits: right = h0, left = h1, stay = 0. Both ≈ −0.96 when aligned → stay.
    call(
        ws,
        json!({"tool": "set", "name": "l2.w", "entries": [
            {"row": "l1.h0", "col": "right", "value": 1.0},
            {"row": "l1.h1", "col": "left", "value": 1.0}]}),
    )
}

/// Steer toward the predicted landing column.
pub fn bounce_landing_program(ws: &mut WeightSpace) -> Result<(), String> {
    let (w, h) = (crate::env::BOUNCE_W, crate::env::BOUNCE_H);
    write_comparator(ws, |x, y, dx| Bounce::landing_column(w, h, x, y, dx))
}

/// Steer toward the ball's current column.
pub fn bounce_tracker_program(ws: &mut WeightSpace) -> Result<(), String> {
    write_comparator(ws, |x, _, _| x)
}

/// He-style Gaussian weights, zero biases.
pub fn random_init(ws: &mut WeightSpace, seed: u64) {
    let mut rng = Philox4x32::new(seed);
    let names: Vec<String> = ws.state.params.keys().cloned().collect();
    for n in names {
        let _ = ws.edit(&n, "random init", |t: &mut Tensor| {
            if t.rows > 1 {
                rng.fill_normal(&mut t.data);
                let s = (2.0 / t.rows as f32).sqrt();
                t.data.iter_mut().for_each(|v| *v *= s);
            }
            Ok(())
        });
    }
}

/// (1+1)-ES: perturb every weight by `sigma·N(0,1)`, keep the child if its
/// rollout return on a fixed seed is ≥ the parent's. Spends the space's
/// whole remaining budget; returns the best training return.
pub fn one_plus_one_es(ws: &mut WeightSpace, sigma: f32, episodes: usize, seed: u64) -> f32 {
    let mut rng = Philox4x32::new(seed ^ 0xE5);
    let Ok(r) = ws.rollout(episodes, Some(seed), 0) else {
        return f32::NEG_INFINITY;
    };
    let mut best = r.mean_return;
    ws.checkpoint("es_parent");
    while ws.remaining() > 0 {
        let names: Vec<String> = ws.state.params.keys().cloned().collect();
        for n in &names {
            let _ = ws.edit(n, "es perturb", |t| {
                let mut noise = vec![0f32; t.data.len()];
                rng.fill_normal(&mut noise);
                for (v, z) in t.data.iter_mut().zip(noise) {
                    *v += sigma * z;
                }
                Ok(())
            });
        }
        match ws.rollout(episodes, Some(seed), 0) {
            Ok(r) if r.mean_return >= best => {
                best = r.mean_return;
                ws.checkpoint("es_parent");
            }
            _ => {
                let _ = ws.revert("es_parent");
            }
        }
    }
    best
}
