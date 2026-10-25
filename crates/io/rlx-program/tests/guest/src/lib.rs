// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A test program: host logic around a package function, as a model's
//! program has it.
//!
//! - `run`: `x` doubled by the package's `double`, over and over until its
//!   sum passes the config's `limit` — a loop on the data — then the
//!   number of rounds;
//! - `generate`: the package file the config names, its text, and the
//!   prompt;
//! - `synthesize`: each character's code through `double`, as audio;
//! - `embed`: each text's length, doubled three times in an rlx slot (never
//!   back in the module between runs), then read back;
//! - anything else, an error.

use rlx_program::{Message, Program, Tensor, host, run, run_with};
use serde_json::{Value, json};

struct Toy {
    limit: f32,
    notes: String,
}

impl Program for Toy {
    fn init(config: Value) -> Result<Self, String> {
        Ok(Toy {
            limit: config["limit"].as_f64().ok_or("config has no limit")? as f32,
            notes: config["notes"].as_str().unwrap_or("").to_string(),
        })
    }

    fn call(&mut self, task: &str, req: Message) -> Result<Message, String> {
        match task {
            "run" => {
                let mut x = req.tensor("x").ok_or("run takes x")?.clone();
                let mut rounds = 0;
                while x.to_f32().iter().sum::<f32>() <= self.limit {
                    x = run("double", vec![x])?.remove(0);
                    x.name = "x".into();
                    rounds += 1;
                }
                Ok(Message::tensors(vec![
                    x,
                    Tensor::f32("rounds", vec![1], &[rounds as f32]),
                ]))
            }
            "generate" => {
                let file = host("file", Message::body(json!({"name": self.notes})))?;
                let bytes = &file.tensor("bytes").ok_or("no bytes")?.data;
                let notes = String::from_utf8_lossy(bytes);
                let prompt = req.body["prompt"].as_str().unwrap_or("");
                Ok(Message::body(json!({"text": format!("{notes}|{prompt}")})))
            }
            "synthesize" => {
                let text = req.body["text"].as_str().unwrap_or("");
                let codes: Vec<f32> = text.chars().map(|c| c as u32 as f32).collect();
                let x = Tensor::f32("x", vec![codes.len()], &codes);
                let mut out = run("double", vec![x])?.remove(0);
                out.name = "audio".into();
                Ok(Message::new(json!({"sample_rate": 8000}), vec![out]))
            }
            "embed" => {
                let lens: Vec<f32> = req.body["texts"]
                    .as_array()
                    .ok_or("embed takes texts")?
                    .iter()
                    .map(|t| t.as_str().map_or(0, str::len) as f32)
                    .collect();
                let x = Tensor::f32("acc", vec![lens.len()], &lens);
                host("slot.set", Message::new(json!({"slot": "acc"}), vec![x]))?;
                for _ in 0..3 {
                    let back = run_with("double", vec![], &[("x", "acc")], &[("output_0", "acc")])?;
                    if !back.is_empty() {
                        return Err("a kept output came back".into());
                    }
                }
                let got = host("slot.get", Message::body(json!({"slot": "acc"})))?;
                let mut e = got.tensors.into_iter().next().ok_or("empty slot")?;
                e.name = "embeddings".into();
                e.shape = vec![lens.len(), 1];
                host("slot.drop", Message::body(json!({"slots": ["acc"]})))?;
                Ok(Message::tensors(vec![e]))
            }
            other => Err(format!("the toy does not {other}")),
        }
    }
}

rlx_program::program!(Toy);
