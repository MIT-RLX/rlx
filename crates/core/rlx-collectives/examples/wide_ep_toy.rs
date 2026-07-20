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

//! WideEP Phase 0/1 toy: 2 experts × 2 ranks, variable-size dispatch/combine.
//!
//! Each rank holds one expert shard and the same microbatch. Routing may
//! send tokens across ranks; outputs must match a single-device dense MoE
//! reference (`moe_demo` logic).
//!
//! ```text
//! cargo run -p rlx-collectives --example wide_ep_toy
//! ```

use rlx_collectives::{MoeEpConfig, moe_ep_ffn, register, register_group, unregister_group};
use rlx_driver::{NetTransport, ProcessGroup};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::sync::Arc;
use std::thread;

fn det(seed: usize, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) * 7 + 11) % 17) as f32 / 17.0 * scale - scale * 0.5)
        .collect()
}

fn dense_moe_reference(x: &[f32], gate_w: &[f32], expert_w: &[f32], m: usize, h: usize, e: usize) -> Vec<f32> {
    let mut reference = vec![0f32; m * h];
    for i in 0..m {
        let mut logits = vec![0f32; e];
        for ei in 0..e {
            for k in 0..h {
                logits[ei] += x[i * h + k] * gate_w[k * e + ei];
            }
        }
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&v| v / sum).collect();
        let mut best_e = 0usize;
        let mut best_p = probs[0];
        for ei in 1..e {
            if probs[ei] > best_p {
                best_p = probs[ei];
                best_e = ei;
            }
        }
        for j in 0..h {
            let mut acc = 0f32;
            for k in 0..h {
                acc += x[i * h + k] * expert_w[(best_e * h + k) * h + j];
            }
            reference[i * h + j] = acc * best_p;
        }
    }
    reference
}

fn main() {
    register();
    let world = 2u32;
    let m = 4usize;
    let h = 8usize;
    let e = 2usize;

    let x_data = det(0, m * h, 0.5);
    let gate_w = det(1, h * e, 0.3);
    let expert_w = det(2, e * h * h, 0.2);
    let reference = dense_moe_reference(&x_data, &gate_w, &expert_w, m, h, e);

    let listeners: Vec<TcpListener> = (0..world)
        .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
        .collect();
    let addrs: Vec<SocketAddr> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    let gid_base = 9300u64;

    let handles: Vec<_> = listeners
        .into_iter()
        .enumerate()
        .map(|(rank, listener)| {
            let addrs = addrs.clone();
            let x_data = x_data.clone();
            let gate_w = gate_w.clone();
            let expert_w = expert_w.clone();
            thread::spawn(move || {
                let rank = rank as u32;
                let t = NetTransport::from_listener(rank, world, listener, addrs, 1 << 20).unwrap();
                let gid = gid_base + rank as u64;
                register_group(gid, Arc::new(ProcessGroup::new(Arc::new(t))));

                let cfg = MoeEpConfig::new(gid, world, rank, e as u32, h as u32, m as u32)
                    .with_placement(vec![0, 1]);
                let mut w_local = vec![0f32; h * h];
                let e_off = rank as usize * h * h;
                w_local.copy_from_slice(&expert_w[e_off..e_off + h * h]);

                let mut g = Graph::new("wide_ep_toy");
                let x = g.input("x", Shape::new(&[m, h], DType::F32));
                let gw = g.param("gate_w", Shape::new(&[h, e], DType::F32));
                let ew = g.param("expert_w_local", Shape::new(&[1, h, h], DType::F32));
                let out = moe_ep_ffn(&mut g, x, gw, ew, &cfg);
                g.set_outputs(vec![out]);

                let mut c = Session::new(Device::Cpu).compile(g);
                c.set_param("gate_w", &gate_w);
                c.set_param("expert_w_local", &w_local);
                let res = c.run(&[("x", &x_data)]);
                unregister_group(gid);
                (rank, res)
            })
        })
        .collect();

    let mut max_err = 0f32;
    for h in handles {
        let (rank, outs) = h.join().unwrap();
        let err = outs[0]
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("rank {rank}: max_err vs dense MoE = {err:.3e}");
        max_err = max_err.max(err);
    }

    if max_err < 1e-5 {
        println!("PASS — WideEP Phase 1 (2 experts × 2 ranks, all_to_all_v) matches dense reference.");
    } else {
        eprintln!("FAIL — max_err={max_err:.3e}");
        std::process::exit(1);
    }
}
