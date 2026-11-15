// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Batched matmul with an operand broadcast along only some batch axes
//! (`[2,1,M,K]·[1,3,K,N]`, `[2,3,M,K]·[3,K,N]`), wgpu vs CPU. The batched
//! kernel takes each operand as the full batch or one repeated matrix, and the
//! lowering used to panic on anything else; `unfuse` now expands such an
//! operand to the output's batch first.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

#[test]
fn partial_batch_broadcast_matches_cpu() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return;
    }
    let (m, k, n) = (5, 7, 4);
    for (sa, sb) in [
        (vec![2, 1, m, k], vec![1, 3, k, n]),
        (vec![2, 3, m, k], vec![1, 3, k, n]),
        (vec![2, 1, m, k], vec![2, 3, k, n]),
        (vec![2, 3, m, k], vec![3, k, n]),
        (vec![1, 8, m, k], vec![8, k, n]),
        (vec![2, 3, m, k], vec![2, 3, k, n]),
    ] {
        let a: Vec<f32> = (0..sa.iter().product::<usize>())
            .map(|i| ((i as f32) * 0.37).sin())
            .collect();
        let b: Vec<f32> = (0..sb.iter().product::<usize>())
            .map(|i| ((i as f32) * 0.23).cos())
            .collect();
        let build = || {
            let mut g = Graph::new("bcast_mm");
            let x = g.input("a", Shape::new(&sa, DType::F32));
            let y = g.input("b", Shape::new(&sb, DType::F32));
            let z = g.mm(x, y);
            g.set_outputs(vec![z]);
            g
        };
        let ins: [(&str, &[f32]); 2] = [("a", &a), ("b", &b)];
        let cpu = Session::new(Device::Cpu).compile(build()).run(&ins);
        let gpu = Session::new(Device::Gpu).compile(build()).run(&ins);
        let d = cpu[0]
            .iter()
            .zip(&gpu[0])
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        eprintln!("{sa:?} x {sb:?}: wgpu vs cpu {d:.2e}");
        assert!(d < 1e-4, "{sa:?} x {sb:?}: wgpu differs from CPU by {d}");
    }
}
