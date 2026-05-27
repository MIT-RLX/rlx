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

//! Lower JUST the embedding stage (gather + add + LN) and compare CPU vs MPSGraph.
//! If this matches but multi-layer BERT doesn't, the bug is in compounding
//! across transformer blocks (an Op outside attention/embeddings).
//!
//! cargo run --example embed_isolation --release \
//!   --features "cpu,metal,blas-accelerate" -p rlx-runtime

#[cfg(all(feature = "cpu", feature = "metal", target_os = "macos"))]
fn main() {
    use rlx_ir::*;
    use rlx_runtime::{Device, Session};

    let vocab = 32;
    let max_seq = 16;
    let tts = 2;
    let b = 4;
    let s = 8;
    let h = 384;

    let build = || {
        let mut g = Graph::new("embed");
        let ids = g.input("ids", Shape::new(&[b, s], DType::F32));
        let pos = g.input("pos", Shape::new(&[b, s], DType::F32));
        let tt = g.input("tt", Shape::new(&[b, s], DType::F32));
        let we = g.param("we", Shape::new(&[vocab, h], DType::F32));
        let pe = g.param("pe", Shape::new(&[max_seq, h], DType::F32));
        let te = g.param("te", Shape::new(&[tts, h], DType::F32));
        let lng = g.param("lng", Shape::new(&[h], DType::F32));
        let lnb = g.param("lnb", Shape::new(&[h], DType::F32));

        // 3-gather case but explicit: (we + pe) + te.
        let we_o = g.gather_(we, ids, 0);
        let pe_o = g.gather_(pe, pos, 0);
        let te_o = g.gather_(te, tt, 0);
        let s1 = g.add(we_o, pe_o);
        let s2 = g.add(s1, te_o);
        g.set_outputs(vec![s2]);
        let _ = (lng, lnb);
        g
    };

    let we_data: Vec<f32> = (0..vocab * h)
        .map(|i| ((i as f32) * 0.001).sin() * 0.1)
        .collect();
    let pe_data: Vec<f32> = (0..max_seq * h)
        .map(|i| ((i as f32) * 0.002).cos() * 0.1)
        .collect();
    let te_data: Vec<f32> = (0..tts * h)
        .map(|i| ((i as f32) * 0.003).sin() * 0.1)
        .collect();
    let ids_data: Vec<f32> = (0..b * s).map(|i| ((i * 7 + 13) % vocab) as f32).collect();
    let pos_data: Vec<f32> = (0..b).flat_map(|_| (0..s).map(|j| j as f32)).collect();
    let tt_data: Vec<f32> = (0..b * s).map(|i| (i % 2) as f32).collect();
    let lng_data = vec![1.0f32; h];
    let lnb_data = vec![0.0f32; h];

    let run_with = |use_mpsg: bool, dev: Device| -> Vec<f32> {
        if use_mpsg {
            rlx_ir::env::set("RLX_USE_MPSGRAPH", "1");
        } else {
            unsafe {
                rlx_ir::env::unset("RLX_USE_MPSGRAPH");
            }
        }
        let session = Session::new(dev);
        let mut compiled = session.compile(build());
        compiled.set_param("we", &we_data);
        compiled.set_param("pe", &pe_data);
        compiled.set_param("te", &te_data);
        compiled.set_param("lng", &lng_data);
        compiled.set_param("lnb", &lnb_data);
        let outs = compiled.run(&[("ids", &ids_data), ("pos", &pos_data), ("tt", &tt_data)]);
        outs.into_iter().next().unwrap_or_default()
    };

    let cpu = run_with(false, Device::Cpu);
    let metal_thunk = run_with(false, Device::Metal);
    let metal_mpsg = run_with(true, Device::Metal);

    println!("CPU thunk[..6]:    {:?}", &cpu[..6]);
    println!("Metal thunk[..6]:  {:?}", &metal_thunk[..6]);
    println!("Metal MPSG[..6]:   {:?}", &metal_mpsg[..6]);

    let diff = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    println!(
        "\nmax_err CPU vs Metal-thunk: {:.3e}",
        diff(&cpu, &metal_thunk)
    );
    println!(
        "max_err CPU vs Metal-MPSG:  {:.3e}",
        diff(&cpu, &metal_mpsg)
    );
}

#[cfg(not(all(feature = "cpu", feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("requires cpu + metal on macOS");
}
