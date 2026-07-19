//! transpose+reverse feeding a host-eval Op::Lstm on MLX vs CPU — isolates the
//! tusnet-phase discrepancy (rot90 before the recurrent cell).
#![cfg(all(feature = "cpu", feature = "mlx"))]
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session, is_available};
fn mk(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 997) as f32) / 498.0 - 1.0)
        .collect()
}
fn build(b: usize, s: usize, h: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("lstm_rev");
    let x = g.input("x", Shape::new(&[b, s, h], f)); // [b, s, h]
    // rot90-ish: transpose (swap s<->h then back) + reverse along the seq axis
    let xt = g.transpose_(x, vec![0, 2, 1]); // [b, h, s]
    let xr = g.add_node(
        Op::Reverse { axes: vec![2] },
        vec![xt],
        Shape::new(&[b, h, s], f),
    );
    let xb = g.transpose_(xr, vec![0, 2, 1]); // [b, s, h]
    let wih = g.input("w_ih", Shape::new(&[4 * h * h], f));
    let whh = g.input("w_hh", Shape::new(&[4 * h * h], f));
    let bias = g.input("bias", Shape::new(&[4 * h], f));
    let out = g.add_node(
        Op::Lstm {
            hidden_size: h,
            num_layers: 1,
            bidirectional: false,
            carry: false,
        },
        vec![xb, wih, whh, bias],
        Shape::new(&[b, s, h], f),
    );
    g.set_outputs(vec![out]);
    g
}
#[test]
fn lstm_rev_mlx_matches_cpu() {
    if !is_available(Device::Mlx) {
        eprintln!("skip: no MLX device");
        return;
    }
    let (b, s, h) = (2usize, 12usize, 8usize);
    let xd = mk(b * s * h, 1);
    let wihd = mk(4 * h * h, 2);
    let whhd = mk(4 * h * h, 3);
    let bd = mk(4 * h, 4);
    let slots: [(&str, &[f32]); 4] = [("x", &xd), ("w_ih", &wihd), ("w_hh", &whhd), ("bias", &bd)];
    let run = |dev| {
        let mut c = Session::new(dev).compile(build(b, s, h));
        c.run(&slots).pop().unwrap()
    };
    let cpu = run(Device::Cpu);
    let mlx = run(Device::Mlx);
    let maxd = cpu
        .iter()
        .zip(&mlx)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let dot: f32 = cpu.iter().zip(&mlx).map(|(a, b)| a * b).sum();
    let nc: f32 = cpu.iter().map(|a| a * a).sum::<f32>().sqrt();
    let nm: f32 = mlx.iter().map(|a| a * a).sum::<f32>().sqrt();
    println!(
        "transpose+reverse+Lstm CPU-vs-MLX  max|delta|={maxd:.3e}  cos={:.6}",
        dot / (nc * nm)
    );
    assert!(maxd < 1e-5, "must match CPU, got max|delta|={maxd}");
}
