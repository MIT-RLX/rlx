//! MLX vs CPU depthwise 1-D ConvTranspose (StyleTTS2 F0/N pool pattern).
#![cfg(all(feature = "cpu", feature = "mlx"))]
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

fn mk(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i.wrapping_mul(2654435761).wrapping_add(seed)) % 1000) as f32) / 500.0 - 1.0)
        .collect()
}

#[test]
fn mlx_depthwise_convtranspose_stride2() {
    if !is_available(Device::Mlx) {
        eprintln!("skip: no MLX device");
        return;
    }
    // ONNX: weight [Cin, Cout/g, k] = [512, 1, 3], groups=512, stride=2,
    // pads=[1,1], output_padding=[1] → L_out = 2*L_in.
    let (b, c, l, k) = (1usize, 512usize, 32usize, 3usize);
    let l_out = 2 * l;
    let x = mk(b * c * l, 1);
    let w = mk(c * k, 2); // [Cin, 1, k]
    let mut g = Graph::new("dw_ct");
    let xi = g.input("x", Shape::new(&[b, c, 1, l], DType::F32));
    let wi = g.input("w", Shape::new(&[c, 1, 1, k], DType::F32));
    let y = g.add_node(
        Op::ConvTranspose2d {
            kernel_size: vec![1, k],
            stride: vec![1, 2],
            padding: vec![0, 1],
            dilation: vec![1, 1],
            output_padding: vec![0, 1],
            groups: c,
        },
        vec![xi, wi],
        Shape::new(&[b, c, 1, l_out], DType::F32),
    );
    g.set_outputs(vec![y]);
    let slots = [("x", x.as_slice()), ("w", w.as_slice())];
    let run = |dev| {
        Session::new(dev)
            .compile(g.clone())
            .run(&slots)
            .pop()
            .unwrap()
    };
    let cpu = run(Device::Cpu);
    let mlx = run(Device::Mlx);
    assert_eq!(cpu.len(), b * c * l_out);
    assert_eq!(mlx.len(), b * c * l_out);
    let maxd = cpu
        .iter()
        .zip(&mlx)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("depthwise CT g={c} maxd={maxd:.3e}");
    assert!(maxd < 1e-4, "MLX depthwise ConvTranspose mismatch {maxd}");
}
