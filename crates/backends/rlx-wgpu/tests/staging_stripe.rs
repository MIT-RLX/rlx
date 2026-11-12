// RLX — versatile ML compiler + runtime.
//
// The staging stripe (`RLX_WGPU_STRADDLE_STAGE`): an op whose operands lie in
// two arena stripes runs on copies of them in one extra stripe, its output
// copied back. Here an activation, a residual add and a transposed 3-D conv
// (no two-window kernel) over a parked weight run staged and must match an
// unstriped run. (The stage reserve of the staging stripe, which lies past
// the arena's size, was once clamped to its first byte: NV-Segment-CTMR's
// 192 x 192 x 128 window gave 5297 wrong labels; this graph does not stage
// into that reserve, so it guards the staging path, not that bug.)

use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_wgpu::backend::WgpuExecutable;

fn ramp(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + seed * 13) % 23) as f32 - 11.0) / 17.0)
        .collect()
}

#[test]
fn staged_conv_transpose_matches_unstriped() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        return;
    }
    let sh = |d: &[usize]| Shape::new(d, DType::F32);
    let (cin, cout, d) = (8, 4, [16, 32, 32]);
    let vol = sh(&[1, cin, d[0], d[1], d[2]]);
    let up = sh(&[1, cout, 2 * d[0], 2 * d[1], 2 * d[2]]);
    let build = || {
        let mut g = Graph::new("staging_stripe");
        let x = g.input("x", vol.clone());
        let w = g.param("w", sh(&[cin, cout, 2, 2, 2]));
        let a = g.activation(rlx_ir::op::Activation::Gelu, x, vol.clone());
        let b = g.activation(rlx_ir::op::Activation::Gelu, a, vol.clone());
        let c = g.add(a, b);
        let y = g.add_node(
            Op::ConvTranspose3d {
                stride: [2; 3],
                padding: [0; 3],
                dilation: [1; 3],
                output_padding: [0; 3],
                groups: 1,
            },
            vec![c, w],
            up.clone(),
        );
        let z = g.activation(rlx_ir::op::Activation::Gelu, y, up.clone());
        g.set_outputs(vec![z, x]);
        g
    };
    let x = ramp(vol.num_elements().unwrap(), 1);
    let w = ramp(cin * cout * 8, 2);
    let ins: [(&str, &[f32]); 2] = [("x", &x), ("w", &w)];
    let whole = WgpuExecutable::compile(build()).run(&ins);

    rlx_ir::env::set("RLX_WGPU_SHARD_CAP_MIB", "4");
    rlx_ir::env::set("RLX_WGPU_SHARD_STAGE_MIB", "1");
    rlx_ir::env::set("RLX_WGPU_ALLOW_SHARD", "1");
    rlx_ir::env::set("RLX_WGPU_SHARD_GPU", "1");
    rlx_ir::env::set("RLX_WGPU_STRADDLE_STAGE", "1");
    // Parked weights, as a model too large for one binding has them: staged
    // beside the operands, in the stage reserve.
    rlx_ir::env::set("RLX_WGPU_ALL_PARAMS_WEIGHT", "1");
    let exe = WgpuExecutable::compile(build());
    let staged = !exe.blocks_on_host();
    drop(exe);
    let striped = WgpuExecutable::compile(build()).run(&ins);
    let worst = striped[0]
        .iter()
        .zip(&whole[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("staged (no host steps) {staged}; max |Δ| {worst:e}");
    assert!(worst <= 1e-5, "striped conv transpose differs by {worst}");
}
