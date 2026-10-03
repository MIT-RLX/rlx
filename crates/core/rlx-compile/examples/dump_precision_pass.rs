use rlx_compile::PrecisionPolicy;
use rlx_compile::precision::AutoMixedPrecision;
use rlx_fusion::pass::Pass;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
fn main() {
    let (b, h, s, d) = (1usize, 2usize, 8usize, 16usize);
    let mut g = Graph::new("a");
    let dim = h * d;
    let q = g.input("q", Shape::new(&[b, s, dim], DType::F32));
    let k = g.input("k", Shape::new(&[b, s, dim], DType::F32));
    let v = g.input("v", Shape::new(&[b, s, dim], DType::F32));
    let shape = rlx_ir::shape::attention_shape(g.shape(q));
    let out = g.attention_kind(q, k, v, h, d, MaskKind::None, shape);
    g.set_outputs(vec![out]);
    let g2 = AutoMixedPrecision::new(PrecisionPolicy::AlwaysF16).run(g);
    for n in g2.nodes() {
        println!(
            "{:>3} {:<28} {:?} {:?}",
            n.id.0,
            format!("{:?}", n.op).chars().take(28).collect::<String>(),
            n.shape.dtype(),
            n.inputs.iter().map(|i| i.0).collect::<Vec<_>>()
        );
    }
    println!(
        "outputs: {:?}",
        g2.outputs
            .iter()
            .map(|o| (o.0, g2.shape(*o).dtype()))
            .collect::<Vec<_>>()
    );
}
