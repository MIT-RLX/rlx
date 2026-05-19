// Tiny [1,2,2,2] @ [1,2,2,2] batched matmul, verify per-head correctness.

use anyhow::Result;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn main() -> Result<()> {
    let (b, h, m, k, n) = (1, 2, 2, 2, 2);
    // Head 0: [1,2;3,4] @ [10,20;30,40] = [70,100; 150,220]
    // Head 1: [5,6;7,8] @ [50,60;70,80] = [670,760; 910,1040]
    let q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let kt = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

    let mut g = Graph::new("bmm");
    let f = DType::F32;
    let q_in = g.input("q", Shape::new(&[b, h, m, k], f));
    let k_in = g.input("k", Shape::new(&[b, h, k, n], f));
    let out = g.matmul(q_in, k_in, Shape::new(&[b, h, m, n], f));
    g.set_outputs(vec![out]);
    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(g);
    let outputs = compiled.run(&[("q", q.as_slice()), ("k", kt.as_slice())]);
    let ir_out = outputs.into_iter().next().unwrap();
    println!("IR  : {ir_out:?}");
    println!("WANT: [70, 100, 150, 220, 670, 760, 910, 1040]");
    Ok(())
}
