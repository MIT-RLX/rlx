use rlx_tensor::rlx;

fn main() {
    let _g = rlx! {
        input x: [2, 4];
        let y = x @ 2.0;   // matmul needs a tensor, not a scalar
        out y;
    };
}
