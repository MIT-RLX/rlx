use rlx_tensor::rlx;

fn main() {
    let _g = rlx! {
        input x: [2, 4];
        param w: [4, 3];
        let y = x @ w;
        out z;   // `z` never declared
    };
}
