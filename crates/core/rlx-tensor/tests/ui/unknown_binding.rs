use rlx_tensor::rlx;

fn main() {
    let _g = rlx! {
        input x: [2, 4];
        let y = x @ w;   // `w` never declared
        out y;
    };
}
