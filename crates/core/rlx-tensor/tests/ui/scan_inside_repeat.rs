use rlx_tensor::rlx;

fn main() {
    let _g = rlx! {
        input x: [4, 4];
        param w: [4, 4];
        repeat 2 {
            scan h = x for 3 { let h = h @ w; }
        }
        out x;
    };
}
