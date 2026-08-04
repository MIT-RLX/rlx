use rlx_tensor::rlx;

fn main() {
    let _g = rlx! {
        input h0: [1, 8];
        scan h = h0 for 4 { }
        out h;
    };
}
