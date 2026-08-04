use rlx_tensor::rlx;

fn main() {
    let _g = rlx! {
        fn block(x, w) { let h = x @ w; }
        input a: [2, 4];
        let o = block(a);   // block takes 2 args
        out o;
    };
}
