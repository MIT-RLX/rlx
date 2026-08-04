use rlx_tensor::rlx;

fn main() {
    let _g = rlx! {
        input x: [4, 4];
        repeat 2 {
            bind w;            // `bind` adopts a name — not allowed inside a loop
            let x = x @ w;
        }
        out x;
    };
}
