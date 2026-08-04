use rlx_tensor::rlx;

fn main() {
    let _g = rlx! {
        input x: [4, 4];
        repeat 2 {
            param w: [4, 4];   // would collide each iteration
            let x = x @ w;
        }
        out x;
    };
}
