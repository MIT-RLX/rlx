//! A half-typed **activation** is stored widened to f32; a half **param**
//! stays packed at 2 B. Ops must route on the slot's real width, not on the
//! declared dtype.
//!
//! `plan_memory_native_in_order` makes that split because the thunks compute
//! in f32. Two places read the dtype instead and broke under bf16 autocast,
//! where a cast activation — not a weight — is routinely a matmul operand:
//!
//! * `Op::MatMul` with a BF16/F16 right-hand took the packed
//!   dequant-on-the-fly GEMM and read f32 bytes as `u16`, panicking with
//!   "range end index N out of range for slice of length 0";
//! * `Op::Cast` to BF16/F16 wrote packed halves into a widened slot, which
//!   reads back as pairs of halves — the corruption the backend's own
//!   `cpu_backend.rs` comment illustrates.
//!
//! Both produced NaN rather than an error, which is why they are pinned here.

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

/// `y = cast_bf16(x) · cast_bf16(w_f32)` — both matmul operands are cast
/// **activations**, the shape autocast produces.
#[test]
fn matmul_on_bf16_activations_is_finite_and_correct() {
    let f = DType::F32;
    let (m, k, n) = (2usize, 3, 4);
    let mut g = Graph::new("t");
    let x = g.input("x", Shape::new(&[m, k], f));
    let w = g.input("w", Shape::new(&[k, n], f));
    let xb = g.cast(x, DType::BF16);
    let wb = g.cast(w, DType::BF16);
    let y = g.matmul(xb, wb, Shape::new(&[m, n], DType::BF16));
    let yf = g.cast(y, f);
    g.set_outputs(vec![yf]);

    let xv: Vec<f32> = (0..m * k).map(|i| 0.5 + i as f32 * 0.25).collect();
    let wv: Vec<f32> = (0..k * n).map(|i| 1.0 - i as f32 * 0.125).collect();
    let mut c = Session::new(Device::Cpu).compile(g);
    let out = c.run(&[("x", &xv[..]), ("w", &wv[..])]);

    assert_eq!(out[0].len(), m * n);
    assert!(
        out[0].iter().all(|v| v.is_finite()),
        "bf16-activation matmul produced non-finite values: {:?}",
        out[0]
    );
    // Reference in f32; bf16 rounding of these exact values is lossless
    // (all are representable), so this is an equality check in disguise.
    for i in 0..m {
        for j in 0..n {
            let want: f32 = (0..k).map(|p| xv[i * k + p] * wv[p * n + j]).sum();
            let got = out[0][i * n + j];
            assert!(
                (got - want).abs() <= 1e-2 * want.abs().max(1.0),
                "({i},{j}): got {got}, want {want}"
            );
        }
    }
}

/// A round trip through a BF16 activation must round the value, not reinterpret
/// the bytes. Reading a widened slot as packed halves gives the documented
/// `[1,2,3,4] -> [2.0038757, 4.007843, 0.0, 0.0]` corruption.
#[test]
fn cast_round_trip_through_a_bf16_activation_rounds_rather_than_repacks() {
    let f = DType::F32;
    let mut g = Graph::new("t");
    let x = g.input("x", Shape::new(&[4], f));
    let b = g.cast(x, DType::BF16);
    let y = g.cast(b, f);
    g.set_outputs(vec![y]);

    let xv = [1.0f32, 2.0, 3.0, 4.0];
    let mut c = Session::new(Device::Cpu).compile(g);
    let out = c.run(&[("x", &xv[..])]);
    assert_eq!(
        out[0], xv,
        "these values are exactly representable in bf16, so the round trip \
         must be the identity; got {:?}",
        out[0]
    );

    // …and a value that is NOT representable must come back rounded, which
    // proves the cast ran rather than being skipped. A bare
    // `Cast(BF16) -> Cast(F32)` pair is elided by the compiler as an identity
    // round trip (it is not one — bf16 loses mantissa), so put an op in
    // between that the elision cannot see through.
    let mut g2 = Graph::new("t2");
    let x2 = g2.input("x", Shape::new(&[1], f));
    let b2 = g2.cast(x2, DType::BF16);
    let z = g2.input("z", Shape::new(&[1], DType::BF16));
    let sum = g2.add(b2, z);
    let y2 = g2.cast(sum, f);
    g2.set_outputs(vec![y2]);
    let mut c2 = Session::new(Device::Cpu).compile(g2);
    let v = [1.234_567_9f32];
    let zero = [0.0f32];
    let got = c2.run(&[("x", &v[..]), ("z", &zero[..])])[0][0];
    let want = half::bf16::from_f32(v[0]).to_f32();
    assert!(
        got.is_finite(),
        "adding through a bf16 activation produced {got}"
    );
    assert!(
        (got - want).abs() <= 1e-3,
        "expected the bf16-rounded value {want} (or close), got {got}"
    );
    // Guard against the old corruption specifically: repacking would put this
    // nowhere near the input.
    assert!(
        (got - v[0]).abs() < 0.01,
        "value was corrupted, not rounded: {got} vs {}",
        v[0]
    );
}

/// A BF16 **param** must still take the packed dequant-on-the-fly path — the
/// fix must not regress bf16-resident weights (the bf16 LM head).
#[test]
fn bf16_param_weights_still_use_the_packed_path() {
    let f = DType::F32;
    let (m, k, n) = (1usize, 4, 4);
    let mut g = Graph::new("t");
    let x = g.input("x", Shape::new(&[m, k], f));
    let w = g.param("w", Shape::new(&[k, n], DType::BF16));
    let y = g.matmul(x, w, Shape::new(&[m, n], f));
    g.set_outputs(vec![y]);

    let xv = [1.0f32, 2.0, 3.0, 4.0];
    let wv: Vec<f32> = (0..k * n).map(|i| 0.25 * (i % 5) as f32).collect();
    let mut c = Session::new(Device::Cpu).compile(g);
    c.set_param("w", &wv);
    let out = c.run(&[("x", &xv[..])]);
    assert!(out[0].iter().all(|v| v.is_finite()), "{:?}", out[0]);
    for j in 0..n {
        let want: f32 = (0..k).map(|p| xv[p] * wv[p * n + j]).sum();
        assert!(
            (out[0][j] - want).abs() <= 1e-2 * want.abs().max(1.0),
            "col {j}: got {}, want {want}",
            out[0][j]
        );
    }
}
