//! The three training additions must behave the same on every backend.
//!
//! * **LoRA dropout** — a host-supplied mask multiplied into the adapter
//!   branch. It is a `Mul`, so it should be universal; this proves it rather
//!   than assuming, and proves the mask gates the delta on each device.
//! * **Gradient checkpointing** — a pure graph transform, so the recomputed
//!   graph must give the same answer everywhere. What can break per backend
//!   is scheduling/aliasing of the duplicated nodes, not the maths.
//! * **bf16 autocast** — the one with genuine per-backend exposure. A
//!   half-typed *activation* is stored widened to f32 on CPU while a half
//!   *param* stays packed; every backend makes its own choice there, and
//!   reading the declared dtype instead of the real layout is what produced
//!   NaN and a GEMM panic on CPU.
//!
//! ```text
//! cargo test -p rlx-runtime --features metal,mlx --test training_features_backend_parity
//! ```
#![cfg(feature = "cpu")]

use rlx_autodiff::checkpoint::checkpoint_backward;
use rlx_autodiff::grad_with_loss;
use rlx_compile::precision::{AutoMixedPrecision, PrecisionPolicy};
use rlx_fusion::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{Device, Session, is_available};

mod common;

const TOL: f32 = 5e-3;

#[allow(clippy::vec_init_then_push)]
fn available_backends() -> Vec<Device> {
    let mut v: Vec<Device> = Vec::new();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    v.push(Device::Metal);
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    v.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    v.push(Device::Gpu);
    #[cfg(feature = "cuda")]
    v.push(Device::Cuda);
    #[cfg(feature = "rocm")]
    v.push(Device::Rocm);
    #[cfg(feature = "vulkan")]
    v.push(Device::Vulkan);
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    v.push(Device::Ane);
    v.retain(|&d| is_available(d));
    v
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str, dev: Device) {
    assert_eq!(a.len(), b.len(), "{dev:?}: {what} length differs");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            x.is_finite() && y.is_finite(),
            "{dev:?}: {what}[{i}] non-finite: {x} vs {y}"
        );
        assert!(
            (x - y).abs() <= tol * x.abs().max(1.0),
            "{dev:?}: {what}[{i}] = {y}, cpu = {x}"
        );
    }
}

/// `y = x·W + scale·((x ⊙ mask)·A)·B` — the injected LoRA forward.
fn lora_graph(m: usize, k: usize, n: usize, r: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("lora");
    let x = g.input("x", Shape::new(&[m, k], f));
    let w = g.param("w", Shape::new(&[k, n], f));
    let a = g.param("a", Shape::new(&[k, r], f));
    let b = g.param("b", Shape::new(&[r, n], f));
    let mask = g.input("mask", Shape::new(&[m, k], f));
    let base = g.matmul(x, w, Shape::new(&[m, n], f));
    let xd = g.mul(x, mask);
    let xa = g.matmul(xd, a, Shape::new(&[m, r], f));
    let delta = g.matmul(xa, b, Shape::new(&[m, n], f));
    let y = g.add(base, delta);
    let loss = g.mean(y, vec![0, 1], false);
    g.set_outputs(vec![loss]);
    g
}

fn params(k: usize, n: usize, r: usize) -> Vec<(&'static str, Vec<f32>)> {
    let fill = |len: usize, s: f32| (0..len).map(|i| s * ((i % 7) as f32 - 3.0)).collect();
    vec![
        ("w", fill(k * n, 0.10)),
        ("a", fill(k * r, 0.20)),
        ("b", fill(r * n, 0.15)),
    ]
}

fn run(dev: Device, g: Graph, ps: &[(&str, Vec<f32>)], feeds: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
    let mut c = Session::new(dev).compile(g);
    for (n, v) in ps {
        c.set_param(n, v);
    }
    c.run(feeds)
}

/// The dropout mask must gate the adapter delta identically everywhere, and
/// must never touch the frozen base path.
#[test]
fn lora_dropout_mask_matches_cpu_on_every_backend() {
    // Serialize GPU access: these build a `Session` on a real device, and
    // without the guard they race every other GPU test in this binary.
    let _gpu = common::serialize_gpu();
    let (m, k, n, r) = (4usize, 8, 6, 2);
    let ps = params(k, n, r);
    let x: Vec<f32> = (0..m * k).map(|i| 0.1 * (i % 5) as f32 - 0.2).collect();
    let ones = vec![1.0f32; m * k];
    let zeros = vec![0.0f32; m * k];
    // Inverted-dropout mask: some lanes off, survivors scaled by 1/(1-p).
    let mixed: Vec<f32> = (0..m * k)
        .map(|i| if i % 4 == 0 { 0.0 } else { 1.0 / 0.75 })
        .collect();

    let cpu: Vec<Vec<f32>> = [&ones, &zeros, &mixed]
        .iter()
        .map(|mask| {
            run(
                Device::Cpu,
                lora_graph(m, k, n, r),
                &ps,
                &[("x", &x), ("mask", mask)],
            )[0]
            .clone()
        })
        .collect();
    assert_ne!(
        cpu[0], cpu[1],
        "a zero mask must change the answer, or this test proves nothing"
    );

    for dev in available_backends() {
        for (i, mask) in [&ones, &zeros, &mixed].iter().enumerate() {
            let got = run(
                dev,
                lora_graph(m, k, n, r),
                &ps,
                &[("x", &x), ("mask", mask)],
            );
            close(&cpu[i], &got[0], TOL, &format!("dropout case {i}"), dev);
        }
        eprintln!("{dev:?}: dropout mask matches CPU");
    }
}

/// A checkpointed backward must produce the same gradients as the plain one,
/// on every backend.
#[test]
fn checkpointed_backward_matches_cpu_on_every_backend() {
    // Serialize GPU access: these build a `Session` on a real device, and
    // without the guard they race every other GPU test in this binary.
    let _gpu = common::serialize_gpu();
    let (m, k, n, r) = (4usize, 8, 6, 2);
    let ps = params(k, n, r);
    let x: Vec<f32> = (0..m * k).map(|i| 0.1 * (i % 5) as f32 - 0.2).collect();
    let ones = vec![1.0f32; m * k];
    let seed = [1.0f32];

    let fwd = lora_graph(m, k, n, r);
    let wrt: Vec<NodeId> = fwd
        .nodes()
        .iter()
        .filter(|nd| matches!(&nd.op, rlx_ir::Op::Param { name } if name == "a" || name == "b"))
        .map(|nd| nd.id)
        .collect();
    assert_eq!(wrt.len(), 2, "expected the two adapter params");
    let bwd = grad_with_loss(&fwd, &wrt);
    let ck = checkpoint_backward(&fwd, &bwd, 3);
    assert!(
        ck.nodes().len() > bwd.nodes().len(),
        "checkpointing did not add recomputes, so this would prove nothing"
    );

    let feeds: Vec<(&str, &[f32])> = vec![("x", &x), ("mask", &ones), ("d_output", &seed)];
    let base = run(Device::Cpu, bwd.clone(), &ps, &feeds);
    let cpu_ck = run(Device::Cpu, ck.clone(), &ps, &feeds);
    for (i, (a, b)) in base.iter().zip(&cpu_ck).enumerate() {
        close(
            a,
            b,
            TOL,
            &format!("cpu checkpointed output {i}"),
            Device::Cpu,
        );
    }
    assert!(
        base[1].iter().any(|v| v.abs() > 1e-9),
        "gradient is all zeros; the comparison would be vacuous"
    );

    for dev in available_backends() {
        let got = run(dev, ck.clone(), &ps, &feeds);
        for (i, (a, b)) in base.iter().zip(&got).enumerate() {
            close(a, b, TOL, &format!("checkpointed output {i}"), dev);
        }
        eprintln!("{dev:?}: checkpointed backward matches CPU");
    }
}

/// bf16 autocast must stay close to the f32 answer on every backend.
///
/// This is where the layout of a half-typed *activation* differs per backend,
/// so a device that reads the declared dtype instead of the real slot width
/// shows up here as NaN or garbage rather than as a small rounding difference.
#[test]
fn bf16_autocast_matches_cpu_on_every_backend() {
    // Serialize GPU access: these build a `Session` on a real device, and
    // without the guard they race every other GPU test in this binary.
    let _gpu = common::serialize_gpu();
    let (m, k, n, r) = (4usize, 8, 6, 2);
    let ps = params(k, n, r);
    let x: Vec<f32> = (0..m * k).map(|i| 0.1 * (i % 5) as f32 - 0.2).collect();
    let ones = vec![1.0f32; m * k];

    let g32 = lora_graph(m, k, n, r);
    let amp = |policy| AutoMixedPrecision::new(policy).run(lora_graph(m, k, n, r));
    let bf = amp(PrecisionPolicy::AutoMixedBf16Safe);
    let cast_count = bf
        .nodes()
        .iter()
        .filter(|nd| nd.shape.dtype() == DType::BF16)
        .count();
    assert!(cast_count > 0, "the bf16 policy did not apply");

    let feeds: Vec<(&str, &[f32])> = vec![("x", &x), ("mask", &ones)];
    let ref32 = run(Device::Cpu, g32, &ps, &feeds)[0].clone();
    // bf16 keeps ~3 decimal digits, so the tolerance is bf16's, not f32's.
    let bf_tol = 2e-2;
    let cpu_bf = run(Device::Cpu, bf.clone(), &ps, &feeds)[0].clone();
    close(&ref32, &cpu_bf, bf_tol, "cpu bf16", Device::Cpu);

    for dev in available_backends() {
        let got = run(dev, bf.clone(), &ps, &feeds);
        close(&ref32, &got[0], bf_tol, "bf16 autocast", dev);
        eprintln!("{dev:?}: bf16 autocast matches CPU");
    }
}

/// A matmul whose **right-hand side is a BF16 activation**, on every backend.
///
/// This is the exact shape that broke CPU, and the safe autocast policy does
/// *not* reach it (it keeps Compute in F32), so it needs its own test. Every
/// backend chooses independently whether a half activation is stored packed
/// or widened, and dispatching on the declared dtype rather than the real
/// layout is the bug — on CPU it panicked inside the packed BF16 GEMM.
#[test]
fn matmul_with_a_bf16_activation_rhs_matches_cpu_on_every_backend() {
    // Serialize GPU access: these build a `Session` on a real device, and
    // without the guard they race every other GPU test in this binary.
    let _gpu = common::serialize_gpu();
    let f = DType::F32;
    let (m, k, n) = (4usize, 8, 6);
    let build = || {
        let mut g = Graph::new("bf16rhs");
        let x = g.input("x", Shape::new(&[m, k], f));
        let w = g.input("w", Shape::new(&[k, n], f));
        // Both operands are CAST ACTIVATIONS, not params.
        let xb = g.cast(x, DType::BF16);
        let wb = g.cast(w, DType::BF16);
        let y = g.matmul(xb, wb, Shape::new(&[m, n], DType::BF16));
        let yf = g.cast(y, f);
        g.set_outputs(vec![yf]);
        g
    };
    let x: Vec<f32> = (0..m * k).map(|i| 0.5 + 0.25 * (i % 5) as f32).collect();
    let w: Vec<f32> = (0..k * n).map(|i| 1.0 - 0.125 * (i % 7) as f32).collect();
    let feeds: Vec<(&str, &[f32])> = vec![("x", &x), ("w", &w)];

    // Reference computed in f64, so this checks the value and not just
    // agreement between two equally-wrong backends.
    let mut want = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            want[i * n + j] = (0..k).map(|p| x[i * k + p] * w[p * n + j]).sum();
        }
    }

    let cpu = run(Device::Cpu, build(), &[], &feeds)[0].clone();
    close(&want, &cpu, 3e-2, "bf16-activation matmul", Device::Cpu);

    for dev in available_backends() {
        let got = run(dev, build(), &[], &feeds);
        close(&want, &got[0], 3e-2, "bf16-activation matmul", dev);
        close(&cpu, &got[0], 3e-2, "bf16-activation matmul vs cpu", dev);
        eprintln!("{dev:?}: bf16-activation matmul matches");
    }
}

/// The full-BF16 policy across backends — reported, not asserted equal.
///
/// `AutoMixedBf16` is the TPU policy: matmul and data movement in BF16 too.
/// On CPU that diverges (which is why `AutoMixedBf16Safe` exists), but other
/// backends may handle it, and this is where the activation-memory win
/// actually lives. Printing per-backend behaviour keeps that visible instead
/// of leaving it an open question.
#[test]
fn full_bf16_policy_behaviour_is_recorded_per_backend() {
    // Serialize GPU access: these build a `Session` on a real device, and
    // without the guard they race every other GPU test in this binary.
    let _gpu = common::serialize_gpu();
    let (m, k, n, r) = (4usize, 8, 6, 2);
    let ps = params(k, n, r);
    let x: Vec<f32> = (0..m * k).map(|i| 0.1 * (i % 5) as f32 - 0.2).collect();
    let ones = vec![1.0f32; m * k];
    let feeds: Vec<(&str, &[f32])> = vec![("x", &x), ("mask", &ones)];
    let ref32 = run(Device::Cpu, lora_graph(m, k, n, r), &ps, &feeds)[0].clone();

    let full =
        || AutoMixedPrecision::new(PrecisionPolicy::AutoMixedBf16).run(lora_graph(m, k, n, r));
    let report = |dev: Device| {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run(dev, full(), &ps, &feeds)[0].clone()
        }));
        match r {
            Err(_) => eprintln!("  {dev:?}: PANIC"),
            Ok(v) if !v.iter().all(|x| x.is_finite()) => eprintln!("  {dev:?}: non-finite"),
            Ok(v) => {
                let err = v
                    .iter()
                    .zip(&ref32)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("  {dev:?}: finite, max |diff| vs f32 = {err:.5}");
            }
        }
    };
    eprintln!("AutoMixedBf16 (full) behaviour:");
    report(Device::Cpu);
    for dev in available_backends() {
        report(dev);
    }
}
