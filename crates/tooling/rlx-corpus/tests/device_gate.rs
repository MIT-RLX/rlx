// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Which FAMILIES did this change break, on which device.**
//!
//! `oracle_gate.rs` runs the corpus on CPU and scores it against the
//! independent f64 evaluator. This is the same loop over **every device the
//! host can instantiate**, and it exists because the device-free gate stack the
//! crate is built around cannot see the defects that actually shipped here.
//!
//! Look at the release this crate landed in: a batch-broadcast attention mask
//! read out of bounds, `KvAppend` used the wrong row stride, wide-hidden LSTM
//! was wrong on Apple GPUs, an op whose operands straddled two arena stripes
//! read zeros. Not one is visible to `verify` → `repr_check` → `plan_check`.
//! They are numerical, they are per-backend, and they need a device.
//!
//! Two things make this different from the ~399 GPU parity tests already in
//! tree, which is the whole reason to add it rather than point at them:
//!
//! * **The authority is not another backend.** Those tests score a GPU against
//!   rlx's own CPU path, which cannot catch a defect both share —
//!   `rms_norm_backward` carried an extra `1/r` in all seven implementations at
//!   once. Here every case is scored against [`rlx_corpus::oracle`]'s f64
//!   evaluator, exactly as the CPU arm is.
//! * **The rollup is by family, not by crate.** "which families did I break" is
//!   the question a compiler change asks, and a suite organized by owning crate
//!   cannot answer it.
//!
//! Backends are opt-in per host (`--features apple` / `gpu` / `cuda` / …); see
//! this crate's `Cargo.toml` for why they are not on by default. With no
//! feature selected this reports CPU only and says so.

use std::collections::BTreeMap;

use rlx_corpus::oracle::{tolerance_for, validate, validate_with_packed};
use rlx_ir::{Graph, Op};
use rlx_runtime::{Device, Session};

/// Deterministic, non-degenerate inputs. Constant values would let a kernel
/// that ignores its indices agree with the reference; values near zero make
/// `Div` blow up to infinity, which the oracle then refuses to score. Held in
/// [0.1, 1.0] with both signs. (Same generator as the CPU arm, deliberately —
/// a device disagreeing with CPU on *different* inputs proves less.)
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mag = 0.1 + (((i * 37 + seed * 101) % 90) as f32) * 0.01;
            if (i + seed).is_multiple_of(2) {
                mag
            } else {
                -mag
            }
        })
        .collect()
}

/// Deterministic bytes for a packed quantized weight.
///
/// Any 34-byte string is a valid `block_q8_0` (f16 scale + 32 int8), so no
/// encoder is needed. The scale's high byte is held in a modest exponent range
/// so a block decodes near 1.0 rather than into f16 subnormals, where a whole
/// row would come out ~0 and a Nop would be indistinguishable from a decode.
fn bytes_fill(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            if i % 34 == 1 {
                0x2c | ((i / 34 + seed) % 3) as u8
            } else if i % 34 == 0 {
                ((i * 31 + seed * 7) % 256) as u8
            } else {
                (((i * 37 + seed * 101) % 251) as i32 - 125) as i8 as u8
            }
        })
        .collect()
}

fn elems(g: &Graph, name: &str) -> Option<usize> {
    g.nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name: nm } | Op::Param { name: nm } if nm == name))
        .and_then(|n| {
            n.shape
                .dims()
                .iter()
                .map(|d| match d {
                    rlx_ir::Dim::Static(k) => Some(*k),
                    rlx_ir::Dim::Dynamic(_) => None,
                })
                .collect::<Option<Vec<usize>>>()
        })
        .map(|d| d.iter().product())
}

/// Devices this build can reach, in a stable order. A backend the build does
/// not contain is not listed; one it contains but cannot instantiate is handled
/// by [`require_device`].
fn devices() -> Vec<(&'static str, Device)> {
    [
        ("cpu", Device::Cpu),
        ("metal", Device::Metal),
        ("mlx", Device::Mlx),
        ("wgpu", Device::Gpu),
        ("vulkan", Device::Vulkan),
        ("cuda", Device::Cuda),
        ("rocm", Device::Rocm),
    ]
    .into_iter()
    .filter(|(_, d)| rlx_runtime::is_available(*d))
    .collect()
}

/// Under `RLX_REQUIRE_DEVICE=1`, a backend that is compiled in but cannot be
/// instantiated is a failure rather than an absence.
///
/// Without it a rig whose card has fallen off the bus reports this gate green
/// having run CPU alone — which is the exact shape of vacuous pass the flag was
/// added for, and the reason `rig.sh` sets it.
fn require_device() {
    if !rlx_ir::env::flag("RLX_REQUIRE_DEVICE") {
        return;
    }
    for (name, d) in [
        ("metal", Device::Metal),
        ("mlx", Device::Mlx),
        ("wgpu", Device::Gpu),
        ("vulkan", Device::Vulkan),
        ("cuda", Device::Cuda),
        ("rocm", Device::Rocm),
    ] {
        assert!(
            !rlx_runtime::feature_compiled(d) || rlx_runtime::is_available(d),
            "RLX_REQUIRE_DEVICE=1, `{name}` is compiled into this build, and it \
             could not be instantiated — the corpus would report ok having never \
             run a case on it"
        );
    }
}

/// How much looser this device's bound must be, and why.
///
/// **CUDA's default GEMM is TF32 on sm_80+**: 10 explicit mantissa bits instead
/// of 23. That is a deliberate, documented performance mode, not a defect — and
/// `tolerance_for` derives an f32 bound, so the two disagree by construction.
/// Measured here on an RTX 3080 Ti: `matmul::prefill` 2.52e-3 against a 2.0e-4
/// bound, `matmul::shared_input_branch` 4.50e-3 against 4.0e-4 — about 11x.
/// `RLX_CUDA_NO_TF32=1` brings CUDA to 19/19 at the full f32 bound, which is
/// what confirms the cause.
///
/// 16x is empirical, not derived: the raw epsilon ratio (2^-11 vs 2^-24) is
/// ~8000x, and a bound that loose would accept almost any wrong answer. This
/// sits just above what TF32 actually produces and ~500x tighter than the
/// theoretical worst case, so a real kernel defect still fails.
///
/// Applied ONLY to graphs containing a matmul — TF32 affects the GEMM path, and
/// widening the bound for an elementwise case would weaken a check for no
/// reason. The factor is printed so a CUDA pass is never mistaken for an
/// f32-accurate one.
fn precision_factor(device: Device, graph: &Graph) -> (f64, &'static str) {
    let has_gemm = graph
        .nodes()
        .iter()
        .any(|n| matches!(n.op, Op::MatMul | Op::DequantMatMul { .. }));
    if device == Device::Cuda && has_gemm && !rlx_ir::env::flag("RLX_CUDA_NO_TF32") {
        (16.0, "TF32 GEMM (set RLX_CUDA_NO_TF32=1 for the f32 bound)")
    } else {
        (1.0, "")
    }
}

/// One (device, family) tally.
#[derive(Default, Clone, Copy)]
struct Tally {
    validated: usize,
    unvalidated: usize,
    failed: usize,
}

#[test]
fn every_family_holds_on_every_available_device() {
    require_device();

    let devices = devices();
    // device -> family -> tally
    let mut grid: BTreeMap<&str, BTreeMap<&str, Tally>> = BTreeMap::new();
    let mut failures: Vec<String> = Vec::new();
    let mut widened: std::collections::BTreeSet<(&str, &str)> = Default::default();

    for (dev_name, dev) in &devices {
        for case in rlx_corpus::cases() {
            let t = grid
                .entry(dev_name)
                .or_default()
                .entry(case.family)
                .or_default();

            let mut inputs: Vec<(&str, Vec<f32>)> = Vec::new();
            let mut params: Vec<(&str, Vec<f32>)> = Vec::new();
            let mut packed: Vec<(&str, Vec<u8>)> = Vec::new();
            let mut resolved = true;
            for (i, node) in case.graph.nodes().iter().enumerate() {
                let is_u8 = node.shape.dtype() == rlx_ir::DType::U8;
                match &node.op {
                    // Packed quantized weight: raw bytes, not f32.
                    Op::Param { name } if is_u8 => match elems(&case.graph, name) {
                        Some(n) => packed.push((
                            Box::leak(name.clone().into_boxed_str()),
                            bytes_fill(n, i + 13),
                        )),
                        None => resolved = false,
                    },
                    Op::Input { name } => match elems(&case.graph, name) {
                        Some(n) => {
                            inputs.push((Box::leak(name.clone().into_boxed_str()), fill(n, i)))
                        }
                        None => resolved = false,
                    },
                    Op::Param { name } => match elems(&case.graph, name) {
                        Some(n) => {
                            params.push((Box::leak(name.clone().into_boxed_str()), fill(n, i + 7)))
                        }
                        None => resolved = false,
                    },
                    _ => {}
                }
            }
            if !resolved {
                t.unvalidated += 1;
                continue;
            }

            // A backend that cannot compile the case is a REPORTED failure, not
            // a skip: the corpus claims every family is compiler-ready, and a
            // panic here is that claim being wrong. Catching it keeps one bad
            // family from hiding the results for every family after it.
            let graph = case.graph.clone();
            let params_c = params.clone();
            let packed_c = packed.clone();
            let inputs_c = inputs.clone();
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let mut exe = Session::new(*dev).compile(graph);
                for (n, v) in &params_c {
                    exe.set_param(n, v);
                }
                for (n, v) in &packed_c {
                    exe.set_param_typed(n, v, rlx_ir::DType::U8);
                }
                let feed: Vec<(&str, &[f32])> =
                    inputs_c.iter().map(|(n, v)| (*n, v.as_slice())).collect();
                exe.run(&feed)
            }));
            let Ok(out) = ran else {
                t.failed += 1;
                failures.push(format!(
                    "{dev_name}: {}::{} panicked during compile/run",
                    case.family, case.name
                ));
                continue;
            };

            let (factor, why) = precision_factor(*dev, &case.graph);
            let tol = tolerance_for(&case.graph) * factor;
            if factor != 1.0 {
                widened.insert((dev_name, why));
            }
            let r = validate_with_packed(&case.graph, &inputs, &params, &packed, &out[0]);
            match (r.authority, r.max_rel_err) {
                (a, Some(err)) if a.is_independent() => {
                    if err > tol {
                        t.failed += 1;
                        failures.push(format!(
                            "{dev_name}: {}::{} max rel {err:.2e} > {tol:.1e} vs {}",
                            case.family,
                            case.name,
                            a.label()
                        ));
                    } else {
                        t.validated += 1;
                    }
                }
                _ => t.unvalidated += 1,
            }
        }
    }

    // Per-device, per-family rollup — the output this gate exists to produce.
    let mut report = String::new();
    for (dev, fams) in &grid {
        let tot: Tally = fams.values().fold(Tally::default(), |a, b| Tally {
            validated: a.validated + b.validated,
            unvalidated: a.unvalidated + b.unvalidated,
            failed: a.failed + b.failed,
        });
        report.push_str(&format!(
            "\n{dev}: {} validated, {} unvalidated, {} FAILED\n",
            tot.validated, tot.unvalidated, tot.failed
        ));
        for (fam, t) in fams {
            let mark = if t.failed == 0 { "ok  " } else { "FAIL" };
            report.push_str(&format!(
                "  {mark} {fam:<22} {}/{}\n",
                t.validated,
                t.validated + t.unvalidated + t.failed
            ));
        }
    }
    for (dev, why) in &widened {
        eprintln!("NOTE: {dev} scored at a WIDENED bound — {why}");
    }
    eprintln!("{report}");

    let device_names: Vec<&str> = devices.iter().map(|(n, _)| *n).collect();
    if device_names == ["cpu"] {
        eprintln!(
            "NOTE: CPU only — no GPU backend is compiled into this build. \
             Select one (`--features apple` on Darwin, `gpu`/`cuda`/`rocm` \
             elsewhere) or this gate covers exactly what oracle_gate.rs already did."
        );
    }

    assert!(
        failures.is_empty(),
        "corpus failed on {} case/device pair(s):\n  {}\n{report}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// **Would this gate actually notice a wrong device result?**
///
/// The test above passing proves nothing on its own — a harness that compiles
/// every case, discards the output and tallies `validated` would look
/// identical. `oracle_gate.rs::the_oracle_detects_a_wrong_answer` proves
/// [`validate`] scores correctly, but not that *this file's plumbing* reaches
/// it: the device could go unselected, `out[0]` could be the wrong output, the
/// tally could count a case nobody ran.
///
/// So: take a real case, run it on every device exactly as the gate does,
/// perturb one element of what the device returned, and require the same
/// `tolerance_for` + `validate` pair to reject it. A device that silently
/// returned nothing, or a harness that scored the wrong buffer, fails here.
#[test]
fn a_wrong_device_result_is_caught() {
    require_device();

    // `matmul` is the right probe: every device runs it, the oracle covers it,
    // and its derived tolerance is tight enough that one perturbed element
    // cannot hide inside it.
    let case = rlx_corpus::cases()
        .into_iter()
        .find(|c| c.family == "matmul")
        .expect("corpus has a matmul family");

    let mut inputs: Vec<(&str, Vec<f32>)> = Vec::new();
    let mut params: Vec<(&str, Vec<f32>)> = Vec::new();
    for (i, node) in case.graph.nodes().iter().enumerate() {
        match &node.op {
            Op::Input { name } => {
                let n = elems(&case.graph, name).expect("static input");
                inputs.push((Box::leak(name.clone().into_boxed_str()), fill(n, i)));
            }
            Op::Param { name } => {
                let n = elems(&case.graph, name).expect("static param");
                params.push((Box::leak(name.clone().into_boxed_str()), fill(n, i + 7)));
            }
            _ => {}
        }
    }

    let mut probed = 0usize;
    for (dev_name, dev) in devices() {
        let mut exe = Session::new(dev).compile(case.graph.clone());
        for (n, v) in &params {
            exe.set_param(n, v);
        }
        let feed: Vec<(&str, &[f32])> = inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        let out = exe.run(&feed);
        let mut actual = out[0].clone();
        assert!(
            !actual.is_empty(),
            "{dev_name} returned an empty output for {}::{} — the main gate would \
             have scored nothing and still reported ok",
            case.family,
            case.name
        );

        let tol = tolerance_for(&case.graph);

        // Unperturbed, this device must pass — otherwise the perturbation below
        // proves nothing (a gate that rejects everything is not a gate).
        let clean = validate(&case.graph, &inputs, &params, &actual);
        let clean_err = clean
            .max_rel_err
            .unwrap_or_else(|| panic!("{dev_name}: no independent authority for a matmul"));
        assert!(
            clean_err <= tol,
            "{dev_name}: unperturbed matmul already exceeds tolerance ({clean_err:.2e} > {tol:.1e})"
        );

        // Now break one element by well over the tolerance and require it seen.
        actual[0] += 1.0;
        let dirty = validate(&case.graph, &inputs, &params, &actual);
        let dirty_err = dirty.max_rel_err.expect("oracle scored the clean run");
        assert!(
            dirty_err > tol,
            "{dev_name}: a result perturbed by 1.0 scored {dirty_err:.2e} <= {tol:.1e} — \
             this gate would not notice a wrong answer from this device"
        );
        probed += 1;
    }

    assert!(probed > 0, "no device was available to probe");
    eprintln!("teeth: {probed} device(s) would surface a wrong result");
}
