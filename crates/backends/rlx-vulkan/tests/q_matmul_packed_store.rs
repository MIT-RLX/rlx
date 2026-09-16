// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **`q_matmul`'s packed-i8 output must survive concurrent writers.**
//!
//! `Op::QMatMul` produces an I8 tensor, and the f32-uniform arena packs four i8
//! codes per 32-bit word. `arena_store_i8` implements a byte store as a
//! read-modify-write of the containing word — its own doc says so and points at
//! `quantize_i8.comp` for the word-owner launch that makes that safe.
//!
//! `q_matmul.comp` did not use that launch. It dispatched one invocation per
//! output element and had each write its own byte, so four invocations raced for
//! every word and only one byte survived. Measured before the fix: exactly every
//! fourth output held a real value and the other three sat at the output zero
//! point — a plausible i8 tensor, three quarters of it stale, with no error
//! anywhere. No test covered `QMatMul` on Vulkan at all.
//!
//! **Scope.** This gate is the race, not the arithmetic: it requires every
//! output to be written and every run to agree. A full value-parity gate against
//! rlx-cpu is *not* asserted here — CPU and Vulkan still disagree on the
//! magnitudes for this graph, and which one is right is an open question this
//! test deliberately does not prejudge. Writing an equality assert against an
//! oracle that may itself be wrong would be worse than writing none.
//!
//! `N` is deliberately not a multiple of 4, so the ragged tail word is shared
//! too.

use rlx_ir::{DType, Graph, Shape};
use rlx_vulkan::backend::VulkanExecutable;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Skip when no Vulkan device is present.
///
/// `rlx_ir::env::skip_unless_device` rather than a bare
/// `if !is_available() { return }`: the bare form reports `ok` on a rig with no
/// device, so a CI box that lost its Vulkan driver would look green. This one
/// honours `RLX_REQUIRE_DEVICE=1` and fails instead. (A local `fn available()`
/// wrapper hides the same problem from `require_device_coverage` without fixing
/// it.)
fn skip() -> bool {
    rlx_ir::env::skip_unless_device("vulkan", true, rlx_vulkan::is_available())
}

fn gpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

const M: usize = 6;
const K: usize = 8;
const N: usize = 7;

/// f32 in, f32 out, packed i8 in the middle: `Quantize` -> `QMatMul` ->
/// `Dequantize`. Keeps I8 off the graph boundary (the executable's `run` takes
/// f32 lanes) while still putting a packed-i8 tensor under the racing writers.
fn build() -> Graph {
    let mut g = Graph::new("qmm");
    let xf = g.input("x", Shape::new(&[M, K], DType::F32));
    let wf = g.input("w", Shape::new(&[K, N], DType::F32));
    let bi = g.input("b", Shape::new(&[N], DType::I32));
    // Scales and `mult` chosen so nothing saturates: codes stay around ±8 and
    // ±6, so |acc| <= 8*8*6 = 384 and `acc * mult` fits an i8 comfortably. A
    // saturating configuration would clamp every output to ±127 and hide
    // exactly the per-element differences this test is looking for.
    let x = g.quantize(xf, 0.25, 0);
    let w = g.quantize(wf, 0.25, 0);
    let y = g.q_matmul(x, w, bi, 0, 0, 0, 0.1, Shape::new(&[M, N], DType::I8));
    // scale 1.0 / zp 0: the f32 output *is* the i8 code.
    let out = g.dequantize(y, 1.0, 0);
    g.set_outputs(vec![out]);
    g
}

fn wave(n: usize, phase: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * phase).sin() * amp).collect()
}

#[test]
fn every_packed_output_is_written_and_runs_agree() {
    if skip() {
        return;
    }
    let _g = gpu_lock();

    let x = wave(M * K, 0.31, 2.0);
    let w = wave(K * N, 0.17, 1.5);
    let b = vec![0.0f32; N];
    let inputs: Vec<(&str, &[f32])> = vec![("x", &x), ("w", &w), ("b", &b)];

    // Run repeatedly: a lost read-modify-write is a race, so one pass can get
    // lucky.
    let mut first: Option<Vec<f32>> = None;
    for pass in 0..8 {
        let got = VulkanExecutable::compile(build()).run(&inputs).remove(0);
        assert_eq!(got.len(), M * N, "pass {pass}: output length");

        // The race signature: with one writer per element, only every fourth
        // byte survived and the rest stayed at the output zero point (0 here,
        // so `dequantize(scale 1, zp 0)` reads back exactly 0.0). Three
        // quarters of the tensor being zero is the failure, so require that
        // fewer than half the elements are zero — real data has a few.
        let zeros = got.iter().filter(|v| **v == 0.0).count();
        assert!(
            zeros * 2 < got.len(),
            "pass {pass}: {zeros}/{} outputs are zero — packed-i8 writes are \
             being lost to a read-modify-write race\n  out={got:?}",
            got.len()
        );
        // Specifically, the lost-write pattern leaves the non-word-owner lanes
        // (index % 4 != 0) untouched. Catch it directly rather than only
        // statistically.
        let tail_zeros = got
            .iter()
            .enumerate()
            .filter(|(i, v)| i % 4 != 0 && **v == 0.0)
            .count();
        let tail_total = got.iter().enumerate().filter(|(i, _)| i % 4 != 0).count();
        assert!(
            tail_zeros < tail_total,
            "pass {pass}: every non-word-owner lane is zero — the classic \
             lost-write signature"
        );

        if let Some(f) = &first {
            assert_eq!(&got, f, "pass {pass}: run-to-run divergence (race)");
        } else {
            first = Some(got);
        }
    }
}

/// The path this test exercises only means anything if `Op::Quantize` itself is
/// right, so pin that against the CPU oracle here too — it is elementwise, so
/// equality is exact.
///
/// This is also what would have caught the push-constant layout bug:
/// `quantize_i8.comp` read its affine table from the wrong offsets (naga gives a
/// bare `uint[]` push array std140's 16-byte stride) so `scale` and
/// `zero_point` both came back 0 and every quantized tensor was zeros.
#[test]
fn quantize_dequantize_roundtrip_matches_cpu() {
    if skip() {
        return;
    }
    let _g = gpu_lock();
    let x = wave(48, 0.31, 2.0);
    let build_rt = |scale: f32, zp: i32| {
        let mut g = Graph::new("rt");
        let xf = g.input("x", Shape::new(&[6, 8], DType::F32));
        let q = g.quantize(xf, scale, zp);
        let d = g.dequantize(q, scale, zp);
        g.set_outputs(vec![d]);
        g
    };
    for (scale, zp) in [(0.05f32, 0i32), (0.05, 3), (0.25, -7), (1.0, 0)] {
        let want = {
            use rlx::prelude::*;
            Session::new(Device::Cpu)
                .compile(build_rt(scale, zp))
                .run(&[("x", &x)])
                .remove(0)
        };
        let got = VulkanExecutable::compile(build_rt(scale, zp))
            .run(&[("x", &x)])
            .remove(0);
        assert_eq!(
            got, want,
            "scale={scale} zp={zp}: vulkan quantize round-trip != cpu\n  vk={got:?}\n cpu={want:?}"
        );
    }
}
