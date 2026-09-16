// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Is your A/B actually an A/B?** — a pre-flight check for any benchmark that
//! selects a code path through an environment variable.
//!
//! Written after a Metal reference-perf harness produced a confident table
//! showing rlx's own sgemm at exactly 1.00x of MPS across nine shapes. Nine
//! shapes agreeing to 1% while per-shape noise ran 20-90% is not a result, it
//! is a signal that both arms measured the same thing. Before trusting any such
//! comparison, prove the knob moves the kernel.
//!
//! # Why bit-equality is the wrong proof
//!
//! The obvious check — "different kernels must give different low bits" — is
//! **invalid**, and this file exists partly to record why. IEEE 754 f32 is
//! deterministic: two entirely different implementations that accumulate the K
//! reduction in the same order produce bit-identical results. rlx's Metal sgemm
//! variants do exactly that by design, and are bit-identical to the CPU thunk at
//! most shapes. So bit-equality proves nothing about which kernel ran, and
//! bit-*difference* only proves the reduction order changed.
//!
//! # What is a valid proof
//!
//! Timing against a variant that is *catastrophically* slower. `naive` and
//! `tiled` are far enough off the tuned paths that the gap cannot be confused
//! with contention, however noisy the machine. If pinning them does not move
//! the clock, the knob is not reaching the kernel.
//!
//! ```sh
//! cargo run --release -p rlx-metal --example probe_ab_validity
//! ```

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("Metal is macOS-only.");
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}

#[cfg(target_os = "macos")]
mod macos {
    use std::time::Instant;

    use rlx_ir::{DType, Graph, Shape};
    use rlx_runtime::{Device, Session};

    /// Load average above which timing on this host is contention, not
    /// measurement.
    ///
    /// The CUDA twin refuses to run when NVML reports the GPU busy. macOS
    /// exposes no equivalent exclusive-access signal for the GPU, but the
    /// symptom is the same and load average catches it: this harness was
    /// developed at load 51 on an 8-core machine, where the SAME pinned variant
    /// timed 1.19 ms and 10.94 ms on consecutive runs. Every ratio taken there
    /// was noise wearing a table's clothes.
    const MAX_LOAD_AVG: f64 = 4.0;

    /// 1-minute load average, or `None` if it cannot be read.
    fn load_avg_1m() -> Option<f64> {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .ok()?;
        // "{ 51.19 45.93 42.83 }"
        String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<f64>().ok())
    }

    /// `Err` when the machine is too busy for the numbers to mean anything.
    fn contention_ok() -> Result<String, String> {
        if rlx_ir::env::flag("RLX_ALLOW_THROTTLE") {
            return Ok("RLX_ALLOW_THROTTLE=1 — gate bypassed".into());
        }
        match load_avg_1m() {
            Some(l) if l > MAX_LOAD_AVG => Err(format!(
                "1-minute load average is {l:.1} (limit {MAX_LOAD_AVG:.1})"
            )),
            Some(l) => Ok(format!("load average {l:.2}")),
            None => Ok("load average unreadable — proceeding unverified".into()),
        }
    }

    const B: usize = 1024;
    const WARMUP: usize = 3;
    const ITERS: usize = 10;

    /// Ratio between the fastest and slowest pinned variant below which the
    /// knob is presumed dead. Variants this far apart cannot be confused with
    /// contention on any machine.
    const MIN_VARIANT_SPREAD: f64 = 1.5;

    /// Names `RLX_METAL_SGEMM_VARIANT` parses. `simd64` is included now that it
    /// has a spelling; it previously had none, so the variant the cascade
    /// actually selects at large shapes could not be pinned or compared.
    const PINNABLE: &[&str] = &["naive", "tiled", "simd4x4", "simd64", "mps"];

    fn matmul_graph(name: &str, m: usize, k: usize, n: usize) -> Graph {
        let mut g = Graph::new(name);
        let x = g.input("x", Shape::new(&[m, k], DType::F32));
        let w = g.param("w", Shape::new(&[k, n], DType::F32));
        let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![y]);
        g
    }

    fn time_ms(dev: Device, g: Graph, xb: &[f32], wb: &[f32]) -> (f64, Vec<f32>) {
        let mut e = Session::new(dev).compile(g);
        e.set_param("w", wb);
        let mut out = Vec::new();
        for _ in 0..WARMUP {
            out = e.run(&[("x", xb)])[0].clone();
        }
        let t0 = Instant::now();
        for _ in 0..ITERS {
            let _ = e.run(&[("x", xb)]);
        }
        (t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64, out)
    }

    fn bit_eq(a: &[f32], b: &[f32]) -> bool {
        !a.is_empty()
            && a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    pub fn run() {
        if !rlx_metal::is_available() {
            println!("Metal not available.");
            return;
        }
        match contention_ok() {
            Ok(note) => println!("host state: {note}"),
            Err(why) => {
                eprintln!(
                    "REFUSING to measure: {why}.\n\
                     A timing taken under contention is not a timing. Free the machine, or set\n\
                     RLX_ALLOW_THROTTLE=1 if you accept the numbers are indicative only."
                );
                std::process::exit(1);
            }
        }

        rlx_ir::env::set("RLX_DISABLE_MPSGRAPH", "1");
        rlx_ir::env::set("RLX_DISABLE_MPSGRAPH_EXECUTABLE", "1");

        let xb: Vec<f32> = (0..B * B).map(|i| ((i % 97) as f32) * 1e-2 - 0.5).collect();
        let wb: Vec<f32> = (0..B * B).map(|i| ((i % 89) as f32) * 1e-2 - 0.5).collect();

        println!("\n1. DOES THE KNOB REACH THE KERNEL?  ({B}^3 f32 matmul)");
        let mut times = Vec::new();
        let mut outs = Vec::new();
        for v in PINNABLE {
            rlx_ir::env::set("RLX_METAL_SGEMM_VARIANT", *v);
            let (ms, out) = time_ms(
                Device::Metal,
                matmul_graph(&format!("pin_{v}"), B, B, B),
                &xb,
                &wb,
            );
            println!("   pinned {v:>8}: {ms:>8.3} ms");
            times.push(ms);
            outs.push(out);
        }
        rlx_ir::env::unset("RLX_METAL_SGEMM_VARIANT");

        let (lo, hi) = times
            .iter()
            .fold((f64::MAX, 0.0f64), |(l, h), &t| (l.min(t), h.max(t)));
        let spread = hi / lo;
        if spread < MIN_VARIANT_SPREAD {
            println!(
                "\n   VERDICT: spread {spread:.2}x — the pin is NOT reaching the kernel.\n   \
                 Any A/B built on this variable has been measuring one path twice."
            );
            return;
        }
        println!("\n   VERDICT: spread {spread:.2}x — the pin works, different kernels ran.");

        println!("\n2. WHY BIT-EQUALITY WOULD HAVE LIED");
        let all_same = outs.windows(2).all(|w| bit_eq(&w[0], &w[1]));
        println!(
            "   the {} pinned variants above are {} to each other",
            PINNABLE.len(),
            if all_same {
                "BIT-IDENTICAL"
            } else {
                "not all bit-identical"
            }
        );
        let (_, cpu_out) = time_ms(Device::Cpu, matmul_graph("cpu_ref", B, B, B), &xb, &wb);
        println!(
            "   and metal is {} to the CPU thunk",
            if bit_eq(&outs[0], &cpu_out) {
                "BIT-IDENTICAL"
            } else {
                "not identical"
            }
        );
        if all_same {
            println!(
                "   => demonstrably different kernels, byte-identical output. Reduction order\n   \
                 is preserved across them, so bit-equality says nothing about which ran."
            );
        }

        println!("\n3. WHAT THE DEFAULT ACTUALLY PICKS");
        let (dflt, _) = time_ms(Device::Metal, matmul_graph("dflt", B, B, B), &xb, &wb);
        let best_pinned = PINNABLE[times
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
            .map(|(i, _)| i)
            .expect("nonempty")];
        println!("   default routing: {dflt:>8.3} ms");
        println!("   best pinnable  : {lo:>8.3} ms  ({best_pinned})");
        if dflt < lo {
            // `Simd64` has no accepted spelling in RLX_METAL_SGEMM_VARIANT, so
            // it cannot be pinned; the default beating every pinnable variant is
            // how it makes itself visible.
            println!(
                "   => the default is {:.2}x faster than anything pinnable, i.e. it selected a\n   \
                 variant with no pin name (Simd64 / Simd64SplitK).",
                lo / dflt
            );
        }
    }
}
