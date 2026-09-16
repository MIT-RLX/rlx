// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Where does a transformer block's Metal time actually go?**
//!
//! Every Apple optimisation in this session was aimed at GEMM. Then
//! `reference_perf` showed rlx's sgemm sitting within noise of
//! `MPSMatrixMultiplication` at all nine shapes, with `pick_sgemm` choosing the
//! faster path every time — so there is no large GEMM gap to close, and a day
//! spent tuning a tile edge on a fallback kernel was aimed at the wrong thing.
//!
//! The obvious next question — *what fraction of a block is even GEMM?* — had
//! no answer in this tree. `rlx-bench` carries matmul, layer-norm and FFT
//! patterns but no transformer block, so nobody had measured the split.
//!
//! This builds a real block (QKV projection, attention, output projection,
//! RMS norms, SwiGLU FFN, residuals) and reports the per-op share via
//! [`rlx_metal::thunk_profile`].
//!
//! # Why this is runnable on a busy machine
//!
//! It reports **percentage of total**, not milliseconds. A share is far more
//! robust to contention than an absolute time: if every op is slowed by the
//! same competing workload, the ratios survive. That is not a licence to quote
//! the ms column — the header says which is which — but it does mean the
//! question "is this backend GEMM-bound or not" can be answered today rather
//! than waiting for an idle window.
//!
//! Weights are synthetic. The *shape* of the graph is what determines the
//! split, not the values in it.
//!
//! ```sh
//! cargo run --release -p rlx-metal --example block_profile
//! cargo run --release -p rlx-metal --example block_profile -- --seq 512
//! ```

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("Metal is macOS-only.");
}

#[cfg(target_os = "macos")]
fn main() {
    use rlx_ir::op::MaskKind;
    use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
    use rlx_metal::backend::MetalExecutable;

    const F: DType = DType::F32;

    /// One decoder block, Llama-shaped: RMSNorm -> QKV -> attention -> out
    /// proj -> residual -> RMSNorm -> SwiGLU FFN -> residual.
    fn block(seq: usize, d_model: usize, heads: usize, d_ff: usize) -> Graph {
        let hd = d_model / heads;
        let mut g = Graph::new("block");
        let x = g.input("x", Shape::new(&[seq, d_model], F));

        let n1_g = g.param("n1_g", Shape::new(&[d_model], F));
        let n1_b = g.param("n1_b", Shape::new(&[d_model], F));
        let h = g.rms_norm(x, n1_g, n1_b, 1e-5);

        // Fused QKV, as a real implementation would: one GEMM, not three.
        let wqkv = g.param("wqkv", Shape::new(&[d_model, 3 * d_model], F));
        let qkv = g.matmul(h, wqkv, Shape::new(&[seq, 3 * d_model], F));

        // Reshape into [1, H, S, D] per operand. `narrow` slices the fused
        // projection; the reshapes are free (no dispatch) and exist so the
        // attention op sees the layout it expects.
        let mut head = |off: usize| {
            let t = g.narrow_(qkv, 1, off * d_model, d_model);
            g.reshape_(t, vec![1, heads as i64, seq as i64, hd as i64])
        };
        let q = head(0);
        let k = head(1);
        let v = head(2);

        let attn = g.add_node(
            Op::Attention {
                num_heads: heads,
                head_dim: hd,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[1, heads, seq, hd], F),
        );
        let attn2 = g.reshape_(attn, vec![seq as i64, d_model as i64]);

        let wo = g.param("wo", Shape::new(&[d_model, d_model], F));
        let proj = g.matmul(attn2, wo, Shape::new(&[seq, d_model], F));
        let res1 = g.add(x, proj);

        let n2_g = g.param("n2_g", Shape::new(&[d_model], F));
        let n2_b = g.param("n2_b", Shape::new(&[d_model], F));
        let h2 = g.rms_norm(res1, n2_g, n2_b, 1e-5);

        // SwiGLU: two up-projections, silu on one, elementwise gate, down.
        let w_up = g.param("w_up", Shape::new(&[d_model, d_ff], F));
        let w_gate = g.param("w_gate", Shape::new(&[d_model, d_ff], F));
        let up = g.matmul(h2, w_up, Shape::new(&[seq, d_ff], F));
        let gate = g.matmul(h2, w_gate, Shape::new(&[seq, d_ff], F));
        let act = g.silu(gate);
        let gated = g.mul(up, act);
        let w_dn = g.param("w_dn", Shape::new(&[d_ff, d_model], F));
        let dn = g.matmul(gated, w_dn, Shape::new(&[seq, d_model], F));
        let out = g.add(res1, dn);

        g.set_outputs(vec![out]);
        g
    }

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    let args: Vec<String> = std::env::args().collect();
    let seq: usize = args
        .iter()
        .position(|a| a == "--seq")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);

    // The profiler is opt-in via env; turn it on for this process so the
    // example does one thing without a wrapper script.
    rlx_ir::env::set("RLX_METAL_THUNK_PROFILE", "1");
    // This example reports for itself, so suppress the backend's per-run
    // summary — otherwise the table prints twice and the two copies cover
    // different sample sets (the reset below happens between them).
    rlx_metal::thunk_profile::set_auto_print(false);

    // Qwen3-0.6B-ish geometry.
    let (d_model, heads, d_ff) = (1024usize, 16usize, 3072usize);

    println!("transformer block on Metal — where does the time go?\n");
    println!("  geometry : seq={seq} d_model={d_model} heads={heads} d_ff={d_ff}");
    println!("  weights  : synthetic (the graph SHAPE decides the split, not the values)");
    println!(
        "  read     : the % column. Absolute ms is contention-sensitive; a share is\n\
         \x20            far less so, because a competing workload slows every op alike."
    );
    println!("  caution  : this is one block, forward only, f32.\n");

    let g = block(seq, d_model, heads, d_ff);
    let dispatches = g
        .nodes()
        .iter()
        .filter(|n| {
            !matches!(
                n.op,
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } | Op::Reshape { .. }
            )
        })
        .count();
    println!(
        "  graph    : {} nodes, {dispatches} dispatching\n",
        g.nodes().len()
    );

    let mut exe = MetalExecutable::compile(g);
    for (name, n) in [
        ("n1_g", d_model),
        ("n1_b", d_model),
        ("wqkv", d_model * 3 * d_model),
        ("wo", d_model * d_model),
        ("n2_g", d_model),
        ("n2_b", d_model),
        ("w_up", d_model * d_ff),
        ("w_gate", d_model * d_ff),
        ("w_dn", d_ff * d_model),
    ] {
        exe.set_param(name, &fill(n, 0xa5a5 ^ n as u64));
    }
    let xv = fill(seq * d_model, 0x5eed);

    // Warm up compilation and first-touch, then reset so the profile is steady
    // state. Folding shader compilation into the numbers would put every op's
    // first dispatch at the top of the table.
    for _ in 0..3 {
        let _ = exe.run(&[("x", &xv)]);
    }
    rlx_metal::thunk_profile::reset();
    let iters = 20usize;
    for _ in 0..iters {
        let _ = exe.run(&[("x", &xv)]);
    }
    if let Some(ms) = rlx_metal::gpu_span::last_ms() {
        println!("  last run device span: {ms:.4} ms (contended — indicative only)\n");
    }
    rlx_metal::thunk_profile::print_summary();

    // ── Score the cost model against what just happened ──────────────────
    //
    // `MetalHwModel::estimate_transformer_forward_ns` predicts this exact
    // shape. Until today its compute term was computed in seconds and labelled
    // nanoseconds, so it was ~1e9x too small and swamped by a constant — a
    // defect that survived because nothing ever compared it to a clock. This is
    // that comparison.
    let hw = rlx_metal::cost::hw_model();
    let predicted_ms = hw.estimate_transformer_forward_ns(1, seq, d_model, d_ff, heads, 1) / 1e6;
    if let Some(measured_ms) = rlx_metal::thunk_profile::total_ms() {
        // The profile accumulated `iters` runs; the model predicts one.
        let per_run = measured_ms / iters as f64;
        let ratio = predicted_ms / per_run;
        println!(
            "\n[cost model] predicted {predicted_ms:.3} ms vs measured {per_run:.3} ms \
             per run  ->  {ratio:.2}x"
        );
        println!(
            "  A model is useful if this is near 1.0 and STABLE across seq. The measured\n\
             \x20 side is contended here, so read the trend across --seq rather than the\n\
             \x20 absolute ratio."
        );
    }
}
