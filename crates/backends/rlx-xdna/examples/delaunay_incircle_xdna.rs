// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// SPECIAL VERSION FOR THE AMD XDNA / Ryzen AI NPU of the Delaunay in-circle predicate.
//
// Same idea as the ANE version (`rlx-coreml/examples/delaunay_incircle_ane.rs`): the
// flip/gather can't run on an NPU, but the in-circle PREDICATE batch is pure dense
// arithmetic. Its compute-heavy core — det = Σ₃ lift·minor over B tests — is a batched
// dot product, i.e. a matmul, which is the AIE array's native INT8/BF16 op. This example
// EMITS the AIE-MLIR for that combine via rlx-xdna's pure-Rust emitter (no Python), the
// same emitter validated bit-exact on the Phoenix NPU. Running it on the NPU needs the
// amd box's `aiecc`/PEANO toolchain (compile → xclbin) + XRT — see xdna_matmul_tiled.
//
// NOTE: rlx is a COMPILER — the ANE and XDNA versions share the SAME rlx_ir::Graph
// (the elementwise determinant DAG). CoreML lowers it to the ANE; rlx-runtime's
// XdnaBackend (Device::Xdna) lowers the same graph to these AIE kernels for the NPU.
//   cargo run -p rlx-xdna --example delaunay_incircle_xdna --release -- <B>
use rlx_xdna::aie::{Eltwise, emit_eltwise_chain, emit_matmul};

fn main() {
    // B = number of in-circle tests in the batch (triangles × test points, host-gathered).
    let b: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let b = b.div_ceil(16) * 16; // AIE vector width 16

    // Stage 1 (elementwise, INT32 AIE vector core): the coordinate lifts / minors are
    // per-element sub/mul — emit a representative fused i32 eltwise chain over the batch.
    let elt = emit_eltwise_chain(
        b,
        512.min(b),
        &[Eltwise::MulScalar(1), Eltwise::AddScalar(0)],
    );

    // Stage 2 (the compute-heavy combine): det[B] = [lift a2,b2,c2] · [minor bc,-ac,ab].
    // A [B,3]·[3,1] contraction = a matmul → the AIE INT8 MAC array's native op. The AIE2
    // MAC tile needs k,n multiples of 8, so pad the 3-wide contraction to [B,8]·[8,8]
    // (the extra lanes carry zeros) — the standard "small-K padded to tile" NPU pattern.
    let mm = emit_matmul(b, 8, 8);

    println!("XDNA in-circle predicate, B={b} tests");
    println!(
        "  stage1 eltwise (lifts/minors) AIE-MLIR: {} lines",
        elt.lines().count()
    );
    println!(
        "  stage2 combine  det=[B,3]·[3,1] matmul AIE-MLIR: {} lines",
        mm.lines().count()
    );
    let dir = "/tmp/rlx_incircle_xdna";
    std::fs::create_dir_all(dir).ok();
    std::fs::write(format!("{dir}/incircle_eltwise.mlir"), &elt).unwrap();
    std::fs::write(format!("{dir}/incircle_combine.mlir"), &mm).unwrap();
    println!("  emitted → {dir}/incircle_{{eltwise,combine}}.mlir");
    println!("\nTo run on the NPU (amd box): AIECC=<mlir_aie/bin/aiecc> PEANO=<llvm-aie> \\");
    println!(
        "  compile_overlay(...) → xclbin, then XRT dispatch (see examples/xdna_matmul_tiled.rs)."
    );
    println!("Precision on the AIE: INT8/BF16 MAC — bf16 certifies ~67% of in-circle signs");
    println!("(measured, examples/predicate_precision), so ~33% fall back to exact on host.");
    assert!(
        !elt.is_empty() && mm.contains("aie."),
        "emitters produced AIE-MLIR"
    );
    println!("\nemit OK (pure-Rust AIE-MLIR generation; execution is toolchain-gated).");
}
