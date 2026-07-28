// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **rlx emits AIE-MLIR** — the first increment of the native codegen path
//! (mirrors `rlx-cerebras`→CSL / `rlx-fpga`→SystemVerilog).
//!
//! Instead of authoring AIE designs with the IRON Python API, rlx generates the
//! `aie` dialect MLIR itself, which the native `aiecc` binary
//! ([`crate::compile`]) then compiles to an overlay — no Python anywhere.
//!
//! This module emits a **DMA passthrough** design (`out = in`, no compute core):
//! `ShimNOC → MemTile → ShimNOC` via two linked ObjectFIFOs. It's the minimal
//! real design — it proves the whole `rlx → AIE-MLIR → aiecc → NPU` pipeline
//! end to end. A compute microkernel (matmul) is the next layer on this seam.

/// Emit the AIE MLIR for an `i32` DMA passthrough of `len` elements. `fifo` is
/// the per-transfer ObjectFIFO chunk (the MemTile buffer size); `len` must be a
/// multiple of `fifo`. The runtime sequence takes the standard 3 buffer args
/// (`arg0` = input, `arg2` = output; `arg1` unused) so it runs through the same
/// `MLIR_AIE(opcode, instr, ninstr, A, B, C)` kernel ABI as the matmul overlay.
pub fn emit_passthrough(len: usize, fifo: usize) -> String {
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %logical_shim_noc = aie.logical_tile<ShimNOCTile>(?, ?)
    %logical_mem = aie.logical_tile<MemTile>(?, ?)
    %logical_shim_noc_0 = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in(%logical_shim_noc, {{%logical_mem}}, 2 : i32) : !aie.objectfifo<memref<{fifo}xi32>>
    aie.objectfifo @in_fwd(%logical_mem, {{%logical_shim_noc_0}}, 2 : i32) : !aie.objectfifo<memref<{fifo}xi32>>
    aie.objectfifo.link [@in] -> [@in_fwd]([] [0])
    aie.runtime_sequence(%arg0: memref<{len}xi32>, %arg1: memref<{len}xi32>, %arg2: memref<{len}xi32>) {{
      %0 = aiex.dma_configure_task_for @in {{
        aie.dma_bd(%arg0 : memref<{len}xi32>, 0, {len}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {len}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @in_fwd {{
        aie.dma_bd(%arg2 : memref<{len}xi32>, 0, {len}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {len}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }}
  }}
}}
"#
    )
}

/// AIE2 vector width for i32 — 512-bit vector registers hold 16 i32 lanes, so
/// the vectorized cores process 16 elements per instruction.
const VEC: usize = 16;

/// A 1-D i32 elementwise op the emitter can lower to an AIE compute core. This
/// is the first real *compute* kernel the rlx→AIE-MLIR compiler generates (the
/// passthrough had no core). Each maps to a per-element `arith` expression.
#[derive(Clone, Copy, Debug)]
pub enum Eltwise {
    /// `out = in + s`
    AddScalar(i32),
    /// `out = in * s`
    MulScalar(i32),
    /// `out = max(0, in)`
    Relu,
}

impl Eltwise {
    /// Short kernel name (for artifacts / bench labels).
    pub fn name(&self) -> &'static str {
        match self {
            Eltwise::AddScalar(_) => "add_scalar",
            Eltwise::MulScalar(_) => "mul_scalar",
            Eltwise::Relu => "relu",
        }
    }
    /// MLIR binding `out` = op(`inp`) on a `vector<VEC x i32>` (the AIE2 vector
    /// unit processes 16 i32 lanes per instruction); `k` disambiguates temporaries
    /// so several ops can be chained (fused) in one vectorized core body.
    fn core_expr_named(&self, inp: &str, out: &str, k: usize) -> String {
        match self {
            Eltwise::AddScalar(s) => format!(
                "          %cs{k} = arith.constant {s} : i32\n          %vs{k} = vector.broadcast %cs{k} : i32 to vector<{VEC}xi32>\n          {out} = arith.addi {inp}, %vs{k} : vector<{VEC}xi32>\n"
            ),
            Eltwise::MulScalar(s) => format!(
                "          %cs{k} = arith.constant {s} : i32\n          %vs{k} = vector.broadcast %cs{k} : i32 to vector<{VEC}xi32>\n          {out} = arith.muli {inp}, %vs{k} : vector<{VEC}xi32>\n"
            ),
            Eltwise::Relu => format!(
                "          %vz{k} = arith.constant dense<0> : vector<{VEC}xi32>\n          {out} = arith.maxsi {inp}, %vz{k} : vector<{VEC}xi32>\n"
            ),
        }
    }
    /// Apply on the host (CPU reference for validation).
    pub fn apply(&self, x: i32) -> i32 {
        match self {
            Eltwise::AddScalar(s) => x.wrapping_add(*s),
            Eltwise::MulScalar(s) => x.wrapping_mul(*s),
            Eltwise::Relu => x.max(0),
        }
    }
}

/// Emit the AIE-MLIR for a 1-D `n`-element i32 elementwise `op` on one compute
/// tile: `ShimNOC → CoreTile → ShimNOC`, applying `op` per element. `n` is
/// streamed through the tile in `chunk`-sized pieces via double-buffered
/// ObjectFIFOs (like the passthrough's MemTile chunking) — so `n` can be large
/// while only `chunk` i32 live in tile memory at once. Requires `n % chunk == 0`;
/// `chunk` must fit the core's local memory (≈ a few K i32). Runtime ABI is the
/// passthrough 3-arg one (`arg0`=input, `arg1` unused, `arg2`=output).
pub fn emit_eltwise(n: usize, chunk: usize, op: Eltwise) -> String {
    emit_eltwise_chain(n, chunk, &[op])
}

/// Emit a FUSED chain of elementwise `ops`, applied in sequence per element in a
/// single core pass — **one** NPU dispatch and **one** DMA round-trip for the
/// whole chain instead of one kernel/dispatch per op. This is rlx's fusion
/// advantage on the NPU: e.g. `[MulScalar(w), AddScalar(b), Relu]` fuses to
/// `relu(w*x + b)` in one kernel. Same streaming/ABI as [`emit_eltwise`].
pub fn emit_eltwise_chain(n: usize, chunk: usize, ops: &[Eltwise]) -> String {
    assert!(chunk > 0 && n % chunk == 0, "n ({n}) must be a multiple of chunk ({chunk})");
    assert!(chunk % VEC == 0, "chunk ({chunk}) must be a multiple of the vector width ({VEC})");
    assert!(!ops.is_empty(), "need at least one op");
    // Thread the value through the chain: %v0 = load; op0 → %v1; op1 → %v2; …
    let mut expr = String::new();
    let mut cur = "%v0".to_string();
    for (k, op) in ops.iter().enumerate() {
        let next = format!("%v{}", k + 1);
        expr += &op.core_expr_named(&cur, &next, k);
        cur = next;
    }
    let store_ssa = cur;
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_in = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim_in, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{chunk}xi32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{chunk}xi32>>
    %0 = aie.core(%core) {{
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %cmax step %c1 {{
        %in_sv = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<{chunk}xi32>>
        %in = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<{chunk}xi32>> -> memref<{chunk}xi32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{chunk}xi32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{chunk}xi32>> -> memref<{chunk}xi32>
        %lo = arith.constant 0 : index
        %hi = arith.constant {chunk} : index
        %st = arith.constant {VEC} : index
        scf.for %i = %lo to %hi step %st {{
          %v0 = vector.load %in[%i] : memref<{chunk}xi32>, vector<{VEC}xi32>
{expr}          vector.store {store_ssa}, %out[%i] : memref<{chunk}xi32>, vector<{VEC}xi32>
        }}
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{n}xi32>, %arg1: memref<{n}xi32>, %arg2: memref<{n}xi32>) {{
      %t0 = aiex.dma_configure_task_for @in0 {{
        aie.dma_bd(%arg0 : memref<{n}xi32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%t0)
      %t2 = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{n}xi32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t0)
    }}
  }}
}}
"#
    )
}

/// Emit AIE-MLIR for a 1-D `n`-element **f32** ReLU on one vectorized compute
/// tile: `ShimNOC → CoreTile → ShimNOC`, `out = max(0.0, in)`. This is the f32
/// twin of the i32 [`emit_eltwise`] path — real rlx activations are f32, and
/// AIE2 has a native f32 vector FPU, so `arith.maximumf` on `vector<16xf32>`
/// legalizes (unlike the i32 vector *multiply* the matmul hit). Same streaming /
/// double-buffered ObjectFIFO / 3-arg runtime ABI as [`emit_eltwise_chain`].
/// Requires `n % chunk == 0`, `chunk % VEC == 0`.
pub fn emit_relu_f32(n: usize, chunk: usize) -> String {
    assert!(chunk > 0 && n % chunk == 0, "n ({n}) must be a multiple of chunk ({chunk})");
    assert!(chunk % VEC == 0, "chunk ({chunk}) must be a multiple of the vector width ({VEC})");
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_in = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim_in, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{chunk}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{chunk}xf32>>
    %0 = aie.core(%core) {{
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %cmax step %c1 {{
        %in_sv = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<{chunk}xf32>>
        %in = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<{chunk}xf32>> -> memref<{chunk}xf32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{chunk}xf32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{chunk}xf32>> -> memref<{chunk}xf32>
        // SCALAR f32 core: the AIE2 aievec vectorizer supports int and bf16 vector
        // ops but NOT f32 (both `aievec.max` and `aievec.cmp` reject f32), so the
        // f32 FPU path is scalar. bf16 is the NPU-native vectorized float type —
        // vectorized f32 activation = cast→bf16→vector→cast-back (perf follow-on).
        %lo = arith.constant 0 : index
        %hi = arith.constant {chunk} : index
        %st = arith.constant 1 : index
        %z = arith.constant 0.0 : f32
        scf.for %i = %lo to %hi step %st {{
          %x = memref.load %in[%i] : memref<{chunk}xf32>
          %y = arith.maximumf %x, %z : f32
          memref.store %y, %out[%i] : memref<{chunk}xf32>
        }}
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{n}xf32>, %arg1: memref<{n}xf32>, %arg2: memref<{n}xf32>) {{
      %t0 = aiex.dma_configure_task_for @in0 {{
        aie.dma_bd(%arg0 : memref<{n}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%t0)
      %t2 = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{n}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t0)
    }}
  }}
}}
"#
    )
}

/// AIE2 vector width for bf16 — 512-bit registers hold 32 bf16 lanes (twice the
/// i32 width), and bf16 is the AIE2's **native vectorized float type** (the f32
/// FPU is scalar-only). So a vectorized float activation runs in bf16.
const VEC_BF16: usize = 32;

/// Emit AIE-MLIR for a 1-D `n`-element **bf16** ReLU on one vectorized compute
/// tile: `out = max(0.0, in)` over `vector<32xbf16>`. This is the *fast* float
/// activation path on AIE2 — unlike f32 (whose vector ops `aievec` rejects),
/// bf16 vector max lowers natively, so this vectorizes 32-wide. Host f32↔bf16
/// cast happens at the I/O boundary (see [`crate::npu_gemm::NpuIoBf16`]).
/// Requires `n % chunk == 0`, `chunk % VEC_BF16 == 0`.
pub fn emit_relu_bf16(n: usize, chunk: usize) -> String {
    assert!(chunk > 0 && n % chunk == 0, "n ({n}) must be a multiple of chunk ({chunk})");
    assert!(
        chunk % VEC_BF16 == 0,
        "chunk ({chunk}) must be a multiple of the bf16 vector width ({VEC_BF16})"
    );
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_in = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim_in, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{chunk}xbf16>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{chunk}xbf16>>
    %0 = aie.core(%core) {{
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %cmax step %c1 {{
        %in_sv = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<{chunk}xbf16>>
        %in = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<{chunk}xbf16>> -> memref<{chunk}xbf16>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{chunk}xbf16>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{chunk}xbf16>> -> memref<{chunk}xbf16>
        %lo = arith.constant 0 : index
        %hi = arith.constant {chunk} : index
        %st = arith.constant {VEC_BF16} : index
        scf.for %i = %lo to %hi step %st {{
          %v0 = vector.load %in[%i] : memref<{chunk}xbf16>, vector<{VEC_BF16}xbf16>
          %vz = arith.constant dense<0.0> : vector<{VEC_BF16}xbf16>
          %v1 = arith.maximumf %v0, %vz : vector<{VEC_BF16}xbf16>
          vector.store %v1, %out[%i] : memref<{chunk}xbf16>, vector<{VEC_BF16}xbf16>
        }}
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{n}xbf16>, %arg1: memref<{n}xbf16>, %arg2: memref<{n}xbf16>) {{
      %t0 = aiex.dma_configure_task_for @in0 {{
        aie.dma_bd(%arg0 : memref<{n}xbf16>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%t0)
      %t2 = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{n}xbf16>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t0)
    }}
  }}
}}
"#
    )
}

/// Emit AIE-MLIR for a `cols`-column elementwise `ops` chain — the data is split
/// evenly across `cols` independent compute tiles (one per AIE column), each a
/// vectorized ShimNOC→CoreTile→ShimNOC pipeline processing `n/cols` elements in
/// `chunk`-sized tiles. Embarrassingly parallel, so throughput scales ~`cols`×
/// over the single-tile [`emit_eltwise_chain`]. Requires `n % cols == 0`,
/// `(n/cols) % chunk == 0`, `chunk % VEC == 0`.
pub fn emit_eltwise_multicol(n: usize, chunk: usize, cols: usize, ops: &[Eltwise]) -> String {
    assert!(cols >= 1 && n % cols == 0, "n ({n}) must be a multiple of cols ({cols})");
    let per_col = n / cols;
    assert!(per_col % chunk == 0, "n/cols ({per_col}) must be a multiple of chunk ({chunk})");
    assert!(chunk % VEC == 0, "chunk ({chunk}) must be a multiple of {VEC}");
    assert!(!ops.is_empty());

    // Shared per-element (vectorized) op chain.
    let mut expr = String::new();
    let mut cur = "%v0".to_string();
    for (k, op) in ops.iter().enumerate() {
        let next = format!("%v{}", k + 1);
        expr += &op.core_expr_named(&cur, &next, k);
        cur = next;
    }
    let store_ssa = cur;

    let mut tiles = String::new();
    let mut fifos = String::new();
    let mut cores = String::new();
    let mut rt = String::new();
    let mut awaits = String::new();
    for c in 0..cols {
        tiles += &format!("    %core{c} = aie.logical_tile<CoreTile>(?, ?)\n");
        tiles += &format!("    %shim{c} = aie.logical_tile<ShimNOCTile>(?, ?)\n");
        fifos += &format!("    aie.objectfifo @in{c}(%shim{c}, {{%core{c}}}, 2 : i32) : !aie.objectfifo<memref<{chunk}xi32>>\n");
        fifos += &format!("    aie.objectfifo @out{c}(%core{c}, {{%shim{c}}}, 2 : i32) : !aie.objectfifo<memref<{chunk}xi32>>\n");
        cores += &format!(
            r#"    %cb{c} = aie.core(%core{c}) {{
      %z = arith.constant 0 : index
      %mx = arith.constant 9223372036854775807 : index
      %o = arith.constant 1 : index
      scf.for %it = %z to %mx step %o {{
        %isv = aie.objectfifo.acquire @in{c}(Consume, 1) : !aie.objectfifosubview<memref<{chunk}xi32>>
        %in = aie.objectfifo.subview.access %isv[0] : !aie.objectfifosubview<memref<{chunk}xi32>> -> memref<{chunk}xi32>
        %osv = aie.objectfifo.acquire @out{c}(Produce, 1) : !aie.objectfifosubview<memref<{chunk}xi32>>
        %out = aie.objectfifo.subview.access %osv[0] : !aie.objectfifosubview<memref<{chunk}xi32>> -> memref<{chunk}xi32>
        %lo = arith.constant 0 : index
        %hi = arith.constant {chunk} : index
        %st = arith.constant {VEC} : index
        scf.for %i = %lo to %hi step %st {{
          %v0 = vector.load %in[%i] : memref<{chunk}xi32>, vector<{VEC}xi32>
{expr}          vector.store {store_ssa}, %out[%i] : memref<{chunk}xi32>, vector<{VEC}xi32>
        }}
        aie.objectfifo.release @in{c}(Consume, 1)
        aie.objectfifo.release @out{c}(Produce, 1)
      }}
      aie.end
    }}
"#
        );
        let off = c * per_col;
        rt += &format!(
            r#"      %ti{c} = aiex.dma_configure_task_for @in{c} {{
        aie.dma_bd(%arg0 : memref<{n}xi32>, {off}, {per_col}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {per_col}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ti{c})
      %to{c} = aiex.dma_configure_task_for @out{c} {{
        aie.dma_bd(%arg2 : memref<{n}xi32>, {off}, {per_col}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {per_col}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to{c})
"#
        );
        awaits += &format!("      aiex.dma_await_task(%to{c})\n");
        awaits += &format!("      aiex.dma_free_task(%ti{c})\n");
    }

    // Device name: npu1_{1,2,3}col are named partitions; 4 columns = the full
    // `npu1` device (there is no `npu1_4col` in the dialect enum).
    let device = if cols <= 3 {
        format!("npu1_{cols}col")
    } else {
        "npu1".to_string()
    };
    format!(
        "module {{\n  aie.device({device}) {{\n{tiles}{fifos}{cores}    aie.runtime_sequence(%arg0: memref<{n}xi32>, %arg1: memref<{n}xi32>, %arg2: memref<{n}xi32>) {{\n{rt}{awaits}    }}\n  }}\n}}\n"
    )
}

/// Emit AIE-MLIR for a single-tile **int8** matmul `C[m,n] = A[m,k] · B[k,n]`
/// (i8 A/B → i32 C), all operands streamed as one tile each (so the elements must
/// fit the core's local memory — keep dims small, e.g. 32³/64³). The core is a
/// scalar MAC; the vectorized AIE2 MAC (`aievec.matmul`) is blocked in pure MLIR
/// by the accumulator-drain (`aievec.srs`) not lowering through Peano — peak-perf
/// matmul needs the C++ `aie::mmul` microkernel (see the core-body note). i8 I/O
/// matches the `RlxXdnaGemm` host path (`NpuGemm`); runtime ABI is the GEMM 3-arg
/// one (`arg0`=A, `arg1`=B, `arg2`=C). Requires `m % 4 == 0`, `k`/`n` `% 8 == 0`.
pub fn emit_matmul(m: usize, k: usize, n: usize) -> String {
    assert!(m % 4 == 0, "matmul m ({m}) must be a multiple of 4 (AIE2 MAC tile)");
    assert!(k % 8 == 0, "matmul k ({k}) must be a multiple of 8 (AIE2 MAC tile)");
    assert!(n % 8 == 0, "matmul n ({n}) must be a multiple of 8 (AIE2 MAC tile)");
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_a = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_b = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_c = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @a0(%shim_a, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{m}x{k}xi8>>
    aie.objectfifo @b0(%shim_b, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{k}x{n}xi8>>
    aie.objectfifo @c0(%core, {{%shim_c}}, 2 : i32) : !aie.objectfifo<memref<{m}x{n}xi32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      scf.for %iter = %z0 to %zmax step %one {{
        %a_sv = aie.objectfifo.acquire @a0(Consume, 1) : !aie.objectfifosubview<memref<{m}x{k}xi8>>
        %A = aie.objectfifo.subview.access %a_sv[0] : !aie.objectfifosubview<memref<{m}x{k}xi8>> -> memref<{m}x{k}xi8>
        %b_sv = aie.objectfifo.acquire @b0(Consume, 1) : !aie.objectfifosubview<memref<{k}x{n}xi8>>
        %B = aie.objectfifo.subview.access %b_sv[0] : !aie.objectfifosubview<memref<{k}x{n}xi8>> -> memref<{k}x{n}xi8>
        %c_sv = aie.objectfifo.acquire @c0(Produce, 1) : !aie.objectfifosubview<memref<{m}x{n}xi32>>
        %C = aie.objectfifo.subview.access %c_sv[0] : !aie.objectfifosubview<memref<{m}x{n}xi32>> -> memref<{m}x{n}xi32>
        %m0 = arith.constant 0 : index
        %mM = arith.constant {m} : index
        %n0 = arith.constant 0 : index
        %nN = arith.constant {n} : index
        %k0 = arith.constant 0 : index
        %kK = arith.constant {k} : index
        // Scalar i8 core (loads i8 A/B, sign-extends to i32, MACs). The VECTORIZED
        // AIE2 int8 MAC path — `vector.contract` (4x8x8 tile) → `aievec.matmul` —
        // lowers cleanly through aie-opt, but the accumulator→memory path does NOT
        // lower through Peano: the matmul result lives in a dedicated AIE2
        // accumulator register whose only legal drain is `aievec.srs`, and every
        // generic route (2D `transfer_write`, per-row `vector.extract`) hits a
        // `G_UNMERGE_VALUES <32 x s32>` / illegal 2D `aievec.srs`. This accumulator
        // bookkeeping is exactly what the C++ `aie::mmul` intrinsics encapsulate —
        // which is why every mlir-aie matmul example ships a `.cc` microkernel
        // rather than emitting pure MLIR. So peak-perf matmul is the C++-microkernel
        // route (external kernel + pre-tiled DMA); the i8 I/O + NpuGemm host path
        // here is kept ready for it. Requires m%4==0, k%8==0, n%8==0.
        scf.for %mi = %m0 to %mM step %one {{
          scf.for %ni = %n0 to %nN step %one {{
            %zero = arith.constant 0 : i32
            %acc = scf.for %ki = %k0 to %kK step %one iter_args(%s = %zero) -> (i32) {{
              %a8 = memref.load %A[%mi, %ki] : memref<{m}x{k}xi8>
              %av = arith.extsi %a8 : i8 to i32
              %b8 = memref.load %B[%ki, %ni] : memref<{k}x{n}xi8>
              %bv = arith.extsi %b8 : i8 to i32
              %p = arith.muli %av, %bv : i32
              %s2 = arith.addi %s, %p : i32
              scf.yield %s2 : i32
            }}
            memref.store %acc, %C[%mi, %ni] : memref<{m}x{n}xi32>
          }}
        }}
        aie.objectfifo.release @a0(Consume, 1)
        aie.objectfifo.release @b0(Consume, 1)
        aie.objectfifo.release @c0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{m}x{k}xi8>, %arg1: memref<{k}x{n}xi8>, %arg2: memref<{m}x{n}xi32>) {{
      %ta = aiex.dma_configure_task_for @a0 {{
        aie.dma_bd(%arg0 : memref<{m}x{k}xi8>, 0, {mk}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {mk}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta)
      %tb = aiex.dma_configure_task_for @b0 {{
        aie.dma_bd(%arg1 : memref<{k}x{n}xi8>, 0, {kn}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {kn}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tb)
      %tc = aiex.dma_configure_task_for @c0 {{
        aie.dma_bd(%arg2 : memref<{m}x{n}xi32>, 0, {mn}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {mn}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tc)
      aiex.dma_await_task(%tc)
      aiex.dma_free_task(%ta)
      aiex.dma_free_task(%tb)
    }}
  }}
}}
"#,
        mk = m * k,
        kn = k * n,
        mn = m * n,
    )
}

/// Pure-Rust vectorized int8 matmul via `vector.contract` → `aievec.matmul` on the
/// AIE2 hardware MAC, **no C++**. Numerically correct (bit-exact vs a CPU i32
/// reference on the amd rig) when driven by the [`tile_a`] / [`tile_b`] /
/// [`untile_c`] / [`matmul_signed_fixup`] host helpers — validated to ~64³.
///
/// SUPERSEDED for production by [`emit_matmul_microkernel`] (the vendor `aie::mmul`
/// overlay), which scales past 64³ and is what `Device::Xdna` runs; this pure-MLIR
/// path is kept as the C++-free reference and for the tiling examples.
///
/// Three tricks make this work end-to-end where the naive version couldn't:
/// (1) the K reduction is **unrolled** into an SSA chain of contracts (a
/// loop-carried 2-D accumulator hits `G_UNMERGE_VALUES`, a dataflow chain is
/// fine); (2) A/B/C are **tile-contiguous**, so the accumulator drains as a
/// `shape_cast`-to-1-D contiguous `vector.store` (a strided 2-D store also
/// unmerges) — this cleared the earlier `srs`/`G_UNMERGE` compile walls;
/// (3) **signedness**: the mlir-aie contract→`aievec.matmul` lowering hard-codes
/// the accumulator's *first* operand as **unsigned** (CONF sgn_x=0) and only the
/// second as signed (sgn_y=1) — signed A would read `-1` as `255`. Rather than
/// fight the pass (feeding `arith.extsi` on the lhs just gets rewritten to
/// `ups`+`cast`, and the whole contract then fails to match the matmul pattern),
/// [`tile_a`] biases A by +128 into the unsigned range and [`matmul_signed_fixup`]
/// removes the resulting `128·Σ_k B` term on the host — the classic u8·s8 GEMM
/// trick, entirely in pure Rust. So this is a correct, fast, C++-free int8 matmul
/// on the NPU. Requires m%4, k%8, n%8 == 0.
///
/// SCALING: A, B, and C are all **tile-resident** (single objectfifo delivery), so
/// the whole problem must fit the 64 KB AIE tile local memory (C at i32 dominates:
/// `4·m·n` bytes, double-buffered) — validated to ~64³; larger dims aiecc-fail at
/// buffer allocation (a clean compile error, never wrong output). Streaming C out
/// per output-row-block (to scale past the tile) was attempted but hit a
/// size-dependent aiecc **output-objectfifo chunking** bug: with C chunked, results
/// are correct at nt∈{4,12,16} (32³/96³/128³) but deterministically wrong at
/// nt∈{6,8} (48³/64³) — a whole output-tile lands with the wrong dot-products,
/// independent of loop-bound/held-vs-streamed-A. Parked pending a look at the
/// generated shim DMA descriptors (compare a good vs bad nt); kept all-resident so
/// the kernel is never silently wrong.
pub fn emit_matmul_tiled(m: usize, k: usize, n: usize) -> String {
    assert!(m % 4 == 0 && k % 8 == 0 && n % 8 == 0, "matmul dims must tile 4×8×8");
    let (mt, kt, nt) = (m / 4, k / 8, n / 8);
    let (na, nb, nc) = (m * k, k * n, m * n);
    let cmap = "{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [\"parallel\", \"parallel\", \"reduction\"], kind = #vector.kind<add>}";
    // Unrolled K: a chain of contracts feeding the accumulator (SSA, not loop-carried).
    let mut chain = String::from("            %czero = arith.constant dense<0> : vector<4x8xi32>\n");
    let mut prev = "%czero".to_string();
    for ki in 0..kt {
        let (aoff, boff) = (ki * 32, ki * nt * 64);
        let acc = format!("%acc{ki}");
        chain += &format!(
            "            %aoc{ki} = arith.constant {aoff} : index\n            %aof{ki} = arith.addi %mikt32, %aoc{ki} : index\n            %av{ki} = vector.load %A[%aof{ki}] : memref<{na}xi8>, vector<32xi8>\n            %a2{ki} = vector.shape_cast %av{ki} : vector<32xi8> to vector<4x8xi8>\n            %boc{ki} = arith.constant {boff} : index\n            %bof{ki} = arith.addi %ni64, %boc{ki} : index\n            %bv{ki} = vector.load %B[%bof{ki}] : memref<{nb}xi8>, vector<64xi8>\n            %b2{ki} = vector.shape_cast %bv{ki} : vector<64xi8> to vector<8x8xi8>\n            {acc} = vector.contract {cmap} %a2{ki}, %b2{ki}, {prev} : vector<4x8xi8>, vector<8x8xi8> into vector<4x8xi32>\n"
        );
        prev = acc;
    }
    let (kt32, nt32) = (kt * 32, nt * 32);
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %sa = aie.logical_tile<ShimNOCTile>(?, ?)
    %sb = aie.logical_tile<ShimNOCTile>(?, ?)
    %sc = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @a0(%sa, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{na}xi8>>
    aie.objectfifo @b0(%sb, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{nb}xi8>>
    aie.objectfifo @c0(%core, {{%sc}}, 2 : i32) : !aie.objectfifo<memref<{nc}xi32>>
    %0 = aie.core(%core) {{
      %z = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %mtN = arith.constant {mt} : index
      %ntN = arith.constant {nt} : index
      %kt32c = arith.constant {kt32} : index
      %nt32c = arith.constant {nt32} : index
      %c64c = arith.constant 64 : index
      %c32c = arith.constant 32 : index
      scf.for %it = %z to %zmax step %one {{
        %as = aie.objectfifo.acquire @a0(Consume, 1) : !aie.objectfifosubview<memref<{na}xi8>>
        %A = aie.objectfifo.subview.access %as[0] : !aie.objectfifosubview<memref<{na}xi8>> -> memref<{na}xi8>
        %bs = aie.objectfifo.acquire @b0(Consume, 1) : !aie.objectfifosubview<memref<{nb}xi8>>
        %B = aie.objectfifo.subview.access %bs[0] : !aie.objectfifosubview<memref<{nb}xi8>> -> memref<{nb}xi8>
        %cs = aie.objectfifo.acquire @c0(Produce, 1) : !aie.objectfifosubview<memref<{nc}xi32>>
        %C = aie.objectfifo.subview.access %cs[0] : !aie.objectfifosubview<memref<{nc}xi32>> -> memref<{nc}xi32>
        scf.for %mi = %z to %mtN step %one {{
          %mikt32 = arith.muli %mi, %kt32c : index
          %mint32 = arith.muli %mi, %nt32c : index
          scf.for %ni = %z to %ntN step %one {{
            %ni64 = arith.muli %ni, %c64c : index
            %ni32 = arith.muli %ni, %c32c : index
{chain}            %cof = arith.addi %mint32, %ni32 : index
            %c1d = vector.shape_cast {prev} : vector<4x8xi32> to vector<32xi32>
            vector.store %c1d, %C[%cof] : memref<{nc}xi32>, vector<32xi32>
          }}
        }}
        aie.objectfifo.release @a0(Consume, 1)
        aie.objectfifo.release @b0(Consume, 1)
        aie.objectfifo.release @c0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{na}xi8>, %arg1: memref<{nb}xi8>, %arg2: memref<{nc}xi32>) {{
      %ta = aiex.dma_configure_task_for @a0 {{
        aie.dma_bd(%arg0 : memref<{na}xi8>, 0, {na}) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta)
      %tb = aiex.dma_configure_task_for @b0 {{
        aie.dma_bd(%arg1 : memref<{nb}xi8>, 0, {nb}) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tb)
      %tc = aiex.dma_configure_task_for @c0 {{
        aie.dma_bd(%arg2 : memref<{nc}xi32>, 0, {nc}) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tc)
      aiex.dma_await_task(%tc)
      aiex.dma_free_task(%ta)
      aiex.dma_free_task(%tb)
    }}
  }}
}}
"#
    )
}

/// Host-side re-tiling for [`emit_matmul_tiled`]: pack a row-major `A[m,k]` into
/// 4×8 tiles (tile-major, row-major within). **A is biased by +128 per element**:
/// the `aievec.matmul` intrinsic this path lowers to reads its *first* operand as
/// **unsigned** i8 (CONF sgn_x=0 — the mlir-aie lowering hard-defaults the lhs to
/// unsigned "activation", only the rhs is signed), so feeding raw signed A reads
/// `-1` as `255`. Biasing to `A+128 ∈ [0,255]` makes the unsigned read exact; the
/// `+128·Σ_k B` term it introduces is removed by [`matmul_signed_fixup`] on the
/// host. This is the standard u8·s8→s8·s8 GEMM trick (gemmlowp / QNNPACK).
pub fn tile_a(a: &[i8], m: usize, k: usize) -> Vec<i8> {
    let (mt, kt) = (m / 4, k / 8);
    let mut out = vec![0i8; m * k];
    for mi in 0..mt {
        for ki in 0..kt {
            let base = (mi * kt + ki) * 32;
            for r in 0..4 {
                for c in 0..8 {
                    let v = a[(mi * 4 + r) * k + (ki * 8 + c)];
                    // bias to unsigned: byte value v+128 ∈ [0,255], stored as i8
                    out[base + r * 8 + c] = ((v as i16 + 128) as u8) as i8;
                }
            }
        }
    }
    out
}

/// Undo the `+128` A-bias from [`tile_a`]: the NPU computed
/// `C'[i][j] = Σ_k (A[i][k]+128)·B[k][j] = C[i][j] + 128·Σ_k B[k][j]`, so subtract
/// `128·colsum_B[j]` (independent of the row `i`) to recover the true signed
/// `C = A·B`. `c` is row-major `[m,n]` (post-[`untile_c`]); `b` is row-major
/// `[k,n]`. O(k·n + m·n) — negligible beside the O(m·k·n) MAC.
pub fn matmul_signed_fixup(c: &mut [i32], b: &[i8], m: usize, k: usize, n: usize) {
    let mut colsum = vec![0i32; n];
    for l in 0..k {
        for j in 0..n {
            colsum[j] += b[l * n + j] as i32;
        }
    }
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] -= 128 * colsum[j];
        }
    }
}

/// Pack `B[k,n]` into 8×8 tiles (tile-major over `[kt, nt]`), **row-major within
/// each tile** (`tile[r][c] = B[ki·8+r][ni·8+c]`). Probe-confirmed: with A=I the
/// raw output equals B row-major, i.e. the intrinsic computes `C = A·B` directly
/// (no Bᵀ), and the k-reduction aligns A's contiguous-in-row k-axis with B's
/// row-stride k-axis.
pub fn tile_b(b: &[i8], k: usize, n: usize) -> Vec<i8> {
    let (kt, nt) = (k / 8, n / 8);
    let mut out = vec![0i8; k * n];
    for ki in 0..kt {
        for ni in 0..nt {
            let base = (ki * nt + ni) * 64;
            for r in 0..8 {
                for c in 0..8 {
                    out[base + r * 8 + c] = b[(ki * 8 + r) * n + (ni * 8 + c)];
                }
            }
        }
    }
    out
}

/// De-tile the 4×8 `C` tiles produced by [`emit_matmul_tiled`] back to row-major.
pub fn untile_c(ct: &[i32], m: usize, n: usize) -> Vec<i32> {
    let (mt, nt) = (m / 4, n / 8);
    let mut out = vec![0i32; m * n];
    for mi in 0..mt {
        for ni in 0..nt {
            let base = (mi * nt + ni) * 32;
            for r in 0..4 {
                for c in 0..8 {
                    out[(mi * 4 + r) * n + (ni * 8 + c)] = ct[base + r * 8 + c];
                }
            }
        }
    }
    out
}

/// MULTI-CORE vectorized int8 matmul: split the output **columns** `n` across
/// `cols` AIE columns, each core computing an all-resident `m×k×(n/cols)` tile via
/// the [`emit_matmul_tiled`] pattern. This (1) scales past the single-tile 64³ wall
/// (each core's C-slice is `m·(n/cols)·4` bytes) and (2) uses all 4 columns for
/// ~cols× throughput. Crucially it **sidesteps the Peano accumulator miscompile**
/// (task #23): that bug fires when Peano fully-unrolls the `ni` loop at
/// `nt = n/8 ∈ {6,8}` (too many live `ACC1024` regs) — keeping each core's
/// `n/cols/8 ≤ 4` keeps the accumulator count in range. Host feeds full A (biased,
/// via [`tile_a`], broadcast to every core), per-column-slice B (via
/// [`tile_b_multicol`]), gathers per-column C (via [`untile_c_multicol`]), then the
/// usual [`matmul_signed_fixup`]. Requires m%4==0, k%8==0, (n/cols)%8==0, cols≤4.
pub fn emit_matmul_multicol(m: usize, k: usize, n: usize, cols: usize) -> String {
    assert!(m % 4 == 0 && k % 8 == 0 && n % 8 == 0, "matmul dims must tile 4×8×8");
    assert!((1..=4).contains(&cols) && n % cols == 0, "n ({n}) must split across cols ({cols}), cols≤4");
    let nc = n / cols;
    assert!(nc % 8 == 0, "n/cols ({nc}) must be a multiple of 8");
    let (mt, kt, ntc) = (m / 4, k / 8, nc / 8);
    let (na, nbc, ncc) = (m * k, k * nc, m * nc);
    let cmap = "{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [\"parallel\", \"parallel\", \"reduction\"], kind = #vector.kind<add>}";
    // Per-core K-unrolled contract chain (n = nc): A from the full resident buffer,
    // B from this core's column-slice buffer.
    let mut chain = String::from("            %czero = arith.constant dense<0> : vector<4x8xi32>\n");
    let mut prev = "%czero".to_string();
    for ki in 0..kt {
        let (aoff, boff) = (ki * 32, ki * ntc * 64);
        let acc = format!("%acc{ki}");
        chain += &format!(
            "            %aoc{ki} = arith.constant {aoff} : index\n            %aof{ki} = arith.addi %mikt32, %aoc{ki} : index\n            %av{ki} = vector.load %A[%aof{ki}] : memref<{na}xi8>, vector<32xi8>\n            %a2{ki} = vector.shape_cast %av{ki} : vector<32xi8> to vector<4x8xi8>\n            %boc{ki} = arith.constant {boff} : index\n            %bof{ki} = arith.addi %ni64, %boc{ki} : index\n            %bv{ki} = vector.load %B[%bof{ki}] : memref<{nbc}xi8>, vector<64xi8>\n            %b2{ki} = vector.shape_cast %bv{ki} : vector<64xi8> to vector<8x8xi8>\n            {acc} = vector.contract {cmap} %a2{ki}, %b2{ki}, {prev} : vector<4x8xi8>, vector<8x8xi8> into vector<4x8xi32>\n"
        );
        prev = acc;
    }
    let (kt32, ntc32) = (kt * 32, ntc * 32);

    let mut tiles = String::new();
    let mut fifos = String::new();
    let mut cores = String::new();
    let mut rt = String::new();
    let mut awaits = String::new();
    for c in 0..cols {
        tiles += &format!("    %core{c} = aie.logical_tile<CoreTile>(?, ?)\n    %shim{c} = aie.logical_tile<ShimNOCTile>(?, ?)\n");
        // @a depth 1: A is read-only + delivered once per run, so it needs no
        // double-buffer — halving its (dominant, full-m·k) tile footprint.
        fifos += &format!("    aie.objectfifo @a{c}(%shim{c}, {{%core{c}}}, 1 : i32) : !aie.objectfifo<memref<{na}xi8>>\n");
        fifos += &format!("    aie.objectfifo @b{c}(%shim{c}, {{%core{c}}}, 2 : i32) : !aie.objectfifo<memref<{nbc}xi8>>\n");
        fifos += &format!("    aie.objectfifo @c{c}(%core{c}, {{%shim{c}}}, 2 : i32) : !aie.objectfifo<memref<{ncc}xi32>>\n");
        cores += &format!(
            r#"    %cb{c} = aie.core(%core{c}) {{
      %z = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %mtN = arith.constant {mt} : index
      %ntN = arith.constant {ntc} : index
      %kt32c = arith.constant {kt32} : index
      %nt32c = arith.constant {ntc32} : index
      %c64c = arith.constant 64 : index
      %c32c = arith.constant 32 : index
      scf.for %it = %z to %zmax step %one {{
        %as = aie.objectfifo.acquire @a{c}(Consume, 1) : !aie.objectfifosubview<memref<{na}xi8>>
        %A = aie.objectfifo.subview.access %as[0] : !aie.objectfifosubview<memref<{na}xi8>> -> memref<{na}xi8>
        %bs = aie.objectfifo.acquire @b{c}(Consume, 1) : !aie.objectfifosubview<memref<{nbc}xi8>>
        %B = aie.objectfifo.subview.access %bs[0] : !aie.objectfifosubview<memref<{nbc}xi8>> -> memref<{nbc}xi8>
        %cs = aie.objectfifo.acquire @c{c}(Produce, 1) : !aie.objectfifosubview<memref<{ncc}xi32>>
        %C = aie.objectfifo.subview.access %cs[0] : !aie.objectfifosubview<memref<{ncc}xi32>> -> memref<{ncc}xi32>
        scf.for %mi = %z to %mtN step %one {{
          %mikt32 = arith.muli %mi, %kt32c : index
          %mint32 = arith.muli %mi, %nt32c : index
          scf.for %ni = %z to %ntN step %one {{
            %ni64 = arith.muli %ni, %c64c : index
            %ni32 = arith.muli %ni, %c32c : index
{chain}            %cof = arith.addi %mint32, %ni32 : index
            %c1d = vector.shape_cast {prev} : vector<4x8xi32> to vector<32xi32>
            vector.store %c1d, %C[%cof] : memref<{ncc}xi32>, vector<32xi32>
          }}
        }}
        aie.objectfifo.release @a{c}(Consume, 1)
        aie.objectfifo.release @b{c}(Consume, 1)
        aie.objectfifo.release @c{c}(Produce, 1)
      }}
      aie.end
    }}
"#
        );
        let (boff, coff) = (c * nbc, c * ncc);
        rt += &format!(
            r#"      %ta{c} = aiex.dma_configure_task_for @a{c} {{
        aie.dma_bd(%arg0 : memref<{na}xi8>, 0, {na}) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta{c})
      %tb{c} = aiex.dma_configure_task_for @b{c} {{
        aie.dma_bd(%arg1 : memref<{nb}xi8>, {boff}, {nbc}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {nbc}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tb{c})
      %tc{c} = aiex.dma_configure_task_for @c{c} {{
        aie.dma_bd(%arg2 : memref<{ncf}xi32>, {coff}, {ncc}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {ncc}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tc{c})
"#,
            nb = k * n,
            ncf = m * n,
        );
        awaits += &format!("      aiex.dma_await_task(%tc{c})\n      aiex.dma_free_task(%ta{c})\n      aiex.dma_free_task(%tb{c})\n");
    }
    let device = if cols <= 3 { format!("npu1_{cols}col") } else { "npu1".to_string() };
    let (nb, ncf) = (k * n, m * n);
    format!(
        "module {{\n  aie.device({device}) {{\n{tiles}{fifos}{cores}    aie.runtime_sequence(%arg0: memref<{na}xi8>, %arg1: memref<{nb}xi8>, %arg2: memref<{ncf}xi32>) {{\n{rt}{awaits}    }}\n  }}\n}}\n"
    )
}

/// Host-side B packing for [`emit_matmul_multicol`]: for each of `cols` column
/// slices `B[:, c·nc : (c+1)·nc]` (nc = n/cols), pack it into 8×8 tiles via the
/// same layout as [`tile_b`], and concatenate the `cols` blocks (block `c` is what
/// core `c` consumes). Result length = k·n.
pub fn tile_b_multicol(b: &[i8], k: usize, n: usize, cols: usize) -> Vec<i8> {
    let nc = n / cols;
    let (kt, ntc) = (k / 8, nc / 8);
    let mut out = vec![0i8; k * n];
    for c in 0..cols {
        let base_c = c * k * nc;
        for ki in 0..kt {
            for ni in 0..ntc {
                let base = base_c + (ki * ntc + ni) * 64;
                for r in 0..8 {
                    for cc in 0..8 {
                        // source column in the full B: c*nc + ni*8 + cc
                        out[base + r * 8 + cc] = b[(ki * 8 + r) * n + (c * nc + ni * 8 + cc)];
                    }
                }
            }
        }
    }
    out
}

/// Host-side C gather for [`emit_matmul_multicol`]: the `cols` per-core C blocks
/// (each `m×nc` in 4×8 tile order, concatenated) are de-tiled and scattered back to
/// the full row-major `C[m,n]` (core `c` owns columns `c·nc … (c+1)·nc`).
pub fn untile_c_multicol(ct: &[i32], m: usize, n: usize, cols: usize) -> Vec<i32> {
    let nc = n / cols;
    let (mt, ntc) = (m / 4, nc / 8);
    let mut out = vec![0i32; m * n];
    for c in 0..cols {
        let base_c = c * m * nc;
        for mi in 0..mt {
            for ni in 0..ntc {
                let base = base_c + (mi * ntc + ni) * 32;
                for r in 0..4 {
                    for cc in 0..8 {
                        out[(mi * 4 + r) * n + (c * nc + ni * 8 + cc)] = ct[base + r * 8 + cc];
                    }
                }
            }
        }
    }
    out
}

/// The **C++ `aie::mmul` microkernel matmul** — the reliable, fast NPU matmul path
/// (~638 GOP/s at d=64, kt=128, cols=4), wired into `Device::Xdna` as the default
/// `Op::MatMul`. Emits an AIE overlay that calls the vendor `matmul_i8_i32` +
/// `zero_i32` kernel (from mlir-aie's `aie_kernels/aie2/mm.cc`, Peano-compiled to
/// `obj` and linked via `link_with` — see [`crate::compile::build_mm_kernel`] +
/// [`crate::compile::compile_overlay_linked`]). Uses hand-written intrinsics, so it
/// SIDESTEPS the Peano aievec-vectorizer fragility (tasks #23/#24) that makes the
/// pure-MLIR [`emit_matmul_tiled`] unreliable, and is bit-exact at any shape.
///
/// Computes `d × (kt·d) × (d·cols)` in one dispatch by combining two levers:
/// - **`cols` AIE columns** — each core owns an output column slice (N split), ~cols× throughput.
/// - **on-core K-accumulation** — each core streams `kt` K-tiles through the kernel,
///   accumulating into ONE resident `d×d` C tile (`zero_i32` once, then `kt`
///   accumulating `matmul_i8_i32` calls). K-tiles stream, so `kt` scales without
///   costing tile memory — bigger `kt·cols` amortizes the ~90µs dispatch floor.
///
/// Signed int8 is handled natively (no +128 bias / fixup). Host tiling:
/// [`tile_a_kacc`] (A, shared), [`tile_b_kacc_multicol`] (B), [`untile_c_multicol`]
/// (C). `obj` = kernel `.o` basename compiled for DIM=`d`. cols≤4;
/// `kt=cols=1` degenerates to a single `d³` tile.
pub fn emit_matmul_microkernel(d: usize, kt: usize, cols: usize, obj: &str) -> String {
    assert!((1..=4).contains(&cols), "cols must be 1..=4");
    let dd = d * d;
    let ka = kt * dd; // per-core A / B stream length (kt tiles)
    let ncf = dd * cols; // full C (and per-core-B block base uses ka)
    let nb = ka * cols; // full B stream length
    let mut tiles = String::new();
    let mut fifos = String::new();
    let mut cores = String::new();
    let mut rt = String::new();
    let mut awaits = String::new();
    for c in 0..cols {
        tiles += &format!("    %core{c} = aie.logical_tile<CoreTile>(?, ?)\n    %shim{c} = aie.logical_tile<ShimNOCTile>(?, ?)\n");
        fifos += &format!("    aie.objectfifo @a{c}(%shim{c}, {{%core{c}}}, 2 : i32) : !aie.objectfifo<memref<{d}x{d}xi8>>\n");
        fifos += &format!("    aie.objectfifo @b{c}(%shim{c}, {{%core{c}}}, 2 : i32) : !aie.objectfifo<memref<{d}x{d}xi8>>\n");
        fifos += &format!("    aie.objectfifo @c{c}(%core{c}, {{%shim{c}}}, 2 : i32) : !aie.objectfifo<memref<{d}x{d}xi32>>\n");
        cores += &format!(
            r#"    %cb{c} = aie.core(%core{c}) {{
      %z = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %ktN = arith.constant {kt} : index
      scf.for %it = %z to %zmax step %one {{
        %cs = aie.objectfifo.acquire @c{c}(Produce, 1) : !aie.objectfifosubview<memref<{d}x{d}xi32>>
        %C = aie.objectfifo.subview.access %cs[0] : !aie.objectfifosubview<memref<{d}x{d}xi32>> -> memref<{d}x{d}xi32>
        %Cf = memref.collapse_shape %C [[0, 1]] : memref<{d}x{d}xi32> into memref<{dd}xi32>
        func.call @zero_i32(%Cf) : (memref<{dd}xi32>) -> ()
        scf.for %ki = %z to %ktN step %one {{
          %as = aie.objectfifo.acquire @a{c}(Consume, 1) : !aie.objectfifosubview<memref<{d}x{d}xi8>>
          %A = aie.objectfifo.subview.access %as[0] : !aie.objectfifosubview<memref<{d}x{d}xi8>> -> memref<{d}x{d}xi8>
          %bs = aie.objectfifo.acquire @b{c}(Consume, 1) : !aie.objectfifosubview<memref<{d}x{d}xi8>>
          %B = aie.objectfifo.subview.access %bs[0] : !aie.objectfifosubview<memref<{d}x{d}xi8>> -> memref<{d}x{d}xi8>
          %Af = memref.collapse_shape %A [[0, 1]] : memref<{d}x{d}xi8> into memref<{dd}xi8>
          %Bf = memref.collapse_shape %B [[0, 1]] : memref<{d}x{d}xi8> into memref<{dd}xi8>
          func.call @matmul_i8_i32(%Af, %Bf, %Cf) : (memref<{dd}xi8>, memref<{dd}xi8>, memref<{dd}xi32>) -> ()
          aie.objectfifo.release @a{c}(Consume, 1)
          aie.objectfifo.release @b{c}(Consume, 1)
        }}
        aie.objectfifo.release @c{c}(Produce, 1)
      }}
      aie.end
    }}
"#
        );
        let (boff, coff) = (c * ka, c * dd);
        rt += &format!(
            r#"      %ta{c} = aiex.dma_configure_task_for @a{c} {{
        aie.dma_bd(%arg0 : memref<{ka}xi8>, 0, {ka}) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta{c})
      %tb{c} = aiex.dma_configure_task_for @b{c} {{
        aie.dma_bd(%arg1 : memref<{nb}xi8>, {boff}, {ka}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {ka}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tb{c})
      %tc{c} = aiex.dma_configure_task_for @c{c} {{
        aie.dma_bd(%arg2 : memref<{ncf}xi32>, {coff}, {dd}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {dd}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tc{c})
"#
        );
        awaits += &format!("      aiex.dma_await_task(%tc{c})\n      aiex.dma_free_task(%ta{c})\n      aiex.dma_free_task(%tb{c})\n");
    }
    let device = if cols <= 3 { format!("npu1_{cols}col") } else { "npu1".to_string() };
    // arg0 = shared A (one `ka`-long stream, all cores read it); arg1 = per-core B
    // (nb = cols·ka); arg2 = gathered C (ncf = cols·dd).
    format!(
        "module {{\n  aie.device({device}) {{\n{tiles}    func.func private @zero_i32(memref<{dd}xi32>) attributes {{link_with = \"{obj}\"}}\n    func.func private @matmul_i8_i32(memref<{dd}xi8>, memref<{dd}xi8>, memref<{dd}xi32>) attributes {{link_with = \"{obj}\"}}\n{fifos}{cores}    aie.runtime_sequence(%arg0: memref<{ka}xi8>, %arg1: memref<{nb}xi8>, %arg2: memref<{ncf}xi32>) {{\n{rt}{awaits}    }}\n  }}\n}}\n"
    )
}

/// Host A packing for [`emit_matmul_microkernel`]: `A[d, kt·d]` → `kt` `d×d` K-tiles
/// in ki order, each packed into 4×8 subtiles (no `+128` bias — the kernel is
/// signed-native) and concatenated. A is shared across cores (each streams the same
/// tiles). Returns the per-core stream (length `kt·d·d`).
pub fn tile_a_kacc(a: &[i8], d: usize, kt: usize) -> Vec<i8> {
    let big_k = kt * d;
    let mut out = Vec::with_capacity(kt * d * d);
    for ki in 0..kt {
        // extract the d×d K-slice A[:, ki*d..(ki+1)*d]
        let mut slice = vec![0i8; d * d];
        for r in 0..d {
            for c in 0..d {
                slice[r * d + c] = a[r * big_k + ki * d + c];
            }
        }
        out.extend_from_slice(&tile_a_raw(&slice, d, d));
    }
    out
}

/// Host B packing for [`emit_matmul_microkernel`]: `B[kt·d, d·cols]` → per core `c`
/// (columns `c·d…`), `kt` `d×d` K-tiles in ki order, each [`tile_b`]-tiled; laid out
/// `[c][ki][tile]`. Length `cols·kt·d·d`.
pub fn tile_b_kacc_multicol(b: &[i8], d: usize, kt: usize, cols: usize) -> Vec<i8> {
    let n = d * cols;
    let mut out = Vec::with_capacity(cols * kt * d * d);
    for c in 0..cols {
        for ki in 0..kt {
            // B[ki*d..(ki+1)*d, c*d..(c+1)*d]
            let mut slice = vec![0i8; d * d];
            for r in 0..d {
                for cc in 0..d {
                    slice[r * d + cc] = b[(ki * d + r) * n + (c * d + cc)];
                }
            }
            out.extend_from_slice(&tile_b(&slice, d, d));
        }
    }
    out
}

/// Pack one `m×k` int8 matrix into 4×8 subtiles (no `+128` bias — the `aie::mmul`
/// kernel is signed-native). Internal helper for [`tile_a_kacc`]; layout matches
/// [`tile_a`] minus the bias.
fn tile_a_raw(a: &[i8], m: usize, k: usize) -> Vec<i8> {
    let (mt, kt) = (m / 4, k / 8);
    let mut out = vec![0i8; m * k];
    for mi in 0..mt {
        for ki in 0..kt {
            let base = (mi * kt + ki) * 32;
            for r in 0..4 {
                for c in 0..8 {
                    out[base + r * 8 + c] = a[(mi * 4 + r) * k + (ki * 8 + c)];
                }
            }
        }
    }
    out
}

// ============================================================================
// Generalized elementwise op framework (pure-Rust AIE-MLIR, parity surface)
// ============================================================================

/// Element dtype an emitter targets on the AIE2 tile. Governs the MLIR element
/// type, the host buffer cell size, and — crucially — the vector width: the
/// AIE2 512-bit vector unit does 16 i32 or 32 bf16 lanes, but has **no f32
/// vector path** (aievec rejects f32 vector ops), so `F32` lowers scalar.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ty {
    I32,
    F32,
    Bf16,
}

impl Ty {
    pub fn mlir(&self) -> &'static str {
        match self {
            Ty::I32 => "i32",
            Ty::F32 => "f32",
            Ty::Bf16 => "bf16",
        }
    }
    /// Host BO cell size in bytes (bf16 = 2, i32/f32 = 4).
    pub fn bytes(&self) -> usize {
        match self {
            Ty::Bf16 => 2,
            _ => 4,
        }
    }
    /// Vectorized lane count on the 512-bit AIE2 vector unit; `F32` → 1 (scalar).
    pub fn lanes(&self) -> usize {
        match self {
            Ty::I32 => 16,
            Ty::Bf16 => 32,
            Ty::F32 => 1,
        }
    }
    pub fn is_float(&self) -> bool {
        matches!(self, Ty::F32 | Ty::Bf16)
    }
}

/// MLIR compute value type for element type `t` at `lanes` width (`vector<Lxt>`
/// when `lanes>1`, else scalar `t`).
fn v_type(t: &str, lanes: usize) -> String {
    if lanes > 1 {
        format!("vector<{lanes}x{t}>")
    } else {
        t.to_string()
    }
}

/// A per-step load of `dst` from `buf[%i]` — `vector.load` when `lanes>1`, else
/// scalar `memref.load`.
fn v_load(t: &str, lanes: usize, dst: &str, buf: &str, chunk: usize) -> String {
    if lanes > 1 {
        format!("          {dst} = vector.load {buf}[%i] : memref<{chunk}x{t}>, {}\n", v_type(t, lanes))
    } else {
        format!("          {dst} = memref.load {buf}[%i] : memref<{chunk}x{t}>\n")
    }
}

/// A per-step store of `src` into `buf[%i]`.
fn v_store(t: &str, lanes: usize, src: &str, buf: &str, chunk: usize) -> String {
    if lanes > 1 {
        format!("          vector.store {src}, {buf}[%i] : memref<{chunk}x{t}>, {}\n", v_type(t, lanes))
    } else {
        format!("          memref.store {src}, {buf}[%i] : memref<{chunk}x{t}>\n")
    }
}

/// A binary elementwise op `out = a ⊙ b` (mirrors `rlx_ir::BinaryOp`). Bitwise /
/// shift ops are integer-only. Emitted as one `arith` op over `Ty::vty()`, so
/// they vectorize for i32/bf16 and fall to scalar for f32 automatically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

impl BinaryOp {
    pub fn name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Max => "max",
            BinaryOp::Min => "min",
            BinaryOp::Mod => "mod",
            BinaryOp::BitAnd => "bitand",
            BinaryOp::BitOr => "bitor",
            BinaryOp::BitXor => "bitxor",
            BinaryOp::Shl => "shl",
            BinaryOp::Shr => "shr",
        }
    }
    pub fn is_bitwise(&self) -> bool {
        matches!(
            self,
            BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::Shl | BinaryOp::Shr
        )
    }
    /// Whether this op has a legal AIE2 **i32 vector** lowering. Verified on the
    /// rig: add/sub/max/min and bitwise-and/or vectorize; mul/div/mod, the shifts,
    /// and bitwise-xor do NOT (no 32-bit vector multiply/divide/shift, and `xori`
    /// gets neither an aievec form nor a Peano-legal raw-vector form) → they fall
    /// back to a scalar core (still correct, just 1-lane). Float (bf16) mul/div DO
    /// vectorize on the native bf16 FPU, so this gate is i32-specific.
    ///
    /// NOTE: `Shr` (arith.shrsi) is scalar here and additionally *deviates* on the
    /// AIE2 for negative operands — the hardware arithmetic right-shift rounds
    /// toward zero (`-125>>1 = -62`), not toward −∞ (`-63`). Fine for the usual
    /// non-negative bit-manipulation use; a caveat for signed data.
    fn i32_vectorizable(&self) -> bool {
        matches!(
            self,
            BinaryOp::Add
                | BinaryOp::Sub
                | BinaryOp::Max
                | BinaryOp::Min
                | BinaryOp::BitAnd
                | BinaryOp::BitOr
        )
    }
    /// The `arith` opcode for this op at dtype `ty`.
    fn opcode(&self, ty: Ty) -> &'static str {
        let f = ty.is_float();
        match (self, f) {
            (BinaryOp::Add, false) => "arith.addi",
            (BinaryOp::Add, true) => "arith.addf",
            (BinaryOp::Sub, false) => "arith.subi",
            (BinaryOp::Sub, true) => "arith.subf",
            (BinaryOp::Mul, false) => "arith.muli",
            (BinaryOp::Mul, true) => "arith.mulf",
            (BinaryOp::Div, false) => "arith.divsi",
            (BinaryOp::Div, true) => "arith.divf",
            (BinaryOp::Max, false) => "arith.maxsi",
            (BinaryOp::Max, true) => "arith.maximumf",
            (BinaryOp::Min, false) => "arith.minsi",
            (BinaryOp::Min, true) => "arith.minimumf",
            (BinaryOp::Mod, false) => "arith.remsi",
            (BinaryOp::Mod, true) => "arith.remf",
            (BinaryOp::BitAnd, _) => "arith.andi",
            (BinaryOp::BitOr, _) => "arith.ori",
            (BinaryOp::BitXor, _) => "arith.xori",
            (BinaryOp::Shl, _) => "arith.shli",
            (BinaryOp::Shr, _) => "arith.shrsi",
        }
    }
    /// Host reference (i32) for validation.
    pub fn apply_i32(&self, a: i32, b: i32) -> i32 {
        match self {
            BinaryOp::Add => a.wrapping_add(b),
            BinaryOp::Sub => a.wrapping_sub(b),
            BinaryOp::Mul => a.wrapping_mul(b),
            BinaryOp::Div => a.wrapping_div(b),
            BinaryOp::Max => a.max(b),
            BinaryOp::Min => a.min(b),
            BinaryOp::Mod => a.wrapping_rem(b),
            BinaryOp::BitAnd => a & b,
            BinaryOp::BitOr => a | b,
            BinaryOp::BitXor => a ^ b,
            BinaryOp::Shl => a.wrapping_shl(b as u32),
            BinaryOp::Shr => a.wrapping_shr(b as u32),
        }
    }
}

/// Emit AIE-MLIR for a 1-D `n`-element **binary** elementwise op `out = a ⊙ b`
/// on one vectorized compute tile. Two input ObjectFIFOs (A via `arg0`, B via
/// `arg1`) feed the core; the result streams out (`arg2`). This maps onto the
/// same 3-buffer `MLIR_AIE` ABI as the matmul (args 3/4/5 = A/B/out), so it runs
/// through `NpuIo::run2`. Requires `n % chunk == 0` and (vectorized dtypes)
/// `chunk % lanes == 0`.
pub fn emit_binary(op: BinaryOp, ty: Ty, n: usize, chunk: usize) -> String {
    assert!(chunk > 0 && n % chunk == 0, "n ({n}) must be a multiple of chunk ({chunk})");
    assert!(!(op.is_bitwise() && ty.is_float()), "bitwise ops are integer-only");
    // Effective vector width: fall back to scalar (1) for i32 ops with no vector
    // lowering, and always for f32 (aievec has no f32 vector path).
    let lanes = if ty == Ty::I32 && !op.i32_vectorizable() {
        1
    } else {
        ty.lanes()
    };
    assert!(chunk % lanes == 0, "chunk ({chunk}) must be a multiple of {lanes} lanes");
    let t = ty.mlir();
    let step = lanes;
    let expr = format!("          %vo = {} %va, %vb : {}\n", op.opcode(ty), v_type(t, lanes));
    let load_a = v_load(t, lanes, "%va", "%A", chunk);
    let load_b = v_load(t, lanes, "%vb", "%B", chunk);
    let store = v_store(t, lanes, "%vo", "%O", chunk);
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %sa = aie.logical_tile<ShimNOCTile>(?, ?)
    %sb = aie.logical_tile<ShimNOCTile>(?, ?)
    %so = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @a0(%sa, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{chunk}x{t}>>
    aie.objectfifo @b0(%sb, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{chunk}x{t}>>
    aie.objectfifo @o0(%core, {{%so}}, 2 : i32) : !aie.objectfifo<memref<{chunk}x{t}>>
    %0 = aie.core(%core) {{
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %iter = %c0 to %cmax step %c1 {{
        %a_sv = aie.objectfifo.acquire @a0(Consume, 1) : !aie.objectfifosubview<memref<{chunk}x{t}>>
        %A = aie.objectfifo.subview.access %a_sv[0] : !aie.objectfifosubview<memref<{chunk}x{t}>> -> memref<{chunk}x{t}>
        %b_sv = aie.objectfifo.acquire @b0(Consume, 1) : !aie.objectfifosubview<memref<{chunk}x{t}>>
        %B = aie.objectfifo.subview.access %b_sv[0] : !aie.objectfifosubview<memref<{chunk}x{t}>> -> memref<{chunk}x{t}>
        %o_sv = aie.objectfifo.acquire @o0(Produce, 1) : !aie.objectfifosubview<memref<{chunk}x{t}>>
        %O = aie.objectfifo.subview.access %o_sv[0] : !aie.objectfifosubview<memref<{chunk}x{t}>> -> memref<{chunk}x{t}>
        %lo = arith.constant 0 : index
        %hi = arith.constant {chunk} : index
        %st = arith.constant {step} : index
        scf.for %i = %lo to %hi step %st {{
{load_a}{load_b}{expr}{store}        }}
        aie.objectfifo.release @a0(Consume, 1)
        aie.objectfifo.release @b0(Consume, 1)
        aie.objectfifo.release @o0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{n}x{t}>, %arg1: memref<{n}x{t}>, %arg2: memref<{n}x{t}>) {{
      %ta = aiex.dma_configure_task_for @a0 {{
        aie.dma_bd(%arg0 : memref<{n}x{t}>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta)
      %tb = aiex.dma_configure_task_for @b0 {{
        aie.dma_bd(%arg1 : memref<{n}x{t}>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tb)
      %to = aiex.dma_configure_task_for @o0 {{
        aie.dma_bd(%arg2 : memref<{n}x{t}>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%ta)
      aiex.dma_free_task(%tb)
    }}
  }}
}}
"#
    )
}

/// A unary elementwise **activation** (subset of `rlx_ir::Activation`), emitted
/// over floats. Simple ops are one `arith`/`math` op; composites (sigmoid, silu,
/// gelu) are built from `math.exp`/`math.tanh` + `arith`. Whether the `math.*`
/// transcendentals lower on AIE2 (Peano) is what the sweep probes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Relu,
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Tanh,
    Recip,
    Sigmoid,
    Silu,
    Gelu,
    Floor,
    Ceil,
    Round,
    Sign,
    Softplus,
    Elu,
    HardSwish,
    HardSigmoid,
    Mish,
    Softsign,
    LogSigmoid,
    Sin,
    Cos,
    Erf,
}

/// Significant-digit count of an f32's shortest decimal (matches MLIR's printer).
fn sig_digits(x: f32) -> usize {
    let s = format!("{:e}", x); // e.g. "4.4715e-2"
    let mant = s.split('e').next().unwrap_or("");
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.trim_end_matches('0').len().max(1)
}

/// Materialize a scalar-f32 constant near `v` as `%{name}`, working around a
/// Peano-`opt` quirk: MLIR's LLVM printer emits an f32 whose shortest decimal is
/// short (≤7 sig digits, e.g. `4.471500e-02`) as a *decimal*, which Peano's
/// (older) `opt` rejects ("floating point constant invalid for type") since the
/// decimal isn't *exactly* the f32; values needing ≥8 digits print as hex, which
/// it accepts. So we nudge to the nearest f32 (searched outward in ULPs, <1e-6
/// relative — negligible) whose shortest decimal has ≥8 sig digits, forcing hex.
/// Emitted as integer bits + `bitcast`. `name` without the leading `%`.
fn fbits(name: &str, v: f32) -> String {
    // Values whose shortest decimal is *exactly* the f32 (integers, dyadic
    // fractions like ±1.0/0.5) print cleanly and MUST NOT be nudged — a nudge
    // would shift an exact bound (e.g. a clamp limit). Only inexact short
    // decimals need the ≥8-sig-digit nudge to force the accepted hex form.
    let exact = format!("{v}").parse::<f64>().map(|d| d == v as f64).unwrap_or(false);
    let base = v.to_bits();
    let mut bits = base;
    if !exact {
        for d in 0..256u32 {
            for cand in [base.wrapping_add(d), base.wrapping_sub(d)] {
                if sig_digits(f32::from_bits(cand)) >= 8 {
                    bits = cand;
                    break;
                }
            }
            if sig_digits(f32::from_bits(bits)) >= 8 {
                break;
            }
        }
    }
    format!(
        "          %{name}_i = arith.constant {} : i32\n          %{name} = arith.bitcast %{name}_i : i32 to f32\n",
        bits as i32
    )
}

/// Pure-`arith` scalar-f32 `exp` — the AIE2 `math.exp` doesn't lower, so rlx
/// emits the numerics itself: range-reduce `x = k·ln2 + r` (k via a round-trip
/// float→int→float floor), `2^k` by exponent-bit construction, and `exp(r)` for
/// `r∈[-0.35,0.35]` by a degree-5 Taylor Horner. `inp`→`outp`; internal SSA is
/// suffixed `_e` so it composes inside a larger body (used once per kernel).
fn approx_exp_f32(inp: &str, outp: &str) -> String {
    format!(
        "          %log2e_e = arith.constant 1.4426950408889634 : f32\n\
         \x20         %ln2_e = arith.constant 0.6931471805599453 : f32\n\
         \x20         %half_e = arith.constant 0.5 : f32\n\
         \x20         %one_e = arith.constant 1.0 : f32\n\
         \x20         %h5_e = arith.constant 0.008333333 : f32\n\
         \x20         %h4_e = arith.constant 0.041666668 : f32\n\
         \x20         %h3_e = arith.constant 0.16666667 : f32\n\
         \x20         %c127_e = arith.constant 127 : i32\n\
         \x20         %c23_e = arith.constant 23 : i32\n\
         \x20         %cm1_e = arith.constant -1 : i32\n\
         \x20         %c0_e = arith.constant 0 : i32\n\
         \x20         %t_e = arith.mulf {inp}, %log2e_e : f32\n\
         \x20         %th_e = arith.addf %t_e, %half_e : f32\n\
         \x20         %kt_e = arith.fptosi %th_e : f32 to i32\n\
         \x20         %ktf_e = arith.sitofp %kt_e : i32 to f32\n\
         \x20         %over_e = arith.cmpf ogt, %ktf_e, %th_e : f32\n\
         \x20         %adj_e = arith.select %over_e, %cm1_e, %c0_e : i32\n\
         \x20         %k_e = arith.addi %kt_e, %adj_e : i32\n\
         \x20         %kf_e = arith.sitofp %k_e : i32 to f32\n\
         \x20         %kln2_e = arith.mulf %kf_e, %ln2_e : f32\n\
         \x20         %r_e = arith.subf {inp}, %kln2_e : f32\n\
         \x20         %pa_e = arith.mulf %h5_e, %r_e : f32\n\
         \x20         %pa2_e = arith.addf %pa_e, %h4_e : f32\n\
         \x20         %pb_e = arith.mulf %pa2_e, %r_e : f32\n\
         \x20         %pb2_e = arith.addf %pb_e, %h3_e : f32\n\
         \x20         %pc_e = arith.mulf %pb2_e, %r_e : f32\n\
         \x20         %pc2_e = arith.addf %pc_e, %half_e : f32\n\
         \x20         %pd_e = arith.mulf %pc2_e, %r_e : f32\n\
         \x20         %pd2_e = arith.addf %pd_e, %one_e : f32\n\
         \x20         %pe_e = arith.mulf %pd2_e, %r_e : f32\n\
         \x20         %er_e = arith.addf %pe_e, %one_e : f32\n\
         \x20         %k127_e = arith.addi %k_e, %c127_e : i32\n\
         \x20         %ksh_e = arith.shli %k127_e, %c23_e : i32\n\
         \x20         %p2k_e = arith.bitcast %ksh_e : i32 to f32\n\
         \x20         {outp} = arith.mulf %p2k_e, %er_e : f32\n"
    )
}

/// Pure-`arith` scalar-f32 `rsqrt` (`1/√x`, x>0) via the fast-inverse-sqrt magic
/// constant + 3 Newton iterations (`r ← r·(1.5 − 0.5·x·r²)`) → ~1e-6. `inp`→
/// `outp`; internal SSA suffixed `_q`.
fn approx_rsqrt_f32(inp: &str, outp: &str) -> String {
    format!(
        "          %magic_q = arith.constant 1597463007 : i32\n\
         \x20         %c1_q = arith.constant 1 : i32\n\
         \x20         %half_q = arith.constant 0.5 : f32\n\
         \x20         %c15_q = arith.constant 1.5 : f32\n\
         \x20         %xi_q = arith.bitcast {inp} : f32 to i32\n\
         \x20         %sh_q = arith.shrsi %xi_q, %c1_q : i32\n\
         \x20         %yi_q = arith.subi %magic_q, %sh_q : i32\n\
         \x20         %r0_q = arith.bitcast %yi_q : i32 to f32\n\
         \x20         %xh_q = arith.mulf {inp}, %half_q : f32\n\
         \x20         %rr1_q = arith.mulf %r0_q, %r0_q : f32\n\
         \x20         %tt1_q = arith.mulf %xh_q, %rr1_q : f32\n\
         \x20         %ss1_q = arith.subf %c15_q, %tt1_q : f32\n\
         \x20         %r1_q = arith.mulf %r0_q, %ss1_q : f32\n\
         \x20         %rr2_q = arith.mulf %r1_q, %r1_q : f32\n\
         \x20         %tt2_q = arith.mulf %xh_q, %rr2_q : f32\n\
         \x20         %ss2_q = arith.subf %c15_q, %tt2_q : f32\n\
         \x20         %r2_q = arith.mulf %r1_q, %ss2_q : f32\n\
         \x20         %rr3_q = arith.mulf %r2_q, %r2_q : f32\n\
         \x20         %tt3_q = arith.mulf %xh_q, %rr3_q : f32\n\
         \x20         %ss3_q = arith.subf %c15_q, %tt3_q : f32\n\
         \x20         {outp} = arith.mulf %r2_q, %ss3_q : f32\n"
    )
}

/// Pure-`arith` scalar-f32 `log` (x>0): split `x = 2^e · m` by exponent-bit
/// extraction (`m∈[1,2)`), reduce `m` to `[1/√2,√2)`, then `log(m) = 2·atanh(s)`
/// with `s=(m−1)/(m+1)` via a degree-7 odd series (`s` is small → fast). Constants
/// are f32-exact doubles so Peano's opt accepts them. `inp`→`outp`; SSA suffix `_l`.
fn approx_log_f32(inp: &str, outp: &str) -> String {
    let consts = format!(
        "{}{}{}{}{}",
        fbits("sqrt2_l", std::f32::consts::SQRT_2),
        fbits("c17_l", 1.0 / 7.0),
        fbits("c15_l", 0.2),
        fbits("c13_l", 1.0 / 3.0),
        fbits("ln2_l", std::f32::consts::LN_2),
    );
    format!(
        "{consts}\
         \x20         %xi_l = arith.bitcast {inp} : f32 to i32\n\
         \x20         %c23_l = arith.constant 23 : i32\n\
         \x20         %eb_l = arith.shrui %xi_l, %c23_l : i32\n\
         \x20         %cff_l = arith.constant 255 : i32\n\
         \x20         %em_l = arith.andi %eb_l, %cff_l : i32\n\
         \x20         %c127_l = arith.constant 127 : i32\n\
         \x20         %e_l = arith.subi %em_l, %c127_l : i32\n\
         \x20         %cmant_l = arith.constant 8388607 : i32\n\
         \x20         %mb0_l = arith.andi %xi_l, %cmant_l : i32\n\
         \x20         %cone_l = arith.constant 1065353216 : i32\n\
         \x20         %mb_l = arith.ori %mb0_l, %cone_l : i32\n\
         \x20         %m_l = arith.bitcast %mb_l : i32 to f32\n\
         \x20         %big_l = arith.cmpf ogt, %m_l, %sqrt2_l : f32\n\
         \x20         %chalf_l = arith.constant 0.5 : f32\n\
         \x20         %mhalf_l = arith.mulf %m_l, %chalf_l : f32\n\
         \x20         %m2_l = arith.select %big_l, %mhalf_l, %m_l : f32\n\
         \x20         %c1i_l = arith.constant 1 : i32\n\
         \x20         %e1_l = arith.addi %e_l, %c1i_l : i32\n\
         \x20         %e2_l = arith.select %big_l, %e1_l, %e_l : i32\n\
         \x20         %onef_l = arith.constant 1.0 : f32\n\
         \x20         %num_l = arith.subf %m2_l, %onef_l : f32\n\
         \x20         %den_l = arith.addf %m2_l, %onef_l : f32\n\
         \x20         %s_l = arith.divf %num_l, %den_l : f32\n\
         \x20         %s2_l = arith.mulf %s_l, %s_l : f32\n\
         \x20         %p1_l = arith.mulf %s2_l, %c17_l : f32\n\
         \x20         %p2_l = arith.addf %p1_l, %c15_l : f32\n\
         \x20         %p3_l = arith.mulf %p2_l, %s2_l : f32\n\
         \x20         %p4_l = arith.addf %p3_l, %c13_l : f32\n\
         \x20         %p5_l = arith.mulf %p4_l, %s2_l : f32\n\
         \x20         %p6_l = arith.addf %p5_l, %onef_l : f32\n\
         \x20         %poly_l = arith.mulf %p6_l, %s_l : f32\n\
         \x20         %two_l = arith.constant 2.0 : f32\n\
         \x20         %logm_l = arith.mulf %poly_l, %two_l : f32\n\
         \x20         %ef_l = arith.sitofp %e2_l : i32 to f32\n\
         \x20         %eln2_l = arith.mulf %ef_l, %ln2_l : f32\n\
         \x20         {outp} = arith.addf %eln2_l, %logm_l : f32\n"
    )
}

/// Pure-`arith` scalar-f32 `sin` — range-reduce `r = x − 2π·round(x/2π)` into
/// `[-π,π]` (round via a float→int→float floor), then a degree-11 odd Taylor
/// Horner (err ≲ 4e-4 on the range). `inp`→`outp`; SSA suffix `_s`.
fn approx_sin_f32(inp: &str, outp: &str) -> String {
    let (invtwopi, twopi, c1, c2, c3, c4, c5) = (
        fbits("invtwopi_s", 0.159_154_94),
        fbits("twopi_s", 6.283_185_5),
        fbits("c1_s", -1.0 / 6.0),
        fbits("c2_s", 1.0 / 120.0),
        fbits("c3_s", -1.0 / 5040.0),
        fbits("c4_s", 1.0 / 362_880.0),
        fbits("c5_s", -1.0 / 39_916_800.0),
    );
    format!(
        "{invtwopi}{twopi}{c1}{c2}{c3}{c4}{c5}\
         \x20         %half_s = arith.constant 0.5 : f32\n\
         \x20         %onef_s = arith.constant 1.0 : f32\n\
         \x20         %cm1f_s = arith.constant -1.0 : f32\n\
         \x20         %c0f_s = arith.constant 0.0 : f32\n\
         \x20         %t_s = arith.mulf {inp}, %invtwopi_s : f32\n\
         \x20         %th_s = arith.addf %t_s, %half_s : f32\n\
         \x20         %ki_s = arith.fptosi %th_s : f32 to i32\n\
         \x20         %kf0_s = arith.sitofp %ki_s : i32 to f32\n\
         \x20         %over_s = arith.cmpf ogt, %kf0_s, %th_s : f32\n\
         \x20         %adj_s = arith.select %over_s, %cm1f_s, %c0f_s : f32\n\
         \x20         %kf_s = arith.addf %kf0_s, %adj_s : f32\n\
         \x20         %ktp_s = arith.mulf %kf_s, %twopi_s : f32\n\
         \x20         %r_s = arith.subf {inp}, %ktp_s : f32\n\
         \x20         %r2_s = arith.mulf %r_s, %r_s : f32\n\
         \x20         %pa_s = arith.mulf %r2_s, %c5_s : f32\n\
         \x20         %pa2_s = arith.addf %pa_s, %c4_s : f32\n\
         \x20         %pb_s = arith.mulf %pa2_s, %r2_s : f32\n\
         \x20         %pb2_s = arith.addf %pb_s, %c3_s : f32\n\
         \x20         %pc_s = arith.mulf %pb2_s, %r2_s : f32\n\
         \x20         %pc2_s = arith.addf %pc_s, %c2_s : f32\n\
         \x20         %pd_s = arith.mulf %pc2_s, %r2_s : f32\n\
         \x20         %pd2_s = arith.addf %pd_s, %c1_s : f32\n\
         \x20         %pe_s = arith.mulf %pd2_s, %r2_s : f32\n\
         \x20         %pe2_s = arith.addf %pe_s, %onef_s : f32\n\
         \x20         {outp} = arith.mulf %pe2_s, %r_s : f32\n"
    )
}

impl UnaryOp {
    pub fn name(&self) -> &'static str {
        match self {
            UnaryOp::Relu => "relu",
            UnaryOp::Neg => "neg",
            UnaryOp::Abs => "abs",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Rsqrt => "rsqrt",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Recip => "recip",
            UnaryOp::Sigmoid => "sigmoid",
            UnaryOp::Silu => "silu",
            UnaryOp::Gelu => "gelu",
            UnaryOp::Floor => "floor",
            UnaryOp::Ceil => "ceil",
            UnaryOp::Round => "round",
            UnaryOp::Sign => "sign",
            UnaryOp::Softplus => "softplus",
            UnaryOp::Elu => "elu",
            UnaryOp::HardSwish => "hardswish",
            UnaryOp::HardSigmoid => "hardsigmoid",
            UnaryOp::Mish => "mish",
            UnaryOp::Softsign => "softsign",
            UnaryOp::LogSigmoid => "logsigmoid",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Erf => "erf",
        }
    }
    /// MLIR body computing `%y` from `%x` at value type `vt` (scalar or vector).
    fn body(&self, vt: &str, lanes: usize) -> String {
        // constant declarator (dense<> for vectors, scalar otherwise)
        let k = |val: &str, name: &str| -> String {
            if lanes > 1 {
                format!("          {name} = arith.constant dense<{val}> : {vt}\n")
            } else {
                format!("          {name} = arith.constant {val} : {vt}\n")
            }
        };
        match self {
            UnaryOp::Relu => format!("{}          %y = arith.maximumf %x, %zero : {vt}\n", k("0.0", "%zero")),
            UnaryOp::Neg => format!("          %y = arith.negf %x : {vt}\n"),
            UnaryOp::Abs => format!("          %y = math.absf %x : {vt}\n"),
            UnaryOp::Recip => format!("{}          %y = arith.divf %one, %x : {vt}\n", k("1.0", "%one")),
            // --- pure-arith transcendentals (scalar f32; guarded in emit_unary) ---
            UnaryOp::Exp => approx_exp_f32("%x", "%y"),
            UnaryOp::Rsqrt => approx_rsqrt_f32("%x", "%y"),
            // √x = x·rsqrt(x)
            UnaryOp::Sqrt => {
                format!("{}          %y = arith.mulf %x, %rs : f32\n", approx_rsqrt_f32("%x", "%rs"))
            }
            // σ(x) = 1/(1+e^-x)
            UnaryOp::Sigmoid => format!(
                "          %oneb = arith.constant 1.0 : f32\n          %nx = arith.negf %x : f32\n{}          %d = arith.addf %e, %oneb : f32\n          %y = arith.divf %oneb, %d : f32\n",
                approx_exp_f32("%nx", "%e")
            ),
            // silu(x) = x·σ(x)
            UnaryOp::Silu => format!(
                "          %oneb = arith.constant 1.0 : f32\n          %nx = arith.negf %x : f32\n{}          %d = arith.addf %e, %oneb : f32\n          %sg = arith.divf %oneb, %d : f32\n          %y = arith.mulf %x, %sg : f32\n",
                approx_exp_f32("%nx", "%e")
            ),
            // tanh(x) = (e^2x−1)/(e^2x+1)
            UnaryOp::Tanh => format!(
                "          %oneb = arith.constant 1.0 : f32\n          %twob = arith.constant 2.0 : f32\n          %x2t = arith.mulf %x, %twob : f32\n{}          %num = arith.subf %e2, %oneb : f32\n          %den = arith.addf %e2, %oneb : f32\n          %y = arith.divf %num, %den : f32\n",
                approx_exp_f32("%x2t", "%e2")
            ),
            // gelu(x) = 0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³)))  (tanh approximation)
            UnaryOp::Gelu => format!(
                "{c044}{csq}          %oneb = arith.constant 1.0 : f32\n          %twob = arith.constant 2.0 : f32\n          %chalfb = arith.constant 0.5 : f32\n          %x2g = arith.mulf %x, %x : f32\n          %x3g = arith.mulf %x2g, %x : f32\n          %cx3g = arith.mulf %x3g, %c044b : f32\n          %in0g = arith.addf %x, %cx3g : f32\n          %innerg = arith.mulf %in0g, %csqb : f32\n          %i2g = arith.mulf %innerg, %twob : f32\n{}          %numg = arith.subf %e2g, %oneb : f32\n          %deng = arith.addf %e2g, %oneb : f32\n          %thg = arith.divf %numg, %deng : f32\n          %t1g = arith.addf %thg, %oneb : f32\n          %hxg = arith.mulf %x, %chalfb : f32\n          %y = arith.mulf %hxg, %t1g : f32\n",
                approx_exp_f32("%i2g", "%e2g"),
                c044 = fbits("c044b", 0.044715),
                csq = fbits("csqb", 0.7978845608)
            ),
            UnaryOp::Log => approx_log_f32("%x", "%y"),
            // ── rounding / sign (pure arith: float→int→float + cmpf/select) ──
            // NOTE: %cz not %c0 — the scaffold already binds %c0 (loop index).
            UnaryOp::Floor => "          %fi = arith.fptosi %x : f32 to i32\n          %ff = arith.sitofp %fi : i32 to f32\n          %cm1 = arith.constant -1.0 : f32\n          %cz = arith.constant 0.0 : f32\n          %gt = arith.cmpf ogt, %ff, %x : f32\n          %adj = arith.select %gt, %cm1, %cz : f32\n          %y = arith.addf %ff, %adj : f32\n".to_string(),
            UnaryOp::Ceil => "          %fi = arith.fptosi %x : f32 to i32\n          %ff = arith.sitofp %fi : i32 to f32\n          %c1p = arith.constant 1.0 : f32\n          %cz = arith.constant 0.0 : f32\n          %lt = arith.cmpf olt, %ff, %x : f32\n          %adj = arith.select %lt, %c1p, %cz : f32\n          %y = arith.addf %ff, %adj : f32\n".to_string(),
            UnaryOp::Round => "          %half = arith.constant 0.5 : f32\n          %xh = arith.addf %x, %half : f32\n          %fi = arith.fptosi %xh : f32 to i32\n          %ff = arith.sitofp %fi : i32 to f32\n          %cm1 = arith.constant -1.0 : f32\n          %cz = arith.constant 0.0 : f32\n          %gt = arith.cmpf ogt, %ff, %xh : f32\n          %adj = arith.select %gt, %cm1, %cz : f32\n          %y = arith.addf %ff, %adj : f32\n".to_string(),
            UnaryOp::Sign => "          %cz = arith.constant 0.0 : f32\n          %c1p = arith.constant 1.0 : f32\n          %cm1 = arith.constant -1.0 : f32\n          %gt = arith.cmpf ogt, %x, %cz : f32\n          %lt = arith.cmpf olt, %x, %cz : f32\n          %pp = arith.select %gt, %c1p, %cz : f32\n          %y = arith.select %lt, %cm1, %pp : f32\n".to_string(),
            // ── piecewise (pure arith; min via −max(−,−) since AIE2 lacks minimumf) ──
            UnaryOp::HardSwish => "          %c3 = arith.constant 3.0 : f32\n          %cz = arith.constant 0.0 : f32\n          %c6 = arith.constant 6.0 : f32\n          %x3 = arith.addf %x, %c3 : f32\n          %mx = arith.maximumf %x3, %cz : f32\n          %nmx = arith.negf %mx : f32\n          %n6 = arith.negf %c6 : f32\n          %mmax = arith.maximumf %nmx, %n6 : f32\n          %mn = arith.negf %mmax : f32\n          %d = arith.divf %mn, %c6 : f32\n          %y = arith.mulf %x, %d : f32\n".to_string(),
            UnaryOp::HardSigmoid => "          %c3 = arith.constant 3.0 : f32\n          %cz = arith.constant 0.0 : f32\n          %c6 = arith.constant 6.0 : f32\n          %c1p = arith.constant 1.0 : f32\n          %x3 = arith.addf %x, %c3 : f32\n          %dd = arith.divf %x3, %c6 : f32\n          %mx = arith.maximumf %dd, %cz : f32\n          %nmx = arith.negf %mx : f32\n          %n1 = arith.negf %c1p : f32\n          %mmax = arith.maximumf %nmx, %n1 : f32\n          %y = arith.negf %mmax : f32\n".to_string(),
            UnaryOp::Softsign => "          %ax = math.absf %x : f32\n          %c1p = arith.constant 1.0 : f32\n          %d = arith.addf %ax, %c1p : f32\n          %y = arith.divf %x, %d : f32\n".to_string(),
            // ── exp/log-based ──
            // softplus(x) = log(1+e^x)
            UnaryOp::Softplus => format!(
                "{}          %c1p = arith.constant 1.0 : f32\n          %u = arith.addf %e, %c1p : f32\n{}",
                approx_exp_f32("%x", "%e"),
                approx_log_f32("%u", "%y")
            ),
            // elu(x) = x>0 ? x : e^x−1
            UnaryOp::Elu => format!(
                "{}          %c1p = arith.constant 1.0 : f32\n          %cz = arith.constant 0.0 : f32\n          %em1 = arith.subf %e, %c1p : f32\n          %gt = arith.cmpf ogt, %x, %cz : f32\n          %y = arith.select %gt, %x, %em1 : f32\n",
                approx_exp_f32("%x", "%e")
            ),
            // mish(x) = x·tanh(softplus(x)) = x·(u²−1)/(u²+1), u=1+e^x (one exp)
            UnaryOp::Mish => format!(
                "{}          %c1p = arith.constant 1.0 : f32\n          %u = arith.addf %e, %c1p : f32\n          %u2 = arith.mulf %u, %u : f32\n          %num = arith.subf %u2, %c1p : f32\n          %den = arith.addf %u2, %c1p : f32\n          %t = arith.divf %num, %den : f32\n          %y = arith.mulf %x, %t : f32\n",
                approx_exp_f32("%x", "%e")
            ),
            // logsigmoid(x) = −softplus(−x) = −log(1+e^−x)
            UnaryOp::LogSigmoid => format!(
                "          %nx = arith.negf %x : f32\n{}          %c1p = arith.constant 1.0 : f32\n          %u = arith.addf %e, %c1p : f32\n{}          %y = arith.negf %lu : f32\n",
                approx_exp_f32("%nx", "%e"),
                approx_log_f32("%u", "%lu")
            ),
            // ── trig / erf ──
            UnaryOp::Sin => approx_sin_f32("%x", "%y"),
            UnaryOp::Cos => format!(
                "{}          %xph = arith.addf %x, %pio2 : f32\n{}",
                fbits("pio2", std::f32::consts::FRAC_PI_2),
                approx_sin_f32("%xph", "%y")
            ),
            // erf(x) via Abramowitz-Stegun 7.1.26: sign(x)·(1 − p(t)·e^{−x²}), t=1/(1+p|x|)
            UnaryOp::Erf => format!(
                "{}{}{}{}{}{}          %ax = math.absf %x : f32\n          %c1e = arith.constant 1.0 : f32\n          %c0e = arith.constant 0.0 : f32\n          %pax = arith.mulf %perf, %ax : f32\n          %den = arith.addf %pax, %c1e : f32\n          %t = arith.divf %c1e, %den : f32\n          %h0 = arith.mulf %t, %a5f : f32\n          %h1 = arith.addf %h0, %a4f : f32\n          %h2 = arith.mulf %h1, %t : f32\n          %h3 = arith.addf %h2, %a3f : f32\n          %h4 = arith.mulf %h3, %t : f32\n          %h5 = arith.addf %h4, %a2f : f32\n          %h6 = arith.mulf %h5, %t : f32\n          %h7 = arith.addf %h6, %a1f : f32\n          %poly = arith.mulf %h7, %t : f32\n          %x2 = arith.mulf %x, %x : f32\n          %nx2 = arith.negf %x2 : f32\n{}          %pex = arith.mulf %poly, %ex : f32\n          %r = arith.subf %c1e, %pex : f32\n          %lt = arith.cmpf olt, %x, %c0e : f32\n          %nr = arith.negf %r : f32\n          %y = arith.select %lt, %nr, %r : f32\n",
                fbits("perf", 0.327_591_1),
                fbits("a1f", 0.254_829_592),
                fbits("a2f", -0.284_496_736),
                fbits("a3f", 1.421_413_741),
                fbits("a4f", -1.453_152_027),
                fbits("a5f", 1.061_405_429),
                approx_exp_f32("%nx2", "%ex")
            ),
        }
    }
    /// Ops emitted via the pure-arith approximations above — scalar f32 only.
    fn needs_approx(&self) -> bool {
        // Everything except the vector-legal relu/neg/abs/recip is scalar-f32:
        // the transcendental approximations, and the rounding/piecewise ops that
        // use cmpf/select/fptosi (no f32 vector path on AIE2).
        !matches!(self, UnaryOp::Relu | UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Recip)
    }
    /// Host reference (f32) for validation.
    pub fn apply_f32(&self, x: f32) -> f32 {
        match self {
            UnaryOp::Relu => x.max(0.0),
            UnaryOp::Neg => -x,
            UnaryOp::Abs => x.abs(),
            UnaryOp::Exp => x.exp(),
            UnaryOp::Log => x.ln(),
            UnaryOp::Sqrt => x.sqrt(),
            UnaryOp::Rsqrt => 1.0 / x.sqrt(),
            UnaryOp::Tanh => x.tanh(),
            UnaryOp::Recip => 1.0 / x,
            UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            UnaryOp::Silu => x / (1.0 + (-x).exp()),
            UnaryOp::Gelu => {
                let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            }
            UnaryOp::Floor => x.floor(),
            UnaryOp::Ceil => x.ceil(),
            UnaryOp::Round => (x + 0.5).floor(),
            UnaryOp::Sign => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            UnaryOp::Softplus => (1.0 + x.exp()).ln(),
            UnaryOp::Elu => {
                if x > 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
            UnaryOp::HardSwish => x * (x + 3.0).max(0.0).min(6.0) / 6.0,
            UnaryOp::HardSigmoid => ((x + 3.0) / 6.0).max(0.0).min(1.0),
            UnaryOp::Mish => x * (x.exp().ln_1p()).tanh(),
            UnaryOp::Softsign => x / (1.0 + x.abs()),
            UnaryOp::LogSigmoid => -(1.0 + (-x).exp()).ln(),
            UnaryOp::Sin => x.sin(),
            UnaryOp::Cos => x.cos(),
            UnaryOp::Erf => {
                // Abramowitz-Stegun 7.1.26 (matches the emitted kernel).
                let s = x.signum();
                let ax = x.abs();
                let t = 1.0 / (1.0 + 0.3275911 * ax);
                let poly = t
                    * (0.254829592
                        + t * (-0.284496736
                            + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
                s * (1.0 - poly * (-ax * ax).exp())
            }
        }
    }
}

/// Emit AIE-MLIR for a 1-D `n`-element **unary** float activation `out = op(in)`
/// on one compute tile (`arg0`=in, `arg2`=out, `arg1` unused — the 1-in/1-out
/// ABI of [`crate::npu_gemm::NpuIoF32`]/`NpuIoBf16`). f32 lowers scalar (no f32
/// vector path); bf16 vectorizes 32-wide. Requires `n % chunk == 0`,
/// `chunk % lanes == 0`.
pub fn emit_unary(op: UnaryOp, ty: Ty, n: usize, chunk: usize) -> String {
    assert!(ty.is_float(), "activations are float ops (use Ty::F32 or Ty::Bf16)");
    assert!(
        !(op.needs_approx() && ty != Ty::F32),
        "{} uses the pure-arith approximation (scalar f32 only)",
        op.name()
    );
    assert!(chunk > 0 && n % chunk == 0, "n ({n}) must be a multiple of chunk ({chunk})");
    let lanes = ty.lanes();
    assert!(chunk % lanes == 0, "chunk ({chunk}) must be a multiple of {lanes} lanes");
    let t = ty.mlir();
    let vt = v_type(t, lanes);
    let step = lanes;
    let load = v_load(t, lanes, "%x", "%in", chunk);
    let body = op.body(&vt, lanes);
    let store = v_store(t, lanes, "%y", "%out", chunk);
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_in = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim_in, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{chunk}x{t}>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{chunk}x{t}>>
    %0 = aie.core(%core) {{
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %iter = %c0 to %cmax step %c1 {{
        %in_sv = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<{chunk}x{t}>>
        %in = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<{chunk}x{t}>> -> memref<{chunk}x{t}>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{chunk}x{t}>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{chunk}x{t}>> -> memref<{chunk}x{t}>
        %lo = arith.constant 0 : index
        %hi = arith.constant {chunk} : index
        %st = arith.constant {step} : index
        scf.for %i = %lo to %hi step %st {{
{load}{body}{store}        }}
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{n}x{t}>, %arg1: memref<{n}x{t}>, %arg2: memref<{n}x{t}>) {{
      %t0 = aiex.dma_configure_task_for @in0 {{
        aie.dma_bd(%arg0 : memref<{n}x{t}>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%t0)
      %t2 = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{n}x{t}>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t0)
    }}
  }}
}}
"#
    )
}

// ============================================================================
// Row-reduction ops: softmax + normalizations (pure-Rust AIE-MLIR, scalar f32)
// ============================================================================
// These reduce over the last axis (`cols`) per row, then apply — the shape at
// the heart of transformer inference. The whole [rows,cols] tile is resident in
// the core (keep rows*cols within tile memory). f32 scalar; they reuse the
// pure-arith `exp`/`rsqrt` primitives. 1-in / 1-out same size → NpuIoF32.

/// Largest divisor of `rows` whose `tile_rows × cols` f32 tile fits the core's
/// streaming budget. The row-reduction kernels stream this many rows at a time
/// (only `tile_rows × cols` is ever resident), so arbitrarily large `rows` work.
fn pick_tile_rows(rows: usize, cols: usize) -> usize {
    // f32 elements per streamed tile — kept small because each of the 2 fifos is
    // double-buffered (4 live buffers), so 4·budget·4B must stay within the tile's
    // 64 KB data memory with headroom for code/stack.
    let budget = 1024usize;
    (1..=rows).rev().find(|&tr| rows % tr == 0 && tr * cols <= budget).unwrap_or(1)
}

/// Emit AIE-MLIR for a numerically-stable row **softmax** over `[rows, cols]`:
/// `out[r,c] = exp(x[r,c]-max_r) / Σ_c exp(x[r,c]-max_r)`. Three passes per row
/// (max, Σexp, write) — the exp calls sit in distinct loop-body regions so their
/// internal SSA doesn't collide. **Row-streamed**: only `tile_rows × cols` is
/// resident, so `rows × cols` may exceed tile memory (the objectfifo chunks the
/// input into `rows/tile_rows` transfers; the forever-loop processes each).
pub fn emit_softmax(rows: usize, cols: usize) -> String {
    let n = rows * cols;
    let tr = pick_tile_rows(rows, cols);
    let exp_sum = approx_exp_f32("%xm", "%e");
    let exp_wr = approx_exp_f32("%xmw", "%ew");
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_in = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim_in, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{tr}x{cols}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{tr}x{cols}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      scf.for %iter = %z0 to %zmax step %one {{
        %in_sv = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<{tr}x{cols}xf32>>
        %in = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<{tr}x{cols}xf32>> -> memref<{tr}x{cols}xf32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{tr}x{cols}xf32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{tr}x{cols}xf32>> -> memref<{tr}x{cols}xf32>
        %rN = arith.constant {tr} : index
        %cN = arith.constant {cols} : index
        %ninf_i = arith.constant -8388608 : i32
        %ninf = arith.bitcast %ninf_i : i32 to f32
        %zerof = arith.constant 0.0 : f32
        %onef = arith.constant 1.0 : f32
        scf.for %r = %z0 to %rN step %one {{
          %maxv = scf.for %c = %z0 to %cN step %one iter_args(%m = %ninf) -> (f32) {{
            %x = memref.load %in[%r, %c] : memref<{tr}x{cols}xf32>
            %m2 = arith.maximumf %m, %x : f32
            scf.yield %m2 : f32
          }}
          %sum = scf.for %c = %z0 to %cN step %one iter_args(%s = %zerof) -> (f32) {{
            %x = memref.load %in[%r, %c] : memref<{tr}x{cols}xf32>
            %xm = arith.subf %x, %maxv : f32
{exp_sum}            %s2 = arith.addf %s, %e : f32
            scf.yield %s2 : f32
          }}
          %inv = arith.divf %onef, %sum : f32
          scf.for %c = %z0 to %cN step %one {{
            %xw = memref.load %in[%r, %c] : memref<{tr}x{cols}xf32>
            %xmw = arith.subf %xw, %maxv : f32
{exp_wr}            %o = arith.mulf %ew, %inv : f32
            memref.store %o, %out[%r, %c] : memref<{tr}x{cols}xf32>
          }}
        }}
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{rows}x{cols}xf32>, %arg1: memref<{rows}x{cols}xf32>, %arg2: memref<{rows}x{cols}xf32>) {{
      %t0 = aiex.dma_configure_task_for @in0 {{
        aie.dma_bd(%arg0 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%t0)
      %t2 = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t0)
    }}
  }}
}}
"#
    )
}

/// Emit AIE-MLIR for a row **RMSNorm** over `[rows, cols]` (no affine gamma):
/// `out[r,c] = x[r,c] · rsqrt(mean_c(x²) + eps)`. Two passes per row (Σx², apply)
/// with the pure-arith `rsqrt`.
pub fn emit_rms_norm(rows: usize, cols: usize, eps: f32) -> String {
    let n = rows * cols;
    let epsc = fbits("eps", eps);
    let rsq = approx_rsqrt_f32("%v", "%rs");
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_in = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim_in, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{rows}x{cols}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{rows}x{cols}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      scf.for %iter = %z0 to %zmax step %one {{
        %in_sv = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<{rows}x{cols}xf32>>
        %in = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<{rows}x{cols}xf32>> -> memref<{rows}x{cols}xf32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{rows}x{cols}xf32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{rows}x{cols}xf32>> -> memref<{rows}x{cols}xf32>
        %rN = arith.constant {rows} : index
        %cN = arith.constant {cols} : index
        %zerof = arith.constant 0.0 : f32
        %colsf = arith.constant {cols}.0 : f32
{epsc}        scf.for %r = %z0 to %rN step %one {{
          %ss = scf.for %c = %z0 to %cN step %one iter_args(%s = %zerof) -> (f32) {{
            %x = memref.load %in[%r, %c] : memref<{rows}x{cols}xf32>
            %x2 = arith.mulf %x, %x : f32
            %s2 = arith.addf %s, %x2 : f32
            scf.yield %s2 : f32
          }}
          %meanss = arith.divf %ss, %colsf : f32
          %v = arith.addf %meanss, %eps : f32
{rsq}          scf.for %c = %z0 to %cN step %one {{
            %xw = memref.load %in[%r, %c] : memref<{rows}x{cols}xf32>
            %o = arith.mulf %xw, %rs : f32
            memref.store %o, %out[%r, %c] : memref<{rows}x{cols}xf32>
          }}
        }}
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{rows}x{cols}xf32>, %arg1: memref<{rows}x{cols}xf32>, %arg2: memref<{rows}x{cols}xf32>) {{
      %t0 = aiex.dma_configure_task_for @in0 {{
        aie.dma_bd(%arg0 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%t0)
      %t2 = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t0)
    }}
  }}
}}
"#
    )
}

/// Emit AIE-MLIR for a row **LayerNorm** over `[rows, cols]` (no affine
/// gamma/beta): `out[r,c] = (x[r,c] − mean_r) · rsqrt(var_r + eps)`. Three passes
/// per row (Σx→mean, Σ(x−mean)²→var, apply).
pub fn emit_layer_norm(rows: usize, cols: usize, eps: f32) -> String {
    let n = rows * cols;
    let epsc = fbits("eps", eps);
    let rsq = approx_rsqrt_f32("%vv", "%rs");
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_in = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim_in, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{rows}x{cols}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{rows}x{cols}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      scf.for %iter = %z0 to %zmax step %one {{
        %in_sv = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<{rows}x{cols}xf32>>
        %in = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<{rows}x{cols}xf32>> -> memref<{rows}x{cols}xf32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{rows}x{cols}xf32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{rows}x{cols}xf32>> -> memref<{rows}x{cols}xf32>
        %rN = arith.constant {rows} : index
        %cN = arith.constant {cols} : index
        %zerof = arith.constant 0.0 : f32
        %colsf = arith.constant {cols}.0 : f32
{epsc}        scf.for %r = %z0 to %rN step %one {{
          %sm = scf.for %c = %z0 to %cN step %one iter_args(%s = %zerof) -> (f32) {{
            %x = memref.load %in[%r, %c] : memref<{rows}x{cols}xf32>
            %s2 = arith.addf %s, %x : f32
            scf.yield %s2 : f32
          }}
          %mean = arith.divf %sm, %colsf : f32
          %sv = scf.for %c = %z0 to %cN step %one iter_args(%s = %zerof) -> (f32) {{
            %x = memref.load %in[%r, %c] : memref<{rows}x{cols}xf32>
            %d = arith.subf %x, %mean : f32
            %d2 = arith.mulf %d, %d : f32
            %s2 = arith.addf %s, %d2 : f32
            scf.yield %s2 : f32
          }}
          %var = arith.divf %sv, %colsf : f32
          %vv = arith.addf %var, %eps : f32
{rsq}          scf.for %c = %z0 to %cN step %one {{
            %xw = memref.load %in[%r, %c] : memref<{rows}x{cols}xf32>
            %dw = arith.subf %xw, %mean : f32
            %o = arith.mulf %dw, %rs : f32
            memref.store %o, %out[%r, %c] : memref<{rows}x{cols}xf32>
          }}
        }}
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{rows}x{cols}xf32>, %arg1: memref<{rows}x{cols}xf32>, %arg2: memref<{rows}x{cols}xf32>) {{
      %t0 = aiex.dma_configure_task_for @in0 {{
        aie.dma_bd(%arg0 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%t0)
      %t2 = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t0)
    }}
  }}
}}
"#
    )
}

/// Emit AIE-MLIR for an **affine RMSNorm** over `[rows, cols]`:
/// `out[r,c] = x[r,c] · rsqrt(mean_c(x²)+eps) · gamma[c] + beta[c]`. `gamma` and
/// `beta` (each `[cols]`) are packed into one `[2*cols]` second buffer (`gamma`
/// then `beta`), so it runs through the generic 3-buffer [`crate::npu_gemm::NpuRun3`]
/// (arg0=x, arg1=gamma‖beta, arg2=out). Classic RMSNorm passes `beta=0`.
pub fn emit_rms_norm_affine(rows: usize, cols: usize, eps: f32) -> String {
    let n = rows * cols;
    let gb2 = 2 * cols;
    let tr = pick_tile_rows(rows, cols);
    let epsc = fbits("eps", eps);
    let rsq = approx_rsqrt_f32("%v", "%rs");
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_x = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_gb = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @x0(%shim_x, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{tr}x{cols}xf32>>
    aie.objectfifo @gb0(%shim_gb, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{gb2}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{tr}x{cols}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %cN = arith.constant {cols} : index
      %colsi = arith.constant {cols} : index
      %zerof = arith.constant 0.0 : f32
      %colsf = arith.constant {cols}.0 : f32
{epsc}      %gb_sv = aie.objectfifo.acquire @gb0(Consume, 1) : !aie.objectfifosubview<memref<{gb2}xf32>>
      %gb = aie.objectfifo.subview.access %gb_sv[0] : !aie.objectfifosubview<memref<{gb2}xf32>> -> memref<{gb2}xf32>
      scf.for %iter = %z0 to %zmax step %one {{
        %x_sv = aie.objectfifo.acquire @x0(Consume, 1) : !aie.objectfifosubview<memref<{tr}x{cols}xf32>>
        %x = aie.objectfifo.subview.access %x_sv[0] : !aie.objectfifosubview<memref<{tr}x{cols}xf32>> -> memref<{tr}x{cols}xf32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{tr}x{cols}xf32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{tr}x{cols}xf32>> -> memref<{tr}x{cols}xf32>
        %rN = arith.constant {tr} : index
        scf.for %r = %z0 to %rN step %one {{
          %ss = scf.for %c = %z0 to %cN step %one iter_args(%s = %zerof) -> (f32) {{
            %xv = memref.load %x[%r, %c] : memref<{tr}x{cols}xf32>
            %x2 = arith.mulf %xv, %xv : f32
            %s2 = arith.addf %s, %x2 : f32
            scf.yield %s2 : f32
          }}
          %meanss = arith.divf %ss, %colsf : f32
          %v = arith.addf %meanss, %eps : f32
{rsq}          scf.for %c = %z0 to %cN step %one {{
            %xw = memref.load %x[%r, %c] : memref<{tr}x{cols}xf32>
            %g = memref.load %gb[%c] : memref<{gb2}xf32>
            %cpc = arith.addi %c, %colsi : index
            %bta = memref.load %gb[%cpc] : memref<{gb2}xf32>
            %nrm = arith.mulf %xw, %rs : f32
            %sc = arith.mulf %nrm, %g : f32
            %o = arith.addf %sc, %bta : f32
            memref.store %o, %out[%r, %c] : memref<{tr}x{cols}xf32>
          }}
        }}
        aie.objectfifo.release @x0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{rows}x{cols}xf32>, %arg1: memref<{gb2}xf32>, %arg2: memref<{rows}x{cols}xf32>) {{
      %tx = aiex.dma_configure_task_for @x0 {{
        aie.dma_bd(%arg0 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tx)
      %tg = aiex.dma_configure_task_for @gb0 {{
        aie.dma_bd(%arg1 : memref<{gb2}xf32>, 0, {gb2}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {gb2}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tg)
      %to = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%tx)
      aiex.dma_free_task(%tg)
    }}
  }}
}}
"#
    )
}

/// Emit AIE-MLIR for an **affine LayerNorm** over `[rows, cols]`:
/// `out[r,c] = (x[r,c]−mean_r)·rsqrt(var_r+eps)·gamma[c] + beta[c]`. Same packed
/// `gamma‖beta` [2*cols] second buffer + [`crate::npu_gemm::NpuRun3`] ABI as
/// [`emit_rms_norm_affine`].
pub fn emit_layer_norm_affine(rows: usize, cols: usize, eps: f32) -> String {
    let n = rows * cols;
    let gb2 = 2 * cols;
    let tr = pick_tile_rows(rows, cols);
    let epsc = fbits("eps", eps);
    let rsq = approx_rsqrt_f32("%vv", "%rs");
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_x = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_gb = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @x0(%shim_x, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{tr}x{cols}xf32>>
    aie.objectfifo @gb0(%shim_gb, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{gb2}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{tr}x{cols}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %cN = arith.constant {cols} : index
      %colsi = arith.constant {cols} : index
      %zerof = arith.constant 0.0 : f32
      %colsf = arith.constant {cols}.0 : f32
{epsc}      %gb_sv = aie.objectfifo.acquire @gb0(Consume, 1) : !aie.objectfifosubview<memref<{gb2}xf32>>
      %gb = aie.objectfifo.subview.access %gb_sv[0] : !aie.objectfifosubview<memref<{gb2}xf32>> -> memref<{gb2}xf32>
      scf.for %iter = %z0 to %zmax step %one {{
        %x_sv = aie.objectfifo.acquire @x0(Consume, 1) : !aie.objectfifosubview<memref<{tr}x{cols}xf32>>
        %x = aie.objectfifo.subview.access %x_sv[0] : !aie.objectfifosubview<memref<{tr}x{cols}xf32>> -> memref<{tr}x{cols}xf32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{tr}x{cols}xf32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{tr}x{cols}xf32>> -> memref<{tr}x{cols}xf32>
        %rN = arith.constant {tr} : index
        scf.for %r = %z0 to %rN step %one {{
          %sm = scf.for %c = %z0 to %cN step %one iter_args(%s = %zerof) -> (f32) {{
            %xv = memref.load %x[%r, %c] : memref<{tr}x{cols}xf32>
            %s2 = arith.addf %s, %xv : f32
            scf.yield %s2 : f32
          }}
          %mean = arith.divf %sm, %colsf : f32
          %sv = scf.for %c = %z0 to %cN step %one iter_args(%s = %zerof) -> (f32) {{
            %xv = memref.load %x[%r, %c] : memref<{tr}x{cols}xf32>
            %d = arith.subf %xv, %mean : f32
            %d2 = arith.mulf %d, %d : f32
            %s2 = arith.addf %s, %d2 : f32
            scf.yield %s2 : f32
          }}
          %var = arith.divf %sv, %colsf : f32
          %vv = arith.addf %var, %eps : f32
{rsq}          scf.for %c = %z0 to %cN step %one {{
            %xw = memref.load %x[%r, %c] : memref<{tr}x{cols}xf32>
            %dw = arith.subf %xw, %mean : f32
            %g = memref.load %gb[%c] : memref<{gb2}xf32>
            %cpc = arith.addi %c, %colsi : index
            %bta = memref.load %gb[%cpc] : memref<{gb2}xf32>
            %nrm = arith.mulf %dw, %rs : f32
            %sc = arith.mulf %nrm, %g : f32
            %o = arith.addf %sc, %bta : f32
            memref.store %o, %out[%r, %c] : memref<{tr}x{cols}xf32>
          }}
        }}
        aie.objectfifo.release @x0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{rows}x{cols}xf32>, %arg1: memref<{gb2}xf32>, %arg2: memref<{rows}x{cols}xf32>) {{
      %tx = aiex.dma_configure_task_for @x0 {{
        aie.dma_bd(%arg0 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tx)
      %tg = aiex.dma_configure_task_for @gb0 {{
        aie.dma_bd(%arg1 : memref<{gb2}xf32>, 0, {gb2}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {gb2}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tg)
      %to = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%tx)
      aiex.dma_free_task(%tg)
    }}
  }}
}}
"#
    )
}

/// Emit AIE-MLIR for **GroupNorm** (NCHW): normalize over `(C/G)×H×W` per group,
/// then a per-**channel** affine `gamma[c]·x̂ + beta[c]`. Each group is a
/// contiguous `group_size = (C/G)·H·W` block, so it's a per-row (`rows = N·G`)
/// mean/var normalize; the affine channel is `c = (r % G)·cg + j/hw`. Single-tile
/// (whole tensor resident). `gamma‖beta` packed as `[2·C]` (arg1); x/out as arg0/2
/// → [`crate::npu_gemm::NpuRun3`]. `cg = C/G`, `hw = H·W`, `C = G·cg`.
pub fn emit_group_norm(rows: usize, group_size: usize, num_groups: usize, cg: usize, hw: usize, eps: f32) -> String {
    let n = rows * group_size;
    let c_total = num_groups * cg;
    let gb2 = 2 * c_total;
    let epsc = fbits("eps", eps);
    let rsq = approx_rsqrt_f32("%vv", "%rs");
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_x = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_gb = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @x0(%shim_x, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{rows}x{group_size}xf32>>
    aie.objectfifo @gb0(%shim_gb, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{gb2}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{rows}x{group_size}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %gsN = arith.constant {group_size} : index
      %rN = arith.constant {rows} : index
      %Gc = arith.constant {num_groups} : index
      %cgc = arith.constant {cg} : index
      %hwc = arith.constant {hw} : index
      %Cc = arith.constant {c_total} : index
      %zerof = arith.constant 0.0 : f32
      %gsf = arith.constant {group_size}.0 : f32
{epsc}      %gb_sv = aie.objectfifo.acquire @gb0(Consume, 1) : !aie.objectfifosubview<memref<{gb2}xf32>>
      %gb = aie.objectfifo.subview.access %gb_sv[0] : !aie.objectfifosubview<memref<{gb2}xf32>> -> memref<{gb2}xf32>
      scf.for %iter = %z0 to %zmax step %one {{
        %x_sv = aie.objectfifo.acquire @x0(Consume, 1) : !aie.objectfifosubview<memref<{rows}x{group_size}xf32>>
        %x = aie.objectfifo.subview.access %x_sv[0] : !aie.objectfifosubview<memref<{rows}x{group_size}xf32>> -> memref<{rows}x{group_size}xf32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{rows}x{group_size}xf32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{rows}x{group_size}xf32>> -> memref<{rows}x{group_size}xf32>
        scf.for %r = %z0 to %rN step %one {{
          %g = arith.remui %r, %Gc : index
          %gcg = arith.muli %g, %cgc : index
          %sm = scf.for %j = %z0 to %gsN step %one iter_args(%s = %zerof) -> (f32) {{
            %xv = memref.load %x[%r, %j] : memref<{rows}x{group_size}xf32>
            %s2 = arith.addf %s, %xv : f32
            scf.yield %s2 : f32
          }}
          %mean = arith.divf %sm, %gsf : f32
          %sv = scf.for %j = %z0 to %gsN step %one iter_args(%s = %zerof) -> (f32) {{
            %xv = memref.load %x[%r, %j] : memref<{rows}x{group_size}xf32>
            %d = arith.subf %xv, %mean : f32
            %d2 = arith.mulf %d, %d : f32
            %s2 = arith.addf %s, %d2 : f32
            scf.yield %s2 : f32
          }}
          %var = arith.divf %sv, %gsf : f32
          %vv = arith.addf %var, %eps : f32
{rsq}          scf.for %j = %z0 to %gsN step %one {{
            %xw = memref.load %x[%r, %j] : memref<{rows}x{group_size}xf32>
            %dw = arith.subf %xw, %mean : f32
            %cig = arith.divui %j, %hwc : index
            %c = arith.addi %gcg, %cig : index
            %g_v = memref.load %gb[%c] : memref<{gb2}xf32>
            %cC = arith.addi %c, %Cc : index
            %b_v = memref.load %gb[%cC] : memref<{gb2}xf32>
            %nrm = arith.mulf %dw, %rs : f32
            %sc = arith.mulf %nrm, %g_v : f32
            %o = arith.addf %sc, %b_v : f32
            memref.store %o, %out[%r, %j] : memref<{rows}x{group_size}xf32>
          }}
        }}
        aie.objectfifo.release @x0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{rows}x{group_size}xf32>, %arg1: memref<{gb2}xf32>, %arg2: memref<{rows}x{group_size}xf32>) {{
      %tx = aiex.dma_configure_task_for @x0 {{
        aie.dma_bd(%arg0 : memref<{rows}x{group_size}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tx)
      %tg = aiex.dma_configure_task_for @gb0 {{
        aie.dma_bd(%arg1 : memref<{gb2}xf32>, 0, {gb2}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {gb2}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tg)
      %to = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{rows}x{group_size}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%tx)
      aiex.dma_free_task(%tg)
    }}
  }}
}}
"#
    )
}

/// Emit AIE-MLIR for **RoPE** (rotary position embedding). `x` is
/// `[rows, head_dim]` where `rows = n_tokens · nh` (one row per (token, head));
/// `cos`/`sin` tables are `[n_tokens, head_dim/2]` (packed cos‖sin into arg1). Per
/// row, token = `r / nh` selects the cos/sin row. For `i < n_rot/2` rotates the
/// pair `(a,b)` — NeoX `(i, i+n_rot/2)` else GptJ `(2i, 2i+1)`:
/// `out[a]=x1·cos−x2·sin`, `out[b]=x2·cos+x1·sin`; dims `[n_rot, head_dim)` pass
/// through (partial rotary). Single-tile scalar f32; matches the vulkan `rope.comp`.
pub fn emit_rope(rows: usize, head_dim: usize, n_rot: usize, nh: usize, neox: bool) -> String {
    let nx = rows * head_dim;
    let n_tokens = rows / nh;
    let tab_half = head_dim / 2;
    let rot_half = n_rot / 2;
    let sin_base = n_tokens * tab_half; // offset of sin within the packed cos‖sin
    let cs_len = 2 * sin_base;
    // per-`i` `(a,b)` pair indices (relative to the row base).
    let ab = if neox {
        "            %bidx = arith.addi %i, %rhc : index\n            %xa = arith.addi %rhd, %i : index\n            %xb = arith.addi %rhd, %bidx : index\n".to_string()
    } else {
        "            %aidx = arith.muli %i, %two : index\n            %bidx = arith.addi %aidx, %one : index\n            %xa = arith.addi %rhd, %aidx : index\n            %xb = arith.addi %rhd, %bidx : index\n".to_string()
    };
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_x = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_cs = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @x0(%shim_x, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{nx}xf32>>
    aie.objectfifo @cs0(%shim_cs, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{cs_len}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{nx}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      %rowsN = arith.constant {rows} : index
      %nhc = arith.constant {nh} : index
      %thc = arith.constant {tab_half} : index
      %hdc = arith.constant {head_dim} : index
      %sbc = arith.constant {sin_base} : index
      %rhc = arith.constant {rot_half} : index
      %rothN = arith.constant {rot_half} : index
      %nrotN = arith.constant {n_rot} : index
      %hdN = arith.constant {head_dim} : index
      %cs_sv = aie.objectfifo.acquire @cs0(Consume, 1) : !aie.objectfifosubview<memref<{cs_len}xf32>>
      %cs = aie.objectfifo.subview.access %cs_sv[0] : !aie.objectfifosubview<memref<{cs_len}xf32>> -> memref<{cs_len}xf32>
      scf.for %iter = %z0 to %zmax step %one {{
        %x_sv = aie.objectfifo.acquire @x0(Consume, 1) : !aie.objectfifosubview<memref<{nx}xf32>>
        %x = aie.objectfifo.subview.access %x_sv[0] : !aie.objectfifosubview<memref<{nx}xf32>> -> memref<{nx}xf32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{nx}xf32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{nx}xf32>> -> memref<{nx}xf32>
        scf.for %r = %z0 to %rowsN step %one {{
          %tok = arith.divui %r, %nhc : index
          %cbase = arith.muli %tok, %thc : index
          %sbase = arith.addi %cbase, %sbc : index
          %rhd = arith.muli %r, %hdc : index
          scf.for %i = %z0 to %rothN step %one {{
            %ci = arith.addi %cbase, %i : index
            %cv = memref.load %cs[%ci] : memref<{cs_len}xf32>
            %si = arith.addi %sbase, %i : index
            %sv = memref.load %cs[%si] : memref<{cs_len}xf32>
{ab}            %x1 = memref.load %x[%xa] : memref<{nx}xf32>
            %x2 = memref.load %x[%xb] : memref<{nx}xf32>
            %x1cv = arith.mulf %x1, %cv : f32
            %x2sv = arith.mulf %x2, %sv : f32
            %oa = arith.subf %x1cv, %x2sv : f32
            %x2cv = arith.mulf %x2, %cv : f32
            %x1sv = arith.mulf %x1, %sv : f32
            %ob = arith.addf %x2cv, %x1sv : f32
            memref.store %oa, %out[%xa] : memref<{nx}xf32>
            memref.store %ob, %out[%xb] : memref<{nx}xf32>
          }}
          scf.for %j = %nrotN to %hdN step %one {{
            %xj = arith.addi %rhd, %j : index
            %pv = memref.load %x[%xj] : memref<{nx}xf32>
            memref.store %pv, %out[%xj] : memref<{nx}xf32>
          }}
        }}
        aie.objectfifo.release @x0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{nx}xf32>, %arg1: memref<{cs_len}xf32>, %arg2: memref<{nx}xf32>) {{
      %tx = aiex.dma_configure_task_for @x0 {{
        aie.dma_bd(%arg0 : memref<{nx}xf32>, 0, {nx}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {nx}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tx)
      %tc = aiex.dma_configure_task_for @cs0 {{
        aie.dma_bd(%arg1 : memref<{cs_len}xf32>, 0, {cs_len}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {cs_len}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tc)
      %to = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{nx}xf32>, 0, {nx}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {nx}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%tx)
      aiex.dma_free_task(%tc)
    }}
  }}
}}
"#
    )
}

/// Emit AIE-MLIR for **fused scaled-dot-product attention** (single head) over
/// `Q,K,V ∈ [seq, d]`: `out = softmax(Q·Kᵀ / √d) · V`. One core, one dispatch —
/// per query row it computes the raw scores into a local `[seq]` scratch buffer,
/// does a numerically-stable softmax in place (pure-arith exp), then the
/// weighted sum over V. Scalar f32 (correct, not fast). `K` and `V` are packed
/// into one `[2·seq·d]` buffer (`K` then `V`) → rides the 3-buffer
/// [`crate::npu_gemm::NpuRun3`] (arg0=Q, arg1=K‖V, arg2=out). Keep `seq·d` +
/// `seq·seq` scratch within tile memory.
pub fn emit_attention(seq: usize, d: usize, num_heads: usize, scale: f32, causal: bool) -> String {
    let hd = num_heads * d; // hidden = heads · head_dim
    let shd = seq * hd; // Q/O element count
    let kv2 = 2 * shd; // packed K‖V
    let scale_decl = fbits("scale", scale);
    let exp_p = approx_exp_f32("%xm", "%e");
    // Causal mask: for query row i, keys j>i are future → score = −∞ (softmax → 0).
    let sc_mask = if causal {
        "              %sc0 = arith.mulf %dot, %scale : f32\n              %fut = arith.cmpi ugt, %j, %i : index\n              %sc = arith.select %fut, %ninf, %sc0 : f32\n"
    } else {
        "              %sc = arith.mulf %dot, %scale : f32\n"
    };
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_q = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_kv = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @q0(%shim_q, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{seq}x{hd}xf32>>
    aie.objectfifo @kv0(%shim_kv, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{kv2}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{seq}x{hd}xf32>>
    %scores = aie.buffer(%core) : memref<{seq}xf32>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %seqN = arith.constant {seq} : index
      %dN = arith.constant {d} : index
      %dc = arith.constant {d} : index
      %HN = arith.constant {num_heads} : index
      %hdc = arith.constant {hd} : index
      %shdc = arith.constant {shd} : index
      %ninf_i = arith.constant -8388608 : i32
      %ninf = arith.bitcast %ninf_i : i32 to f32
      %zerof = arith.constant 0.0 : f32
      %onef = arith.constant 1.0 : f32
{scale_decl}      scf.for %iter = %z0 to %zmax step %one {{
        %q_sv = aie.objectfifo.acquire @q0(Consume, 1) : !aie.objectfifosubview<memref<{seq}x{hd}xf32>>
        %Q = aie.objectfifo.subview.access %q_sv[0] : !aie.objectfifosubview<memref<{seq}x{hd}xf32>> -> memref<{seq}x{hd}xf32>
        %kv_sv = aie.objectfifo.acquire @kv0(Consume, 1) : !aie.objectfifosubview<memref<{kv2}xf32>>
        %KV = aie.objectfifo.subview.access %kv_sv[0] : !aie.objectfifosubview<memref<{kv2}xf32>> -> memref<{kv2}xf32>
        %o_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{seq}x{hd}xf32>>
        %O = aie.objectfifo.subview.access %o_sv[0] : !aie.objectfifosubview<memref<{seq}x{hd}xf32>> -> memref<{seq}x{hd}xf32>
        scf.for %h = %z0 to %HN step %one {{
          %ho = arith.muli %h, %dc : index
          scf.for %i = %z0 to %seqN step %one {{
            %maxv = scf.for %j = %z0 to %seqN step %one iter_args(%m = %ninf) -> (f32) {{
              %jhd = arith.muli %j, %hdc : index
              %jbase = arith.addi %jhd, %ho : index
              %dot = scf.for %k = %z0 to %dN step %one iter_args(%a = %zerof) -> (f32) {{
                %qk = arith.addi %ho, %k : index
                %qv = memref.load %Q[%i, %qk] : memref<{seq}x{hd}xf32>
                %kidx = arith.addi %jbase, %k : index
                %kvv = memref.load %KV[%kidx] : memref<{kv2}xf32>
                %p = arith.mulf %qv, %kvv : f32
                %a2 = arith.addf %a, %p : f32
                scf.yield %a2 : f32
              }}
{sc_mask}              memref.store %sc, %scores[%j] : memref<{seq}xf32>
              %m2 = arith.maximumf %m, %sc : f32
              scf.yield %m2 : f32
            }}
            %sum = scf.for %j = %z0 to %seqN step %one iter_args(%s = %zerof) -> (f32) {{
              %sv = memref.load %scores[%j] : memref<{seq}xf32>
              %xm = arith.subf %sv, %maxv : f32
{exp_p}              memref.store %e, %scores[%j] : memref<{seq}xf32>
              %s2 = arith.addf %s, %e : f32
              scf.yield %s2 : f32
            }}
            %inv = arith.divf %onef, %sum : f32
            scf.for %k = %z0 to %dN step %one {{
              %ok = arith.addi %ho, %k : index
              %acc = scf.for %j = %z0 to %seqN step %one iter_args(%a = %zerof) -> (f32) {{
                %sv = memref.load %scores[%j] : memref<{seq}xf32>
                %jhd = arith.muli %j, %hdc : index
                %vbase = arith.addi %shdc, %jhd : index
                %vh = arith.addi %vbase, %ho : index
                %vidx = arith.addi %vh, %k : index
                %vv = memref.load %KV[%vidx] : memref<{kv2}xf32>
                %p = arith.mulf %sv, %vv : f32
                %a2 = arith.addf %a, %p : f32
                scf.yield %a2 : f32
              }}
              %o = arith.mulf %acc, %inv : f32
              memref.store %o, %O[%i, %ok] : memref<{seq}x{hd}xf32>
            }}
          }}
        }}
        aie.objectfifo.release @q0(Consume, 1)
        aie.objectfifo.release @kv0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{seq}x{hd}xf32>, %arg1: memref<{kv2}xf32>, %arg2: memref<{seq}x{hd}xf32>) {{
      %tq = aiex.dma_configure_task_for @q0 {{
        aie.dma_bd(%arg0 : memref<{seq}x{hd}xf32>, 0, {shd}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {shd}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tq)
      %tkv = aiex.dma_configure_task_for @kv0 {{
        aie.dma_bd(%arg1 : memref<{kv2}xf32>, 0, {kv2}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {kv2}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tkv)
      %to = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{seq}x{hd}xf32>, 0, {shd}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {shd}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%tq)
      aiex.dma_free_task(%tkv)
    }}
  }}
}}
"#
    )
}

/// A row **reduction** op (mirrors `rlx_ir::ReduceOp`) over the last axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

impl ReduceOp {
    pub fn name(&self) -> &'static str {
        match self {
            ReduceOp::Sum => "sum",
            ReduceOp::Mean => "mean",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
            ReduceOp::Prod => "prod",
        }
    }
    /// Host reference over one row.
    pub fn apply(&self, row: &[f32]) -> f32 {
        match self {
            ReduceOp::Sum => row.iter().sum(),
            ReduceOp::Mean => row.iter().sum::<f32>() / row.len() as f32,
            ReduceOp::Max => row.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            ReduceOp::Min => row.iter().cloned().fold(f32::INFINITY, f32::min),
            ReduceOp::Prod => row.iter().product(),
        }
    }
}

/// Emit AIE-MLIR for a row **reduction** `op` over `[rows, cols]`, writing the
/// per-row scalar result **broadcast across the row** into `out[r, :]` (so it
/// ships through the same 1-in/1-out same-size ABI; a caller reads column 0).
pub fn emit_reduce(op: ReduceOp, rows: usize, cols: usize) -> String {
    let n = rows * cols;
    // (init SSA, combine-body producing %a2, finalize line, result SSA). `min`
    // is done as −max(−x): AIE2 lowers `arith.maximumf` but not `arith.minimumf`.
    let bin = |o: &str| format!("            %a2 = {o} %a, %x : f32\n");
    let (init, combine, finalize, res_ssa) = match op {
        ReduceOp::Sum => ("%zerof", bin("arith.addf"), "", "%acc"),
        ReduceOp::Mean => {
            ("%zerof", bin("arith.addf"), "          %res = arith.divf %acc, %colsf : f32\n", "%res")
        }
        ReduceOp::Max => ("%ninf", bin("arith.maximumf"), "", "%acc"),
        ReduceOp::Min => (
            "%ninf",
            "            %nx = arith.negf %x : f32\n            %a2 = arith.maximumf %a, %nx : f32\n".to_string(),
            "          %res = arith.negf %acc : f32\n",
            "%res",
        ),
        ReduceOp::Prod => ("%onef", bin("arith.mulf"), "", "%acc"),
    };
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_in = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim_in, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{rows}x{cols}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{rows}x{cols}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      scf.for %iter = %z0 to %zmax step %one {{
        %in_sv = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<{rows}x{cols}xf32>>
        %in = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<{rows}x{cols}xf32>> -> memref<{rows}x{cols}xf32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{rows}x{cols}xf32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{rows}x{cols}xf32>> -> memref<{rows}x{cols}xf32>
        %rN = arith.constant {rows} : index
        %cN = arith.constant {cols} : index
        %zerof = arith.constant 0.0 : f32
        %onef = arith.constant 1.0 : f32
        %colsf = arith.constant {cols}.0 : f32
        %ninf_i = arith.constant -8388608 : i32
        %ninf = arith.bitcast %ninf_i : i32 to f32
        %pinf_i = arith.constant 2139095040 : i32
        %pinf = arith.bitcast %pinf_i : i32 to f32
        scf.for %r = %z0 to %rN step %one {{
          %acc = scf.for %c = %z0 to %cN step %one iter_args(%a = {init}) -> (f32) {{
            %x = memref.load %in[%r, %c] : memref<{rows}x{cols}xf32>
{combine}            scf.yield %a2 : f32
          }}
{finalize}          scf.for %c = %z0 to %cN step %one {{
            memref.store {res_ssa}, %out[%r, %c] : memref<{rows}x{cols}xf32>
          }}
        }}
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{rows}x{cols}xf32>, %arg1: memref<{rows}x{cols}xf32>, %arg2: memref<{rows}x{cols}xf32>) {{
      %t0 = aiex.dma_configure_task_for @in0 {{
        aie.dma_bd(%arg0 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%t0)
      %t2 = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t0)
    }}
  }}
}}
"#
    )
}

/// `ArgMax` (`is_max`) / `ArgMin` over the last axis: `out[r] = argextreme_j x[r,j]`
/// as an **f32-encoded index** (the index value as a float; ties → smaller index).
/// Per-row `scf.for` with a `(best_val, best_idx)` tuple accumulator, then the
/// index broadcast across the row (exec reads col 0, like [`emit_reduce`]).
pub fn emit_argmax(rows: usize, cols: usize, is_max: bool) -> String {
    let n = rows * cols;
    // argmin(x) = argmax(−x): always compare against −∞ with `ogt` (avoids a +∞
    // constant, which Peano's `opt` rejects), negating x for the min case.
    let load_x = if is_max {
        format!("            %x = memref.load %in[%r, %c] : memref<{rows}x{cols}xf32>\n")
    } else {
        format!("            %xr = memref.load %in[%r, %c] : memref<{rows}x{cols}xf32>\n            %x = arith.negf %xr : f32\n")
    };
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_in = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim_in, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{rows}x{cols}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{rows}x{cols}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %rN = arith.constant {rows} : index
      %cN = arith.constant {cols} : index
      %ninf_i = arith.constant -8388608 : i32
      %ninf = arith.bitcast %ninf_i : i32 to f32
      scf.for %iter = %z0 to %zmax step %one {{
        %in_sv = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<{rows}x{cols}xf32>>
        %in = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<{rows}x{cols}xf32>> -> memref<{rows}x{cols}xf32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{rows}x{cols}xf32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{rows}x{cols}xf32>> -> memref<{rows}x{cols}xf32>
        scf.for %r = %z0 to %rN step %one {{
          %res:2 = scf.for %c = %z0 to %cN step %one iter_args(%bv = %ninf, %bi = %z0) -> (f32, index) {{
{load_x}            %better = arith.cmpf ogt, %x, %bv : f32
            %nbv = arith.select %better, %x, %bv : f32
            %nbi = arith.select %better, %c, %bi : index
            scf.yield %nbv, %nbi : f32, index
          }}
          %idx_i = arith.index_cast %res#1 : index to i32
          %idx_f = arith.sitofp %idx_i : i32 to f32
          scf.for %c = %z0 to %cN step %one {{
            memref.store %idx_f, %out[%r, %c] : memref<{rows}x{cols}xf32>
          }}
        }}
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{rows}x{cols}xf32>, %arg1: memref<{rows}x{cols}xf32>, %arg2: memref<{rows}x{cols}xf32>) {{
      %t0 = aiex.dma_configure_task_for @in0 {{
        aie.dma_bd(%arg0 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%t0)
      %t2 = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{rows}x{cols}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t0)
    }}
  }}
}}
"#
    )
}

/// Cumulative-scan op along the last axis (matches `Op::Cumsum`/`CumProd`/`CumMax`).
#[derive(Clone, Copy)]
pub enum ScanOp {
    Sum,
    Prod,
    Max,
}

/// Per-row **prefix scan** over the last axis: `out[r,j] = op(x[r, 0..=j])`
/// (inclusive) or `op(x[r, 0..j])` with the identity at `j=0` (exclusive). Scalar
/// f32, whole `[rows,cols]` resident. Sum→`addf`/0, Prod→`mulf`/1, Max→`maximumf`
/// /−∞ (via bitcast; AIE2 lowers `maximumf`, not `minimumf`).
pub fn emit_scan(kind: ScanOp, rows: usize, cols: usize, exclusive: bool) -> String {
    let n = rows * cols;
    let (op, init) = match kind {
        ScanOp::Sum => ("arith.addf", "        %init = arith.constant 0.0 : f32\n".to_string()),
        ScanOp::Prod => ("arith.mulf", "        %init = arith.constant 1.0 : f32\n".to_string()),
        ScanOp::Max => (
            "arith.maximumf",
            "        %ninf_i = arith.constant -8388608 : i32\n        %init = arith.bitcast %ninf_i : i32 to f32\n".to_string(),
        ),
    };
    let inner = if exclusive {
        format!("            memref.store %acc, %out[%f] : memref<{n}xf32>\n            %na = {op} %acc, %v : f32\n            scf.yield %na : f32\n")
    } else {
        format!("            %na = {op} %acc, %v : f32\n            memref.store %na, %out[%f] : memref<{n}xf32>\n            scf.yield %na : f32\n")
    };
    let body = format!(
        "        %rowsN = arith.constant {rows} : index\n        %colsN = arith.constant {cols} : index\n{init}        scf.for %r = %z0 to %rowsN step %one {{\n          %rc = arith.muli %r, %colsN : index\n          %fin = scf.for %j = %z0 to %colsN step %one iter_args(%acc = %init) -> (f32) {{\n            %f = arith.addi %rc, %j : index\n            %v = memref.load %in[%f] : memref<{n}xf32>\n{inner}          }}\n        }}\n"
    );
    dm_unary(n, &body)
}

// ============================================================================
// Data-movement / shape ops (pure-Rust AIE-MLIR). Dtype-agnostic byte moves on
// f32 cells (4 bytes) — index-copy kernels that treat a tensor as [outer, axis,
// inner] and remap the middle axis. arg0 = in, arg2 = out; arg1 unused (dummy).
// ============================================================================

/// Generic axis index-copy: `out[o, a, i] = in[o, src_a(a), i]` over the
/// `[outer, in_axis, inner]` view, where `src` is MLIR computing `%sa` (source
/// axis index, `index` type) from `%a` (output axis index, `0..out_axis`). The
/// engine behind reverse / narrow / slice / tile / expand. `n_in`/`n_out` differ.
fn emit_axis_copy(outer: usize, in_axis: usize, inner: usize, out_axis: usize, src: &str) -> String {
    let n_in = outer * in_axis * inner;
    let n_out = outer * out_axis * inner;
    let in_stride = in_axis * inner;
    let out_stride = out_axis * inner;
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_in = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim_in, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{n_in}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{n_out}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %outerN = arith.constant {outer} : index
      %outaxisN = arith.constant {out_axis} : index
      %innerN = arith.constant {inner} : index
      %instrideC = arith.constant {in_stride} : index
      %outstrideC = arith.constant {out_stride} : index
      %innerC = arith.constant {inner} : index
      scf.for %iter = %z0 to %zmax step %one {{
        %in_sv = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<{n_in}xf32>>
        %in = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<{n_in}xf32>> -> memref<{n_in}xf32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{n_out}xf32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{n_out}xf32>> -> memref<{n_out}xf32>
        scf.for %o = %z0 to %outerN step %one {{
          %oii = arith.muli %o, %instrideC : index
          %ooi = arith.muli %o, %outstrideC : index
          scf.for %a = %z0 to %outaxisN step %one {{
{src}            %sai = arith.muli %sa, %innerC : index
            %ibase = arith.addi %oii, %sai : index
            %oai = arith.muli %a, %innerC : index
            %obase = arith.addi %ooi, %oai : index
            scf.for %i = %z0 to %innerN step %one {{
              %sf = arith.addi %ibase, %i : index
              %of = arith.addi %obase, %i : index
              %v = memref.load %in[%sf] : memref<{n_in}xf32>
              memref.store %v, %out[%of] : memref<{n_out}xf32>
            }}
          }}
        }}
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{n_in}xf32>, %arg1: memref<1xf32>, %arg2: memref<{n_out}xf32>) {{
      %ti = aiex.dma_configure_task_for @in0 {{
        aie.dma_bd(%arg0 : memref<{n_in}xf32>, 0, {n_in}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n_in}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ti)
      %to = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{n_out}xf32>, 0, {n_out}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n_out}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%ti)
    }}
  }}
}}
"#
    )
}

/// `Pad` the axis by `before`/`after` with `fill`: fill the whole output, then
/// copy `in[…,a,…] → out[…,before+a,…]`. Constant mode only.
pub fn emit_pad(outer: usize, in_axis: usize, inner: usize, before: usize, after: usize, fill: f32) -> String {
    emit_pad_impl(outer, in_axis, inner, before, before + in_axis + after, fill)
}

fn emit_pad_impl(outer: usize, in_axis: usize, inner: usize, before: usize, out_axis: usize, fill: f32) -> String {
    let n_in = outer * in_axis * inner;
    let n_out = outer * out_axis * inner;
    let (in_stride, out_stride) = (in_axis * inner, out_axis * inner);
    let boff = before * inner;
    let fillc = fbits("fill", fill);
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_in = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim_in, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{n_in}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{n_out}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %noutN = arith.constant {n_out} : index
      %outerN = arith.constant {outer} : index
      %inaxisN = arith.constant {in_axis} : index
      %innerN = arith.constant {inner} : index
      %instrideC = arith.constant {in_stride} : index
      %outstrideC = arith.constant {out_stride} : index
      %innerC = arith.constant {inner} : index
      %boffC = arith.constant {boff} : index
{fillc}      scf.for %iter = %z0 to %zmax step %one {{
        %in_sv = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<{n_in}xf32>>
        %in = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<{n_in}xf32>> -> memref<{n_in}xf32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{n_out}xf32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{n_out}xf32>> -> memref<{n_out}xf32>
        scf.for %fk = %z0 to %noutN step %one {{
          memref.store %fill, %out[%fk] : memref<{n_out}xf32>
        }}
        scf.for %o = %z0 to %outerN step %one {{
          %oii = arith.muli %o, %instrideC : index
          %ooi = arith.muli %o, %outstrideC : index
          %obb = arith.addi %ooi, %boffC : index
          scf.for %a = %z0 to %inaxisN step %one {{
            %ai = arith.muli %a, %innerC : index
            %ib = arith.addi %oii, %ai : index
            %ob = arith.addi %obb, %ai : index
            scf.for %i = %z0 to %innerN step %one {{
              %sf = arith.addi %ib, %i : index
              %df = arith.addi %ob, %i : index
              %v = memref.load %in[%sf] : memref<{n_in}xf32>
              memref.store %v, %out[%df] : memref<{n_out}xf32>
            }}
          }}
        }}
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{n_in}xf32>, %arg1: memref<1xf32>, %arg2: memref<{n_out}xf32>) {{
      %ti = aiex.dma_configure_task_for @in0 {{
        aie.dma_bd(%arg0 : memref<{n_in}xf32>, 0, {n_in}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n_in}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ti)
      %to = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{n_out}xf32>, 0, {n_out}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n_out}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%ti)
    }}
  }}
}}
"#
    )
}

/// `Reverse` along the axis: `out[…,a,…] = in[…,in_axis-1-a,…]`.
pub fn emit_reverse(outer: usize, axis: usize, inner: usize) -> String {
    let amax = axis - 1;
    let src = format!("            %amax = arith.constant {amax} : index\n            %sa = arith.subi %amax, %a : index\n");
    emit_axis_copy(outer, axis, inner, axis, &src)
}

/// `Narrow` along the axis: `out[…,a,…] = in[…,start+a,…]`, `a in 0..len`.
pub fn emit_narrow(outer: usize, in_axis: usize, inner: usize, start: usize, len: usize) -> String {
    let src = format!("            %startc = arith.constant {start} : index\n            %sa = arith.addi %a, %startc : index\n");
    emit_axis_copy(outer, in_axis, inner, len, &src)
}

/// `Slice` with stride: `out[…,a,…] = in[…,start+a·step,…]` (step may be < 0).
pub fn emit_slice(outer: usize, in_axis: usize, inner: usize, start: usize, len: usize, step: i64) -> String {
    // i32 arithmetic so a negative step works, then cast back to index.
    let src = format!(
        "            %ai = arith.index_cast %a : index to i32\n            %starti = arith.constant {start} : i32\n            %stepi = arith.constant {step} : i32\n            %mul = arith.muli %ai, %stepi : i32\n            %sai32 = arith.addi %mul, %starti : i32\n            %sa = arith.index_cast %sai32 : i32 to index\n"
    );
    emit_axis_copy(outer, in_axis, inner, len, &src)
}

/// `Tile` (repeat) the axis `reps` times: `out[…,a,…] = in[…,a mod in_axis,…]`.
pub fn emit_tile(outer: usize, in_axis: usize, inner: usize, reps: usize) -> String {
    let src = format!("            %axisc = arith.constant {in_axis} : index\n            %sa = arith.remui %a, %axisc : index\n");
    emit_axis_copy(outer, in_axis, inner, in_axis * reps, &src)
}

/// `Expand` (broadcast) a size-1 axis to `out_axis`: `src_a = 0`.
pub fn emit_expand(outer: usize, inner: usize, out_axis: usize) -> String {
    let src = "            %sa = arith.constant 0 : index\n";
    emit_axis_copy(outer, 1, inner, out_axis, src)
}

/// A 1-in / 1-out data-movement core over flat f32 buffers: `body` computes the
/// per-index copy given `%in`/`%out` memrefs and loop bound `%nN`. arg0=in,
/// arg2=out (both `n`), arg1 dummy. Used by transpose/trilu/clamp/cast.
fn dm_unary(n: usize, body: &str) -> String {
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %shim_in = aie.logical_tile<ShimNOCTile>(?, ?)
    %shim_out = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @in0(%shim_in, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{n}xf32>>
    aie.objectfifo @out0(%core, {{%shim_out}}, 2 : i32) : !aie.objectfifo<memref<{n}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %nN = arith.constant {n} : index
      scf.for %iter = %z0 to %zmax step %one {{
        %in_sv = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<{n}xf32>>
        %in = aie.objectfifo.subview.access %in_sv[0] : !aie.objectfifosubview<memref<{n}xf32>> -> memref<{n}xf32>
        %out_sv = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{n}xf32>>
        %out = aie.objectfifo.subview.access %out_sv[0] : !aie.objectfifosubview<memref<{n}xf32>> -> memref<{n}xf32>
{body}        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{n}xf32>, %arg1: memref<1xf32>, %arg2: memref<{n}xf32>) {{
      %ti = aiex.dma_configure_task_for @in0 {{
        aie.dma_bd(%arg0 : memref<{n}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ti)
      %to = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{n}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%ti)
    }}
  }}
}}
"#
    )
}

/// 2-in scalar-f32 single-tile scaffold: `arg0`=A[n], `arg1`=B[n], `arg2`=O[n]
/// (all resident, so n ≤ ~tile memory). Same 2-in ABI as [`emit_binary`] →
/// [`crate::npu_gemm::NpuIo::run2`]. `body` is the `scf.for %i` loop over
/// %A/%B/%O (`memref<{n}xf32>`) + %z0/%one/%nN.
fn dm_binary2(n: usize, body: &str) -> String {
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %sa = aie.logical_tile<ShimNOCTile>(?, ?)
    %sb = aie.logical_tile<ShimNOCTile>(?, ?)
    %so = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @a0(%sa, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{n}xf32>>
    aie.objectfifo @b0(%sb, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{n}xf32>>
    aie.objectfifo @o0(%core, {{%so}}, 2 : i32) : !aie.objectfifo<memref<{n}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %nN = arith.constant {n} : index
      scf.for %iter = %z0 to %zmax step %one {{
        %a_sv = aie.objectfifo.acquire @a0(Consume, 1) : !aie.objectfifosubview<memref<{n}xf32>>
        %A = aie.objectfifo.subview.access %a_sv[0] : !aie.objectfifosubview<memref<{n}xf32>> -> memref<{n}xf32>
        %b_sv = aie.objectfifo.acquire @b0(Consume, 1) : !aie.objectfifosubview<memref<{n}xf32>>
        %B = aie.objectfifo.subview.access %b_sv[0] : !aie.objectfifosubview<memref<{n}xf32>> -> memref<{n}xf32>
        %o_sv = aie.objectfifo.acquire @o0(Produce, 1) : !aie.objectfifosubview<memref<{n}xf32>>
        %O = aie.objectfifo.subview.access %o_sv[0] : !aie.objectfifosubview<memref<{n}xf32>> -> memref<{n}xf32>
{body}        aie.objectfifo.release @a0(Consume, 1)
        aie.objectfifo.release @b0(Consume, 1)
        aie.objectfifo.release @o0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{n}xf32>, %arg1: memref<{n}xf32>, %arg2: memref<{n}xf32>) {{
      %ta = aiex.dma_configure_task_for @a0 {{
        aie.dma_bd(%arg0 : memref<{n}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta)
      %tb = aiex.dma_configure_task_for @b0 {{
        aie.dma_bd(%arg1 : memref<{n}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tb)
      %to = aiex.dma_configure_task_for @o0 {{
        aie.dma_bd(%arg2 : memref<{n}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%ta)
      aiex.dma_free_task(%tb)
    }}
  }}
}}
"#
    )
}

/// Elementwise `Compare`: `O[i] = (A[i] <pred> B[i]) ? 1.0 : 0.0`. `pred` is a
/// `cmpf` predicate (`oeq`/`one`/`olt`/`ole`/`ogt`/`oge`). 2-in ABI = [`emit_binary`].
pub fn emit_compare(pred: &str, n: usize) -> String {
    let body = format!(
        "        %onef = arith.constant 1.0 : f32\n        %zerof = arith.constant 0.0 : f32\n        scf.for %i = %z0 to %nN step %one {{\n          %a = memref.load %A[%i] : memref<{n}xf32>\n          %b = memref.load %B[%i] : memref<{n}xf32>\n          %m = arith.cmpf {pred}, %a, %b : f32\n          %o = arith.select %m, %onef, %zerof : f32\n          memref.store %o, %O[%i] : memref<{n}xf32>\n        }}\n"
    );
    dm_binary2(n, &body)
}

/// 3-in scalar-f32 single-tile scaffold for ops with two extra inputs packed into
/// `arg1`: `arg0`=A[n], `arg1`=P[2n] (two n-length operands B‖C), `arg2`=O[n].
/// `body` is the `scf.for %i` loop using %A/%P/%O + %z0/%one/%nN/%nNi (n as index).
fn dm_ternary_packed(n: usize, body: &str) -> String {
    let n2 = 2 * n;
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %sa = aie.logical_tile<ShimNOCTile>(?, ?)
    %sp = aie.logical_tile<ShimNOCTile>(?, ?)
    %so = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @a0(%sa, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{n}xf32>>
    aie.objectfifo @p0(%sp, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{n2}xf32>>
    aie.objectfifo @o0(%core, {{%so}}, 2 : i32) : !aie.objectfifo<memref<{n}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %nN = arith.constant {n} : index
      %nNi = arith.constant {n} : index
      scf.for %iter = %z0 to %zmax step %one {{
        %a_sv = aie.objectfifo.acquire @a0(Consume, 1) : !aie.objectfifosubview<memref<{n}xf32>>
        %A = aie.objectfifo.subview.access %a_sv[0] : !aie.objectfifosubview<memref<{n}xf32>> -> memref<{n}xf32>
        %p_sv = aie.objectfifo.acquire @p0(Consume, 1) : !aie.objectfifosubview<memref<{n2}xf32>>
        %P = aie.objectfifo.subview.access %p_sv[0] : !aie.objectfifosubview<memref<{n2}xf32>> -> memref<{n2}xf32>
        %o_sv = aie.objectfifo.acquire @o0(Produce, 1) : !aie.objectfifosubview<memref<{n}xf32>>
        %O = aie.objectfifo.subview.access %o_sv[0] : !aie.objectfifosubview<memref<{n}xf32>> -> memref<{n}xf32>
{body}        aie.objectfifo.release @a0(Consume, 1)
        aie.objectfifo.release @p0(Consume, 1)
        aie.objectfifo.release @o0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{n}xf32>, %arg1: memref<{n2}xf32>, %arg2: memref<{n}xf32>) {{
      %ta = aiex.dma_configure_task_for @a0 {{
        aie.dma_bd(%arg0 : memref<{n}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta)
      %tp = aiex.dma_configure_task_for @p0 {{
        aie.dma_bd(%arg1 : memref<{n2}xf32>, 0, {n2}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n2}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tp)
      %to = aiex.dma_configure_task_for @o0 {{
        aie.dma_bd(%arg2 : memref<{n}xf32>, 0, {n}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%ta)
      aiex.dma_free_task(%tp)
    }}
  }}
}}
"#
    )
}

/// Elementwise `Where`: `O[i] = (cond[i] != 0) ? a[i] : b[i]`. `arg0`=cond,
/// `arg1`=a‖b packed (the exec concatenates the two branch inputs). Signed f32.
pub fn emit_where(n: usize) -> String {
    let body = format!(
        "        %zf = arith.constant 0.0 : f32\n        scf.for %i = %z0 to %nN step %one {{\n          %cnd = memref.load %A[%i] : memref<{n}xf32>\n          %ai = memref.load %P[%i] : memref<{}xf32>\n          %ipn = arith.addi %i, %nNi : index\n          %bi = memref.load %P[%ipn] : memref<{}xf32>\n          %m = arith.cmpf une, %cnd, %zf : f32\n          %o = arith.select %m, %ai, %bi : f32\n          memref.store %o, %O[%i] : memref<{n}xf32>\n        }}\n",
        2 * n,
        2 * n
    );
    dm_ternary_packed(n, &body)
}

/// Elementwise fused multiply-add `Fma`: `O[i] = a[i]*b[i] + c[i]`. `arg0`=a,
/// `arg1`=b‖c packed. (AIE2 has no lowering `math.fma` → explicit mul+add.)
pub fn emit_fma(n: usize) -> String {
    let body = format!(
        "        scf.for %i = %z0 to %nN step %one {{\n          %ai = memref.load %A[%i] : memref<{n}xf32>\n          %bi = memref.load %P[%i] : memref<{}xf32>\n          %ipn = arith.addi %i, %nNi : index\n          %ci = memref.load %P[%ipn] : memref<{}xf32>\n          %ab = arith.mulf %ai, %bi : f32\n          %o = arith.addf %ab, %ci : f32\n          memref.store %o, %O[%i] : memref<{n}xf32>\n        }}\n",
        2 * n,
        2 * n
    );
    dm_ternary_packed(n, &body)
}

/// 2-D `Transpose`: `out[j, i] = in[i, j]` (`[rows, cols] → [cols, rows]`).
pub fn emit_transpose2d(rows: usize, cols: usize) -> String {
    let n = rows * cols;
    let body = format!(
        "        %rowsN = arith.constant {rows} : index\n        %colsN = arith.constant {cols} : index\n        scf.for %i = %z0 to %rowsN step %one {{\n          %ic = arith.muli %i, %colsN : index\n          scf.for %j = %z0 to %colsN step %one {{\n            %inf = arith.addi %ic, %j : index\n            %jr = arith.muli %j, %rowsN : index\n            %outf = arith.addi %jr, %i : index\n            %v = memref.load %in[%inf] : memref<{n}xf32>\n            memref.store %v, %out[%outf] : memref<{n}xf32>\n          }}\n        }}\n"
    );
    dm_unary(n, &body)
}

/// `Trilu` on the last two axes `[rows, cols]`: keep the upper (or lower)
/// triangle relative to `diagonal`, zero the rest. `keep = j ≥ i+diag` (upper)
/// or `j ≤ i+diag` (lower).
pub fn emit_trilu(rows: usize, cols: usize, upper: bool, diagonal: i64) -> String {
    let n = rows * cols;
    let cmp = if upper { "sge" } else { "sle" };
    let body = format!(
        "        %rowsN = arith.constant {rows} : index\n        %colsN = arith.constant {cols} : index\n        %diag = arith.constant {diagonal} : i32\n        %zf = arith.constant 0.0 : f32\n        scf.for %i = %z0 to %rowsN step %one {{\n          %ic = arith.muli %i, %colsN : index\n          %ii32 = arith.index_cast %i : index to i32\n          %thr = arith.addi %ii32, %diag : i32\n          scf.for %j = %z0 to %colsN step %one {{\n            %f = arith.addi %ic, %j : index\n            %ji32 = arith.index_cast %j : index to i32\n            %keep = arith.cmpi {cmp}, %ji32, %thr : i32\n            %v = memref.load %in[%f] : memref<{n}xf32>\n            %o = arith.select %keep, %v, %zf : f32\n            memref.store %o, %out[%f] : memref<{n}xf32>\n          }}\n        }}\n"
    );
    dm_unary(n, &body)
}

/// Elementwise `Clamp(x, lo, hi)` = `min(max(x, lo), hi)` (min via −max).
pub fn emit_clamp(n: usize, lo: f32, hi: f32) -> String {
    let (loc, hic) = (fbits("lo", lo), fbits("hi", hi));
    let body = format!(
        "{loc}{hic}        scf.for %k = %z0 to %nN step %one {{\n          %v = memref.load %in[%k] : memref<{n}xf32>\n          %mx = arith.maximumf %v, %lo : f32\n          %nmx = arith.negf %mx : f32\n          %nhi = arith.negf %hi : f32\n          %mm = arith.maximumf %nmx, %nhi : f32\n          %o = arith.negf %mm : f32\n          memref.store %o, %out[%k] : memref<{n}xf32>\n        }}\n"
    );
    // fbits emits at 10-space indent; dm_unary body sits at 8 — fine (MLIR ignores).
    dm_unary(n, &body)
}

/// `Cast` between f32 and i32 (the two dtypes that share the 4-byte cell).
/// `f2i` = f32→i32 (`fptosi`), else i32→f32 (`sitofp`). The buffers stay f32
/// memrefs; the caller reinterprets the produced/consumed i32 bits.
pub fn emit_cast(n: usize, f2i: bool) -> String {
    let conv = if f2i {
        // in reinterpreted as f32, out stored as i32 bits.
        "          %v = memref.load %in[%k] : memref<{n}xf32>\n          %iv = arith.fptosi %v : f32 to i32\n          %ob = arith.bitcast %iv : i32 to f32\n          memref.store %ob, %out[%k] : memref<{n}xf32>\n"
    } else {
        // in reinterpreted as i32 bits, out stored as f32.
        "          %vb = memref.load %in[%k] : memref<{n}xf32>\n          %iv = arith.bitcast %vb : f32 to i32\n          %o = arith.sitofp %iv : i32 to f32\n          memref.store %o, %out[%k] : memref<{n}xf32>\n"
    };
    let body = format!("        scf.for %k = %z0 to %nN step %one {{\n{conv}        }}\n").replace("{n}", &n.to_string());
    dm_unary(n, &body)
}

/// `Concat` of two tensors along the axis: `out = [A ‖ B]` over `[outer, ·,
/// inner]`. arg0=A, arg1=B, arg2=out (all distinct sizes) → `NpuRun3`.
pub fn emit_concat2(outer: usize, a_axis: usize, b_axis: usize, inner: usize) -> String {
    let (na, nb) = (outer * a_axis * inner, outer * b_axis * inner);
    let out_axis = a_axis + b_axis;
    let nc = outer * out_axis * inner;
    let (as_, bs, os) = (a_axis * inner, b_axis * inner, out_axis * inner);
    let aoff = a_axis * inner;
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %sa = aie.logical_tile<ShimNOCTile>(?, ?)
    %sb = aie.logical_tile<ShimNOCTile>(?, ?)
    %so = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @a0(%sa, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{na}xf32>>
    aie.objectfifo @b0(%sb, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{nb}xf32>>
    aie.objectfifo @o0(%core, {{%so}}, 2 : i32) : !aie.objectfifo<memref<{nc}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %outerN = arith.constant {outer} : index
      %aAxisN = arith.constant {a_axis} : index
      %bAxisN = arith.constant {b_axis} : index
      %innerN = arith.constant {inner} : index
      %asC = arith.constant {as_} : index
      %bsC = arith.constant {bs} : index
      %osC = arith.constant {os} : index
      %aoffC = arith.constant {aoff} : index
      %innerC = arith.constant {inner} : index
      scf.for %iter = %z0 to %zmax step %one {{
        %a_sv = aie.objectfifo.acquire @a0(Consume, 1) : !aie.objectfifosubview<memref<{na}xf32>>
        %A = aie.objectfifo.subview.access %a_sv[0] : !aie.objectfifosubview<memref<{na}xf32>> -> memref<{na}xf32>
        %b_sv = aie.objectfifo.acquire @b0(Consume, 1) : !aie.objectfifosubview<memref<{nb}xf32>>
        %B = aie.objectfifo.subview.access %b_sv[0] : !aie.objectfifosubview<memref<{nb}xf32>> -> memref<{nb}xf32>
        %o_sv = aie.objectfifo.acquire @o0(Produce, 1) : !aie.objectfifosubview<memref<{nc}xf32>>
        %O = aie.objectfifo.subview.access %o_sv[0] : !aie.objectfifosubview<memref<{nc}xf32>> -> memref<{nc}xf32>
        scf.for %o = %z0 to %outerN step %one {{
          %oosc = arith.muli %o, %osC : index
          %oas = arith.muli %o, %asC : index
          scf.for %a = %z0 to %aAxisN step %one {{
            %ai = arith.muli %a, %innerC : index
            %ib = arith.addi %oas, %ai : index
            %ob = arith.addi %oosc, %ai : index
            scf.for %i = %z0 to %innerN step %one {{
              %sf = arith.addi %ib, %i : index
              %df = arith.addi %ob, %i : index
              %v = memref.load %A[%sf] : memref<{na}xf32>
              memref.store %v, %O[%df] : memref<{nc}xf32>
            }}
          }}
          %obs = arith.muli %o, %bsC : index
          scf.for %a = %z0 to %bAxisN step %one {{
            %ai = arith.muli %a, %innerC : index
            %ib = arith.addi %obs, %ai : index
            %aoi = arith.addi %aoffC, %ai : index
            %ob = arith.addi %oosc, %aoi : index
            scf.for %i = %z0 to %innerN step %one {{
              %sf = arith.addi %ib, %i : index
              %df = arith.addi %ob, %i : index
              %v = memref.load %B[%sf] : memref<{nb}xf32>
              memref.store %v, %O[%df] : memref<{nc}xf32>
            }}
          }}
        }}
        aie.objectfifo.release @a0(Consume, 1)
        aie.objectfifo.release @b0(Consume, 1)
        aie.objectfifo.release @o0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{na}xf32>, %arg1: memref<{nb}xf32>, %arg2: memref<{nc}xf32>) {{
      %ta = aiex.dma_configure_task_for @a0 {{
        aie.dma_bd(%arg0 : memref<{na}xf32>, 0, {na}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {na}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta)
      %tb = aiex.dma_configure_task_for @b0 {{
        aie.dma_bd(%arg1 : memref<{nb}xf32>, 0, {nb}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {nb}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tb)
      %to = aiex.dma_configure_task_for @o0 {{
        aie.dma_bd(%arg2 : memref<{nc}xf32>, 0, {nc}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {nc}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%ta)
      aiex.dma_free_task(%tb)
    }}
  }}
}}
"#
    )
}

/// `Gather` along the axis: `out[o, j, i] = data[o, idx[j], i]`, `j in 0..num_idx`.
/// arg0=data, arg1=idx (f32-encoded indices), arg2=out → `NpuRun3`.
pub fn emit_gather(outer: usize, in_axis: usize, inner: usize, num_idx: usize) -> String {
    let n_data = outer * in_axis * inner;
    let n_out = outer * num_idx * inner;
    let (in_stride, out_stride) = (in_axis * inner, num_idx * inner);
    format!(
        r#"module {{
  aie.device(npu1_1col) {{
    %core = aie.logical_tile<CoreTile>(?, ?)
    %sd = aie.logical_tile<ShimNOCTile>(?, ?)
    %si = aie.logical_tile<ShimNOCTile>(?, ?)
    %so = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.objectfifo @d0(%sd, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{n_data}xf32>>
    aie.objectfifo @i0(%si, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{num_idx}xf32>>
    aie.objectfifo @o0(%core, {{%so}}, 2 : i32) : !aie.objectfifo<memref<{n_out}xf32>>
    %0 = aie.core(%core) {{
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %outerN = arith.constant {outer} : index
      %numN = arith.constant {num_idx} : index
      %innerN = arith.constant {inner} : index
      %instrideC = arith.constant {in_stride} : index
      %outstrideC = arith.constant {out_stride} : index
      %innerC = arith.constant {inner} : index
      scf.for %iter = %z0 to %zmax step %one {{
        %d_sv = aie.objectfifo.acquire @d0(Consume, 1) : !aie.objectfifosubview<memref<{n_data}xf32>>
        %D = aie.objectfifo.subview.access %d_sv[0] : !aie.objectfifosubview<memref<{n_data}xf32>> -> memref<{n_data}xf32>
        %i_sv = aie.objectfifo.acquire @i0(Consume, 1) : !aie.objectfifosubview<memref<{num_idx}xf32>>
        %IDX = aie.objectfifo.subview.access %i_sv[0] : !aie.objectfifosubview<memref<{num_idx}xf32>> -> memref<{num_idx}xf32>
        %o_sv = aie.objectfifo.acquire @o0(Produce, 1) : !aie.objectfifosubview<memref<{n_out}xf32>>
        %O = aie.objectfifo.subview.access %o_sv[0] : !aie.objectfifosubview<memref<{n_out}xf32>> -> memref<{n_out}xf32>
        scf.for %o = %z0 to %outerN step %one {{
          %oii = arith.muli %o, %instrideC : index
          %ooi = arith.muli %o, %outstrideC : index
          scf.for %j = %z0 to %numN step %one {{
            %ixf = memref.load %IDX[%j] : memref<{num_idx}xf32>
            %ixi = arith.fptosi %ixf : f32 to i32
            %ix = arith.index_cast %ixi : i32 to index
            %sai = arith.muli %ix, %innerC : index
            %ib = arith.addi %oii, %sai : index
            %oaj = arith.muli %j, %innerC : index
            %ob = arith.addi %ooi, %oaj : index
            scf.for %i = %z0 to %innerN step %one {{
              %sf = arith.addi %ib, %i : index
              %df = arith.addi %ob, %i : index
              %v = memref.load %D[%sf] : memref<{n_data}xf32>
              memref.store %v, %O[%df] : memref<{n_out}xf32>
            }}
          }}
        }}
        aie.objectfifo.release @d0(Consume, 1)
        aie.objectfifo.release @i0(Consume, 1)
        aie.objectfifo.release @o0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{n_data}xf32>, %arg1: memref<{num_idx}xf32>, %arg2: memref<{n_out}xf32>) {{
      %td = aiex.dma_configure_task_for @d0 {{
        aie.dma_bd(%arg0 : memref<{n_data}xf32>, 0, {n_data}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n_data}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%td)
      %tidx = aiex.dma_configure_task_for @i0 {{
        aie.dma_bd(%arg1 : memref<{num_idx}xf32>, 0, {num_idx}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {num_idx}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tidx)
      %to = aiex.dma_configure_task_for @o0 {{
        aie.dma_bd(%arg2 : memref<{n_out}xf32>, 0, {n_out}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n_out}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%td)
      aiex.dma_free_task(%tidx)
    }}
  }}
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_mlir_shape() {
        let m = emit_passthrough(4096, 1024);
        assert!(m.contains("aie.device(npu1_1col)"));
        assert!(m.contains("aie.objectfifo @in"));
        assert!(m.contains("aie.objectfifo.link [@in] -> [@in_fwd]"));
        assert!(m.contains("memref<4096xi32>"));
        assert!(m.contains("memref<1024xi32>"));
        assert!(m.contains("aie.runtime_sequence"));
    }

    #[test]
    fn eltwise_mlir_shape() {
        // 8192 elems streamed through 1024-i32 tile buffers.
        let m = emit_eltwise(8192, 1024, Eltwise::Relu);
        assert!(m.contains("aie.core("));
        assert!(m.contains("arith.maxsi")); // int8 relu = max(x, 0) on the vector ALU
        assert!(m.contains("memref<1024xi32>")); // tile buffer = chunk
        assert!(m.contains("memref<8192xi32>")); // host buffer = n
        assert!(m.contains("aie.objectfifo @in0"));
        assert!(m.contains("aie.objectfifo @out0"));
        assert_eq!(Eltwise::AddScalar(5).apply(3), 8);
        assert_eq!(Eltwise::Relu.apply(-2), 0);
    }
}
