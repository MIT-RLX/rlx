// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

use std::collections::HashMap;

// ── Step enum ─────────────────────────────────────────────────────────
// Copy of `rlx-cuda::backend::Step` — same variants, same fields.
// Kept private to the crate; the public surface is `RocmExecutable`.

#[derive(Clone)]
pub(crate) enum Step {
    Matmul {
        m: u32,
        k: u32,
        n: u32,
        a_off_f32: u32,
        b_off_f32: u32,
        c_off_f32: u32,
        batch: u32,
        a_batch_stride: u32,
        b_batch_stride: u32,
        c_batch_stride: u32,
        has_bias: u32,
        bias_off_f32: u32,
        act_id: u32,
    },
    /// Native FP8 (FNUZ) tensor-core GEMM via hipBLASLt. TN: lhs[m,k]·rhs[n,k]ᵀ.
    /// All offsets are BYTES (codes u8; scales/out/bias f32).
    ScaledMatMul {
        m: u32,
        k: u32,
        n: u32,
        lhs_byte_off: u32,
        rhs_byte_off: u32,
        lhs_scale_byte_off: u32,
        rhs_scale_byte_off: u32,
        out_byte_off: u32,
        has_bias: u32,
        bias_byte_off: u32,
        lhs_e5m2: u32,
        rhs_e5m2: u32,
    },
    /// Per-tensor amax → f32 scale for a tensor about to be FP8-quantized.
    ScaledQuantScale {
        x_off_f32: u32,
        scale_off_f32: u32,
        n: u32,
        max_finite: f32,
    },
    /// Encode f32 → FP8 codes (per-tensor scale). `e5m2`: 0=E4M3, 1=E5M2.
    ScaledQuantizeFp8 {
        x_off_f32: u32,
        scale_off_f32: u32,
        out_byte_off: u32,
        n: u32,
        e5m2: u32,
    },
    /// Decode-and-accumulate GEMM fallback (non-tensor-core) for block / FP4 /
    /// FP6 configs hipBLASLt can't do.
    ScaledMatMulDecode {
        m: u32,
        k: u32,
        n: u32,
        lhs_byte_off: u32,
        rhs_byte_off: u32,
        lhs_scale_byte_off: u32,
        rhs_scale_byte_off: u32,
        out_off_f32: u32,
        lhs_fmt: u32,
        rhs_fmt: u32,
        scale_mode: u32,
        block: u32,
        has_bias: u32,
        bias_off_f32: u32,
    },
    /// General (all-format/all-layout) scale producer.
    ScaledQuantScaleGeneral {
        x_off_f32: u32,
        scale_byte_off: u32,
        rows: u32,
        cols: u32,
        fmt: u32,
        scale_mode: u32,
        block: u32,
    },
    /// General (all-format/all-layout) quantize producer.
    ScaledQuantizeGeneral {
        x_off_f32: u32,
        scale_byte_off: u32,
        out_byte_off: u32,
        rows: u32,
        cols: u32,
        fmt: u32,
        scale_mode: u32,
        block: u32,
    },
    ScaledDequantizeGeneral {
        codes_byte_off: u32,
        scale_byte_off: u32,
        out_off_f32: u32,
        rows: u32,
        cols: u32,
        fmt: u32,
        scale_mode: u32,
        block: u32,
    },
    Binary {
        n: u32,
        a_off: u32,
        b_off: u32,
        c_off: u32,
        op: u32,
    },
    Compare {
        n: u32,
        a_off: u32,
        b_off: u32,
        c_off: u32,
        op: u32,
    },
    Unary {
        n: u32,
        in_off: u32,
        out_off: u32,
        op: u32,
    },
    Where {
        n: u32,
        cond_off: u32,
        x_off: u32,
        y_off: u32,
        out_off: u32,
    },
    /// Element-wise fused multiply-add: `out = a * b + c` (single rounding).
    Fma {
        n: u32,
        a_off: u32,
        b_off: u32,
        c_off: u32,
        out_off: u32,
    },
    Reduce {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
        op: u32,
    },
    Softmax {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
    },
    ReluBackward {
        n: u32,
        x_off: u32,
        dy_off: u32,
        dx_off: u32,
    },
    ActivationBackward {
        n: u32,
        x_off: u32,
        dy_off: u32,
        dx_off: u32,
        op: u32,
    },
    SoftmaxCrossEntropy {
        outer: u32,
        inner: u32,
        logits_off: u32,
        targets_off: u32,
        out_off: u32,
    },
    SoftmaxCrossEntropyWithLogits {
        outer: u32,
        inner: u32,
        logits_off: u32,
        labels_off: u32,
        out_off: u32,
    },
    SoftmaxCrossEntropyBackward {
        outer: u32,
        inner: u32,
        logits_off: u32,
        labels_off: u32,
        d_loss_off: u32,
        out_off: u32,
    },
    LayerNorm {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
        gamma_off: u32,
        beta_off: u32,
        eps_bits: u32,
        op: u32,
    },
    FusedResidualLn {
        outer: u32,
        inner: u32,
        in_off: u32,
        residual_off: u32,
        bias_off: u32,
        gamma_off: u32,
        beta_off: u32,
        out_off: u32,
        eps_bits: u32,
        has_bias: u32,
    },
    AdaLayerNorm {
        outer: u32,
        inner: u32,
        in_off: u32,
        scale_off: u32,
        shift_off: u32,
        out_off: u32,
        eps_bits: u32,
        layer_norm: u32,
        meta_idx: usize,
    },
    GatedResidual {
        total: u32,
        inner: u32,
        x_off: u32,
        y_off: u32,
        gate_off: u32,
        out_off: u32,
        meta_idx: usize,
    },
    AdaLayerNormBackward {
        mod_rows: u32,
        seq_per_mod: u32,
        inner: u32,
        x_off: u32,
        scale_off: u32,
        dy_off: u32,
        out_off: u32,
        eps_bits: u32,
        layer_norm: u32,
    },
    GatedResidualBackward {
        mod_rows: u32,
        seq_per_mod: u32,
        inner: u32,
        y_off: u32,
        gate_off: u32,
        dy_off: u32,
        out_off: u32,
    },
    Gather {
        n_out: u32,
        n_idx: u32,
        dim: u32,
        vocab: u32,
        in_off: u32,
        idx_off: u32,
        out_off: u32,
    },
    GatherAxis {
        total: u32,
        outer: u32,
        axis_dim: u32,
        num_idx: u32,
        trailing: u32,
        table_off: u32,
        idx_off: u32,
        out_off: u32,
    },
    Narrow {
        total: u32,
        outer: u32,
        inner: u32,
        axis_in_size: u32,
        axis_out_size: u32,
        start: u32,
        in_off: u32,
        out_off: u32,
    },
    Argmax {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
    },
    Transpose {
        rank: u32,
        out_total: u32,
        in_off: u32,
        out_off: u32,
        meta_idx: usize,
    },
    Expand {
        rank: u32,
        out_total: u32,
        in_off: u32,
        out_off: u32,
        meta_idx: usize,
    },
    Concat {
        total: u32,
        outer: u32,
        inner: u32,
        axis_in_size: u32,
        axis_out_size: u32,
        start: u32,
        in_off: u32,
        out_off: u32,
    },
    Attention {
        batch: u32,
        heads: u32,
        seq_q: u32,
        seq_k: u32,
        head_dim: u32,
        q_off: u32,
        k_off: u32,
        v_off: u32,
        out_off: u32,
        mask_off: u32,
        mask_kind: u32,
        scale_bits: u32,
        softcap_bits: u32,
        window: u32,
        seq_q_stride: u32,
        seq_k_stride: u32,
        mask_batch_stride: u32,
        mask_head_stride: u32,
        q_batch_stride: u32,
        q_head_stride: u32,
        q_seq_stride: u32,
        k_batch_stride: u32,
        k_head_stride: u32,
        k_seq_stride: u32,
        v_batch_stride: u32,
        v_head_stride: u32,
        v_seq_stride: u32,
        o_batch_stride: u32,
        o_head_stride: u32,
        o_seq_stride: u32,
    },
    AttentionBackward {
        batch: u32,
        heads: u32,
        seq_q: u32,
        seq_k: u32,
        head_dim: u32,
        q_off: u32,
        k_off: u32,
        v_off: u32,
        dy_off: u32,
        out_off: u32,
        mask_off: u32,
        mask_kind: u32,
        scale_bits: u32,
        window: u32,
        wrt: u32,
    },
    Rope {
        n_total: u32,
        seq: u32,
        head_dim: u32,
        half: u32,
        in_off: u32,
        cos_off: u32,
        sin_off: u32,
        out_off: u32,
        last_dim: u32,
        interleaved: u32,
    },
    Cumsum {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
        exclusive: u32,
    },
    TopK {
        outer: u32,
        inner: u32,
        k: u32,
        in_off: u32,
        out_off: u32,
    },
    GroupedMatmul {
        m: u32,
        k: u32,
        n: u32,
        num_experts: u32,
        in_off: u32,
        w_off: u32,
        idx_off: u32,
        out_off: u32,
    },
    ScatterAddZero {
        out_off: u32,
        out_total: u32,
    },
    ScatterAddAcc {
        out_off: u32,
        upd_off: u32,
        idx_off: u32,
        num_updates: u32,
        trailing: u32,
        out_dim: u32,
    },
    DequantMatmul {
        m: u32,
        k: u32,
        n: u32,
        block_size: u32,
        scheme_id: u32,
        x_off: u32,
        w_off: u32,
        scale_off: u32,
        zp_off: u32,
        out_off: u32,
    },
    DequantMatmulGguf {
        m: u32,
        k: u32,
        n: u32,
        scheme_id: u32,
        x_byte_off: u32,
        w_byte_off: u32,
        out_byte_off: u32,
    },
    DequantGroupedMatmulGguf {
        m: u32,
        k: u32,
        n: u32,
        num_experts: u32,
        scheme_id: u32,
        x_byte_off: u32,
        w_byte_off: u32,
        idx_byte_off: u32,
        out_byte_off: u32,
    },
    Sample {
        outer: u32,
        inner: u32,
        in_off: u32,
        out_off: u32,
        top_k: u32,
        top_p_bits: u32,
        temp_bits: u32,
        seed_lo: u32,
        seed_hi: u32,
    },
    RngNormal {
        dst_byte_off: u32,
        len: u32,
        mean: f32,
        scale: f32,
        key: u64,
        op_seed: Option<f32>,
    },
    RngUniform {
        dst_byte_off: u32,
        len: u32,
        low: f32,
        high: f32,
        key: u64,
        op_seed: Option<f32>,
    },
    SelectiveScan {
        batch: u32,
        seq: u32,
        hidden: u32,
        state_size: u32,
        x_off: u32,
        delta_off: u32,
        a_off: u32,
        b_off: u32,
        c_off: u32,
        out_off: u32,
    },
    /// Gated-DeltaNet — host scan between GPU segments.
    Fft {
        src_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        n_complex: u32,
        inverse: bool,
        norm_tag: u32,
        dtype_tag: u32,
        use_gpu: bool,
    },
    LogMelHost {
        spec_byte_off: u32,
        filt_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        n_fft: u32,
        n_bins: u32,
        n_mels: u32,
    },
    LogMelBackwardHost {
        spec_byte_off: u32,
        filt_byte_off: u32,
        dy_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        n_fft: u32,
        n_bins: u32,
        n_mels: u32,
    },
    WelchPeaksHost {
        spec_byte_off: u32,
        dst_byte_off: u32,
        welch_batch: u32,
        n_fft: u32,
        n_segments: u32,
        k: u32,
    },
    /// Native GPU WelchPeaks (in-arena, no D2H).
    WelchPeaksGpu {
        spec_off: u32,
        dst_off: u32,
        welch_batch: u32,
        n_fft: u32,
        n_segments: u32,
        k: u32,
        n_bins: u32,
    },
    /// Ternary-pruned radix-2 butterfly stage (interleaved C64).
    FftButterflyStage {
        state_off: u32,
        out_off: u32,
        gate_off: u32,
        rev_off: u32,
        tw_re_off: u32,
        tw_im_off: u32,
        batch: u32,
        n_fft: u32,
        stage: u32,
    },
    Im2ColHost {
        x_byte_off: u32,
        col_byte_off: u32,
        n: u32,
        c_in: u32,
        h: u32,
        w: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw_dil: u32,
        use_gpu: bool,
    },
    /// Host-staged batch-general reverse/flip.
    ReverseHost {
        src_byte_off: u32,
        dst_byte_off: u32,
        dims: Vec<u32>,
        rev_mask: Vec<bool>,
        elem_bytes: u32,
    },
    /// Host-staged ArgMax/ArgMin (f32-encoded indices).
    ArgReduceHost {
        src_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        reduced: u32,
        inner: u32,
        is_max: bool,
    },
    /// Native axial 2-D RoPE (SAM2-style).
    AxialRope2d {
        in_off: u32,
        out_off: u32,
        batch: u32,
        seq: u32,
        hidden: u32,
        end_x: u32,
        end_y: u32,
        head_dim: u32,
        num_heads: u32,
        theta: f32,
        repeat_factor: u32,
    },
    GatedDeltaNet {
        q_byte_off: u32,
        k_byte_off: u32,
        v_byte_off: u32,
        g_byte_off: u32,
        beta_byte_off: u32,
        state_byte_off: u32,
        dst_byte_off: u32,
        batch: u32,
        seq: u32,
        heads: u32,
        state_size: u32,
        use_carry: bool,
    },
    Lstm {
        x_byte_off: u32,
        w_ih_byte_off: u32,
        w_hh_byte_off: u32,
        bias_byte_off: u32,
        h0_byte_off: u32,
        c0_byte_off: u32,
        dst_byte_off: u32,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
        num_layers: u32,
        bidirectional: bool,
        carry: bool,
    },
    /// Native ROCm GRU (L=1 / unidir / no-carry / hidden ≤ 1024).
    Gru {
        x_byte_off: u32,
        w_ih_byte_off: u32,
        w_hh_byte_off: u32,
        b_ih_byte_off: u32,
        b_hh_byte_off: u32,
        dst_byte_off: u32,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
    },
    /// Host-staged GRU fallback (multi-layer / bidir / carry / hidden > 1024).
    GruHost {
        x_byte_off: u32,
        w_ih_byte_off: u32,
        w_hh_byte_off: u32,
        b_ih_byte_off: u32,
        b_hh_byte_off: u32,
        h0_byte_off: u32,
        dst_byte_off: u32,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
        num_layers: u32,
        bidirectional: bool,
        carry: bool,
    },
    /// Native ROCm Elman RNN (L=1 / unidir / no-carry / hidden ≤ 1024).
    Rnn {
        x_byte_off: u32,
        w_ih_byte_off: u32,
        w_hh_byte_off: u32,
        bias_byte_off: u32,
        dst_byte_off: u32,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
        relu: bool,
    },
    /// Host-staged Elman RNN fallback.
    RnnHost {
        x_byte_off: u32,
        w_ih_byte_off: u32,
        w_hh_byte_off: u32,
        bias_byte_off: u32,
        h0_byte_off: u32,
        dst_byte_off: u32,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
        num_layers: u32,
        bidirectional: bool,
        carry: bool,
        relu: bool,
    },
    /// Native ROCm Mamba-2 SSD scan (`state_size ≤ 256`).
    Mamba2 {
        x_byte_off: u32,
        dt_byte_off: u32,
        a_byte_off: u32,
        b_byte_off: u32,
        c_byte_off: u32,
        dst_byte_off: u32,
        batch: u32,
        seq: u32,
        heads: u32,
        head_dim: u32,
        state_size: u32,
    },
    /// Host-staged Mamba-2 fallback (`state_size > 256` or force-host).
    Mamba2Host {
        x_byte_off: u32,
        dt_byte_off: u32,
        a_byte_off: u32,
        b_byte_off: u32,
        c_byte_off: u32,
        dst_byte_off: u32,
        batch: u32,
        seq: u32,
        heads: u32,
        head_dim: u32,
        state_size: u32,
    },
    /// General `Op::Scan` recurrence (e.g. IIR biquad) via host fallback
    /// (D2H → CPU body loop → H2D). Not graph-capture-safe.
    ScanHost {
        desc: rlx_cpu::thunk::ScanHostDesc,
    },
    HostOp {
        desc: rlx_cpu::thunk::HostOpDesc,
    },
    /// Native CPU ScatterNd / ScatterElements / GatherNd / GatherElements
    /// via full-arena D2H (correct for `I64` indices; no mini-graph rebuild).
    CpuIndexing {
        thunk: rlx_cpu::thunk::IndexingThunk,
    },
    /// Core Riemannian / SPD-manifold op (`Op::BiMap`, `ReEig`, `LogEig`,
    /// `SpdBatchNorm`, `SpdKarcherMean`, and their backwards) via host fallback
    /// (D2H → CPU F64 reference → H2D; see [`crate::spd_host`]). No ROCm
    /// eigendecomposition kernel; runs the exact `rlx-cpu` thunk kernels. Not
    /// graph-capture-safe (forces eager + a stream sync).
    SpdHost {
        op: rlx_ir::Op,
        out_off: usize,
        out_shape: rlx_ir::Shape,
        /// `(f32_offset, declared_shape)` per operand, in graph-input order.
        inputs: Vec<(usize, rlx_ir::Shape)>,
    },
    /// Native batched symmetric eigendecomposition (`Op::Eigh` / `Op::EighBatch`,
    /// `n ≤ 32`) via hipSOLVER `SsyevjBatched` — on-device, no host round-trip.
    /// `in_off` = input `A [batch·n·n]`, `out_off` = packed `[batch·(n·n+n)]`,
    /// both f32 arena offsets. See [`crate::eigh_native`]. Allocates scratch, so
    /// eager (not hipGraph-capture-safe).
    EighNative {
        in_off: usize,
        out_off: usize,
        n: usize,
        batch: usize,
    },
    /// Native dense linear solve (`Op::DenseSolve`, F32) via hipSOLVER
    /// `Sgetrf`+`Sgetrs`. See [`crate::dense_solve_native`]. Eager (scratch).
    DenseSolveNative {
        a_off: usize,
        b_off: usize,
        x_off: usize,
        n: usize,
        nrhs: usize,
    },
    /// Native batched dense solve (`Op::BatchedDenseSolve`, F32) via hipBLAS
    /// `SgetrfBatched`+`SgetrsBatched`. See [`crate::dense_solve_native`].
    BatchedDenseSolveNative {
        a_off: usize,
        b_off: usize,
        x_off: usize,
        batch: usize,
        n: usize,
        nrhs: usize,
    },
    Llada2GroupLimitedGate {
        sig_off: u32,
        route_off: u32,
        out_off: u32,
        n_elems: u32,
        attrs: [u8; 20],
    },
    /// Fused multi-scale deformable attention (host compute between GPU segments).
    MsDeformAttnHost {
        in_offs: Vec<(u32, u32)>, // (f32_off, f32_len) per input
        out_off: u32,
        out_len: u32,
        attrs: Vec<u8>,
    },
    /// Host-delegate `collective.*` op (all_reduce / all_gather / reduce_scatter
    /// / Megatron f/g). Staged off-GPU and run through the registered rlx-cpu
    /// collective kernel. See `crate::collective_host`. Offsets/lengths are in
    /// f32 elements (ROCm arena convention).
    CollectiveHost {
        name: String,
        in_off: u32,
        in_len: u32,
        out_off: u32,
        out_len: u32,
        attrs: Vec<u8>,
    },
    UmapKnn {
        pairwise_off: u32,
        out_off: u32,
        n: u32,
        k: u32,
    },
    /// Raw-GPU custom op (`RocmGpuKernel`): hipRTC-compiled HIP-C launched
    /// against the arena buffer, no host roundtrip. Offsets baked at compile
    /// time; the kernel is resolved by `name` (hipRTC-compiled on first launch).
    RocmGpuKernel {
        name: String,
        out_off: u32,
        out_len: u32,
        in_offs: Vec<(u32, u32)>, // (f32_off, f32_len) per input, ≤ MAX_INPUTS
    },
    GaussianSplatRender {
        positions_off: u32,
        positions_len: u32,
        scales_off: u32,
        scales_len: u32,
        rotations_off: u32,
        rotations_len: u32,
        opacities_off: u32,
        opacities_len: u32,
        colors_off: u32,
        colors_len: u32,
        sh_coeffs_off: u32,
        sh_coeffs_len: u32,
        meta_off: u32,
        dst_off: u32,
        dst_len: u32,
        width: u32,
        height: u32,
        tile_size: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
    },
    GaussianSplatRenderBackward {
        positions_off: u32,
        positions_len: u32,
        scales_off: u32,
        scales_len: u32,
        rotations_off: u32,
        rotations_len: u32,
        opacities_off: u32,
        opacities_len: u32,
        colors_off: u32,
        colors_len: u32,
        sh_coeffs_off: u32,
        sh_coeffs_len: u32,
        meta_off: u32,
        d_loss_off: u32,
        d_loss_len: u32,
        packed_off: u32,
        packed_len: u32,
        width: u32,
        height: u32,
        tile_size: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
        loss_grad_clip: f32,
        sh_band: u32,
        max_anisotropy: f32,
    },
    GaussianSplatPrepare {
        positions_off: u32,
        positions_len: u32,
        scales_off: u32,
        scales_len: u32,
        rotations_off: u32,
        rotations_len: u32,
        opacities_off: u32,
        opacities_len: u32,
        colors_off: u32,
        colors_len: u32,
        sh_coeffs_off: u32,
        sh_coeffs_len: u32,
        meta_off: u32,
        meta_len: u32,
        prep_off: u32,
        prep_len: u32,
        width: u32,
        height: u32,
        tile_size: u32,
        radius_scale: f32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
    },
    GaussianSplatRasterize {
        prep_off: u32,
        prep_len: u32,
        meta_off: u32,
        meta_len: u32,
        dst_off: u32,
        dst_len: u32,
        count: u32,
        width: u32,
        height: u32,
        tile_size: u32,
        alpha_cutoff: f32,
        max_splat_steps: u32,
        transmittance_threshold: f32,
        max_list_entries: u32,
    },
    RmsNormBackwardInput {
        x_byte_off: u32,
        gamma_byte_off: u32,
        beta_byte_off: u32,
        dy_byte_off: u32,
        dx_byte_off: u32,
        rows: u32,
        h: u32,
        eps_bits: u32,
    },
    RmsNormBackwardGamma {
        x_byte_off: u32,
        gamma_byte_off: u32,
        beta_byte_off: u32,
        dy_byte_off: u32,
        dgamma_byte_off: u32,
        rows: u32,
        h: u32,
        eps_bits: u32,
    },
    RmsNormBackwardBeta {
        x_byte_off: u32,
        gamma_byte_off: u32,
        beta_byte_off: u32,
        dy_byte_off: u32,
        dbeta_byte_off: u32,
        rows: u32,
        h: u32,
        eps_bits: u32,
    },
    RopeBackward {
        dy_byte_off: u32,
        cos_byte_off: u32,
        sin_byte_off: u32,
        dx_byte_off: u32,
        batch: u32,
        seq: u32,
        hidden: u32,
        head_dim: u32,
        n_rot: u32,
        cos_len: u32,
    },
    CumsumBackward {
        dy_byte_off: u32,
        dx_byte_off: u32,
        rows: u32,
        cols: u32,
        exclusive: bool,
    },
    GatherBackward {
        dy_byte_off: u32,
        indices_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        axis_dim: u32,
        num_idx: u32,
        trailing: u32,
    },
    MaxPool2dBackward {
        x_byte_off: u64,
        dy_byte_off: u64,
        dx_byte_off: u64,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
    },
    Conv2dBackwardInput {
        // u64: arena can exceed 4 GiB; u32 byte offsets wrap (same as CUDA).
        dy_byte_off: u64,
        w_byte_off: u64,
        dx_byte_off: u64,
        n: u32,
        c_in: u32,
        h: u32,
        w_in: u32,
        c_out: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw: u32,
        groups: u32,
    },
    Conv2dBackwardWeight {
        x_byte_off: u64,
        dy_byte_off: u64,
        dw_byte_off: u64,
        n: u32,
        c_in: u32,
        h: u32,
        w: u32,
        c_out: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw_dil: u32,
        groups: u32,
    },
    Pool1d {
        n: u32,
        c: u32,
        l: u32,
        l_out: u32,
        kl: u32,
        sl: u32,
        pl: u32,
        op: u32,
        in_off: u32,
        out_off: u32,
    },
    Pool2d {
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        op: u32,
        in_off: u32,
        out_off: u32,
    },
    Pool3d {
        n: u32,
        c: u32,
        d: u32,
        h: u32,
        w: u32,
        d_out: u32,
        h_out: u32,
        w_out: u32,
        kd: u32,
        kh: u32,
        kw: u32,
        sd: u32,
        sh: u32,
        sw: u32,
        pd: u32,
        ph: u32,
        pw: u32,
        op: u32,
        in_off: u32,
        out_off: u32,
    },
    Conv1d {
        n: u32,
        c_in: u32,
        c_out: u32,
        l: u32,
        l_out: u32,
        kl: u32,
        sl: u32,
        pl: u32,
        dl: u32,
        groups: u32,
        in_off: u32,
        w_off: u32,
        out_off: u32,
    },
    Conv2d {
        n: u32,
        c_in: u32,
        c_out: u32,
        h: u32,
        w: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw: u32,
        groups: u32,
        in_off: u32,
        w_off: u32,
        out_off: u32,
        // Fused bias + activation epilogue (from `Op::FusedConvBiasAct`).
        // `has_bias=0` / `act_id=0xFFFF` for a plain `Op::Conv`. MIOpen has
        // no fused conv-bias-act path; when set, run `conv_bias_act_epilogue`
        // after MIOpen or the direct conv kernel (same as CUDA without cuDNN).
        has_bias: u32,
        bias_off_f32: u32,
        act_id: u32,
        // Optional residual add before the activation, from
        // `Op::FusedConvBiasAct { has_residual: true }`.
        has_residual: u32,
        residual_off_f32: u32,
    },
    Conv3d {
        n: u32,
        c_in: u32,
        c_out: u32,
        d: u32,
        h: u32,
        w: u32,
        d_out: u32,
        h_out: u32,
        w_out: u32,
        kd: u32,
        kh: u32,
        kw: u32,
        sd: u32,
        sh: u32,
        sw: u32,
        pd: u32,
        ph: u32,
        pw: u32,
        dd: u32,
        dh: u32,
        dw: u32,
        groups: u32,
        in_off: u32,
        w_off: u32,
        out_off: u32,
    },
    LayerNorm2d {
        src_off: u32,
        g_off: u32,
        b_off: u32,
        dst_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        eps_bits: u32,
    },
    ConvTranspose2d {
        src_off: u32,
        w_off: u32,
        dst_off: u32,
        n: u32,
        c_in: u32,
        h: u32,
        w_in: u32,
        c_out: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw: u32,
        groups: u32,
    },
    /// NCDHW ConvTranspose3d (PyTorch weight layout).
    ConvTranspose3d {
        n: u32,
        c_in: u32,
        c_out: u32,
        d: u32,
        h: u32,
        w: u32,
        d_out: u32,
        h_out: u32,
        w_out: u32,
        kd: u32,
        kh: u32,
        kw: u32,
        sd: u32,
        sh: u32,
        sw: u32,
        pd: u32,
        ph: u32,
        pw: u32,
        dd: u32,
        dh: u32,
        dw: u32,
        groups: u32,
        in_off: u32,
        w_off: u32,
        out_off: u32,
    },
    /// Fused SwiGLU: `up * silu(gate)` on concatenated `[..., 2*n_half]`.
    FusedSwiGLU {
        in_off: u32,
        out_off: u32,
        n_half: u32,
        total: u32,
        gate_first: u32,
    },
    GroupNorm {
        src_off: u32,
        g_off: u32,
        b_off: u32,
        dst_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        num_groups: u32,
        eps_bits: u32,
    },
    GroupNormBackwardInput {
        x_off: u32,
        gamma_off: u32,
        dy_off: u32,
        out_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        num_groups: u32,
        eps_bits: u32,
    },
    GroupNormBackwardGamma {
        x_off: u32,
        dy_off: u32,
        out_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        num_groups: u32,
        eps_bits: u32,
    },
    GroupNormBackwardBeta {
        dy_off: u32,
        out_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
    },
    /// Channels-last BatchNormInference (frozen μ/σ²). Inputs
    /// `[x, gamma, beta, mean, var]`.
    BatchNormInference {
        src_off: u32,
        g_off: u32,
        b_off: u32,
        mean_off: u32,
        var_off: u32,
        dst_off: u32,
        count: u32,
        channels: u32,
        eps_bits: u32,
    },
    /// Channels-last BatchNormInference dx. Inputs `[x, gamma, mean, var, dy]`
    /// (x/mean unused at runtime).
    BatchNormInferenceBackwardInput {
        gamma_off: u32,
        var_off: u32,
        dy_off: u32,
        out_off: u32,
        count: u32,
        channels: u32,
        eps_bits: u32,
    },
    /// Channels-last BatchNormInference dγ. Inputs `[x, mean, var, dy]`.
    BatchNormInferenceBackwardGamma {
        x_off: u32,
        mean_off: u32,
        var_off: u32,
        dy_off: u32,
        out_off: u32,
        count: u32,
        channels: u32,
        eps_bits: u32,
    },
    /// Channels-last BatchNormInference dβ. Input `[dy]`.
    BatchNormInferenceBackwardBeta {
        dy_off: u32,
        out_off: u32,
        count: u32,
        channels: u32,
    },
    /// Last-axis LayerNorm backward w.r.t. input. Inputs `[x, gamma, dy]`.
    LayerNormBackwardInput {
        x_off: u32,
        gamma_off: u32,
        dy_off: u32,
        out_off: u32,
        rows: u32,
        h: u32,
        eps_bits: u32,
    },
    /// Last-axis LayerNorm backward w.r.t. gamma. Inputs `[x, dy]`.
    LayerNormBackwardGamma {
        x_off: u32,
        dy_off: u32,
        out_off: u32,
        rows: u32,
        h: u32,
        eps_bits: u32,
    },
    /// Native `Op::FakeQuantize` Fixed (scale input).
    /// Also used for `Op::FakeQuantizeLSQ` (same forward).
    FakeQuantizeFixed {
        in_off: u32,
        scale_off: u32,
        out_off: u32,
        n: u32,
        chan_dim: u32,
        inner: u32,
        q_max_bits: u32,
    },
    /// Native `Op::FakeQuantize` PerBatch (derive scale from max abs).
    FakeQuantizePerBatch {
        in_off: u32,
        out_off: u32,
        n: u32,
        chan_dim: u32,
        inner: u32,
        q_max_bits: u32,
    },
    /// Native `Op::FakeQuantize` EMA (running scale state, mutated in place).
    FakeQuantizeEma {
        in_off: u32,
        scale_off: u32,
        out_off: u32,
        n: u32,
        chan_dim: u32,
        inner: u32,
        q_max_bits: u32,
        decay_bits: u32,
    },
    /// Native INT8 `Op::Quantize`. Affine table in `meta_buffers[meta_idx]`
    /// as `[scale_bits, zp_i32, …]` per channel. `q_byte_off` is an arena
    /// byte offset (I8 slot).
    QuantizeI8 {
        in_off: u32,
        q_byte_off: u32,
        n: u32,
        chan_dim: u32,
        inner: u32,
        meta_idx: usize,
    },
    /// Native INT8 `Op::Dequantize`. Affine packing matches `QuantizeI8`.
    DequantizeI8 {
        q_byte_off: u32,
        out_off: u32,
        n: u32,
        chan_dim: u32,
        inner: u32,
        meta_idx: usize,
    },
    /// Native real-INT8 `Op::QMatMul`. `x`/`w`/`out` are packed i8 byte
    /// offsets; `bias_off` is an f32-lane index (I32 stored as float).
    QMatMul {
        m: u32,
        k: u32,
        n: u32,
        x_byte_off: u32,
        w_byte_off: u32,
        bias_off: u32,
        out_byte_off: u32,
        x_zp: i32,
        w_zp: i32,
        out_zp: i32,
        mult_bits: u32,
    },
    /// Native real-INT8 `Op::QConv2d` (NCHW). Packed i8 x/w/out; bias is
    /// f32-lane I32 (same convention as `QMatMul`).
    QConv2d {
        n: u32,
        c_in: u32,
        c_out: u32,
        h: u32,
        w: u32,
        h_out: u32,
        w_out: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw: u32,
        groups: u32,
        x_byte_off: u32,
        w_byte_off: u32,
        bias_off: u32,
        out_byte_off: u32,
        x_zp: i32,
        w_zp: i32,
        out_zp: i32,
        mult_bits: u32,
    },
    /// Native `Op::FakeQuantizeLSQBackwardX` (STE-clipped).
    FakeQuantizeLsqBwdX {
        x_off: u32,
        scale_off: u32,
        dy_off: u32,
        dx_off: u32,
        n: u32,
        chan_dim: u32,
        inner: u32,
        q_max_bits: u32,
    },
    /// Native `Op::FakeQuantizeLSQBackwardScale` (per-channel ψ reduce).
    FakeQuantizeLsqBwdScale {
        x_off: u32,
        scale_off: u32,
        dy_off: u32,
        dscale_off: u32,
        n: u32,
        chan_dim: u32,
        inner: u32,
        q_max_bits: u32,
    },
    /// Native `Op::FakeQuantizeBackward` (PerBatch scale + STE).
    /// `ste_kind`: 0 Identity, 1 ClippedIdentity, 2 Tanh, 3 HardTanh.
    FakeQuantizeBackward {
        x_off: u32,
        dy_off: u32,
        dx_off: u32,
        n: u32,
        chan_dim: u32,
        inner: u32,
        q_max_bits: u32,
        ste_kind: u32,
    },
    ResizeNearest2x {
        src_off: u32,
        dst_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
    },
    /// Standalone complex `Op::Cast` on the simulated-complex f32-lane arena
    /// (`complex_cast.cu`, shared with rlx-cuda). `mode` picks one of six pure
    /// lane-move directions (real↔C64, real↔C128, C64↔C128); `n` is the
    /// complex-element count. Byte offsets are u32 (rocm arena convention); the
    /// dispatch widens `byte_off/4` to u64 before launch because the kernel
    /// declares its offset params `unsigned long long` — passing a u32 would
    /// leave the high word as stack garbage.
    ComplexCast {
        n: u32,
        in_byte_off: u32,
        out_byte_off: u32,
        mode: u32,
    },
    /// Element-wise C64 binary (`binary_c64.cu`, shared with rlx-cuda):
    /// add/sub/mul/div reading BOTH `[re, im]` lanes per element, with modulo
    /// broadcast (`n_a`/`n_b` are the operands' complex-element counts). `n` is
    /// the output complex-element count; byte offsets are u32 (widened to u64 at
    /// launch to match the kernel's `unsigned long long` params).
    BinaryC64 {
        n: u32,
        a_byte_off: u32,
        b_byte_off: u32,
        c_byte_off: u32,
        op: u32,
        n_a: u32,
        n_b: u32,
    },
    /// `|z|² = re² + im²` (`complex_wirtinger.cu` / `complex_norm_sq`).
    /// `n` is the complex-element count; output is real F32 (one lane per elem).
    ComplexNormSq {
        n: u32,
        src_byte_off: u32,
        dst_byte_off: u32,
    },
    /// Wirtinger VJP of ComplexNormSq: `dz = g · z` (`complex_norm_sq_backward`).
    /// `z` is C64, `g` is real F32, `dz` is C64.
    ComplexNormSqBackward {
        n: u32,
        z_byte_off: u32,
        g_byte_off: u32,
        dz_byte_off: u32,
    },
    /// Element-wise C64 conjugate: `(re, -im)` (`conjugate_c64`).
    ConjugateC64 {
        n: u32,
        src_byte_off: u32,
        dst_byte_off: u32,
    },
    FusedBinaryUnary {
        n: u32,
        a_off: u32,
        b_off: u32,
        out_off: u32,
        bin_op: u32,
        un_op: u32,
    },
    /// PLAN L2 — interpreted N-ary element-wise chain. The chain
    /// encoding (input_offs[8] + chain[64]) lives in `meta_buffers`
    /// and is indexed via `meta_idx`. One thread per output element;
    /// each thread walks the chain in registers and writes the final
    /// result to `arena[dst_off + i]`. Caps: 16 steps, 8 inputs.
    /// Kernel source shared with rlx-cuda (`elementwise_region.cu`).
    /// `input_offs` mirrors what's packed in `meta` and is kept in
    /// the Step so the multi-stream scheduler can resolve
    /// producer-consumer dependencies without unpacking metadata.
    ElementwiseRegion {
        len: u32,
        num_inputs: u32,
        num_steps: u32,
        dst_off: u32,
        input_offs: [u32; 16],
        /// PLAN L2 quality fast path: per-input scalar-broadcast bitfield.
        scalar_input_mask: u32,
        /// PLAN L2 quality general broadcast: per-input element count.
        /// `0` ⇒ no broadcast (kernel reads gid); `>0` ⇒ kernel reads
        /// `arena[input_offs[i] + (gid % input_modulus[i])]`.
        input_modulus: [u32; 16],
        meta_idx: usize,
        spatial_prologue: bool,
        prologue_w: u32,
        prologue_h: u32,
        prologue_nc: u32,
    },
    BatchElementwiseRegion {
        slice_len: u32,
        num_batch: u32,
        num_steps: u32,
        base_dst_off: u32,
        slice_elems: u32,
        batch_input_offs: [u32; 64],
        batch_offs_idx: usize,
        meta_idx: usize,
        scalar_input_mask: u32,
        input_modulus: [u32; 16],
    },
}

pub(crate) fn step_name(step: &Step) -> &'static str {
    match step {
        Step::Matmul { .. } => "rlx::Matmul",
        Step::ScaledMatMul { .. } => "rlx::ScaledMatMul",
        Step::ScaledQuantScale { .. } => "rlx::ScaledQuantScale",
        Step::ScaledQuantizeFp8 { .. } => "rlx::ScaledQuantizeFp8",
        Step::ScaledMatMulDecode { .. } => "rlx::ScaledMatMulDecode",
        Step::ScaledQuantScaleGeneral { .. } => "rlx::ScaledQuantScaleGeneral",
        Step::ScaledQuantizeGeneral { .. } => "rlx::ScaledQuantizeGeneral",
        Step::ScaledDequantizeGeneral { .. } => "rlx::ScaledDequantizeGeneral",
        Step::Binary { .. } => "rlx::Binary",
        Step::Compare { .. } => "rlx::Compare",
        Step::Unary { .. } => "rlx::Unary",
        Step::Where { .. } => "rlx::Where",
        Step::Fma { .. } => "rlx::Fma",
        Step::Reduce { .. } => "rlx::Reduce",
        Step::Softmax { .. } => "rlx::Softmax",
        Step::ReluBackward { .. } => "rlx::ReluBackward",
        Step::ActivationBackward { .. } => "rlx::ActivationBackward",
        Step::SoftmaxCrossEntropy { .. } => "rlx::SoftmaxCrossEntropy",
        Step::SoftmaxCrossEntropyWithLogits { .. } => "rlx::SoftmaxCrossEntropyWithLogits",
        Step::SoftmaxCrossEntropyBackward { .. } => "rlx::SoftmaxCrossEntropyBackward",
        Step::LayerNorm { .. } => "rlx::LayerNorm",
        Step::FusedResidualLn { .. } => "rlx::FusedResidualLN",
        Step::AdaLayerNorm { .. } => "rlx::AdaLayerNorm",
        Step::GatedResidual { .. } => "rlx::GatedResidual",
        Step::AdaLayerNormBackward { .. } => "rlx::AdaLayerNormBackward",
        Step::GatedResidualBackward { .. } => "rlx::GatedResidualBackward",
        Step::Gather { .. } => "rlx::Gather",
        Step::GatherAxis { .. } => "rlx::GatherAxis",
        Step::Narrow { .. } => "rlx::Narrow",
        Step::Concat { .. } => "rlx::Concat",
        Step::Transpose { .. } => "rlx::Transpose",
        Step::Expand { .. } => "rlx::Expand",
        Step::Argmax { .. } => "rlx::Argmax",
        Step::Attention { .. } => "rlx::Attention",
        Step::AttentionBackward { .. } => "rlx::AttentionBackward",
        Step::Rope { .. } => "rlx::Rope",
        Step::Cumsum { .. } => "rlx::Cumsum",
        Step::TopK { .. } => "rlx::TopK",
        Step::GroupedMatmul { .. } => "rlx::GroupedMatmul",
        Step::ScatterAddZero { .. } => "rlx::ScatterAdd::zero",
        Step::ScatterAddAcc { .. } => "rlx::ScatterAdd::acc",
        Step::DequantMatmul { .. } => "rlx::DequantMatmul",
        Step::DequantMatmulGguf { .. } => "rlx::DequantMatmulGguf",
        Step::DequantGroupedMatmulGguf { .. } => "rlx::DequantGroupedMatmulGguf",
        Step::Sample { .. } => "rlx::Sample",
        Step::RngNormal { .. } => "rlx::RngNormal",
        Step::RngUniform { .. } => "rlx::RngUniform",
        Step::SelectiveScan { .. } => "rlx::SelectiveScan",
        Step::Fft { .. } => "rlx::Fft",
        Step::LogMelHost { .. } => "rlx::LogMelHost",
        Step::LogMelBackwardHost { .. } => "rlx::LogMelBackwardHost",
        Step::WelchPeaksHost { .. } => "rlx::WelchPeaksHost",
        Step::WelchPeaksGpu { .. } => "rlx::WelchPeaksGpu",
        Step::FftButterflyStage { .. } => "rlx::FftButterflyStage",
        Step::Im2ColHost { .. } => "rlx::Im2ColHost",
        Step::ReverseHost { .. } => "rlx::ReverseHost",
        Step::ArgReduceHost { .. } => "rlx::ArgReduceHost",
        Step::AxialRope2d { .. } => "rlx::AxialRope2d",
        Step::GatedDeltaNet { .. } => "rlx::GatedDeltaNet",
        Step::Lstm { .. } => "rlx::Lstm",
        Step::Gru { .. } => "rlx::Gru",
        Step::GruHost { .. } => "rlx::GruHost",
        Step::Rnn { .. } => "rlx::Rnn",
        Step::RnnHost { .. } => "rlx::RnnHost",
        Step::Mamba2 { .. } => "rlx::Mamba2",
        Step::Mamba2Host { .. } => "rlx::Mamba2Host",
        Step::ScanHost { .. } => "rlx::ScanHost",
        Step::HostOp { .. } => "rlx::HostOp",
        Step::CpuIndexing { .. } => "rlx::CpuIndexing",
        Step::SpdHost { .. } => "rlx::SpdHost",
        Step::EighNative { .. } => "rlx::EighNative",
        Step::DenseSolveNative { .. } => "rlx::DenseSolveNative",
        Step::BatchedDenseSolveNative { .. } => "rlx::BatchedDenseSolveNative",
        Step::Llada2GroupLimitedGate { .. } => "rlx::Llada2GroupLimitedGate",
        Step::MsDeformAttnHost { .. } => "rlx::MsDeformAttnHost",
        Step::CollectiveHost { .. } => "rlx::CollectiveHost",
        Step::UmapKnn { .. } => "rlx::UmapKnn",
        Step::RocmGpuKernel { .. } => "rlx::RocmGpuKernel",
        Step::GaussianSplatRender { .. } => "rlx::GaussianSplatRender",
        Step::GaussianSplatRenderBackward { .. } => "rlx::GaussianSplatRenderBackward",
        Step::GaussianSplatPrepare { .. } => "rlx::GaussianSplatPrepare",
        Step::GaussianSplatRasterize { .. } => "rlx::GaussianSplatRasterize",
        Step::RmsNormBackwardInput { .. } => "rlx::RmsNormBackwardInput",
        Step::RmsNormBackwardGamma { .. } => "rlx::RmsNormBackwardGamma",
        Step::RmsNormBackwardBeta { .. } => "rlx::RmsNormBackwardBeta",
        Step::RopeBackward { .. } => "rlx::RopeBackward",
        Step::CumsumBackward { .. } => "rlx::CumsumBackward",
        Step::GatherBackward { .. } => "rlx::GatherBackward",
        Step::MaxPool2dBackward { .. } => "rlx::MaxPool2dBackward",
        Step::Conv2dBackwardInput { .. } => "rlx::Conv2dBackwardInput",
        Step::Conv2dBackwardWeight { .. } => "rlx::Conv2dBackwardWeight",
        Step::Pool1d { .. } => "rlx::Pool1d",
        Step::Pool2d { .. } => "rlx::Pool2d",
        Step::Pool3d { .. } => "rlx::Pool3d",
        Step::Conv1d { .. } => "rlx::Conv1d",
        Step::Conv2d { .. } => "rlx::Conv2d",
        Step::Conv3d { .. } => "rlx::Conv3d",
        Step::LayerNorm2d { .. } => "rlx::LayerNorm2d",
        Step::ConvTranspose2d { .. } => "rlx::ConvTranspose2d",
        Step::ConvTranspose3d { .. } => "rlx::ConvTranspose3d",
        Step::FusedSwiGLU { .. } => "rlx::FusedSwiGLU",
        Step::GroupNorm { .. } => "rlx::GroupNorm",
        Step::GroupNormBackwardInput { .. } => "rlx::GroupNormBackwardInput",
        Step::GroupNormBackwardGamma { .. } => "rlx::GroupNormBackwardGamma",
        Step::GroupNormBackwardBeta { .. } => "rlx::GroupNormBackwardBeta",
        Step::BatchNormInference { .. } => "rlx::BatchNormInference",
        Step::BatchNormInferenceBackwardInput { .. } => "rlx::BatchNormInferenceBackwardInput",
        Step::BatchNormInferenceBackwardGamma { .. } => "rlx::BatchNormInferenceBackwardGamma",
        Step::BatchNormInferenceBackwardBeta { .. } => "rlx::BatchNormInferenceBackwardBeta",
        Step::LayerNormBackwardInput { .. } => "rlx::LayerNormBackwardInput",
        Step::LayerNormBackwardGamma { .. } => "rlx::LayerNormBackwardGamma",
        Step::FakeQuantizeFixed { .. } => "rlx::FakeQuantizeFixed",
        Step::FakeQuantizePerBatch { .. } => "rlx::FakeQuantizePerBatch",
        Step::FakeQuantizeEma { .. } => "rlx::FakeQuantizeEma",
        Step::QuantizeI8 { .. } => "rlx::QuantizeI8",
        Step::DequantizeI8 { .. } => "rlx::DequantizeI8",
        Step::QMatMul { .. } => "rlx::QMatMul",
        Step::QConv2d { .. } => "rlx::QConv2d",
        Step::FakeQuantizeLsqBwdX { .. } => "rlx::FakeQuantizeLsqBwdX",
        Step::FakeQuantizeLsqBwdScale { .. } => "rlx::FakeQuantizeLsqBwdScale",
        Step::FakeQuantizeBackward { .. } => "rlx::FakeQuantizeBackward",
        Step::ResizeNearest2x { .. } => "rlx::ResizeNearest2x",
        Step::ComplexCast { .. } => "rlx::ComplexCast",
        Step::BinaryC64 { .. } => "rlx::BinaryC64",
        Step::ComplexNormSq { .. } => "rlx::ComplexNormSq",
        Step::ComplexNormSqBackward { .. } => "rlx::ComplexNormSqBackward",
        Step::ConjugateC64 { .. } => "rlx::ConjugateC64",
        Step::FusedBinaryUnary { .. } => "rlx::FusedBinaryUnary",
        Step::ElementwiseRegion { .. } => "rlx::ElementwiseRegion",
        Step::BatchElementwiseRegion { .. } => "rlx::BatchElementwiseRegion",
    }
}

// ── step_offsets (port from rlx-cuda) ─────────────────────────────────

pub(crate) fn step_offsets(step: &Step) -> (Vec<u32>, Vec<u32>) {
    match step {
        Step::ScanHost { desc } => {
            let mut reads = vec![(desc.outer_init_off / 4) as u32];
            reads.extend(desc.bcast_outer.iter().map(|&(o, _)| (o / 4) as u32));
            reads.extend(desc.xs_outer.iter().map(|&(o, _)| (o / 4) as u32));
            (reads, vec![(desc.outer_final_off / 4) as u32])
        }
        Step::HostOp { desc } => {
            let reads = desc.inputs.iter().map(|&(o, _)| (o / 4) as u32).collect();
            (reads, vec![(desc.out_byte_off / 4) as u32])
        }
        Step::CpuIndexing { thunk } => {
            let regions = rlx_cpu::thunk::indexing_thunk_regions(thunk.inner());
            let reads: Vec<u32> = regions[..regions.len().saturating_sub(1)]
                .iter()
                .map(|&(o, _)| (o / 4) as u32)
                .collect();
            let writes = regions
                .last()
                .map(|&(o, _)| vec![(o / 4) as u32])
                .unwrap_or_default();
            (reads, writes)
        }
        Step::SpdHost {
            out_off, inputs, ..
        } => {
            // Offsets already stored in f32 elements.
            let reads = inputs.iter().map(|&(o, _)| o as u32).collect();
            (reads, vec![*out_off as u32])
        }
        Step::EighNative {
            in_off, out_off, ..
        } => (vec![*in_off as u32], vec![*out_off as u32]),
        Step::DenseSolveNative {
            a_off,
            b_off,
            x_off,
            ..
        }
        | Step::BatchedDenseSolveNative {
            a_off,
            b_off,
            x_off,
            ..
        } => (vec![*a_off as u32, *b_off as u32], vec![*x_off as u32]),
        Step::Matmul {
            a_off_f32,
            b_off_f32,
            c_off_f32,
            has_bias,
            bias_off_f32,
            ..
        } => {
            let mut r = vec![*a_off_f32, *b_off_f32];
            if *has_bias != 0 {
                r.push(*bias_off_f32);
            }
            (r, vec![*c_off_f32])
        }
        Step::ScaledMatMul {
            lhs_byte_off,
            rhs_byte_off,
            lhs_scale_byte_off,
            rhs_scale_byte_off,
            out_byte_off,
            has_bias,
            bias_byte_off,
            ..
        } => {
            let mut r = vec![
                *lhs_byte_off / 4,
                *rhs_byte_off / 4,
                *lhs_scale_byte_off / 4,
                *rhs_scale_byte_off / 4,
            ];
            if *has_bias != 0 {
                r.push(*bias_byte_off / 4);
            }
            (r, vec![*out_byte_off / 4])
        }
        Step::ScaledQuantScale {
            x_off_f32,
            scale_off_f32,
            ..
        } => (vec![*x_off_f32], vec![*scale_off_f32]),
        Step::ScaledQuantizeFp8 {
            x_off_f32,
            scale_off_f32,
            out_byte_off,
            ..
        } => (vec![*x_off_f32, *scale_off_f32], vec![*out_byte_off / 4]),
        Step::ScaledMatMulDecode {
            lhs_byte_off,
            rhs_byte_off,
            lhs_scale_byte_off,
            rhs_scale_byte_off,
            out_off_f32,
            has_bias,
            bias_off_f32,
            ..
        } => {
            let mut r = vec![
                *lhs_byte_off / 4,
                *rhs_byte_off / 4,
                *lhs_scale_byte_off / 4,
                *rhs_scale_byte_off / 4,
            ];
            if *has_bias != 0 {
                r.push(*bias_off_f32);
            }
            (r, vec![*out_off_f32])
        }
        Step::ScaledQuantScaleGeneral {
            x_off_f32,
            scale_byte_off,
            ..
        } => (vec![*x_off_f32], vec![*scale_byte_off / 4]),
        Step::ScaledQuantizeGeneral {
            x_off_f32,
            scale_byte_off,
            out_byte_off,
            ..
        } => (
            vec![*x_off_f32, *scale_byte_off / 4],
            vec![*out_byte_off / 4],
        ),
        Step::ScaledDequantizeGeneral {
            codes_byte_off,
            scale_byte_off,
            out_off_f32,
            ..
        } => (
            vec![*codes_byte_off / 4, *scale_byte_off / 4],
            vec![*out_off_f32],
        ),
        Step::Binary {
            a_off,
            b_off,
            c_off,
            ..
        }
        | Step::Compare {
            a_off,
            b_off,
            c_off,
            ..
        } => (vec![*a_off, *b_off], vec![*c_off]),
        Step::Unary {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::Where {
            cond_off,
            x_off,
            y_off,
            out_off,
            ..
        } => (vec![*cond_off, *x_off, *y_off], vec![*out_off]),
        Step::Fma {
            a_off,
            b_off,
            c_off,
            out_off,
            ..
        } => (vec![*a_off, *b_off, *c_off], vec![*out_off]),
        Step::Reduce {
            in_off, out_off, ..
        }
        | Step::Softmax {
            in_off, out_off, ..
        }
        | Step::Argmax {
            in_off, out_off, ..
        }
        | Step::Cumsum {
            in_off, out_off, ..
        }
        | Step::Sample {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::ReluBackward {
            x_off,
            dy_off,
            dx_off,
            ..
        }
        | Step::ActivationBackward {
            x_off,
            dy_off,
            dx_off,
            ..
        } => (vec![*x_off, *dy_off], vec![*dx_off]),
        Step::SoftmaxCrossEntropy {
            logits_off,
            targets_off,
            out_off,
            ..
        }
        | Step::SoftmaxCrossEntropyWithLogits {
            logits_off,
            labels_off: targets_off,
            out_off,
            ..
        } => (vec![*logits_off, *targets_off], vec![*out_off]),
        Step::SoftmaxCrossEntropyBackward {
            logits_off,
            labels_off,
            d_loss_off,
            out_off,
            ..
        } => (vec![*logits_off, *labels_off, *d_loss_off], vec![*out_off]),
        Step::RngNormal { dst_byte_off, .. } | Step::RngUniform { dst_byte_off, .. } => {
            (vec![], vec![*dst_byte_off / 4])
        }
        Step::TopK {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::LayerNorm {
            in_off,
            gamma_off,
            beta_off,
            out_off,
            ..
        } => (vec![*in_off, *gamma_off, *beta_off], vec![*out_off]),
        Step::FusedResidualLn {
            in_off,
            residual_off,
            bias_off,
            gamma_off,
            beta_off,
            out_off,
            has_bias,
            ..
        } => {
            let mut r = vec![*in_off, *residual_off, *gamma_off, *beta_off];
            if *has_bias != 0 {
                r.push(*bias_off);
            }
            (r, vec![*out_off])
        }
        Step::AdaLayerNorm {
            in_off,
            scale_off,
            shift_off,
            out_off,
            ..
        } => (vec![*in_off, *scale_off, *shift_off], vec![*out_off]),
        Step::GatedResidual {
            x_off,
            y_off,
            gate_off,
            out_off,
            ..
        } => (vec![*x_off, *y_off, *gate_off], vec![*out_off]),
        Step::AdaLayerNormBackward {
            x_off,
            scale_off,
            dy_off,
            out_off,
            ..
        } => (vec![*x_off, *scale_off, *dy_off], vec![*out_off]),
        Step::GatedResidualBackward {
            y_off,
            gate_off,
            dy_off,
            out_off,
            ..
        } => (vec![*y_off, *gate_off, *dy_off], vec![*out_off]),
        Step::Gather {
            in_off,
            idx_off,
            out_off,
            ..
        } => (vec![*in_off, *idx_off], vec![*out_off]),
        Step::GatherAxis {
            table_off,
            idx_off,
            out_off,
            ..
        } => (vec![*table_off, *idx_off], vec![*out_off]),
        Step::Narrow {
            in_off, out_off, ..
        }
        | Step::Concat {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::Transpose {
            in_off, out_off, ..
        }
        | Step::Expand {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::Attention {
            q_off,
            k_off,
            v_off,
            mask_off,
            mask_kind,
            out_off,
            ..
        } => {
            let mut r = vec![*q_off, *k_off, *v_off];
            if *mask_kind == 2 || *mask_kind == 4 {
                r.push(*mask_off);
            }
            (r, vec![*out_off])
        }
        Step::AttentionBackward {
            q_off,
            k_off,
            v_off,
            dy_off,
            mask_off,
            mask_kind,
            out_off,
            ..
        } => {
            let mut r = vec![*q_off, *k_off, *v_off, *dy_off];
            if *mask_kind == 2 || *mask_kind == 4 {
                r.push(*mask_off);
            }
            (r, vec![*out_off])
        }
        Step::Rope {
            in_off,
            cos_off,
            sin_off,
            out_off,
            ..
        } => (vec![*in_off, *cos_off, *sin_off], vec![*out_off]),
        Step::GroupedMatmul {
            in_off,
            w_off,
            idx_off,
            out_off,
            ..
        } => (vec![*in_off, *w_off, *idx_off], vec![*out_off]),
        Step::ScatterAddZero { out_off, .. } => (vec![], vec![*out_off]),
        Step::ScatterAddAcc {
            upd_off,
            idx_off,
            out_off,
            ..
        } => (vec![*upd_off, *idx_off, *out_off], vec![*out_off]),
        Step::DequantMatmul {
            x_off,
            w_off,
            scale_off,
            zp_off,
            out_off,
            scheme_id,
            ..
        } => {
            let mut r = vec![*x_off, *w_off, *scale_off];
            if *scheme_id == 1 {
                r.push(*zp_off);
            }
            (r, vec![*out_off])
        }
        Step::DequantMatmulGguf {
            x_byte_off,
            w_byte_off,
            out_byte_off,
            ..
        } => (vec![x_byte_off / 4, w_byte_off / 4], vec![out_byte_off / 4]),
        Step::DequantGroupedMatmulGguf {
            x_byte_off,
            w_byte_off,
            idx_byte_off,
            out_byte_off,
            ..
        } => (
            vec![x_byte_off / 4, w_byte_off / 4, idx_byte_off / 4],
            vec![out_byte_off / 4],
        ),
        Step::SelectiveScan {
            x_off,
            delta_off,
            a_off,
            b_off,
            c_off,
            out_off,
            ..
        } => (
            vec![*x_off, *delta_off, *a_off, *b_off, *c_off],
            vec![*out_off],
        ),
        Step::Fft {
            src_byte_off,
            dst_byte_off,
            ..
        } => (vec![*src_byte_off / 4], vec![*dst_byte_off / 4]),
        Step::LogMelHost {
            spec_byte_off,
            filt_byte_off,
            dst_byte_off,
            ..
        } => (
            vec![*spec_byte_off / 4, *filt_byte_off / 4],
            vec![*dst_byte_off / 4],
        ),
        Step::LogMelBackwardHost {
            spec_byte_off,
            filt_byte_off,
            dy_byte_off,
            dst_byte_off,
            ..
        } => (
            vec![*spec_byte_off / 4, *filt_byte_off / 4, *dy_byte_off / 4],
            vec![*dst_byte_off / 4],
        ),
        Step::WelchPeaksHost {
            spec_byte_off,
            dst_byte_off,
            ..
        } => (vec![*spec_byte_off / 4], vec![*dst_byte_off / 4]),
        Step::WelchPeaksGpu {
            spec_off, dst_off, ..
        } => (vec![*spec_off], vec![*dst_off]),
        Step::FftButterflyStage {
            state_off,
            out_off,
            gate_off,
            rev_off,
            tw_re_off,
            tw_im_off,
            ..
        } => (
            vec![*state_off, *gate_off, *rev_off, *tw_re_off, *tw_im_off],
            vec![*out_off],
        ),
        Step::Im2ColHost {
            x_byte_off,
            col_byte_off,
            ..
        } => (vec![*x_byte_off / 4], vec![*col_byte_off / 4]),
        Step::ReverseHost {
            src_byte_off,
            dst_byte_off,
            ..
        }
        | Step::ArgReduceHost {
            src_byte_off,
            dst_byte_off,
            ..
        } => (vec![*src_byte_off / 4], vec![*dst_byte_off / 4]),
        Step::AxialRope2d {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::GatedDeltaNet {
            q_byte_off,
            k_byte_off,
            v_byte_off,
            g_byte_off,
            beta_byte_off,
            state_byte_off,
            dst_byte_off,
            use_carry,
            ..
        } => {
            let mut reads = vec![
                q_byte_off / 4,
                k_byte_off / 4,
                v_byte_off / 4,
                g_byte_off / 4,
                beta_byte_off / 4,
            ];
            if *use_carry {
                reads.push(state_byte_off / 4);
            }
            let mut writes = vec![dst_byte_off / 4];
            if *use_carry {
                writes.push(state_byte_off / 4);
            }
            (reads, writes)
        }
        Step::Lstm {
            x_byte_off,
            w_ih_byte_off,
            w_hh_byte_off,
            bias_byte_off,
            h0_byte_off,
            c0_byte_off,
            dst_byte_off,
            carry,
            ..
        } => {
            let mut reads = vec![
                x_byte_off / 4,
                w_ih_byte_off / 4,
                w_hh_byte_off / 4,
                bias_byte_off / 4,
            ];
            let mut writes = vec![dst_byte_off / 4];
            if *carry {
                // h0/c0 are read and (decode) written back in place.
                reads.push(h0_byte_off / 4);
                reads.push(c0_byte_off / 4);
                writes.push(h0_byte_off / 4);
                writes.push(c0_byte_off / 4);
            }
            (reads, writes)
        }
        Step::Gru {
            x_byte_off,
            w_ih_byte_off,
            w_hh_byte_off,
            b_ih_byte_off,
            b_hh_byte_off,
            dst_byte_off,
            ..
        } => (
            vec![
                x_byte_off / 4,
                w_ih_byte_off / 4,
                w_hh_byte_off / 4,
                b_ih_byte_off / 4,
                b_hh_byte_off / 4,
            ],
            vec![dst_byte_off / 4],
        ),
        Step::GruHost {
            x_byte_off,
            w_ih_byte_off,
            w_hh_byte_off,
            b_ih_byte_off,
            b_hh_byte_off,
            h0_byte_off,
            dst_byte_off,
            carry,
            ..
        } => {
            let mut reads = vec![
                x_byte_off / 4,
                w_ih_byte_off / 4,
                w_hh_byte_off / 4,
                b_ih_byte_off / 4,
                b_hh_byte_off / 4,
            ];
            let mut writes = vec![dst_byte_off / 4];
            if *carry {
                reads.push(h0_byte_off / 4);
                writes.push(h0_byte_off / 4);
            }
            (reads, writes)
        }
        Step::Rnn {
            x_byte_off,
            w_ih_byte_off,
            w_hh_byte_off,
            bias_byte_off,
            dst_byte_off,
            ..
        } => (
            vec![
                x_byte_off / 4,
                w_ih_byte_off / 4,
                w_hh_byte_off / 4,
                bias_byte_off / 4,
            ],
            vec![dst_byte_off / 4],
        ),
        Step::RnnHost {
            x_byte_off,
            w_ih_byte_off,
            w_hh_byte_off,
            bias_byte_off,
            h0_byte_off,
            dst_byte_off,
            carry,
            ..
        } => {
            let mut reads = vec![
                x_byte_off / 4,
                w_ih_byte_off / 4,
                w_hh_byte_off / 4,
                bias_byte_off / 4,
            ];
            let mut writes = vec![dst_byte_off / 4];
            if *carry {
                reads.push(h0_byte_off / 4);
                writes.push(h0_byte_off / 4);
            }
            (reads, writes)
        }
        Step::Mamba2 {
            x_byte_off,
            dt_byte_off,
            a_byte_off,
            b_byte_off,
            c_byte_off,
            dst_byte_off,
            ..
        }
        | Step::Mamba2Host {
            x_byte_off,
            dt_byte_off,
            a_byte_off,
            b_byte_off,
            c_byte_off,
            dst_byte_off,
            ..
        } => (
            vec![
                x_byte_off / 4,
                dt_byte_off / 4,
                a_byte_off / 4,
                b_byte_off / 4,
                c_byte_off / 4,
            ],
            vec![dst_byte_off / 4],
        ),
        Step::GaussianSplatRender {
            positions_off,
            positions_len: _,
            scales_off,
            scales_len: _,
            rotations_off,
            rotations_len: _,
            opacities_off,
            opacities_len: _,
            colors_off,
            colors_len: _,
            sh_coeffs_off,
            sh_coeffs_len: _,
            meta_off,
            dst_off,
            dst_len: _,
            ..
        } => (
            vec![
                positions_off / 4,
                scales_off / 4,
                rotations_off / 4,
                opacities_off / 4,
                colors_off / 4,
                sh_coeffs_off / 4,
                meta_off / 4,
            ],
            vec![dst_off / 4],
        ),
        Step::GaussianSplatRenderBackward {
            positions_off,
            positions_len: _,
            scales_off,
            scales_len: _,
            rotations_off,
            rotations_len: _,
            opacities_off,
            opacities_len: _,
            colors_off,
            colors_len: _,
            sh_coeffs_off,
            sh_coeffs_len: _,
            meta_off,
            d_loss_off,
            d_loss_len: _,
            packed_off,
            packed_len: _,
            ..
        } => (
            vec![
                positions_off / 4,
                scales_off / 4,
                rotations_off / 4,
                opacities_off / 4,
                colors_off / 4,
                sh_coeffs_off / 4,
                meta_off / 4,
                d_loss_off / 4,
            ],
            vec![packed_off / 4],
        ),
        Step::GaussianSplatPrepare {
            positions_off,
            scales_off,
            rotations_off,
            opacities_off,
            colors_off,
            sh_coeffs_off,
            meta_off,
            prep_off,
            ..
        } => (
            vec![
                positions_off / 4,
                scales_off / 4,
                rotations_off / 4,
                opacities_off / 4,
                colors_off / 4,
                sh_coeffs_off / 4,
                meta_off / 4,
            ],
            vec![prep_off / 4],
        ),
        Step::GaussianSplatRasterize {
            prep_off,
            meta_off,
            dst_off,
            ..
        } => (vec![prep_off / 4, meta_off / 4], vec![dst_off / 4]),
        Step::RmsNormBackwardInput {
            x_byte_off,
            gamma_byte_off,
            beta_byte_off,
            dy_byte_off,
            dx_byte_off,
            ..
        } => (
            vec![
                x_byte_off / 4,
                gamma_byte_off / 4,
                beta_byte_off / 4,
                dy_byte_off / 4,
            ],
            vec![dx_byte_off / 4],
        ),
        Step::RmsNormBackwardGamma {
            x_byte_off,
            gamma_byte_off,
            beta_byte_off,
            dy_byte_off,
            dgamma_byte_off,
            ..
        } => (
            vec![
                x_byte_off / 4,
                gamma_byte_off / 4,
                beta_byte_off / 4,
                dy_byte_off / 4,
            ],
            vec![dgamma_byte_off / 4],
        ),
        Step::RmsNormBackwardBeta {
            x_byte_off,
            gamma_byte_off,
            beta_byte_off,
            dy_byte_off,
            dbeta_byte_off,
            ..
        } => (
            vec![
                x_byte_off / 4,
                gamma_byte_off / 4,
                beta_byte_off / 4,
                dy_byte_off / 4,
            ],
            vec![dbeta_byte_off / 4],
        ),
        Step::RopeBackward {
            dy_byte_off,
            cos_byte_off,
            sin_byte_off,
            dx_byte_off,
            ..
        } => (
            vec![dy_byte_off / 4, cos_byte_off / 4, sin_byte_off / 4],
            vec![dx_byte_off / 4],
        ),
        Step::CumsumBackward {
            dy_byte_off,
            dx_byte_off,
            ..
        } => (vec![dy_byte_off / 4], vec![dx_byte_off / 4]),
        Step::GatherBackward {
            dy_byte_off,
            indices_byte_off,
            dst_byte_off,
            ..
        } => (
            vec![dy_byte_off / 4, indices_byte_off / 4],
            vec![dst_byte_off / 4],
        ),
        Step::MaxPool2dBackward {
            x_byte_off,
            dy_byte_off,
            dx_byte_off,
            ..
        } => (
            vec![(*x_byte_off / 4) as u32, (*dy_byte_off / 4) as u32],
            vec![(*dx_byte_off / 4) as u32],
        ),
        Step::Conv2dBackwardInput {
            dy_byte_off,
            w_byte_off,
            dx_byte_off,
            ..
        } => (
            vec![(*dy_byte_off / 4) as u32, (*w_byte_off / 4) as u32],
            vec![(*dx_byte_off / 4) as u32],
        ),
        Step::Conv2dBackwardWeight {
            x_byte_off,
            dy_byte_off,
            dw_byte_off,
            ..
        } => (
            vec![(*x_byte_off / 4) as u32, (*dy_byte_off / 4) as u32],
            vec![(*dw_byte_off / 4) as u32],
        ),
        Step::Pool1d {
            in_off, out_off, ..
        }
        | Step::Pool2d {
            in_off, out_off, ..
        }
        | Step::Pool3d {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::Conv1d {
            in_off,
            w_off,
            out_off,
            ..
        }
        | Step::Conv2d {
            in_off,
            w_off,
            out_off,
            ..
        }
        | Step::Conv3d {
            in_off,
            w_off,
            out_off,
            ..
        } => (vec![*in_off, *w_off], vec![*out_off]),
        Step::LayerNorm2d {
            src_off,
            g_off,
            b_off,
            dst_off,
            ..
        } => (vec![*src_off, *g_off, *b_off], vec![*dst_off]),
        Step::ConvTranspose2d {
            src_off,
            w_off,
            dst_off,
            ..
        } => (vec![*src_off, *w_off], vec![*dst_off]),
        Step::ConvTranspose3d {
            in_off,
            w_off,
            out_off,
            ..
        } => (vec![*in_off, *w_off], vec![*out_off]),
        Step::FusedSwiGLU {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::GroupNorm {
            src_off,
            g_off,
            b_off,
            dst_off,
            ..
        } => (vec![*src_off, *g_off, *b_off], vec![*dst_off]),
        Step::GroupNormBackwardInput {
            x_off,
            gamma_off,
            dy_off,
            out_off,
            ..
        } => (vec![*x_off, *gamma_off, *dy_off], vec![*out_off]),
        Step::GroupNormBackwardGamma {
            x_off,
            dy_off,
            out_off,
            ..
        } => (vec![*x_off, *dy_off], vec![*out_off]),
        Step::GroupNormBackwardBeta {
            dy_off, out_off, ..
        } => (vec![*dy_off], vec![*out_off]),
        Step::BatchNormInference {
            src_off,
            g_off,
            b_off,
            mean_off,
            var_off,
            dst_off,
            ..
        } => (
            vec![*src_off, *g_off, *b_off, *mean_off, *var_off],
            vec![*dst_off],
        ),
        Step::BatchNormInferenceBackwardInput {
            gamma_off,
            var_off,
            dy_off,
            out_off,
            ..
        } => (vec![*gamma_off, *var_off, *dy_off], vec![*out_off]),
        Step::BatchNormInferenceBackwardGamma {
            x_off,
            mean_off,
            var_off,
            dy_off,
            out_off,
            ..
        } => (vec![*x_off, *mean_off, *var_off, *dy_off], vec![*out_off]),
        Step::BatchNormInferenceBackwardBeta {
            dy_off, out_off, ..
        } => (vec![*dy_off], vec![*out_off]),
        Step::LayerNormBackwardInput {
            x_off,
            gamma_off,
            dy_off,
            out_off,
            ..
        } => (vec![*x_off, *gamma_off, *dy_off], vec![*out_off]),
        Step::LayerNormBackwardGamma {
            x_off,
            dy_off,
            out_off,
            ..
        } => (vec![*x_off, *dy_off], vec![*out_off]),
        Step::FakeQuantizeFixed {
            in_off,
            scale_off,
            out_off,
            ..
        } => (vec![*in_off, *scale_off], vec![*out_off]),
        Step::FakeQuantizePerBatch {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::FakeQuantizeEma {
            in_off,
            scale_off,
            out_off,
            ..
        } => (vec![*in_off, *scale_off], vec![*out_off, *scale_off]),
        Step::QuantizeI8 {
            in_off, q_byte_off, ..
        } => (vec![*in_off], vec![*q_byte_off / 4]),
        Step::DequantizeI8 {
            q_byte_off,
            out_off,
            ..
        } => (vec![*q_byte_off / 4], vec![*out_off]),
        Step::QMatMul {
            x_byte_off,
            w_byte_off,
            bias_off,
            out_byte_off,
            ..
        } => (
            vec![*x_byte_off / 4, *w_byte_off / 4, *bias_off],
            vec![*out_byte_off / 4],
        ),
        Step::QConv2d {
            x_byte_off,
            w_byte_off,
            bias_off,
            out_byte_off,
            ..
        } => (
            vec![*x_byte_off / 4, *w_byte_off / 4, *bias_off],
            vec![*out_byte_off / 4],
        ),
        Step::FakeQuantizeLsqBwdX {
            x_off,
            scale_off,
            dy_off,
            dx_off,
            ..
        } => (vec![*x_off, *scale_off, *dy_off], vec![*dx_off]),
        Step::FakeQuantizeLsqBwdScale {
            x_off,
            scale_off,
            dy_off,
            dscale_off,
            ..
        } => (vec![*x_off, *scale_off, *dy_off], vec![*dscale_off]),
        Step::FakeQuantizeBackward {
            x_off,
            dy_off,
            dx_off,
            ..
        } => (vec![*x_off, *dy_off], vec![*dx_off]),
        Step::ResizeNearest2x {
            src_off, dst_off, ..
        } => (vec![*src_off], vec![*dst_off]),
        Step::ComplexCast {
            in_byte_off,
            out_byte_off,
            ..
        } => (vec![*in_byte_off / 4], vec![*out_byte_off / 4]),
        Step::BinaryC64 {
            a_byte_off,
            b_byte_off,
            c_byte_off,
            ..
        } => (
            vec![*a_byte_off / 4, *b_byte_off / 4],
            vec![*c_byte_off / 4],
        ),
        Step::ComplexNormSq {
            src_byte_off,
            dst_byte_off,
            ..
        } => (vec![*src_byte_off / 4], vec![*dst_byte_off / 4]),
        Step::ComplexNormSqBackward {
            z_byte_off,
            g_byte_off,
            dz_byte_off,
            ..
        } => (
            vec![*z_byte_off / 4, *g_byte_off / 4],
            vec![*dz_byte_off / 4],
        ),
        Step::ConjugateC64 {
            src_byte_off,
            dst_byte_off,
            ..
        } => (vec![*src_byte_off / 4], vec![*dst_byte_off / 4]),
        Step::FusedBinaryUnary {
            a_off,
            b_off,
            out_off,
            ..
        } => (vec![*a_off, *b_off], vec![*out_off]),
        Step::ElementwiseRegion {
            dst_off,
            input_offs,
            num_inputs,
            ..
        } => {
            let n = (*num_inputs as usize).min(input_offs.len());
            (input_offs[..n].to_vec(), vec![*dst_off])
        }
        Step::BatchElementwiseRegion {
            base_dst_off,
            batch_input_offs,
            num_batch,
            ..
        } => {
            let n = (*num_batch as usize).min(64);
            (batch_input_offs[..n].to_vec(), vec![*base_dst_off])
        }
        Step::Llada2GroupLimitedGate {
            sig_off,
            route_off,
            out_off,
            ..
        } => (vec![*sig_off, *route_off], vec![*out_off]),
        Step::MsDeformAttnHost {
            in_offs, out_off, ..
        } => (in_offs.iter().map(|(o, _)| *o).collect(), vec![*out_off]),
        Step::RocmGpuKernel {
            in_offs, out_off, ..
        } => (in_offs.iter().map(|(o, _)| *o).collect(), vec![*out_off]),
        Step::CollectiveHost {
            in_off, out_off, ..
        } => (vec![*in_off], vec![*out_off]),
        Step::UmapKnn {
            pairwise_off,
            out_off,
            ..
        } => (vec![*pairwise_off], vec![*out_off]),
    }
}

// ── fuse_elementwise_chains (port from rlx-cuda) ──────────────────────

pub(crate) fn fuse_elementwise_chains(schedule: Vec<Step>) -> Vec<Step> {
    let mut consumer_counts: HashMap<u32, usize> = HashMap::new();
    for step in &schedule {
        let (reads, _) = step_offsets(step);
        for r in &reads {
            *consumer_counts.entry(*r).or_insert(0) += 1;
        }
    }
    let mut out = Vec::with_capacity(schedule.len());
    let mut i = 0;
    while i < schedule.len() {
        if i + 1 < schedule.len() {
            let pair = (&schedule[i], &schedule[i + 1]);
            if let (
                Step::Binary {
                    n,
                    a_off,
                    b_off,
                    c_off,
                    op: bin_op,
                },
                Step::Unary {
                    n: n2,
                    in_off,
                    out_off,
                    op: un_op,
                },
            ) = pair
            {
                let single_consumer = consumer_counts.get(c_off).copied() == Some(1);
                // Only fuse real activations (ids 0–16). Cast unary steps use
                // ids ≥100 which fused_binary_unary.cu does not implement — a
                // fused cast would silently drop the trunc/saturate.
                if n == n2 && c_off == in_off && single_consumer && *un_op <= 16 {
                    out.push(Step::FusedBinaryUnary {
                        n: *n,
                        a_off: *a_off,
                        b_off: *b_off,
                        out_off: *out_off,
                        bin_op: *bin_op,
                        un_op: *un_op,
                    });
                    i += 2;
                    continue;
                }
            }
        }
        out.push(schedule[i].clone());
        i += 1;
    }
    out
}

impl Step {
    /// True when this Step variant honors active-extent dispatch (PLAN L1).
    /// Initial coverage matches rlx-cuda's: simple element-wise +
    /// reductions + softmax + LayerNorm + cumsum. Matmul and the
    /// rest still default to unsafe.
    pub fn safe_for_active_extent(&self) -> bool {
        matches!(
            self,
            Step::Binary { .. }
                | Step::Compare { .. }
                | Step::Unary { .. }
                | Step::Where { .. }
                | Step::Fma { .. }
                | Step::Reduce { .. }
                | Step::Softmax { .. }
                | Step::ReluBackward { .. }
                | Step::ActivationBackward { .. }
                | Step::SoftmaxCrossEntropy { .. }
                | Step::SoftmaxCrossEntropyWithLogits { .. }
                | Step::SoftmaxCrossEntropyBackward { .. }
                | Step::LayerNorm { .. }
                | Step::BatchNormInference { .. }
                | Step::BatchNormInferenceBackwardInput { .. }
                | Step::FusedResidualLn { .. }
                | Step::AdaLayerNorm { .. }
                | Step::GatedResidual { .. }
                | Step::AdaLayerNormBackward { .. }
                | Step::GatedResidualBackward { .. }
                | Step::Cumsum { .. }
                | Step::FusedBinaryUnary { .. }
                | Step::ElementwiseRegion { .. }
                | Step::BatchElementwiseRegion { .. }
        )
    }

    /// False when the step performs host-side work or stream sync during dispatch.
    pub fn graph_capture_safe(&self) -> bool {
        match self {
            Step::Im2ColHost { use_gpu, .. } | Step::Fft { use_gpu, .. } => *use_gpu,
            Step::GatedDeltaNet { .. }
            | Step::Llada2GroupLimitedGate { .. }
            | Step::MsDeformAttnHost { .. }
            | Step::CollectiveHost { .. }
            | Step::UmapKnn { .. }
            | Step::LogMelHost { .. }
            | Step::LogMelBackwardHost { .. }
            | Step::WelchPeaksHost { .. }
            | Step::RngNormal { .. }
            | Step::RngUniform { .. }
            | Step::ScanHost { .. }
            | Step::HostOp { .. }
            | Step::CpuIndexing { .. }
            | Step::SpdHost { .. }
            | Step::EighNative { .. }
            | Step::DenseSolveNative { .. }
            | Step::BatchedDenseSolveNative { .. }
            | Step::ReverseHost { .. }
            | Step::ArgReduceHost { .. }
            | Step::GruHost { .. }
            | Step::RnnHost { .. }
            | Step::Mamba2Host { .. }
            | Step::GaussianSplatRender { .. }
            | Step::GaussianSplatRenderBackward { .. }
            | Step::GaussianSplatPrepare { .. }
            | Step::GaussianSplatRasterize { .. } => false,
            _ => true,
        }
    }
}

pub(crate) fn schedule_graph_capture_safe(schedule: &[Step]) -> bool {
    schedule.iter().all(Step::graph_capture_safe)
}

pub(crate) fn step_is_tail_host(step: &Step) -> bool {
    matches!(
        step,
        Step::LogMelHost { .. } | Step::LogMelBackwardHost { .. } | Step::WelchPeaksHost { .. }
    )
}
