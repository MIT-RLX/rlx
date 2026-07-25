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

use rlx_ir::Op;

use crate::kernels::{
    ActivationBackwardParams, AdaLayerNormBackwardParams, AdaLayerNormParams, ArgmaxParams,
    AttentionBwdParams, AttentionParams, AxialRope2dParams, BatchElementwiseRegionParams,
    BinaryC64Params, BinaryParams, CastParams, ComplexCastParams, ComplexWirtingerParams,
    Conv1dParams, Conv2dParams, Conv3dBwdInputParams, Conv3dBwdWeightParams, Conv3dParams,
    CopyParams, CumsumBwdParams, CumsumParams, DequantMatmulMlxParams, DequantMatmulParams,
    ElementwiseRegionParams, ExpandParams, FakeQuantizeParams, FftButterflyStageParams, FmaParams,
    FusedResidualLnParams, FusedResidualLnTeeParams, FusedResidualRmsNormParams,
    GatedDeltaNetParams, GatedResidualBackwardParams, GatedResidualParams, GatherAxisParams,
    GatherBwdParams, GatherParams, GroupNormBwdParams, GroupedMatmulParams, GruParams,
    Im2Col2dParams, LayerNormBwdParams, LayerNormParams, Mamba2Params, MatmulQkvParams,
    MaxPool2dBwdParams, MaxPool3dBwdParams, NarrowConcatParams, Pool1dParams, Pool2dParams,
    Pool3dParams, ReduceParams, RmsNormBwdParams, RnnParams, RopeBwdParams, RopeParams,
    SampleParams, ScatterAddParams, SceBwdParams, SceParams, SelectiveScanParams, SoftmaxParams,
    TopKParams, TransposeParams, UmapKnnParams, UnaryParams, WelchPeaksGpuParams, WhereParams,
};

use super::helpers::{MatmulCompute, MatmulQkvKind};

/// f32 → f16 element-wise cast, mirroring an arena region into the
/// f16 shadow buffer. Used as a pre-pass before `matmul_coop16` so
/// the matmul's A operand (a runtime activation, not a Param) is
/// readable as f16.
///
/// Currently unused — the matmul_coop16 kernel stages A through
/// workgroup-shared memory directly from the f32 arena. Kept for
/// future paths that may want a one-shot cast (e.g. before a chain
/// of f16-only kernels operating on a fixed activation region).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CastF32ToF16Params {
    pub src_off: u32, // f32-element offset into arena (also f16-element offset)
    pub len: u32,
    pub _p0: u32,
    pub _p1: u32,
}
unsafe impl bytemuck::Pod for CastF32ToF16Params {}
unsafe impl bytemuck::Zeroable for CastF32ToF16Params {}

/// One dispatch step in the compiled schedule.
///
/// `dead_code` is allowed at the enum level: several variants carry
/// fields (mask_buf, meta_idx, compute_precision discriminants) that
/// are only consulted at compile time during bind-group construction,
/// or are kept to extend buffer lifetimes (mask_buf). A few variants
/// (CastF32ToF16, Copy, the unreachable F16 compute_precision) are
/// retained for future paths.
#[allow(dead_code)]
pub(crate) enum Step {
    CastF32ToF16 {
        params: CastF32ToF16Params,
    },
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
        act_id: u32, // 0xFFFF = no activation
        // True iff input B is a Param node — i.e. a model weight that
        // doesn't change between `run()` calls. Read from the f16
        // shadow buffer (half memory bandwidth) when set + the device
        // exposes SHADER_F16. Set at compile time; consulted only by
        // the dispatch arm.
        b_is_param: bool,
        // Compute precision for the inner FMA. F32 = full precision
        // (the historical / default path). F16 = mixed-precision
        // (operands cast to f16, multiply in f16 for 2× ALU on Apple,
        // accumulator in f32). Set at compile time from the IR's
        // dtype after AutoMixedPrecision policy.
        compute_precision: MatmulCompute,
    },
    Binary {
        params: BinaryParams,
    },
    Compare {
        params: BinaryParams,
    },
    Unary {
        params: UnaryParams,
        f16_mirror: bool,
    },
    Where {
        params: WhereParams,
    },
    Fma {
        params: FmaParams,
    },
    ReluBackward {
        params: ActivationBackwardParams,
    },
    ActivationBackward {
        params: ActivationBackwardParams,
    },
    Reduce {
        params: ReduceParams,
    },
    Softmax {
        params: SoftmaxParams,
    },
    SoftmaxCrossEntropy {
        params: SceParams,
    },
    SoftmaxCrossEntropyWithLogits {
        params: SceParams,
    },
    SoftmaxCrossEntropyBackward {
        params: SceBwdParams,
    },
    LayerNorm {
        params: LayerNormParams,
    },
    Cumsum {
        params: CumsumParams,
    },
    /// Native multi-kernel f32 FFT (gpu-fft dispatch strategy).
    FftGpu {
        src_off: u32,
        dst_off: u32,
        outer: u32,
        n: u32,
        inverse: u32,
        norm_scale: f32,
    },
    /// Explicit host FFT (D2H → rlx-cpu → H2D). Used when the native
    /// WGSL kernel cannot handle dtype / size / non-pow-2 constraints.
    FftHost {
        src_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        n_complex: u32,
        inverse: bool,
        norm_tag: u32,
        dtype_tag: u32,
    },
    /// Core Riemannian / SPD-manifold op (BiMap / ReEig / LogEig /
    /// SpdBatchNorm / SpdKarcherMean + backwards) — no WGSL eigen kernel, so
    /// each operand's arena span is read back, computed in F64 via
    /// `rlx_cpu::spd`, and the f32 result written back. See `crate::spd_host`.
    SpdHost {
        op: Op,
        inputs: Vec<crate::spd_host::SpdInput>,
        out_shape: rlx_ir::Shape,
        out_byte_off: usize,
    },
    /// General `Op::Scan` recurrence — D2H the input span → run the compiled
    /// body loop on the CPU → H2D. Enables IIR (`biquad`/`sosfilt`) on wgpu.
    ScanHost {
        desc: rlx_cpu::thunk::ScanHostDesc,
    },
    /// ScanBackward / ScanBackwardXs via arena readback + shared [`HostOpDesc`].
    HostOp {
        desc: rlx_cpu::thunk::HostOpDesc,
    },
    /// Native CPU ScatterNd / ScatterElements / GatherNd / GatherElements
    /// via span readback (correct for `I64` indices; no mini-graph rebuild).
    CpuIndexing {
        thunk: rlx_cpu::thunk::IndexingThunk,
    },
    /// Welch PSD top-K — D2H → rlx-cpu → H2D.
    WelchPeaksHost {
        spec_byte_off: u32,
        dst_byte_off: u32,
        welch_batch: u32,
        n_fft: u32,
        n_segments: u32,
        k: u32,
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
    /// NCHW im2col host path (D2H → rlx-cpu → H2D).
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
    },
    /// Host `Op::Conv2dBackwardWeight` (D2H → CPU → H2D). wgpu has no native
    /// conv-backward kernel and the autodiff decomposition corrupts the grad on
    /// wgpu; computing it on the CPU from the arena avoids the decomposition.
    Conv2dBackwardWeightHost {
        x_byte_off: u32,
        dy_byte_off: u32,
        dw_byte_off: u32,
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
    /// Host `Op::Conv2dBackwardInput` (D2H → CPU → H2D) — grad w.r.t. the conv
    /// input, for backprop through a conv into an earlier layer.
    Conv2dBackwardInputHost {
        dy_byte_off: u32,
        w_byte_off: u32,
        dx_byte_off: u32,
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
        dw_dil: u32,
        groups: u32,
    },
    /// Host fill for [`Op::RngNormal`] (fill → H2D).
    RngNormalHost {
        dst_byte_off: u32,
        len: u32,
        mean: f32,
        scale: f32,
        key: u64,
        op_seed: Option<f32>,
    },
    /// Host fill for [`Op::RngUniform`] (fill → H2D).
    RngUniformHost {
        dst_byte_off: u32,
        len: u32,
        low: f32,
        high: f32,
        key: u64,
        op_seed: Option<f32>,
    },
    /// Host-side buffer copy (recorded into a command encoder) used to
    /// stage small param tensors into the tail scratch region so kernels
    /// can bind a ≤4GiB window of the arena.
    BufferCopy {
        // u64: staging copies for >4 GiB GGUF decode arenas address tensors past
        // the 4 GiB mark; a u32 byte offset truncates and stages garbage.
        src_byte_off: u64,
        dst_byte_off: u64,
        bytes: u32,
    },
    Copy {
        params: CopyParams,
    },
    /// Numeric `Op::Cast` compute pass (`cast.wgsl`): float→int truncate+
    /// saturate, →Bool, or identity, on the f32-uniform arena. Distinct from
    /// `Copy` (which is a value-preserving move) so the cast semantics are
    /// explicit; identity casts still take the cheaper `BufferCopy` path.
    Cast {
        params: CastParams,
    },
    /// Standalone complex `Op::Cast` (`complex_cast.wgsl`): real↔C64,
    /// real↔C128, C64↔C128 lane moves on the f32-uniform arena. Kept off the
    /// fused-region path (the region kernel is scalar-per-f32-lane and cannot
    /// re-pair complex lanes). Dispatched over the complex-element index.
    ComplexCast {
        params: ComplexCastParams,
    },
    /// C64 element-wise binary (`binary_c64.wgsl`): Add/Sub/Mul/Div with each
    /// thread reading both `[re, im]` lanes of its operands. Excluded from
    /// `fuse_elementwise_chains` (the fused scalar-per-thread kernel can't read
    /// the partner `im` lane). Dispatched over the complex-element index.
    BinaryC64 {
        params: BinaryC64Params,
    },
    /// `|z|² = re² + im²` (`complex_wirtinger.wgsl` / `complex_norm_sq`).
    /// `n` is the complex-element count; output is real F32.
    ComplexNormSq {
        params: ComplexWirtingerParams,
    },
    /// Wirtinger VJP of ComplexNormSq: `dz = g · z` (`complex_norm_sq_backward`).
    ComplexNormSqBackward {
        params: ComplexWirtingerParams,
    },
    /// Element-wise C64 conjugate: `(re, -im)` (`conjugate_c64`).
    ConjugateC64 {
        params: ComplexWirtingerParams,
    },
    /// Ternary-pruned radix-2 butterfly stage (`fft_butterfly_stage.wgsl`).
    FftButterflyStage {
        params: FftButterflyStageParams,
    },
    /// PLAN L2 — fused N-ary element-wise region. Lowered from
    /// `Op::ElementwiseRegion` by `MarkElementwiseRegions`. Kernel
    /// interprets the chain encoding per-element (saves N kernel
    /// dispatches + N global-memory round-trips vs the decomposed
    /// atomic ops).
    ElementwiseRegion {
        params: ElementwiseRegionParams,
    },
    BatchElementwiseRegion {
        params: BatchElementwiseRegionParams,
    },
    Transpose {
        params: TransposeParams,
        meta_idx: usize,
    },
    /// Host permutation for a virtual arena whose input and output cannot
    /// share one storage binding window.
    TransposeHost {
        in_byte_off: usize,
        out_byte_off: usize,
        in_dims: Vec<u32>,
        out_dims: Vec<u32>,
        in_strides: Vec<usize>,
    },
    /// Host slice for a virtual arena whose source and destination straddle
    /// storage binding windows.
    NarrowHost {
        in_byte_off: usize,
        out_byte_off: usize,
        outer: u32,
        inner: u32,
        axis_in_size: u32,
        start: u32,
        axis_out_size: u32,
    },
    Narrow {
        params: NarrowConcatParams,
    },
    Concat {
        params: NarrowConcatParams,
    }, // one Step per input
    /// Host Concat when inputs/output cannot share one ≤4 GiB bind window
    /// (sharded arenas / F5-scale DiT). Reads each operand separately so the
    /// span need not be contiguous.
    ConcatHost {
        dst_byte_off: usize,
        outer: u32,
        inner: u32,
        total_axis: u32,
        /// `(src_byte_off, axis_len, numel)` per input.
        inputs: Vec<(usize, u32, u32)>,
    },
    /// Host-merge selected Concat pieces into an output that other inputs already
    /// filled via GPU Concat (cross-shard leftovers).
    ConcatHostPieces {
        dst_byte_off: usize,
        outer: u32,
        inner: u32,
        total_axis: u32,
        inputs: Vec<(usize, u32, u32)>,
        /// Axis start position for each entry in `inputs`.
        starts: Vec<u32>,
    },
    Gather {
        params: GatherParams,
    },
    GatherAxis {
        params: GatherAxisParams,
    },
    Attention {
        params: AttentionParams,
        mask_buf: Option<wgpu::Buffer>,
    },
    AttentionBackward {
        params: AttentionBwdParams,
        mask_buf: Option<wgpu::Buffer>,
    },
    Rope {
        params: RopeParams,
    },
    Expand {
        params: ExpandParams,
        meta_idx: usize,
    },
    /// Host broadcast when GPU Expand cannot cover the full output window
    /// (sharded F5 DiT: large Expand was only filling a prefix).
    ExpandHost {
        in_byte_off: usize,
        out_byte_off: usize,
        in_dims: Vec<u32>,
        out_dims: Vec<u32>,
    },
    Argmax {
        params: ArgmaxParams,
    },
    Pool2d {
        params: Pool2dParams,
    },
    Conv2d {
        params: Conv2dParams,
    },
    /// GPU im2col: gather a conv's receptive fields into a `col` matrix in the
    /// arena scratch tail. Always immediately followed by a `Step::Matmul`
    /// (weight @ col → NCHW output) that reuses the tiled f32 GEMM kernels.
    Im2ColGpu {
        params: Im2Col2dParams,
    },
    /// 2D register-blocked 1D conv (`conv1d_tiled.wgsl`) — same `Conv2dParams`
    /// as `Conv2d` but a `TCO×TL` output tile per thread that reuses the input
    /// across output channels. Used for the `one_d` (kw==1) vocoder convs.
    Conv2dTiled {
        params: Conv2dParams,
    },
    Pool1d {
        params: Pool1dParams,
    },
    Pool3d {
        params: Pool3dParams,
    },
    Conv1d {
        params: Conv1dParams,
    },
    Conv3d {
        params: Conv3dParams,
    },
    ScatterAdd {
        params: ScatterAddParams,
    },
    TopK {
        params: TopKParams,
    },
    WelchPeaksGpu {
        params: WelchPeaksGpuParams,
    },
    GroupedMatmul {
        params: GroupedMatmulParams,
    },
    Sample {
        params: SampleParams,
    },
    SelectiveScan {
        params: SelectiveScanParams,
    },
    /// Native WGSL Mamba-2 SSD scan (state_size ≤ 256).
    Mamba2 {
        params: Mamba2Params,
    },
    /// Native WGSL GRU (single-layer/unidir/no-carry, hidden ≤ 256).
    Gru {
        params: GruParams,
    },
    /// Native WGSL Elman RNN (single-layer/unidir/no-carry, hidden ≤ 256).
    Rnn {
        params: RnnParams,
    },
    /// Host-staged GRU fallback (multi-layer / bidir / carry / hidden > 256).
    GruHost {
        x: u32,
        w_ih: u32,
        w_hh: u32,
        b_ih: u32,
        b_hh: u32,
        h0: u32,
        dst: u32,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
        num_layers: u32,
        bidirectional: bool,
        carry: bool,
    },
    /// Host-staged Elman RNN fallback.
    RnnHost {
        x: u32,
        w_ih: u32,
        w_hh: u32,
        bias: u32,
        h0: u32,
        dst: u32,
        batch: u32,
        seq: u32,
        input_size: u32,
        hidden: u32,
        num_layers: u32,
        bidirectional: bool,
        carry: bool,
        relu: bool,
    },
    DequantMatmul {
        params: DequantMatmulParams,
    },
    /// Int8 block dequant+matmul on the host (Kitten / ONNX Int8BlockAsym).
    ///
    /// The wgpu SPIR-V `dequant_matmul` path is wrong on discrete NVIDIA
    /// (Kitten NSF → ~0.05 DC mush) while Apple Metal is fine. rlx-vulkan
    /// already hosts all non-GGUF DequantMatMul; mirror that here.
    DequantMatmulInt8Host {
        m: u32,
        k: u32,
        n: u32,
        block_size: u32,
        is_asymmetric: bool,
        x_byte_off: u64,
        w_byte_off: u64,
        scale_byte_off: u64,
        zp_byte_off: u64,
        out_byte_off: u64,
    },
    /// MLX affine / mxfp4 / mxfp8 on-device WGSL dequant-matmul.
    DequantMatmulMlx {
        params: DequantMatmulMlxParams,
    },
    /// MLX host fallback (`RLX_MLX_DEQUANT_GPU_DISABLE` or bisect).
    DequantMatmulMlxHost {
        m: u32,
        k: u32,
        n: u32,
        scheme: rlx_ir::quant::QuantScheme,
        x_byte_off: u64,
        w_byte_off: u64,
        scale_byte_off: u64,
        zp_byte_off: u64,
        out_byte_off: u64,
    },
    /// NCHW Conv2d on the host (discrete Vulkan/DX12 wgpu).
    ///
    /// Kitten's vocoder is dominated by 1×L "2d" convs; the SPIR-V tiled/direct
    /// paths still collapse NSF on NVIDIA even when Int8 dequant is hosted.
    /// Absolute byte offsets (may be weight-tagged).
    Conv2dHost {
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
        in_byte_off: u64,
        w_byte_off: u64,
        out_byte_off: u64,
    },
    /// Split-binding embedding gather for >4 GiB arenas. The table and the
    /// idx/output slots are more than one ≤4 GiB binding window apart, so the
    /// single-arena-binding `Step::Gather` cannot reach the output. Runs as a
    /// host segment (its own submission + copy-back), like `DequantMatmulGguf`.
    /// BYTE offsets are u64 (arena exceeds 4 GiB).
    GatherSplit {
        n_out: u32,
        n_idx: u32,
        dim: u32,
        vocab: u32,
        table_byte_off: u64,
        idx_byte_off: u64,
        out_byte_off: u64,
    },
    /// GGUF K-quant — host fused dequant+matmul between GPU segments.
    DequantMatmulGguf {
        m: u32,
        k: u32,
        n: u32,
        scheme_id: u32,
        // Arena BYTE offsets must be u64: GGUF decode arenas exceed 4 GiB
        // (Orpheus-3B Q4_K_M is ~10 GiB), so a u32 byte offset truncates for
        // any tensor past the 4 GiB mark and the host dequant reads garbage.
        x_byte_off: u64,
        w_byte_off: u64,
        out_byte_off: u64,
    },
    /// GGUF K-quant — host fused dequant+grouped matmul between GPU segments.
    DequantGroupedMatmulGguf {
        m: u32,
        k: u32,
        n: u32,
        num_experts: u32,
        scheme_id: u32,
        x_byte_off: u64,
        w_byte_off: u64,
        idx_byte_off: u64,
        out_byte_off: u64,
    },
    /// MLX-affine MoE grouped matmul on the host (no SPIR-V kernel; mirrors
    /// [`Step::DequantMatmulMlxHost`] with a stacked-expert weight + idx).
    DequantGroupedMatmulMlxHost {
        m: u32,
        k: u32,
        n: u32,
        num_experts: u32,
        scheme: rlx_ir::quant::QuantScheme,
        x_byte_off: u64,
        w_byte_off: u64,
        scale_byte_off: u64,
        zp_byte_off: u64,
        idx_byte_off: u64,
        out_byte_off: u64,
    },
    /// Gated-DeltaNet scan (qwen35 / Bonsai linear layers).
    /// `use_gpu=false` forces host readback (`RLX_WGPU_GDN_HOST=1`).
    GatedDeltaNet {
        params: GatedDeltaNetParams,
        /// Host-path byte offsets (only used when `use_gpu` is false).
        /// `u64` so logical addresses on >4 GiB sharded arenas are not truncated.
        q_byte_off: u64,
        k_byte_off: u64,
        v_byte_off: u64,
        g_byte_off: u64,
        beta_byte_off: u64,
        state_byte_off: u64,
        dst_byte_off: u64,
        use_gpu: bool,
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
    ConvTranspose2d {
        src_byte_off: u32,
        weight_byte_off: u32,
        dst_byte_off: u32,
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
    /// Native WGSL `Op::ConvTranspose3d` (reuses [`Conv3dParams`] layout).
    ConvTranspose3d {
        params: Conv3dParams,
    },
    /// Host-staged NCDHW `Op::ConvTranspose3d`.
    ConvTranspose3dHost {
        src_byte_off: u32,
        weight_byte_off: u32,
        dst_byte_off: u32,
        n: u32,
        c_in: u32,
        d: u32,
        h: u32,
        w_in: u32,
        c_out: u32,
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
    },
    /// Host-staged NCHW GroupNorm (readback → CPU → writeback).
    GroupNormHost {
        src_byte_off: u32,
        gamma_byte_off: u32,
        beta_byte_off: u32,
        dst_byte_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        num_groups: u32,
        eps: f32,
    },
    /// Host-staged NCHW LayerNorm2d (readback → CPU → writeback).
    LayerNorm2dHost {
        src_byte_off: u32,
        gamma_byte_off: u32,
        beta_byte_off: u32,
        dst_byte_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        eps: f32,
    },
    /// Host-staged nearest 2× upsample on NCHW (readback → CPU → writeback).
    ResizeNearest2xHost {
        src_byte_off: u32,
        dst_byte_off: u32,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
    },
    /// Host-staged batch-general reverse/flip (readback → CPU → writeback).
    ReverseHost {
        src_byte_off: u32,
        dst_byte_off: u32,
        dims: Vec<u32>,
        rev_mask: Vec<bool>,
        elem_bytes: u32,
    },
    /// Host-staged ArgMax/ArgMin (readback → CPU → writeback).
    ArgReduceHost {
        src_byte_off: u32,
        dst_byte_off: u32,
        outer: u32,
        reduced: u32,
        inner: u32,
        is_max: bool,
    },
    /// Host-staged `Op::AxialRope2d` (readback → CPU → writeback).
    AxialRope2dHost {
        src_byte_off: u32,
        dst_byte_off: u32,
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
    /// Native WGSL `Op::AxialRope2d`.
    AxialRope2d {
        params: AxialRope2dParams,
    },
    /// Native WGSL `Op::FakeQuantize` Fixed (scale input).
    FakeQuantizeFixed {
        params: FakeQuantizeParams,
    },
    /// Native WGSL `Op::FakeQuantize` PerBatch (derive scale from max abs).
    FakeQuantizePerBatch {
        params: FakeQuantizeParams,
    },
    /// Native WGSL GroupNorm backward w.r.t. input.
    GroupNormBackwardInput {
        params: GroupNormBwdParams,
    },
    /// Native WGSL GroupNorm backward w.r.t. gamma.
    GroupNormBackwardGamma {
        params: GroupNormBwdParams,
    },
    /// Native WGSL GroupNorm backward w.r.t. beta.
    GroupNormBackwardBeta {
        params: GroupNormBwdParams,
    },
    /// Native WGSL `Op::MaxPool2dBackward` (f32 element offsets in params).
    MaxPool2dBackward {
        params: MaxPool2dBwdParams,
    },
    /// Native WGSL `Op::MaxPool3dBackward` (f32 element offsets in params).
    MaxPool3dBackward {
        params: MaxPool3dBwdParams,
    },
    /// Native WGSL `Op::Conv3dBackwardInput` (f32 element offsets in params).
    Conv3dBackwardInput {
        params: Conv3dBwdInputParams,
    },
    /// Native WGSL `Op::Conv3dBackwardWeight` (f32 element offsets in params).
    Conv3dBackwardWeight {
        params: Conv3dBwdWeightParams,
    },
    Llada2GroupLimitedGate {
        sig_byte_off: u32,
        route_byte_off: u32,
        out_byte_off: u32,
        n_elems: u32,
        attrs: [u8; 20],
    },
    UmapKnn {
        params: UmapKnnParams,
    },
    /// Raw-GPU custom op (`WgpuGpuKernel`): a WGSL compute dispatch straight
    /// against the arena buffer, no host roundtrip. Its bind group + storage
    /// params buffer live in the parallel `bind_groups` / `uniforms` vecs like
    /// any other GPU step; `name` resolves the cached pipeline at dispatch.
    WgpuGpuKernel {
        name: String,
        workgroups: (u32, u32, u32),
    },
    /// Small-`n` host k-NN (partial arena read/write; avoids GPU launch overhead).
    UmapKnnHost {
        pairwise_byte_off: u32,
        out_byte_off: u32,
        n: u32,
        k: u32,
    },
    /// Fused multi-scale deformable attention (host compute over arena buffers).
    MsDeformAttnHost {
        in_offs: Vec<(u32, u32)>, // (byte_off, byte_len) per input
        out_byte_off: u32,
        out_bytes: u32,
        attrs: Vec<u8>,
    },
    /// Host-delegate `collective.*` op (all_reduce / all_gather / reduce_scatter
    /// / Megatron f/g). Staged off-GPU and run through the registered rlx-cpu
    /// collective kernel. See `crate::collective_host`.
    CollectiveHost {
        name: String,
        in_byte_off: u32,
        in_bytes: u32,
        out_byte_off: u32,
        out_bytes: u32,
        attrs: Vec<u8>,
    },
    /// Generic host-delegate for any `onnx.*` custom op with a registered rlx-cpu
    /// reference kernel but no wgpu shader (Einsum / Mod / ScatterND / …). Stages
    /// each operand off-GPU (dtype-aware, per its `Shape`), runs the CPU
    /// reference, and writes the result back. See `crate::custom_host`.
    CustomHost {
        name: String,
        in_specs: Vec<(u32, rlx_ir::Shape)>, // (byte_off, shape)
        out_byte_off: u32,
        out_shape: rlx_ir::Shape,
        attrs: Vec<u8>,
    },
    /// 3D Gaussian splat forward (CPU reference between segments).
    #[cfg(feature = "splat")]
    GaussianSplatRender {
        positions_byte_off: u32,
        positions_len: u32,
        scales_byte_off: u32,
        scales_len: u32,
        rotations_byte_off: u32,
        rotations_len: u32,
        opacities_byte_off: u32,
        opacities_len: u32,
        colors_byte_off: u32,
        colors_len: u32,
        sh_coeffs_byte_off: u32,
        sh_coeffs_len: u32,
        meta_byte_off: u32,
        dst_byte_off: u32,
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
    /// Backward splat — host round-trip via rlx-cpu/splat.
    #[cfg(feature = "splat")]
    GaussianSplatRenderBackward {
        positions_byte_off: u32,
        positions_len: u32,
        scales_byte_off: u32,
        scales_len: u32,
        rotations_byte_off: u32,
        rotations_len: u32,
        opacities_byte_off: u32,
        opacities_len: u32,
        colors_byte_off: u32,
        colors_len: u32,
        sh_coeffs_byte_off: u32,
        sh_coeffs_len: u32,
        meta_byte_off: u32,
        d_loss_byte_off: u32,
        d_loss_len: u32,
        packed_byte_off: u32,
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
    #[cfg(feature = "splat")]
    GaussianSplatPrepare {
        positions_byte_off: u32,
        positions_len: u32,
        scales_byte_off: u32,
        scales_len: u32,
        rotations_byte_off: u32,
        rotations_len: u32,
        opacities_byte_off: u32,
        opacities_len: u32,
        colors_byte_off: u32,
        colors_len: u32,
        sh_coeffs_byte_off: u32,
        sh_coeffs_len: u32,
        meta_byte_off: u32,
        meta_len: u32,
        prep_byte_off: u32,
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
    #[cfg(feature = "splat")]
    GaussianSplatRasterize {
        prep_byte_off: u32,
        prep_len: u32,
        meta_byte_off: u32,
        meta_len: u32,
        dst_byte_off: u32,
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
        params: RmsNormBwdParams,
    },
    RmsNormBackwardGamma {
        params: RmsNormBwdParams,
    },
    RmsNormBackwardBeta {
        params: RmsNormBwdParams,
    },
    LayerNormBackwardInput {
        params: LayerNormBwdParams,
    },
    LayerNormBackwardGammaPartial {
        params: LayerNormBwdParams,
        num_workgroups: u32,
    },
    LayerNormBackwardGammaReduce {
        params: LayerNormBwdParams,
    },
    RopeBackward {
        params: RopeBwdParams,
    },
    CumsumBackward {
        params: CumsumBwdParams,
    },
    GatherBackward {
        params: GatherBwdParams,
    },
    FusedResidualLn {
        params: FusedResidualLnParams,
    },
    /// Split-write QKV matmul. Replaces a (FusedMatMulBiasAct → Narrow×3)
    /// pattern with one dispatch that writes Q, K, V into separate
    /// contiguous buffers from a single matmul pass. See
    /// `kernels/matmul_qkv.wgsl`.
    MatmulQkv {
        params: MatmulQkvParams,
        kind: MatmulQkvKind,
    },
    /// `fused_residual_ln_tee` — does (Add → LN) but writes the sum to
    /// a separate arena slot (the eliminated Add's old slot). Fires
    /// when the Add has multi-consumer downstream (vision pre-norm).
    FusedResidualLnTee {
        params: FusedResidualLnTeeParams,
    },
    FusedResidualRmsNorm {
        params: FusedResidualRmsNormParams,
    },
    AdaLayerNorm {
        params: AdaLayerNormParams,
    },
    GatedResidual {
        params: GatedResidualParams,
    },
    AdaLayerNormBackward {
        params: AdaLayerNormBackwardParams,
    },
    GatedResidualBackward {
        params: GatedResidualBackwardParams,
    },
}

impl Step {
    /// True when this Step variant honors active-extent dispatch (PLAN L1).
    /// Coverage: simple element-wise + reductions + matmul + linalg
    /// + reductions/argmax/topk/sample + gather + conv + pool +
    /// scatter (zero output + scale num_updates) + macros gated to
    /// batch=1 (Attention, SelectiveScan).
    pub fn safe_for_active_extent(&self) -> bool {
        match self {
            Step::Binary { .. }
            | Step::Compare { .. }
            | Step::Unary { .. }
            | Step::Where { .. }
            | Step::Fma { .. }
            | Step::ReluBackward { .. }
            | Step::ActivationBackward { .. }
            | Step::Reduce { .. }
            | Step::Softmax { .. }
            | Step::SoftmaxCrossEntropy { .. }
            | Step::SoftmaxCrossEntropyWithLogits { .. }
            | Step::SoftmaxCrossEntropyBackward { .. }
            | Step::LayerNorm { .. }
            | Step::FusedResidualLn { .. }
            | Step::FusedResidualLnTee { .. }
            | Step::FusedResidualRmsNorm { .. }
            | Step::AdaLayerNorm { .. }
            | Step::GatedResidual { .. }
            | Step::AdaLayerNormBackward { .. }
            | Step::GatedResidualBackward { .. }
            | Step::Cumsum { .. }
            | Step::Copy { .. }
            | Step::Cast { .. }
            | Step::ComplexCast { .. }
            | Step::BinaryC64 { .. }
            | Step::ComplexNormSq { .. }
            | Step::ComplexNormSqBackward { .. }
            | Step::ConjugateC64 { .. }
            | Step::FftButterflyStage { .. }
            | Step::ElementwiseRegion { .. }
            | Step::BatchElementwiseRegion { .. }
            | Step::Argmax { .. }
            | Step::TopK { .. }
            | Step::WelchPeaksGpu { .. }
            | Step::Sample { .. }
            | Step::Gather { .. }
            | Step::GatherAxis { .. }
            | Step::GatherSplit { .. }
            | Step::GroupedMatmul { .. }
            | Step::DequantMatmul { .. }
            | Step::DequantMatmulMlx { .. }
            | Step::DequantMatmulMlxHost { .. }
            | Step::DequantMatmulGguf { .. }
            | Step::DequantMatmulInt8Host { .. }
            | Step::Conv2dHost { .. }
            | Step::DequantGroupedMatmulGguf { .. }
            | Step::DequantGroupedMatmulMlxHost { .. }
            | Step::Lstm { .. }
            | Step::ConvTranspose2d { .. }
            | Step::ConvTranspose3dHost { .. }
            | Step::GroupNormHost { .. }
            | Step::LayerNorm2dHost { .. }
            | Step::ResizeNearest2xHost { .. }
            | Step::ReverseHost { .. }
            | Step::ArgReduceHost { .. }
            | Step::AxialRope2dHost { .. }
            | Step::GruHost { .. }
            | Step::RnnHost { .. }
            | Step::Llada2GroupLimitedGate { .. }
            | Step::UmapKnn { .. }
            | Step::UmapKnnHost { .. }
            | Step::MsDeformAttnHost { .. }
            | Step::CollectiveHost { .. }
            | Step::Conv1d { .. }
            | Step::Conv2d { .. }
            | Step::Conv2dTiled { .. }
            | Step::Conv3d { .. }
            | Step::ConvTranspose3d { .. }
            | Step::Pool1d { .. }
            | Step::Pool2d { .. }
            | Step::Pool3d { .. }
            | Step::MaxPool2dBackward { .. }
            | Step::MaxPool3dBackward { .. }
            | Step::Conv3dBackwardInput { .. }
            | Step::Conv3dBackwardWeight { .. }
            | Step::AxialRope2d { .. }
            | Step::FakeQuantizeFixed { .. }
            | Step::FakeQuantizePerBatch { .. }
            | Step::GroupNormBackwardInput { .. }
            | Step::GroupNormBackwardGamma { .. }
            | Step::GroupNormBackwardBeta { .. }
            | Step::ScatterAdd { .. }
            | Step::BufferCopy { .. } => true,
            // FFT: full-extent transform per row, no active-extent
            // scaling. Marking true so a graph that mixes FFT with
            // active-extent-safe ops still gets the optimization for
            // the rest of the schedule.
            Step::FftGpu { .. }
            | Step::FftHost { .. }
            | Step::ScanHost { .. }
            | Step::HostOp { .. }
            | Step::CpuIndexing { .. }
            | Step::ConcatHost { .. }
            | Step::ConcatHostPieces { .. }
            | Step::TransposeHost { .. }
            | Step::NarrowHost { .. }
            | Step::ExpandHost { .. } => true,
            // SPD ops transform full square matrices (no bucket/seq axis to
            // scale); mark true so a mixed graph still gets the fast path for
            // its other ops.
            Step::SpdHost { .. } => true,
            Step::Im2ColHost { .. }
            | Step::Conv2dBackwardWeightHost { .. }
            | Step::Conv2dBackwardInputHost { .. }
            | Step::RngNormalHost { .. }
            | Step::RngUniformHost { .. }
            | Step::WelchPeaksHost { .. }
            | Step::LogMelHost { .. }
            | Step::LogMelBackwardHost { .. } => true,
            // Matmul: c_batch_stride is set at compile time at full m,
            // independent of params.m. With scaled m, threads with
            // global_row >= m early-return; per-batch output offsets
            // stay correct. Safe at any batch.
            Step::Matmul { .. } => true,
            // im2col params (offsets + spatial extents) are baked at compile
            // time for a fixed conv shape; they cannot be active-extent
            // scaled. Returning false disables the fast path for any graph
            // that contains an im2col conv (conv-heavy models don't use it).
            Step::Im2ColGpu { .. } => false,
            // Generic host custom-op reads exact per-node shapes off the arena;
            // disable the active-extent optimization when one is present.
            Step::CustomHost { .. } => false,
            // Raw-GPU custom op: window/params/workgroups are baked at compile
            // time from the full shapes — no active-extent scaling.
            Step::WgpuGpuKernel { .. } => false,
            // Same active-extent reasoning as Matmul: per-batch output
            // strides are baked at compile time, scaling m only adjusts
            // the per-thread bound check.
            Step::MatmulQkv { .. } => true,
            Step::CastF32ToF16 { .. } => true,
            // Attention: WGSL kernel uses `seq_q_stride`/`seq_k_stride`
            // (full extent, set at compile time) for per-(batch, head)
            // offset math, and `params.seq_q`/`params.seq_k` for loop
            // bounds only. Scaling seq_q/seq_k shrinks the iteration
            // without corrupting per-head strides. Safe at any batch.
            Step::Attention { .. } => true,
            Step::AttentionBackward { .. } => true,
            // SelectiveScan: WGSL kernel uses `params.seq_stride`
            // (full extent, set at compile time) for per-batch stride
            // math; `params.seq` is the loop bound only. Safe at any
            // batch under active-extent scaling of seq.
            Step::SelectiveScan { .. } => true,
            // GatedDeltaNet: same seq_stride discipline as SelectiveScan.
            // Host fallback does not scale (full arena readback).
            Step::GatedDeltaNet { use_gpu: true, .. } => true,
            Step::GatedDeltaNet { use_gpu: false, .. } => true,
            // Mamba2: same seq_stride discipline as SelectiveScan.
            Step::Mamba2 { .. } => true,
            // GRU/RNN: per-batch workgroups; seq_stride is full-extent, seq is
            // the loop bound only. Safe under active-extent scaling.
            Step::Gru { .. } => true,
            Step::Rnn { .. } => true,
            // Narrow + Concat: kernel iterates `params.total` in
            // row-major order with outer as the leading dim. Scaling
            // total by actual/upper effectively scales outer by the
            // same factor (since total = outer * axis_size * inner).
            // Output positions past scaled_total stay untouched.
            // **Conservative assumption**: bucket axis is outer.
            // Cases where the bucket axis is the narrow/concat axis
            // itself are unsafe — fall back to full extent there.
            Step::Narrow { .. } => true,
            Step::Concat { .. } => true,
            // Rope: WGSL kernel uses `seq_stride` (full extent, set
            // at compile time) for per-batch buffer offset math and
            // explicit `batch` for index decomposition. `params.seq`
            // and `params.n_total` are runtime-scaled iteration
            // bounds. Safe at any batch.
            Step::Rope { .. } => true,
            // Transpose: precomputed `bucket_outermost` flag in
            // params (set to 1 at compile time iff `perm[0] == 0`).
            // Active path scales `out_total` by `actual / upper`
            // proportional to `out_dim_0`. Other transposes (where
            // bucket axis moves) fall back to full extent.
            Step::Transpose { params, .. } => params.bucket_outermost == 1,
            // Expand: when `bucket_outermost==1`, run/dispatch scale `out_total`.
            // When 0 (broadcast into the bucket axis), launch stays full-extent
            // but must NOT veto the whole-graph active-extent gate — Bonsai-27B
            // has many such expands and was stuck computing full max_seq.
            Step::Expand { .. } => true,
            // Training backward ops: not used in inference; disable
            // active-extent fast path until individually audited.
            Step::RmsNormBackwardInput { .. }
            | Step::RmsNormBackwardGamma { .. }
            | Step::RmsNormBackwardBeta { .. }
            | Step::LayerNormBackwardInput { .. }
            | Step::LayerNormBackwardGammaPartial { .. }
            | Step::LayerNormBackwardGammaReduce { .. }
            | Step::RopeBackward { .. }
            | Step::CumsumBackward { .. }
            | Step::GatherBackward { .. } => false,
            #[cfg(feature = "splat")]
            Step::GaussianSplatRender { .. }
            | Step::GaussianSplatRenderBackward { .. }
            | Step::GaussianSplatPrepare { .. }
            | Step::GaussianSplatRasterize { .. } => false,
        }
    }
}

pub(crate) fn step_name(step: &Step) -> &'static str {
    match step {
        Step::CastF32ToF16 { .. } => "cast_f32_to_f16",
        Step::Matmul { .. } => "matmul",
        Step::Binary { .. } => "binary",
        Step::Compare { .. } => "compare",
        Step::Unary { .. } => "unary",
        Step::Where { .. } => "where",
        Step::Fma { .. } => "fma",
        Step::ReluBackward { .. } => "relu_backward",
        Step::ActivationBackward { .. } => "activation_backward",
        Step::Reduce { .. } => "reduce",
        Step::Softmax { .. } => "softmax",
        Step::SoftmaxCrossEntropy { .. } => "softmax_cross_entropy",
        Step::SoftmaxCrossEntropyWithLogits { .. } => "softmax_cross_entropy_with_logits",
        Step::SoftmaxCrossEntropyBackward { .. } => "softmax_cross_entropy_backward",
        Step::LayerNorm { .. } => "layer_norm",
        Step::Cumsum { .. } => "cumsum",
        Step::FftGpu { .. } => "fft_gpu",
        Step::FftHost { .. } => "fft_host",
        Step::WelchPeaksHost { .. } => "welch_peaks_host",
        Step::LogMelHost { .. } => "log_mel_host",
        Step::LogMelBackwardHost { .. } => "log_mel_backward_host",
        Step::Im2ColHost { .. } => "im2col_host",
        Step::Conv2dBackwardWeightHost { .. } => "conv2d_backward_weight_host",
        Step::Conv2dBackwardInputHost { .. } => "conv2d_backward_input_host",
        Step::RngNormalHost { .. } => "rng_normal_host",
        Step::RngUniformHost { .. } => "rng_uniform_host",
        Step::BufferCopy { .. } => "buffer_copy",
        Step::Copy { .. } => "copy",
        Step::Cast { .. } => "cast",
        Step::ComplexCast { .. } => "complex_cast",
        Step::BinaryC64 { .. } => "binary_c64",
        Step::ComplexNormSq { .. } => "complex_norm_sq",
        Step::ComplexNormSqBackward { .. } => "complex_norm_sq_backward",
        Step::FftButterflyStage { .. } => "fft_butterfly_stage",
        Step::ConjugateC64 { .. } => "conjugate_c64",
        Step::Transpose { .. } => "transpose",
        Step::TransposeHost { .. } => "transpose_host",
        Step::NarrowHost { .. } => "narrow_host",
        Step::Narrow { .. } => "narrow",
        Step::Concat { .. } => "concat",
        Step::ConcatHost { .. } => "concat_host",
        Step::ConcatHostPieces { .. } => "concat_host_pieces",
        Step::Gather { .. } => "gather",
        Step::GatherAxis { .. } => "gather_axis",
        Step::Attention { .. } => "attention",
        Step::AttentionBackward { .. } => "attention_bwd",
        Step::Rope { .. } => "rope",
        Step::Expand { .. } => "expand",
        Step::ExpandHost { .. } => "expand_host",
        Step::Argmax { .. } => "argmax",
        Step::Pool2d { .. } => "pool2d",
        Step::Conv2d { .. } => "conv2d",
        Step::Conv2dTiled { .. } => "conv2d_tiled",
        Step::Im2ColGpu { .. } => "im2col_gpu",
        Step::Pool1d { .. } => "pool1d",
        Step::Pool3d { .. } => "pool3d",
        Step::Conv1d { .. } => "conv1d",
        Step::Conv3d { .. } => "conv3d",
        Step::ScatterAdd { .. } => "scatter_add",
        Step::TopK { .. } => "topk",
        Step::WelchPeaksGpu { .. } => "welch_peaks_gpu",
        Step::GroupedMatmul { .. } => "grouped_matmul",
        Step::Sample { .. } => "sample",
        Step::SelectiveScan { .. } => "selective_scan",
        Step::Mamba2 { .. } => "mamba2",
        Step::Gru { .. } => "gru",
        Step::Rnn { .. } => "rnn",
        Step::GruHost { .. } => "gru_host",
        Step::RnnHost { .. } => "rnn_host",
        Step::DequantMatmul { .. } => "dequant_matmul",
        Step::DequantMatmulMlx { .. } => "dequant_matmul_mlx",
        Step::DequantMatmulMlxHost { .. } => "dequant_matmul_mlx_host",
        Step::GatherSplit { .. } => "gather_split",
        Step::DequantMatmulGguf { .. } => "dequant_matmul_gguf",
        Step::DequantMatmulInt8Host { .. } => "dequant_matmul_int8_host",
        Step::Conv2dHost { .. } => "conv2d_host",
        Step::DequantGroupedMatmulGguf { .. } => "dequant_grouped_matmul_gguf",
        Step::DequantGroupedMatmulMlxHost { .. } => "dequant_grouped_matmul_mlx_host",
        Step::GatedDeltaNet { .. } => "gated_delta_net",
        Step::Lstm { .. } => "lstm",
        Step::ConvTranspose2d { .. } => "conv_transpose2d",
        Step::ConvTranspose3d { .. } => "conv_transpose3d",
        Step::ConvTranspose3dHost { .. } => "conv_transpose3d_host",
        Step::GroupNormHost { .. } => "group_norm_host",
        Step::LayerNorm2dHost { .. } => "layer_norm2d_host",
        Step::ResizeNearest2xHost { .. } => "resize_nearest2x_host",
        Step::ReverseHost { .. } => "reverse_host",
        Step::ArgReduceHost { .. } => "argreduce_host",
        Step::AxialRope2dHost { .. } => "axial_rope2d_host",
        Step::AxialRope2d { .. } => "axial_rope2d",
        Step::FakeQuantizeFixed { .. } => "fake_quantize_fixed",
        Step::FakeQuantizePerBatch { .. } => "fake_quantize_perbatch",
        Step::GroupNormBackwardInput { .. } => "group_norm_backward_input",
        Step::GroupNormBackwardGamma { .. } => "group_norm_backward_gamma",
        Step::GroupNormBackwardBeta { .. } => "group_norm_backward_beta",
        Step::MaxPool2dBackward { .. } => "maxpool2d_backward",
        Step::MaxPool3dBackward { .. } => "maxpool3d_backward",
        Step::Conv3dBackwardInput { .. } => "conv3d_backward_input",
        Step::Conv3dBackwardWeight { .. } => "conv3d_backward_weight",
        Step::Llada2GroupLimitedGate { .. } => "llada2_group_limited_gate",
        Step::UmapKnn { .. } => "umap_knn",
        Step::WgpuGpuKernel { .. } => "wgpu_gpu_kernel",
        Step::UmapKnnHost { .. } => "umap_knn_host",
        Step::MsDeformAttnHost { .. } => "ms_deform_attn_host",
        Step::CollectiveHost { .. } => "collective_host",
        Step::CustomHost { .. } => "custom_host",
        Step::ScanHost { .. } => "scan_host",
        Step::HostOp { .. } => "host_op",
        Step::CpuIndexing { .. } => "cpu_indexing",
        Step::SpdHost { .. } => "spd_host",
        #[cfg(feature = "splat")]
        Step::GaussianSplatRender { .. } => "gaussian_splat_render",
        #[cfg(feature = "splat")]
        Step::GaussianSplatRenderBackward { .. } => "gaussian_splat_render_backward",
        #[cfg(feature = "splat")]
        Step::GaussianSplatPrepare { .. } => "gaussian_splat_prepare",
        #[cfg(feature = "splat")]
        Step::GaussianSplatRasterize { .. } => "gaussian_splat_rasterize",
        Step::RmsNormBackwardInput { .. } => "rms_norm_backward_input",
        Step::RmsNormBackwardGamma { .. } => "rms_norm_backward_gamma",
        Step::RmsNormBackwardBeta { .. } => "rms_norm_backward_beta",
        Step::LayerNormBackwardInput { .. } => "layer_norm_backward_input",
        Step::LayerNormBackwardGammaPartial { .. } => "layer_norm_backward_gamma_partial",
        Step::LayerNormBackwardGammaReduce { .. } => "layer_norm_backward_gamma_reduce",
        Step::RopeBackward { .. } => "rope_backward",
        Step::CumsumBackward { .. } => "cumsum_backward",
        Step::GatherBackward { .. } => "gather_backward",
        Step::FusedResidualLn { .. } => "fused_residual_ln",
        Step::FusedResidualLnTee { .. } => "fused_residual_ln_tee",
        Step::FusedResidualRmsNorm { .. } => "fused_residual_rms_norm",
        Step::AdaLayerNorm { .. } => "ada_layer_norm",
        Step::GatedResidual { .. } => "gated_residual",
        Step::AdaLayerNormBackward { .. } => "ada_layer_norm_backward",
        Step::GatedResidualBackward { .. } => "gated_residual_backward",
        Step::MatmulQkv { .. } => "matmul_qkv",
        Step::ElementwiseRegion { .. } => "elementwise_region",
        Step::BatchElementwiseRegion { .. } => "batch_elementwise_region",
    }
}

pub(crate) fn step_is_tail_host(step: &Step) -> bool {
    matches!(
        step,
        Step::WelchPeaksHost { .. } | Step::LogMelHost { .. } | Step::LogMelBackwardHost { .. }
    )
}

pub(crate) fn step_runs_on_host(step: &Step) -> bool {
    match step {
        Step::GatedDeltaNet { use_gpu, .. } => !*use_gpu,
        Step::GatherSplit { .. }
        | Step::DequantMatmulGguf { .. }
        | Step::DequantMatmulInt8Host { .. }
        | Step::DequantMatmulMlxHost { .. }
        | Step::Conv2dHost { .. }
        | Step::DequantGroupedMatmulGguf { .. }
        | Step::DequantGroupedMatmulMlxHost { .. }
        | Step::Lstm { .. }
        | Step::ConvTranspose2d { .. }
        | Step::ConvTranspose3dHost { .. }
        | Step::GroupNormHost { .. }
        | Step::LayerNorm2dHost { .. }
        | Step::ResizeNearest2xHost { .. }
        | Step::ReverseHost { .. }
        | Step::ArgReduceHost { .. }
        | Step::AxialRope2dHost { .. }
        | Step::GruHost { .. }
        | Step::RnnHost { .. }
        | Step::Llada2GroupLimitedGate { .. }
        | Step::UmapKnnHost { .. }
        | Step::MsDeformAttnHost { .. }
        | Step::CollectiveHost { .. }
        | Step::CustomHost { .. }
        | Step::FftHost { .. }
        | Step::ScanHost { .. }
        | Step::HostOp { .. }
        | Step::CpuIndexing { .. }
        | Step::ConcatHost { .. }
        | Step::ConcatHostPieces { .. }
        | Step::TransposeHost { .. }
        | Step::NarrowHost { .. }
        | Step::ExpandHost { .. }
        | Step::SpdHost { .. }
        | Step::Im2ColHost { .. }
        | Step::Conv2dBackwardWeightHost { .. }
        | Step::Conv2dBackwardInputHost { .. }
        | Step::RngNormalHost { .. }
        | Step::RngUniformHost { .. }
        | Step::BufferCopy { .. } => true,
        #[cfg(feature = "splat")]
        Step::GaussianSplatRender { .. }
        | Step::GaussianSplatRenderBackward { .. }
        | Step::GaussianSplatPrepare { .. }
        | Step::GaussianSplatRasterize { .. } => true,
        _ => false,
    }
}

pub(crate) fn step_needs_pass_flush(step: &Step, prev: &Step) -> bool {
    match step {
        Step::CastF32ToF16 { .. } => matches!(
            prev,
            Step::Unary {
                f16_mirror: false,
                ..
            }
        ),
        Step::Matmul {
            compute_precision: MatmulCompute::CoopF16Vk,
            ..
        }
        | Step::MatmulQkv {
            kind: MatmulQkvKind::CoopF16Vk,
            ..
        } => matches!(prev, Step::Unary { .. } | Step::CastF32ToF16 { .. }),
        // Discrete Vulkan/DX12: Kitten NSF collapses when Binary/Unary follow
        // MatMul in the same compute pass (peak ~0.05). End the pass so the
        // matmul writes are visible before elementwise. Metal is fine without.
        Step::Binary { .. }
        | Step::Unary { .. }
        | Step::Where { .. }
        | Step::Fma { .. }
        | Step::Compare { .. }
            if crate::device::coop_discrete_backend() =>
        {
            matches!(
                prev,
                Step::Matmul { .. }
                    | Step::MatmulQkv { .. }
                    | Step::BufferCopy { .. }
                    | Step::Gather { .. }
                    | Step::Reduce { .. }
            )
        }
        _ => false,
    }
}
