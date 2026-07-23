// C64 |z|² → F32. Dispatched over complex-element index k ∈ [0, n).
// Interleaved [re, im] f32 lanes (matches CUDA complex_wirtinger.cu).
__kernel void complex_norm_sq(__global float* arena,
                              uint n, uint src_off, uint dst_off) {
    uint k = get_global_id(0);
    if (k >= n) return;
    float re = arena[src_off + 2u * k];
    float im = arena[src_off + 2u * k + 1u];
    arena[dst_off + k] = re * re + im * im;
}
