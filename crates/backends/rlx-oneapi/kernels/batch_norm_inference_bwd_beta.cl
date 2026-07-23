// BatchNormInferenceBackwardBeta: one work-item per channel.
__kernel void batch_norm_inference_bwd_beta(__global float* arena,
                                            uint count, uint channels,
                                            uint dy_off, uint out_off) {
    uint c = get_global_id(0);
    if (c >= channels) return;
    float acc = 0.0f;
    for (uint row = 0u; row < count; row++) {
        acc += arena[dy_off + row * channels + c];
    }
    arena[out_off + c] = acc;
}
