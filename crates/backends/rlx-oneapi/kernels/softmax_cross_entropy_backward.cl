// Softmax cross-entropy backward (integer labels), one work-item per row:
//   dlogits[n,c] = (softmax(logits[n])[c] - [c==label]) * d_loss[n]
__kernel void softmax_cross_entropy_backward(__global float* arena,
                                             uint outer, uint inner,
                                             uint logits_off, uint labels_off,
                                             uint d_loss_off, uint out_off) {
    uint row = get_global_id(0);
    if (row >= outer || inner == 0u) return;
    uint lbase = logits_off + row * inner;
    uint obase = out_off + row * inner;

    float m = arena[lbase];
    for (uint i = 1; i < inner; i++) m = fmax(m, arena[lbase + i]);

    float s = 0.0f;
    for (uint i = 0; i < inner; i++) s += exp(arena[lbase + i] - m);
    float inv = 1.0f / s;
    float scale = arena[d_loss_off + row];
    uint label = (uint)arena[labels_off + row];
    if (label >= inner) label = inner - 1u;

    for (uint k = 0; k < inner; k++) {
        float p = exp(arena[lbase + k] - m) * inv;
        float oh = (k == label) ? 1.0f : 0.0f;
        arena[obase + k] = (p - oh) * scale;
    }
}
