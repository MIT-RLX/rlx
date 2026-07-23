// Integer-label softmax cross-entropy along the last axis.
// loss[n] = logsumexp(logits[n]) - logits[n, label]
// One work-item per row. Labels are f32-encoded class indices.
__kernel void softmax_cross_entropy_with_logits(__global float* arena,
                                                uint outer, uint inner,
                                                uint logits_off, uint labels_off,
                                                uint out_off) {
    uint row = get_global_id(0);
    if (row >= outer || inner == 0u) return;
    uint lbase = logits_off + row * inner;

    float m = arena[lbase];
    for (uint i = 1; i < inner; i++) m = fmax(m, arena[lbase + i]);

    float s = 0.0f;
    for (uint i = 0; i < inner; i++) s += exp(arena[lbase + i] - m);

    uint label = (uint)arena[labels_off + row];
    if (label >= inner) label = inner - 1u;
    arena[out_off + row] = (m + log(s)) - arena[lbase + label];
}
