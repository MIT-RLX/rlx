// Dense / soft-label softmax cross-entropy along the last axis.
// loss[n] = logsumexp(logits[n]) - Σ_c targets[n,c]·logits[n,c]
// One work-item per row.
__kernel void softmax_cross_entropy(__global float* arena,
                                    uint outer, uint inner,
                                    uint logits_off, uint targets_off, uint out_off) {
    uint row = get_global_id(0);
    if (row >= outer || inner == 0u) return;
    uint lbase = logits_off + row * inner;
    uint tbase = targets_off + row * inner;

    float m = arena[lbase];
    for (uint i = 1; i < inner; i++) m = fmax(m, arena[lbase + i]);

    float s = 0.0f;
    float dot = 0.0f;
    for (uint i = 0; i < inner; i++) {
        float v = arena[lbase + i];
        s += exp(v - m);
        dot += arena[tbase + i] * v;
    }
    arena[out_off + row] = (m + log(s)) - dot;
}
