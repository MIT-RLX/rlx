// Numerically-stable softmax along a (possibly strided) axis. One work-item per
// (outer, inner) position; it walks the `axis_len` slice at stride `inner`.
__kernel void softmax(__global float* arena,
                      uint outer, uint axis_len, uint inner,
                      uint off_x, uint off_out) {
    uint gid = get_global_id(0);
    uint total = outer * inner;
    if (gid >= total) return;
    uint o = gid / inner;
    uint i = gid % inner;
    uint base = off_x + o * axis_len * inner + i;
    float m = -INFINITY;
    for (uint j = 0; j < axis_len; j++) m = fmax(m, arena[base + j * inner]);
    float s = 0.0f;
    for (uint j = 0; j < axis_len; j++) s += exp(arena[base + j * inner] - m);
    uint obase = off_out + o * axis_len * inner + i;
    float invs = 1.0f / s;
    for (uint j = 0; j < axis_len; j++)
        arena[obase + j * inner] = exp(arena[base + j * inner] - m) * invs;
}
