// Fused Q1_0 GEMV (prism-ml Bonsai-27B decode) for mlx::fast::metal_kernel.
//
// out[j] = sum_k x[k] * dequant(w[j, k]) — one thread per output column j
// (grid = n). Reads the packed 1-bit weight row directly (18 bytes / 128
// elems: f16 scale `d` + 16 sign bytes, LSB-first, bit1 -> +d bit0 -> -d),
// so the ~28x f32 blow-up never materializes (Bonsai-27B: 3.8 GiB packed).
//
// Inputs: `w` (u8 packed, row-major [n, k] blocks), `x` (f32, reshaped to
// [k] by the host so `x_shape[0] == k`). Helper `q1_read_f16` injected via
// the host `header` arg. Mentioning `x_shape` makes MLX inject it.

const uint j = thread_position_in_grid.x;
uint kdim = uint(x_shape[0]);
uint nblocks = kdim / 128u;
uint wbase = j * nblocks * 18u;
float acc = 0.0f;
for (uint b = 0u; b < nblocks; ++b) {
    uint off = wbase + b * 18u;
    float d = q1_read_f16(w, off);
    float nd = -d;
    const device uchar* qs = w + off + 2u;
    uint kb = b * 128u;
    for (uint byte = 0u; byte < 16u; ++byte) {
        uint bits = uint(qs[byte]);
        for (uint bit = 0u; bit < 8u; ++bit) {
            float wv = (((bits >> bit) & 1u) != 0u) ? d : nd;
            acc += x[kb + byte * 8u + bit] * wv;
        }
    }
}
out[j] = acc;
