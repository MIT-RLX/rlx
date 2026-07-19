// Q1_0 (prism-ml Bonsai-27B) body for mlx::fast::metal_kernel.
//
// Helper `q1_read_f16` is injected via the host `header` arg — MLX pastes
// the body into a context where nested function definitions are illegal.
//
// Layout: f16 d | 16 sign-bit bytes (18 bytes / 128 elems).
// Bit LSB-first within each byte; 1 -> +d, 0 -> -d.

const uint gid = thread_position_in_grid.x;
uint off = gid * 18u;
float d = q1_read_f16(w, off);
float neg_d = -d;
const device uchar* qs = w + off + 2u;
device float* dst = out + gid * 128u;
for (uint j = 0u; j < 128u; ++j) {
    uint bit = (uint(qs[j >> 3u]) >> (j & 7u)) & 1u;
    dst[j] = (bit != 0u) ? d : neg_d;
}
