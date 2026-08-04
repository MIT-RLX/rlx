// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * gamma + beta, per row of length n.
// Carries (x, gamma, beta) like Op::RmsNorm. One work-item per row.
__kernel void rmsnorm(__global float* arena,
                      uint rows, uint n,
                      uint off_x, uint off_gamma, uint off_beta, uint off_out,
                      float eps) {
    uint row = get_global_id(0);
    if (row >= rows) return;
    uint base = off_x + row * n;
    float n_inv = 1.0f / (float)n;
    // Two-pass: mean(x²) = mean((x−mean)²) + mean². Matches the CPU oracle.
    float sum = 0.0f;
    for (uint j = 0; j < n; j++) { sum += arena[base + j]; }
    float mean = sum * n_inv;
    float dev = 0.0f;
    for (uint j = 0; j < n; j++) { float d = arena[base + j] - mean; dev += d * d; }
    float inv = rsqrt(dev * n_inv + mean * mean + eps);
    uint obase = off_out + row * n;
    for (uint j = 0; j < n; j++)
        arena[obase + j] = arena[base + j] * inv * arena[off_gamma + j] + arena[off_beta + j];
}
