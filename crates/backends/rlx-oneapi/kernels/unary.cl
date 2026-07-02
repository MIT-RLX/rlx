// Elementwise activation over the f32-uniform arena. `act` matches the
// `act_id` mapping in src/backend.rs (Activation enum order).
__kernel void unary(__global float* arena,
                    uint n, uint off_x, uint off_out, uint act) {
    uint gid = get_global_id(0);
    if (gid >= n) return;
    float x = arena[off_x + gid];
    float r;
    switch (act) {
        case 0u: r = 0.5f * x * (1.0f + erf(x * 0.70710678118f)); break;   // Gelu (exact)
        case 1u: {                                                         // GeluApprox (tanh)
            float c = 0.7978845608f;
            float t = tanh(c * (x + 0.044715f * x * x * x));
            r = 0.5f * x * (1.0f + t);
        } break;
        case 2u: r = x / (1.0f + exp(-x)); break;                          // Silu
        case 3u: r = fmax(x, 0.0f); break;                                 // Relu
        case 4u: r = 1.0f / (1.0f + exp(-x)); break;                       // Sigmoid
        case 5u: r = tanh(x); break;                                       // Tanh
        case 6u: r = exp(x); break;                                        // Exp
        case 7u: r = log(x); break;                                        // Log
        case 8u: r = sqrt(x); break;                                       // Sqrt
        case 9u: r = rsqrt(x); break;                                      // Rsqrt
        case 10u: r = -x; break;                                           // Neg
        case 11u: r = fabs(x); break;                                      // Abs
        case 12u: r = sin(x); break;                                       // Sin
        case 13u: r = cos(x); break;                                       // Cos
        case 14u: r = tan(x); break;                                       // Tan
        case 15u: r = atan(x); break;                                      // Atan
        case 16u: r = round(x); break;                                     // Round
        default: r = x; break;
    }
    arena[off_out + gid] = r;
}
