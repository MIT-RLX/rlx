// Elementwise activation over the f32-uniform arena. `act` matches the
// `act_id` mapping in src/backend.rs (Activation enum order).
//
// Cast op ids (see `classify_cast` in src/backend.rs, kept in sync with
// unary.cu / unary.comp). In the f32-uniform arena the dst dtype's value is
// written back as an f32 lane:
//   100=f32->i8  101=f32->i16 102=f32->i32 103=f32->i64
//   104=f32->u8  105=f32->u32 106=(x!=0)->bool
// float->int truncates toward zero + saturates (Rust `as` / rlx-cpu); NaN->0.
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
        case 17u: r = 1.0f / x; break;                                     // Recip / vvrecf
        case 18u: r = floor(x); break;                                     // Floor
        case 19u: r = ceil(x); break;                                      // Ceil
        case 20u: r = sign(x); break;                                      // Sign
        case 21u: r = log1p(exp(x)); break;                                // Softplus
        case 22u: r = x > 0.0f ? x : (exp(x) - 1.0f); break;               // Elu (alpha=1)
        case 23u: r = erf(x); break;                                       // Erf
        case 24u: r = x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f; break;       // HardSwish
        case 25u: r = clamp(x / 6.0f + 0.5f, 0.0f, 1.0f); break;           // HardSigmoid
        case 26u: r = x * tanh(fmax(x, 0.0f) + log1p(exp(-fabs(x)))); break; // Mish
        case 27u: r = x / (1.0f + fabs(x)); break;                         // Softsign
        case 28u: r = fmin(x, 0.0f) - log1p(exp(-fabs(x))); break;         // LogSigmoid
        // f32 -> int: truncate toward zero, saturate to dst range, NaN -> 0.
        case 100u: r = isnan(x) ? 0.0f : clamp(trunc(x), -128.0f, 127.0f); break;
        case 101u: r = isnan(x) ? 0.0f : clamp(trunc(x), -32768.0f, 32767.0f); break;
        case 102u: r = isnan(x) ? 0.0f : clamp(trunc(x), -2147483648.0f, 2147483647.0f); break;
        case 103u: r = isnan(x) ? 0.0f : clamp(trunc(x), -9223372036854775808.0f, 9223372036854775807.0f); break;
        case 104u: r = isnan(x) ? 0.0f : clamp(trunc(x), 0.0f, 255.0f); break;
        case 105u: r = isnan(x) ? 0.0f : clamp(trunc(x), 0.0f, 4294967295.0f); break;
        // -> Bool: x != 0 ? 1 : 0 (NaN is non-zero -> 1, matching Rust).
        case 106u: r = (x != 0.0f) ? 1.0f : 0.0f; break;
        default: r = x; break;
    }
    arena[off_out + gid] = r;
}
