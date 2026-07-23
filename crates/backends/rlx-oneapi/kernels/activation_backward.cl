// Element-wise activation / ReLU backward.
// op ids match CUDA / wgpu (NOT forward unary.cl ids):
//   0=relu 1=sigmoid 2=tanh 3=exp 4=log 5=sqrt 6=rsqrt
//   7=neg  8=abs     9=gelu 10=silu 11=gelu_approx
//   12=round 13=sin 14=cos 15=tan 16=atan 17=recip
float erf_approx(float x) {
    float s = (x >= 0.0f) ? 1.0f : -1.0f;
    float xa = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * xa);
    float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f
                + t * (-1.453152027f + t * 1.061405429f))));
    return s * (1.0f - poly * exp(-xa * xa));
}

__kernel void activation_backward(__global float* arena,
                                  uint n,
                                  uint x_off, uint dy_off, uint dx_off,
                                  uint op) {
    uint i = get_global_id(0);
    if (i >= n) return;
    float x = arena[x_off + i];
    float dy = arena[dy_off + i];
    float dx = dy;
    switch (op) {
        case 0u: // relu
            dx = (x > 0.0f) ? dy : 0.0f;
            break;
        case 1u: { // sigmoid
            float xc = clamp(x, -88.0f, 88.0f);
            float s = 1.0f / (1.0f + exp(-xc));
            dx = s * (1.0f - s) * dy;
        } break;
        case 2u: { // tanh
            float t = tanh(clamp(x, -15.0f, 15.0f));
            dx = (1.0f - t * t) * dy;
        } break;
        case 3u: // exp
            dx = exp(x) * dy;
            break;
        case 4u: // log
            dx = dy / x;
            break;
        case 5u: { // sqrt
            float s = sqrt(x);
            dx = (s > 0.0f) ? (0.5f * dy / s) : 0.0f;
        } break;
        case 6u: { // rsqrt
            float s = sqrt(x);
            dx = (s > 0.0f) ? (-0.5f * dy / (x * s)) : 0.0f;
        } break;
        case 7u: // neg
            dx = -dy;
            break;
        case 8u: // abs
            dx = (x > 0.0f) ? dy : ((x < 0.0f) ? -dy : 0.0f);
            break;
        case 9u: { // gelu
            const float INV_SQRT2 = 0.7071067811865475f;
            const float INV_SQRT_2PI = 0.3989422804014327f;
            float phi = 0.5f * (1.0f + erf_approx(x * INV_SQRT2));
            float pdf = INV_SQRT_2PI * exp(-0.5f * x * x);
            dx = (phi + x * pdf) * dy;
        } break;
        case 10u: { // silu
            float xc = clamp(x, -88.0f, 88.0f);
            float s = 1.0f / (1.0f + exp(-xc));
            dx = s * (1.0f + x * (1.0f - s)) * dy;
        } break;
        case 11u: { // gelu_approx
            const float C = 0.7978845608028654f;
            const float A = 0.044715f;
            float inner = clamp(C * (x + A * x * x * x), -15.0f, 15.0f);
            float t = tanh(inner);
            float dinner = C * (1.0f + 3.0f * A * x * x);
            float d = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * dinner;
            dx = d * dy;
        } break;
        case 12u: // round STE
            dx = dy;
            break;
        case 13u: // sin
            dx = cos(x) * dy;
            break;
        case 14u: // cos
            dx = -sin(x) * dy;
            break;
        case 15u: { // tan
            float t = tan(x);
            dx = (1.0f + t * t) * dy;
        } break;
        case 16u: // atan
            dx = dy / (1.0f + x * x);
            break;
        case 17u: // recip: -1/x²
            dx = -dy / (x * x);
            break;
        default:
            dx = dy;
            break;
    }
    arena[dx_off + i] = dx;
}
