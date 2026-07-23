// Dual-buffer arena helpers for packed-quant kernels (uint word view).
const uint WEIGHT_TAG = 0x80000000u;

layout(std430, binding = 0) buffer ActArenaU { uint act_u[]; };
layout(std430, binding = 1) buffer WeightArenaU { uint weight_u[]; };

uint arena_load_u(uint tagged_off) {
    if ((tagged_off & WEIGHT_TAG) != 0u) {
        return weight_u[tagged_off & ~WEIGHT_TAG];
    }
    return act_u[tagged_off];
}

void arena_store_u(uint tagged_off, uint val) {
    if ((tagged_off & WEIGHT_TAG) != 0u) {
        weight_u[tagged_off & ~WEIGHT_TAG] = val;
    } else {
        act_u[tagged_off] = val;
    }
}

float arena_load_f(uint tagged_off) {
    return uintBitsToFloat(arena_load_u(tagged_off));
}

void arena_store_f(uint tagged_off, float val) {
    arena_store_u(tagged_off, floatBitsToUint(val));
}

// Signed i8 at byte index `i` relative to a 4-byte-aligned tagged word base
// (same convention as CUDA packed-I8 slots / QuantizeI8).
int arena_load_i8(uint tagged_word_base, uint i) {
    uint w = arena_load_u(tagged_word_base + (i >> 2u));
    uint b = (w >> ((i & 3u) * 8u)) & 0xFFu;
    return int(b) - 256 * int((b >> 7u) & 1u);
}

// Packed i8 store via non-atomic RMW. naga's GLSL front-end has no
// `atomicCompSwap`; concurrent writers to the same uint word can race.
// Quantize uses a word-owner launch (see quantize_i8.comp) to avoid that.
void arena_store_i8(uint tagged_word_base, uint i, int val) {
    uint word_tagged = tagged_word_base + (i >> 2u);
    uint shift = (i & 3u) * 8u;
    uint mask = 0xFFu << shift;
    uint bits = (uint(val) & 0xFFu) << shift;
    uint old = arena_load_u(word_tagged);
    arena_store_u(word_tagged, (old & ~mask) | bits);
}

// Rust `f32::round` (half away from zero). Use float(bool) — naga rejects bool−bool.
float q_round_half_away(float x) {
    float sgn = float(x > 0.0) - float(x < 0.0);
    return sgn * floor(abs(x) + 0.5);
}
