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
