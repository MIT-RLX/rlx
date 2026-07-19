// Dual-buffer arena helpers (included by build.rs). Push-constant offsets carry
// WEIGHT_TAG (bit 31) when the slot lives in binding 1 (weights); otherwise
// binding 0 (activations).
const uint WEIGHT_TAG = 0x80000000u;

layout(std430, binding = 0) buffer ActArena { float act_data[]; };
layout(std430, binding = 1) buffer WeightArena { float weight_data[]; };

float arena_load(uint tagged_off) {
    if ((tagged_off & WEIGHT_TAG) != 0u) {
        return weight_data[tagged_off & ~WEIGHT_TAG];
    }
    return act_data[tagged_off];
}

void arena_store(uint tagged_off, float val) {
    if ((tagged_off & WEIGHT_TAG) != 0u) {
        weight_data[tagged_off & ~WEIGHT_TAG] = val;
    } else {
        act_data[tagged_off] = val;
    }
}
