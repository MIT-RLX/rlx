// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// C ABI exposed by csrc/coreml_shim.m. Consumed from Rust via the FFI
// declarations in src/ffi.rs. Tensor I/O is contiguous row-major f32 or f16.
#ifndef RLX_COREML_SHIM_H
#define RLX_COREML_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RlxCoremlModel RlxCoremlModel;

// compute_units: 0 = all, 1 = cpuOnly, 2 = cpuAndGpu, 3 = cpuAndNeuralEngine.
// Compiles the .mlpackage at `mlpackage_path` to .mlmodelc, then loads it.
// Returns NULL on failure and writes a message into `err` (NUL-terminated).
// `compiled_cache_path`: if non-empty and already present, the compiled
// .mlmodelc there is reused (the expensive compile is skipped); otherwise
// the freshly compiled model is copied to it. Pass NULL/"" to disable.
RlxCoremlModel *rlx_coreml_load(const char *mlpackage_path, int compute_units,
                                const char *compiled_cache_path, char *err,
                                int err_len);

// Runs one prediction. Inputs/outputs are matched by name.
// `in_dtypes[i]`: 0 = float32, 1 = float16 (NULL → all float32).
// Output buffers are always float32; f16 model outputs are converted.
// `out_data[i]` must hold at least `out_len[i]` floats.
// Returns 0 on success, non-zero on error (message written to `err`).
int rlx_coreml_predict(RlxCoremlModel *model, int n_inputs,
                       const char *const *in_names, const void *const *in_data,
                       const int64_t *const *in_shapes, const int *in_ranks,
                       const int *in_dtypes, int n_outputs,
                       const char *const *out_names, float *const *out_data,
                       const int *out_len, char *err, int err_len);

void rlx_coreml_free(RlxCoremlModel *model);

// Reports which devices the loaded model's ops were scheduled onto. Fills
// `counts[0..3]` with op counts for {CPU, GPU, ANE, unknown}. Returns 0 on
// success. Uses MLComputePlan when available (macOS 14.4+), else returns -1.
int rlx_coreml_compute_plan(RlxCoremlModel *model, int *counts, char *err,
                            int err_len);

// --- introspection (no model needed) ---
int rlx_coreml_ane_available(void);
void rlx_coreml_chip_brand(char *buf, int len);
void rlx_coreml_chip_model(char *buf, int len);
void rlx_coreml_os_version(char *buf, int len);

#ifdef __cplusplus
}
#endif

#endif // RLX_COREML_SHIM_H
