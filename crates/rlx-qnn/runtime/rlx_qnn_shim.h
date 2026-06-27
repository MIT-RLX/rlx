/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * GPL-3.0-only. See the crate root for the full license text.
 *
 * Minimal stable C ABI for the rlx-qnn FFI runtime backend. The shim compiles
 * against the real QNN AI Engine Direct headers ($QNN_SDK_ROOT/include/QNN) and
 * dlopen's a backend library (libQnnCpu.so / libQnnHtp.so) at run time, so the
 * vtable layout is taken from the SDK rather than hand-transcribed into Rust.
 * Mirrors the rlx-mlx-sys "vendored C shim + stable C ABI" precedent.
 */
#ifndef RLX_QNN_SHIM_H
#define RLX_QNN_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Build and execute a single f32 matmul on `backend_lib`:
 *
 *     out[M,N] = in0[M,K] * in1[K,N]      (row-major, no transpose)
 *
 * Returns 0 on success. On failure returns a negative step code identifying
 * where it broke (see RLX_QNN_E_* below), so the Rust side can surface a
 * precise error. `err_out` (optional, may be NULL) receives the QNN
 * Qnn_ErrorHandle_t from the failing call.
 */
int rlx_qnn_matmul_f32(const char *backend_lib,
                       uint32_t M, uint32_t K, uint32_t N,
                       const float *in0, const float *in1, float *out,
                       uint64_t *err_out);

/* ── General multi-op graph execution ─────────────────────────────────────
 *
 * Build and execute an arbitrary supported subgraph. One `RlxQnnTensor` per
 * graph node (index = node index); `RlxQnnNode`s reference tensors by index.
 * `ttype` is a `QNN_TENSOR_TYPE_*` value: 0 APP_WRITE (runtime input), 1
 * APP_READ (output), 3 NATIVE (intermediate), 4 STATIC (baked weight).
 *
 * `data` is the host buffer, by tensor role:
 *   STATIC    — weight data, read at graph-build time
 *   APP_WRITE — input activations, read at execute
 *   APP_READ  — output buffer, written at execute
 *   NATIVE    — NULL
 */
typedef struct {
    const char *name;
    int32_t ttype;
    uint32_t rank;
    const uint32_t *dims; /* `rank` entries */
    float *data;
    uint32_t num_elems; /* element count; byte size = num_elems * 4 */
    int32_t dtype;       /* 0 = float32, 1 = int32, 2 = sfixed8 (quantized int8) */
    float q_scale;       /* quantization scale (dtype 2 only) */
    int32_t q_offset;    /* quantization zero-point/offset (dtype 2 only) */
} RlxQnnTensor;

typedef struct {
    const char *name;
    const char *op_type;    /* QNN op type, e.g. "MatMul", "ElementWiseAdd" */
    const uint32_t *inputs; /* tensor indices */
    uint32_t num_inputs;
    uint32_t output; /* tensor index (single output) */
    int32_t axis;    /* Softmax axis (>= 0); -1 = no axis param */
    const uint32_t *perm; /* uint32 tensor-param data: Transpose `perm` or
                           * norm `axes` (NULL if unused) */
    uint32_t perm_len;    /* length of `perm` (0 if unused) */
    float eps;            /* LayerNorm epsilon (ignored unless the op is a norm) */
} RlxQnnNode;

int rlx_qnn_run_graph(const char *backend_lib,
                      RlxQnnTensor *tensors, uint32_t num_tensors,
                      const RlxQnnNode *nodes, uint32_t num_nodes,
                      uint64_t *err_out);

/* Step codes (negated on return). */
#define RLX_QNN_OK 0
#define RLX_QNN_E_DLOPEN 1     /* dlopen(backend_lib) failed                  */
#define RLX_QNN_E_GETPROC 2    /* QnnInterface_getProviders symbol missing    */
#define RLX_QNN_E_PROVIDERS 3  /* getProviders failed / returned none         */
#define RLX_QNN_E_BACKEND 4    /* backendCreate failed                        */
#define RLX_QNN_E_CONTEXT 5    /* contextCreate failed                        */
#define RLX_QNN_E_GRAPH 6      /* graphCreate failed                          */
#define RLX_QNN_E_TENSOR 7     /* tensorCreateGraphTensor failed              */
#define RLX_QNN_E_ADDNODE 8    /* graphAddNode failed                         */
#define RLX_QNN_E_FINALIZE 9   /* graphFinalize failed                        */
#define RLX_QNN_E_EXECUTE 10   /* graphExecute failed                         */

#ifdef __cplusplus
}
#endif

#endif /* RLX_QNN_SHIM_H */
