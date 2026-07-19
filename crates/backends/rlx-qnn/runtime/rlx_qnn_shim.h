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
/* Must match Qnn_ScaleOffset_t layout (scale then offset). */
typedef struct {
    float scale;
    int32_t offset;
} RlxQnnScaleOffset;

typedef struct {
    const char *name;
    int32_t ttype;
    uint32_t rank;
    const uint32_t *dims; /* `rank` entries */
    float *data;
    uint32_t num_elems; /* element count; byte size depends on dtype */
    int32_t dtype;       /* 0 = f32, 1 = i32, 2 = sfixed8, 3 = int4-as-bw4 (sfixed8+bitwidth=4) */
    float q_scale;       /* per-tensor scale (dtype 2/3 when q_num_scales == 0) */
    int32_t q_offset;    /* per-tensor zero-point/offset (dtype 2/3 when q_num_scales == 0) */
    int32_t q_axis;      /* AXIS_SCALE_OFFSET axis; ignored when q_num_scales == 0 */
    uint32_t q_num_scales; /* 0 = per-tensor SCALE_OFFSET; >0 = AXIS_SCALE_OFFSET */
    const RlxQnnScaleOffset *q_scale_offsets; /* q_num_scales entries; NULL if 0 */
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

/* One-shot build+execute (legacy). Prefer session_* for reuse / binary I/O. */
int rlx_qnn_run_graph(const char *backend_lib,
                      RlxQnnTensor *tensors, uint32_t num_tensors,
                      const RlxQnnNode *nodes, uint32_t num_nodes,
                      uint64_t *err_out);

/* ── Persistent session (finalize once, execute many) ─────────────────────
 *
 * `rlx_qnn_session_create` builds + finalizes the graph and keeps backend /
 * context / graph / tensor ids alive. `rlx_qnn_session_execute` rebinds
 * APP_WRITE / APP_READ client buffers and runs. Context binary save/load is
 * the M3 perf path (`createFromBinary` + `graphRetrieve`).
 */
typedef struct RlxQnnSession RlxQnnSession;

int rlx_qnn_session_create(const char *backend_lib,
                           RlxQnnTensor *tensors, uint32_t num_tensors,
                           const RlxQnnNode *nodes, uint32_t num_nodes,
                           RlxQnnSession **out,
                           uint64_t *err_out);

int rlx_qnn_session_execute(RlxQnnSession *sess,
                            RlxQnnTensor *tensors, uint32_t num_tensors,
                            uint64_t *err_out);

/* Serialize a finalized context. On success `*out_buf` is malloc'd
 * (`written` bytes); free with `rlx_qnn_binary_free`. */
int rlx_qnn_session_save_binary(RlxQnnSession *sess,
                                void **out_buf, uint64_t *written,
                                uint64_t *err_out);

/* Deserialize a context binary (style-2). I/O tensors come from
 * libQnnSystem.so metadata; `execute` matches APP_WRITE/APP_READ by name. */
int rlx_qnn_session_load_binary(const char *backend_lib,
                                const void *binary, uint64_t binary_size,
                                RlxQnnSession **out,
                                uint64_t *err_out);

void rlx_qnn_session_free(RlxQnnSession *sess);
void rlx_qnn_binary_free(void *buf);

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
#define RLX_QNN_E_BINARY 11    /* getBinary / createFromBinary / System meta  */
#define RLX_QNN_E_SYSTEM 12    /* dlopen(libQnnSystem) / System providers     */

#ifdef __cplusplus
}
#endif

#endif /* RLX_QNN_SHIM_H */
