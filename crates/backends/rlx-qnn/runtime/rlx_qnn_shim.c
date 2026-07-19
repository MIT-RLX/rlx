/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * GPL-3.0-only. See the crate root for the full license text.
 *
 * QNN AI Engine Direct FFI shim — dynamic (style-1) graph build + execute, with
 * persistent sessions and context-binary save/load. Compiled against the real
 * SDK headers; backend libraries are resolved at run time via dlopen.
 */

#define _POSIX_C_SOURCE 200809L

#include "rlx_qnn_shim.h"

#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "QnnInterface.h"
#include "QnnLog.h"
#include "QnnOpDef.h"
#include "QnnTypes.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"

typedef Qnn_ErrorHandle_t (*GetProvidersFn)(const QnnInterface_t ***, uint32_t *);
typedef Qnn_ErrorHandle_t (*SystemGetProvidersFn)(const QnnSystemInterface_t ***,
                                                    uint32_t *);

struct RlxQnnSession {
    void *lib;
    void *sys_lib;
    const QNN_INTERFACE_VER_TYPE *qi;
    Qnn_BackendHandle_t backend;
    Qnn_ContextHandle_t context;
    Qnn_GraphHandle_t graph;
    Qnn_LogHandle_t logger;
    Qnn_Tensor_t *qt;
    uint32_t num_tensors;
    uint32_t *in_idx;
    uint32_t n_in;
    uint32_t *out_idx;
    uint32_t n_out;
    int from_binary;
    char **owned_names;
    uint32_t **owned_dims;
    uint32_t n_owned;
    char *graph_name;
};

/* Forward QNN backend diagnostics (e.g. why an op config is rejected) to
 * stderr — invaluable when graphAddNode/graphFinalize fail. */
static void rlx_qnn_log_cb(const char *fmt, QnnLog_Level_t level,
                           uint64_t timestamp, va_list argp) {
    (void)level;
    (void)timestamp;
    fprintf(stderr, "[QNN] ");
    vfprintf(stderr, fmt, argp);
}

/* Build a v1 f32, rank-2, raw-buffer graph tensor with undefined quantization
 * — the same shape the qnn-converter emits, set field-by-field (plain C). */
static Qnn_Tensor_t make_tensor(const char *name, Qnn_TensorType_t type,
                                uint32_t *dims, uint32_t rank) {
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.id = 0;
    t.v1.name = name;
    t.v1.type = type;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType = QNN_DATATYPE_FLOAT_32;
    t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    t.v1.quantizeParams.scaleOffsetEncoding.scale = 0.0f;
    t.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
    t.v1.rank = rank;
    t.v1.dimensions = dims;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf.data = NULL;
    t.v1.clientBuf.dataSize = 0;
    return t;
}

/* Total clientBuf byte size. dtype 3 = int4-as-bw4 (1 byte/elem, lower nibble). */
static uint32_t rlx_tensor_data_size(const RlxQnnTensor *t) {
    if (t->dtype == 2 || t->dtype == 3) {
        return t->num_elems;
    }
    return t->num_elems * 4u;
}

static RlxQnnTensor *find_rlx_tensor(RlxQnnTensor *tensors, uint32_t num_tensors,
                                     const char *name, int32_t ttype) {
    for (uint32_t i = 0; i < num_tensors; ++i) {
        if (tensors[i].name && name && strcmp(tensors[i].name, name) == 0 &&
            tensors[i].ttype == ttype) {
            return &tensors[i];
        }
    }
    return NULL;
}

static int deep_copy_qnn_tensor(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src,
                                char **owned_name, uint32_t **owned_dims) {
    Qnn_Tensor_t init = QNN_TENSOR_INIT;
    *dst = init;
    *owned_name = NULL;
    *owned_dims = NULL;
    dst->version = src->version;

    if (src->version == QNN_TENSOR_VERSION_1) {
        if (src->v1.name) {
            *owned_name = strdup(src->v1.name);
            if (!*owned_name) return -1;
            dst->v1.name = *owned_name;
        }
        dst->v1.id = src->v1.id;
        dst->v1.type = src->v1.type;
        dst->v1.dataFormat = src->v1.dataFormat;
        dst->v1.dataType = src->v1.dataType;
        dst->v1.quantizeParams = src->v1.quantizeParams;
        dst->v1.rank = src->v1.rank;
        if (src->v1.rank > 0 && src->v1.dimensions) {
            *owned_dims = (uint32_t *)malloc(src->v1.rank * sizeof(uint32_t));
            if (!*owned_dims) return -1;
            memcpy(*owned_dims, src->v1.dimensions,
                   src->v1.rank * sizeof(uint32_t));
            dst->v1.dimensions = *owned_dims;
        }
        dst->v1.memType = src->v1.memType;
        dst->v1.clientBuf.data = NULL;
        dst->v1.clientBuf.dataSize = 0;
        return 0;
    }

    if (src->version == QNN_TENSOR_VERSION_2) {
        if (src->v2.name) {
            *owned_name = strdup(src->v2.name);
            if (!*owned_name) return -1;
            dst->v2.name = *owned_name;
        }
        dst->v2.id = src->v2.id;
        dst->v2.type = src->v2.type;
        dst->v2.dataFormat = src->v2.dataFormat;
        dst->v2.dataType = src->v2.dataType;
        dst->v2.quantizeParams = src->v2.quantizeParams;
        dst->v2.rank = src->v2.rank;
        if (src->v2.rank > 0 && src->v2.dimensions) {
            *owned_dims = (uint32_t *)malloc(src->v2.rank * sizeof(uint32_t));
            if (!*owned_dims) return -1;
            memcpy(*owned_dims, src->v2.dimensions,
                   src->v2.rank * sizeof(uint32_t));
            dst->v2.dimensions = *owned_dims;
        }
        dst->v2.memType = src->v2.memType;
        dst->v2.clientBuf.data = NULL;
        dst->v2.clientBuf.dataSize = 0;
        return 0;
    }

    return -1;
}

static const char *tensor_name_v1(const Qnn_Tensor_t *t) {
    if (t->version == QNN_TENSOR_VERSION_1) return t->v1.name;
    if (t->version == QNN_TENSOR_VERSION_2) return t->v2.name;
    return NULL;
}

static int extract_graph_io_v1(const QnnSystemContext_GraphInfoV1_t *gi,
                                 const char **graph_name,
                                 const Qnn_Tensor_t **inputs, uint32_t *n_in,
                                 const Qnn_Tensor_t **outputs, uint32_t *n_out) {
    if (!gi) return -1;
    *graph_name = gi->graphName;
    *inputs = gi->graphInputs;
    *n_in = gi->numGraphInputs;
    *outputs = gi->graphOutputs;
    *n_out = gi->numGraphOutputs;
    return 0;
}

static int extract_graph_io(const QnnSystemContext_GraphInfo_t *gi,
                            const char **graph_name,
                            const Qnn_Tensor_t **inputs, uint32_t *n_in,
                            const Qnn_Tensor_t **outputs, uint32_t *n_out) {
    if (!gi) return -1;
    if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
        return extract_graph_io_v1(&gi->graphInfoV1, graph_name, inputs, n_in,
                                   outputs, n_out);
    }
    if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        const QnnSystemContext_GraphInfoV2_t *g2 = &gi->graphInfoV2;
        *graph_name = g2->graphName;
        *inputs = g2->graphInputs;
        *n_in = g2->numGraphInputs;
        *outputs = g2->graphOutputs;
        *n_out = g2->numGraphOutputs;
        return 0;
    }
    if (gi->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        const QnnSystemContext_GraphInfoV3_t *g3 = &gi->graphInfoV3;
        *graph_name = g3->graphName;
        *inputs = g3->graphInputs;
        *n_in = g3->numGraphInputs;
        *outputs = g3->graphOutputs;
        *n_out = g3->numGraphOutputs;
        return 0;
    }
    return -1;
}

static int resolve_system_lib_path(const char *backend_lib, char *out, size_t out_sz) {
    const char *env = getenv("RLX_QNN_SYSTEM_LIB");
    if (env && env[0]) {
        snprintf(out, out_sz, "%s", env);
        return 0;
    }
    const char *slash = strrchr(backend_lib, '/');
    if (slash) {
        size_t dir_len = (size_t)(slash - backend_lib + 1);
        if (dir_len + strlen("libQnnSystem.so") < out_sz) {
            memcpy(out, backend_lib, dir_len);
            strcpy(out + dir_len, "libQnnSystem.so");
            return 0;
        }
    }
    snprintf(out, out_sz, "libQnnSystem.so");
    return 0;
}

void rlx_qnn_session_free(RlxQnnSession *sess) {
    if (!sess) return;

    if (sess->n_owned > 0) {
        for (uint32_t i = 0; i < sess->n_owned; ++i) {
            free(sess->owned_names[i]);
            free(sess->owned_dims[i]);
        }
    }
    free(sess->owned_names);
    free(sess->owned_dims);
    free(sess->graph_name);
    free(sess->qt);
    free(sess->in_idx);
    free(sess->out_idx);

    if (sess->context && sess->qi && sess->qi->contextFree) {
        sess->qi->contextFree(sess->context, NULL);
    }
    if (sess->backend && sess->qi && sess->qi->backendFree) {
        sess->qi->backendFree(sess->backend);
    }
    if (sess->sys_lib) dlclose(sess->sys_lib);
    if (sess->lib) dlclose(sess->lib);
    free(sess);
}

void rlx_qnn_binary_free(void *buf) { free(buf); }

int rlx_qnn_matmul_f32(const char *backend_lib,
                       uint32_t M, uint32_t K, uint32_t N,
                       const float *in0, const float *in1, float *out,
                       uint64_t *err_out) {
    if (err_out) *err_out = 0;
    int rc = RLX_QNN_OK;

    void *lib = dlopen(backend_lib, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "rlx_qnn: dlopen(%s) failed: %s\n", backend_lib, dlerror());
        return -RLX_QNN_E_DLOPEN;
    }

    GetProvidersFn get_providers =
        (GetProvidersFn)dlsym(lib, "QnnInterface_getProviders");
    if (!get_providers) { dlclose(lib); return -RLX_QNN_E_GETPROC; }

    const QnnInterface_t **providers = NULL;
    uint32_t num_providers = 0;
    if (get_providers(&providers, &num_providers) != QNN_SUCCESS ||
        num_providers == 0 || providers == NULL) {
        dlclose(lib);
        return -RLX_QNN_E_PROVIDERS;
    }
    /* The versioned core vtable (e.g. v2_xx) — accessor macro from the header. */
    const QNN_INTERFACE_VER_TYPE *qi = &providers[0]->QNN_INTERFACE_VER_NAME;

    Qnn_BackendHandle_t backend = NULL;
    Qnn_ContextHandle_t context = NULL;
    Qnn_GraphHandle_t graph = NULL;
    Qnn_ErrorHandle_t e;

#define FAIL(step)                       \
    do {                                 \
        if (err_out) *err_out = e;       \
        rc = -(step);                    \
        goto cleanup;                    \
    } while (0)

    e = qi->backendCreate(NULL, NULL, &backend);
    if (e != QNN_SUCCESS) FAIL(RLX_QNN_E_BACKEND);

    e = qi->contextCreate(backend, NULL, NULL, &context);
    if (e != QNN_SUCCESS) FAIL(RLX_QNN_E_CONTEXT);

    e = qi->graphCreate(context, "matmul_graph", NULL, &graph);
    if (e != QNN_SUCCESS) FAIL(RLX_QNN_E_GRAPH);

    /* Graph tensors — createGraphTensor assigns each a backend id. */
    uint32_t dims_in0[2] = {M, K};
    uint32_t dims_in1[2] = {K, N};
    uint32_t dims_out[2] = {M, N};
    Qnn_Tensor_t t_in0 = make_tensor("in0", QNN_TENSOR_TYPE_APP_WRITE, dims_in0, 2);
    Qnn_Tensor_t t_in1 = make_tensor("in1", QNN_TENSOR_TYPE_APP_WRITE, dims_in1, 2);
    Qnn_Tensor_t t_out = make_tensor("out", QNN_TENSOR_TYPE_APP_READ, dims_out, 2);
    if ((e = qi->tensorCreateGraphTensor(graph, &t_in0)) != QNN_SUCCESS) FAIL(RLX_QNN_E_TENSOR);
    if ((e = qi->tensorCreateGraphTensor(graph, &t_in1)) != QNN_SUCCESS) FAIL(RLX_QNN_E_TENSOR);
    if ((e = qi->tensorCreateGraphTensor(graph, &t_out)) != QNN_SUCCESS) FAIL(RLX_QNN_E_TENSOR);

    /* MatMul node: out = in0 * in1, no transpose. */
    Qnn_Param_t params[2];
    params[0].paramType = QNN_PARAMTYPE_SCALAR;
    params[0].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;
    params[0].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    params[0].scalarParam.bool8Value = 0;
    params[1].paramType = QNN_PARAMTYPE_SCALAR;
    params[1].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
    params[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
    params[1].scalarParam.bool8Value = 0;

    Qnn_Tensor_t node_inputs[2] = {t_in0, t_in1};
    Qnn_Tensor_t node_outputs[1] = {t_out};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = "matmul_0";
    op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
    op.v1.typeName = QNN_OP_MAT_MUL;
    op.v1.numOfParams = 2;
    op.v1.params = params;
    op.v1.numOfInputs = 2;
    op.v1.inputTensors = node_inputs;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = node_outputs;
    if ((e = qi->graphAddNode(graph, op)) != QNN_SUCCESS) FAIL(RLX_QNN_E_ADDNODE);

    if ((e = qi->graphFinalize(graph, NULL, NULL)) != QNN_SUCCESS) FAIL(RLX_QNN_E_FINALIZE);

    /* Bind host buffers and execute. Tensors carry the ids assigned above, so
     * the runtime matches them to the graph's APP_WRITE / APP_READ tensors. */
    t_in0.v1.clientBuf.data = (void *)in0;
    t_in0.v1.clientBuf.dataSize = (uint32_t)(M * K * sizeof(float));
    t_in1.v1.clientBuf.data = (void *)in1;
    t_in1.v1.clientBuf.dataSize = (uint32_t)(K * N * sizeof(float));
    t_out.v1.clientBuf.data = (void *)out;
    t_out.v1.clientBuf.dataSize = (uint32_t)(M * N * sizeof(float));

    Qnn_Tensor_t exec_in[2] = {t_in0, t_in1};
    Qnn_Tensor_t exec_out[1] = {t_out};
    e = qi->graphExecute(graph, exec_in, 2, exec_out, 1, NULL, NULL);
    if (e != QNN_SUCCESS) FAIL(RLX_QNN_E_EXECUTE);

#undef FAIL
cleanup:
    if (context && qi->contextFree) qi->contextFree(context, NULL);
    if (backend && qi->backendFree) qi->backendFree(backend);
    dlclose(lib);
    return rc;
}

int rlx_qnn_session_create(const char *backend_lib,
                           RlxQnnTensor *tensors, uint32_t num_tensors,
                           const RlxQnnNode *nodes, uint32_t num_nodes,
                           RlxQnnSession **out,
                           uint64_t *err_out) {
    if (err_out) *err_out = 0;
    if (out) *out = NULL;
    int rc = RLX_QNN_OK;

    RlxQnnSession *sess = (RlxQnnSession *)calloc(1, sizeof(RlxQnnSession));
    if (!sess) return -RLX_QNN_E_TENSOR;

    sess->lib = dlopen(backend_lib, RTLD_NOW | RTLD_LOCAL);
    if (!sess->lib) {
        fprintf(stderr, "rlx_qnn: dlopen(%s) failed: %s\n", backend_lib, dlerror());
        rlx_qnn_session_free(sess);
        return -RLX_QNN_E_DLOPEN;
    }

    GetProvidersFn get_providers =
        (GetProvidersFn)dlsym(sess->lib, "QnnInterface_getProviders");
    if (!get_providers) {
        rlx_qnn_session_free(sess);
        return -RLX_QNN_E_GETPROC;
    }

    const QnnInterface_t **providers = NULL;
    uint32_t num_providers = 0;
    if (get_providers(&providers, &num_providers) != QNN_SUCCESS ||
        num_providers == 0 || providers == NULL) {
        rlx_qnn_session_free(sess);
        return -RLX_QNN_E_PROVIDERS;
    }
    sess->qi = &providers[0]->QNN_INTERFACE_VER_NAME;

    sess->qt = (Qnn_Tensor_t *)calloc(num_tensors, sizeof(Qnn_Tensor_t));
    if (!sess->qt) {
        rlx_qnn_session_free(sess);
        return -RLX_QNN_E_TENSOR;
    }
    sess->num_tensors = num_tensors;

    Qnn_ErrorHandle_t e = QNN_SUCCESS;

#define FAILG(step)                \
    do {                           \
        if (err_out) *err_out = e; \
        rc = -(step);              \
        goto cleanup_g;            \
    } while (0)

    if (sess->qi->logCreate) {
        sess->qi->logCreate(rlx_qnn_log_cb, QNN_LOG_LEVEL_WARN, &sess->logger);
    }
    e = sess->qi->backendCreate(sess->logger, NULL, &sess->backend);
    if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_BACKEND);
    e = sess->qi->contextCreate(sess->backend, NULL, NULL, &sess->context);
    if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_CONTEXT);
    e = sess->qi->graphCreate(sess->context, "rlx_graph", NULL, &sess->graph);
    if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_GRAPH);

    /* Create every graph tensor; STATIC tensors carry their data now. */
    for (uint32_t i = 0; i < num_tensors; ++i) {
        RlxQnnTensor *t = &tensors[i];
        sess->qt[i] = make_tensor(t->name, (Qnn_TensorType_t)t->ttype,
                                  (uint32_t *)t->dims, t->rank);
        if (t->dtype == 1) {
            sess->qt[i].v1.dataType = QNN_DATATYPE_INT_32;
        } else if (t->dtype == 2) {
            sess->qt[i].v1.dataType = QNN_DATATYPE_SFIXED_POINT_8;
            sess->qt[i].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            if (t->q_num_scales > 0 && t->q_scale_offsets != NULL) {
                /* Per-channel / per-axis scales (e.g. Linear weight [K,N], axis=1). */
                sess->qt[i].v1.quantizeParams.quantizationEncoding =
                    QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
                sess->qt[i].v1.quantizeParams.axisScaleOffsetEncoding.axis = t->q_axis;
                sess->qt[i].v1.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets =
                    t->q_num_scales;
                sess->qt[i].v1.quantizeParams.axisScaleOffsetEncoding.scaleOffset =
                    (Qnn_ScaleOffset_t *)t->q_scale_offsets;
            } else {
                sess->qt[i].v1.quantizeParams.quantizationEncoding =
                    QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
                sess->qt[i].v1.quantizeParams.scaleOffsetEncoding.scale = t->q_scale;
                sess->qt[i].v1.quantizeParams.scaleOffsetEncoding.offset = t->q_offset;
            }
        } else if (t->dtype == 3) {
            /* Int4 precision in an 8-bit container (CPU rejects packed
             * SFIXED_POINT_4). Values occupy the low 4 bits; upper bits ignored. */
            sess->qt[i].v1.dataType = QNN_DATATYPE_SFIXED_POINT_8;
            sess->qt[i].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            sess->qt[i].v1.quantizeParams.quantizationEncoding =
                QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET;
            sess->qt[i].v1.quantizeParams.bwScaleOffsetEncoding.bitwidth = 4;
            sess->qt[i].v1.quantizeParams.bwScaleOffsetEncoding.scale = t->q_scale;
            sess->qt[i].v1.quantizeParams.bwScaleOffsetEncoding.offset = t->q_offset;
        }
        if (t->ttype == QNN_TENSOR_TYPE_STATIC) {
            sess->qt[i].v1.clientBuf.data = t->data;
            sess->qt[i].v1.clientBuf.dataSize = rlx_tensor_data_size(t);
        }
        e = sess->qi->tensorCreateGraphTensor(sess->graph, &sess->qt[i]);
        if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_TENSOR);
    }

    /* Add nodes. MatMul carries transpose params; the rest carry none. */
    for (uint32_t n = 0; n < num_nodes; ++n) {
        const RlxQnnNode *nd = &nodes[n];
        Qnn_Tensor_t node_inputs[8];
        for (uint32_t j = 0; j < nd->num_inputs && j < 8; ++j) {
            node_inputs[j] = sess->qt[nd->inputs[j]];
        }
        Qnn_Tensor_t node_outputs[1] = {sess->qt[nd->output]};

        Qnn_Param_t params[6];
        uint32_t num_params = 0;
        if (strcmp(nd->op_type, QNN_OP_MAT_MUL) == 0) {
            params[0].paramType = QNN_PARAMTYPE_SCALAR;
            params[0].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0;
            params[0].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
            params[0].scalarParam.bool8Value = 0;
            params[1].paramType = QNN_PARAMTYPE_SCALAR;
            params[1].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1;
            params[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
            params[1].scalarParam.bool8Value = 0;
            num_params = 2;
        } else if ((strcmp(nd->op_type, QNN_OP_SOFTMAX) == 0 ||
                    strcmp(nd->op_type, QNN_OP_CONCAT) == 0) &&
                   nd->axis >= 0) {
            /* Softmax and Concat both take a single uint32 "axis" scalar param. */
            params[0].paramType = QNN_PARAMTYPE_SCALAR;
            params[0].name = QNN_OP_SOFTMAX_PARAM_AXIS;
            params[0].scalarParam.dataType = QNN_DATATYPE_UINT_32;
            params[0].scalarParam.uint32Value = (uint32_t)nd->axis;
            num_params = 1;
        } else if (strcmp(nd->op_type, QNN_OP_GATHER) == 0 && nd->axis >= 0) {
            /* Gather's "axis" is a *signed* int32 scalar. */
            params[0].paramType = QNN_PARAMTYPE_SCALAR;
            params[0].name = QNN_OP_GATHER_PARAM_AXIS;
            params[0].scalarParam.dataType = QNN_DATATYPE_INT_32;
            params[0].scalarParam.int32Value = nd->axis;
            num_params = 1;
        }
        /* Transpose's `perm` is a static uint32 tensor *param* (not a graph
         * tensor): embed it directly in the op config. */
        uint32_t perm_dims[2]; /* rank-1 (perm/axes) or rank-2 (ranges) param dims */
        char perm_name[80];
        Qnn_Tensor_t perm_tensor;
        /* Conv2d needs 3 registered tensor params, declared here so they outlive
         * the if-chain (referenced at graphAddNode below). */
        uint32_t cv_d1[1] = {2}, cv_d2[2] = {2, 2};
        char cv_n0[80], cv_n1[80], cv_n2[80];
        Qnn_Tensor_t cv_t0, cv_t1, cv_t2;
        if (strcmp(nd->op_type, QNN_OP_TRANSPOSE) == 0 && nd->perm_len > 0) {
            perm_dims[0] = nd->perm_len;
            snprintf(perm_name, sizeof(perm_name), "%s_perm", nd->name);
            perm_tensor = make_tensor(perm_name, QNN_TENSOR_TYPE_STATIC, perm_dims, 1);
            perm_tensor.v1.dataType = QNN_DATATYPE_UINT_32;
            perm_tensor.v1.clientBuf.data = (void *)nd->perm;
            perm_tensor.v1.clientBuf.dataSize = (uint32_t)(nd->perm_len * sizeof(uint32_t));
            /* Register the param tensor so it gets a valid backend id — the CPU
             * backend rejects unregistered (id-0) param tensors. */
            e = sess->qi->tensorCreateGraphTensor(sess->graph, &perm_tensor);
            if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_TENSOR);
            params[0].paramType = QNN_PARAMTYPE_TENSOR;
            params[0].name = QNN_OP_TRANSPOSE_PARAM_PERM;
            params[0].tensorParam = perm_tensor;
            num_params = 1;
        } else if ((strcmp(nd->op_type, QNN_OP_LAYER_NORM) == 0 ||
                    strcmp(nd->op_type, QNN_OP_RMS_NORM) == 0) &&
                   nd->perm_len > 0) {
            /* epsilon (float scalar) + axes (uint32 tensor param). The param
             * names ("epsilon"/"axes") are identical for LayerNorm and RmsNorm. */
            params[0].paramType = QNN_PARAMTYPE_SCALAR;
            params[0].name = QNN_OP_LAYER_NORM_PARAM_EPSILON;
            params[0].scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
            params[0].scalarParam.floatValue = nd->eps;
            perm_dims[0] = nd->perm_len;
            snprintf(perm_name, sizeof(perm_name), "%s_axes", nd->name);
            perm_tensor = make_tensor(perm_name, QNN_TENSOR_TYPE_STATIC, perm_dims, 1);
            perm_tensor.v1.dataType = QNN_DATATYPE_UINT_32;
            perm_tensor.v1.clientBuf.data = (void *)nd->perm;
            perm_tensor.v1.clientBuf.dataSize = (uint32_t)(nd->perm_len * sizeof(uint32_t));
            e = sess->qi->tensorCreateGraphTensor(sess->graph, &perm_tensor);
            if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_TENSOR);
            params[1].paramType = QNN_PARAMTYPE_TENSOR;
            params[1].name = QNN_OP_LAYER_NORM_PARAM_AXES;
            params[1].tensorParam = perm_tensor;
            num_params = 2;
        } else if (strcmp(nd->op_type, QNN_OP_STRIDED_SLICE) == 0 && nd->perm_len > 0) {
            /* `ranges` is a rank-2 [n_dims, 3] uint32 tensor param: begin, end,
             * stride per dim. (begin/end/shrink masks left at their 0 default.) */
            perm_dims[0] = nd->perm_len / 3u;
            perm_dims[1] = 3u;
            snprintf(perm_name, sizeof(perm_name), "%s_ranges", nd->name);
            perm_tensor = make_tensor(perm_name, QNN_TENSOR_TYPE_STATIC, perm_dims, 2);
            /* StridedSlice ranges are signed int32 (begin/end/stride may be
             * negative). Our begin/end/stride are non-negative, so the uint32
             * bit pattern reads back correctly as int32. */
            perm_tensor.v1.dataType = QNN_DATATYPE_INT_32;
            perm_tensor.v1.clientBuf.data = (void *)nd->perm;
            perm_tensor.v1.clientBuf.dataSize = (uint32_t)(nd->perm_len * sizeof(uint32_t));
            e = sess->qi->tensorCreateGraphTensor(sess->graph, &perm_tensor);
            if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_TENSOR);
            params[0].paramType = QNN_PARAMTYPE_TENSOR;
            params[0].name = QNN_OP_STRIDED_SLICE_PARAM_RANGES;
            params[0].tensorParam = perm_tensor;
            num_params = 1;
        } else if (strcmp(nd->op_type, QNN_OP_TILE) == 0 && nd->perm_len > 0) {
            /* `multiples`: a rank-1 uint32 tensor param, one repeat count per
             * input axis. */
            perm_dims[0] = nd->perm_len;
            snprintf(perm_name, sizeof(perm_name), "%s_mult", nd->name);
            perm_tensor = make_tensor(perm_name, QNN_TENSOR_TYPE_STATIC, perm_dims, 1);
            perm_tensor.v1.dataType = QNN_DATATYPE_UINT_32;
            perm_tensor.v1.clientBuf.data = (void *)nd->perm;
            perm_tensor.v1.clientBuf.dataSize = (uint32_t)(nd->perm_len * sizeof(uint32_t));
            e = sess->qi->tensorCreateGraphTensor(sess->graph, &perm_tensor);
            if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_TENSOR);
            params[0].paramType = QNN_PARAMTYPE_TENSOR;
            params[0].name = QNN_OP_TILE_PARAM_MULTIPLES;
            params[0].tensorParam = perm_tensor;
            num_params = 1;
        } else if ((strcmp(nd->op_type, QNN_OP_REDUCE_MEAN) == 0 ||
                    strcmp(nd->op_type, QNN_OP_REDUCE_SUM) == 0 ||
                    strcmp(nd->op_type, QNN_OP_REDUCE_MAX) == 0) &&
                   nd->perm_len > 0) {
            /* axes (uint32 tensor) + keep_dims (bool scalar, from `axis`). The
             * "axes"/"keep_dims" param names are identical across Reduce ops. */
            perm_dims[0] = nd->perm_len;
            snprintf(perm_name, sizeof(perm_name), "%s_axes", nd->name);
            perm_tensor = make_tensor(perm_name, QNN_TENSOR_TYPE_STATIC, perm_dims, 1);
            perm_tensor.v1.dataType = QNN_DATATYPE_UINT_32;
            perm_tensor.v1.clientBuf.data = (void *)nd->perm;
            perm_tensor.v1.clientBuf.dataSize = (uint32_t)(nd->perm_len * sizeof(uint32_t));
            e = sess->qi->tensorCreateGraphTensor(sess->graph, &perm_tensor);
            if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_TENSOR);
            params[0].paramType = QNN_PARAMTYPE_TENSOR;
            params[0].name = QNN_OP_REDUCE_MEAN_PARAM_AXES;
            params[0].tensorParam = perm_tensor;
            params[1].paramType = QNN_PARAMTYPE_SCALAR;
            params[1].name = QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS;
            params[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8;
            params[1].scalarParam.bool8Value = nd->axis > 0 ? 1 : 0;
            num_params = 2;
        } else if (strcmp(nd->op_type, QNN_OP_CONV_2D) == 0 && nd->perm_len == 8) {
            /* perm = [strideH,strideW, padT,padB,padL,padR, dilH,dilW]; group = axis.
             * Three registered uint32 tensor params + a group scalar. */
            snprintf(cv_n0, sizeof(cv_n0), "%s_stride", nd->name);
            cv_t0 = make_tensor(cv_n0, QNN_TENSOR_TYPE_STATIC, cv_d1, 1);
            cv_t0.v1.dataType = QNN_DATATYPE_UINT_32;
            cv_t0.v1.clientBuf.data = (void *)nd->perm;
            cv_t0.v1.clientBuf.dataSize = 2 * sizeof(uint32_t);
            e = sess->qi->tensorCreateGraphTensor(sess->graph, &cv_t0);
            if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_TENSOR);
            snprintf(cv_n1, sizeof(cv_n1), "%s_pad", nd->name);
            cv_t1 = make_tensor(cv_n1, QNN_TENSOR_TYPE_STATIC, cv_d2, 2);
            cv_t1.v1.dataType = QNN_DATATYPE_UINT_32;
            cv_t1.v1.clientBuf.data = (void *)(nd->perm + 2);
            cv_t1.v1.clientBuf.dataSize = 4 * sizeof(uint32_t);
            e = sess->qi->tensorCreateGraphTensor(sess->graph, &cv_t1);
            if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_TENSOR);
            snprintf(cv_n2, sizeof(cv_n2), "%s_dil", nd->name);
            cv_t2 = make_tensor(cv_n2, QNN_TENSOR_TYPE_STATIC, cv_d1, 1);
            cv_t2.v1.dataType = QNN_DATATYPE_UINT_32;
            cv_t2.v1.clientBuf.data = (void *)(nd->perm + 6);
            cv_t2.v1.clientBuf.dataSize = 2 * sizeof(uint32_t);
            e = sess->qi->tensorCreateGraphTensor(sess->graph, &cv_t2);
            if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_TENSOR);
            params[0].paramType = QNN_PARAMTYPE_TENSOR;
            params[0].name = QNN_OP_CONV_2D_PARAM_STRIDE;
            params[0].tensorParam = cv_t0;
            params[1].paramType = QNN_PARAMTYPE_TENSOR;
            params[1].name = QNN_OP_CONV_2D_PARAM_PAD_AMOUNT;
            params[1].tensorParam = cv_t1;
            params[2].paramType = QNN_PARAMTYPE_TENSOR;
            params[2].name = QNN_OP_CONV_2D_PARAM_DILATION;
            params[2].tensorParam = cv_t2;
            params[3].paramType = QNN_PARAMTYPE_SCALAR;
            params[3].name = QNN_OP_CONV_2D_PARAM_GROUP;
            params[3].scalarParam.dataType = QNN_DATATYPE_UINT_32;
            params[3].scalarParam.uint32Value = (uint32_t)(nd->axis < 1 ? 1 : nd->axis);
            num_params = 4;
        }

        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = nd->name;
        op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
        op.v1.typeName = nd->op_type;
        op.v1.numOfParams = num_params;
        op.v1.params = num_params ? params : NULL;
        op.v1.numOfInputs = nd->num_inputs;
        op.v1.inputTensors = node_inputs;
        op.v1.numOfOutputs = 1;
        op.v1.outputTensors = node_outputs;
        e = sess->qi->graphAddNode(sess->graph, op);
        if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_ADDNODE);
    }

    e = sess->qi->graphFinalize(sess->graph, NULL, NULL);
    if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_FINALIZE);

    sess->n_in = 0;
    sess->n_out = 0;
    for (uint32_t i = 0; i < num_tensors; ++i) {
        if (tensors[i].ttype == QNN_TENSOR_TYPE_APP_WRITE) ++sess->n_in;
        else if (tensors[i].ttype == QNN_TENSOR_TYPE_APP_READ) ++sess->n_out;
    }
    sess->in_idx = (uint32_t *)calloc(sess->n_in, sizeof(uint32_t));
    sess->out_idx = (uint32_t *)calloc(sess->n_out, sizeof(uint32_t));
    if ((sess->n_in && !sess->in_idx) || (sess->n_out && !sess->out_idx)) {
        rc = -RLX_QNN_E_TENSOR;
        goto cleanup_g;
    }
    uint32_t ii = 0, oo = 0;
    for (uint32_t i = 0; i < num_tensors; ++i) {
        if (tensors[i].ttype == QNN_TENSOR_TYPE_APP_WRITE) sess->in_idx[ii++] = i;
        else if (tensors[i].ttype == QNN_TENSOR_TYPE_APP_READ) sess->out_idx[oo++] = i;
    }
    sess->from_binary = 0;

#undef FAILG
cleanup_g:
    if (rc != RLX_QNN_OK) {
        rlx_qnn_session_free(sess);
        return rc;
    }
    if (out) *out = sess;
    return RLX_QNN_OK;
}

int rlx_qnn_session_execute(RlxQnnSession *sess,
                            RlxQnnTensor *tensors, uint32_t num_tensors,
                            uint64_t *err_out) {
    if (err_out) *err_out = 0;
    if (!sess || !sess->qi || !sess->graph) return -RLX_QNN_E_EXECUTE;

    Qnn_Tensor_t *exec_in =
        (Qnn_Tensor_t *)calloc(sess->n_in, sizeof(Qnn_Tensor_t));
    Qnn_Tensor_t *exec_out =
        (Qnn_Tensor_t *)calloc(sess->n_out, sizeof(Qnn_Tensor_t));
    if ((sess->n_in && !exec_in) || (sess->n_out && !exec_out)) {
        free(exec_in);
        free(exec_out);
        return -RLX_QNN_E_EXECUTE;
    }

    if (!sess->from_binary) {
        for (uint32_t j = 0; j < sess->n_in; ++j) {
            uint32_t idx = sess->in_idx[j];
            if (idx >= num_tensors) {
                free(exec_in);
                free(exec_out);
                return -RLX_QNN_E_EXECUTE;
            }
            sess->qt[idx].v1.clientBuf.data = tensors[idx].data;
            sess->qt[idx].v1.clientBuf.dataSize = rlx_tensor_data_size(&tensors[idx]);
            exec_in[j] = sess->qt[idx];
        }
        for (uint32_t j = 0; j < sess->n_out; ++j) {
            uint32_t idx = sess->out_idx[j];
            if (idx >= num_tensors) {
                free(exec_in);
                free(exec_out);
                return -RLX_QNN_E_EXECUTE;
            }
            sess->qt[idx].v1.clientBuf.data = tensors[idx].data;
            sess->qt[idx].v1.clientBuf.dataSize = rlx_tensor_data_size(&tensors[idx]);
            exec_out[j] = sess->qt[idx];
        }
    } else {
        for (uint32_t j = 0; j < sess->n_in; ++j) {
            uint32_t idx = sess->in_idx[j];
            const char *name = tensor_name_v1(&sess->qt[idx]);
            RlxQnnTensor *rt = find_rlx_tensor(tensors, num_tensors, name,
                                               QNN_TENSOR_TYPE_APP_WRITE);
            if (!rt) {
                free(exec_in);
                free(exec_out);
                return -RLX_QNN_E_EXECUTE;
            }
            if (sess->qt[idx].version == QNN_TENSOR_VERSION_1) {
                sess->qt[idx].v1.clientBuf.data = rt->data;
                sess->qt[idx].v1.clientBuf.dataSize = rlx_tensor_data_size(rt);
            } else {
                sess->qt[idx].v2.clientBuf.data = rt->data;
                sess->qt[idx].v2.clientBuf.dataSize = rlx_tensor_data_size(rt);
            }
            exec_in[j] = sess->qt[idx];
        }
        for (uint32_t j = 0; j < sess->n_out; ++j) {
            uint32_t idx = sess->out_idx[j];
            const char *name = tensor_name_v1(&sess->qt[idx]);
            RlxQnnTensor *rt = find_rlx_tensor(tensors, num_tensors, name,
                                               QNN_TENSOR_TYPE_APP_READ);
            if (!rt) {
                free(exec_in);
                free(exec_out);
                return -RLX_QNN_E_EXECUTE;
            }
            if (sess->qt[idx].version == QNN_TENSOR_VERSION_1) {
                sess->qt[idx].v1.clientBuf.data = rt->data;
                sess->qt[idx].v1.clientBuf.dataSize = rlx_tensor_data_size(rt);
            } else {
                sess->qt[idx].v2.clientBuf.data = rt->data;
                sess->qt[idx].v2.clientBuf.dataSize = rlx_tensor_data_size(rt);
            }
            exec_out[j] = sess->qt[idx];
        }
    }

    Qnn_ErrorHandle_t e = sess->qi->graphExecute(sess->graph, exec_in, sess->n_in,
                                                 exec_out, sess->n_out, NULL, NULL);
    free(exec_in);
    free(exec_out);
    if (e != QNN_SUCCESS) {
        if (err_out) *err_out = e;
        return -RLX_QNN_E_EXECUTE;
    }
    return RLX_QNN_OK;
}

int rlx_qnn_session_save_binary(RlxQnnSession *sess,
                                void **out_buf, uint64_t *written,
                                uint64_t *err_out) {
    if (err_out) *err_out = 0;
    if (out_buf) *out_buf = NULL;
    if (written) *written = 0;
    if (!sess || !sess->qi || !sess->context) return -RLX_QNN_E_BINARY;
    if (!sess->qi->contextGetBinarySize || !sess->qi->contextGetBinary) {
        return -RLX_QNN_E_BINARY;
    }

    Qnn_ContextBinarySize_t sz = 0;
    Qnn_ErrorHandle_t e = sess->qi->contextGetBinarySize(sess->context, &sz);
    if (e != QNN_SUCCESS || sz == 0) {
        if (err_out) *err_out = e;
        return -RLX_QNN_E_BINARY;
    }

    void *buf = malloc((size_t)sz);
    if (!buf) return -RLX_QNN_E_BINARY;

    Qnn_ContextBinarySize_t got = 0;
    e = sess->qi->contextGetBinary(sess->context, buf, sz, &got);
    if (e != QNN_SUCCESS) {
        if (err_out) *err_out = e;
        free(buf);
        return -RLX_QNN_E_BINARY;
    }

    if (out_buf) *out_buf = buf;
    else free(buf);
    if (written) *written = (uint64_t)got;
    return RLX_QNN_OK;
}

int rlx_qnn_session_load_binary(const char *backend_lib,
                                const void *binary, uint64_t binary_size,
                                RlxQnnSession **out,
                                uint64_t *err_out) {
    if (err_out) *err_out = 0;
    if (out) *out = NULL;
    int rc = RLX_QNN_OK;

    if (!binary || binary_size == 0) return -RLX_QNN_E_BINARY;

    RlxQnnSession *sess = (RlxQnnSession *)calloc(1, sizeof(RlxQnnSession));
    if (!sess) return -RLX_QNN_E_BINARY;

    Qnn_ErrorHandle_t e = QNN_SUCCESS;

#define FAILB(step)                \
    do {                           \
        if (err_out) *err_out = e; \
        rc = -(step);              \
        goto cleanup_b;            \
    } while (0)

    sess->lib = dlopen(backend_lib, RTLD_NOW | RTLD_LOCAL);
    if (!sess->lib) {
        fprintf(stderr, "rlx_qnn: dlopen(%s) failed: %s\n", backend_lib, dlerror());
        rlx_qnn_session_free(sess);
        return -RLX_QNN_E_DLOPEN;
    }

    GetProvidersFn get_providers =
        (GetProvidersFn)dlsym(sess->lib, "QnnInterface_getProviders");
    if (!get_providers) {
        rlx_qnn_session_free(sess);
        return -RLX_QNN_E_GETPROC;
    }

    const QnnInterface_t **providers = NULL;
    uint32_t num_providers = 0;
    if (get_providers(&providers, &num_providers) != QNN_SUCCESS ||
        num_providers == 0 || providers == NULL) {
        rlx_qnn_session_free(sess);
        return -RLX_QNN_E_PROVIDERS;
    }
    sess->qi = &providers[0]->QNN_INTERFACE_VER_NAME;

    if (sess->qi->logCreate) {
        sess->qi->logCreate(rlx_qnn_log_cb, QNN_LOG_LEVEL_WARN, &sess->logger);
    }
    e = sess->qi->backendCreate(sess->logger, NULL, &sess->backend);
    if (e != QNN_SUCCESS) FAILB(RLX_QNN_E_BACKEND);

    char sys_path[1024];
    resolve_system_lib_path(backend_lib, sys_path, sizeof(sys_path));
    sess->sys_lib = dlopen(sys_path, RTLD_NOW | RTLD_LOCAL);
    if (!sess->sys_lib) {
        fprintf(stderr, "rlx_qnn: dlopen(%s) failed: %s\n", sys_path, dlerror());
        FAILB(RLX_QNN_E_SYSTEM);
    }

    SystemGetProvidersFn sys_get_providers = (SystemGetProvidersFn)dlsym(
        sess->sys_lib, "QnnSystemInterface_getProviders");
    if (!sys_get_providers) FAILB(RLX_QNN_E_SYSTEM);

    const QnnSystemInterface_t **sys_providers = NULL;
    uint32_t num_sys_providers = 0;
    if (sys_get_providers(&sys_providers, &num_sys_providers) != QNN_SUCCESS ||
        num_sys_providers == 0 || sys_providers == NULL) {
        FAILB(RLX_QNN_E_SYSTEM);
    }
    const QNN_SYSTEM_INTERFACE_VER_TYPE *si =
        &sys_providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    if (!si->systemContextCreate || !si->systemContextGetBinaryInfo ||
        !si->systemContextFree) {
        FAILB(RLX_QNN_E_SYSTEM);
    }

    QnnSystemContext_Handle_t sys = NULL;
    e = si->systemContextCreate(&sys);
    if (e != QNN_SUCCESS || !sys) FAILB(RLX_QNN_E_SYSTEM);

    const QnnSystemContext_BinaryInfo_t *binary_info = NULL;
    Qnn_ContextBinarySize_t info_size = 0;
    e = si->systemContextGetBinaryInfo(sys, (void *)binary, binary_size,
                                       &binary_info, &info_size);
    if (e != QNN_SUCCESS || !binary_info) {
        si->systemContextFree(sys);
        FAILB(RLX_QNN_E_BINARY);
    }

    const QnnSystemContext_GraphInfo_t *graphs = NULL;
    uint32_t num_graphs = 0;
    if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        num_graphs = binary_info->contextBinaryInfoV1.numGraphs;
        graphs = binary_info->contextBinaryInfoV1.graphs;
    } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        num_graphs = binary_info->contextBinaryInfoV2.numGraphs;
        graphs = binary_info->contextBinaryInfoV2.graphs;
    } else if (binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        num_graphs = binary_info->contextBinaryInfoV3.numGraphs;
        graphs = binary_info->contextBinaryInfoV3.graphs;
    } else {
        si->systemContextFree(sys);
        FAILB(RLX_QNN_E_BINARY);
    }

    if (num_graphs == 0 || !graphs) {
        si->systemContextFree(sys);
        FAILB(RLX_QNN_E_BINARY);
    }

    const char *graph_name = "rlx_graph";
    const Qnn_Tensor_t *graph_inputs = NULL;
    const Qnn_Tensor_t *graph_outputs = NULL;
    uint32_t n_in = 0, n_out = 0;
    if (extract_graph_io(&graphs[0], &graph_name, &graph_inputs, &n_in,
                         &graph_outputs, &n_out) != 0) {
        si->systemContextFree(sys);
        FAILB(RLX_QNN_E_BINARY);
    }

    sess->n_in = n_in;
    sess->n_out = n_out;
    sess->num_tensors = n_in + n_out;
    sess->qt = (Qnn_Tensor_t *)calloc(sess->num_tensors, sizeof(Qnn_Tensor_t));
    sess->owned_names = (char **)calloc(sess->num_tensors, sizeof(char *));
    sess->owned_dims = (uint32_t **)calloc(sess->num_tensors, sizeof(uint32_t *));
    sess->in_idx = (uint32_t *)calloc(n_in, sizeof(uint32_t));
    sess->out_idx = (uint32_t *)calloc(n_out, sizeof(uint32_t));
    if (!sess->qt || !sess->owned_names || !sess->owned_dims ||
        (n_in && !sess->in_idx) || (n_out && !sess->out_idx)) {
        si->systemContextFree(sys);
        rc = -RLX_QNN_E_BINARY;
        goto cleanup_b;
    }
    sess->n_owned = sess->num_tensors;

    for (uint32_t i = 0; i < n_in; ++i) {
        if (deep_copy_qnn_tensor(&sess->qt[i], &graph_inputs[i],
                                 &sess->owned_names[i],
                                 &sess->owned_dims[i]) != 0) {
            si->systemContextFree(sys);
            rc = -RLX_QNN_E_BINARY;
            goto cleanup_b;
        }
        sess->in_idx[i] = i;
    }
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t dst = n_in + i;
        if (deep_copy_qnn_tensor(&sess->qt[dst], &graph_outputs[i],
                                 &sess->owned_names[dst],
                                 &sess->owned_dims[dst]) != 0) {
            si->systemContextFree(sys);
            rc = -RLX_QNN_E_BINARY;
            goto cleanup_b;
        }
        sess->out_idx[i] = dst;
    }

    if (graph_name) {
        sess->graph_name = strdup(graph_name);
    }

    si->systemContextFree(sys);
    sys = NULL;
    dlclose(sess->sys_lib);
    sess->sys_lib = NULL;

    if (!sess->qi->contextCreateFromBinary || !sess->qi->graphRetrieve) {
        FAILB(RLX_QNN_E_BINARY);
    }

    e = sess->qi->contextCreateFromBinary(
        sess->backend, NULL, NULL, binary, (Qnn_ContextBinarySize_t)binary_size,
        &sess->context, NULL);
    if (e != QNN_SUCCESS) FAILB(RLX_QNN_E_BINARY);

    const char *retrieve_name = sess->graph_name ? sess->graph_name : "rlx_graph";
    e = sess->qi->graphRetrieve(sess->context, retrieve_name, &sess->graph);
    if (e != QNN_SUCCESS) FAILB(RLX_QNN_E_BINARY);

    sess->from_binary = 1;

#undef FAILB
cleanup_b:
    if (rc != RLX_QNN_OK) {
        rlx_qnn_session_free(sess);
        return rc;
    }
    if (out) *out = sess;
    return RLX_QNN_OK;
}

int rlx_qnn_run_graph(const char *backend_lib,
                      RlxQnnTensor *tensors, uint32_t num_tensors,
                      const RlxQnnNode *nodes, uint32_t num_nodes,
                      uint64_t *err_out) {
    RlxQnnSession *sess = NULL;
    int rc = rlx_qnn_session_create(backend_lib, tensors, num_tensors, nodes,
                                    num_nodes, &sess, err_out);
    if (rc != RLX_QNN_OK) return rc;
    rc = rlx_qnn_session_execute(sess, tensors, num_tensors, err_out);
    rlx_qnn_session_free(sess);
    return rc;
}
