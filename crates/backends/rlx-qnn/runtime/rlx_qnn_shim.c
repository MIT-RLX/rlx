/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * GPL-3.0-only. See the crate root for the full license text.
 *
 * QNN AI Engine Direct FFI shim — dynamic (style-1) graph build + execute for a
 * single f32 MatMul. Compiled against the real SDK headers; the backend library
 * is resolved at run time via dlopen so the build stays driverless.
 */

#include "rlx_qnn_shim.h"

#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "QnnInterface.h"
#include "QnnLog.h"
#include "QnnOpDef.h"
#include "QnnTypes.h"

typedef Qnn_ErrorHandle_t (*GetProvidersFn)(const QnnInterface_t ***, uint32_t *);

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

int rlx_qnn_run_graph(const char *backend_lib,
                      RlxQnnTensor *tensors, uint32_t num_tensors,
                      const RlxQnnNode *nodes, uint32_t num_nodes,
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
    const QNN_INTERFACE_VER_TYPE *qi = &providers[0]->QNN_INTERFACE_VER_NAME;

    Qnn_BackendHandle_t backend = NULL;
    Qnn_ContextHandle_t context = NULL;
    Qnn_GraphHandle_t graph = NULL;
    Qnn_Tensor_t *qt = (Qnn_Tensor_t *)calloc(num_tensors, sizeof(Qnn_Tensor_t));
    Qnn_Tensor_t *exec_in = (Qnn_Tensor_t *)calloc(num_tensors, sizeof(Qnn_Tensor_t));
    Qnn_Tensor_t *exec_out = (Qnn_Tensor_t *)calloc(num_tensors, sizeof(Qnn_Tensor_t));
    Qnn_ErrorHandle_t e = QNN_SUCCESS;

#define FAILG(step)                \
    do {                           \
        if (err_out) *err_out = e; \
        rc = -(step);              \
        goto cleanup_g;            \
    } while (0)

    if (!qt || !exec_in || !exec_out) { rc = -RLX_QNN_E_TENSOR; goto cleanup_g; }

    Qnn_LogHandle_t logger = NULL;
    if (qi->logCreate) {
        qi->logCreate(rlx_qnn_log_cb, QNN_LOG_LEVEL_WARN, &logger);
    }
    e = qi->backendCreate(logger, NULL, &backend);
    if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_BACKEND);
    e = qi->contextCreate(backend, NULL, NULL, &context);
    if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_CONTEXT);
    e = qi->graphCreate(context, "rlx_graph", NULL, &graph);
    if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_GRAPH);

    /* Create every graph tensor; STATIC tensors carry their data now. */
    for (uint32_t i = 0; i < num_tensors; ++i) {
        RlxQnnTensor *t = &tensors[i];
        qt[i] = make_tensor(t->name, (Qnn_TensorType_t)t->ttype, (uint32_t *)t->dims, t->rank);
        if (t->dtype == 1) {
            qt[i].v1.dataType = QNN_DATATYPE_INT_32;
        } else if (t->dtype == 2) {
            qt[i].v1.dataType = QNN_DATATYPE_SFIXED_POINT_8;
            qt[i].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
            qt[i].v1.quantizeParams.quantizationEncoding =
                QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            qt[i].v1.quantizeParams.scaleOffsetEncoding.scale = t->q_scale;
            qt[i].v1.quantizeParams.scaleOffsetEncoding.offset = t->q_offset;
        }
        if (t->ttype == QNN_TENSOR_TYPE_STATIC) {
            qt[i].v1.clientBuf.data = t->data;
            qt[i].v1.clientBuf.dataSize = (uint32_t)(t->num_elems * sizeof(float));
        }
        e = qi->tensorCreateGraphTensor(graph, &qt[i]);
        if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_TENSOR);
    }

    /* Add nodes. MatMul carries transpose params; the rest carry none. */
    for (uint32_t n = 0; n < num_nodes; ++n) {
        const RlxQnnNode *nd = &nodes[n];
        Qnn_Tensor_t node_inputs[8];
        for (uint32_t j = 0; j < nd->num_inputs && j < 8; ++j) {
            node_inputs[j] = qt[nd->inputs[j]];
        }
        Qnn_Tensor_t node_outputs[1] = {qt[nd->output]};

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
            e = qi->tensorCreateGraphTensor(graph, &perm_tensor);
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
            e = qi->tensorCreateGraphTensor(graph, &perm_tensor);
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
            e = qi->tensorCreateGraphTensor(graph, &perm_tensor);
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
            e = qi->tensorCreateGraphTensor(graph, &perm_tensor);
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
            e = qi->tensorCreateGraphTensor(graph, &perm_tensor);
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
            e = qi->tensorCreateGraphTensor(graph, &cv_t0);
            if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_TENSOR);
            snprintf(cv_n1, sizeof(cv_n1), "%s_pad", nd->name);
            cv_t1 = make_tensor(cv_n1, QNN_TENSOR_TYPE_STATIC, cv_d2, 2);
            cv_t1.v1.dataType = QNN_DATATYPE_UINT_32;
            cv_t1.v1.clientBuf.data = (void *)(nd->perm + 2);
            cv_t1.v1.clientBuf.dataSize = 4 * sizeof(uint32_t);
            e = qi->tensorCreateGraphTensor(graph, &cv_t1);
            if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_TENSOR);
            snprintf(cv_n2, sizeof(cv_n2), "%s_dil", nd->name);
            cv_t2 = make_tensor(cv_n2, QNN_TENSOR_TYPE_STATIC, cv_d1, 1);
            cv_t2.v1.dataType = QNN_DATATYPE_UINT_32;
            cv_t2.v1.clientBuf.data = (void *)(nd->perm + 6);
            cv_t2.v1.clientBuf.dataSize = 2 * sizeof(uint32_t);
            e = qi->tensorCreateGraphTensor(graph, &cv_t2);
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
        e = qi->graphAddNode(graph, op);
        if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_ADDNODE);
    }

    e = qi->graphFinalize(graph, NULL, NULL);
    if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_FINALIZE);

    /* Bind runtime I/O buffers and execute. */
    uint32_t n_in = 0, n_out = 0;
    for (uint32_t i = 0; i < num_tensors; ++i) {
        if (tensors[i].ttype == QNN_TENSOR_TYPE_APP_WRITE) {
            qt[i].v1.clientBuf.data = tensors[i].data;
            qt[i].v1.clientBuf.dataSize = (uint32_t)(tensors[i].num_elems * sizeof(float));
            exec_in[n_in++] = qt[i];
        } else if (tensors[i].ttype == QNN_TENSOR_TYPE_APP_READ) {
            qt[i].v1.clientBuf.data = tensors[i].data;
            qt[i].v1.clientBuf.dataSize = (uint32_t)(tensors[i].num_elems * sizeof(float));
            exec_out[n_out++] = qt[i];
        }
    }
    e = qi->graphExecute(graph, exec_in, n_in, exec_out, n_out, NULL, NULL);
    if (e != QNN_SUCCESS) FAILG(RLX_QNN_E_EXECUTE);

#undef FAILG
cleanup_g:
    if (context && qi->contextFree) qi->contextFree(context, NULL);
    if (backend && qi->backendFree) qi->backendFree(backend);
    free(qt);
    free(exec_in);
    free(exec_out);
    dlclose(lib);
    return rc;
}
