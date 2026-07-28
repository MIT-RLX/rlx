// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Emit the artifact set for a [`Model`]: `qnn_model.cpp`, `verify.py`,
//! `run_qnn.sh`, and `run_qnn_context.sh` (offline context-binary path).
//!
//! Supported: a single rank-2 matmul, a two-node `Linear` graph
//! (`MatMul` → `ElementWiseAdd`), a three-node `LinearRelu` graph
//! (`MatMul` → `ElementWiseAdd` → `Relu`), a two-node `MatMulSoftmax`
//! graph (`MatMul` → `Softmax`), a two-layer `Mlp2`
//! (`LinearRelu` → `Linear`), or `LinearStatic` (activation `APP_WRITE`,
//! weight/bias baked as `STATIC`). Intermediate tensors are `NATIVE`;
//! the final tensor is `APP_READ`. The C++ is composed with the
//! `qnn_wrapper_api` surface — the `qnn-onnx-converter` emits — so it
//! builds into a `.so` with `qnn-model-lib-generator` and runs under
//! `qnn-net-run`.

use std::path::Path;

use crate::cpp::Cpp;
use crate::model::{Layer, Model};

/// One emitted file: its relative path and its full contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    pub path: String,
    pub contents: String,
}

/// Build every artifact for `model` in memory (no filesystem I/O).
pub fn collect_artifacts(model: &Model) -> Result<Vec<Artifact>, String> {
    let mut arts = match model.layers.as_slice() {
        [Layer::MatMul { m, k, n, .. }] => collect_matmul_artifacts(*m, *k, *n)?,
        [Layer::Linear { m, k, n, .. }] => collect_linear_artifacts(*m, *k, *n)?,
        [Layer::LinearRelu { m, k, n, .. }] => collect_linear_relu_artifacts(*m, *k, *n)?,
        [Layer::MatMulSoftmax { m, k, n, .. }] => collect_matmul_softmax_artifacts(*m, *k, *n)?,
        [Layer::LinearStatic { m, k, n, w, b, .. }] => {
            collect_linear_static_artifacts(*m, *k, *n, w, b)?
        }
        [
            Layer::LinearRelu { m, k, n: h, .. },
            Layer::Linear {
                m: m2, k: h2, n, ..
            },
        ] if m == m2 && h == h2 => collect_mlp2_artifacts(*m, *k, *h, *n)?,
        other => {
            return Err(format!(
                "rlx-qnn codegen: unsupported layer layout ({} layers)",
                other.len()
            ));
        }
    };
    // Offline context-binary path (style-2): model.so → .bin → --retrieve_context.
    arts.push(Artifact {
        path: "run_qnn_context.sh".to_string(),
        contents: emit_run_qnn_context_sh(),
    });
    Ok(arts)
}

fn collect_matmul_artifacts(m: usize, k: usize, n: usize) -> Result<Vec<Artifact>, String> {
    if m == 0 || k == 0 || n == 0 {
        return Err(format!(
            "MatMul dims must be non-zero; got M={m} K={k} N={n}"
        ));
    }

    Ok(vec![
        Artifact {
            path: "qnn_model.cpp".to_string(),
            contents: emit_qnn_model_cpp(m, k, n),
        },
        Artifact {
            path: "verify.py".to_string(),
            contents: emit_verify_py(m, k, n),
        },
        Artifact {
            path: "run_qnn.sh".to_string(),
            contents: emit_run_qnn_sh(),
        },
    ])
}

fn collect_linear_artifacts(m: usize, k: usize, n: usize) -> Result<Vec<Artifact>, String> {
    if m == 0 || k == 0 || n == 0 {
        return Err(format!(
            "Linear dims must be non-zero; got M={m} K={k} N={n}"
        ));
    }

    Ok(vec![
        Artifact {
            path: "qnn_model.cpp".to_string(),
            contents: emit_linear_qnn_model_cpp(m, k, n),
        },
        Artifact {
            path: "verify.py".to_string(),
            contents: emit_linear_verify_py(m, k, n),
        },
        Artifact {
            path: "run_qnn.sh".to_string(),
            contents: emit_run_qnn_sh(),
        },
    ])
}

fn collect_linear_relu_artifacts(m: usize, k: usize, n: usize) -> Result<Vec<Artifact>, String> {
    if m == 0 || k == 0 || n == 0 {
        return Err(format!(
            "LinearRelu dims must be non-zero; got M={m} K={k} N={n}"
        ));
    }

    Ok(vec![
        Artifact {
            path: "qnn_model.cpp".to_string(),
            contents: emit_linear_relu_qnn_model_cpp(m, k, n),
        },
        Artifact {
            path: "verify.py".to_string(),
            contents: emit_linear_relu_verify_py(m, k, n),
        },
        Artifact {
            path: "run_qnn.sh".to_string(),
            contents: emit_run_qnn_sh(),
        },
    ])
}

fn collect_matmul_softmax_artifacts(m: usize, k: usize, n: usize) -> Result<Vec<Artifact>, String> {
    if m == 0 || k == 0 || n == 0 {
        return Err(format!(
            "MatMulSoftmax dims must be non-zero; got M={m} K={k} N={n}"
        ));
    }

    Ok(vec![
        Artifact {
            path: "qnn_model.cpp".to_string(),
            contents: emit_matmul_softmax_qnn_model_cpp(m, k, n),
        },
        Artifact {
            path: "verify.py".to_string(),
            contents: emit_matmul_softmax_verify_py(m, k, n),
        },
        Artifact {
            path: "run_qnn.sh".to_string(),
            contents: emit_run_qnn_sh(),
        },
    ])
}

fn collect_mlp2_artifacts(m: usize, k: usize, h: usize, n: usize) -> Result<Vec<Artifact>, String> {
    if m == 0 || k == 0 || h == 0 || n == 0 {
        return Err(format!(
            "Mlp2 dims must be non-zero; got M={m} K={k} H={h} N={n}"
        ));
    }

    Ok(vec![
        Artifact {
            path: "qnn_model.cpp".to_string(),
            contents: emit_mlp2_qnn_model_cpp(m, k, h, n),
        },
        Artifact {
            path: "verify.py".to_string(),
            contents: emit_mlp2_verify_py(m, k, h, n),
        },
        Artifact {
            path: "run_qnn.sh".to_string(),
            contents: emit_run_qnn_sh(),
        },
    ])
}

fn collect_linear_static_artifacts(
    m: usize,
    k: usize,
    n: usize,
    w: &[f32],
    b: &[f32],
) -> Result<Vec<Artifact>, String> {
    if m == 0 || k == 0 || n == 0 {
        return Err(format!(
            "LinearStatic dims must be non-zero; got M={m} K={k} N={n}"
        ));
    }
    if w.len() != k * n {
        return Err(format!("LinearStatic w len {} != K*N={}", w.len(), k * n));
    }
    if b.len() != m * n {
        return Err(format!("LinearStatic b len {} != M*N={}", b.len(), m * n));
    }

    Ok(vec![
        Artifact {
            path: "qnn_model.cpp".to_string(),
            contents: emit_linear_static_qnn_model_cpp(m, k, n, w, b),
        },
        Artifact {
            path: "verify.py".to_string(),
            contents: emit_linear_static_verify_py(m, k, n, w, b),
        },
        Artifact {
            path: "run_qnn.sh".to_string(),
            contents: emit_run_qnn_sh(),
        },
    ])
}

fn format_c_float_array(name: &str, vals: &[f32]) -> String {
    let mut s = format!("static float {name}[] = {{\n");
    for (i, v) in vals.iter().enumerate() {
        if i % 8 == 0 {
            s.push_str("  ");
        }
        s.push_str(&format!("{v:.8e}f"));
        if i + 1 < vals.len() {
            s.push(',');
            if (i + 1) % 8 == 0 {
                s.push('\n');
            } else {
                s.push(' ');
            }
        }
    }
    s.push_str("\n};\n");
    s
}

fn format_py_float_list(vals: &[f32]) -> String {
    let mut s = String::from("[");
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{v:.8e}"));
    }
    s.push(']');
    s
}

/// Emit all artifacts for `model` into `out_dir` (created if absent).
pub fn emit_model(model: &Model, out_dir: &Path) -> std::io::Result<()> {
    let artifacts = collect_artifacts(model).map_err(std::io::Error::other)?;
    std::fs::create_dir_all(out_dir)?;
    for a in &artifacts {
        std::fs::write(out_dir.join(&a.path), &a.contents)?;
    }
    Ok(())
}

/// `qnn_model.cpp` — compose the matmul graph with the `qnn_wrapper_api`
/// surface: two `APP_WRITE` input tensors, one `MatMul` node, one `APP_READ`
/// output. `qnn-model-lib-generator` turns this into `libqnn_model.so`.
fn emit_qnn_model_cpp(m: usize, k: usize, n: usize) -> String {
    let mut c = Cpp::new();
    c.banner("Generated by rlx-qnn — QNN AI Engine Direct model graph.");
    c.comment("out[M,N] = in0[M,K] * in1[K,N], row-major f32.");
    c.comment("Composed with the qnn_wrapper_api surface (the qnn-onnx-converter shape).");
    c.comment(
        "Build:  qnn-model-lib-generator -c qnn_model.cpp -t x86_64-linux-clang -o model_libs",
    );
    c.comment(
        "Run:    qnn-net-run --backend libQnnCpu.so --model libqnn_model.so ...  (see run_qnn.sh)",
    );
    c.blank();
    c.line("#include \"QnnModel.hpp\"");
    c.line("#include \"QnnOpDef.h\"");
    c.blank();
    c.comment("Have the backend validate each op node as it is added.");
    c.line("#define DO_GRAPH_NODE_VALIDATIONS 1");
    c.blank();
    c.line("using namespace qnn_wrapper_api;");
    c.blank();
    c.line("extern \"C\" {");
    c.blank();
    c.line("QNN_API");
    c.line("ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,");
    c.line("                                    QNN_INTERFACE_VER_TYPE interface,");
    c.line("                                    Qnn_ContextHandle_t contextHandle,");
    c.line("                                    const GraphConfigInfo_t** graphsConfigInfo,");
    c.line("                                    const uint32_t numGraphsConfigInfo,");
    c.line("                                    GraphInfoPtr_t** graphsInfo,");
    c.line("                                    uint32_t* numGraphsInfo,");
    c.line("                                    bool debug,");
    c.line("                                    QnnLog_Callback_t logCallback,");
    c.line("                                    QnnLog_Level_t maxLogLevel) {");
    c.block(|c| {
        c.line("ModelError_t err = MODEL_NO_ERROR;");
        c.blank();
        c.line("QnnModel matmul_graph;");
        c.line("const QnnGraph_Config_t** graphConfigs = nullptr;");
        c.line("VALIDATE(getQnnGraphConfigFromInfo(\"matmul_graph\", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);");
        c.line("VALIDATE(matmul_graph.initialize(backendHandle, interface, contextHandle, \"matmul_graph\", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);");
        c.blank();
        c.comment("graph input in0 [M, K]");
        c.line(&format!("uint32_t dimensions_in0[] = {{{m}, {k}}};"));
        c.tensor_v1(
            "VALIDATE(matmul_graph.addTensor(\"in0\", ",
            "in0",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in0",
            "), err);",
        );
        c.blank();
        c.comment("graph input in1 [K, N]");
        c.line(&format!("uint32_t dimensions_in1[] = {{{k}, {n}}};"));
        c.tensor_v1(
            "VALIDATE(matmul_graph.addTensor(\"in1\", ",
            "in1",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in1",
            "), err);",
        );
        c.blank();
        c.comment("MatMul node: out[M, N] = in0 * in1  (no transpose)");
        c.line("const char* inputs_matmul_0[] = {\"in0\", \"in1\"};");
        c.line(&format!("uint32_t dimensions_out[] = {{{m}, {n}}};"));
        c.line("Qnn_Tensor_t outputs_matmul_0[] = {");
        c.block(|c| {
            c.tensor_v1("", "out", "QNN_TENSOR_TYPE_APP_READ", "dimensions_out", "");
        });
        c.line("};");
        c.line("Qnn_Param_t params_matmul_0[] = {");
        c.block(|c| {
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}},");
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}");
        });
        c.line("};");
        c.line("VALIDATE(matmul_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"matmul_0\",                    // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("QNN_OP_MAT_MUL,                // QNN op type");
            c.line("params_matmul_0, 2,            // params + count");
            c.line("inputs_matmul_0, 2,            // input tensor names + count");
            c.line("outputs_matmul_0, 1            // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("Collect the constructed graph into the output variables. The");
        c.comment("graph is finalized later by the runtime (e.g. qnn-net-run).");
        c.line("QnnModel* models[] = {&matmul_graph};");
        c.line("uint32_t numModels = 1;");
        c.line("VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);");
        c.line("*numGraphsInfo = numModels;");
        c.blank();
        c.line("return err;");
    });
    c.line("}");
    c.blank();
    c.line("QNN_API");
    c.line("ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo) {");
    c.block(|c| {
        c.line("return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);");
    });
    c.line("}");
    c.blank();
    c.line("}  // extern \"C\"");
    c.into_string()
}

/// `qnn_model.cpp` — two-node graph: MatMul → ElementWiseAdd.
fn emit_linear_qnn_model_cpp(m: usize, k: usize, n: usize) -> String {
    let mut c = Cpp::new();
    c.banner("Generated by rlx-qnn — QNN AI Engine Direct linear graph.");
    c.comment("out[M,N] = in0[M,K] * in1[K,N] + in2[M,N], row-major f32.");
    c.comment("Composed with the qnn_wrapper_api surface (the qnn-onnx-converter shape).");
    c.comment(
        "Build:  qnn-model-lib-generator -c qnn_model.cpp -t x86_64-linux-clang -o model_libs",
    );
    c.comment(
        "Run:    qnn-net-run --backend libQnnCpu.so --model libqnn_model.so ...  (see run_qnn.sh)",
    );
    c.blank();
    c.line("#include \"QnnModel.hpp\"");
    c.line("#include \"QnnOpDef.h\"");
    c.blank();
    c.comment("Have the backend validate each op node as it is added.");
    c.line("#define DO_GRAPH_NODE_VALIDATIONS 1");
    c.blank();
    c.line("using namespace qnn_wrapper_api;");
    c.blank();
    c.line("extern \"C\" {");
    c.blank();
    c.line("QNN_API");
    c.line("ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,");
    c.line("                                    QNN_INTERFACE_VER_TYPE interface,");
    c.line("                                    Qnn_ContextHandle_t contextHandle,");
    c.line("                                    const GraphConfigInfo_t** graphsConfigInfo,");
    c.line("                                    const uint32_t numGraphsConfigInfo,");
    c.line("                                    GraphInfoPtr_t** graphsInfo,");
    c.line("                                    uint32_t* numGraphsInfo,");
    c.line("                                    bool debug,");
    c.line("                                    QnnLog_Callback_t logCallback,");
    c.line("                                    QnnLog_Level_t maxLogLevel) {");
    c.block(|c| {
        c.line("ModelError_t err = MODEL_NO_ERROR;");
        c.blank();
        c.line("QnnModel linear_graph;");
        c.line("const QnnGraph_Config_t** graphConfigs = nullptr;");
        c.line("VALIDATE(getQnnGraphConfigFromInfo(\"linear_graph\", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);");
        c.line("VALIDATE(linear_graph.initialize(backendHandle, interface, contextHandle, \"linear_graph\", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);");
        c.blank();
        c.comment("graph input in0 [M, K]");
        c.line(&format!("uint32_t dimensions_in0[] = {{{m}, {k}}};"));
        c.tensor_v1(
            "VALIDATE(linear_graph.addTensor(\"in0\", ",
            "in0",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in0",
            "), err);",
        );
        c.blank();
        c.comment("graph input in1 [K, N]");
        c.line(&format!("uint32_t dimensions_in1[] = {{{k}, {n}}};"));
        c.tensor_v1(
            "VALIDATE(linear_graph.addTensor(\"in1\", ",
            "in1",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in1",
            "), err);",
        );
        c.blank();
        c.comment("graph input in2 [M, N] (bias / addend)");
        c.line(&format!("uint32_t dimensions_in2[] = {{{m}, {n}}};"));
        c.tensor_v1(
            "VALIDATE(linear_graph.addTensor(\"in2\", ",
            "in2",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in2",
            "), err);",
        );
        c.blank();
        c.comment("MatMul node: mm[M, N] = in0 * in1  (no transpose)");
        c.line("const char* inputs_matmul_0[] = {\"in0\", \"in1\"};");
        c.line(&format!("uint32_t dimensions_mm[] = {{{m}, {n}}};"));
        c.line("Qnn_Tensor_t outputs_matmul_0[] = {");
        c.block(|c| {
            c.tensor_v1("", "mm", "QNN_TENSOR_TYPE_NATIVE", "dimensions_mm", "");
        });
        c.line("};");
        c.line("Qnn_Param_t params_matmul_0[] = {");
        c.block(|c| {
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}},");
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}");
        });
        c.line("};");
        c.line("VALIDATE(linear_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"matmul_0\",                    // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("QNN_OP_MAT_MUL,                // QNN op type");
            c.line("params_matmul_0, 2,            // params + count");
            c.line("inputs_matmul_0, 2,            // input tensor names + count");
            c.line("outputs_matmul_0, 1            // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("ElementWiseAdd node: out[M, N] = mm + in2");
        c.line("const char* inputs_add_0[] = {\"mm\", \"in2\"};");
        c.line(&format!("uint32_t dimensions_out[] = {{{m}, {n}}};"));
        c.line("Qnn_Tensor_t outputs_add_0[] = {");
        c.block(|c| {
            c.tensor_v1("", "out", "QNN_TENSOR_TYPE_APP_READ", "dimensions_out", "");
        });
        c.line("};");
        c.line("VALIDATE(linear_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"add_0\",                       // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("\"ElementWiseAdd\",            // QNN op type");
            c.line("nullptr, 0,                    // params + count");
            c.line("inputs_add_0, 2,               // input tensor names + count");
            c.line("outputs_add_0, 1               // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("Collect the constructed graph into the output variables. The");
        c.comment("graph is finalized later by the runtime (e.g. qnn-net-run).");
        c.line("QnnModel* models[] = {&linear_graph};");
        c.line("uint32_t numModels = 1;");
        c.line("VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);");
        c.line("*numGraphsInfo = numModels;");
        c.blank();
        c.line("return err;");
    });
    c.line("}");
    c.blank();
    c.line("QNN_API");
    c.line("ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo) {");
    c.block(|c| {
        c.line("return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);");
    });
    c.line("}");
    c.blank();
    c.line("}  // extern \"C\"");
    c.into_string()
}

/// `qnn_model.cpp` — Linear with STATIC weight/bias (activation-only input).
fn emit_linear_static_qnn_model_cpp(m: usize, k: usize, n: usize, w: &[f32], b: &[f32]) -> String {
    let mut c = Cpp::new();
    c.banner("Generated by rlx-qnn — QNN AI Engine Direct linear (STATIC weights).");
    c.comment("out[M,N] = in0[M,K] * W[K,N] + b[M,N]; W/b baked as QNN_TENSOR_TYPE_STATIC.");
    c.comment("Composed with the qnn_wrapper_api surface (the qnn-onnx-converter shape).");
    c.comment(
        "Build:  qnn-model-lib-generator -c qnn_model.cpp -t x86_64-linux-clang -o model_libs",
    );
    c.comment(
        "Run:    qnn-net-run --backend libQnnCpu.so --model libqnn_model.so ...  (see run_qnn.sh)",
    );
    c.blank();
    c.line("#include \"QnnModel.hpp\"");
    c.line("#include \"QnnOpDef.h\"");
    c.blank();
    c.comment("Have the backend validate each op node as it is added.");
    c.line("#define DO_GRAPH_NODE_VALIDATIONS 1");
    c.blank();
    c.line("using namespace qnn_wrapper_api;");
    c.blank();
    c.comment("Baked STATIC weight [K,N] and bias [M,N] (seed-0 LCG; see verify.py).");
    c.raw(&format_c_float_array("w_data", w));
    c.raw(&format_c_float_array("b_data", b));
    c.blank();
    c.line("extern \"C\" {");
    c.blank();
    c.line("QNN_API");
    c.line("ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,");
    c.line("                                    QNN_INTERFACE_VER_TYPE interface,");
    c.line("                                    Qnn_ContextHandle_t contextHandle,");
    c.line("                                    const GraphConfigInfo_t** graphsConfigInfo,");
    c.line("                                    const uint32_t numGraphsConfigInfo,");
    c.line("                                    GraphInfoPtr_t** graphsInfo,");
    c.line("                                    uint32_t* numGraphsInfo,");
    c.line("                                    bool debug,");
    c.line("                                    QnnLog_Callback_t logCallback,");
    c.line("                                    QnnLog_Level_t maxLogLevel) {");
    c.block(|c| {
        c.line("ModelError_t err = MODEL_NO_ERROR;");
        c.blank();
        c.line("QnnModel linear_static_graph;");
        c.line("const QnnGraph_Config_t** graphConfigs = nullptr;");
        c.line("VALIDATE(getQnnGraphConfigFromInfo(\"linear_static_graph\", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);");
        c.line("VALIDATE(linear_static_graph.initialize(backendHandle, interface, contextHandle, \"linear_static_graph\", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);");
        c.blank();
        c.comment("graph input in0 [M, K] (activation)");
        c.line(&format!("uint32_t dimensions_in0[] = {{{m}, {k}}};"));
        c.tensor_v1(
            "VALIDATE(linear_static_graph.addTensor(\"in0\", ",
            "in0",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in0",
            "), err);",
        );
        c.blank();
        c.comment("STATIC weight W [K, N]");
        c.line(&format!("uint32_t dimensions_w[] = {{{k}, {n}}};"));
        c.tensor_v1_buf(
            "VALIDATE(linear_static_graph.addTensor(\"w\", ",
            "w",
            "QNN_TENSOR_TYPE_STATIC",
            "dimensions_w",
            "w_data",
            "sizeof(w_data)",
            "), err);",
        );
        c.blank();
        c.comment("STATIC bias b [M, N]");
        c.line(&format!("uint32_t dimensions_b[] = {{{m}, {n}}};"));
        c.tensor_v1_buf(
            "VALIDATE(linear_static_graph.addTensor(\"b\", ",
            "b",
            "QNN_TENSOR_TYPE_STATIC",
            "dimensions_b",
            "b_data",
            "sizeof(b_data)",
            "), err);",
        );
        c.blank();
        c.comment("MatMul node: mm[M, N] = in0 * w");
        c.line("const char* inputs_matmul_0[] = {\"in0\", \"w\"};");
        c.line(&format!("uint32_t dimensions_mm[] = {{{m}, {n}}};"));
        c.line("Qnn_Tensor_t outputs_matmul_0[] = {");
        c.block(|c| {
            c.tensor_v1("", "mm", "QNN_TENSOR_TYPE_NATIVE", "dimensions_mm", "");
        });
        c.line("};");
        c.line("Qnn_Param_t params_matmul_0[] = {");
        c.block(|c| {
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}},");
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}");
        });
        c.line("};");
        c.line("VALIDATE(linear_static_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"matmul_0\",                    // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("QNN_OP_MAT_MUL,                // QNN op type");
            c.line("params_matmul_0, 2,            // params + count");
            c.line("inputs_matmul_0, 2,            // input tensor names + count");
            c.line("outputs_matmul_0, 1            // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("ElementWiseAdd node: out[M, N] = mm + b");
        c.line("const char* inputs_add_0[] = {\"mm\", \"b\"};");
        c.line(&format!("uint32_t dimensions_out[] = {{{m}, {n}}};"));
        c.line("Qnn_Tensor_t outputs_add_0[] = {");
        c.block(|c| {
            c.tensor_v1("", "out", "QNN_TENSOR_TYPE_APP_READ", "dimensions_out", "");
        });
        c.line("};");
        c.line("VALIDATE(linear_static_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"add_0\",                       // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("\"ElementWiseAdd\",            // QNN op type");
            c.line("nullptr, 0,                    // params + count");
            c.line("inputs_add_0, 2,               // input tensor names + count");
            c.line("outputs_add_0, 1               // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("Collect the constructed graph into the output variables. The");
        c.comment("graph is finalized later by the runtime (e.g. qnn-net-run).");
        c.line("QnnModel* models[] = {&linear_static_graph};");
        c.line("uint32_t numModels = 1;");
        c.line("VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);");
        c.line("*numGraphsInfo = numModels;");
        c.blank();
        c.line("return err;");
    });
    c.line("}");
    c.blank();
    c.line("QNN_API");
    c.line("ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo) {");
    c.block(|c| {
        c.line("return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);");
    });
    c.line("}");
    c.blank();
    c.line("}  // extern \"C\"");
    c.into_string()
}

/// `qnn_model.cpp` — three-node graph: MatMul → ElementWiseAdd → Relu.
fn emit_linear_relu_qnn_model_cpp(m: usize, k: usize, n: usize) -> String {
    let mut c = Cpp::new();
    c.banner("Generated by rlx-qnn — QNN AI Engine Direct linear+relu graph.");
    c.comment("out[M,N] = relu(in0[M,K] * in1[K,N] + in2[M,N]), row-major f32.");
    c.comment("Composed with the qnn_wrapper_api surface (the qnn-onnx-converter shape).");
    c.comment(
        "Build:  qnn-model-lib-generator -c qnn_model.cpp -t x86_64-linux-clang -o model_libs",
    );
    c.comment(
        "Run:    qnn-net-run --backend libQnnCpu.so --model libqnn_model.so ...  (see run_qnn.sh)",
    );
    c.blank();
    c.line("#include \"QnnModel.hpp\"");
    c.line("#include \"QnnOpDef.h\"");
    c.blank();
    c.comment("Have the backend validate each op node as it is added.");
    c.line("#define DO_GRAPH_NODE_VALIDATIONS 1");
    c.blank();
    c.line("using namespace qnn_wrapper_api;");
    c.blank();
    c.line("extern \"C\" {");
    c.blank();
    c.line("QNN_API");
    c.line("ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,");
    c.line("                                    QNN_INTERFACE_VER_TYPE interface,");
    c.line("                                    Qnn_ContextHandle_t contextHandle,");
    c.line("                                    const GraphConfigInfo_t** graphsConfigInfo,");
    c.line("                                    const uint32_t numGraphsConfigInfo,");
    c.line("                                    GraphInfoPtr_t** graphsInfo,");
    c.line("                                    uint32_t* numGraphsInfo,");
    c.line("                                    bool debug,");
    c.line("                                    QnnLog_Callback_t logCallback,");
    c.line("                                    QnnLog_Level_t maxLogLevel) {");
    c.block(|c| {
        c.line("ModelError_t err = MODEL_NO_ERROR;");
        c.blank();
        c.line("QnnModel linear_relu_graph;");
        c.line("const QnnGraph_Config_t** graphConfigs = nullptr;");
        c.line("VALIDATE(getQnnGraphConfigFromInfo(\"linear_relu_graph\", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);");
        c.line("VALIDATE(linear_relu_graph.initialize(backendHandle, interface, contextHandle, \"linear_relu_graph\", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);");
        c.blank();
        c.comment("graph input in0 [M, K]");
        c.line(&format!("uint32_t dimensions_in0[] = {{{m}, {k}}};"));
        c.tensor_v1(
            "VALIDATE(linear_relu_graph.addTensor(\"in0\", ",
            "in0",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in0",
            "), err);",
        );
        c.blank();
        c.comment("graph input in1 [K, N]");
        c.line(&format!("uint32_t dimensions_in1[] = {{{k}, {n}}};"));
        c.tensor_v1(
            "VALIDATE(linear_relu_graph.addTensor(\"in1\", ",
            "in1",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in1",
            "), err);",
        );
        c.blank();
        c.comment("graph input in2 [M, N] (bias / addend)");
        c.line(&format!("uint32_t dimensions_in2[] = {{{m}, {n}}};"));
        c.tensor_v1(
            "VALIDATE(linear_relu_graph.addTensor(\"in2\", ",
            "in2",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in2",
            "), err);",
        );
        c.blank();
        c.comment("MatMul node: mm[M, N] = in0 * in1  (no transpose)");
        c.line("const char* inputs_matmul_0[] = {\"in0\", \"in1\"};");
        c.line(&format!("uint32_t dimensions_mm[] = {{{m}, {n}}};"));
        c.line("Qnn_Tensor_t outputs_matmul_0[] = {");
        c.block(|c| {
            c.tensor_v1("", "mm", "QNN_TENSOR_TYPE_NATIVE", "dimensions_mm", "");
        });
        c.line("};");
        c.line("Qnn_Param_t params_matmul_0[] = {");
        c.block(|c| {
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}},");
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}");
        });
        c.line("};");
        c.line("VALIDATE(linear_relu_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"matmul_0\",                    // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("QNN_OP_MAT_MUL,                // QNN op type");
            c.line("params_matmul_0, 2,            // params + count");
            c.line("inputs_matmul_0, 2,            // input tensor names + count");
            c.line("outputs_matmul_0, 1            // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("ElementWiseAdd node: add[M, N] = mm + in2");
        c.line("const char* inputs_add_0[] = {\"mm\", \"in2\"};");
        c.line(&format!("uint32_t dimensions_add[] = {{{m}, {n}}};"));
        c.line("Qnn_Tensor_t outputs_add_0[] = {");
        c.block(|c| {
            c.tensor_v1("", "add", "QNN_TENSOR_TYPE_NATIVE", "dimensions_add", "");
        });
        c.line("};");
        c.line("VALIDATE(linear_relu_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"add_0\",                       // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("\"ElementWiseAdd\",            // QNN op type");
            c.line("nullptr, 0,                    // params + count");
            c.line("inputs_add_0, 2,               // input tensor names + count");
            c.line("outputs_add_0, 1               // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("Relu node: out[M, N] = relu(add)");
        c.line("const char* inputs_relu_0[] = {\"add\"};");
        c.line(&format!("uint32_t dimensions_out[] = {{{m}, {n}}};"));
        c.line("Qnn_Tensor_t outputs_relu_0[] = {");
        c.block(|c| {
            c.tensor_v1("", "out", "QNN_TENSOR_TYPE_APP_READ", "dimensions_out", "");
        });
        c.line("};");
        c.line("VALIDATE(linear_relu_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"relu_0\",                      // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("\"Relu\",                        // QNN op type");
            c.line("nullptr, 0,                    // params + count");
            c.line("inputs_relu_0, 1,              // input tensor names + count");
            c.line("outputs_relu_0, 1              // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("Collect the constructed graph into the output variables. The");
        c.comment("graph is finalized later by the runtime (e.g. qnn-net-run).");
        c.line("QnnModel* models[] = {&linear_relu_graph};");
        c.line("uint32_t numModels = 1;");
        c.line("VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);");
        c.line("*numGraphsInfo = numModels;");
        c.blank();
        c.line("return err;");
    });
    c.line("}");
    c.blank();
    c.line("QNN_API");
    c.line("ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo) {");
    c.block(|c| {
        c.line("return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);");
    });
    c.line("}");
    c.blank();
    c.line("}  // extern \"C\"");
    c.into_string()
}

/// `qnn_model.cpp` — two-node graph: MatMul → Softmax(axis=1).
fn emit_matmul_softmax_qnn_model_cpp(m: usize, k: usize, n: usize) -> String {
    let mut c = Cpp::new();
    c.banner("Generated by rlx-qnn — QNN AI Engine Direct matmul+softmax graph.");
    c.comment("out[M,N] = softmax(in0[M,K] * in1[K,N], axis=1), row-major f32.");
    c.comment("Composed with the qnn_wrapper_api surface (the qnn-onnx-converter shape).");
    c.comment(
        "Build:  qnn-model-lib-generator -c qnn_model.cpp -t x86_64-linux-clang -o model_libs",
    );
    c.comment(
        "Run:    qnn-net-run --backend libQnnCpu.so --model libqnn_model.so ...  (see run_qnn.sh)",
    );
    c.blank();
    c.line("#include \"QnnModel.hpp\"");
    c.line("#include \"QnnOpDef.h\"");
    c.blank();
    c.comment("Have the backend validate each op node as it is added.");
    c.line("#define DO_GRAPH_NODE_VALIDATIONS 1");
    c.blank();
    c.line("using namespace qnn_wrapper_api;");
    c.blank();
    c.line("extern \"C\" {");
    c.blank();
    c.line("QNN_API");
    c.line("ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,");
    c.line("                                    QNN_INTERFACE_VER_TYPE interface,");
    c.line("                                    Qnn_ContextHandle_t contextHandle,");
    c.line("                                    const GraphConfigInfo_t** graphsConfigInfo,");
    c.line("                                    const uint32_t numGraphsConfigInfo,");
    c.line("                                    GraphInfoPtr_t** graphsInfo,");
    c.line("                                    uint32_t* numGraphsInfo,");
    c.line("                                    bool debug,");
    c.line("                                    QnnLog_Callback_t logCallback,");
    c.line("                                    QnnLog_Level_t maxLogLevel) {");
    c.block(|c| {
        c.line("ModelError_t err = MODEL_NO_ERROR;");
        c.blank();
        c.line("QnnModel matmul_softmax_graph;");
        c.line("const QnnGraph_Config_t** graphConfigs = nullptr;");
        c.line("VALIDATE(getQnnGraphConfigFromInfo(\"matmul_softmax_graph\", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);");
        c.line("VALIDATE(matmul_softmax_graph.initialize(backendHandle, interface, contextHandle, \"matmul_softmax_graph\", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);");
        c.blank();
        c.comment("graph input in0 [M, K]");
        c.line(&format!("uint32_t dimensions_in0[] = {{{m}, {k}}};"));
        c.tensor_v1(
            "VALIDATE(matmul_softmax_graph.addTensor(\"in0\", ",
            "in0",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in0",
            "), err);",
        );
        c.blank();
        c.comment("graph input in1 [K, N]");
        c.line(&format!("uint32_t dimensions_in1[] = {{{k}, {n}}};"));
        c.tensor_v1(
            "VALIDATE(matmul_softmax_graph.addTensor(\"in1\", ",
            "in1",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in1",
            "), err);",
        );
        c.blank();
        c.comment("MatMul node: mm[M, N] = in0 * in1  (no transpose)");
        c.line("const char* inputs_matmul_0[] = {\"in0\", \"in1\"};");
        c.line(&format!("uint32_t dimensions_mm[] = {{{m}, {n}}};"));
        c.line("Qnn_Tensor_t outputs_matmul_0[] = {");
        c.block(|c| {
            c.tensor_v1("", "mm", "QNN_TENSOR_TYPE_NATIVE", "dimensions_mm", "");
        });
        c.line("};");
        c.line("Qnn_Param_t params_matmul_0[] = {");
        c.block(|c| {
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}},");
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}");
        });
        c.line("};");
        c.line("VALIDATE(matmul_softmax_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"matmul_0\",                    // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("QNN_OP_MAT_MUL,                // QNN op type");
            c.line("params_matmul_0, 2,            // params + count");
            c.line("inputs_matmul_0, 2,            // input tensor names + count");
            c.line("outputs_matmul_0, 1            // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("Softmax node: out[M, N] = softmax(mm, axis=1)");
        c.line("const char* inputs_softmax_0[] = {\"mm\"};");
        c.line(&format!("uint32_t dimensions_out[] = {{{m}, {n}}};"));
        c.line("Qnn_Tensor_t outputs_softmax_0[] = {");
        c.block(|c| {
            c.tensor_v1("", "out", "QNN_TENSOR_TYPE_APP_READ", "dimensions_out", "");
        });
        c.line("};");
        c.line("Qnn_Param_t params_softmax_0[] = {");
        c.block(|c| {
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_SOFTMAX_PARAM_AXIS,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}");
        });
        c.line("};");
        c.line("VALIDATE(matmul_softmax_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"softmax_0\",                   // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("QNN_OP_SOFTMAX,                // QNN op type");
            c.line("params_softmax_0, 1,           // params + count");
            c.line("inputs_softmax_0, 1,           // input tensor names + count");
            c.line("outputs_softmax_0, 1           // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("Collect the constructed graph into the output variables. The");
        c.comment("graph is finalized later by the runtime (e.g. qnn-net-run).");
        c.line("QnnModel* models[] = {&matmul_softmax_graph};");
        c.line("uint32_t numModels = 1;");
        c.line("VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);");
        c.line("*numGraphsInfo = numModels;");
        c.blank();
        c.line("return err;");
    });
    c.line("}");
    c.blank();
    c.line("QNN_API");
    c.line("ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo) {");
    c.block(|c| {
        c.line("return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);");
    });
    c.line("}");
    c.blank();
    c.line("}  // extern \"C\"");
    c.into_string()
}

/// `qnn_model.cpp` — five-node MLP: MatMul→Add→Relu→MatMul→Add.
fn emit_mlp2_qnn_model_cpp(m: usize, k: usize, h: usize, n: usize) -> String {
    let mut c = Cpp::new();
    c.banner("Generated by rlx-qnn — QNN AI Engine Direct two-layer MLP graph.");
    c.comment("out[M,N] = relu(in0[M,K]*in1[K,H]+in2[M,H])*in3[H,N]+in4[M,N], f32.");
    c.comment("Composed with the qnn_wrapper_api surface (the qnn-onnx-converter shape).");
    c.comment(
        "Build:  qnn-model-lib-generator -c qnn_model.cpp -t x86_64-linux-clang -o model_libs",
    );
    c.comment(
        "Run:    qnn-net-run --backend libQnnCpu.so --model libqnn_model.so ...  (see run_qnn.sh)",
    );
    c.blank();
    c.line("#include \"QnnModel.hpp\"");
    c.line("#include \"QnnOpDef.h\"");
    c.blank();
    c.comment("Have the backend validate each op node as it is added.");
    c.line("#define DO_GRAPH_NODE_VALIDATIONS 1");
    c.blank();
    c.line("using namespace qnn_wrapper_api;");
    c.blank();
    c.line("extern \"C\" {");
    c.blank();
    c.line("QNN_API");
    c.line("ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,");
    c.line("                                    QNN_INTERFACE_VER_TYPE interface,");
    c.line("                                    Qnn_ContextHandle_t contextHandle,");
    c.line("                                    const GraphConfigInfo_t** graphsConfigInfo,");
    c.line("                                    const uint32_t numGraphsConfigInfo,");
    c.line("                                    GraphInfoPtr_t** graphsInfo,");
    c.line("                                    uint32_t* numGraphsInfo,");
    c.line("                                    bool debug,");
    c.line("                                    QnnLog_Callback_t logCallback,");
    c.line("                                    QnnLog_Level_t maxLogLevel) {");
    c.block(|c| {
        c.line("ModelError_t err = MODEL_NO_ERROR;");
        c.blank();
        c.line("QnnModel mlp2_graph;");
        c.line("const QnnGraph_Config_t** graphConfigs = nullptr;");
        c.line("VALIDATE(getQnnGraphConfigFromInfo(\"mlp2_graph\", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);");
        c.line("VALIDATE(mlp2_graph.initialize(backendHandle, interface, contextHandle, \"mlp2_graph\", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);");
        c.blank();
        c.comment("graph input in0 [M, K]");
        c.line(&format!("uint32_t dimensions_in0[] = {{{m}, {k}}};"));
        c.tensor_v1(
            "VALIDATE(mlp2_graph.addTensor(\"in0\", ",
            "in0",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in0",
            "), err);",
        );
        c.blank();
        c.comment("graph input in1 [K, H] (w1)");
        c.line(&format!("uint32_t dimensions_in1[] = {{{k}, {h}}};"));
        c.tensor_v1(
            "VALIDATE(mlp2_graph.addTensor(\"in1\", ",
            "in1",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in1",
            "), err);",
        );
        c.blank();
        c.comment("graph input in2 [M, H] (b1)");
        c.line(&format!("uint32_t dimensions_in2[] = {{{m}, {h}}};"));
        c.tensor_v1(
            "VALIDATE(mlp2_graph.addTensor(\"in2\", ",
            "in2",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in2",
            "), err);",
        );
        c.blank();
        c.comment("graph input in3 [H, N] (w2)");
        c.line(&format!("uint32_t dimensions_in3[] = {{{h}, {n}}};"));
        c.tensor_v1(
            "VALIDATE(mlp2_graph.addTensor(\"in3\", ",
            "in3",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in3",
            "), err);",
        );
        c.blank();
        c.comment("graph input in4 [M, N] (b2)");
        c.line(&format!("uint32_t dimensions_in4[] = {{{m}, {n}}};"));
        c.tensor_v1(
            "VALIDATE(mlp2_graph.addTensor(\"in4\", ",
            "in4",
            "QNN_TENSOR_TYPE_APP_WRITE",
            "dimensions_in4",
            "), err);",
        );
        c.blank();
        c.comment("MatMul node: mm0[M, H] = in0 * in1");
        c.line("const char* inputs_matmul_0[] = {\"in0\", \"in1\"};");
        c.line(&format!("uint32_t dimensions_mm0[] = {{{m}, {h}}};"));
        c.line("Qnn_Tensor_t outputs_matmul_0[] = {");
        c.block(|c| {
            c.tensor_v1("", "mm0", "QNN_TENSOR_TYPE_NATIVE", "dimensions_mm0", "");
        });
        c.line("};");
        c.line("Qnn_Param_t params_matmul_0[] = {");
        c.block(|c| {
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}},");
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}");
        });
        c.line("};");
        c.line("VALIDATE(mlp2_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"matmul_0\",                    // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("QNN_OP_MAT_MUL,                // QNN op type");
            c.line("params_matmul_0, 2,            // params + count");
            c.line("inputs_matmul_0, 2,            // input tensor names + count");
            c.line("outputs_matmul_0, 1            // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("ElementWiseAdd node: add0[M, H] = mm0 + in2");
        c.line("const char* inputs_add_0[] = {\"mm0\", \"in2\"};");
        c.line(&format!("uint32_t dimensions_add0[] = {{{m}, {h}}};"));
        c.line("Qnn_Tensor_t outputs_add_0[] = {");
        c.block(|c| {
            c.tensor_v1("", "add0", "QNN_TENSOR_TYPE_NATIVE", "dimensions_add0", "");
        });
        c.line("};");
        c.line("VALIDATE(mlp2_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"add_0\",                       // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("\"ElementWiseAdd\",            // QNN op type");
            c.line("nullptr, 0,                    // params + count");
            c.line("inputs_add_0, 2,               // input tensor names + count");
            c.line("outputs_add_0, 1               // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("Relu node: relu0[M, H] = relu(add0)");
        c.line("const char* inputs_relu_0[] = {\"add0\"};");
        c.line(&format!("uint32_t dimensions_relu0[] = {{{m}, {h}}};"));
        c.line("Qnn_Tensor_t outputs_relu_0[] = {");
        c.block(|c| {
            c.tensor_v1("", "relu0", "QNN_TENSOR_TYPE_NATIVE", "dimensions_relu0", "");
        });
        c.line("};");
        c.line("VALIDATE(mlp2_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"relu_0\",                      // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("\"Relu\",                        // QNN op type");
            c.line("nullptr, 0,                    // params + count");
            c.line("inputs_relu_0, 1,              // input tensor names + count");
            c.line("outputs_relu_0, 1              // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("MatMul node: mm1[M, N] = relu0 * in3");
        c.line("const char* inputs_matmul_1[] = {\"relu0\", \"in3\"};");
        c.line(&format!("uint32_t dimensions_mm1[] = {{{m}, {n}}};"));
        c.line("Qnn_Tensor_t outputs_matmul_1[] = {");
        c.block(|c| {
            c.tensor_v1("", "mm1", "QNN_TENSOR_TYPE_NATIVE", "dimensions_mm1", "");
        });
        c.line("};");
        c.line("Qnn_Param_t params_matmul_1[] = {");
        c.block(|c| {
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}},");
            c.line("{.paramType = QNN_PARAMTYPE_SCALAR, .name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,");
            c.line(" .scalarParam = {.dataType = QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}");
        });
        c.line("};");
        c.line("VALIDATE(mlp2_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"matmul_1\",                    // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("QNN_OP_MAT_MUL,                // QNN op type");
            c.line("params_matmul_1, 2,            // params + count");
            c.line("inputs_matmul_1, 2,            // input tensor names + count");
            c.line("outputs_matmul_1, 1            // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("ElementWiseAdd node: out[M, N] = mm1 + in4");
        c.line("const char* inputs_add_1[] = {\"mm1\", \"in4\"};");
        c.line(&format!("uint32_t dimensions_out[] = {{{m}, {n}}};"));
        c.line("Qnn_Tensor_t outputs_add_1[] = {");
        c.block(|c| {
            c.tensor_v1("", "out", "QNN_TENSOR_TYPE_APP_READ", "dimensions_out", "");
        });
        c.line("};");
        c.line("VALIDATE(mlp2_graph.addNode(");
        c.block(|c| {
            c.line("QNN_OPCONFIG_VERSION_1,        // Op_Config version");
            c.line("\"add_1\",                       // node name");
            c.line("QNN_OP_PACKAGE_NAME_QTI_AISW,  // op package");
            c.line("\"ElementWiseAdd\",            // QNN op type");
            c.line("nullptr, 0,                    // params + count");
            c.line("inputs_add_1, 2,               // input tensor names + count");
            c.line("outputs_add_1, 1               // output tensors + count");
        });
        c.line("), err);");
        c.blank();
        c.comment("Collect the constructed graph into the output variables. The");
        c.comment("graph is finalized later by the runtime (e.g. qnn-net-run).");
        c.line("QnnModel* models[] = {&mlp2_graph};");
        c.line("uint32_t numModels = 1;");
        c.line("VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);");
        c.line("*numGraphsInfo = numModels;");
        c.blank();
        c.line("return err;");
    });
    c.line("}");
    c.blank();
    c.line("QNN_API");
    c.line("ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo) {");
    c.block(|c| {
        c.line("return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);");
    });
    c.line("}");
    c.blank();
    c.line("}  // extern \"C\"");
    c.into_string()
}

/// `verify.py` for Linear — `out = in0 @ in1 + in2`.
fn emit_linear_verify_py(m: usize, k: usize, n: usize) -> String {
    format!(
        r#"#!/usr/bin/env python3
# Generated by rlx-qnn. Host harness for the emitted QNN linear model.
#
#   python3 verify.py --gen     # write in0.raw, in1.raw, in2.raw, input_list.txt, expected.raw
#   python3 verify.py --check   # compare qnn-net-run output vs numpy
#
# run_qnn.sh calls both around qnn-model-lib-generator + qnn-net-run.

import argparse
import numpy as np

M, K, N = {m}, {k}, {n}


def gen():
    rng = np.random.default_rng(0)
    in0 = rng.standard_normal((M, K)).astype(np.float32)
    in1 = rng.standard_normal((K, N)).astype(np.float32)
    in2 = rng.standard_normal((M, N)).astype(np.float32)
    in0.tofile("in0.raw")
    in1.tofile("in1.raw")
    in2.tofile("in2.raw")
    (in0 @ in1 + in2).astype(np.float32).tofile("expected.raw")
    with open("input_list.txt", "w") as f:
        f.write("in0:=in0.raw in1:=in1.raw in2:=in2.raw\n")
    print("wrote in0.raw, in1.raw, in2.raw, input_list.txt, expected.raw")


def check():
    out = np.fromfile("output/Result_0/out.raw", dtype=np.float32).reshape(M, N)
    expected = np.fromfile("expected.raw", dtype=np.float32).reshape(M, N)
    np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)
    print("SUCCESS!")


parser = argparse.ArgumentParser()
parser.add_argument("--gen", action="store_true", help="write inputs + expected")
parser.add_argument("--check", action="store_true", help="check qnn-net-run output")
args = parser.parse_args()
if args.gen:
    gen()
elif args.check:
    check()
else:
    parser.error("pass --gen or --check")
"#
    )
}

/// `verify.py` for LinearStatic — only `in0` is a runtime input; W/b match C++.
fn emit_linear_static_verify_py(m: usize, k: usize, n: usize, w: &[f32], b: &[f32]) -> String {
    let w_list = format_py_float_list(w);
    let b_list = format_py_float_list(b);
    format!(
        r#"#!/usr/bin/env python3
# Generated by rlx-qnn. Host harness for LinearStatic (STATIC W/b).
#
#   python3 verify.py --gen     # write in0.raw, input_list.txt, expected.raw
#   python3 verify.py --check   # compare qnn-net-run output vs numpy
#
# run_qnn.sh calls both around qnn-model-lib-generator + qnn-net-run.

import argparse
import numpy as np

M, K, N = {m}, {k}, {n}
W = np.array({w_list}, dtype=np.float32).reshape(K, N)
B = np.array({b_list}, dtype=np.float32).reshape(M, N)


def gen():
    rng = np.random.default_rng(0)
    in0 = rng.standard_normal((M, K)).astype(np.float32)
    in0.tofile("in0.raw")
    (in0 @ W + B).astype(np.float32).tofile("expected.raw")
    with open("input_list.txt", "w") as f:
        f.write("in0:=in0.raw\n")
    print("wrote in0.raw, input_list.txt, expected.raw")


def check():
    out = np.fromfile("output/Result_0/out.raw", dtype=np.float32).reshape(M, N)
    expected = np.fromfile("expected.raw", dtype=np.float32).reshape(M, N)
    np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)
    print("SUCCESS!")


parser = argparse.ArgumentParser()
parser.add_argument("--gen", action="store_true", help="write inputs + expected")
parser.add_argument("--check", action="store_true", help="check qnn-net-run output")
args = parser.parse_args()
if args.gen:
    gen()
elif args.check:
    check()
else:
    parser.error("pass --gen or --check")
"#
    )
}

/// `verify.py` for LinearRelu — `out = maximum(in0 @ in1 + in2, 0)`.
fn emit_linear_relu_verify_py(m: usize, k: usize, n: usize) -> String {
    format!(
        r#"#!/usr/bin/env python3
# Generated by rlx-qnn. Host harness for the emitted QNN linear+relu model.
#
#   python3 verify.py --gen     # write in0.raw, in1.raw, in2.raw, input_list.txt, expected.raw
#   python3 verify.py --check   # compare qnn-net-run output vs numpy
#
# run_qnn.sh calls both around qnn-model-lib-generator + qnn-net-run.

import argparse
import numpy as np

M, K, N = {m}, {k}, {n}


def gen():
    rng = np.random.default_rng(0)
    in0 = rng.standard_normal((M, K)).astype(np.float32)
    in1 = rng.standard_normal((K, N)).astype(np.float32)
    in2 = rng.standard_normal((M, N)).astype(np.float32)
    in0.tofile("in0.raw")
    in1.tofile("in1.raw")
    in2.tofile("in2.raw")
    np.maximum(in0 @ in1 + in2, 0).astype(np.float32).tofile("expected.raw")
    with open("input_list.txt", "w") as f:
        f.write("in0:=in0.raw in1:=in1.raw in2:=in2.raw\n")
    print("wrote in0.raw, in1.raw, in2.raw, input_list.txt, expected.raw")


def check():
    out = np.fromfile("output/Result_0/out.raw", dtype=np.float32).reshape(M, N)
    expected = np.fromfile("expected.raw", dtype=np.float32).reshape(M, N)
    np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)
    print("SUCCESS!")


parser = argparse.ArgumentParser()
parser.add_argument("--gen", action="store_true", help="write inputs + expected")
parser.add_argument("--check", action="store_true", help="check qnn-net-run output")
args = parser.parse_args()
if args.gen:
    gen()
elif args.check:
    check()
else:
    parser.error("pass --gen or --check")
"#
    )
}

/// `verify.py` for MatMulSoftmax — `out = softmax(in0 @ in1, axis=1)`.
fn emit_matmul_softmax_verify_py(m: usize, k: usize, n: usize) -> String {
    format!(
        r#"#!/usr/bin/env python3
# Generated by rlx-qnn. Host harness for the emitted QNN matmul+softmax model.
#
#   python3 verify.py --gen     # write in0.raw, in1.raw, input_list.txt, expected.raw
#   python3 verify.py --check   # compare qnn-net-run output vs numpy
#
# run_qnn.sh calls both around qnn-model-lib-generator + qnn-net-run.

import argparse
import numpy as np

M, K, N = {m}, {k}, {n}


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def gen():
    rng = np.random.default_rng(0)
    in0 = rng.standard_normal((M, K)).astype(np.float32)
    in1 = rng.standard_normal((K, N)).astype(np.float32)
    in0.tofile("in0.raw")
    in1.tofile("in1.raw")
    softmax(in0 @ in1, axis=1).astype(np.float32).tofile("expected.raw")
    with open("input_list.txt", "w") as f:
        f.write("in0:=in0.raw in1:=in1.raw\n")
    print("wrote in0.raw, in1.raw, input_list.txt, expected.raw")


def check():
    out = np.fromfile("output/Result_0/out.raw", dtype=np.float32).reshape(M, N)
    expected = np.fromfile("expected.raw", dtype=np.float32).reshape(M, N)
    np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)
    print("SUCCESS!")


parser = argparse.ArgumentParser()
parser.add_argument("--gen", action="store_true", help="write inputs + expected")
parser.add_argument("--check", action="store_true", help="check qnn-net-run output")
args = parser.parse_args()
if args.gen:
    gen()
elif args.check:
    check()
else:
    parser.error("pass --gen or --check")
"#
    )
}

/// `verify.py` for Mlp2 — `out = relu(in0@in1+in2)@in3+in4`.
fn emit_mlp2_verify_py(m: usize, k: usize, h: usize, n: usize) -> String {
    format!(
        r#"#!/usr/bin/env python3
# Generated by rlx-qnn. Host harness for the emitted QNN two-layer MLP model.
#
#   python3 verify.py --gen     # write in0..in4.raw, input_list.txt, expected.raw
#   python3 verify.py --check   # compare qnn-net-run output vs numpy
#
# run_qnn.sh calls both around qnn-model-lib-generator + qnn-net-run.

import argparse
import numpy as np

M, K, H, N = {m}, {k}, {h}, {n}


def gen():
    rng = np.random.default_rng(0)
    in0 = rng.standard_normal((M, K)).astype(np.float32)
    in1 = rng.standard_normal((K, H)).astype(np.float32)
    in2 = rng.standard_normal((M, H)).astype(np.float32)
    in3 = rng.standard_normal((H, N)).astype(np.float32)
    in4 = rng.standard_normal((M, N)).astype(np.float32)
    in0.tofile("in0.raw")
    in1.tofile("in1.raw")
    in2.tofile("in2.raw")
    in3.tofile("in3.raw")
    in4.tofile("in4.raw")
    hidden = np.maximum(in0 @ in1 + in2, 0)
    (hidden @ in3 + in4).astype(np.float32).tofile("expected.raw")
    with open("input_list.txt", "w") as f:
        f.write("in0:=in0.raw in1:=in1.raw in2:=in2.raw in3:=in3.raw in4:=in4.raw\n")
    print("wrote in0.raw..in4.raw, input_list.txt, expected.raw")


def check():
    out = np.fromfile("output/Result_0/out.raw", dtype=np.float32).reshape(M, N)
    expected = np.fromfile("expected.raw", dtype=np.float32).reshape(M, N)
    np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)
    print("SUCCESS!")


parser = argparse.ArgumentParser()
parser.add_argument("--gen", action="store_true", help="write inputs + expected")
parser.add_argument("--check", action="store_true", help="check qnn-net-run output")
args = parser.parse_args()
if args.gen:
    gen()
elif args.check:
    check()
else:
    parser.error("pass --gen or --check")
"#
    )
}

/// `verify.py` — two modes. `--gen` writes `in0.raw` / `in1.raw` /
/// `input_list.txt` / `expected.raw` (numpy, seed 0); `--check` reshapes the
/// `qnn-net-run` output and asserts it against numpy. Validates the model on
/// the QNN x86 reference backend without Snapdragon hardware.
fn emit_verify_py(m: usize, k: usize, n: usize) -> String {
    format!(
        r#"#!/usr/bin/env python3
# Generated by rlx-qnn. Host harness for the emitted QNN matmul model.
#
#   python3 verify.py --gen     # write in0.raw, in1.raw, input_list.txt, expected.raw
#   python3 verify.py --check   # compare qnn-net-run output vs numpy in0 @ in1
#
# run_qnn.sh calls both around qnn-model-lib-generator + qnn-net-run.

import argparse
import numpy as np

M, K, N = {m}, {k}, {n}


def gen():
    rng = np.random.default_rng(0)
    in0 = rng.standard_normal((M, K)).astype(np.float32)
    in1 = rng.standard_normal((K, N)).astype(np.float32)
    in0.tofile("in0.raw")
    in1.tofile("in1.raw")
    (in0 @ in1).astype(np.float32).tofile("expected.raw")
    with open("input_list.txt", "w") as f:
        f.write("in0:=in0.raw in1:=in1.raw\n")
    print("wrote in0.raw, in1.raw, input_list.txt, expected.raw")


def check():
    out = np.fromfile("output/Result_0/out.raw", dtype=np.float32).reshape(M, N)
    expected = np.fromfile("expected.raw", dtype=np.float32).reshape(M, N)
    np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)
    print("SUCCESS!")


parser = argparse.ArgumentParser()
parser.add_argument("--gen", action="store_true", help="write inputs + expected")
parser.add_argument("--check", action="store_true", help="check qnn-net-run output")
args = parser.parse_args()
if args.gen:
    gen()
elif args.check:
    check()
else:
    parser.error("pass --gen or --check")
"#
    )
}

/// `run_qnn.sh` — generate inputs, build the model lib, run it on a QNN
/// backend, and check the output. Default backend is x86 `libQnnCpu.so`;
/// set `RLX_QNN_BACKEND_LIB` to `libQnnHtp.so` for the x86 HTP functional
/// simulator (no Snapdragon silicon required).
fn emit_run_qnn_sh() -> String {
    "#!/usr/bin/env bash\n\
     # Generated by rlx-qnn. Build + run on a QNN backend (default: libQnnCpu.so).\n\
     # HTP functional sim (x86): export RLX_QNN_BACKEND_LIB=$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so\n\
     set -e\n\
     \n\
     : \"${QNN_SDK_ROOT:?set QNN_SDK_ROOT to your Qualcomm AI Engine Direct SDK}\"\n\
     export PYTHONPATH=\"${PYTHONPATH:-}\"\n\
     # shellcheck disable=SC1090\n\
     source \"$QNN_SDK_ROOT/bin/envsetup.sh\"\n\
     \n\
     TARGET=x86_64-linux-clang\n\
     BACKEND=\"${RLX_QNN_BACKEND_LIB:-$QNN_SDK_ROOT/lib/$TARGET/libQnnCpu.so}\"\n\
     export LD_LIBRARY_PATH=\"$QNN_SDK_ROOT/lib/$TARGET${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"\n\
     \n\
     # 1. inputs + expected output (numpy, seed 0)\n\
     python3 verify.py --gen\n\
     \n\
     # 2. compile the composed graph into a shared library (lib<name>.so)\n\
     MODEL_LIB=qnn_model\n\
     qnn-model-lib-generator -c qnn_model.cpp -t $TARGET -l $MODEL_LIB -o model_libs\n\
     \n\
     # 3. run on the selected backend\n\
     qnn-net-run \\\n\
       --backend \"$BACKEND\" \\\n\
       --model model_libs/$TARGET/lib${MODEL_LIB}.so \\\n\
       --input_list input_list.txt \\\n\
       --output_dir output\n\
     \n\
     # 4. check device output against numpy\n\
     python3 verify.py --check\n\
     echo \"qnn-net-run OK (backend=$BACKEND)\"\n"
        .to_string()
}

/// `run_qnn_context.sh` — build the model lib, serialize a context `.bin` via
/// `qnn-context-binary-generator`, then execute with `qnn-net-run
/// --retrieve_context` (style-2 / M3 offline path). Same numpy check as
/// `run_qnn.sh`. Backend defaults to `libQnnCpu.so`; override with
/// `RLX_QNN_BACKEND_LIB` (e.g. x86 `libQnnHtp.so` functional simulator).
fn emit_run_qnn_context_sh() -> String {
    "#!/usr/bin/env bash\n\
     # Generated by rlx-qnn. Offline context-binary path (style-2):\n\
     #   qnn_model.cpp → libqnn_model.so → model.bin → qnn-net-run --retrieve_context\n\
     # Default backend: libQnnCpu.so. HTP x86 sim: set RLX_QNN_BACKEND_LIB to libQnnHtp.so.\n\
     set -e\n\
     \n\
     : \"${QNN_SDK_ROOT:?set QNN_SDK_ROOT to your Qualcomm AI Engine Direct SDK}\"\n\
     export PYTHONPATH=\"${PYTHONPATH:-}\"\n\
     # shellcheck disable=SC1090\n\
     source \"$QNN_SDK_ROOT/bin/envsetup.sh\"\n\
     \n\
     TARGET=x86_64-linux-clang\n\
     BACKEND=\"${RLX_QNN_BACKEND_LIB:-$QNN_SDK_ROOT/lib/$TARGET/libQnnCpu.so}\"\n\
     export LD_LIBRARY_PATH=\"$QNN_SDK_ROOT/lib/$TARGET${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\"\n\
     MODEL_LIB=qnn_model\n\
     MODEL_SO=\"model_libs/$TARGET/lib${MODEL_LIB}.so\"\n\
     # Tool appends .bin — pass stem only (model → model.bin).\n\
     CTX_STEM=model\n\
     CTX_BIN=${CTX_STEM}.bin\n\
     \n\
     # 1. inputs + expected output (numpy, seed 0)\n\
     python3 verify.py --gen\n\
     \n\
     # 2. compile the composed graph into a shared library\n\
     qnn-model-lib-generator -c qnn_model.cpp -t $TARGET -l $MODEL_LIB -o model_libs\n\
     \n\
     # 3. serialize a context binary from the model lib\n\
     qnn-context-binary-generator \\\n\
       --model \"$MODEL_SO\" \\\n\
       --backend \"$BACKEND\" \\\n\
       --binary_file \"$CTX_STEM\" \\\n\
       --output_dir .\n\
     \n\
     # 4. execute from the cached context (no --model)\n\
     rm -rf output\n\
     qnn-net-run \\\n\
       --backend \"$BACKEND\" \\\n\
       --retrieve_context \"$CTX_BIN\" \\\n\
       --input_list input_list.txt \\\n\
       --output_dir output\n\
     \n\
     # 5. check device output against numpy\n\
     python3 verify.py --check\n\
     echo \"context-binary path OK ($CTX_BIN, backend=$BACKEND)\"\n"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact<'a>(arts: &'a [Artifact], path: &str) -> &'a str {
        &arts
            .iter()
            .find(|a| a.path == path)
            .expect("artifact present")
            .contents
    }

    /// Net (open − close) for `()`, `{}`, `[]`, ignoring `//` comments and
    /// `"…"` string literals. `(0, 0, 0)` iff balanced. Guards the
    /// hand-written designated-initializer brace nesting in `cpp::tensor_v1`.
    fn delimiter_residual(src: &str) -> (i32, i32, i32) {
        let (mut paren, mut brace, mut brack) = (0i32, 0i32, 0i32);
        for line in src.lines() {
            let chars: Vec<char> = line.chars().collect();
            let mut in_string = false;
            let mut i = 0;
            while i < chars.len() {
                let c = chars[i];
                if in_string {
                    if c == '\\' {
                        i += 2;
                        continue;
                    }
                    if c == '"' {
                        in_string = false;
                    }
                    i += 1;
                    continue;
                }
                match c {
                    '"' => in_string = true,
                    '/' if i + 1 < chars.len() && chars[i + 1] == '/' => break,
                    '(' => paren += 1,
                    ')' => paren -= 1,
                    '{' => brace += 1,
                    '}' => brace -= 1,
                    '[' => brack += 1,
                    ']' => brack -= 1,
                    _ => {}
                }
                i += 1;
            }
        }
        (paren, brace, brack)
    }

    #[test]
    fn linear_qnn_model_cpp_is_delimiter_balanced() {
        for (m, k, n) in [(1, 1, 1), (3, 5, 7), (8, 16, 4), (32, 64, 32)] {
            let arts = collect_artifacts(&Model::linear("lin", m, k, n)).unwrap();
            let cpp = artifact(&arts, "qnn_model.cpp");
            assert_eq!(
                delimiter_residual(cpp),
                (0, 0, 0),
                "linear qnn_model.cpp unbalanced for {m}x{k}x{n}"
            );
        }
    }

    #[test]
    fn linear_model_carries_dims_and_nodes() {
        let model = Model::linear("lin", 3, 5, 7);
        let arts = collect_artifacts(&model).unwrap();
        let cpp = artifact(&arts, "qnn_model.cpp");
        assert!(cpp.contains("linear_graph"));
        assert!(cpp.contains("uint32_t dimensions_in0[] = {3, 5};"));
        assert!(cpp.contains("uint32_t dimensions_in1[] = {5, 7};"));
        assert!(cpp.contains("uint32_t dimensions_in2[] = {3, 7};"));
        assert!(cpp.contains("QNN_OP_MAT_MUL"));
        assert!(cpp.contains("\"ElementWiseAdd\""));
        assert!(!cpp.contains("\"Relu\""));
        assert!(cpp.contains("QNN_TENSOR_TYPE_NATIVE"));
        assert!(cpp.contains(".name           = \"mm\","));
        assert!(cpp.contains(".name           = \"out\","));
        assert!(!cpp.contains(".name           = \"add\","));
    }

    #[test]
    fn linear_verify_uses_dims_and_checks() {
        let model = Model::linear("lin", 4, 6, 8);
        let arts = collect_artifacts(&model).unwrap();
        let v = artifact(&arts, "verify.py");
        assert!(v.contains("M, K, N = 4, 6, 8"));
        assert!(v.contains("in0 @ in1 + in2"));
        assert!(v.contains("in2:=in2.raw"));
        assert!(!v.contains("np.maximum"));
    }

    #[test]
    fn linear_relu_qnn_model_cpp_is_delimiter_balanced() {
        for (m, k, n) in [(1, 1, 1), (3, 5, 7), (8, 16, 4), (32, 64, 32)] {
            let arts = collect_artifacts(&Model::linear_relu("lr", m, k, n)).unwrap();
            let cpp = artifact(&arts, "qnn_model.cpp");
            assert_eq!(
                delimiter_residual(cpp),
                (0, 0, 0),
                "linear_relu qnn_model.cpp unbalanced for {m}x{k}x{n}"
            );
        }
    }

    #[test]
    fn linear_relu_model_carries_dims_and_nodes() {
        let model = Model::linear_relu("lr", 3, 5, 7);
        let arts = collect_artifacts(&model).unwrap();
        let cpp = artifact(&arts, "qnn_model.cpp");
        assert!(cpp.contains("linear_relu_graph"));
        assert!(cpp.contains("uint32_t dimensions_in0[] = {3, 5};"));
        assert!(cpp.contains("uint32_t dimensions_in1[] = {5, 7};"));
        assert!(cpp.contains("uint32_t dimensions_in2[] = {3, 7};"));
        assert!(cpp.contains("QNN_OP_MAT_MUL"));
        assert!(cpp.contains("\"ElementWiseAdd\""));
        assert!(cpp.contains("\"Relu\""));
        assert!(cpp.contains("QNN_TENSOR_TYPE_NATIVE"));
        assert!(cpp.contains(".name           = \"mm\","));
        assert!(cpp.contains(".name           = \"add\","));
        assert!(cpp.contains(".name           = \"out\","));
    }

    #[test]
    fn linear_relu_verify_uses_dims_and_checks() {
        let model = Model::linear_relu("lr", 4, 6, 8);
        let arts = collect_artifacts(&model).unwrap();
        let v = artifact(&arts, "verify.py");
        assert!(v.contains("M, K, N = 4, 6, 8"));
        assert!(v.contains("np.maximum(in0 @ in1 + in2, 0)"));
        assert!(v.contains("in2:=in2.raw"));
    }

    #[test]
    fn matmul_softmax_qnn_model_cpp_is_delimiter_balanced() {
        for (m, k, n) in [(1, 1, 1), (3, 5, 7), (8, 16, 4), (32, 64, 32)] {
            let arts = collect_artifacts(&Model::matmul_softmax("ms", m, k, n)).unwrap();
            let cpp = artifact(&arts, "qnn_model.cpp");
            assert_eq!(
                delimiter_residual(cpp),
                (0, 0, 0),
                "matmul_softmax qnn_model.cpp unbalanced for {m}x{k}x{n}"
            );
        }
    }

    #[test]
    fn matmul_softmax_model_carries_dims_and_nodes() {
        let model = Model::matmul_softmax("ms", 3, 5, 7);
        let arts = collect_artifacts(&model).unwrap();
        let cpp = artifact(&arts, "qnn_model.cpp");
        assert!(cpp.contains("matmul_softmax_graph"));
        assert!(cpp.contains("uint32_t dimensions_in0[] = {3, 5};"));
        assert!(cpp.contains("uint32_t dimensions_in1[] = {5, 7};"));
        assert!(cpp.contains("QNN_OP_MAT_MUL"));
        assert!(cpp.contains("QNN_OP_SOFTMAX"));
        assert!(cpp.contains("QNN_OP_SOFTMAX_PARAM_AXIS"));
        assert!(cpp.contains("uint32Value = 1"));
        assert!(cpp.contains("QNN_TENSOR_TYPE_NATIVE"));
        assert!(cpp.contains(".name           = \"mm\","));
        assert!(cpp.contains(".name           = \"out\","));
    }

    #[test]
    fn matmul_softmax_verify_uses_dims_and_checks() {
        let model = Model::matmul_softmax("ms", 4, 6, 8);
        let arts = collect_artifacts(&model).unwrap();
        let v = artifact(&arts, "verify.py");
        assert!(v.contains("M, K, N = 4, 6, 8"));
        assert!(v.contains("softmax(in0 @ in1, axis=1)"));
        assert!(!v.contains("in2:="));
    }

    #[test]
    fn mlp2_qnn_model_cpp_is_delimiter_balanced() {
        for (m, k, h, n) in [(1, 1, 1, 1), (3, 5, 8, 4), (8, 16, 32, 4)] {
            let arts = collect_artifacts(&Model::mlp2("mlp", m, k, h, n)).unwrap();
            let cpp = artifact(&arts, "qnn_model.cpp");
            assert_eq!(
                delimiter_residual(cpp),
                (0, 0, 0),
                "mlp2 qnn_model.cpp unbalanced for {m}x{k}x{h}x{n}"
            );
        }
    }

    #[test]
    fn mlp2_model_carries_dims_and_nodes() {
        let model = Model::mlp2("mlp", 3, 5, 8, 4);
        let arts = collect_artifacts(&model).unwrap();
        let cpp = artifact(&arts, "qnn_model.cpp");
        assert!(cpp.contains("mlp2_graph"));
        assert!(cpp.contains("uint32_t dimensions_in0[] = {3, 5};"));
        assert!(cpp.contains("uint32_t dimensions_in1[] = {5, 8};"));
        assert!(cpp.contains("uint32_t dimensions_in2[] = {3, 8};"));
        assert!(cpp.contains("uint32_t dimensions_in3[] = {8, 4};"));
        assert!(cpp.contains("uint32_t dimensions_in4[] = {3, 4};"));
        assert!(cpp.contains("\"matmul_0\""));
        assert!(cpp.contains("\"matmul_1\""));
        assert!(cpp.contains("\"Relu\""));
        assert!(cpp.contains("\"ElementWiseAdd\""));
        assert!(cpp.contains(".name           = \"relu0\","));
        assert!(cpp.contains(".name           = \"out\","));
    }

    #[test]
    fn mlp2_verify_uses_dims_and_checks() {
        let model = Model::mlp2("mlp", 4, 6, 8, 3);
        let arts = collect_artifacts(&model).unwrap();
        let v = artifact(&arts, "verify.py");
        assert!(v.contains("M, K, H, N = 4, 6, 8, 3"));
        assert!(v.contains("np.maximum(in0 @ in1 + in2, 0)"));
        assert!(v.contains("hidden @ in3 + in4"));
        assert!(v.contains("in4:=in4.raw"));
    }

    #[test]
    fn linear_static_qnn_model_cpp_is_delimiter_balanced() {
        for (m, k, n) in [(1, 1, 1), (3, 5, 7), (8, 16, 4)] {
            let arts = collect_artifacts(&Model::linear_static("ls", m, k, n)).unwrap();
            let cpp = artifact(&arts, "qnn_model.cpp");
            assert_eq!(
                delimiter_residual(cpp),
                (0, 0, 0),
                "linear_static qnn_model.cpp unbalanced for {m}x{k}x{n}"
            );
        }
    }

    #[test]
    fn linear_static_bakes_weights_and_single_input() {
        let model = Model::linear_static("ls", 3, 5, 7);
        let arts = collect_artifacts(&model).unwrap();
        let cpp = artifact(&arts, "qnn_model.cpp");
        assert!(cpp.contains("linear_static_graph"));
        assert!(cpp.contains("static float w_data[]"));
        assert!(cpp.contains("static float b_data[]"));
        assert!(cpp.contains("QNN_TENSOR_TYPE_STATIC"));
        assert!(cpp.contains("QNN_TENSOR_TYPE_APP_WRITE"));
        assert!(cpp.contains(".name           = \"w\","));
        assert!(cpp.contains(".name           = \"b\","));
        assert!(cpp.contains("(void*)w_data"));
        let v = artifact(&arts, "verify.py");
        assert!(v.contains("in0:=in0.raw"));
        assert!(!v.contains("in1:="));
        assert!(v.contains("in0 @ W + B"));
        let ctx = artifact(&arts, "run_qnn_context.sh");
        assert!(ctx.contains("qnn-context-binary-generator"));
        assert!(ctx.contains("--retrieve_context"));
    }

    #[test]
    fn linear_static_custom_weights_appear_in_cpp() {
        let w = vec![1.25f32, -2.5, 3.75, 4.0, 5.5, 6.25]; // 3x2
        let b = vec![0.125f32, -0.25, 0.5, -0.75]; // 2x2
        let model = Model::linear_static_with_weights("ls", 2, 3, 2, w, b);
        let arts = collect_artifacts(&model).unwrap();
        let cpp = artifact(&arts, "qnn_model.cpp");
        assert!(cpp.contains("1.25000000e0f"), "cpp={cpp}");
        assert!(cpp.contains("-2.50000000e0f"));
        assert!(cpp.contains("1.25000000e-1f"));
        let v = artifact(&arts, "verify.py");
        assert!(v.contains("1.25000000e0"));
        assert!(v.contains("1.25000000e-1"));
    }

    #[test]
    fn qnn_model_cpp_is_delimiter_balanced() {
        // The MatMul tensor literals nest five braces each; a miscount silently
        // emits C++ that qnn-model-lib-generator rejects. Check several shapes.
        for (m, k, n) in [(1, 1, 1), (3, 5, 7), (8, 16, 4), (32, 64, 32)] {
            let arts = collect_artifacts(&Model::single_matmul("mm", m, k, n)).unwrap();
            let cpp = artifact(&arts, "qnn_model.cpp");
            assert_eq!(
                delimiter_residual(cpp),
                (0, 0, 0),
                "qnn_model.cpp unbalanced for {m}x{k}x{n}"
            );
        }
    }

    #[test]
    fn emits_full_artifact_set() {
        let model = Model::single_matmul("mm", 3, 5, 7);
        let arts = collect_artifacts(&model).unwrap();
        let paths: Vec<&str> = arts.iter().map(|a| a.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "qnn_model.cpp",
                "verify.py",
                "run_qnn.sh",
                "run_qnn_context.sh"
            ]
        );
    }

    #[test]
    fn qnn_model_carries_dims_and_node() {
        let model = Model::single_matmul("mm", 3, 5, 7);
        let arts = collect_artifacts(&model).unwrap();
        let cpp = artifact(&arts, "qnn_model.cpp");
        assert!(cpp.contains("QnnModel_composeGraphs"));
        assert!(cpp.contains("#define DO_GRAPH_NODE_VALIDATIONS 1"));
        assert!(cpp.contains("uint32_t dimensions_in0[] = {3, 5};"));
        assert!(cpp.contains("uint32_t dimensions_in1[] = {5, 7};"));
        assert!(cpp.contains("uint32_t dimensions_out[] = {3, 7};"));
        assert!(cpp.contains("QNN_OP_MAT_MUL"));
        assert!(cpp.contains("matmul_graph.addNode("));
        assert!(cpp.contains(".v1 = {"));
        assert!(cpp.contains(".name           = \"in0\","));
        // Real wrapper API has no freeze(); finalize happens in the runtime.
        assert!(!cpp.contains("freeze"));
        assert!(cpp.contains("getGraphInfoFromModels(*models, numModels, graphsInfo)"));
        assert!(cpp.contains("QnnModel_freeGraphsInfo"));
    }

    #[test]
    fn verify_uses_dims_and_checks() {
        let model = Model::single_matmul("mm", 4, 6, 8);
        let arts = collect_artifacts(&model).unwrap();
        let v = artifact(&arts, "verify.py");
        assert!(v.contains("M, K, N = 4, 6, 8"));
        assert!(v.contains("in0 @ in1"));
        assert!(v.contains("np.testing.assert_allclose"));
    }

    #[test]
    fn run_sh_invokes_qnn_tools() {
        let model = Model::single_matmul("mm", 2, 2, 2);
        let arts = collect_artifacts(&model).unwrap();
        let sh = artifact(&arts, "run_qnn.sh");
        assert!(sh.contains("qnn-model-lib-generator"));
        assert!(sh.contains("qnn-net-run"));
        assert!(sh.contains("libQnnCpu.so"));
        assert!(sh.contains("RLX_QNN_BACKEND_LIB"));
        let ctx = artifact(&arts, "run_qnn_context.sh");
        assert!(ctx.contains("qnn-context-binary-generator"));
        assert!(ctx.contains("--retrieve_context"));
        assert!(ctx.contains("model.bin"));
        assert!(ctx.contains("RLX_QNN_BACKEND_LIB"));
    }

    #[test]
    fn rejects_zero_dims() {
        let model = Model::single_matmul("bad", 0, 4, 4);
        assert!(collect_artifacts(&model).is_err());
    }

    #[test]
    fn emit_model_writes_files() {
        let dir = tempfile::tempdir().unwrap();
        let model = Model::single_matmul("mm", 2, 3, 4);
        emit_model(&model, dir.path()).unwrap();
        for f in [
            "qnn_model.cpp",
            "verify.py",
            "run_qnn.sh",
            "run_qnn_context.sh",
        ] {
            assert!(dir.path().join(f).exists(), "{f} written");
        }
    }

    #[test]
    fn linear_static_custom_weights_context_e2e() {
        // Offline context-binary path with caller-supplied Constants (not seed-0).
        let sdk = match std::env::var("QNN_SDK_ROOT") {
            Ok(s) if !s.is_empty() => s,
            _ => {
                eprintln!("skip: QNN_SDK_ROOT unset");
                return;
            }
        };
        let backend = std::path::Path::new(&sdk).join("lib/x86_64-linux-clang/libQnnCpu.so");
        if !backend.exists() {
            eprintln!("skip: no libQnnCpu.so");
            return;
        }

        let (m, k, n) = (2usize, 4, 3);
        let w: Vec<f32> = (0..k * n).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let b: Vec<f32> = (0..m * n).map(|i| 0.01 * (i as f32 - 2.0)).collect();
        let model = Model::linear_static_with_weights("ls_e2e", m, k, n, w, b);
        let dir = tempfile::tempdir().unwrap();
        emit_model(&model, dir.path()).unwrap();

        let status = std::process::Command::new("bash")
            .arg("run_qnn_context.sh")
            .current_dir(dir.path())
            .env("QNN_SDK_ROOT", &sdk)
            .status()
            .expect("spawn run_qnn_context.sh");
        assert!(status.success(), "run_qnn_context.sh failed: {status}");
    }
}
