// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// C ABI shim over XRT's C++ API for running an MLIR-AIE INT8 GEMM overlay on
// the XDNA NPU. The NPU requires the modern register_xclbin + hw_context flow,
// which XRT exposes only in C++ (the C API's load_axlf path returns "Operation
// not supported" on amdxdna) — so rlx dlopens these `extern "C"` entry points.
//
// The handle API separates the expensive ONE-TIME setup (device open,
// register_xclbin, hw_context, kernel, BO alloc, instruction upload, arg bind)
// from the hot PER-CALL path (sync inputs, start, wait, sync output). That
// amortization is the latency win — cold setup is ~150 ms, warm run is ~ms.
// Kernel ABI mirrors mlir-aie's host: MLIR_AIE(opcode=3, instr, ninstr, A, B, C),
// BOs bound by kernel.group_id(1/3/4/5).
//
// Build (needs XRT dev headers + libxrt; g++):
//   g++ -O2 -fPIC -shared -std=c++17 -I$XILINX_XRT/include \
//       -o librlx_xdna_shim.so csrc/xrt_gemm_shim.cpp \
//       -L$XILINX_XRT/lib/x86_64-linux-gnu -Wl,-rpath,$XILINX_XRT/lib/x86_64-linux-gnu \
//       -lxrt_coreutil

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#ifndef XCL_BO_FLAGS_CACHEABLE
#define XCL_BO_FLAGS_CACHEABLE (1U << 24)
#endif
#ifndef XRT_BO_FLAGS_HOST_ONLY
#define XRT_BO_FLAGS_HOST_ONLY (1U << 29)
#endif

// Persistent GEMM context: everything created once, reused per call.
struct RlxXdnaGemm {
  xrt::device device;
  xrt::xclbin xclbin;
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::bo bo_instr, bo_a, bo_b, bo_c;
  xrt::run run;
  std::vector<xrt::bo> wbos; // resident weight blocks (tiled path)
  size_t a_bytes, b_bytes, c_bytes;

  RlxXdnaGemm(const char *xclbin_path, const uint32_t *insts, size_t ninstr,
              int M, int K, int N)
      : device(0), // device 0 == the NPU
        xclbin(std::string(xclbin_path)),
        // register the overlay, then open a hardware context on its UUID
        context((device.register_xclbin(xclbin),
                 xrt::hw_context(device, xclbin.get_uuid()))),
        kernel(context, "MLIR_AIE"),
        bo_instr(device, ninstr * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
                 kernel.group_id(1)),
        bo_a(device, (size_t)M * K, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3)),
        bo_b(device, (size_t)K * N, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4)),
        bo_c(device, (size_t)M * N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
             kernel.group_id(5)),
        run(kernel), a_bytes((size_t)M * K), b_bytes((size_t)K * N),
        c_bytes((size_t)M * N * sizeof(int32_t)) {
    // Upload the instruction stream once; it never changes across calls.
    std::memcpy(bo_instr.map<void *>(), insts, ninstr * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    // Bind kernel args once (BOs are referenced by handle, so re-syncing their
    // contents per call is all that's needed).
    unsigned int opcode = 3;
    run.set_arg(0, opcode);
    run.set_arg(1, bo_instr);
    run.set_arg(2, (int)ninstr);
    run.set_arg(3, bo_a);
    run.set_arg(4, bo_b);
    run.set_arg(5, bo_c);
  }
};

// Open a persistent context (one-time setup). Returns null on failure.
extern "C" RlxXdnaGemm *rlx_xdna_gemm_open(const char *xclbin_path,
                                           const uint32_t *insts, size_t ninstr,
                                           int M, int K, int N) {
  try {
    return new RlxXdnaGemm(xclbin_path, insts, ninstr, M, K, N);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_gemm_open: %s\n", e.what());
    return nullptr;
  }
}

// Hot path: upload A/B, run on the NPU, read C back. Reuses all handles.
extern "C" int rlx_xdna_gemm_run(RlxXdnaGemm *h, const int8_t *A,
                                 const int8_t *B, int32_t *C) {
  if (!h) return 2;
  try {
    std::memcpy(h->bo_a.map<void *>(), A, h->a_bytes);
    std::memcpy(h->bo_b.map<void *>(), B, h->b_bytes);
    h->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    h->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    h->run.start();
    h->run.wait();
    h->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(C, h->bo_c.map<void *>(), h->c_bytes);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_gemm_run: %s\n", e.what());
    return 1;
  }
}

// Upload the weight B into its resident BO once (reused across run_a calls).
extern "C" int rlx_xdna_gemm_set_weight(RlxXdnaGemm *h, const int8_t *B) {
  if (!h) return 2;
  try {
    std::memcpy(h->bo_b.map<void *>(), B, h->b_bytes);
    h->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_gemm_set_weight: %s\n", e.what());
    return 1;
  }
}

// Hot path with RESIDENT weight: upload only A, run, read C (B stays on-device
// from a prior set_weight). This is the LLM-decode shape — reuse the weight.
extern "C" int rlx_xdna_gemm_run_a(RlxXdnaGemm *h, const int8_t *A, int32_t *C) {
  if (!h) return 2;
  try {
    std::memcpy(h->bo_a.map<void *>(), A, h->a_bytes);
    h->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    h->run.start();
    h->run.wait();
    h->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(C, h->bo_c.map<void *>(), h->c_bytes);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_gemm_run_a: %s\n", e.what());
    return 1;
  }
}

// Upload weight block `idx` into its own resident BO (allocated on first use).
// For the tiled path: each (K-block, N-block) of the weight stays on-device.
extern "C" int rlx_xdna_gemm_set_weight_block(RlxXdnaGemm *h, int idx,
                                              const int8_t *B) {
  if (!h || idx < 0) return 2;
  try {
    while ((int)h->wbos.size() <= idx) {
      h->wbos.push_back(xrt::bo(h->device, h->b_bytes, XRT_BO_FLAGS_HOST_ONLY,
                                h->kernel.group_id(4)));
    }
    std::memcpy(h->wbos[idx].map<void *>(), B, h->b_bytes);
    h->wbos[idx].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_gemm_set_weight_block: %s\n", e.what());
    return 1;
  }
}

// Run one tile against resident weight block `idx`: upload only A, bind that
// block as the B arg, run, read the partial C. Host accumulates over K-blocks.
extern "C" int rlx_xdna_gemm_run_block(RlxXdnaGemm *h, int idx, const int8_t *A,
                                       int32_t *C) {
  if (!h || idx < 0 || idx >= (int)h->wbos.size()) return 2;
  try {
    std::memcpy(h->bo_a.map<void *>(), A, h->a_bytes);
    h->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    h->run.set_arg(4, h->wbos[idx]); // resident B block
    h->run.start();
    h->run.wait();
    h->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(C, h->bo_c.map<void *>(), h->c_bytes);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_gemm_run_block: %s\n", e.what());
    return 1;
  }
}

extern "C" void rlx_xdna_gemm_close(RlxXdnaGemm *h) { delete h; }

// Self-contained runner for a DMA-passthrough overlay (i32 in -> out) emitted
// by rlx_xdna::aie. Same MLIR_AIE(opcode, instr, ninstr, A, B, C) ABI; the
// design copies arg0 (A) -> arg2 (C), B unused. Verifies rlx-emitted AIE-MLIR
// runs on the NPU. Returns 0 on success.
extern "C" int rlx_xdna_run_passthrough(const char *xclbin_path,
                                        const uint32_t *insts, size_t ninstr,
                                        int n, const int32_t *in, int32_t *out) {
  try {
    auto device = xrt::device(0);
    auto xclbin = xrt::xclbin(std::string(xclbin_path));
    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(context, "MLIR_AIE");

    const size_t bytes = (size_t)n * sizeof(int32_t);
    auto bo_instr = xrt::bo(device, ninstr * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_in = xrt::bo(device, bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_b = xrt::bo(device, bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_out = xrt::bo(device, bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    std::memcpy(bo_instr.map<void *>(), insts, ninstr * sizeof(uint32_t));
    std::memcpy(bo_in.map<void *>(), in, bytes);
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    unsigned int opcode = 3;
    auto run = xrt::run(kernel);
    run.set_arg(0, opcode);
    run.set_arg(1, bo_instr);
    run.set_arg(2, (int)ninstr);
    run.set_arg(3, bo_in);
    run.set_arg(4, bo_b);
    run.set_arg(5, bo_out);
    run.start();
    run.wait();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(out, bo_out.map<void *>(), bytes);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_run_passthrough: %s\n", e.what());
    return 1;
  }
}

// One-shot convenience = open + run + close (used by the simple path).
extern "C" int rlx_xdna_gemm_i8(const char *xclbin_path, const uint32_t *insts,
                                size_t ninstr, int M, int K, int N,
                                const int8_t *A, const int8_t *B, int32_t *C) {
  RlxXdnaGemm *h = rlx_xdna_gemm_open(xclbin_path, insts, ninstr, M, K, N);
  if (!h) return 2;
  int rc = rlx_xdna_gemm_run(h, A, B, C);
  rlx_xdna_gemm_close(h);
  return rc;
}

// Persistent i32-in / i32-out context — for the rlx-emitted elementwise kernels
// (and any 1-in/1-out overlay). Same MLIR_AIE(opcode, instr, ninstr, in, B, out)
// ABI as passthrough: `open` pays the one-time setup, `run` is the hot path
// (sync in, run, sync out) so it benchmarks warm like the GEMM path.
struct RlxXdnaIO {
  xrt::device device;
  xrt::xclbin xclbin;
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::bo bo_instr, bo_in, bo_b, bo_out;
  xrt::run run;
  size_t nbytes;

  RlxXdnaIO(const char *xclbin_path, const uint32_t *insts, size_t ninstr, int n)
      : device(0),
        xclbin(std::string(xclbin_path)),
        context((device.register_xclbin(xclbin),
                 xrt::hw_context(device, xclbin.get_uuid()))),
        kernel(context, "MLIR_AIE"),
        bo_instr(device, ninstr * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
                 kernel.group_id(1)),
        bo_in(device, (size_t)n * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
              kernel.group_id(3)),
        bo_b(device, (size_t)n * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
             kernel.group_id(4)),
        bo_out(device, (size_t)n * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
               kernel.group_id(5)),
        run(kernel), nbytes((size_t)n * sizeof(int32_t)) {
    std::memcpy(bo_instr.map<void *>(), insts, ninstr * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    unsigned int opcode = 3;
    run.set_arg(0, opcode);
    run.set_arg(1, bo_instr);
    run.set_arg(2, (int)ninstr);
    run.set_arg(3, bo_in);
    run.set_arg(4, bo_b);
    run.set_arg(5, bo_out);
  }
};

extern "C" RlxXdnaIO *rlx_xdna_io_open(const char *xclbin_path,
                                       const uint32_t *insts, size_t ninstr,
                                       int n) {
  try {
    return new RlxXdnaIO(xclbin_path, insts, ninstr, n);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_io_open: %s\n", e.what());
    return nullptr;
  }
}

extern "C" int rlx_xdna_io_run(RlxXdnaIO *h, const int32_t *in, int32_t *out) {
  if (!h) return 2;
  try {
    std::memcpy(h->bo_in.map<void *>(), in, h->nbytes);
    h->bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    h->run.start();
    h->run.wait();
    h->bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(out, h->bo_out.map<void *>(), h->nbytes);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_io_run: %s\n", e.what());
    return 1;
  }
}

// Two-input variant: writes BOTH bo_in (arg3=A) and bo_b (arg4=B) before the
// dispatch, for binary elementwise ops (out = A op B). Reuses RlxXdnaIO, whose
// three BOs are already all sized n*4 — so no new struct/allocation is needed.
extern "C" int rlx_xdna_io_run2(RlxXdnaIO *h, const int32_t *a, const int32_t *b,
                                int32_t *out) {
  if (!h) return 2;
  try {
    std::memcpy(h->bo_in.map<void *>(), a, h->nbytes);
    std::memcpy(h->bo_b.map<void *>(), b, h->nbytes);
    h->bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    h->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    h->run.start();
    h->run.wait();
    h->bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(out, h->bo_out.map<void *>(), h->nbytes);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_io_run2: %s\n", e.what());
    return 1;
  }
}

extern "C" void rlx_xdna_io_close(RlxXdnaIO *h) { delete h; }

// Persistent i32 A·B→C context — for the rlx-emitted i32 matmul overlays (all
// three operands i32, unlike RlxXdnaGemm's i8 A/B). GEMM 3-arg ABI.
struct RlxXdnaMM32 {
  xrt::device device;
  xrt::xclbin xclbin;
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::bo bo_instr, bo_a, bo_b, bo_c;
  xrt::run run;
  size_t a_bytes, b_bytes, c_bytes;

  RlxXdnaMM32(const char *xclbin_path, const uint32_t *insts, size_t ninstr,
              int M, int K, int N)
      : device(0),
        xclbin(std::string(xclbin_path)),
        context((device.register_xclbin(xclbin),
                 xrt::hw_context(device, xclbin.get_uuid()))),
        kernel(context, "MLIR_AIE"),
        bo_instr(device, ninstr * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
                 kernel.group_id(1)),
        bo_a(device, (size_t)M * K * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3)),
        bo_b(device, (size_t)K * N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4)),
        bo_c(device, (size_t)M * N * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5)),
        run(kernel), a_bytes((size_t)M * K * sizeof(int32_t)),
        b_bytes((size_t)K * N * sizeof(int32_t)),
        c_bytes((size_t)M * N * sizeof(int32_t)) {
    std::memcpy(bo_instr.map<void *>(), insts, ninstr * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    unsigned int opcode = 3;
    run.set_arg(0, opcode);
    run.set_arg(1, bo_instr);
    run.set_arg(2, (int)ninstr);
    run.set_arg(3, bo_a);
    run.set_arg(4, bo_b);
    run.set_arg(5, bo_c);
  }
};

extern "C" RlxXdnaMM32 *rlx_xdna_mm32_open(const char *xclbin_path,
                                           const uint32_t *insts, size_t ninstr,
                                           int M, int K, int N) {
  try {
    return new RlxXdnaMM32(xclbin_path, insts, ninstr, M, K, N);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_mm32_open: %s\n", e.what());
    return nullptr;
  }
}

extern "C" int rlx_xdna_mm32_run(RlxXdnaMM32 *h, const int32_t *A,
                                 const int32_t *B, int32_t *C) {
  if (!h) return 2;
  try {
    std::memcpy(h->bo_a.map<void *>(), A, h->a_bytes);
    std::memcpy(h->bo_b.map<void *>(), B, h->b_bytes);
    h->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    h->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    h->run.start();
    h->run.wait();
    h->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(C, h->bo_c.map<void *>(), h->c_bytes);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_mm32_run: %s\n", e.what());
    return 1;
  }
}

extern "C" void rlx_xdna_mm32_close(RlxXdnaMM32 *h) { delete h; }

// Generic 3-buffer runner: three host buffers of ARBITRARY byte sizes bound to
// kernel args 3/4/5. Covers any overlay whose runtime_sequence takes 3 buffers
// of independent sizes (e.g. affine norm: x[rows*cols], gamma+beta[2*cols],
// out[rows*cols]). One persistent context; `run` writes a+b, dispatches, reads c.
struct RlxXdnaRun3 {
  xrt::device device;
  xrt::xclbin xclbin;
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::bo bo_instr, bo_a, bo_b, bo_c;
  xrt::run run;
  size_t na, nb, nc;

  RlxXdnaRun3(const char *xclbin_path, const uint32_t *insts, size_t ninstr,
              size_t na_, size_t nb_, size_t nc_)
      : device(0), xclbin(std::string(xclbin_path)),
        context((device.register_xclbin(xclbin),
                 xrt::hw_context(device, xclbin.get_uuid()))),
        kernel(context, "MLIR_AIE"),
        bo_instr(device, ninstr * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
                 kernel.group_id(1)),
        bo_a(device, na_, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3)),
        bo_b(device, nb_, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4)),
        bo_c(device, nc_, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5)),
        run(kernel), na(na_), nb(nb_), nc(nc_) {
    std::memcpy(bo_instr.map<void *>(), insts, ninstr * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    unsigned int opcode = 3;
    run.set_arg(0, opcode);
    run.set_arg(1, bo_instr);
    run.set_arg(2, (int)ninstr);
    run.set_arg(3, bo_a);
    run.set_arg(4, bo_b);
    run.set_arg(5, bo_c);
  }
};

extern "C" RlxXdnaRun3 *rlx_xdna_run3_open(const char *xclbin_path,
                                           const uint32_t *insts, size_t ninstr,
                                           size_t na, size_t nb, size_t nc) {
  try {
    return new RlxXdnaRun3(xclbin_path, insts, ninstr, na, nb, nc);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_run3_open: %s\n", e.what());
    return nullptr;
  }
}

extern "C" int rlx_xdna_run3_run(RlxXdnaRun3 *h, const void *a, const void *b,
                                 void *c) {
  if (!h) return 2;
  try {
    std::memcpy(h->bo_a.map<void *>(), a, h->na);
    std::memcpy(h->bo_b.map<void *>(), b, h->nb);
    h->bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    h->bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    h->run.start();
    h->run.wait();
    h->bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(c, h->bo_c.map<void *>(), h->nc);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "rlx_xdna_run3_run: %s\n", e.what());
    return 1;
  }
}

extern "C" void rlx_xdna_run3_close(RlxXdnaRun3 *h) { delete h; }
