// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// LD_PRELOAD ioctl/mmap interposer — captures the GROUND-TRUTH amdxdna ABI the
// known-good XRT path uses, so the direct (no-XRT) path in `src/direct.rs` can
// mirror it exactly. It intercepts every `ioctl` and dumps the amdxdna ones
// (type 'd', nr in [0x40..0x4a]) with their struct contents, plus follows the
// EXEC_CMD command BO to hexdump the actual ert command packet.
//
// Build + run against the working XRT example:
//   gcc -O2 -shared -fPIC -o xdna_trace.so xdna_ioctl_trace.c -ldl
//   LD_PRELOAD=$PWD/xdna_trace.so LD_LIBRARY_PATH=<xrt lib> \
//     RLX_XDNA_SHIM=... XCLBIN=... INSTS=... M=512 K=512 N=512 ITERS=1 \
//     ./npu_gemm_check
//
// All output goes to stderr prefixed with "XT|" so it's easy to grep.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

// ── amdxdna UAPI structs (mirrors /usr/include/drm/amdxdna_accel.h) ──────────
struct create_hwctx {
  uint64_t ext, ext_flags, qos_p;
  uint32_t umq_bo, log_buf_bo, max_opc, num_tiles, mem_size, umq_doorbell, handle,
      syncobj_handle;
};
struct config_hwctx {
  uint32_t handle, param_type;
  uint64_t param_val;
  uint32_t param_val_size, pad;
};
struct create_bo {
  uint64_t flags, vaddr, size;
  uint32_t type, handle;
};
struct get_bo_info {
  uint64_t ext, ext_flags;
  uint32_t handle, pad;
  uint64_t map_offset, vaddr, xdna_addr;
};
struct exec_cmd {
  uint64_t ext, ext_flags;
  uint32_t hwctx, type;
  uint64_t cmd_handles, args;
  uint32_t cmd_count, arg_count;
  uint64_t seq;
};
struct cu_config {
  uint32_t cu_bo;
  uint8_t cu_func;
  uint8_t pad[3];
};
struct config_cu_param {
  uint16_t num_cus;
  uint16_t pad[3];
  struct cu_config cu[16];
};

static const char *BO_TYPE[] = {"INVALID", "SHMEM", "DEV_HEAP", "DEV", "CMD"};

// ── handle → mmap address table, so EXEC_CMD can dump the command BO ─────────
#define MAXH 4096
static uint64_t g_handle_off[MAXH];   // GET_BO_INFO: handle -> map_offset
static uint64_t g_handle_xdna[MAXH];  // GET_BO_INFO: handle -> xdna_addr
static uint64_t g_handle_size[MAXH];  // CREATE_BO: handle -> size
static uint64_t g_last_create_size;   // CREATE_BO in size, bound to handle at out
static uint64_t g_off_addr[256];      // mmap: offset ...
static uint64_t g_off_key[256];       // ... -> keyed by offset
static size_t g_off_addr_sz[256];
static int g_noff = 0;

static void *(*real_mmap)(void *, size_t, int, int, int, off_t) = NULL;
static int (*real_ioctl)(int, unsigned long, ...) = NULL;

static void ensure(void) {
  if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
  if (!real_mmap) real_mmap = dlsym(RTLD_NEXT, "mmap");
}

static uint64_t addr_for_handle(uint32_t h) {
  if (h >= MAXH) return 0;
  uint64_t off = g_handle_off[h];
  for (int i = 0; i < g_noff; i++)
    if (g_off_key[i] == off) return g_off_addr[i];
  return 0;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
  ensure();
  void *r = real_mmap(addr, len, prot, flags, fd, off);
  if ((uint64_t)off >= 0x100000000ULL) { // DRM BO map offsets are high
    uint64_t a = (uint64_t)r;
    // report alignment of the returned userptr (trailing-zero bits)
    int alnlog = a ? __builtin_ctzll(a) : 64;
    fprintf(stderr, "XT= mmap fd=%d off=0x%llx len=%zu -> ptr=0x%llx align=2^%d (%lluKiB)\n",
            fd, (unsigned long long)off, len, (unsigned long long)a, alnlog,
            (unsigned long)(1ULL << alnlog) / 1024);
  }
  if (r != MAP_FAILED && g_noff < 256) {
    g_off_key[g_noff] = (uint64_t)off;
    g_off_addr[g_noff] = (uint64_t)r;
    g_off_addr_sz[g_noff] = len;
    g_noff++;
  }
  return r;
}

static void dump_words(const char *tag, uint64_t addr, int words) {
  if (!addr) {
    fprintf(stderr, "XT|   %s: <no mapping>\n", tag);
    return;
  }
  const uint32_t *p = (const uint32_t *)addr;
  fprintf(stderr, "XT|   %s (first %d words @0x%llx):\n", tag, words,
          (unsigned long long)addr);
  for (int i = 0; i < words; i += 8) {
    fprintf(stderr, "XT|     [%02d]", i);
    for (int j = i; j < i + 8 && j < words; j++)
      fprintf(stderr, " %08x", p[j]);
    fprintf(stderr, "\n");
  }
}

int ioctl(int fd, unsigned long request, ...) {
  ensure();
  va_list ap;
  va_start(ap, request);
  void *arg = va_arg(ap, void *);
  va_end(ap);

  unsigned type = (request >> 8) & 0xff;
  unsigned nr = request & 0xff;
  int is_xdna = (type == 'd' && nr >= 0x40 && nr <= 0x4a);
  int id = nr - 0x40;

  // Catch-all: every DRM ioctl in order (nr + amdxdna id if applicable), so the
  // full setup sequence is visible — not just the ones we decode in detail.
  if (type == 'd')
    fprintf(stderr, "XT= drm ioctl fd=%d nr=0x%02x%s\n", fd, nr,
            is_xdna ? (id == 0   ? " [CREATE_HWCTX]"
                       : id == 2 ? " [CONFIG_HWCTX]"
                       : id == 3 ? " [CREATE_BO]"
                       : id == 4 ? " [GET_BO_INFO]"
                       : id == 6 ? " [EXEC_CMD]"
                       : id == 7 ? " [GET_INFO]"
                       : id == 8 ? " [SET_STATE]"
                                 : " [amdxdna]")
                     : "");

  // Pre-call decode (inputs).
  if (is_xdna) {
    if (id == 0) { // CREATE_HWCTX
      struct create_hwctx *c = arg;
      fprintf(stderr,
              "XT| CREATE_HWCTX in: ext=0x%llx ext_flags=0x%llx qos_p=0x%llx "
              "max_opc=%u num_tiles=%u mem_size=%u umq_bo=%u log_buf_bo=%u\n",
              (unsigned long long)c->ext, (unsigned long long)c->ext_flags,
              (unsigned long long)c->qos_p, c->max_opc, c->num_tiles, c->mem_size,
              c->umq_bo, c->log_buf_bo);
      if (c->qos_p) {
        // struct amdxdna_qos_info: gops,fps,dma_bandwidth,latency,frame_exec_time,priority
        uint32_t *q = (uint32_t *)(uintptr_t)c->qos_p;
        fprintf(stderr,
                "XT|   qos: gops=%u fps=%u dma_bw=%u latency=%u frame_exec=%u priority=%u\n",
                q[0], q[1], q[2], q[3], q[4], q[5]);
      }
    } else if (id == 2) { // CONFIG_HWCTX
      struct config_hwctx *c = arg;
      fprintf(stderr, "XT| CONFIG_HWCTX: handle=%u param_type=%u val_size=%u\n",
              c->handle, c->param_type, c->param_val_size);
      if (c->param_type == 0 && c->param_val) { // CONFIG_CU
        struct config_cu_param *cu = (void *)(uintptr_t)c->param_val;
        int n = cu->num_cus > 16 ? 16 : cu->num_cus;
        fprintf(stderr, "XT|   CONFIG_CU num_cus=%u\n", cu->num_cus);
        for (int i = 0; i < n; i++) {
          uint32_t cb = cu->cu[i].cu_bo;
          fprintf(stderr, "XT|     cu[%d]: cu_bo=%u cu_func=%u\n", i, cb, cu->cu[i].cu_func);
          // Dump the PDI the firmware is about to load: it lives in the dev heap
          // at heap_host + (cu_xdna - heap_xdna). Heap is handle 1.
          uint64_t heap_addr = addr_for_handle(1);
          uint64_t sz = cb < MAXH ? g_handle_size[cb] : 0;
          if (heap_addr && cb < MAXH && g_handle_xdna[cb]) {
            const uint8_t *pdi = (const uint8_t *)(heap_addr + (g_handle_xdna[cb] - g_handle_xdna[1]));
            uint64_t sum = 0;
            for (uint64_t k = 0; k < sz; k++) sum += pdi[k];
            fprintf(stderr, "XT|     PDI %llu bytes sum=0x%llx head=", (unsigned long long)sz, (unsigned long long)sum);
            for (int k = 0; k < 8; k++) fprintf(stderr, "%02x", pdi[k]);
            fprintf(stderr, "\n");
          }
        }
      }
    } else if (id == 7) { // GET_INFO (in)
      struct { uint32_t param; uint32_t buffer_size; uint64_t buffer; } *g = arg;
      fprintf(stderr, "XT| GET_INFO in: param=%u buffer_size=%u\n", g->param,
              g->buffer_size);
    } else if (id == 3) { // CREATE_BO (in)
      struct create_bo *b = arg;
      g_last_create_size = b->size;
      const char *tn = b->type <= 4 ? BO_TYPE[b->type] : "?";
      fprintf(stderr, "XT| CREATE_BO in: type=%s size=%llu flags=0x%llx vaddr=0x%llx\n",
              tn, (unsigned long long)b->size, (unsigned long long)b->flags,
              (unsigned long long)b->vaddr);
    } else if (id == 6) { // EXEC_CMD (in)
      struct exec_cmd *e = arg;
      fprintf(stderr,
              "XT| EXEC_CMD in: hwctx=%u type=%u cmd_count=%u arg_count=%u "
              "cmd_handles=0x%llx args=0x%llx\n",
              e->hwctx, e->type, e->cmd_count, e->arg_count,
              (unsigned long long)e->cmd_handles, (unsigned long long)e->args);
      // cmd_handles: if count==1 it may BE the handle, else a pointer to an array.
      uint32_t cmdh;
      if (e->cmd_count == 1)
        cmdh = (uint32_t)e->cmd_handles;
      else
        cmdh = ((uint32_t *)(uintptr_t)e->cmd_handles)[0];
      fprintf(stderr, "XT|   cmd BO handle[0]=%u\n", cmdh);
      dump_words("cmd BO packet", addr_for_handle(cmdh), 32);
      // args array (BO handles patched into the command).
      if (e->arg_count && e->args) {
        uint32_t *a = (uint32_t *)(uintptr_t)e->args;
        fprintf(stderr, "XT|   args[%u]:", e->arg_count);
        for (uint32_t i = 0; i < e->arg_count && i < 16; i++)
          fprintf(stderr, " %u", a[i]);
        fprintf(stderr, "\n");
        // Dump the INSTRUCTION-STREAM BO (args[0]) content — it lives in the dev
        // heap at heap_host + (instr_xdna - heap_xdna). If XRT patched operand
        // addresses into it, its checksum will differ from the raw insts.bin.
        uint32_t ih = a[0];
        uint64_t heap_addr = addr_for_handle(1);
        if (heap_addr && ih < MAXH && g_handle_xdna[ih] && g_handle_size[ih]) {
          const uint8_t *ins = (const uint8_t *)(heap_addr + (g_handle_xdna[ih] - g_handle_xdna[1]));
          uint64_t sz = g_handle_size[ih], sum = 0;
          for (uint64_t k = 0; k < sz; k++) sum += ins[k];
          fprintf(stderr, "XT|   INSTR bo=%u %llu bytes sum=0x%llx head=", ih,
                  (unsigned long long)sz, (unsigned long long)sum);
          for (int k = 0; k < 8; k++) fprintf(stderr, "%02x", ins[k]);
          fprintf(stderr, " tail=");
          for (uint64_t k = (sz >= 8 ? sz - 8 : 0); k < sz; k++) fprintf(stderr, "%02x", ins[k]);
          fprintf(stderr, "\n");
        }
      }
    }
  }

  int ret = real_ioctl(fd, request, arg);

  // Post-call decode (outputs).
  if (is_xdna && ret == 0) {
    if (id == 0) {
      struct create_hwctx *c = arg;
      fprintf(stderr,
              "XT| CREATE_HWCTX out: handle=%u syncobj=%u umq_bo=%u doorbell=0x%x\n",
              c->handle, c->syncobj_handle, c->umq_bo, c->umq_doorbell);
    } else if (id == 3) {
      struct create_bo *b = arg;
      if (b->handle < MAXH) g_handle_size[b->handle] = g_last_create_size;
      fprintf(stderr, "XT| CREATE_BO out: handle=%u\n", b->handle);
    } else if (id == 4) { // GET_BO_INFO out
      struct get_bo_info *g = arg;
      if (g->handle < MAXH) {
        g_handle_off[g->handle] = g->map_offset;
        g_handle_xdna[g->handle] = g->xdna_addr;
      }
      fprintf(stderr,
              "XT| GET_BO_INFO: handle=%u map_offset=0x%llx xdna_addr=0x%llx\n",
              g->handle, (unsigned long long)g->map_offset,
              (unsigned long long)g->xdna_addr);
    } else if (id == 6) {
      struct exec_cmd *e = arg;
      fprintf(stderr, "XT| EXEC_CMD out: seq=%llu\n", (unsigned long long)e->seq);
    }
  } else if (is_xdna && ret != 0) {
    fprintf(stderr, "XT| ioctl id=%d FAILED ret=%d\n", id, ret);
  }
  return ret;
}
