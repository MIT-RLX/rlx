// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// The helper process. It holds the driver-extension connection and republishes
// it over a UNIX socket, so a client needs no IOKit code and no entitlement.
//
//   helper server <socket-path>
//
// Protocol: see docs/egpu.md. Request 33 bytes, response 17 bytes, both packed
// little-endian. MMIO_WRITE sends no response; MMIO_READ sends a response and
// then the payload.

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOMessage.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

// Must match RlxPciSelector in RlxPciDriverUserClient.iig.
enum { SEL_READ_CFG = 0, SEL_WRITE_CFG = 1, SEL_RESET = 2, SEL_PREPARE_DMA = 3 };

// Must match the wire protocol the rlx client speaks.
enum {
  CMD_MAP_BAR = 1,
  CMD_MAP_SYSMEM_FD = 2,
  CMD_CFG_READ = 3,
  CMD_CFG_WRITE = 4,
  CMD_RESET = 5,
  CMD_MMIO_READ = 6,
  CMD_MMIO_WRITE = 7,
};

// The IOService name the driver registers. Keep in step with SetName() in
// RlxPciDriver.cpp and with DextConfig::service() on the rlx side.
#define SERVICE_NAME "rlxpci"

#define MAX_BARS 6
#define MAX_SYSMEM 128
#define BULK_BUF_SIZE (64 << 20)

typedef struct __attribute__((packed)) {
  uint8_t cmd;
  uint32_t dev_id, bar;
  uint64_t arg0, arg1, arg2;
} request_t;

typedef struct __attribute__((packed)) {
  uint8_t status;
  uint64_t resp0, resp1;
} response_t;

static io_connect_t g_conn = IO_OBJECT_NULL;
static uint8_t *g_bulk;
static struct { mach_vm_address_t addr; mach_vm_size_t size; } g_bars[MAX_BARS];
static struct { void *addr; size_t size; int fd; char name[32]; } g_sysmem[MAX_SYSMEM];
static int g_sysmem_count;

static void recv_all(int fd, void *buf, size_t len) {
  for (size_t off = 0; off < len;) {
    ssize_t n = recv(fd, (uint8_t *)buf + off, len - off, 0);
    if (n <= 0) return;
    off += (size_t)n;
  }
}

// MMIO must be touched with aligned 32-bit volatile accesses; memcpy is free to
// split a word and a device register does not tolerate that.
static void mmio_copy(volatile uint32_t *dst, const volatile uint32_t *src, size_t len) {
  for (size_t i = 0; i < len / 4; i++) dst[i] = src[i];
}

static int send_response(int fd, response_t *resp, int pass_fd) {
  char control[CMSG_SPACE(sizeof(int))];
  struct iovec iov = {resp, sizeof(*resp)};
  struct msghdr msg = {.msg_iov = &iov, .msg_iovlen = 1};

  if (pass_fd >= 0) {
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);
    struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
    *c = (struct cmsghdr){.cmsg_level = SOL_SOCKET,
                          .cmsg_type = SCM_RIGHTS,
                          .cmsg_len = CMSG_LEN(sizeof(int))};
    memcpy(CMSG_DATA(c), &pass_fd, sizeof(int));
  }
  return sendmsg(fd, &msg, 0) > 0 ? 0 : -1;
}

static void send_error(int fd, const char *text) {
  response_t resp = {.status = 1, .resp0 = strlen(text)};
  send_response(fd, &resp, -1);
  send(fd, text, strlen(text), 0);
}

// Exit when the device disappears: an unplugged eGPU leaves a helper holding a
// dead connection, and a client that reconnects to it hangs instead of failing.
static void on_terminate(void *refcon, io_service_t service, uint32_t type, void *arg) {
  (void)refcon; (void)service; (void)arg;
  if (type == kIOMessageServiceIsTerminated) _exit(0);
}

static io_connect_t open_driver(void) {
  io_service_t service =
      IOServiceGetMatchingService(kIOMainPortDefault, IOServiceNameMatching(SERVICE_NAME));
  if (!service) return IO_OBJECT_NULL;

  static io_object_t notification;
  if (!notification) {
    IONotificationPortRef port = IONotificationPortCreate(kIOMainPortDefault);
    IONotificationPortSetDispatchQueue(port, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0));
    IOServiceAddInterestNotification(port, service, kIOGeneralInterest, on_terminate, NULL,
                                     &notification);
  }

  io_connect_t conn = IO_OBJECT_NULL;
  kern_return_t kr = IOServiceOpen(service, mach_task_self(), 0, &conn);
  IOObjectRelease(service);
  return kr == KERN_SUCCESS ? conn : IO_OBJECT_NULL;
}

static int map_bar(uint32_t bar, response_t *resp) {
  if (bar >= MAX_BARS) return -1;
  if (!g_bars[bar].addr &&
      IOConnectMapMemory64(g_conn, bar, mach_task_self(), &g_bars[bar].addr, &g_bars[bar].size,
                           kIOMapAnywhere))
    return -1;
  resp->resp0 = g_bars[bar].addr;
  resp->resp1 = g_bars[bar].size;
  return 0;
}

static int map_sysmem_fd(uint64_t size, response_t *resp, int *out_fd) {
  if (g_sysmem_count >= MAX_SYSMEM) return -1;

  // IOMemoryDescriptor arguments need >4096 bytes, and the mapping must be page
  // aligned; 16 KiB is the smallest that satisfies both on every Apple part.
  size_t bytes = (size + 0xfff) & ~0xfffULL;
  if (bytes < 0x4000) bytes = 0x4000;

  int idx = g_sysmem_count;
  char name[32];
  snprintf(name, sizeof(name), "/rlxpci_%d", idx);
  shm_unlink(name);

  int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
  if (fd < 0) return -1;
  if (ftruncate(fd, (off_t)bytes) < 0) { close(fd); shm_unlink(name); return -1; }

  void *ptr = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) { close(fd); shm_unlink(name); return -1; }

  uint8_t segments[8192] = {0};
  size_t out_size = sizeof(segments);
  if (IOConnectCallStructMethod(g_conn, SEL_PREPARE_DMA, ptr, bytes, segments, &out_size) !=
      KERN_SUCCESS) {
    munmap(ptr, bytes); close(fd); shm_unlink(name);
    return -1;
  }
  memcpy(ptr, segments, out_size);

  g_sysmem[idx].addr = ptr;
  g_sysmem[idx].size = bytes;
  g_sysmem[idx].fd = fd;
  strncpy(g_sysmem[idx].name, name, sizeof(g_sysmem[idx].name) - 1);
  g_sysmem_count++;

  *resp = (response_t){.resp0 = bytes, .resp1 = (uint64_t)idx};
  *out_fd = fd;
  return 0;
}

static int valid_bar(uint32_t bar, uint64_t off, uint64_t len) {
  return bar < MAX_BARS && g_bars[bar].addr && len <= BULK_BUF_SIZE &&
         off + len >= off && off + len <= g_bars[bar].size;
}

static void cleanup(void) {
  for (int i = 0; i < MAX_BARS; i++)
    if (g_bars[i].addr) {
      IOConnectUnmapMemory64(g_conn, i, mach_task_self(), g_bars[i].addr);
      g_bars[i].addr = 0;
    }
  for (int i = 0; i < g_sysmem_count; i++) {
    munmap(g_sysmem[i].addr, g_sysmem[i].size);
    close(g_sysmem[i].fd);
    shm_unlink(g_sysmem[i].name);
  }
  g_sysmem_count = 0;
  if (g_conn != IO_OBJECT_NULL) { IOServiceClose(g_conn); g_conn = IO_OBJECT_NULL; }
}

static void serve(int fd) {
  int bufsize = BULK_BUF_SIZE;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));

  g_conn = open_driver();
  if (g_conn == IO_OBJECT_NULL) {
    request_t discard;
    recv(fd, &discard, sizeof(discard), 0);
    send_error(fd, "no driver extension has claimed a GPU: check System Settings > General > "
                   "Login Items & Extensions > Driver Extensions, and that a display-class "
                   "device is on the PCIe tunnel");
    return;
  }

  request_t req;
  while (recv(fd, &req, sizeof(req), 0) == sizeof(req)) {
    response_t resp = {0};
    uint64_t in[3];

    switch (req.cmd) {
    case CMD_MAP_BAR:
      resp.status = map_bar(req.bar, &resp) ? 1 : 0;
      break;

    case CMD_MAP_SYSMEM_FD: {
      int pass = -1;
      resp.status = map_sysmem_fd(req.arg0, &resp, &pass) ? 1 : 0;
      send_response(fd, &resp, pass);
      continue;
    }

    case CMD_CFG_READ: {
      in[0] = req.arg0; in[1] = req.arg1;
      uint64_t out[2]; uint32_t out_count = 2;
      resp.status = IOConnectCallScalarMethod(g_conn, SEL_READ_CFG, in, 2, out, &out_count) ? 1 : 0;
      if (!resp.status) resp.resp0 = out[0];
      break;
    }

    case CMD_CFG_WRITE:
      in[0] = req.arg0; in[1] = req.arg1; in[2] = req.arg2;
      resp.status = IOConnectCallScalarMethod(g_conn, SEL_WRITE_CFG, in, 3, NULL, NULL) ? 1 : 0;
      break;

    case CMD_RESET:
      resp.status = IOConnectCallScalarMethod(g_conn, SEL_RESET, NULL, 0, NULL, NULL) ? 1 : 0;
      break;

    case CMD_MMIO_READ:
      if (!valid_bar(req.bar, req.arg0, req.arg1)) { resp.status = 1; break; }
      mmio_copy((volatile uint32_t *)g_bulk,
                (const volatile uint32_t *)(g_bars[req.bar].addr + req.arg0), req.arg1);
      resp.resp0 = req.arg1;
      send_response(fd, &resp, -1);
      send(fd, g_bulk, req.arg1, 0);
      continue;

    case CMD_MMIO_WRITE:
      // The payload always follows, so it must be drained even when the write
      // is rejected — otherwise the stream desynchronises and every later
      // request reads garbage.
      recv_all(fd, g_bulk, req.arg1);
      if (valid_bar(req.bar, req.arg0, req.arg1))
        mmio_copy((volatile uint32_t *)(g_bars[req.bar].addr + req.arg0),
                  (const volatile uint32_t *)g_bulk, req.arg1);
      continue;

    default:
      resp.status = 1;
    }
    send_response(fd, &resp, -1);
  }
  cleanup();
}

int main(int argc, char **argv) {
  if (argc != 3 || strcmp(argv[1], "server") != 0) {
    fprintf(stderr, "usage: %s server <socket-path>\n", argv[0]);
    return 2;
  }
  const char *path = argv[2];

  g_bulk = malloc(BULK_BUF_SIZE);
  if (!g_bulk) { perror("malloc"); return 1; }

  int server = socket(AF_UNIX, SOCK_STREAM, 0);
  if (server < 0) { perror("socket"); return 1; }

  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
  unlink(path);
  if (bind(server, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
  if (listen(server, 1) < 0) { perror("listen"); return 1; }
  printf("listening on %s\n", path);

  for (;;) {
    int client = accept(server, NULL, NULL);
    if (client < 0) { if (errno == EINTR) continue; perror("accept"); break; }
    serve(client);
    close(client);
  }
  close(server);
  unlink(path);
  cleanup();
  return 0;
}
