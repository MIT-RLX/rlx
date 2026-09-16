/* RLX — versatile ML compiler + runtime.
 * Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
 * SPDX-License-Identifier: MIT OR Apache-2.0
 *
 * C ABI for the RLX distributed node. See crates/bindings/rlx-ffi/src/lib.rs.
 *
 * All entry points are thread-safe. Only one node may run per process.
 * rlx_node_start() returns as soon as the serving thread is spawned — poll
 * rlx_node_status() to learn whether the mesh was actually joined.
 */
#ifndef RLX_NODE_H
#define RLX_NODE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RLX_NODE_OK           0
#define RLX_NODE_ERR_BUSY    (-1)  /* a node is already running            */
#define RLX_NODE_ERR_ARG     (-2)  /* null pointer or non-UTF-8 string     */
#define RLX_NODE_ERR_CONFIG  (-3)  /* bad rank/world, or peers != world    */
#define RLX_NODE_ERR_SPAWN   (-4)  /* could not spawn the serving thread   */
#define RLX_NODE_ERR_MODE    (-5)  /* mode was not "infer" or "train"      */

/* Join a mesh as worker `rank` of `world`.
 *
 * peers  — comma-separated "host:port" indexed by rank; "" = UDP discovery.
 * device — "auto", or a backend name ("cpu", "metal", "gpu", ...).
 * mode   — "infer" (serve a shipped stage) or "train" (join a data-parallel
 *          run); "" means "infer".
 *
 * A training rank cannot drop out partway: the gradient reduce is a barrier,
 * so stopping one stalls every other rank. rlx_node_stop() is honoured between
 * inference activations but NOT mid-training-run.
 */
int rlx_node_start(int rank, int world, const char *peers, const char *device,
                   const char *mode);

/* Write node state into buf: "idle" | "running" | "stopping" | "ok: ..." |
 * "error: ...". Returns bytes written (excluding NUL), or a negative code.
 * Truncates to fit; the result is always NUL-terminated. */
int rlx_node_status(char *buf, size_t cap);

/* Ask the node to leave the mesh after its current activation. Cooperative:
 * a node parked in recv exits when its peer sends or the link drops. */
int rlx_node_stop(void);

/* Write the last error detail into buf. Returns bytes written. */
int rlx_node_last_error(char *buf, size_t cap);

/* Platform tag this library was built for ("ios", "android", ...).
 * Static storage — do not free. */
const char *rlx_node_platform(void);

#ifdef __cplusplus
}
#endif

#endif /* RLX_NODE_H */
