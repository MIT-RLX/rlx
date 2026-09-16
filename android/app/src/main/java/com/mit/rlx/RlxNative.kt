package com.mit.rlx

/**
 * JNI bindings shared by the demo activity and instrumented tests.
 */
object RlxNative {
    init {
        // Dynamic OpenBLAS builds only — static OpenBLAS is linked into rlx_jni.
        try {
            System.loadLibrary("openblas")
        } catch (_: UnsatisfiedLinkError) {
            // Expected for scalar / static-OpenBLAS builds.
        }
        System.loadLibrary("rlx_jni")
    }

    /** Tiny matmul→bias→GELU demo graph. */
    external fun runInference(): FloatArray

    external fun backendName(): String

    /** Embedded MNIST MLP logits (length 10) for the bundled sample digit. */
    external fun runMnist(): FloatArray

    /** Argmax class for [runMnist]. */
    external fun mnistPredict(): Int

    /** Ground-truth label of the bundled MNIST sample. */
    external fun mnistExpectedLabel(): Int

    // ── distributed node ───────────────────────────────────────────────

    /**
     * Join an RLX mesh as worker [rank] of [world].
     *
     * [peers] is a comma-separated `host:port` list indexed by rank; [device]
     * is `auto` or a backend name (`cpu`, `gpu`, …); [mode] is `infer` or
     * `train`. Returns immediately — the node serves on its own thread, so
     * poll [nodeStatus].
     *
     * A training rank cannot drop out partway: the gradient reduce is a
     * barrier, so stopping one stalls every other rank.
     *
     * Throws if a node is already running or the arguments are inconsistent.
     */
    external fun nodeStart(
        rank: Int,
        world: Int,
        peers: String,
        device: String,
        mode: String,
    ): String

    /** `idle` | `running` | `stopping` | `ok: …` | `error: …`. */
    external fun nodeStatus(): String

    /**
     * Ask the node to leave the mesh after its current activation.
     * Cooperative: a node parked in `recv` exits when its peer sends or the
     * link drops.
     */
    external fun nodeStop()
}
