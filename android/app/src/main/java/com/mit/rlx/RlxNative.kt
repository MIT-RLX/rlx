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
}
