// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Inference: build a graph, compile it for a device, set its weights, run it.
// A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].

package org.rlx.kt.examples

import org.rlx.kt.Graph
import org.rlx.kt.Session

fun weights(n: Int, seed: Int) = FloatArray(n) { i -> ((i * 7919 + seed * 104729) % 1000) / 1000f - 0.5f }

fun main(args: Array<String>) {
    val device = args.firstOrNull() ?: "cpu"
    val g = Graph("classifier")
    val x = g.input("x", longArrayOf(1, 4))
    val hidden = g.relu(g.matmul(x, g.param("w1", longArrayOf(4, 8))))
    g.setOutputs(intArrayOf(g.softmax(g.matmul(hidden, g.param("w2", longArrayOf(8, 3))))))

    Session(device = device).compile(g).use { model -> // compile consumes g
        println("compiled on: ${model.device()}")
        model.setParams(mapOf("w1" to weights(32, 1), "w2" to weights(24, 2)))
        val probs = model.run(mapOf("x" to floatArrayOf(0.5f, -1f, 2f, 0.25f)))[0]
        println("probs: " + probs.joinToString(" ") { "%.4f".format(it) })
        println("probs sum: %.4f".format(probs.sum()))
        println("output shapes: ${model.outputShapes().map { it.toList() }} dtypes: ${model.outputDtypes()}")
    }
}
