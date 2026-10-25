// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Picking a backend: what this build has, what suits a graph, and the ways to
// choose — by name, by cost model, by policy, by benchmark, per run.

package org.rlx.kt.examples

import kotlin.math.abs
import org.rlx.kt.FlexibleSession
import org.rlx.kt.Graph
import org.rlx.kt.RLX
import org.rlx.kt.Router
import org.rlx.kt.Runner
import org.rlx.kt.Session

private fun model(): Graph {
    val g = Graph("mlp")
    val x = g.input("x", longArrayOf(4, 16))
    g.setOutputs(intArrayOf(g.tanh(g.matmul(x, g.param("w", longArrayOf(16, 16))))))
    return g
}

fun main() {
    val x = mapOf("x" to FloatArray(64) { (it % 7) / 7f })
    val w = FloatArray(256) { (it % 5) / 10f - 0.2f }
    val available = RLX.devices()
    println("devices: ${available.joinToString(" ")}")
    println("metal available: ${RLX.isAvailable("metal")}")
    println("fastest for graph: ${RLX.fastestDeviceFor(model())}")
    var reference: FloatArray? = null
    for (device in available) {
        val c = Session(device = device).compile(model())
        c.setParam("w", w)
        val y = c.run(x)[0]
        val ref = reference ?: y.also { reference = it }
        println("session $device: max|Δ| vs first = %.2e".format(y.indices.maxOf { abs(y[it] - ref[it]) }))
    }
    val flex = FlexibleSession(mapOf("prefer" to listOf("metal", "cpu")))
    println("flexible session picked: ${flex.compile(model()).device()}")
    val runner = Runner(model())
    runner.setParam("w", w)
    println("runner warmed: ${runner.warmAll().joinToString(" ")}")
    for (row in runner.benchmark(x, runs = 3)) println("  bench ${row["device"]}: exec_ms=${row["exec_ms"]}")
    println("runner ran on: ${runner.run(x).device}")
    println("run_try chain: ${runner.runTry(listOf("metal", "cpu"), x).device}")
    val router = Router(model())
    router.setParam("w", w)
    println("router ran on: ${router.runChain(x).device}")
}
