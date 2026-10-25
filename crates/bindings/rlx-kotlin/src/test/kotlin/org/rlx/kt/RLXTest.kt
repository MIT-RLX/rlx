// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx.kt

import java.io.File
import kotlin.concurrent.thread

// The Kotlin binding against the real library; exits non-zero on the first failure.

private var checks = 0
private fun check(ok: Boolean, what: String) {
    checks++
    if (!ok) throw AssertionError("failed: $what")
}

private fun lossGraph(): Pair<Graph, Int> {
    val g = Graph("mse")
    val x = g.input("x", longArrayOf(1, 2))
    val y = g.input("y", longArrayOf(1, 2))
    val w = g.param("w", longArrayOf(2, 2))
    val d = g.sub(g.matmul(x, w), y)
    g.setOutputs(intArrayOf(g.mean(g.mul(d, d), longArrayOf(0, 1))))
    return g to w
}

fun main() {
    // inference
    var (g, w) = lossGraph()
    Session().compile(g).use { m ->
        check(g.isConsumed(), "compile consumes")
        m.setParam("w", floatArrayOf(1f, 0f, 0f, 1f))
        check(m.run(mapOf("x" to floatArrayOf(1f, 2f), "y" to floatArrayOf(0f, 0f)))[0][0] == 2.5f, "loss 2.5")
        check(m.device() == "cpu", "device")
    }
    // errors
    try { Graph().relu(42); check(false, "bad node throws") } catch (e: RlxException) { check(e.kind == RlxException.Kind.RANGE, "range") }
    // training
    lossGraph().also { g = it.first; w = it.second }
    val bwd = Session().compile(RLX.grad(g, intArrayOf(w)))
    bwd.setParam("w", floatArrayOf(1f, 0f, 0f, 1f))
    check(bwd.run(mapOf("x" to floatArrayOf(1f, 2f), "y" to floatArrayOf(0f, 0f), "d_output" to floatArrayOf(1f)))[1].contentEquals(floatArrayOf(1f, 2f, 2f, 4f)), "dw")
    val t = Trainer(g.clone(), listOf("w"), mapOf("w" to FloatArray(4)), mapOf("optimizer" to mapOf("kind" to "adamw", "lr" to 0.1)))
    val batch = mapOf("x" to floatArrayOf(1f, 1f), "y" to floatArrayOf(1f, -1f))
    val r = t.run(300, onStep = { _, l -> l > 1e-6f }) { batch }
    check((r["last_loss"] as Float) < 1e-4f, "trainer converges")
    val opt = Optimizer("sgd", mapOf("lr" to 0.5))
    val p = floatArrayOf(1f, -1f)
    opt.step(mapOf("p" to p), mapOf("p" to floatArrayOf(2f, -2f)))
    check(p.contentEquals(floatArrayOf(0f, 0f)), "optimizer in place")
    // loading
    val file = File.createTempFile("rlx-kt", ".gguf")
    val wv = floatArrayOf(0.5f, -1f, 2f, 0.25f)
    GgufWriter().use { wr -> wr.addF32("w", longArrayOf(4), "F32", wv); check(wr.write(file.path, architecture = "demo") == 1L, "write") }
    Gguf.open(file.path).use { f ->
        check(f.architecture() == "demo", "architecture")
        check(f.tensor("w").contentEquals(wv), "tensor")
        check(f.info("w")["dtype"] == "F32", "info")
    }
    file.delete()
    // backends and a two-rank mesh
    check(RLX.devices().contains("cpu") && RLX.isAvailable("cpu"), "cpu")
    val runner = Runner(lossGraph().first, mapOf("only" to listOf("cpu")))
    runner.setParam("w", floatArrayOf(1f, 0f, 0f, 1f))
    val res = runner.run(mapOf("x" to floatArrayOf(1f, 2f), "y" to floatArrayOf(0f, 0f)))
    check(res.device == "cpu" && res.outputs[0][0] == 2.5f, "runner")
    val peers = RLX.loopbackPeers(2)
    var served: Map<String, Any?>? = null
    val worker = thread {
        val wg = Group.connect(1, 2, mapOf("peers" to peers))
        wg.allReduce(floatArrayOf(3f), "max")
        served = wg.serveWorker(mapOf("steps" to 1))
    }
    val mesh = Group.connect(0, 2, mapOf("peers" to peers))
    check(mesh.allReduce(floatArrayOf(5f), "max")[0] == 5f, "all_reduce")
    check(mesh.collectCaps().size == 1, "caps")
    val s = Graph()
    val sx = s.input("x", longArrayOf(2))
    s.setOutputs(intArrayOf(s.add(sx, sx)))
    mesh.shipStage(1, s, mapOf("device" to "cpu"))
    mesh.sendActivation(1, floatArrayOf(1f, 2f))
    check(mesh.recvActivation(1).contentEquals(floatArrayOf(2f, 4f)), "activation")
    worker.join()
    check((served!!["activations"] as Number).toInt() == 1, "worker served")
    println("RLXTest: $checks checks passed")
}
