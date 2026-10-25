// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
// the same fit by hand with grad and an Optimizer.

package org.rlx.kt.examples

import java.io.File
import kotlin.math.PI
import kotlin.math.cos
import org.rlx.kt.Graph
import org.rlx.kt.Optimizer
import org.rlx.kt.RLX
import org.rlx.kt.Session
import org.rlx.kt.Trainer

private val W_TRUE = floatArrayOf(2f, -1f, 0.5f)
private const val BATCH = 8

private fun data(step: Long): Map<String, FloatArray> {
    val x = FloatArray(BATCH * 3)
    val y = FloatArray(BATCH)
    for (i in 0 until BATCH) for (k in 0 until 3) {
        x[i * 3 + k] = ((step * 31 + i * 7 + k * 3) % 17) / 8.5f - 1
        y[i] += x[i * 3 + k] * W_TRUE[k]
    }
    return mapOf("x" to x, "y" to y)
}

private fun fmt(v: FloatArray) = v.joinToString(" ") { "%.3f".format(it) }

fun main(args: Array<String>) {
    val device = args.firstOrNull() ?: "cpu"
    val g = Graph("regression")
    val x = g.input("x", longArrayOf(BATCH.toLong(), 3))
    val y = g.input("y", longArrayOf(BATCH.toLong(), 1))
    val w = g.param("w", longArrayOf(3, 1))
    val d = g.sub(g.matmul(x, w), y)
    g.setOutputs(intArrayOf(g.mean(g.mul(d, d), longArrayOf(0, 1))))

    val trainer = Trainer(g.clone(), listOf("w"), mapOf("w" to FloatArray(3)),
        mapOf("device" to device, "optimizer" to mapOf("kind" to "adamw", "lr" to 0.1)))
    val first = trainer.evaluate(data(0))
    trainer.run(400, lr = { s -> (0.1 * 0.5 * (1 + cos(PI * s / 400))).toFloat() }, onStep = { _, loss -> loss > 1e-6f }) { data(it) }
    println("trainer loss: %.4f -> %.6f in %d steps".format(first, trainer.evaluate(data(0)), trainer.steps()))
    println("trainer w: ${fmt(trainer.params()["w"]!!)}")

    val path = File.createTempFile("rlx-example-train-kt", ".gguf")
    val written = trainer.save(path.path)
    val restored = Trainer(g.clone(), listOf("w"), mapOf("w" to FloatArray(3)), mapOf("device" to device))
    println("checkpoint: $written tensors, restored step ${restored.load(path.path)["step"]}")
    path.delete()
    restored.freeze(listOf("w"))
    val before = restored.params()["w"]!!
    restored.step(data(1))
    println("frozen: ${restored.frozen()} unchanged: ${restored.params()["w"]!!.contentEquals(before)}")
    restored.unfreeze()

    val backward = Session(device = device).compile(RLX.grad(g, intArrayOf(w))) // outputs [loss, dw]
    val opt = Optimizer("sgd", mapOf("lr" to 0.2))
    val params = FloatArray(3)
    var loss = 0f
    for (step in 0L until 300) {
        backward.setParam("w", params)
        val outs = backward.run(data(step) + ("d_output" to floatArrayOf(1f)))
        loss = outs[0][0]
        opt.step(mapOf("w" to params), mapOf("w" to outs[1]))
    }
    println("manual loss: %.6f w: %s".format(loss, fmt(params)))
}
