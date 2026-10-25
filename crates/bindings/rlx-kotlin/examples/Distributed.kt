// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Distributed: three ranks as threads on loopback; across machines (a phone, a
// laptop, a server), one rank per host with the same peers. Collectives,
// pipelined inference (a stage shipped per worker), data-parallel training.

package org.rlx.kt.examples

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.max
import org.rlx.kt.Graph
import org.rlx.kt.Group
import org.rlx.kt.RLX

private const val WORLD = 3
private const val PER_RANK = 16
private val tmp = kotlin.io.path.createTempDirectory("rlx-dist-kt").toFile()

private fun blob(name: String, values: FloatArray): String {
    val b = ByteBuffer.allocate(4 * values.size).order(ByteOrder.LITTLE_ENDIAN)
    values.forEach { b.putFloat(it) }
    val f = File(tmp, name)
    f.writeBytes(b.array())
    return "file://${f.path}"
}

private fun matvec(x: FloatArray, w: FloatArray) = FloatArray(4) { j -> max(0f, (0 until 4).sumOf { i -> (x[i] * w[i * 4 + j]).toDouble() }.toFloat()) }

@Suppress("UNCHECKED_CAST")
private fun weightsOf(report: Map<String, Any?>) =
    ((report["params"] as Map<String, Any?>)["w"] as List<Number>).map { it.toFloat() }.toFloatArray()

private fun spec(rank: Int, uris: Map<String, String>) = mapOf(
    "params" to listOf(mapOf("name" to "w", "uri" to uris["w"])),
    "data" to listOf(
        mapOf("input" to "X", "uri" to uris["X"], "elem" to 3, "shard_start" to rank * PER_RANK, "shard_len" to PER_RANK),
        mapOf("input" to "y", "uri" to uris["y"], "elem" to 1, "shard_start" to rank * PER_RANK, "shard_len" to PER_RANK),
    ),
    "batch" to 4, "lr_per_epoch" to List(40) { 0.2 }, "device" to "cpu",
)

fun main() {
    val stageW = mapOf(1 to FloatArray(16) { (it * 5 % 7) / 7f - 0.3f }, 2 to FloatArray(16) { (it * 3 % 5) / 5f - 0.2f })
    val wTrue = floatArrayOf(1.5f, -2f, 0.5f)
    val xAll = FloatArray(PER_RANK * WORLD * 3) { (it * 37 % 23) / 11.5f - 1 }
    val yAll = FloatArray(PER_RANK * WORLD) { i -> (0 until 3).sumOf { k -> (xAll[i * 3 + k] * wTrue[k]).toDouble() }.toFloat() }

    val peers = RLX.loopbackPeers(WORLD)
    val options = mapOf("peers" to peers, "device" to "cpu")
    val results = arrayOfNulls<FloatArray>(WORLD)
    val workers = (1 until WORLD).map { rank ->
        thread {
            val g = Group.connect(rank, WORLD, options)
            g.allReduce(floatArrayOf(rank + 1f), "sum")
            g.allGather(floatArrayOf(rank + 1f))
            g.broadcast(0, floatArrayOf(0f)) // the root's length; it overwrites
            g.serveWorker(mapOf("steps" to 2))
            results[rank] = weightsOf(g.serveTrainer())
        }
    }

    val g = Group.connect(0, WORLD, options)
    println("mesh: rank ${g.rank()} of ${g.world()}, leader: ${g.isLeader()}")
    println("all_reduce sum: ${g.allReduce(floatArrayOf(1f), "sum").toList()}")
    println("all_gather: ${g.allGather(floatArrayOf(1f)).toList()}")
    println("broadcast: ${g.broadcast(0, floatArrayOf(42f)).toList()}")
    println("workers: " + g.collectCaps().joinToString(", ") { "rank ${it["rank"]} on ${it["platform"]}" })
    val stage = Graph("stage")
    val sx = stage.input("x", longArrayOf(1, 4))
    stage.setOutputs(intArrayOf(stage.relu(stage.matmul(sx, stage.param("w", longArrayOf(4, 4))))))
    for (r in 1 until WORLD) g.shipStage(r, stage, mapOf("device" to "cpu", "weights" to mapOf("w" to blob("w$r.bin", stageW[r]!!))))
    for (xv in listOf(floatArrayOf(1f, -1f, 0.5f, 2f), floatArrayOf(0.25f, 0.5f, -0.75f, 1f))) {
        g.sendActivation(1, xv)
        val yv = g.recvActivation(WORLD - 1)
        val want = matvec(matvec(xv, stageW[1]!!), stageW[2]!!)
        println("pipeline max|Δ| vs local = %.1e".format(yv.indices.maxOf { abs(yv[it] - want[it]) }))
    }

    g.collectCaps()
    val lr = Graph("lr")
    val bx = lr.input("X", longArrayOf(4, 3))
    val by = lr.input("y", longArrayOf(4, 1))
    val wv = lr.param("w", longArrayOf(3, 1))
    val d = lr.sub(lr.matmul(bx, wv), by)
    lr.setOutputs(intArrayOf(lr.mean(lr.mul(d, d), longArrayOf(0, 1))))
    val bwd = RLX.grad(lr, intArrayOf(wv))
    val uris = mapOf("X" to blob("X.bin", xAll), "y" to blob("y.bin", yAll), "w" to blob("w0.bin", FloatArray(3)))
    for (r in 1 until WORLD) g.shipTrain(r, bwd, spec(r, uris))
    val mine = g.runTrain(bwd, spec(0, uris))
    @Suppress("UNCHECKED_CAST") val m = mine["metrics"] as Map<String, Any?>
    println("training loss: %.4f -> %.2e".format((m["first_loss"] as Number).toDouble(), (m["last_loss"] as Number).toDouble()))
    val w0 = weightsOf(mine)
    println("rank 0 w: %.3f %.3f %.3f (true: 1.5 -2 0.5)".format(w0[0], w0[1], w0[2]))
    workers.forEach { it.join() }
    for (r in 1 until WORLD) println("rank $r w agrees: ${w0.indices.all { abs(results[r]!![it] - w0[it]) < 1e-5 }}")
    tmp.deleteRecursively()
}
